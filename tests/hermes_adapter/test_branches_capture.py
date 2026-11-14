"""Owner delegation is captured safely before the outbox or transport sees it."""

import json
from types import SimpleNamespace

import httpx
import pytest

from colonymind_hermes.capture import Capture, SessionMap, TurnOutbox, deliver
from colonymind_hermes.client import Settings
from colonymind_hermes.tools import Tools
from conftest import build_home


@pytest.fixture
def home(tmp_path):
    return build_home(tmp_path, 1, "http://127.0.0.1:1")


@pytest.fixture
def adapter(home):
    settings = Settings(sidecar_url="http://127.0.0.1:1", key_file=home.key_file, api_key="test-key",
                        home=home.instance, hermes_home=home.hermes,
                        outbox_path=home.hermes / "state" / "branches.sqlite3")
    calls = []

    def request(path, **kwargs):
        calls.append((path, kwargs))
        return httpx.Response(200, json={"branches": [], "tool_available": True, "answer": "kept"})

    client = SimpleNamespace(post=request, get=request,
                             resolve_contact=lambda platform, sender, **kw: {
                                 "contact_id": "p-01" if sender == "1001" else "p-03"})
    sessions = SessionMap(settings, client)
    outbox = TurnOutbox(settings.outbox_path)
    return SimpleNamespace(settings=settings, sessions=sessions, outbox=outbox, calls=calls,
                           client=client, capture=Capture(sessions, outbox, settings))


def finish(adapter, sender="1001", platform="telegram", child="child-1"):
    adapter.capture.pre_llm_call(session_id="parent", sender_id=sender, platform=platform,
                                 user_message="Read the report", turn_id="owner-turn")
    adapter.capture.pre_llm_call(session_id=child, parent_session_id="parent", platform="subagent",
                                 user_message="Read the report", turn_id="child-turn")
    adapter.capture.post_llm_call(session_id=child, platform="subagent", user_message="Read the report",
                                  assistant_response="The count is seven", conversation_history=[
                                      {"role": "user", "content": "Read the report"},
                                      {"role": "tool", "content": "password: synthetic-value-8412"},
                                      {"role": "assistant", "content": "The count is seven"}])


def test_owner_child_enqueues_full_material_once_and_masks_before_disk(adapter):
    finish(adapter)
    finish(adapter)
    branches = [r for r in adapter.outbox.rows() if r["payload"].get("kind") == "branch"]
    assert len(branches) == 1
    payload = branches[0]["payload"]
    assert payload["parent_session"] == "parent" and payload["child_session"] == "child-1"
    assert [m["role"] for m in payload["messages"]] == ["user", "tool", "assistant"]
    assert payload["requester"] == "p-01"
    assert payload["sender"] == {"platform": "telegram", "user_id": "1001"}
    assert "synthetic-value-8412" not in json.dumps(payload)
    assert "synthetic-value-8412".encode() not in adapter.outbox.path.read_bytes()
    assert adapter.calls == []


@pytest.mark.parametrize("sender,platform", [("2003", "telegram"), ("", "telegram"), ("", "cron")])
def test_non_owner_parent_child_is_never_queued(adapter, sender, platform):
    finish(adapter, sender, platform)
    assert adapter.outbox.rows() == []


def test_disabled_faculty_has_no_capture_or_consult_tool(adapter, home):
    home.write_mind(faculties={"working_memory": False})
    finish(adapter)
    assert adapter.outbox.rows() == []
    assert "colonymind_consult" not in [s["name"] for s, _ in Tools(
        adapter.client, adapter.sessions, adapter.settings).handlers()]


def test_branch_delivery_uses_branch_route_and_bound_requester(adapter):
    finish(adapter)
    row, = [r for r in adapter.outbox.rows() if r["payload"].get("kind") == "branch"]
    assert deliver(adapter.client, adapter.sessions, adapter.settings, row["payload"])
    path, options = adapter.calls[-1]
    assert path == "/v1/memory/branches"
    assert options["json"]["requester"] == "p-01"
    assert "kind" not in options["json"]


def test_consult_is_registered_and_uses_session_not_argument_identity(adapter):
    adapter.sessions.observe(session_id="parent", platform="cli")
    handlers = {s["name"]: h for s, h in Tools(adapter.client, adapter.sessions, adapter.settings).handlers()}
    assert "colonymind_consult" in handlers
    answer = json.loads(handlers["colonymind_consult"]({"handles": ["br-0123456789ab"], "question": "Count?",
                                                       "requester": "p-03"}, session_id="parent"))
    assert answer["answer"] == "kept"
    path, options = adapter.calls[-1]
    assert path == "/v1/memory/branches/consult"
    assert options["json"] == {"handles": ["br-0123456789ab"], "question": "Count?", "requester": "p-01",
                               "session_id": "parent", "sender": {"platform": "cli", "user_id": ""}}


def test_guest_cannot_consult_even_when_tool_arguments_claim_owner(adapter):
    adapter.sessions.observe(session_id="guest", platform="telegram", sender_id="2003")
    adapter.client.resolve_contact = lambda *a, **kw: {"contact_id": "p-03"}
    tools = Tools(adapter.client, adapter.sessions, adapter.settings)
    answer = json.loads(tools.consult({"handles": ["br-0123456789ab"], "question": "Count?",
                                       "requester": "p-01"}, session_id="guest"))
    assert answer["unavailable"] is True and answer["retry"] is False
    assert adapter.calls == []


def test_oversized_child_span_is_not_written(adapter):
    from colonymind_hermes.capture import BRANCH_SPAN_BYTES
    adapter.capture.pre_llm_call(session_id="parent", platform="cli")
    adapter.capture.pre_llm_call(session_id="child", platform="subagent", parent_session_id="parent")
    adapter.capture.post_llm_call(session_id="child", conversation_history=[
        {"role": "tool", "content": "x" * (BRANCH_SPAN_BYTES + 1)}])
    assert adapter.outbox.rows() == []


def test_consult_next_owner_turn_queues_one_reaction_with_no_network(adapter):
    adapter.capture.pre_llm_call(session_id="parent", platform="cli", turn_id="turn-1")
    Tools(adapter.client, adapter.sessions, adapter.settings).consult(
        {"handles": ["br-0123456789ab"], "question": "Count?"}, session_id="parent")
    adapter.calls.clear()
    adapter.capture.pre_llm_call(session_id="parent", platform="cli", turn_id="turn-2", user_message="Correct")
    adapter.capture.pre_llm_call(session_id="parent", platform="cli", turn_id="turn-2", user_message="Correct")
    row, = adapter.outbox.rows()
    assert row["payload"]["kind"] == "branch_reaction"
    assert row["payload"]["text"] == "Correct" and row["payload"]["turn_id"] == "turn-2"
    assert adapter.calls == []


def test_consult_availability_follows_withdrawal_and_summary_only(adapter, home):
    tools = Tools(adapter.client, adapter.sessions, adapter.settings)
    assert tools.consult_available()
    adapter.client.get = lambda *a, **kw: httpx.Response(200, json={"branches": [], "tool_available": False})
    assert not tools.consult_available()
    import yaml
    path = home.instance / "colonymind.yaml"
    configuration = yaml.safe_load(path.read_text())
    configuration["working_memory"] = {"summary_only": True}
    path.write_text(yaml.safe_dump(configuration))
    assert "colonymind_consult" not in [s["name"] for s, _ in tools.handlers()]


def test_session_manifest_waits_for_its_durable_checkpoint(adapter):
    adapter.outbox.enqueue("checkpoint-1", {"session_id": "parent"})
    payload = {"kind": "branch_session", "session_id": "parent", "requester": "p-01",
               "sender": {"platform": "cli", "user_id": ""}, "checkpoint_turn_id": "checkpoint-1"}
    assert deliver(adapter.client, adapter.sessions, adapter.settings, payload) is False
    assert adapter.calls == []
    adapter.outbox.drain(lambda payload: True)
    assert deliver(adapter.client, adapter.sessions, adapter.settings, payload)
    assert adapter.calls[-1][0] == "/v1/memory/branches/session"
    assert "checkpoint_turn_id" not in adapter.calls[-1][1]["json"]


def test_consult_refusal_preserves_the_reason_and_every_handle(adapter):
    adapter.sessions.observe(session_id="parent", platform="cli")
    adapter.client.post = lambda *a, **kw: httpx.Response(413, json={"detail": {
        "code": "token_cap", "handles": ["br-0123456789ab", "br-abcdef123456"]}})
    result = json.loads(Tools(adapter.client, adapter.sessions, adapter.settings).consult(
        {"handles": ["br-0123456789ab", "br-abcdef123456"], "question": "Count?"}, session_id="parent"))
    assert result["unavailable"] is True and result["retry"] is False
    assert result["reason"] == "token_cap: br-0123456789ab, br-abcdef123456"


def test_terminal_rejection_discards_branch_material_from_outbox(adapter):
    finish(adapter)
    adapter.client.post = lambda *a, **kw: httpx.Response(403)
    adapter.outbox.drain(lambda payload: deliver(adapter.client, adapter.sessions, adapter.settings, payload))
    row, = adapter.outbox.rows()
    assert row["state"] == "delivered" and row["payload"] == {}


def test_expired_pending_branch_is_never_delivered_after_an_outage(adapter, monkeypatch):
    finish(adapter)
    row, = adapter.outbox.rows()
    monkeypatch.setattr("colonymind_hermes.capture.time.time", lambda: row["payload"]["queued_at"] + 15 * 86400)
    assert deliver(adapter.client, adapter.sessions, adapter.settings, row["payload"])
    assert adapter.calls == []


def test_structured_tool_arguments_are_masked_before_json_escaping_and_disk(adapter):
    adapter.capture.pre_llm_call(session_id="parent", platform="cli")
    adapter.capture.pre_llm_call(session_id="child", parent_session_id="parent", platform="subagent")
    secret = "synthetic-value-8412"
    adapter.capture.post_llm_call(session_id="child", conversation_history=[{
        "role": "assistant", "content": "Checking the report", "tool_calls": [{
            "id": "call-1", "type": "function", "function": {
                "name": "read_report", "arguments": json.dumps({"password": secret, "report": "weekly"})}}]},
        {"role": "tool", "content": "Seven entries"}])
    row, = adapter.outbox.rows()
    assert secret not in json.dumps(row["payload"])
    assert secret.encode() not in adapter.outbox.path.read_bytes()
    assert "weekly" in row["payload"]["messages"][0]["content"]


def test_redaction_cannot_expand_a_span_past_the_capture_limit(adapter):
    from colonymind_hermes.capture import BRANCH_SPAN_BYTES
    adapter.capture.pre_llm_call(session_id="parent", platform="cli")
    adapter.capture.pre_llm_call(session_id="child", parent_session_id="parent", platform="subagent")
    content = "password: 8412\n" * (BRANCH_SPAN_BYTES // 16)
    assert len(content.encode()) < BRANCH_SPAN_BYTES
    adapter.capture.post_llm_call(session_id="child", conversation_history=[{"role": "tool", "content": content}])
    assert adapter.outbox.rows() == []


def test_disabling_faculty_consumes_the_next_turn_without_later_reaction(adapter, home):
    adapter.capture.pre_llm_call(session_id="parent", platform="cli", turn_id="turn-1")
    Tools(adapter.client, adapter.sessions, adapter.settings).consult(
        {"handles": ["br-0123456789ab"], "question": "Count?"}, session_id="parent")
    home.write_mind(faculties={"working_memory": False})
    adapter.capture.pre_llm_call(session_id="parent", platform="cli", turn_id="turn-2", user_message="Correct")
    home.write_mind(faculties={"working_memory": True})
    adapter.calls.clear()
    adapter.capture.pre_llm_call(session_id="parent", platform="cli", turn_id="turn-3", user_message="Wrong")
    assert adapter.outbox.rows() == []
    assert adapter.calls == []


@pytest.mark.parametrize("chat_type", ["group", "channel"])
def test_owner_shared_chat_child_is_not_captured_or_consulted(adapter, chat_type):
    adapter.capture.pre_llm_call(session_id="parent", sender_id="1001", platform="telegram")
    adapter.sessions.get("parent").chat_type = chat_type
    adapter.capture.pre_llm_call(session_id="child", parent_session_id="parent", platform="subagent")
    adapter.capture.post_llm_call(session_id="child", assistant_response="Shared result")
    result = json.loads(Tools(adapter.client, adapter.sessions, adapter.settings).consult(
        {"handles": ["br-0123456789ab"], "question": "Count?"}, session_id="parent"))
    assert result["unavailable"] is True and result["retry"] is False
    assert adapter.outbox.rows() == [] and adapter.calls == []


def test_registered_consult_checks_live_withdrawal_and_faculty_off(adapter, home, monkeypatch, request):
    import colonymind_hermes

    registered = {}
    ctx = SimpleNamespace(register_hook=lambda *args: None, register_command=lambda *args, **kwargs: None,
                          on_unload=request.addfinalizer,
                          register_system_prompt_section=lambda *args: None,
                          register_tool=lambda **kwargs: registered.update({kwargs["name"]: kwargs}))
    monkeypatch.setenv("COLONYMIND_BODY_THREAD", "0")
    monkeypatch.setattr(colonymind_hermes, "_BODY", None)
    monkeypatch.setattr(colonymind_hermes, "_REGISTERED_TOOLS", None)
    monkeypatch.setattr(colonymind_hermes, "load_settings", lambda: adapter.settings)
    monkeypatch.setattr(colonymind_hermes, "ColonyMindClient", lambda settings: adapter.client)
    monkeypatch.setattr(colonymind_hermes.secrets_module, "register", lambda *args: None)
    colonymind_hermes.register(ctx)
    assert adapter.calls == []
    entry = registered["colonymind_consult"]
    assert entry["check_fn"]()
    adapter.client.get = lambda *args, **kwargs: httpx.Response(200, json={"tool_available": False})
    assert not entry["check_fn"]()
    adapter.client.post = lambda *args, **kwargs: httpx.Response(409, json={"detail": {"code": "withdrawn"}})
    entry["handler"].__self__.sessions.observe(session_id="parent", platform="cli")
    response = json.loads(entry["handler"]({"handles": ["br-0123456789ab"], "question": "Count?"},
                                            session_id="parent"))
    assert response["unavailable"] is True and response["reason"] == "withdrawn"
    home.write_mind(faculties={"working_memory": False})
    adapter.calls.clear()
    assert not entry["check_fn"]()
    response = json.loads(entry["handler"]({"handles": ["br-0123456789ab"], "question": "Count?"},
                                            session_id="parent"))
    assert response["unavailable"] is True and response["retry"] is False
    assert adapter.calls == []


@pytest.mark.parametrize("arguments,expected", [
    ({"handles": [], "question": "Count?"}, "handles must be"),
    ({"handles": ["br-0123456789ab"], "question": " "}, "question must be"),
])
def test_consult_bad_arguments_can_be_corrected_without_network(adapter, arguments, expected):
    adapter.sessions.observe(session_id="parent", platform="cli")
    result = json.loads(Tools(adapter.client, adapter.sessions, adapter.settings).consult(
        arguments, session_id="parent"))
    assert expected in result["error"]
    assert "unavailable" not in result and adapter.calls == []
