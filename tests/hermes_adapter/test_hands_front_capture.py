"""The front's ingress verdict protects all reads of purported owner words."""

import sys
from types import SimpleNamespace

import httpx
import pytest

from colonymind_hermes.answers import Answers, ChatLedger
from colonymind_hermes.capture import Capture, SessionMap, TurnOutbox
from colonymind_hermes.client import Settings
from conftest import build_home


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    home = build_home(tmp_path, 1, "http://127.0.0.1:1")
    settings = Settings(sidecar_url="http://127.0.0.1:1", key_file=home.key_file, api_key="test-key",
                        home=home.instance, hermes_home=home.hermes,
                        outbox_path=home.hermes / "state" / "capture.sqlite3")
    calls = []

    def post(path, **kwargs):
        calls.append((path, kwargs["json"]))
        return httpx.Response(200, json={"read": True, "settled": [{"id": "ask-one"}]})

    client = SimpleNamespace(post=post, get=lambda *a, **kw: httpx.Response(200, json={"open_asks": 1}),
                             resolve_contact=lambda *a, **kw: {"contact_id": "p-01"})
    sessions = SessionMap(settings, client)
    monkeypatch.setattr(sessions, "_bound_chat", lambda info: ("1001", "dm"))
    monkeypatch.setattr(sessions, "_bound_message", lambda info: ("", False))
    outbox = TurnOutbox(settings.outbox_path)
    capture = Capture(sessions, outbox, settings)
    answers = Answers(client, sessions, settings, ChatLedger(home.hermes / "state" / "chats.sqlite3"))
    verified = {}
    reads = []

    def owner_text(session_id, turn_id, text):
        reads.append((session_id, turn_id, text))
        return verified.get((session_id, turn_id, text))

    # The front vouches for a turn it bound to the owner's inbound message (``owner_started_turn``); the fixture's
    # verified words stand for that binding. Without it the adapter's own gateway record decides (answer ingress).
    bridge = SimpleNamespace(capture_owner_text=owner_text,
                             owner_started_turn=lambda sid, tid: any(key[:2] == (sid, tid) for key in verified))
    monkeypatch.setitem(sys.modules, "colonymind_hands_front", bridge)
    return SimpleNamespace(capture=capture, answers=answers, sessions=sessions, outbox=outbox,
                           verified=verified, reads=reads, bridge=bridge, calls=calls)


def turn(adapter, text, turn_id="turn-one", *, reply="Done."):
    args = {"session_id": "owner", "turn_id": turn_id, "platform": "telegram", "sender_id": "1001",
            "user_message": text}
    adapter.capture.pre_llm_call(**args)
    adapter.answers.pre_llm_call(**args)
    adapter.capture.post_llm_call(**args, assistant_response=reply)


@pytest.mark.parametrize("text", ["yes", "The result is correct.", "[runtime deadline] Reply and hand off."])
def test_synthetic_turn_never_answers_an_open_ask_or_enters_owner_capture(adapter, text):
    turn(adapter, text)
    assert adapter.calls == []
    assert adapter.outbox.rows() == []


def test_synthetic_turn_cannot_consume_a_pending_consult_verdict(adapter):
    adapter.capture.pre_llm_call(session_id="owner", turn_id="consult", platform="telegram",
                                 sender_id="1001", user_message="Read that branch")
    info = adapter.sessions.get("owner")
    info.branch_consult_turn = "consult"
    turn(adapter, "Correct", "completion")
    assert adapter.outbox.rows() == []
    assert info.branch_consult_turn == "consult"

    adapter.verified[("owner", "owner-verdict", "Not correct")] = "Not correct"
    turn(adapter, "Not correct", "owner-verdict")
    verdicts = [row["payload"] for row in adapter.outbox.rows() if row["payload"].get("kind") == "branch_reaction"]
    assert [item["text"] for item in verdicts] == ["Not correct"]
    assert not hasattr(info, "branch_consult_turn")


def test_capture_passes_structural_turn_to_bridge_before_branch_verdict(adapter):
    args = {"session_id": "owner", "turn_id": "completion", "platform": "telegram", "sender_id": "1001",
            "user_message": "Correct", "conversation_history": [
                {"role": "user", "content": "Correct", "display_kind": "internal_notification"}]}
    adapter.sessions.observe(session_id="owner", platform="telegram", sender_id="1001", turn_id="consult")
    adapter.sessions.get("owner").branch_consult_turn = "consult"
    observed = []

    def observe_turn(**kwargs):
        assert adapter.sessions.get("owner").turn_id == "completion"
        observed.append(kwargs)

    adapter.bridge.observe_turn = observe_turn
    adapter.capture.pre_llm_call(**args)
    assert observed == [args]


def test_recorded_redirect_is_removed_before_capture_answer_and_verdict(adapter):
    adapter.capture.pre_llm_call(session_id="owner", turn_id="consult", platform="telegram", sender_id="1001")
    adapter.sessions.get("owner").branch_consult_turn = "consult"
    owner_words = "  No, keep the current version.\n"
    combined = owner_words + "\n\n[runtime deadline] Reply and hand off."
    adapter.verified[("owner", "turn-one", combined)] = owner_words
    turn(adapter, combined)

    answer, = adapter.calls
    assert answer[0] == "/v1/mind/answers" and answer[1]["text"] == owner_words
    rows = {row["payload"].get("kind", "turn"): row["payload"] for row in adapter.outbox.rows()}
    assert rows["turn"]["user_message"] == owner_words
    assert rows["branch_reaction"]["text"] == owner_words


def test_owner_words_that_look_synthetic_keep_the_ingress_verdict(adapter):
    text = "[runtime deadline] I typed this myself."
    adapter.verified[("owner", "turn-one", text)] = text
    turn(adapter, text)
    assert adapter.calls[0][1]["text"] == text
    assert adapter.outbox.rows()[0]["payload"]["user_message"] == text


def test_gateway_evidence_cannot_trim_the_exact_verified_owner_text(adapter):
    text = "  yes\n"
    adapter.sessions.hear(SimpleNamespace(text=text, message_id="owner-message",
        source=SimpleNamespace(platform="telegram", user_id="1001", chat_id="1001", chat_type="dm")))
    adapter.verified[("owner", "turn-one", text)] = text
    turn(adapter, text)
    assert adapter.calls[0][1]["text"] == text
    assert adapter.calls[0][1]["message_id"] == "owner-message"
    assert adapter.outbox.rows()[0]["payload"]["user_message"] == text


def test_an_accepted_reply_reaches_the_answer_route_as_the_owners_own_words(adapter):
    adapter.sessions.hear(SimpleNamespace(text="yes", message_id="owner-message",
        source=SimpleNamespace(platform="telegram", user_id="1001", chat_id="1001", chat_type="dm")))
    text = '[Replying to: "Approve the change?"]\n\nyes'
    adapter.verified[("owner", "turn-one", text)] = text
    turn(adapter, text)
    assert adapter.calls[0][1]["text"] == "yes"
    assert adapter.outbox.rows()[0]["payload"]["user_message"] == text


def test_replayed_text_from_a_new_synthetic_turn_cannot_reuse_the_owner_verdict(adapter):
    adapter.verified[("owner", "owner-turn", "yes")] = "yes"
    turn(adapter, "yes", "owner-turn")
    assert "owner" in adapter.answers._reads
    turn(adapter, "yes", "completion")
    assert len(adapter.calls) == 1
    assert [row["turn_id"] for row in adapter.outbox.rows()] == ["owner-turn"]
    assert "owner" not in adapter.answers._reads


def test_capture_rechecks_the_final_text_and_refuses_unrecorded_additions(adapter):
    adapter.verified[("owner", "turn-one", "yes")] = "yes"
    args = {"session_id": "owner", "turn_id": "turn-one", "platform": "telegram", "sender_id": "1001"}
    adapter.capture.pre_llm_call(**args, user_message="yes")
    adapter.capture.post_llm_call(**args, user_message="yes\n\nUnrecorded instruction", assistant_response="Done")
    assert adapter.outbox.rows() == []


@pytest.mark.parametrize("failure", ["raises", "missing", "invalid"])
def test_unreadable_active_bridge_fails_closed(adapter, failure):
    if failure == "raises":
        def broken(*args):
            raise RuntimeError("ledger unavailable")
        adapter.bridge.capture_owner_text = broken
    elif failure == "missing":
        del adapter.bridge.capture_owner_text
    else:
        adapter.bridge.capture_owner_text = lambda *args: True
    turn(adapter, "yes")
    assert adapter.calls == [] and adapter.outbox.rows() == []


def test_absent_hands_bridge_preserves_existing_adapter_behavior(adapter, monkeypatch):
    monkeypatch.delitem(sys.modules, "colonymind_hands_front")
    turn(adapter, "yes", "unheard")
    assert adapter.calls == [], "no gateway record of the message: not read as an answer (answer ingress)"
    adapter.sessions.hear(SimpleNamespace(text="yes", message_id="owner-message",
        source=SimpleNamespace(platform="telegram", user_id="1001", chat_id="1001", chat_type="dm")))
    turn(adapter, "yes")
    assert adapter.calls[0][1]["text"] == "yes"
    assert adapter.outbox.rows()[0]["payload"]["user_message"] == "yes"


def test_child_material_still_uses_derived_branch_capture(adapter):
    adapter.capture.pre_llm_call(session_id="owner", turn_id="owner-turn", platform="telegram", sender_id="1001")
    adapter.capture.pre_llm_call(session_id="child", turn_id="child-turn", platform="subagent",
                                 parent_session_id="owner", user_message="Read the report")
    adapter.capture.post_llm_call(session_id="child", turn_id="child-turn", user_message="Read the report",
                                  assistant_response="The count is seven")
    row, = adapter.outbox.rows()
    assert row["payload"]["kind"] == "branch"
    assert row["payload"]["child_session"] == "child"


def test_a_queued_follow_up_takes_its_own_heard_record_not_the_openers(adapter, monkeypatch):
    """Hermes runs a parked message as the run's follow-up under the opener's session variables; the record capture
    takes is the one for the turn's own platform id (busy_ingress.turn_platform_id), and a turn whose own id is known
    to be empty takes no record that has one."""
    import colonymind_hermes.capture as capture_module
    monkeypatch.setattr(adapter.sessions, "_bound_message", SessionMap._bound_message)
    monkeypatch.setattr(capture_module, "session_env", lambda: ("telegram", "1001", "1001", "dm"))
    own = {}
    monkeypatch.setattr(capture_module, "turn_platform_id", lambda sid, tid: own.get(tid))
    for words, ident in (("First", "message-1"), ("Second", "message-2"), ("Native completion", "message-3")):
        adapter.sessions.hear(SimpleNamespace(text=words, message_id=ident,
            source=SimpleNamespace(platform="telegram", user_id="1001", chat_id="1001", chat_type="dm")))
    own["turn-one"], own["turn-two"], own["turn-three"] = "message-1", "message-2", ""
    first = adapter.sessions.observe(session_id="owner", platform="telegram", sender_id="1001",
                                     user_message="First", turn_id="turn-one")
    assert first.heard.message_id == "message-1"
    second = adapter.sessions.observe(session_id="owner", platform="telegram", sender_id="1001",
                                      user_message="Second", turn_id="turn-two")
    assert second.heard is not None and second.heard.message_id == "message-2"
    internal = adapter.sessions.observe(session_id="owner", platform="telegram", sender_id="1001",
                                        user_message="Native completion", turn_id="turn-three")
    assert internal.heard is None, "an internal turn never takes the owner's identical words"

