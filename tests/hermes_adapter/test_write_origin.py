"""Synthetic write-origin contract tests; no service endpoints."""
import contextvars
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from colonymind_hermes import capture, guard, write_origin
from test_unknown_standing import lane  # noqa: F401


def start(lane, *, platform="companion", sender="owner-1", session="session", turn="turn", words="go ahead", tool="web_search", args=None):
    """A message arrives at the gateway, its turn starts and the turn calls a tool. The fixture binds no gateway
    session, so the turn takes the gateway's record of the message by its words alone."""
    gate, sessions = lane
    sessions.settings.owner_handles = lambda: {platform: ["owner-1"]}
    sessions.hear(SimpleNamespace(text=words, message_id="message-" + turn,
        source=SimpleNamespace(platform=platform, user_id=sender, chat_id=session, chat_type="dm")))
    sessions.observe(session_id=session, platform=platform, sender_id=sender, turn_id=turn, user_message=words)
    gate.pre_tool_call(tool, args or {}, session, turn_id=turn)
    return gate


@pytest.mark.parametrize("platform", ["companion", "whatsapp", "rcs", "voice", "telegram", "sms"])
def test_owner_channels(lane, platform):
    gate = start(lane, platform=platform, args={"z": 2, "a": "é"})
    origin = guard.call_origin()
    assert origin == {"origin": "owner_channel", "channel": platform, "turn_id": "turn", "session_id": "session",
        "tool": "web_search", "args_hash": hashlib.sha256(
            '{"args":{"a":"é","z":2},"tool":"web_search"}'.encode()).hexdigest(), "verified": True}
    origin["origin"] = "other"
    assert guard.call_origin()["origin"] == "owner_channel"
    gate.post_tool_call(session_id="session", turn_id="turn")
    assert guard.call_origin() is None


def test_a_message_without_words_and_without_a_bound_id_is_other(lane):
    """A message with no words of the sender's (an attachment or a voice note sent with no caption) digests like
    every other such message, so words cannot tie the gateway's record of it to a turn: only the id the gateway
    bound for the turn can. With none bound the turn takes no record, and its call is not the owner's."""
    start(lane, words="")
    assert lane[1].get("session").heard is None
    assert guard.call_origin()["origin"] == "other"


@pytest.mark.parametrize("sender,origin", [("owner-1", "owner_channel"), ("guest-1", "other")])
def test_group_speaker(lane, monkeypatch, sender, origin):
    start(lane)
    monkeypatch.setattr(capture, "session_env", lambda: ("companion", sender, "group", "group"))
    lane[0].pre_tool_call("colonymind_self", {}, "session", turn_id="turn")
    assert guard.call_origin()["origin"] == origin


def test_unknown_and_inherited_owner_are_other(lane):
    gate, sessions = lane
    gate.pre_tool_call("web_search", {}, "unknown", turn_id="unknown-turn")
    assert guard.call_origin()["origin"] == "other"
    assert guard.call_origin()["verified"] is False
    sessions.inherit("child", "", standing="owner", platform="companion")
    gate.pre_tool_call("web_search", {}, "child", turn_id="child-turn")
    assert guard.call_origin()["origin"] == "other"


def test_child_and_bound_sender_without_turn_ingress_are_other(lane, monkeypatch):
    gate, sessions = lane
    monkeypatch.setattr(capture, "session_env", lambda: ("companion", "owner-1", "chat", "dm"))
    sessions.settings.owner_handles = lambda: {"companion": ["owner-1"]}
    gate.pre_tool_call("web_search", {}, "direct", turn_id="direct-turn")
    assert guard.call_origin()["origin"] == "other"
    sessions.inherit("child", "", standing="owner", platform="companion")
    gate.pre_tool_call("web_search", {}, "child", turn_id="child-turn")
    assert guard.call_origin()["origin"] == "other"


@pytest.mark.parametrize("saved,expected", [("owner", "autonomy"), ("mind", "other"), ("unknown", "other")])
@pytest.mark.parametrize("kind", ["cron", "kanban"])
def test_automation_requires_owner_record(lane, monkeypatch, saved, expected, kind):
    gate, sessions = lane
    sessions.standings.put(kind, "job", saved, "companion")
    if kind == "kanban":
        monkeypatch.setenv("HERMES_KANBAN_TASK", "job")
    else:
        monkeypatch.setattr(gate, "_cron_job", lambda *a: "job")
    gate.pre_tool_call("web_search", {}, "scheduled", task_id="cron:job:run", turn_id="run-turn")
    assert guard.call_origin()["origin"] == expected
    assert ("run_id" in guard.call_origin()) == (expected == "autonomy")


def test_args_and_headers_cannot_spoof_and_calls_clear(lane):
    gate = start(lane, sender="guest-1", args={"origin": "owner_channel", "verified": True,
        "headers": {"X-ColonyMind-Origin": "owner_channel"}})
    assert guard.call_origin()["origin"] == "other"
    copied = contextvars.copy_context()
    gate.post_tool_call(session_id="session", turn_id="turn")
    assert copied.run(guard.call_origin) is None
    gate.pre_tool_call("web_search", {}, "orphan", turn_id="other", headers={"origin": "owner_channel"})
    assert guard.call_origin()["origin"] == "other"
    assert guard.call_origin()["session_id"] == "orphan"


def result(lane, content, *, session="session", turn="turn"):
    lane[0].post_tool_call(result=content, session_id=session, turn_id=turn)
    lane[0].pre_tool_call("web_search", {}, session, turn_id=turn)


def test_addresses_only_results_minus_owner_and_contacts_cached(lane):
    gate = start(lane, words="Use OWN@EXAMPLE.TEST", args={"query": "arg@example.test"})
    calls = []
    def lookup(platform, address, **kw):
        calls.append((platform, address, kw))
        return {"contact_id": "known"} if address == "known@example.test" else None
    gate.client.resolve_contact = lookup
    assert guard.untrusted_addresses() == set()
    result(lane, {"body": "OWN@example.test KNOWN@example.test Fresh@Example.Test fresh@example.test"})
    assert guard.untrusted_addresses() == {"fresh@example.test"}
    assert guard.untrusted_addresses() == {"fresh@example.test"}
    assert len(calls) == 2 and all(p == "email" and 0 < kw["timeout"] <= 0.2 for p, _, kw in calls)


def test_lookup_failure_turn_isolation_end_and_cap(lane):
    gate = start(lane)
    def failed(*a, **kw):
        raise RuntimeError("synthetic failure")
    gate.client.resolve_contact = failed
    result(lane, [{"type": "text", "text": "unknown@example.test"}])
    assert guard.untrusted_addresses() == {"unknown@example.test"}
    start(lane, session="second", turn="next")
    assert guard.untrusted_addresses() == set()
    result(lane, " ".join(f"person{i}@example.test" for i in range(300)), session="second", turn="next")
    assert guard.untrusted_addresses() == {"*"}
    assert len(gate.write_origin.turns[("second", "next")]["seen"]) == write_origin.ADDRESS_CAP
    gate.write_origin.end("second", "next")
    assert guard.untrusted_addresses() == set() and guard.call_origin() is None
    assert ("second", "next") not in gate.write_origin.turns
    start(lane, session="second", turn="later")
    assert guard.untrusted_addresses() == set()


def test_late_results_and_owner_words_do_not_cross_turns(lane):
    gate = start(lane, turn="old", words="owner-old@example.test")
    gate.client.resolve_contact = lambda *a, **kw: None
    start(lane, turn="new", words="owner-new@example.test")
    gate.post_tool_call(result="late@example.test", session_id="session", turn_id="old")
    gate.pre_tool_call("web_search", {}, "session", turn_id="new")
    assert guard.untrusted_addresses() == set()
    result(lane, "owner-old@example.test owner-new@example.test", turn="new")
    assert guard.untrusted_addresses() == {"owner-old@example.test"}


def test_post_without_turn_uses_call_identity_not_latest_session(lane):
    gate = start(lane, turn="old")
    lane[1].observe(session_id="session", platform="companion", sender_id="owner-1", turn_id="new")
    gate.write_origin.begin("session", "new")
    gate.post_tool_call(result="late@example.test", session_id="session")
    gate.pre_tool_call("web_search", {}, "session", turn_id="new")
    assert guard.untrusted_addresses() == set()
    assert gate.write_origin.turns[("session", "old")]["seen"] == {"late@example.test"}


@pytest.mark.parametrize("outcome", [{"completed": True}, {"failed": True}, {"interrupted": True}])
def test_native_settlement_discards_turn_memory(lane, outcome):
    gate = start(lane)
    result(lane, "read@example.test")
    observer = SimpleNamespace(sessions=lane[1])
    # A missing task id still clears provenance before transport-specific cleanup.
    capture.Capture.on_native_turn_settled(observer, session_id="session", turn_id="turn", outcome=outcome)
    assert gate.write_origin.turns == {} and guard.call_origin() is None


def test_contexts_do_not_share_current_calls(lane):
    start(lane)
    def other():
        lane[0].pre_tool_call("web_search", {}, "other", turn_id="second")
        assert guard.call_origin()["session_id"] == "other"
    contextvars.Context().run(other)
    assert guard.call_origin()["session_id"] == "session"


@pytest.fixture
def actions(lane):
    gate = start(lane)
    state = SimpleNamespace(posts=[], response={"state": "asked", "ask_id": "ask-1"})
    def post(path, **kwargs):
        assert path == "/v1/mind/owner-actions"
        state.posts.append(kwargs["json"])
        return httpx.Response(200, json=state.response, request=httpx.Request("POST", "https://test.invalid"))
    gate.client.post = post
    def call(args, tool="google_send"):
        gate.pre_tool_call(tool, args, "session", turn_id="turn")
        return guard.request_owner_action(tool, args, "body and secret must never be terms")
    state.call = call
    return state


@pytest.mark.parametrize("state", ["asked", "approved", "denied"])
def test_action_delegates_settlement_to_sidecar(actions, state):
    actions.response = {"state": state, "ask_id": "ask-1"}
    assert actions.call({"to": "friend@example.test"}) == actions.response


def test_changed_args_hashes_and_terms_have_no_body_or_secrets(actions):
    args = {"attendees": [{"email": "Friend@Example.Test"}], "title": "password: synthetic-private-password",
            "body": "PRIVATE-BODY", "headers": {"Authorization": "Bearer private-key"}}
    actions.call(args, "google_calendar_invite")
    actions.call({**args, "body": "DIFFERENT-BODY"}, "google_calendar_invite")
    assert actions.posts[0]["args_hash"] != actions.posts[1]["args_hash"]
    assert actions.posts[0]["summary"]["recipients"] == ["friend@example.test"]
    assert actions.posts[0]["tool"] == "google_calendar_invite"
    terms = json.dumps(actions.posts)
    assert all(secret not in terms for secret in ("PRIVATE-BODY", "DIFFERENT-BODY", "synthetic-private-password", "private-key"))
    assert set(actions.posts[0]) == {"tool", "args_hash", "summary", "reason"}


def test_unavailable_and_wrong_current_call_never_approve(actions, lane):
    args = {"to": "friend@example.test"}
    actions.call(args)
    def failure(*a, **kw):
        raise RuntimeError("offline")
    lane[0].client.post = failure
    assert actions.call(args) == {"state": "denied"}
    assert guard.request_owner_action("different_tool", args, "") == {"state": "denied"}
    assert guard.request_owner_action("google_send", {**args, "body": "changed"}, "") == {"state": "denied"}
