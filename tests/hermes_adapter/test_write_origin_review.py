"""Security review reproductions, using synthetic HTTP and the real hands front."""
import importlib.util
import sys
from types import SimpleNamespace

import httpx
import pytest

from colonymind_hermes import guard, write_origin
from test_hands_front import rig as front_rig, owner_turn, review_turn  # noqa: F401
from test_tool_origins import mcp  # noqa: F401
from test_unknown_standing import lane  # noqa: F401
from test_write_origin import start, result

pytestmark = pytest.mark.requires_sidecar
# Without the sidecar the module is still collected: the marker skips its tests and counts each of them.
if importlib.util.find_spec("colonymind") is not None:
    from colonymind.mind.tool_origins import ToolOrigins


def test_overflow_must_not_make_a_result_recipient_trusted(lane, mcp):
    gate = start(lane, words="attacker@example.test")
    gate.client.resolve_contact = lambda *a, **kw: {"contact_id": "known"}
    result(lane, " ".join(f"filler{i}@example.test" for i in range(256)) + " attacker@example.test")
    assert guard.untrusted_addresses() == {"*"}
    assert len(gate.write_origin.turns[("session", "turn")]["seen"]) == 256
    origins = ToolOrigins()
    def post(path, **kw):
        origins.register(kw["json"])
        return httpx.Response(200, request=httpx.Request("POST", "https://test.invalid"))
    gate.client.post = post
    assert gate.pre_tool_call(mcp("google", "send"), {}, "session", turn_id="turn") is None
    claimed = origins.claim({"tool": "send", "args_hash": write_origin._hash("send", {})})
    assert claimed["untrusted_addresses"] == ["*"]


def test_prior_owner_record_cannot_authorize_after_registration_failure(lane, mcp):
    origins = ToolOrigins()
    tool = mcp("google", "send")
    gate, sessions = lane
    def post(path, **kw):
        if path == "/v1/mind/tool-origins":
            origins.register(kw["json"])
        else:
            assert path == "/v1/mind/guard"
        return httpx.Response(200, json={"allow": True}, request=httpx.Request("POST", "https://test.invalid"))
    gate.client.post = post
    args = {"to": "friend@example.test", "subject": "same call"}
    start(lane, tool=tool, args=args)
    gate.post_tool_call(session_id="session", turn_id="turn")
    gate.write_origin.end("session", "turn")
    sessions.observe(session_id="guest", platform="companion", sender_id="guest-1", turn_id="guest-turn")
    gate.settings.guests = lambda: {"tools": [tool], "executor_platforms": []}
    paths = []
    def unavailable(path, **kw):
        paths.append(path)
        if path == "/v1/mind/tool-origins":
            raise httpx.ConnectError("synthetic outage")
        assert path == "/v1/mind/guard"
        return httpx.Response(200, json={"allow": True}, request=httpx.Request("POST", "https://test.invalid"))
    gate.client.post = unavailable
    verdict = gate.pre_tool_call(tool, args, "guest", turn_id="guest-turn")
    # An unreachable server cannot erase its stale slot: refuse dispatch before MCP can claim it.
    assert verdict and verdict["action"] == "block"
    assert paths == ["/v1/mind/guard", "/v1/mind/tool-origins"]
    assert guard.call_origin() is None
    gate.client.post = post
    assert gate.pre_tool_call(tool, args, "guest", turn_id="guest-turn") is None
    claimed = origins.claim({"tool": "send", "args_hash": write_origin._hash("send", args)})
    assert claimed["origin"] == "other"


@pytest.mark.parametrize("owner_started", [False, True])
def test_actual_hands_notification_requires_owner_start(lane, front_rig, mcp, owner_started):
    words = "Owner asks" if owner_started else "Native completion"
    (owner_turn if owner_started else review_turn)(front_rig)
    assert front_rig.guard.owner_started_turn("owner-session", "turn-1") is owner_started
    gate, sessions = lane
    sessions.settings.owner_handles = lambda: {"sms": ["5550100"]}
    sessions.observe(session_id="owner-session", platform="sms", sender_id="5550100", turn_id="turn-1",
                     user_message=words)
    tool, args, origins = mcp("google", "delete_email"), {"message_id": "synthetic"}, ToolOrigins()
    def post(path, **kw):
        origins.register(kw["json"])
        return httpx.Response(200, request=httpx.Request("POST", "https://test.invalid"))
    gate.client.post = post
    front_verdict = front_rig.guard.pre_tool_call(session_id="owner-session", turn_id="turn-1",
                                                tool_name=tool, args=args, tool_call_id="call")
    assert front_verdict.get("action") != "block"
    assert gate.pre_tool_call(tool, args, "owner-session", turn_id="turn-1") is None
    claimed = origins.claim({"tool": "delete_email", "args_hash": write_origin._hash("delete_email", args)})
    assert claimed["origin"] == ("owner_channel" if owner_started else "other")


@pytest.mark.parametrize("verdict", [False, None, "true", "error", "missing"])
def test_installed_hands_bridge_cannot_fall_back_to_session_sender(lane, monkeypatch, verdict):
    def started(*args):
        if verdict == "error":
            raise RuntimeError("synthetic ledger failure")
        return verdict
    bridge = SimpleNamespace() if verdict == "missing" else SimpleNamespace(owner_started_turn=started)
    monkeypatch.setitem(sys.modules, "colonymind_hands_front", bridge)
    start(lane)
    assert guard.call_origin()["origin"] == "other"


@pytest.mark.parametrize("platform", ["companion", "sms", "internal", "system", "notification", "cron", "relay"])
def test_owner_session_without_triggering_message_is_other(lane, platform):
    gate, sessions = lane
    sessions.settings.owner_handles = lambda: {platform: ["owner-1"]}
    sessions.observe(session_id="session", platform=platform, sender_id="owner-1", turn_id="turn")
    gate.pre_tool_call("web_search", {}, "session", turn_id="turn")
    assert guard.call_origin()["origin"] == "other"


def test_exact_cap_and_repeated_addresses_do_not_overflow(lane):
    gate = start(lane)
    gate.client.resolve_contact = lambda *a, **kw: None
    addresses = {f"filler{i}@example.test" for i in range(256)}
    result(lane, " ".join(sorted(addresses)))
    result(lane, " ".join(sorted(addresses)))
    assert guard.untrusted_addresses() == addresses
    result(lane, "attacker@example.test")
    assert guard.untrusted_addresses() == {"*"}
