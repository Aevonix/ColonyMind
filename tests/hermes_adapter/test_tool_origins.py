"""Gateway registration uses Hermes' real MCP naming and handler binding, without transport calls."""
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from colonymind_hermes import client, guard
from test_write_origin import lane, start, result  # noqa: F401


@pytest.fixture
def mcp(monkeypatch):
    from tools.registry import registry
    from tools.mcp_tool_registration import _tool_candidates
    entries = {}
    def register(server, bare):
        tool = SimpleNamespace(name=bare, description="synthetic test", inputSchema={"type": "object"})
        candidate = _tool_candidates(server, [tool], lambda _: True, 1)[0]
        entries[candidate.registry_name] = SimpleNamespace(toolset="mcp-" + server, handler=candidate.handler)
        return candidate.registry_name
    monkeypatch.setattr(registry, "get_entry", entries.get)
    return register


@pytest.fixture
def posts(lane):
    calls = []
    def post(path, **kw):
        calls.append((path, kw))
        return httpx.Response(200, json={"registered": True}, request=httpx.Request("POST", "https://test.invalid"))
    lane[0].client.post = post
    return calls


@pytest.mark.parametrize("server,bare", [("google", "google_send"), ("google", "calendar__invite"),
    ("google", "send-email.v2"), ("google", "long_name_" * 12), ("google_team-west", "invite__guest")])
def test_original_bare_name_and_canonical_hash(lane, mcp, posts, server, bare):
    lane[0].settings.tool_origin_servers = (server,)
    registered = mcp(server, bare)
    args = {"z": [2, {"b": True, "a": "é"}], "a": "日本語", "body": "full body"}
    start(lane, tool=registered, args=args)
    assert len(posts) == 1
    path, kw = posts[0]
    assert path == "/v1/mind/tool-origins" and kw["timeout"] == 0.5
    payload = kw["json"]
    digest = hashlib.sha256(json.dumps({"tool": bare, "args": args}, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    assert payload == {"tool": bare, "args_hash": digest, "origin": "owner_channel", "channel": "companion",
                       "session_id": "session", "turn_id": "turn", "untrusted_addresses": []}
    assert registered.startswith("mcp__") and len(registered) <= 64


def test_only_configured_servers_and_actual_mcp_registrations(lane, mcp, posts):
    start(lane, tool=mcp("other", "send"))
    start(lane, tool="mcp__google__forged")
    start(lane, tool="google_send")
    assert posts == []
    start(lane, tool=mcp("google", "send"))
    assert len(posts) == 1
    lane[0].settings.tool_origin_servers = ()
    start(lane, tool=mcp("google", "send"))
    assert len(posts) == 1
    lane[0].settings.tool_origin_servers = ("other",)
    start(lane, tool=mcp("other", "send"))
    assert len(posts) == 2


def test_registration_carries_turn_addresses_and_precedes_execution(lane, mcp, posts):
    gate = start(lane, words="owner@example.test")
    gate.client.resolve_contact = lambda *a, **kw: None
    result(lane, "owner@example.test read@example.test")
    registered = mcp("google", "send")
    assert gate.pre_tool_call(registered, {}, "session", turn_id="turn") is None
    assert posts[-1][1]["json"]["untrusted_addresses"] == ["read@example.test"]
    # Execution starts only after the pre-hook has returned and the POST has completed.
    assert len(posts) == 1


@pytest.mark.parametrize("failure", ["timeout", "offline", "status"])
def test_sidecar_failure_logs_once_and_blocks_dispatch(lane, mcp, caplog, failure):
    tool = mcp("google", "send")
    gate = start(lane)
    def post(path, **kw):
        assert kw["timeout"] == 0.5
        if failure == "status":
            return httpx.Response(503, request=httpx.Request("POST", "https://test.invalid"))
        raise httpx.ReadTimeout("synthetic") if failure == "timeout" else httpx.ConnectError("synthetic")
    gate.client.post = post
    assert gate.pre_tool_call(tool, {}, "session", turn_id="turn")["action"] == "block"
    assert guard.call_origin() is None
    logs = [record.message for record in caplog.records if "tool-origin registration failed" in record.message]
    assert len(logs) == 1 and "\n" not in logs[0]


def test_server_configuration_defaults_and_override(tmp_path):
    base = {"key_file": str(tmp_path / "api.key")}
    for configured, expected in [({}, ("google",)), ({"tool_origin_servers": []}, ()),
                                ({"tool_origin_servers": ["work_google", "calendar"]}, ("work_google", "calendar")),
                                ({"tool_origin_servers": None}, ()), ({"tool_origin_servers": "google"}, ())]:
        settings = client.load_settings({"plugins": {"colonymind": base | configured}})
        assert settings.tool_origin_servers == expected


def test_exact_hermes_registration_rule(mcp):
    assert mcp("google", "send") == "mcp__google__send"
    assert mcp("google_team-west", "send-email.v2") == "mcp__google_team_west__send_email_v2"
    bare = "long_name_" * 12
    full = "mcp__google__" + bare
    assert mcp("google", bare) == full[:55] + "_" + hashlib.sha256(full.encode()).hexdigest()[:8]
