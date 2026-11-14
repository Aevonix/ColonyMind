"""The recorded guest-worker probe, through the real guard and isolated device API."""
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest
import yaml

from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import devices as routes, mind as mind_routes
from colonymind_hermes.client import ACTING_SESSION, Settings
from colonymind_hermes.tools import Tools
from onekey import KEY, AUTH
from test_device_capabilities import rig, grant, invocation, pinned_clock  # noqa: F401
from test_hands_authority_sessions import app  # noqa: F401

needs_hermes = pytest.mark.skipif(importlib.util.find_spec("hermes_time") is None,
                                  reason="needs stock Hermes in the test interpreter")
pytestmark = needs_hermes


@pytest.fixture
def boundary(rig, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tests/hermes_adapter"))
    from test_unknown_standing import lane

    gate, sessions = lane.__wrapped__(tmp_path, monkeypatch)
    grant(rig, capability="home", method="write", mode="allow")
    mind_routes.set_mind(rig.mind)
    api = FastAPI()
    api.add_middleware(ApiKeyMiddleware, api_key=KEY)
    api.include_router(routes.router)
    calls = []

    async def backend(method, path, kwargs):
        async with AsyncClient(transport=ASGITransport(app=api), base_url="http://test",
                               headers={**AUTH, "X-ColonyMind-Session": ACTING_SESSION.get()}) as client:
            return await client.request(method, path, **kwargs)

    def post(path, **kwargs):
        assert path == "/v1/mind/guard"
        body = kwargs["json"]
        answer = asyncio.run(rig.mind.guard(tool=body["tool"], args=body["args"], run=body["run"],
                                           session_id=body["session"], task_id=body["task_id"]))
        return SimpleNamespace(status_code=200, is_success=True, json=lambda: answer)

    def request(method, path, **kwargs):
        calls.append((method, path, ACTING_SESSION.get()))
        return asyncio.run(backend(method, path, {k: v for k, v in kwargs.items() if k != "timeout"}))

    gate.client.post = post
    gate.client.request = request
    gate.client.has_mind_routes = lambda: True
    yield SimpleNamespace(gate=gate, sessions=sessions, calls=calls,
                          tools=Tools(gate.client, sessions, gate.settings))
    mind_routes.set_mind(None)


@pytest.fixture(params=[(platform, binding) for platform in ("telegram", "whatsapp", "sms", "rcs", "companion")
                        for binding in ("handle", "contact")])
def owner_binding(boundary, tmp_path, monkeypatch, request):
    from gateway.session_context import clear_session_vars, set_session_vars
    from colonymind_hermes import capture, tools

    platform, binding = request.param
    sender = "+15550000011" if platform in {"sms", "rcs"} else "owner-1"
    identity = tmp_path / "identity.yaml"
    identity.write_text(yaml.safe_dump({"owner": {"handles": {platform: [sender]} if binding == "handle" else {}}}))
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump({"owner": {"contact_id": "owner-1"}}))
    settings = Settings("http://test", tmp_path / "api.key", KEY, tmp_path,
                        tmp_path / "hermes", tmp_path / "outbox.sqlite3")
    contacts = {sender: "owner-1" if binding == "contact" else "guest-1"}
    boundary.sessions.settings = boundary.tools.settings = settings
    boundary.sessions.client.resolve_contact = lambda platform, sender, **kw: {
        "contact_id": contacts.get(sender, "guest-1")}
    monkeypatch.setattr(capture, "session_env", tools.session_env)
    tokens = set_session_vars(platform=platform, user_id=sender, chat_id=sender, chat_type="dm",
                              session_id="owner-session")

    def revoke():
        identity.write_text("owner:\n  handles: {}\n")
        contacts[sender] = "guest-1"

    try:
        yield SimpleNamespace(platform=platform, sender=sender, revoke=revoke)
    finally:
        clear_session_vars(tokens)


def test_owner_dm_channels_reach_device_api(boundary, rig, owner_binding):
    binding = owner_binding
    boundary.sessions.observe(session_id="owner-session", platform=binding.platform, sender_id=binding.sender)
    for args in ({"operation": "request", "invocation": invocation(
            rig, capability="home", command="home.execute").model_dump(mode="json")},
            {"operation": "inventory"}, {"operation": "status", "id": "inv-study"}):
        result = json.loads(boundary.tools.device_tool(args, session_id="owner-session"))
        assert "devices" in result if args["operation"] == "inventory" else result["state"] == "ready"
    assert boundary.calls == [("POST", routes.MODEL, "owner-session"),
                              ("GET", "/v1/mind/device-capabilities", "owner-session"),
                              ("GET", routes.MODEL + "/inv-study", "owner-session")]
    assert ACTING_SESSION.get() == ""


def test_revoked_warm_owner_refused_on_next_turn(boundary, rig, owner_binding):
    binding, sessions = owner_binding, boundary.sessions
    sessions.observe(session_id="owner-session", platform=binding.platform, sender_id=binding.sender,
                     turn_id="before")
    assert sessions.standing("owner-session")[0] == "owner"
    assert "devices" in json.loads(boundary.tools.device_tool({"operation": "inventory"}, session_id="owner-session"))
    binding.revoke()
    sessions.observe(session_id="owner-session", platform=binding.platform, sender_id=binding.sender,
                     turn_id="after")
    assert sessions.sender_is_owner(binding.platform, binding.sender) is False
    # The shared standing cache is deliberately retained; the device gate must bypass it.
    if binding.platform not in {"sms", "rcs"}:
        assert sessions.standing("owner-session")[0] == "owner"
    boundary.calls.clear()
    for operation in ("inventory", "status", "request"):
        result = json.loads(boundary.tools.device_tool({"operation": operation, "id": "inv-study", "invocation":
            invocation(rig, capability="home", command="home.execute").model_dump(mode="json")},
            session_id="owner-session"))
        assert result == {"unavailable": True, "retry": False,
                          "reason": "device requests need the owner's own direct conversation"}
    assert boundary.calls == [] and rig.service.invocations("phone-study") == []


@pytest.mark.parametrize("kind", ["kanban", "cron", "delegated", "child-cli", "cli"])
@pytest.mark.parametrize("restart", [False, True])
def test_senderless_owner_tasks_refused_even_with_owner_binding(boundary, rig, monkeypatch, kind, restart):
    from colonymind_hermes import tools
    from colonymind_hermes.capture import SessionMap

    gate, sessions = boundary.gate, boundary.sessions
    sessions.observe(session_id="parent", platform="telegram", sender_id="owner-1", turn_id="creating-turn")
    if kind in {"kanban", "cron"}:
        tool = "kanban_create" if kind == "kanban" else "cronjob_manage"
        args = {} if kind == "kanban" else {"action": "create"}
        assert gate.pre_tool_call(tool, args, "parent", tool_call_id="creation") is None
        result = {"task_id": "task"} if kind == "kanban" else {"success": True, "job_id": "job"}
        gate.post_tool_call(tool, args, result, session_id="parent", tool_call_id="creation")
        if kind == "kanban":
            monkeypatch.setenv("HERMES_KANBAN_TASK", "task")
            session, platform, task = "worker", "cli", ""
        else:
            session, platform, task = "cron_job_20300101_120000", "cron", "cron:job:run"
        sessions.observe(session_id=session, platform=platform, task_id=task)
    elif kind == "cli":
        session = "cli"
        sessions.observe(session_id=session, platform="cli")
    else:
        session = "child"
        gate.subagent_start(child_session_id=session, parent_session_id="parent")
        sessions.observe(session_id=session, platform="cli" if kind == "child-cli" else "subagent",
                         parent_session_id="parent")
    assert sessions.standing(session)[0] == "owner"
    if restart:
        boundary.tools.sessions = SessionMap(sessions.settings, sessions.client)
    # A child can inherit the parent's gateway context as well as its recorded authority.
    monkeypatch.setattr(tools, "session_env", lambda: ("telegram", "owner-1", "owner-1", "dm"))
    for operation in ("inventory", "status", "request"):
        result = json.loads(boundary.tools.device_tool({"operation": operation, "id": "inv-study", "invocation":
            invocation(rig, capability="home", command="home.execute").model_dump(mode="json")}, session_id=session))
        assert result == {"unavailable": True, "retry": False,
                          "reason": "device requests need the owner's own direct conversation"}
    assert boundary.calls == [] and rig.service.invocations("phone-study") == []


@pytest.mark.parametrize("current_platform,current_sender", [("telegram", "guest-1"), ("whatsapp", "owner-1")])
def test_stale_owner_session_cannot_override_current_sender(boundary, rig, monkeypatch,
                                                          current_platform, current_sender):
    from colonymind_hermes import tools

    boundary.sessions.observe(session_id="group", platform="telegram", sender_id="owner-1")
    assert boundary.sessions.standing("group")[0] == "owner"
    monkeypatch.setattr(tools, "session_env", lambda: (current_platform, current_sender, "group", "group"))
    result = json.loads(boundary.tools.device_tool({"operation": "inventory"}, session_id="group"))
    assert result["unavailable"] is True and result["retry"] is False
    assert boundary.calls == [] and rig.service.invocations("phone-study") == []


@pytest.mark.parametrize("kind,origin", [
    ("kanban", "guest"), ("kanban", "unknown"), ("kanban", "mind"), ("kanban", "system"),
    ("delegated", "guest"), ("child-cli", "guest"), ("child-cli", "unknown"), ("direct", "guest"),
])
@pytest.mark.parametrize("operation", ["inventory", "status", "request"])
def test_non_owner_origin_never_reaches_device_api(boundary, rig, monkeypatch, kind, origin, operation):
    gate, sessions = boundary.gate, boundary.sessions
    session = "guest-worker"
    if kind == "kanban":
        monkeypatch.setenv("HERMES_KANBAN_TASK", "guest-owned-task")
        sessions.standings.put("kanban", "guest-owned-task", origin, "telegram", "guest-parent")
        sessions.observe(session_id=session, platform="cli")
    elif kind == "direct":
        sessions.observe(session_id=session, platform="telegram", sender_id="guest-1")
    else:
        sessions.inherit(session, "guest-parent", standing=origin, platform="telegram")
        sessions.observe(session_id=session, platform="cli" if kind == "child-cli" else "subagent",
                         parent_session_id="guest-parent")
    args = {"operation": operation, "id": "inv-study", "invocation": invocation(
        rig, capability="home", command="home.execute").model_dump(mode="json")}
    verdict = gate.pre_tool_call("colonymind_device_invoke", args, session)
    if kind == "kanban" and origin == "guest" and operation == "request":
        # The original probe passed the real guard under a standing device allow.
        assert (verdict or {}).get("action") != "block"
    # Also exercise direct dispatch: a handler cannot depend on the pre-call hook refusing it.
    result = json.loads(boundary.tools.device_tool(args, session_id=session))
    assert result["unavailable"] is True and result["retry"] is False
    assert boundary.calls == []
    assert rig.service.invocations("phone-study") == []


def test_owner_session_reaches_real_device_api_with_its_session(boundary, rig, monkeypatch):
    from colonymind_hermes import tools
    monkeypatch.setattr(tools, "session_env", lambda: ("telegram", "owner-1", "owner-1", "dm"))
    boundary.sessions.observe(session_id="owner-session", platform="telegram", sender_id="owner-1")
    args = {"operation": "request", "invocation": invocation(
        rig, capability="home", command="home.execute").model_dump(mode="json")}
    verdict = boundary.gate.pre_tool_call("colonymind_device_invoke", args, "owner-session")
    assert (verdict or {}).get("action") != "block"
    result = json.loads(boundary.tools.device_tool(args, session_id="owner-session"))
    assert result["state"] == "ready"
    assert boundary.calls == [("POST", routes.MODEL, "owner-session")]
    assert ACTING_SESSION.get() == ""


@pytest.mark.parametrize("method,path", [
    ("POST", routes.MODEL), ("GET", "/v1/mind/device-capabilities"), ("GET", routes.MODEL + "/inv-study"),
])
async def test_device_routes_refuse_registered_child_before_validation(app, method, path):
    app.add_middleware(ApiKeyMiddleware, api_key=KEY)
    app.include_router(routes.router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", headers=AUTH) as client:
        result = await client.request(method, path, json={}, headers={"X-ColonyMind-Session": "child-session"})
    assert result.status_code == 403
    assert result.json()["detail"]["code"] == "hand_session_refused"


@pytest.mark.parametrize("live", [("", "", "", ""), ("telegram", "", "", "dm"), ("", "owner-1", "owner-1", "dm")])
def test_cached_owner_session_with_empty_live_sender_is_refused(boundary, rig, monkeypatch, live):
    from colonymind_hermes import tools

    boundary.sessions.observe(session_id="owner-session", platform="telegram", sender_id="owner-1",
                              turn_id="owner-turn")
    # The cached SessionInfo names the owner, but this turn carries no live sender binding.
    monkeypatch.setattr(tools, "session_env", lambda: live)
    for operation in ("inventory", "status", "request"):
        result = json.loads(boundary.tools.device_tool({"operation": operation, "id": "inv-study", "invocation":
                                                        invocation(rig, capability="home", command="home.execute").model_dump(mode="json")},
                                                       session_id="owner-session"))
        assert result == {"unavailable": True, "retry": False,
                          "reason": "device requests need the owner's own direct conversation"}
    assert boundary.calls == [] and rig.service.invocations("phone-study") == []
