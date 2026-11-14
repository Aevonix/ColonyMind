"""Real isolated Mind stores and authenticated routes; no native adapters or live state."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import asyncio
import hashlib
import json
import re
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
import pytest

from colonymind import devices
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import devices as routes, mind as mind_routes
from colonymind.mind.authority import Policy, Authority
from colonymind.mind import permit_asks
from test_hands_permit_answers import TABLE, HUB, HUB_AUTHORITY, answer_body
from test_mind_loop import Fixture, pinned_clock  # noqa: F401
from owner_answer import owner_answers
from onekey import KEY, AUTH


REPORT = {
    "display_name": "Study iPhone", "platform": "ios", "app_version": "1.0", "state": "online",
    "capabilities": [
        {"capability": "health", "os_grants": {"read": "granted", "write": "unknown"}, "commands": ["health.read"]},
        {"capability": "alarms", "os_grants": {"read": "granted", "write": "granted"},
         "commands": ["alarm.list", "alarm.set"]},
        {"capability": "home", "os_grants": {"read": "granted", "write": "granted"},
         "commands": ["home.execute"]},
    ],
}


@pytest.fixture
def rig(tmp_path):
    fixture = Fixture(tmp_path)
    service = devices.Devices(fixture.mind)
    service.report("phone-study", devices.DeviceReport.model_validate(REPORT))
    yield SimpleNamespace(fixture=fixture, mind=fixture.mind, service=service)
    fixture.store.close()


def invocation(rig, ident="inv-study", capability="alarms", command="alarm.set", **extra):
    return devices.Invoke.model_validate({
        "id": ident, "device_id": "phone-study", "capability": capability, "command": command,
        "parameters": {"hour": 9}, "deadline": rig.mind.clock() + timedelta(minutes=5), **extra})


def grant(rig, mode="allow", method="write", capability="alarms"):
    return rig.service.grant("phone-study", capability, devices.GrantEdit(
        expected_revision=rig.service.device("phone-study")["revision"], method=method, mode=mode))


def test_report_persists_and_keeps_os_and_agent_grants_independent(rig):
    before = rig.service.device("phone-study")
    assert before["capabilities"][0]["authorization"] == {"read": "ask", "write": "ask"}
    same = rig.service.report("phone-study", devices.DeviceReport.model_validate(REPORT))
    assert same["revision"] == before["revision"]
    grant(rig)
    report = deepcopy(REPORT)
    report["capabilities"][1]["os_grants"]["write"] = "denied"
    rig.service.report("phone-study", devices.DeviceReport.model_validate(report))
    reopened = devices.Devices(rig.fixture.build())
    assert reopened.device("phone-study")["capabilities"][1]["authorization"]["write"] == "allow"
    assert reopened.device("phone-study")["capabilities"][1]["os_grants"]["write"] == "denied"
    assert len(reopened.list_devices()) == 1


@pytest.mark.parametrize("state", ["unknown", "not_determined", "limited", "denied", "restricted", "unsupported"])
def test_missing_os_permission_never_creates_an_approval(rig, state):
    report = deepcopy(REPORT)
    report["capabilities"][1]["os_grants"]["write"] = state
    rig.service.report("phone-study", devices.DeviceReport.model_validate(report))
    value = rig.service.request(invocation(rig))
    assert value["state"] == "os_permission_required" and value["approval"] is None
    with pytest.raises(HTTPException):
        rig.service.claim("phone-study", value["id"])


@pytest.mark.parametrize("capability,command,method", [
    ("health", "health.read", "read"), ("alarms", "alarm.set", "write"), ("home", "home.execute", "write")])
def test_initial_capabilities_obey_their_exact_read_or_write_grant(rig, capability, command, method):
    grant(rig, capability=capability, method=method)
    value = rig.service.request(invocation(rig, capability=capability, command=command))
    assert value["state"] == "ready" and value["approval"] is None
    assert rig.service.claim("phone-study", value["id"])["state"] == "dispatched"
    result = rig.service.result("phone-study", value["id"], devices.Result(ok=True, payload={"changed": True}))
    assert result["state"] == "succeeded"


@pytest.mark.asyncio
async def test_ask_uses_the_existing_explicit_owner_answer_and_spends_one_exact_permit(rig):
    body = invocation(rig)
    value = rig.service.request(body)
    assert value["state"] == "awaiting_approval" and value["result"] is None
    ask = rig.mind.store.get(value["approval"])
    assert permit_asks.is_permit_ask(ask)
    assert ask.kind == "note" and ask.expires_at == body.deadline
    assert rig.service.request(body)["approval"] == ask.id
    with pytest.raises(HTTPException):
        rig.service.result("phone-study", body.id, devices.Result(ok=True))
    code, answer = await rig.mind.answer_explicit(ask.id, choice="yes", channel="companion",
        contact_id=rig.mind.owner_id, authority=HUB_AUTHORITY, key="device-answer",
        request_hash=permit_asks.request_hash(ask.id, answer_body()))
    assert code == 200 and answer["status"] == "approved"
    assert rig.service.invocation(body.id)["state"] == "ready"
    history = rig.mind.store.get_history(ask.id, limit=200)
    assert not any(item.action == "permit_used" for item in history)
    rig.service.claim("phone-study", body.id)
    with pytest.raises(HTTPException):
        rig.service.claim("phone-study", body.id)
    assert rig.service.request(body)["state"] == "dispatched"
    assert len([item for item in rig.mind.store.get_history(ask.id, limit=200)
                if item.action == "permit_used"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("capability,command", [
    ("health", "health.read"), ("alarms", "alarm.set"), ("home", "home.execute")])
async def test_parameter_values_never_reach_approval_or_autobiography(rig, caplog, capability, command):
    marker = "HEALTH_PAYLOAD_REVIEW_CANARY"
    parameters = {"measurement": {"samples": [marker], "units": "PRIVATE_UNITS_CANARY"}}
    value = rig.service.request(invocation(rig, capability=capability, command=command, parameters=parameters))
    assert value["state"] == "awaiting_approval"
    ask = rig.mind.store.get(value["approval"])
    terms = permit_asks.terms(ask, rig.mind.store)
    args = terms["call"]["arguments"]
    assert {key: args[key] for key in ("id", "device_id", "capability", "command")} == {
        key: value[key] for key in ("id", "device_id", "capability", "command")}
    digest = hashlib.sha256(devices.dumps(parameters).encode()).hexdigest()
    assert args["parameters_digest"] == digest
    assert set(args) == {"id", "device_id", "capability", "command", "parameters_digest"}
    assert digest in terms["reason"]
    await owner_answers(rig.mind, ask.id)
    assert rig.service.claim("phone-study", value["id"])["parameters"] == parameters
    rig.service.result("phone-study", value["id"], devices.Result(ok=True))
    with rig.fixture.ledger._connect() as conn:
        sources = conn.execute("SELECT messages_json FROM turn_sources WHERE session_id='mind'").fetchall()
    assert sources
    records = json.dumps([terms, rig.mind.store.get(ask.id).context,
                          rig.mind.store.get(ask.id).decision_reason,
                          [row["messages_json"] for row in sources]], default=str) + caplog.text
    for private_text in (marker, "PRIVATE_UNITS_CANARY", "measurement", "samples"):
        assert private_text not in records


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [
    {"parameters": {"measurement": "different"}}, {"id": "inv-other"}, {"device_id": "phone-other"},
    {"capability": "home"}, {"command": "home.read"}, {"device_revision": 100}, {"deadline": None}])
async def test_digest_binding_preserves_every_exact_invocation_term(rig, changed):
    value = rig.service.request(invocation(rig, capability="health", command="health.read"))
    await owner_answers(rig.mind, value["approval"])
    assert rig.service._authorize(value)["allow"]
    if "deadline" in changed:
        changed = {"deadline": (rig.mind.clock() + timedelta(minutes=4)).isoformat()}
    assert not rig.service._authorize({**value, **changed})["allow"]
    assert rig.service.claim("phone-study", value["id"])["state"] == "dispatched"


@pytest.mark.asyncio
@pytest.mark.parametrize("read_ready", [False, True])
async def test_expired_effect_permit_cannot_be_claimed(rig, read_ready):
    rig.mind.authority = Authority(replace(rig.mind.authority.policy, permit_hours=1 / 3600),
                                   rig.mind.store, owner_id=rig.mind.owner_id, clock=rig.mind.clock)
    value = rig.service.request(invocation(rig))
    await owner_answers(rig.mind, value["approval"])
    if read_ready:
        for _ in range(2):
            assert rig.service.invocation(value["id"])["state"] == "ready"
        # A fresh service must not turn a saved ready flag into execution authority.
        rig.service = devices.Devices(rig.mind)
    rig.fixture.now += timedelta(seconds=2)
    with pytest.raises(HTTPException) as error:
        rig.service.claim("phone-study", value["id"])
    assert error.value.status_code == 409
    row = rig.service.invocation(value["id"])
    assert row["state"] != "dispatched" and "dispatched_at" not in row
    assert not any(item.action == "permit_used" for item in rig.mind.store.get_history(value["approval"], limit=200))


@pytest.mark.asyncio
async def test_valid_permit_is_consumed_at_claim_and_cannot_be_replayed(rig):
    rig.mind.authority = Authority(replace(rig.mind.authority.policy, permit_hours=1 / 3600),
                                   rig.mind.store, owner_id=rig.mind.owner_id, clock=rig.mind.clock)
    value = rig.service.request(invocation(rig))
    await owner_answers(rig.mind, value["approval"])
    for _ in range(2):
        assert rig.service.invocations("phone-study")[0]["state"] == "ready"
    rig.fixture.now += timedelta(milliseconds=500)
    assert rig.service.claim("phone-study", value["id"])["state"] == "dispatched"
    used = [item for item in rig.mind.store.get_history(value["approval"], limit=200)
            if item.action == "permit_used"]
    assert len(used) == 1 and used[0].timestamp == rig.mind.clock()
    with pytest.raises(HTTPException) as error:
        rig.service.claim("phone-study", value["id"])
    assert error.value.status_code == 409
    assert len([item for item in rig.mind.store.get_history(value["approval"], limit=200)
                if item.action == "permit_used"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("invalidation", ["undo", "spent", "off", "deny_tool", "deny_parameter"])
async def test_ready_invocation_revalidates_execution_authority_at_claim(rig, invalidation):
    value = rig.service.request(invocation(rig))
    await owner_answers(rig.mind, value["approval"])
    assert rig.service.invocation(value["id"])["state"] == "ready"
    ask = rig.mind.store.get(value["approval"])
    if invalidation == "undo":
        await rig.mind._undo_ask(ask, by="owner", now=rig.mind.clock(), answer="undo")
    elif invalidation == "spent":
        rig.mind.store.transition(ask.id, ask.status, action="permit_used", at=rig.mind.clock())
    else:
        policy = rig.mind.authority.policy
        if invalidation == "off":
            policy = replace(policy, level="off")
        elif invalidation == "deny_tool":
            policy = replace(policy, deny_tools=frozenset({devices.TOOL}))
        else:
            policy = replace(policy, deny_text=(re.compile(r'"hour": 9'),))
        rig.mind.authority = Authority(policy, rig.mind.store, owner_id=rig.mind.owner_id, clock=rig.mind.clock)
    with pytest.raises(HTTPException) as error:
        rig.service.claim("phone-study", value["id"])
    assert error.value.status_code == 409
    assert "dispatched_at" not in rig.service.invocation(value["id"])


@pytest.mark.asyncio
async def test_existing_owner_message_answer_also_releases_invocation(rig):
    value = rig.service.request(invocation(rig))
    await owner_answers(rig.mind, value["approval"])
    assert rig.service.claim("phone-study", value["id"])["state"] == "dispatched"


@pytest.mark.asyncio
async def test_another_invocation_cannot_reuse_a_recent_yes(rig):
    first = rig.service.request(invocation(rig))
    await owner_answers(rig.mind, first["approval"])
    assert rig.service.invocation(first["id"])["state"] == "ready"
    second = rig.service.request(invocation(rig, "inv-second"))
    assert second["state"] == "awaiting_approval" and second["approval"] != first["approval"]


@pytest.mark.asyncio
async def test_denial_and_an_operator_yes_do_not_authorize(rig):
    value = rig.service.request(invocation(rig))
    rig.mind.store.transition(value["approval"], "approved", action="queued", at=rig.mind.clock(),
                              details={"by": "console", "code": "claimed"})
    assert rig.service.invocation(value["id"])["state"] == "awaiting_approval"
    second = rig.service.request(invocation(rig, "inv-other"))
    await owner_answers(rig.mind, second["approval"], "no")
    assert rig.service.invocation(second["id"])["state"] == "denied"


def test_deny_is_separate_per_device_capability_and_method(rig):
    grant(rig, "deny")
    assert rig.service.request(invocation(rig))["state"] == "denied"
    other = rig.service.request(invocation(rig, "inv-read", command="alarm.list"))
    assert other["state"] == "awaiting_approval"
    rig.service.report("phone-other", devices.DeviceReport.model_validate(REPORT))
    assert rig.service.request(invocation(rig, "inv-phone", device_id="phone-other"))["state"] == "awaiting_approval"


@pytest.mark.parametrize("mutation", ["revoke", "os", "disconnect", "remove"])
def test_changes_invalidate_even_ready_invocations(rig, mutation):
    grant(rig)
    value = rig.service.request(invocation(rig))
    if mutation == "revoke":
        grant(rig, "deny")
    else:
        report = deepcopy(REPORT)
        if mutation == "os":
            report["capabilities"][1]["os_grants"]["write"] = "denied"
        elif mutation == "disconnect":
            report["state"] = "disconnected"
        else:
            report["capabilities"] = []
        rig.service.report("phone-study", devices.DeviceReport.model_validate(report))
    assert rig.service.invocation(value["id"])["state"] == "cancelled"
    with pytest.raises(HTTPException):
        rig.service.claim("phone-study", value["id"])


def test_stale_grant_edit_and_changed_invocation_arguments_are_rejected(rig):
    prior = rig.service.device("phone-study")["revision"]
    grant(rig)
    with pytest.raises(HTTPException) as error:
        rig.service.grant("phone-study", "alarms", devices.GrantEdit(expected_revision=prior, method="write", mode="deny"))
    assert error.value.status_code == 409
    rig.service.request(invocation(rig))
    with pytest.raises(HTTPException) as error:
        rig.service.request(invocation(rig, parameters={"hour": 10}))
    assert error.value.status_code == 409


def test_deadline_result_failure_and_ambiguous_late_completion(rig):
    grant(rig)
    first = rig.service.request(invocation(rig))
    rig.service.claim("phone-study", first["id"])
    rig.mind.clock = lambda: rig.fixture.now + timedelta(minutes=6)
    assert rig.service.invocation(first["id"])["state"] == "timed_out"
    result = devices.Result(ok=False, error="os_permission_required", card={"kind": "open_settings"})
    value = rig.service.result("phone-study", first["id"], result)
    assert value["state"] == "timed_out" and value["late"] and value["result"]["card"]
    assert rig.service.result("phone-study", first["id"], result) == value
    with pytest.raises(HTTPException):
        rig.service.result("phone-study", first["id"], devices.Result(ok=True))


def test_unclaimed_expiry_never_dispatches(rig):
    value = rig.service.request(invocation(rig))
    rig.mind.clock = lambda: rig.fixture.now + timedelta(minutes=6)
    assert rig.service.invocation(value["id"])["state"] == "timed_out"
    assert rig.mind.store.get(value["approval"]).status == "expired"
    with pytest.raises(HTTPException):
        rig.service.claim("phone-study", value["id"])


@pytest.mark.parametrize("seconds", [-1, 0, 3601])
def test_invalid_deadlines(rig, seconds):
    with pytest.raises(HTTPException) as error:
        rig.service.request(invocation(rig, deadline=rig.mind.clock() + timedelta(seconds=seconds)))
    assert error.value.status_code == 422


def test_standing_allow_does_not_lift_existing_off_or_deny_rules(rig):
    grant(rig)
    rig.mind.authority = Authority(Policy(level="off"), rig.mind.store, owner_id=rig.mind.owner_id)
    assert rig.service.request(invocation(rig))["state"] == "denied"


def test_standing_allow_obeys_existing_tool_deny_rule(rig):
    grant(rig)
    rig.mind.authority = Authority(Policy(level="trusted", deny_tools=frozenset({devices.TOOL})),
                                   rig.mind.store, owner_id=rig.mind.owner_id)
    assert rig.service.request(invocation(rig))["state"] == "denied"


def test_device_uuid_identity_is_reused_without_a_second_registration_system(rig):
    ident = "10000000-0000-4000-8000-000000000000"
    rig.service.report(ident, devices.DeviceReport.model_validate(REPORT))
    value = rig.service.request(invocation(rig, device_id=ident))
    assert value["device_id"] == ident


def test_native_failure_is_separate_from_approval(rig):
    grant(rig)
    value = rig.service.request(invocation(rig))
    rig.service.claim("phone-study", value["id"])
    result = rig.service.result("phone-study", value["id"], devices.Result(ok=False, error="os_permission_required"))
    assert result["state"] == "failed" and result["result"]["error"] == "os_permission_required"
    assert result["approval"] is None


def test_model_tool_can_only_read_and_request_through_the_backend(rig, monkeypatch):
    from colonymind_hermes import tools as hermes_tools
    from colonymind_hermes.tools import Tools, DEVICE_SCHEMA
    monkeypatch.setattr(hermes_tools, "session_env", lambda: ("companion", "owner", "owner", "dm"))
    calls = []

    def request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        return SimpleNamespace(status_code=200, is_success=True, text='{"state":"awaiting_approval"}')

    tool = Tools(SimpleNamespace(has_mind_routes=lambda: True, request=request),
                 SimpleNamespace(get=lambda session: SimpleNamespace(platform="companion", sender_id="owner"),
                                 sender_is_owner=lambda platform, sender, **kw: True),
                 SimpleNamespace(mind=lambda: {}))
    assert DEVICE_SCHEMA["name"] == devices.TOOL
    assert "awaiting_approval" in tool.device_tool({"operation": "request", "invocation": {"id": "inv-tool"}})
    assert calls[0][0:2] == ("POST", routes.MODEL)
    for operation in ("grant", "answer", "claim", "result"):
        tool.device_tool({"operation": operation})
    assert len(calls) == 1
    tool.sessions.sender_is_owner = lambda platform, sender, **kw: False
    tool.device_tool({"operation": "inventory"})
    assert len(calls) == 1


@pytest.fixture
async def api(rig):
    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware, api_key=KEY, credentials=TABLE)
    app.include_router(routes.router)
    app.include_router(mind_routes.router)
    mind_routes.set_mind(rig.mind)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", headers=AUTH) as client:
        yield client
    mind_routes.set_mind(None)


@pytest.mark.asyncio
async def test_app_routes_and_separate_answer_credential(api, rig):
    from test_hands_permit_answers import LANE_SECRET
    app_headers = {"Authorization": "Bearer " + LANE_SECRET}
    inventory = await api.get(routes.APP, headers=app_headers)
    assert inventory.status_code == 200 and inventory.json()["devices"][0]["id"] == "phone-study"
    assert (await api.get(routes.APP + "/phone-study", headers=app_headers)).status_code == 200
    assert (await api.get(routes.APP + "/phone-study/capabilities/health", headers=app_headers)).json()["os_grants"]["read"] == "granted"
    report = await api.put(routes.APP + "/phone-study", json=REPORT, headers=app_headers)
    assert report.status_code == 200
    value = (await api.post(routes.MODEL, json=invocation(rig).model_dump(mode="json"))).json()
    answered = await api.post(f"/v1/mind/permit-asks/{value['approval']}/answer", json=answer_body(),
                             headers={**HUB, "Idempotency-Key": "device-api-answer"})
    assert answered.status_code == 200
    pending = await api.get(routes.APP + "/phone-study/invocations", headers=app_headers)
    assert pending.json()["invocations"][0]["state"] == "ready"
    path = routes.APP + "/phone-study/invocations/inv-study"
    replies = await asyncio.gather(*[api.post(path + "/claim", headers=app_headers) for _ in range(2)])
    assert sorted(reply.status_code for reply in replies) == [200, 409]
    assert (await api.post(path + "/result", json={"ok": True}, headers=app_headers)).json()["state"] == "succeeded"


@pytest.mark.asyncio
async def test_agent_guest_rcs_and_answer_credentials_cannot_write_app_reports(api, rig):
    from test_hands_permit_answers import GUEST_SECRET, RCS_ONLY_SECRET
    for headers in (AUTH, HUB, {"Authorization": "Bearer " + GUEST_SECRET}, {"Authorization": "Bearer " + RCS_ONLY_SECRET}):
        assert (await api.put(routes.APP + "/phone-study", json=REPORT, headers=headers)).status_code == 403
        assert (await api.get(routes.APP, headers=headers)).status_code == 403
    assert (await api.post(routes.MODEL, json=invocation(rig).model_dump(mode="json"),
                           headers={"Authorization": "Bearer " + GUEST_SECRET})).status_code == 403


@pytest.mark.asyncio
async def test_sensitive_payload_never_echoed_or_persisted(api, rig):
    value = invocation(rig).model_dump(mode="json")
    sensitive = "+" + "1" + "2" * 10
    value["parameters"] = {"phone": sensitive}
    reply = await api.post(routes.MODEL, json=value)
    assert reply.status_code == 422 and sensitive not in reply.text
    assert rig.mind.mind_state.items(devices.INVOCATION) == []


@pytest.mark.asyncio
async def test_app_grant_endpoint_requires_current_revision_and_os_report_cannot_widen_it(api, rig):
    from test_hands_permit_answers import LANE_SECRET
    headers = {"Authorization": "Bearer " + LANE_SECRET}
    path = routes.APP + "/phone-study/capabilities/alarms/authorization"
    edit = {"expected_revision": rig.service.device("phone-study")["revision"], "method": "write", "mode": "allow"}
    assert (await api.patch(path, json=edit, headers=AUTH)).status_code == 403
    assert (await api.patch(path, json=edit, headers=headers)).status_code == 200
    assert (await api.patch(path, json=edit, headers=headers)).status_code == 409
    report = deepcopy(REPORT)
    report["capabilities"][1]["authorization"] = {"read": "allow", "write": "allow"}
    assert (await api.put(routes.APP + "/phone-study", json=report, headers=headers)).status_code == 422


@pytest.mark.asyncio
async def test_result_wrong_device_and_unadvertised_command(api, rig):
    grant(rig)
    value = rig.service.request(invocation(rig))
    with pytest.raises(HTTPException) as error:
        rig.service.claim("phone-other", value["id"])
    assert error.value.status_code == 404
    value = rig.service.request(invocation(rig, "inv-unsupported", command="alarm.update"))
    assert value["state"] == "unavailable" and value["approval"] is None
