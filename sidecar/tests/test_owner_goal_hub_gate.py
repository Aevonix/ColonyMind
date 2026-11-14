"""Production server wiring, using only synthetic marker data and local ASGI requests."""
from dataclasses import replace
import hashlib
import json
import sqlite3
from types import SimpleNamespace
import uuid

from httpx import ASGITransport, AsyncClient
import pytest

from colonymind.api.credentials import CredentialTable, secret_digest
from colonymind.api.routers import mind as mind_routes
from colonymind.initiatives.store import InitiativeStore
from colonymind.owner_goals.authority import ActionContext, Audience, VerifiedAction, get_action_verifier
from colonymind.owner_goals.hub import verify_companion_owner_action
from colonymind.owner_goals.store import OwnerGoalsStore
from colonymind.owner_goals.validation import GoalError, dumps

ROOT = "/v1/mind/owner-goals"


def marker():
    return "marker-" + str(uuid.uuid4())


@pytest.fixture
async def hub(tmp_path, monkeypatch, request):
    from colonymind.api import auth

    owner, instance = marker(), marker()
    monkeypatch.setenv("COLONYMIND_OWNER_PERSON_ID", owner)
    tokens = {name: marker() for name in (
        "companion-hub", "other-hub", "non-owner", "other-owner", "wildcard", "read-only")}
    channels = {"companion-hub": ["companion"], "other-hub": ["companion"],
                "non-owner": [], "other-owner": ["whatsapp"], "wildcard": "any",
                "read-only": ["companion"]}
    channels["companion-hub"] = getattr(request, "param", ["companion"])
    table = CredentialTable.from_config({name: {
        "sha256": secret_digest(token), "owner": channels[name], "contacts": "all",
        "access": ["read"] if name == "read-only" else ["manage"],
    } for name, token in tokens.items()})
    monkeypatch.setattr(CredentialTable, "load", classmethod(lambda cls: table))
    monkeypatch.setattr(auth, "configured_api_key", lambda: instance)
    from colonymind.server import create_app

    intentions = InitiativeStore(tmp_path)
    goals = OwnerGoalsStore(intentions)
    mind = SimpleNamespace(store=intentions, owner_goals=goals)
    monkeypatch.setattr(mind_routes, "get_mind", lambda: mind)
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield SimpleNamespace(client=client, app=app, mind=mind, goals=goals,
                              owner=owner, channels=channels, tokens=tokens | {"instance": instance})
    intentions.close()


def creation():
    return {"title": marker(), "description": marker(), "category": marker()}


async def send(hub, method, path=ROOT, body=None, *, credential="companion-hub", key=None, headers=None):
    auth = {} if credential is None else {"Authorization": "Bearer " + hub.tokens[credential]}
    return await hub.client.request(method, path, json=body, headers={
        **auth, "Idempotency-Key": key or str(uuid.uuid4()), **(headers or {})})


def assert_code(response, status, code):
    assert response.status_code == status, response.text
    assert response.json()["detail"]["code"] == code


async def test_hub_goal_lifecycle_and_durable_replay(hub):
    listing = await send(hub, "GET")
    assert listing.status_code == 200, listing.text
    assert listing.json()["goals"] == []
    operations = []

    async def mutate(method, path, body, status):
        key = str(uuid.uuid4())
        response = await send(hub, method, path, body, key=key)
        assert response.status_code == status, response.text
        operations.append((method, path, body, key, response.json(), status))
        return response.json()

    body = creation() | {"sensitivity": "Secret"}
    goal = (await mutate("POST", ROOT, body, 201))["goal"]
    path = ROOT + "/" + goal["id"]
    assert (await send(hub, "GET", path)).json()["goal"]["title"] == body["title"]
    assert (await send(hub, "GET")).json()["goals"][0]["id"] == goal["id"]
    goal = (await mutate("PATCH", path, {"expected_revision": goal["revision"],
        "title": marker(), "muted": True, "status": "completed", "sensitivity": "Private"}, 200))["goal"]
    assert goal["muted"] and goal["status"] == "completed" and goal["sensitivity"] == "Private"
    progress = {"at": "2026-10-17T12:00:00Z", "text": marker()}
    goal = (await mutate("POST", path + "/progress", progress, 201))["goal"]
    assert (await send(hub, "GET", path + "/progress")).json()["items"][0]["text"] == progress["text"]
    assert (await send(hub, "GET", path + "/activity")).status_code == 200
    await mutate("DELETE", path, {"expected_revision": goal["revision"]}, 200)
    assert_code(await send(hub, "GET", path), 404, "owner_goal_not_found")
    assert (await send(hub, "GET")).json()["goals"] == []

    # Reopen the store: replay uses retained effects before checking deleted targets/revisions.
    hub.mind.owner_goals = OwnerGoalsStore(hub.goals.db_path)
    for method, route, original, key, result, status in operations:
        response = await send(hub, method, route, original, key=key)
        assert response.status_code == status, response.text
        assert response.json() == result
    assert_code(await send(hub, "POST", ROOT, body | {"title": marker()},
                           key=operations[0][3]), 409, "idempotency_conflict")
    with hub.goals.connection() as db:
        assert db.execute("SELECT count(*) FROM owner_goal_receipts").fetchone()[0] == 4
        assert db.execute("SELECT count(*) FROM owner_goal_records WHERE kind='owner_goal_progress'").fetchone()[0] == 1


def seed_proposal(hub, kind, target_id=None):
    """An already verified historical proposal; the production agent seam remains closed."""
    payload = creation() if kind == "goal" else {"at": "2026-10-17T12:00:00Z", "text": marker()}
    body = {"kind": kind, "target_id": target_id, "payload": payload}
    context = ActionContext(hub.owner, marker(), Audience(hub.owner, frozenset({hub.owner}), True),
                            "POST", ROOT + "/proposals", str(uuid.uuid4()), body,
                            hashlib.sha256(dumps(["POST", ROOT + "/proposals", body]).encode()).hexdigest())

    def historical_verifier(ctx, **kwargs):
        return VerifiedAction(ctx.owner_id, ctx.principal_id, marker(), "proposal_only")

    return hub.goals.mutate(context, "propose", historical_verifier)[1]["proposal"]


@pytest.mark.parametrize("kind", ["goal", "progress"])
async def test_hub_confirms_exact_proposal_and_replays(hub, kind):
    goal = (await send(hub, "POST", body=creation())).json()["goal"]
    proposal = seed_proposal(hub, kind, goal["id"] if kind == "progress" else None)
    path = ROOT + "/proposals/" + proposal["id"]
    assert (await send(hub, "GET", ROOT + "/proposals")).json()["proposals"][0] == proposal
    assert (await send(hub, "GET", path)).json()["proposal"] == proposal
    key, body = str(uuid.uuid4()), {"expected_revision": proposal["revision"]}
    assert_code(await send(hub, "POST", path + "/confirm", {"expected_revision": 0}),
                409, "revision_conflict")
    confirmed = await send(hub, "POST", path + "/confirm", body, key=key)
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["proposal"]["state"] == "confirmed"
    with hub.goals.connection() as db:
        row = db.execute("SELECT resolved_action FROM owner_goal_receipts WHERE operation_id=?", (key,)).fetchone()
        retained = json.loads(hub.goals._plaintext(row[0]))
        assert retained["proposal"]["state"] == "proposed"
        assert retained["proposal"]["revision"] == proposal["revision"]
    hub.mind.owner_goals = OwnerGoalsStore(hub.goals.db_path)
    assert (await send(hub, "POST", path + "/confirm", body, key=key)).json() == confirmed.json()
    assert_code(await send(hub, "POST", path + "/confirm", body), 409, "proposal_already_confirmed")


@pytest.mark.parametrize("credential,status,code", [
    ("non-owner", 403, "not_owner"), ("other-owner", 503, "owner_goals_unavailable"),
    ("wildcard", 503, "owner_goals_unavailable"), ("instance", 503, "owner_goals_unavailable"),
    ("read-only", 403, "route_not_granted"),
])
@pytest.mark.parametrize("method,suffix,body", [
    ("GET", "", None), ("POST", "", {}), ("PATCH", "/{id}", {"expected_revision": 1}),
    ("POST", "/{id}/progress", {}), ("POST", "/proposals/{id}/confirm", {"expected_revision": 1}),
    ("DELETE", "/{id}", {"expected_revision": 1}),
])
async def test_other_credentials_keep_refusals_despite_spoofing(hub, credential, status, code, method, suffix, body):
    path = ROOT + suffix.format(id=marker())
    assert_code(await send(hub, method, path, body, credential=credential), status, code)
    spoof = {"Owner-Goal-Action": marker(), "X-Owner": hub.owner, "X-Owner-Platforms": "companion",
             "X-Credential-Id": "companion-hub", "X-Owner-Goal-Audience": "owner"}
    if body is not None:
        body = body | {"by": "owner", "owner_id": hub.owner, "owner": ["companion"],
                       "mode": "owner", "source_id": marker()}
    response = await send(hub, method, path, body,
                          credential=credential, headers=spoof)
    assert_code(response, status, code)
    with hub.goals.connection() as db:
        assert db.execute("SELECT count(*) FROM owner_goal_receipts").fetchone()[0] == 0


@pytest.mark.parametrize("provenance", [False, True])
async def test_proposal_only_stays_denied_even_for_hub(hub, provenance):
    body = {"kind": "goal", "target_id": None, "payload": creation()}
    headers = {}
    if provenance:
        body["source_id"] = marker()
        headers["Owner-Goal-Action"] = marker()
    response = await send(hub, "POST", ROOT + "/proposals", body, headers=headers)
    assert_code(response, 403 if provenance else 422, "not_owner" if provenance else "owner_turn_required")


@pytest.mark.parametrize("field", ["by", "owner_id", "owner_goal_audience", "mode"])
async def test_hub_caller_authority_fields_remain_invalid(hub, field):
    assert_code(await send(hub, "POST", body=creation() | {field: "owner"}), 422, "invalid_request")


async def test_receipts_are_scoped_to_authenticated_principal(hub):
    body, key = creation(), str(uuid.uuid4())
    first = await send(hub, "POST", body=body, key=key)
    second = await send(hub, "POST", body=body, key=key, credential="other-hub")
    assert first.status_code == second.status_code == 201
    assert first.json()["goal"]["id"] != second.json()["goal"]["id"]
    assert (await send(hub, "POST", body=body, key=key)).json() == first.json()


@pytest.mark.parametrize("hub", [[], ["whatsapp"]], indirect=True)
async def test_hub_name_without_companion_grant_has_no_authority(hub):
    status, code = (503, "owner_goals_unavailable") if hub.channels["companion-hub"] else (403, "not_owner")
    assert_code(await send(hub, "GET"), status, code)
    assert_code(await send(hub, "POST", body=creation()), status, code)


async def test_verifier_binds_context_effects_receipt_and_transaction(hub):
    phases, connections = [], []

    def inspect(context, *, resolved_action, receipt, connection):
        phases.append(resolved_action.get("phase", "resolved"))
        connections.append(connection)
        assert connection.in_transaction
        grant = verify_companion_owner_action(context, resolved_action=resolved_action,
                                               receipt=receipt, connection=connection)
        if resolved_action.get("phase") == "authorize":
            return grant
        # A grant for different invocation data or a mixed audience cannot authorize execution.
        for field, value in {
            "owner_id": marker(), "principal_id": marker(), "method": "DELETE", "path": ROOT + "/" + marker(),
            "request_hash": marker(), "operation_id": str(uuid.uuid4()), "body": creation(),
            "audience": replace(context.audience, person_ids=frozenset({context.owner_id, marker()})),
        }.items():
            with pytest.raises(GoalError, match="not_owner"):
                verify_companion_owner_action(replace(context, **{field: value}),
                    resolved_action=resolved_action, receipt=receipt, connection=connection)
        retained = {"owner_id": context.owner_id, "principal_id": context.principal_id,
                    "operation_id": context.operation_id, "request_hash": context.request_hash,
                    "evidence_id": grant.evidence_id}
        assert verify_companion_owner_action(context, resolved_action=resolved_action,
            receipt=retained, connection=connection) == grant
        for field in retained:
            with pytest.raises(GoalError, match="not_owner"):
                verify_companion_owner_action(context, resolved_action=resolved_action,
                    receipt=retained | {field: marker()}, connection=connection)
        with pytest.raises(GoalError, match="not_owner"):
            verify_companion_owner_action(context, resolved_action=resolved_action | {"payload": creation()},
                receipt=retained, connection=connection)
        # Even otherwise matching invocation/evidence cannot execute outside a transaction.
        idle = sqlite3.connect(":memory:")
        try:
            with pytest.raises(GoalError, match="not_owner"):
                verify_companion_owner_action(context, resolved_action=resolved_action,
                    receipt=retained, connection=idle)
        finally:
            idle.close()
        return grant

    async def dependency():
        return inspect

    hub.app.dependency_overrides[get_action_verifier] = dependency
    response = await send(hub, "POST", body=creation())
    assert response.status_code == 201, response.text
    assert phases == ["authorize", "resolved"]
    assert connections[0] is connections[1]
