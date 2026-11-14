"""Atomic pickup and credential boundaries for app task requests."""
import asyncio
from datetime import timedelta
import sqlite3

from fastapi import FastAPI, HTTPException
import httpx
import pytest

from colonymind.api.credentials import CredentialTable, secret_digest
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import hands
from colonymind.hands import controller, definitions, wire
from colonymind.hands.definitions_schema import SCHEMA, QUEUE_SCHEMA
from colonymind.hands.store import HandsStore, SCHEMA_VERSION, _SCHEMA
from test_hands_controller import c  # noqa: F401
from test_hands_definitions import events  # noqa: F401

pytestmark = pytest.mark.asyncio


async def queued(c):
    definition = await c.save_definition(definitions.CreateDefinition(name='Reviewer'), 'create')
    task = await c.queue_definition_task(definition['id'], definitions.TaskBrief(brief='Inspect'), 'enqueue')
    return task['id']


async def test_atomic_claims_across_independent_controller_connections(c):
    tid = await queued(c)
    peer = controller.Controller(c.store.path, config=c.config, clock=c.clock)
    await peer.connect()
    try:
        results = await asyncio.gather(c.claim_definition_tasks(definitions.QueueClaim(parent_session='parent'), 'one'),
            peer.claim_definition_tasks(definitions.QueueClaim(parent_session='other'), 'two'))
        assert sum(len(result['tasks']) for result in results) == 1
        assert next(result['tasks'][0] for result in results if result['tasks'])['task_id'] == tid
    finally:
        await peer.close()


async def test_unplanned_claim_expires_back_to_queue_and_rejects_old_token(c):
    tid = await queued(c)
    claim = (await c.claim_definition_tasks(definitions.QueueClaim(parent_session='parent'), 'claim'))['tasks'][0]
    c.test_time[0] += timedelta(seconds=31)
    assert await c.claim_definition_tasks(definitions.QueueClaim(), 'no-parent') == {'tasks': []}
    assert (await c.task_view(tid))['dispatch_state'] == 'waiting_for_main_session'
    row = await c.one('SELECT * FROM definition_task_dispatch WHERE task_id=?', (tid,))
    assert row['claim_token'] is None and row['lease_until'] is None
    with pytest.raises(HTTPException) as error:
        await c.definition_claim(tid, claim['claim_token'])
    assert error.value.status_code == 409
    fresh = (await c.claim_definition_tasks(definitions.QueueClaim(parent_session='parent'), 'again'))['tasks'][0]
    assert fresh['claim_token'] != claim['claim_token']


async def test_claim_requires_matching_parent_and_can_only_plan_once(c):
    tid = await queued(c)
    claim = (await c.claim_definition_tasks(definitions.QueueClaim(parent_session='parent'), 'claim'))['tasks'][0]
    body = wire.PlanRequest(parent_session='other', turn='turn', origin='own', origin_ref='real-ingress',
        tasks=[{'goal': 'Inspect', 'continue': tid, 'queue_claim': claim['claim_token']}])
    with pytest.raises(HTTPException, match='definition_claim_invalid'):
        await c.plan(body, 'wrong-parent')
    body = body.model_copy(update={'parent_session': 'parent'})
    await c.plan(body, 'first')
    with pytest.raises(HTTPException, match='definition_claim_already_planned'):
        await c.plan(body, 'second')
    assert len(await c.all('SELECT * FROM hands')) == 1


async def test_queue_routes_require_existing_runtime_instance_key_and_refuse_children(c, monkeypatch):
    tid = await queued(c)
    app = FastAPI()
    app.include_router(hands.runtime_router)
    app.include_router(hands.router)
    app.state.hands_store = c.store
    table = CredentialTable.from_config({'companion-hands': {
        'owner': [], 'contacts': 'own', 'access': ['hands'], 'sha256': secret_digest('hands-secret')}})
    app.add_middleware(ApiKeyMiddleware, api_key='instance-secret', credentials=table)
    async def current():
        return c
    monkeypatch.setattr(controller, 'get_controller', current)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as http:
        for action, body in [('claim', {'parent_session': 'parent'}),
            ('bind', {'task_id': tid, 'claim_token': 'token', 'hand_id': 'h-123'}),
            ('release', {'task_id': tid, 'claim_token': 'token', 'code': 'slots'})]:
            route = '/v1/hands/definitions/queue/' + action
            for authorization in (None, 'hands-secret'):
                response = await http.post(route, json=body, headers={'Idempotency-Key': action,
                    **({'Authorization': 'Bearer ' + authorization} if authorization else {})})
                assert response.status_code in {401, 403}
        response = await http.post('/v1/hands/definitions/queue/claim', json={'parent_session': 'parent'},
            headers={'Authorization': 'Bearer instance-secret', 'Idempotency-Key': 'valid'})
        assert response.status_code == 200 and response.json()['tasks'][0]['task_id'] == tid
        # Register a child through the normal plan/lease/bind path.
        claim = response.json()['tasks'][0]
        body = wire.PlanRequest(parent_session='parent', turn='turn', origin='own', origin_ref='actual',
            tasks=[{'goal': 'Inspect', 'continue': tid, 'queue_claim': claim['claim_token']}])
        plan = (await c.plan(body, 'plan'))['plans'][0]
        epoch = {'owner_pid': 123, 'owner_started_at': 1.0}
        await c.dispatched(plan['hand_id'], wire.DispatchedRequest(epoch=epoch, delegation_id='child',
            lease_token=plan['lease_token']), 'dispatched')
        await c.bind(plan['hand_id'], wire.BindRequest(epoch=epoch, executor_ref='child', child_session='child'), 'bind')
        for action in ('claim', 'release', 'bind'):
            response = await http.post('/v1/hands/definitions/queue/' + action, json={}, headers={
                'Authorization': 'Bearer instance-secret', 'Idempotency-Key': 'child-' + action,
                'X-ColonyMind-Session': 'child'})
            assert response.status_code == 403


async def test_v2_upgrade_keeps_queue_and_event_namespace(tmp_path):
    path = tmp_path / 'v2.db'
    with sqlite3.connect(path) as db:
        db.executescript(_SCHEMA + SCHEMA + '\nPRAGMA user_version=2;')
        identity = db.execute('SELECT id FROM hands_change_identity').fetchone()[0]
        db.execute("INSERT INTO tasks(id,origin,origin_ref,kind) VALUES('t-existing','own','ref','deep')")
    stores = [HandsStore(path) for _ in range(2)]
    try:
        await asyncio.gather(*(store.connect() for store in stores))
        for store in stores:
            assert (await store.db.execute_fetchall('PRAGMA user_version'))[0][0] == SCHEMA_VERSION
            assert (await store.db.execute_fetchall('SELECT id FROM hands_change_identity'))[0][0] == identity
            assert len(await store.db.execute_fetchall('SELECT * FROM tasks')) == 1
            assert len(await store.db.execute_fetchall('SELECT * FROM hands_changes')) == 1
    finally:
        await asyncio.gather(*(store.close() for store in stores))


async def test_v3_upgrade_does_not_invent_owner_start_for_legacy_requests(tmp_path):
    path = tmp_path / 'v3.db'
    with sqlite3.connect(path) as db:
        db.executescript(_SCHEMA + SCHEMA + QUEUE_SCHEMA + '\nPRAGMA user_version=3;')
        db.execute("INSERT INTO tasks(id,origin,origin_ref,kind) VALUES('t-old','own','ref','deep')")
        db.execute("INSERT INTO definition_task_queue(task_id,brief) VALUES('t-old','Inspect')")
        db.execute("INSERT INTO definition_task_dispatch(task_id,claim_token,lease_until,parent_session) "
                   "VALUES('t-old','old-token','9999','old-parent')")
    stores = [HandsStore(path) for _ in range(3)]
    try:
        await asyncio.gather(*(store.connect() for store in stores))
        for store in stores:
            row, = await store.db.execute_fetchall('SELECT * FROM definition_task_queue')
            assert not row['owner_started'] and row['initiating_principal'] is None
            claim, = await store.db.execute_fetchall('SELECT * FROM definition_task_dispatch')
            assert claim['dispatch_state'] == 'awaiting_owner_start' and claim['claim_token'] is None
    finally:
        await asyncio.gather(*(store.close() for store in stores))
