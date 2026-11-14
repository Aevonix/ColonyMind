"""Terminal work cannot stay in the polling index or reuse a leased claim."""
import asyncio
import sqlite3

from fastapi import HTTPException
import pytest

from colonymind.hands import definitions, wire
from colonymind.hands.definitions_schema import SCHEMA, QUEUE_SCHEMA, OWNER_QUEUE_SCHEMA
from colonymind.hands.store import HandsStore, SCHEMA_VERSION, _SCHEMA
from test_hands_controller import c  # noqa: F401
from test_hands_definitions import events  # noqa: F401
from test_hands_definition_queue import queued

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize('transition', ['cancel', 'done', 'failed', 'revise', 'revise-stop', 'stop'])
async def test_terminal_or_revised_request_drops_eligibility_and_claim(c, transition):
    tid = await queued(c)
    claim = (await c.claim_definition_tasks(definitions.QueueClaim(parent_session='parent'), 'claim'))['tasks'][0]
    plan = (await c.plan(wire.PlanRequest(parent_session='parent', turn='turn', origin='own', origin_ref='ref',
        tasks=[{'goal': 'Inspect', 'continue': tid, 'queue_claim': claim['claim_token']}]), 'plan'))['plans'][0]
    if transition == 'cancel':
        await c.cancel(tid, wire.CancelRequest(reason='Owner cancelled'), 'cancel')
    elif transition.startswith('revise'):
        await c.revise(tid, wire.ReviseRequest(goal='Changed goal', stop=transition == 'revise-stop'), 'revise')
    elif transition == 'stop':
        await c.stop(plan['hand_id'], wire.StopRequest(reason='Owner stopped'), 'stop')
    else:
        await c.update('tasks', 'id', tid, state=transition)
        await c.store.db.commit()
    row = await c.one('SELECT * FROM definition_task_queue WHERE task_id=?', (tid,))
    assert row['owner_started'] == 1 and row['dispatched'] == 0 and row['eligible'] == 0
    dispatch = await c.one('SELECT * FROM definition_task_dispatch WHERE task_id=?', (tid,))
    assert dispatch['claim_token'] is None and dispatch['lease_until'] is None
    assert await c.claim_definition_tasks(definitions.QueueClaim(parent_session='parent'), 'again') == {'tasks': []}
    with pytest.raises(HTTPException, match='definition_claim_invalid'):
        await c.bind_definition_claim(definitions.QueueBind(task_id=tid, hand_id=plan['hand_id'],
                                                            claim_token=claim['claim_token']), 'late-bind')
    assert (await c.pending_definition_tasks())['tasks'] == []


async def test_v4_migration_removes_retained_terminal_rows_and_preserves_evidence(tmp_path):
    path = tmp_path / 'v4.db'
    states = ['open', 'waiting', 'cancelled', 'done', 'failed']
    with sqlite3.connect(path) as db:
        # A real v4 HandsStore is already in WAL mode before concurrent opens.
        db.executescript(_SCHEMA + SCHEMA + QUEUE_SCHEMA + OWNER_QUEUE_SCHEMA
                         + '\nPRAGMA journal_mode=WAL; PRAGMA user_version=4;')
        for state in states:
            db.execute("INSERT INTO tasks(id,origin,origin_ref,kind,state) VALUES(?,'own','ref','deep',?)", (state, state))
            db.execute("INSERT INTO definition_task_queue(task_id,brief,owner_started,initiating_principal,initiating_credential) "
                       "VALUES(?,'Original brief',1,'owner-id','credential-id')", (state,))
            db.execute("INSERT INTO definition_task_dispatch(task_id,claim_token,lease_until,parent_session) "
                       "VALUES(?,?,'9999','original-parent')", (state, 'token-' + state))
        identity = db.execute('SELECT id FROM hands_change_identity').fetchone()[0]
        history = db.execute('SELECT * FROM hands_changes').fetchall()
    stores = [HandsStore(path) for _ in range(3)]
    try:
        await asyncio.gather(*(store.connect() for store in stores))
        for store in stores:
            await store.migrate()
            assert (await store.db.execute_fetchall('PRAGMA user_version'))[0][0] == SCHEMA_VERSION
            rows = await store.db.execute_fetchall('SELECT * FROM definition_task_queue')
            assert {r['task_id'] for r in rows if r['eligible']} == {'open', 'waiting'}
            assert all((r['brief'], r['initiating_principal'], r['initiating_credential'], r['owner_started'], r['dispatched'])
                       == ('Original brief', 'owner-id', 'credential-id', 1, 0) for r in rows)
            dispatch = await store.db.execute_fetchall('SELECT * FROM definition_task_dispatch')
            assert {r['task_id'] for r in dispatch if r['claim_token']} == {'open', 'waiting'}
            assert (await store.db.execute_fetchall('SELECT id FROM hands_change_identity'))[0][0] == identity
            assert [tuple(r) for r in await store.db.execute_fetchall('SELECT * FROM hands_changes')] == history
            assert (await store.db.execute_fetchall('PRAGMA integrity_check'))[0][0] == 'ok'
            assert not await store.db.execute_fetchall('PRAGMA foreign_key_check')
    finally:
        await asyncio.gather(*(store.close() for store in stores))
