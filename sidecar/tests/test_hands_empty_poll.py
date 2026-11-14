"""Review query-cost reproduction, exercising the actual claim route queries."""
import pytest

from colonymind.hands.definitions import QueueClaim
from test_hands_controller import c  # noqa: F401
from test_hands_definitions import events  # noqa: F401

pytestmark = pytest.mark.asyncio


async def poll_cost(c):
    steps, statements = [0], []
    def step():
        steps[0] += 1
        return 0
    await c.store.db.set_progress_handler(step, 1)
    await c.store.db.set_trace_callback(statements.append)
    try:
        assert await c.claim_definition_tasks(QueueClaim(parent_session='owner'), 'empty') == {'tasks': []}
    finally:
        await c.store.db.set_progress_handler(None, 0)
        await c.store.db.set_trace_callback(None)
    return steps[0], statements


@pytest.mark.parametrize('history', ['done', 'cancelled'])
async def test_empty_poll_work_is_constant_with_retained_history(c, record_property, history):
    empty, _ = await poll_cost(c)
    await c.store.db.executemany("INSERT INTO tasks(id,origin,origin_ref,kind,state) "
        "VALUES(?,'own','historical','deep','open')", ((f't-{i}',) for i in range(100000)))
    await c.store.db.executemany('INSERT INTO definition_task_queue(task_id,brief,dispatched,owner_started,eligible) '
        'VALUES(?,?,?,?,1)', ((f't-{i}', 'Historical task', int(history == 'done'), 1) for i in range(100000)))
    await c.store.db.executemany("INSERT INTO definition_task_dispatch(task_id,dispatch_state) VALUES(?,'dispatched')",
        ((f't-{i}',) for i in range(100000)))
    await c.execute('UPDATE tasks SET state=?', (history,))
    assert not await c.one('SELECT task_id FROM definition_task_queue WHERE eligible=1')
    await c.execute('UPDATE hands_changes SET prepared=1,published=1')
    await c.store.db.commit()
    retained, statements = await poll_cost(c)
    record_property('empty_poll_vm_steps', empty)
    record_property('retained_100k_poll_vm_steps', retained)
    print(f'empty={empty} VM steps; retained_100k={retained} VM steps; SQL={statements}')
    assert retained <= empty + 100
    assert not any('BEGIN' in sql.upper() or 'HANDS_CHANGES' in sql.upper() for sql in statements)
    for sql, index in [
        ('SELECT * FROM hands_changes WHERE prepared=0 ORDER BY id', 'hands_changes_unprepared'),
        ('SELECT * FROM hands_changes WHERE published=0 AND prepared=1 ORDER BY id', 'hands_changes_unpublished'),
    ]:
        plans = await c.all('EXPLAIN QUERY PLAN ' + sql)
        assert any(index in plan['detail'] for plan in plans), plans
        steps = [0]
        def step():
            steps[0] += 1
            return 0
        await c.store.db.set_progress_handler(step, 1)
        assert not await c.all(sql)
        await c.store.db.set_progress_handler(None, 0)
        assert steps[0] < 100


async def test_empty_poll_skips_writer_and_outbox(c, monkeypatch):
    async def unexpected(*args, **kwargs):
        pytest.fail('empty claim entered writer/outbox maintenance')
    monkeypatch.setattr(c, 'write', unexpected)
    monkeypatch.setattr(c, 'prepare_changes', unexpected)
    monkeypatch.setattr(c, 'publish_changes', unexpected)
    await poll_cost(c)
