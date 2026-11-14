from copy import deepcopy
import pytest
from colonymind.config import DEFAULTS
from colonymind.hands.controller import Controller
from colonymind.hands.definitions import CreateDefinition, TaskBrief, QueueClaim
from colonymind.hands.wire import CancelRequest


@pytest.mark.asyncio
async def test_empty_poll_constant_after_retained_pre_dispatch_cancellations(tmp_path, monkeypatch):
    monkeypatch.setenv('COLONYMIND_EVENT_JOURNAL_DIR', str(tmp_path/'events'))
    c=Controller(tmp_path/'cancelled.db', config=deepcopy(DEFAULTS['hands']))
    await c.connect()
    async def cost():
        steps=[0]
        def step():
            steps[0]+=1
            return 0
        await c.store.db.set_progress_handler(step, 1)
        try:
            assert await c.claim_definition_tasks(QueueClaim(parent_session='owner'), 'empty') == {'tasks':[]}
        finally:
            await c.store.db.set_progress_handler(None, 0)
        return steps[0]
    try:
        initial=await cost()
        definition=await c.save_definition(CreateDefinition(name='Cancelled work'), 'definition')
        task=await c.queue_definition_task(definition['id'], TaskBrief(brief='Inspect sources'), 'enqueue')
        await c.cancel(task['id'], CancelRequest(reason='Owner cancelled before pickup'), 'cancel')
        queue=await c.one('SELECT * FROM definition_task_queue WHERE task_id=?', (task['id'],))
        assert queue['dispatched']==0 and queue['owner_started']==1
        one=await cost()
        count=100000
        await c.store.db.executemany("INSERT INTO tasks(id,origin,origin_ref,kind,state) VALUES(?,'own','cancelled-history','deep','cancelled')",
                                    ((f't-retained-{i}',) for i in range(count)))
        await c.store.db.executemany('INSERT INTO task_definitions(task_id,definition_id,version,snapshot) '
                                    'SELECT ?,definition_id,version,snapshot FROM task_definitions WHERE task_id=?',
                                    ((f't-retained-{i}', task['id']) for i in range(count)))
        await c.store.db.executemany('INSERT INTO definition_task_queue(task_id,brief,dispatched,initiating_principal,initiating_credential,owner_started) '
                                    'SELECT ?,brief,dispatched,initiating_principal,initiating_credential,owner_started '
                                    'FROM definition_task_queue WHERE task_id=?',
                                    ((f't-retained-{i}', task['id']) for i in range(count)))
        await c.store.db.executemany('INSERT INTO definition_task_dispatch(task_id) VALUES(?)',
                                    ((f't-retained-{i}',) for i in range(count)))
        await c.execute('UPDATE hands_changes SET prepared=1,published=1')
        await c.store.db.commit()
        retained=await cost()
        print({'initial_steps':initial, 'one_real_cancellation_steps':one,
               '100k_retained_cancellations_steps':retained})
        assert retained <= initial+100, 'An empty poll scans retained cancelled task history'
    finally:
        await c.close()
