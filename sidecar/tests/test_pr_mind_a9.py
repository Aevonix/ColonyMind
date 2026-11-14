"""Board blockers reach the duty concerns without inventing autonomous work."""
from colonymind.mind.drives import DriveInputs, duty
from test_mind_consolidate import fx, OWNER


def test_blocked_tasks_raise_owner_concerns_and_exclude_worker_tasks(fx):
    inputs = DriveInputs(now=fx.now, owner_id=OWNER, worker_profile='worker', blocked_tasks=[
        {'id': 'waiting', 'title': 'Review the draft', 'assignee': 'owner'},
        {'id': 'worker-task', 'title': 'Worker draft', 'assignee': 'worker'},
        {'title': 'Missing identity'}])
    fx.mind._raise_concerns(inputs, fx.now)
    concerns = fx.mind.concerns.open()
    blocked = [c for c in concerns if c.dedup_key.startswith('blocked_task:')]
    assert len(blocked) == 1
    assert blocked[0].dedup_key == 'blocked_task:waiting'
    assert blocked[0].kind == 'obligation' and blocked[0].drive == 'duty'
    assert blocked[0].detail['recipient'] == OWNER and blocked[0].detail['kind'] == 'message'
    assert blocked[0].sources == ['kanban:waiting']
    fx.mind._raise_concerns(inputs, fx.now)
    assert len([c for c in fx.mind.concerns.open() if c.dedup_key == 'blocked_task:waiting']) == 1


def test_blocked_task_does_not_address_an_unknown_owner(fx):
    result = duty(DriveInputs(now=fx.now, owner_id='', blocked_tasks=[{'id': 'waiting'}]))
    assert result[1] == []
