"""Synthetic checks through stock Hermes dispatch; only a local fake sidecar."""
from conftest import home, sidecar, probe, worker_env
from test_guard import GUARD_CODE


def test_stock_dispatch_compressed_delegate_preserves_guest(home, sidecar):
    result=probe(GUARD_CODE + '''
guest('group', '2003')
invoke_hook('subagent_start', parent_session_id='group', parent_turn_id='turn', child_session_id='child')
invoke_hook('pre_llm_call', session_id='child', platform='subagent', sender_id='',
            parent_session_id='group', turn_id='child-turn', task_id='t', user_message='synthetic',
            conversation_history=[], is_first_turn=True, model='m')
owner('group')
from hermes_state_registry import acquire, release_or_close
db=acquire()
db.create_session('group', 'telegram')
db.create_session('child', 'subagent', parent_session_id='group')
db.create_session('compressed', 'subagent', parent_session_id='child')
release_or_close(db)
emit(before=(check('terminal', {}, 'child')['action'] == 'block'),
     after=(check('terminal', {}, 'compressed')['action'] == 'block'))
''',home)
    assert result['before']
    assert result['after']


def test_stock_dispatch_owner_task_keeps_owner_tools(home, sidecar):
    result=probe(GUARD_CODE + '''
colonymind_hermes._BODY.sessions.standings.put('kanban', 'task-01', 'owner')
emit(allowed=check('write_file', {'path':'identity.yaml', 'content':''}, 'worker')['action'] is None)
''',home,env=worker_env(home))
    assert result['allowed']
