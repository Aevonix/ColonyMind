"""Process-stop results follow the selected native registry capability."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hands_fake_hermes import load_hands
from hermes_source import HERMES_ROOT, assert_source_module, assert_supported_source


@pytest.mark.parametrize('result,confirmed', [
    ({'status':'killed'}, True), ({'status':'already_exited'}, True),
    ({'status':'error', 'process_running':True}, False), ({'error':'kill failed'}, False),
    ({'status':'killed', 'process_running':True}, False), (None, False),
])
def test_stop_records_only_confirmed_exit(monkeypatch, result, confirmed):
    revision = assert_supported_source()
    monkeypatch.syspath_prepend(str(HERMES_ROOT))
    process_module = importlib.import_module('tools.process_registry')
    assert_source_module(process_module)
    reconcile = importlib.import_module(load_hands().__name__ + '.reconcile')
    process = SimpleNamespace(id='process-1', pid=123, host_start_time=40,
                              pid_scope='host', exited=False, owner_task_id='child-task')
    killed = []
    def kill(sid, **kwargs):
        killed.append(sid)
        return result
    registry = SimpleNamespace(get=lambda sid:process, kill_process=kill)
    capability = getattr(process_module.ProcessRegistry, '_post_kill_survivors', None)
    assert callable(capability) is (revision in {'S', 'S217'})
    if capability:
        registry._post_kill_survivors = capability
    hand = SimpleNamespace(plan={'process_handles':{'process-1':{
        'pid':123, 'started_at':40, 'owner_task_id':'child-task'}}})
    assert reconcile.stop_recorded_processes(hand, registry=registry, start_time=lambda pid:41) == []
    process.owner_task_id = 'other-task'
    assert reconcile.stop_recorded_processes(hand, registry=registry, start_time=lambda pid:40) == []
    assert not killed
    process.owner_task_id = 'child-task'
    expected = ['process-1'] if revision not in {'S', 'S217'} or confirmed else []
    assert reconcile.stop_recorded_processes(hand, registry=registry, start_time=lambda pid:40) == expected
    assert killed == ['process-1']


def test_process_stop_matches_earlier_layout():
    """Compare function outputs from an earlier layout without requiring Git history."""
    reconcile = importlib.import_module(load_hands().__name__ + '.reconcile')
    fixture = json.loads((Path(__file__).parent / 'fixtures' / 'process_stop_earlier_layout.json').read_text())
    process = SimpleNamespace(pid=123, host_start_time=40, owner_task_id='task', pid_scope='host', exited=False)
    hand = SimpleNamespace(plan={'process_handles':{'child':{'pid':123,'started_at':40,'owner_task_id':'task'}}})
    for case in fixture['cases']:
        registry = SimpleNamespace(get=lambda sid:process, kill_process=lambda *args, **kwargs:case['kill_result'])
        options = {'registry':registry, 'start_time':lambda pid:40}
        assert reconcile.stop_recorded_processes(hand, **options) == case['stopped']
