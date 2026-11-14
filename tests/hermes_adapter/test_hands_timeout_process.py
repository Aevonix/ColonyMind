"""Native timeout and process capabilities preserve Hands wall accounting."""
from concurrent.futures import TimeoutError
import importlib
from types import SimpleNamespace

import pytest

from hands_fake_hermes import Parent
from hermes_source import HERMES_ROOT, assert_source_module, assert_supported_source
from test_hands_dispatch import setup


@pytest.mark.parametrize('progressing', [False, True])
def test_child_timeout_semantics_l_s(setup, monkeypatch, progressing):
    """Run the actual native wait with deterministic time and an unfinished worker."""
    revision = assert_supported_source()
    monkeypatch.syspath_prepend(str(HERMES_ROOT))
    native_run = importlib.import_module('tools.delegate_tool_child_run')
    delegate = importlib.import_module('tools.delegate_tool')
    pool = importlib.import_module('tools.daemon_pool')
    native_async = importlib.import_module('tools.async_delegation')
    assert_source_module(native_run)
    runtime, native, client = setup
    runtime.dispatch_tasks([{'goal':'bounded work'}], parent=Parent())
    hand = runtime.registry.all()[0]
    hand.plan['budgets']['wall_seconds'] = 12
    hand.started_at = 0
    hand.activity = 'generating'
    native.inactivity_timeout = hasattr(native_run._ChildRun, 'wait_liveness_aware')
    assert native.inactivity_timeout is (revision in {'S', 'S217'})
    budgets = importlib.import_module(runtime.__module__.rsplit('.', 1)[0] + '.budgets')
    clock = SimpleNamespace(now=0.0)
    child = SimpleNamespace(session_id=hand.child_session_id, progress=0, closed=False)
    child.get_activity_summary = lambda: {'api_call_count':1, 'last_activity_ts':child.progress}
    child.close = lambda: setattr(child, 'closed', True)
    clock_module = SimpleNamespace(monotonic=lambda:clock.now, time=lambda:clock.now)
    monkeypatch.setattr(native_run, 'time', clock_module)
    monkeypatch.setattr(delegate, '_get_child_timeout', lambda:10)
    monkeypatch.setattr(delegate, '_get_subagent_approval_callback', lambda:None)
    monkeypatch.setattr(native_run, '_signal_child_stop', lambda child:None)
    deferred = []
    monkeypatch.setattr(native_run, '_defer_close_after_timeout', lambda child, future:deferred.append(child))

    class Future:
        def done(self):
            return False

        def add_done_callback(self, callback):
            pass

        def result(self, timeout=None):
            assert timeout == 10 and revision not in {'S', 'S217'}
            clock.now += timeout
            if progressing:
                child.progress = clock.now
            budgets.tick(runtime, now=clock.now)
            raise TimeoutError()

    class Pool:
        def __init__(self, **kwargs):
            pass

        def submit(self, *args):
            return Future()

        def shutdown(self, **kwargs):
            assert kwargs == {'wait':False}

    class Settled:
        def wait(self, timeout=None):
            assert timeout is not None
            clock.now += timeout
            # Four progress signals outlast the 12 second wall budget. Once
            # activity stops, native inactivity still expires ten seconds later.
            if progressing and clock.now <= 20:
                child.progress = clock.now
            budgets.tick(runtime, now=clock.now)

        def is_set(self):
            return False

    monkeypatch.setattr(pool, 'DaemonThreadPoolExecutor', Pool)
    run = native_run._ChildRun(child, Parent(), 0, 'work', None, None, child_start=0)
    if revision in {'S', 'S217'}:
        run.heartbeat = SimpleNamespace(settled=Settled(), stale_threshold_seconds=None)
    result, entry, close_deferred = run.await_child()
    assert result is None and entry['status'] == 'timeout' and close_deferred
    assert deferred == [child] and not child.closed
    if revision in {'S', 'S217'} and progressing:
        assert clock.now == 30
        assert hand.stop_reason == 'wall_budget'
        assert len(native.stops) == 1
    else:
        assert clock.now == 10
        budgets.tick(runtime, now=12)
        assert hand.stop_reason == 'wall_budget'
    # Native finalization releases capacity even while the timed-out worker has
    # not returned. A late result cannot finalize that slot twice.
    ident = hand.delegation_id
    records = {ident:{'delegation_id':ident, 'status':'running'}}
    emitted = []
    monkeypatch.setattr(native_async, '_records', records)
    monkeypatch.setattr(native_async, '_push_completion_event', lambda *args:emitted.append(args))
    monkeypatch.setattr(native_async, '_prune_completed_locked', lambda:None)
    native_async._finalize(ident, {'results':[entry]}, 'completed')
    native_async._finalize(ident, {'results':[]}, 'completed')
    assert records[ident]['status'] not in native_async._ACTIVE_STATES
    assert len(emitted) == 1
    runtime.subagent_stop(child_session_id=hand.child_session_id, child_status=entry['status'])
    recorded = [body for path, body, _ in client.writes if path.endswith('/result')]
    assert recorded[0]['status'] == ('done' if revision not in {'S', 'S217'} else 'failed')
