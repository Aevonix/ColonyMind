"""An interrupted native turn releases transport binding before its successor."""

from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest

from colonymind_hermes.capture import Capture
from colonymind_hermes.client import SidecarUnavailable


def test_settled_hook_posts_only_the_exact_interrupted_native_tuple():
    calls = []
    client = SimpleNamespace(post=lambda path, **kwargs: calls.append((path, kwargs)) or httpx.Response(200))
    capture = Capture(SimpleNamespace(client=client), None, None)
    native = dict(session_id='session', task_id='task', turn_id='turn')
    for outcome in [None, {}, {'completed': True}, {'interrupted': False}, {'interrupted': 'true'}]:
        capture.on_native_turn_settled(**native, outcome=outcome)
    capture.on_native_turn_settled(session_id='session', outcome={'interrupted': True})
    assert calls == []
    capture.on_native_turn_settled(**native, outcome={'completed': False, 'interrupted': True})
    assert calls == [('/v1/host/transport/ingress/native-settled',
                      {'json': {'native_turn': native, 'interrupted': True}, 'timeout': 2})]


@pytest.mark.parametrize('status', [401, 403, 409, 503, None])
def test_binding_release_failure_does_not_raise_from_native_settlement(status, caplog):
    calls = []
    def post(*args, **kwargs):
        calls.append((args, kwargs))
        if status is None:
            raise OSError('fixture')
        return httpx.Response(status)
    capture = Capture(SimpleNamespace(client=SimpleNamespace(post=post)), None, None)
    capture.on_native_turn_settled(session_id='session', task_id='task', turn_id='turn',
                                  outcome={'interrupted': True})
    assert 'interrupted transport binding release failed' in caplog.text
    assert len(calls) == 1


@pytest.fixture
def bound_transport():
    transport = _bound_transport()
    try:
        yield transport
    finally:
        transport.connection.close()


def _bound_transport(native=None):
    # Load the real receipt store without requiring the sidecar package and all
    # its optional dependencies in the adapter test environment.
    path = Path(__file__).resolve().parents[2] / 'sidecar/colonymind/contacts/transport_ingress.py'
    spec = importlib.util.spec_from_file_location('binding_receipt_store', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    connection = sqlite3.connect(':memory:')
    connection.row_factory = sqlite3.Row
    module.ensure_schema(connection)
    store = module.TransportIngress(connection)
    receipt = store.admit(producer='transport', account_id='account', epoch='epoch', sequence=1,
        event_id='message', contact_id='owner', occurred_at=100, journal_ref='epoch:1',
        payload_digest='a' * 64, media_available=True)['receipt_id']
    batch = dict(producer='transport', receipt_ids=[receipt], batch_id='batch')
    native = native or dict(session_id='session', task_id='task', turn_id='original')
    assert store.handoff(**batch)['may_dispatch'] is True
    assert store.handoff(**batch, native_turn=native)['native_bound'] is True
    return SimpleNamespace(connection=connection, store=store, receipt=receipt, batch=batch, native=native)


@pytest.mark.parametrize('committed_before_timeout', [False, True])
@pytest.mark.parametrize('already_captured', [False, True])
def test_release_timeout_retries_exact_tuple_before_successor_binds(
        bound_transport, committed_before_timeout, already_captured, caplog):
    transport = bound_transport
    if already_captured:
        transport.store.complete(native_turn=transport.native,
            source_versions={'original': 'version'}, outcome='captured')
    calls, replies = [], []

    def post(path, **kwargs):
        calls.append((path, deepcopy(kwargs)))
        if len(calls) == 1 and not committed_before_timeout:
            raise SidecarUnavailable('ReadTimeout', delivered=True)
        result = transport.store.native_settled(producer='key', host=True, **kwargs['json'])
        if len(calls) == 1:
            raise SidecarUnavailable('ReadTimeout', delivered=True)
        replies.append(result)
        return httpx.Response(200, json=result)

    capture = Capture(SimpleNamespace(client=SimpleNamespace(post=post)), None, None)
    capture.on_native_turn_settled(**transport.native, outcome={'interrupted': True})
    following = dict(transport.native, turn_id='continuation')
    result = transport.store.handoff(**transport.batch, native_turn=following)
    assert result['native_bound'] is True and result['may_dispatch'] is False
    assert result['processing_complete'] is already_captured
    assert len(calls) == 2 and calls[0] == calls[1]
    assert calls[0] == ('/v1/host/transport/ingress/native-settled',
        {'json': {'native_turn': transport.native, 'interrupted': True}, 'timeout': 2})
    assert replies == [{'released': 1, 'processing_complete': False}]
    assert 'interrupted transport binding release failed' not in caplog.text
    # A retry of the old tuple cannot release a successor that already bound.
    assert transport.store.native_settled(producer='key', host=True,
        native_turn=transport.native, interrupted=True)['released'] == 0
    with pytest.raises(ValueError, match='ingress_not_ready' if already_captured else 'native_turn_conflict'):
        transport.store.handoff(**transport.batch, native_turn=transport.native)


@pytest.mark.parametrize('delivered', [False, True])
def test_release_transport_failure_has_bounded_attempts_and_keeps_binding(
        bound_transport, delivered, caplog):
    calls = []

    def post(path, **kwargs):
        calls.append((path, deepcopy(kwargs)))
        raise SidecarUnavailable('ConnectError' if not delivered else 'ReadTimeout', delivered=delivered)

    capture = Capture(SimpleNamespace(client=SimpleNamespace(post=post)), None, None)
    capture.on_native_turn_settled(**bound_transport.native, outcome={'interrupted': True})
    assert len(calls) == 3 and all(call == calls[0] for call in calls)
    assert 'interrupted transport binding release failed' in caplog.text
    assert bound_transport.store.get(bound_transport.receipt)['native_interrupted'] == 0
    with pytest.raises(ValueError, match='native_turn_conflict'):
        bound_transport.store.handoff(**bound_transport.batch,
            native_turn=dict(bound_transport.native, turn_id='continuation'))


def test_zero_release_does_not_retry_or_unlock_an_unmatched_native_tuple(bound_transport):
    calls = []

    def post(path, **kwargs):
        result = bound_transport.store.native_settled(producer='key', host=True, **kwargs['json'])
        calls.append(result)
        return httpx.Response(200, json=result)

    capture = Capture(SimpleNamespace(client=SimpleNamespace(post=post)), None, None)
    capture.on_native_turn_settled(**dict(bound_transport.native, task_id='other'),
                                  outcome={'interrupted': True})
    assert calls == [{'released': 0, 'processing_complete': False}]
    with pytest.raises(ValueError, match='native_turn_conflict'):
        bound_transport.store.handoff(**bound_transport.batch,
            native_turn=dict(bound_transport.native, turn_id='continuation'))


def test_stock_capture_settlement_waits_for_native_cleanup_and_owns_reload(monkeypatch):
    from agent import conversation_loop
    from agent.turn_facade import TurnFacadeMixin
    from hermes_cli import plugins
    from colonymind_hermes import native_settlement
    monkeypatch.setattr(plugins, 'VALID_HOOKS', set(plugins.VALID_HOOKS) - {'on_native_turn_settled'})
    events = []
    def original(agent, user_message, **kwargs):
        if user_message == 'unadmitted':
            return {'interrupted': True}
        try:
            return conversation_loop.run_conversation(agent, user_message, **kwargs)
        finally:
            events.append('released')
    def loop(agent, user_message, **kwargs):
        agent.session_id, agent._current_task_id, agent._current_turn_id = 'rotated', 'task', user_message
        events.append('running')
        if user_message == 'raised':
            raise InterruptedError('fixture')
        return {'interrupted': True, 'completed': False}
    monkeypatch.setattr(TurnFacadeMixin, 'run_conversation', original)
    monkeypatch.setattr(conversation_loop, 'run_conversation', loop)
    unloads = []
    ctx = SimpleNamespace(on_unload=unloads.append)
    def observer(**kwargs):
        assert events[-1] == 'released'
        assert kwargs['session_id'] == 'rotated' and kwargs['task_id'] == 'task'
        events.append(('observed', kwargs['turn_id'], kwargs['outcome']['interrupted']))
    capture = SimpleNamespace(on_native_turn_settled=observer, native_turn_settled=lambda **kw: None)
    native_settlement.register(ctx, capture)
    wrapper = TurnFacadeMixin.run_conversation
    native_settlement.register(ctx, capture)
    assert TurnFacadeMixin.run_conversation is wrapper
    unloads[0]()
    assert TurnFacadeMixin.run_conversation is wrapper
    agent = SimpleNamespace(_current_turn_id='old')
    result = wrapper(agent, 'turn')
    assert result == {'interrupted': True, 'completed': False}
    assert events == ['running', 'released', ('observed', 'turn', True)]
    wrapper(agent, 'unadmitted')
    assert len(events) == 3
    with pytest.raises(InterruptedError):
        wrapper(agent, 'raised')
    assert events[-1] == ('observed', 'raised', True)
    unloads[1]()
    assert TurnFacadeMixin.run_conversation is original
    assert conversation_loop.run_conversation is loop


def test_native_capture_observer_cannot_change_success_or_restore_new_owner(monkeypatch):
    from agent import conversation_loop
    from agent.turn_facade import TurnFacadeMixin
    from hermes_cli import plugins
    from colonymind_hermes import native_settlement
    monkeypatch.setattr(plugins, 'VALID_HOOKS', set(plugins.VALID_HOOKS) - {'on_native_turn_settled'})
    result = {'completed': True}
    def original(agent, *args):
        return conversation_loop.run_conversation(agent, *args)
    def loop(agent, *args):
        agent._current_turn_id = 'new'
        return result
    monkeypatch.setattr(TurnFacadeMixin, 'run_conversation', original)
    monkeypatch.setattr(conversation_loop, 'run_conversation', loop)
    def broken(**kw):
        raise RuntimeError('observer')
    frozen, unloads = [], []
    native_settlement.register(SimpleNamespace(on_unload=unloads.append), SimpleNamespace(
        on_native_turn_settled=broken, native_turn_settled=lambda **kw: frozen.append(kw)))
    assert TurnFacadeMixin.run_conversation(SimpleNamespace(), 'text') is result
    assert len(frozen) == 1
    newer = lambda *a: None
    monkeypatch.setattr(TurnFacadeMixin, 'run_conversation', newer)
    unloads[0]()
    assert TurnFacadeMixin.run_conversation is newer


def test_stock_capture_reload_replaces_disposed_marker_copied_by_newer_wrapper(monkeypatch):
    from functools import wraps
    from agent import conversation_loop
    from agent.turn_facade import TurnFacadeMixin
    from hermes_cli import plugins
    from colonymind_hermes import native_settlement

    monkeypatch.setattr(plugins, 'VALID_HOOKS', set(plugins.VALID_HOOKS) - {'on_native_turn_settled'})
    def original(agent, message):
        return conversation_loop.run_conversation(agent, message)
    def loop(agent, message):
        agent.session_id, agent._current_task_id, agent._current_turn_id = 'session', 'task', message
        return {'completed': True}
    monkeypatch.setattr(TurnFacadeMixin, 'run_conversation', original)
    monkeypatch.setattr(conversation_loop, 'run_conversation', loop)
    observed, unloads = [], []
    capture = SimpleNamespace(on_native_turn_settled=lambda **values: observed.append(values),
                              native_turn_settled=lambda **values: None)
    ctx = SimpleNamespace(on_unload=unloads.append)
    native_settlement.register(ctx, capture)
    old_wrapper = TurnFacadeMixin.run_conversation
    @wraps(old_wrapper)
    def newer(agent, message):
        return old_wrapper(agent, message)
    monkeypatch.setattr(TurnFacadeMixin, 'run_conversation', newer)
    unloads[0]()
    assert TurnFacadeMixin.run_conversation is newer and conversation_loop.run_conversation is loop
    native_settlement.register(ctx, capture)
    current = TurnFacadeMixin.run_conversation
    assert current is not newer and conversation_loop.run_conversation is not loop
    unloads[0]()
    assert TurnFacadeMixin.run_conversation is current
    assert current(SimpleNamespace(), 'turn') == {'completed': True}
    assert len(observed) == 1 and observed[0]['turn_id'] == 'turn'
    unloads[1]()
    assert TurnFacadeMixin.run_conversation is newer and conversation_loop.run_conversation is loop


def test_stock_capture_settlement_keeps_profile_owners_and_unload_separate(monkeypatch):
    from contextvars import ContextVar
    from agent import conversation_loop
    from agent.turn_facade import TurnFacadeMixin
    from hermes_cli import plugins
    import hermes_constants
    from colonymind_hermes import native_settlement

    profile = ContextVar('capture_test_profile', default='first')
    monkeypatch.setattr(hermes_constants, 'hermes_home_key', profile.get)
    monkeypatch.setattr(plugins, 'VALID_HOOKS', set(plugins.VALID_HOOKS) - {'on_native_turn_settled'})
    def original(agent, message):
        return conversation_loop.run_conversation(agent, message)
    def loop(agent, message):
        agent.session_id, agent._current_task_id, agent._current_turn_id = 'session', 'task', message
        return {'completed': True}
    monkeypatch.setattr(TurnFacadeMixin, 'run_conversation', original)
    monkeypatch.setattr(conversation_loop, 'run_conversation', loop)
    observations, unloads = [], {}
    for home in ('first', 'second'):
        profile.set(home)
        unloads[home] = []
        capture = SimpleNamespace(on_native_turn_settled=lambda home=home, **values:
            observations.append((home, profile.get(), values['turn_id'])), native_turn_settled=lambda **values: None)
        native_settlement.register(SimpleNamespace(on_unload=unloads[home].append), capture)
    wrapper, wrapped_loop = TurnFacadeMixin.run_conversation, conversation_loop.run_conversation
    try:
        for home in ('first', 'second', 'unowned'):
            profile.set(home)
            assert wrapper(SimpleNamespace(), home)['completed'] is True
        assert observations == [('first', 'first', 'first'), ('second', 'second', 'second')]
        unloads['first'][0]()
        assert TurnFacadeMixin.run_conversation is wrapper and conversation_loop.run_conversation is wrapped_loop
        profile.set('first')
        wrapper(SimpleNamespace(), 'after-unload')
        assert len(observations) == 2
    finally:
        for callbacks in unloads.values():
            for callback in callbacks:
                callback()
    assert TurnFacadeMixin.run_conversation is original and conversation_loop.run_conversation is loop


@pytest.mark.parametrize('abandonment', ['before', 'during', 'disposed'])
def test_stock_capture_abandoned_registration_leaves_native_methods_alone(monkeypatch, abandonment):
    from agent import conversation_loop
    from agent.turn_facade import TurnFacadeMixin
    from hermes_cli import plugins
    from colonymind_hermes import native_settlement
    monkeypatch.setattr(plugins, 'VALID_HOOKS', set(plugins.VALID_HOOKS) - {'on_native_turn_settled'})
    original, loop = TurnFacadeMixin.run_conversation, conversation_loop.run_conversation
    ctx = SimpleNamespace(_load_abandoned=abandonment == 'before')
    def on_unload(callback):
        if abandonment == 'during':
            ctx._load_abandoned = True
        elif not ctx._load_abandoned:
            callback()
    ctx.on_unload = on_unload
    capture = SimpleNamespace(on_native_turn_settled=lambda **values: None,
                              native_turn_settled=lambda **values: None)
    native_settlement.register(ctx, capture)
    assert TurnFacadeMixin.run_conversation is original and conversation_loop.run_conversation is loop


@pytest.mark.parametrize('release_reply', ['success', 'timeout_before_commit', 'timeout_after_commit', 'exhausted'])
@pytest.mark.parametrize('already_captured', [False, True])
def test_selected_native_facade_releases_before_capture_and_keeps_exact_tuple(
        tmp_path, release_reply, already_captured):
    if not os.environ.get('HERMES_SOURCE_ROOT'):
        pytest.skip('set HERMES_SOURCE_ROOT to qualify native facade cleanup')
    from hermes_source import assert_supported_source
    assert_supported_source()
    env = dict(os.environ, HERMES_HOME=str(tmp_path), PYTHON_DOTENV_DISABLED='1',
               HERMES_DISABLE_LAZY_INSTALLS='1')
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'native_facade',
                             release_reply, str(int(already_captured))],
                            cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def _native_facade_probe(release_reply, already_captured):
    from contextlib import ExitStack, nullcontext
    from unittest.mock import patch

    sys.path.insert(0, os.environ['HERMES_SOURCE_ROOT'])
    from agent import conversation_loop, turn_facade
    from agent.turn_context import _bind_turn_identity
    from hermes_cli import lifecycle, plugins
    from hermes_source import assert_source_module, assert_supported_source
    from colonymind_hermes import native_settlement

    revision = assert_supported_source()
    assert_source_module(turn_facade)
    original = turn_facade.TurnFacadeMixin.run_conversation
    events, observed, posts, hooks, unloads = [], [], [], {}, []
    current = {}
    ctx = SimpleNamespace(on_unload=unloads.append,
        register_hook=lambda name, callback: hooks.setdefault(name, []).append(callback))
    def post(path, **kwargs):
        assert 'lease released' in events
        assert 'work frozen' not in events
        posts.append((path, deepcopy(kwargs)))
        if release_reply == 'exhausted' or (release_reply == 'timeout_before_commit' and len(posts) == 1):
            raise SidecarUnavailable('ReadTimeout', delivered=True)
        result = current['transport'].store.native_settled(producer='key', host=True, **kwargs['json'])
        if release_reply == 'timeout_after_commit' and len(posts) == 1:
            raise SidecarUnavailable('ReadTimeout', delivered=True)
        return httpx.Response(200, json=result)
    capture = Capture(SimpleNamespace(client=SimpleNamespace(post=post)), None, None)
    def observe(**native):
        assert events[-1] == 'turn finished', events
        assert native['session_id'] == 'rotated-session'
        assert native['task_id'] == 'task'
        assert native['turn_id'] == current['bound_turn']
        assert not {'messages', 'final_response'} & set(native.get('outcome') or {})
        observed.append(native)
        capture.on_native_turn_settled(**native)
    observer = SimpleNamespace(on_native_turn_settled=observe,
                               native_turn_settled=lambda **native: events.append('work frozen'))

    class Lease:
        def start(self):
            events.append('lease started')
        def stop_refresher(self):
            events.append('refresher stopped')
        def join_threads(self):
            events.append('threads joined')
        def clear_interrupt(self):
            events.append('interrupt cleared')
        def release(self):
            events.append('lease released')
            if current['mode'] == 'successor':
                # A new turn can bind immediately after the old lease releases.
                current['agent']._current_task_id = 'successor-task'
                current['agent']._current_turn_id = 'successor-turn'

    def admission(agent, **values):
        early = {'interrupted': True, 'messages': []} if current['mode'] == 'unadmitted' else None
        return SimpleNamespace(early_result=early, lease=None if early else Lease(),
                               conversation_history=values['conversation_history'])

    def loop(agent, user_message, system_message, history, task_id, stream_callback, persist_user_message, **kwargs):
        task, turn = _bind_turn_identity(agent, task_id, stream_callback, persist_user_message, None, None)
        assert task == 'task'
        current['bound_turn'] = turn
        agent.session_id = 'rotated-session'
        current['transport'] = _bound_transport(dict(session_id=agent.session_id, task_id=task, turn_id=turn))
        if already_captured:
            current['transport'].store.complete(native_turn=current['transport'].native,
                source_versions={'original': 'version'}, outcome='captured')
        events.append('loop returned early')
        if current['mode'] == 'raised':
            raise current['error']
        return current['result']

    coordinator = SimpleNamespace(acquire_conversation=lambda **kwargs: object(),
        register_session_initializer=lambda *args, **kwargs: None,
        begin_turn=lambda *args, **kwargs: SimpleNamespace(relay_enabled=False),
        finish_logical_calls=lambda *args, **kwargs: events.append('logical calls finished'),
        end_turn=lambda *args, **kwargs: events.append('relay ended'),
        release_conversation=lambda *args: events.append('relay released'))
    with ExitStack() as patches:
        def replace(name, value):
            patches.enter_context(patch(name, value))
        replace('agent.background_review.cancel_background_review_for_live_turn', lambda agent: None)
        replace('agent.turn_facade_lease.admit_durable_turn_lease', admission)
        replace('agent.relay_runtime.SESSION_COORDINATOR', coordinator)
        replace('agent.relay_runtime.current_profile_key', lambda: 'fixture')
        replace('agent.auxiliary_client.scoped_runtime_main', lambda *args: nullcontext())
        replace('agent.interrupt_scope.track_in_interrupt_scope', lambda *args: nullcontext())
        replace('agent.prompt_cache_scope.declared_conversation_scope_safe', lambda agent: None)
        replace('agent.review_idle_queue.QUEUE', SimpleNamespace(
            note_turn_started=lambda: events.append('turn started'),
            note_turn_finished=lambda: events.append('turn finished')))
        replace('agent.conversation_loop.run_conversation', loop)
        if revision in {'S', 'S217'}:
            replace('agent.relay_cwd.resolve_relay_scope_cwds', lambda *args: (None, None))
        replace('hermes_cli.lifecycle.has_hook', lambda name: bool(hooks.get(name)))
        replace('hermes_cli.lifecycle.invoke_hook',
                lambda name, **kwargs: [fn(**kwargs) for fn in hooks.get(name, [])])
        native_settlement.register(ctx, observer)
        assert ('on_native_turn_settled' in hooks) is (revision not in {'S', 'S217'})
        assert (turn_facade.TurnFacadeMixin.run_conversation is original) is (revision not in {'S', 'S217'})
        try:
            for mode in ('interrupted', 'success', 'raised', 'unadmitted', 'detached', 'successor'):
                events.clear()
                observed.clear()
                posts.clear()
                current.clear()
                current.update(mode=mode, error=InterruptedError('fixture'), result={
                    'interrupted': mode != 'success', 'completed': mode == 'success', 'failed': False,
                    'final_response': 'fixture private response', 'messages': [{'content': 'fixture private words'}]})
                agent = SimpleNamespace(session_id='session', platform='sms',
                    _current_turn_id='older-turn', _persist_disabled=mode == 'detached',
                    _conversation_root_id=lambda: 'root', _reset_activity_labels_after_turn=lambda: None)
                current['agent'] = agent
                if mode == 'raised':
                    try:
                        turn_facade.TurnFacadeMixin.run_conversation(agent, 'Owner input', task_id='task')
                    except InterruptedError as error:
                        assert error is current['error']
                    else:
                        raise AssertionError('native exception must propagate')
                else:
                    result = turn_facade.TurnFacadeMixin.run_conversation(agent, 'Owner input', task_id='task')
                    if mode != 'unadmitted':
                        assert result is current['result']
                if mode in {'unadmitted', 'detached'}:
                    assert observed == [] and posts == []
                    assert 'work frozen' not in events
                else:
                    assert len(observed) == 1 and events.count('work frozen') == 1
                    # A host without a settlement outcome keeps its behavior when the loop raises;
                    # stock cleanup also has an explicit interrupted exception verdict.
                    interrupted = mode != 'success' and (mode != 'raised' or revision in {'S', 'S217'})
                    attempts = {'success': 1, 'timeout_before_commit': 2, 'timeout_after_commit': 2,
                                'exhausted': 3}[release_reply]
                    assert len(posts) == (attempts if interrupted else 0)
                    if posts:
                        native = dict(session_id='rotated-session', task_id='task', turn_id=current['bound_turn'])
                        assert all(call == posts[0] for call in posts)
                        assert posts[0] == ('/v1/host/transport/ingress/native-settled',
                            {'json': {'native_turn': native, 'interrupted': True}, 'timeout': 2})
                        transport = current['transport']
                        successor = dict(native, task_id='successor-task', turn_id='successor-turn')
                        if release_reply == 'exhausted':
                            assert transport.store.get(transport.receipt)['native_interrupted'] == 0
                            with pytest.raises(ValueError, match='ingress_not_ready' if already_captured
                                               else 'native_turn_conflict'):
                                transport.store.handoff(**transport.batch, native_turn=successor)
                        else:
                            result = transport.store.handoff(**transport.batch, native_turn=successor)
                            assert result['native_bound'] is True and result['may_dispatch'] is False
                            assert result['processing_complete'] is already_captured
                            assert transport.store.native_settled(producer='key', host=True,
                                native_turn=native, interrupted=True)['released'] == 0
                if 'transport' in current:
                    current['transport'].connection.close()
                assert not any(name == 'on_session_end' for name in hooks)
        finally:
            for unload in reversed(unloads):
                unload()
        assert turn_facade.TurnFacadeMixin.run_conversation is original
        assert conversation_loop.run_conversation is loop


if __name__ == '__main__':
    from unittest.mock import patch
    with patch('socket.socket.connect', side_effect=AssertionError('native capture probe must remain offline')):
        _native_facade_probe(sys.argv[2], bool(int(sys.argv[3])))
