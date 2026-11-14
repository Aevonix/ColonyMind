"""Selected native lease release, profile routing and front retry boundaries."""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from importlib import import_module
from pathlib import Path
import os
import sys
import threading
from types import SimpleNamespace as NS

import pytest

from hands_fake_hermes import FakeContext, attach_turn_leases, hold_turn_lease, load_hands
from test_hands_front import event, owner_turn, review_turn, rig  # noqa: F401
from test_hands_review_requests import request


@pytest.fixture
def lease(tmp_path, monkeypatch):
    if not os.environ.get('HERMES_SOURCE_ROOT'):
        pytest.skip('set HERMES_SOURCE_ROOT to qualify native lease release')
    package = load_hands()
    module = import_module(package.__name__ + '.settlement')
    native = NS(persistent=NS(run_generation=1))
    runner = NS(_peek_session_state=lambda key: native if key == 'route' else None,
        _is_session_run_current=lambda key, generation: key == 'route' and generation == native.persistent.run_generation)
    registry = attach_turn_leases(runner, native)
    native_hooks = []
    monkeypatch.setitem(sys.modules, 'hermes_cli.lifecycle', NS(
        has_hook=lambda name: True, invoke_hook=lambda name, **values: native_hooks.append((name, values))))
    calls = []
    def observe(**values):
        assert registry.released, 'the registry must release the token first'
        assert values['run_generation'] not in native.turn.lease_tokens
        calls.append(values)
    observer = module.Settlement(observe, tmp_path)
    original = runner._release_turn_lease
    observer.install(runner)
    yield NS(**locals())
    observer.close()


def test_release_wrapper_notifies_only_after_success(lease):
    original_calls = []
    def original(*args):
        original_calls.append(args)
        return lease.original(*args)
    lease.observer.wrapper.original = original
    token = hold_turn_lease(lease.runner, lease.native)
    assert lease.runner._release_turn_lease('route', 1) is True
    assert lease.registry.released == [token]
    assert len(lease.calls) == 1
    assert original_calls == [('route', 1)]
    assert lease.runner._release_turn_lease('route', 1) is False
    assert lease.runner._release_turn_lease('missing', 1) is False
    assert len(original_calls) == 3
    for result, error in [(False, None), (True, RuntimeError('registry failure'))]:
        lease.native.persistent.run_generation += 1
        hold_turn_lease(lease.runner, lease.native)
        lease.registry.result, lease.registry.error = result, error
        assert lease.runner._release_turn_lease('route', lease.native.persistent.run_generation) is False
        assert len(lease.calls) == 1
    lease.registry.error = None
    lease.native.persistent.run_generation += 1
    hold_turn_lease(lease.runner, lease.native)
    def broken(**values):
        raise RuntimeError('observer failure')
    lease.observer.callback = broken
    assert lease.runner._release_turn_lease('route', lease.native.persistent.run_generation) is True
    # An exception raised by the original method itself remains an exception.
    def raised(*args):
        raise LookupError('native failure')
    lease.observer.wrapper.original = raised
    with pytest.raises(LookupError, match='native failure'):
        lease.runner._release_turn_lease('route', 9)


def test_release_wrapper_uses_token_session_after_rotation(lease):
    hold_turn_lease(lease.runner, lease.native)
    assert lease.runner._rebind_turn_lease('route', 1, 'rotated-session')
    # No session lookup after release may replace the authoritative token identity.
    lease.runner.session_store = NS(config=NS(multiplex_profiles=False),
                                   get=lambda *args: pytest.fail('must not read the session store'))
    assert lease.runner._release_turn_lease('route', 1) is True
    assert lease.calls[0]['session_id'] == 'rotated-session'


def test_l_fork_hook_does_not_double_settle(lease, monkeypatch):
    front = import_module(lease.package.__name__ + '.front')
    dispatch = import_module(lease.package.__name__ + '.dispatch')
    ctx, calls = FakeContext(), []
    runtime = NS(config={}, client=NS(settings=NS(hermes_home=lease.tmp_path)),
                 gateway=lambda: lease.runner)
    monkeypatch.setattr(dispatch, 'get_runtime', lambda context: runtime)
    monkeypatch.setattr(front.FrontGuard, 'on_gateway_turn_settled', lambda self, **values: calls.append(values))
    front.register(ctx)
    try:
        assert 'on_session_end' in ctx.hooks
        assert 'on_native_turn_settled' not in ctx.hooks
        assert 'on_gateway_turn_settled' not in ctx.hooks
        hold_turn_lease(lease.runner, lease.native)
        assert lease.runner._release_turn_lease('route', 1) is True
        assert len(calls) == 1
        from hermes_source import assert_supported_source
        assert len(lease.native_hooks) == (1 if assert_supported_source() not in {'S', 'S217'} else 0)
    finally:
        for unload in reversed(ctx.unloads):
            unload()


@pytest.mark.parametrize('abandonment', ['before', 'during', 'closed', 'closes_during', 'disposed'])
def test_abandoned_front_registration_preserves_active_bridge_and_observer(rig, monkeypatch, abandonment):
    dispatch = import_module(rig.front.__package__ + '.dispatch')
    monkeypatch.setattr(dispatch, 'get_runtime', lambda context: rig.runtime)
    rig.runtime._closed = threading.Event()
    if abandonment == 'closed':
        rig.runtime._closed.set()
    guards = []
    original_guard = rig.front.FrontGuard
    class ObservedGuard(original_guard):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            guards.append(self)
    monkeypatch.setattr(rig.front, 'FrontGuard', ObservedGuard)
    class Context(FakeContext):
        _load_abandoned = abandonment == 'before'

        def on_unload(self, callback):
            if abandonment == 'during':
                self._load_abandoned = True
            if self._load_abandoned:
                return  # The native S registrar silently ignores this call.
            if abandonment == 'closes_during':
                rig.runtime._closed.set()
            if abandonment == 'disposed':
                callback()
                return
            super().on_unload(callback)

        def register_hook(self, *args, **kwargs):
            if not self._load_abandoned:
                super().register_hook(*args, **kwargs)

        def register_middleware(self, *args, **kwargs):
            if not self._load_abandoned:
                super().register_middleware(*args, **kwargs)

    ctx = Context()
    bridge = sys.modules[rig.front._BRIDGE]
    release = rig.gateway._release_turn_lease
    owners = dict(rig.guard.settlement.wrapper.owners)
    rig.front.register(ctx)
    assert sys.modules[rig.front._BRIDGE] is bridge
    assert rig.runtime.ingress is rig.guard
    assert not rig.guard.closed
    assert rig.gateway._release_turn_lease is release
    assert rig.guard.settlement.wrapper.owners == owners
    assert all(guard.closed for guard in guards)
    assert not ctx.hooks and not ctx.middleware
    for unload in ctx.unloads:
        unload()


def test_settlement_wrapper_reload_and_unload(lease):
    wrapped = lease.runner._release_turn_lease
    clearing = lease.runner._release_running_agent_state
    original_clear = lease.observer.wrapper.original_clear
    assert lease.observer.install(lease.runner)
    assert lease.runner._release_turn_lease is wrapped
    assert lease.runner._release_running_agent_state is clearing
    newer_calls = []
    newer = lease.module.Settlement(lambda **values: newer_calls.append(values), lease.tmp_path)
    assert newer.install(lease.runner)
    lease.observer.close()
    assert lease.runner._release_turn_lease is wrapped
    assert lease.runner._release_running_agent_state is clearing
    hold_turn_lease(lease.runner, lease.native)
    assert lease.runner._release_turn_lease('route', 1) is True
    assert len(newer_calls) == 1 and not lease.calls
    newer.close()
    assert lease.runner._release_turn_lease == lease.original
    assert lease.runner._release_running_agent_state == original_clear
    # A later wrapper belonging to another component must never be overwritten.
    other = lease.module.Settlement(lambda **values: None)
    other.install(lease.runner)
    replacement = lambda *args: True
    lease.runner._release_turn_lease = replacement
    lease.runner._release_running_agent_state = replacement
    other.close()
    assert lease.runner._release_turn_lease is replacement
    assert lease.runner._release_running_agent_state is replacement


def test_early_cleanup_capture_uses_profile_owner_and_cannot_change_native_behavior(lease, monkeypatch):
    active_home = ContextVar('early_cleanup_test_home', default=None)
    @contextmanager
    def scope(home, **kwargs):
        token = active_home.set(str(home))
        try:
            yield
        finally:
            active_home.reset(token)
    monkeypatch.setitem(sys.modules, 'gateway.run', NS(_profile_runtime_scope=scope))
    lease.runner.session_store = NS(config=NS(multiplex_profiles=True),
        _named_profile_for_key=lambda key: 'owner', _profile_home_for_key=lambda key: lease.tmp_path)
    prepared = []
    lease.runner._running_agents = {'route': object()}
    def prepare(**values):
        assert 'route' in lease.runner._running_agents
        assert not lease.registry.released and not lease.calls
        prepared.append((active_home.get(), values['session_id']))
    lease.observer.prepare = prepare
    other = lease.module.Settlement(lambda **values: pytest.fail('wrong profile settled'), lease.tmp_path / 'other',
        prepare=lambda **values: pytest.fail('wrong profile observed'))
    other.install(lease.runner)
    try:
        hold_turn_lease(lease.runner, lease.native)
        assert lease.runner._release_running_agent_state('route', run_generation=1) is True
        assert prepared == [(str(lease.tmp_path), 'owner-session')]
        assert lease.runner._release_turn_lease('route', 1) is True
        assert len(lease.calls) == 1
        hold_turn_lease(lease.runner, lease.native)
        def broken(**values):
            raise RuntimeError('observer failure')
        lease.observer.prepare = broken
        assert lease.runner._release_running_agent_state('route', run_generation=1) is True
        def raised(*args, **kwargs):
            raise LookupError('native cleanup failure')
        lease.observer.wrapper.original_clear = raised
        with pytest.raises(LookupError, match='native cleanup failure'):
            lease.runner._release_running_agent_state('route', run_generation=1)
    finally:
        other.close()


def test_early_cleanup_reload_replaces_disposed_marker_copied_by_later_wrapper(lease):
    from functools import wraps
    previous = lease.runner._release_turn_lease
    original_clear = lease.observer.wrapper.original_clear
    outer_calls, prepared, settled = [], [], []
    @wraps(previous)
    def outer(*args, **kwargs):
        outer_calls.append(args)
        return previous(*args, **kwargs)
    lease.runner._release_turn_lease = outer
    lease.observer.close()
    assert lease.runner._release_running_agent_state == original_clear
    newer = lease.module.Settlement(lambda **values: settled.append(values), lease.tmp_path,
        prepare=lambda **values: prepared.append(values))
    newer.install(lease.runner)
    try:
        assert newer.wrapper is not previous
        assert newer.wrapper.original is outer
        hold_turn_lease(lease.runner, lease.native)
        assert lease.runner._release_running_agent_state('route', run_generation=1)
        assert len(prepared) == 1
        assert lease.runner._release_turn_lease('route', 1)
        assert len(settled) == len(outer_calls) == 1 and not lease.calls
    finally:
        newer.close()
    assert lease.runner._release_turn_lease is outer
    assert lease.runner._release_running_agent_state == original_clear


def test_profile_scope_survives_cross_thread_settlement_and_stale_callback_is_cancelled(lease, monkeypatch):
    active_home = ContextVar('settlement_test_home', default=None)
    @contextmanager
    def scope(home, **kwargs):
        token = active_home.set(str(home))
        try:
            yield
        finally:
            active_home.reset(token)
    monkeypatch.setitem(sys.modules, 'gateway.run', NS(_profile_runtime_scope=scope))
    lease.runner.session_store = NS(config=NS(multiplex_profiles=True),
        _named_profile_for_key=lambda key: 'owner', _profile_home_for_key=lambda key: lease.tmp_path)
    wrong = lease.module.Settlement(lambda **values: pytest.fail('wrong profile notified'), lease.tmp_path / 'other')
    wrong.install(lease.runner)
    seen = []
    lease.observer.callback = lambda **values: seen.append((active_home.get(), values['session_id']))
    async def run():
        lease.runner._gateway_loop = asyncio.get_running_loop()
        hold_turn_lease(lease.runner, lease.native, 'rotated-session')
        # Block this loop until the release worker has enqueued the callback.
        worker = threading.Thread(target=lambda: lease.runner._release_turn_lease('route', 1))
        worker.start()
        worker.join(2)
        assert not worker.is_alive() and not seen
        await asyncio.sleep(0)
        assert seen == [(str(lease.tmp_path), 'rotated-session')]
        lease.native.persistent.run_generation = 2
        hold_turn_lease(lease.runner, lease.native)
        worker = threading.Thread(target=lambda: lease.runner._release_turn_lease('route', 2))
        worker.start()
        worker.join(2)
        lease.native.persistent.run_generation = 3
        await asyncio.sleep(0)
        assert len(seen) == 1
    try:
        asyncio.run(run())
    finally:
        wrong.close()


def test_running_agent_removed_is_not_settled(rig):
    review_turn(rig)
    rig.pending['route'] = event('Owner asks now', 'message-2')
    assert request(rig)[0] == []
    rig.pending.clear()
    owner_turn(rig, 'Owner asks now', 'turn-2', 'message-2')
    async def cleanup():
        removed, resume = asyncio.Event(), asyncio.Event()
        async def finish():
            rig.gateway._running_agents.pop('route')
            removed.set()
            await resume.wait()
            assert rig.gateway._release_turn_lease('route', 1) is True
        task = asyncio.create_task(finish())
        await removed.wait()
        assert not rig.ctx.injected
        resume.set()
        await task
    asyncio.run(cleanup())
    assert [row['content'] for row in rig.ctx.injected] == ['Native completion']


def test_internal_dispatch_installs_before_ingress_early_return(rig):
    rig.guard.settlement._remove()
    assert not hasattr(rig.gateway._release_turn_lease, '_colonymind_hands_settlement')
    rig.guard.pre_gateway_dispatch(event=event(internal=True), gateway=rig.gateway)
    assert hasattr(rig.gateway._release_turn_lease, '_colonymind_hands_settlement')


def test_a_gateway_without_a_turn_lease_function_is_warned_about_once(tmp_path, caplog):
    """A Hermes whose gateway lacks a function the observer wraps: nothing is installed and the log names the
    function, once, though every inbound message asks for the install again."""
    module = import_module(load_hands().__name__ + '.settlement')
    observer = module.Settlement(lambda **_: None, tmp_path)
    gateway = NS(_held_turn_lease=lambda key, generation: None)
    with caplog.at_level('WARNING', logger=module.logger.name):
        assert observer.install(None) is False          # no gateway yet: nothing to say
        assert not caplog.records
        assert [observer.install(gateway) for _ in range(3)] == [False, False, False]
    said = [record.getMessage() for record in caplog.records if record.name == module.logger.name]
    assert len(said) == 1 and '_release_turn_lease' in said[0] and '_held_turn_lease' not in said[0]
    assert not hasattr(gateway, '_release_turn_lease')


def test_concurrent_front_call_counts_and_duplicate_tool_ids(rig):
    from concurrent.futures import ThreadPoolExecutor
    owner_turn(rig)
    state = rig.guard._state('owner-session', 'turn-1')
    for n in range(7):
        assert rig.guard.pre_tool_call(session_id=state.session, turn_id=state.turn,
            tool_name='edit_file', tool_call_id=f'call-{n}').get('action') != 'block'
    barrier = threading.Barrier(16)
    def call(number):
        barrier.wait()
        return rig.guard.pre_tool_call(session_id=state.session, turn_id=state.turn,
            tool_name='edit_file', tool_call_id='last' if number < 8 else f'extra-{number}')
    with ThreadPoolExecutor(max_workers=16) as workers:
        results = list(workers.map(call, range(16)))
    accepted = [number for number, result in enumerate(results) if result.get('action') != 'block']
    # Seven calls, the eight that share one id counted once, and eight more: all pass, none is capped.
    assert state.calls == {'read': 0, 'write': 16}
    assert len(state.tool_ids) == 16
    assert accepted == list(range(16))
