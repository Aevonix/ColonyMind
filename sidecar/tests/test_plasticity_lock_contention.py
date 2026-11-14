"""The nightly's thread/process lock must never suspend the sidecar event loop."""
import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from colonymind.plasticity.locking import StateLock
from colonymind.plasticity.learners import action_success, contact_reply_timing, query_adapter, reply_timing, salience
from colonymind.plasticity.nightly import Runtime


@asynccontextmanager
async def busy(lock, *, process=False):
    """A finite hold makes the unpatched regression fail rather than hang."""
    release, held = threading.Event(), threading.Event()
    child = None
    if process:
        # A fresh interpreter, not fork: no inherited mutex/descriptor state.
        child = subprocess.Popen([sys.executable, '-c', '''
import fcntl, pathlib, sys, time
root = pathlib.Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=True)
with (root / '.lock').open('a+b') as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    (root / 'held').touch()
    time.sleep(.6)
''', str(lock.root)])
        deadline = time.monotonic() + 5
        while not (lock.root / 'held').exists():
            assert child.poll() is None and time.monotonic() < deadline
            await asyncio.sleep(.001)
    else:
        def hold():
            with lock:
                held.set()
                release.wait(.6)
        holder = threading.Thread(target=hold)
        holder.start()
        while not held.is_set():
            await asyncio.sleep(.001)
    beats = []

    async def heartbeat():
        while True:
            beats.append(time.monotonic())
            await asyncio.sleep(.005)

    task = asyncio.create_task(heartbeat())
    await asyncio.sleep(.015)
    start = time.monotonic()
    try:
        yield
        elapsed = time.monotonic() - start
        await asyncio.sleep(.025)
        assert elapsed < .15, f'loop-thread call waited {elapsed:.3f}s for the plasticity lock'
        assert len(beats) >= 3
        assert max(b - a for a, b in zip(beats, beats[1:])) < .15
    finally:
        release.set()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        if child is not None:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)
        else:
            holder.join(timeout=2)
            assert not holder.is_alive()


@pytest.mark.asyncio
@pytest.mark.parametrize('process', [False, True], ids=['thread', 'process'])
async def test_prepare_falls_back_and_recovers(tmp_path, process):
    from colonymind.initiatives.store import InitiativeStore
    from colonymind.mind import outreach
    from colonymind.mind.concerns import Concerns
    from colonymind.turns.idempotency import TurnIdempotencyLedger
    from test_mind_outreach import declared, finding, inputs
    from test_plasticity_reply_timing import NOW, H, runtime, turn

    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    turn(ledger, 'last', NOW - H)
    InitiativeStore(state_dir=tmp_path).close()
    service = runtime(tmp_path, [NOW])
    concerns = Concerns(tmp_path / 'mind.db')
    mind = NS(plasticity=service, faculties={'plasticity': True}, owner_id='Owner', mind_state=concerns.state)
    try:
        choose = service.loaded['timing'].driver.forced.choose
        service.loaded['timing'].driver.forced.choose = lambda *a, **kw: replace(choose(*a, **kw), forced=False)
        service.pin = lambda *a: NS(closure=NS(mode='rebuild'), state={})
        service.loaded['timing'].driver.learner.predict = lambda state, frames: np.array([
            [.2 if datetime.fromisoformat(frames[0]['at']).hour == NOW.hour else .8]])
        state = inputs(now=NOW, interests=[declared()], findings=[finding()])
        # Also check a reused snapshot cannot retain a learned hold on contention.
        state.reply_p.update({0: .2, 1: .8, 2: .8})
        state.timing_holds['stale'] = NOW
        async with busy(service.lock, process=process):
            reply_timing.prepare(mind, state)
        assert state.reply_p == {} and state.timing_holds == {}
        assert outreach.candidates(state)[1]  # today's outreach rules permit it
        reply_timing.prepare(mind, state)
        assert state.reply_p == {0: .2, 1: .8, 2: .8}
        assert not outreach.candidates(state)[1]  # learned hold works after release
    finally:
        service.close()
        concerns.close()


def seam(case, tmp_path, monkeypatch):
    """Real public entry points and lock-taking methods, tiny post-lock fixtures."""
    lock = StateLock(tmp_path / 'plasticity')
    touched = Mock()
    pin = NS(closure=NS(mode='rebuild', id='learned'), state={})
    binding = NS(store=NS(erasure_pending=lambda: False, revisions=NS(pin=lambda _: pin, base=pin.closure)))
    runtime = NS(lock=lock, enabled=True, loaded={'learner': binding}, find=lambda _: 'learner',
                 config={'mind.budgets': {}}, pin=lambda *a: pin)
    candidate = NS(kind='task', multiplier_keys=lambda: ['task'], extra={}, open_ended=False,
                   topic='', evidence=[], parent_goal_id=None, lesson_ids=[], type='task', drive='curiosity',
                   success_check='', id='row')
    mind = NS(plasticity=runtime, faculties={'plasticity': True})
    driver = binding.driver = NS()
    binding.comparator = NS(records=lambda: [], _path=lambda ref: Path(ref))
    if case in ('reply_sent', 'contact_outcome'):
        module = reply_timing if case == 'reply_sent' else contact_reply_timing
        driver = binding.driver = object.__new__(module.Driver)
        driver.store = NS(name='learner')
        driver.rows = NS(snapshot=lambda: (touched() or [], []))
        driver.clock = lambda: None
        if case == 'reply_sent':
            call = lambda: reply_timing.sent(mind, NS(type=next(iter(reply_timing.OUTREACH_TYPES))))
        else:
            monkeypatch.setattr(contact_reply_timing, '_runtime', lambda: (runtime, 'learner'))
            call = lambda: contact_reply_timing.outcome('row')
        return lock, call, None, None, touched
    if case == 'salience_factor':
        driver = binding.driver = object.__new__(salience.Driver)
        driver.prediction = lambda *a: touched() or {'factor': 1.25}
        driver._pending = {'choice': NS(forced=False)}
        return lock, lambda: salience.factor(runtime, 'learner', candidate, None, context={}), 1., 1.25, touched
    if case == 'salience_observe':
        driver = binding.driver = object.__new__(salience.Driver)
        # Stop after entering the lock by using the existing erasure fallback.
        driver.pass_view = lambda *a: touched() or {'erasure_epoch': 0}
        driver.rows = NS(erasure_epoch=1)
        driver.store = NS(root=lock.root)
        ranking = NS(context={}, eligible=[])
        return lock, lambda: salience.observe(runtime, 'learner', ranking, None), None, None, touched
    if case == 'action_factor':
        driver.rows = NS(frame=lambda *a: (touched() or {}, []))
        driver.learner = NS(predict=lambda *a: np.array([[.1]]))
        return lock, lambda: action_success.factor(runtime, 'learner', candidate, None), 1., .5, touched
    if case == 'action_note':
        driver.store = NS(erasure_pending=lambda: touched() or True)
        return lock, lambda: action_success.note(runtime, 'learner', candidate, None), None, None, touched
    if case == 'contact_probability':
        monkeypatch.setattr(contact_reply_timing, '_runtime', lambda: (runtime, 'learner'))
        driver.rows = NS(ref=lambda fid: touched() or fid)
        binding.comparator.records = lambda: [{'ref_hash': 'row', 'output': .8, 'closure': 'learned'}]
        return lock, lambda: contact_reply_timing.probability('row', {}), (.5, contact_reply_timing.BASE_BASIS), (.8, contact_reply_timing.BASE_BASIS), touched
    if case == 'recall_create':
        driver = binding.driver = object.__new__(query_adapter.Driver)
        driver.queries = NS()
        driver.pending_erasure = lambda: touched() or {}
        call = lambda: query_adapter.RecallRequest.create(runtime, owner=True, turn_id='turn')
        return lock, call, None, query_adapter.RecallRequest, touched
    service = object.__new__(Runtime)
    service.lock, service.enabled, service.loaded = lock, True, runtime.loaded
    service.chains = [NS(name='chain', members=['learner'])]
    service.joints = NS(pin=lambda *a: touched() or {'learner': pin})
    service._publish = lambda: None
    if case == 'runtime_pin':
        return lock, lambda: service.pin('learner', 'owner'), None, pin, touched
    if case == 'runtime_pin_chain':
        return lock, lambda: service.pin_chain('chain', 'owner'), None, {'learner': pin}, touched
    if case == 'runtime_serve':
        binding.source = NS(source=NS())
        binding.comparator.serve = lambda *a, **kw: touched() or 'learned'
        return lock, lambda: service.serve('learner', 'row', {}, predict=lambda *a: 'base'), 'base', 'learned', touched
    if case == 'runtime_outcome':
        binding.comparator.outcome = lambda *a, **kw: touched() or 'recorded'
        return lock, lambda: service.outcome('learner', 'row'), None, 'recorded', touched
    raise AssertionError(case)


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['reply_sent', 'salience_factor', 'salience_observe', 'action_factor',
    'action_note', 'contact_probability', 'contact_outcome', 'recall_create', 'runtime_pin',
    'runtime_pin_chain', 'runtime_serve', 'runtime_outcome'])
async def test_optional_seams_skip_busy_lock_then_resume(tmp_path, monkeypatch, case):
    lock, call, fallback, learned, touched = seam(case, tmp_path, monkeypatch)
    touched.return_value = None
    async with busy(lock):
        result = call()
    assert result == fallback
    touched.assert_not_called()
    result = call()
    if isinstance(learned, type):
        assert isinstance(result, learned)
    else:
        assert result == learned
    touched.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('process', [False, True], ids=['thread', 'process'])
async def test_direct_lock_refuses_wait_and_releases_failed_acquisition(tmp_path, process):
    lock = StateLock(tmp_path)
    async with busy(lock, process=process):
        with pytest.raises(BlockingIOError):
            with lock:
                pytest.fail('entered contended critical section')
    # Distinct objects share a reentrant lease; failed flock must close/reset it.
    with lock:
        with StateLock(tmp_path):
            assert lock._lease.depth == 2
    assert lock._lease.depth == 0 and lock._lease.handle is None


@pytest.mark.asyncio
async def test_worker_still_waits_for_owner(tmp_path):
    lock = StateLock(tmp_path)
    started, entered = threading.Event(), threading.Event()
    def worker():
        started.set()
        with lock:
            entered.set()
    with lock:
        thread = threading.Thread(target=worker)
        thread.start()
        while not started.is_set():
            await asyncio.sleep(.001)
        await asyncio.sleep(.03)
        assert not entered.is_set()
    while not entered.is_set():
        await asyncio.sleep(.001)
    thread.join(timeout=2)


@pytest.mark.asyncio
async def test_salience_does_not_record_a_learned_pass_after_fallback(tmp_path):
    lock = StateLock(tmp_path)
    driver = object.__new__(salience.Driver)
    driver.prediction = Mock(side_effect=AssertionError('must not predict this pass'))
    driver.pass_view = Mock(side_effect=AssertionError('must not observe this pass'))
    runtime = NS(enabled=True, lock=lock, loaded={'salience': NS(driver=driver)})
    context = {}
    candidate = NS(multiplier_keys=lambda: ['task'])
    async with busy(lock):
        assert salience.factor(runtime, 'salience', candidate, None, context=context) == 1.
    assert salience.factor(runtime, 'salience', candidate, None, context=context) == 1.
    salience.observe(runtime, 'salience', NS(context=context), None)
    driver.prediction.assert_not_called()
    driver.pass_view.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['startup', 'recall_attach', 'service_sample'])
async def test_async_lock_work_waits_off_loop(tmp_path, monkeypatch, operation):
    from colonymind.plasticity import nightly
    from colonymind.plasticity.learners import service_health
    from test_plasticity_reply_timing import NOW

    lock = StateLock(tmp_path)
    completed = threading.Event()
    runtime = NS(lock=lock, enabled=False, loaded={}, close=Mock(), _publish=lambda: None)
    def locked_work(*args, **kwargs):
        with lock:
            completed.set()
            return runtime

    if operation == 'startup':
        monkeypatch.setattr(nightly.Runtime, 'from_config', locked_work)
        async def run():
            async with nightly.lifespan({}, data_dir=tmp_path):
                pass
    elif operation == 'recall_attach':
        driver = object.__new__(query_adapter.Driver)
        driver.queries = NS()
        driver.pending_erasure = lambda: completed.set() or {}
        driver.resolve_outcomes = lambda: None
        async def run():
            await driver.live(runtime)
    else:
        driver = object.__new__(service_health.ServiceHealth)
        runtime.enabled, runtime.loaded = True, {'service': None}
        driver.store = NS(name='service')
        driver.clock = lambda: NOW
        driver._pass = locked_work
        driver._live_state = lambda: {'warnings': []}
        async def run():
            await driver.sample(runtime, mind=NS(plasticity=runtime, faculties={}))

    task = None
    try:
        async with busy(lock):
            task = asyncio.create_task(run())
            await asyncio.sleep(.04)
            assert not completed.is_set()
        deadline = time.monotonic() + 3
        while not completed.is_set():
            assert time.monotonic() < deadline
            await asyncio.sleep(.005)
        if operation != 'recall_attach':
            await task
            if operation == 'startup':
                runtime.close.assert_called_once()
    finally:
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['instinct', 'source'])
async def test_required_erasure_propagates_contention_for_retry(tmp_path, path):
    from colonymind.instinct.log import Store, bind_erasure
    from colonymind.turns.erasure_hooks import bind_source_erasure, notify_source_erasure

    service = object.__new__(Runtime)
    service.lock = StateLock(tmp_path / 'plasticity')
    entered = Mock(side_effect=ValueError('entered erasure'))
    if path == 'instinct':
        service._recover_moves = entered
        store = Store(tmp_path / 'instinct.db')
        unbind = bind_erasure(store.path, service.erase)
        call = lambda: store.forget(['decision'])
    else:
        service._resume_source_erasures = entered
        unbind = bind_source_erasure(service.erase_sources)
        call = lambda: notify_source_erasure(['source'])
    try:
        async with busy(service.lock):
            with pytest.raises(BlockingIOError):
                call()
        entered.assert_not_called()
        with pytest.raises(ValueError, match='entered erasure'):
            call()
        entered.assert_called_once()
    finally:
        unbind()


@pytest.mark.asyncio
async def test_worker_waits_for_other_process(tmp_path):
    lock = StateLock(tmp_path)
    entered = threading.Event()
    def worker():
        with lock:
            entered.set()
    async with busy(lock, process=True):
        thread = threading.Thread(target=worker)
        thread.start()
        await asyncio.sleep(.03)
        assert not entered.is_set()
    deadline = time.monotonic() + 3
    while not entered.is_set():
        assert time.monotonic() < deadline
        await asyncio.sleep(.005)
    thread.join(timeout=2)
    assert not thread.is_alive()


@pytest.mark.asyncio
async def test_cancelled_startup_drains_worker_and_closes_runtime(tmp_path, monkeypatch):
    from colonymind.plasticity import nightly
    lock = StateLock(tmp_path)
    runtime = NS(close=Mock())
    started = threading.Event()
    def initialize(*a, **kw):
        started.set()
        with lock:
            return runtime
    monkeypatch.setattr(Runtime, 'from_config', initialize)
    async def run():
        async with nightly.lifespan({}, data_dir=tmp_path):
            pytest.fail('cancelled startup yielded')
    async with busy(lock):
        task = asyncio.create_task(run())
        while not started.is_set():
            await asyncio.sleep(.001)
        task.cancel()
        await asyncio.sleep(.02)
        assert not task.done()
    with pytest.raises(asyncio.CancelledError):
        await task
    runtime.close.assert_called_once()
