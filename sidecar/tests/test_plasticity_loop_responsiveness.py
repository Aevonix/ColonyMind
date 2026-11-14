"""Live learner lock waits leave the event loop available."""

import asyncio
from contextlib import suppress
import threading
import time
from types import SimpleNamespace

import pytest

from colonymind.plasticity.learners import action_success, contact_reply_timing, reply_timing, salience


@pytest.mark.asyncio
@pytest.mark.parametrize('module', [reply_timing, contact_reply_timing, salience, action_success])
async def test_live_poll_does_not_block_loop_behind_runtime_lock(module):
    lock = threading.RLock()
    held = threading.Event()
    finished = threading.Event()

    def hold_lock():
        with lock:
            held.set()
            time.sleep(2)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    await asyncio.to_thread(held.wait)
    comparator = SimpleNamespace(records=lambda: [])
    runtime = SimpleNamespace(enabled=True, find=lambda name: 'learner', lock=lock,
                              loaded={'learner': SimpleNamespace(comparator=comparator)})
    driver = object.__new__(module.Driver)
    driver.store = SimpleNamespace(name='learner', erasure_pending=lambda: False)
    driver.rows = SimpleNamespace(snapshot=lambda: ([], []) if module in (reply_timing, contact_reply_timing) else [],
                                  observe=lambda: None)
    driver.clock, driver.started = (lambda: None), None    # contact_reply_timing's pass serves new sends first
    original_poll = driver.poll

    def poll(runtime):
        try:
            original_poll(runtime)
        finally:
            finished.set()

    driver.poll = poll
    loop = asyncio.get_running_loop()
    fired = loop.create_future()
    started = loop.time()
    task = asyncio.create_task(driver.live(runtime))
    timer = loop.call_later(.05, lambda: fired.set_result(loop.time() - started))
    try:
        assert await fired < .1
        assert not finished.is_set()
    finally:
        timer.cancel()
        await asyncio.to_thread(holder.join)
        await asyncio.to_thread(finished.wait, 1)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
