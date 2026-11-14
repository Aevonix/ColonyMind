"""Shutdown waits for threaded maintenance before closing its shared runtime."""

import asyncio
from datetime import datetime, timezone
from threading import Event
from types import SimpleNamespace

import pytest

from colonymind.plasticity import nightly

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)
POLL = 0.01  # Test assumption: frequent scheduling also avoids relying on cross-thread loop wakeups.
DEADLINE = 2.0  # Test assumption: a blocked worker must be released within two seconds.


async def _until(predicate):
    limit = asyncio.get_running_loop().time() + DEADLINE
    while not predicate():
        assert asyncio.get_running_loop().time() < limit, "the worker did not reach the expected state"
        await asyncio.sleep(POLL)


async def _cancel_sleep(seconds):
    raise asyncio.CancelledError


@pytest.mark.asyncio
async def test_lifespan_does_not_close_runtime_while_the_nightly_worker_is_still_running(tmp_path, monkeypatch):
    started, release, finished = Event(), Event(), Event()
    closed = []
    runtime = SimpleNamespace(enabled=True, loaded={"probe": object()}, _clock=lambda: NOW,
                              close=lambda: closed.append(finished.is_set()))
    monkeypatch.setattr(nightly.Runtime, "from_config", lambda *args, **kwargs: runtime)

    def tick(self, now):
        started.set()
        assert release.wait(DEADLINE)
        finished.set()

    monkeypatch.setattr(nightly.NightlyTask, "tick", tick)
    leave = asyncio.Event()
    mind = SimpleNamespace(tz="UTC", consolidation=SimpleNamespace(boundary_minute=lambda: 0))

    async def lifespan():
        async with nightly.lifespan({}, data_dir=tmp_path, mind=mind):
            await leave.wait()

    task = asyncio.create_task(lifespan())
    try:
        await _until(started.is_set)
        leave.set()
        await asyncio.sleep(POLL)
        assert not closed
        assert not task.done()
    finally:
        leave.set()
        release.set()
        await _until(task.done)
        await task
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, ValueError("fixture worker detail")])
async def test_repeated_cancellation_waits_for_worker_and_preserves_cancelled_error(monkeypatch, caplog, failure):
    started, release, finished = Event(), Event(), Event()
    task = nightly.NightlyTask(SimpleNamespace(_clock=lambda: NOW))

    def tick(now):
        started.set()
        assert release.wait(DEADLINE)
        finished.set()
        if failure is not None:
            raise failure

    monkeypatch.setattr(task, "tick", tick)
    running = asyncio.create_task(task.run())
    try:
        await _until(started.is_set)
        running.cancel()
        await asyncio.sleep(POLL)
        assert not running.done()
        running.cancel()
        await asyncio.sleep(POLL)
        assert not running.done()
    finally:
        release.set()
        await _until(running.done)
        with pytest.raises(asyncio.CancelledError):
            await running
    assert finished.is_set()
    if failure is not None:
        assert "plasticity maintenance failed (ValueError)" in caplog.text
        assert "fixture worker detail" not in caplog.text


@pytest.mark.asyncio
async def test_worker_failures_keep_the_existing_log_and_loop_behavior(monkeypatch, caplog):
    task = nightly.NightlyTask(SimpleNamespace(_clock=lambda: NOW))

    def tick(now):
        raise RuntimeError("fixture worker detail")

    monkeypatch.setattr(task, "tick", tick)
    running = asyncio.create_task(task.run(sleep=_cancel_sleep))
    await _until(running.done)
    with pytest.raises(asyncio.CancelledError):
        await running
    assert "plasticity maintenance failed (RuntimeError)" in caplog.text
    assert "fixture worker detail" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_off_and_unconfigured_runtime_remain_idle(enabled, caplog):
    task = nightly.NightlyTask(SimpleNamespace(enabled=enabled, loaded={}, _clock=lambda: NOW))
    running = asyncio.create_task(task.run(sleep=_cancel_sleep))
    await _until(running.done)
    with pytest.raises(asyncio.CancelledError):
        await running
    assert not caplog.records
