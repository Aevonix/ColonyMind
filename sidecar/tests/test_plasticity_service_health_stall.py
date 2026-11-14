"""The five-minute service-health pass keeps the sidecar's event loop serving."""
import asyncio
import gc
import os
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest

from colonymind.plasticity import _io
from colonymind.plasticity.learners import service_health
from colonymind.plasticity.locking import StateLockBusy
from test_plasticity_service_health import NOW, calls, runtime


def _frames(series, start, count):
    return [({"at": (start + timedelta(seconds=300 * (i + 1))).isoformat(),
              "previous_at": (start + timedelta(seconds=300 * i)).isoformat(),
              "streams": {"value": [float(i % 7)] * 12}, "series": series}, float(i % 5)) for i in range(count)]


def _key_reads(monkeypatch):
    reads = []
    original = Path.read_bytes

    def counted(path):
        if path.name == "comparator.key":
            reads.append(path)
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", counted)
    return reads


@pytest.mark.hardware_latency
async def test_pass_runs_off_the_event_loop_while_a_large_sample_file_is_processed(tmp_path, monkeypatch, hardware_latency):
    now = [NOW]
    service = runtime(tmp_path, now)
    try:
        driver = service.loaded["health"].driver
        start = NOW - timedelta(days=13)
        pairs = [pair for n in range(6) for pair in _frames("router.unit%d.latency_p95_ms" % n, start, 3500)]
        driver.samples.add_many(pairs)
        assert len(driver.samples._read()) == 21000
        calls(monkeypatch, tmp_path, now[0] - timedelta(seconds=1))
        await driver.sample(service)
        now[0] += timedelta(seconds=300)
        calls(monkeypatch, tmp_path, now[0] - timedelta(seconds=1), latency=200)
        measured = service_health.collect
        loop_thread = threading.get_ident()

        def slow_collect(*args, **kwargs):
            assert threading.get_ident() != loop_thread
            time.sleep(.5)  # A blocking walk that would freeze every caller if it ran on the loop.
            return measured(*args, **kwargs)
        monkeypatch.setattr(service_health, "collect", slow_collect)
        ticks, gaps, cpu_gaps, running = [], [], [], [True]
        # The 21,000 samples above make one full collection cost about 240 ms, a
        # process-wide pause no offload avoids; freeze them so the bar times the pass.
        gc.collect()
        gc.freeze()

        async def ticker():
            last = time.monotonic()
            last_cpu = time.thread_time()
            while running[0]:
                await asyncio.sleep(.01)
                cpu = time.thread_time()
                ticks.append(time.monotonic())
                gaps.append(ticks[-1] - last)
                cpu_gaps.append(cpu - last_cpu)
                last = ticks[-1]
                last_cpu = cpu
        task = asyncio.create_task(ticker())
        began = time.monotonic()
        await driver.sample(service, offload=service_health.off_loop)
        took = time.monotonic() - began
        running[0] = False
        await task
        assert took >= .5
        assert ticks
        assert driver._live_state()["at"] == now[0].isoformat()
        assert len(driver.samples._read()) == 21004
        hardware_latency([gap * 1000 for gap in gaps], [gap * 1000 for gap in cpu_gaps],
                         threshold_ms=250, statistic="max", enforced=os.environ.get("GITHUB_ACTIONS") != "true")
        if os.environ.get("GITHUB_ACTIONS") != "true":
            assert len(ticks) >= 20
    finally:
        gc.unfreeze()
        service.close()


async def test_live_offloads_each_pass(tmp_path, monkeypatch):
    now = [NOW]
    service = runtime(tmp_path, now)
    try:
        driver = service.loaded["health"].driver
        seen = []

        async def sample(runtime, **kwargs):
            seen.append(kwargs.get("offload"))
            runtime.enabled = False

        async def no_wait(seconds):
            assert 0 < seconds <= 300
        monkeypatch.setattr(driver, "sample", sample)
        monkeypatch.setattr("colonymind.plasticity.learners.service_health.asyncio.sleep", no_wait)
        await driver.live(service)
        assert seen == [service_health.off_loop]
    finally:
        service.close()


async def test_cancelled_caller_waits_for_the_running_pass():
    finished = []

    def work():
        time.sleep(.2)
        finished.append(True)
    task = asyncio.create_task(service_health.off_loop(work))
    await asyncio.sleep(.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert finished == [True]


async def test_comparator_key_is_read_once_per_pass(tmp_path, monkeypatch):
    now = [NOW]
    service = runtime(tmp_path, now)
    try:
        driver = service.loaded["health"].driver
        driver.samples.add_many(_frames("router.primary.call_count", NOW - timedelta(days=2), 400))
        calls(monkeypatch, tmp_path, now[0] - timedelta(seconds=1))
        await driver.sample(service)
        now[0] += timedelta(seconds=300)
        calls(monkeypatch, tmp_path, now[0] - timedelta(seconds=1), latency=200)
        reads = _key_reads(monkeypatch)
        driver._pass(service, now[0])
        assert len(reads) == 1
        assert len(driver.samples._read()) == 404  # The four labelled series joined the retained rows.
        reads.clear()
        assert len(list(driver.source.rows(driver.objective))) == 404 and len(reads) == 1
    finally:
        service.close()


def test_retention_keeps_fourteen_days_byte_compatible_and_skips_unchanged_rewrites(tmp_path):
    now = [NOW]
    service = runtime(tmp_path, now)
    try:
        samples = service.loaded["health"].driver.samples
        old = _frames("router.primary.error_share", NOW - timedelta(days=15), 3)
        kept = _frames("router.primary.error_share", NOW - timedelta(days=14) + timedelta(minutes=1), 3)
        samples.add_many(old + kept)
        assert [row["at"] for row in samples._read()] == [frame["at"] for frame, _ in kept]
        fresh = _frames("router.other.error_share", NOW - timedelta(hours=1), 2)
        samples.add_many(fresh)
        expected = [{"frame": frame, "label": float(label)} for frame, label in kept + fresh]
        assert samples.path.read_bytes() == _io.canonical(expected)
        before = os.stat(samples.path)
        samples.add_many([])
        samples.add_many(fresh)
        after = os.stat(samples.path)
        assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
        now[0] += timedelta(minutes=8)
        samples.add_many([])
        assert [row["at"] for row in samples._read()] == [frame["at"] for frame, _ in kept[1:] + fresh]
    finally:
        service.close()


@pytest.mark.hardware_latency
async def test_loop_side_lock_users_skip_busy_steps_without_waiting(tmp_path, monkeypatch, hardware_latency):
    now = [NOW]
    service = runtime(tmp_path, now)
    try:
        driver = service.loaded["health"].driver
        driver.samples.add_many([pair for n in range(4) for pair in _frames(
            "router.unit%d.call_count" % n, NOW - timedelta(days=12), 3000)])
        calls(monkeypatch, tmp_path, now[0] - timedelta(seconds=1))
        await driver.sample(service)
        now[0] += timedelta(seconds=300)
        calls(monkeypatch, tmp_path, now[0] - timedelta(seconds=1), latency=200)
        read = service_health.Samples._read
        held = []

        def take_lock():
            with service.lock:
                pass

        def slow_read(self, key=None):
            # Reading and keying a large retained file. During the pass's snapshot read a probe
            # thread takes the state lock: a pass holding it across that read keeps the probe
            # blocked for the whole 0.4 s join on any runner, so the check needs no timing bar.
            if sys._getframe(1).f_code.co_name != "snapshot":
                time.sleep(.5)
                return read(self, key)
            began = time.monotonic()
            probe = threading.Thread(target=take_lock, daemon=True)
            probe.start()
            probe.join(.4)
            held.append(probe.is_alive())
            time.sleep(max(0, .5 - (time.monotonic() - began)))
            return read(self, key)
        monkeypatch.setattr(service_health.Samples, "_read", slow_read)
        stalls, cpu_stalls, running = [], [], [True]

        async def recall():  # Loop-side code that takes the state lock synchronously, as requests do.
            while running[0]:
                began = time.monotonic()
                began_cpu = time.thread_time()
                try:
                    with service.lock:
                        pass
                except StateLockBusy:
                    pass  # The request uses its base path for this pass.
                stalls.append(time.monotonic() - began)
                cpu_stalls.append(time.thread_time() - began_cpu)
                await asyncio.sleep(.01)
        task = asyncio.create_task(recall())
        began = time.monotonic()
        await driver.sample(service, offload=service_health.off_loop)
        took = time.monotonic() - began
        running[0] = False
        await task
        assert took >= .5 and stalls
        assert held and not any(held)
        assert driver._live_state()["at"] == now[0].isoformat()
        assert len(driver.samples._read()) == 12004
        hardware_latency([stall * 1000 for stall in stalls], [stall * 1000 for stall in cpu_stalls],
                         threshold_ms=250, statistic="max", enforced=os.environ.get("GITHUB_ACTIONS") != "true")
        if os.environ.get("GITHUB_ACTIONS") != "true":
            assert len(stalls) >= 10
    finally:
        service.close()


def test_commit_rereads_samples_changed_after_the_snapshot(tmp_path):
    now = [NOW]
    service = runtime(tmp_path, now)
    try:
        samples = service.loaded["health"].driver.samples
        kept = _frames("router.primary.latency_p95_ms", NOW - timedelta(hours=3), 3)
        samples.add_many(kept)
        snapshot = samples.snapshot()
        gone = samples.reference(kept[0][0])
        samples.erase([gone])
        samples.add_many(_frames("router.other.latency_p95_ms", NOW - timedelta(hours=1), 1), snapshot=snapshot)
        refs = [row["ref"] for row in samples._read()]
        assert gone not in refs and len(refs) == 3
    finally:
        service.close()


async def test_outcome_erased_during_the_pass_is_neither_labelled_nor_kept(tmp_path, monkeypatch):
    now = [NOW]
    service = runtime(tmp_path, now)
    try:
        driver = service.loaded["health"].driver
        comparator = service.loaded["health"].comparator
        calls(monkeypatch, tmp_path, now[0] - timedelta(seconds=1))
        await driver.sample(service)
        pending = [series["pending"]["frame"] for series in driver._live_state()["series"].values()]
        now[0] += timedelta(seconds=300)
        calls(monkeypatch, tmp_path, now[0] - timedelta(seconds=1), latency=200)
        records, erased = comparator.records, []

        def records_then_erase():
            found = records()
            if not erased:  # An erasure lands after the pass has read the served items.
                erased.append(driver.samples.reference(pending[0]))
                service.erase(erased)
            return found
        monkeypatch.setattr(comparator, "records", records_then_erase)
        driver._pass(service, now[0])
        monkeypatch.setattr(comparator, "records", records)
        refs = {row["ref"] for row in driver.samples._read()}
        assert erased[0] not in refs and len(refs) == len(pending) - 1
        labelled = [r for r in comparator.records() if r["outcome"] is not None]
        assert len(labelled) == len(pending) - 1
        assert driver._live_state()["at"] == now[0].isoformat()
    finally:
        service.close()


def test_directory_size_skips_in_flight_temporary_files(tmp_path):
    (tmp_path / "kept.bin").write_bytes(b"x" * 100)
    (tmp_path / ".numeric-samples.json.0123abcd.tmp").write_bytes(b"x" * 5000)
    (tmp_path / "report.json.new").write_bytes(b"x" * 5000)
    gate = tmp_path / "plasticity" / "numeric-gate-abc"
    gate.mkdir(parents=True)
    (gate / "revision").write_bytes(b"x" * 5000)
    result = service_health.collect(tmp_path, since=NOW - timedelta(seconds=300), until=NOW, key=b"fixture key")
    assert result["store.bytes"]["value"] == 100.
