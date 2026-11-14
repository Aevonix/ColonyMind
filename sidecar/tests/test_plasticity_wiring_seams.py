"""Public learner seams, using CPU fixtures and no external services.

Row counts and async deadlines are test assumptions, not product settings.
"""
import asyncio
import sqlite3
import subprocess
import sys

import pytest

from colonymind.initiatives.store import InitiativeStore
from colonymind.mind import Mind
from colonymind.plasticity import nightly
from colonymind.plasticity.comparator import LiveComparator
from colonymind.plasticity.synthetic import SyntheticSource, make_rows
from colonymind.turns import TurnIdempotencyLedger
from test_plasticity_wiring_registration import declaration as checked_declaration
from test_plasticity_wiring_runtime import NOW, config


def declaration(**changes):
    return checked_declaration(**{"dependents": ["threshold"], **changes})


def runtime(tmp_path, declarations=None, **kwargs):
    cfg = config(tmp_path, declarations or {"probe": declaration()})
    return nightly.Runtime(cfg, clock=lambda: NOW, **kwargs)


def test_package_loads_public_modules_in_order_once_before_resolving_factories(tmp_path):
    from colonymind.plasticity import learners

    assert callable(learners.load)
    modules = tmp_path / "modules"
    modules.mkdir()
    for name in ("zeta", "alpha"):
        (modules / f"{name}.py").write_text(
            "from colonymind.plasticity.nightly import register_factory, _synthetic\n"
            f"register_factory('{name}', _synthetic)\n"
        )
    (modules / "_private.py").write_text("raise AssertionError('private module imported')\n")
    cfg = config(tmp_path, {"probe": declaration(implementation="alpha", base="fixture@1")})
    script = """
import sys
from colonymind.config import load_config
from colonymind.plasticity import learners, nightly
learners.__path__ = [sys.argv[1]]
registered = []
register = nightly.register_factory
def record(name, factory):
    registered.append(name)
    register(name, factory)
nightly.register_factory = record
runtime = nightly.Runtime(load_config(sys.argv[2], environ={}))
try:
    assert list(runtime.loaded) == ['probe'], runtime.rejected
    learners.load()
    learners.load()
    assert registered == ['alpha', 'zeta'], registered
finally:
    runtime.close()
"""
    result = subprocess.run([sys.executable, "-c", script, str(modules), str(cfg.home)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_builtin_synthetic_implementation_loads_many_times(tmp_path):
    spec = declaration(implementation="synthetic", base="fixture@1")
    service = runtime(tmp_path, {"first": spec, "second": spec})
    try:
        assert list(service.loaded) == ["first", "second"] and not service.rejected
        assert service.find("synthetic") == "first"
    finally:
        service.close()


@pytest.mark.parametrize("implementation", ["fixture"])
def test_find_and_duplicate_implementation_refusal_are_named_in_report(tmp_path, implementation):
    spec = declaration(implementation=implementation, base="fixture@1")
    service = runtime(tmp_path, {"first": spec, "second": spec},
                      factories={"fixture": nightly._synthetic})
    try:
        assert list(service.loaded) == ["first"]
        assert service.find(implementation) == "first"
        assert service.find("missing") is None
        report = service.report()
        assert "second" in report["rejected"]
        assert implementation in report["rejected"]["second"]
        assert "first" in report["rejected"]["second"]
    finally:
        service.close()


def test_disabled_learner_does_not_reserve_an_implementation(tmp_path):
    service = runtime(tmp_path, {"first": declaration(enabled=False), "second": declaration()})
    try:
        assert service.find("synthetic") == "second"
        assert not service.rejected
    finally:
        service.close()


def test_comparator_dispatches_by_registered_kind_and_defaults_to_live(tmp_path, monkeypatch):
    seen = []

    class KindComparator(LiveComparator):
        def __init__(self, controller, **kwargs):
            seen.append(kwargs)
            super().__init__(controller, **kwargs)

    monkeypatch.setitem(nightly.COMPARATORS, "supervised-refit", KindComparator)
    service = runtime(tmp_path)
    try:
        assert type(service.loaded["probe"].comparator) is KindComparator
        assert {"key", "lock", "clock", "bar", "pin", "rollback"} == set(seen[0])
    finally:
        service.close()
    monkeypatch.delitem(nightly.COMPARATORS, "supervised-refit")
    service = runtime(tmp_path)
    try:
        assert type(service.loaded["probe"].comparator) is LiveComparator
    finally:
        service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_mind_handle_defaults_to_none_and_is_owned_by_lifespan(tmp_path, enabled):
    store = InitiativeStore(state_dir=tmp_path)
    mind = Mind(config={}, store=store, state_dir=tmp_path, owner_id=None, backups=False)
    cfg = config(tmp_path)
    cfg.data["mind"]["faculties"]["plasticity"] = enabled
    try:
        assert mind.plasticity is None
        async with nightly.lifespan(cfg, data_dir=tmp_path, mind=mind) as service:
            assert mind.plasticity is service
        assert mind.plasticity is None
    finally:
        store.close()


@pytest.mark.asyncio
async def test_live_runs_restarts_after_sixty_seconds_and_cancels_at_shutdown(tmp_path, monkeypatch, caplog):
    started, restarted, waiting, release, cancelled = (asyncio.Event() for _ in range(5))
    calls, delays = [], []

    async def live(service):
        calls.append(service)
        if len(calls) == 1:
            started.set()
            raise ValueError("private fixture error detail")
        restarted.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    async def sleep(seconds):
        delays.append(seconds)
        waiting.set()
        await release.wait()

    async def maintenance(task):
        await asyncio.Future()

    service = runtime(tmp_path)
    monkeypatch.setattr(service.loaded["probe"].driver, "live", live, raising=False)
    monkeypatch.setattr(nightly.Runtime, "from_config", lambda *args, **kwargs: service)
    monkeypatch.setattr(nightly.NightlyTask, "run", maintenance)
    monkeypatch.setattr(nightly.asyncio, "sleep", sleep)
    async with nightly.lifespan(service.config, data_dir=tmp_path) as loaded:
        await asyncio.wait_for(started.wait(), 2)
        await asyncio.wait_for(waiting.wait(), 2)
        assert delays == [60]
        assert calls == [loaded]
        assert not restarted.is_set()
        release.set()
        await asyncio.wait_for(restarted.wait(), 2)
        assert calls == [loaded, loaded]
    assert cancelled.is_set()
    assert "ValueError" in caplog.text
    assert "private fixture error detail" not in caplog.text


@pytest.mark.asyncio
async def test_live_faculty_off_and_shutdown_during_restart_delay(tmp_path, monkeypatch):
    waiting, cancelled = asyncio.Event(), asyncio.Event()
    calls = []
    yield_to_tasks = asyncio.sleep

    async def live(service):
        calls.append(service)
        raise RuntimeError("fixture detail")

    async def sleep(seconds):
        waiting.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    async def maintenance(task):
        await asyncio.Future()

    service = runtime(tmp_path)
    monkeypatch.setattr(service.loaded["probe"].driver, "live", live, raising=False)
    monkeypatch.setattr(nightly.Runtime, "from_config", lambda *args, **kwargs: service)
    monkeypatch.setattr(nightly.NightlyTask, "run", maintenance)
    monkeypatch.setattr(nightly.asyncio, "sleep", sleep)
    service.enabled = False
    async with nightly.lifespan(service.config, data_dir=tmp_path):
        await yield_to_tasks(0)
        assert calls == []
    service.enabled = True
    async with nightly.lifespan(service.config, data_dir=tmp_path):
        await asyncio.wait_for(waiting.wait(), 2)
    assert cancelled.is_set()
    assert calls == [service]


@pytest.mark.asyncio
async def test_live_starts_each_optional_driver_once_and_all_stop_before_close(tmp_path, monkeypatch):
    service = runtime(tmp_path, {
        "first": declaration(),
        "second": declaration(implementation="other", base="fixture@1"),
        "plain": declaration(implementation="plain", base="fixture@1"),
    }, factories={"other": nightly._synthetic, "plain": nightly._synthetic})
    started, finished = set(), set()
    ready = asyncio.Event()
    closed = []

    def loop(name):
        async def live(loaded):
            assert loaded is service
            assert name not in started
            started.add(name)
            if len(started) == 2:
                ready.set()
            if name == "first":
                finished.add(name)
                return
            try:
                await asyncio.Future()
            finally:
                finished.add(name)
        return live

    async def maintenance(task):
        await asyncio.Future()

    close = service.close

    def close_after_tasks():
        closed.append(set(finished))
        close()

    for name in ("first", "second"):
        monkeypatch.setattr(service.loaded[name].driver, "live", loop(name), raising=False)
    monkeypatch.setattr(service, "close", close_after_tasks)
    monkeypatch.setattr(nightly.Runtime, "from_config", lambda *args, **kwargs: service)
    monkeypatch.setattr(nightly.NightlyTask, "run", maintenance)
    async with nightly.lifespan(service.config, data_dir=tmp_path):
        await asyncio.wait_for(ready.wait(), 2)
    assert started == finished == {"first", "second"}
    assert closed == [finished]


def test_runtime_collects_all_row_sources_and_erases_once(tmp_path, monkeypatch):
    calls = []

    class RestingRows(SyntheticSource):
        def __init__(self, ref):
            super().__init__()
            self.ref = ref

        def refs_resting_on(self, ids):
            calls.append(tuple(ids))
            return [self.ref, "row-shared"]

    service = runtime(tmp_path, {
        "first": declaration(),
        "second": declaration(implementation="other", base="fixture@1"),
        "plain": declaration(implementation="plain", base="fixture@1"),
    }, factories={"other": nightly._synthetic, "plain": nightly._synthetic},
        sources={"first": RestingRows("row-a"), "second": RestingRows("row-b"), "plain": SyntheticSource()})
    erased = []
    monkeypatch.setattr(service, "erase", lambda refs: erased.append(tuple(refs)))
    try:
        service.erase_sources(iter(["turn-a"]))
        assert calls == [("turn-a",), ("turn-a",)]
        assert len(erased) == 1
        assert set(erased[0]) == {"row-a", "row-b", "row-shared"}
    finally:
        service.close()
