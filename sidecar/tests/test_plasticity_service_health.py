"""Service samples, durable pairing, and the warn-only senses boundary."""
import copy
import json
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from colonymind import usage
from test_plasticity_wiring_runtime import learner_config as load_config
from colonymind.plasticity import _io
from colonymind.plasticity.gates import Check, GateReport
from colonymind.plasticity.learners.service_health import DECLARATION, collect
from colonymind.plasticity.nightly import Runtime
from colonymind.plasticity.numeric_comparator import NumericComparator
from colonymind.plasticity.registration import check_learner
from colonymind.reports.plasticity import lines
from test_mind_loop import pinned_clock  # noqa: F401

NOW = datetime(2026, 1, 10, 12, tzinfo=timezone.utc)


def runtime(tmp_path, now, **changes):
    declaration = copy.deepcopy(DECLARATION)
    declaration.update(changes)
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump({"plasticity": {"learners": {"health": declaration}}}))
    return Runtime(load_config(tmp_path, environ={}), clock=lambda: now[0])


def calls(monkeypatch, directory, at, *, count=5, latency=100, endpoint="primary", client="router", error=False):
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(directory))
    monkeypatch.setattr(usage, "now_utc", lambda: at)
    for _ in range(count):
        usage.record(client=client, task="fixture words", role="private-model-name", endpoint=endpoint,
                     latency_ms=latency, outcome="error" if error else "ok")
    assert usage.flush()


def test_declaration_loads_real_factory_and_numeric_comparator(tmp_path):
    assert check_learner("health", DECLARATION).kind == "numeric-online"
    service = runtime(tmp_path, [NOW])
    try:
        assert not service.rejected
        assert service.find("service_health") == "health"
        assert isinstance(service.loaded["health"].comparator, NumericComparator)
        assert service.loaded["health"].comparator.bar == .02
    finally:
        service.close()


def test_usage_streams_aliases_sparse_intervals_and_no_external_reads(tmp_path, monkeypatch):
    calls(monkeypatch, tmp_path, NOW - timedelta(seconds=1), endpoint="https://private.invalid/model", latency=123)
    calls(monkeypatch, tmp_path, NOW - timedelta(seconds=1), client="embedder", latency=25)
    calls(monkeypatch, tmp_path, NOW - timedelta(seconds=1), endpoint="sparse", count=4)
    (tmp_path / "outside-link").symlink_to("/etc/passwd")
    result = collect(tmp_path, since=NOW - timedelta(seconds=300), until=NOW, key=b"fixture key")
    assert len(result) == 7
    assert result["embedder.primary.latency_p95_ms"]["value"] == 25.
    assert result["embedder.primary.call_count"]["value"] == 5.
    assert result["embedder.primary.error_share"]["value"] == 0.
    assert result["store.bytes"]["value"] > 0
    assert all("sparse" not in stream for stream in result)
    assert "private.invalid" not in json.dumps(result)
    # A linked database must not let the sampler leave its own state directory.
    (tmp_path / "usage.db").unlink()
    (tmp_path / "usage.db").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="state directory"):
        collect(tmp_path, since=NOW - timedelta(seconds=300), until=NOW, key=b"fixture key")


async def test_forecasts_pair_next_interval_restart_and_off_arm(tmp_path, monkeypatch):
    now = [NOW]
    service = runtime(tmp_path, now)
    driver = service.loaded["health"].driver
    calls(monkeypatch, tmp_path, now[0] - timedelta(seconds=1), endpoint="https://private.invalid/model")
    await driver.sample(service)
    assert len(service.loaded["health"].comparator.records()) == 4
    assert all(row["outcome"] is None for row in service.loaded["health"].comparator.records())
    await driver.sample(service)
    assert len(service.loaded["health"].comparator.records()) == 4
    service.close()
    service = runtime(tmp_path, now)
    driver = service.loaded["health"].driver
    now[0] += timedelta(seconds=300)
    calls(monkeypatch, tmp_path, now[0] - timedelta(seconds=1), endpoint="https://private.invalid/model", latency=200)
    await driver.sample(service)
    records = service.loaded["health"].comparator.records()
    assert len(records) == 8 and sum(r["outcome"] is not None for r in records) == 4
    assert len(list(driver.source.rows(driver.objective))) == 4
    for path in service.root.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            assert all(text not in data for text in (b"private.invalid", b"private-model-name", b"fixture words"))
    before = {path: path.read_bytes() for path in service.root.rglob("*") if path.is_file()}
    service.enabled = False
    now[0] += timedelta(seconds=300)
    await driver.sample(service)
    assert {path: path.read_bytes() for path in service.root.rglob("*") if path.is_file()} == before
    service.close()
    cfg = load_config(tmp_path, environ={})
    cfg.data["mind"]["faculties"]["plasticity"] = False
    off = Runtime(cfg, clock=lambda: now[0])
    assert off.loaded == {} and off.run() == []
    off.close()


async def test_gap_does_not_label_old_forecast_or_join_warning_streak(tmp_path, monkeypatch):
    now = [NOW]
    service = runtime(tmp_path, now)
    try:
        driver = service.loaded["health"].driver
        calls(monkeypatch, tmp_path, now[0] - timedelta(seconds=1))
        await driver.sample(service)
        now[0] += timedelta(seconds=900)
        calls(monkeypatch, tmp_path, now[0] - timedelta(seconds=1), latency=300)
        await driver.sample(service)
        assert all(row["outcome"] is None for row in service.loaded["health"].comparator.records())
        assert not list(driver.source.rows(driver.objective))
    finally:
        service.close()


async def test_training_refs_erase_the_same_numeric_comparator_items(tmp_path, monkeypatch):
    now = [NOW]
    service = runtime(tmp_path, now)
    try:
        driver = service.loaded["health"].driver
        for _ in range(2):
            calls(monkeypatch, tmp_path, now[0] - timedelta(seconds=1))
            await driver.sample(service)
            now[0] += timedelta(seconds=300)
        rows = list(driver.source.rows(driver.objective))
        comparator = service.loaded["health"].comparator
        hashes = {comparator._path(row["ref"]).stem for row in rows}
        assert hashes <= {r["ref_hash"] for r in comparator.records() if r["outcome"] is not None}
        assert service.erase([row["ref"] for row in rows])["clean"]
        assert not hashes & {r["ref_hash"] for r in comparator.records()}
        assert not list(driver.source.rows(driver.objective))
    finally:
        service.close()


@pytest.mark.parametrize("senses", [True, False])
async def test_latency_step_posts_one_note_per_day_and_fixed_arm_lead(tmp_path, monkeypatch, senses):
    from test_mind_loop import Fixture, OWNER
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fx = Fixture(tmp_path, config={"faculties": {"outreach": False, "senses": senses}})
    now = [fx.now]
    service = runtime(tmp_path, now)
    fx.mind.plasticity = service
    driver = service.loaded["health"].driver
    try:
        # A qualified mean-reverting revision gives persistence a live comparator.
        state = driver.learner.base()
        state["W"][0, 0] = .5
        controller = service.loaded["health"].controller
        closure = controller.store.revisions.compose(state=_io.state_digest(state), objective=driver.objective.identity,
            calibration=("numeric-fixture",), dependents={}, parent=controller.revisions.base.id,
            manifests=(), periods=(), audience="owner", mode="stack")
        controller.propose(closure, state)
        for i in range(38):
            if i == 34:
                controller.consider(closure.id, GateReport(closure.id, "fixture", (Check("fixture", True),)))
            now[0] = fx.now
            calls(monkeypatch, tmp_path, now[0] - timedelta(seconds=1), latency=100 if i < 34 else 1000,
                  error=i >= 34)
            await driver.sample(service, mind=fx.mind)
            fx.shift(minutes=5)
        for _ in range(3):
            await fx.mind.tick(force=True)
        health = service.comparator("health")["health"]
        warnings = driver.warning_records()
        latency = [r for r in warnings if r["subject"] == "router.primary" and r["feature"] == "latency_p95_ms"]
        assert len(latency) == 1
        assert latency[0]["fixed_at"] is not None
        assert latency[0]["lead_seconds"] == 300.
        if senses:
            notes = [c for c in fx.mind.concerns.open() if c.detail.get("feature") == "latency_p95_ms"]
            assert len(notes) == 1 and notes[0].detail["note_only"]
            assert health["notes"] >= 1
            misses = [p for p in fx.expectations.store.resolved_since(0, include_notes=True)
                      if p.subject == "router.primary" and p.detail.get("feature") == "latency_p95_ms"]
            assert len(misses) == 1 and misses[0].source_kind == "numeric_surprise"
        else:
            assert not fx.mind.concerns.open() and health["ignored"] >= 1
        assert fx.mind.dispatch() == [] and await fx.mind.outbox_ready() == []
        assert fx.store.list(limit=100) == []
        assert fx.mind.owner_gate.ledger.recent(since=fx.now - timedelta(days=1)) == []
        assert fx.mind.goals.open() == []
        assert not fx.mind.mind_state.items("outreach.timing_hold:")
        assert "forecasts served" in " ".join(lines(service.report()))
    finally:
        service.close()
        fx.store.close()


async def test_live_loop_samples_and_stops_with_faculty(tmp_path, monkeypatch):
    now = [NOW]
    service = runtime(tmp_path, now)
    try:
        driver = service.loaded["health"].driver
        calls(monkeypatch, tmp_path, now[0] - timedelta(seconds=1))
        async def sleep(seconds):
            assert 0 < seconds <= 300
            service.enabled = False
        monkeypatch.setattr("colonymind.plasticity.learners.service_health.asyncio.sleep", sleep)
        await driver.live(service)
        assert service.loaded["health"].comparator.records()
    finally:
        service.close()


async def test_warning_batches_respect_senses_cap_and_keep_active_notes(tmp_path, monkeypatch):
    from test_mind_loop import Fixture, OWNER
    from colonymind.mind.observations import MAX_BATCH, for_mind
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fx = Fixture(tmp_path, config={"faculties": {"outreach": False}})
    service = runtime(tmp_path, [fx.now])
    try:
        driver = service.loaded["health"].driver
        state = driver._live_state()
        state["at"] = fx.now.isoformat()
        state["warnings"] = [{"id": str(i), "subject": "router.unit" + str(i), "feature": "latency_p95_ms",
            "at": fx.now.isoformat(), "severity": 5., "delivered": False,
            "fixed_at": None, "lead_seconds": None} for i in range(MAX_BATCH + 6)]
        driver._save_live(state)
        await driver.sample(service, mind=fx.mind)
        assert len(fx.mind.concerns.open()) == 6
        assert len(for_mind(fx.mind).active(source="service-health", viewer=OWNER)) == 6
        assert all(w["delivered"] for w in driver.warning_records())
    finally:
        service.close()
        fx.store.close()


async def test_running_mind_plasticity_switch_stops_sampling(tmp_path, monkeypatch):
    from test_mind_loop import Fixture, OWNER
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fx = Fixture(tmp_path)
    service = runtime(tmp_path, [fx.now])
    try:
        fx.mind.plasticity = service
        fx.mind.faculties["plasticity"] = False
        before = {path: path.read_bytes() for path in service.root.rglob("*") if path.is_file()}
        await service.loaded["health"].driver.sample(service, mind=fx.mind)
        assert {path: path.read_bytes() for path in service.root.rglob("*") if path.is_file()} == before
    finally:
        service.close()
        fx.store.close()


def test_erased_outcomes_cannot_return_after_disabled_runtime_reloads(tmp_path):
    service = runtime(tmp_path, [NOW])
    service.serve("health", "erased-value", {}, predict=lambda pin, inputs: 1.)
    service.outcome("health", "erased-value", observed=1.)
    service.close()
    cfg = load_config(tmp_path, environ={})
    cfg.data["mind"]["faculties"]["plasticity"] = False
    off = Runtime(cfg, clock=lambda: NOW)
    try:
        assert off.erase(["erased-value"])["clean"]
    finally:
        off.close()
    reloaded = runtime(tmp_path, [NOW])
    try:
        assert reloaded.loaded["health"].comparator.records() == []
    finally:
        reloaded.close()
