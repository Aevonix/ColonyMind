"""The configured plasticity runtime and its nightly boundary, on CPU only."""
from datetime import timedelta
from types import SimpleNamespace

import pytest
import yaml

from colonymind.config import ConfigError, load_config
from colonymind.plasticity import _io
from colonymind.plasticity.manifests import ROW_KEYS
from colonymind.plasticity.synthetic import EPOCH, OBJECTIVE, SyntheticSource, make_rows

NOW = EPOCH + timedelta(days=10)


def learner_config(*args, **kwargs):
    """Isolate a learner fixture from the new core reflector declaration."""
    cfg = load_config(*args, **kwargs)
    cfg.data["plasticity"]["learners"].pop("behaviour-tool-failure", None)
    return cfg


def config(tmp_path, learners=None, **plasticity):
    data = {"plasticity": {"learners": learners or {}, **plasticity}}
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump(data))
    return learner_config(tmp_path, environ={})


def declaration(**changes):
    return {"implementation": "synthetic", "kind": "supervised_refit", "area": "instinct",
            "mode": "rebuild", "dependents": ["threshold"], **changes}


def runtime(tmp_path, learners=None, **kwargs):
    from colonymind.plasticity.nightly import Runtime
    cfg = config(tmp_path, learners or {"probe": declaration()})
    return Runtime.from_config(cfg, check_learner=lambda name, value: SimpleNamespace(name=name),
                               clock=lambda: NOW, **kwargs)


def test_configuration_ships_with_idle_reflector_target(tmp_path):
    from colonymind.plasticity.nightly import Runtime
    cfg = load_config(tmp_path, environ={})
    assert cfg.get("mind.faculties.plasticity") is True
    assert cfg.get("plasticity.directory") == "plasticity"
    service = Runtime.from_config(cfg)
    assert list(service.loaded) == ["behaviour-tool-failure"]
    assert service.run()[0]["why"] == ["awaiting_lesson"]


@pytest.mark.parametrize("directory", ["../outside", "/tmp/outside", "."])
def test_state_directory_stays_under_data_directory(tmp_path, directory):
    with pytest.raises(ConfigError, match="plasticity.directory"):
        config(tmp_path, directory=directory)


def test_bad_registration_does_not_block_other_learners(tmp_path):
    from colonymind.plasticity.nightly import Runtime
    cfg = config(tmp_path, {"bad": declaration(), "good": declaration()})
    calls = []

    def check(name, declared):
        calls.append(name)
        if name == "bad":
            raise ValueError("budget: a declaration is required")
        return SimpleNamespace(name=name)

    service = Runtime.from_config(cfg, check_learner=check, clock=lambda: NOW)
    assert calls == ["bad", "good"]
    assert list(service.loaded) == ["good"]
    assert "budget" in service.report()["rejected"]["bad"]


def test_faculty_off_never_loads_or_runs_and_reads_base(tmp_path):
    from colonymind.plasticity.nightly import Runtime
    service = runtime(tmp_path, sources={"probe": SyntheticSource(make_rows(60))})
    service.run()
    cfg = load_config(tmp_path, environ={})
    cfg.data["mind"]["faculties"]["plasticity"] = False
    off = Runtime.from_config(cfg, check_learner=lambda *args: pytest.fail("loaded while off"))
    assert not off.loaded
    assert off.run() == []
    assert off.serve("probe", "request", "input", predict=lambda pin, value: pin is None) is True


def test_rows_forward_points_and_since_and_only_contract_keys(monkeypatch):
    from colonymind.plasticity.rows import InstinctRows
    from colonymind.instinct import outcomes
    calls = []
    row = make_rows(1)[0]

    def replay(store, sources, *, point, since):
        calls.append((store, sources, point, since))
        yield row

    monkeypatch.setattr(outcomes, "replay_rows", replay)
    rows = list(InstinctRows("log", "sources").rows(OBJECTIVE, since=NOW))
    assert calls == [("log", "sources", OBJECTIVE.points[0], NOW)]
    assert set(rows[0]) == set(ROW_KEYS)
    assert rows[0]["state"] == row["state"]


def test_yaml_learner_runs_whole_night_and_never_persists_state_text(tmp_path):
    rows = make_rows(60)
    service = runtime(tmp_path, sources={"probe": SyntheticSource(rows)})
    result = service.run()
    assert result[0]["result"] == "promoted"
    binding = service.loaded["probe"]
    assert binding.store.periods.retained()
    assert binding.store.revisions.serving().mode == "rebuild"
    for path in service.root.rglob("*"):
        if path.is_file():
            assert rows[0]["state"].encode() not in path.read_bytes()
    assert service.report()["learners"][0]["events"]


def test_nightly_refusal_keeps_previous_and_prunes_dropped_manifests(tmp_path, monkeypatch):
    from colonymind.plasticity.gates import Check, GateReport
    source = SyntheticSource(make_rows(60))
    service = runtime(tmp_path, sources={"probe": source})
    service.run()
    binding = service.loaded["probe"]
    before = binding.store.revisions.serving().id
    source.add(make_rows(30, prefix="later"))
    monkeypatch.setattr(binding.driver, "gate", lambda closure, build: GateReport(
        closure.id, "fixture", (Check("retention", False),)))
    result = service.run()
    assert result[0]["result"] == "dropped"
    assert binding.store.revisions.serving().id == before
    dropped = binding.store.periods.periods()[-1]
    assert dropped.state == "discarded"
    assert not dropped.manifests


def test_chain_promotes_and_reads_as_one(tmp_path):
    declarations = {name: declaration(chain="pair") for name in ("first", "second")}
    service = runtime(tmp_path, declarations, sources={
        name: SyntheticSource(make_rows(60, prefix=name)) for name in declarations})
    assert all(row["result"] == "promoted" for row in service.run())
    pins = service.pin_chain("pair", "owner")
    assert all(pin.closure.mode != "base" for pin in pins.values())
    assert service.joints.records("pair")[-1]["members"] == {
        name: pin.closure.id for name, pin in pins.items()}
    service.joints.begin("pair", {name: pin.closure.id for name, pin in pins.items()})
    assert all(pin.closure.mode == "base" for pin in service.pin_chain("pair", "owner").values())


def test_chain_failed_gate_moves_no_member(tmp_path, monkeypatch):
    from colonymind.plasticity.gates import Check, GateReport
    service = runtime(tmp_path, {name: declaration(chain="pair") for name in ("first", "second")},
                      sources={name: SyntheticSource(make_rows(60, prefix=name))
                               for name in ("first", "second")})
    monkeypatch.setattr(service.loaded["second"].driver, "gate", lambda closure, build: GateReport(
        closure.id, "fixture", (Check("retention", False),)))
    result = {row["learner"]: row for row in service.run()}
    assert result["second"]["result"] == "dropped"
    # The member that passed its own gates is not withdrawn for the chain alone: it waits as a candidate.
    assert (result["first"]["result"], result["first"]["why"]) == ("unchanged", ["chain_waiting"])
    assert service.loaded["first"].store.revisions.status(result["first"]["closure"]).state == "candidate"
    assert all(pin.closure.mode == "base" for pin in service.pin_chain("pair", "owner").values())


def test_compaction_keeps_fourteen_closures_and_erasure_leaves_no_row(tmp_path):
    source = SyntheticSource(make_rows(60))
    service = runtime(tmp_path, sources={"probe": source})
    service.run()
    binding = service.loaded["probe"]
    from colonymind.plasticity.compaction import compact
    from colonymind.plasticity.builds import state_digest
    import numpy as np
    for index in range(18):
        state = {"w": np.array([index], dtype=float)}
        closure = binding.store.revisions.compose(state=state_digest(state), objective="fixture",
                                                   dependents={"threshold": str(index)}, audience="guest")
        binding.store.revisions.write(closure, state)
    compact(binding.store)
    assert len(binding.store.revisions.retained()) == 14
    erased = service.erase(["case-1"])
    assert erased["clean"]
    assert not any(b'"case-1"' in path.read_bytes() for path in service.root.rglob("*") if path.is_file())


def test_nightly_boundary_matches_existing_window_and_survives_restart(tmp_path):
    from colonymind.plasticity.nightly import NightlyTask
    service = runtime(tmp_path, sources={"probe": SyntheticSource(make_rows(60))})
    start = NOW.replace(hour=21)
    task = NightlyTask(service, tz="UTC", boundary_minute=22 * 60, started_at=start)
    assert task.tick(start + timedelta(minutes=30)) == []
    assert task.tick(start + timedelta(hours=1))[0]["result"] == "promoted"
    other = NightlyTask(service, tz="UTC", boundary_minute=22 * 60, started_at=start)
    assert other.tick(start + timedelta(hours=2)) == []


def test_state_lock_instances_share_thread_and_process_lock(tmp_path):
    from colonymind.plasticity.locking import StateLock
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    entered = Event()
    def run():
        with StateLock(tmp_path):
            entered.set()
    with ThreadPoolExecutor() as pool:
        with StateLock(tmp_path):
            with StateLock(tmp_path):
                future = pool.submit(run)
                assert not entered.wait(0.05)
        future.result(timeout=2)
    assert entered.is_set()


@pytest.mark.asyncio
async def test_lifespan_starts_nightly_in_minds_window_and_closes(tmp_path, monkeypatch):
    import asyncio
    from colonymind.plasticity import nightly
    cfg = config(tmp_path)
    seen = []
    service = SimpleNamespace(enabled=True, loaded={"probe": object()},
                              _clock=lambda: NOW, close=lambda: seen.append("closed"))
    monkeypatch.setattr(nightly.Runtime, "from_config", lambda *a, **kw: service)

    async def run(task):
        seen.append((task.tz, task.boundary_minute))
        try:
            await asyncio.Future()
        finally:
            seen.append("cancelled")

    monkeypatch.setattr(nightly.NightlyTask, "run", run)
    mind = SimpleNamespace(tz="UTC", consolidation=SimpleNamespace(boundary_minute=lambda: 123))
    async with nightly.lifespan(cfg, data_dir=tmp_path, mind=mind) as loaded:
        assert loaded is service
        await asyncio.sleep(0)
    assert seen[-2:] == ["cancelled", "closed"]
    assert seen[0][1] == 123


def test_erasure_while_off_removes_comparator_records_from_retained_catalog(tmp_path):
    from colonymind.plasticity.nightly import Runtime
    service = runtime(tmp_path, sources={"probe": SyntheticSource(make_rows(60))})
    service.run()
    service.serve("probe", "case-1", "input", predict=lambda pin, value: 1)
    assert service.loaded["probe"].comparator.records()
    cfg = load_config(tmp_path, environ={})
    cfg.data["mind"]["faculties"]["plasticity"] = False
    off = Runtime.from_config(cfg)
    assert off.erase(["case-1"])["clean"]
    assert service.loaded["probe"].comparator.records() == []


def test_erasure_retry_after_manifest_rewrite_rebuilds_before_finishing(tmp_path, monkeypatch):
    source = SyntheticSource(make_rows(60))
    service = runtime(tmp_path, sources={"probe": source})
    service.run()
    binding = service.loaded["probe"]
    build = binding.driver.build
    monkeypatch.setattr(binding.driver, "build", lambda: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(OSError, match="interrupted"):
        service.erase(["case-1"])
    assert binding.store.revisions.serving().mode == "base"
    with pytest.raises(ValueError, match="pending erasure"):
        service.run()
    monkeypatch.setattr(binding.driver, "build", build)
    assert service.erase(["case-1"])["clean"]
    assert not (binding.store.directory / "erasures" / "PENDING").exists()
    assert binding.store.revisions.serving().mode == "rebuild"


def test_dropped_manifest_compaction_removes_states_and_never_readmits_its_rows(tmp_path, monkeypatch):
    from colonymind.plasticity.gates import Check, GateReport
    source = SyntheticSource(make_rows(60))
    service = runtime(tmp_path, sources={"probe": source})
    service.run()
    binding = service.loaded["probe"]
    source.add(make_rows(30, prefix="failed"))
    gate = binding.driver.gate
    monkeypatch.setattr(binding.driver, "gate", lambda closure, build: GateReport(
        closure.id, "fixture", (Check("retention", False),)))
    dropped = service.run()[0]["closure"]
    assert not binding.store.revisions.exists(dropped)
    monkeypatch.setattr(binding.driver, "gate", gate)
    source.add(make_rows(30, prefix="fresh"))
    service.run()
    assert all(not entry.ref.startswith("failed-") for manifest in binding.store.manifests
               for entry in manifest.entries)


def test_teacher_drop_and_no_accumulation_change_only_their_named_inputs(tmp_path):
    source = SyntheticSource([*make_rows(60), *make_rows(20, prefix="model", teacher="deliberation")])
    service = runtime(tmp_path, {"probe": declaration(disabled_teachers=["deliberation"], accumulation=False)},
                      sources={"probe": source})
    service.run()
    binding = service.loaded["probe"]
    assert binding.last_build.rows == 60
    source.add(make_rows(30, prefix="fresh"))
    service.run()
    assert binding.last_build.rows == 30


def test_bad_chain_name_rejects_only_that_learner(tmp_path):
    service = runtime(tmp_path, {"bad": declaration(chain="../wrong"), "good": declaration()})
    assert set(service.loaded) == {"good"}
    assert "chain" in service.rejected["bad"]


def test_report_after_removing_last_learner_is_idle(tmp_path):
    from colonymind.plasticity.nightly import read_report
    service = runtime(tmp_path, sources={"probe": SyntheticSource(make_rows(60))})
    service.run()
    cfg = config(tmp_path)
    report = read_report(cfg)
    assert report["learners"] == []
    assert "no learners" in report["note"].lower()


def test_shuffled_arm_keeps_all_manifest_rows_when_source_grows(tmp_path):
    from colonymind.plasticity.builds import training_rows
    source = SyntheticSource(make_rows(60))
    service = runtime(tmp_path, {"probe": declaration(control="shuffled")}, sources={"probe": source})
    driver = service.loaded["probe"].driver
    driver.period()
    source.add(make_rows(60, start=60))
    driver.period()
    trained = training_rows(driver.objective, driver.store.current_manifests(), driver.source)
    assert len(trained) == 120


def test_build_error_discards_the_period_it_closed(tmp_path, monkeypatch):
    service = runtime(tmp_path, sources={"probe": SyntheticSource(make_rows(60))})
    binding = service.loaded["probe"]
    monkeypatch.setattr(binding.driver, "build", lambda: (_ for _ in ()).throw(ValueError("failure")))
    assert service.run()[0]["result"] == "error"
    assert all(period.state == "discarded" for period in binding.store.periods.periods())


def test_no_accumulation_stack_names_base_when_the_new_periods_audience_widens(tmp_path):
    source = SyntheticSource(make_rows(60))
    service = runtime(tmp_path, {"probe": declaration(mode="stack", accumulation=False)},
                      sources={"probe": source})
    assert service.run()[0]["result"] == "promoted"
    binding = service.loaded["probe"]
    first = binding.store.revisions.serving()
    source.add(make_rows(60, start=60, audience="guest"))
    assert service.run()[0]["result"] == "promoted"
    second = binding.store.revisions.serving()
    assert second.parent == binding.store.revisions.base.id
    assert second.audience == "guest"
    assert binding.controller.previous(second.id).id == first.id


def test_disabled_erasure_retry_discards_learning_without_loading_learner(tmp_path, monkeypatch):
    from colonymind.plasticity.nightly import Runtime, read_report
    service = runtime(tmp_path, sources={"probe": SyntheticSource(make_rows(60))})
    service.run()
    binding = service.loaded["probe"]
    monkeypatch.setattr(binding.driver, "build", lambda: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(OSError):
        service.erase(["case-1"])
    assert read_report(service.config)["learners"][0]["erasure_pending"] is True
    cfg = load_config(tmp_path, environ={})
    cfg.data["mind"]["faculties"]["plasticity"] = False
    off = Runtime.from_config(cfg, check_learner=lambda *args: pytest.fail("learner loaded"))
    assert off.erase(["case-1"])["clean"]
    assert not (binding.store.directory / "erasures" / "PENDING").exists()
    assert not binding.store.revisions.retained()


def test_chain_observation_failure_keeps_unvisited_members_previous_period(tmp_path, monkeypatch):
    sources = {name: SyntheticSource(make_rows(60, prefix=name)) for name in ("first", "second")}
    service = runtime(tmp_path, {name: declaration(chain="pair") for name in sources}, sources=sources)
    service.run()
    before = service.loaded["second"].store.periods.periods()[-1]
    monkeypatch.setattr(service.loaded["first"].driver, "period",
                        lambda: (_ for _ in ()).throw(OSError("observation failed")))
    assert all(row["result"] == "error" for row in service.run())
    assert service.loaded["second"].store.periods.get(before.id).state == "closed"


def test_promotion_hook_failure_never_serves_the_discarded_period(tmp_path, monkeypatch):
    service = runtime(tmp_path, sources={"probe": SyntheticSource(make_rows(60))})
    binding = service.loaded["probe"]
    monkeypatch.setattr(binding.controller.lineage, "carry",
                        lambda *args: (_ for _ in ()).throw(OSError("carry failed")))
    assert service.run()[0]["result"] == "error"
    assert binding.store.revisions.serving().mode == "base"
    assert service.serve("probe", "after-failure", "input", predict=lambda pin, value: pin.closure.mode) == "base"


def test_erasure_unserves_before_source_deletion_and_stays_erased_after_restart(tmp_path, monkeypatch):
    from colonymind.plasticity.nightly import Runtime
    source = SyntheticSource(make_rows(60))
    service = runtime(tmp_path, sources={"probe": source})
    service.run()
    binding = service.loaded["probe"]
    dirty = binding.store.revisions.serving().id

    def delete(refs):
        assert binding.store.revisions.serving().id != dirty
        # Emulate a reader over retained raw sources. Weight erasure must not
        # depend on deleting those sources to keep the row out after restart.

    monkeypatch.setattr(source, "erase", delete)
    service.erase(["case-1"])
    fresh = Runtime.from_config(service.config, check_learner=lambda *args: SimpleNamespace(),
                                sources={"probe": source}, clock=lambda: NOW)
    assert all(row["ref"] != "case-1" for row in fresh.loaded["probe"].source.rows(OBJECTIVE))


@pytest.mark.parametrize("chain", [False, True])
def test_fresh_failed_gate_on_unchanged_serving_bytes_rolls_back(tmp_path, monkeypatch, chain):
    from colonymind.plasticity.gates import Check, GateReport
    names = ("first", "second") if chain else ("probe",)
    service = runtime(tmp_path, {name: declaration(**({"chain": "pair"} if chain else {})) for name in names},
                      sources={name: SyntheticSource(make_rows(60, prefix=name)) for name in names})
    assert all(row["result"] == "promoted" for row in service.run())
    monkeypatch.setattr(service.loaded[names[0]].driver, "gate", lambda closure, build: GateReport(
        closure.id, "fresh", (Check("retention", False),)))
    result = service.run()
    assert all(row["result"] == "rolled_back" for row in result)
    assert all(service.pin(name, "owner").closure.mode == "base" for name in names)
