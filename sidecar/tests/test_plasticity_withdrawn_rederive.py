"""A night whose derivation names a closure withdrawn before: unchanged, never an error, never served."""
import logging
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest
import yaml

from test_plasticity_wiring_runtime import learner_config as load_config
from colonymind.plasticity import _io
from colonymind.plasticity.compaction import discarded_rows
from colonymind.plasticity.gates import Check, GateReport
from colonymind.plasticity.states import StateError, WithdrawnClosure
from colonymind.plasticity.synthetic import EPOCH, SyntheticSource, make_rows

NOW = EPOCH + timedelta(days=10)


def runtime(tmp_path, source, *, clock=lambda: NOW, **declared):
    from colonymind.plasticity.nightly import Runtime
    data = {"plasticity": {"learners": {"probe": {
        "implementation": "synthetic", "kind": "supervised_refit", "area": "instinct", "mode": "rebuild",
        "dependents": ["threshold"], **declared}}}}
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump(data))
    return Runtime.from_config(load_config(tmp_path, environ={}),
                               check_learner=lambda name, value: SimpleNamespace(name=name),
                               clock=clock, sources={"probe": source})


def chained(tmp_path, sources):
    """Two synthetic rebuild learners qualified together as the chain ``pair``."""
    from colonymind.plasticity.nightly import Runtime
    data = {"plasticity": {"learners": {name: {
        "implementation": "synthetic", "kind": "supervised_refit", "area": "instinct", "mode": "rebuild",
        "dependents": ["threshold"], "chain": "pair"} for name in sources}}}
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump(data))
    return Runtime.from_config(load_config(tmp_path, environ={}),
                               check_learner=lambda name, value: SimpleNamespace(name=name), clock=lambda: NOW,
                               sources=sources)


def by_learner(results):
    return {row["learner"]: row for row in results}


def failing(closure, build):
    return GateReport(closure.id, "fixture", (Check("calibration", False),))


def test_rederiving_a_withdrawn_closure_is_unchanged_not_an_error(tmp_path, monkeypatch, caplog):
    # A learner with nothing to learn yet: the first night's closure fails its gates and is withdrawn. Every
    # later night with no new rows derives that same closure id again.
    service = runtime(tmp_path, SyntheticSource([]))
    binding = service.loaded["probe"]
    monkeypatch.setattr(binding.driver, "gate", failing)
    first = service.run()[0]
    assert first["result"] == "dropped"
    withdrawn = first["closure"]
    assert binding.store.revisions.status(withdrawn).state == "withdrawn"
    events = binding.controller.events()

    monkeypatch.setattr(binding.driver, "gate", lambda closure, build: pytest.fail("a withdrawn derivation was gated"))
    with caplog.at_level(logging.INFO, logger="colonymind.plasticity.nightly"):
        for _ in range(2):
            assert service.run() == [{"learner": "probe", "closure": withdrawn, "result": "unchanged",
                                      "why": ["withdrawn_before"]}]
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)
    assert _io.read_json(service.root / "nightly.json")["learners"][0]["result"] == "unchanged"
    # Withdrawn stays withdrawn, nothing serves it and no transition is reported.
    assert binding.store.revisions.status(withdrawn).state == "withdrawn"
    assert binding.store.revisions.serving().mode == "base"
    assert service.pin("probe", "owner").closure.mode == "base"
    assert binding.controller.events() == events
    # The unchanged nights admitted nothing, so their empty periods are discarded rather than piling up.
    periods = binding.store.periods.periods()
    assert [period.state for period in periods] == ["discarded"] * 3
    assert [period.reason for period in periods[1:]] == ["unchanged night: nothing to learn"] * 2
    # The protective refusal still holds for anyone proposing the withdrawn identity directly.
    closure = binding.store.revisions.compose(
        state=binding.last_build.digest, objective=binding.driver.objective.identity,
        calibration=binding.driver.refit(binding.last_build.digest),
        dependents={"threshold": f"threshold-{binding.last_build.digest[:8]}"},
        parent=binding.store.revisions.serving().id, manifests=binding.last_build.manifests,
        periods=tuple(period.id for period in binding.store.periods.retained()), audience="guest")
    assert closure.id == withdrawn
    with pytest.raises(StateError, match="withdrawn") as refused:
        binding.controller.propose(closure, binding.last_build.state)
    assert isinstance(refused.value, WithdrawnClosure) and refused.value.code == "withdrawn"


def test_new_rows_after_a_withdrawal_produce_a_new_candidate(tmp_path, monkeypatch):
    source = SyntheticSource([])
    service = runtime(tmp_path, source)
    binding = service.loaded["probe"]
    gate = binding.driver.gate
    monkeypatch.setattr(binding.driver, "gate", failing)
    withdrawn = service.run()[0]["closure"]
    assert service.run()[0]["result"] == "unchanged"
    assert binding.store.periods.periods()[-1].state == "discarded"  # it admitted nothing

    source.add(make_rows(60, prefix="fresh"))
    refused = service.run()[0]
    assert refused["result"] == "dropped" and refused["closure"] != withdrawn  # a new candidate, judged

    source.add(make_rows(60, prefix="later"))
    monkeypatch.setattr(binding.driver, "gate", gate)
    promoted = service.run()[0]
    assert promoted["result"] == "promoted" and promoted["closure"] not in (withdrawn, refused["closure"])
    serving = binding.store.revisions.serving()
    assert serving.id == promoted["closure"]
    assert binding.last_build.rows == 60  # the refused night's rows were discarded with its closure
    assert binding.store.revisions.status(withdrawn).state == "withdrawn"


def test_withdrawn_before_leaves_the_serving_closure_alone(tmp_path, monkeypatch):
    # Same learned bytes under another calibration (as a refit cutoff gives) is another closure: withdraw it
    # while an earlier closure serves, then derive it again.
    service = runtime(tmp_path, SyntheticSource(make_rows(60)))
    binding = service.loaded["probe"]
    assert service.run()[0]["result"] == "promoted"
    serving = binding.store.revisions.serving()
    monkeypatch.setattr(binding.driver, "refit", lambda digest: ["calibration-other"])
    monkeypatch.setattr(binding.driver, "gate", failing)
    dropped = service.run()[0]
    assert dropped["result"] == "dropped" and dropped["closure"] != serving.id
    monkeypatch.setattr(binding.driver, "gate", lambda closure, build: pytest.fail("a withdrawn derivation was gated"))
    events = binding.controller.events()

    assert service.run() == [{"learner": "probe", "closure": dropped["closure"], "result": "unchanged",
                              "why": ["withdrawn_before"]}]
    assert binding.store.revisions.serving().id == serving.id
    assert binding.store.revisions.status(serving.id).state == "serving"
    assert service.pin("probe", "owner").closure.id == serving.id
    assert binding.controller.events() == events
    assert binding.store.periods.periods()[-1].state == "discarded"  # every row was decided on the first night


def test_a_refused_state_change_reports_its_fixed_code_not_its_message(tmp_path, monkeypatch, caplog):
    service = runtime(tmp_path, SyntheticSource(make_rows(60)))
    binding = service.loaded["probe"]
    build = binding.driver.build
    # A build claiming a wider audience than its owner rows: the controller refuses it.
    monkeypatch.setattr(binding.driver, "build", lambda: replace(build(), audience="guest"))
    with caplog.at_level(logging.WARNING, logger="colonymind.plasticity.nightly"):
        result = service.run()
    assert result == [{"learner": "probe", "result": "error", "why": ["StateError", "audience_wider"]}]
    warning = next(record.getMessage() for record in caplog.records if record.levelno == logging.WARNING)
    assert warning == "plasticity nightly failed (StateError, audience_wider)"
    assert binding.store.revisions.serving().mode == "base"


def test_a_chain_member_withdrawn_before_moves_no_member(tmp_path, monkeypatch):
    names = ("first", "second")
    service = chained(tmp_path, {name: SyntheticSource(make_rows(60, prefix=name)) for name in names})
    assert all(row["result"] == "promoted" for row in service.run())
    serving = {name: service.loaded[name].store.revisions.serving().id for name in names}
    records = service.joints.records("pair")
    # ``first`` keeps deriving its serving closure and passes; ``second`` derives another closure, refused once.
    monkeypatch.setattr(service.loaded["first"].driver, "gate", lambda closure, build: GateReport(
        closure.id, "fixture", (Check("retention", True),)))
    second = service.loaded["second"].driver
    monkeypatch.setattr(second, "refit", lambda digest: ["calibration-other"])
    monkeypatch.setattr(second, "gate", failing)
    withdrawn = {row["learner"]: row for row in service.run()}["second"]
    assert withdrawn["result"] == "dropped"
    monkeypatch.setattr(second, "gate", lambda closure, build: pytest.fail("a withdrawn derivation was gated"))

    result = {row["learner"]: row for row in service.run()}
    assert result["second"] == {"learner": "second", "closure": withdrawn["closure"], "result": "unchanged",
                                "why": ["withdrawn_before"]}
    # The chain cannot move as one; its serving member is not judged, keeps serving and is not rolled back.
    assert result["first"] == {"learner": "first", "closure": serving["first"], "result": "unchanged",
                               "why": ["chain_waiting"]}
    assert {name: service.loaded[name].store.revisions.serving().id for name in names} == serving
    assert {name: pin.closure.id for name, pin in service.pin_chain("pair", "owner").items()} == serving
    assert service.joints.records("pair") == records


# -- chains: a night the chain cannot move judges no member for the chain alone ------------------------------------


def test_alternating_rows_across_a_chain_lose_nothing_and_promote_once_both_members_have_rows(tmp_path,
                                                                                               monkeypatch):
    sources = {"first": SyntheticSource([]), "second": SyntheticSource([])}
    service = chained(tmp_path, sources)
    first, second = service.loaded["first"], service.loaded["second"]
    records = len(service.joints.records("pair"))

    # Nothing to learn yet: each member fails its own gates and is dropped, as before.
    night = by_learner(service.run())
    assert night["first"]["result"] == night["second"]["result"] == "dropped"

    # ``first`` learns while ``second`` re-derives its withdrawn closure: the chain waits, nothing is judged.
    sources["first"].add(make_rows(60, prefix="first-a"))
    night = by_learner(service.run())
    assert night["second"]["why"] == ["withdrawn_before"]
    waiting = night["first"]
    assert (waiting["result"], waiting["why"]) == ("unchanged", ["chain_waiting"])
    assert first.store.revisions.status(waiting["closure"]).state == "candidate"
    assert first.store.periods.periods()[-1].state == "closed" and first.last_build.rows == 60

    # More rows for ``first`` while ``second`` still has none: its rows grow, nothing is discarded, and the
    # candidate it replaces is deleted, never withdrawn.
    sources["first"].add(make_rows(60, prefix="first-b"))
    night = by_learner(service.run())
    assert night["first"]["why"] == ["chain_waiting"] and night["first"]["closure"] != waiting["closure"]
    assert first.last_build.rows == 120
    assert not first.store.revisions.exists(waiting["closure"])
    assert first.store.revisions.status(waiting["closure"]).state != "withdrawn"
    assert [period.state for period in first.store.periods.periods()] == ["discarded", "closed", "closed"]
    waiting = night["first"]
    assert first.store.revisions.serving().mode == second.store.revisions.serving().mode == "base"
    assert len(service.joints.records("pair")) == records

    # ``second`` learns too: the chain moves as one, promoting the very candidate ``first`` left waiting.
    sources["second"].add(make_rows(60, prefix="second-a"))
    night = by_learner(service.run())
    assert night["first"] == {"learner": "first", "closure": waiting["closure"], "result": "promoted", "why": []}
    assert night["second"]["result"] == "promoted"
    serving = {name: service.loaded[name].store.revisions.serving().id for name in sources}
    assert serving == {"first": waiting["closure"], "second": night["second"]["closure"]}
    assert first.last_build.rows == 120 and second.last_build.rows == 60
    assert len(service.joints.records("pair")) == records + 1
    assert service.joints.records("pair")[-1]["members"] == serving
    assert {name: pin.closure.id for name, pin in service.pin_chain("pair", "owner").items()} == serving

    # The other member's turn: ``second`` grows, ``first`` re-derives what it serves, the chain moves again. (The
    # synthetic gate refuses a re-gated serving closure on calibration, so ``first`` passes here as a fixture.)
    monkeypatch.setattr(first.driver, "gate", lambda closure, build: GateReport(
        closure.id, "fixture", (Check("retention", True),)))
    sources["second"].add(make_rows(60, prefix="second-b"))
    night = by_learner(service.run())
    assert night["first"]["closure"] == serving["first"] and night["second"]["result"] == "promoted"
    assert second.last_build.rows == 120 and len(service.joints.records("pair")) == records + 2


def test_a_member_whose_only_failed_check_is_the_chain_waits_and_keeps_its_rows(tmp_path, monkeypatch):
    sources = {"first": SyntheticSource(make_rows(60, prefix="first")),
               "second": SyntheticSource(make_rows(60, prefix="second"))}
    service = chained(tmp_path, sources)
    first, second = service.loaded["first"], service.loaded["second"]
    monkeypatch.setattr(second.driver, "gate", failing)

    night = by_learner(service.run())
    # The sibling that failed its own gates is judged (dropped, its period discarded) ...
    assert night["second"]["result"] == "dropped" and night["second"]["why"] == ["calibration"]
    assert second.store.periods.periods()[-1].state == "discarded"
    # ... the member that passed its own gates is not withdrawn for the chain: it waits, with its period.
    candidate = night["first"]["closure"]
    assert night["first"] == {"learner": "first", "closure": candidate, "result": "unchanged",
                              "why": ["chain_waiting"]}
    assert first.store.revisions.status(candidate).state == "candidate"
    assert first.store.periods.periods()[-1].state == "closed"

    monkeypatch.undo()
    sources["second"].add(make_rows(60, prefix="second-later"))
    night = by_learner(service.run())
    assert night["first"] == {"learner": "first", "closure": candidate, "result": "promoted", "why": []}
    assert night["second"]["result"] == "promoted"
    assert first.store.revisions.serving().id == candidate and first.last_build.rows == 60


# -- erasure: a clean rebuild that derives a withdrawn closure completes -----------------------------------------------


def erased_learner(tmp_path, source, **declared):
    """A learner whose first night (nothing to learn) was withdrawn, then that promoted 60 rows: the rows erasure
    removes, after which a clean rebuild derives the withdrawn first-night closure again."""
    service = runtime(tmp_path, source, **declared)
    withdrawn = service.run()[0]
    assert withdrawn["result"] == "dropped"
    rows = make_rows(60, prefix="erased")
    source.add(rows)
    promoted = service.run()[0]
    assert promoted["result"] == "promoted"
    store = service.loaded["probe"].store
    hashes = {entry.state_hash for manifest in store.manifests for entry in manifest.entries}
    return service, withdrawn["closure"], promoted["closure"], sorted(row["ref"] for row in rows), hashes


def test_an_erasure_whose_rebuild_derives_a_withdrawn_closure_completes_clean(tmp_path):
    from colonymind.plasticity.erasure import scan
    service, withdrawn, trained, refs, hashes = erased_learner(tmp_path, SyntheticSource([]))
    binding = service.loaded["probe"]

    report = service.erase(refs)
    assert report["clean"] is True and report["left"] == ()
    assert report["contaminated"] == {"probe": (trained,)}
    # Nothing that learned from the erased rows survives or serves, and no marker is left behind.
    assert not binding.store.revisions.exists(trained) and binding.store.revisions.find(trained) is None
    assert binding.store.revisions.serving().mode == "base"
    assert service.pin("probe", "owner").closure.mode == "base"
    assert not (binding.store.directory / "erasures" / "PENDING").exists()
    assert not binding.store.rebuild_required.exists() and not binding.store.erasure_pending()
    assert list(scan(service.root, set(refs) | hashes)) == []
    # The withdrawn identity stays withdrawn; a retry and the next night have nothing to do.
    assert binding.store.revisions.status(withdrawn).state == "withdrawn"
    assert service.erase(refs)["clean"] is True
    assert service.run() == [{"learner": "probe", "closure": withdrawn, "result": "unchanged",
                              "why": ["withdrawn_before"]}]


def test_a_deferred_erasure_rebuild_that_derives_a_withdrawn_closure_clears_its_marker(tmp_path):
    from colonymind.plasticity.erasure import scan
    service, withdrawn, trained, refs, hashes = erased_learner(tmp_path, SyntheticSource([]))
    service.close()
    # Erased while the learner is off: its rebuild is deferred to the next load.
    off = runtime(tmp_path, SyntheticSource([]), enabled=False)
    assert off.erase(refs)["clean"] is True
    stored = off._eraser().stores["probe"]
    assert stored.rebuild_required.exists() and stored.revisions.serving().mode == "base"
    off.close()

    back = runtime(tmp_path, SyntheticSource([]))
    binding = back.loaded["probe"]
    assert not binding.store.rebuild_required.exists() and not binding.store.erasure_pending()
    assert binding.store.revisions.serving().mode == "base"
    assert back.pin("probe", "owner").closure.mode == "base"
    assert binding.store.revisions.find(trained) is None
    assert list(scan(back.root, set(refs) | hashes)) == []
    assert back.run() == [{"learner": "probe", "closure": withdrawn, "result": "unchanged",
                           "why": ["withdrawn_before"]}]


# -- periods: an unchanged night keeps its period only where a later build reads it ---------------------------------


@pytest.mark.parametrize("declared, kept", [
    ({"mode": "rebuild"}, True),                          # every build reads every retained period
    ({"mode": "stack"}, False),                           # a stacked build reads the newest period alone
    ({"mode": "rebuild", "accumulation": False}, False),  # so does a non-accumulating one
])
def test_an_unchanged_nights_admitted_rows_are_kept_only_where_a_later_build_reads_them(tmp_path, monkeypatch,
                                                                                         declared, kept):
    service = runtime(tmp_path, source := SyntheticSource([]), **declared)
    binding = service.loaded["probe"]
    monkeypatch.setattr(binding.driver, "gate", failing)
    withdrawn = service.run()[0]["closure"]
    # New admitted rows that leave the learned state as it was (a learner they cannot move), so the night
    # derives the withdrawn closure again.
    unchanged = binding.last_build
    monkeypatch.setattr(service, "_build", lambda binding: unchanged)
    rows = make_rows(20, prefix="later")
    source.add(rows)
    assert service.run() == [{"learner": "probe", "closure": withdrawn, "result": "unchanged",
                              "why": ["withdrawn_before"]}]
    period = binding.store.periods.periods()[-1]
    if kept:
        assert period.state == "closed" and period in binding.store.periods.retained()
        assert len(binding.store.manifests_of([period])[0].admitted()) == 20
    else:
        # Discarded as a judged night's period is: compaction pruned its manifest, its rows are excluded.
        assert (period.state, period.reason) == ("discarded", "unchanged night: unused by later builds")
        assert {_io.digest([row["ref"], row["teacher"]]) for row in rows} <= discarded_rows(binding.store)


@pytest.mark.parametrize("mode", ["rebuild", "stack"])
def test_an_unchanged_nights_quarantined_rows_stay_eligible_and_train_once_they_mature(tmp_path, mode):
    clock = [NOW]
    source = SyntheticSource([])
    service = runtime(tmp_path, source, clock=lambda: clock[0], mode=mode)
    binding = service.loaded["probe"]
    withdrawn = service.run()[0]
    assert withdrawn["result"] == "dropped"
    # Rows too young to train: quarantined, nothing admitted, so the night derives the withdrawn closure again.
    source.add(make_rows(30, prefix="young", at=(NOW - timedelta(hours=1)).isoformat()))
    assert service.run()[0]["why"] == ["withdrawn_before"]
    period = binding.store.periods.periods()[-1]
    verdicts = {entry.verdict for manifest in binding.store.manifests_of([period]) for entry in manifest.entries}
    assert verdicts == {"quarantine"} and period.state == "closed"
    # A day later they are offered again, admitted and trained on.
    clock[0] = NOW + timedelta(days=2)
    later = service.run()[0]
    assert later["closure"] != withdrawn["closure"] and binding.last_build.rows == 30
