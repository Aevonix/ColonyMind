"""A revision's states (docs/PLASTICITY.md, States and promotion): candidate, serving, superseded and withdrawn.
Promotion is automatic inside the gates, including the first revision; rollback follows a failed later gate."""

from __future__ import annotations

from datetime import timedelta

import pytest

from colonymind.plasticity.gates import Check, GateReport
from colonymind.plasticity.revisions import RevisionError
from colonymind.plasticity.states import STATES, Controller, StateError
from colonymind.plasticity.store import LearnerStore
from colonymind.plasticity.synthetic import EPOCH, Cycle, SyntheticSource, make_rows

NOW = EPOCH + timedelta(days=10)


def _cycle(root, *, live=True, rows=None):
    store = LearnerStore(root, "probe", base="synthetic-base@1", declared=("threshold",), clock=lambda: NOW)
    controller = Controller(store, live=live, clock=lambda: NOW)
    return Cycle(store, controller, source=SyntheticSource(make_rows(80) if rows is None else rows),
                 clock=lambda: NOW)


def _failing(closure_id):
    return GateReport(closure=closure_id, suite="suite-1", checks=(Check("retention", False, "declined"),))


def test_the_states_are_a_closed_list():
    assert STATES == ("candidate", "serving", "superseded", "withdrawn")


def test_promotion_is_automatic_inside_the_gates(tmp_path):
    cycle = _cycle(tmp_path)
    first, report, outcome = cycle.step()
    assert report.passed and outcome == "promoted"
    assert cycle.store.revisions.serving().id == first.id
    assert cycle.store.revisions.status(first.id).state == "serving"
    cycle.source.add(make_rows(40, prefix="more"))
    second, _, outcome = cycle.step()
    assert outcome == "promoted" and cycle.store.revisions.status(first.id).state == "superseded"


def test_a_first_live_revision_serves_immediately_and_old_gate_types_are_removed(tmp_path):
    from colonymind.plasticity import states
    from colonymind.plasticity.revisions import Status

    cycle = _cycle(tmp_path)
    first, report, outcome = cycle.step()
    assert report.passed and outcome == "promoted"
    assert cycle.store.revisions.serving().id == first.id
    assert not hasattr(states, "FIRST_GO") and not hasattr(states, "Go")
    assert not hasattr(Status(), "held")
    assert not hasattr(cycle.controller, "record_go")


def test_automatic_rollback_on_a_failed_later_gate(tmp_path):
    cycle = _cycle(tmp_path, live=False)
    first, _, _ = cycle.step()
    cycle.source.add(make_rows(40, prefix="more"))
    second, _, _ = cycle.step()
    assert cycle.controller.recheck(GateReport(closure=second.id, suite="suite-1",
                                               checks=(Check("retention", True, ""),))) == "kept"
    assert cycle.controller.recheck(_failing(second.id)) == "rolled_back"
    assert cycle.store.revisions.serving().id == first.id
    assert cycle.store.revisions.status(second.id).state == "withdrawn"
    assert cycle.store.revisions.status(first.id).state == "serving"
    assert cycle.controller.recheck(_failing(first.id)) == "rolled_back"
    assert cycle.store.revisions.serving().id == cycle.store.revisions.base.id


def test_rollback_returns_to_the_actual_previous_revision_even_when_a_build_names_another_parent(tmp_path):
    cycle = _cycle(tmp_path)
    first, _, _ = cycle.step()
    cycle.source.add(make_rows(40, prefix="more"))
    cycle.period()
    build = cycle.build()
    closure = _closure(cycle, build, parent=cycle.store.revisions.base.id)
    cycle.controller.propose(closure, build.state)
    passing = GateReport(closure.id, "fixture", (Check("retention", True, ""),))
    cycle.controller.consider(closure.id, passing)
    assert cycle.controller.previous(closure.id).id == first.id
    assert cycle.controller.rollback() == "rolled_back"
    assert cycle.store.revisions.serving().id == first.id


def test_retry_after_a_promotion_pointer_swap_keeps_the_original_comparator(tmp_path, monkeypatch):
    cycle = _cycle(tmp_path)
    first, _, _ = cycle.step()
    cycle.source.add(make_rows(40, prefix="more"))
    cycle.period()
    build = cycle.build()
    second = cycle.propose(build)
    report = cycle.gate(second, build)
    original = cycle.store.revisions.set_status

    def crash(closure, state, **kwargs):
        if state == "serving":
            raise OSError("power lost")
        return original(closure, state, **kwargs)

    monkeypatch.setattr(cycle.store.revisions, "set_status", crash)
    with pytest.raises(OSError):
        cycle.controller.consider(second.id, report)
    monkeypatch.undo()
    assert cycle.controller.consider(second.id, report) == "promoted"
    assert cycle.controller.previous(second.id).id == first.id


def _trained_refs(cycle):
    return {entry.ref for manifest in cycle.store.current_manifests() for entry in manifest.admitted()}


def test_a_rollback_discards_the_periods_the_withdrawn_revision_introduced(tmp_path):
    cycle = _cycle(tmp_path, live=False)
    first, _, _ = cycle.step()
    cycle.source.add(make_rows(40, prefix="bad"))
    second, _, _ = cycle.step()
    introduced = cycle.last_period
    assert cycle.controller.recheck(_failing(second.id)) == "rolled_back"
    assert cycle.store.periods.get(introduced).state == "discarded"
    assert [period.state for period in cycle.store.periods.periods()] == ["closed", "discarded"]
    assert not any(ref.startswith("bad-") for ref in _trained_refs(cycle))  # the next rebuild never retrains them
    cycle.source.add(make_rows(40, prefix="later"))
    third, _, outcome = cycle.step()
    assert outcome == "promoted" and not any(ref.startswith("bad-") for ref in _trained_refs(cycle))
    assert cycle.store.revisions.serving().id == third.id


def test_a_rollback_interrupted_after_the_pointer_moved_has_already_withdrawn_and_discarded(tmp_path,
                                                                                            monkeypatch):
    cycle = _cycle(tmp_path, live=False)
    first, _, _ = cycle.step()
    cycle.source.add(make_rows(40, prefix="bad"))
    second, _, _ = cycle.step()
    introduced = cycle.last_period
    set_status = cycle.store.revisions.set_status

    def crash(closure_id, state, **options):
        if state == "serving":
            raise OSError("power lost")
        return set_status(closure_id, state, **options)

    monkeypatch.setattr(cycle.store.revisions, "set_status", crash)
    with pytest.raises(OSError):
        cycle.controller.recheck(_failing(second.id))
    monkeypatch.undo()
    # A retry finishes the original intent instead of withdrawing the restored revision.
    assert cycle.store.revisions.serving().id == first.id
    assert cycle.store.revisions.status(second.id).state == "withdrawn"
    assert cycle.store.periods.get(introduced).state == "discarded"
    assert cycle.controller.recheck(_failing(second.id)) == "rolled_back"
    assert cycle.store.revisions.serving().id == first.id
    assert cycle.store.revisions.pending_rollback() is None
    assert len([event for event in cycle.controller.events() if event["event"] == "rolled_back"]) == 1


def test_a_failed_candidate_discards_the_period_it_introduced_even_when_the_caller_names_none(tmp_path):
    cycle = _cycle(tmp_path, live=False)
    first, _, _ = cycle.step()
    cycle.source.add(make_rows(40, prefix="bad"))
    cycle.period()
    introduced = cycle.last_period
    build = cycle.build()
    candidate = cycle.propose(build)
    assert cycle.controller.consider(candidate.id, _failing(candidate.id)) == "dropped"
    assert cycle.store.periods.get(introduced).state == "discarded"
    assert not any(ref.startswith("bad-") for ref in _trained_refs(cycle))
    assert cycle.store.revisions.serving().id == first.id


def test_a_promoted_revision_keeps_its_periods_when_a_newer_candidate_fails(tmp_path):
    cycle = _cycle(tmp_path)
    first, _, outcome = cycle.step()
    assert outcome == "promoted"
    kept = cycle.last_period
    cycle.source.add(make_rows(40, prefix="bad"))
    cycle.period()
    build = cycle.build()
    candidate = cycle.propose(build)
    assert cycle.controller.consider(candidate.id, _failing(candidate.id)) == "dropped"
    assert cycle.store.periods.get(kept).state == "closed"  # the serving revision learned it and passed its gates
    assert cycle.store.periods.get(cycle.last_period).state == "discarded"


def test_a_candidate_trained_on_a_period_a_rollback_discarded_never_serves(tmp_path):
    cycle = _cycle(tmp_path, live=False)
    first, _, _ = cycle.step()
    cycle.source.add(make_rows(40, prefix="bad"))
    second, _, _ = cycle.step()
    cycle.source.add(make_rows(40, prefix="next"))
    cycle.period()
    build = cycle.build()
    pending = cycle.propose(build)  # built on the serving revision's periods, before its later gate failed
    passing = cycle.gate(pending, build)
    assert passing.passed
    cycle.controller.recheck(_failing(second.id))
    assert cycle.controller.consider(pending.id, passing) == "dropped"
    assert cycle.store.revisions.status(pending.id).state == "withdrawn"
    assert cycle.store.revisions.serving().id == first.id


def test_a_report_must_name_the_closure_it_judges(tmp_path):
    cycle = _cycle(tmp_path, live=False)
    first, _, _ = cycle.step()
    with pytest.raises(StateError, match="serving"):
        cycle.controller.recheck(_failing("some-other-closure"))
    cycle.source.add(make_rows(10, prefix="more"))
    cycle.period()
    build = cycle.build()
    candidate = cycle.propose(build)
    with pytest.raises(StateError, match="report"):
        cycle.controller.consider(candidate.id, _failing(first.id))


def test_a_withdrawn_revision_never_promotes(tmp_path):
    cycle = _cycle(tmp_path, live=False)
    first, _, _ = cycle.step()
    cycle.source.add(make_rows(10, prefix="more"))
    second, _, _ = cycle.step()
    cycle.controller.recheck(_failing(second.id))
    passing = GateReport(closure=second.id, suite="suite-1", checks=(Check("retention", True, ""),))
    with pytest.raises(StateError, match="withdrawn"):
        cycle.controller.consider(second.id, passing)


def test_a_closure_never_claims_a_wider_audience_than_its_rows(tmp_path):
    cycle = _cycle(tmp_path, live=False)
    cycle.period()
    build = cycle.build()
    assert build.audience == "owner"
    serving = cycle.store.revisions.serving()
    closure = cycle.store.revisions.compose(
        state=build.digest, objective=cycle.objective.identity, calibration=("calibration-x",),
        dependents={"threshold": "t"}, parent=serving.id, manifests=build.manifests,
        periods=tuple(period.id for period in cycle.store.periods.retained()), audience="guest", mode="rebuild")
    with pytest.raises(StateError, match="audience"):
        cycle.controller.propose(closure, build.state)
    assert not cycle.store.revisions.exists(closure.id)


def _closure(cycle, build, **changes):
    fields = dict(state=build.digest, objective=cycle.objective.identity, calibration=("calibration-x",),
                  dependents={"threshold": "t"}, parent=cycle.store.revisions.serving().id, manifests=build.manifests,
                  periods=tuple(period.id for period in cycle.store.periods.retained()), audience="owner",
                  mode="rebuild")
    fields.update(changes)
    return cycle.store.revisions.compose(**fields)


def test_the_learner_store_refuses_a_wider_audience_even_past_the_controller(tmp_path):
    cycle = _cycle(tmp_path, live=False)
    cycle.period()
    build = cycle.build()
    closure = _closure(cycle, build, audience="guest")
    with pytest.raises(RevisionError, match="audience"):
        cycle.store.revisions.write(closure, build.state)
    assert not cycle.store.revisions.exists(closure.id)
    with pytest.raises(RevisionError, match="audience"):
        cycle.store.revisions.write(_closure(cycle, build, audience="everyone"), build.state)


def test_a_stacked_closure_inherits_the_audience_of_the_state_it_starts_from(tmp_path):
    store = LearnerStore(tmp_path, "probe", base="synthetic-base@1", mode="stack", declared=("threshold",),
                         clock=lambda: NOW)
    cycle = Cycle(store, Controller(store, live=False, clock=lambda: NOW), source=SyntheticSource(make_rows(40)),
                  clock=lambda: NOW)
    first, _, outcome = cycle.step()
    assert outcome == "promoted" and first.audience == "owner"
    cycle.source.add(make_rows(40, prefix="public", audience="guest"))
    cycle.period()
    build = cycle.build()
    assert build.audience == "owner"  # it carries everything the state it started from learned
    closure = _closure(cycle, build, mode="stack", parent=first.id,
                       periods=(cycle.store.periods.retained()[-1].id,), audience="guest")
    with pytest.raises(RevisionError, match="audience"):
        cycle.store.revisions.write(closure, build.state)


def test_a_closure_names_the_periods_its_manifests_came_from(tmp_path):
    cycle = _cycle(tmp_path, live=False)
    cycle.period()
    build = cycle.build()
    with pytest.raises(RevisionError, match="period"):
        cycle.store.revisions.write(_closure(cycle, build, periods=()), build.state)
    with pytest.raises(RevisionError, match="manifest"):
        cycle.store.revisions.write(_closure(cycle, build, manifests=build.manifests + ("not-kept",)), build.state)
