"""The automatic gates before a revision serves (docs/PLASTICITY.md, Gates), their evaluation hygiene and the
statistics they decide with. A failed gate drops the period, and the previous revision keeps serving."""

from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import pytest

from colonymind.plasticity import _io, gates
from colonymind.plasticity.admission import Admission
from colonymind.plasticity.budgets import BudgetCheck
from colonymind.plasticity.gates import (DECLINE_BAR, MIN_POOLED, Evidence, SetError, SetLedger, decline_bound,
                                        evaluate, gain_lower_bound, pooled_suite, rule_of_three)
from colonymind.plasticity.states import Controller
from colonymind.plasticity.store import LearnerStore
from colonymind.plasticity.synthetic import (EPOCH, OBJECTIVE, Cycle, SyntheticSource, SyntheticTemplate, make_rows)

NOW = EPOCH + timedelta(days=10)


def _suite(seeds=range(10)):
    return pooled_suite([SyntheticTemplate()], seeds=seeds)


def _evidence(**changes):
    size = MIN_POOLED
    fields = dict(reference=[True] * size, base=[True] * size, candidate=[True] * size, control=[False] * size,
                  canaries_base={"canary-1": 0.5}, canaries_candidate={"canary-1": 0.5},
                  served=np.linspace(0, 1, 7), offline=np.linspace(0, 1, 7), metrics={"accuracy": 0.9},
                  calibration=("cal-2",), parent_calibration=("cal-1",), state={"w": np.zeros(3)})
    fields.update(changes)
    if "budget" not in changes:
        fields["budget"] = _budget(fields["state"], used=1.0)
    return Evidence(**fields)


def _budget(state, *, used):
    return BudgetCheck(learner="probe", state=_io.state_digest(state), type="anchored_output", limit=2.0, used=used)


def _evaluate(tmp_path, evidence, **options):
    sets = SetLedger(tmp_path / "sets")
    suite = _suite()
    sets.register(suite.id, "gate_suite", suite.refs())
    return evaluate("closure-1", suite, evidence, sets=sets, **options)


# -- the statistics (D 6.4's arithmetic) -----------------------------------------------------------------------------

def test_the_rule_of_three():
    assert rule_of_three(100) == pytest.approx(0.03)
    # 100 items and no error at all: the bound on a decline is still 3 points.
    assert decline_bound(100, losses=0, gains=0) == pytest.approx(0.03)
    assert decline_bound(1000, losses=0, gains=0) == pytest.approx(0.003)


def test_the_bound_at_1000_paired_items_and_3_percent_disagreement():
    # About 0.9 points when no net decline is seen: 1,000 items resolve a 1-point bar.
    assert decline_bound(1000, losses=15, gains=15) == pytest.approx(0.009, abs=5e-4)
    assert decline_bound(1000, losses=15, gains=15) <= DECLINE_BAR


def test_the_bound_never_tightens_as_losses_grow():
    bounds = [decline_bound(100, losses=losses, gains=0) for losses in range(0, 6)]
    assert bounds == sorted(bounds) and bounds[1] > bounds[0]
    assert gain_lower_bound(1000, wins=15, losses=15) == pytest.approx(-decline_bound(1000, losses=15, gains=15))
    with pytest.raises(ValueError):
        decline_bound(0, losses=0, gains=0)
    with pytest.raises(ValueError):
        decline_bound(10, losses=8, gains=8)


# -- sets: selection and evaluation kept apart -----------------------------------------------------------------------

def test_a_set_used_for_selection_can_never_be_named_as_an_evaluation_set(tmp_path):
    sets = SetLedger(tmp_path)
    sets.register("suite-a", "gate_suite", ["item-1", "item-2"])
    sets.use("suite-a", "selection", revision="r1")
    with pytest.raises(SetError, match="selection"):
        sets.use("suite-a", "evaluation", revision="r1")
    with pytest.raises(SetError, match="selection"):
        SetLedger(tmp_path).use("suite-a", "evaluation", revision="r2")  # the use ledger persists


def test_a_sealed_set_is_never_used_for_selection(tmp_path):
    sets = SetLedger(tmp_path)
    sets.register("sealed-a", "sealed", ["item-9"])
    with pytest.raises(SetError, match="sealed"):
        sets.use("sealed-a", "selection", revision="r1")
    sets.use("sealed-a", "evaluation", revision="r1")
    with pytest.raises(SetError, match="unknown"):
        sets.use("never-registered", "evaluation", revision="r1")


def test_gate_suites_and_sealed_sets_share_no_item_and_keep_their_role(tmp_path):
    sets = SetLedger(tmp_path)
    sets.register("sealed-a", "sealed", ["item-1"])
    with pytest.raises(SetError, match="sealed"):
        sets.register("suite-a", "gate_suite", ["item-1", "item-2"])
    with pytest.raises(SetError, match="role"):
        sets.register("sealed-a", "gate_suite", ["item-1"])


def test_a_set_id_registered_again_with_other_items_is_refused(tmp_path):
    # Otherwise the new items would silently stay unprotected: trained on, then evaluated under the set's id.
    sets = SetLedger(tmp_path)
    sets.register("sealed-a", "sealed", ["item-1"])
    sets.register("sealed-a", "sealed", ["item-1"])
    with pytest.raises(SetError, match="items"):
        sets.register("sealed-a", "sealed", ["item-1", "item-2"])
    assert sets.protected_refs() == frozenset({"item-1"})


def test_a_suite_whose_items_are_not_all_registered_under_its_id_cannot_gate(tmp_path):
    sets = SetLedger(tmp_path / "sets")
    suite = _suite()
    sets.register(suite.id, "gate_suite", sorted(suite.refs())[:10])
    with pytest.raises(SetError, match="registered"):
        evaluate("closure-1", suite, _evidence(), sets=sets)
    assert sets.uses(suite.id) == []


def test_every_registered_set_joins_the_protected_set_that_admission_refuses(tmp_path):
    sets = SetLedger(tmp_path)
    sets.register("suite-a", "gate_suite", ["case-1"])
    sets.register("sealed-a", "sealed", ["case-2"])
    assert sets.protected_refs() == frozenset({"case-1", "case-2"})
    source = SyntheticSource()
    admission = Admission(OBJECTIVE, learner="probe", source_of=source.source_of,
                          protected_refs=sets.protected_refs(), clock=lambda: NOW)
    admitted = admission.admit(make_rows(3), period="p1")
    assert [row["ref"] for row in admitted.rows] == ["case-0"]


def test_a_set_registered_after_admission_was_set_up_is_refused_too(tmp_path):
    sets = SetLedger(tmp_path)
    admission = Admission(OBJECTIVE, learner="probe", source_of=SyntheticSource().source_of,
                          protected_refs=sets.protected_refs, clock=lambda: NOW)
    sets.register("sealed-a", "sealed", ["case-1"])
    admitted = admission.admit(make_rows(3), period="p1")
    assert [row["ref"] for row in admitted.rows] == ["case-0", "case-2"]


def test_gating_on_a_suite_records_it_as_used_for_selection(tmp_path):
    sets = SetLedger(tmp_path / "sets")
    suite = _suite()
    sets.register(suite.id, "gate_suite", suite.refs())
    evaluate("closure-1", suite, _evidence(), sets=sets)
    with pytest.raises(SetError):
        sets.use(suite.id, "evaluation", revision="closure-1")


# -- pooled suites ---------------------------------------------------------------------------------------------------

def test_pooled_suites_hold_at_least_1000_items_from_templates_at_new_seeds():
    assert MIN_POOLED == 1000
    suite = _suite()
    assert len(suite.items) == 1000 and suite.seeds == tuple(range(10))
    with pytest.raises(ValueError, match="1000"):
        _suite(range(9))
    with pytest.raises(ValueError, match="frozen"):
        pooled_suite([SyntheticTemplate()], seeds=range(10), frozen_seeds={3})
    with pytest.raises(ValueError, match="twice"):
        pooled_suite([SyntheticTemplate()], seeds=[1] * 11)
    assert _suite().id == suite.id and _suite(range(10, 20)).id != suite.id


def test_a_suite_under_the_pooled_minimum_cannot_gate(tmp_path):
    sets = SetLedger(tmp_path)
    small = gates.Suite(id="small", items=_suite().items[:100], seeds=(0,))
    sets.register(small.id, "gate_suite", small.refs())
    size = 100
    with pytest.raises(ValueError, match="1000"):
        evaluate("closure-1", small, _evidence(reference=[True] * size, candidate=[True] * size,
                                               control=[False] * size), sets=sets)


def test_a_suite_that_repeats_an_item_cannot_gate_whatever_its_length(tmp_path):
    # A repeated item is one observation counted many times: the bound would tighten on nothing new.
    items = _suite().items
    for suite in (gates.Suite(id="one-item", items=(items[0],) * MIN_POOLED, seeds=(0,)),
                  gates.Suite(id="one-short", items=items[:MIN_POOLED - 1] + (items[0],), seeds=(0,)),
                  gates.Suite(id="one-state", items=tuple(gates.Item(id=f"copy-{index}", state=items[0].state,
                                                                     label=items[0].label)
                                                          for index in range(MIN_POOLED)), seeds=(0,))):
        sets = SetLedger(tmp_path / suite.id)
        sets.register(suite.id, "gate_suite", suite.refs())
        with pytest.raises(ValueError, match="more than once"):
            evaluate("closure-1", suite, _evidence(), sets=sets)
        assert sets.uses(suite.id) == []  # refused before it counts as a use


def test_a_pooled_suite_refuses_a_state_generated_twice_under_other_ids():
    class Echo:
        name = "echo"

        def generate(self, seed):
            return [gates.Item(id=f"echo-{seed}-{index}", state="the same state", label="yes") for index in range(2)]

    with pytest.raises(ValueError, match="more than once"):
        pooled_suite([Echo()], seeds=[1], minimum=1)


# -- each gate -------------------------------------------------------------------------------------------------------

def test_a_clean_candidate_passes_every_gate(tmp_path):
    report = _evaluate(tmp_path, _evidence())
    assert report.passed and report.failed == ()
    assert {check.name for check in report.checks} == {"retention", "specificity", "canaries", "calibration",
                                                       "budget", "finite", "served_offline"}


def test_a_decline_inside_the_bar_whose_bound_is_not_fails(tmp_path):
    reference = [True] * MIN_POOLED
    candidate = [True] * MIN_POOLED
    for index in range(15):  # 15 gains
        reference[index] = False
    for index in range(15, 35):  # 20 losses
        candidate[index] = False
    point = (20 - 15) / MIN_POOLED
    assert point < DECLINE_BAR
    report = _evaluate(tmp_path, _evidence(reference=reference, candidate=candidate))
    assert report.failed == ("retention",)


def test_retention_is_judged_against_the_base_too_so_small_declines_never_ratchet(tmp_path):
    # Each night's revision within the bar of the one before it could still drift far from the base: the decline
    # against the base is bounded by the same bar.
    drifted = [index >= 40 for index in range(MIN_POOLED)]  # 40 items the base got right, forgotten over nights
    report = _evaluate(tmp_path, _evidence(reference=drifted, candidate=drifted))
    assert report.failed == ("retention",)
    assert "base" in {check.name: check.detail for check in report.checks}["retention"]
    assert _evaluate(tmp_path / "2", _evidence(reference=drifted, candidate=drifted,
                                               base=drifted)).failed == ()


def test_a_candidate_that_does_not_beat_its_shuffled_control_fails(tmp_path):
    control = [True] * MIN_POOLED
    report = _evaluate(tmp_path, _evidence(control=control))
    assert report.failed == ("specificity",)
    half = [index % 2 == 0 for index in range(MIN_POOLED)]
    assert _evaluate(tmp_path / "2", _evidence(candidate=half, reference=half, base=half, control=half)).failed == (
        "specificity",)


def test_a_canary_exposed_above_the_base_fails(tmp_path):
    report = _evaluate(tmp_path, _evidence(canaries_candidate={"canary-1": 0.51}))
    assert report.failed == ("canaries",)
    assert _evaluate(tmp_path / "2", _evidence(canaries_candidate={})).failed == ("canaries",)


@pytest.mark.parametrize("change", [
    {"state": {"w": np.array([0.0, math.nan])}},
    {"state": {"w": np.array([math.inf])}},
    {"metrics": {"accuracy": math.nan}},
])
def test_non_finite_values_fail(tmp_path, change):
    assert _evaluate(tmp_path, _evidence(**change)).failed == ("finite",)


def test_served_and_offline_outputs_must_agree(tmp_path):
    served = np.linspace(0, 1, 7)
    served[3] += 1e-3
    assert _evaluate(tmp_path, _evidence(served=served)).failed == ("served_offline",)
    assert _evaluate(tmp_path / "2", _evidence(served=np.zeros(3))).failed == ("served_offline",)


@pytest.mark.parametrize("calibration", [(), ("cal-1",)])
def test_thresholds_are_never_carried_over_from_other_weights(tmp_path, calibration):
    assert _evaluate(tmp_path, _evidence(calibration=calibration)).failed == ("calibration",)


def test_a_missing_or_exceeded_budget_fails(tmp_path):
    over = _budget({"w": np.zeros(3)}, used=2.5)
    assert _evaluate(tmp_path, _evidence(budget=over)).failed == ("budget",)
    assert _evaluate(tmp_path / "2", _evidence(budget=None)).failed == ("budget",)


def test_a_budget_check_of_another_state_does_not_count(tmp_path):
    other = _budget({"w": np.ones(3)}, used=0.5)
    assert other.ok
    assert _evaluate(tmp_path, _evidence(budget=other)).failed == ("budget",)


def test_evidence_must_cover_every_suite_item(tmp_path):
    with pytest.raises(ValueError, match="item"):
        _evaluate(tmp_path, _evidence(candidate=[True] * 10))


# -- a failed gate keeps the previous revision -----------------------------------------------------------------------

def test_a_failed_gate_drops_the_period_and_the_previous_revision_keeps_serving(tmp_path):
    store = LearnerStore(tmp_path, "probe", base="synthetic-base@1", declared=("threshold",), clock=lambda: NOW)
    cycle = Cycle(store, Controller(store, live=False, clock=lambda: NOW), source=SyntheticSource(make_rows(80)),
                  clock=lambda: NOW)
    first, _, outcome = cycle.step()
    assert outcome == "promoted"
    cycle.source.add(make_rows(40, prefix="more"))
    second, report, outcome = cycle.step(canaries_candidate={"canary-0": 1.0})
    assert not report.passed and outcome == "dropped"
    assert store.revisions.serving().id == first.id
    assert store.revisions.status(second.id).state == "withdrawn"
    assert [period.state for period in store.periods.periods()] == ["closed", "discarded"]
    # The dropped period's rows are not in the next build.
    assert all(not ref.startswith("more") for manifest in store.current_manifests() for ref in manifest.refs())


def test_the_calibration_refit_hook_gives_this_revisions_records_or_none():
    assert gates.refit_calibration(lambda state: [f"calibration-{state}"], "s1") == ("calibration-s1",)
    assert gates.refit_calibration(lambda state: None, "s1") == ()
    assert gates.refit_calibration(lambda state: ["", "calibration-1"], "s1") == ()
    with pytest.raises(TypeError):
        gates.refit_calibration(lambda state: "calibration-1", "s1")


def test_a_sealed_set_cannot_gate(tmp_path):
    sets = SetLedger(tmp_path)
    suite = _suite()
    sets.register(suite.id, "sealed", suite.refs())
    with pytest.raises(SetError, match="sealed"):
        evaluate("closure-1", suite, _evidence(), sets=sets)
