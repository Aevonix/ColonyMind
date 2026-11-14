"""The period ledger and the budget ledger (docs/PLASTICITY.md, Periods and Budgets)."""

from __future__ import annotations

import math
from datetime import timedelta

import pytest

from colonymind.plasticity.budgets import BUDGET_TYPES, Budget, BudgetCheck, BudgetError, BudgetLedger
from colonymind.plasticity.periods import PeriodError, PeriodLedger
from colonymind.plasticity.synthetic import EPOCH

NOW = EPOCH + timedelta(days=10)


def test_the_period_ledger_opens_closes_and_discards(tmp_path):
    ledger = PeriodLedger(tmp_path, "probe", clock=lambda: NOW)
    first = ledger.open()
    assert first.id == "p000001" and first.state == "open" and first.opened_at == NOW.isoformat()
    with pytest.raises(PeriodError, match="open"):
        ledger.open()
    ledger.add_manifest(first.id, "m1")
    closed = ledger.close(first.id)
    assert closed.state == "closed" and closed.manifests == ("m1",)
    with pytest.raises(PeriodError, match="closed"):
        ledger.add_manifest(first.id, "m2")
    second = ledger.open()
    ledger.close(second.id, delta="d2")
    discarded = ledger.discard(second.id, reason="gate")
    assert discarded.state == "discarded" and discarded.reason == "gate" and discarded.delta == "d2"
    with pytest.raises(PeriodError):
        ledger.close(second.id)
    with pytest.raises(PeriodError, match="unknown"):
        ledger.discard("p000009", reason="x")
    reopened = PeriodLedger(tmp_path, "probe", clock=lambda: NOW)
    assert [period.id for period in reopened.retained()] == ["p000001"]
    assert [period.state for period in reopened.periods()] == ["closed", "discarded"]
    assert reopened.open().id == "p000003"


def test_an_erasure_rewrites_a_periods_manifest(tmp_path):
    ledger = PeriodLedger(tmp_path, "probe", clock=lambda: NOW)
    period = ledger.open()
    ledger.add_manifest(period.id, "m1")
    ledger.add_manifest(period.id, "m2")
    ledger.close(period.id)
    assert ledger.replace_manifest(period.id, "m1", "m1b").manifests == ("m1b", "m2")
    with pytest.raises(PeriodError):
        ledger.replace_manifest(period.id, "m9", "m9b")


def test_the_core_records_the_engines_budget_check_and_refuses(tmp_path):
    assert BUDGET_TYPES == ("absolute", "relative_dense", "low_rank", "operator_norm", "anchored_output",
                            "ladder")
    ledger = BudgetLedger(tmp_path)
    ledger.declare(Budget("probe", "anchored_output", 2.0))
    within = ledger.record(BudgetCheck("probe", "s1", "anchored_output", 2.0, 1.5))
    assert within.ok
    over = ledger.record(BudgetCheck("probe", "s2", "anchored_output", 2.0, 2.5))
    assert not over.ok
    broken = ledger.record(BudgetCheck("probe", "s3", "anchored_output", 2.0, math.nan))
    assert not broken.ok
    assert BudgetLedger(tmp_path).result("probe", "s2") == over
    assert ledger.result("probe", "never-checked") is None
    with pytest.raises(BudgetError, match="declared"):
        ledger.record(BudgetCheck("other", "s1", "anchored_output", 2.0, 1.0))
    with pytest.raises(BudgetError, match="type"):
        ledger.record(BudgetCheck("probe", "s1", "operator_norm", 2.0, 1.0))
    with pytest.raises(BudgetError, match="limit"):
        ledger.record(BudgetCheck("probe", "s1", "anchored_output", 3.0, 1.0))


def test_a_check_against_a_declaration_since_changed_no_longer_counts(tmp_path):
    ledger = BudgetLedger(tmp_path)
    ledger.declare(Budget("probe", "anchored_output", 2.0))
    ledger.record(BudgetCheck("probe", "s1", "anchored_output", 2.0, 1.5))
    ledger.declare(Budget("probe", "anchored_output", 1.0))  # tightened: 1.5 no longer fits
    assert ledger.result("probe", "s1") is None
    ledger.declare(Budget("probe", "relative_dense", 2.0))  # another type: a check of the old type says nothing
    assert ledger.result("probe", "s1") is None
    ledger.declare(Budget("probe", "anchored_output", 2.0))  # the declaration it was checked against, again
    assert ledger.result("probe", "s1") == BudgetCheck("probe", "s1", "anchored_output", 2.0, 1.5)


@pytest.mark.parametrize("budget", [Budget("probe", "made_up", 1.0), Budget("probe", "anchored_output", 0.0),
                                    Budget("probe", "anchored_output", math.inf), Budget("", "anchored_output", 1.0)])
def test_a_budget_is_a_known_type_with_a_positive_finite_limit(tmp_path, budget):
    with pytest.raises(BudgetError):
        BudgetLedger(tmp_path).declare(budget)
