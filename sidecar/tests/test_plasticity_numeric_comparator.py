"""Numeric live comparison uses paired losses and the existing rollback controller."""
import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from colonymind.plasticity import _io
from colonymind.plasticity.gates import Check, GateReport
from colonymind.plasticity.nightly import COMPARATORS
from colonymind.plasticity.numeric_comparator import NumericComparator, configure
from colonymind.plasticity.states import Controller
from colonymind.plasticity.store import LearnerStore
from colonymind.reports.plasticity import comparator_line

NOW = datetime(2026, 1, 10, tzinfo=timezone.utc)


def promote(controller, number):
    state = {"W": np.array([[number]], dtype=np.float32)}
    closure = controller.store.revisions.compose(state=_io.state_digest(state), objective="numeric",
        calibration=(f"scale-{number}",), dependents={}, parent=controller.store.revisions.serving().id,
        manifests=(), periods=(), audience="owner", mode="stack")
    controller.propose(closure, state)
    controller.consider(closure.id, GateReport(closure.id, "test", (Check("finite", True),)))
    return closure


def setup(tmp_path, **options):
    clock = options.pop("clock", lambda: NOW)
    store = LearnerStore(tmp_path, "numeric", base="persistence@1", mode="stack", clock=clock)
    controller = Controller(store, clock=clock)
    first = promote(controller, 1)
    configure(controller, options.pop("declaration", {}), **options)
    return controller, first, NumericComparator(controller, key=b"numeric test key", clock=clock)


def test_registry_finite_numbers_no_input_text_and_standardised_loss(tmp_path):
    _, first, comparator = setup(tmp_path, scales=[10., .5])
    assert COMPARATORS["numeric-online"] is NumericComparator
    comparator.serve("private-ref", {"sentence": "fixture secret sentence"},
                     lambda pin, inputs: [20., 1.] if pin.closure.id == first.id else [10., .5])
    result = comparator.outcome("private-ref", observed=[0., 0.], teacher="realized_value")
    saved, = comparator.records()
    assert saved["output"] == [20., 1.] and saved["reference_output"] == [10., .5]
    assert saved["outcome"]["served_loss"] == 4.
    assert saved["outcome"]["reference_loss"] == 1.
    assert result["labelled"] == 1 and not result["rolled_back"]
    paths = list(comparator.directory.glob("numeric-*.jsonl"))
    assert len(paths) == 1 and paths[0].name == "numeric-2026-01-10.jsonl"
    assert "private-ref" not in paths[0].read_text() and "fixture secret" not in paths[0].read_text()
    for value in ("private output", float("nan"), float("inf"), [1., "words"]):
        with pytest.raises(ValueError):
            comparator.serve("invalid", {}, lambda pin, inputs: value)


def test_ten_percent_excess_rolls_back_previous_with_report(tmp_path):
    controller, first, comparator = setup(tmp_path)
    second = promote(controller, 2)
    for i in range(200):
        comparator.serve(f"r{i}", {}, lambda pin, inputs: math.sqrt(1.1) if pin.closure.id == second.id else 1.)
    results = [comparator.outcome(f"r{i}", observed=0., teacher="verified_outcome") for i in range(200)]
    assert all(not result["rolled_back"] for result in results[:29])
    assert results[29]["rolled_back"] and results[29]["lower_bound"] == pytest.approx(.1)
    assert controller.store.revisions.serving().id == first.id
    assert sum(result["rolled_back"] for result in results) == 1
    assert "Rolled back" in comparator_line(comparator.summary())
    assert any(event["event"] == "rolled_back" for event in controller.events())


def test_inside_bar_and_stale_outcomes_do_not_rollback_new_revision(tmp_path):
    controller, first, comparator = setup(tmp_path)
    for i in range(200):
        comparator.serve(f"r{i}", {}, lambda pin, inputs: math.sqrt(1.01) if pin.closure.id == first.id else 1.)
        assert not comparator.outcome(f"r{i}", observed=0.)["rolled_back"]
    for i in range(30):
        comparator.serve(f"stale{i}", {}, lambda pin, inputs: 10. if pin.closure.id == first.id else 1.)
    newest = promote(controller, 2)
    for i in range(30):
        assert not comparator.outcome(f"stale{i}", observed=0.)["rolled_back"]
    assert controller.store.revisions.serving().id == newest.id


def test_proof_idempotence_brier_window_and_erasure(tmp_path):
    now = [NOW]
    _, _, comparator = setup(tmp_path, variant="logistic", clock=lambda: now[0],
                             declaration={"comparator": {"window_items": 40, "window_hours": 2, "bar": .02}})
    for i in range(50):
        comparator.serve(str(i), {}, lambda pin, inputs: .25)
        result = comparator.outcome(str(i), observed=1.)
    assert result["labelled"] == 40
    assert comparator.records()[0]["outcome"]["served_loss"] == .5625
    assert comparator.outcome("49", observed=1.)["labelled"] == 40
    with pytest.raises(ValueError, match="changed"):
        comparator.outcome("49", observed=0.)
    with pytest.raises(ValueError, match="proof"):
        comparator.outcome("49", observed=1., teacher="stronger_model")
    with pytest.raises(ValueError):
        comparator.serve("bad", {}, lambda pin, inputs: 2.)
    comparator.erase([str(i) for i in range(49)])
    assert len(comparator.records()) == 1
    now[0] += timedelta(hours=3)
    assert comparator.summary()["labelled"] == 0
    now[0] += timedelta(days=14)
    assert comparator.records() == []
    assert not list(comparator.directory.glob("numeric-*.jsonl"))


def test_mutating_predictor_and_duplicate_refs_are_refused(tmp_path):
    _, _, comparator = setup(tmp_path)
    def mutate(pin, inputs):
        inputs["x"] += 1
        return 1.
    with pytest.raises(ValueError, match="unchanged"):
        comparator.serve("r", {"x": 0}, mutate)
    comparator.serve("r", {}, lambda pin, inputs: 1.)
    with pytest.raises(ValueError, match="already"):
        comparator.serve("r", {}, lambda pin, inputs: 1.)
    assert len(comparator.records()) == 1
    assert json.loads(next(comparator.directory.glob("*.jsonl")).read_text())["output"] == 1.
