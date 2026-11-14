"""The same live item scores both closures; only paired real outcomes can trigger rollback."""

from datetime import timedelta
from threading import Event, Thread

import pytest

from colonymind.plasticity.comparator import LiveComparator
from colonymind.plasticity.erasure import Erasure
from colonymind.plasticity.gates import Check, GateReport
from colonymind.plasticity.revisions import RevisionError
from colonymind.plasticity.states import Controller, StateError
from colonymind.plasticity.store import LearnerStore
from colonymind.plasticity.synthetic import EPOCH, Cycle, SyntheticSource, make_rows

NOW = EPOCH + timedelta(days=10)
KEY = b"comparator-test-key"


def _setup(root, **options):
    clock = options.pop("clock", lambda: NOW)
    store = LearnerStore(root, "probe", base="synthetic-base@1", declared=("threshold",), clock=clock)
    cycle = Cycle(store, Controller(store, clock=clock), source=SyntheticSource(make_rows(80)), clock=clock)
    first, _, _ = cycle.step()
    comparator = LiveComparator(cycle.controller, key=KEY, clock=clock, **options)
    return cycle, first, comparator


def _label(comparator, index, *, served=True, reference=True):
    ref = f"decision-{index}"
    comparator.serve(ref, "fixture input sentence", lambda pin, value: pin.closure.id)
    return comparator.outcome(ref, served_correct=served, reference_correct=reference)


def test_every_served_item_scores_both_outputs_on_the_same_input_and_logs_no_text(tmp_path):
    cycle, first, comparator = _setup(tmp_path)
    calls = []
    inputs = {"sentence": "a private fixture input sentence"}

    def predict(pin, value):
        calls.append((pin.closure.id, value))
        return "a private fixture output sentence"

    assert comparator.serve("decision-1", inputs, predict) == "a private fixture output sentence"
    assert [item[0] for item in calls] == [first.id, cycle.store.revisions.base.id]
    assert all(item[1] is inputs for item in calls)
    record, = comparator.records()
    assert record["learner"] == "probe"
    assert record["closure"] == first.id and record["reference"] == cycle.store.revisions.base.id
    assert record["output"]["hash"] and record["reference_output"]["hash"]
    assert record["outcome"] is None
    comparator.outcome("decision-1", served_correct=True, reference_correct=False)
    assert comparator.records()[0]["outcome"]["served_correct"] is True
    for path in comparator.directory.rglob("*"):
        if path.is_file():
            text = path.read_text()
            assert "private fixture" not in text and "decision-1" not in text


def test_a_measured_regression_rolls_back_once_and_writes_a_report(tmp_path):
    cycle, first, comparator = _setup(tmp_path)
    cycle.source.add(make_rows(40, prefix="more"))
    second, _, _ = cycle.step()
    for index in range(20):
        result = _label(comparator, index, served=False, reference=True)
        if result["rolled_back"]:
            break
    assert result["lower_bound"] > result["bar"]
    assert result["closure"] == second.id and result["reference"] == first.id
    assert cycle.store.revisions.serving().id == first.id
    assert cycle.store.revisions.status(second.id).state == "withdrawn"
    assert cycle.store.periods.get(cycle.last_period).state == "discarded"
    assert any(event["event"] == "rolled_back" and "comparator" in event["reason"]
               for event in cycle.controller.events())
    passing = GateReport(second.id, "live", (Check("retention", True, ""),))
    with pytest.raises(StateError, match="withdrawn"):
        cycle.controller.consider(second.id, passing)


def test_identical_outputs_and_declines_within_the_declared_bar_keep_serving(tmp_path):
    cycle, first, comparator = _setup(tmp_path, bar=0.1)
    for index in range(220):
        result = _label(comparator, index, served=index % 20 != 19)
        assert not result["rolled_back"]
    assert cycle.store.revisions.serving().id == first.id
    assert result["labelled"] == 200
    assert result["decline"] == pytest.approx(0.05)


def test_rolling_window_limits_by_labelled_count_and_served_age(tmp_path):
    moment = [NOW]
    _, _, comparator = _setup(tmp_path, clock=lambda: moment[0])
    for index in range(201):
        _label(comparator, index)
    assert comparator.summary()["labelled"] == 200
    moment[0] += timedelta(days=8)
    _label(comparator, "fresh")
    comparator.serve("unlabelled", "input", lambda pin, value: 1.0)
    assert comparator.summary()["labelled"] == 1


def test_model_teachers_never_supply_live_proof_and_repeated_outcomes_do_not_count_twice(tmp_path):
    _, _, comparator = _setup(tmp_path)
    comparator.serve("decision", "input", lambda pin, value: True)
    with pytest.raises(ValueError, match="proof"):
        comparator.outcome("decision", served_correct=False, reference_correct=True, teacher="stronger_model")
    result = comparator.outcome("decision", served_correct=True, reference_correct=True)
    assert result["labelled"] == 1
    assert comparator.outcome("decision", served_correct=True, reference_correct=True)["labelled"] == 1
    with pytest.raises(ValueError, match="boolean"):
        comparator.outcome("decision", served_correct="yes", reference_correct=True)


def test_an_erasure_removes_both_outputs_and_its_outcome(tmp_path):
    _, _, comparator = _setup(tmp_path)
    _label(comparator, 1)
    comparator.erase({"decision-1"})
    assert comparator.records() == [] and comparator.summary()["labelled"] == 0


def test_rollback_waits_for_the_same_lock_as_erasure(tmp_path):
    cycle, first, comparator = _setup(tmp_path)
    for index in range(3):
        _label(comparator, index, served=False, reference=True)
    comparator.serve("regression", "input", lambda pin, value: False)
    erasure = Erasure({"probe": cycle.store})
    started, finished = Event(), Event()

    def outcome():
        started.set()
        comparator.outcome("regression", served_correct=False, reference_correct=True)
        finished.set()

    with erasure.lock:
        worker = Thread(target=outcome)
        worker.start()
        assert started.wait(1)
        assert not finished.wait(0.05)
        assert cycle.store.revisions.serving().id == first.id
    worker.join(2)
    assert finished.is_set()
    assert cycle.store.revisions.serving().id == cycle.store.revisions.base.id


@pytest.mark.parametrize("interrupted", ["intent", "discard", "withdrawal", "report", "pointer"])
def test_interrupted_rollback_stops_serving_failed_weights_and_next_request_finishes_once(
        tmp_path, monkeypatch, interrupted):
    cycle, first, comparator = _setup(tmp_path)
    cycle.source.add(make_rows(40, prefix="more"))
    failed, _, _ = cycle.step()
    for index in range(3):
        _label(comparator, index, served=False, reference=True)
    comparator.serve("fourth", "input", lambda pin, value: pin.closure.id)

    def crash(*args, **kwargs):
        raise OSError("power lost")

    if interrupted == "intent":
        monkeypatch.setattr(cycle.store.revisions, "begin_rollback", crash)
    elif interrupted == "discard":
        monkeypatch.setattr(cycle.store.periods, "discard", crash)
    elif interrupted == "withdrawal":
        original = cycle.store.revisions.set_status

        def withdraw(closure, state, **kwargs):
            if state == "withdrawn":
                crash()
            return original(closure, state, **kwargs)

        monkeypatch.setattr(cycle.store.revisions, "set_status", withdraw)
    elif interrupted == "report":
        monkeypatch.setattr(cycle.controller, "_event", crash)
    else:
        monkeypatch.setattr(cycle.store.revisions, "promote", crash)

    with pytest.raises(OSError, match="power lost"):
        comparator.outcome("fourth", served_correct=False, reference_correct=True)
    if interrupted != "intent":
        # The durable intent is authoritative before the period or status was written.
        assert cycle.store.revisions.pin("owner").closure.id != failed.id
        assert cycle.store.revisions.status(failed.id).state == "withdrawn"
    monkeypatch.undo()

    reopened = LearnerStore(tmp_path, "probe", base="synthetic-base@1", declared=("threshold",), clock=lambda: NOW)
    if interrupted != "intent":
        with pytest.raises(RevisionError, match="withdrawn"):
            reopened.revisions.promote(failed.id)
    controller = Controller(reopened, clock=lambda: NOW)
    comparator = LiveComparator(controller, key=KEY, clock=lambda: NOW)
    assert comparator.serve("after", "input", lambda pin, value: pin.closure.id) == first.id
    assert reopened.revisions.pending_rollback() is None
    assert reopened.periods.get(cycle.last_period).state == "discarded"
    assert reopened.revisions.status(failed.id).state == "withdrawn"
    assert controller.recover_rollback() == "kept"
    assert not comparator.outcome("fourth", served_correct=False, reference_correct=True)["rolled_back"]
    assert reopened.revisions.serving().id == first.id
    events = [event for event in controller.events() if event["event"] == "rolled_back"]
    assert len(events) == 1 and events[0]["closure"] == failed.id
