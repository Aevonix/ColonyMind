"""Literal wire fixtures and bounded numeric learning. All series settings are test assumptions."""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from colonymind.plasticity.numeric import (
    NumericLearner, Revision, Surprise, bounded_step, load_revision, metrics, save_revision,
)
from colonymind.plasticity.builds import TrainingRow
from colonymind.plasticity.objectives import NEXT_VALUE

FIXTURES = Path(__file__).parent / "fixtures" / "numeric_revision_1"
EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def layout(variant="value"):
    if variant == "logistic":
        return dict(schema="numeric-revision-1", learner="probe", variant=variant,
                    features=[{"input": "x"}, {"time": "hod_sin"}, {"const": 1}],
                    outputs=[{"event": "reply"}], standardisation={"x": {"mean": 0., "scale": 1.}},
                    anchor={"type": "constant", "p": .3}, tz="UTC")
    return dict(schema="numeric-revision-1", learner="probe", variant=variant,
                features=[{"stream": "a", "lag": 1}, {"time": "hod_sin"},
                          {"time": "hod_cos"}, {"const": 1}], outputs=[{"stream": "a"}],
                standardisation={"a": {"mean": 0., "scale": 1.}},
                anchor={"type": "persistence"}, tz="UTC")


def row(index, value, target, **extra):
    state = dict(at=(EPOCH + timedelta(hours=index)).isoformat(), streams={"a": [value]}, **extra)
    return TrainingRow(str(index), "next_value", json.dumps(state), target, "realized_value", False,
                       "normalised_squared_error", 1., "owner")


@pytest.mark.parametrize("variant,inputs,expected", [
    ("value", dict(at="2026-01-01T06:00:00Z", streams={"a": [12., 11.], "b": [.75]}),
     {"a": 12.13, "b": .7375}),
    ("logistic", dict(at="2026-01-05T00:00:00Z", inputs={"hours_since_owner_turn": 8., "channel=a": 1.}),
     {"reply_60m": .5498340}),
])
def test_literal_revisions_and_round_trip(tmp_path, variant, inputs, expected):
    revision = load_revision(FIXTURES / variant)
    assert revision.forecast(**inputs) == pytest.approx(expected, rel=1e-6)
    save_revision(tmp_path / "revision", revision.metadata, revision.weights)
    restored = load_revision(tmp_path / "revision")
    assert restored.metadata == revision.metadata
    assert restored.metadata["weights_digest"] == hashlib.sha256(restored.weights.tobytes()).hexdigest()
    assert restored.weights.dtype == np.float32 and restored.weights.flags.c_contiguous
    np.testing.assert_array_equal(restored.weights, revision.weights)
    with pytest.raises((ValueError, OSError)):
        save_revision(tmp_path / "revision", revision.metadata, revision.weights)


def test_anchor_predictions_and_validation():
    for variant in ("value", "logistic"):
        learner = NumericLearner(layout(variant))
        rev = learner.revision(learner.base())
        got = rev.forecast(at=EPOCH, streams={"a": [3.]}, inputs={"x": 4.})
        assert list(got.values()) == [3. if variant == "value" else .3]
    m = layout()
    m["standardisation"]["a"] = {"mean": 10., "scale": 2.}
    learner = NumericLearner(m)
    assert learner.revision(learner.base()).forecast(at=EPOCH, streams={"a": [12.123456789]})["a"] == 12.123456789
    with pytest.raises(ValueError):
        Revision(layout(), np.zeros((1, 4)))
    bad = layout()
    bad["standardisation"]["a"]["scale"] = 0
    with pytest.raises(ValueError):
        NumericLearner(bad)
    with pytest.raises(ValueError):
        NumericLearner(layout()).revision({"W": np.full((1, 4), np.nan, dtype=np.float32)})


def test_fifty_step_literal_engine_trace():
    trace = json.loads((FIXTURES / "step_trace.json").read_text())
    anchor = np.asarray(trace["anchor"], dtype=np.float32)
    weight = anchor.copy()
    assert len(trace["steps"]) == 50
    for step in trace["steps"]:
        weight = bounded_step(weight, anchor, step["x"], step["target"], rate=trace["rate"], clip=trace["clip"])
        np.testing.assert_allclose(weight, step["W"], rtol=1e-6, atol=1e-8)
        expected = np.asarray(step["W"], dtype=np.float64)
        # Relative matrix error is insensitive to near-zero individual entries.
        assert np.linalg.norm(weight - expected) / np.linalg.norm(expected) <= 1e-6


def test_value_and_logistic_budgets_project_every_step():
    value = NumericLearner(layout(), rate=10., clip=10., bound=.5)
    logistic = NumericLearner(layout("logistic"), rate=10., clip=10., bound=2.)
    for learner in (value, logistic):
        state = learner.base()
        for i in range(100):
            frame = dict(at=EPOCH.isoformat(), streams={"a": [3.]}, inputs={"x": 3.})
            state = learner.step(state, learner.base(), frame, 100. if learner.variant == "value" else 1.)
            assert learner.budget_used(state) <= learner.bound + 1e-6
        if learner.variant == "logistic":
            w = state["W"][0]
            assert abs(float(w[2])) + 3 * abs(float(w[0])) + abs(float(w[1])) <= 2.000001


def test_pair_breaks_and_safety_pause():
    learner = NumericLearner(layout(), max_gap=3600, safety={"a": 2.})
    for extra in ({"gap": True}, {"actuation": True}, {"alarm": True},
                  {"safety": {"a": 3.}}, {"previous_at": (EPOCH - timedelta(hours=2)).isoformat()}):
        result = learner.fit(learner.base(), [row(0, 1., 2., **extra)], objective=NEXT_VALUE, seed=0)
        np.testing.assert_array_equal(result["W"], learner.base()["W"])
    changed = learner.fit(learner.base(), [row(0, 1., 2., safety={"a": 4.})], objective=NEXT_VALUE, seed=0)
    assert not np.array_equal(changed["W"], learner.base()["W"])


def test_daily_cycle_arms_metrics_and_time_order():
    learner = NumericLearner(layout(), rate=.05, clip=.5)
    values = 2 + np.sin(np.arange(24 * 30 + 1) * 2 * np.pi / 24)
    train = [row(i + 1, values[i], values[i + 1]) for i in range(24 * 20)]
    test = [row(i + 1, values[i], values[i + 1]) for i in range(24 * 20, 24 * 30)]
    fitted = learner.fit(learner.base(), train[::-1], objective=NEXT_VALUE, seed=0)
    ordered = learner.fit(learner.base(), train, objective=NEXT_VALUE, seed=0)
    np.testing.assert_array_equal(fitted["W"], ordered["W"])
    arms = learner.arms(train, test, start=fitted, objective=NEXT_VALUE, seed=7)
    assert set(arms) == {"learned", "persistence", "seasonal_day", "seasonal_week", "ridge",
                         "time_shuffled", "no_accumulation"}
    truth = np.array([item.label for item in test])[:, None]
    errors = {name: np.mean((pred - truth) ** 2) for name, pred in arms.items()}
    assert errors["learned"] < errors["persistence"]
    assert errors["time_shuffled"] >= errors["persistence"]
    assert errors["seasonal_day"] < 1e-10 and errors["seasonal_week"] < 1e-10
    report = metrics([.2, .8], [0., 1.], variant="logistic", reference=[.5, .5])
    assert report["brier"]["mean"] == pytest.approx(.04)
    assert report["log_loss"]["mean"] == pytest.approx(-np.log(.8))
    assert report["brier"]["low"] <= .04 <= report["brier"]["high"]


def test_surprise_warms_up_and_scores_before_updating():
    surprise = Surprise(floor=.1)
    for _ in range(30):
        assert surprise.score(0.) is None
    assert surprise.score(4.) == pytest.approx(40.)


def test_lifetime_projection_keeps_the_period_envelope_too():
    m = layout("logistic")
    m["features"] = [{"input": str(i)} for i in range(10)]
    m["standardisation"] = {str(i): {"mean": 0., "scale": 1.} for i in range(10)}
    learner = NumericLearner(m, rate=10., clip=.01, bound=2.)
    start = learner.base()
    start["W"][0, 0] = .66
    frame = {"at": EPOCH.isoformat(), "inputs": {str(i): 3. for i in range(10)}}
    result = learner.step(start, start, frame, 1.)
    assert np.max(np.abs(result["W"] - start["W"])) <= .010001
    assert learner.budget_used(result) <= 2.000001
