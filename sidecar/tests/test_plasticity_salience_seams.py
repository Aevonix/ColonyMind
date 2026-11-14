"""The salience observer receives the actual eligible ranking pass.

These replace the two original red probes for missing ranking context.
All numeric fixtures are builder assumptions chosen to separate rank and gate.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

from colonymind.mind import learned
from colonymind.mind.rank import Candidate, eligible


NOW = datetime(2026, 9, 30, 12, tzinfo=timezone.utc)


def observer(monkeypatch):
    calls = []
    runtime = SimpleNamespace(find=lambda implementation: "salience-fixture")

    def observe(runtime, name, ranking, now):
        calls.append(ranking)

    monkeypatch.setattr(learned, "SPECS", {})
    learned.register(learned.FactorSpec("salience", 0.5, 1.5, lambda *args: 1.0, observe=observe))
    return learned.LearnedFactors(SimpleNamespace(plasticity=runtime)), calls


def candidate(drive):
    return Candidate(type="research", drive=drive, kind="task", title="Fixture",
                     dedup_key="fixture:" + drive, salience=0.8, cost=0.1)


def observe_pass(factors, candidates, *, drives, threshold=0.6):
    view = factors.at(NOW, {"drives": drives, "base": drives, "threshold": threshold,
                            "affect": None, "last_owner_turn": None})
    ranked = eligible(candidates, drives=drives, threshold=threshold, learned=view)
    view.observe(view.ranking(ranked))
    return ranked


def test_propensity_observer_can_distinguish_actual_rank_order(monkeypatch):
    """The observer sees the exact served scores and order after weight reversal."""
    factors, calls = observer(monkeypatch)
    candidates = [candidate("curiosity"), candidate("mastery")]
    first = observe_pass(factors, candidates, drives={"curiosity": 1.0, "mastery": 0.5})
    second = observe_pass(factors, candidates, drives={"curiosity": 0.5, "mastery": 1.0})
    assert [item.drive for item, _ in first] == ["curiosity", "mastery"]
    assert [item.drive for item, _ in second] == ["mastery", "curiosity"]
    assert len(calls) == 2
    assert calls[0].eligible == tuple(first)
    assert calls[1].eligible == tuple(second)
    assert calls[0].context["drives"] != calls[1].context["drives"]


def test_propensity_observer_can_distinguish_eligible_candidates(monkeypatch):
    """A threshold-rejected item is absent even though its factor was evaluated."""
    factors, calls = observer(monkeypatch)
    item = candidate("curiosity")
    first = observe_pass(factors, [item], drives={"curiosity": 1.0}, threshold=0.6)
    second = observe_pass(factors, [item], drives={"curiosity": 1.0}, threshold=0.8)
    assert first and second == []
    assert len(calls) == 2
    assert calls[0].eligible == tuple(first)
    assert calls[1].eligible == ()
    assert calls[0].context["threshold"] == 0.6
    assert calls[1].context["threshold"] == 0.8
