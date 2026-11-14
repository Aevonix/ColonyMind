"""The shared calibration, learned-factor and learner-report seams.

Numeric fixtures are synthetic assumptions chosen to exercise the seam bounds.
"""

from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from colonymind.mind.rank import Candidate, eligible, pick, rank, score
from test_mind_drives_loop import fx, idle  # noqa: F401
from test_mind_loop import pinned_clock  # noqa: F401

NOW = datetime(2026, 9, 30, 12, tzinfo=timezone.utc)

# Calibration seam tests.

@pytest.mark.parametrize("readout, suffix", [(None, ""), ("", ""), (False, ""), (3, ""),
                                             ("letters-r1", "/letters-r1"),
                                             ("names-free/chat/r2", "/names-free/chat/r2")])
def test_calibration_kind_readout(readout, suffix):
    from colonymind.instinct.providers import calibration_kind

    provider = SimpleNamespace(kind="fixture-provider")
    assert calibration_kind(provider) == "fixture-provider"
    provider.readout = readout
    assert calibration_kind(provider) == "fixture-provider" + suffix


async def test_calibration_record_and_answer_fingerprint_use_readout():
    from test_instinct_core import Provider, engine, record, yes_no
    from colonymind.instinct.registry import Tier

    point = yes_no(tiers=(Tier("fast", may_settle=("yes", "no"), state="live"),))
    provider = Provider(probe={"yes": 0.99, "no": 0.01})
    provider.readout = "letters-r1"
    bare = record(point)
    bound = replace(bare, provider_kind=provider.kind + "/letters-r1")
    other = replace(bare, provider_kind=provider.kind + "/letters-all")
    for records, expected in [([bare, other], "no_record"), ([bare, other, bound], "settled")]:
        service = engine([point], fast=provider, records=records)
        entry = {}
        result = await service._ask(point, point.tier("fast"), provider, "fixture", {}, entry)
        assert result == expected
        assert entry["fingerprint"] == bound.fingerprint
        assert service.record_for(point, "fast", provider, provider.revision) == (
            bound if expected == "settled" else None)


async def test_bakeoff_answers_carry_calibration_identity(tmp_path, monkeypatch):
    import json
    from unittest.mock import AsyncMock
    from test_instinct_bakeoff import measure

    provider = SimpleNamespace(kind="fixture-provider", readout="letters-r1", model="fixture-model")
    arms = SimpleNamespace(ask=AsyncMock(return_value={"base": {}}),
                           write=lambda path, payload: path.write_text(json.dumps(payload)))
    monkeypatch.setattr(measure, "_bakeoff_modules", lambda: (arms, None, None, None))
    monkeypatch.setattr(measure, "_frozen", lambda *args: {"probe": []})
    monkeypatch.setattr(measure, "_identity", lambda *args: {})
    monkeypatch.setattr(measure, "build_provider", lambda *args, **kwargs: provider)
    path = tmp_path / "answers.json"
    await measure.decide_bakeoff(SimpleNamespace(provider="local", sets=[], perturb=["base"],
                                                concurrency=1, out=str(path)))
    assert json.loads(path.read_text())["provider"] == {
        "kind": "fixture-provider/letters-r1", "model": "fixture-model"}


# Learner report seam tests.

def test_report_uses_optional_learner_hooks(monkeypatch):
    import sys
    from types import ModuleType
    from colonymind.reports.plasticity import collect, lines, to_json

    module = ModuleType("colonymind.plasticity.learners.fixture_report")
    calls = []

    def report(config, name, *, runtime=None):
        calls.append((config, name, runtime))
        return {"forecasts": 7}

    module.report = report
    module.report_lines = lambda detail: [f"Fixture forecasts: {detail['forecasts']}."]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = {"plasticity.learners": {"fixture": {"implementation": "fixture_report"},
                                      "plain": {"implementation": "uninstalled_fixture"}}}
    runtime = SimpleNamespace(report=lambda: {"learners": [{"name": "fixture"}, {"name": "plain"}]})
    result = collect(config, runtime=runtime)
    assert result["learners"][0]["detail"] == {"forecasts": 7}
    assert calls == [(config, "fixture", runtime)]
    assert lines(result)[0].endswith("Fixture forecasts: 7.")
    assert len(lines(result)) == 2
    assert '"forecasts": 7' in to_json(result)


@pytest.mark.parametrize("useful, ending", [(None, "awaiting usefulness verdicts"), (2.0, "2")])
def test_reply_timing_report_text_is_unchanged(monkeypatch, useful, ending):
    from colonymind.plasticity.learners import reply_timing
    from colonymind.reports.plasticity import collect, lines

    detail = {"holds": 2, "forced": 1, "served": 5, "outcomes": 4, "interruptions_per_useful": useful,
              "served_brier": 0.2, "served_log_loss": 0.3,
              "constant_brier": 0.25, "constant_log_loss": 0.6931,
              "hour_mark_brier": 0.4, "hour_mark_log_loss": 0.5}
    monkeypatch.setattr(reply_timing, "report", lambda *args, **kwargs: detail)
    runtime = SimpleNamespace(report=lambda: {"learners": [{"name": "timing"}]})
    config = {"plasticity.learners": {"timing": {"implementation": "reply_timing"}}}
    result = collect(config, runtime=runtime)
    assert result["learners"][0]["reply_timing"] == detail
    assert lines(result) == [
        "timing: 0 promoted, 0 refused, 0 rolled back; 0 erasures. "
        "The live comparison has no labelled outcomes yet. "
        "Reply timing: 2 holds, 1 forced runs (last 24 h); 5 served, 4 outcomes. "
        "served: Brier 0.2000, log loss 0.3000. constant 0.5: Brier 0.2500, log loss 0.6931. "
        "hour mark: Brier 0.4000, log loss 0.5000. Interruptions per useful notice: " + ending + "."]


# Learned-factor seam tests.

@pytest.fixture
def learned_module(monkeypatch):
    from colonymind.mind import learned

    monkeypatch.setattr(learned, "SPECS", {})
    return learned


def discretionary(**kwargs):
    return Candidate(**{"type": "research", "drive": "curiosity", "kind": "task", "title": "Research",
                        "dedup_key": "research:fixture", "salience": 0.8, "cost": 0.15, **kwargs})


def factor_view(module, implementations):
    runtime = SimpleNamespace(find=lambda implementation: implementations.get(implementation))
    return module.LearnedFactors(SimpleNamespace(plasticity=runtime)).at(NOW)


def test_empty_specs_keep_rank_pick_and_eligibility_identical(learned_module):
    research = discretionary()
    overdue = discretionary(type="commitment_overdue", drive="duty", dedup_key="commitment:c:overdue")
    candidates = [research, overdue]
    view = factor_view(learned_module, {})
    weights = {"duty": 1.0, "curiosity": 0.5}
    feedback = SimpleNamespace(multiplier=lambda key: 0.75)
    affect = SimpleNamespace(score_factor=lambda candidate: 1.2, threshold_factor=lambda candidate: 1.0)
    for options in ({"drives": weights}, {"drives": weights, "feedback": feedback, "affect": affect}):
        for operation in (rank, pick, eligible):
            assert operation(candidates, learned=view, **options) == operation(candidates, **options)
    assert [c.dedup_key for c, _ in eligible(candidates, drives=weights, learned=view)] == [
        "commitment:c:overdue", "research:fixture"]
    assert view.explain(research) == {}


@pytest.mark.parametrize("raw, clamped", [(-100.0, 0.5), (100.0, 1.5)])
def test_factor_range_and_registration(learned_module, raw, clamped):
    calls = []

    def factor(runtime, learner, candidate, now):
        calls.append((learner, candidate, now))
        return raw

    spec = learned_module.FactorSpec("fixture", 0.5, 1.5, factor)
    learned_module.register(spec)
    with pytest.raises(ValueError):
        learned_module.register(spec)
    candidate = discretionary()
    view = factor_view(learned_module, {"fixture": "loaded"})
    assert view.factor(candidate) == clamped
    assert view.explain(candidate) == {"fixture": clamped}
    assert calls == [("loaded", candidate, NOW)]
    assert factor_view(learned_module, {}).factor(candidate) == 1.0
    absent = learned_module.LearnedFactors(SimpleNamespace(plasticity=None)).at(NOW)
    assert absent.factor(candidate) == 1.0 and absent.explain(candidate) == {}
    assert absent.notes(SimpleNamespace()) == []


@pytest.mark.parametrize("value", [0.01, 100.0])
def test_factor_product_is_bounded_and_protects_owed_work_and_checkins(learned_module, value):
    learned_module.register(learned_module.FactorSpec("first", 0.5, 1.5, lambda *args: value))
    learned_module.register(learned_module.FactorSpec("second", 0.5, 1.0, lambda *args: value))
    view = factor_view(learned_module, {"first": "one", "second": "two"})
    feedback = SimpleNamespace(multiplier=lambda key: value)
    candidate = discretionary(salience=1.0, cost=0.0)
    expected = 0.25 if value < 1 else 2.25
    assert score(candidate, feedback=feedback, learned=view) == expected
    assert view.explain(candidate) == {"first": min(1.5, max(0.5, value)),
                                       "second": min(1.0, max(0.5, value))}
    for owed in (replace(candidate, source_type="commitment"),
                 replace(candidate, type="outreach_answer"), replace(candidate, type="outreach_followup")):
        assert view.factor(owed) == 1.0 and view.explain(owed) == {}
        assert score(owed, feedback=feedback, learned=view) == 1.0
    assert eligible([candidate], drives={"curiosity": 0}, feedback=feedback, learned=view) == []
    for kind in ("check_in", "commitment_check_in"):
        checkin = replace(candidate, type=kind, drive="social", kind="message", recipient="contact-1")
        # Even a factor above one cannot make an otherwise ineligible check-in due.
        assert eligible([checkin], threshold=1.1, feedback=feedback, learned=view) == []
        assert eligible([checkin], threshold=0.9, feedback=feedback, learned=view) == [(checkin, expected)]


def test_raising_and_nan_factors_are_neutral_and_log_only_exception_type(learned_module, caplog):
    def broken(*args):
        raise RuntimeError("fixture exception contents")

    learned_module.register(learned_module.FactorSpec("broken", 0.5, 1.5, broken, note=broken))
    learned_module.register(learned_module.FactorSpec("nan", 0.5, 1.5, lambda *args: float("nan")))
    view = factor_view(learned_module, {"broken": "first", "nan": "second"})
    candidate = discretionary()
    assert view.factor(candidate) == 1.0
    assert view.explain(candidate) == {"broken": 1.0, "nan": 1.0}
    assert view.notes(SimpleNamespace()) == []
    assert "RuntimeError" in caplog.text and "fixture exception contents" not in caplog.text


async def test_off_arm_dispatch_body_unchanged(fx, learned_module):
    fx.mind.add_interest("local history")
    await idle(fx)
    item, = fx.mind.dispatch()
    row = fx.store.get(item["id"])
    assert item["body"] == row.context["body"]
    assert "learned" not in row.context


async def test_factor_explanation_reaches_intention_and_dispatch_notes_are_bounded(fx, learned_module):
    seen = []

    def factor(runtime, learner, candidate, now):
        seen.append(candidate.dedup_key)
        return 1.5

    for implementation, note in (("fixture", "Use the check."), ("duplicate", "Use the check."),
                                 ("second", "Verify the result."), ("third", "Not included.")):
        learned_module.register(learned_module.FactorSpec(
            implementation, 0.5, 1.5, factor if implementation == "fixture" else lambda *args: 1.0,
            note=lambda *args, text=note: text))
    from threading import RLock
    fx.mind.plasticity = SimpleNamespace(enabled=True, lock=RLock(), find=lambda implementation: (
        implementation if implementation in learned_module.SPECS else None))
    fx.mind.add_interest("local history")
    summary, = await idle(fx)
    assert len(summary["formed"]) == 1
    item, = fx.mind.dispatch()
    row = fx.store.get(item["id"])
    assert row.context["learned"] == {"fixture": 1.5, "duplicate": 1.0, "second": 1.0, "third": 1.0}
    assert seen == [row.dedup_key]
    assert item["body"].endswith("Use the check.\n\nVerify the result.")
    assert item["body"].count("Use the check.") == 1 and "Not included." not in item["body"]
    assert fx.mind.dispatch() == [item]
    fx.store.update(row.id, context={**row.context, "body": item["body"]})
    assert fx.mind.dispatch() == [item]


def test_context_is_opt_in_and_snapshots_ranking_inputs(learned_module):
    from dataclasses import FrozenInstanceError
    from colonymind.mind.affect import AffectView

    legacy_calls, context_calls = [], []

    def legacy(*args):
        legacy_calls.append(args)
        return 1.0

    def contextual(*args, context):
        context_calls.append(context)
        return 1.0

    learned_module.register(learned_module.FactorSpec("legacy", 0.5, 1.5, legacy, note=legacy))
    learned_module.register(learned_module.FactorSpec(
        "contextual", 0.5, 1.5, contextual, note=contextual, wants_context=True))
    drives, base = {"curiosity": 0.4}, {"curiosity": 0.5}
    affect = AffectView(route={"priority": "rules"}, worry=0.2, curiosity=0.3)
    supplied = {"drives": drives, "base": base, "threshold": 0.7,
                "affect": affect, "last_owner_turn": NOW}
    runtime = SimpleNamespace(find=lambda implementation: implementation)
    view = learned_module.LearnedFactors(SimpleNamespace(plasticity=runtime)).at(NOW, supplied)
    drives["curiosity"] = 0.0
    base.clear()
    affect.route["priority"] = "state"
    supplied["threshold"] = 10.0
    assert view.factor(discretionary()) == 1.0
    view.notes(SimpleNamespace())
    assert len(legacy_calls) == 2 and all(len(args) == 4 for args in legacy_calls)
    assert context_calls == [view.context, view.context]
    assert view.context["drives"] == {"curiosity": 0.4}
    assert view.context["base"] == {"curiosity": 0.5}
    assert view.context["threshold"] == 0.7
    assert view.context["last_owner_turn"] == NOW
    assert view.context["affect"].route == {"priority": "rules"}
    with pytest.raises(TypeError):
        view.context["threshold"] = 0.0
    with pytest.raises(TypeError):
        view.context["drives"]["curiosity"] = 0.0
    with pytest.raises(TypeError):
        view.context["affect"].route["priority"] = "state"
    with pytest.raises(FrozenInstanceError):
        view.context["affect"].worry = 0.0


def test_default_context_keeps_legacy_dispatch_free_of_new_mind_reads(learned_module):
    def unreadable():
        raise AssertionError("legacy dispatch must not read ranking context")

    mind = SimpleNamespace(plasticity=None, effective_weights=unreadable,
                           drive_weights={"curiosity": 0.5}, act_threshold=0.7,
                           feelings=SimpleNamespace(view=unreadable), _owner_last_turn=unreadable)
    view = learned_module.LearnedFactors(mind).at(NOW)
    assert dict(view.context) == {"implementations": ()}
    assert view.notes(SimpleNamespace()) == []


def test_ranking_rescore_preserves_feedback_affect_and_other_factors(learned_module):
    from colonymind.mind.affect import AffectView

    learned_module.register(learned_module.FactorSpec("first", 0.5, 1.5, lambda *args: 1.5))
    learned_module.register(learned_module.FactorSpec("second", 0.5, 1.0, lambda *args: 0.5))
    runtime = SimpleNamespace(find=lambda implementation: implementation)
    context = {"drives": {"curiosity": 0.7}, "base": {"curiosity": 1.0},
               "threshold": 0.1, "affect": AffectView(route={}, curiosity=0.2), "last_owner_turn": NOW}
    view = learned_module.LearnedFactors(SimpleNamespace(plasticity=runtime)).at(NOW, context)
    multiplier = [2.0]
    feedback = view.feedback(SimpleNamespace(multiplier=lambda key: multiplier[0]))
    candidate = discretionary()
    ranked = eligible([candidate], threshold=0.1, drives=context["drives"], affect=context["affect"],
                      feedback=feedback, learned=view)
    ranking = view.ranking(ranked)
    expected = score(candidate, drives=context["drives"], affect=context["affect"],
                     feedback=SimpleNamespace(multiplier=lambda key: 1.0))
    multiplier[0] = 0.01
    candidate.salience = 0.01
    assert ranking.eligible == tuple(ranked)
    assert ranking.rescore(candidate, {}) == ranked[0][1]
    assert ranking.rescore(candidate, {"first": 1.0}) == pytest.approx(expected)
    assert ranking.rescore(candidate, {"first": 100.0, "second": 100.0}) == pytest.approx(expected * 2.25)
    assert ranking.rescore(candidate, {"unloaded": 0.5}) == ranked[0][1]


async def test_observer_runs_once_after_formation_and_isolates_faults(fx, learned_module, caplog):
    from threading import RLock

    observations = []

    def broken(*args):
        raise RuntimeError("fixture private observer detail")

    def observe(runtime, learner, ranking, now):
        observations.append((ranking, now))
        assert all(fx.store.get(intention_id) is not None for _, intention_id in ranking.acted)

    learned_module.register(learned_module.FactorSpec("broken", 0.5, 1.5, lambda *args: 1.0,
                                                    observe=broken))
    learned_module.register(learned_module.FactorSpec("fixture", 0.5, 1.5, lambda *args: 1.0,
                                                    observe=observe))
    fx.mind.plasticity = SimpleNamespace(enabled=True, lock=RLock(), find=lambda implementation: implementation)
    fx.mind.add_interest("local history")
    summary, = await idle(fx)
    assert len(observations) == 1
    ranking, now = observations[0]
    assert now == fx.now and len(ranking.eligible) == 1 and len(ranking.acted) == 1
    candidate, intention_id = ranking.acted[0]
    assert candidate is ranking.eligible[0][0]
    assert intention_id == summary["formed"][0]["id"]
    assert ranking.context["drives"] == fx.mind.effective_weights()
    assert ranking.context["base"] == fx.mind.drive_weights
    assert ranking.context["threshold"] == fx.mind.act_threshold
    assert "RuntimeError" in caplog.text and "fixture private observer detail" not in caplog.text
    await idle(fx)
    assert len(observations) == 2
    assert observations[-1][0].eligible == () and observations[-1][0].acted == ()


async def test_observer_excludes_unformed_candidate_from_acted(fx, learned_module, monkeypatch):
    from threading import RLock
    from unittest.mock import AsyncMock

    observed = []
    learned_module.register(learned_module.FactorSpec(
        "fixture", 0.5, 1.5, lambda *args: 1.0, observe=lambda runtime, name, ranking, now: observed.append(ranking)))
    fx.mind.plasticity = SimpleNamespace(enabled=True, lock=RLock(), find=lambda implementation: implementation)
    fx.mind.add_interest("local history")
    monkeypatch.setattr(fx.mind, "_form", AsyncMock(return_value=None))
    summary, = await idle(fx)
    assert summary["formed"] == []
    assert len(observed) == 1 and len(observed[0].eligible) == 1 and observed[0].acted == ()


def test_opted_in_neutral_overrides_use_baseline_with_other_loaded_factors(learned_module):
    learned_module.register(learned_module.FactorSpec("action_success", 0.5, 1.0, lambda *args: 0.5))
    contexts = []

    def exploratory(*args, context):
        contexts.append(context)
        return learned_module.FactorValue(1.0, overrides={name: 1.0 for name in context["implementations"]})

    learned_module.register(learned_module.FactorSpec("salience", 0.5, 1.5, exploratory, wants_context=True))
    view = factor_view(learned_module, {"action_success": "success", "salience": "salience"})
    candidate = discretionary()
    feedback = SimpleNamespace(multiplier=lambda key: 0.75)
    assert score(candidate, feedback=feedback, learned=view) == score(candidate, feedback=feedback)
    assert view.explain(candidate) == {"action_success": 1.0, "salience": 1.0}
    assert contexts[0]["implementations"] == ("action_success", "salience")
    with pytest.raises(ValueError):
        learned_module.FactorValue(1.0, overrides={"action_success": 0.5})


def test_legacy_factor_cannot_request_neutral_overrides(learned_module):
    learned_module.register(learned_module.FactorSpec("action_success", 0.5, 1.0, lambda *args: 0.5))
    learned_module.register(learned_module.FactorSpec(
        "legacy", 0.5, 1.5, lambda *args: learned_module.FactorValue(1.0, overrides={"action_success": 1.0})))
    view = factor_view(learned_module, {"action_success": "success", "legacy": "legacy"})
    assert view.factor(discretionary()) == 0.5


@pytest.mark.parametrize("decision", ["ask", "defer", "drop"])
async def test_observer_acted_excludes_intentions_authority_did_not_allow(fx, learned_module, monkeypatch, decision):
    from threading import RLock
    from colonymind.mind.authority import Verdict

    observations = []
    learned_module.register(learned_module.FactorSpec(
        "fixture", 0.5, 1.5, lambda *args: 1.0,
        observe=lambda runtime, name, ranking, now: observations.append(ranking)))
    fx.mind.plasticity = SimpleNamespace(enabled=True, lock=RLock(), find=lambda implementation: implementation)
    monkeypatch.setattr(fx.mind.authority, "decide", lambda **kwargs: Verdict(
        decision=decision, cls="internal", reason="Fixture authority decision"))
    fx.mind.add_interest("local history")
    summary, = await idle(fx)
    assert len(summary["formed"]) == 1 and summary["formed"][0]["decision"] == decision
    assert len(observations) == 1 and len(observations[0].eligible) == 1
    assert observations[0].acted == ()
