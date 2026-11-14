"""The replay gate must reject missing evidence and surviving mutations."""

import pytest

from .catalog import MissingObservation, SCENARIOS, SafetyViolation, check


def test_exactly_fourteen_numbered_scenarios():
    assert [scenario.number for scenario in SCENARIOS] == list(range(1, 15))
    assert all(scenario.steps and scenario.guard for scenario in SCENARIOS)


@pytest.mark.parametrize("number", range(1, 15))
def test_no_scenario_accepts_absent_observations(number):
    with pytest.raises(SafetyViolation, match="missing observation"):
        check(number, {})


def test_engine_release_requires_a_measured_decrease():
    observed = dict(process_gone=True, stream_closed=True, engine_before=1,
                    engine_after=1, state="stopped")
    with pytest.raises(SafetyViolation, match="engine_release"):
        check(2, observed)


@pytest.mark.parametrize("peak", [-1, True, "0", float("nan")])
def test_invalid_counts_are_not_safety_evidence(peak):
    with pytest.raises(SafetyViolation, match="invalid observation"):
        check(1, {"peak_hands": peak, "inline": 0, "answers": 4,
                  "paths": ["owner-a", "owner-b", "escalation", "continuation"]})


def test_unknown_delivery_is_not_a_receipt():
    observed = dict(lines=1, delivery_ids=["line-a", "line-a"],
                    receipt=None, attempts=2)
    with pytest.raises(SafetyViolation, match="notice_receipt"):
        check(5, observed)


def test_platform_refusal_tracks_explicit_rcs_launch_without_admitting_other_channels():
    observed = dict(launched_platforms=["whatsapp", "sms", "rcs"], refused=["api", "voice", "unknown"],
                    reasons=["platform"] * 3, dispatches=0, inline=0, front_continued=True)
    check(14, observed)
    with pytest.raises(SafetyViolation, match="platform_refusal"):
        check(14, {**observed, "launched_platforms": ["whatsapp", "sms"]})
    with pytest.raises(SafetyViolation, match="launch_platforms"):
        check(14, {**observed, "launched_platforms": ["api"]})


def test_final_intent_and_obligations_are_both_required():
    observed = dict(versions=[1, 2, 3], conflicts=["superseded", "superseded"],
                    outage=True, recovered=True, final_version=3, final_goal="latest",
                    expected_goal="latest", obligations=["open"])
    with pytest.raises(SafetyViolation, match="obligations_closed"):
        check(13, observed)


def test_a_superseded_obligation_needs_a_closed_successor():
    observed = dict(versions=[1, 2, 3], conflicts=["superseded", "superseded"],
                    outage=True, recovered=True, final_version=3, final_goal="latest",
                    expected_goal="latest", obligations=["superseded", "superseded", "communicated"])
    check(13, observed)
    for chain in (["communicated", "superseded"], ["superseded"], ["superseded", "open", "communicated"]):
        with pytest.raises(SafetyViolation, match="obligations_closed"):
            check(13, {**observed, "obligations": chain})


@pytest.mark.parametrize("baseline,elapsed,passes", [
    ([0.026, 0.048, 0.062, 0.037, 0.058], 1.548, True),
    ([0.026, 0.048, 0.062, 0.037, 0.058], 1.549, False),
    ([1, 1.1, 1.2], 2.6, True),
    ([1, 1.1, 1.2], 2.601, False),
    ([0.03, 0.04, 15], 1.541, False),
    ([0.2, 0.1, 0.4, 0.3], 1.75, True),
    ([0.2, 0.1, 0.4, 0.3], 1.751, False),
    ([0, 0, 0], 1.5, True),
    ([0, 0, 0], 1.501, False),
    ([0, 0, 0], -0.001, False),
])
def test_owner_first_uses_no_review_median_plus_margin(baseline, elapsed, passes):
    observed = dict(next_turn="owner", redirected=True, elapsed=elapsed, baseline=baseline)
    if passes:
        check(10, observed)
    else:
        with pytest.raises(SafetyViolation, match="owner_first"):
            check(10, observed)


def test_owner_first_requires_no_review_baseline_observation():
    with pytest.raises(MissingObservation, match="missing observation.*baseline"):
        check(10, dict(next_turn="owner", redirected=True, elapsed=0.1))


@pytest.mark.parametrize("baseline", [
    None, [], [0.01, 0.02], "0.01,0.02,0.03", (0.01, 0.02, 0.03),
    [0.01, 0.02, None], [0.01, 0.02, True], [0.01, 0.02, "0.03"],
    [0.01, 0.02, -0.03], [0.01, 0.02, float("nan")],
    [0.01, 0.02, float("inf")], [0.01, 0.02, float("-inf")],
])
def test_owner_first_refuses_invalid_baseline_samples(baseline):
    with pytest.raises(MissingObservation, match="invalid observation.*baseline"):
        check(10, dict(next_turn="owner", redirected=True, elapsed=0.1, baseline=baseline))


@pytest.mark.parametrize("overrides", [
    {"elapsed": None}, {"elapsed": True}, {"elapsed": "0.1"},
    {"elapsed": float("nan")}, {"elapsed": float("inf")},
    {"next_turn": None}, {"next_turn": "unknown"},
    {"redirected": None}, {"redirected": 1},
])
def test_owner_first_refuses_invalid_turn_observations(overrides):
    observed = dict(next_turn="owner", redirected=True, elapsed=0.1, baseline=[0.01] * 3)
    with pytest.raises(MissingObservation, match="invalid observation"):
        check(10, {**observed, **overrides})


@pytest.mark.parametrize("overrides", [{"next_turn": "review"}, {"redirected": False}])
def test_owner_first_keeps_next_turn_and_redirect_requirements(overrides):
    observed = dict(next_turn="owner", redirected=True, elapsed=0.1, baseline=[0.01] * 3)
    with pytest.raises(SafetyViolation, match="owner_first"):
        check(10, {**observed, **overrides})


def test_owner_first_accepts_observed_request_drop_without_native_redirect():
    check(10, dict(next_turn="owner", redirected=False, request_dropped=True,
                   elapsed=0.1, baseline=[0.01] * 3))


@pytest.mark.parametrize("request_dropped", [None, 0, 1, "true", [], {}])
def test_owner_first_requires_boolean_drop_observation_when_present(request_dropped):
    observed = dict(next_turn="owner", redirected=True, elapsed=0.1, baseline=[0.01] * 3)
    with pytest.raises(MissingObservation, match="invalid observation.*request_dropped"):
        check(10, {**observed, "request_dropped": request_dropped})


@pytest.mark.parametrize("overrides", [
    {"request_dropped": False}, {"next_turn": "review"}, {"elapsed": 1.511},
])
def test_observed_request_drop_keeps_owner_first_requirements(overrides):
    observed = dict(next_turn="owner", redirected=False, request_dropped=True,
                    elapsed=0.1, baseline=[0.01] * 3)
    with pytest.raises(SafetyViolation, match="owner_first"):
        check(10, {**observed, **overrides})
