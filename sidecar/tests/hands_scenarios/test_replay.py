"""Tests first for the clone transport and its fail-closed acceptance gate."""

import sys

import pytest

from .replay import evaluate, replay


def envelope(request, observation, **overrides):
    return {"protocol": "hands-scenarios/1", "run_id": request["run_id"],
            "scenario": request["scenario"], "stage": request["stage"],
            "guard_disabled": request["guard_disabled"], "target": "rc-clone",
            "isolated": True, "evidence": ["trace.jsonl"],
            "observed": observation, **overrides}


def request(disabled=False):
    return {"run_id": "test-run", "scenario": 14, "stage": 1,
            "guard_disabled": disabled}


def observed():
    return {"refused": ["rcs", "api", "voice", "unknown"],
            "reasons": ["platform"] * 4, "dispatches": 0,
            "inline": 0, "front_continued": True}


def test_baseline_pass_and_mutation_kill_are_distinct():
    baseline = request()
    assert evaluate(baseline, envelope(baseline, observed()))["status"] == "passed"
    mutant = request(True)
    assert evaluate(mutant, envelope(mutant, observed()))["status"] == "mutation_survived"
    bad = {**observed(), "dispatches": 1}
    result = evaluate(mutant, envelope(mutant, bad))
    assert result["status"] == "mutation_killed" and result["assertion"] == "platform_refusal"


@pytest.mark.parametrize("overrides", [{"run_id": "old-run"}, {"stage": 2},
    {"target": "fake"}, {"isolated": False}, {"evidence": []},
    {"guard_disabled": True}, {"observed": {}}, {"error": "driver crashed"}])
def test_mismatched_or_incomplete_report_cannot_pass(overrides):
    body = request()
    assert evaluate(body, envelope(body, observed(), **overrides))["status"] == "error"


def test_missing_observations_do_not_count_as_killed_mutation():
    body = request(True)
    assert evaluate(body, envelope(body, {}))["status"] == "error"


def test_an_unrelated_failure_is_not_a_mutation_kill():
    body = {**request(True), "scenario": 2}
    response = envelope(body, {"process_gone": True, "stream_closed": True,
        "engine_before": 1, "engine_after": 1, "state": "stopped"})
    assert evaluate(body, response)["status"] == "failed"


@pytest.mark.parametrize("overrides", [{"guard_disabled": 0}, {"stage": True}])
def test_envelope_types_cannot_impersonate_matching_values(overrides):
    body = request()
    assert evaluate(body, envelope(body, observed(), **overrides))["status"] == "error"


def test_transport_sends_steps_and_requires_successful_process(tmp_path):
    script = tmp_path / "driver.py"
    script.write_text("import json, sys\nrequest = json.load(sys.stdin)\n"
                      "assert request['steps'] and request['guard']\n"
                      "print(json.dumps({'error': 'clone not connected'}))\n")
    report = replay([sys.executable, str(script)], stage=1, numbers=[14])
    assert report["ok"] is False
    assert len(report["results"]) == 2
    assert all(item["status"] == "error" for item in report["results"])


def test_partial_selection_cannot_claim_stage_acceptance(tmp_path):
    script = tmp_path / "driver.py"
    script.write_text("import json, sys\nr = json.load(sys.stdin)\n"
        "o = " + repr(observed()) + "\n"
        "if r['guard_disabled']: o['dispatches'] = 1\n"
        "r.update(target='rc-clone', isolated=True, evidence=['trace.jsonl'], observed=o)\n"
        "print(json.dumps(r))\n")
    report = replay([sys.executable, str(script)], stage=1, numbers=[14])
    assert report["ok"] is False and report["complete"] is False
    assert [item["status"] for item in report["results"]] == ["passed", "mutation_killed"]


def test_unconfigured_clone_is_never_green():
    with pytest.raises(ValueError, match="clone command"):
        replay([], stage=1)


@pytest.mark.parametrize("disabled,overrides,status", [
    (False, {}, "passed"),
    (False, {"elapsed": 1.541}, "failed"),
    (False, {"redirected": False, "request_dropped": True}, "passed"),
    (True, {"redirected": False}, "mutation_killed"),
    (True, {"redirected": False, "request_dropped": False}, "mutation_killed"),
    (True, {"redirected": False, "request_dropped": 1}, "error"),
    (True, {"baseline": [0.03, None, 0.05], "redirected": False}, "error"),
])
def test_owner_first_replay_requires_measured_baseline(disabled, overrides, status):
    body = {**request(disabled), "scenario": 10}
    observation = dict(next_turn="owner", redirected=True, elapsed=1.54,
                       baseline=[0.03, 0.04, 0.05])
    result = evaluate(body, envelope(body, {**observation, **overrides}))
    assert result["status"] == status
