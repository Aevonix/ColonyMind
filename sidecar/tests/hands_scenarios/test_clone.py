"""Integrator-only: every scenario and mutation on an isolated live Hermes clone."""

import json
import os

import pytest

from .catalog import SCENARIOS
from .replay import replay


@pytest.mark.hands_clone
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda case: f"s{case.number:02}")
def test_rc_clone_stage_one(scenario):
    configured = os.environ.get("HANDS_CLONE_COMMAND")
    if not configured:
        pytest.skip("Integrator: set HANDS_CLONE_COMMAND to a JSON argv for the isolated live Hermes clone driver")
    command = json.loads(configured)
    assert isinstance(command, list) and all(isinstance(arg, str) for arg in command)
    # A subset deliberately cannot return overall acceptance. Both results must
    # still prove this scenario, with the same assertions used in local CI.
    report = replay(command, stage=1, numbers=[scenario.number])
    assert [result["status"] for result in report["results"]] == ["passed", "mutation_killed"], report
