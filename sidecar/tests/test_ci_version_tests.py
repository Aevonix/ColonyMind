"""One job of the CI workflow runs the tests of the command that checks and sets the version.

``tests/test_set_version.py`` stands at the root of the repository, outside the sidecar's test directory: it runs
only where a job names it. Among its tests is the one that says every place of this checkout holds one version.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
JOBS = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"]
TESTS = "tests/test_set_version.py"


def test_one_job_runs_the_version_tests_beside_its_own_and_its_skip_check_reads_their_outcomes():
    assert (ROOT / TESTS).is_file()
    named = [(name, step) for name, job in JOBS.items() for step in job.get("steps", [])
             if {"pytest", TESTS} <= set(str(step.get("run", "")).split())]
    assert [name for name, _ in named] == ["hermes-adapter"]
    [(name, step)] = named
    words = step["run"].split()
    # The file is one more path on the line that runs the job's own tests: no job was made for it.
    assert "tests/hermes_adapter" in words
    [report] = [word.partition("=")[2] for word in words if word.startswith("--junitxml=")]
    later = JOBS[name]["steps"][JOBS[name]["steps"].index(step) + 1:]
    assert [str(other.get("run", "")) for other in later].count(f"python scripts/check_sidecar_skips.py {report}") == 1
