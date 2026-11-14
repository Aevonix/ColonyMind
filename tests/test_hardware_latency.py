"""The hardware policy never masks functional failures and enforces its exact bar."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from scripts.hardware_latency import hardware_latency


@pytest.mark.parametrize("github_actions,latency,functional,expected,optimize", [
    ("true", 2, True, 0, ""), ("false", 2, True, 1, ""), ("false", .5, True, 0, ""), ("true", .5, False, 1, ""),
    # An optimized interpreter strips plain asserts; the bound must still hold.
    ("false", 2, True, 1, "1"), ("false", .5, True, 0, "1"),
])
@pytest.mark.parametrize("statistic", ["p99", "p95", "max"])
def test_hardware_latency_reporting_and_enforcement(tmp_path, github_actions, latency, functional, expected, optimize, statistic):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "conftest.py").write_text('''
from scripts.hardware_latency import (configure, hardware_latency, pytest_addoption,
                                      pytest_sessionfinish, pytest_terminal_summary)
def pytest_configure(config):
    configure(config)
    config.addinivalue_line('markers', 'hardware_latency: controlled hardware qualification')
''')
    (tmp_path / "test_gate.py").write_text(f'''
import os
import pytest
@pytest.mark.hardware_latency
def test_gate(hardware_latency):
    assert {functional!r}, 'functional failure'
    samples = [{latency}] * 100 if {statistic!r} != 'max' else [.1] * 999 + [{latency}]
    hardware_latency(samples, [.1] * len(samples), threshold_ms=1, statistic={statistic!r},
                     enforced=os.environ.get("GITHUB_ACTIONS") != "true")
''')
    report = tmp_path / "latency.json"
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
               f"--hardware-latency-report={report}"]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(root), "GITHUB_ACTIONS": github_actions,
                                 "PYTHONOPTIMIZE": optimize, "PYTEST_ADDOPTS": ""}, timeout=30)
    assert result.returncode == expected, result.stdout + result.stderr
    measurements = json.loads(report.read_text())["measurements"]
    if functional:
        row, = measurements
        assert row["enforced"] is (github_actions != "true") and row["within_bar"] is (latency < 1)
        assert row["threshold_ms"] == 1
        assert row["statistic"] == statistic and row["wall_max_ms"] == latency
    else:
        assert not measurements and "functional failure" in result.stdout


@pytest.mark.hardware_latency
@pytest.mark.parametrize('slow_samples', [15, 16])
def test_voice_affect_nearest_rank_p95_keeps_exact_bar(hardware_latency, request, monkeypatch, slow_samples):
    monkeypatch.setattr(request.config, '_hardware_latency_measurements', [], raising=False)
    samples = [.5] * (300 - slow_samples) + [20] * slow_samples
    if slow_samples == 15:
        hardware_latency(samples, samples, threshold_ms=20, statistic='p95')
    else:
        with pytest.raises(pytest.fail.Exception, match='p95 20.000 ms >= 20 ms'):
            hardware_latency(samples, samples, threshold_ms=20, statistic='p95')
    row = request.config._hardware_latency_measurements[-1]
    assert row['wall_p95_ms'] == (.5 if slow_samples == 15 else 20)
    assert row['within_bar'] is (slow_samples == 15)
