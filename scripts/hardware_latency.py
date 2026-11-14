"""Pytest support for explicit controlled-hardware latency qualification."""
import json
import math
from pathlib import Path
import platform
import statistics
import sys

import pytest


def pytest_addoption(parser):
    # Both conftests can be loaded by an explicit sidecar + adapter invocation.
    if getattr(parser, "_colonymind_hardware_latency_options", False):
        return
    parser._colonymind_hardware_latency_options = True
    group = parser.getgroup("hardware latency")
    group.addoption("--qualify-hardware-latency", action="store_true",
                    help="enforce hardware_latency bars on a controlled runner")
    group.addoption("--hardware-profile", help="controlled runner's qualification profile identifier")
    group.addoption("--hardware-latency-report", help="write measured hardware latency JSON")


def configure(config):
    config._hardware_latency_measurements = []
    if config.getoption("qualify_hardware_latency") and not (
        config.getoption("hardware_profile") and config.getoption("hardware_latency_report")
    ):
        raise pytest.UsageError("hardware qualification requires --hardware-profile and --hardware-latency-report")


@pytest.fixture
def hardware_latency(request, record_property):
    """Keep functional assertions unconditional; qualify only the measured latency."""
    # Explicit failures, not assert: this helper is not assertion-rewritten, and
    # PYTHONOPTIMIZE would strip an assert and silently skip the bound.
    if not request.node.get_closest_marker("hardware_latency"):
        pytest.fail("mark hardware-qualified tests explicitly")

    def measure(wall_ms, cpu_ms, *, threshold_ms, enforced=True, statistic="p99"):
        if statistic not in {"p99", "p95", "max"}:
            raise ValueError(f"unsupported latency statistic: {statistic}")
        wall_p99 = statistics.quantiles(wall_ms, n=100)[98] if len(wall_ms) > 1 else wall_ms[0]
        # Preserve the voice-affect gate's nearest-rank p95 exactly.
        wall_p95 = sorted(wall_ms)[math.ceil(len(wall_ms) * .95) - 1]
        cpu_p95 = sorted(cpu_ms)[math.ceil(len(cpu_ms) * .95) - 1]
        wall_value = {"max": max(wall_ms), "p99": wall_p99, "p95": wall_p95}[statistic]
        row = {"nodeid": request.node.nodeid, "samples": len(wall_ms), "threshold_ms": threshold_ms,
               "wall_p99_ms": wall_p99,
               "wall_p95_ms": wall_p95, "thread_cpu_p95_ms": cpu_p95,
               "thread_cpu_p99_ms": statistics.quantiles(cpu_ms, n=100)[98] if len(cpu_ms) > 1 else cpu_ms[0],
               "statistic": statistic, "wall_max_ms": max(wall_ms), "thread_cpu_max_ms": max(cpu_ms),
               "wall_samples_ms": wall_ms, "thread_cpu_samples_ms": cpu_ms,
               "within_bar": wall_value < threshold_ms, "enforced": enforced}
        request.config._hardware_latency_measurements.append(row)
        record_property("hardware_latency", json.dumps(row, sort_keys=True))
        if enforced and not row["within_bar"]:
            pytest.fail(f"hardware-qualified latency {statistic} {wall_value:.3f} ms >= {threshold_ms} ms")

    return measure


def pytest_sessionfinish(session):
    config = session.config
    path = config.getoption("hardware_latency_report")
    if path:
        report = {"profile": config.getoption("hardware_profile"), "python": sys.version,
                  "platform": platform.platform(), "processor": platform.processor(),
                  "measurements": config._hardware_latency_measurements}
        Path(path).write_text(json.dumps(report, indent=2) + "\n")


def pytest_terminal_summary(terminalreporter):
    for row in terminalreporter.config._hardware_latency_measurements:
        mode = "enforced" if row["enforced"] else "report only"
        statistic = row["statistic"]
        terminalreporter.write_line(
            f"hardware_latency {row['nodeid']}: wall {statistic}={row[f'wall_{statistic}_ms']:.3f} ms; "
            f"thread CPU {statistic}={row[f'thread_cpu_{statistic}_ms']:.3f} ms; bar <{row['threshold_ms']} ms; "
            f"within_bar={row['within_bar']}; {mode}")
