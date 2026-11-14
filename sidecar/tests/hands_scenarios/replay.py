"""Replay all scenarios through an integrator's isolated rc-clone driver.

The driver receives one JSON request on stdin and returns observations on stdout.
It owns clone-specific Hermes lifecycle and platform instrumentation. This runner
does not substitute fake execution for a clone run or treat an unavailable driver
as a pass. Each request must start fresh isolated state and restore its mutation.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import uuid

from .catalog import MissingObservation, SCENARIOS, SafetyViolation, check


def evaluate(request, response):
    result = {"scenario": request["scenario"], "guard_disabled": request["guard_disabled"]}
    required = ("protocol", "run_id", "scenario", "stage", "guard_disabled")
    if (not isinstance(response, dict) or response.get("error")
            or any(type(response.get(key)) is not type(request.get(key, "hands-scenarios/1"))
                   or response.get(key) != request.get(key, "hands-scenarios/1") for key in required)
            or response.get("target") != "rc-clone" or response.get("isolated") is not True
            or not isinstance(response.get("evidence"), list) or not response["evidence"]):
        return {**result, "status": "error", "reason": "invalid clone evidence envelope"}
    scenario = SCENARIOS[request["scenario"] - 1]
    try:
        check(scenario.number, response.get("observed"))
    except (MissingObservation, TypeError, ValueError, KeyError):
        return {**result, "status": "error", "reason": "missing or invalid observations"}
    except SafetyViolation as exc:
        killed = request["guard_disabled"] and exc.code == scenario.assertion
        return {**result, "status": "mutation_killed" if killed else "failed", "assertion": exc.code}
    return {**result, "status": "mutation_survived" if request["guard_disabled"] else "passed"}


def replay(command, *, stage=1, numbers=None, timeout=300):
    """Timeout of 300 seconds per case is a runner assumption, configurable by CLI."""
    if not isinstance(command, (list, tuple)) or not command or not all(isinstance(arg, str) and arg for arg in command):
        raise ValueError("an explicit clone command is required")
    if type(stage) is not int or stage not in (1, 2, 4, 8):
        raise ValueError("stage must be 1, 2, 4 or 8")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    numbers = list(range(1, 15)) if numbers is None else list(numbers)
    if not numbers or len(set(numbers)) != len(numbers) or any(n not in range(1, 15) for n in numbers):
        raise ValueError("select distinct scenario numbers from 1 to 14")
    run_id, results = uuid.uuid4().hex, []
    for number in numbers:
        scenario = SCENARIOS[number - 1]
        for disabled in (False, True):
            request = {**asdict(scenario), "protocol": "hands-scenarios/1", "run_id": run_id,
                       "scenario": number, "stage": stage, "guard_disabled": disabled,
                       "max_running": stage, "max_concurrent_children": stage + 1}
            raw = ""
            try:
                process = subprocess.run(command, input=json.dumps(request), text=True,
                                         capture_output=True, timeout=timeout, check=False)
                raw = process.stdout
                if process.returncode != 0:
                    result = {"status": "error", "reason": "clone driver failed",
                              "exit_code": process.returncode}
                else:
                    response = json.loads(raw)
                    result = evaluate(request, response)
                    result["evidence"] = response.get("evidence", []) if isinstance(response, dict) else []
                    result["observed"] = response.get("observed") if isinstance(response, dict) else None
            except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
                result = {"status": "error", "reason": type(exc).__name__}
            results.append({**result, "scenario": number, "guard_disabled": disabled,
                            "stdout_sha256": hashlib.sha256(raw.encode()).hexdigest()})
    complete = set(numbers) == set(range(1, 15))
    return {"protocol": "hands-scenarios/1", "run_id": run_id, "target": "rc-clone", "stage": stage,
            "complete": complete, "results": results,
            "ok": complete and all(row["status"] == ("mutation_killed" if row["guard_disabled"] else "passed")
                                   for row in results)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--scenario", type=int, action="append", dest="numbers")
    parser.add_argument("--timeout", type=float, default=300, help="Per-case seconds; default is an assumption")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Clone driver argv, following --")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        report = replay(command, stage=args.stage, numbers=args.numbers, timeout=args.timeout)
    except ValueError as exc:
        parser.error(str(exc))
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{sum(r['status'] == 'passed' for r in report['results'])}/14 baseline scenarios passed; "
          f"{sum(r['status'] == 'mutation_killed' for r in report['results'])}/14 mutations killed")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
