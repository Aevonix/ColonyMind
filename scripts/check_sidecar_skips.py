#!/usr/bin/env python3
"""Check guarded pytest skips and every outcome in the adapter JUnit report."""
from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


SIDECAR_ABSENT = "sidecar absent: requires_sidecar"
SKIP_REASONS = {
    "requires_live_qualification": "requires the test's external live qualification prerequisite",
    "requires_sidecar": SIDECAR_ABSENT,
}


PATCH_HOOK_XFAIL_NODEID = (
    "tests/hermes_adapter/test_work_capture.py::test_maximum_resource_patch_hook_stays_under_one_ms"
)
# Only the documented GC latency limit is an accepted expected failure for this test.
PATCH_HOOK_XFAIL_PREFIX = "GC latency limit: "


class MarkedTests:
    def __init__(self):
        self.cases = []
        self.incomplete = False

    def pytest_collection_finish(self, session):
        for item in session.items:
            module, *classes, name = item.nodeid.split("::")
            classname = ".".join([module.removesuffix(".py").replace("/", "."), *classes])
            self.cases.append((classname, name))

    def pytest_collectreport(self, report):
        # A module-level import skip would hide its tests from the marker count.
        if report.failed or report.skipped:
            self.incomplete = True


def collect_requires_sidecar():
    import pytest

    marked = MarkedTests()
    output = io.StringIO()
    # Script invocation puts scripts/, rather than the repository, on the import path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        result = pytest.main(
            ["tests/hermes_adapter", "--collect-only", "-m", "requires_sidecar", "-q"],
            plugins=[marked],
        )
    if marked.incomplete or result not in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
        raise ValueError("marker collection did not complete:\n" + output.getvalue())
    return marked.cases


def check_report(path: Path, marked_sidecar=()) -> list[str]:
    cases = list(ET.parse(path).getroot().iter("testcase"))
    problems = []
    if not cases:
        problems.append("no test outcomes in report")
    expected = {reason: set() for reason in SKIP_REASONS.values()}
    expected[SIDECAR_ABSENT] = set(marked_sidecar)
    skipped = {reason: set() for reason in SKIP_REASONS.values()}
    seen = set()
    seen_identities = set()
    for case in cases:
        identity = (case.get("classname"), case.get("name"))
        skip = case.find("skipped")
        if skip is not None and "could not import 'colonymind" in (skip.get("message") or ""):
            problems.append(f"sidecar import skip without requires_sidecar: {identity}")
        if not all(identity) or identity in seen_identities:
            problems.append(f"missing or duplicate JUnit identity: {identity}")
        seen_identities.add(identity)
        properties = {p.get("name"): p.get("value") for p in case.findall("properties/property")}
        nodeid = properties.get("guard_nodeid")
        if not nodeid or nodeid in seen:
            problems.append(f"missing or duplicate node id: {nodeid}")
            continue
        seen.add(nodeid)
        marker = properties.get("guard_marker")
        reason = properties.get("guard_expected_skip")
        if marker:
            if marker not in SKIP_REASONS or reason not in ("", SKIP_REASONS[marker]):
                problems.append(f"invalid guarded marker/reason: {nodeid}")
            elif marker == "requires_sidecar":
                if bool(reason) != (identity in expected[SIDECAR_ABSENT]):
                    problems.append(f"sidecar expectation disagrees with collected marker: {nodeid}")
            elif reason:
                expected[reason].add(identity)
        elif reason:
            problems.append(f"skip expectation without marker: {nodeid}")
        if case.find("failure") is not None or case.find("error") is not None:
            problems.append(f"failed or errored: {nodeid}")
        if skip is not None:
            message = skip.get("message", "")
            if marker and not reason:
                problems.append(f"guarded test expected to run but skipped: {nodeid}: {message}")
            allowed_xfail = (
                nodeid == PATCH_HOOK_XFAIL_NODEID
                and identity == ("tests.hermes_adapter.test_work_capture",
                                 "test_maximum_resource_patch_hook_stays_under_one_ms")
                and skip.get("type") == "pytest.xfail"
                and message.startswith(PATCH_HOOK_XFAIL_PREFIX)
                and not marker and not reason
            )
            if allowed_xfail:
                continue
            if message in skipped:
                skipped[message].add(identity)
            else:
                problems.append(f"unguarded adapter skip: {nodeid}: {message}")
    for reason in expected:
        if skipped[reason] != expected[reason]:
            problems.append(f"{reason}: expected {sorted(expected[reason])}, skipped {sorted(skipped[reason])}")
    return problems


def check_skips(xml_path, marked):
    problems = check_report(Path(xml_path), marked_sidecar=marked)
    if problems:
        raise ValueError("\n".join(problems))


def main() -> int:
    try:
        path = Path(sys.argv[1])
        marked = collect_requires_sidecar() if importlib.util.find_spec("colonymind") is None else []
        problems = check_report(path, marked_sidecar=marked)
    except (IndexError, OSError, ValueError, ET.ParseError) as error:
        print(f"cannot check skip report: {error}", file=sys.stderr)
        return 1
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print("Guarded skips equal the marked tests with absent prerequisites; no failures, errors or unguarded adapter skips.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
