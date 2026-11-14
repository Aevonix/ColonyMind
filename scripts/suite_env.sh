#!/usr/bin/env bash
# Build the full-suite environment and run the sidecar and adapter suites in it.
#
# The hands, init and upgrade tests only run for real when Hermes core, the adapter and the sidecar are importable
# together and the tests know which Hermes they face. Without that they fail on imports or skip, and a suite count
# says nothing. Every count this prints names its environment; failures are compared with
# docs/internal/suite-known-failures.json both ways.
#
# Expects:
#   PYTHON312     a CPython 3.12 interpreter (the sidecar's documented interpreter)
#   HERMES_SRC    a Hermes checkout at a qualified revision (tests/hermes_adapter/hermes_source.py PINS)
#   HERMES_PYTHON the interpreter of a separate Hermes install whose metadata states its release version
#                 (run clones this unmarked seed; template-marked and unversioned installs are refused)
#   SUITE_VENV    existing dependency environment, or where build creates one
#   SUITE_PRIVATE_ROOT caller-supplied directory for temporary files and qualification outputs;
#                 every ancestor must pass the qualification output preflight (CI uses runner.temp)
#   HERMES_RUNTIME_REQUIREMENTS optional requirements file for a source-only compatibility
#                 run where Hermes' declared dependency markers omit this test interpreter
# Usage: scripts/suite_env.sh [build|run|both]   (default: both)
set -euo pipefail
umask 077

: "${PYTHON312:?set PYTHON312 to a CPython 3.12 interpreter}"
: "${HERMES_SRC:?set HERMES_SRC to a Hermes checkout at a qualified revision}"
: "${HERMES_PYTHON:?set HERMES_PYTHON to a stock-Hermes interpreter}"
: "${SUITE_VENV:?set SUITE_VENV to the suite environment directory}"
: "${SUITE_PRIVATE_ROOT:?set SUITE_PRIVATE_ROOT to an explicit private output root with qualifying ancestors}"
repo="$(cd "$(dirname "$0")/.." && pwd)"
# Nothing in this run writes bytecode into the checkouts it imports from. The variable, not -B: -B binds only the
# interpreter it is passed to, while tests start child interpreters (probes, qualification runs) that inherit the
# environment and would otherwise write caches beside another tree's sources.
export PYTHONDONTWRITEBYTECODE=1
export PYTHON_DOTENV_DISABLED=1 COLONYMIND_SKIP_DOTENV=1 LITELLM_LOCAL_MODEL_COST_MAP=True
unset COLONYMIND_URL COLONYMIND_API_KEY
export TMPDIR="$SUITE_PRIVATE_ROOT"
mode="${1:-both}"

check_tree() {
  if [ -f "$SUITE_VENV/SUITE-TREE" ] && [ "$(cat "$SUITE_VENV/SUITE-TREE")" != "$repo" ]; then
    echo "SUITE_VENV was not built for this tree; build a separate environment for the tested tree" >&2
    exit 2
  fi
}

import_proof() {
  "$SUITE_VENV/bin/python" -B - "$repo" <<'PY'
from pathlib import Path
import colonymind, colonymind_hermes, sys
tree = Path(sys.argv[1]).resolve()
paths = {m.__name__: Path(m.__file__).resolve() for m in (colonymind, colonymind_hermes)}
print("== import-proof: " + "; ".join(f"{name}.__file__={path}" for name, path in paths.items()), flush=True)
if not all(path.is_relative_to(tree) for path in paths.values()):
    sys.exit("Import provenance mismatch: both packages must be under the tested tree; refusing suite counts")
PY
}

source_overlay() {
  # Borrow installed dependencies without changing an editable install in a shared venv.
  # The package map is the same one the wheel build uses; imports are checked below.
  "$SUITE_VENV/bin/python" -I - "$repo" "$1" <<'PY'
from pathlib import Path
import sys, tomllib
tree, overlay = map(Path, sys.argv[1:])
overlay.mkdir()
config = tree / "pyproject.toml"
mapping = tomllib.loads(config.read_text()).get("tool", {}).get("setuptools", {}).get("package-dir", {}) if config.exists() else {}
for name, relative in mapping.items():
    if "." not in name:
        (overlay / name).symlink_to(tree / relative, target_is_directory=True)
PY
  export PYTHONPATH="$1:$repo/sidecar:$repo:$HERMES_SRC${PYTHONPATH:+:$PYTHONPATH}"
}

build() {
  if [ -e "$SUITE_VENV/SUITE-TREE" ]; then check_tree; fi
  "$PYTHON312" -m venv "$SUITE_VENV"
  local py="$SUITE_VENV/bin/python"
  "$py" -m pip install --quiet --upgrade pip
  "$py" -m pip install --quiet -e "$repo/sidecar[dev]" -e "${repo}[test]"
  # Match CI's stock editable install, including the installed provenance metadata.
  # A bare .pth file cannot establish which distribution belongs to the reviewed tree.
  # Init and upgrade tests still use the separate HERMES_PYTHON supplied below.
  "$py" -m pip install --quiet -e "$HERMES_SRC"
  if [ -n "${HERMES_RUNTIME_REQUIREMENTS:-}" ]; then
    "$py" -m pip install --quiet -r "$HERMES_RUNTIME_REQUIREMENTS"
  fi
  "$py" -m pip check
  "$py" -B -c "import tools.terminal_tool, colonymind, colonymind_hermes, aiosqlite"
  import_proof
  printf '%s\n' "$repo" > "$SUITE_VENV/SUITE-TREE"
}

run() {
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PIP_NO_INDEX=1
  local py="$SUITE_VENV/bin/python" out hermes_version hermes_prefix hermes_copy
  check_tree
  out="$(mktemp -d "$SUITE_PRIVATE_ROOT/colonymind-suite.XXXXXX")"
  chmod 0700 "$out"
  export TMPDIR="$out"
  # The run's own home, before anything of the tree is imported: the import proof, git and both phases read and
  # change nothing in the home of the user who runs the suite. (Each phase's conftest sets a home of its own.)
  mkdir "$out/home"
  export HOME="$out/home"
  unset XDG_CONFIG_HOME XDG_DATA_HOME XDG_CACHE_HOME XDG_STATE_HOME
  source_overlay "$out/imports"
  import_proof
  hermes_prefix="$("$HERMES_PYTHON" -I -c 'import sys; print(sys.prefix)')"
  if [ -e "$hermes_prefix/TEMPLATE-DO-NOT-USE-DIRECTLY" ]; then
    echo "HERMES_PYTHON must be a per-run copy; this one is marked as a template" >&2
    exit 2
  fi
  if [ ! -f "$hermes_prefix/pyvenv.cfg" ]; then
    echo "HERMES_PYTHON must belong to a virtual environment" >&2
    exit 2
  fi
  # This suite copies and verifies its Hermes seed by package metadata. It requires a release version there,
  # even though the installer also accepts Hermes' own install stamp when metadata is 0.0.0.
  # Reject placeholder metadata here before copy_suite_hermes.py performs the same seed check.
  hermes_version="$("$HERMES_PYTHON" -I -c 'import importlib.metadata as m; print(m.version("hermes-agent"))')"
  if [ "$hermes_version" = "0.0.0" ]; then
    echo "HERMES_PYTHON reports hermes-agent 0.0.0: install Hermes with its release version in its metadata" >&2
    exit 2
  fi
  # Qualification outputs require private ancestors, including the temp root.
  hermes_copy="$out/hermes-venv"
  "$py" -I "$repo/scripts/copy_suite_hermes.py" "$hermes_prefix" "$hermes_copy" "$hermes_version"
  export HERMES_SOURCE_ROOT="$HERMES_SRC" COLONYMIND_TEST_HERMES_PYTHON="$hermes_copy/bin/python" COLUMNS=400
  echo "== tree $(git -C "$repo" rev-parse HEAD); Hermes $(git -C "$HERMES_SRC" rev-parse HEAD 2>/dev/null || cat "$HERMES_SRC/SOURCE_REVISION"); test Hermes $hermes_version"
  echo "== venv $SUITE_VENV ($("$py" -V)); adapter $("$py" -c 'import importlib.metadata as m; print(m.version("colonymind-hermes"))'); TMPDIR and qualification output root $TMPDIR; bytecode not written (PYTHONDONTWRITEBYTECODE=1)"
  echo "== .pyc files outside __pycache__ in the trees it imports: $(find "$repo" "$HERMES_SRC" -name '*.pyc' -not -path '*/__pycache__/*' 2>/dev/null | wc -l)"
  local sidecar_status=0 adapter_status=0
  # Both phases load the result recorder by its module name from the import overlay, which is on both paths.
  ln -s "$repo/scripts/suite_results.py" "$out/imports/suite_results.py"
  (cd "$repo/sidecar" && "$py" -B -m pytest -q -p no:cacheprovider -p suite_results -rfE --junitxml="$out/sidecar.xml" \
    --suite-results="$out/sidecar-results.json" tests > "$out/sidecar.txt" 2>&1) || sidecar_status=$?
  (cd "$repo" && "$py" -B -m pytest -q -p no:cacheprovider -p scripts.suite_poll_loop -p suite_results -rfE \
    --junitxml="$out/adapter.xml" --suite-results="$out/adapter-results.json" tests > "$out/adapter.txt" 2>&1) || adapter_status=$?
  # A test or a fixture that calls pytest.exit ends the session there, with whatever status the call names.
  local phase
  for phase in sidecar adapter; do
    if "$py" -c 'import json, sys; sys.exit(not json.load(open(sys.argv[1], encoding="utf-8"))["ended_early"])' \
      "$out/$phase-results.json" 2>/dev/null; then
      echo "== full output: $out"
      echo "Suite did not complete normally; refusing suite counts ($phase: the session was ended early by pytest.exit)" >&2
      return 2
    fi
  done
  if [ "$sidecar_status" -gt 1 ] || [ "$adapter_status" -gt 1 ]; then
    echo "== full output: $out"
    echo "Suite did not complete normally; refusing suite counts (sidecar=$sidecar_status adapter=$adapter_status)" >&2
    return 2
  fi
  # pytest-timeout's thread method uses os._exit(1), just like an ordinary
  # failing exit status, but never reaches the JUnit session-finish writer.
  # Each output directory is fresh, so only a complete report proves this run
  # finished. Refuse aborted phases before consulting the known-failure list.
  # Which tests failed comes from pytest's own reports through the recorder,
  # never from terminal text: every failure and error in a JUnit report must
  # be a recorded test, and every recorded test a failure or error there.
  if ! "$py" - "$out" "$sidecar_status" "$adapter_status" <<'PY'
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET

out = Path(sys.argv[1])
failing = set()
for phase, status in zip(("sidecar", "adapter"), map(int, sys.argv[2:])):
    try:
        root = ET.parse(out / f"{phase}.xml").getroot()
        suites = root.findall("testsuite")
        if root.tag != "testsuites" or not suites:
            raise ValueError("missing test suites")
        failures = tests = 0
        reported = set()
        for suite in suites:
            counts = {name: int(suite.attrib[name]) for name in ("tests", "failures", "errors", "skipped")}
            if any(value < 0 for value in counts.values()):
                raise ValueError("negative result count")
            cases = suite.findall("testcase")
            entries = {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0}
            for case in cases:
                for tag, name in (("failure", "failures"), ("error", "errors"), ("skipped", "skipped")):
                    entries[name] += len(case.findall(tag))
            # Failures, errors and skips must equal the entries. pytest's own test count does not (subtests, a
            # teardown error after a reported call): it must be zero exactly when there is no entry, and never
            # below the number of tests the entries name. A run that only planned its fixtures (--setup-plan,
            # --setup-only) writes an entry for every test and counts none of them, so no test body ran.
            fits = all(entries[name] == counts[name] for name in ("failures", "errors", "skipped"))
            named = len({(case.get("classname"), case.get("name")) for case in cases})
            if not fits or bool(counts["tests"]) != bool(entries["tests"]) or counts["tests"] < named:
                raise ValueError(f"declared counts {counts} do not fit the test entries {entries}")
            tests += counts["tests"]
            failures += counts["failures"] + counts["errors"]
            reported.update((case.get("classname"), case.get("name")) for case in cases
                            if case.find("failure") is not None or case.find("error") is not None)
        if not tests:
            raise ValueError("no test ran")
        if bool(failures) != bool(status):
            raise ValueError("report disagrees with exit status")
        results = json.loads((out / f"{phase}-results.json").read_text(encoding="utf-8"))
        # A module that could not be collected ran none of its tests, whatever status pytest gave.
        if results["uncollected"]:
            raise ValueError("could not collect " + ", ".join(results["uncollected"]))
        recorded = results["failed"]
        if {tuple(junit) for junit in recorded.values()} != reported:
            raise ValueError("recorded failed tests disagree with the report")
    except (OSError, ET.ParseError, KeyError, TypeError, AttributeError, ValueError) as exc:
        sys.exit(f"Suite did not complete normally; refusing suite counts ({phase}: {exc})")
    failing.update(("sidecar/" if phase == "sidecar" else "") + name for name in recorded)
(out / "failing.txt").write_text("".join(name + "\n" for name in failing), encoding="utf-8")
PY
  then
    echo "== full output: $out"
    return 2
  fi
  # One complete id per line, in sort(1) order.
  sort -o "$out/failing.txt" "$out/failing.txt"
  echo "== sidecar: $(tail -n 1 "$out/sidecar.txt")"
  echo "== adapter: $(tail -n 1 "$out/adapter.txt")"
  local status=0
  echo "== full output: $out"
  "$py" - "$repo/docs/internal/suite-known-failures.json" "$out/failing.txt" <<'PY' || status=$?
from collections import Counter
import json, sys


def unique(pairs):
    twice = [key for key, count in Counter(key for key, _ in pairs).items() if count > 1]
    if twice:
        raise ValueError(f"the key {json.dumps(twice[0])} appears twice")
    return dict(pairs)


try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        known = json.load(stream, object_pairs_hook=unique)
    known = known.get("failures") if isinstance(known, dict) else None
    if not isinstance(known, dict) or not all(isinstance(cause, str) for cause in known.values()):
        raise ValueError("it is not one JSON object whose failures member maps test ids to text causes")
except (OSError, ValueError) as exc:
    print(f"Refusing suite counts: known-failure record {sys.argv[1]}: {exc}", file=sys.stderr)
    sys.exit(2)
# One complete id per line: nothing is trimmed, and only a line end separates two ids.
with open(sys.argv[2], encoding="utf-8", newline="") as stream:
    failing = {name for name in stream.read().split("\n") if name}
new, fixed = sorted(failing - set(known)), sorted(set(known) - failing)
for name in new:
    print("NEW   ", name)
for name in fixed:
    print("PASSES", name, f"({known[name]})")
print(f"== known failures: {len(failing & set(known))} of {len(known)} still fail; {len(new)} new; {len(fixed)} now pass")
sys.exit(1 if new else 0)
PY
  return "$status"
}

case "$mode" in
  build) build ;;
  run) run ;;
  both) [ -x "$SUITE_VENV/bin/python" ] || build; run ;;
  *) echo "usage: $0 [build|run|both]" >&2; exit 2 ;;
esac
