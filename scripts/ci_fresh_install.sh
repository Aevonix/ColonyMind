#!/usr/bin/env bash
# Fresh-install check against stock Hermes (build plan M1, acceptance tests 2, 3, 5, 6 and 9).
#
# Expects:
#   HERMES_SRC   a stock hermes-agent checkout (installed here, editable, into its own venv)
#   DIST_DIR     the built colonymind and colonymind_hermes wheels
# Optional HERMES_INSTALLER_HOME reuses a stock installer-made home, with uv on PATH.
# Otherwise everything is created under a temporary directory.
set -euo pipefail

: "${HERMES_SRC:?set HERMES_SRC to a stock hermes-agent checkout}"
: "${DIST_DIR:?set DIST_DIR to the directory holding the built wheels}"

work="$(mktemp -d)"
if [ -n "${HERMES_INSTALLER_HOME:-}" ]; then
  export HERMES_HOME="$HERMES_INSTALLER_HOME"
else
  export HOME="$work/home"
  export HERMES_HOME="$work/home/.hermes"
fi
export COLONYMIND_HOME="$work/home/.colonymind"
export COLONYMIND_INIT_NO_SERVICE=1
mkdir -p "$HERMES_HOME"

# The source-and-pip layout keeps its original setup. The installer already
# owns its interpreter and dependency environment; init discovers that one.
if [ -z "${HERMES_INSTALLER_HOME:-}" ]; then
  python -m venv "$work/hermes-venv"
  "$work/hermes-venv/bin/python" -m pip install --quiet --upgrade pip
  "$work/hermes-venv/bin/python" -m pip install --quiet -e "$HERMES_SRC"
  export PATH="$work/hermes-venv/bin:$PATH"
fi

# The sidecar in its own environment, from the built wheels (pipx in spirit).
python -m venv "$work/colonymind-venv"
"$work/colonymind-venv/bin/python" -m pip install --quiet --upgrade pip
sidecar_wheel="$(ls "$DIST_DIR"/colonymind-*.whl)"
adapter_wheel="$(ls "$DIST_DIR"/colonymind_hermes-*.whl)"
"$work/colonymind-venv/bin/python" -m pip install --quiet "$sidecar_wheel"
colonymind="$work/colonymind-venv/bin/colonymind"
discovered="$("$work/colonymind-venv/bin/python" -c 'from colonymind.init import resolve_hermes_python; print(resolve_hermes_python(None, None)[0])')"
if [ -z "${HERMES_INSTALLER_HOME:-}" ]; then
  # Source and pip: the interpreter is the one built above. Discovery must return it, every command is
  # given it by name, and pip check must pass with no exception.
  hermes_python="$work/hermes-venv/bin/python"
  if [ ! "$(dirname "$discovered")" -ef "$work/hermes-venv/bin" ]; then
    echo "discovery returned $discovered, not $hermes_python"; exit 1
  fi
  python_option=(--hermes-python "$hermes_python")
  check_dependencies() {
    "$hermes_python" -m pip check
  }
else
  # Hermes' installer: init discovers the environment the installer recorded. A conflict among Hermes' own
  # packages is a warning there; one that involves the adapter fails.
  hermes_python="$discovered"
  python_option=()
  check_dependencies() {
    "$work/colonymind-venv/bin/python" - "$hermes_python" <<'PY'
import sys
from pathlib import Path
from colonymind.init import pip_check, dependency_conflict_detail
python = Path(sys.argv[1])
ok, output = pip_check(python)
print(output)
if not ok:
    unrelated, detail = dependency_conflict_detail(python, output)
    assert unrelated, detail
    print("WARNING: dependency conflict in Hermes' own environment:", detail)
PY
  }
fi

# Another memory provider selected: init replaces it, says so, and uninstall puts it back.
cat > "$HERMES_HOME/config.yaml" <<'YAML'
model:
  provider: custom
  default: ci-model
  base_url: http://127.0.0.1:9/v1
memory:
  provider: example-memory
YAML

echo "== init"
"$colonymind" init --non-interactive --owner-name Owner --agent-name Agent \
  --hermes-home "$HERMES_HOME" ${python_option[@]+"${python_option[@]}"} \
  --adapter-source "$adapter_wheel" --no-service | tee "$work/init.log"
grep -q "replaces memory provider example-memory; 'colonymind init --uninstall' restores it" "$work/init.log"
grep -q "plugins.hook_callback_timeout: 0 is Hermes-wide" "$work/init.log"
grep -q "keyword recall only" "$work/init.log"

echo "== pip check in Hermes' environment (acceptance 2)"
check_dependencies

echo "== Hermes is unmodified: RECORD hashes or a clean checkout (acceptance 5)"
"$hermes_python" -I "$(dirname "$0")/check_hermes_record.py"

echo "== stock Hermes loads the config and the profile (acceptance 7, 9)"
"$hermes_python" -I - <<'PY'
import importlib.util
assert importlib.util.find_spec("colonymind") is None, "the sidecar must stay in its own environment"
from hermes_cli.config import load_config
from hermes_cli.profiles import profile_exists
config = load_config()
assert "colonymind" in config["plugins"]["enabled"], config["plugins"]
assert config["memory"]["provider"] == "colonymind-memory"
assert config["plugins"]["hook_callback_timeout"] == 0
assert profile_exists("colonymind-act")
import importlib.metadata as m
names = {e.name for e in m.entry_points(group="hermes_agent.plugins")}
assert "colonymind" in names, names
print("stock Hermes sees the adapter and the worker profile")
PY

echo "== upgrade twice (acceptance 3)"
"$colonymind" upgrade ${python_option[@]+"${python_option[@]}"} --adapter-source "$adapter_wheel"
before="$(find "$HERMES_HOME" "$COLONYMIND_HOME" -type f ! -path '*/backups/*' ! -name '*.db-shm' ! -name '*.db-wal' -exec sha256sum {} + | sort)"
second="$("$colonymind" upgrade ${python_option[@]+"${python_option[@]}"})"
echo "$second"
echo "$second" | grep -q "nothing to do"
after="$(find "$HERMES_HOME" "$COLONYMIND_HOME" -type f ! -path '*/backups/*' ! -name '*.db-shm' ! -name '*.db-wal' -exec sha256sum {} + | sort)"
test "$before" = "$after"
check_dependencies

echo "== doctor (the sidecar is not running here, so only that check may fail)"
"$colonymind" doctor --json > "$work/doctor.json" || true
"$work/colonymind-venv/bin/python" - "$work/doctor.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
failing = [r for r in report["results"] if r["status"] == "fail" and not r["name"].startswith("sidecar")]
assert not failing, failing
# No embedding endpoint here: recall is keyword-only and never abstains, and doctor says both, with the fix.
checks = {r["name"]: r for r in report["results"]}
assert checks["recall-mode"]["status"] == "warn", checks["recall-mode"]
assert checks["recall-mode"]["detail"].startswith("keyword recall only"), checks["recall-mode"]
assert "--embed-url auto" in checks["recall-mode"]["remedy"], checks["recall-mode"]
assert checks["recall-cutoff"]["detail"].startswith("no calibrated cutoff: recall never abstains")
print("doctor:", report["summary"])
PY

echo "== non-loopback bind without a key is refused (acceptance 6)"
mv "$COLONYMIND_HOME/api.key" "$COLONYMIND_HOME/api.key.off"
if "$colonymind" start --host 0.0.0.0 --port 7999 >/dev/null 2>&1; then
  echo "expected the bind to be refused"; exit 1
fi
mv "$COLONYMIND_HOME/api.key.off" "$COLONYMIND_HOME/api.key"

echo "== uninstall leaves stock Hermes (acceptance 9)"
"$colonymind" init --uninstall ${python_option[@]+"${python_option[@]}"} | tee "$work/uninstall.log"
grep -q "memory.provider restored to example-memory" "$work/uninstall.log"
"$hermes_python" -I - <<'PY'
from hermes_cli.config import load_config
from hermes_cli.profiles import profile_exists
config = load_config()
assert "colonymind" not in (config.get("plugins") or {}).get("enabled", []), config.get("plugins")
assert config["memory"]["provider"] == "example-memory", config.get("memory")
assert not profile_exists("colonymind-act")
import importlib.metadata as m
try:
    m.version("colonymind-hermes")
except m.PackageNotFoundError:
    print("adapter removed; stock Hermes starts with this config")
else:
    raise SystemExit("adapter still installed")
PY
"$hermes_python" -I "$(dirname "$0")/check_hermes_record.py"
echo "fresh install check passed"
