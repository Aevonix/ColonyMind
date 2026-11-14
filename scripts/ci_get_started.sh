#!/usr/bin/env bash
# The README's "Get started" commands, run as a reader copies them, on a fresh machine with stock Hermes.
# scripts/get_started_commands.py reads them from the page and each runs as written.
#
# Expects:
#   HERMES_SRC   a stock hermes-agent checkout (installed here, editable, into its own environment)
#   DIST_DIR     the built colonymind and colonymind_hermes wheels
#   uv           on PATH (the page's first command is uv's)
#
# What the page leaves to the reader comes from the environment, never from a changed command: the built wheels
# stand in for the package index (UV_FIND_LINKS, and a constraint that pins this build's version), init's answers
# are its COLONYMIND_INIT_* variables, and Hermes' model is an endpoint on a closed local port. A runner has no user
# service manager, so init skips the service (COLONYMIND_INIT_NO_SERVICE) and the sidecar is started the way the
# service starts it (`colonymind start`) before the gateway command. With no gateway service installed,
# `hermes gateway restart` runs the gateway in the foreground; it is left running while the checks read it. Then:
# the installed tool is this build, doctor passes every check but the service's, Hermes' own
# `hermes colonymind-memory status` reaches the sidecar through the memory provider, and the gateway runs with no
# ColonyMind plugin refused.
set -euo pipefail

: "${HERMES_SRC:?set HERMES_SRC to a stock hermes-agent checkout}"
: "${DIST_DIR:?set DIST_DIR to the directory holding the built wheels}"

root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
export HOME="$work/home"
export HERMES_HOME="$HOME/.hermes"
unset COLONYMIND_HOME COLONYMIND_API_KEY
mkdir -p "$HERMES_HOME"
log() { echo "== $*"; }

sidecar_wheel="$(ls "$DIST_DIR"/colonymind-*.whl)"
adapter_wheel="$(ls "$DIST_DIR"/colonymind_hermes-*.whl)"
version="$(basename "$sidecar_wheel" | cut -d- -f2)"

gateway=""
sidecar=""
stop() {
  for pid in $gateway $sidecar; do
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
  done
}
trap stop EXIT

log "stock Hermes in its own environment"
uv venv --quiet --python 3.12 "$work/hermes-venv"
uv pip install --quiet --python "$work/hermes-venv/bin/python" -e "$HERMES_SRC"
export PATH="$HOME/.local/bin:$work/hermes-venv/bin:$PATH"
cat > "$HERMES_HOME/config.yaml" <<'YAML'
model:
  provider: custom
  default: ci-model
  base_url: http://127.0.0.1:9/v1
YAML

# The reader's side of the page: this build in place of the index's, and init's answers.
printf 'colonymind==%s\n' "$version" > "$work/constraints.txt"
export UV_FIND_LINKS="$DIST_DIR" UV_CONSTRAINT="$work/constraints.txt"
port="$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
export COLONYMIND_INIT_NON_INTERACTIVE=1 COLONYMIND_INIT_OWNER_NAME=Owner COLONYMIND_INIT_AGENT_NAME=Agent \
  COLONYMIND_INIT_PORT="$port" COLONYMIND_INIT_ADAPTER_SOURCE="$adapter_wheel" COLONYMIND_INIT_NO_SERVICE=1

wait_for_sidecar() {
  python3 - "$port" "$HOME/.colonymind/api.key" <<'PY'
import sys, time, urllib.request
port, key_file = sys.argv[1], sys.argv[2]
deadline = time.monotonic() + 120
while True:
    try:
        key = open(key_file, encoding="utf-8").read().strip()
        request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/host/health",
                                         headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(request, timeout=5) as response:
            if response.status == 200:
                break
    except OSError:
        pass
    if time.monotonic() > deadline:
        raise SystemExit("the sidecar did not answer its health check within 120 s")
    time.sleep(1)
print("the sidecar answers on port", port)
PY
}

wait_for_gateway() {
  python3 - "$HERMES_HOME/gateway_state.json" "$gateway" <<'PY'
import json, os, sys, time
state_file, pid = sys.argv[1], int(sys.argv[2])
deadline = time.monotonic() + 180
while True:
    try:
        if json.load(open(state_file, encoding="utf-8")).get("gateway_state") == "running":
            break
    except (OSError, ValueError):
        pass
    try:
        os.kill(pid, 0)
    except OSError:
        raise SystemExit("the gateway exited before it was running")
    if time.monotonic() > deadline:
        raise SystemExit("the gateway was not running within 180 s")
    time.sleep(1)
print("the gateway is running")
PY
}

mapfile -t commands < <(python3 "$root/scripts/get_started_commands.py" "$root/README.md")
[ "${#commands[@]}" -gt 0 ]
for command in "${commands[@]}"; do
  log "$command"
  case "$command" in
    "hermes gateway "*)
      # init starts the sidecar as a user service; a runner has none, so it starts here the way the service does.
      (cd "$HOME/.colonymind" && exec colonymind start) > "$work/sidecar.log" 2>&1 &
      sidecar=$!
      wait_for_sidecar
      bash -c "$command" > "$work/gateway.log" 2>&1 &
      gateway=$!
      wait_for_gateway
      ;;
    *)
      bash -c "$command"
      ;;
  esac
done

log "the installed tool is this build ($version)"
uv tool list | tee "$work/tools.txt"
grep -q "^colonymind v$version\$" "$work/tools.txt"

log "doctor: every check passes but the service's (a runner has no user service manager)"
colonymind doctor --json > "$work/doctor.json" || true
python3 - "$work/doctor.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
failing = [result for result in report["results"] if result["status"] == "fail" and result["name"] != "service"]
assert not failing, failing
print("doctor:", report["summary"])
PY

log "Hermes reaches the sidecar through the memory provider"
hermes colonymind-memory status

log "the gateway runs with no ColonyMind plugin refused"
kill -0 "$gateway"
if grep -h "Plugin 'colonymind[a-z-]*' not loaded" "$HERMES_HOME"/logs/*.log "$work/gateway.log" 2>/dev/null; then
  echo "a ColonyMind plugin was refused"
  exit 1
fi
grep -h "Plugin discovery complete" "$HERMES_HOME"/logs/agent.log | tail -1 || true

echo "the Get started commands work on a fresh machine"
