#!/usr/bin/env bash
# Build plan M2 acceptance on stock Hermes: the first closed loop end to end, the ask path,
# the 72 h expiry, the digest, the off switch without a model, `mind why` and the dismissal
# multiplier (colonymind.devtools.mind_loop_probe), against a scripted model
# (colonymind.devtools.fake_model), both run from the built sidecar wheel, a real gateway whose
# platforms are the benchmark `capture` platform and a loopback webhook route, and
# `colonymind init` from the built wheels.
#
# Expects:
#   HERMES_SRC   a stock hermes-agent checkout
#   DIST_DIR     the built colonymind and colonymind_hermes wheels
# Optional: FAKE_MODEL_DUE_SECONDS (how soon a captured promise is due; default 45).
set -euo pipefail
: "${HERMES_SRC:?set HERMES_SRC to a stock hermes-agent checkout}"
: "${DIST_DIR:?set DIST_DIR to the directory holding the built wheels}"
here="$(cd "$(dirname "$0")" && pwd)"
work="${WORK_DIR:-$(mktemp -d)}"
mkdir -p "$work"
export WORK="$work"
if [ -n "${HERMES_INSTALLER_HOME:-}" ]; then
  export HERMES_HOME="$HERMES_INSTALLER_HOME"
else
  export HOME="$work/home"
  export HERMES_HOME="$work/home/.hermes"
fi
export COLONYMIND_HOME="$work/home/.colonymind"
export COLONYMIND_INIT_NO_SERVICE=1
export HERMES_DISABLE_TELEMETRY=1
export FAKE_MODEL_DUE_SECONDS="${FAKE_MODEL_DUE_SECONDS:-45}"
export FAKE_MODEL_LOG="$work/model.log"
export CAPTURE_OUTBOX="$work/outbox.json"
export CAPTURE_HOME_CHANNEL=owner
mkdir -p "$HERMES_HOME"
start=$(date +%s)
log() { echo "[$(( $(date +%s) - start ))s] $*"; }
cleanup() {
  for name in gateway sidecar model; do
    if [ -f "$work/$name.pid" ]; then kill "$(cat "$work/$name.pid")" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT

port_free() { python -c "import socket,sys; s=socket.socket(); s.settimeout(0.2); sys.exit(0 if s.connect_ex(('127.0.0.1', int(sys.argv[1]))) else 1)" "$1"; }
MODEL_PORT=18865; while ! port_free $MODEL_PORT; do MODEL_PORT=$((MODEL_PORT + 1)); done
SIDECAR_PORT=7851; while ! port_free $SIDECAR_PORT; do SIDECAR_PORT=$((SIDECAR_PORT + 1)); done
WEBHOOK_PORT=18944; while ! port_free $WEBHOOK_PORT; do WEBHOOK_PORT=$((WEBHOOK_PORT + 1)); done
export MODEL_URL="http://127.0.0.1:$MODEL_PORT/v1"
export SIDECAR_URL="http://127.0.0.1:$SIDECAR_PORT"
export WEBHOOK_URL="http://127.0.0.1:$WEBHOOK_PORT"

log "== the sidecar from the built wheels, in its own environment"
python -m venv "$work/colonymind-venv"
"$work/colonymind-venv/bin/python" -m pip install --quiet --upgrade pip
"$work/colonymind-venv/bin/python" -m pip install --quiet "$(ls "$DIST_DIR"/colonymind-*.whl)"
colonymind="$work/colonymind-venv/bin/colonymind"
export COLONYMIND_BIN="$colonymind"
# The scripted model and the probe are the wheel's own (colonymind.devtools), not repository copies.
colonymind_python="$work/colonymind-venv/bin/python"

cat > "$work/model.sh" <<SH
#!/usr/bin/env bash
case "\$1" in
  start) FAKE_MODEL_LOG="$FAKE_MODEL_LOG" FAKE_MODEL_DUE_SECONDS="$FAKE_MODEL_DUE_SECONDS" \\
           "$colonymind_python" -m colonymind.devtools.fake_model "$MODEL_PORT" >> "$work/model.out" 2>&1 &
         echo \$! > "$work/model.pid"
         for _ in \$(seq 1 30); do curl -sf "$MODEL_URL/models" > /dev/null && exit 0; sleep 0.5; done; exit 1;;
  stop) kill "\$(cat "$work/model.pid")" 2>/dev/null || true; sleep 1;;
esac
SH
log "== scripted model on 127.0.0.1:$MODEL_PORT (promises due after ${FAKE_MODEL_DUE_SECONDS}s)"
bash "$work/model.sh" start

log "== stock Hermes in its own environment ($HERMES_SRC)"
if [ -z "${HERMES_INSTALLER_HOME:-}" ]; then
  python -m venv "$work/hermes-venv"
  "$work/hermes-venv/bin/python" -m pip install --quiet --upgrade pip
  "$work/hermes-venv/bin/python" -m pip install --quiet -e "$HERMES_SRC"
  export PATH="$work/hermes-venv/bin:$PATH"
fi
discovered="$("$colonymind_python" -c 'from colonymind.init import resolve_hermes_python; print(resolve_hermes_python(None, None)[0])')"
# The release snapshot's webhook dependency (aiohttp); the guest turn arrives on this platform.
if [ -z "${HERMES_INSTALLER_HOME:-}" ]; then
  # Source and pip: the interpreter is the one built above. Discovery must return it, and init is given it by name.
  hermes_python="$work/hermes-venv/bin/python"
  if [ ! "$(dirname "$discovered")" -ef "$work/hermes-venv/bin" ]; then
    echo "discovery returned $discovered, not $hermes_python"; exit 1
  fi
  python_option=(--hermes-python "$hermes_python")
  "$hermes_python" -m pip install --quiet "aiohttp==3.14.4"
else
  # Hermes' installer: init discovers the environment the installer recorded; it may come without pip.
  hermes_python="$discovered"
  python_option=()
  if "$hermes_python" -m pip --version >/dev/null 2>&1; then
    "$hermes_python" -m pip install --quiet "aiohttp==3.14.4"
  else
    uv pip install --python "$hermes_python" --quiet "aiohttp==3.14.4"
  fi
fi
export HERMES_PYTHON="$hermes_python"

cat > "$HERMES_HOME/config.yaml" <<YAML
model:
  provider: custom
  default: scripted-model
  base_url: $MODEL_URL
plugins:
  enabled: [capture]
platforms:
  capture:
    enabled: true
  webhook:
    enabled: true
    extra:
      host: 127.0.0.1
      port: $WEBHOOK_PORT
      routes:
        guest:
          secret: INSECURE_NO_AUTH
          prompt: "{text}"
          deliver: log
kanban:
  dispatch_in_gateway: true
  dispatch_interval_seconds: 5
  default_assignee: default
tools:
  tool_search:
    enabled: off      # plugin tools offered directly, not behind tool_search (the model is scripted)
YAML
printf 'OPENAI_API_KEY=local-no-key\n' > "$HERMES_HOME/.env"
# The benchmark capture platform: every delivery lands in one JSON array (nothing is transmitted).
mkdir -p "$HERMES_HOME/plugins"
cp -r "$here/../benchmarks/paired/capture_platform" "$HERMES_HOME/plugins/capture"
echo '[]' > "$CAPTURE_OUTBOX"

log "== colonymind init --non-interactive (the three-command path, with the owner's capture handle; the owner"
log "   speaks on the CLI and the capture platform, so the key is granted both)"
"$colonymind" init --non-interactive --autonomy standard --owner-name Owner --agent-name Agent --owner-handle capture=owner \
  --owner-platform cli --owner-platform capture \
  --hermes-home "$HERMES_HOME" ${python_option[@]+"${python_option[@]}"} \
  --adapter-source "$(ls "$DIST_DIR"/colonymind_hermes-*.whl)" --no-service --port "$SIDECAR_PORT"
# No quiet hours, a digest that is due at once and no pause before an ask enters the conversation (the probe's owner
# speaks seconds before each ask forms): the acceptance list must not depend on the wall clock.
"$work/colonymind-venv/bin/python" - <<'PY'
import os, yaml
path = os.path.join(os.environ["COLONYMIND_HOME"], "colonymind.yaml")
config = yaml.safe_load(open(path))
config.setdefault("mind", {}).update({"quiet_hours": "", "digest_hour": 0, "asks": {"quiet_minutes": 0}})
open(path, "w").write(yaml.safe_dump(config, sort_keys=False))
PY

cat > "$work/sidecar.sh" <<SH
#!/usr/bin/env bash
# restart: stop the sidecar and start it again with the caller's environment
# (COLONYMIND_MIND_CLOCK_OFFSET_SECONDS shifts the mind's clock for the 72 h expiry check).
running() {
  local state
  state=\$(ps -o stat= -p "\$1" 2>/dev/null) || return 1
  [[ -n "\$state" && "\$state" != Z* ]]
}
if [ -f "$work/sidecar.pid" ]; then
  kill "\$(cat "$work/sidecar.pid")" 2>/dev/null || true
  if command -v ps > /dev/null; then
    for _ in \$(seq 1 120); do running "\$(cat "$work/sidecar.pid")" || break; sleep 0.5; done
    if running "\$(cat "$work/sidecar.pid")"; then
      echo "the old sidecar did not stop"; exit 1
    fi
  else
    # Without ps a process that ended cannot be told from one still running: wait on the signal alone, say so.
    echo "ps is missing: not checked that the old sidecar stopped"
    for _ in \$(seq 1 120); do kill -0 "\$(cat "$work/sidecar.pid")" 2>/dev/null || break; sleep 0.5; done
  fi
fi
cd "$work"
# Keep a parent alive to reap the sidecar on every restart.
(
  "$colonymind" start &
  child=\$!
  echo "\$child" > "$work/sidecar.pid"
  wait "\$child"
) </dev/null >> "$work/sidecar.log" 2>&1 &
for _ in \$(seq 1 120); do curl -sf "$SIDECAR_URL/v1/host/health" > /dev/null && exit 0; sleep 1; done
echo "the sidecar did not come up"; tail -40 "$work/sidecar.log"; exit 1
SH
log "== sidecar"
bash "$work/sidecar.sh" restart

cat > "$work/gateway.sh" <<SH
#!/usr/bin/env bash
if [ -f "$work/gateway.pid" ]; then
  kill -INT "\$(cat "$work/gateway.pid")" 2>/dev/null || true
  for _ in \$(seq 1 40); do kill -0 "\$(cat "$work/gateway.pid")" 2>/dev/null || break; sleep 0.5; done
  kill -9 "\$(cat "$work/gateway.pid")" 2>/dev/null || true
fi
[ "\$1" = stop ] && exit 0
cd "$work"
hermes gateway run >> "$work/gateway.log" 2>&1 &
echo \$! > "$work/gateway.pid"
for _ in \$(seq 1 120); do curl -sf "$WEBHOOK_URL/health" > /dev/null && exit 0; sleep 1; done
echo "the gateway did not come up"; tail -60 "$work/gateway.log"; exit 1
SH
log "== gateway (capture platform, loopback webhook route, kanban dispatcher every 5 s)"
bash "$work/gateway.sh" start
sleep 5
kill -0 "$(cat "$work/gateway.pid")" || { echo "the gateway died:"; tail -60 "$work/gateway.log"; exit 1; }

log "== the M2 acceptance list"
rc=0
"$colonymind_python" -m colonymind.devtools.mind_loop_probe || rc=$?
if [ "$rc" != 0 ]; then
  echo "--- gateway.log (tail)"; tail -80 "$work/gateway.log"
  echo "--- sidecar.log (tail)"; tail -80 "$work/sidecar.log"
  echo "--- outbox.json"; cat "$CAPTURE_OUTBOX"
  for diagnostic in outcome-recall.json chat-results.jsonl chat.log model.log; do
    if [ -f "$work/$diagnostic" ]; then
      echo "--- $diagnostic"
      cat "$work/$diagnostic"
    else
      echo "--- $diagnostic: missing"
    fi
  done
fi
log "acceptance finished with rc=$rc in $(( $(date +%s) - start )) s (work dir $work)"
exit $rc
