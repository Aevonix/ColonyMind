#!/usr/bin/env bash
# Build plan M1, acceptance test 1: stock Hermes, `colonymind init --non-interactive`,
# the sidecar and the gateway started, "remember X" in session A and X recalled in
# session B, all through `hermes chat -q` against a scripted model (colonymind.devtools.fake_model,
# run from the built sidecar wheel).
#
# Expects:
#   HERMES_SRC   a stock hermes-agent checkout
#   DIST_DIR     the built colonymind and colonymind_hermes wheels
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
export HERMES_DISABLE_TELEMETRY=1
mkdir -p "$HERMES_HOME"
start=$(date +%s)
log() { echo "[$(( $(date +%s) - start ))s] $*"; }
pids=()
cleanup() {
  local result=$?
  # Hermes 0.22 chat can start a persistent host gateway. The next proof step
  # keeps this installer home but writes a new config, so stop that gateway too.
  if [ -n "${HERMES_INSTALLER_HOME:-}" ]; then
    timeout 60 hermes gateway stop || result=1
  fi
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]:-}"; do wait "$pid" 2>/dev/null || true; done
  exit "$result"
}
trap cleanup EXIT

port_free() { python -c "import socket,sys; s=socket.socket(); s.settimeout(0.2); sys.exit(0 if s.connect_ex(('127.0.0.1', int(sys.argv[1]))) else 1)" "$1"; }
MODEL_PORT=18765; while ! port_free $MODEL_PORT; do MODEL_PORT=$((MODEL_PORT + 1)); done
SIDECAR_PORT=7841; while ! port_free $SIDECAR_PORT; do SIDECAR_PORT=$((SIDECAR_PORT + 1)); done

log "== stock Hermes in its own environment"
if [ -z "${HERMES_INSTALLER_HOME:-}" ]; then
  python -m venv "$work/hermes-venv"
  "$work/hermes-venv/bin/python" -m pip install --quiet --upgrade pip
  "$work/hermes-venv/bin/python" -m pip install --quiet -e "$HERMES_SRC"
  export PATH="$work/hermes-venv/bin:$PATH"
fi

log "== the sidecar from the built wheels, in its own environment"
python -m venv "$work/colonymind-venv"
"$work/colonymind-venv/bin/python" -m pip install --quiet --upgrade pip
"$work/colonymind-venv/bin/python" -m pip install --quiet "$(ls "$DIST_DIR"/colonymind-*.whl)"
colonymind="$work/colonymind-venv/bin/colonymind"

log "== scripted model on 127.0.0.1:$MODEL_PORT (the wheel's own, colonymind.devtools)"
FAKE_MODEL_LOG="$work/model.log" "$work/colonymind-venv/bin/python" -m colonymind.devtools.fake_model "$MODEL_PORT" \
  > "$work/model.out" 2>&1 &
pids+=($!)

cat > "$HERMES_HOME/config.yaml" <<YAML
model:
  provider: custom
  default: scripted-model
  base_url: http://127.0.0.1:$MODEL_PORT/v1
YAML
printf 'OPENAI_API_KEY=local-no-key\n' > "$HERMES_HOME/.env"

log "== colonymind init --non-interactive (the owner speaks on the CLI, so the key is granted it)"
"$colonymind" init --non-interactive --owner-name Owner --agent-name Agent --owner-platform cli \
  --hermes-home "$HERMES_HOME" \
  --adapter-source "$(ls "$DIST_DIR"/colonymind_hermes-*.whl)" --no-service --port "$SIDECAR_PORT"

log "== sidecar"
# A simple command, no subshell: the recorded pid is the sidecar's own, so the exit trap really stops it.
cd "$work"
"$colonymind" start > "$work/sidecar.log" 2>&1 &
pids+=($!)
cd "$OLDPWD"
for _ in $(seq 1 90); do curl -sf "http://127.0.0.1:$SIDECAR_PORT/v1/host/health" > /dev/null && break; sleep 1; done
curl -sf "http://127.0.0.1:$SIDECAR_PORT/v1/host/health" > /dev/null

log "== gateway starts with the adapter, then is stopped"
(cd "$work" && timeout 25 hermes gateway run > "$work/gateway.log" 2>&1 || true) &
gw=$!
sleep 15
kill -INT $gw 2>/dev/null || true; wait $gw 2>/dev/null || true
if grep -q "Traceback" "$work/gateway.log"; then echo "gateway traceback:"; grep -A12 Traceback "$work/gateway.log" | head -40; exit 1; fi

token="mem-$(python -c 'import secrets; print(secrets.token_hex(4))')"
log "== session A: remember $token"
(cd "$work" && timeout 300 hermes chat -q "Please remember this token for me: $token. Just acknowledge." > "$work/session-a.log" 2>&1)
grep -q "$token" "$work/session-a.log"

log "== memory formation (asynchronous extraction and review through the scripted model)"
owner="$("$work/colonymind-venv/bin/python" -c "import yaml; print(yaml.safe_load(open('$COLONYMIND_HOME/colonymind.yaml'))['owner']['contact_id'])")"
key="$(cat "$COLONYMIND_HOME/api.key")"
formed=""
for i in $(seq 1 90); do
  body="$(curl -s -H "Authorization: Bearer $key" -H "content-type: application/json" \
    -X POST "http://127.0.0.1:$SIDECAR_PORT/v1/host/context/assemble" \
    -d "{\"identity\":{\"host_id\":\"hermes\"},\"context\":{\"session_id\":\"probe-$i\",\"contact_id\":\"$owner\"},\"incoming_message\":{\"role\":\"user\",\"content\":\"What was the token I asked you to remember earlier?\"}}")"
  case "$body" in *"$token"*) formed=$i; break;; esac
  sleep 2
done
test -n "$formed" || { echo "the token never became recallable through /context/assemble"; tail -40 "$work/sidecar.log"; exit 1; }
log "recallable after $formed polls of 2 s"

log "== session B: recall in a fresh session"
(cd "$work" && timeout 300 hermes chat -q "What was the token I asked you to remember earlier? Reply with it." > "$work/session-b.log" 2>&1)
grep -q "$token" "$work/session-b.log" || { echo "session B did not recall the token:"; cat "$work/session-b.log"; exit 1; }
"$work/colonymind-venv/bin/python" - "$work/model.log" "$token" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
def text(c):
    return " ".join(str(b.get("text") or "") for b in c if isinstance(b, dict)) if isinstance(c, list) else (c or "")
prompts = ["\n".join(text(m.get("content")) for m in r["body"].get("messages") or [])
           for r in rows if any(m.get("role") == "user" and text(m.get("content")).startswith("What was the token")
                                and "<memory-context>" in text(m.get("content")) for m in r["body"].get("messages") or [])]
assert prompts and all(sys.argv[2] in p for p in prompts), "session B's prompt did not carry the recalled token"
print("session B's prompt carried the recalled token; the answer repeated it")
PY
log "acceptance test 1 passed in $(( $(date +%s) - start )) s"
