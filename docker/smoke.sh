#!/usr/bin/env bash
# The container quickstart's CI check (the docker job in ci.yml), on the image docker/Dockerfile built:
#
#  1. the Hermes inside is the release CI qualifies (the base image's own provenance record names HERMES_REF), and
#     the image carries no instance: nothing is written before the container's first start;
#  2. a container on two fresh volumes sets itself up and turns healthy, and `colonymind doctor` inside it passes
#     (warnings allowed; the one failure accepted is a dependency conflict Hermes' own image already carries,
#     compared with that image pulled by the digest the quickstart names);
#  3. stock Hermes inside it remembers through ColonyMind: told something in one `hermes chat` session, it answers
#     from memory in the next (the scripted model, colonymind.devtools.fake_model, on the container's loopback);
#  4. `colonymind demo`, the keyless mind loop, passes inside it, answered the way a person answers its ask;
#  5. a restart changes nothing (the same key, the adapter's Hermes keys once) and doctor still passes;
#  6. a new container on the same volumes with another HERMES_UID/HERMES_GID takes the instance over and is healthy.
#
# Containers are stopped before they are removed; nothing is removed by force.
#
# Expects:
#   IMAGE        the image to check (loaded into the local Docker)
#   HERMES_REF   the Hermes commit CI qualifies (ci.yml's env)
set -euo pipefail
: "${IMAGE:?set IMAGE to the image docker/Dockerfile built}"
: "${HERMES_REF:?set HERMES_REF to the Hermes commit CI qualifies}"
work="$(mktemp -d)"
name="colonymind-smoke-$$"
volumes=("$name-hermes" "$name-instance")
start=$(date +%s)
log() { echo "[$(( $(date +%s) - start ))s] $*"; }

cleanup() {
  docker stop --time 30 "$name" > /dev/null 2>&1 || true
  docker rm "$name" > /dev/null 2>&1 || true
  for volume in "${volumes[@]}"; do docker volume rm "$volume" > /dev/null 2>&1 || true; done
}
trap cleanup EXIT

run_container() {  # extra `docker run` options...
  docker run --detach --name "$name" \
    --volume "${volumes[0]}:/opt/data" --volume "${volumes[1]}:/var/lib/colonymind" \
    --env COLONYMIND_INIT_OWNER_NAME=Owner --env COLONYMIND_INIT_OWNER_HANDLES=telegram=5550100 \
    --env COLONYMIND_CONTAINER_MODEL_URL=http://127.0.0.1:18865/v1 --env COLONYMIND_CONTAINER_MODEL=scripted-model \
    "$@" "$IMAGE" sleep infinity > /dev/null
}

wait_healthy() {  # up to 240 s (a chosen bound: a first start runs init before the sidecar comes up)
  for _ in $(seq 1 120); do
    case "$(docker inspect --format '{{.State.Health.Status}}' "$name")" in
      healthy) log "healthy"; return 0;;
      unhealthy) break;;
    esac
    [ "$(docker inspect --format '{{.State.Running}}' "$name")" = true ] || break
    sleep 2
  done
  echo "the container did not turn healthy:"; docker logs "$name" 2>&1 | tail -100
  return 1
}

conflicts() {  # the dependency conflicts `uv pip check` reports in an image's Hermes environment
  docker run --rm --network none --entrypoint uv "$1" pip check --python /opt/hermes/.venv/bin/python 2>&1 \
    | grep '^The package' | sort || true
}

base="$(docker image inspect --format \
  '{{ index .Config.Labels "org.opencontainers.image.base.name" }}@{{ index .Config.Labels "org.opencontainers.image.base.digest" }}' \
  "$IMAGE")"
log "== the base image, for the conflicts it carries itself: $base"
docker pull --quiet "$base" > /dev/null
conflicts "$base" > "$work/base-conflicts.txt"
conflicts "$IMAGE" > "$work/image-conflicts.txt"
if ! cmp -s "$work/base-conflicts.txt" "$work/image-conflicts.txt"; then
  echo "the image's Hermes environment has conflicts its base does not:"
  diff "$work/base-conflicts.txt" "$work/image-conflicts.txt" || true
  exit 1
fi

doctor_passes() {  # `colonymind doctor` inside the container: no failure but Hermes' own conflict, if its base has one
  docker exec "$name" colonymind doctor || true
  docker exec "$name" colonymind doctor --json > "$work/doctor.json" || true
  python3 - "$work/doctor.json" "$work/base-conflicts.txt" <<'PY'
import json, sys
results = json.load(open(sys.argv[1]))["results"]
failed = sorted(r["name"] for r in results if r["status"] == "fail")
inherited = {"pip-check"} if open(sys.argv[2]).read().strip() else set()
unexpected = [name for name in failed if name not in inherited]
passed = sum(r["status"] == "pass" for r in results)
print(f"doctor: {passed} pass, failing {failed or 'nothing'}"
      + (" (Hermes' own image already fails pip-check)" if set(failed) & inherited else ""))
sys.exit(1 if unexpected or not passed else 0)
PY
}

log "== the Hermes inside is the qualified release"
revision="$(docker run --rm --network none --entrypoint cat "$IMAGE" /etc/hermes/image-provenance.json \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["revision"])')"
[ "$revision" = "$HERMES_REF" ] || { echo "the image's Hermes is $revision, not the qualified $HERMES_REF"; exit 1; }
log "== the image carries no instance"
leftovers="$(docker run --rm --network none --entrypoint find "$IMAGE" /var/lib/colonymind -mindepth 1)"
[ -z "$leftovers" ] || { echo "the image carries instance files: $leftovers"; exit 1; }

log "== first start on fresh volumes"
docker volume create "${volumes[0]}" > /dev/null
docker volume create "${volumes[1]}" > /dev/null
run_container
wait_healthy
doctor_passes

log "== stock Hermes remembers through ColonyMind across sessions (scripted model on the container's loopback)"
docker exec --detach --user hermes --workdir /tmp "$name" /opt/colonymind/.venv/bin/python -m colonymind.devtools.fake_model 18865
for _ in $(seq 1 30); do docker exec "$name" curl -sf http://127.0.0.1:18865/v1/models > /dev/null && break; sleep 1; done
token="mem-$(python3 -c 'import secrets; print(secrets.token_hex(4))')"
docker exec --workdir /opt/data "$name" timeout 300 hermes chat -q "Please remember this token for me: $token. Just acknowledge." \
  > "$work/session-a.log" 2>&1 || { cat "$work/session-a.log"; exit 1; }
grep -q "$token" "$work/session-a.log" || { echo "session A:"; cat "$work/session-a.log"; exit 1; }
# Formation is asynchronous: ask the sidecar (inside the container, with the key that never leaves it) until the
# token is recallable, then ask Hermes in a fresh session.
docker exec --user hermes --env TOKEN="$token" "$name" sh -c '
  owner="$(/opt/colonymind/.venv/bin/python -c "import yaml; print(yaml.safe_load(open(\"/var/lib/colonymind/colonymind.yaml\"))[\"owner\"][\"contact_id\"])")"
  for i in $(seq 1 90); do
    body="$(curl -s -H "Authorization: Bearer $(cat /var/lib/colonymind/api.key)" -H "content-type: application/json" \
      -X POST http://127.0.0.1:7777/v1/host/context/assemble -d "{\"identity\":{\"host_id\":\"hermes\"},\"context\":{\"session_id\":\"smoke-$i\",\"contact_id\":\"$owner\"},\"incoming_message\":{\"role\":\"user\",\"content\":\"What was the token I asked you to remember earlier?\"}}")"
    case "$body" in *"$TOKEN"*) echo "recallable after $i polls of 2 s"; exit 0;; esac
    sleep 2
  done
  echo "the token never became recallable"; exit 1'
docker exec --workdir /opt/data "$name" timeout 300 hermes chat -q "What was the token I asked you to remember earlier? Reply with it." \
  > "$work/session-b.log" 2>&1 || { cat "$work/session-b.log"; exit 1; }
grep -q "$token" "$work/session-b.log" || { echo "session B did not recall the token:"; cat "$work/session-b.log"; exit 1; }
log "session B answered from memory"

log "== colonymind demo, the keyless mind loop, inside the container"
python3 - "$name" <<'PY'
import re
import subprocess
import sys
import time

BOUND = 180  # the demo's own CI bound (scripts/ci_demo.sh): a chosen bound, not a measured rule
started = time.monotonic()
demo = subprocess.Popen(["docker", "exec", "--interactive", sys.argv[1], "colonymind", "demo"], stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
lines, code = [], None
for line in demo.stdout:
    print(line, end="", flush=True)
    lines.append(line)
    shown = re.match(r"  \| - ([A-Z0-9]{3,8}): ", line)   # an ask the notice names with its code
    if shown:
        code = shown.group(1)
    if line.startswith("Type your reply as the owner"):
        demo.stdin.write(f"yes {code}\n" if code else "yes\n")
        demo.stdin.flush()
rc = demo.wait(timeout=60)
elapsed = time.monotonic() - started
passed = rc == 0 and elapsed <= BOUND and "outcome approved" in "".join(lines)
print(f"demo {'passed' if passed else 'FAILED'} (exit {rc}) in {elapsed:.1f} s; bound {BOUND} s")
sys.exit(0 if passed else 1)
PY

log "== a restart changes nothing"
key_before="$(docker exec "$name" sha256sum /var/lib/colonymind/api.key)"
docker restart --time 30 "$name" > /dev/null
wait_healthy
[ "$(docker exec "$name" sha256sum /var/lib/colonymind/api.key)" = "$key_before" ] || { echo "the restart replaced api.key"; exit 1; }
docker exec --user hermes "$name" /opt/hermes/.venv/bin/python -c '
import sys, yaml
config = yaml.safe_load(open("/opt/data/config.yaml"))
enabled = config["plugins"]["enabled"]
external = config["skills"]["external_dirs"]
sys.exit(0 if enabled.count("colonymind") == 1 and external.count("/var/lib/colonymind/skills") == 1
         and config["memory"]["provider"] == "colonymind-memory" else f"Hermes keys after a restart: {enabled} {external}")'
doctor_passes

log "== another HERMES_UID/HERMES_GID takes the instance over"
docker stop --time 30 "$name" > /dev/null
docker rm "$name" > /dev/null
run_container --env HERMES_UID=4242 --env HERMES_GID=4242
wait_healthy
owners="$(docker exec "$name" sh -c 'find /var/lib/colonymind \( ! -uid 4242 -o ! -gid 4242 \) -print | head -5')"
[ -z "$owners" ] || { echo "not handed over to 4242:4242: $owners"; exit 1; }
[ "$(docker exec "$name" stat -c %a /var/lib/colonymind/api.key)" = 600 ] || { echo "api.key is not mode 600"; exit 1; }
doctor_passes

log "container quickstart passed"
