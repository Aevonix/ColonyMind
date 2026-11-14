#!/usr/bin/env bash
# The keyless demo from the built sidecar wheel: `colonymind demo` in a fresh environment, answered the way a
# person answers the notice it shows, within its bound (180 s: a chosen bound, not a measured rule). Prints
# the runtime.
#
# Expects:
#   DIST_DIR   the built colonymind wheel
set -euo pipefail
: "${DIST_DIR:?set DIST_DIR to the directory holding the built wheels}"
work="$(mktemp -d)"
log() { echo "== $*"; }

log "the sidecar from the built wheel, in its own environment"
python -m venv "$work/venv"
"$work/venv/bin/python" -m pip install --quiet --upgrade pip
"$work/venv/bin/python" -m pip install --quiet "$(ls "$DIST_DIR"/colonymind-*.whl)"

log "colonymind demo (no keys, a scripted model on loopback)"
cd "$work"
timeout 240 "$work/venv/bin/python" - "$work/venv/bin/colonymind" <<'PY'
import re
import subprocess
import sys
import time

BOUND = 180
started = time.monotonic()
demo = subprocess.Popen([sys.argv[1], "demo"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, bufsize=1)
lines, code = [], None
for line in demo.stdout:
    print(line, end="", flush=True)
    lines.append(line)
    shown = re.match(r"  \| - ([A-Z0-9]{3,8}): ", line)   # an ask the notice names with its code
    if shown:
        code = shown.group(1)
    if line.startswith("Type your reply as the owner"):
        # The person's answer, as the notice asks for it: the word, and the code when the notice shows one.
        demo.stdin.write(f"yes {code}\n" if code else "yes\n")
        demo.stdin.flush()
rc = demo.wait(timeout=60)
elapsed = time.monotonic() - started
passed = rc == 0 and elapsed <= BOUND and "outcome approved" in "".join(lines)
print(f"demo {'passed' if passed else 'FAILED'} (exit {rc}) in {elapsed:.1f} s; bound {BOUND} s")
sys.exit(0 if passed else 1)
PY
