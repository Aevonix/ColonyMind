"""The install proofs preserve verifiable hashes for editable snapshots."""
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace

import pytest

from scripts import check_hermes_record as record


@pytest.mark.parametrize("modified", [False, True])
def test_snapshot_still_checks_installed_record(tmp_path, capsys, modified):
    checkout = tmp_path / "workspace"
    checkout.mkdir()
    installed = tmp_path / "site-packages"
    installed.mkdir()
    original = b"editable loader"
    digest = base64.urlsafe_b64encode(hashlib.sha256(original).digest()).decode().rstrip("=")
    (installed / "loader.pth").write_bytes(b"changed" if modified else original)
    files = {"direct_url.json": json.dumps({"url": checkout.as_uri(), "dir_info": {"editable": True}}),
             "RECORD": f"loader.pth,sha256={digest},15\n"}
    dist = SimpleNamespace(version="0.0.0", read_text=files.get, locate_file=lambda _: installed)
    assert record.check_editable_checkout(dist) == int(modified)
    output = capsys.readouterr().out
    if modified:
        assert "modified: loader.pth" in output
    else:
        assert "1 installed files match RECORD" in output
        # A pass that could not look at the checkout ends with a line that says so.
        assert output.splitlines()[-1].startswith("WARN: Source integrity cannot be verified")


def test_proof_scripts_parse():
    root = Path(__file__).resolve().parents[2]
    for name in ("ci_fresh_install.sh", "ci_recall_probe.sh", "ci_mind_loop.sh"):
        subprocess.run(["bash", "-n", str(root / "scripts" / name)], check=True)
    ast.parse((root / "scripts/check_hermes_record.py").read_text())


# A stand-in for `hermes gateway run`: its first start exits 75 as Hermes does when another gateway holds the host
# lock; every later start serves the work directory, whose file `health` answers the helper's health check. It ends
# on SIGINT, as the gateway does (a job started in the background of a script inherits SIGINT ignored).
FAKE_HERMES = """#!/usr/bin/env bash
if [ ! -f "$FAKE_STATE/started" ]; then
  touch "$FAKE_STATE/started"
  echo "Another gateway already owns this host"
  exit "$FAKE_FIRST_EXIT"
fi
exec python3 -c 'import http.server, os, signal
signal.signal(signal.SIGINT, signal.SIG_DFL)
address = ("127.0.0.1", int(os.environ["FAKE_PORT"]))
http.server.ThreadingHTTPServer(address, http.server.SimpleHTTPRequestHandler).serve_forever()'
"""


def gateway_helper(work: Path, port: int) -> Path:
    """The mind loop's gateway.sh, as scripts/ci_mind_loop.sh writes it for this work directory and webhook port."""
    script = (Path(__file__).resolve().parents[2] / "scripts" / "ci_mind_loop.sh").read_text()
    body = script.split('cat > "$work/gateway.sh" <<SH\n', 1)[1].split("\nSH\n", 1)[0]
    rendered = subprocess.run(["bash", "-c", f"cat <<SH\n{body}\nSH\n"], check=True, capture_output=True, text=True,
                              env={"PATH": os.environ["PATH"], "work": str(work),
                                   "WEBHOOK_URL": f"http://127.0.0.1:{port}"}).stdout
    helper = work / "gateway.sh"
    helper.write_text(rendered)
    return helper


@pytest.mark.parametrize("first_exit", [75, 1])
def test_mind_loop_gateway_helper_starts_again_only_after_exit_75(tmp_path, first_exit):
    """Exit 75 is Hermes' lost host-lock race, which its service units retry after 5 s; any other exit fails."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    work = tmp_path / "work"
    (tmp_path / "bin").mkdir()
    work.mkdir()
    (work / "health").write_text("ok")
    fake = tmp_path / "bin" / "hermes"
    fake.write_text(FAKE_HERMES)
    fake.chmod(0o755)
    helper = gateway_helper(work, port)
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}", "FAKE_STATE": str(tmp_path),
           "FAKE_PORT": str(port), "FAKE_FIRST_EXIT": str(first_exit)}
    try:
        result = subprocess.run(["bash", str(helper), "start"], env=env, capture_output=True, text=True, timeout=60)
    finally:
        subprocess.run(["bash", str(helper), "stop"], env=env, timeout=60, check=True)
    if first_exit == 75:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "the gateway exited 75" in result.stdout
    else:
        assert result.returncode == 1
        assert "the gateway exited with 1" in result.stdout
        assert "exited 75" not in result.stdout
