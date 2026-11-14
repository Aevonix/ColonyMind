"""The job watch's HTTP reads against a real local server, and one whole ``watch --once --dry-run`` pass.

Live on 11-04 no pass completed: ``fetch(..., body=True)`` handed httpx 0.28 a ``bytearray``, which it takes for a
stream, so ``.json()`` raised ``httpx.ResponseNotRead``; ``check_dependencies`` reads ``/v1/mind/status`` that way
on every pass, and ``main()`` does not catch it. Every earlier test stubbed ``fetch``, so none saw it. These tests
use a real HTTP server on 127.0.0.1 and the real ``fetch``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from contextlib import closing, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sqlite3

import pytest
import yaml

from colonymind.jobwatch.network import fetch

KEY = "watch-test-key"
COUNTERS = {"context_reads_failed": 0, "loop_stalls": 0}


@contextmanager
def server():
    """A real HTTP server: health, mind status (JSON, key required), an endpoint, and an oversize body."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, status, body, kind="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            seen.append((self.path, self.headers.get("X-API-Key")))
            if self.path == "/v1/host/health":
                self._send(200, b'{"status": "ok"}')
            elif self.path == "/v1/mind/status":
                if self.headers.get("X-API-Key") != KEY:
                    self._send(401, b'{"detail": "key"}')
                else:
                    self._send(200, json.dumps({"enabled": True, "health_counters": COUNTERS}).encode())
            elif self.path == "/ok":
                self._send(200, b"fine", "text/plain")
            elif self.path == "/big":
                self._send(200, b"x" * (4 * 1024 * 1024 + 1), "text/plain")
            else:
                self._send(404, b"{}")

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", seen
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(5)


def test_fetch_reads_a_json_body_from_a_real_server():
    with server() as (url, seen):
        assert fetch(url + "/v1/mind/status", 5, key=KEY, body=True) == (
            200, {"enabled": True, "health_counters": COUNTERS})
        assert fetch(url + "/v1/host/health", 5, key=KEY) == (200, None)       # no body asked: none parsed
        assert fetch(url + "/v1/mind/status", 5, body=True) == (401, {"detail": "key"})
        with pytest.raises(ValueError, match="response_too_large"):
            fetch(url + "/big", 5, body=True)
    assert seen[0] == ("/v1/mind/status", KEY) and seen[2] == ("/v1/mind/status", None)


def _instance(tmp_path, url):
    """A temp instance whose sidecar is the local server, and a Hermes home with native-shaped job records."""
    home, hermes = tmp_path / "instance", tmp_path / "hermes"
    home.mkdir()
    port = int(url.rsplit(":", 1)[1])
    (home / "colonymind.yaml").write_text(yaml.safe_dump({
        "sidecar": {"host": "127.0.0.1", "port": port},
        "watch": {"hermes_home": str(hermes), "tmux": False, "health_timeout_s": 5,
                  "endpoints": [{"name": "local ok", "url": url + "/ok"}]}}))
    (home / "api.key").write_text(KEY + "\n")
    os.chmod(home / "api.key", 0o600)
    results = hermes / "logs" / "process-results"
    results.mkdir(parents=True)
    (hermes / "processes.json").write_text(json.dumps([]))
    for number in range(3):
        (results / f"proc_{number}.json").write_text(json.dumps({
            "id": f"proc_{number}", "command": "make all", "started_at": 1, "exit_code": 2,
            "completion_reason": "exited", "output": "private\nerror-secret\n"}))
    with closing(sqlite3.connect(hermes / "kanban.db")) as conn:
        conn.execute("CREATE TABLE task_runs (id,task_id,status,worker_pid,started_at,ended_at,max_runtime_seconds)")
        conn.commit()
    return home, hermes


def test_watch_once_dry_run_completes_a_whole_pass(tmp_path):
    with server() as (url, seen):
        home, hermes = _instance(tmp_path, url)
        before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
        for name in ("COLONYMIND_HOME", "COLONYMIND_STATE_DIR", "COLONYMIND_API_KEY"):
            env.pop(name, None)
        result = subprocess.run([sys.executable, "-m", "colonymind", "--instance", str(home), "watch", "--once",
                                 "--dry-run"], capture_output=True, text=True, timeout=120, env=env)
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    findings = json.loads(result.stdout)
    checks = {item["check"] for item in findings}
    assert "repeat" in checks, findings                     # the three failed runs were read from the fixtures
    assert not checks & {"sidecar", "endpoint"}, findings   # health, mind status and the endpoint all answered
    assert ("/v1/host/health", KEY) in seen and ("/v1/mind/status", KEY) in seen and ("/ok", None) in seen
    assert "error-secret" not in result.stdout and "private" not in result.stdout
    after = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before and not (home / "jobwatch.db").exists()                 # a dry run writes nothing
