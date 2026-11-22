"""``/v1/recorder``: a caller in another process records lines with the instance's API key, and gets a 200 for
good and for bad input."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from onekey import AUTH, KEY
from colonymind import recorder
from colonymind.api.errors import install_exception_handlers
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import recorder as recorder_router

ROUTES = ("begin", "finish", "once", "mark")


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(recorder_router.router)
    app.add_middleware(ApiKeyMiddleware, api_key=KEY)
    return app


def client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://local", headers=AUTH)


async def test_the_four_routes_record_lines(app):
    async with client(app) as http:
        begun = await http.post("/v1/recorder/begin", json={"kind": "event", "source": "whatsapp", "ref": "ingress:17"})
        line = begun.json()["id"]
        finished = await http.post("/v1/recorder/finish", json={
            "id": line, "result": "silent", "choice": "ignore", "chooser": "model", "detail": "group_rule"})
        once = await http.post("/v1/recorder/once", json={
            "kind": "check", "source": "check.mail", "result": "failed", "choice": "tell_owner", "chooser": "rule",
            "detail": "TimeoutError", "took_ms": 840})
        marked = await http.post("/v1/recorder/mark", json={"ref": "ingress:17", "label": "owner_acted"})
    assert [response.status_code for response in (begun, finished, once, marked)] == [200, 200, 200, 200]
    assert finished.json() == {"ok": True} and marked.json() == {"count": 1}
    check, event = recorder.recent()
    assert check["id"] == once.json()["id"]
    assert (check["kind"], check["source"], check["result"], check["detail"], check["took_ms"]) == (
        "check", "check.mail", "failed", "TimeoutError", 840)
    assert (event["id"], event["ref"], event["choice"], event["chooser"], event["result"], event["detail"]) == (
        line, "ingress:17", "ignore", "model", "silent", "group_rule")
    assert event["marks"] == ["owner_acted"]


async def test_an_owner_correction_is_stored_against_the_decision(app):
    recorder.once("event", "sms", result="silent", choice="ignore", chooser="model", ref="ingress:5")
    recorder.once("event", "sms", result="sent", choice="reply", chooser="model", ref="ingress:6")
    async with client(app) as http:
        should = await http.post("/v1/recorder/mark", json={"ref": "ingress:5", "label": "owner_said_should_have"})
        no_need = await http.post("/v1/recorder/mark", json={"ref": "ingress:6", "label": "owner_said_no_need"})
    assert should.json() == no_need.json() == {"count": 1}
    assert {line["ref"]: line["marks"] for line in recorder.recent()} == {
        "ingress:5": ["owner_said_should_have"], "ingress:6": ["owner_said_no_need"]}
    assert {"owner_said_should_have", "owner_said_no_need"} <= set(recorder.MARKS)


async def test_bad_input_answers_200(app):
    bodies = (b"", b"not json", b"[1, 2]", b'"text"', b"{}", json.dumps({
        "id": 7, "kind": ["event"], "source": None, "ref": {"a": 1}, "result": 3.5, "choice": True, "chooser": [],
        "detail": {}, "label": 9, "took_ms": "soon"}).encode())
    async with client(app) as http:
        for route in ROUTES:
            for body in bodies:
                response = await http.post(f"/v1/recorder/{route}", content=body,
                                           headers={"Content-Type": "application/json"})
                assert response.status_code == 200, (route, body)
        unknown = await http.post("/v1/recorder/finish", json={"id": "no-such-line", "result": "sent"})
        strange = await http.post("/v1/recorder/once", json={"kind": "party", "source": "A Person", "result": "fine"})
    assert unknown.json() == {"ok": True}
    line = recorder.recent(1)[0]
    assert line["id"] == strange.json()["id"]
    assert (line["kind"], line["source"], line["result"]) == ("other", "other", "other")


def test_the_engine_serves_the_four_routes(monkeypatch):
    monkeypatch.setenv("COLONYMIND_API_KEY", KEY)
    from colonymind.server import create_app
    assert {f"/v1/recorder/{route}" for route in ROUTES} <= set(create_app().openapi()["paths"])


async def test_another_key_is_refused(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://local") as http:
        response = await http.post("/v1/recorder/once", json={"kind": "event", "source": "sms", "result": "sent"},
                                   headers={"Authorization": "Bearer wrong-key"})
    assert response.status_code == 401 and recorder.recent() == []


CALLER = """
import json, sys, urllib.request
statuses = []
for route, body in json.loads(sys.argv[3]):
    request = urllib.request.Request(sys.argv[1] + route, data=body.encode(), method="POST",
                                     headers={"Authorization": "Bearer " + sys.argv[2],
                                              "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        statuses.append(response.status)
print(json.dumps(statuses))
"""


def test_a_caller_in_another_process_gets_200(app):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.started
        port = server.servers[0].sockets[0].getsockname()[1]
        calls = [("/v1/recorder/once", json.dumps({"kind": "check", "source": "check.mail", "result": "done"})),
                 ("/v1/recorder/begin", json.dumps({"kind": "event", "source": "sms"})),
                 ("/v1/recorder/once", "this is not json"),
                 ("/v1/recorder/finish", json.dumps({"id": 12})),
                 ("/v1/recorder/mark", json.dumps(["ref"]))]
        done = subprocess.run([sys.executable, "-c", CALLER, f"http://127.0.0.1:{port}", KEY, json.dumps(calls)],
                              capture_output=True, text=True, timeout=30)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [200, 200, 200, 200, 200]
    assert [(line["kind"], line["source"]) for line in recorder.recent()] == [
        ("", ""), ("event", "sms"), ("check", "check.mail")]
