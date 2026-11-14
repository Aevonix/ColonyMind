"""Slow journal reads must not hold the host's asyncio loop."""
import asyncio
import json
from types import SimpleNamespace
import threading
import time

from fastapi import FastAPI
from fastapi import WebSocketDisconnect
from httpx import ASGITransport, AsyncClient
import pytest

from colonymind.api.routers import host
from colonymind.events import journal


@pytest.fixture
def host_app(monkeypatch):
    for name in ("_telemetry", "_commitment_store", "_embedder", "_embed_failure",
                 "_contacts_store", "_comms_log", "_expectations", "_self_model"):
        monkeypatch.setattr(host, name, None)
    monkeypatch.setattr(host, "_mind", lambda: None)
    app = FastAPI()
    app.include_router(host.router)
    return app


@pytest.mark.parametrize("route", [
    "timeline?since=6h&types=conversation.turn&contact_id=a&limit=4",
    "events/replay?afterSeq=0&limit=4",
    "comms/recent?limit=4",
    "self/expectations?limit=4",
    "self/journal?limit=4",
    "self/journal?today=true",
])
async def test_slow_read_leaves_concurrent_health_under_100ms(host_app, monkeypatch, route):
    loop_thread = threading.get_ident()
    read_threads = []

    def slow(value):
        def read(*args, **kwargs):
            read_threads.append(threading.get_ident())
            time.sleep(0.4)
            return value
        return read

    # Patching both readers also lets this regression test demonstrate the old
    # timeline/replay stall when run against the baseline checkout.
    monkeypatch.setattr(journal, "timeline_events", slow({"events": []}), raising=False)
    monkeypatch.setattr(journal, "replay_events", slow({"events": []}))
    monkeypatch.setattr(host, "_comms_log", SimpleNamespace(
        recent_snapshot=slow({"entries": [], "by_channel": {}}),
        recent=slow([]), rollup=lambda **kw: {}))
    monkeypatch.setattr(host, "_expectations", SimpleNamespace(snapshot=slow({"pending": []})))
    monkeypatch.setattr(host, "_self_model", SimpleNamespace(journal=SimpleNamespace(
        recent=slow([]), today=slow([]))))
    async with AsyncClient(transport=ASGITransport(app=host_app), base_url="http://test") as client:
        assert (await client.get("/v1/host/health")).status_code == 200

        async def concurrent_health():
            started = time.perf_counter()
            await asyncio.sleep(0.01)
            response = await client.get("/v1/host/health")
            return response, time.perf_counter() - started

        health_task = asyncio.create_task(concurrent_health())
        read_task = asyncio.create_task(client.get("/v1/host/" + route))
        try:
            response, elapsed = await health_task
            assert response.status_code == 200
            assert elapsed < 0.1, f"health took {elapsed:.3f}s during {route}"
            assert read_threads and all(t != loop_thread for t in read_threads)
            assert not read_task.done()
        finally:
            result = await read_task
        assert result.status_code == 200


async def test_health_source_probe_runs_offloop(host_app, monkeypatch):
    from colonymind import turns

    original = turns.get_turn_idempotency_ledger
    thread_ids = []

    def slow(*args, **kwargs):
        thread_ids.append(threading.get_ident())
        time.sleep(0.3)
        return original(*args, **kwargs)

    async with AsyncClient(transport=ASGITransport(app=host_app), base_url="http://test") as client:
        assert (await client.get("/v1/host/health")).status_code == 200
        monkeypatch.setattr(turns, "get_turn_idempotency_ledger", slow)
        async def heartbeat():
            started = time.perf_counter()
            await asyncio.sleep(0.01)
            return time.perf_counter() - started

        heartbeat_task = asyncio.create_task(heartbeat())
        health_task = asyncio.create_task(client.get("/v1/host/health"))
        try:
            assert await heartbeat_task < 0.1
        finally:
            response = await health_task
    assert response.status_code == 200
    assert thread_ids and all(t != threading.get_ident() for t in thread_ids)


class _Socket:
    async def accept(self):
        pass

    async def receive_text(self):
        return json.dumps({"type": "auth", "token": "offline-test-key"})

    async def send_json(self, frame):
        if frame["type"] == "replay_complete":
            raise WebSocketDisconnect()


async def test_websocket_cursor_read_is_offloop(host_app, monkeypatch):
    monkeypatch.setenv("COLONYMIND_API_KEY", "offline-test-key")
    threads = []

    def slow_cursor():
        threads.append(threading.get_ident())
        time.sleep(0.3)
        return 0

    monkeypatch.setattr(journal, "current_sequence", slow_cursor)
    monkeypatch.setattr(journal, "replay_events", lambda *args, **kw: {"events": []})

    async def heartbeat():
        started = time.perf_counter()
        await asyncio.sleep(0.01)
        return time.perf_counter() - started

    heartbeat_task = asyncio.create_task(heartbeat())
    stream_task = asyncio.create_task(host.events_ws(_Socket()))
    try:
        assert await heartbeat_task < 0.1
    finally:
        await stream_task
    assert threads and all(t != threading.get_ident() for t in threads)
    assert not host._event_subscribers


async def test_cancelled_websocket_cursor_read_removes_subscriber(host_app, monkeypatch):
    monkeypatch.setenv("COLONYMIND_API_KEY", "offline-test-key")
    started = threading.Event()
    release = threading.Event()

    def cursor():
        started.set()
        release.wait(timeout=2)
        return 0

    monkeypatch.setattr(journal, "current_sequence", cursor)
    stream_task = asyncio.create_task(host.events_ws(_Socket()))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        stream_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stream_task
        assert not host._event_subscribers
    finally:
        release.set()


from colonymind.memory import curated
async def test_comms_worker_connection_preserves_response_and_writer(host_app, tmp_path, monkeypatch):
    from colonymind.contacts.comms import CommsLog
    from colonymind.turns import TurnIdempotencyLedger
    ledger = TurnIdempotencyLedger(tmp_path / 'sources.db')
    log = CommsLog(str(tmp_path / 'comms.db'), source_ledger=ledger)
    log.log('a', channel='sms', direction='in', summary='A conversation')
    log.log('b', channel='email', direction='out', summary='A follow-up')
    expected = {'available': True, 'window_days': 30, 'entries': log.recent(limit=4), 'by_channel': log.rollup(since_days=30)}
    for entry in expected['entries']:
        entry['display_name'] = None
    ledger.record_source('erased-turn', contact_id='a', session_id='session', messages=[{'role': 'user', 'content': 'An erased source'}], derive_claims=False)
    lineage, _ = log.source_input('erased-turn', 'a')
    log.log('a', summary='An erased source', source_lineage=lineage)
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger)
    curated.forget_sources(ledger, 'a', source_ids=['erased-turn'])
    assert historical_bytes(ledger) == _native_before_withdrawal
    monkeypatch.setattr(host, '_comms_log', log)
    async with AsyncClient(transport=ASGITransport(app=host_app), base_url='http://test') as client:
        response = await client.get('/v1/host/comms/recent?limit=4')
    assert response.status_code == 200
    assert response.json() == expected
    log.log('a', summary='Writer still usable')
    assert len(log.recent()) == 3
