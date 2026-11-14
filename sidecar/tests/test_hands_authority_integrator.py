"""Integrator-only socket test; run with COLONYMIND_TEST_HANDS_SOCKETS=1."""
from contextlib import asynccontextmanager
import os
import socket
import threading
import time

from fastapi import FastAPI
import httpx
import pytest

from colonymind.api.routers import host, mind, people
from colonymind.hands import sessions
from colonymind.hands.store import HandsStore
from test_hands_permits import EPOCH, seed


@pytest.mark.skipif(os.environ.get("COLONYMIND_TEST_HANDS_SOCKETS") != "1",
                    reason="integrator: local sockets and a normal asyncio wakeup socket are required")
def test_real_socket_refuses_hand_mutations_and_preserves_owner_lane(tmp_path):
    import uvicorn

    @asynccontextmanager
    async def lifespan(app):
        store = HandsStore(tmp_path / "hands.db")
        await store.connect()
        await seed(store)
        await sessions.register(store, "child-session", "h-000000000001", parent_session="owner-session", epoch=EPOCH)
        app.state.hands_store = store
        try:
            yield
        finally:
            await store.close()

    app = FastAPI(lifespan=lifespan)
    for router in (host.router, mind.router, people.router):
        app.include_router(router)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    worker = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    worker.start()
    try:
        deadline = time.monotonic() + 10  # Test assumption: ten seconds for local startup.
        while not server.started and worker.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)  # Test assumption: short startup polling interval.
        assert server.started
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False) as client:
            for route in ("/v1/host/memory/index", "/v1/host/memory/curated/sources/forget",
                          "/v1/mind/people/changes", "/v1/mind/answers"):
                response = client.post(route, json={}, headers={"X-ColonyMind-Session": "child-session"})
                assert response.status_code == 403 and response.json()["detail"]["code"] == "hand_session_refused"
            response = client.post("/v1/host/memory/index", json={}, headers={"X-ColonyMind-Session": "owner-session"})
            assert response.status_code == 422  # Normal body validation still runs for the owner.
    finally:
        server.should_exit = True
        worker.join(timeout=10)  # Test assumption: ten seconds for server shutdown.
        listener.close()
        assert not worker.is_alive()
