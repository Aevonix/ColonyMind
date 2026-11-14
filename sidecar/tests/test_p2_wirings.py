"""Runtime context composition."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import colonymind.api.routers.host as host_mod


@asynccontextmanager
async def _client():
    app = FastAPI()
    app.include_router(host_mod.router)
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://test") as c:
        yield c


async def test_context_assemble_omits_static_architecture_preserves_identity():
    async with _client() as c:
        ctx = {"session_id": "s1", "contact_id": "c1"}
        health = await c.get("/v1/host/health")
        assert health.status_code == 200
        r = await c.post("/v1/host/context/assemble", json={
            "identity": {"host_id": "test"}, "context": ctx,
            "incoming_message": {"role": "user",
                                 "content": "what are your capabilities and architecture?"},
        })
        assert r.status_code == 200
        sections = {s["id"]: s["body"] for s in r.json()["sections"]}
        assert "colonymind-self-knowledge" not in sections
        assert "colonymind-identity" not in sections  # chain ids are /health data, not turn context

        r2 = await c.post("/v1/host/context/assemble", json={
            "identity": {"host_id": "test"}, "context": ctx,
            "incoming_message": {"role": "user",
                                 "content": "What are you working on?"},
        })
        assert r2.status_code == 200
        later = {s["id"]: s["body"] for s in r2.json()["sections"]}
        assert "colonymind-self-knowledge" not in later and "colonymind-identity" not in later
        assert (await c.get("/v1/host/health")).json()["capabilities"] == health.json()["capabilities"]
