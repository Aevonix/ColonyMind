"""Regression tests for the 2026-08-10 functional validation sweep.

Every test here reproduces a defect that was observed by RUNNING the system
(not inferred from reading code): the skills invoke path awaiting a
non-awaitable, briefings swallowed into empty 200s (the briefings are gone since), safety-gate context never
populated, memory endpoints whose "backend down" was indistinguishable from
"no data", health advertising a dead memory capability and two unhandled 500s.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from colonymind.api.routers import host


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(host.router)
    return app


def _client(app: FastAPI, **kwargs) -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=app, **kwargs), base_url="http://test",
    )


# ---------------------------------------------------------------------------
# 5. Memory endpoints: backend-down must be distinguishable from "no data"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_reports_canonical_memory(app, monkeypatch):
    async with _client(app) as client:
        response = await client.get('/v1/host/health')
    assert response.status_code == 200, response.text
    assert 'memory' in response.json()['capabilities']
    assert 'Canonical source ledger readable' in response.json()['notes']['memory']


@pytest.mark.asyncio
async def test_health_does_not_claim_unavailable_canonical_memory(app, monkeypatch):
    from colonymind import turns
    def unavailable(*args): raise OSError('fixture unavailable')
    monkeypatch.setattr(turns, 'get_turn_idempotency_ledger', unavailable)
    async with _client(app) as client:
        response = await client.get('/v1/host/health')
    assert response.status_code == 200
    data = response.json()
    assert 'memory' not in data['capabilities']
    assert data['status'] == 'degraded'
    assert 'Canonical source ledger unavailable' in data['notes']['memory']


# ---------------------------------------------------------------------------
# 8. Doctor covers canonical source-store availability
# ---------------------------------------------------------------------------

