"""Tests for the audit-cleanup pass.

Covers the behaviours introduced by the April 2026 cleanup:
- IMAPProvider missing → email_reply condition returns cleanly.
- ApiKeyMiddleware refuses /v1/host/configure in dev mode.
- Neo4j update_person rejects unknown property names.
"""

from __future__ import annotations


import pytest



# ── A1: missing IMAPProvider is handled gracefully ────────────────────────────


# ── B2: /configure refused without COLONYMIND_API_KEY ─────────────────────────────


@pytest.mark.asyncio
async def test_middleware_refuses_configure_in_dev_mode():
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from colonymind.api.middleware import ApiKeyMiddleware

    app = FastAPI()

    @app.post("/v1/host/configure")
    async def _configure():
        return {"ok": True}

    @app.get("/v1/host/health")
    async def _health():
        return {"ok": True}

    app.add_middleware(ApiKeyMiddleware, api_key=None)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://localhost"
    ) as client:
        health = await client.get("/v1/host/health")
        assert health.status_code == 200

        configure = await client.post("/v1/host/configure", json={})
        assert configure.status_code == 503
        assert "COLONYMIND_API_KEY" in configure.json()["detail"]


@pytest.mark.asyncio
async def test_middleware_accepts_valid_bearer():
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from colonymind.api.middleware import ApiKeyMiddleware

    app = FastAPI()

    @app.post("/v1/host/configure")
    async def _configure():
        return {"ok": True}

    app.add_middleware(ApiKeyMiddleware, api_key="s3cret")

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://localhost"
    ) as client:
        unauthed = await client.post("/v1/host/configure", json={})
        assert unauthed.status_code == 401

        authed = await client.post(
            "/v1/host/configure",
            json={},
            headers={"Authorization": "Bearer s3cret"},
        )
        assert authed.status_code == 200


# ── B4: PII hash is stable and short ──────────────────────────────────────────


# ── Body size middleware ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_body_size_middleware_rejects_oversized_payload():
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from colonymind.api.middleware import BodySizeLimitMiddleware

    app = FastAPI()

    @app.post("/echo")
    async def _echo(body: dict):
        return body

    app.add_middleware(BodySizeLimitMiddleware, max_bytes=64)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://localhost"
    ) as client:
        small = await client.post("/echo", json={"a": 1})
        assert small.status_code == 200

        big = await client.post("/echo", content=b"x" * 128)
        assert big.status_code == 413
        assert "exceeds limit" in big.json()["detail"]


@pytest.mark.asyncio
async def test_body_size_middleware_allows_missing_content_length():
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from colonymind.api.middleware import BodySizeLimitMiddleware

    app = FastAPI()

    @app.get("/ping")
    async def _ping():
        return {"ok": True}

    app.add_middleware(BodySizeLimitMiddleware, max_bytes=64)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://localhost"
    ) as client:
        resp = await client.get("/ping")
        assert resp.status_code == 200
