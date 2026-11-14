"""OC5 request boundaries use durable child identities, never model text."""
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from colonymind.api.routers import host, mind, people, branches, curated_memory
from colonymind.hands import sessions
from colonymind.hands.store import HandsStore
from test_hands_permits import EPOCH, seed


@pytest.fixture
async def app(tmp_path):
    store = HandsStore(tmp_path / "hands.db")
    await store.connect()
    await seed(store)
    await sessions.register(store, "child-session", "h-000000000001", parent_session="owner-session", epoch=EPOCH)
    app = FastAPI()
    app.state.hands_store = store
    for router in (host.router, host.v2_router, mind.router, people.router, branches.router, curated_memory.router):
        app.include_router(router)
    yield app
    await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path", [
    ("POST", "/v1/host/memory/index"), ("POST", "/v1/host/memory/curated/sources/forget"),
    ("POST", "/v1/host/turns/sync"), ("POST", "/v1/host/contacts"),
    ("POST", "/v1/mind/people/changes"), ("POST", "/v1/mind/people/link"),
    ("POST", "/v1/mind/answers"), ("POST", "/v1/mind/asks/ABC/yes"),
    ("POST", "/v1/mind/grants"), ("POST", "/v1/memory/branches"),
])
async def test_registered_hand_mutations_and_answers_refused_before_body_validation(app, method, path):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.request(method, path, json={}, headers={"X-ColonyMind-Session": "child-session"})
        assert response.status_code == 403, response.text
        assert response.json()["detail"]["code"] == "hand_session_refused"


@pytest.mark.asyncio
async def test_body_and_nested_session_ids_refused_but_owner_and_branch_parent_are_not(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for body in ({"session_id": "child-session"}, {"context": {"session_id": "child-session"}},
                     {"requester": "child-session"}):
            response = await client.post("/v1/host/memory/index", json=body)
            assert response.status_code == 403
        response = await client.post("/v1/host/memory/index", json={"session_id": "owner-session",
            "text": "child-session", "child_session": "child-session"})
        assert response.status_code != 403


@pytest.mark.parametrize("selector", ["session", "session_id", "requester", "X-ColonyMind-Session"])
@pytest.mark.parametrize("values", [("child-session", "owner-session"), ("owner-session", "child-session")])
async def test_all_repeated_selector_values_are_checked(app, selector, values):
    pairs = [(selector, value) for value in values]
    headers, params = (pairs, []) if selector == "X-ColonyMind-Session" else ([], pairs)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/host/memory/index", json={}, headers=headers, params=params)
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "hand_session_refused"


@pytest.mark.asyncio
async def test_registry_survives_restart_rotation_and_retirement(app):
    store = app.state.hands_store
    await sessions.register(store, "child-rotated", "h-000000000001", parent_session="owner-session", epoch=EPOCH)
    await sessions.retire(store, "child-session")
    reopened = HandsStore(store.path)
    await reopened.connect()
    try:
        for name in ("child-session", "child-rotated"):
            assert (await sessions.lookup(reopened, name))["hand_id"] == "h-000000000001"
        assert await sessions.lookup(reopened, "owner-session") is None
        with pytest.raises(ValueError):
            await sessions.register(reopened, "owner-session", "h-000000000001", parent_session="owner-session", epoch=EPOCH)
    finally:
        await reopened.close()
