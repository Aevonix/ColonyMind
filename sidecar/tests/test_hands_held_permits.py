"""The front reads durable held targets without borrowing execution authority."""

from contextlib import asynccontextmanager
from dataclasses import replace
import json

import httpx
import pytest
from fastapi import FastAPI

from colonymind.api.auth import key_authority
from colonymind.api.routers.hands import runtime_router
from colonymind.hands import approvals, sessions
from colonymind.hands.store import HandsStore


TARGET = {"kind": "file", "target": "/outside/report.txt", "value": "a" * 64}
CALL = {"tool": "terminal", "arguments": {"command": "rm /outside/report.txt"}}


@pytest.fixture
async def store(tmp_path):
    current = HandsStore(tmp_path / "hands.db")
    await current.connect()
    yield current
    await current.close()


async def hold_row(store, number, state, *, parent="owner-session", hand_state="running"):
    hand, task = f"h-{number:012x}", f"t-{number:012x}"
    await store.db.execute("INSERT INTO tasks(id,origin,origin_ref,kind) VALUES (?, 'owner_turn', 'owner', 'coding')",
                           (task,))
    await store.db.execute(
        "INSERT INTO hands(id,task_id,attempt,intent_version,name,kind,effort,parent_session,executor,epoch,state) "
        "VALUES (?,?,1,1,'build','coding',1,?,'harness','{}',?)", (hand, task, parent, hand_state))
    await store.db.execute(
        "INSERT INTO permits(id,ask_id,hand_id,task_id,call,call_digest,workdir,targets,state) "
        "VALUES (?, 'ask', ?, ?, ?, ?, '/workspace', ?, ?)",
        (f"permit-{number}", hand, task, json.dumps(CALL), f"{number:064x}", json.dumps([TARGET]), state))
    await store.db.commit()
    return {"permit_id": f"permit-{number}", "ask_id": "ask", "hand_id": hand,
            "parent_session": parent, "state": state, "call": CALL, "workdir": "/workspace",
            "call_digest": f"{number:064x}", "targets": [TARGET]}


async def test_held_targets_are_parent_scoped_and_do_not_consume_or_refresh(store):
    waiting = await hold_row(store, 1, "waiting")
    granted = await hold_row(store, 2, "granted", hand_state="stopped")
    await hold_row(store, 3, "waiting", parent="another-session")
    for number, state in enumerate(("consumed", "refused", "stale", "lapsed"), 4):
        await hold_row(store, number, state)
    before = [tuple(row) for row in await store.db.execute_fetchall("SELECT * FROM permits ORDER BY id")]
    await store.db.execute("PRAGMA query_only=ON")
    service = approvals.Approvals(store, None)
    for _ in range(2):
        response = await service.held_permits("owner-session")
        assert response.model_dump() == {"permits": [waiting, granted], "refreshed": False}
    assert [tuple(row) for row in await store.db.execute_fetchall("SELECT * FROM permits ORDER BY id")] == before


@pytest.mark.parametrize("state", ["waiting", "granted"])
async def test_held_targets_survive_reopening_without_a_mind_service(store, monkeypatch, state):
    held = await hold_row(store, 1, state)
    await store.close()

    @asynccontextmanager
    async def reopened():
        current = HandsStore(store.path)
        await current.connect()
        try:
            await current.db.execute("PRAGMA query_only=ON")
            yield current
        finally:
            await current.close()

    monkeypatch.setattr(sessions, "open_store", reopened)
    assert (await approvals.held_permits("owner-session")).model_dump() == {"permits": [held], "refreshed": False}
    assert (await approvals.held_permits("another-session")).model_dump() == {"permits": [], "refreshed": False}


async def test_reopened_store_with_no_held_permits_is_empty(store):
    await hold_row(store, 1, "consumed")
    await store.close()
    await store.connect()
    response = await approvals.Approvals(store, None).held_permits("owner-session")
    assert response.model_dump() == {"permits": [], "refreshed": False}


def client_for(authority):
    app = FastAPI()
    app.include_router(runtime_router)

    async def authenticated(scope, receive, send):
        scope.setdefault("state", {})["colonymind_authority"] = authority
        await app(scope, receive, send)

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=authenticated), base_url="http://test")


async def test_held_targets_route_is_an_owner_read_without_an_execution_key(store, monkeypatch):
    held = await hold_row(store, 1, "waiting")
    monkeypatch.setattr(approvals, "held_permits", approvals.Approvals(store, None).held_permits,
                        raising=False)
    async with client_for(key_authority()) as client:
        response = await client.get("/v1/hands/permits/held", params={"session": "owner-session"})
    assert response.status_code == 200, response.text
    assert response.json() == {"permits": [held], "refreshed": False}


@pytest.mark.parametrize("authority", [None, replace(key_authority(), owner_platforms=frozenset()),
                                      replace(key_authority(), principal_id="credential:guest"),
                                      replace(key_authority(), scopes=frozenset({"hands:read"}))])
async def test_held_targets_route_requires_the_owner_credential(authority):
    async with client_for(authority) as client:
        response = await client.get("/v1/hands/permits/held", params={"session": "owner-session"})
    assert response.status_code == 403, response.text


@pytest.mark.parametrize("params", [{}, {"session": ""}, {"session": "  "}])
async def test_held_targets_route_requires_a_parent_session(params):
    async with client_for(key_authority()) as client:
        response = await client.get("/v1/hands/permits/held", params=params)
    assert response.status_code == 422, response.text
