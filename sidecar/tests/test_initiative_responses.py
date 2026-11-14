"""Initiative list and detail preserve their fields and expose the latest model response."""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from colonymind.api.routers import host as host_router
from colonymind.api.credentials import CredentialTable, default_credentials
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.initiatives.store import InitiativeStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    value = InitiativeStore(state_dir=tmp_path)
    monkeypatch.setattr(host_router, "_initiative_store", value)
    monkeypatch.setattr(host_router, "_feedback_store", None)
    yield value
    value.close()


@pytest.fixture
def app(store):
    value = FastAPI()
    value.include_router(host_router.router)
    return value


async def test_existing_initiative_fields_are_unchanged(store, app):
    row = store.create(type="task", description="Review the draft", priority=0.7,
                       entity_id="p-02", preferred_agent_id="agent-1", dedup_key="review-draft",
                       context={"topic": "draft"})
    expected = {
        "id": row.id, "initiative_type": "task", "title": "Review the draft", "description": "Review the draft",
        "priority": 70, "status": "pending", "timeout_seconds": row.timeout_seconds,
        "context": {"topic": "draft"}, "context_durability": "durable", "entity_id": "p-02",
        "target_agent_id": "agent-1", "assigned_agent_id": None, "dedup_key": "review-draft",
        "result": None, "error_message": None, "created_at": row.created_at.isoformat(),
        "acknowledged_at": None, "completed_at": None, "failed_at": None, "expires_at": None,
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://host") as client:
        detail = await client.get(f"/v1/host/initiatives/{row.id}")
        listing = await client.get("/v1/host/initiatives")
    assert detail.status_code == listing.status_code == 200
    values = listing.json()
    assert values["total"] == 1
    for value in (detail.json(), values["initiatives"][0]):
        value.pop("last_response", None)
        assert value == expected


async def test_list_and_detail_show_responses_and_read_history_once(store, app):
    rows = [store.create(type="task", description=f"Review draft {index}") for index in range(3)]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://host") as client:
        for row, action in zip(rows, ("acknowledged", "snoozed")):
            response = await client.post(f"/v1/host/initiatives/{row.id}/respond", json={"action": action})
            assert response.status_code == 200
        expected = {
            row.id: {"action": action, "at": store.get_history(row.id)[0].timestamp.isoformat()}
            for row, action in zip(rows, ("acknowledged", "snoozed"))}
        expected[rows[2].id] = None

        queries = []
        store._db.set_trace_callback(queries.append)
        try:
            listing = await client.get("/v1/host/initiatives")
            assert listing.status_code == 200
            assert {item["id"]: item["last_response"] for item in listing.json()["initiatives"]} == expected
            history_queries = [query for query in queries if "assignment_history" in query]
            assert len(history_queries) == 1
            assert "MAX(" in history_queries[0] and "GROUP BY" in history_queries[0]

            for row in rows:
                queries.clear()
                detail = await client.get(f"/v1/host/initiatives/{row.id}")
                assert detail.status_code == 200
                assert detail.json()["last_response"] == expected[row.id]
                # Detail also reads the latest supplied activation origin.
                assert len([query for query in queries if "assignment_history" in query]) == 2

            queries.clear()
            page = await client.get("/v1/host/initiatives", params={"status": "pending", "limit": 1})
            assert page.status_code == 200 and len(page.json()["initiatives"]) == 1
            [query] = [query for query in queries if "assignment_history" in query]
            [item] = page.json()["initiatives"]
            assert item["id"] in query
            assert all(row.id not in query for row in rows if row.id != item["id"])
        finally:
            store._db.set_trace_callback(None)


async def test_list_and_ideas_leave_out_mind_rows_and_skip_a_malformed_row(store, app):
    app.add_middleware(ApiKeyMiddleware, api_key="ideas-test-key", credentials=CredentialTable.from_config(default_credentials()))
    sent = store.create(type="check_in", description="Message sent")
    odd = store.create(type="task", description="Odd row")
    plain = store.create(type="task", description="Plain row")
    store._db.execute("UPDATE initiatives SET kind='message', status='sent' WHERE id=?", [sent.id])
    store._db.execute("UPDATE initiatives SET status='not-a-status' WHERE id=?", [odd.id])
    store._db.commit()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://host",
                           headers={"Authorization": "Bearer ideas-test-key"}) as client:
        listing = await client.get("/v1/host/initiatives", params={"limit": 1000})
        ideas = await client.get("/v1/host/initiatives/ideas", params={"limit": 1000})
    assert listing.status_code == ideas.status_code == 200
    assert {item["id"]: item["status"] for item in listing.json()["initiatives"]} == {plain.id: "pending"}
    assert listing.json()["total"] == 1
    cards = {card["id"] for section in ideas.json()["sections"] for card in section["cards"]}
    assert cards == set()
    assert store.count() == 2
