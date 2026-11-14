"""The optional idea catalog extends the initiative lifecycle and preserves old clients."""

import json
import sqlite3
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from colonymind.api.credentials import CredentialTable, default_credentials
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import host
from colonymind.feedback import TypeFeedbackStore
from colonymind.initiatives.models import InitiativeStatus, MIND_STATUSES
from colonymind.initiatives.store import InitiativeStore

pytestmark = pytest.mark.asyncio
ROOT = "/v1/host/initiatives"


def marker():
    return f"marker-{uuid4()}"


@pytest.fixture
def store(tmp_path, monkeypatch):
    value = InitiativeStore(tmp_path)
    feedback = TypeFeedbackStore()
    monkeypatch.setattr(host, "_initiative_store", value)
    monkeypatch.setattr(host, "_feedback_store", feedback)
    monkeypatch.delenv("COLONYMIND_SHARE_BASE_URL", raising=False)
    yield value
    feedback._conn.close()
    value.close()


@pytest.fixture
async def client(store):
    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware, api_key="ideas-test-key", credentials=CredentialTable.from_config(default_credentials()))
    app.include_router(host.router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test",
                           headers={"Authorization": "Bearer ideas-test-key"}) as value:
        yield value


def create(store, **kwargs):
    return store.create(type=kwargs.pop("type", "TASK_SUGGESTION"), description=marker(), **kwargs)


async def test_legacy_catalog_rows_are_excluded_without_lifecycle_mutation(store, client):
    rows = [create(store, priority=priority) for priority in (0.2, 0.9, 0.9)]
    rows.append(create(store, type="RESEARCH_DEEP_DIVE"))
    for contract in (None, 2):
        page = (await client.get(ROOT + "/ideas", params={"contract": contract} if contract else {})).json()
        assert page["sections"] == []
    assert all(store.get(row.id).status == "pending" for row in rows)
    assert all(store.get_history(row.id) == [] for row in rows)


async def test_empty_catalog_and_invalid_cursors(store, client):
    assert (await client.get(ROOT + "/ideas")).json() == {
        "sections": [], "viewer_state": {"has_built_idea": False},
        "pagination": {"next_cursor": None, "has_next_page": False}}
    for cursor in ("", "!", "bnVsbA==", "e30=", "WzFd", "x" * 4097):
        assert (await client.get(ROOT + "/ideas", params={"cursor": cursor})).status_code == 422
    for limit in (0, 1001):
        assert (await client.get(ROOT + "/ideas", params={"limit": limit})).status_code == 422


STATE_CASES = {
    "pending": "available", "assigned": "executing", "acknowledged": "available",
    "in_progress": "executing", "completed": "built", "failed": "failed", "cancelled": "dismissed",
    "proposed": "available", "asked": "available", "approved": "executing", "dispatched": "executing",
    "sending": "executing", "done": "built", "sent": "built", "expired": "failed",
    "uncertain": "failed", "dropped": "dismissed",
}


@pytest.mark.parametrize("status,state", STATE_CASES.items())
async def test_feed_state_mapping_for_every_status(store, client, status, state):
    assert {value.value for value in InitiativeStatus} | MIND_STATUSES | {"in_progress"} == set(STATE_CASES)
    row = create(store)
    store.update(row.id, status=status)
    response = await client.get(ROOT + "/ideas/state", params={"ids": row.id})
    assert response.status_code == 200
    assert response.json() == {"states": [{"id": row.id, "exists": False, "state": None, "status": None}]}


async def test_feed_state_uses_response_history_and_current_status(store, client):
    row = create(store)
    for body, state, status in (
        ({"action": "acknowledged"}, "available", "acknowledged"),
        ({"mode": "full"}, "executing", "acknowledged"),
        ({"action": "acknowledged"}, "executing", "acknowledged"),
        ({"action": "snoozed"}, "available", "pending"),
        ({"action": "approved"}, "executing", "acknowledged"),
        ({"action": "actioned"}, "built", "completed"),
        ({"action": "acknowledged"}, "built", "acknowledged"),
        ({"action": "dismissed"}, "dismissed", "cancelled"),
    ):
        assert (await client.post(f"{ROOT}/{row.id}/respond", json=body)).status_code == 200
        assert (await client.get(ROOT + "/ideas/state", params={"ids": row.id})).json() == {
            "states": [{"id": row.id, "exists": False, "state": None, "status": None}]}
    store.update(row.id, status="failed")
    assert store.idea_states([row.id])[0]["state"] is None
    store.update(row.id, status="pending")
    assert store.idea_states([row.id])[0]["state"] is None


@pytest.mark.parametrize("native_ack", [False, True])
async def test_feed_state_uses_native_agent_history(store, client, native_ack):
    row = create(store)
    store.update(row.id, status="assigned", assigned_agent_id="agent-1")
    if native_ack:
        store.update(row.id, status="acknowledged")
    else:
        # A response acknowledgment must also preserve native assignment evidence.
        response = await client.post(f"{ROOT}/{row.id}/respond", json={"action": "acknowledged"})
        assert response.status_code == 200
    response = await client.get(ROOT + "/ideas/state", params={"ids": row.id})
    assert response.status_code == 200
    assert response.json() == {"states": [
        {"id": row.id, "exists": False, "state": None, "status": None}]}


async def test_feed_state_cap_unknown_ids_order_and_duplicates(store, client):
    row = create(store)
    ids = [marker() for _ in range(99)] + [row.id]
    response = await client.get(ROOT + "/ideas/state", params={"ids": ",".join(ids)})
    assert response.status_code == 200
    states = response.json()["states"]
    assert states[:-1] == [{"id": value, "exists": False, "state": None, "status": None} for value in ids[:-1]]
    assert states[-1] == {"id": row.id, "exists": False, "state": None, "status": None}
    duplicate = await client.get(ROOT + "/ideas/state", params={"ids": f" {row.id}, {row.id} "})
    assert duplicate.json()["states"] == [states[-1], states[-1]]
    for values in (ids + [marker()], [row.id] * 101, [""], [row.id, ""]):
        assert (await client.get(ROOT + "/ideas/state", params={"ids": ",".join(values)})).status_code == 422
    assert (await client.get(ROOT + "/ideas/state")).status_code == 422
    assert store.get(row.id).status == "pending" and store.get_history(row.id) == []


@pytest.mark.parametrize("surface", ["feed", "ideas", "chat"])
@pytest.mark.parametrize("with_unit", [True, False])
async def test_origin_round_trip_and_legacy_followup_preserves_association(store, client, surface, with_unit):
    row = create(store)
    origin = {"surface": surface, **({"unit_id": marker()} if with_unit else {})}
    details = {"marker": marker(), "origin": {"surface": "chat", "unit_id": marker()}}
    response = await client.post(f"{ROOT}/{row.id}/respond", json={"mode": "full", "origin": origin, "details": details})
    assert response.json() == {"success": True, "initiative_id": row.id, "status": "acknowledged"}
    history = store.get_history(row.id)
    assert len(history) == 1 and history[0].action == "llm_approved" and history[0].origin == origin
    assert history[0].details == {**details, "mode": "full", "item_ids": None}
    assert (await client.get(f"{ROOT}/{row.id}")).json()["origin"] == origin
    assert "origin" not in (await client.get(ROOT)).json()["initiatives"][0]
    assert (await client.post(f"{ROOT}/{row.id}/respond", json={"action": "actioned"})).status_code == 200
    assert (await client.get(f"{ROOT}/{row.id}")).json()["origin"] == origin
    # The latest supplied origin is shown on detail; every association remains in history.
    next_origin = {"surface": "chat", "unit_id": marker()}
    assert (await client.post(f"{ROOT}/{row.id}/respond", json={"action": "acknowledged", "origin": next_origin})).status_code == 200
    assert (await client.get(f"{ROOT}/{row.id}")).json()["origin"] == next_origin
    assert origin in [entry.origin for entry in store.get_history(row.id)]


@pytest.mark.parametrize("origin", [{}, {"surface": "invalid"}, {"surface": "feed", "unit_id": ""},
                                   {"surface": "feed", "unit_id": 42}, {"surface": "feed", "extra": True}, []])
async def test_invalid_origin_is_rejected_before_mutation(store, client, origin):
    row = create(store)
    assert (await client.post(f"{ROOT}/{row.id}/respond", json={"mode": "full", "origin": origin})).status_code == 422
    assert store.get(row.id).status == "pending" and store.get_history(row.id) == []


async def test_create_metadata_and_stable_included_items(store, client):
    presentation = {"labels": [marker()], "badges": [marker()], "action_label": marker(),
                    "icon": {"media_id": marker(), "mime_type": "image/webp"},
                    "preview": {"url": "https://example.invalid/" + marker(), "mime_type": "image/png"},
                    "fit_reason": marker(), "build_summary": marker(), "detail_description": marker()}
    items = [{"type": marker(), "title": marker(), "summary": marker(), "build_summary": marker(),
              "selectable": True, "is_selected": False},
             {"id": marker(), "type": marker(), "title": marker(), "selectable": False}]
    body = {"initiative_type": "TASK_SUGGESTION", "title": marker(), "description": marker(),
            "presentation": presentation, "items": items, "dedup_key": marker()}
    response = await client.post(ROOT, json=body)
    assert response.status_code == 200
    initiative_id = response.json()["id"]
    first = (await client.get(f"{ROOT}/{initiative_id}")).json()
    assert first["presentation"] == presentation
    assert first["items"][0]["id"] and first["items"][0]["is_selected"] is False
    assert first["items"][1]["id"] == items[1]["id"] and not first["items"][1]["selectable"]
    assert (await client.post(ROOT, json=body)).json()["id"] == initiative_id
    assert (await client.get(f"{ROOT}/{initiative_id}")).json() == first
    assert store.get(initiative_id).presentation == presentation
    assert (await client.get(ROOT + "/ideas")).json()["sections"] == []
    assert "presentation" not in (await client.get(ROOT)).json()["initiatives"][0]


async def test_create_refuses_duplicate_item_ids_and_missing_media_reference(store, client):
    item = {"id": marker(), "type": marker(), "title": marker()}
    body = {"initiative_type": "TASK_SUGGESTION", "title": marker(), "description": marker()}
    assert (await client.post(ROOT, json={**body, "items": [item, item]})).status_code == 422
    assert (await client.post(ROOT, json={**body, "presentation": {"icon": {"mime_type": "image/webp"}}})).status_code == 422
    assert store.count() == 0


async def test_legacy_list_and_get_bytes_and_respond_body(store, client):
    row = create(store, context={"topic": marker()})
    expected = {"id": row.id, "initiative_type": row.type, "title": row.description,
                "description": row.description, "priority": 50, "status": "pending", "timeout_seconds": 300,
                "context": row.context, "context_durability": "durable", "entity_id": None,
                "target_agent_id": None, "assigned_agent_id": None, "dedup_key": None, "result": None,
                "error_message": None, "created_at": row.created_at.isoformat(), "acknowledged_at": None,
                "completed_at": None, "failed_at": None, "expires_at": None, "last_response": None}
    encode = lambda value: json.dumps(value, separators=(",", ":")).encode()
    assert (await client.get(f"{ROOT}/{row.id}")).content == encode(expected)
    assert (await client.get(ROOT)).content == encode({"initiatives": [expected], "total": 1})
    for action, status in (("approved", "acknowledged"), ("actioned", "completed"),
                           ("dismissed", "cancelled"), ("snoozed", "pending"),
                           ("acknowledged", "acknowledged"), (marker(), "acknowledged")):
        details = {"marker": marker()}
        response = await client.post(f"{ROOT}/{row.id}/respond", json={"action": action, "details": details})
        assert response.content == encode({"success": True, "initiative_id": row.id, "status": status})
        history = store._db.execute("SELECT action, details FROM assignment_history ORDER BY id DESC LIMIT 1").fetchone()
        assert history["action"] == f"llm_{action}" and json.loads(history["details"]) == details
        assert "origin" not in (await client.get(f"{ROOT}/{row.id}")).json()
        assert all(entry.origin is None for entry in store.get_history(row.id))


async def test_execute_modes_record_selection_with_existing_acknowledgement(store, client):
    item = {"id": marker(), "type": marker(), "title": marker(), "selectable": True}
    row = create(store, items=[item])
    for body in ({"mode": "selected_items", "item_ids": [item["id"]]}, {"mode": "full"},
                 {"action": "approved", "mode": "full", "details": {"marker": marker()}}):
        response = await client.post(f"{ROOT}/{row.id}/respond", json=body)
        assert response.json() == {"success": True, "initiative_id": row.id, "status": "acknowledged"}
        assert store.get(row.id).job_id is None
        details = json.loads(store._db.execute("SELECT details FROM assignment_history ORDER BY id DESC LIMIT 1").fetchone()[0])
        assert details == {**body.get("details", {}), "mode": body["mode"], "item_ids": body.get("item_ids")}
    assert host._feedback_store.multiplier(f"{row.type}:{row.drive}") == 1.0
    assert not (await client.get(ROOT + "/ideas")).json()["viewer_state"]["has_built_idea"]


async def test_invalid_selections_do_not_mutate_status_or_history(store, client):
    selected, fixed = marker(), marker()
    row = create(store, items=[{"id": selected, "type": marker(), "title": marker(), "selectable": True},
                               {"id": fixed, "type": marker(), "title": marker(), "selectable": False}])
    foreign = create(store, items=[{"id": marker(), "type": marker(), "title": marker()}]).items[0]["id"]
    bodies = [{}, {"mode": "invalid"}, {"mode": "selected_items"}] + [
        {"mode": "selected_items", "item_ids": ids} for ids in ([], [fixed], [foreign], [marker()], [selected, selected])
    ] + [{"mode": "full", "item_ids": [selected]}, {"action": "approved", "item_ids": [selected]},
         {"action": "dismissed", "mode": "full"}]
    for body in bodies:
        assert (await client.post(f"{ROOT}/{row.id}/respond", json=body)).status_code == 422
    assert store.get(row.id).status == "pending" and store.get_history(row.id) == []


async def test_feedback_is_preference_only_up_and_existing_dismiss_down(store, client):
    row = create(store)
    for _ in range(2):
        response = await client.post(f"{ROOT}/{row.id}/feedback", json={"direction": "up", "surface": marker()})
        assert response.json()["status"] == "pending"
    assert store.get(row.id).status == "pending" and store.get(row.id).job_id is None
    assert host._feedback_store.multiplier(f"{row.type}:{row.drive}") == pytest.approx(1.12)
    assert host._feedback_store.snapshot()[0]["actioned"] == 0
    body = {"direction": "down", "surface": marker(), "reason": marker()}
    assert (await client.post(f"{ROOT}/{row.id}/feedback", json=body)).json()["status"] == "cancelled"
    assert host._feedback_store.multiplier(f"{row.type}:{row.drive}") == pytest.approx(0.85)
    assert store.get_history(row.id)[0].action in {"feedback_up", "llm_dismissed"}
    history = store._db.execute("SELECT details FROM assignment_history ORDER BY id DESC LIMIT 1").fetchone()
    assert json.loads(history[0]) == body
    assert (await client.get(ROOT + "/ideas")).json()["sections"] == []


@pytest.mark.parametrize("direction", ["up", "down"])
async def test_reason_limit_counts_unicode_codepoints_and_never_truncates(store, client, direction):
    row = create(store)
    prefix = marker()
    reason = prefix + "\U0001f680" * (600 - len(prefix))
    body = {"direction": direction, "surface": marker(), "reason": reason}
    assert (await client.post(f"{ROOT}/{row.id}/feedback", json={**body, "reason": reason + "e"})).status_code == 422
    assert store.get(row.id).status == "pending" and not store.get_history(row.id)
    assert (await client.post(f"{ROOT}/{row.id}/feedback", json=body)).status_code == 200
    assert store.get_history(row.id)[0].details["reason"] == reason


async def test_engagement_append_only_idempotent_and_conflicting_reuse_refused(store, client):
    row, other = create(store), create(store)
    body = {"receipt_id": marker(), "kind": "impression", "visible_fraction": 0.9,
            "surface": marker(), "anchor": marker()}
    endpoint = f"{ROOT}/{row.id}/engagement"
    assert (await client.post(endpoint, json=body)).json() == {"receipt_id": body["receipt_id"], "recorded": True}
    before = tuple(store._db.execute("SELECT * FROM initiative_engagement").fetchone())
    assert (await client.post(endpoint, json=body)).json()["recorded"] is False
    for changes in ({"kind": "open"}, {"visible_fraction": 0.1}, {"surface": marker()}, {"anchor": marker()}):
        assert (await client.post(endpoint, json={**body, **changes})).status_code == 409
    assert (await client.post(f"{ROOT}/{other.id}/engagement", json=body)).status_code == 409
    assert tuple(store._db.execute("SELECT * FROM initiative_engagement").fetchone()) == before
    assert (await client.post(endpoint, json={**body, "receipt_id": marker(), "kind": "open"})).json()["recorded"]
    assert store._db.execute("SELECT count(*) FROM initiative_engagement").fetchone()[0] == 2
    assert store.get(row.id).status == "pending" and store.get_history(row.id) == []
    for changes in ({"kind": "other"}, {"visible_fraction": -0.1}, {"visible_fraction": 1.1}, {"receipt_id": ""}):
        assert (await client.post(endpoint, json={**body, **changes})).status_code == 422


async def test_share_stable_snapshot_and_configured_url_only(store, client, monkeypatch):
    private = marker()
    row = create(store, context={"private": private}, presentation={"detail_description": private},
                 items=[{"title": private, "type": marker()}])
    first = (await client.post(f"{ROOT}/{row.id}/share", headers={"Host": "attacker.invalid"})).json()
    assert first == {"share_id": first["share_id"], "share_url": None,
                     "content": {"title": row.description[:100], "summary": row.description}}
    assert private not in json.dumps(first)
    store.update(row.id, description=marker())
    assert (await client.post(f"{ROOT}/{row.id}/share")).json() == first
    monkeypatch.setenv("COLONYMIND_SHARE_BASE_URL", "https://example.invalid/" + marker())
    resolved = (await client.get(f"{ROOT}/shares/{first['share_id']}")).json()
    assert resolved["share_url"].startswith("https://example.invalid/")
    assert resolved["share_url"].endswith(f"{ROOT}/shares/{first['share_id']}")
    assert resolved["content"] == first["content"]
    for invalid in ("/relative", "javascript:alert(1)", "https://example.invalid/?query=1", "http://["):
        monkeypatch.setenv("COLONYMIND_SHARE_BASE_URL", invalid)
        assert (await client.post(f"{ROOT}/{row.id}/share")).json()["share_url"] is None
    assert (await client.get(f"{ROOT}/shares/{marker()}")).status_code == 404


async def test_new_routes_use_existing_host_auth_and_share_is_not_public(store):
    row = create(store)
    share = store.share(row.id, row.description, row.description)
    app = FastAPI()
    app.include_router(host.router)
    secret = marker()
    app.add_middleware(ApiKeyMiddleware, api_key=secret,
                       credentials=CredentialTable.from_config(default_credentials()))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for path in (ROOT, ROOT + "/ideas", ROOT + f"/ideas/state?ids={row.id}",
                     f"{ROOT}/{row.id}", f"{ROOT}/shares/{share['share_id']}"):
            assert (await client.get(path)).status_code == 401
            assert (await client.get(path, headers={"Authorization": f"Bearer {secret}"})).status_code == 200
        for suffix in ("respond", "feedback", "engagement", "share"):
            assert (await client.post(f"{ROOT}/{row.id}/{suffix}", json={})).status_code == 401


async def test_unknown_initiatives_are_hidden_by_new_routes(store, client):
    for initiative_id in (marker(), marker()):
        for suffix, body in (("share", {}), ("feedback", {"direction": "up", "surface": marker()}),
                             ("engagement", {"receipt_id": marker(), "kind": "open", "visible_fraction": 1,
                                             "surface": marker(), "anchor": marker()}),
                             ("respond", {"mode": "full"})):
            assert (await client.post(f"{ROOT}/{initiative_id}/{suffix}", json=body)).status_code == 404
    assert (await client.get(ROOT + "/ideas")).json()["sections"] == []


async def test_migration_of_old_store_and_reopening_preserves_new_content(tmp_path, monkeypatch):
    # Pre-context SQLite schema, independent of the current initialization code.
    old_id, description = marker(), marker()
    with sqlite3.connect(tmp_path / "initiatives.db") as db:
        db.execute("""CREATE TABLE initiatives (
            id TEXT PRIMARY KEY, dedup_key TEXT UNIQUE, type TEXT NOT NULL, description TEXT NOT NULL,
            priority REAL DEFAULT 0.5, rationale TEXT, action_hint TEXT, entity_id TEXT,
            source_type TEXT, source_id TEXT, created_by TEXT, status TEXT DEFAULT 'pending',
            assigned_agent_id TEXT, assigned_agent_name TEXT, assigned_at TEXT, acknowledged_at TEXT,
            completed_at TEXT, cancelled_at TEXT, cancelled_by TEXT, cancelled_reason TEXT,
            failed_at TEXT, failed_reason TEXT, attempt_count INTEGER DEFAULT 0, max_attempts INTEGER DEFAULT 3,
            timeout_seconds INTEGER DEFAULT 300, last_attempt_at TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            expires_at TEXT, delivery_mode TEXT DEFAULT 'websocket', delivery_attempts INTEGER DEFAULT 0,
            last_delivery_at TEXT, delivery_failed_at TEXT, delivery_failed_reason TEXT,
            result TEXT, result_metadata TEXT DEFAULT '{}', preferred_agent_id TEXT, stale_reason TEXT, recovery_reason TEXT
        )""")
        db.execute("INSERT INTO initiatives(id,type,description) VALUES (?,?,?)", [old_id, "TASK_SUGGESTION", description])
        db.execute("""CREATE TABLE assignment_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, initiative_id TEXT NOT NULL, agent_id TEXT NOT NULL,
            agent_name TEXT, action TEXT NOT NULL, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, details TEXT,
            FOREIGN KEY (initiative_id) REFERENCES initiatives(id)
        )""")
        db.execute("INSERT INTO assignment_history(initiative_id,agent_id,action,details) VALUES (?,?,?,?)",
                   [old_id, marker(), "llm_acknowledged", json.dumps({"marker": description})])
    store = InitiativeStore(tmp_path)
    try:
        old = store.get(old_id)
        assert old.description == description and old.presentation == {} and old.items == []
        assert store.get_history(old_id)[0].origin is None
        assert store.get_history(old_id)[0].details == {"marker": description}
        monkeypatch.setattr(host, "_initiative_store", store)
        monkeypatch.setattr(host, "_feedback_store", None)
        app = FastAPI()
        app.add_middleware(ApiKeyMiddleware, api_key="migration-test-key", credentials=CredentialTable.from_config(default_credentials()))
        app.include_router(host.router)
        origin = {"surface": "feed", "unit_id": marker()}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test",
                               headers={"Authorization": "Bearer migration-test-key"}) as client:
            assert "origin" not in (await client.get(f"{ROOT}/{old_id}")).json()
            assert (await client.get(ROOT + "/ideas/state", params={"ids": old_id})).json()["states"][0]["state"] is None
            assert (await client.post(f"{ROOT}/{old_id}/respond", json={"action": "approved", "origin": origin})).status_code == 200
            assert (await client.get(f"{ROOT}/{old_id}")).json()["origin"] == origin
        row = create(store, presentation={"fit_reason": marker()}, items=[{"title": marker(), "type": marker()}])
        receipt = {"receipt_id": marker(), "kind": "open", "visible_fraction": 1, "surface": marker(), "anchor": marker()}
        store.record_engagement(row.id, receipt)
        share = store.share(row.id, row.description, row.description)
    finally:
        store.close()
    reopened = InitiativeStore(tmp_path)
    try:
        assert reopened.get(old_id).description == description
        assert reopened.last_response_origin(old_id) == origin
        assert reopened.idea_states([old_id])[0]["state"] is None
        assert len(reopened.get_history(old_id)) == 2
        assert origin in [entry.origin for entry in reopened.get_history(old_id)]
        assert reopened.get(row.id).items == row.items
        assert reopened.get(row.id).presentation == row.presentation
        assert not reopened.record_engagement(row.id, receipt)
        assert reopened.share(row.id, marker(), marker()) == share
    finally:
        reopened.close()


async def test_contract_2_legacy_exclusion_on_both_routes(store, client):
    rows = [store.create(type="research", description=f"Legacy idea {index}") for index in range(3)]
    first = (await client.get(ROOT, params={"contract": 2, "limit": 2})).json()
    assert first == (await client.get(f"{ROOT}/ideas", params={"contract": 2, "limit": 2})).json()
    assert first["sections"] == []
    assert first["viewerState"] == {"hasBuiltIdea": False}
    assert first["pagination"] == {"hasNextPage": False, "nextCursor": None}
    assert len((await client.get(ROOT)).json()["initiatives"]) == len(rows)


async def test_large_legacy_inventory_stays_excluded_under_relay_cap(store, client):
    rows = [store.create(type="research", description="x" * 4000) for _ in range(839)]
    response = await client.get(ROOT, params={"contract": 2, "limit": 1000})
    assert response.status_code == 200 and len(response.content) < 1_048_576
    assert response.json()["sections"] == []
    assert all(store.get(r.id).status == "pending" and store.get_history(r.id) == [] for r in rows)


async def test_contract_2_actions_answer_with_the_receipts_the_client_validates(store, client):
    row = store.create(type="research", description="Idea", items=[{"type": "step", "title": "A"}, {"type": "step", "title": "B"}])
    first, second = (item["id"] for item in store.get(row.id).items)
    context = {"target": "IDEA", "surface": "idea_cards", "lane": "explore", "anchor": "research"}
    for body in ({"action": "approved", "mode": "full"},
                 {"action": "approved", "mode": "selectedItems", "itemIds": [first]}):
        assert (await client.post(f"{ROOT}/{row.id}/respond", json=body)).status_code == 404
    assert (await client.post(f"{ROOT}/{row.id}/respond", json={"mode": "selectedItems", "itemIds": [second]})).status_code == 422
    up = await client.post(f"{ROOT}/{row.id}/feedback", json={"signal": "UP", "reason": "nice", **context})
    assert up.status_code == 200 and up.json()["ok"] is True
    long = await client.post(f"{ROOT}/{row.id}/feedback", json={"signal": "UP", "reason": "a" * 601, **context})
    ups = [entry.details for entry in store.get_history(row.id) if entry.action == "feedback_up"]
    assert long.status_code == 200 and len(ups) == 2 and sorted(details["reason"] or "" for details in ups) == ["", "nice"]
    seen = await client.post(f"{ROOT}/{row.id}/engagement",
                             json={"event": "visible", "visibleFraction": 0.95, **context})
    assert seen.status_code == 200 and seen.json()["ok"] is True and seen.json()["recorded"] is True
    blank = {**context, "anchor": ""}
    assert (await client.post(f"{ROOT}/{row.id}/engagement",
                              json={"event": "open", "visibleFraction": 1.0, **blank})).status_code == 200
    assert (await client.post(f"{ROOT}/{row.id}/feedback", json={"signal": "UP", **blank})).status_code == 200
    for bad in ({"signal": "UP", **context, "lane": "other"}, {"signal": "SIDEWAYS", **context},
                {"signal": "UP", **context, "anchor": "a" * 129}, {"signal": "UP", **context, "extra": 1}):
        assert (await client.post(f"{ROOT}/{row.id}/feedback", json=bad)).status_code == 422
    assert (await client.post(f"{ROOT}/{row.id}/engagement",
                              json={"event": "hover", "visibleFraction": 1, **context})).status_code == 422
    down = await client.post(f"{ROOT}/{row.id}/feedback", json={"signal": "DOWN", **context})
    assert down.status_code == 200 and down.json()["ok"] is True and store.get(row.id).status == "cancelled"


def ios_engagement_body():
    return {"target": "IDEA", "surface": "idea_cards", "lane": "explore", "anchor": "research",
            "event": "visible", "visibleFraction": 0.95, "value": 0.95, "anchor_kind": "section",
            "metadata": {"source": "ideas", "card_type": "research"}}


async def test_contract_2_engagement_accepts_ios_context(store, client):
    row = create(store)
    response = await client.post(f"{ROOT}/{row.id}/engagement", json=ios_engagement_body())
    assert response.status_code == 200
    assert response.json()["ok"] is True and response.json()["recorded"] is True
    receipts = store._db.execute("SELECT * FROM initiative_engagement").fetchall()
    assert len(receipts) == 1
    receipt = dict(receipts[0])
    assert receipt["receipt_id"] == response.json()["receipt_id"]
    assert receipt["kind"] == "impression" and receipt["visible_fraction"] == 0.95
    assert receipt["surface"] == "idea_cards" and receipt["anchor"] == "research"
    assert not {"value", "anchor_kind", "metadata"}.intersection(receipt)


async def test_contract_2_engagement_rejects_excess_metadata_keys(store, client):
    row = create(store)
    body = {**ios_engagement_body(), "metadata": {str(n): n for n in range(17)}}
    response = await client.post(f"{ROOT}/{row.id}/engagement", json=body)
    assert response.status_code == 422
    assert response.json()["detail"] == "invalid idea engagement"
    assert store._db.execute("SELECT count(*) FROM initiative_engagement").fetchone()[0] == 0


async def test_contract_2_engagement_rejects_string_value(store, client):
    row = create(store)
    response = await client.post(f"{ROOT}/{row.id}/engagement", json={**ios_engagement_body(), "value": "x"})
    assert response.status_code == 422
    assert response.json()["detail"] == "invalid idea engagement"
    assert store._db.execute("SELECT count(*) FROM initiative_engagement").fetchone()[0] == 0


async def test_contract_2_engagement_rejects_unknown_key(store, client):
    row = create(store)
    response = await client.post(f"{ROOT}/{row.id}/engagement", json={**ios_engagement_body(), "unknown": 1})
    assert response.status_code == 422
    assert response.json()["detail"] == "invalid idea card context"
    assert store._db.execute("SELECT count(*) FROM initiative_engagement").fetchone()[0] == 0


@pytest.mark.parametrize("context,accepted", [
    ({"value": 0}, True), ({"value": -1.5}, True), ({"value": True}, False), ({"value": None}, False),
    ({"value": float("inf")}, False), ({"value": float("-inf")}, False), ({"value": float("nan")}, False),
    ({"anchor_kind": "a" * 80}, True), ({"anchor_kind": "a" * 81}, False), ({"anchor_kind": None}, False),
    ({"metadata": {}}, True), ({"metadata": {str(n): n for n in range(16)}}, True),
    ({"metadata": {"a" * 80: "x" * 500}}, True), ({"metadata": {"a" * 81: 0}}, False),
    ({"metadata": {"a": "x" * 501}}, False),
    ({"metadata": {"a": True, "b": 1, "c": 1.5, "d": None}}, True),
    ({"metadata": {"a": []}}, False), ({"metadata": {"a": {}}}, False),
    ({"metadata": []}, False), ({"metadata": None}, False),
    ({"metadata": {**{str(n): "x" * 500 for n in range(8)}, "8": "x" * 15}}, True),
    ({"metadata": {**{str(n): "x" * 500 for n in range(8)}, "8": "x" * 16}}, False),
])
async def test_contract_2_engagement_validates_optional_context(store, client, context, accepted):
    row = create(store)
    response = await client.post(f"{ROOT}/{row.id}/engagement",
                                 content=json.dumps({**ios_engagement_body(), **context}),
                                 headers={"Content-Type": "application/json"})
    assert response.status_code == (200 if accepted else 422)
    if not accepted:
        assert response.json()["detail"] == "invalid idea engagement"
    assert store._db.execute("SELECT count(*) FROM initiative_engagement").fetchone()[0] == int(accepted)
