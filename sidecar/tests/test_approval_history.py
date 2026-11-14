"""B12: real recorded asks, revision-bound paging, private owner reads and restart."""
import asyncio
import base64
from datetime import datetime, timedelta, timezone
import json
import sqlite3
import threading
from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from onekey import AUTH, KEY
from colonymind.api.credentials import CredentialTable, secret_digest
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import host
from colonymind.initiatives.store import InitiativeStore
from colonymind.mind import approval_history

PATH = "/v1/host/queue/approvals/history"
NOW = datetime(2026, 10, 18, 12, tzinfo=timezone.utc)
TABLE = CredentialTable.from_config({
    "api-key": {"owner": ["companion"], "contacts": "all"},
    "reader": {"sha256": secret_digest("scoped-reader"), "owner": ["companion"],
               "contacts": "all", "scopes": ["approvals:read"]},
    "decider": {"sha256": secret_digest("scoped-decider"), "owner": ["companion"],
                "contacts": "all", "scopes": ["approvals:decide"]},
    "guest": {"sha256": secret_digest("guest-secret"), "owner": [],
              "contacts": "all", "access": ["manage"]},
    "context": {"sha256": secret_digest("context-secret"), "owner": ["companion"],
                "contacts": "all", "access": ["read"]},
    "answer": {"sha256": secret_digest("answer-secret"), "owner": ["companion"],
               "access": ["answer"]},
    "hands": {"sha256": secret_digest("hands-secret"), "access": ["hands"]},
})


@pytest.fixture
async def rig(tmp_path, monkeypatch):
    store = InitiativeStore(tmp_path)
    mind = SimpleNamespace(store=store, clock=lambda: NOW)
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    monkeypatch.setattr(host, "_mind", lambda: mind)
    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware, api_key=KEY, credentials=TABLE)
    app.include_router(host.router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", headers=AUTH) as client:
        yield SimpleNamespace(store=store, mind=mind, client=client, path=tmp_path, app=app)
    mind.store.close()


def ask(store, key, *, at=NOW, **extra):
    fields = dict(kind="task", type="research", title="Review " + key, drive="curiosity", cls="self",
                  decision="ask", decision_reason="existing reason", status="asked", dedup_key=key,
                  ask_code="ABC234", created_at=at, expires_at=NOW + timedelta(hours=1), context={})
    fields.update(extra)
    return store.create_intention(**fields)[0]


def settle(store, row, action, *, status="cancelled", **extra):
    return store.transition(row.id, status, action=action, at=NOW, **extra)


async def page(rig, **params):
    response = await rig.client.get(PATH, params=params)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    return response.json()


async def test_empty_unknown_ids_non_asks_and_capability(rig):
    empty = await page(rig)
    assert empty["items"] == empty["sessions"] == []
    assert empty["counts"] == {"pending": 0, "settled": 0, "total": 0}
    assert empty["has_more"] is False and empty["next_cursor"] is None
    assert "approval_history" in host.supported_capabilities()
    ordinary = ask(rig.store, "automatic-action", ask_code=None, decision="act", status="approved")
    assert (await page(rig))["items"] == []  # no synthesized approval from an action
    for ident in ("unknown", ordinary.id):
        response = await rig.client.get(PATH + "/" + ident)
        assert response.status_code == 404
        assert response.json() == {"detail": {"code": "unknown_approval"}}
        assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("secret,status,code", [
    ("invalid", 401, None), ("guest-secret", 403, "not_owner"),
    ("scoped-decider", 403, "route_not_granted"), ("context-secret", 403, "route_not_granted"),
    ("answer-secret", 403, "route_not_granted"), ("hands-secret", 403, None),
])
async def test_reads_refuse_missing_authority_before_private_data(rig, secret, status, code):
    row = ask(rig.store, "private-request")
    for path in (PATH, PATH + "/" + row.id):
        response = await rig.client.get(path, headers={"Authorization": "Bearer " + secret})
        assert response.status_code == status
        assert response.headers["cache-control"] == "no-store"
        if code:
            assert response.json()["detail"]["code"] == code
        assert row.description not in response.text
    assert (await rig.client.get(PATH, headers={"Authorization": ""})).status_code == 401


async def test_existing_read_scope_can_read_but_cannot_decide(rig):
    row = ask(rig.store, "readable")
    headers = {"Authorization": "Bearer scoped-reader"}
    assert (await rig.client.get(PATH, headers=headers)).status_code == 200
    assert (await rig.client.get(PATH + "/" + row.id, headers=headers)).status_code == 200
    assert (await rig.client.post(PATH + "/" + row.id, headers=headers)).status_code == 403
    assert rig.store.get(row.id).status == "asked"


async def test_development_mode_cannot_read_private_history(rig):
    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware, credentials=TABLE)
    app.include_router(host.router)
    async with AsyncClient(transport=ASGITransport(app=app, client=("127.0.0.1", 42)),
                           base_url="http://localhost") as client:
        assert (await client.get(PATH)).status_code == 403
        assert (await client.get(PATH + "/anything")).status_code == 403


@pytest.mark.parametrize("action,kwargs,expected", [
    ("queued", {"status": "approved", "details": {"by": "owner", "answer": {"path": "explicit"}}}, "allowed"),
    ("queued", {"status": "approved", "decision": "act"}, "auto"),
    ("outcome_denied", {"outcome": "denied"}, "denied"),
    ("outcome_expired", {"status": "expired", "outcome": "expired"}, "timed_out"),
    ("outcome_cancelled", {"outcome": "cancelled"}, "cancelled"),
    ("outcome_failed", {"status": "failed", "outcome": "failed"}, "failed"),
    ("outcome_done", {"status": "done", "outcome": "done"}, "done"),
    ("outcome_uncertain", {"status": "uncertain", "outcome": "uncertain"}, "uncertain"),
    ("dropped", {"status": "dropped", "outcome": "denied"}, "denied"),
])
async def test_closed_outcomes_come_from_recorded_events(rig, action, kwargs, expected):
    row = ask(rig.store, "outcome")
    settle(rig.store, row, action, **kwargs)
    result = (await page(rig))["items"][0]
    assert result["outcome"] == expected and result["uncertainty"] is None
    assert result["state"] == "settled" and result["open"] is False
    assert result["settled_at"] == NOW.isoformat()
    assert set(result).isdisjoint({"terms", "settlement", "raw_outcome"})


async def test_clock_expiry_missing_and_unknown_records_never_invent_decisions(rig):
    expired = ask(rig.store, "unswept", expires_at=NOW - timedelta(seconds=1))
    missing = ask(rig.store, "missing", status="approved")
    unknown = ask(rig.store, "unknown")
    settle(rig.store, unknown, "outcome_future_value", status="future_value")
    ambiguous = ask(rig.store, "ambiguous")
    settle(rig.store, ambiguous, "queued", status="approved")
    before = rig.store._db.total_changes
    result = {item["id"]: item for item in (await page(rig))["items"]}
    assert result[expired.id]["state"] == "pending" and result[expired.id]["open"] is False
    assert result[expired.id]["outcome"] is None and result[expired.id]["settled_at"] is None
    for row in (missing, ambiguous):
        assert result[row.id]["outcome"] == "uncertain"
        assert result[row.id]["uncertainty"] == "missing_settlement"
    assert result[missing.id]["settled_at"] is None
    assert result[unknown.id]["uncertainty"] == "unrecognized_settlement"
    assert rig.store._db.total_changes == before
    assert rig.store.get(expired.id).status == "asked"


async def test_pending_first_tied_times_counts_and_no_duplicates_through_restart(rig):
    settled = [ask(rig.store, "history-" + str(i), at=NOW + timedelta(minutes=1)) for i in range(3)]
    for row in settled:
        settle(rig.store, row, "outcome_denied", outcome="denied")
    pending = [ask(rig.store, "pending-" + str(i)) for i in range(3)]
    deduped = ask(rig.store, "pending-0")
    assert deduped.id == pending[0].id
    first = await page(rig, limit=2)
    expected = sorted(row.id for row in pending) + sorted(row.id for row in settled)
    assert [item["id"] for item in first["items"]] == expected[:2]
    assert first["counts"] == {"pending": 3, "settled": 3, "total": 6}
    assert first["sessions"] == [{"session_id": None, **first["counts"]}]
    assert "Review" not in base64.urlsafe_b64decode(first["next_cursor"]).decode()
    ids = [item["id"] for item in first["items"]]
    cursor = first["next_cursor"]
    second = await page(rig, limit=2, cursor=cursor)
    rig.store.close()
    rig.store = rig.mind.store = InitiativeStore(rig.path)
    rig.mind.clock = lambda: NOW + timedelta(days=1)
    assert await page(rig, limit=2, cursor=cursor) == second
    assert second["items"][0]["open"] is True  # fixed snapshot_at, despite clock advancing
    while cursor:
        value = await page(rig, limit=2, cursor=cursor)
        assert value["revision"] == first["revision"]
        assert value["counts"] == first["counts"]
        ids.extend(item["id"] for item in value["items"])
        cursor = value["next_cursor"]
    assert ids == expected
    assert (await page(rig, state="settled"))["counts"] == first["counts"]
    assert (await page(rig, session_id="unknown"))["counts"]["total"] == 0


@pytest.mark.parametrize("mutation", ["insert", "settle", "history", "delete"])
async def test_cursor_refuses_changed_revision(rig, mutation):
    rows = [ask(rig.store, str(i)) for i in range(3)]
    first = await page(rig, limit=1)
    if mutation == "insert":
        ask(rig.store, "new")
    elif mutation == "settle":
        settle(rig.store, rows[-1], "outcome_denied", outcome="denied")
    elif mutation == "history":
        settle(rig.store, rows[-1], "ask_shown", status="asked", details={"message": "private"})
    else:
        rig.store._db.execute("DELETE FROM initiatives WHERE id=?", (rows[-1].id,))
        rig.store._db.commit()
    response = await rig.client.get(PATH, params={"limit": 1, "cursor": first["next_cursor"]})
    assert response.status_code == 409 and response.json()["detail"]["code"] == "approval_history_changed"
    assert response.headers["cache-control"] == "no-store"


async def test_cursor_query_binding_validation_and_paging_edges(rig):
    for i in range(4):
        ask(rig.store, str(i))
    exact = await page(rig, limit=4)
    assert exact["next_cursor"] is None and exact["has_more"] is False
    cursor = (await page(rig, limit=2))["next_cursor"]
    for params in ({"limit": 1}, {"limit": 2, "state": "pending"}, {"limit": 2, "session_id": "other"}):
        response = await rig.client.get(PATH, params={**params, "cursor": cursor})
        assert response.status_code == 422 and response.json()["detail"]["code"] == "invalid_cursor"
    for token in ("!invalid", "null", "W10=", "e30="):
        response = await rig.client.get(PATH, params={"cursor": token})
        assert response.status_code == 422 and response.json()["detail"]["code"] == "invalid_cursor"
    decoded = json.loads(base64.urlsafe_b64decode(cursor))
    for changes in ({"v": 2}, {"v": True}, {"revision": "x" * 64}, {"position": -1},
                    {"position": 1}, {"position": 4}, {"snapshot_at": "bad"}):
        token = base64.urlsafe_b64encode(json.dumps({**decoded, **changes}).encode()).decode()
        response = await rig.client.get(PATH, params={"limit": 2, "cursor": token})
        assert response.status_code == 422 and response.json()["detail"]["code"] == "invalid_cursor"
    for params in ({"limit": 0}, {"limit": 201}, {"state": "other"}, {"session_id": ""}, {"cursor": ""}):
        assert (await rig.client.get(PATH, params=params)).status_code == 422


async def test_settlement_is_not_lost_after_500_history_entries_or_replaced_by_execution(rig):
    row = ask(rig.store, "long-history")
    settle(rig.store, row, "queued", status="approved", details={"by": "owner", "answer": {"path": "explicit"}})
    rig.store._db.executemany("INSERT INTO assignment_history(initiative_id,agent_id,action,timestamp,details) "
                              "VALUES (?,?,?,?,?)", [(row.id, "mind", "ask_shown", NOW.isoformat(), "{}")
                                                     for _ in range(510)])
    rig.store._db.commit()
    settle(rig.store, row, "outcome_failed", status="failed", outcome="failed")
    detail = (await rig.client.get(PATH + "/" + row.id)).json()["approval"]
    assert detail["outcome"] == "allowed" and detail["raw_outcome"] == "failed"
    assert detail["settlement"]["outcome"] == "approved"
    settle(rig.store, row, "ask_revived", status="asked", outcome=None)
    assert (await page(rig))["items"][0]["outcome"] is None
    settle(rig.store, row, "outcome_denied", outcome="denied")
    assert (await page(rig))["items"][0]["outcome"] == "denied"


async def test_earlier_automatic_action_is_not_the_later_requests_settlement(rig):
    row = ask(rig.store, "reconsidered", status="approved", decision="act")
    settle(rig.store, row, "queued", status="approved", decision="act")
    settle(rig.store, row, "asked", status="asked", decision="ask")
    settle(rig.store, row, "outcome_denied", outcome="denied")
    assert (await page(rig))["items"][0]["outcome"] == "denied"


async def test_pagination_reads_more_than_old_500_request_bound(rig):
    for i in range(503):
        ask(rig.store, str(i))
    ids, cursor = [], None
    while True:
        value = await page(rig, limit=200, **({"cursor": cursor} if cursor else {}))
        assert value["counts"]["total"] == 503
        ids.extend(item["id"] for item in value["items"])
        cursor = value["next_cursor"]
        if cursor is None:
            break
    assert len(ids) == len(set(ids)) == 503


async def test_select_count_stays_bounded_with_3000_asks_and_full_text_notices(rig, monkeypatch):
    from colonymind.mind.outbox import ASK_IN_FULL

    def create(index):
        row = ask(rig.store, f"bounded-{index}", context={"full_view": {"attempts": 2}})
        notice = ask(rig.store, f"{ASK_IN_FULL}:{row.id}:3", kind="message", type=ASK_IN_FULL,
                     ask_code=None, status="sent", context={"ask_id": row.id})
        if index % 2:
            # Also resolve an explicitly named notice without a matching retry key.
            rig.store.update(row.id, context={"full_view": {"notice": notice.id}})
        return row

    statements = []
    connect = sqlite3.connect

    def traced(*args, **kwargs):
        conn = connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(approval_history.sqlite3, "connect", traced)

    def read(**query):
        statements.clear()
        value = approval_history.read(rig.store, now=NOW, limit=1, **query)
        selects = [sql for sql in statements if sql.lstrip().upper().startswith("SELECT")]
        assert len(selects) <= 4, selects
        return value, len(selects)

    first = create(0)
    _, small_count = read()
    for index in range(1, 3000):
        last = create(index)
    result, large_count = read()
    assert result["counts"]["total"] == 3000
    assert large_count == small_count
    read(cursor=result["next_cursor"])
    for row in (first, last):
        detail, _ = read(approval_id=row.id)
        assert detail["approval"]["terms"]["full_text_delivered"] is True
    unknown, _ = read(session_id="unknown")
    assert unknown["items"] == []


async def test_detail_retains_request_terms_and_excludes_arbitrary_private_context(rig):
    words = "A retained message to its recipient, in full. " * 30
    row = ask(rig.store, "message", kind="message", context={"text": words, "unrelated": {"private": "never expose"}})
    settle(rig.store, row, "queued", status="approved", details={"by": "owner"})
    response = await rig.client.get(PATH + "/" + row.id)
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    detail = response.json()["approval"]
    assert detail["terms"]["text"] == words and detail["terms"]["reason"] == "existing reason"
    assert detail["terms"]["call"] == {"tool": None, "arguments": {}, "workdir": None, "targets": []}
    assert "never expose" not in response.text
    summary = (await page(rig))["items"][0]
    assert all(detail[key] == value for key, value in summary.items())


async def test_unavailable_or_malformed_data_is_a_private_503_not_empty_history(rig, monkeypatch, caplog):
    row = ask(rig.store, "malformed")
    rig.store._db.execute("UPDATE initiatives SET context=? WHERE id=?", ('{"private":"do not log"', row.id))
    rig.store._db.commit()
    response = await rig.client.get(PATH)
    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "approval_history_unavailable"}}
    assert "do not log" not in response.text + caplog.text
    monkeypatch.setattr(host, "_mind", lambda: None)
    assert "approval_history" not in host.supported_capabilities()
    assert (await rig.client.get(PATH)).status_code == 503
    monkeypatch.setattr(host, "_mind", lambda: SimpleNamespace(store=SimpleNamespace(_db_path=rig.path / "missing.db"),
                                                               clock=lambda: NOW))
    assert (await rig.client.get(PATH)).status_code == 503
    assert not (rig.path / "missing.db").exists()


async def test_malformed_response_data_is_refused_without_private_validation_diagnostics(rig, monkeypatch, caplog):
    ask(rig.store, "private")
    original = approval_history.read

    def malformed(*args, **kwargs):
        value = original(*args, **kwargs)
        value["items"][0]["hand_id"] = {"private": ["never include me in validation logs"]}
        return value

    monkeypatch.setattr(approval_history, "read", malformed)
    response = await rig.client.get(PATH)
    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "approval_history_unavailable"}}
    assert "never include me" not in response.text + caplog.text


@pytest.mark.parametrize("call", [
    {"tool": "terminal", "args": ["private malformed argument"], "targets": "private malformed targets"},
    {"tool": "terminal", "args": ["private malformed argument"], "targets": []},
    {"tool": "terminal", "args": {}, "targets": "private malformed targets"},
    {"tool": "terminal", "args": {}, "targets": ["private malformed target"]},
    {"tool": "terminal", "args": {}, "targets": [{"kind": "file", "target": {"private": "target"}, "value": "x"}]},
])
async def test_malformed_persisted_nested_terms_are_a_redacted_503(rig, caplog, call):
    row = ask(rig.store, "malformed-call", context={"effect": {"call": json.dumps(call)}})
    response = await rig.client.get(PATH + "/" + row.id)
    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "approval_history_unavailable"}}
    assert response.headers["cache-control"] == "no-store"
    assert "private" not in response.text + caplog.text


async def test_database_work_runs_off_the_event_loop(rig, monkeypatch):
    original = approval_history.read
    started, release = threading.Event(), threading.Event()
    loop_thread = threading.get_ident()

    def blocked(*args, **kwargs):
        assert threading.get_ident() != loop_thread
        started.set()
        assert release.wait(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(approval_history, "read", blocked)
    task = asyncio.create_task(rig.client.get(PATH))
    try:
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set() and not task.done()
    finally:
        release.set()
        assert (await task).status_code == 200
