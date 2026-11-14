"""Owner authentication, bounded views and immutable work event HTTP behavior."""
from datetime import datetime, timedelta, timezone
from dataclasses import replace
from contextlib import closing
import sqlite3

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from colonymind.api.auth import anonymous_authority, key_authority
from colonymind.api.routers import work

def detail_path(ident):
    return "/v1/host/work/" + work._encode_id(ident)


NOW = datetime.now(timezone.utc)
SINCE = (NOW - timedelta(days=1)).isoformat()
UNTIL = NOW.isoformat()


class Store:
    def __init__(self):
        self.calls = []
        self.rows = [dict(id=f"root-{index}", contact_id="owner-1", session_id="session-1",
                          turn_id=f"root-{index}", status=status, created_at=SINCE, title=None, revision=1)
                     for index, status in enumerate(("proven", "failed", "unproven"))]
        self.events = {}

    def list_roots(self, **kwargs):
        self.calls.append(kwargs)
        rows = self.rows
        if kwargs["status"]:
            rows = [row for row in rows if row["status"] == kwargs["status"]]
        return rows[kwargs["offset"]:kwargs["offset"] + kwargs["limit"]]

    def get_root(self, id, **kwargs):
        self.calls.append(kwargs)
        row = next((row for row in self.rows if row["id"] == id), None)
        if row is None:
            return None
        effect = dict(id="effect-1", operation_id="op-1", tool="terminal", action="execute",
                      resource_ref="resource:1", kind="execute", status="ran",
                      verification="successful_call_only", source_refs=[
                          {"source_id": "evidence-source", "source_version": "a" * 64}])
        return {**row, "effects": [effect] * 257, "children": [], "raw_arguments": "never returned"}

    def summary(self, **kwargs):
        self.calls.append(kwargs)
        return {"total": 3, "counts": {"proven": 1, "failed": 1, "unproven": 1},
                "owner_corrected": 0, "unknown_effects": [dict(tool="external", action="call", count=2)]}

    def admit_event(self, body, **kwargs):
        self.calls.append(kwargs)
        if body.root_turn_id == "missing":
            raise ValueError("work_root_missing")
        if body.root_turn_id == "parent-pending":
            raise ValueError("work_parent_missing")
        if body.root_turn_id == "mismatch":
            raise ValueError("work_child_parent_mismatch")
        if body.root_turn_id == "invalid":
            raise ValueError("private diagnostic must not leave the server")
        old = self.events.get(body.event_id)
        if old is not None and old != body:
            raise ValueError("work_event_conflict")
        self.events[body.event_id] = body
        return {"accepted": True, "duplicate": old is not None}


@pytest.fixture
def service(monkeypatch):
    store = Store()
    monkeypatch.setattr(work, "get_work_store", lambda **kwargs: store)
    monkeypatch.setenv("COLONYMIND_OWNER_PERSON_ID", "owner-1")
    return store


def app(authority=None):
    result = FastAPI()
    if authority is not None:
        @result.middleware("http")
        async def authorize(request, call_next):
            request.state.colonymind_authority = authority
            return await call_next(request)
    result.include_router(work.router)
    return result


@pytest.mark.asyncio
async def test_cold_reads_never_initialize_or_write_either_database(tmp_path, monkeypatch):
    from colonymind.turns import TurnIdempotencyLedger
    import colonymind.work
    from colonymind.work.store import WorkStore
    from test_work_projection import envelope
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(tmp_path))
    monkeypatch.setenv('COLONYMIND_OWNER_PERSON_ID', 'owner')
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    ledger.record_source('root', contact_id='owner', session_id='session',
        messages=[{'role': 'user', 'content': 'Update the report'}],
        work={'envelope': envelope(), 'principal_id': 'api-key'}, derive_claims=False)
    WorkStore(ledger).reconcile()

    def no_writer(*args, **kwargs):
        pytest.fail('cold GET initialized a writable work store')

    monkeypatch.setattr(colonymind.work, '_store', no_writer)
    connect = sqlite3.connect
    opened, statements = [], []

    def traced(database, *args, **kwargs):
        opened.append(str(database))
        conn = connect(database, *args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    with closing(ledger._connect()) as writer:
        writer.execute('BEGIN IMMEDIATE')
        monkeypatch.setattr(sqlite3, 'connect', traced)
        async with AsyncClient(transport=ASGITransport(app=app(key_authority())), base_url='http://example.test') as client:
            listing = await client.get('/v1/host/work')
            summary = await client.get('/v1/host/work/summary')
            detail = await client.get(detail_path('root'))
        writer.rollback()
    assert listing.status_code == summary.status_code == detail.status_code == 200
    assert listing.json()['enabled'] is True and len(listing.json()['items']) == 1
    assert summary.json()['total'] == 1 and detail.json()['status'] == 'proven'
    assert opened and all(path.endswith('?mode=ro') for path in opened)
    assert not any(query.upper().startswith(('CREATE ', 'ALTER ', 'INSERT ', 'UPDATE ', 'DELETE ',
                                            'BEGIN IMMEDIATE', 'BEGIN EXCLUSIVE')) for query in statements)
    assert all('mode=ro' in query for query in statements if query.startswith('ATTACH '))


@pytest.mark.asyncio
@pytest.mark.parametrize('ledger_exists', [False, True])
async def test_cold_missing_projection_stays_enabled_without_creating_databases(tmp_path, monkeypatch, ledger_exists):
    from colonymind.turns import TurnIdempotencyLedger
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(tmp_path))
    monkeypatch.setenv('COLONYMIND_OWNER_PERSON_ID', 'owner')
    ledger_path = tmp_path / 'turn-idempotency.db'
    if ledger_exists:
        TurnIdempotencyLedger(ledger_path)
    async with AsyncClient(transport=ASGITransport(app=app(key_authority())), base_url='http://example.test') as client:
        listing = await client.get('/v1/host/work')
        summary = await client.get('/v1/host/work/summary')
        detail = await client.get(detail_path('missing'))
    assert listing.status_code == summary.status_code == 200 and detail.status_code == 404
    assert listing.json()['enabled'] is True and listing.json()['items'] == []
    assert summary.json()['enabled'] is True and summary.json()['total'] == 0
    assert ledger_path.exists() is ledger_exists
    assert not (tmp_path / 'colonymind-work.db').exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("authority", [
    None,
    anonymous_authority(),
    replace(key_authority(), owner_platforms=frozenset()),
    replace(key_authority(), principal_id="issued-owner", credential_id="issued-owner"),
    replace(key_authority(), scopes=frozenset({"manage"})),
])
async def test_only_the_owner_instance_credential_reaches_work(service, authority):
    async with AsyncClient(transport=ASGITransport(app=app(authority)), base_url="http://local") as client:
        for path in ("", "/summary", "/root-0"):
            response = await client.get("/v1/host/work" + path)
            assert response.status_code == 403
        response = await client.post("/v1/host/work/events", json=event())
        assert response.status_code == 403
    assert service.calls == []


@pytest.mark.asyncio
async def test_global_middleware_rejects_missing_credential(service):
    from colonymind.api.credentials import CredentialTable, default_credentials
    from colonymind.api.middleware import ApiKeyMiddleware
    instance = app()
    instance.add_middleware(ApiKeyMiddleware, api_key="fixture-only-key",
                            credentials=CredentialTable.from_config(default_credentials()))
    async with AsyncClient(transport=ASGITransport(app=instance), base_url="http://local") as client:
        assert (await client.get("/v1/host/work")).status_code == 401
        response = await client.get("/v1/host/work", headers={"X-API-Key": "fixture-only-key"})
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_owner_paging_filters_and_summary_remain_separate(service):
    async with AsyncClient(transport=ASGITransport(app=app(key_authority())), base_url="http://local") as client:
        params = {"since": SINCE, "until": UNTIL, "limit": 2}
        first = (await client.get("/v1/host/work", params=params)).json()
        assert [row["turn_id"] for row in first["items"]] == ["root-0", "root-1"]
        assert first["next_offset"] == 2
        second = (await client.get("/v1/host/work", params={**params, "offset": 2})).json()
        assert [row["turn_id"] for row in second["items"]] == ["root-2"]
        assert second["next_offset"] is None
        filtered = (await client.get("/v1/host/work", params={**params, "status": "failed"})).json()
        assert [row["turn_id"] for row in filtered["items"]] == ["root-1"]
        summary = (await client.get("/v1/host/work/summary", params=params)).json()
        assert summary["total"] == 3
        assert summary["counts"] == {"proven": 1, "partial": 0, "unproven": 1, "failed": 1}
        assert summary["rates"]["failed"] == pytest.approx(1 / 3)
        assert summary["roots_per_day"] == 3
        assert summary["owner_corrected_rate"] == 0
        assert summary["unknown_effects"] == [{"tool": "external", "action": "call", "count": 2}]
    assert all(call["contact_id"] == "owner-1" for call in service.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("params", [
    {"limit": 0}, {"limit": 101}, {"offset": -1}, {"offset": 10001}, {"status": "observed"},
    {"since": "2026-01-01"}, {"since": "2026-01-01T00:00:00"}, {"until": "invalid"},
    {"since": UNTIL, "until": SINCE}, {"since": SINCE, "until": SINCE},
    {"since": (NOW - timedelta(days=367)).isoformat()},
    {"until": (NOW + timedelta(days=1)).isoformat()},
])
async def test_invalid_list_bounds_never_read_the_store(service, params):
    async with AsyncClient(transport=ASGITransport(app=app(key_authority())), base_url="http://local") as client:
        response = await client.get("/v1/host/work", params=params)
        assert response.status_code == 422, response.text
    assert service.calls == []


@pytest.mark.asyncio
async def test_summary_date_bounds_and_detail_size(service):
    async with AsyncClient(transport=ASGITransport(app=app(key_authority())), base_url="http://local") as client:
        response = await client.get("/v1/host/work/summary", params={"since": "2026-01-01T00:00:00"})
        assert response.status_code == 422
        response = await client.get("/v1/host/work/summary")
        summary = response.json()
        assert datetime.fromisoformat(summary["until"]) - datetime.fromisoformat(summary["since"]) == timedelta(days=7)
        response = await client.get(detail_path("root-0"))
        assert response.status_code == 200
        assert len(response.json()["effects"]) == 256
        assert response.json()["effects_truncated"] is True
        assert response.json()["effects"][0]["verification"] == "successful_call_only"
        assert response.json()["effects"][0]["source_refs"][0]["source_id"] == "evidence-source"
        assert "raw_arguments" not in response.text
        assert (await client.get(detail_path("missing"))).status_code == 404
        assert (await client.get("/v1/host/work/" + "x" * 1371)).status_code == 422


def event(root="root-0", ident="event-1"):
    return {"event_id": ident, "root_turn_id": root, "session_id": "session-child", "source_refs": [],
            "work": {"version": 1, "root_turn_id": root, "operations": [], "children": [], "truncated": False}}


@pytest.mark.asyncio
async def test_event_retry_conflict_and_parent_not_yet_admitted(service):
    async with AsyncClient(transport=ASGITransport(app=app(key_authority())), base_url="http://local") as client:
        first = await client.post("/v1/host/work/events", json=event())
        assert first.status_code == 200 and first.json()["duplicate"] is False
        retry = await client.post("/v1/host/work/events", json=event())
        assert retry.status_code == 200 and retry.json()["duplicate"] is True
        changed = event()
        changed["work"]["truncated"] = True
        assert (await client.post("/v1/host/work/events", json=changed)).status_code == 409
        assert (await client.post("/v1/host/work/events", json=event("missing", "event-2"))).status_code == 404
        assert (await client.post("/v1/host/work/events", json=event("parent-pending", "event-3"))).status_code == 404
        assert (await client.post("/v1/host/work/events", json=event("mismatch", "event-4"))).status_code == 422
        refused = await client.post("/v1/host/work/events", json=event("invalid", "event-5"))
        assert refused.status_code == 422 and refused.json()["detail"]["code"] == "work_event_invalid"
        assert "private diagnostic" not in refused.text


@pytest.mark.asyncio
async def test_disabled_views_and_events_do_not_initialize_projection(monkeypatch):
    monkeypatch.setattr(work, "get_work_store", lambda **kwargs: None)
    async with AsyncClient(transport=ASGITransport(app=app(key_authority())), base_url="http://local") as client:
        listing = (await client.get("/v1/host/work")).json()
        assert listing["enabled"] is False and listing["items"] == []
        summary = (await client.get("/v1/host/work/summary")).json()
        assert summary["enabled"] is False and summary["total"] == 0
        assert summary["unknown_effects"] == []
        assert (await client.get(detail_path("root-0"))).status_code == 404
        receipt = (await client.post("/v1/host/work/events", json=event())).json()
        assert receipt["accepted"] is False and receipt["skipped_reason"] == "work_records_disabled"


def retained_work(ledger, ident, *, operations=None, children=(), at=SINCE, contact="owner-1"):
    if operations is None:
        operations = [{'operation_id': 'call', 'tool_call_id': 'call', 'tool': 'terminal',
                       'action': 'execute', 'effect': 'execute', 'success': True}]
    ledger.record_source(ident, contact_id=contact, session_id="owner-session",
        messages=[{"role": "user", "content": "Please handle this request."}],
        occurred_at=at, derive_claims=False, work={"principal_id": key_authority().principal_id,
            "envelope": {"version": 1, "root_turn_id": ident, "operations": list(operations),
                         "children": list(children), "truncated": False}})


@pytest.mark.asyncio
async def test_real_projection_paging_at_maximum_and_exclusive_time_end(tmp_path, monkeypatch):
    from colonymind.turns import TurnIdempotencyLedger
    from colonymind.work.store import WorkStore
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    for index in range(102):
        retained_work(ledger, f"root-{index:03}")
    retained_work(ledger, "at-end", at=UNTIL)
    retained_work(ledger, "other-person", contact="guest-1")
    store = WorkStore(ledger)
    store.reconcile()
    monkeypatch.setenv("COLONYMIND_OWNER_PERSON_ID", "owner-1")
    monkeypatch.setattr(work, "get_work_store", lambda **kwargs: store)
    async with AsyncClient(transport=ASGITransport(app=app(key_authority())), base_url="http://local") as client:
        params = {"since": SINCE, "until": UNTIL, "limit": 100}
        first = await client.get("/v1/host/work", params=params)
        assert first.status_code == 200, first.text
        page = first.json()
        assert len(page["items"]) == 100 and page["next_offset"] == 100
        second = (await client.get("/v1/host/work", params={**params, "offset": 100})).json()
        assert len(second["items"]) == 2 and second["next_offset"] is None
        ids = {row["turn_id"] for row in page["items"] + second["items"]}
        assert len(ids) == 102 and "at-end" not in ids and "other-person" not in ids
        summary = (await client.get("/v1/host/work/summary", params=params)).json()
        assert summary["total"] == 102 and summary["counts"]["unproven"] == 102
        assert (await client.get(detail_path("other-person"))).status_code == 404


@pytest.mark.asyncio
async def test_real_late_event_updates_status_and_preserves_detail_truncation(tmp_path, monkeypatch):
    from colonymind.turns import TurnIdempotencyLedger
    from colonymind.work.store import WorkStore
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    operations = [{"operation_id": f"op-{i}", "tool_call_id": f"call-{i}", "tool": "terminal",
                   "action": "execute", "effect": "execute", "resource_ref": f"resource:{i}",
                   "success": True, "evidence": {"kind": "none"}, "source_refs": []} for i in range(256)]
    children = [{"turn_id": f"child-{i}", "parent_turn_id": "root-0"} for i in range(256)]
    retained_work(ledger, "root-0", operations=operations, children=children)
    store = WorkStore(ledger)
    store.reconcile()
    monkeypatch.setenv("COLONYMIND_OWNER_PERSON_ID", "owner-1")
    monkeypatch.setattr(work, "get_work_store", lambda **kwargs: store)
    async with AsyncClient(transport=ASGITransport(app=app(key_authority())), base_url="http://local") as client:
        before = await client.get(detail_path("root-0"))
        assert before.status_code == 200 and before.json()["status"] == "unproven"
        late = event()
        late["work"]["parent_turn_id"] = "root-0"
        late["work"]["operations"] = [{**operations[0], "operation_id": "op-late",
                                        "tool_call_id": "call-late", "success": False}]
        late["work"]["children"] = [{"turn_id": "child-late", "parent_turn_id": "root-0"}]
        admitted = await client.post("/v1/host/work/events", json=late)
        assert admitted.status_code == 200, admitted.text
        assert admitted.json()["accepted"] is True
        replay = await client.post("/v1/host/work/events", json=late)
        assert replay.status_code == 200 and replay.json()["duplicate"] is True
        assert store.wait_pending(timeout=2)
        after = (await client.get(detail_path("root-0"))).json()
        assert after["status"] == "failed"
        assert len(after["effects"]) == len(after["children"]) == 256
        assert after["effects_truncated"] is True and after["children_truncated"] is True
        assert "evidence_json" not in after and all("evidence_json" not in row for row in after["effects"])


@pytest.mark.asyncio
async def test_unknown_kind_summary_is_bounded_and_event_metadata_is_closed(service, monkeypatch):
    monkeypatch.setattr(service, "summary", lambda **kwargs: {
        "total": 1, "counts": {"unproven": 1}, "unknown_effects": [
            {"tool": f"external-{index}", "action": "call", "count": 1} for index in range(101)]})
    async with AsyncClient(transport=ASGITransport(app=app(key_authority())), base_url="http://local") as client:
        summary = (await client.get("/v1/host/work/summary")).json()
        assert len(summary["unknown_effects"]) == 100 and summary["unknown_effects_truncated"] is True
        body = event()
        body["raw_arguments"] = "unretained tool payload"
        assert (await client.post("/v1/host/work/events", json=body)).status_code == 422
        body = event()
        body["work"]["root_turn_id"] = "different-root"
        assert (await client.post("/v1/host/work/events", json=body)).status_code == 422


@pytest.mark.asyncio
async def test_opaque_identifiers_keep_reserved_slash_and_unicode_roots_reachable(service):
    originals = ["summary", "session/turn", "w:literal-host-id", "\u6d4b\u8bd5" * 128, "\U0001f9ed" * 256]
    service.rows = [{**service.rows[0], "id": ident, "turn_id": ident} for ident in originals]
    async with AsyncClient(transport=ASGITransport(app=app(key_authority())), base_url="http://local") as client:
        listing = (await client.get("/v1/host/work")).json()
        assert [row["turn_id"] for row in listing["items"]] == originals
        for row in listing["items"]:
            assert row["id"].startswith("w:") and "/" not in row["id"]
            response = await client.get("/v1/host/work/" + row["id"])
            assert response.status_code == 200
            assert response.json()["turn_id"] == row["turn_id"]
        assert "total" in (await client.get("/v1/host/work/summary")).json()
        for ident in ["root-0", "w:", "w:YQ==", "w:YR", "w:_w", "w:" + "A" * 1368,
                      work._encode_id("x" * 257)]:
            response = await client.get("/v1/host/work/" + ident)
            assert response.status_code == 422, (ident, response.text)
