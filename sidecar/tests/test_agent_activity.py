"""B11: durable current state, replay boundary, restart, and existing host auth."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from colonymind.api.credentials import CredentialTable, secret_digest
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import activity, host
from colonymind.events.activity import ActivityStore, ProducerSnapshot
from colonymind.events.journal import replay_events
from colonymind.turns.idempotency import TurnIdempotencyLedger

PRODUCER = "hermes-" + "a" * 64
OTHER = "hermes-" + "b" * 64
PATH = "/v1/host/activity"
AUTH = {"Authorization": "Bearer test-key"}


def body(revision=1, code="thinking", *, scope="thread-1", count=None, producer=PRODUCER):
    return {"producer_id": producer, "revision": revision, "activities": [] if code is None else [
        {"scope": scope, "activity_code": code, "count": count if count is not None else (
            0 if code in {"idle", "error"} else 1)}]}


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_EVENT_JOURNAL_DIR", str(tmp_path / "events"))
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    store = ActivityStore(ledger)
    monkeypatch.setattr(activity, "store", lambda: ActivityStore(ledger))
    return store


def client(key="test-key", credentials=None):
    app = FastAPI()
    app.include_router(host.router)
    app.add_middleware(ApiKeyMiddleware, api_key=key, credentials=credentials or CredentialTable.from_config({}))
    return TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 1234))


def apply(state, value):
    return state.ingest(ProducerSnapshot.model_validate(value))


def by_scope(state):
    return {row["scope"]: row for row in state.snapshot()["activities"]}


def test_empty_is_unknown_and_capabilities_are_explicit(state):
    with client() as c:
        response = c.get(PATH, headers=AUTH)
        snapshot = response.json()
    assert response.headers["cache-control"] == "no-store"
    assert snapshot["available"] is False and snapshot["activities"] == [] and snapshot["cursor"] == 0
    caps = snapshot["capabilities"]
    assert caps["activity_codes"] == ["idle", "thinking", "tool_running", "waiting_approval", "error"]
    assert caps["disconnect_clears_busy"] == "consumer" and caps["text_expiry_seconds"] is None
    assert caps["extensions"] == [
        {"id": "desktop", "supported": False, "read_route": None},
        {"id": "mailbox", "supported": True, "read_route": "/v1/agent-mail/mailboxes"},
        {"id": "hands", "supported": True, "read_route": "/v1/hands/"}]
    assert "agent_activity_v1" in host.supported_capabilities()


def test_state_transitions_are_explicit_and_per_scope_sequences_survive_restart(state, monkeypatch):
    seen = []
    for revision, code in enumerate(["thinking", "tool_running", "waiting_approval", "tool_running", "thinking", "error"], 1):
        apply(state, body(revision, code))
        rows = by_scope(state)
        assert set(rows) == {None, "thread-1"}
        assert rows[None]["activity_code"] == code and rows["thread-1"]["seq"] == revision
        assert rows[None]["count"] == (0 if code == "error" else 1)
        assert rows[None]["cta"] == ({"kind": "open_approvals"} if code == "waiting_approval" else None)
        seen.append(rows["thread-1"])
    # Moving the wall clock alone never clears text or busy/error state.
    import colonymind.events.activity as implementation
    with monkeypatch.context() as frozen:
        frozen.setattr(implementation, "now_utc", lambda: pytest.fail("snapshot must not age out state"))
        restarted = ActivityStore(TurnIdempotencyLedger(state.ledger.db_path))
        assert by_scope(restarted)["thread-1"] == seen[-1]
    apply(restarted, body(7, None))
    final = by_scope(restarted)
    assert all(row["activity_code"] == "idle" and row["count"] == 0 for row in final.values())
    assert final["thread-1"]["seq"] == 7


def test_parallel_scopes_and_installations_do_not_clear_one_another(state):
    first = body()
    first["activities"].append({"scope": "thread-2", "activity_code": "waiting_approval", "count": 2})
    apply(state, first)
    apply(state, body(1, "tool_running", scope="thread-1", count=3, producer=OTHER))
    rows = by_scope(state)
    assert rows[None]["activity_code"] == "waiting_approval" and rows[None]["count"] == 6
    assert rows["thread-1"]["activity_code"] == "tool_running" and rows["thread-1"]["count"] == 4
    apply(state, body(2, None))
    rows = by_scope(state)
    assert rows[None]["count"] == 3 and rows["thread-2"]["activity_code"] == "idle"
    assert rows["thread-1"]["activity_code"] == "tool_running"


def test_duplicate_conflict_and_out_of_order_receipts(state):
    with client() as c:
        first = c.post(PATH + "/snapshot", json=body(), headers=AUTH)
        assert first.json() == {"accepted": True, "duplicate": False, "superseded": False, "revision": 1}
        snapshot = state.snapshot()
        assert c.post(PATH + "/snapshot", json=body(), headers=AUTH).json()["duplicate"] is True
        assert state.snapshot() == snapshot
        conflict = c.post(PATH + "/snapshot", json=body(1, "error"), headers=AUTH)
        assert conflict.status_code == 409 and conflict.json()["detail"]["code"] == "activity_revision_conflict"
        c.post(PATH + "/snapshot", json=body(3, None), headers=AUTH)
        prior = state.snapshot()
        late = c.post(PATH + "/snapshot", json=body(2, "waiting_approval"), headers=AUTH)
        assert late.json() == {"accepted": False, "duplicate": False, "superseded": True, "revision": 3}
        assert state.snapshot() == prior


def test_snapshot_reconnect_cursor_and_replay_pagination(state):
    apply(state, body())
    snapshot = state.snapshot()
    apply(state, body(2, "tool_running"))
    apply(state, body(3, None))
    with client() as c:
        cursor, frames = snapshot["cursor"], []
        while True:
            page = c.get("/v1/host/events/replay", params={"types": "agent.status", "afterSeq": cursor, "limit": 1}, headers=AUTH).json()
            frames.extend(page["events"])
            if page["events"]:
                cursor = page["lastSeq"]
            if not page["hasMore"]:
                break
        empty = c.get("/v1/host/events/replay", params={"afterSeq": cursor, "limit": 1}, headers=AUTH).json()
        assert empty["events"] == [] and not empty["hasMore"] and empty["lastSeq"] == 0
        for params in [{"limit": 0}, {"limit": 1001}, {"afterSeq": -1}, {"untilSeq": -1}]:
            assert c.get("/v1/host/events/replay", params=params, headers=AUTH).status_code == 422
    assert len(frames) == 4
    assert [frame["seq"] for frame in frames] == sorted({frame["seq"] for frame in frames})
    assert {frame["data"]["scope"] for frame in frames} == {None, "thread-1"}
    assert all(frame["type"] == "agent.status" and frame["data"]["seq"] > 1 for frame in frames)


def test_snapshot_retained_when_journal_history_is_pruned(state, monkeypatch):
    monkeypatch.setenv("COLONYMIND_EVENT_JOURNAL_RETENTION", "1")
    apply(state, body())
    apply(state, body(2, "waiting_approval"))
    replay = replay_events(after_seq=0)
    assert len(replay["events"]) == 1 and replay["firstAvailableSeq"] > 1
    assert by_scope(state)["thread-1"]["activity_code"] == "waiting_approval"


def test_failed_publication_is_recoverable_after_store_restart(state, monkeypatch):
    original = host.broadcast_event
    monkeypatch.setattr(host, "broadcast_event", lambda _: None)
    with client() as c:
        failed = c.post(PATH + "/snapshot", json=body(), headers=AUTH)
        assert failed.status_code == 503 and failed.json()["detail"]["code"] == "activity_unavailable"
        assert c.get(PATH, headers=AUTH).status_code == 503
        monkeypatch.setattr(host, "broadcast_event", original)
        restarted = ActivityStore(TurnIdempotencyLedger(state.ledger.db_path))
        snapshot = restarted.snapshot()
        assert snapshot["available"] and snapshot["activities"][0]["activity_code"] == "thinking"
        assert c.post(PATH + "/snapshot", json=body(), headers=AUTH).json()["duplicate"] is True
    assert len(replay_events(after_seq=0)["events"]) == 2


def test_journal_receipt_recovers_crash_between_emit_and_pending_ack(state, monkeypatch):
    original = host.broadcast_event
    emitted = []
    def crash(frame):
        published = original(frame)
        emitted.append(published)
        if len(emitted) == 1:
            raise OSError("simulated process death after journal commit")
        return published
    monkeypatch.setattr(host, "broadcast_event", crash)
    with pytest.raises(OSError):
        apply(state, body())
    monkeypatch.setattr(host, "broadcast_event", original)
    ActivityStore(TurnIdempotencyLedger(state.ledger.db_path)).snapshot()
    frames = replay_events(after_seq=0)["events"]
    assert len(frames) == 2 and frames[0]["seq"] == emitted[0]["seq"]
    with closing(state.ledger._connect()) as conn:
        assert conn.execute("SELECT COUNT(*) FROM activity_publications").fetchone()[0] == 0


def test_concurrent_ingestion_cannot_regress_latest_revision(state):
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda revision: apply(state, body(revision, "thinking" if revision < 20 else "error")), range(1, 21)))
    assert by_scope(state)[None]["activity_code"] == "error"
    events = replay_events(after_seq=0)["events"]
    for scope in (None, "thread-1"):
        seqs = [frame["data"]["seq"] for frame in events if frame["data"]["scope"] == scope]
        assert seqs == sorted(set(seqs))


@pytest.mark.parametrize("change", [
    {"revision": 0}, {"revision": True}, {"producer_id": "bad"},
    {"activities": [{"scope": None, "activity_code": "idle", "count": 0}]},
    {"activities": [{"scope": "thread", "activity_code": "speaking", "count": 1}]},
    {"activities": [{"scope": "thread", "activity_code": "thinking", "count": 0}]},
    {"activities": [{"scope": "thread", "activity_code": "error", "count": 1}]},
    {"activities": [{"scope": "thread", "activity_code": "idle", "count": False}]},
    {"activities": [{"scope": "thread", "activity_code": "thinking", "count": 1, "text": "PRIVATE-NESTED"}]},
    {"activities": [{"scope": "thread", "activity_code": "thinking", "count": 1}] * 2},
    {"activities": [{"scope": "x" * 257, "activity_code": "idle", "count": 0}]},
])
def test_invalid_snapshot_does_not_reflect_private_input_or_mutate_state(state, change):
    with client() as c:
        response = c.post(PATH + "/snapshot", json={**body(), **change}, headers=AUTH)
    assert response.status_code == 422
    assert response.json() == {"detail": {"code": "invalid_activity_snapshot"}}
    assert not state.snapshot()["available"]


def test_auth_refuses_missing_wrong_and_underprivileged_credentials(state):
    table = CredentialTable.from_config({
        "reader": {"sha256": secret_digest("reader"), "access": ["read"], "contacts": "own"},
        "events": {"sha256": secret_digest("events"), "scopes": ["events:read"], "contacts": "all"},
        "writer": {"sha256": secret_digest("writer"), "scopes": ["turns:write"], "contacts": "all"},
    })
    with client(credentials=table) as c:
        for path, method in [(PATH, "GET"), (PATH + "/snapshot", "POST")]:
            for headers in [{}, {"Authorization": "Bearer wrong"}]:
                assert c.request(method, path, headers=headers).status_code == 401
            assert c.request(method, path, headers={"Authorization": "Bearer reader"}).status_code == 403
        assert c.get(PATH, headers={"Authorization": "Bearer events"}).status_code == 200
        assert c.post(PATH + "/snapshot", json=body(), headers={"Authorization": "Bearer events"}).status_code == 403
        assert c.post(PATH + "/snapshot", json=body(), headers={"Authorization": "Bearer writer"}).status_code == 200
        assert c.get(PATH, headers={"Authorization": "Bearer writer"}).status_code == 403
    with client(key=None) as c:
        assert c.get(PATH).status_code == 403
        assert c.post(PATH + "/snapshot", json=body()).status_code == 403


def test_pruned_journal_receipt_acknowledges_pending_without_republishing_content(state, monkeypatch):
    original = host.broadcast_event
    emitted = []
    def crash(frame):
        published = original(frame)
        emitted.append(published)
        raise OSError("crash before pending ack")
    monkeypatch.setattr(host, "broadcast_event", crash)
    with pytest.raises(OSError):
        apply(state, body())
    monkeypatch.setenv("COLONYMIND_EVENT_JOURNAL_RETENTION", "1")
    monkeypatch.setattr(host, "broadcast_event", original)
    original({"type": "other.event", "payload": {"safe": True}})
    snapshot = ActivityStore(TurnIdempotencyLedger(state.ledger.db_path)).snapshot()
    assert snapshot["available"] and snapshot["activities"][0]["activity_code"] == "thinking"
    retained = replay_events(after_seq=0)["events"]
    assert len(retained) == 1 and retained[0]["data"]["scope"] == "thread-1"
    assert snapshot["cursor"] == retained[0]["seq"]


def test_live_frame_uses_the_pinned_payload_and_is_durable_before_delivery(state):
    seen = []
    class Subscriber:
        def publish(self, frame):
            assert replay_events(after_seq=frame["seq"] - 1)["events"][0]["data"] == frame["payload"]
            seen.append(frame)
    subscriber = Subscriber()
    with host._event_broadcast_lock:
        host._event_subscribers.append(subscriber)
    try:
        apply(state, body(code="waiting_approval"))
    finally:
        with host._event_broadcast_lock:
            host._event_subscribers.remove(subscriber)
    assert len(seen) == 2 and {frame["payload"]["scope"] for frame in seen} == {None, "thread-1"}
    assert all(set(frame["payload"]) == {"scope", "activity_code", "text", "emoji", "count", "cta", "seq", "at"}
               for frame in seen)
    assert all(frame["payload"]["cta"] == {"kind": "open_approvals"} for frame in seen)


@pytest.mark.parametrize("operation", ["ingest", "snapshot"])
def test_journal_stall_never_locks_the_turn_ledger(state, monkeypatch, operation):
    import sqlite3
    import threading
    original = host.broadcast_event
    if operation == "snapshot":
        monkeypatch.setattr(host, "broadcast_event", lambda _: None)
        with pytest.raises(OSError):
            apply(state, body())
    entered, release = threading.Event(), threading.Event()
    def paused(frame):
        entered.set()
        assert release.wait(5)
        return original(frame)
    monkeypatch.setattr(host, "broadcast_event", paused)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(lambda: apply(state, body()) if operation == "ingest" else state.snapshot())
        try:
            assert entered.wait(3)
            with sqlite3.connect(state.ledger.db_path, timeout=.05) as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("CREATE TABLE IF NOT EXISTS isolation_probe (value INTEGER)")
                conn.execute("INSERT INTO isolation_probe VALUES (1)")
        finally:
            release.set()
        future.result(timeout=3)


def test_publication_outage_coalesces_one_latest_record_per_scope(state, monkeypatch):
    monkeypatch.setattr(host, "broadcast_event", lambda _: None)
    for revision in range(1, 101):
        with pytest.raises(OSError):
            apply(state, body(revision, "thinking" if revision % 2 else "waiting_approval"))
    with closing(state.ledger._connect()) as conn:
        rows = conn.execute("SELECT * FROM activity_publications").fetchall()
        assert len(rows) == 2
        assert all(json.loads(row["payload_json"])["seq"] == 100 for row in rows)
        assert conn.execute("SELECT COUNT(*) FROM activity_states").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM activity_producers").fetchone()[0] == 1


def test_idle_history_and_pending_storage_are_capped_with_monotone_reintroduction(state, monkeypatch):
    import colonymind.events.activity as implementation
    monkeypatch.setattr(implementation, "MAX_SCOPES", 4)
    monkeypatch.setattr(host, "broadcast_event", lambda _: None)
    first_seq = None
    for revision in range(1, 101):
        with pytest.raises(OSError):
            apply(state, body(revision, scope=f"scope-{revision}"))
        with closing(state.ledger._connect()) as conn:
            assert conn.execute("SELECT COUNT(*) FROM activity_states").fetchone()[0] <= 5
            assert conn.execute("SELECT COUNT(*) FROM activity_publications").fetchone()[0] <= 5
            if first_seq is None:
                first_seq = json.loads(conn.execute("SELECT payload_json FROM activity_states WHERE scope='scope-1'").fetchone()[0])["seq"]
    with pytest.raises(OSError):
        apply(state, body(101, scope="scope-1"))
    with closing(state.ledger._connect()) as conn:
        restored = json.loads(conn.execute("SELECT payload_json FROM activity_states WHERE scope='scope-1'").fetchone()[0])
        assert restored["seq"] > first_seq


def test_legacy_append_only_backlog_migrates_to_latest_per_scope(state):
    ledger = TurnIdempotencyLedger(state.ledger.db_path.with_name("legacy.db"))
    with closing(ledger._connect()) as conn, conn:
        conn.execute("CREATE TABLE activity_publications (id INTEGER PRIMARY KEY AUTOINCREMENT, payload_json TEXT NOT NULL)")
        conn.execute("CREATE TABLE activity_states (scope TEXT PRIMARY KEY, seq INTEGER NOT NULL, payload_json TEXT NOT NULL)")
        for seq in range(1, 101):
            payload = {"scope": "legacy", "seq": seq, "activity_code": "thinking", "count": 1}
            conn.execute("INSERT INTO activity_publications(payload_json) VALUES (?)", (json.dumps(payload),))
        conn.execute("INSERT INTO activity_states VALUES ('legacy',100,?)", (json.dumps(payload),))
    ActivityStore(ledger)
    with closing(ledger._connect()) as conn:
        rows = conn.execute("SELECT * FROM activity_publications").fetchall()
        assert len(rows) == 1 and json.loads(rows[0]["payload_json"])["seq"] == 100
