"""Current agent activity and publication intents in the existing turn ledger.

Operational labels only. This is an observation, never an execution authority.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
import hashlib
import fcntl
import os
import json
from functools import lru_cache
import threading
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from colonymind.util.temporal import now_utc

MAX_SCOPES, MAX_PRODUCERS = 4096, 64


@lru_cache(maxsize=128)
def _activity_lock(path):
    """Serialize only activity work, including publication and snapshot cursors."""
    return threading.RLock()


CODES = ("idle", "thinking", "tool_running", "waiting_approval", "error")
PRIORITY = {"idle": 0, "error": 1, "thinking": 2, "tool_running": 3, "waiting_approval": 4}
LABELS = {"idle": ("Ready", ""), "thinking": ("Thinking", "💭"),
          "tool_running": ("Using a tool", "🛠️"),
          "waiting_approval": ("Waiting for approval", "✋"), "error": ("Run failed", "⚠️")}


def encoded(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    scope: str = Field(min_length=1, max_length=256)
    activity_code: Literal["idle", "thinking", "tool_running", "waiting_approval", "error"]
    count: int = Field(ge=0, le=1000000)

    @model_validator(mode="after")
    def count_matches_state(self):
        if (self.activity_code in {"idle", "error"}) != (self.count == 0):
            raise ValueError("invalid_activity_count")
        return self


class ProducerSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    producer_id: str = Field(pattern=r"^hermes-[a-f0-9]{64}$")
    revision: int = Field(ge=1, le=9223372036854775807)
    activities: list[Observation] = Field(max_length=4096)

    @model_validator(mode="after")
    def unique_scopes(self):
        if len({row.scope for row in self.activities}) != len(self.activities):
            raise ValueError("duplicate_activity_scope")
        return self


def capabilities():
    return {"activity_codes": list(CODES), "event_type": "agent.status",
            "snapshot_route": "/v1/host/activity", "disconnect_clears_busy": "consumer",
            "text_expiry_seconds": None, "extensions": [
                {"id": "desktop", "supported": False, "read_route": None},
                {"id": "mailbox", "supported": True, "read_route": "/v1/agent-mail/mailboxes"},
                {"id": "hands", "supported": True, "read_route": "/v1/hands/"}]}


class ActivityStore:
    def __init__(self, ledger):
        self.ledger = ledger
        self._lock = _activity_lock(str(ledger.db_path.resolve()))
        with self._serialized(), closing(ledger._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("""CREATE TABLE IF NOT EXISTS activity_producers (
                producer_id TEXT PRIMARY KEY, revision INTEGER NOT NULL,
                digest TEXT NOT NULL, activities_json TEXT NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS activity_states (
                scope TEXT PRIMARY KEY, seq INTEGER NOT NULL, payload_json TEXT NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS activity_publications (
                id INTEGER PRIMARY KEY, scope TEXT NOT NULL UNIQUE, payload_json TEXT NOT NULL)""")
            # Upgrade the former append-only backlog in place. Retain only the
            # newest publication for each scope before adding the unique index.
            if "scope" not in {r[1] for r in conn.execute("PRAGMA table_info(activity_publications)")}:
                conn.execute("ALTER TABLE activity_publications ADD COLUMN scope TEXT")
                latest = {}
                for row in conn.execute("SELECT * FROM activity_publications ORDER BY id"):
                    latest[json.loads(row["payload_json"])["scope"] or ""] = row["id"]
                for scope, ident in latest.items():
                    conn.execute("UPDATE activity_publications SET scope=? WHERE id=?", (scope, ident))
                conn.execute("DELETE FROM activity_publications WHERE scope IS NULL")
                conn.execute("CREATE UNIQUE INDEX activity_publication_scope ON activity_publications(scope)")
            conn.execute("CREATE TABLE IF NOT EXISTS activity_sequence (id INTEGER PRIMARY KEY, value INTEGER NOT NULL)")
            conn.execute("INSERT OR IGNORE INTO activity_sequence SELECT 1, COALESCE(MAX(seq),0) FROM activity_states")
            self._prune(conn)

    @contextmanager
    def _serialized(self):
        # Coordinate activity across sidecar threads/processes independently of
        # SQLite's turn locks. The file contains no state or event data.
        with self._lock:
            path = self.ledger.db_path.with_suffix(".activity.lock")
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                os.close(fd)

    @staticmethod
    def _prune(conn):
        # Idle scopes are a bounded reconnect cache, not a permanent history.
        # Never evict active/error state to admit another scope.
        rows = conn.execute("SELECT scope,payload_json FROM activity_states WHERE scope!='' ORDER BY seq DESC,scope").fetchall()
        excess = len(rows) - MAX_SCOPES
        for row in reversed(rows):
            if excess <= 0:
                break
            if json.loads(row["payload_json"])["activity_code"] == "idle":
                conn.execute("DELETE FROM activity_states WHERE scope=?", (row["scope"],))
                conn.execute("DELETE FROM activity_publications WHERE scope=?", (row["scope"],))
                excess -= 1
        if excess > 0:
            raise OSError("activity_capacity")
        conn.execute("DELETE FROM activity_publications WHERE scope NOT IN (SELECT scope FROM activity_states)")

    @staticmethod
    def _combined(rows):
        return {"activity_code": max((r["activity_code"] for r in rows),
                                     key=PRIORITY.get, default="idle"),
                "count": sum(r["count"] for r in rows)}

    def ingest(self, snapshot: ProducerSnapshot):
        with self._serialized():
            return self._ingest(snapshot)

    def _ingest(self, snapshot: ProducerSnapshot):
        value = snapshot.model_dump()
        digest = hashlib.sha256(encoded(value).encode()).hexdigest()
        with closing(self.ledger._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            old = conn.execute("SELECT * FROM activity_producers WHERE producer_id=?",
                               (snapshot.producer_id,)).fetchone()
            duplicate = old is not None and old["revision"] == snapshot.revision
            superseded = old is not None and old["revision"] > snapshot.revision
            if duplicate and old["digest"] != digest:
                raise ValueError("activity_revision_conflict")
            revision = old["revision"] if superseded else snapshot.revision
            if not duplicate and not superseded:
                if old is None and conn.execute("SELECT COUNT(*) FROM activity_producers").fetchone()[0] >= MAX_PRODUCERS:
                    raise OSError("activity_capacity")
                conn.execute("INSERT OR REPLACE INTO activity_producers VALUES (?,?,?,?)",
                             (snapshot.producer_id, snapshot.revision, digest, encoded(value["activities"])))
                grouped = {}
                for producer in conn.execute("SELECT activities_json FROM activity_producers"):
                    for row in json.loads(producer[0]):
                        grouped.setdefault(row["scope"], []).append(row)
                current = {scope: self._combined(rows) for scope, rows in grouped.items()}
                current[""] = self._combined(list(current.values()))
                previous = {r["scope"]: r for r in conn.execute("SELECT * FROM activity_states")}
                at = now_utc().isoformat()
                highwater = conn.execute("SELECT value FROM activity_sequence WHERE id=1").fetchone()[0]
                for scope in sorted(set(previous) | set(current)):
                    state = current.get(scope, {"activity_code": "idle", "count": 0})
                    before = previous.get(scope)
                    if before and scope and all(json.loads(before["payload_json"])[k] == v
                                                for k, v in state.items()):
                        continue
                    code = state["activity_code"]
                    payload = {"scope": scope or None, **state, "text": LABELS[code][0],
                               "emoji": LABELS[code][1],
                               "cta": {"kind": "open_approvals"} if code == "waiting_approval" else None,
                               "seq": before["seq"] + 1 if before else highwater + 1, "at": at}
                    data = encoded(payload)
                    conn.execute("INSERT OR REPLACE INTO activity_states VALUES (?,?,?)",
                                 (scope, payload["seq"], data))
                    conn.execute("INSERT OR REPLACE INTO activity_publications(scope,payload_json) VALUES (?,?)", (scope, data))
                conn.execute("UPDATE activity_sequence SET value=(SELECT MAX(seq) FROM activity_states) WHERE id=1")
                self._prune(conn)
        # State and publication intent are committed before any live frame.
        self._flush()
        return {"accepted": not superseded, "duplicate": duplicate,
                "superseded": superseded, "revision": revision}

    def _flush(self):
        from colonymind.api.routers.host import broadcast_event
        # The activity lock orders publication, but NO SQLite transaction or
        # connection spans journal I/O. Turn ingestion can write throughout.
        with closing(self.ledger._connect()) as conn:
            pending = conn.execute("SELECT * FROM activity_publications ORDER BY scope").fetchall()
        for row in pending:
            payload = json.loads(row["payload_json"])
            scope_key = hashlib.sha256(encoded(payload["scope"]).encode()).hexdigest()
            published = broadcast_event({"type": "agent.status", "payload": payload,
                "occurred_at": payload["at"], "event_key": f"activity:{scope_key}:{payload['seq']}"})
            if not isinstance(published, dict):
                raise OSError("activity_publication_unavailable")
            with closing(self.ledger._connect()) as conn, conn:
                conn.execute("DELETE FROM activity_publications WHERE scope=? AND payload_json=?",
                             (row["scope"], row["payload_json"]))

    def snapshot(self):
        from colonymind.events.journal import current_sequence
        # Exclude activity writers through flush/read/cursor using only their
        # own lock; journal publication never owns the turn-ledger write lock.
        with self._serialized():
            self._flush()
            with closing(self.ledger._connect()) as conn:
                activities = [json.loads(r[0]) for r in conn.execute(
                    "SELECT payload_json FROM activity_states ORDER BY scope")]
                available = conn.execute("SELECT 1 FROM activity_producers LIMIT 1").fetchone() is not None
            return {"schema_version": 1, "available": available, "activities": activities,
                    "cursor": current_sequence(), "capabilities": capabilities()}
