"""SQLite schema for the durable task controller.

It creates every controller table, plus the replay ledger and the session
bindings. There are no controller transitions or task executions in this
module. Schema creation is atomic and repeatable.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from pathlib import Path
from enum import Enum

import aiosqlite

from .types import (
    DerivationState,
    EvidenceLevel,
    HandActivity,
    HandState,
    Kind,
    ObligationState,
    PermitState,
    Priority,
    ReservationState,
    Risk,
    TaskState,
)


SCHEMA_VERSION = 5
# In the instance directory beside instinct.db and usage.db (COLONYMIND_STATE_DIR is the home).
DB_FILE = "hands.db"
BUSY_TIMEOUT_MS = 5000  # Assumption: wait up to five seconds for a local writer.


def _choices(enum: type[Enum]) -> str:
    """SQL literals from fixed internal enums, never caller-supplied values."""
    return ", ".join(f"'{item.value}'" for item in enum)


# Assumptions inherited from the design: at most two lock waiters. Its event
# payload limit of 2 KB is interpreted as 2048 UTF-8 bytes. Event string values
# are bounded recursively, including strings inside nested arrays and objects.
_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY NOT NULL,
    origin TEXT NOT NULL CHECK (origin IN ('owner_turn', 'own', 'mind', 'voice', 'harness')),
    origin_ref TEXT NOT NULL,
    priority TEXT NOT NULL DEFAULT 'owner' CHECK (priority IN ({_choices(Priority)})),
    kind TEXT NOT NULL CHECK (kind IN ({_choices(Kind)})),
    risk TEXT NOT NULL DEFAULT 'normal' CHECK (risk IN ({_choices(Risk)})),
    needs TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(needs) AND json_type(needs) = 'array'),
    intent_version INTEGER NOT NULL DEFAULT 1 CHECK (intent_version >= 1),
    intents TEXT NOT NULL DEFAULT '{{}}' CHECK (json_valid(intents) AND json_type(intents) = 'object'),
    resources TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(resources) AND json_type(resources) = 'array'),
    depends_on TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(depends_on) AND json_type(depends_on) = 'array'),
    state TEXT NOT NULL DEFAULT 'open' CHECK (state IN ({_choices(TaskState)})),
    budget TEXT NOT NULL DEFAULT '{{}}' CHECK (json_valid(budget) AND json_type(budget) = 'object'),
    usage TEXT NOT NULL DEFAULT '{{}}' CHECK (json_valid(usage) AND json_type(usage) = 'object'),
    failures_in_row INTEGER NOT NULL DEFAULT 0 CHECK (failures_in_row >= 0),
    cooldown_until TEXT,
    obligation_id TEXT REFERENCES obligations(id) ON DELETE SET NULL,
    owner_spoke_since TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    closed_at TEXT
);

CREATE TABLE IF NOT EXISTS hands (
    id TEXT PRIMARY KEY NOT NULL,
    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    attempt INTEGER NOT NULL CHECK (attempt >= 1),
    intent_version INTEGER NOT NULL CHECK (intent_version >= 1),
    name TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ({_choices(Kind)})),
    effort INTEGER NOT NULL CHECK (effort IN (0, 1, 2, 3)),
    role TEXT,
    contract TEXT CHECK (contract IS NULL OR (json_valid(contract) AND json_type(contract) = 'object')),
    families_used TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(families_used) AND json_type(families_used) = 'array'),
    model TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(model) AND json_type(model) = 'array'),
    origin_ref TEXT,
    priority TEXT NOT NULL DEFAULT 'owner' CHECK (priority IN ({_choices(Priority)})),
    parent_session TEXT NOT NULL,
    child_session TEXT,
    executor TEXT NOT NULL CHECK (executor IN ('hermes_delegation', 'hermes_kanban', 'harness')),
    executor_ref TEXT,
    epoch TEXT NOT NULL CHECK (json_valid(epoch) AND json_type(epoch) = 'object'),
    state TEXT NOT NULL DEFAULT 'planned' CHECK (state IN ({_choices(HandState)})),
    activity TEXT CHECK (activity IS NULL OR activity IN ({_choices(HandActivity)})),
    activity_since TEXT,
    outcome TEXT,
    evidence TEXT CHECK (evidence IS NULL OR (json_valid(evidence) AND json_type(evidence) = 'object')),
    "check" TEXT NOT NULL DEFAULT '[]' CHECK (json_valid("check") AND json_type("check") = 'array'),
    branch TEXT,
    continues TEXT REFERENCES hands(id) ON DELETE SET NULL,
    escalated_from TEXT REFERENCES hands(id) ON DELETE SET NULL,
    budgets TEXT NOT NULL DEFAULT '{{}}' CHECK (json_valid(budgets) AND json_type(budgets) = 'object'),
    usage TEXT NOT NULL DEFAULT '{{}}' CHECK (json_valid(usage) AND json_type(usage) = 'object'),
    decisions TEXT NOT NULL DEFAULT '{{}}' CHECK (json_valid(decisions) AND json_type(decisions) = 'object'),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    started_at TEXT,
    ended_at TEXT,
    last_event_at TEXT,
    UNIQUE (task_id, attempt)
);

CREATE TABLE IF NOT EXISTS work_refresh_tasks (
    task_id TEXT PRIMARY KEY NOT NULL, generation INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS work_refresh_origins (
    origin_ref TEXT PRIMARY KEY NOT NULL, generation INTEGER NOT NULL DEFAULT 1
);
CREATE TRIGGER IF NOT EXISTS work_refresh_hand_insert AFTER INSERT ON hands
BEGIN
    INSERT INTO work_refresh_tasks(task_id) VALUES (NEW.task_id)
    ON CONFLICT(task_id) DO UPDATE SET generation=generation+1;
END;
CREATE TRIGGER IF NOT EXISTS work_refresh_hand_update
AFTER UPDATE OF state,evidence,intent_version,decisions ON hands
BEGIN
    INSERT INTO work_refresh_tasks(task_id) VALUES (NEW.task_id)
    ON CONFLICT(task_id) DO UPDATE SET generation=generation+1;
END;
CREATE TRIGGER IF NOT EXISTS work_refresh_task_update
AFTER UPDATE OF intent_version,origin,origin_ref ON tasks
BEGIN
    INSERT INTO work_refresh_tasks(task_id) VALUES (NEW.id)
    ON CONFLICT(task_id) DO UPDATE SET generation=generation+1;
END;

CREATE TABLE IF NOT EXISTS reservations (
    hand_id TEXT PRIMARY KEY NOT NULL REFERENCES hands(id) ON DELETE CASCADE,
    token TEXT NOT NULL UNIQUE,
    family TEXT NOT NULL,
    "class" TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'held' CHECK (state IN ({_choices(ReservationState)}))
);

CREATE TABLE IF NOT EXISTS locks (
    resource TEXT PRIMARY KEY NOT NULL,
    holder_hand TEXT NOT NULL REFERENCES hands(id) ON DELETE CASCADE,
    since TEXT NOT NULL,
    waiters TEXT NOT NULL DEFAULT '[]'
        CHECK (json_valid(waiters) AND json_type(waiters) = 'array' AND json_array_length(waiters) <= 2)
);

CREATE TABLE IF NOT EXISTS effects (
    hand_id TEXT NOT NULL REFERENCES hands(id) ON DELETE CASCADE,
    key TEXT PRIMARY KEY NOT NULL,
    tool TEXT NOT NULL,
    call_hash TEXT NOT NULL,
    workdir TEXT NOT NULL,
    intent_at TEXT NOT NULL,
    outcome TEXT NOT NULL DEFAULT 'unknown' CHECK (outcome IN ('ok', 'error', 'unknown')),
    exit_code INTEGER,
    result_hash TEXT,
    done_at TEXT
);

CREATE TABLE IF NOT EXISTS obligations (
    id TEXT PRIMARY KEY NOT NULL,
    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    owner_ref TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'open' CHECK (state IN ({_choices(ObligationState)})),
    result_ready_at TEXT,
    reviewed_at TEXT,
    accepted_at TEXT,
    communicated_at TEXT,
    communicated_ref TEXT,
    delivered_at TEXT,
    delivery_ref TEXT,
    delivery TEXT NOT NULL DEFAULT 'unknown' CHECK (delivery IN ('unknown', 'receipt')),
    closed_reason TEXT
);

CREATE TABLE IF NOT EXISTS permits (
    id TEXT PRIMARY KEY NOT NULL,
    ask_id TEXT NOT NULL,
    hand_id TEXT NOT NULL REFERENCES hands(id) ON DELETE CASCADE,
    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    call TEXT NOT NULL CHECK (json_valid(call) AND json_type(call) = 'object'),
    call_digest TEXT NOT NULL,
    workdir TEXT NOT NULL,
    targets TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(targets) AND json_type(targets) = 'array'),
    owner_answer_ref TEXT,
    granted_at TEXT,
    expires_at TEXT,
    replay TEXT NOT NULL DEFAULT 'once' CHECK (replay = 'once'),
    state TEXT NOT NULL DEFAULT 'waiting' CHECK (state IN ({_choices(PermitState)}))
);

CREATE TABLE IF NOT EXISTS derivations (
    derived_kind TEXT NOT NULL CHECK (derived_kind IN ('autobiography', 'lesson', 'relay')),
    derived_id TEXT NOT NULL,
    hand_id TEXT NOT NULL REFERENCES hands(id) ON DELETE CASCADE,
    evidence_level TEXT NOT NULL CHECK (evidence_level IN ({_choices(EvidenceLevel)})),
    evidence_ref TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'live' CHECK (state IN ({_choices(DerivationState)})),
    PRIMARY KEY (derived_kind, derived_id, hand_id)
);

CREATE TABLE IF NOT EXISTS events (
    hand_id TEXT NOT NULL REFERENCES hands(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL CHECK (seq >= 0),
    at TEXT NOT NULL,
    type TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{{}}'
        CHECK (json_valid(payload) AND json_type(payload) = 'object' AND length(CAST(payload AS BLOB)) <= 2048),
    source TEXT NOT NULL,
    key TEXT NOT NULL UNIQUE,
    PRIMARY KEY (hand_id, seq)
);

CREATE TRIGGER IF NOT EXISTS hands_events_payload_insert
BEFORE INSERT ON events
WHEN json_valid(NEW.payload) AND EXISTS (
    SELECT 1 FROM json_tree(NEW.payload) WHERE type = 'text' AND length(atom) > 280
)
BEGIN
    SELECT RAISE(ABORT, 'event text exceeds 280 characters');
END;

CREATE TRIGGER IF NOT EXISTS hands_events_payload_update
BEFORE UPDATE OF payload ON events
WHEN json_valid(NEW.payload) AND EXISTS (
    SELECT 1 FROM json_tree(NEW.payload) WHERE type = 'text' AND length(atom) > 280
)
BEGIN
    SELECT RAISE(ABORT, 'event text exceeds 280 characters');
END;

CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY NOT NULL,
    hand_id TEXT NOT NULL REFERENCES hands(id) ON DELETE CASCADE,
    asked_at TEXT NOT NULL,
    text TEXT CHECK (text IS NULL OR length(text) <= 500),
    assumption TEXT,
    blocking INTEGER NOT NULL DEFAULT 0 CHECK (blocking IN (0, 1)),
    state TEXT NOT NULL DEFAULT 'open',
    answer TEXT,
    answered_by TEXT,
    answered_at TEXT
);

CREATE TABLE IF NOT EXISTS notices (
    id TEXT PRIMARY KEY NOT NULL,
    budget TEXT NOT NULL CHECK (budget IN ('routine', 'actionable')),
    line TEXT NOT NULL,
    queued_at TEXT NOT NULL,
    sent_at TEXT,
    platform_message_id TEXT,
    hand_id TEXT NOT NULL REFERENCES hands(id) ON DELETE CASCADE,
    attempt INTEGER NOT NULL CHECK (attempt >= 1)
);

CREATE TABLE IF NOT EXISTS idempotency (
    key TEXT PRIMARY KEY NOT NULL CHECK (length(trim(key)) > 0),
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    response TEXT NOT NULL CHECK (json_valid(response)),
    status_code INTEGER NOT NULL CHECK (status_code BETWEEN 100 AND 599),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hand_sessions (
    session_id TEXT PRIMARY KEY NOT NULL,
    hand_id TEXT NOT NULL REFERENCES hands(id) ON DELETE CASCADE,
    parent_session TEXT NOT NULL,
    epoch TEXT NOT NULL CHECK (json_valid(epoch) AND json_type(epoch) = 'object'),
    registered_at TEXT NOT NULL,
    retired_at TEXT
);

-- Minimal authority identities outlive operational rows; they carry no task or result content.
CREATE TABLE IF NOT EXISTS hand_session_tombstones (
    session_id TEXT PRIMARY KEY NOT NULL,
    hand_id TEXT NOT NULL,
    parent_session TEXT NOT NULL,
    epoch TEXT NOT NULL CHECK (json_valid(epoch) AND json_type(epoch) = 'object')
);

-- One bounded maintenance summary, independent of the rows that retention removes.
CREATE TABLE IF NOT EXISTS maintenance (
    id TEXT PRIMARY KEY NOT NULL,
    at TEXT NOT NULL,
    payload TEXT NOT NULL CHECK (json_valid(payload) AND json_type(payload) = 'object')
);

CREATE INDEX IF NOT EXISTS idx_tasks_admission ON tasks(state, origin, priority);
CREATE INDEX IF NOT EXISTS idx_hands_admission ON hands(state, priority);
CREATE INDEX IF NOT EXISTS idx_hands_parent ON hands(parent_session, state);
CREATE INDEX IF NOT EXISTS idx_hands_child ON hands(child_session);
CREATE INDEX IF NOT EXISTS idx_reservations_expiry ON reservations(state, expires_at);
CREATE INDEX IF NOT EXISTS idx_reservations_family ON reservations(family, "class", state);
CREATE INDEX IF NOT EXISTS idx_effects_open ON effects(hand_id, done_at);
CREATE INDEX IF NOT EXISTS idx_obligations_open ON obligations(state);
-- Not unique per task: a revision can supersede one obligation and open another; tasks.obligation_id is current.
CREATE INDEX IF NOT EXISTS idx_obligations_task ON obligations(task_id, state);
CREATE INDEX IF NOT EXISTS idx_permits_call ON permits(task_id, call_digest, state);
CREATE INDEX IF NOT EXISTS idx_derivations_hand ON derivations(hand_id, state);
CREATE INDEX IF NOT EXISTS idx_events_retention ON events(at);
CREATE INDEX IF NOT EXISTS idx_questions_hand ON questions(hand_id, state);
CREATE INDEX IF NOT EXISTS idx_notices_pending ON notices(sent_at, queued_at);
CREATE INDEX IF NOT EXISTS idx_idempotency_retention ON idempotency(created_at);
CREATE INDEX IF NOT EXISTS idx_hand_sessions_hand ON hand_sessions(hand_id);
"""


class HandsStore:
    """Open a private controller database and apply the initial schema.

    Assumptions for the initial seam: references use TEXT, structured payloads
    use JSON, creation timestamps are UTC, and a question starts ``open``. The
    supporting replay and session tables use the shapes in ``types.py``. A
    refused hand can have no resolved role. The contract JSON holds the planned
    provider and version plus the actual route history in ``used``.
    ``:memory:`` is supported for callers that do not need file durability.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path) if str(path) == ":memory:" else str(Path(path).expanduser())
        self._db: aiosqlite.Connection | None = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("HandsStore is not connected")
        return self._db

    def _secure_files(self) -> None:
        if self.path == ":memory:":
            return
        for suffix in ("", "-wal", "-shm", "-journal"):
            path = Path(self.path + suffix)
            try:
                path.chmod(0o600)
            except FileNotFoundError:
                pass

    async def connect(self) -> None:
        """Connect once; a failed setup closes the connection before raising."""
        if self._db is not None:
            return
        if self.path != ":memory:":
            path = Path(self.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                os.fchmod(descriptor, 0o600)
            finally:
                os.close(descriptor)
            self._secure_files()
        db = await aiosqlite.connect(self.path)
        self._db = db
        try:
            db.row_factory = aiosqlite.Row
            await db.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
            # Concurrent startup/migration can make this PRAGMA return BUSY
            # immediately even with busy_timeout. Retry within the same bound.
            loop = asyncio.get_running_loop()
            deadline = loop.time() + BUSY_TIMEOUT_MS / 1000
            while True:
                try:
                    async with db.execute("PRAGMA journal_mode=WAL") as cursor:
                        await cursor.fetchone()
                    break
                except sqlite3.OperationalError as error:
                    code = getattr(error, "sqlite_errorcode", 0) & 0xff
                    if code not in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED} or loop.time() >= deadline:
                        raise
                    await asyncio.sleep(0.01)
            await db.execute("PRAGMA foreign_keys=ON")
            await self.migrate()
            self._secure_files()
        except BaseException:
            self._db = None
            await db.close()
            raise

    async def migrate(self) -> None:
        """Create all tables atomically; repeated calls preserve the same schema."""
        db = self.db
        try:
            await db.execute("BEGIN IMMEDIATE")
            async with db.execute("PRAGMA user_version") as cursor:
                row = await cursor.fetchone()
            if row[0] > SCHEMA_VERSION:
                raise RuntimeError("HandsStore schema is newer than this implementation")
            from .definitions_schema import SCHEMA as definitions_schema, QUEUE_SCHEMA, OWNER_QUEUE_SCHEMA, ELIGIBLE_QUEUE_SCHEMA
            # executescript implicitly commits before executing, dropping the lock.
            statement = ""
            schema = (_SCHEMA + (definitions_schema if row[0] < 2 else "")
                      + (QUEUE_SCHEMA if row[0] < 3 else "") + (OWNER_QUEUE_SCHEMA if row[0] < 4 else "")
                      + (ELIGIBLE_QUEUE_SCHEMA if row[0] < 5 else ""))
            for line in schema.splitlines(True):
                statement += line
                if sqlite3.complete_statement(statement):
                    await db.execute(statement)
                    statement = ""
            await db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            await db.commit()
        except BaseException:
            await db.rollback()
            raise
        self._secure_files()

    async def close(self) -> None:
        """Close the connection; repeated closes are harmless."""
        db, self._db = self._db, None
        if db is not None:
            await db.close()
