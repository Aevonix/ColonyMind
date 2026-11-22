"""Initiative store for multi-agent ColonyMind.

Provides:
- InitiativeStore: SQLite persistence for initiatives
- Assignment history tracking
- Dead letter queue
- Timeout and expiry checks
"""


from colonymind.memory.producers import ReadableConnection
import base64
import hashlib
import json
import logging
import shutil
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .models import AssignmentHistory, InitiativeStatus, StoredInitiative, MIND_TERMINAL_STATUSES, OWNER_KINDS
from colonymind import recorder
from colonymind.util.temporal import now_utc

logger = logging.getLogger(__name__)

# One recorder ``card`` line per piece of the agent's own work: the line's source per intention kind, and how a
# row's decision and final status read as the line's choice and result (any other ending came to nothing).
CARD_SOURCES = {"task": "mind.task"}
CARD_CHOICES = {"act": "act", "ask": "ask_owner", "drop": "ignore"}
CARD_RESULTS = {"done": "done", "failed": "failed", "uncertain": "failed"}


@dataclass(frozen=True)
class SettleGuard:
    """A transition that settles a row only while it is still in one of ``expect`` (compare-and-set in SQLite).

    The status change, its history row and ``writes(status, at)`` (more statements: an explicit answer's
    idempotency record) commit in ONE SQLite transaction, or none of them does. Two answers racing on one ask (an
    RCS reply and an app answer, in this process or another) therefore settle it once: the first commit wins and
    the second finds the row moved on (``TransitionRefused``)."""
    expect: Tuple[str, ...] = ("asked",)
    writes: Optional[Callable[[str, datetime], Sequence[Tuple[str, Sequence[Any]]]]] = None


class TransitionRefused(RuntimeError):
    """A guarded transition found its row no longer in the expected status: nothing was written. ``current`` is the
    row as it is now (None when it is gone)."""

    def __init__(self, initiative_id: str, current: Optional[StoredInitiative]):
        self.initiative_id = initiative_id
        self.current = current
        super().__init__(f"{initiative_id} is already {current.status if current is not None else 'gone'}")


# Maximum pending initiatives before rejecting new ones
MAX_PENDING_INITIATIVES = 1000
MAX_IDEA_STATE_IDS = 100
IDEA_COLUMNS = {"presentation": "TEXT", "items": "TEXT"}

# Intention and audit columns (architecture 5.2). The table is the intention
# store and the only audit log; ``colonymind upgrade`` and every store open add
# the missing ones in place.
MIND_COLUMNS: Dict[str, str] = {
    "cls": "TEXT", "decision": "TEXT", "decision_reason": "TEXT", "drive": "TEXT",
    "kind": "TEXT", "ask_code": "TEXT", "invalidates_if": "TEXT", "success_check": "TEXT",
    "expectation_id": "TEXT", "parent_goal_id": "TEXT", "hermes_kind": "TEXT",
    "hermes_ref": "TEXT", "outcome": "TEXT", "verified": "TEXT", "verdict": "TEXT",
    "lesson_ids": "TEXT", "cost_tokens": "INTEGER DEFAULT 0", "due_at": "TIMESTAMP",
}


def missing_mind_columns(conn: sqlite3.Connection) -> List[str]:
    """The intention columns an existing ``initiatives`` table still lacks."""
    try:
        present = {row[1] for row in conn.execute("PRAGMA table_info(initiatives)").fetchall()}
    except sqlite3.DatabaseError:
        return []
    if not present:
        return []
    return [name for name in MIND_COLUMNS if name not in present]


#: SQLite's primary result codes for a damaged file; only these are recovered at open.
_SQLITE_CORRUPT, _SQLITE_NOTADB = 11, 26


def _damaged(exc: sqlite3.DatabaseError) -> bool:
    """Whether SQLite reported the file itself as damaged (not locked, busy or unreadable)."""
    code = getattr(exc, "sqlite_errorcode", None)
    return code is not None and code & 0xFF in {_SQLITE_CORRUPT, _SQLITE_NOTADB}


def get_state_dir() -> Path:
    """Get ColonyMind state directory."""
    import os
    state_dir = os.environ.get("COLONYMIND_STATE_DIR")
    if state_dir:
        return Path(state_dir)
    return Path.home() / ".colonymind" / "data"


class InitiativeStore:
    """Manages initiative persistence with SQLite."""

    _db = ReadableConnection()

    def __init__(self, state_dir: Optional[Path] = None):
        self._state_dir = Path(state_dir) if state_dir else get_state_dir()
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._state_dir / "initiatives.db"
        self._backup_path = self._state_dir / "initiatives.db.backup"
        self._dlq_path = self._state_dir / "dead-letter-queue.jsonl"

        self._db = self._init_db()

    def _init_db(self) -> sqlite3.Connection:
        """Open the store; recover only a damaged file, and never delete it.

        A locked, busy or unreadable store (any other error) is raised as it is:
        its rows are intact and the caller retries once the other process lets go.
        A file SQLite reports as damaged (SQLITE_CORRUPT, SQLITE_NOTADB) is renamed
        aside with its WAL and shared-memory files, then the backup is restored when
        one exists, else the store starts empty.
        """
        try:
            return self._connect()
        except sqlite3.DatabaseError as exc:
            if not _damaged(exc):
                raise
            aside = self._rename_aside()
            logger.warning("initiatives.db is damaged (%s); kept as %s", exc, aside.name)
            if self._backup_path.exists():
                from colonymind.backup import _copy_private
                _copy_private(self._backup_path, self._db_path)
                logger.warning("Restored initiatives.db from %s", self._backup_path.name)
            else:
                logger.warning("No initiatives.db backup available, starting empty")
            return self._connect()

    def _rename_aside(self) -> Path:
        """Move the damaged store and its -wal/-shm files to ``initiatives.db.corrupt-<UTC stamp>``."""
        stamp = now_utc().strftime("%Y%m%dT%H%M%S%fZ")
        aside = self._db_path.with_name(f"{self._db_path.name}.corrupt-{stamp}")
        for suffix in ("", "-wal", "-shm"):
            source = self._db_path.with_name(self._db_path.name + suffix)
            if source.exists():
                source.rename(aside.with_name(aside.name + suffix))
        return aside

    def _connect(self) -> sqlite3.Connection:
        """Connect to database with WAL mode."""
        from colonymind.memory import producers
        reader = producers.connection(self._db_path)
        if reader is not None:
            return reader
        # check_same_thread=False allows TestClient to access the DB from
        # a different thread (test thread vs event loop thread).
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        try:
            conn.row_factory = sqlite3.Row

            # WAL mode for better crash recovery
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

            self._create_tables(conn)
        except BaseException:
            conn.close()
            raise
        return conn

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """Create initiatives and history tables."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS initiatives (
                id TEXT PRIMARY KEY,
                dedup_key TEXT UNIQUE,
                type TEXT NOT NULL,
                description TEXT NOT NULL,
                priority REAL DEFAULT 0.5,
                rationale TEXT,
                action_hint TEXT,
                entity_id TEXT,
                
                source_type TEXT,
                source_id TEXT,
                created_by TEXT,
                
                status TEXT DEFAULT 'pending',
                assigned_agent_id TEXT,
                assigned_agent_name TEXT,
                assigned_at TIMESTAMP,
                acknowledged_at TIMESTAMP,
                completed_at TIMESTAMP,
                cancelled_at TIMESTAMP,
                cancelled_by TEXT,
                cancelled_reason TEXT,
                failed_at TIMESTAMP,
                failed_reason TEXT,
                
                attempt_count INTEGER DEFAULT 0,
                max_attempts INTEGER DEFAULT 3,
                timeout_seconds INTEGER DEFAULT 300,
                last_attempt_at TIMESTAMP,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                
                delivery_mode TEXT DEFAULT 'websocket',
                delivery_attempts INTEGER DEFAULT 0,
                last_delivery_at TIMESTAMP,
                delivery_failed_at TIMESTAMP,
                delivery_failed_reason TEXT,
                
                result TEXT,
                result_metadata TEXT DEFAULT '{}',
                
                preferred_agent_id TEXT,
                stale_reason TEXT,
                recovery_reason TEXT,
                job_id TEXT,  -- v0.13.0: linked task-queue job
                context TEXT  -- v0.16.0: situational context snapshot (JSON)
            )
            """
        )

        # Column migrations: v0.13.0 job_id, v0.16.0 context.
        # Old rows keep context NULL — the API serializer maps that to {}.
        try:
            cursor = conn.execute("PRAGMA table_info(initiatives)")
            columns = {row[1] for row in cursor.fetchall()}
            if "job_id" not in columns:
                conn.execute("ALTER TABLE initiatives ADD COLUMN job_id TEXT")
                conn.commit()
                logger.info("Migrated initiatives table: added job_id column")
            if "context" not in columns:
                conn.execute("ALTER TABLE initiatives ADD COLUMN context TEXT")
                conn.commit()
                logger.info("Migrated initiatives table: added context column")
            if "dedup_base" not in columns:
                # v0.21.23: logical (un-bucketed) dedup key for the cross-period
                # in-flight guard. Old rows keep it NULL (they behave as before).
                conn.execute("ALTER TABLE initiatives ADD COLUMN dedup_base TEXT")
                conn.commit()
                logger.info("Migrated initiatives table: added dedup_base column")
            for name in missing_mind_columns(conn):
                conn.execute(f"ALTER TABLE initiatives ADD COLUMN {name} {MIND_COLUMNS[name]}")
                conn.commit()
                logger.info("Migrated initiatives table: added %s column", name)
            for name, column_type in IDEA_COLUMNS.items():
                if name not in columns:
                    conn.execute(f"ALTER TABLE initiatives ADD COLUMN {name} {column_type}")
        except Exception as exc:
            # Two processes opening at once race to the same ALTER; the loser's
            # duplicate-column error is harmless. Anything else is caught below.
            logger.warning("Initiative migration check failed: %s", exc)
        missing = missing_mind_columns(conn)
        if missing:
            raise sqlite3.OperationalError(
                f"initiatives.db still lacks {', '.join(missing)}; another process may hold its write lock")
        present = {row[1] for row in conn.execute("PRAGMA table_info(initiatives)")}
        if set(IDEA_COLUMNS) - present:
            raise sqlite3.OperationalError("initiatives.db still lacks idea content columns")

        # Defense in depth: owner records live in separate tables. This view also
        # excludes owner kinds if a legacy connection predates the migration.
        conn.execute("""CREATE VIEW IF NOT EXISTS generic_initiatives AS
            SELECT * FROM initiatives WHERE kind IS NULL OR kind NOT IN
            ('owner_goal','owner_subgoal','owner_goal_progress')""")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_initiatives_kind ON initiatives(kind, status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_initiatives_ask_code ON initiatives(ask_code)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_initiatives_hermes_ref ON initiatives(hermes_ref)")

        # Indexes
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_initiatives_status ON initiatives(status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_initiatives_assigned ON initiatives(assigned_agent_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_initiatives_dedup ON initiatives(dedup_key)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_initiatives_dedup_base ON initiatives(dedup_base)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_initiatives_priority ON initiatives(priority DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_initiatives_created ON initiatives(created_at DESC)"
        )

        # Assignment history table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assignment_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                initiative_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                agent_name TEXT,
                action TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                details TEXT,
                origin TEXT,
                
                FOREIGN KEY (initiative_id) REFERENCES initiatives(id)
            )
            """
        )
        history_columns = {row[1] for row in conn.execute("PRAGMA table_info(assignment_history)")}
        if "origin" not in history_columns:
            conn.execute("ALTER TABLE assignment_history ADD COLUMN origin TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_initiative ON assignment_history(initiative_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_agent ON assignment_history(agent_id)"
        )

        # Explicit answers to an ask by its id (``POST /v1/mind/permit-asks/{id}/answer``): one row per caller
        # idempotency key, with the digest of the request it was first sent with and the result it got. A settling
        # answer's row commits in the same transaction as the ask's own settlement (``SettleGuard.writes``).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ask_answer_requests (
                scope TEXT NOT NULL,
                key TEXT NOT NULL,
                ask_id TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                response TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (scope, key)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ask_answer_requests_ask ON ask_answer_requests(ask_id)")

        conn.execute("""CREATE INDEX IF NOT EXISTS idx_initiatives_ideas
            ON initiatives(type, priority DESC, created_at DESC, id)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS initiative_engagement (
            receipt_id TEXT PRIMARY KEY, initiative_id TEXT NOT NULL,
            kind TEXT NOT NULL, visible_fraction REAL NOT NULL,
            surface TEXT NOT NULL, anchor TEXT NOT NULL, created_at TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS initiative_shares (
            share_id TEXT PRIMARY KEY, initiative_id TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL, summary TEXT NOT NULL
        )""")
        conn.commit()
        from colonymind.owner_goals.schema import initialize
        initialize(conn)

    # ------------------------------------------------------------------
    # CRUD Operations
    # ------------------------------------------------------------------

    def create(self, *args, **kwargs) -> StoredInitiative:
        """Create an initiative. Back-compat shim returning just the initiative; the
        dispatch loop uses :meth:`create_with_outcome` to learn whether it was newly
        created vs deduped."""
        return self.create_with_outcome(*args, **kwargs)[0]

    def create_with_outcome(
        self,
        type: str,
        description: str,
        priority: float = 0.5,
        rationale: str = "",
        action_hint: Optional[str] = None,
        entity_id: Optional[str] = None,
        dedup_key: Optional[str] = None,
        dedup_base: Optional[str] = None,
        source_type: Optional[str] = None,
        source_id: Optional[str] = None,
        created_by: Optional[str] = None,
        timeout_seconds: int = 300,
        expires_at: Optional[datetime] = None,
        preferred_agent_id: Optional[str] = None,
        job_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        presentation: Optional[Dict[str, Any]] = None,
        items: Optional[List[Dict[str, Any]]] = None,
        **extra,
    ) -> Tuple[StoredInitiative, str]:
        """Create an initiative and report WHAT HAPPENED so the caller knows whether to
        dispatch. The outcome is one of:

          "created"          — a fresh row was inserted; dispatch it.
          "reactivated"      — a previously FAILED instance was reset to pending; dispatch it.
          "deduped_active"   — an active instance already exists (same dedup_key, or any active
                               one sharing dedup_base across a period rollover); do NOT dispatch.
          "deduped_terminal" — this period's instance already completed/cancelled; do NOT
                               dispatch (a recurring type re-arms on its next-period dedup_key).

        This replaces the old id-comparison heuristic in the loop, which could never tell a
        fresh create from a dedup hit (the store assigns its own uuid).
        """
        if extra.get("kind") in OWNER_KINDS:
            raise ValueError("Owner kinds require the owner-goals store")
        # Check pending limit
        pending_count = self.count(status=["pending"])
        if pending_count >= MAX_PENDING_INITIATIVES:
            raise ValueError(
                f"Too many pending initiatives (max {MAX_PENDING_INITIATIVES})"
            )

        # Cross-period in-flight guard: never create a second instance of the same logical
        # work while one is still active, even after the period (dedup_key) rolls over.
        if dedup_base:
            active = self.get_active_by_dedup_base(dedup_base)
            if active:
                logger.debug(
                    "dedup_base %s already has active initiative %s; suppressing",
                    dedup_base, active.id,
                )
                return active, "deduped_active"


        # Period-key dedup (at most one row per dedup_key; UNIQUE).
        if dedup_key:
            existing = self.get_by_dedup_key(dedup_key)
            if existing:
                if existing.is_active:
                    logger.info(
                        "Initiative with dedup_key %s already exists: %s",
                        dedup_key, existing.id,
                    )
                    return existing, "deduped_active"
                # FAILED reactivates so the work can be retried.
                if existing.status == InitiativeStatus.FAILED.value:
                    logger.info(
                        "Reactivating failed initiative %s with dedup_key %s",
                        existing.id, dedup_key,
                    )
                    reactivated = self.update(
                        existing.id,
                        status=InitiativeStatus.PENDING.value,
                        failed_at=None,
                        failed_reason=None,
                        attempt_count=0,
                        assigned_agent_id=None,
                        assigned_agent_name=None,
                        assigned_at=None,
                        acknowledged_at=None,
                    )
                    return reactivated, "reactivated"
                # Completed/cancelled for THIS period — already ran; don't re-fire within it.
                # A recurring type gets a new dedup_key next period and re-arms there.
                logger.debug(
                    "dedup_key %s already ran this period (status=%s); suppressing",
                    dedup_key, existing.status,
                )
                return existing, "deduped_terminal"

        # Reject priorities outside the supported range.
        priority = max(0.0, min(1.0, priority))
        
        initiative_id = str(uuid.uuid4())
        now = now_utc()
        stored_items = []
        item_ids = set()
        for item in items or []:
            item = {"selectable": True, "is_selected": True, "summary": "", "build_summary": "", **item}
            item["id"] = item.get("id") or str(uuid.uuid4())
            if item["id"] in item_ids:
                raise ValueError("item ids must be unique within an initiative")
            item_ids.add(item["id"])
            stored_items.append(item)

        self._db.execute(
            """
            INSERT INTO initiatives (
                id, dedup_key, dedup_base, type, description, priority, rationale,
                action_hint, entity_id, source_type, source_id, created_by,
                timeout_seconds, expires_at, preferred_agent_id, job_id,
                context, created_at, presentation, items
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                initiative_id,
                dedup_key,
                dedup_base,
                type,
                description,
                priority,
                rationale,
                action_hint,
                entity_id,
                source_type,
                source_id,
                created_by,
                timeout_seconds,
                expires_at.isoformat() if expires_at else None,
                preferred_agent_id,
                job_id,
                json.dumps(context) if context is not None else None,
                now.isoformat(),
                json.dumps(presentation) if presentation else None,
                json.dumps(stored_items) if stored_items else None,
            ],
        )
        self._db.commit()

        return self.get(initiative_id), "created"

    def ideas_page(self, limit: int, cursor: Optional[str] = None):
        """Positive eligibility: only the tick's source-validated owner offers.

        Cursors bind an offset to the current ordered catalog. Changes refuse
        the cursor so a client can restart rather than silently skip offers.
        """
        engine = getattr(self, "_ideas", None)
        rows = engine.catalog() if engine else []
        fingerprint = hashlib.sha256(json.dumps([r.id for r in rows]).encode()).hexdigest()[:24]
        offset = 0
        if cursor is not None:
            parts = cursor.split(".")
            if (len(parts) != 3 or parts[:2] != ["ideas2", fingerprint] or not parts[2].isdigit()
                    or not 0 < int(parts[2]) < len(rows)):
                raise ValueError("invalid_or_stale_ideas_cursor")
            offset = int(parts[2])
        end = min(len(rows), offset + max(1, min(limit, 6)))
        next_cursor = f"ideas2.{fingerprint}.{end}" if end < len(rows) else None
        has_built = bool(engine and any((r.context["idea"].get("execution") or {}).get("taskId") for r in engine.rows()))
        return rows[offset:end], next_cursor, has_built

    def idea_states(self, ids: Sequence[str]) -> List[Dict[str, Any]]:
        """Current eligible states; legacy/missing/stale ids disclose no card."""
        if not 1 <= len(ids) <= MAX_IDEA_STATE_IDS or any(not value for value in ids):
            raise ValueError(f"ids must contain 1 to {MAX_IDEA_STATE_IDS} nonempty ids")
        engine = getattr(self, "_ideas", None)
        result = []
        for ident in ids:
            row = self.get(ident)
            eligible = engine is not None and engine.valid(row, accepted=True)
            card = engine.card(row) if eligible else {}
            result.append({"id": ident, "exists": eligible, "state": card.get("buildState"),
                           "status": row.status if eligible else None,
                           **{k: card[k] for k in ("buildStatus", "build_status", "buildState") if k in card},
                           **{k: card[k] for k in ("executionId", "taskId") if k in card}})
        return result

    def record_engagement(self, initiative_id: str, receipt: Dict[str, Any]) -> bool:
        """Append once per client receipt id; a conflicting reuse is refused."""
        if self.get(initiative_id) is None:
            return False
        values = [initiative_id, receipt["kind"], receipt["visible_fraction"], receipt["surface"], receipt["anchor"]]
        cursor = self._db.execute("""INSERT INTO initiative_engagement
            (receipt_id, initiative_id, kind, visible_fraction, surface, anchor, created_at)
            VALUES (?,?,?,?,?,?,?) ON CONFLICT(receipt_id) DO NOTHING""",
            [receipt["receipt_id"], *values, now_utc().isoformat()])
        self._db.commit()
        existing = self._db.execute("""SELECT initiative_id, kind, visible_fraction, surface, anchor
            FROM initiative_engagement WHERE receipt_id=?""", [receipt["receipt_id"]]).fetchone()
        if list(existing) != values:
            raise ValueError("receipt_id already belongs to a different receipt")
        return cursor.rowcount == 1

    def share(self, initiative_id: str, title: str, summary: str) -> Optional[Dict[str, Any]]:
        """Freeze only the card's title and summary; repeated sharing keeps its id."""
        if self.get(initiative_id) is None:
            return None
        self._db.execute("""INSERT INTO initiative_shares (share_id, initiative_id, title, summary)
            VALUES (?,?,?,?) ON CONFLICT(initiative_id) DO NOTHING""",
            [str(uuid.uuid4()), initiative_id, title, summary])
        self._db.commit()
        return self.resolve_share(self._db.execute(
            "SELECT share_id FROM initiative_shares WHERE initiative_id=?", [initiative_id]
        ).fetchone()[0])

    def resolve_share(self, share_id: str) -> Optional[Dict[str, Any]]:
        row = self._db.execute("""SELECT s.share_id, s.title, s.summary FROM initiative_shares s
            JOIN generic_initiatives i ON i.id=s.initiative_id WHERE s.share_id=?""", [share_id]).fetchone()
        return dict(row) if row else None

    def get(self, initiative_id: str) -> Optional[StoredInitiative]:
        """Get initiative by ID."""
        cursor = self._db.execute(
            "SELECT * FROM generic_initiatives WHERE id = ?",
            [initiative_id],
        )
        row = cursor.fetchone()
        if row:
            return StoredInitiative.from_row(dict(row))
        return None

    def get_by_dedup_key(self, dedup_key: str) -> Optional[StoredInitiative]:
        """Get initiative by dedup key."""
        cursor = self._db.execute(
            "SELECT * FROM generic_initiatives WHERE dedup_key = ?",
            [dedup_key],
        )
        row = cursor.fetchone()
        if row:
            return StoredInitiative.from_row(dict(row))
        return None

    def get_active_by_dedup_base(self, dedup_base: str) -> Optional[StoredInitiative]:
        """The most recent ACTIVE initiative sharing this logical (un-bucketed) key, or None.
        Used to suppress a duplicate when a recurring initiative's period rolls over while a
        prior instance is still in flight."""
        cursor = self._db.execute(
            "SELECT * FROM generic_initiatives WHERE dedup_base = ? "
            "AND status IN ('pending', 'assigned', 'acknowledged') "
            "ORDER BY created_at DESC LIMIT 1",
            [dedup_base],
        )
        row = cursor.fetchone()
        if row:
            return StoredInitiative.from_row(dict(row))
        return None

    def finished_under_base(self, dedup_base: str, *, other_than: str) -> bool:
        """Whether a mind task under this stable key was done (``outcome='done'``) under another period's key
        than ``other_than``: recurring self-work (``research:<topic>``) that already finished in an earlier
        period."""
        return self._db.execute(
            "SELECT 1 FROM generic_initiatives WHERE dedup_base = ? AND kind = 'task' AND outcome = 'done' "
            "AND dedup_key != ? LIMIT 1", [dedup_base, other_than]).fetchone() is not None

    # Columns accepted by update(). Kept in sync with _create_tables() so a
    # typo or a renamed column fails loudly at the API boundary instead of
    # being swallowed by an outer except clause that calls the failure
    # "non-fatal" (see autonomy/loop.py:_phase_ghost_cleanup for the bug class
    # this whitelist was added to prevent).
    _UPDATABLE_COLUMNS = frozenset({
        "dedup_key", "dedup_base", "type", "description", "priority", "rationale",
        "action_hint", "entity_id", "source_type", "source_id", "created_by",
        "status", "assigned_agent_id", "assigned_agent_name", "assigned_at",
        "acknowledged_at", "completed_at", "cancelled_at", "cancelled_by",
        "cancelled_reason", "failed_at", "failed_reason",
        "attempt_count", "max_attempts", "timeout_seconds", "last_attempt_at",
        "expires_at",
        "delivery_mode", "delivery_attempts", "last_delivery_at",
        "delivery_failed_at", "delivery_failed_reason",
        "result", "result_metadata",
        "preferred_agent_id", "stale_reason", "recovery_reason", "job_id",
        "context", *MIND_COLUMNS,
    })
    _TIMESTAMP_COLUMNS = frozenset({
        "assigned_at", "acknowledged_at", "completed_at", "cancelled_at", "failed_at",
        "expires_at", "last_attempt_at", "last_delivery_at", "delivery_failed_at", "due_at",
    })

    def update(self, initiative_id: str, **updates) -> Optional[StoredInitiative]:
        """Update initiative fields."""
        if updates.get("kind") in OWNER_KINDS:
            raise ValueError("Owner kinds require the owner-goals store")
        if self.get(initiative_id) is None:
            return None
        if not updates:
            return self.get(initiative_id)

        set_parts, params = self._assignments(updates)

        if not set_parts:
            return self.get(initiative_id)

        params.append(initiative_id)
        query = f"UPDATE initiatives SET {', '.join(set_parts)} WHERE id = ? AND (kind IS NULL OR kind NOT IN ('owner_goal','owner_subgoal','owner_goal_progress'))"

        self._db.execute(query, params)
        self._db.commit()

        return self.get(initiative_id)

    def _assignments(self, updates: Dict[str, Any]) -> Tuple[List[str], List[Any]]:
        """``update``'s SET clause and its parameters (JSON and timestamp columns encoded)."""
        if updates.get("kind") in OWNER_KINDS:
            raise ValueError("Owner kinds require the owner-goals store")
        unknown = set(updates) - self._UPDATABLE_COLUMNS
        if unknown:
            raise ValueError(
                f"Unknown initiative column(s): {sorted(unknown)}"
            )

        # Build SET clause
        set_parts = []
        params = []

        for key, value in updates.items():
            if key in ("result_metadata", "context", "success_check", "lesson_ids"):
                set_parts.append(f"{key} = ?")
                if value is None:
                    params.append(None)
                else:
                    params.append(json.dumps(value) if not isinstance(value, str) else value)
            elif key in self._TIMESTAMP_COLUMNS:
                set_parts.append(f"{key} = ?")
                if isinstance(value, datetime):
                    params.append(value.isoformat())
                else:
                    params.append(value)
            else:
                set_parts.append(f"{key} = ?")
                params.append(value)

        return set_parts, params

    def list(
        self,
        status: Optional[List[str]] = None,
        type: Optional[str] = None,
        assigned_agent_id: Optional[str] = None,
        job_id: Optional[str] = None,
        created_before: Optional[datetime] = None,
        created_after: Optional[datetime] = None,
        acknowledged_before: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[StoredInitiative]:
        """List initiatives with filters."""
        # Owner records have their own audience-bound reader and lifecycle;
        # mind rows (those with a ``kind``) are read through ``intentions``.
        query = "SELECT * FROM generic_initiatives WHERE kind IS NULL"
        params: List[Any] = []

        if status:
            placeholders = ",".join("?" * len(status))
            query += f" AND status IN ({placeholders})"
            params.extend(status)

        if type:
            query += " AND type = ?"
            params.append(type)

        if assigned_agent_id:
            query += " AND assigned_agent_id = ?"
            params.append(assigned_agent_id)

        if job_id:
            query += " AND job_id = ?"
            params.append(job_id)

        if created_before:
            query += " AND created_at < ?"
            params.append(created_before.isoformat())

        if created_after:
            query += " AND created_at > ?"
            params.append(created_after.isoformat())

        if acknowledged_before:
            query += " AND acknowledged_at < ?"
            params.append(acknowledged_before.isoformat())

        query += " ORDER BY priority DESC, created_at ASC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = self._db.execute(query, params)
        return [StoredInitiative.from_row(dict(row)) for row in cursor.fetchall()]

    def count(
        self,
        status: Optional[List[str]] = None,
        assigned_agent_id: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> int:
        """Count initiatives with filters."""
        query = "SELECT COUNT(*) FROM generic_initiatives WHERE kind IS NULL"
        params: List[Any] = []

        if status:
            placeholders = ",".join("?" * len(status))
            query += f" AND status IN ({placeholders})"
            params.extend(status)

        if assigned_agent_id:
            query += " AND assigned_agent_id = ?"
            params.append(assigned_agent_id)

        if job_id:
            query += " AND job_id = ?"
            params.append(job_id)

        cursor = self._db.execute(query, params)
        return cursor.fetchone()[0]

    # ------------------------------------------------------------------
    # Status Transitions
    # ------------------------------------------------------------------


    def complete(
        self,
        initiative_id: str,
        agent_id: str,
        result: Optional[str] = None,
        result_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[StoredInitiative]:
        """Mark initiative as completed."""
        now = now_utc()

        initiative = self.get(initiative_id)
        if not initiative:
            return None

        # Check if expired
        if initiative.is_expired:
            return self.update(
                initiative_id,
                status=InitiativeStatus.FAILED.value,
                failed_at=now,
                failed_reason="initiative_expired",
            )

        updated = self.update(
            initiative_id,
            status=InitiativeStatus.COMPLETED.value,
            completed_at=now,
            result=result,
            result_metadata=result_metadata or {},
        )

        if updated:
            self.log_history(
                initiative_id,
                action="completed",
                agent_id=agent_id,
                details={"result": result},
            )

        return updated

    def fail(
        self,
        initiative_id: str,
        agent_id: str,
        reason: str,
        retry: bool = False,
    ) -> Optional[StoredInitiative]:
        """Mark initiative as failed."""
        now = now_utc()

        initiative = self.get(initiative_id)
        if not initiative:
            return None

        new_attempt_count = initiative.attempt_count + 1

        # If retry requested and attempts remaining, reset to pending
        if retry and new_attempt_count < initiative.max_attempts:
            updated = self.update(
                initiative_id,
                status=InitiativeStatus.PENDING.value,
                assigned_agent_id=None,
                assigned_agent_name=None,
                assigned_at=None,
                acknowledged_at=None,
                attempt_count=new_attempt_count,
                last_attempt_at=now,
            )

            self.log_history(
                initiative_id,
                action="retry_scheduled",
                agent_id=agent_id,
                details={"reason": reason, "attempt": new_attempt_count},
            )
        else:
            # Mark as failed
            updated = self.update(
                initiative_id,
                status=InitiativeStatus.FAILED.value,
                failed_at=now,
                failed_reason=reason,
                attempt_count=new_attempt_count,
                last_attempt_at=now,
            )

            self.log_history(
                initiative_id,
                action="failed",
                agent_id=agent_id,
                details={"reason": reason, "attempt": new_attempt_count},
            )

            # Add to dead letter queue if max attempts reached
            if new_attempt_count >= initiative.max_attempts:
                self._add_to_dlq(initiative, reason)

        return updated

    def retry(self, initiative_id: str, actor_id: str) -> Optional[StoredInitiative]:
        """Requeue a failed initiative and its history in one transaction."""
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise ValueError("Retry requires an actor")
        # Do not borrow the shared connection: another operation must never
        # commit this transition without its history or inherit a failed write.
        with closing(sqlite3.connect(self._db_path, timeout=2)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM generic_initiatives WHERE id=? AND status='failed'",
                             (initiative_id,)).fetchone()
            if row is None:
                return None
            details = {
                "attempt_count": row["attempt_count"],
                "previous_agent_id": row["assigned_agent_id"],
                "failed_reason": row["failed_reason"],
                "failed_at": row["failed_at"],
            }
            db.execute("""UPDATE initiatives SET status='pending', assigned_agent_id=NULL,
                failed_reason=NULL, failed_at=NULL WHERE id=? AND (kind IS NULL OR kind NOT IN ('owner_goal','owner_subgoal','owner_goal_progress'))""", (initiative_id,))
            db.execute("""INSERT INTO assignment_history(initiative_id,agent_id,action,details)
                VALUES(?,?,'retry',?)""", (initiative_id, actor_id, json.dumps(details)))
            return StoredInitiative.from_row(dict(db.execute(
                "SELECT * FROM generic_initiatives WHERE id=?", (initiative_id,)).fetchone()))

    def cancel(
        self,
        initiative_id: str,
        cancelled_by: str,
        reason: Optional[str] = None,
    ) -> Optional[StoredInitiative]:
        """Cancel an initiative."""
        now = now_utc()

        initiative = self.get(initiative_id)
        if not initiative or not initiative.is_active:
            return None

        updated = self.update(
            initiative_id,
            status=InitiativeStatus.CANCELLED.value,
            cancelled_at=now,
            cancelled_by=cancelled_by,
            cancelled_reason=reason,
        )

        if updated:
            self.log_history(
                initiative_id,
                action="cancelled",
                agent_id=cancelled_by,
                details={"reason": reason},
            )

        return updated


    # ------------------------------------------------------------------
    # Intentions (the mind): one row is an intention and its audit entry
    # ------------------------------------------------------------------

    MIND_ACTOR = "mind"

    def create_intention(
        self,
        *,
        kind: str,
        type: str,
        title: str,
        drive: str,
        cls: str,
        decision: str,
        decision_reason: str,
        status: str,
        dedup_key: Optional[str],
        rationale: str = "",
        recipient: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        priority: float = 0.5,
        expires_at: Optional[datetime] = None,
        due_at: Optional[datetime] = None,
        ask_code: Optional[str] = None,
        invalidates_if: Optional[str] = None,
        success_check: Optional[Dict[str, Any]] = None,
        hermes_kind: str = "none",
        source_type: Optional[str] = None,
        source_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
    ) -> Tuple[StoredInitiative, str]:
        """Save an intention; ``dedup_key`` makes a repeat return the existing row.

        The outcome is ``created`` or ``deduped``: an obligation is reported
        once, whatever became of the earlier intention (architecture 3.3).
        """
        if kind in OWNER_KINDS:
            raise ValueError("Owner kinds require the owner-goals store")
        if dedup_key:
            existing = self.get_by_dedup_key(dedup_key)
            if existing is not None:
                return existing, "deduped"
        intention_id = str(uuid.uuid4())
        now = created_at or now_utc()
        self._db.execute(
            """
            INSERT INTO initiatives (
                id, dedup_key, type, description, priority, rationale, entity_id,
                source_type, source_id, created_by, status, expires_at, context, created_at,
                kind, cls, decision, decision_reason, drive, ask_code, invalidates_if,
                success_check, hermes_kind, due_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                intention_id, dedup_key, type, title, max(0.0, min(1.0, float(priority))),
                rationale, recipient, source_type, source_id, self.MIND_ACTOR, status,
                expires_at.isoformat() if expires_at else None,
                json.dumps(context) if context is not None else None, now.isoformat(),
                kind, cls, decision, decision_reason, drive, ask_code, invalidates_if,
                json.dumps(success_check) if success_check is not None else None,
                hermes_kind, due_at.isoformat() if due_at else None,
            ],
        )
        # A piece of the agent's own work opens its recorder line here; the first history row keeps the line id.
        line = recorder.begin("card", CARD_SOURCES[kind], ref=f"initiative:{intention_id}") \
            if kind in CARD_SOURCES else None
        self._db.execute(
            "INSERT INTO assignment_history (initiative_id, agent_id, action, timestamp, details) "
            "VALUES (?, ?, ?, ?, ?)",
            [intention_id, self.MIND_ACTOR, f"decided_{decision}", now.isoformat(),
             json.dumps({"status": status, "reason": decision_reason, **({"line": line} if line else {})})],
        )
        self._db.commit()
        created = self.get(intention_id)
        if status in MIND_TERMINAL_STATUSES:
            self.finish_card(created, detail=f"decided_{decision}")
        return created, "created"

    def card_line(self, initiative_id: str) -> Optional[str]:
        """The recorder line opened when this intention was raised (its first history row keeps the id), or None."""
        return next((str(item.details["line"]) for item in self.get_history(initiative_id, limit=1000)
                     if isinstance(item.details, dict) and item.details.get("line")), None)

    def finish_card(self, row: Optional[StoredInitiative], *, detail: str, result: Optional[str] = None) -> None:
        """Close the recorder line of a piece of the agent's own work: what came of it (``result``, else read from
        the row's status), how it was decided, and ``detail``, the code label of what settled it."""
        line = self.card_line(row.id) if row is not None and row.kind in CARD_SOURCES else None
        if line:
            recorder.finish(line, result=result or CARD_RESULTS.get(row.status, "silent"),
                            choice=CARD_CHOICES.get(row.decision or "", ""), chooser="rule", detail=detail)

    def transition(self, initiative_id: str, status: str, *, action: str,
                   details: Optional[Dict[str, Any]] = None, at: Optional[datetime] = None,
                   guard: Optional[SettleGuard] = None, **updates) -> Optional[StoredInitiative]:
        """Move an intention to ``status`` and log the transition with its time.

        With ``guard``, only while the row is still in one of ``guard.expect``: the update, its history row and the
        guard's own writes are one SQLite transaction, and a row that moved on raises ``TransitionRefused`` with
        nothing written."""
        now = at or now_utc()
        if guard is not None:
            return self._guarded_transition(initiative_id, status, action=action, details=details, at=now,
                                            guard=guard, updates=updates)
        settles = status in MIND_TERMINAL_STATUSES and \
            getattr(self.get(initiative_id), "status", status) not in MIND_TERMINAL_STATUSES
        row = self.update(initiative_id, status=status, **updates)
        if row is not None:
            self._db.execute(
                "INSERT INTO assignment_history (initiative_id, agent_id, action, timestamp, details) "
                "VALUES (?, ?, ?, ?, ?)",
                [initiative_id, self.MIND_ACTOR, action, now.isoformat(),
                 json.dumps(details) if details else None],
            )
            self._db.commit()
            if settles:
                self.finish_card(row, detail=str((details or {}).get("label") or action))
        return row

    def _guarded_transition(self, initiative_id: str, status: str, *, action: str,
                            details: Optional[Dict[str, Any]], at: datetime, guard: SettleGuard,
                            updates: Dict[str, Any]) -> StoredInitiative:
        """``transition`` under ``guard``: compare-and-set on the status, then the history row and the guard's
        writes, committed together. The UPDATE is the transaction's first statement, so SQLite's write lock (and
        ``busy_timeout`` for another process holding it) decides which of two racing answers commits first."""
        expect = tuple(dict.fromkeys(str(value) for value in guard.expect))
        if not expect:
            raise ValueError("a settle guard names the statuses it expects")
        set_parts, params = self._assignments({"status": status, **updates})
        marks = ",".join("?" * len(expect))
        try:
            cursor = self._db.execute(
                f"UPDATE initiatives SET {', '.join(set_parts)} WHERE id = ? AND (kind IS NULL OR kind NOT IN ('owner_goal','owner_subgoal','owner_goal_progress')) AND status IN ({marks})",
                [*params, initiative_id, *expect])
            if cursor.rowcount != 1:
                self._db.rollback()
                raise TransitionRefused(initiative_id, self.get(initiative_id))
            self._db.execute(
                "INSERT INTO assignment_history (initiative_id, agent_id, action, timestamp, details) "
                "VALUES (?, ?, ?, ?, ?)",
                [initiative_id, self.MIND_ACTOR, action, at.isoformat(), json.dumps(details) if details else None],
            )
            for statement, values in (guard.writes(status, at) if guard.writes is not None else ()):
                self._db.execute(statement, list(values))
            self._db.commit()
        except TransitionRefused:
            raise
        except BaseException:
            self._db.rollback()
            raise
        row = self.get(initiative_id)
        if status in MIND_TERMINAL_STATUSES and not MIND_TERMINAL_STATUSES.intersection(expect):
            self.finish_card(row, detail=str((details or {}).get("label") or action))
        return row

    # -- explicit answers' idempotency records (``ask_answer_requests``) ------------------------------------------

    def answer_request(self, scope: str, key: str) -> Optional[Dict[str, Any]]:
        """The recorded explicit answer request under this caller scope and idempotency key, or None."""
        row = self._db.execute("SELECT * FROM ask_answer_requests WHERE EXISTS (SELECT 1 FROM generic_initiatives i WHERE i.id=ask_id) AND scope = ? AND key = ?",
                               [scope, key]).fetchone()
        if row is None:
            return None
        value = dict(row)
        value["response"] = json.loads(value["response"])
        return value

    @staticmethod
    def answer_request_write(*, scope: str, key: str, ask_id: str, request_hash: str, status_code: int,
                             response: Dict[str, Any], at: datetime) -> Tuple[str, Sequence[Any]]:
        """The statement that records an explicit answer request (for a ``SettleGuard``'s writes, or alone). A key
        already recorded fails the statement (its primary key), so the transaction it is in writes nothing."""
        return ("INSERT INTO ask_answer_requests (scope, key, ask_id, request_hash, status_code, response, created_at) "
                "SELECT ?, ?, ?, ?, ?, ?, ? WHERE EXISTS (SELECT 1 FROM generic_initiatives WHERE id=?)",
                [scope, key, ask_id, request_hash, int(status_code), json.dumps(response, sort_keys=True),
                 at.isoformat(), ask_id])

    def answer_requests_since(self, since: datetime) -> List[Dict[str, Any]]:
        """The explicit answer requests recorded at or after ``since``, oldest first (``created_at`` is the
        settlement's own time for one that settled the ask)."""
        floor = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
        values = []
        # A coarse bound in SQL (the times are stored as UTC ISO strings), the exact one parsed.
        coarse = (floor.astimezone(timezone.utc) - timedelta(days=1)).isoformat()
        for row in self._db.execute("SELECT * FROM ask_answer_requests WHERE EXISTS (SELECT 1 FROM generic_initiatives i WHERE i.id=ask_id) AND created_at >= ? ORDER BY created_at",
                                    [coarse]).fetchall():
            value = dict(row)
            try:
                at = datetime.fromisoformat(str(value["created_at"]).replace("Z", "+00:00"))
            except ValueError:
                continue
            at = at if at.tzinfo else at.replace(tzinfo=timezone.utc)
            if at >= floor:
                value["created_at"], value["response"] = at, json.loads(value["response"])
                values.append(value)
        return values

    def record_answer_request(self, **fields: Any) -> bool:
        """Record an explicit answer request that settled nothing (``answer_request_write``); False when its key was
        recorded already."""
        statement, values = self.answer_request_write(**fields)
        try:
            cursor = self._db.execute(statement, list(values))
            self._db.commit()
            return cursor.rowcount == 1
        except sqlite3.IntegrityError:
            self._db.rollback()
            return False

    def intentions(
        self,
        status: Optional[List[str]] = None,
        kind: Optional[List[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
        recipient: Optional[str] = None,
        types: Optional[List[str]] = None,
    ) -> List[StoredInitiative]:
        """Mind rows (those with a ``kind``), newest first; ``recipient`` and ``types`` select before the limit."""
        query = "SELECT * FROM generic_initiatives WHERE kind IS NOT NULL"
        params: List[Any] = []
        if recipient:
            query += " AND entity_id = ?"
            params.append(recipient)
        if types:
            query += f" AND type IN ({','.join('?' * len(types))})"
            params.extend(types)
        if status:
            query += f" AND status IN ({','.join('?' * len(status))})"
            params.extend(status)
        if kind:
            query += f" AND kind IN ({','.join('?' * len(kind))})"
            params.extend(kind)
        if since is not None:
            query += " AND created_at >= ?"
            params.append(since.isoformat())
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        return [StoredInitiative.from_row(dict(row)) for row in self._db.execute(query, params).fetchall()]

    def findings(self, states: List[str], types: List[str], limit: int = 500) -> List[StoredInitiative]:
        """Done task rows of ``types`` whose finding outreach (``result_metadata.outreach.state``) is one of
        ``states``, newest first, whatever their age: a finding is found by its state, never by when it formed."""
        if not states or not types:
            return []
        query = (f"SELECT * FROM generic_initiatives WHERE kind = 'task' AND status = 'done' "
                 f"AND type IN ({','.join('?' * len(types))}) "
                 f"AND json_extract(result_metadata, '$.outreach.state') IN ({','.join('?' * len(states))}) "
                 "ORDER BY created_at DESC LIMIT ?")
        params: List[Any] = [*types, *states, max(1, int(limit))]
        return [StoredInitiative.from_row(dict(row)) for row in self._db.execute(query, params).fetchall()]

    def lesson_rows(self, since: Optional[datetime] = None, limit: int = 20000) -> List[StoredInitiative]:
        """Mind rows that carried a lesson (``lesson_ids``), newest first: the joins lessons are scored by."""
        query = ("SELECT * FROM generic_initiatives WHERE kind IS NOT NULL AND lesson_ids IS NOT NULL "
                 "AND lesson_ids NOT IN ('', '[]')")
        params: List[Any] = []
        if since is not None:
            query += " AND created_at >= ?"
            params.append(since.isoformat())
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        return [StoredInitiative.from_row(dict(row)) for row in self._db.execute(query, params).fetchall()]

    def get_by_ask_code(self, code: str) -> Optional[StoredInitiative]:
        row = self._db.execute(
            "SELECT * FROM generic_initiatives WHERE ask_code = ? AND status = 'asked'", [code.upper()]
        ).fetchone()
        return StoredInitiative.from_row(dict(row)) if row else None

    def get_by_hermes_ref(self, hermes_ref: str) -> Optional[StoredInitiative]:
        row = self._db.execute(
            "SELECT * FROM generic_initiatives WHERE hermes_ref = ? ORDER BY created_at DESC LIMIT 1", [hermes_ref]
        ).fetchone()
        return StoredInitiative.from_row(dict(row)) if row else None

    def open_ask_codes(self) -> List[str]:
        return [row[0] for row in self._db.execute(
            "SELECT ask_code FROM generic_initiatives WHERE status = 'asked' AND ask_code IS NOT NULL").fetchall()]

    def count_transitions(self, action: str, since: datetime, *, kind: Optional[str] = None,
                          recipient: Optional[str] = None, exclude_types: Tuple[str, ...] = (),
                          include_types: Tuple[str, ...] = ()) -> int:
        """Transitions of one kind since ``since``, from the history: the budget counters. ``include_types``
        counts only those types, ``exclude_types`` leaves those out."""
        query = ("SELECT COUNT(*) FROM assignment_history h JOIN generic_initiatives i ON i.id = h.initiative_id "
                 "WHERE h.action = ? AND h.timestamp >= ?")
        params: List[Any] = [action, since.isoformat()]
        if kind:
            query += " AND i.kind = ?"
            params.append(kind)
        if recipient:
            query += " AND i.entity_id = ?"
            params.append(recipient)
        if exclude_types:
            query += f" AND i.type NOT IN ({','.join('?' * len(exclude_types))})"
            params.extend(exclude_types)
        if include_types:
            query += f" AND i.type IN ({','.join('?' * len(include_types))})"
            params.extend(include_types)
        return int(self._db.execute(query, params).fetchone()[0])

    def last_transition_at(self, action: str, *, recipient: Optional[str] = None,
                           type: Optional[str] = None, types: Tuple[str, ...] = ()) -> Optional[datetime]:
        query = ("SELECT MAX(h.timestamp) FROM assignment_history h JOIN generic_initiatives i ON i.id = h.initiative_id "
                 "WHERE h.action = ?")
        params: List[Any] = [action]
        if recipient:
            query += " AND i.entity_id = ?"
            params.append(recipient)
        if type:
            query += " AND i.type = ?"
            params.append(type)
        if types:
            query += f" AND i.type IN ({','.join('?' * len(types))})"
            params.extend(types)
        value = self._db.execute(query, params).fetchone()[0]
        if not value:
            return None
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    def failures_since(self, cls: str, since: datetime) -> List[StoredInitiative]:
        rows = self._db.execute(
            "SELECT * FROM generic_initiatives WHERE kind IS NOT NULL AND cls = ? AND outcome = 'failed' "
            "AND failed_at >= ? ORDER BY failed_at DESC", [cls, since.isoformat()],
        ).fetchall()
        return [StoredInitiative.from_row(dict(row)) for row in rows]

    def tokens_since(self, since: datetime) -> int:
        value = self._db.execute(
            "SELECT COALESCE(SUM(cost_tokens), 0) FROM generic_initiatives WHERE kind IS NOT NULL AND created_at >= ?",
            [since.isoformat()],
        ).fetchone()[0]
        return int(value or 0)

    def prune_intentions(self, before: datetime) -> Dict[str, int]:
        """Delete terminal mind rows older than ``before``; returns counts by outcome."""
        from .models import MIND_TERMINAL_STATUSES
        statuses = sorted(MIND_TERMINAL_STATUSES)
        placeholders = ",".join("?" * len(statuses))
        rows = self._db.execute(
            f"SELECT id, outcome, status FROM generic_initiatives WHERE kind IS NOT NULL "
            f"AND status IN ({placeholders}) AND created_at < ?", [*statuses, before.isoformat()],
        ).fetchall()
        counts: Dict[str, int] = {}
        for row in rows:
            key = row["outcome"] or row["status"]
            counts[key] = counts.get(key, 0) + 1
        ids = [row["id"] for row in rows]
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            marks = ",".join("?" * len(chunk))
            self._db.execute(f"DELETE FROM assignment_history WHERE initiative_id IN ({marks})", chunk)
            self._db.execute(f"DELETE FROM ask_answer_requests WHERE EXISTS (SELECT 1 FROM generic_initiatives i WHERE i.id=ask_id) AND ask_id IN ({marks})", chunk)
            self._db.execute(f"DELETE FROM initiatives WHERE id IN ({marks}) AND (kind IS NULL OR kind NOT IN ('owner_goal','owner_subgoal','owner_goal_progress'))", chunk)
        self._db.commit()
        return counts

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def log_history(
        self,
        initiative_id: str,
        action: str,
        agent_id: str,
        agent_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        origin: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log assignment history."""
        if self.get(initiative_id) is None:
            return
        self._db.execute(
            """
            INSERT INTO assignment_history (
                initiative_id, agent_id, agent_name, action, details, origin
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                initiative_id,
                agent_id,
                agent_name,
                action,
                json.dumps(details) if details else None,
                json.dumps(origin) if origin is not None else None,
            ],
        )
        self._db.commit()

    def last_response_origin(self, initiative_id: str) -> Optional[Dict[str, Any]]:
        """Latest supplied respond origin; later legacy responses keep the association."""
        row = self._db.execute("""SELECT origin FROM assignment_history
            WHERE initiative_id=? AND origin IS NOT NULL AND action GLOB 'llm_*'
            AND EXISTS (SELECT 1 FROM generic_initiatives i WHERE i.id=initiative_id)
            ORDER BY id DESC LIMIT 1""", [initiative_id]).fetchone()
        return json.loads(row["origin"]) if row else None

    def last_responses(self, ids: Sequence[str]) -> Dict[str, Dict[str, str]]:
        """Latest model response per initiative, read in one grouped query for the page.

        History insertion order need not match its timestamps. Join the maximum timestamp back to
        its row; equal timestamps use the last inserted row. GLOB treats the underscore literally.
        """
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self._db.execute(
            f"""
            WITH latest AS (
                SELECT initiative_id, MAX(julianday(timestamp)) AS at
                FROM assignment_history
                WHERE EXISTS (SELECT 1 FROM generic_initiatives i WHERE i.id=initiative_id) AND initiative_id IN ({marks}) AND action GLOB 'llm_*'
                GROUP BY initiative_id
            )
            SELECT h.initiative_id, h.action, h.timestamp
            FROM assignment_history h JOIN latest l
                ON h.initiative_id = l.initiative_id AND julianday(h.timestamp) = l.at
            WHERE h.action GLOB 'llm_*'
            ORDER BY h.id
            """, list(ids)).fetchall()
        responses: Dict[str, Dict[str, str]] = {}
        latest_at: Dict[str, datetime] = {}
        for row in rows:
            at = datetime.fromisoformat(row["timestamp"])
            stamp = at.replace(tzinfo=timezone.utc) if at.tzinfo is None else at.astimezone(timezone.utc)
            initiative_id = row["initiative_id"]
            # SQLite's date functions round to milliseconds; compare tied rows at full precision.
            if initiative_id not in latest_at or stamp >= latest_at[initiative_id]:
                latest_at[initiative_id] = stamp
                responses[initiative_id] = {"action": row["action"][4:], "at": at.isoformat()}
        return responses

    def get_history(
        self,
        initiative_id: str,
        limit: int = 50,
    ) -> List[AssignmentHistory]:
        """Get assignment history for an initiative."""
        cursor = self._db.execute(
            """
            SELECT * FROM assignment_history
            WHERE EXISTS (SELECT 1 FROM generic_initiatives i WHERE i.id=initiative_id) AND initiative_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            [initiative_id, limit],
        )
        return [AssignmentHistory.from_row(dict(row)) for row in cursor.fetchall()]

    def get_agent_history(
        self,
        agent_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[AssignmentHistory]:
        """Get assignment history for an agent."""
        cursor = self._db.execute(
            """
            SELECT * FROM assignment_history
            WHERE EXISTS (SELECT 1 FROM generic_initiatives i WHERE i.id=initiative_id) AND agent_id = ?
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            [agent_id, limit, offset],
        )
        return [AssignmentHistory.from_row(dict(row)) for row in cursor.fetchall()]


    # ------------------------------------------------------------------
    # Dead Letter Queue
    # ------------------------------------------------------------------

    def _add_to_dlq(
        self,
        initiative: StoredInitiative,
        reason: str,
    ) -> None:
        """Add failed initiative to dead letter queue."""
        if initiative.kind in OWNER_KINDS:
            return
        entry = {
            "initiative_id": initiative.id,
            "type": initiative.type,
            "description": initiative.description,
            "reason": reason,
            "attempt_count": initiative.attempt_count,
            "failed_at": now_utc().isoformat(),
        }

        with open(self._dlq_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        logger.warning(
            "Added initiative %s to dead letter queue: %s",
            initiative.id,
            reason,
        )


    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def delete_old(
        self,
        status: List[str],
        before: datetime,
    ) -> int:
        """Delete old initiatives with given statuses."""
        placeholders = ",".join("?" * len(status))
        cursor = self._db.execute(
            f"""
            DELETE FROM initiatives
            WHERE status IN ({placeholders})
            AND (kind IS NULL OR kind NOT IN ('owner_goal','owner_subgoal','owner_goal_progress'))
            AND created_at < ?
            """,
            [*status, before.isoformat()],
        )
        self._db.commit()
        return cursor.rowcount

    def backup(self) -> None:
        """Copy the store to ``initiatives.db.backup`` through SQLite, WAL commits included
        (a plain file copy of a WAL store misses whatever is not yet checkpointed)."""
        from colonymind.backup import _private_file
        _private_file(self._backup_path)
        with closing(sqlite3.connect(self._backup_path)) as target:
            self._db.backup(target)

    def close(self) -> None:
        """Close connection and create backup."""
        self.backup()
        self._db.close()
