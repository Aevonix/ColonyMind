"""Durable server-side idempotency for host turn ingestion.

The ledger is intentionally small and independent of graph/vector stores. A
turn ID is reserved durably before any downstream effect runs. Identical
replays return the stored response; different canonical content for the same
ID is a conflict. An interrupted first attempt is marked ``ambiguous`` and is
never automatically replayed because some downstream effects may already have
occurred.

A first attempt whose process died before it could report (a hard kill between
``reserve`` and ``complete``/``mark_ambiguous``) leaves a ``processing`` row
nobody will ever finish. Such a row is reclaimed by the next identical retry
once its lease expires, so a durable outbox retry can still ingest the turn
instead of being told forever that it is in progress.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional
from colonymind.util.temporal import ledger_stamp

logger = logging.getLogger(__name__)

# A keyword lane ranks this many of its best full-text matches first, before it reads their sources (my choice):
# enough for a person's results even with duplicate chunks; a lane reads further only when they fall short.
MATCH_WINDOW = 256

# How long a 'processing' reservation is trusted before an identical retry may
# take it over. Comfortably longer than any live ingestion so a slow first
# attempt is not duplicated, short enough that a hard-killed one is retried.
PROCESSING_LEASE_SECONDS = 300.0


class SourceErased(ValueError):
    """The requested source or an exact message copy was deliberately erased."""


class SourceInputPending(ValueError):
    """An exact admitted parent has not reached the canonical ledger yet."""


def source_message_hash(session_id: str, message: Mapping[str, Any]) -> str:
    # Ledger-owned normalization metadata retains the input message identity.
    # Public message schemas cannot supply this top-level field.
    original = message.get("_source_message_hash")
    if isinstance(original, str) and re.fullmatch(r"[0-9a-f]{64}", original):
        return original
    return canonical_turn_digest({"session_id": session_id, "role": message.get("role"), "content": message.get("content")})


class ReservationOutcome(str, Enum):
    CREATED = "created"
    REPLAYED = "replayed"
    CONFLICT = "conflict"
    IN_PROGRESS = "in_progress"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class Reservation:
    outcome: ReservationOutcome
    response: Optional[dict[str, Any]] = None


def _without_null_work_capture(payload: Any) -> Any:
    """Keep pre-capture digests stable, including derived-session turn wrappers."""
    if not isinstance(payload, dict):
        return payload
    if isinstance(payload.get("turn"), dict):
        payload = {**payload, "turn": _without_null_work_capture(payload["turn"])}
    work = payload.get("work")
    if isinstance(work, dict) and isinstance(work.get("operations"), list):
        capture_fields = {"request_id", "model", "error_type", "failure_kind", "duration_ms"}
        operations = [
            {key: value for key, value in operation.items()
             if key not in capture_fields or value is not None}
            if isinstance(operation, dict) else operation for operation in work["operations"]
        ]
        payload = {**payload, "work": {**work, "operations": operations}}
    return payload


def canonical_turn_digest(payload: Any) -> str:
    """Hash the normalized turn envelope, preserving list order and values."""
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json", exclude_none=False)
    payload = _without_null_work_capture(payload)
    # Preserve digests already stored before the additive checkpoint field.
    if isinstance(payload, dict) and payload.get("checkpoint_messages") is None:
        payload = {key: value for key, value in payload.items() if key != "checkpoint_messages"}
    if isinstance(payload, dict) and payload.get("assistant_source_refs") is None:
        payload = {key: value for key, value in payload.items() if key != "assistant_source_refs"}
    if isinstance(payload, dict) and payload.get("assistant_input_refs") is None:
        payload = {key: value for key, value in payload.items() if key != "assistant_input_refs"}
    if isinstance(payload, dict) and payload.get("source_only") is None:
        payload = {key: value for key, value in payload.items() if key != "source_only"}
    if isinstance(payload, dict) and payload.get("work") is None:
        payload = {key: value for key, value in payload.items() if key != "work"}
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _lexical_chunks(messages):
    """The existing FTS writer's exact chunks, also used to recover ownership."""
    from colonymind.turns.audio import source_text
    for message in messages:
        content = source_text(message.get("content"))
        if not content.strip():
            continue
        # Overlap preserves boundary phrases; full messages remain canonical.
        for start in range(0, len(content), 1800):
            yield message, content, content[start:start + 2000]


def _scoped_matches(conn, *, table: str, select: str, joins: str, expression: str, contact_id: str,
                    session_id: str, window: int | None = None):
    """A keyword lane's matches in the viewer's scope, best first (bm25, then row): rows of ``select`` over
    ``table AS f`` and ``joins`` (which name the source ``s``), each with its full-text ``rowid``.
    ``joins`` must use explicit CROSS JOIN clauses on unique keys, with the full-text table first,
    to keep FTS lookups outside the source scan even on older SQLite planners.

    Scoring every match is cheap; reading every match's source to check its scope is not. So the first ``window``
    matches are ranked in the full-text table alone and only their sources are read; only when a caller consumes
    them all and more matched does the scope-checked query run over every match, yielding those not already
    yielded. The order is the one query's order either way."""
    window = MATCH_WINDOW if window is None else window
    scope = ("s.contact_id=? AND (s.scope='person' OR s.session_id=?) "
             "AND NOT EXISTS (SELECT 1 FROM source_attribution_invalidations i WHERE i.source_id=s.turn_id)")
    top = [row[0] for row in conn.execute(
        f"SELECT rowid FROM {table} WHERE {table} MATCH ? ORDER BY bm25({table}), rowid LIMIT ?",
        (expression, window + 1))]
    first = top[:window]
    if first:
        rows = {row["rowid"]: row for row in conn.execute(
            f"SELECT {select}, f.rowid AS rowid FROM {table} AS f {joins} "
            f"WHERE f.rowid IN ({','.join('?' * len(first))}) AND {scope}", (*first, contact_id, session_id))}
        for rowid in first:
            if rowid in rows:
                yield rows[rowid]
    if len(top) > window:
        read = set(first)
        for row in conn.execute(
                f"SELECT {select}, f.rowid AS rowid FROM {table} AS f {joins} "
                f"WHERE {table} MATCH ? AND {scope} ORDER BY bm25({table}), f.rowid",
                (expression, contact_id, session_id)):
            if row["rowid"] not in read:
                yield row


def _stem_chunks(messages) -> list[tuple[str, str]]:
    """A source's keyword chunks, ``(role, text)`` in order: a stemmed row names its chunk by its place here."""
    return [(message["role"], chunk) for message, _, chunk in _lexical_chunks(messages)]


class _ErasureRules:
    """A contact's erasure rules, indexed so each message costs a few lookups.

    A message is erased by a rule when it is an exact copy (same session and
    message hash) of an erased message, or when it depends on the erased
    source: an assistant message's ``_supplied_sources`` or a native tool
    observation's ``_observation_sources`` names the source, in any version
    for a whole erasure or in the recorded version for a partial one. Testing
    every message against every rule made one forget on a long history cost
    seconds (source rows times rules); the index gives the same causes.
    """

    __slots__ = ('count', 'exact', 'whole', 'versions')

    def __init__(self, rules):
        self.count = 0
        self.exact: dict[str, dict[str, set[str]]] = {}
        self.whole: set[str] = set()
        self.versions: dict[str, set[str]] = {}
        for rule in rules:
            self.count += 1
            turn_id = rule['turn_id']
            by_hash = self.exact.setdefault(rule['session_id'], {})
            for message_hash in json.loads(rule['message_hashes_json']):
                by_hash.setdefault(message_hash, set()).add(turn_id)
            if rule['whole_source']:
                self.whole.add(turn_id)
            else:
                self.versions.setdefault(turn_id, set()).add(rule['source_version'])

    @classmethod
    def of(cls, rules):
        return rules if isinstance(rules, cls) else cls(rules)

    def causes(self, message, session_id, prior_hashes=()) -> set[str]:
        """The erasures that remove ``message``. ``prior_hashes`` are the identities of the other forms the
        message has had (computed in memory, never stored): as it arrived, before a secret in it was sealed, and
        as it was sealed before the owner released a value in it. An exact copy of an erased message is erased
        in any of its forms."""
        refs = (message.get('_supplied_sources', []) if message.get('role') == 'assistant' else
                message.get('_observation_sources', []) if message.get('role') == 'tool' and
                message.get('_native_tool_observation') == 'native-tool-observation-v1' else [])
        causes = set()
        if not self.count:
            return causes
        exact = self.exact.get(session_id)
        if exact:
            causes.update(exact.get(source_message_hash(session_id, message), ()))
            for prior_hash in prior_hashes:
                causes.update(exact.get(prior_hash, ()))
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            source = ref.get('source_id')
            if not isinstance(source, str):
                continue
            if source in self.whole:
                causes.add(source)
            elif source in self.versions:
                version = ref.get('source_version')
                if isinstance(version, str) and version in self.versions[source]:
                    causes.add(source)
        return causes


class TurnIdempotencyLedger:
    """SQLite reservation ledger safe across threads and sidecar processes."""

    def __init__(self, db_path: str | Path, *,
                 processing_lease_seconds: float = PROCESSING_LEASE_SECONDS) -> None:
        self.db_path = Path(db_path)
        self.processing_lease_seconds = float(processing_lease_seconds)
        from colonymind.memory import producers
        if producers.active():
            self._initialized = True
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_lock = threading.Lock()
        self._initialized = False
        self._initialize()

    @property
    def vault(self):
        """The sealed store next to this ledger (``secrets-vault.db`` in the same state directory)."""
        from colonymind.secrets.vault import get_vault
        return get_vault(self.db_path.parent)

    def _connect(self) -> sqlite3.Connection:
        from colonymind.memory import producers
        reader = producers.connection(self.db_path)
        if reader is not None:
            return reader
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        os.chmod(self.db_path, 0o600)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA secure_delete=ON")
        return conn

    def _read_connect(self, *, check_same_thread: bool = True) -> sqlite3.Connection:
        """Pure evidence reads need neither writer pragmas nor file-mode writes."""
        from colonymind.memory import producers
        reader = producers.connection(self.db_path)
        if reader is not None:
            return reader
        conn = sqlite3.connect(self.db_path.absolute().as_uri() + '?mode=ro',
                               uri=True, timeout=10.0, check_same_thread=check_same_thread)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def read_snapshot(self):
        """One read transaction on its own read-only connection, closed on exit.

        A nested use opens another connection. An error or an early return still
        rolls the transaction back and closes. A context worker keeps its
        deadline-bound connection.
        """
        with closing(self._read_connect(check_same_thread=False)) as conn:
            conn.execute('PRAGMA query_only=ON')
            try:
                conn.execute('BEGIN')
                yield conn
            finally:
                conn.rollback()

    def _initialize(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            with closing(self._connect()) as conn, conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=FULL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS turn_ingestion (
                        turn_id TEXT PRIMARY KEY,
                        content_sha256 TEXT NOT NULL,
                        state TEXT NOT NULL CHECK (
                            state IN ('processing', 'completed', 'ambiguous')
                        ),
                        response_json TEXT,
                        created_at TEXT NOT NULL DEFAULT (
                            strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                        ),
                        completed_at TEXT,
                        error TEXT
                    )
                    """
                )
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS turn_sources (
                        turn_id TEXT PRIMARY KEY,
                        content_sha256 TEXT NOT NULL,
                        contact_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        scope TEXT NOT NULL CHECK(scope IN ('person', 'session')),
                        messages_json TEXT NOT NULL,
                        occurred_at TEXT,
                        ingested_at TEXT NOT NULL DEFAULT (
                            strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                        )
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS source_observation_origin ON turn_sources
                    (contact_id, json_extract(messages_json, '$[0]._observation_sources[0].source_id'), turn_id)
                    WHERE json_extract(messages_json, '$[0]._native_tool_observation') = 'native-tool-observation-v1'
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS source_observation_tool_call ON turn_sources
                    (contact_id, json_extract(messages_json, '$[0].provenance.native.tool_call_id'))
                    WHERE json_extract(messages_json, '$[0]._native_tool_observation') = 'native-tool-observation-v1'
                """)
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS turn_source_search USING fts5(
                        turn_id UNINDEXED, role UNINDEXED, content,
                        tokenize='unicode61'
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS source_erasures (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        contact_id TEXT NOT NULL,
                        turn_id TEXT NOT NULL UNIQUE,
                        session_id TEXT NOT NULL,
                        message_hashes_json TEXT NOT NULL,
                        erased_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                    )
                """)
                # Keep predecessor tombstone semantics and cursor allocation.
                # A partial event uses an opaque event ID in source_erasures;
                # predecessor code still removes its exact message hashes but
                # does not mistake the surviving source for a whole tombstone.
                conn.execute('''CREATE TABLE IF NOT EXISTS source_erasure_revisions (
                    sequence INTEGER PRIMARY KEY, source_turn_id TEXT NOT NULL,
                    source_version TEXT NOT NULL, source_scope TEXT NOT NULL,
                    whole_source INTEGER NOT NULL CHECK(whole_source IN (0,1)),
                    UNIQUE(source_turn_id,source_version,whole_source))''')
                conn.execute("CREATE TABLE IF NOT EXISTS source_projection_erasures (turn_id TEXT NOT NULL, source_turn_id TEXT NOT NULL, PRIMARY KEY(turn_id, source_turn_id))")
                from .projection_backlog import initialize as initialize_backlog
                initialize_backlog(conn)
                from colonymind.beliefs.source_projection import initialize
                initialize(conn)
                from colonymind.turns.media import initialize as initialize_media
                initialize_media(conn)
                from colonymind.turns.source_vectors import initialize as initialize_vectors
                initialize_vectors(conn)
                from colonymind.memory.time_lane import initialize as initialize_time_lane
                initialize_time_lane(conn)
                from colonymind.self_model.judgments import initialize as initialize_judgments
                initialize_judgments(conn)
                from colonymind.self_model.appraisals import initialize as initialize_appraisals
                initialize_appraisals(conn)
                from colonymind.commitments.extract import initialize as initialize_commitments
                initialize_commitments(conn)
                from colonymind.turns.source_attribution import initialize as initialize_attribution
                initialize_attribution(conn)
                from colonymind.turns.source_annotations import initialize as initialize_annotations
                initialize_annotations(conn)
                from colonymind.turns.source_channels import initialize as initialize_channels
                initialize_channels(conn)
                from colonymind.pages.store import initialize as initialize_pages
                initialize_pages(conn)
                from colonymind.turns.group_chats import initialize as initialize_group_chats
                initialize_group_chats(conn)
            # Curated storage is optional. Its DDL must not roll back native initialization.
            try:
                from colonymind.memory.curated import initialize as initialize_curated
                with closing(self._connect()) as conn, conn:
                    conn.execute('BEGIN')
                    initialize_curated(conn)
                from colonymind.memory.curated import enable
                enable(self)
                from colonymind.memory.curated import backfill_source_fences
                backfill_source_fences(self)
            except Exception as exc:
                from colonymind.memory.curated import disable
                disable(self, exc)
                logger.warning("curated_backend_unavailable")
            self._initialize_keyword_stems()
            self._initialize_claim_keys()
            self._initialize_source_replies()
            self._initialized = True

    def _initialize_keyword_stems(self) -> None:
        """Create the stemmed keyword index (``turns.keywords``) in one transaction with its triggers and, on a store
        from before it, the queue of every source; then index what is queued (resumed after a build cut short)."""
        from . import keywords
        with closing(self._connect()) as conn:
            present = keywords.ready(conn)
        if not present:
            with closing(self._connect()) as conn, conn:
                conn.execute("BEGIN IMMEDIATE")
                keywords.initialize(conn)
        self._index_pending_stems()

    def _index_pending_stems(self, *, batch: int = 200) -> int:
        """Index the queued sources in the stemmed keyword index, ``batch`` to a transaction; the number indexed."""
        from . import keywords
        started, indexed = time.monotonic(), 0
        while True:
            with closing(self._connect()) as conn, conn:
                if not conn.execute(f"SELECT 1 FROM {keywords.PENDING} LIMIT 1").fetchone():
                    break
                conn.execute("BEGIN IMMEDIATE")
                rows = conn.execute(f"""SELECT p.turn_id, s.messages_json FROM {keywords.PENDING} AS p
                    LEFT JOIN turn_sources AS s ON s.turn_id = p.turn_id LIMIT ?""", (batch,)).fetchall()
                for turn_id, messages_json in rows:
                    try:
                        messages = json.loads(messages_json) if messages_json is not None else None
                        chunks = _stem_chunks(messages) if isinstance(messages, list) else None
                    except (ValueError, TypeError, KeyError, AttributeError):
                        logger.warning("stemmed keyword index: source %s is not readable; left unindexed", turn_id)
                        chunks = None
                    if chunks is None:
                        conn.execute(f"DELETE FROM {keywords.PENDING} WHERE turn_id=?", (turn_id,))
                        continue
                    keywords.index(conn, turn_id, chunks)
                    indexed += 1
        if indexed:
            logger.info("stemmed keyword index: indexed %d sources in %.2f s", indexed, time.monotonic() - started)
        return indexed

    def rebuild_keyword_stems(self) -> int:
        """Rebuild the stemmed keyword index from the canonical sources; the number of sources indexed. Running it
        again gives the same rows."""
        from . import keywords
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            keywords.queue_all(conn)
        return self._index_pending_stems()

    def merge_keyword_indexes(self, *, pages: int = 16, max_steps: int = 2000, pause: float = 0.05,
                              sleep=time.sleep, stop: threading.Event | None = None) -> dict[str, int]:
        """Merge the full-text keyword indexes' segments a step at a time; the steps taken per table.

        Deleting or rewriting a source leaves its old index entries behind until segments merge, and merges
        otherwise run only as new rows arrive, so a store whose sources are relexed or erased slows every keyword
        search as the dead entries pile up. Each step is one short write transaction that merges about ``pages``
        pages. The first step of a table uses a negative count, which starts a merge of segments of every level
        (the work ``optimize`` does in one long transaction); later steps use a positive count, which continues that
        merge rather than starting a new one when a writer adds a segment in between (FTS5 ``merge``). Writers wait
        for a step through the busy timeout; ``pause`` between steps lets them in. A table is done when a step
        changes nothing more (``total_changes`` moves by less than two, the documented test), after ``max_steps``,
        or when ``stop`` is set (checked before every step). The indexed rows are unchanged."""
        from . import keywords
        steps: dict[str, int] = {}
        for table in ("turn_source_search", keywords.TABLE):
            taken = 0
            while taken < max_steps and not (stop is not None and stop.is_set()):
                count = -abs(int(pages)) if taken == 0 else abs(int(pages))
                with closing(self._connect()) as conn, conn:
                    conn.execute("BEGIN IMMEDIATE")
                    before = conn.total_changes
                    conn.execute(f"INSERT INTO {table}({table}, rank) VALUES('merge', ?)", (count,))
                    progressed = conn.total_changes - before >= 2
                taken += 1
                if not progressed:
                    break
                sleep(pause)
            steps[table] = taken
        return steps

    def _initialize_claim_keys(self) -> None:
        """Create the claim key index (``turns.claim_keys``) in one transaction with its triggers and, on a store from
        before it, the queue of every claim; then index what is queued (resumed after a build cut short, and claims
        an older version wrote)."""
        from . import claim_keys
        with closing(self._connect()) as conn:
            present = claim_keys.ready(conn)
        if not present:
            with closing(self._connect()) as conn, conn:
                conn.execute("BEGIN IMMEDIATE")
                claim_keys.initialize(conn)
        self._index_pending_claim_keys()

    def _index_pending_claim_keys(self, *, batch: int = 500) -> int:
        """Index the queued claims in the claim key index, ``batch`` to a transaction; the number that now have
        rows."""
        from . import claim_keys
        started, indexed = time.monotonic(), 0
        while True:
            with closing(self._connect()) as conn, conn:
                conn.execute("BEGIN IMMEDIATE")
                taken, done = claim_keys.index_pending(conn, limit=batch)
            indexed += done
            if not taken:
                break
        if indexed:
            logger.info("claim key index: indexed %d claims in %.2f s", indexed, time.monotonic() - started)
        return indexed

    def rebuild_claim_keys(self) -> int:
        """Rebuild the claim key index from the claims; the number of claims that have rows. Running it again gives
        the same rows."""
        from . import claim_keys
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            claim_keys.queue_all(conn)
        return self._index_pending_claim_keys()

    def _initialize_source_replies(self) -> None:
        """Create the reply map (``turns.source_replies``) in one transaction with its triggers and, on a store from
        before it, the queue of every source; then derive what is queued (resumed after a build cut short, and
        sources an older version wrote)."""
        from . import source_replies
        with closing(self._connect()) as conn:
            present = source_replies.ready(conn)
        if not present:
            with closing(self._connect()) as conn, conn:
                conn.execute("BEGIN IMMEDIATE")
                source_replies.initialize(conn)
        self._index_pending_source_replies()

    def _index_pending_source_replies(self, *, batch: int = 200) -> int:
        """Derive the queued sources' rows in the reply map, ``batch`` to a transaction; the number derived."""
        from . import source_replies
        started, indexed = time.monotonic(), 0
        while True:
            with closing(self._connect()) as conn, conn:
                conn.execute("BEGIN IMMEDIATE")
                taken, done = source_replies.index_pending(conn, limit=batch)
            indexed += done
            if not taken:
                break
        if indexed:
            logger.info("reply map: indexed %d sources in %.2f s", indexed, time.monotonic() - started)
        return indexed

    def rebuild_source_replies(self) -> int:
        """Rebuild the reply map from the canonical sources; the number of sources derived. Running it again gives
        the same rows."""
        from . import source_replies
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            source_replies.queue_all(conn)
        return self._index_pending_source_replies()

    def index_pending(self) -> int:
        """Index what the derived indexes have queued: the stemmed keyword index, the claim key index and the reply
        map; the number of sources and claims indexed. The ledger does this as it writes and when it opens. A writer
        that changes the store in SQL calls it when it is done: the secrets migration and a release re-point every
        identity a rewritten message had, and each claim that moves with it is queued (``turns.claim_keys``), not
        located again until the queue is indexed."""
        return self._index_pending_stems() + self._index_pending_claim_keys() + self._index_pending_source_replies()

    def append_source_annotation(self, **kwargs):
        """Append an attributed correction without rewriting its original source."""
        from .source_annotations import append
        return append(self, **kwargs)

    def record_source(
        self, turn_id: str, *, contact_id: str, session_id: str,
        messages: list[dict[str, Any]], scope: str = "person",
        occurred_at: str | None = None,
        timezone_name: str | None = None,
        derive_claims: bool = True,
        channel_id: str | None = None,
        raw_hashes: Mapping[str, str] | None = None,
        staged=None,
        work=None,
        work_result=None,
    ) -> bool:
        """Atomically retain source JSON and its rebuildable lexical index.

        Checkpoint sources are session-scoped because a full transcript does
        not attest the speaker of every old message. Ordinary attributed turns
        can be recalled by that person across sessions. No inference or affect
        update occurs in this storage path. True means newly committed.

        Reviewed historical imports can set derive_claims=False to retain
        quotations without scheduling assertion learning. Text indexing still
        uses the same source ledger and semantic projection queue.

        Every writer's messages are sealed first (``colonymind.secrets.seal``),
        every text they carry: a detected secret's reference stands in the
        text before the digest, the insert, the lexical index and every queue
        (claims, opinions, appraisals, commitments, vectors). The values are
        staged and stored in the vault only once the source is admitted (with
        ``staged``, a caller's own staged values), so a replay of an erased
        or conflicting source changes nothing there. Erasure is checked on
        every form of each message: the sealed one, the one that arrived
        (``raw_hashes`` maps a caller's already sealed message to its arrival
        identity), the one its references resolve to and the one it had
        before a value was released, so an exact copy stays erased whenever
        and however its secret was sealed.
        """
        missing = [name for name, value in (("turn_id", turn_id), ("contact_id", contact_id),
                                            ("session_id", session_id), ("messages", messages)) if not value]
        if missing:
            raise ValueError("source requires " + ", ".join(missing))
        from colonymind.secrets.seal import Scope, Staged, redact_sensitive_messages
        stage = staged if staged is not None else Staged()   # a caller's stage: what is stored here leaves it
        arrived = messages
        messages = redact_sensitive_messages(
            arrived, scope=Scope(contact_id=contact_id, session_id=session_id, source_turn=turn_id),
            vault=self.vault, stage=stage)
        if work is not None:
            from colonymind.secrets.seal import map_leaves, seal_text
            from colonymind.work.models import WorkEnvelope
            envelope = map_leaves(work['envelope'], lambda value: seal_text(value,
                scope=Scope(contact_id=contact_id, session_id=session_id, source_turn=turn_id),
                vault=self.vault, stage=stage), skip=('source_refs', 'intended_sha256', 'actual_sha256'))
            work = {**work, 'envelope': WorkEnvelope.model_validate(envelope).model_dump(exclude_none=True)}
        prior: dict[str, set[str]] = {}
        for sealed_hash, raw_hash in (raw_hashes or {}).items():
            prior.setdefault(sealed_hash, set()).add(raw_hash)
        for before, after in zip(arrived, messages):
            if isinstance(before, Mapping) and before.get('content') != after.get('content'):
                prior.setdefault(source_message_hash(session_id, after), set()).add(
                    source_message_hash(session_id, before))
        if scope not in {"person", "session"}:
            raise ValueError("invalid source scope")
        if occurred_at is not None:
            if not isinstance(occurred_at, str) or len(occurred_at) > 64:
                raise ValueError("source occurrence time must be an ISO timestamp")
            occurred = datetime.fromisoformat(occurred_at)
            if occurred.tzinfo is None:
                raise ValueError("source occurrence time requires a timezone")
            occurred_at = occurred.astimezone(timezone.utc).isoformat()
        encoded = json.dumps(messages, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode()) > 8 * 1024 * 1024:
            raise ValueError("source exceeds 8 MiB")
        digest = canonical_turn_digest({
            "messages": messages, "contact_id": contact_id,
            "session_id": session_id, "scope": scope,
            "occurred_at": occurred_at,
        })
        messages = json.loads(encoded)  # Resolving lineage never mutates caller-owned input.
        # A retained value can have several sealed identities, including one
        # the vault already forgot. Resolve those aliases before reserving the
        # native writer; only their hashes enter the current source fence.
        from colonymind.secrets.vault import VaultUnavailable
        forms_error = None
        try:
            self._add_erasure_forms(prior, messages, session_id, contact_id, stage)
        except VaultUnavailable as error:
            forms_error = error
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            rules = _ErasureRules(self._erasure_rules(conn, contact_id))
            # Source IDs are globally unique. Reattributing an erased source
            # must not resurrect it under a different contact after restore or
            # a reviewed historical mapping change. Exact message-copy rules
            # below remain scoped to their original contact and session.
            if conn.execute("SELECT 1 FROM source_erasures WHERE turn_id=?", (turn_id,)).fetchone():
                raise SourceErased("source was erased")
            for message in messages:
                if message.get('_supplied_inputs'):
                    if message.get('role') != 'assistant':
                        raise ValueError('invalid_source_input_dependency')
                    resolved = self._resolve_input_dependencies(conn, contact_id, session_id,
                                                               message['_supplied_inputs'], turn_id=turn_id)
                    existing = message.get('_supplied_sources', [])
                    message['_supplied_sources'] = list({(ref['source_id'], ref['source_version']): ref
                                                         for ref in [*existing, *resolved]}.values())
            self._validate_dependencies(conn, turn_id, contact_id, session_id, messages)
            if forms_error is not None:
                from colonymind.memory import curated
                if rules.exact.get(session_id) or curated._state(
                        conn, self, contact_id, source_only=True).forgotten_hashes:
                    raise forms_error
            retained = self._retained_messages(messages, session_id, rules, prior)
            if retained != messages:
                for cause in self._erasure_causes(messages, session_id, rules, prior):
                    conn.execute("INSERT OR IGNORE INTO source_projection_erasures(turn_id,source_turn_id) VALUES (?,?)", (turn_id, cause))
                self._append_erasure(conn, turn_id=turn_id, contact_id=contact_id,
                    session_id=session_id, scope=scope, messages=messages,
                    removed=[message for message in messages if message not in retained], whole=False)
            messages = retained
            if not messages:
                raise SourceErased("source contains only erased messages")
            encoded = json.dumps(messages, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
            row = conn.execute("SELECT content_sha256,contact_id FROM turn_sources WHERE turn_id=?", (turn_id,)).fetchone()
            if row:
                if row[0] != digest:
                    raise ValueError("source id already contains different evidence")
                from .source_channels import record as record_channel
                # Exact replay must preserve a subsequent reviewed owner correction.
                record_channel(conn, turn_id=turn_id, contact_id=row[1],
                    messages=messages, occurred_at=occurred_at, channel_id=channel_id)
                if work is not None:
                    from colonymind.work.ledger import try_admit_turn
                    try_admit_turn(conn, result=work_result, work=work['envelope'], source_id=turn_id, contact_id=contact_id,
                        session_id=session_id, messages=messages, principal_id=work['principal_id'],
                        occurred_at=occurred_at or ledger_stamp())
                return False
            # Admitted: the values its retained messages carry are stored now, before its row (never those of a
            # message erasure removed, and nothing for a replay or a rejected source).
            kept = json.dumps(messages, ensure_ascii=True, sort_keys=True)
            stage.commit(only=lambda ref: ref in kept)
            from colonymind.turns.media import normalize_messages, SourceMedia
            messages = normalize_messages(conn, SourceMedia(self).store, turn_id, session_id, messages)
            encoded = json.dumps(messages, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
            conn.execute(
                "INSERT INTO turn_sources "
                "(turn_id, content_sha256, contact_id, session_id, scope, messages_json, occurred_at, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (turn_id, digest, contact_id, session_id, scope, encoded, occurred_at, ledger_stamp()),
            )
            from colonymind.memory.curated import index_source_fences
            index_source_fences(self, conn, turn_id, contact_id, session_id, messages, prior=prior)
            if work is not None:
                from colonymind.work.ledger import try_admit_turn
                admitted = try_admit_turn(conn, result=work_result, work=work['envelope'], source_id=turn_id, contact_id=contact_id,
                    session_id=session_id, messages=messages, principal_id=work['principal_id'],
                    occurred_at=occurred_at or ledger_stamp())
                if admitted:
                    admitted_work = json.dumps(work['envelope'], ensure_ascii=True, sort_keys=True)
                    stage.commit(only=lambda ref: ref in admitted_work)
            from .source_channels import record as record_channel
            record_channel(conn, turn_id=turn_id, contact_id=contact_id,
                messages=messages, occurred_at=occurred_at, channel_id=channel_id)
            self._index_messages(conn, turn_id, messages)
            from colonymind.beliefs.source_projection import enqueue
            if derive_claims:
                enqueue(conn, turn_id, messages, scope=scope, timezone_name=timezone_name)
            from colonymind.self_model.judgments import enqueue as enqueue_opinions
            enqueue_opinions(conn, turn_id, contact_id, messages, scope=scope, derive_claims=derive_claims)
            if derive_claims:
                from colonymind.self_model.appraisals import enqueue as enqueue_appraisals
                enqueue_appraisals(conn, turn_id, contact_id, messages, scope=scope)
            if derive_claims:
                from colonymind.commitments.extract import enqueue as enqueue_commitments
                enqueue_commitments(conn, turn_id, contact_id, messages, scope=scope, timezone_name=timezone_name)
            from colonymind.turns.source_vectors import enqueue as enqueue_vectors
            enqueue_vectors(conn, turn_id)
        if scope == "person":
            # The memory lanes' cost per captured message (``colonymind.usage``) counts this capture as a number
            # only, kept apart from the source, so a later erasure of the source never rewrites what it cost.
            from colonymind import usage
            usage.captured(sum(1 for message in messages if message.get("role") == "user"))
        return True

    @staticmethod
    def _index_messages(conn: sqlite3.Connection, turn_id: str, messages: list[dict[str, Any]]) -> None:
        """Index one source, already written, in both keyword tables (as written, and stemmed: ``turns.keywords``),
        and derive its rows in the reply map (``turns.source_replies``)."""
        chunks = _stem_chunks(messages)
        for role, chunk in chunks:
            conn.execute(
                "INSERT INTO turn_source_search(turn_id, role, content) VALUES (?, ?, ?)",
                (turn_id, role, chunk),
            )
        from .keywords import index as index_stems
        index_stems(conn, turn_id, chunks)
        from .source_replies import index as index_replies
        index_replies(conn, turn_id)


    @staticmethod
    def _erasure_rules(conn: sqlite3.Connection, contact_id: str) -> list[dict[str, Any]]:
        # Opaque whole-source events publish curated fences to hosts; native
        # capture retains history and uses the curated read fence instead.
        return [dict(row) for row in conn.execute(
            '''SELECT e.sequence,e.contact_id,e.session_id,e.message_hashes_json,
                coalesce(r.source_turn_id,e.turn_id) AS turn_id,
                coalesce(r.whole_source,1) AS whole_source,r.source_version
                FROM source_erasures e LEFT JOIN source_erasure_revisions r USING(sequence)
                WHERE e.contact_id=?
                AND (r.whole_source IS NOT 1 OR e.turn_id=r.source_turn_id)''', (contact_id,)
        )]

    def _add_erasure_forms(self, prior: dict[str, set[str]], messages, session_id: str, contact_id: str,
                           stage=None) -> None:
        """Every other identity each message has had (``seal.erasure_forms``: its references resolved, each value
        under every reference it has had, a released or a forgotten one included), added to ``prior`` in memory
        for the erasure check. Raises ``VaultUnavailable`` when a form cannot be worked out now (the key
        unavailable, or a value of this source withheld for that reason): an erasure that cannot be checked
        never lets its message in."""
        from colonymind.secrets.seal import erasure_forms
        from colonymind.secrets.vault import VaultUnavailable
        if stage is not None and getattr(stage, "withheld", False):
            raise VaultUnavailable("a value of this source was withheld, so its erasure cannot be checked now")
        for message in messages:
            if not isinstance(message, Mapping) or "content" not in message:
                continue
            forms = erasure_forms(message["content"], contact_id=contact_id, vault=self.vault, stage=stage)
            if forms:
                prior.setdefault(source_message_hash(session_id, message), set()).update(
                    source_message_hash(session_id, {"role": message.get("role"), "content": form}) for form in forms)

    @staticmethod
    def _retained_messages(messages: list[dict[str, Any]], session_id: str, rules,
                           prior: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        rules = _ErasureRules.of(rules)
        prior = prior or {}
        return [message for message in messages if not rules.causes(
            message, session_id, prior.get(source_message_hash(session_id, message), ()) if prior else ())]

    @staticmethod
    def _erasure_causes(messages, session_id, rules, prior: Mapping[str, Any] | None = None):
        rules = _ErasureRules.of(rules)
        prior = prior or {}
        causes = set()
        for message in messages:
            causes |= rules.causes(message, session_id,
                                   prior.get(source_message_hash(session_id, message), ()) if prior else ())
        return causes

    @staticmethod
    def _resolve_input_dependencies(conn, contact_id, session_id, refs, *, turn_id=''):
        if not isinstance(refs, list):
            raise ValueError('invalid_source_input_dependency')
        result = []
        for ref in refs:
            if (not isinstance(ref, dict) or set(ref) != {'source_id', 'input_message_hash'}
                    or not isinstance(ref['source_id'], str) or not 1 <= len(ref['source_id']) <= 256
                    or ref['source_id'] == turn_id or not isinstance(ref['input_message_hash'], str)
                    or not re.fullmatch('[0-9a-f]{64}', ref['input_message_hash'])):
                raise ValueError('invalid_source_input_dependency')
            # An old revision can have been erased while delivery was offline.
            # Check that exact input first, even when some parent content remains.
            erased = conn.execute('''SELECT e.message_hashes_json,r.source_version
                FROM source_erasures e JOIN source_erasure_revisions r USING(sequence)
                WHERE r.source_turn_id=? AND e.contact_id=?
                AND (r.source_scope='person' OR e.session_id=?) ORDER BY sequence''',
                (ref['source_id'], contact_id, session_id)).fetchall()
            version = next((row['source_version'] for row in erased
                            if ref['input_message_hash'] in json.loads(row['message_hashes_json'])), None)
            if version is None:
                parent = conn.execute('''SELECT * FROM turn_sources WHERE turn_id=? AND contact_id=?
                    AND NOT EXISTS (SELECT 1 FROM source_attribution_invalidations i
                        WHERE i.source_id=turn_sources.turn_id)''', (ref['source_id'], contact_id)).fetchone()
                if parent is None:
                    if conn.execute('SELECT 1 FROM turn_sources WHERE turn_id=?', (ref['source_id'],)).fetchone():
                        raise ValueError('invalid_source_input_dependency')
                    raise SourceInputPending('source_input_parent_pending')
                messages = json.loads(parent['messages_json'])
                if (not (parent['scope'] == 'person' or parent['session_id'] == session_id)
                        or not any(message.get('role') == 'user' and
                            source_message_hash(parent['session_id'], message) == ref['input_message_hash']
                            for message in messages)):
                    raise ValueError('invalid_source_input_dependency')
                version = canonical_turn_digest(messages)
            result.append({'source_id': ref['source_id'], 'source_version': version})
        return result

    def resolve_input_dependencies(self, *, contact_id, session_id, refs, turn_id=''):
        """Preflight before reserving effects; record_source rechecks atomically."""
        with closing(self._connect()) as conn:
            return self._resolve_input_dependencies(conn, contact_id, session_id, refs, turn_id=turn_id)

    @staticmethod
    def _validate_dependencies(conn, turn_id, contact_id, session_id, messages):
        for message in messages:
            refs = message.get('_supplied_sources', [])
            observation = message.get('_observation_sources', [])
            if observation:
                if (message.get('role') != 'tool' or refs
                        or message.get('_native_tool_observation') != 'native-tool-observation-v1'):
                    raise ValueError('invalid_source_dependency')
                refs = observation
            # The complete source envelope already has an 8 MiB bound. A
            # separate count cap would reject valid long native histories only
            # after generation, or tempt callers to silently drop dependencies.
            if not isinstance(refs, list) or (refs and message.get('role') != 'assistant' and not observation):
                raise ValueError('invalid_source_dependency')
            for ref in refs:
                if (not isinstance(ref, dict) or set(ref) != {'source_id', 'source_version'}
                        or not isinstance(ref['source_id'], str) or not 1 <= len(ref['source_id']) <= 256
                        or ref['source_id'] == turn_id or not isinstance(ref['source_version'], str)
                        or not re.fullmatch('[0-9a-f]{64}', ref['source_version'])):
                    raise ValueError('invalid_source_dependency')
                parent = conn.execute('''SELECT * FROM turn_sources WHERE turn_id=? AND contact_id=?
                    AND NOT EXISTS (SELECT 1 FROM source_attribution_invalidations i
                        WHERE i.source_id=turn_sources.turn_id)''',
                                      (ref['source_id'], contact_id)).fetchone()
                if (parent and (parent['scope'] == 'person' or parent['session_id'] == session_id)
                        and canonical_turn_digest(json.loads(parent['messages_json'])) == ref['source_version']):
                    continue
                # A delayed answer may truthfully reference the exact revision
                # already erased. Accept its envelope, then redact that answer.
                erased = conn.execute('''SELECT 1 FROM source_erasures e JOIN source_erasure_revisions r USING(sequence)
                    WHERE r.source_turn_id=? AND e.contact_id=? AND r.source_version=?
                    AND (r.source_scope='person' OR e.session_id=?)''',
                    (ref['source_id'], contact_id, ref['source_version'], session_id)).fetchone()
                if not erased:
                    raise ValueError('invalid_source_dependency')

    @staticmethod
    def _append_erasure(conn, *, turn_id, contact_id, session_id, scope, messages, removed, whole, event_id=None):
        version = canonical_turn_digest(messages)
        if conn.execute('''SELECT 1 FROM source_erasure_revisions WHERE
                source_turn_id=? AND source_version=? AND whole_source=?''', (turn_id, version, int(whole))).fetchone():
            return False
        hashes = set() if event_id is not None else {h for row in conn.execute('''SELECT e.message_hashes_json FROM source_erasures e
            LEFT JOIN source_erasure_revisions r USING(sequence)
            WHERE coalesce(r.source_turn_id,e.turn_id)=?''', (turn_id,))
                  for h in json.loads(row[0])}
        for message in removed:
            # Clients can resend either the stored form or the original media
            # row. The helper accepts only a valid preserved input identity.
            hashes.add(source_message_hash(session_id, {
                'role': message.get('role'), 'content': message.get('content')}))
            hashes.add(source_message_hash(session_id, message))
        event_id = event_id or (turn_id if whole else 'erased-message:' + uuid.uuid4().hex)
        if not whole:
            while conn.execute('SELECT 1 FROM turn_sources WHERE turn_id=? UNION SELECT 1 FROM source_erasures WHERE turn_id=?',
                               (event_id, event_id)).fetchone():
                event_id = 'erased-message:' + uuid.uuid4().hex
        cursor = conn.execute('''INSERT OR IGNORE INTO source_erasures
            (contact_id,turn_id,session_id,message_hashes_json) VALUES (?,?,?,?)''',
            (contact_id, event_id, session_id, json.dumps(sorted(hashes))))
        if cursor.rowcount:
            conn.execute('''INSERT INTO source_erasure_revisions
                (sequence,source_turn_id,source_version,source_scope,whole_source) VALUES (?,?,?,?,?)''',
                (cursor.lastrowid, turn_id, version, scope, int(whole)))
            return True
        return False

    def source_references(self, turn_ids, *, contact_id, session_id):
        """Structured selected-source revisions, never parsed from generated prose."""
        with closing(self._connect()) as conn:
            return self._source_references(conn, turn_ids, contact_id=contact_id, session_id=session_id)

    @staticmethod
    def _source_references(conn, turn_ids, *, contact_id, session_id):
        refs = []
        for turn_id in dict.fromkeys(turn_ids):
            row = conn.execute('''SELECT messages_json FROM turn_sources WHERE turn_id=?
                AND contact_id=? AND (scope='person' OR session_id=?)
                AND NOT EXISTS (SELECT 1 FROM source_attribution_invalidations i
                    WHERE i.source_id=turn_sources.turn_id)''',
                (turn_id, contact_id, session_id)).fetchone()
            if row:
                refs.append({'source_id': turn_id, 'source_version': canonical_turn_digest(json.loads(row[0]))})
        return refs

    def is_source_erased(self, turn_id: str, contact_id: str | None = None) -> bool:
        with closing(self._read_connect()) as conn:
            sql, args = "SELECT 1 FROM source_erasures WHERE turn_id=?", [turn_id]
            if contact_id is not None:
                sql += " AND contact_id=?"
                args.append(contact_id)
            return conn.execute(sql, args).fetchone() is not None

    def is_projection_erased(self, turn_id: str) -> bool:
        with closing(self._read_connect()) as conn:
            return conn.execute("SELECT 1 FROM source_projection_erasures WHERE turn_id=?", (turn_id,)).fetchone() is not None

    def retained_messages(self, *, contact_id: str, session_id: str, messages: list[dict[str, Any]],
                          raw_hashes: Mapping[str, str] | None = None, staged=None) -> list[dict[str, Any]]:
        """The messages no erasure removes, checked in every form each has had (``record_source``). Raises
        ``VaultUnavailable`` when an exact erasure of the session cannot be checked now."""
        with closing(self._connect()) as conn:
            rules = _ErasureRules(self._erasure_rules(conn, contact_id))
        prior: dict[str, set[str]] = {sealed: {raw} for sealed, raw in (raw_hashes or {}).items()}
        if rules.exact.get(session_id):
            self._add_erasure_forms(prior, messages, session_id, contact_id, staged)
        return self._retained_messages(messages, session_id, rules, prior)

    def has_exact_erasures(self, *, session_id: str) -> bool:
        """Whether an erasure by exact message names anything in this session, for any contact (its check then
        needs the sealed store)."""
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT message_hashes_json FROM source_erasures WHERE session_id=?",
                                (session_id,)).fetchall()
        for (hashes,) in rows:
            try:
                if json.loads(hashes or "[]"):
                    return True
            except ValueError:
                return True
        return False

    def erasure_watermark(self, contact_id: str) -> int:
        from colonymind.memory import curated
        with closing(self._connect()) as conn:
            return curated.watermark(self, contact_id, conn=conn)

    def erasure_feed(self, contact_id: str, after: int = 0, limit: int = 250) -> dict[str, Any]:
        with closing(self._connect()) as conn:
            head = conn.execute("SELECT coalesce(max(sequence), 0) FROM source_erasures WHERE contact_id=?", (contact_id,)).fetchone()[0]
            if after > head:
                raise ValueError("erasure cursor exceeds server history; restore requires reconciliation")
            rows = conn.execute('''SELECT e.*,r.source_turn_id,r.source_version,r.source_scope,coalesce(r.whole_source,1) AS whole_source
                FROM source_erasures e LEFT JOIN source_erasure_revisions r USING(sequence)
                WHERE e.contact_id=? AND e.sequence>? ORDER BY e.sequence LIMIT ?''', (contact_id, after, max(1, min(limit, 500)))).fetchall()
            events = [{"sequence": row["sequence"],
                       "turn_id": (row['source_turn_id'] or row['turn_id']) if row['whole_source'] else row['turn_id'],
                       "event_id": row['turn_id'], "session_id": row["session_id"],
                       "message_hashes": json.loads(row["message_hashes_json"]),
                       "whole_source": bool(row['whole_source']),
                       'source_version': None if row['source_scope'] == 'curated-item' else row['source_version'],
                       'source_turn_id': None if row['source_scope'] == 'curated-item' else row['source_turn_id'] or row['turn_id']}
                      for row in rows]
            through = events[-1]["sequence"] if events else after
            return {"contact_id": contact_id, "head": head, "through": through, "events": events, "complete": through == head}


    def search_sources(self, query: str, *, contact_id: str, session_id: str, limit: int=5) -> list[dict[str, Any]]:
        """Recall direct source excerpts inside the already-authorized viewer."""
        from colonymind.memory import curated
        return curated.fenced_read(self, lambda: self._read_search_sources(query, contact_id=contact_id, session_id=session_id, limit=limit),
                                  empty=[])

    def _read_search_sources(
        self, query: str, *, contact_id: str, session_id: str, limit: int = 5,
    ) -> list[dict[str, Any]]:
        if not contact_id:
            return []
        from colonymind.turns.audio import evidence_metadata
        from . import keywords

        with closing(self._connect()) as conn:
            # Identifiers, then the question's words: its rarest when it has more than the lane searches.
            expression = keywords.match_expression(keywords.query_terms(conn, query))
            if expression is None:
                return []

            @lru_cache(maxsize=8)
            def canonical_chunks(turn_id):
                # Reconstruct one source envelope (capped at 8 MiB) only when a
                # ranked candidate from that turn is consumed, never per FTS match.
                row = conn.execute("SELECT session_id, messages_json FROM turn_sources WHERE turn_id=?",
                                   (turn_id,)).fetchone()
                chunks, messages = {}, {}
                for message, text, chunk in _lexical_chunks(json.loads(row["messages_json"]) if row else []):
                    # A long message can have thousands of chunks. Hash its original
                    # envelope and recover its modality once, not once per chunk.
                    key = id(message)
                    if key not in messages:
                        messages[key] = (source_message_hash(row["session_id"], message), evidence_metadata(message))
                    message_hash, metadata = messages[key]
                    owners = chunks.setdefault((message.get('role'), chunk), {})
                    owners[message_hash] = {'source_message_hash': message_hash,
                        **metadata,
                        **({'excerpt_truncated': True} if chunk != text else {})}
                return {key: list(owners.values()) for key, owners in chunks.items()}

            candidates = _scoped_matches(
                conn, table="turn_source_search", expression=expression, contact_id=contact_id,
                session_id=session_id, joins="CROSS JOIN turn_sources AS s ON s.turn_id=f.turn_id",
                select="f.turn_id, f.role, f.content, s.contact_id, s.session_id, s.scope, s.occurred_at, "
                       "s.ingested_at")
            # Candidates are ranked in SQL and consumed lazily. Exact chunk
            # ownership and duplicate projection rows are still resolved before a
            # row spends the canonical result budget, so stale rows never do; this
            # also reads predecessor-written indexes without a migration or rebuild.
            budget, result, seen = max(1, min(limit, 20)), [], set()
            for row in candidates:
                value = {key: row[key] for key in row.keys() if key != "rowid"}
                for ownership in canonical_chunks(value['turn_id']).get((value['role'], value['content']), ()):
                    key = (value['turn_id'], value['role'], value['content'], ownership['source_message_hash'])
                    if key not in seen:
                        seen.add(key)
                        candidate = {**value, **ownership}
                        from colonymind.memory import curated
                        if curated.is_forgotten(self, candidate, conn=conn):
                            continue
                        result.append(candidate)
                        if len(result) == budget:
                            return result
        return result

    def search_stems(self, query: str, *, contact_id: str, session_id: str, limit: int=5) -> list[dict[str, Any]]:
        """The stemmed keyword lane (``turns.keywords``): excerpts shaped as ``search_sources`` returns them, found in
        the stemmed index with CJK pieces, inside the already-authorized viewer.

        A row names its chunk by place (source, speaker, index among the source's chunks); the excerpt is read back
        from the canonical source, and a row whose indexed text is not the rewrite of the chunk at its place never
        spends the result budget."""
        from colonymind.memory import curated
        return curated.fenced_read(self, lambda: self._read_search_stems(query, contact_id=contact_id, session_id=session_id, limit=limit),
                                  empty=[])

    def _read_search_stems(
        self, query: str, *, contact_id: str, session_id: str, limit: int = 5,
    ) -> list[dict[str, Any]]:
        if not contact_id:
            return []
        from colonymind.turns.audio import evidence_metadata
        from . import keywords

        with closing(self._connect()) as conn:
            expression = keywords.match_expression(keywords.query_terms(conn, query, stemmed=True))
            if expression is None:
                return []

            @lru_cache(maxsize=8)
            def canonical_chunks(turn_id):
                # One source's chunks in order, each with its message's identity: read only when a ranked row
                # from that source is consumed.
                row = conn.execute("SELECT session_id, messages_json FROM turn_sources WHERE turn_id=?",
                                   (turn_id,)).fetchone()
                chunks, messages = [], {}
                for message, text, chunk in _lexical_chunks(json.loads(row["messages_json"]) if row else []):
                    key = id(message)
                    if key not in messages:
                        messages[key] = (source_message_hash(row["session_id"], message), evidence_metadata(message))
                    message_hash, metadata = messages[key]
                    chunks.append((message.get("role"), chunk, {
                        'source_message_hash': message_hash, **metadata,
                        **({'excerpt_truncated': True} if chunk != text else {})}))
                return chunks

            candidates = _scoped_matches(
                conn, table=keywords.TABLE, expression=expression, contact_id=contact_id, session_id=session_id,
                joins=f"CROSS JOIN {keywords.CHUNKS} AS m ON m.id = f.rowid CROSS JOIN turn_sources AS s ON s.turn_id = m.turn_id",
                select="m.turn_id, m.role, m.chunk, f.content AS indexed, s.contact_id, s.session_id, s.scope, "
                       "s.occurred_at, s.ingested_at")
            budget, result, seen = max(1, min(limit, 20)), [], set()
            for row in candidates:
                chunks = canonical_chunks(row['turn_id'])
                if not 0 <= row['chunk'] < len(chunks):
                    continue
                role, chunk, ownership = chunks[row['chunk']]
                if role != row['role'] or keywords.cjk_bigrams(chunk) != row['indexed']:
                    continue
                key = (row['turn_id'], role, chunk, ownership['source_message_hash'])
                if key in seen:
                    continue
                seen.add(key)
                candidate = {'turn_id': row['turn_id'], 'role': role, 'content': chunk,
                               **{name: row[name] for name in ('contact_id', 'session_id', 'scope', 'occurred_at',
                                                               'ingested_at')},
                               **ownership}
                from colonymind.memory import curated
                if curated.is_forgotten(self, candidate, conn=conn):
                    continue
                result.append(candidate)
                if len(result) == budget:
                    return result
        return result

    def search_claim_keys(self, query: str, *, contact_id: str, session_id: str, limit: int=5) -> list[dict[str, Any]]:
        """The claim key lane (``turns.claim_keys``): the places claims are stated whose key (subject, predicate,
        value) matches ``query``, inside the already-authorized viewer, each as a source excerpt shaped as
        ``search_sources`` returns them: the words of the message the claim quotes (its evidence), read back from the
        canonical source. The key only locates; recall expands the excerpt into the claim's grounded bundle.

        A row whose indexed text is not the key of the claim at its place, or whose quotation is no longer the
        message's words at the claim's span, never spends the result budget."""
        from colonymind.memory import curated
        return curated.fenced_read(self, lambda: self._read_search_claim_keys(query, contact_id=contact_id, session_id=session_id, limit=limit),
                                  empty=[])

    def _read_search_claim_keys(
        self, query: str, *, contact_id: str, session_id: str, limit: int = 5,
    ) -> list[dict[str, Any]]:
        if not contact_id:
            return []
        from colonymind.turns.audio import evidence_metadata, source_text
        from . import claim_keys, keywords

        with closing(self._connect()) as conn:
            # The stemmed lane's choice of terms: the key index is written the same way.
            expression = keywords.match_expression(keywords.query_terms(conn, query, stemmed=True))
            if expression is None:
                return []

            @lru_cache(maxsize=8)
            def messages(turn_id):
                # One source's messages by identity: read only when a ranked row from that source is consumed.
                row = conn.execute("SELECT session_id, messages_json FROM turn_sources WHERE turn_id=?",
                                   (turn_id,)).fetchone()
                return {source_message_hash(row["session_id"], message): message
                        for message in json.loads(row["messages_json"])} if row else {}

            candidates = _scoped_matches(
                conn, table=claim_keys.TABLE, expression=expression, contact_id=contact_id, session_id=session_id,
                joins=f"CROSS JOIN {claim_keys.ROWS} AS k ON k.id = f.rowid CROSS JOIN turn_sources AS s ON s.turn_id = k.turn_id",
                select="k.claim_id, k.turn_id, k.message_hash, f.content AS indexed, s.contact_id, s.session_id, "
                       "s.scope, s.occurred_at, s.ingested_at")
            budget, result, seen = max(1, min(limit, 20)), [], set()
            for row in candidates:
                found = claim_keys.located(conn, row["claim_id"], row["turn_id"], row["message_hash"])
                if found is None or claim_keys.key_text(*found) != row["indexed"]:
                    continue
                data = found[0]
                message = messages(row["turn_id"]).get(row["message_hash"])
                if message is None or message.get("role") != "user":
                    continue
                text, evidence = source_text(message.get("content")), data.get("evidence")
                start, end = data.get("span_start"), data.get("span_end")
                if (not isinstance(evidence, str) or not evidence.strip() or not isinstance(start, int)
                        or not isinstance(end, int) or text[start:end] != evidence):
                    continue
                key = (row["turn_id"], evidence, row["message_hash"])
                if key in seen:
                    continue
                seen.add(key)
                candidate = {"turn_id": row["turn_id"], "role": "user", "content": evidence,
                               **{name: row[name] for name in ("contact_id", "session_id", "scope", "occurred_at",
                                                               "ingested_at")},
                               "source_message_hash": row["message_hash"], **evidence_metadata(message),
                               **({"excerpt_truncated": True} if evidence != text else {})}
                from colonymind.memory import curated
                if curated.is_forgotten(self, dict(candidate, source_claim_id=row['claim_id']), conn=conn):
                    continue
                result.append(candidate)
                if len(result) == budget:
                    return result
        return result

    def reserve(self, turn_id: str, content_sha256: str) -> Reservation:
        turn_id = (turn_id or "").strip()
        if not turn_id or len(turn_id) > 256:
            raise ValueError("turn_id must contain 1..256 non-whitespace characters")
        if len(content_sha256) != 64:
            raise ValueError("content_sha256 must be a SHA-256 hex digest")

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT content_sha256, state, response_json, "
                "(julianday('now') - julianday(created_at)) * 86400.0 AS age_seconds "
                "FROM turn_ingestion WHERE turn_id=?",
                (turn_id,),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO turn_ingestion "
                    "(turn_id, content_sha256, state) VALUES (?, ?, 'processing')",
                    (turn_id, content_sha256),
                )
                reservation = Reservation(ReservationOutcome.CREATED)
            elif row["content_sha256"] != content_sha256:
                reservation = Reservation(ReservationOutcome.CONFLICT)
            elif row["state"] == "completed":
                response = None
                if row["response_json"]:
                    response = json.loads(row["response_json"])
                reservation = Reservation(ReservationOutcome.REPLAYED, response=response)
            elif row["state"] == "ambiguous":
                reservation = Reservation(ReservationOutcome.AMBIGUOUS)
            elif (row["age_seconds"] is not None
                    and row["age_seconds"] > self.processing_lease_seconds):
                # The first attempt never completed or reported failure within
                # its lease: its process is gone. Renew the lease under the same
                # transaction so exactly one identical retry takes it over.
                conn.execute(
                    "UPDATE turn_ingestion "
                    "SET created_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), error=NULL "
                    "WHERE turn_id=? AND state='processing'",
                    (turn_id,),
                )
                reservation = Reservation(ReservationOutcome.CREATED)
            else:
                reservation = Reservation(ReservationOutcome.IN_PROGRESS)
            conn.commit()
            return reservation
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def complete(
        self,
        turn_id: str,
        content_sha256: str,
        response: Mapping[str, Any],
    ) -> None:
        response_json = json.dumps(
            dict(response), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        with closing(self._connect()) as conn, conn:
            changed = conn.execute(
                """
                UPDATE turn_ingestion
                SET state='completed', response_json=?,
                    completed_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    error=NULL
                WHERE turn_id=? AND content_sha256=? AND state='processing'
                """,
                (response_json, turn_id, content_sha256),
            ).rowcount
            if changed != 1:
                row = conn.execute(
                    "SELECT content_sha256, state FROM turn_ingestion WHERE turn_id=?",
                    (turn_id,),
                ).fetchone()
                if not row or row["content_sha256"] != content_sha256:
                    raise RuntimeError("turn reservation changed before completion")
                if row["state"] != "completed":
                    raise RuntimeError(f"turn cannot complete from state {row['state']}")

    def mark_ambiguous(
        self,
        turn_id: str,
        content_sha256: str,
        error: BaseException,
    ) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                UPDATE turn_ingestion
                SET state='ambiguous', error=?
                WHERE turn_id=? AND content_sha256=? AND state='processing'
                """,
                (f"{type(error).__name__}: {error}"[:1000], turn_id, content_sha256),
            )

    def release(self, turn_id: str, content_sha256: str) -> None:
        """Give up a reservation whose attempt failed before it stored anything it cannot repeat, so the next
        identical request is taken in as new rather than refused as ambiguous. Only a caller whose every effect is
        an idempotent ledger write may release (``turns.observed``); a turn's effects are not."""
        with closing(self._connect()) as conn, conn:
            conn.execute("DELETE FROM turn_ingestion WHERE turn_id=? AND content_sha256=? AND state='processing'",
                         (turn_id, content_sha256))

    def get(self, turn_id: str) -> Optional[dict[str, Any]]:
        """Read-only reconciliation view used by tests and future tooling."""
        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT * FROM turn_ingestion WHERE turn_id=?", (turn_id,)
            ).fetchone()
        return dict(row) if row else None


@lru_cache(maxsize=16)
def _cached_ledger(db_path: str) -> TurnIdempotencyLedger:
    return TurnIdempotencyLedger(db_path)


KEYWORD_MERGE_INTERVAL_S = 6 * 3600.0


async def keyword_index_maintenance(ledger, *, sleep=None, first_delay: float = 600.0,
                                    interval: float = KEYWORD_MERGE_INTERVAL_S) -> None:
    """The sidecar's keyword index task: :meth:`TurnIdempotencyLedger.merge_keyword_indexes` off the event loop,
    every ``interval`` seconds. ``ledger`` is the ledger or a callable returning it, called on each pass so an
    unreadable ledger is a logged failed pass, not a failed start. Cancelling the task stops the merge between steps.
    Runs until cancelled; a failed pass is logged and the next one runs on schedule."""
    import asyncio
    sleep = sleep or asyncio.sleep
    await sleep(first_delay)
    while True:
        stop = threading.Event()
        try:
            current = ledger() if callable(ledger) and not hasattr(ledger, "merge_keyword_indexes") else ledger
            started = time.monotonic()
            steps = await asyncio.to_thread(current.merge_keyword_indexes, stop=stop)
            if any(count > 1 for count in steps.values()):
                logger.info("keyword index merge: %s steps in %.2f s", steps, time.monotonic() - started)
        except asyncio.CancelledError:
            stop.set()
            raise
        except Exception as error:
            logger.warning("keyword index merge failed (%s)", type(error).__name__)
        await sleep(interval)


def get_turn_idempotency_ledger(state_dir: str | Path) -> TurnIdempotencyLedger:
    path = Path(state_dir).resolve() / "turn-idempotency.db"
    from colonymind.memory import producers
    if producers.active():
        # A read-only facade must never populate the writer's initialized-store cache.
        return TurnIdempotencyLedger(path)
    return _cached_ledger(str(path))
