"""The sole durable store of owner recall query vectors, retained for thirty days.

Writes require a currently loaded query adapter. The store contains float32 vectors,
turn ids and source dependencies, never query text. Source erasure can be bound after
the runtime's listener, or called by a row source after it has captured training refs.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
from datetime import datetime, timedelta
import os
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any, Callable, Iterable

import numpy as np

from colonymind.util.temporal import now_utc


DB_FILE = "query-vectors.db"
# Plan assumption (recall-1, D22): derived query vectors are retained thirty days.
RETENTION_DAYS = 30
RETENTION = timedelta(days=RETENTION_DAYS)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_vectors (
    turn_id TEXT PRIMARY KEY,
    audience TEXT NOT NULL CHECK (audience = 'owner'),
    created_at REAL NOT NULL,
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    vector BLOB NOT NULL,
    CHECK (length(vector) = dimension * 4)
);
CREATE INDEX IF NOT EXISTS query_vectors_created ON query_vectors(created_at);
CREATE TABLE IF NOT EXISTS query_sources (
    turn_id TEXT NOT NULL REFERENCES query_vectors(turn_id) ON DELETE CASCADE,
    source_id TEXT NOT NULL,
    PRIMARY KEY (turn_id, source_id)
);
CREATE INDEX IF NOT EXISTS query_sources_source ON query_sources(source_id);
"""


class QueryLog:
    """An owner-only derived SQLite store, created lazily on an admitted write.

    ``runtime.find('query_adapter')`` is checked for every write. With no runtime,
    no loaded learner, or a non-owner audience, recording is a non-mutating no-op.
    Reads remain available to the learner after it unloads, until retention expires.

    By default the learner's row source owns erasure ordering: capture row refs,
    call ``erase_sources``, then return the refs to the runtime for unserving and
    rebuilding. Set ``bind_erasure=True`` only when the runtime's ref resolver has
    already been bound, or derives its refs independently of this store.
    """

    def __init__(self, path: Path, *, runtime: Any = None,
                 clock: Callable[[], datetime] | None = None, bind_erasure: bool = False):
        self.path = Path(path)
        self.runtime = runtime
        self._clock = clock or now_utc
        self._lock = RLock()
        self._unbind: Callable[[], None] | None = None
        if bind_erasure:
            self.bind_erasure()

    def bind_erasure(self) -> None:
        """Bind an idempotent source-erasure listener after the runtime's resolver."""
        from ..turns.erasure_hooks import bind_source_erasure

        with self._lock:
            if self._unbind is None:
                self._unbind = bind_source_erasure(self.erase_sources)

    def close(self) -> None:
        """Unbind the listener; SQLite connections are scoped to each operation."""
        with self._lock:
            if self._unbind is not None:
                self._unbind()
                self._unbind = None

    def sources_current(self, sources: Iterable[str]) -> bool:
        """Retain query history while withholding withdrawn training evidence."""
        from colonymind import get_state_dir
        from colonymind.memory import curated

        directory = getattr(self.runtime, 'data_dir', None)
        path = Path(directory or self.path.parent) / 'turn-idempotency.db'
        if not path.is_file():
            path = get_state_dir() / 'turn-idempotency.db'
        if not path.is_file():
            return True
        with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            sources = list(sources)
            return curated.fenced_read(conn,
                lambda: not any(curated.is_forgotten(conn, source) for source in sources), empty=False)

    @staticmethod
    def _sources(conn, turn_id):
        return (turn_id, *(row[0] for row in conn.execute(
            'SELECT source_id FROM query_sources WHERE turn_id=?', (turn_id,))))

    @contextmanager
    def _transaction(self, *, create: bool = False):
        with self._lock:
            if not create and not self.path.exists():
                yield None
                return
            if create:
                self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
                os.close(fd)
            self.path.chmod(0o600)
            # mode=rw prevents a read racing a removed file from recreating it.
            connection = sqlite3.connect(self.path.resolve().as_uri() + "?mode=rw", uri=True)
            with closing(connection) as conn, conn:
                # Deleted and overwritten vectors must not survive in free pages,
                # WAL files or persistent rollback journals after the operation.
                conn.execute("PRAGMA journal_mode=DELETE")
                conn.execute("PRAGMA secure_delete=ON")
                conn.execute("PRAGMA foreign_keys=ON")
                conn.execute("PRAGMA temp_store=MEMORY")
                if create:
                    conn.executescript(_SCHEMA)
                conn.execute("BEGIN IMMEDIATE")
                yield conn

    def _prune(self, conn: sqlite3.Connection) -> int:
        cutoff = (self._clock() - RETENTION).timestamp()
        return conn.execute("DELETE FROM query_vectors WHERE created_at <= ?", (cutoff,)).rowcount

    def record(self, turn_id: str, vector: Any, *, audience: str | None,
               source_ids: Iterable[str] = ()) -> bool:
        """Record one float32 query per turn only while an owner adapter is loaded.

        Re-recording a turn replaces its vector and dependencies while preserving
        its original retention deadline. Returns whether a record was written.
        """
        if (audience != "owner" or self.runtime is None
                or getattr(self.runtime, "enabled", True) is not True
                or self.runtime.find("query_adapter") is None):
            return False
        if not isinstance(turn_id, str) or not turn_id.strip():
            raise ValueError("query log requires a turn id")
        values = np.asarray(vector, dtype="<f4")
        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
            raise ValueError("query vector must be a nonempty finite one-dimensional vector")
        sources = tuple(sorted(set(source_ids)))
        if any(not isinstance(source, str) or not source.strip() for source in sources):
            raise ValueError("query log requires nonempty source ids")
        if not self.sources_current((turn_id, *sources)):
            return False
        with self._transaction(create=True) as conn:
            self._prune(conn)
            conn.execute("""INSERT INTO query_vectors (turn_id, audience, created_at, dimension, vector)
                            VALUES (?, 'owner', ?, ?, ?)
                            ON CONFLICT(turn_id) DO UPDATE SET dimension=excluded.dimension,
                                                               vector=excluded.vector""",
                         (turn_id, self._clock().timestamp(), values.size, values.tobytes()))
            conn.execute("DELETE FROM query_sources WHERE turn_id=?", (turn_id,))
            conn.executemany("INSERT INTO query_sources (turn_id, source_id) VALUES (?, ?)",
                             ((turn_id, source) for source in sources))
        return True

    def get(self, turn_id: str, *, audience: str | None = "owner") -> np.ndarray | None:
        """Return a copy of a retained owner's query vector, or None."""
        if audience != "owner":
            return None
        with self._transaction() as conn:
            if conn is None:
                return None
            self._prune(conn)
            row = conn.execute("SELECT vector FROM query_vectors WHERE turn_id=? AND audience='owner'",
                               (turn_id,)).fetchone()
            if row is not None and not self.sources_current(self._sources(conn, turn_id)):
                return None
        return None if row is None else np.frombuffer(row[0], dtype="<f4").copy()

    def turn_ids(self) -> tuple[str, ...]:
        """Retained recall ids in stable order, for the learner's row source."""
        with self._transaction() as conn:
            if conn is None:
                return ()
            self._prune(conn)
            return tuple(row[0] for row in conn.execute("SELECT turn_id FROM query_vectors ORDER BY turn_id")
                         if self.sources_current(self._sources(conn, row[0])))

    @staticmethod
    def _resting(conn: sqlite3.Connection, ids: Iterable[str]) -> tuple[str, ...]:
        ids = tuple(dict.fromkeys(ids))
        found: set[str] = set()
        # Builder assumption: batches of 400 ids keep both IN clauses below the
        # older SQLite limit of 999 parameters, even for large erasure closures.
        for offset in range(0, len(ids), 400):
            batch = ids[offset:offset + 400]
            marks = ",".join("?" for _ in batch)
            found.update(row[0] for row in conn.execute(
                f"SELECT turn_id FROM query_vectors WHERE turn_id IN ({marks}) "
                f"UNION SELECT turn_id FROM query_sources WHERE source_id IN ({marks})", batch + batch))
        return tuple(sorted(found))

    def refs_resting_on(self, ids: Iterable[str]) -> tuple[str, ...]:
        """Recall turn ids depending on the named turns or passage sources.

        The learner maps these ids to its training row refs before deleting them.
        This covers retained vectors only: retention also removes their source
        mappings. A learner must additionally resolve retained manifest row refs
        from their original provenance, including rows whose vector has expired.
        A log-only lookup cannot establish that learned weights are clean.
        """
        with self._transaction() as conn:
            if conn is None:
                return ()
            self._prune(conn)
            return self._resting(conn, ids)

    def erase_sources(self, ids: Iterable[str]) -> int:
        """Securely remove queries for named turns or sources; safe to repeat."""
        with self._transaction() as conn:
            if conn is None:
                return 0
            turns = self._resting(conn, ids)
            conn.executemany("DELETE FROM query_vectors WHERE turn_id=?", ((turn,) for turn in turns))
            return len(turns)

    def prune(self) -> int:
        """Securely remove vectors at least thirty days old without creating a DB."""
        with self._transaction() as conn:
            return 0 if conn is None else self._prune(conn)


__all__ = ["DB_FILE", "RETENTION_DAYS", "RETENTION", "QueryLog"]
