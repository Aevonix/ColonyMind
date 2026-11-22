"""The recorder's store: one table of lines in one small SQLite file.

A line is one row of code labels and timings. ``ref`` is the key of a row in
another of this instance's stores; nothing from those stores is copied here.

Every call opens its own short-lived connection, so any thread and any
process can write. A write waits at most ``BUSY_S`` for a lock. ``write`` and
``read`` never raise: a call that fails changes nothing and is owed as one
``error`` line, written by the next write that works.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
import uuid
from collections import deque
from contextlib import closing
from pathlib import Path

DB_FILE = "recorder.db"
BUSY_S = 0.25  # the longest a call waits for another writer
OWED = 2000  # failures kept in memory until a write works

SCHEMA = """
CREATE TABLE IF NOT EXISTS lines (
    id       TEXT PRIMARY KEY,
    kind     TEXT NOT NULL,
    source   TEXT NOT NULL,
    ref      TEXT NOT NULL DEFAULT '',
    choice   TEXT NOT NULL DEFAULT '',
    chooser  TEXT NOT NULL DEFAULT '',
    result   TEXT NOT NULL DEFAULT '',
    detail   TEXT NOT NULL DEFAULT '',
    marks    TEXT NOT NULL DEFAULT '',
    began_at REAL NOT NULL,
    took_ms  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_lines_began ON lines(began_at);
CREATE INDEX IF NOT EXISTS idx_lines_source ON lines(source, began_at);
CREATE INDEX IF NOT EXISTS idx_lines_ref ON lines(ref) WHERE ref != '';
CREATE INDEX IF NOT EXISTS idx_lines_open ON lines(kind, source, began_at) WHERE result = '';
"""

INSERT = (
    "INSERT INTO lines (id, kind, source, ref, choice, chooser, result, detail, marks, began_at, took_ms) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def new_line(
    kind: str,
    source: str,
    *,
    ref: str = "",
    choice: str = "",
    chooser: str = "",
    result: str = "",
    detail: str = "",
    began_at: float | None = None,
    took_ms: int | None = None,
    line_id: str = "",
) -> tuple:
    """The values of one new row, in ``INSERT`` order. The first is its id."""
    began = time.time() if began_at is None else began_at
    return (line_id or uuid.uuid4().hex, kind, source, ref, choice, chooser, result, detail, "", began, took_ms)


class Store:
    """The file of lines. ``write`` and ``read`` never raise."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._ready = False
        self._owed: deque[tuple] = deque(maxlen=OWED)
        self.pruned_on: int | None = None  # the day, in days since the epoch, old lines were last dropped

    def _connect(self) -> sqlite3.Connection:
        created = not self.path.exists()
        if created:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=BUSY_S)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA synchronous=NORMAL")
            if created or not self._ready:
                with self._lock:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.executescript(SCHEMA)
                    self._ready = True
            if created:
                try:
                    os.chmod(self.path, 0o600)
                except OSError:
                    pass
        except BaseException:
            conn.close()
            raise
        return conn

    def owe(self, source: str, error: BaseException) -> None:
        """Remember one failure of the recorder itself: it becomes an ``error`` line with the next write that works."""
        with self._lock:
            self._owed.append(new_line("error", source, result="failed", detail=type(error).__name__))

    def write(self, sql: str, params: tuple = ()) -> int:
        """Run one statement in its own transaction and return how many rows it changed (0 when it failed)."""
        try:
            with closing(self._connect()) as conn, conn:
                count = conn.execute(sql, params).rowcount
                with self._lock:
                    owed = list(self._owed)
                conn.executemany(INSERT, owed)
            with self._lock:
                for row in owed:
                    if row in self._owed:
                        self._owed.remove(row)
            return count
        except Exception as error:  # noqa: BLE001 - recording never fails the caller
            self.owe("recorder.store", error)
            return 0

    def rows(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """The rows of one query. Raises when the store cannot be read."""
        with closing(self._connect()) as conn:
            return conn.execute(sql, params).fetchall()

    def read(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """The rows of one query (none when it failed)."""
        try:
            return self.rows(sql, params)
        except Exception as error:  # noqa: BLE001
            self.owe("recorder.store", error)
            return []

    def prune(self, before: float) -> int:
        """Drop the lines begun before ``before`` (epoch seconds). Returns how many."""
        return self.write("DELETE FROM lines WHERE began_at < ?", (before,))


_STORES: dict[str, Store] = {}


def store_at(state_dir: str | os.PathLike[str]) -> Store:
    """The store of one state directory: one object per file, so what it owes is kept."""
    path = Path(state_dir) / DB_FILE
    store = _STORES.get(str(path))
    if store is None:
        store = _STORES.setdefault(str(path), Store(path))
    return store
