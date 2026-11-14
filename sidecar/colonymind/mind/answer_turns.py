"""``answer_turns``: every owner message the answer route read, once (docs/MIND.md, Asks).

A table in ``mind.db``. One row per message, keyed ``<platform>:<chat>:<message id>`` (a delivery id is unique only
within its chat on some platforms; else ``turn:<turn id>``), with its turn id indexed too: a message that reaches
the sidecar by two routes (the plugin before the model runs, the turn's ingest after it) is read once, whichever
came first. A row keeps what the reading did, never the message's words: the kind (``answered``, ``undone``,
``clarified``, ``shown``, ``later``, ``unbound``, ``conversation``), the path and the reader, the asks it bound and a
result with each ask's id and status. ``answered`` rows are what an undo looks for (in the same chat, arrived inside
``answers.UNDO_MINUTES`` before the undo itself); ``unbound`` rows are the asks report's "clean yes or no bound to no
ask". Rows older than ``RETENTION`` are dropped (my choice: past any undo, clarification or list window, and the
report's week).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

TABLE = "answer_turns"
RETENTION = timedelta(days=30)
# What a read did; those in ``AS_ANSWER`` mean "read as an answer" (the question and outreach readers skip the turn).
# ``revoked``: the owner's "ask me again about <thing>" took standing allows back.
ANSWERED, UNDONE, CLARIFIED, SHOWN, LATER, REVOKED, UNBOUND, CONVERSATION = (
    "answered", "undone", "clarified", "shown", "later", "revoked", "unbound", "conversation")
AS_ANSWER = frozenset({ANSWERED, UNDONE, CLARIFIED, SHOWN, LATER, REVOKED})
KINDS = (ANSWERED, UNDONE, CLARIFIED, SHOWN, LATER, REVOKED, UNBOUND, CONVERSATION)

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
  key         TEXT PRIMARY KEY,
  turn_id     TEXT,
  platform    TEXT,
  chat_id     TEXT,
  message_id  TEXT,
  target      TEXT,
  arrived_at  REAL,
  read_at     REAL,
  route       TEXT,
  kind        TEXT,
  path        TEXT,
  reader      TEXT,
  asks_json   TEXT,
  result_json TEXT,
  undone_at   REAL
);
CREATE INDEX IF NOT EXISTS idx_{TABLE}_turn ON {TABLE}(turn_id);
CREATE INDEX IF NOT EXISTS idx_{TABLE}_chat ON {TABLE}(platform, chat_id, read_at);
CREATE INDEX IF NOT EXISTS idx_{TABLE}_read ON {TABLE}(read_at);
"""


def _ts(value: datetime) -> float:
    return (value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)).timestamp()


def _dt(value: Any) -> Optional[datetime]:
    return datetime.fromtimestamp(float(value), timezone.utc) if value is not None else None


@dataclass
class Read:
    """One message as the answer route read it."""
    key: str
    turn_id: str = ""
    platform: str = ""
    chat_id: str = ""
    message_id: str = ""
    target: str = ""                 # what it reacted or replied to (a delivery id), if anything
    arrived_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    route: str = "answers"           # answers (the plugin, before the model) | ingest (the turn, after it)
    kind: str = CONVERSATION
    path: str = ""
    reader: str = ""
    asks: List[str] = field(default_factory=list)
    result: Dict[str, Any] = field(default_factory=dict)
    undone_at: Optional[datetime] = None

    @property
    def as_answer(self) -> bool:
        return self.kind in AS_ANSWER

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Read":
        def loads(value: Any, default: Any) -> Any:
            try:
                return json.loads(value) if value else default
            except ValueError:
                return default
        return cls(key=row["key"], turn_id=row["turn_id"] or "", platform=row["platform"] or "",
                   chat_id=row["chat_id"] or "", message_id=row["message_id"] or "", target=row["target"] or "",
                   arrived_at=_dt(row["arrived_at"]), read_at=_dt(row["read_at"]), route=row["route"] or "",
                   kind=row["kind"] or CONVERSATION, path=row["path"] or "", reader=row["reader"] or "",
                   asks=list(loads(row["asks_json"], [])), result=dict(loads(row["result_json"], {})),
                   undone_at=_dt(row["undone_at"]))


class AnswerTurns:
    """The table over a ``mind.db`` connection (the workspace's own, shared across the mind)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._lock = threading.Lock()
        with self._lock:
            conn.executescript(SCHEMA)
            conn.commit()

    def _rows(self, query: str, params: Sequence[Any]) -> List[Read]:
        with self._lock:
            cursor = self.conn.execute(query, list(params))
            cursor.row_factory = sqlite3.Row
            return [Read.from_row(row) for row in cursor.fetchall()]

    def find(self, key: str, *, turn_id: str = "") -> Optional[Read]:
        """The read of this message: by its key, else by its turn id (the other route's key)."""
        rows = self._rows(f"SELECT * FROM {TABLE} WHERE key = ? OR (? != '' AND turn_id = ?) ORDER BY read_at LIMIT 1",
                          [key, turn_id or "", turn_id or ""])
        return rows[0] if rows else None

    def for_turn(self, turn_id: str) -> Optional[Read]:
        if not turn_id:
            return None
        rows = self._rows(f"SELECT * FROM {TABLE} WHERE turn_id = ? ORDER BY read_at LIMIT 1", [turn_id])
        return rows[0] if rows else None

    def record(self, read: Read) -> bool:
        """Write one read; False when the message was read already (its key taken)."""
        with self._lock:
            cursor = self.conn.execute(
                f"INSERT OR IGNORE INTO {TABLE} (key, turn_id, platform, chat_id, message_id, target, arrived_at, "
                "read_at, route, kind, path, reader, asks_json, result_json, undone_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [read.key, read.turn_id or None, read.platform, read.chat_id, read.message_id, read.target,
                 _ts(read.arrived_at) if read.arrived_at else None, _ts(read.read_at) if read.read_at else None,
                 read.route, read.kind, read.path, read.reader, json.dumps(list(read.asks)),
                 json.dumps(read.result, ensure_ascii=False, default=str), None])
            self.conn.commit()
            return cursor.rowcount == 1

    def approvals(self, *, platform: str, chat_id: str, since: datetime, before: datetime) -> List[Read]:
        """The messages in this chat that arrived from ``since`` to just before ``before`` (an undo's own arrival:
        it takes back only what came before it, however late either is read), approved something and were not
        undone, newest first."""
        return [row for row in self.answered(platform=platform, chat_id=chat_id, since=since, before=before)
                if row.result.get("approved")]

    def answered(self, *, platform: str, chat_id: str, since: datetime, before: datetime) -> List[Read]:
        """The messages in this chat that arrived from ``since`` to just before ``before`` and settled an ask (a yes
        or a no), not undone, newest first."""
        return self._rows(f"SELECT * FROM {TABLE} WHERE platform = ? AND chat_id = ? AND arrived_at >= ? "
                          "AND arrived_at < ? AND kind = ? AND undone_at IS NULL "
                          "ORDER BY arrived_at DESC, read_at DESC",
                          [platform, chat_id, _ts(since), _ts(before), ANSWERED])

    def approved_by_reaction(self, *, platform: str, chat_id: str, target: str, since: datetime,
                             before: datetime) -> List[Read]:
        """The reactions on ``target`` (a delivery id in this chat) that arrived from ``since`` to just before
        ``before`` (the reaction's change or removal), approved something and were not undone."""
        rows = self._rows(f"SELECT * FROM {TABLE} WHERE platform = ? AND chat_id = ? AND target = ? "
                          "AND path = 'reaction' AND arrived_at >= ? AND arrived_at < ? AND kind = ? "
                          "AND undone_at IS NULL ORDER BY arrived_at DESC, read_at DESC",
                          [platform, chat_id, target, _ts(since), _ts(before), ANSWERED])
        return [row for row in rows if row.result.get("approved")]

    def mark_undone(self, key: str, at: datetime) -> None:
        with self._lock:
            self.conn.execute(f"UPDATE {TABLE} SET undone_at = ? WHERE key = ?", [_ts(at), key])
            self.conn.commit()

    def events(self, since: datetime) -> List[Dict[str, Any]]:
        """The asks report's message-level events since ``since``: each clean yes or no bound to no open ask."""
        return [{"kind": "unbound_answer", "at": row.read_at}
                for row in self._rows(f"SELECT * FROM {TABLE} WHERE kind = ? AND read_at >= ? ORDER BY read_at",
                                      [UNBOUND, _ts(since)])]

    def prune(self, before: datetime) -> int:
        with self._lock:
            cursor = self.conn.execute(f"DELETE FROM {TABLE} WHERE read_at < ?", [_ts(before)])
            self.conn.commit()
            return cursor.rowcount


__all__ = ["ANSWERED", "AS_ANSWER", "AnswerTurns", "CLARIFIED", "CONVERSATION", "KINDS", "LATER", "RETENTION",
           "REVOKED", "Read", "SHOWN", "TABLE", "UNBOUND", "UNDONE"]
