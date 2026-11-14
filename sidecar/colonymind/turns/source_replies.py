"""The reply map: from a user message to the replies that name it as their direct input (docs/RECALL-HYBRID.md,
Returning no useful memory: "Conversation pairs").

A reply names its input in one of the two ways ``SourceClaimProjection._direct_input`` reads: it is the next message
of its own source after a user message, or its ``_supplied_inputs`` names one user message of an earlier source. The
second reference is written only inside the reply, so finding the reply of a found user message would mean reading
every later source. This map is the reverse, derived from each source when it is written: ``(input source, input
message hash, reply source, reply message hash)``. It only proposes. Recall checks each reply it names as the forward
reading does, and pairs a user message only with its single reply.

A source derives only the rows in which it is the reply, and only when it is a person's own source: a checkpoint's
copies are never paired. The rows follow the sources in SQL, whoever changes them: triggers on ``turn_sources`` delete
every row that names a source (as input or as reply) when the source is deleted or its messages change, and queue every
new or changed source. The ledger derives a source's rows when it writes the source, and what is still queued when it
opens (the upgrade build, a build cut short, sources an older version wrote). A reply that names an earlier version of
its input no longer reads as its direct reply, so losing its row when the input changes loses nothing.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

TABLE = "source_replies"
PENDING = "source_replies_pending"

_FORGET = (f"DELETE FROM {TABLE} WHERE input_turn=old.turn_id OR reply_turn=old.turn_id; "
           f"DELETE FROM {PENDING} WHERE turn_id=old.turn_id;")
_QUEUE = f"INSERT OR IGNORE INTO {PENDING}(turn_id) VALUES (new.turn_id);"
_SCHEMA = (
    f"CREATE TABLE IF NOT EXISTS {TABLE} (input_turn TEXT NOT NULL, input_hash TEXT NOT NULL, "
    "reply_turn TEXT NOT NULL, reply_hash TEXT NOT NULL, PRIMARY KEY (input_turn, input_hash, reply_turn, reply_hash))",
    f"CREATE INDEX IF NOT EXISTS {TABLE}_reply ON {TABLE}(reply_turn)",
    f"CREATE TABLE IF NOT EXISTS {PENDING} (turn_id TEXT PRIMARY KEY)",
    f"CREATE TRIGGER IF NOT EXISTS {TABLE}_queue AFTER INSERT ON turn_sources BEGIN {_QUEUE} END",
    f"CREATE TRIGGER IF NOT EXISTS {TABLE}_change AFTER UPDATE OF turn_id, messages_json ON turn_sources "
    f"BEGIN {_FORGET} {_QUEUE} END",
    f"CREATE TRIGGER IF NOT EXISTS {TABLE}_forget AFTER DELETE ON turn_sources BEGIN {_FORGET} END",
)
_OBJECTS = {TABLE, PENDING, f"{TABLE}_reply", f"{TABLE}_queue", f"{TABLE}_change", f"{TABLE}_forget"}
#: The one read of the map: the replies that name one user message. It is answered from the primary key's index.
REPLIES_SQL = (f"SELECT reply_turn, reply_hash FROM {TABLE} WHERE input_turn=? AND input_hash=? "
               "ORDER BY reply_turn, reply_hash")


def ready(conn: sqlite3.Connection) -> bool:
    """Whether the map and everything that keeps it are in the store."""
    names = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE name IN (%s)" % ",".join("?" * len(_OBJECTS)), sorted(_OBJECTS))}
    return names == _OBJECTS


def initialize(conn: sqlite3.Connection) -> bool:
    """Create the map, its queue and the triggers that keep them with ``turn_sources``, in the caller's transaction.
    True when the map was created now: then every existing source is queued."""
    created = not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE,)).fetchone()
    for statement in _SCHEMA:
        conn.execute(statement)
    if created:
        conn.execute(f"INSERT OR IGNORE INTO {PENDING}(turn_id) SELECT turn_id FROM turn_sources")
    return created


def _named_input(message: dict[str, Any]):
    """The one ``(source, message hash)`` a reply's ``_supplied_inputs`` names; None when it names none or several."""
    refs = message.get("_supplied_inputs")
    if not isinstance(refs, list) or len(refs) != 1 or not isinstance(refs[0], dict):
        return None
    ref = refs[0]
    if set(ref) != {"source_id", "input_message_hash"} or not all(isinstance(value, str) for value in ref.values()):
        return None
    return ref["source_id"], ref["input_message_hash"]


def derive(turn_id: str, session_id: str, messages) -> list[tuple[str, str, str, str]]:
    """The rows one person-scope source derives: each assistant message with the user message it answers, the one
    its ``_supplied_inputs`` names, or else the message before it in the same source."""
    from .idempotency import source_message_hash
    rows = []
    for index, message in enumerate(messages if isinstance(messages, list) else []):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        reply = source_message_hash(session_id, message)
        if message.get("_supplied_inputs") is not None:
            named = _named_input(message)
            if named is not None:
                rows.append((*named, turn_id, reply))
        elif index and isinstance(messages[index - 1], dict) and messages[index - 1].get("role") == "user":
            rows.append((turn_id, source_message_hash(session_id, messages[index - 1]), turn_id, reply))
    return rows


def index(conn: sqlite3.Connection, turn_id: str) -> bool:
    """Derive one source's rows from the source as it is, in place of any it had, and take it off the queue. False
    when the source is gone."""
    conn.execute(f"DELETE FROM {TABLE} WHERE reply_turn=?", (turn_id,))
    conn.execute(f"DELETE FROM {PENDING} WHERE turn_id=?", (turn_id,))
    row = conn.execute("SELECT session_id, scope, messages_json FROM turn_sources WHERE turn_id=?",
                       (turn_id,)).fetchone()
    if row is None:
        return False
    if row[1] == "person":
        try:
            messages = json.loads(row[2])
        except (TypeError, ValueError):
            messages = None
        conn.executemany(f"INSERT OR IGNORE INTO {TABLE}(input_turn, input_hash, reply_turn, reply_hash) "
                         "VALUES (?, ?, ?, ?)", derive(turn_id, row[0], messages))
    return True


def index_pending(conn: sqlite3.Connection, *, limit: int | None = None) -> tuple[int, int]:
    """Derive the queued sources' rows (at most ``limit``), in the caller's transaction: ``(taken, indexed)``, the
    sources taken off the queue and those still in the store."""
    sql = f"SELECT turn_id FROM {PENDING} ORDER BY turn_id" + (" LIMIT ?" if limit is not None else "")
    queued = [row[0] for row in conn.execute(sql, (limit,) if limit is not None else ())]
    return len(queued), sum(index(conn, turn_id) for turn_id in queued)


def replies(conn: sqlite3.Connection, input_turn: str, input_hash: str) -> list[tuple[str, str]]:
    """The ``(reply source, reply message hash)`` pairs the map holds for one user message, unchecked."""
    return [(row[0], row[1]) for row in conn.execute(REPLIES_SQL, (input_turn, input_hash))]


def queue_all(conn: sqlite3.Connection) -> None:
    """Empty the map and queue every source (a rebuild from the canonical sources)."""
    conn.execute(f"DELETE FROM {TABLE}")
    conn.execute(f"INSERT OR IGNORE INTO {PENDING}(turn_id) SELECT turn_id FROM turn_sources")


__all__ = ["TABLE", "PENDING", "REPLIES_SQL", "ready", "initialize", "derive", "index", "index_pending", "replies",
           "queue_all"]
