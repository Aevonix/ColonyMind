"""The claim key index: the words of each claim's key, where they locate evidence (docs/SOURCE-CLAIMS.md, Recall and
time; docs/RECALL-HYBRID.md, Keyword lanes).

A claim is reachable only when the words or the meaning of the message that states it match a question. The index
holds each claim's subject, predicate (its underscores read as spaces) and value, written with English stemming
(``porter unicode61``) after every run of CJK ideographs, kana or hangul is rewritten as its overlapping two-character
pieces, as the stemmed keyword lane writes messages (``turns.keywords``). So "who is my employer" reaches
``employer: Acme`` from "I just started at Acme", whose words it does not share. It helps only a question that uses the
key's words: "where do I work" shares none with ``employer``.

A row locates one place a claim is stated: the claim, its source and its message (the claim's anchor, and each
witness message of a folded claim). The key only locates. The recall lane (``TurnIdempotencyLedger.search_claim_keys``)
gives the message's words the claim quotes as a source hit, and recall expands it as any other: what is shown is the
grounded assertion bundle, never the key. A row whose indexed text is not the key of the claim at its place now never
reaches the candidates.

The rows follow the claims in SQL, whoever changes them. Triggers on ``source_claims`` and ``source_claim_witnesses``
delete a claim's rows when the claim or one of its witnesses is deleted or moved, and queue the claim; a trigger on
``turn_sources`` deletes the rows of a deleted source. Rows are written only from the claims as they are, when the
queue is drained: at every claim commit, at the end of an erasure, when a writer that changes the store in SQL is done
(``TurnIdempotencyLedger.index_pending``: the secrets migration and a release, whose re-pointed messages move their
claims), and when the ledger opens (the upgrade build, and claims an older version wrote). A rejected claim is never
in ``source_claims``, so it has none. A retracted or superseded claim keeps its rows: expansion picks the value
current at the question's time.

An episode gets no rows: its subject and predicate are fixed words and its value is the reported message itself,
which the keyword lanes already search.
"""
from __future__ import annotations

import json
import sqlite3

from .keywords import cjk_bigrams

TABLE = "claim_key_search"
ROWS = "claim_key_rows"
PENDING = "claim_key_pending"


def _forget(claim: str) -> str:
    return (f"DELETE FROM {TABLE} WHERE rowid IN (SELECT id FROM {ROWS} WHERE claim_id={claim}); "
            f"DELETE FROM {ROWS} WHERE claim_id={claim};")


def _queue(claim: str) -> str:
    return f"INSERT OR IGNORE INTO {PENDING}(claim_id) VALUES ({claim});"


_SCHEMA = (
    f"CREATE VIRTUAL TABLE IF NOT EXISTS {TABLE} USING fts5(content, tokenize='porter unicode61')",
    f"CREATE TABLE IF NOT EXISTS {ROWS} (id INTEGER PRIMARY KEY AUTOINCREMENT, claim_id TEXT NOT NULL, "
    "turn_id TEXT NOT NULL, message_hash TEXT NOT NULL)",
    f"CREATE INDEX IF NOT EXISTS {ROWS}_claim ON {ROWS}(claim_id)",
    f"CREATE INDEX IF NOT EXISTS {ROWS}_source ON {ROWS}(turn_id)",
    f"CREATE TABLE IF NOT EXISTS {PENDING} (claim_id TEXT PRIMARY KEY)",
    f"CREATE TRIGGER IF NOT EXISTS {TABLE}_claim_new AFTER INSERT ON source_claims BEGIN {_queue('new.id')} END",
    f"CREATE TRIGGER IF NOT EXISTS {TABLE}_claim_change AFTER UPDATE OF id, turn_id, message_hash, subject_key, "
    f"predicate, data_json ON source_claims BEGIN {_forget('old.id')} {_queue('new.id')} END",
    f"CREATE TRIGGER IF NOT EXISTS {TABLE}_claim_gone AFTER DELETE ON source_claims "
    f"BEGIN {_forget('old.id')} DELETE FROM {PENDING} WHERE claim_id=old.id; END",
    f"CREATE TRIGGER IF NOT EXISTS {TABLE}_witness_new AFTER INSERT ON source_claim_witnesses "
    f"BEGIN {_queue('new.claim_id')} END",
    f"CREATE TRIGGER IF NOT EXISTS {TABLE}_witness_change AFTER UPDATE OF claim_id, turn_id, message_hash, data_json "
    f"ON source_claim_witnesses BEGIN {_forget('old.claim_id')} {_queue('old.claim_id')} {_queue('new.claim_id')} END",
    f"CREATE TRIGGER IF NOT EXISTS {TABLE}_witness_gone AFTER DELETE ON source_claim_witnesses "
    f"BEGIN {_forget('old.claim_id')} {_queue('old.claim_id')} END",
    f"CREATE TRIGGER IF NOT EXISTS {TABLE}_source_gone AFTER DELETE ON turn_sources BEGIN "
    f"DELETE FROM {TABLE} WHERE rowid IN (SELECT id FROM {ROWS} WHERE turn_id=old.turn_id); "
    f"DELETE FROM {ROWS} WHERE turn_id=old.turn_id; END",
)
_OBJECTS = {TABLE, ROWS, PENDING, f"{ROWS}_claim", f"{ROWS}_source", f"{TABLE}_claim_new", f"{TABLE}_claim_change",
            f"{TABLE}_claim_gone", f"{TABLE}_witness_new", f"{TABLE}_witness_change", f"{TABLE}_witness_gone",
            f"{TABLE}_source_gone"}


def ready(conn: sqlite3.Connection) -> bool:
    """Whether the index and everything that keeps it are in the store."""
    names = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE name IN (%s)" % ",".join("?" * len(_OBJECTS)), sorted(_OBJECTS))}
    return names == _OBJECTS


def initialize(conn: sqlite3.Connection) -> bool:
    """Create the index, its location rows, its queue and the triggers that keep them with the claims, in the
    caller's transaction. True when the index was created now: then every existing claim is queued."""
    created = not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE,)).fetchone()
    for statement in _SCHEMA:
        conn.execute(statement)
    if created:
        conn.execute(f"DELETE FROM {ROWS}")
        conn.execute(f"INSERT OR IGNORE INTO {PENDING}(claim_id) SELECT id FROM source_claims")
    return created


def key_text(data, predicate) -> str:
    """The indexed words of one place a claim is stated: its subject and value as that message's claim states them,
    and the claim's predicate with its underscores read as spaces, CJK runs rewritten as their pieces. Empty for an
    episode (see the module docstring)."""
    if not isinstance(data, dict) or data.get("representation") == "episode":
        return ""
    parts = (data.get("subject"), str(predicate or "").replace("_", " "), data.get("value"))
    return cjk_bigrams(" ".join(part.strip() for part in parts if isinstance(part, str) and part.strip()))


def located(conn: sqlite3.Connection, claim_id: str, turn_id: str, message_hash: str):
    """``(data, predicate)`` of the claim as the message at ``(turn_id, message_hash)`` states it: the claim's own
    record when it is anchored there, else that witness's; None when the claim is not stated there."""
    row = conn.execute("SELECT data_json, predicate, subject_key FROM source_claims WHERE id=? AND turn_id=? "
                       "AND message_hash=?", (claim_id, turn_id, message_hash)).fetchone()
    if row is None:
        row = conn.execute("SELECT w.data_json, c.predicate, c.subject_key FROM source_claim_witnesses w "
                           "JOIN source_claims c ON c.id=w.claim_id WHERE w.claim_id=? AND w.turn_id=? "
                           "AND w.message_hash=?", (claim_id, turn_id, message_hash)).fetchone()
    if row is None or str(row[2] or "").startswith("episode:"):
        return None
    try:
        data = json.loads(row[0])
    except (TypeError, ValueError):
        return None
    return (data, row[1]) if isinstance(data, dict) else None


def index(conn: sqlite3.Connection, claim_id: str) -> bool:
    """Write one claim's rows from the claim as it is, in place of any it had, and take it off the queue: one row
    for its anchor and one for each witness message. False when it has none (it is gone, or an episode)."""
    forget(conn, claim_id)
    conn.execute(f"DELETE FROM {PENDING} WHERE claim_id=?", (claim_id,))
    anchor = conn.execute("SELECT turn_id, message_hash FROM source_claims WHERE id=?", (claim_id,)).fetchone()
    if anchor is None:
        return False
    places = [(anchor[0], anchor[1]), *((row[0], row[1]) for row in conn.execute(
        "SELECT turn_id, message_hash FROM source_claim_witnesses WHERE claim_id=? ORDER BY turn_id, message_hash",
        (claim_id,)))]
    written = False
    for turn_id, message_hash in dict.fromkeys(places):
        found = located(conn, claim_id, turn_id, message_hash)
        text = key_text(*found) if found is not None else ""
        if not text:
            continue
        row = conn.execute(f"INSERT INTO {ROWS}(claim_id, turn_id, message_hash) VALUES (?, ?, ?)",
                           (claim_id, turn_id, message_hash))
        conn.execute(f"INSERT INTO {TABLE}(rowid, content) VALUES (?, ?)", (row.lastrowid, text))
        written = True
    return written


def index_pending(conn: sqlite3.Connection, *, limit: int | None = None) -> tuple[int, int]:
    """Index the queued claims (at most ``limit``), in the caller's transaction: ``(taken, indexed)``, the claims
    taken off the queue and those that now have rows."""
    sql = f"SELECT claim_id FROM {PENDING} ORDER BY claim_id" + (" LIMIT ?" if limit is not None else "")
    queued = [row[0] for row in conn.execute(sql, (limit,) if limit is not None else ())]
    return len(queued), sum(index(conn, claim_id) for claim_id in queued)


def forget(conn: sqlite3.Connection, claim_id: str) -> None:
    """Delete one claim's rows."""
    conn.execute(f"DELETE FROM {TABLE} WHERE rowid IN (SELECT id FROM {ROWS} WHERE claim_id=?)", (claim_id,))
    conn.execute(f"DELETE FROM {ROWS} WHERE claim_id=?", (claim_id,))


def queue_all(conn: sqlite3.Connection) -> None:
    """Empty the index and queue every claim (a rebuild from the claims)."""
    conn.execute(f"DELETE FROM {TABLE}")
    conn.execute(f"DELETE FROM {ROWS}")
    conn.execute(f"INSERT OR IGNORE INTO {PENDING}(claim_id) SELECT id FROM source_claims")


__all__ = ["TABLE", "ROWS", "PENDING", "ready", "initialize", "key_text", "located", "index", "index_pending",
           "forget", "queue_all"]
