"""Verify the owner turn behind a hand ask without changing its delivery route."""

from contextlib import closing
import json
import sqlite3

from colonymind.hands.store import DB_FILE
from colonymind.hands.wire import MessageRef
from colonymind.turns.source_attribution import is_invalidated


def initialize(conn):
    """Keep verified request links with the asks, independent of controller retention."""
    conn.execute("""CREATE TABLE IF NOT EXISTS hand_ask_origins (
        ask_id TEXT PRIMARY KEY, hand_id TEXT NOT NULL, owner_id TEXT NOT NULL,
        parent_session TEXT NOT NULL, origin_ref TEXT NOT NULL)""")
    conn.commit()


def owner_started(row, *, store, state_dir, owner_id, ledger, answer_turns, verified):
    """Every ask in the notice must trace to a controller task and its owner's turn.

    Missing or legacy records keep the ordinary pause. Unreadable stores raise so
    the outbox holds this pull without recording a verdict based on a failed read.
    A previously verified link survives controller cleanup. Its captured owner
    source is still checked on every evaluation. Neither notice text nor the
    note's claimed origin establishes who started it.
    """
    context = row.context if isinstance(row.context, dict) else {}
    ids = context.get("asks") or ([context["ask_id"]] if context.get("ask_id") else [])
    if not isinstance(ids, list) or not ids or not owner_id or ledger is None:
        return False
    hands = []
    for ident in ids:
        note = store.get(ident) if isinstance(ident, str) else None
        source = note.context if note is not None and isinstance(note.context, dict) else {}
        hand = source.get("held_from")
        if (note is None or (note.kind, note.type, note.cls) != ("note", "floor_effect", "floor")
                or source.get("origin") != "hand" or not isinstance(hand, str) or not hand.strip()):
            return False
        hands.append((ident, hand))
    path = state_dir / DB_FILE
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1) if path.exists() else None
    try:
        if db is not None:
            db.row_factory = sqlite3.Row
        checked = []
        with closing(ledger._connect()) as turns:
            for ident, hand in hands:
                task = db.execute("SELECT t.origin,t.origin_ref,h.parent_session FROM hands h "
                                  "JOIN tasks t ON t.id=h.task_id WHERE h.id=?", (hand,)).fetchone() if db else None
                if task is None:
                    saved = verified.execute("SELECT origin_ref,parent_session FROM hand_ask_origins "
                        "WHERE ask_id=? AND hand_id=? AND owner_id=?", (ident, hand, owner_id)).fetchone()
                    if saved is not None:
                        task = {"origin": "owner_turn", "origin_ref": saved[0], "parent_session": saved[1]}
                if task is None or task["origin"] != "owner_turn":
                    return False
                try:
                    reference = MessageRef.model_validate_json(task["origin_ref"])
                except ValueError:
                    return False
                if not _owner_turn(reference, task["parent_session"], owner_id, answer_turns, turns):
                    return False
                checked.append((ident, hand, owner_id, task["parent_session"], task["origin_ref"]))
        verified.executemany("INSERT INTO hand_ask_origins VALUES (?,?,?,?,?) ON CONFLICT(ask_id) DO UPDATE SET "
            "hand_id=excluded.hand_id,owner_id=excluded.owner_id,parent_session=excluded.parent_session,"
            "origin_ref=excluded.origin_ref WHERE hand_id!=excluded.hand_id OR owner_id!=excluded.owner_id "
            "OR parent_session!=excluded.parent_session OR origin_ref!=excluded.origin_ref", checked)
        verified.commit()
    finally:
        if db is not None:
            db.close()
    return True


def _owner_turn(reference, session, owner_id, answer_turns, turns):
    read = answer_turns.find(f"{reference.platform}:{reference.chat}:{reference.message_id}")
    if read is not None:
        if (read.platform, read.chat_id, read.message_id) != (
                reference.platform, reference.chat, reference.message_id):
            return False
        turn = read.turn_id
    elif reference.platform in {"api", "cli"} and reference.chat == session:
        # Direct API and console turns can name their session and turn instead
        # of a chat delivery id. They need the same captured sender attribution.
        turn = reference.message_id
    else:
        return False
    source = turns.execute("SELECT contact_id,session_id,scope,messages_json FROM turn_sources WHERE turn_id=?",
                           (turn,)).fetchone()
    if (source is None or source["contact_id"] != owner_id or source["session_id"] != session
            or source["scope"] != "person" or is_invalidated(turns, turn)):
        return False
    channel = turns.execute("SELECT platform FROM source_channels WHERE turn_id=?",
                            (turn,)).fetchone()
    if channel is not None and channel["platform"] != reference.platform:
        return False
    return any(isinstance(message, dict) and message.get("role") == "user"
               for message in json.loads(source["messages_json"]))
