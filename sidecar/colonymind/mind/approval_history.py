"""Page existing asks and their recorded settlements without writing a snapshot.

Revision-bound cursors deliberately refuse changed data. Nothing inferred from an
execution journal, a wall-clock expiry, or an absent session becomes history.
"""
from __future__ import annotations

import base64
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from fastapi import HTTPException

from colonymind.api.auth import owner_person_id
from colonymind.config import instance_home
from colonymind.hands.store import DB_FILE
from colonymind.initiatives.models import AssignmentHistory, StoredInitiative
from colonymind.mind import permit_asks
from colonymind.mind.outbox import ASK_IN_FULL


_ASKS = "kind IS NOT NULL AND ask_code IS NOT NULL AND ask_code != ''"
_OUTCOMES = {"denied": "denied", "expired": "timed_out", "cancelled": "cancelled",
             "failed": "failed", "done": "done", "uncertain": "uncertain", "dropped": "denied"}


def _error(status, code):
    # No exception text, query values, or stored private details enter diagnostics.
    return HTTPException(status, detail={"code": code})


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def _iso(value):
    return _utc(value).isoformat() if value is not None else None


@contextmanager
def _read(path):
    # mode=ro refuses missing stores instead of creating an empty history database.
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=1)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        yield conn


class _Recorded:
    """Snapshot adapter for the existing terms/settlement readers."""
    def __init__(self, rows, histories):
        self.rows = {row.id: row for row in rows}
        self.dedup = {row.dedup_key: row for row in rows}
        self.histories = histories

    def get(self, ident):
        return self.rows.get(ident)

    def get_by_dedup_key(self, key):
        return self.dedup.get(key)

    def get_history(self, ident, limit=500):
        # This projection has read the complete history. No 500-row truncation can
        # discard the approval or a revival and change its apparent outcome.
        return self.histories.get(ident, [])


def _period(history):
    ordered = sorted(history, key=lambda item: (_utc(item.timestamp), item.id))
    starts = [i for i, item in enumerate(ordered) if item.action in {"asked", "ask_revived"}]
    return ordered[starts[-1] + 1:] if starts else ordered


def _decision(row, history):
    if row.status == "asked":
        return None, None, None
    event = next((item for item in _period(history) if item.action in {"queued", "dropped"}
                  or item.action.startswith("outcome_")), None)
    if event is None:
        return "uncertain", "missing_settlement", None
    if event.action == "queued":
        if event.details.get("by"):
            return "allowed", None, event
        if row.decision == "act":
            return "auto", None, event
        return "uncertain", "missing_settlement", event
    recorded = event.action.removeprefix("outcome_")
    if recorded in _OUTCOMES:
        return _OUTCOMES[recorded], None, event
    return "uncertain", "unrecognized_settlement", event


def _sessions(conn, rows):
    links = {}
    if conn.execute("SELECT 1 FROM sqlite_master WHERE name='hand_ask_origins'").fetchone():
        links = {item["ask_id"]: (item["hand_id"], item["parent_session"])
                 for item in conn.execute("SELECT ask_id,hand_id,parent_session FROM hand_ask_origins WHERE owner_id=?",
                                          (owner_person_id(),))}
    hands = {row.context.get("held_from") for row in rows if permit_asks.is_permit_ask(row)}
    parents = {}
    path = instance_home() / DB_FILE
    # No hands store is required for ordinary/legacy asks. Existing but unreadable
    # controller data is an error, not evidence that a session is unknown.
    if hands and path.exists():
        with _read(path) as db:
            parents = {item["id"]: item["parent_session"]
                       for item in db.execute("SELECT id,parent_session FROM hands") if item["id"] in hands}
    result = {}
    for row in rows:
        hand = row.context.get("held_from") if permit_asks.is_permit_ask(row) else None
        saved = links.get(row.id)
        result[row.id] = saved[1] if saved and saved[0] == hand else parents.get(hand)
    return result


def _snapshot(store, at):
    with _read(store._db_path) as conn:
        raw = [dict(row) for row in conn.execute(f"SELECT * FROM generic_initiatives WHERE {_ASKS} ORDER BY id")]
        rows = [StoredInitiative.from_row(item) for item in raw]
        for data, row in zip(raw, rows):
            context = json.loads(data["context"]) if data["context"] else {}
            if context is not None and not isinstance(context, dict):
                raise ValueError("malformed stored request")
            row.context = context or {}
            _utc(data["created_at"])
            if data["expires_at"] is not None:
                _utc(data["expires_at"])
        histories = {row.id: [] for row in rows}
        raw_history = {row.id: [] for row in rows}
        for item in conn.execute(f"SELECT h.* FROM assignment_history h JOIN generic_initiatives i "
                                 f"ON i.id=h.initiative_id WHERE {_ASKS} ORDER BY h.id"):
            data = dict(item)
            # Validate recorded timestamps before from_row's legacy fallback to now.
            _utc(data["timestamp"])
            history = AssignmentHistory.from_row(data)
            if history.details is None:
                history.details = {}
            if not isinstance(history.details, dict):
                raise ValueError("malformed stored history")
            histories[history.initiative_id].append(history)
            raw_history[history.initiative_id].append(data)
        notices = []
        if any(isinstance(row.context.get("full_view"), dict) for row in rows):
            # Both explicit notice IDs and retry dedup keys resolve in memory.
            notices = [StoredInitiative.from_row(dict(item)) for item in conn.execute(
                "SELECT * FROM generic_initiatives WHERE type=?", (ASK_IN_FULL,))]
        recorded = _Recorded(rows + notices, histories)
        sessions = _sessions(conn, rows)
        items = []
        for data, row in zip(raw, rows):
            normalized, uncertainty, event = _decision(row, histories[row.id])
            # Keep the existing receipt reader within this request period and the
            # actual deciding event, so subsequent work outcomes cannot replace it.
            receipt = None
            if event is not None:
                receipt = permit_asks.settlement(_Recorded([row], {row.id: [event]}), row)
            terms = permit_asks.terms(row, recorded)
            if row.kind == "message":
                # asked_words intentionally serves open asks only. Historical
                # detail must still show the exact retained message being asked about.
                from colonymind.mind.audit import stored_text
                terms["text"] = stored_text(row)
            session = sessions[row.id]
            revision = _digest([data, raw_history[row.id], session, terms])
            items.append({"id": row.id, "revision": revision,
                          "kind": "hands_permit" if permit_asks.is_permit_ask(row) else "ask",
                          "type": row.type, "code": row.ask_code, "title": row.description,
                          "hand_id": row.context.get("held_from") if permit_asks.is_permit_ask(row) else None,
                          "session_id": session, "state": "pending" if row.status == "asked" else "settled",
                          "status": row.status, "outcome": normalized, "uncertainty": uncertainty,
                          "open": permit_asks.is_open(row, at), "created_at": _iso(row.created_at),
                          "expires_at": _iso(row.expires_at), "settled_at": _iso(event.timestamp) if event else None,
                          "raw_outcome": row.outcome, "terms": terms, "settlement": receipt})
    return _digest([[item["id"], item["revision"]] for item in items]), items


def _cursor(value, binding):
    try:
        data = json.loads(base64.b64decode(value.encode("ascii"), altchars=b"-_", validate=True))
        if (not isinstance(data, dict) or set(data) != {"v", "revision", "position", "snapshot_at", "query"}
                or type(data["v"]) is not int or data["v"] != 1 or data["query"] != binding
                or type(data["position"]) is not int or data["position"] <= 0
                or not isinstance(data["revision"], str) or len(data["revision"]) != 64
                or any(c not in "0123456789abcdef" for c in data["revision"])
                or not isinstance(data["snapshot_at"], str)):
            raise ValueError()
        _utc(data["snapshot_at"])
        return data
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        raise _error(422, "invalid_cursor") from None


def _token(revision, position, at, binding):
    return base64.urlsafe_b64encode(_json({"v": 1, "revision": revision, "position": position,
                                          "snapshot_at": at, "query": binding}).encode()).decode()


def read(store, *, now, limit=50, state="all", session_id=None, cursor=None, approval_id=None):
    """One read-only transaction, off the request loop; no cache, writes or timers."""
    binding = [limit, state, session_id]
    continuation = _cursor(cursor, binding) if cursor is not None else None
    at = _utc(continuation["snapshot_at"] if continuation else now)
    try:
        revision, items = _snapshot(store, at)
    except (sqlite3.Error, OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
        raise _error(503, "approval_history_unavailable") from None
    if approval_id is not None:
        item = next((item for item in items if item["id"] == approval_id), None)
        if item is None:
            raise _error(404, "unknown_approval")
        return {"revision": revision, "snapshot_at": _iso(at), "approval": item}
    if continuation and continuation["revision"] != revision:
        raise _error(409, "approval_history_changed")
    items = [item for item in items if session_id is None or item["session_id"] == session_id]
    counts = {"pending": 0, "settled": 0, "total": len(items)}
    sessions = {}
    for item in items:
        counts[item["state"]] += 1
        group = sessions.setdefault(item["session_id"], {"session_id": item["session_id"],
                                                        "pending": 0, "settled": 0, "total": 0})
        group[item["state"]] += 1
        group["total"] += 1
    items = [item for item in items if state == "all" or item["state"] == state]
    # Stable tie order, then descending timestamps, then pending first.
    items.sort(key=lambda item: item["id"])
    items.sort(key=lambda item: _utc(item["created_at"]), reverse=True)
    items.sort(key=lambda item: item["state"] != "pending")
    position = continuation["position"] if continuation else 0
    if continuation and (position >= len(items) or position % limit):
        raise _error(422, "invalid_cursor")
    end = position + limit
    more = end < len(items)
    return {"revision": revision, "snapshot_at": _iso(at), "items": items[position:end], "counts": counts,
            "sessions": sorted(sessions.values(), key=lambda group: (group["session_id"] is None,
                                                                     group["session_id"] or "")),
            "has_more": more, "next_cursor": _token(revision, end, _iso(at), binding) if more else None}
