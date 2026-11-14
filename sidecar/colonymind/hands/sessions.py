"""Durable hand-session identities and refusal of direct authority mutations."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path

import aiosqlite
from fastapi import HTTPException, Request

from colonymind.config import instance_home
from colonymind.util.temporal import now_utc
from .store import DB_FILE, HandsStore


def transaction_lock(store: HandsStore) -> asyncio.Lock:
    # aiosqlite serializes statements, not multi-statement transactions.
    if not hasattr(store, "_authority_lock"):
        store._authority_lock = asyncio.Lock()
    return store._authority_lock


@asynccontextmanager
async def open_store():
    """Use the frozen controller database without a second persistence schema."""
    store = HandsStore(instance_home() / DB_FILE)
    await store.connect()
    try:
        yield store
    finally:
        await store.close()


async def register(store: HandsStore, session_id: str, hand_id: str, *, parent_session: str, epoch: dict) -> None:
    """Bind a child identity once. Old identities stay refused after rotation."""
    if not session_id or session_id == parent_session:
        raise ValueError("a hand needs its own child session")
    async with transaction_lock(store):
        await store.db.execute("BEGIN IMMEDIATE")
        try:
            rows = await store.db.execute_fetchall("SELECT * FROM hands WHERE id=?", (hand_id,))
            if not rows or rows[0]["parent_session"] != parent_session or json.loads(rows[0]["epoch"]) != epoch:
                raise ValueError("hand parent or executor epoch does not match")
            found = await lookup(store, session_id)
            if found and (found["hand_id"] != hand_id or found["epoch"] != epoch):
                raise ValueError("session already belongs to a hand")
            await store.db.execute(
                "INSERT OR IGNORE INTO hand_sessions(session_id,hand_id,parent_session,epoch,registered_at) "
                "VALUES (?,?,?,?,?)", (session_id, hand_id, parent_session, json.dumps(epoch, sort_keys=True),
                                       now_utc().isoformat()))
            await store.db.commit()
        except BaseException:
            await store.db.rollback()
            raise


async def lookup(store: HandsStore, session_id: str) -> dict | None:
    return await _lookup(store.db, session_id)


async def _lookup(db: aiosqlite.Connection, session_id: str) -> dict | None:
    rows = await db.execute_fetchall("SELECT * FROM hand_sessions WHERE session_id=?", (session_id,))
    if rows:
        row = dict(rows[0])
        row["epoch"] = json.loads(row["epoch"])
        return row
    rows = await db.execute_fetchall("SELECT * FROM hand_session_tombstones WHERE session_id=?", (session_id,))
    if rows:
        row = dict(rows[0])
        row["epoch"] = json.loads(row["epoch"])
        return row
    # Also covers the bind transition until the controller registers all aliases.
    rows = await db.execute_fetchall("SELECT id, parent_session, epoch FROM hands WHERE child_session=?",
                                    (session_id,))
    return {"hand_id": rows[0]["id"], "parent_session": rows[0]["parent_session"],
            "epoch": json.loads(rows[0]["epoch"])} if rows else None


async def retire(store: HandsStore, session_id: str) -> None:
    """Retirement records history; it never turns a child into an owner session."""
    async with transaction_lock(store):
        await store.db.execute("UPDATE hand_sessions SET retired_at=COALESCE(retired_at,?) WHERE session_id=?",
                               (now_utc().isoformat(), session_id))
        await store.db.commit()


async def check_hand_sessions(request: Request, *, body: object) -> None:
    """Inspect structured request identity, never free text or child capture targets.

    The caller supplies the decoded body so bounded routes control reading and parsing.
    The header is the adapter seam for routes whose existing body has no session.
    A full-shell caller can omit or forge it; this is the documented trusted-shell gap.
    """
    identities = set(request.headers.getlist("X-ColonyMind-Session"))
    for key in ("session", "session_id", "requester"):
        identities.update(request.query_params.getlist(key))
    if isinstance(body, dict):
        for item in (body, body.get("context"), body.get("metadata"), body.get("turn")):
            if isinstance(item, dict):
                identities.update(item.get(key) for key in ("session", "session_id", "requester")
                                  if isinstance(item.get(key), str))
    identities.discard("")
    if not identities:
        return
    # Before the controller exists there cannot be a registered hand. Do not
    # create a database merely because an ordinary conversation names a session.
    supplied = getattr(request.app.state, "hands_store", None)
    if supplied is None and not Path(instance_home() / DB_FILE).exists():
        return
    try:
        if supplied is not None:
            found = any([await lookup(supplied, sid) for sid in identities])
        else:
            # A refusal lookup must not migrate or otherwise write controller state.
            uri = (instance_home() / DB_FILE).resolve().as_uri() + "?mode=ro"
            async with aiosqlite.connect(uri, uri=True) as db:
                db.row_factory = aiosqlite.Row
                await db.execute("BEGIN")
                found = any([await _lookup(db, sid) for sid in identities])
    except Exception:
        raise HTTPException(503, detail={"code": "hand_session_unavailable"}) from None
    if found:
        raise HTTPException(403, detail={"code": "hand_session_refused",
            "message": "hand sessions cannot write memory, change people or answer owner asks"})


async def refuse_hand_session(request: Request):
    """Refuse the child before the handler can read or change owner authority."""
    body = None
    if request.method not in {"GET", "HEAD"}:
        try:
            body = await request.json()
        except (ValueError, UnicodeError):
            pass
    await check_hand_sessions(request, body=body)
