"""``/v1/mind/pages``: the agent's pages, read and written.

The owner reads every page. Any other viewer reads the agent's own page and the page about themselves, which
is what a turn with them carries. The agent writes the lines that are its own to set: its boundaries with a
person (from the owner's conversation for anyone, from a person's own conversation for that person) and its
own likes and dislikes. A written line names the turn it was written in; without one it is refused. The
other lines are written by the nightly pass (``pages.night``).

Interface (JSON; the plugin's ``colonymind_people`` and ``colonymind_self`` are the clients):
  GET  /{subject}?contact_id=      {subject, lines: [{id, kind, text, sources, writer, written_at}], text}
  POST /{subject}/lines            {kind, text?, drop?, source?, contact_id?} -> {ok, added, dropped, lines, text}
``{subject}`` is ``self`` or a reference to a person, as ``/v1/mind/people`` reads one. ``contact_id`` is the
viewer: the session's contact. ``text`` adds a line, ``drop`` removes one by its id, both change one.

A call from a turn in a group chat names the chat as the turn's context does (``chat_type``, ``chat``, and its
``session_id``: query parameters of the GET, body fields of the POST). It is shown a line only when every turn
the line cites took place in that chat (``turns.group_chats``), and the line it wrote.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from colonymind.hands.sessions import refuse_hand_session

from . import store

router = APIRouter(prefix="/v1/mind/pages", tags=["pages"])

#: The kinds of line the agent sets itself: on its own page, and on a person's.
OWN_KINDS, PERSON_KINDS = ("like", "dislike"), ("boundary",)


class LineBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str = Field(min_length=1, max_length=16)
    text: str = Field(default="", max_length=2000)
    drop: str = Field(default="", max_length=32)
    source: str = Field(default="", max_length=256)
    contact_id: Optional[str] = Field(default=None, max_length=256)
    session_id: Optional[str] = Field(default=None, max_length=256)
    chat_type: Optional[str] = Field(default=None, max_length=64)
    chat: Optional[str] = Field(default=None, max_length=512)


def _ledger() -> Any:
    from colonymind import get_state_dir
    from colonymind.turns import get_turn_idempotency_ledger
    return get_turn_idempotency_ledger(get_state_dir())


async def _subject(request: Request, reference: str, viewer: Optional[str]) -> str:
    """The page a reference names, when this viewer may open it."""
    if reference == store.SELF:
        return store.SELF
    from colonymind.api.routers import people
    owner_view = people._is_owner_viewer(request, viewer)
    contact = await people._resolve(reference, candidates_shown=owner_view)
    if not owner_view and contact.contact_id != viewer:
        raise HTTPException(status_code=403, detail={
            "code": "not_your_page", "message": "only the owner's conversation opens another person's page"})
    return contact.contact_id


async def page_text(subject: str) -> str:
    """One page as text, each line with its id and the turns it rests on ("" for a page with no lines)."""
    return store.render(await asyncio.to_thread(store.lines, _ledger(), subject), detail=True)


def _shown(subject: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"subject": subject, "lines": rows, "text": store.render(rows, detail=True) or "(no lines)"}


def _refused(code: str, message: str) -> HTTPException:
    return HTTPException(status_code=422, detail={"code": code, "message": message})


def _lines(subject: str, chat_type: Optional[str], chat: Optional[str], session_id: Optional[str], *,
           kinds: Optional[Sequence[str]] = None, written: Any = None) -> List[Dict[str, Any]]:
    """The page's standing lines, in page order. A turn in a group chat is given the ones whose every cited turn
    took place in that chat, and the line it wrote (``written``, a line id)."""
    from colonymind.turns import group_chats
    ledger = _ledger()
    rows = store.lines(ledger, subject, kinds=kinds)
    group = group_chats.of_turn({"chat_type": chat_type, "chat": chat})
    if group is None:
        return rows
    inside = {row["id"] for row in group_chats.within(ledger, group, session_id, rows,
                                                      turns=lambda row: row["sources"])}
    return [row for row in rows if row["id"] in inside or row["id"] == written]


@router.get("/{subject}")
async def page(request: Request, subject: str, contact_id: Optional[str] = None, session_id: Optional[str] = None,
               chat_type: Optional[str] = None, chat: Optional[str] = None) -> Dict[str, Any]:
    """One page, each line with its id and the turns it rests on."""
    who = await _subject(request, subject, contact_id)
    return _shown(who, await asyncio.to_thread(_lines, who, chat_type, chat, session_id))


@router.post("/{subject}/lines", dependencies=[Depends(refuse_hand_session)])
async def write(request: Request, subject: str, body: LineBody) -> Dict[str, Any]:
    """Set one of the agent's own lines: ``text`` adds one, ``drop`` removes one, both change one."""
    who = await _subject(request, subject, body.contact_id)
    kinds = OWN_KINDS if who == store.SELF else PERSON_KINDS
    text, drop, source = store.clean(body.text), body.drop.strip(), body.source.strip()
    if body.kind not in kinds:
        raise _refused("kind", f"kind is one of: {', '.join(kinds)}")
    if not text and not drop:
        raise _refused("nothing_to_set", "text (a line to add) or drop (the id of a line to remove) is required")
    if text and not source:
        raise _refused("no_source", "a line names the turn it was written in, and this turn has no id")
    ledger = _ledger()
    dropped = bool(drop) and await asyncio.to_thread(store.drop, ledger, who, drop, kinds=(body.kind,))
    line = await asyncio.to_thread(store.add, ledger, who, body.kind, text, [source]) if text else None
    rows = await asyncio.to_thread(_lines, who, body.chat_type, body.chat, body.session_id, kinds=kinds,
                                   written=line["id"] if line else None)
    return {"ok": bool(line or dropped), "added": line["id"] if line else None, "dropped": dropped,
            **_shown(who, rows)}
