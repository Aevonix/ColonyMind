"""The owner's own answer to an open ask, the way the Hermes plugin hands it to the sidecar (``Mind.read_answer``).

A yes comes only from the owner's own message in their chat (docs/MIND.md, Asks): the console (``Mind.answer``) may
only say no. Tests that need the owner to answer an ask go through the one path an answer takes in production: the
ask put in front of the owner as its own message (``Mind.record_shown``), then their message as a quoted reply to it
in their direct chat, read by the answer route.
"""

from __future__ import annotations

import itertools
from datetime import timedelta
from typing import Any, Optional

PLATFORM, CHAT = "whatsapp", "owner-chat"
_numbers = itertools.count(1)


def open_ask(mind: Any, ident: str) -> Optional[Any]:
    """The open ask an id or a code names, or None."""
    row = mind.store.get(str(ident)) if ident else None
    if row is None or not row.ask_code:
        row = mind.store.get_by_ask_code(str(ident or "").strip().upper())
    return row if row is not None and row.status == "asked" else None


async def owner_answers(mind: Any, ident: str, text: str = "yes", *, contact_id: str | None = None,
                        platform: str = PLATFORM, chat: str = CHAT, chat_type: str = "dm") -> Optional[Any]:
    """``text`` from ``contact_id`` (the owner by default) as a quoted reply to the open ask ``ident`` (its id or its
    code), read by the answer route; the ask's row after it, or None when no open ask has it. A message the route
    read and refused to settle (a yes before a long ask's full text arrived, say) raises ``PermissionError`` with the
    route's reason; one it did not read at all (someone else's, a group's) leaves the ask as it was."""
    row = open_ask(mind, ident)
    if row is None:
        return None
    number = next(_numbers)
    shown = f"ask-message-{number}"
    now = mind.clock()
    mind.record_shown([row.id], platform=platform, message_id=shown, chat=chat, at=now - timedelta(seconds=30))
    result = await mind.read_answer({
        "platform": platform, "contact_id": contact_id or mind.owner_id, "chat_id": chat, "chat_type": chat_type,
        "message_id": f"owner-message-{number}", "turn_id": f"owner-turn-{number}", "arrived_at": now.isoformat(),
        "text": text, "reply_to_message_id": shown})
    if result.get("refused"):
        raise PermissionError(result["refused"][0]["reason"])
    return mind.store.get(row.id)


__all__ = ["CHAT", "PLATFORM", "open_ask", "owner_answers"]
