"""Which sessions took place in which group chat, beside the canonical sources.

A turn in a group chat names its chat when its context is assembled, and the session is then noted for that chat.
The chat's sessions are the group's own history: a group turn's automatic recall reads nothing else, and the turn
carries one line that says so. A memory search and a page read made from such a turn name the chat too and are
narrowed the same way. There is one memory. This is a filter at read time over the same sources; a turn in a
direct chat is never narrowed.
"""
from __future__ import annotations

from contextlib import closing
from typing import Any, Callable, Iterable, List, Mapping, Optional, Set

from .source_annotations import source_ids

#: The chat kinds that are one person's own chat with the agent. Every other named kind is a group chat.
DIRECT = frozenset({"", "dm", "direct", "private"})
#: The sections a group turn carries: the clock, and the recall and the pages read from the group's own history.
KEPT = frozenset({"temporal-context", "colonymind-memory", "colonymind-page-person", "colonymind-page-self"})
#: The one line that tells the agent its recall was narrowed.
NOTICE = {"id": "colonymind-recall-narrowed", "title": "Recall narrowed", "priority": 97,
          "body": "This turn is in a group chat. Recall read this group's own history only; the rest of your "
                  "memory was withheld."}


def initialize(conn) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS group_chat_sessions (
        chat TEXT NOT NULL, session_id TEXT NOT NULL, PRIMARY KEY(chat, session_id))""")


def of_turn(metadata: Mapping[str, Any]) -> Optional[str]:
    """The group chat a turn belongs to, from its context metadata (``chat_type``, ``chat``): None in a direct
    chat and for a turn that names no chat kind, else the chat, empty when the host named a group kind only."""
    if str(metadata.get("chat_type") or "").strip().lower() in DIRECT:
        return None
    chat = metadata.get("chat")
    return chat.strip()[:512] if isinstance(chat, str) else ""


def note(chat: str, session_id: Optional[str]) -> None:
    """Remember that ``session_id`` takes place in ``chat``."""
    from colonymind.contacts.digest import source_ledger
    ledger = source_ledger()
    if ledger is not None and chat and session_id:
        with closing(ledger._connect()) as conn, conn:
            conn.execute("INSERT OR IGNORE INTO group_chat_sessions VALUES (?,?)", (chat, session_id))


def turns_of(row: Mapping[str, Any]) -> Set[str]:
    """Every source turn a recall candidate names: its own, the ones its corrections and dependencies visited,
    a conversation pair's, and a media locator's."""
    ids = set(source_ids(row))
    for name in ("_annotation_source_refs", "_conversation_source_refs", "source_anchors"):
        ids.update(ref["source_id"] for ref in row.get(name) or () if isinstance(ref, dict) and ref.get("source_id"))
    read = row.get("source_read")
    if isinstance(read, dict) and read.get("source_id"):
        ids.add(read["source_id"])
    return ids


def within(ledger, chat: str, session_id: Optional[str], rows: Iterable[Mapping[str, Any]], *,
           turns: Callable[[Mapping[str, Any]], Iterable[str]] = turns_of) -> List[Any]:
    """The ``rows`` whose every source turn took place in ``chat``, whose current session is ``session_id``.
    A row that names no turn is not from the group's history and is left out."""
    rows = list(rows)
    named = [set(turns(row)) for row in rows]
    wanted = sorted(set().union(*named))
    if ledger is None or not wanted:
        return []
    inside: Set[str] = set()
    with closing(ledger._read_connect()) as conn:
        for start in range(0, len(wanted), 500):
            batch = wanted[start:start + 500]
            inside.update(found[0] for found in conn.execute(
                f"SELECT turn_id FROM turn_sources WHERE turn_id IN ({','.join('?' * len(batch))}) AND (session_id=?"
                " OR session_id IN (SELECT session_id FROM group_chat_sessions WHERE chat=?))",
                (*batch, session_id or None, chat)))
    return [row for row, ids in zip(rows, named) if ids and ids <= inside]
