"""What of a page a turn carries.

In a turn with a person the assembled context carries the agent's page on that person, kept whole, and the
agent's page on itself, whose lines are chosen against the turn's message like every other optional item
(``memory.compass``). Both count toward the context budget (``memory.blocks``).

Every line says what kind of line it is, so a page that is cut still reads right.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from . import store

PERSON_ID, PERSON_TITLE = "colonymind-page-person", "Your page on this person"
SELF_ID, SELF_TITLE = "colonymind-page-self", "Your page on yourself"
#: kind -> how a line of that kind starts in a turn, on a person's page and on the agent's own.
PERSON_LEADS = {"boundary": "Your boundary with them", "about": "About them"}
SELF_LEADS = {"about": "About you", "like": "You like", "dislike": "You dislike"}


def _ledger() -> Any:
    from colonymind import get_state_dir
    from colonymind.turns import get_turn_idempotency_ledger
    return (get_turn_idempotency_ledger(get_state_dir())
            if (Path(get_state_dir()) / "turn-idempotency.db").exists() else None)


def sections(contact_id: Optional[str], *, people: bool = True) -> List[Dict[str, Any]]:
    """The page sections of a turn with ``contact_id`` (``id``, ``title``, ``body``, ``priority``): the page
    on that person, then the agent's own. ``people`` off leaves the person's page out. A page with no
    standing line gives no section."""
    ledger = _ledger()
    if ledger is None:
        return []
    person = contact_id if people and contact_id != store.SELF else None
    found = []
    for subject, ident, title, leads, priority in ((person, PERSON_ID, PERSON_TITLE, PERSON_LEADS, 85),
                                                   (store.SELF, SELF_ID, SELF_TITLE, SELF_LEADS, 82)):
        rows = store.lines(ledger, subject, kinds=tuple(leads)) if subject else []
        if rows:
            found.append({"id": ident, "title": title, "priority": priority,
                          "body": "\n".join(f"- {leads[row['kind']]}: {row['text']}" for row in rows)})
    return found
