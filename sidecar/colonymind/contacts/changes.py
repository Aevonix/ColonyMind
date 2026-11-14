"""Changes to who may be contacted, and the owner's yes that lets a widening one apply (architecture 7.4, 7.10).

Nothing the owner writes in a conversation authorizes a change that widens who gets messaged, and nothing
here reads their words for one. Their request in a turn files the change as their one-tap ask
(``Mind.request_people_change``: a ``people_change`` owner question with a code), and the change applies
once their own reply to that ask says a plain yes (``Mind.answer``, read by ``colonymind.mind.answers``). What
widens:

- ``may_contact`` raised: never to ask, never to auto, ask to auto;
- a cadence raised: one set where there was none, or fewer minutes than before;
- any merge into the owner: the folded record's handles would speak as the owner, and what is recorded
  about it would be theirs;
- any other merge that gives a record's handles more reach than they had: the folded record's handles
  reached at a higher permission (the keeper's) or with a new or faster cadence, or the keeper's handles
  given a cadence it lacked. ``never`` on either side still wins.

Free, applied at once: a lowered permission, an unchanged value, a cleared or slower cadence, a merge that
widens nothing, and an obvious dedup (a verified handle of each record shares one normalized address: the
same phone number or email on two gateways) between two records neither of which is the owner.

How the owner's words are read as a yes or a no is ``colonymind.mind.answers``'s, the one place words decide.
"""

from __future__ import annotations

import asyncio
import re
import weakref
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PERMISSION_RANK = {"never": 0, "ask": 1, "auto": 2}
OPERATIONS = ("set_permission", "set_cadence", "merge")
_EMAIL_TEXT = re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+")


def every(minutes: int) -> str:
    """``minutes`` as the owner would say it, exactly: ``every 90 minutes``, ``every 2 days``, ``every week``."""
    for unit, size in (("week", 10080), ("day", 1440), ("hour", 60)):
        if minutes % size == 0:
            count = minutes // size
            return f"every {unit}" if count == 1 else f"every {count} {unit}s"
    return "every minute" if minutes == 1 else f"every {minutes} minutes"


# -- what a change does --------------------------------------------------------------------------------------

def _field(record: Any, name: str) -> Any:
    return record.get(name) if isinstance(record, Mapping) else getattr(record, name, None)


def effective_permission(record: Any, *, owner_id: Optional[str]) -> str:
    """``may_contact`` as it governs sends: the owner is ``auto`` by identity, anyone else its column
    (``ask`` without a usable value)."""
    if owner_id and _field(record, "contact_id") == owner_id:
        return "auto"
    value = _field(record, "may_contact")
    return str(value) if value in PERMISSION_RANK else "ask"


def permission_change(current: str, wanted: str) -> str:
    """``raise``, ``lower`` or ``unchanged`` for a ``may_contact`` move."""
    before, after = PERMISSION_RANK.get(current, 1), PERMISSION_RANK[wanted]
    return "raise" if after > before else "lower" if after < before else "unchanged"


def _cadence(value: Any) -> Optional[int]:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def faster(new: Optional[int], old: Optional[int]) -> bool:
    """True when cadence ``new`` checks in more often than ``old`` (a cadence where there was none counts)."""
    return new is not None and (old is None or new < old)


def cadence_change(current: Optional[int], wanted: Optional[int]) -> str:
    """``raise`` (a cadence where there was none, or fewer minutes), ``lower`` (cleared, or more minutes) or
    ``unchanged``."""
    current, wanted = _cadence(current), _cadence(wanted)
    if current == wanted:
        return "unchanged"
    return "raise" if faster(wanted, current) else "lower"


def merge_reach(keep: Any, drop: Any, keep_handles: Sequence[Any], drop_handles: Sequence[Any], *,
                owner_id: Optional[str]) -> List[str]:
    """What folding ``drop`` into ``keep`` would widen, as short clauses for the owner's ask ("its" is the
    folded record); [] when nothing gains reach.

    Folded into the owner, a record always widens: its handles become the owner's own (whoever writes from
    them is the owner, whatever either record's permission), and what is recorded about it becomes theirs. Else
    the fold (``SQLiteContactStore.merge``) keeps the keeper's permission unless either record is ``never``,
    and the keeper's cadence, else the folded one's: the folded record's handles are then reached as the
    keeper is, and the keeper's handles with the cadence the result has."""
    if owner_id and _field(keep, "contact_id") == owner_id:
        return ["its addresses become the owner's own: whoever writes from them is treated as the owner"
                if drop_handles else "what is recorded about it becomes the owner's own"]
    drop_may = effective_permission(drop, owner_id=owner_id)
    kept_column = effective_permission({"may_contact": _field(keep, "may_contact")}, owner_id=None)
    result_may = "never" if "never" in (kept_column, drop_may) else kept_column
    keep_cadence, drop_cadence = _cadence(_field(keep, "cadence_minutes")), _cadence(_field(drop, "cadence_minutes"))
    result_cadence = keep_cadence if keep_cadence is not None else drop_cadence
    reach: List[str] = []
    if drop_handles and result_may != "never":
        if PERMISSION_RANK[result_may] > PERMISSION_RANK[drop_may]:
            reach.append(f"its addresses reached at may_contact={result_may} (now {drop_may})")
        if faster(result_cadence, drop_cadence):
            reach.append(f"its addresses checked in with {every(result_cadence)}")
    if keep_handles and result_may != "never" and faster(result_cadence, keep_cadence):
        reach.append(f"the kept addresses checked in with {every(result_cadence)}")
    return reach


def _identity(handle: Any) -> Optional[Tuple[str, str]]:
    """The address a handle names across gateways: ``("phone", "+<digits>")`` or ``("email", lower)``."""
    from .store import canonical_handle
    gateway, address = str(_field(handle, "gateway") or ""), str(_field(handle, "address") or "")
    kind, key = canonical_handle(gateway, address)
    if kind in {"phone", "email"} and key:
        return kind, key
    if _EMAIL_TEXT.fullmatch(address.strip()):
        return "email", address.strip().lower()
    return None


def obvious_dedup(keep_handles: Iterable[Any], drop_handles: Iterable[Any]) -> bool:
    """True when a verified handle of each record names one address (the same phone number or email on two
    gateways): the two records are one person the owner already confirmed on both sides."""
    kept = {_identity(h) for h in keep_handles if _field(h, "verified")} - {None}
    return bool(kept & ({_identity(h) for h in drop_handles if _field(h, "verified")} - {None}))


# -- what an ask binds, and one change at a time ------------------------------------------------------------

async def snapshot(store: Any, records: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
    """What a change is decided on, per record: its stored ``may_contact`` and ``cadence_minutes`` and its
    handles (``gateway:address``, sorted). An ask carries it as ``from``, and its yes applies only to the
    records as they were: the write compares the permission or cadence and the handle set in the same
    statement (``SQLiteContactStore.set_may_contact``), so a handle attached to the contact while the ask
    waited, or even while the yes is written, is a different change and is asked again."""
    state: Dict[str, Dict[str, Any]] = {}
    for record in records:
        contact_id = str(_field(record, "contact_id") or "")
        handles = await store.get_handles(contact_id)
        state[contact_id] = {
            "may_contact": _field(record, "may_contact"), "cadence_minutes": _field(record, "cadence_minutes"),
            "handles": sorted(f"{_field(h, 'gateway')}:{_field(h, 'address')}" for h in handles)}
    return state


_LOCKS: "weakref.WeakKeyDictionary[Any, asyncio.Lock]" = weakref.WeakKeyDictionary()


def change_lock(store: Any) -> asyncio.Lock:
    """The one lock owner changes to a contact store are decided and written under (the people route, the
    direct routes and the owner's yes), so two of them never interleave between a decision and its write.
    An opt-out does not wait for it: the writes are compare-and-set (``expected``) and lose to it instead."""
    lock = _LOCKS.get(store)
    if lock is None:
        lock = _LOCKS[store] = asyncio.Lock()
    return lock


__all__ = ["OPERATIONS", "PERMISSION_RANK", "cadence_change", "change_lock", "effective_permission", "every",
           "faster", "merge_reach", "obvious_dedup", "permission_change", "snapshot"]
