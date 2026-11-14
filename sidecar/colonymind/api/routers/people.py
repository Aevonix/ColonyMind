"""``/v1/mind/people``: who people are, and the owner's say over them (architecture 4.7, 7.4, 7.10).

Reads are for everyone; a caller that names a viewer ``contact_id`` other than the owner sees
only who a person is (id, name, tier), never their handles, digest or permission, and only for
the one person a reference names: no listing, partial matches or 404 candidates. The owner's view
and every mutation (``permission``, ``cadence``, ``merge``) need a request that speaks for the
owner: a credential granted the owner (docs/CREDENTIALS.md), from the owner's session (the viewer
is the owner) or naming no viewer (the key holder's CLI). ``by`` labels the audit and authorizes
nothing. Anyone may propose a link, which only files a candidate the owner confirms.
``may_contact`` is raised nowhere else (an opt-out only lowers it).

Being in the owner's session is not the owner's yes (``contacts.changes``): a change that widens who gets
messaged (a raised permission, a new or faster cadence, a merge that gives a record's handles more reach, any
merge into the owner) never applies on what the turn said, and no route has a field for words. ``/changes``
(the plugin's route) files it as the owner's ask, which their own answer in their chat applies; the direct
routes answer 409 ``needs_owner_ask``. Nor is the operator's own call (a credential granted the owner that
names no viewer: the local CLI, a script with the key) the owner's yes: it cites no owner turn, and an agent
with a shell can hold that key, so a widening from it is refused 403 ``approve_required`` for every credential.
Narrowings, unchanged values, merges that widen nothing and verified-handle dedups apply at once, from either.

Interface (JSON; the plugin's ``colonymind_people`` and ``colonymind people`` are the clients):
  GET  /?q=&contact_id=&limit=            who: [{contact_id, display_name, trust_tier, may_contact,
                                          cadence_minutes, last_interaction_at, handles}]
  GET  /proposals                         pending link proposals (a name suggested a handle)
  GET  /{who}?contact_id=                 inspect: the record with its digest, handles, open
                                          proposals and permission history
  POST /changes                           {operation: set_permission|set_cadence|merge, contact_id,
                                          value|minutes|drop, viewer, by?} ->
                                          {ok, status: applied|asked|unchanged, code?, text}
  POST /{who}/permission                  {may_contact: never|ask|auto, contact_id?, by?, reason?}
  POST /{who}/cadence                     {minutes: int|null, contact_id?, by?}
  POST /merge                             {keep, drop, contact_id?, by?}
  POST /link                              {contact_id, gateway, address, evidence_refs?, by?}
``{who}``, ``keep``, ``drop`` and the link's ``contact_id`` are references: a contact id, a
phone number, an email, ``gateway:address`` or a unique name. With the mind's people faculty
off, ``merge``, ``link`` and ``cadence`` answer 409 ``people_off``.
"""

from __future__ import annotations

import json
import logging
from contextlib import closing
from typing import Any, Dict, List, Optional

from fastapi import Depends, APIRouter, HTTPException, Request
from colonymind.hands.sessions import refuse_hand_session
from pydantic import BaseModel, ConfigDict, Field

from colonymind.contacts import changes as people_changes
from colonymind.contacts.models import MAY_CONTACT
from colonymind.contacts.store import ContactChanged
from colonymind.api.public_spec import PUBLIC

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/mind/people", tags=["people"])

PUBLIC_FIELDS = ("contact_id", "display_name", "trust_tier")
PERMISSION_ACTIONS = ("may_contact_set", "opt_out", "cadence_set")
SOURCE_CONFLICTS = {"source_erased", "source_attribution_preimage_changed"}


class PermissionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    may_contact: str = Field(min_length=1, max_length=8)
    contact_id: Optional[str] = Field(default=None, max_length=256)
    by: str = Field(default="owner", max_length=64)
    reason: str = Field(default="", max_length=400)


class CadenceBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    minutes: Optional[int] = None
    contact_id: Optional[str] = Field(default=None, max_length=256)
    by: str = Field(default="owner", max_length=64)


class MergeBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    keep: str = Field(min_length=1, max_length=256)
    drop: str = Field(min_length=1, max_length=256)
    contact_id: Optional[str] = Field(default=None, max_length=256)
    by: str = Field(default="owner", max_length=64)


class ChangeBody(BaseModel):
    """A change from the owner's session (the plugin's ``colonymind_people``): ``contact_id`` is the person
    changed (the keeper of a merge) and ``viewer`` the session's contact (the owner). Nothing the turn said
    has a field here: no words stand for the owner's yes, which only their reply to the ask's code gives."""
    model_config = ConfigDict(extra="forbid")
    operation: str = Field(min_length=1, max_length=32)
    contact_id: str = Field(min_length=1, max_length=256)
    value: Optional[str] = Field(default=None, max_length=8)
    minutes: Optional[int] = None
    drop: Optional[str] = Field(default=None, max_length=256)
    viewer: Optional[str] = Field(default=None, max_length=256)
    by: str = Field(default="owner", max_length=64)


class LinkBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contact_id: str = Field(min_length=1, max_length=256)
    gateway: str = Field(min_length=1, max_length=64)
    address: str = Field(min_length=1, max_length=512)
    evidence_refs: List[str] = Field(default_factory=list, max_length=10)
    by: str = Field(default="owner", max_length=256)


# -- wiring ------------------------------------------------------------------------------

def _store() -> Any:
    from colonymind.api.routers import host
    if host._contacts_store is None:
        raise HTTPException(status_code=503, detail={"code": "contacts_unavailable",
                                                     "message": "the contact store is not running"})
    return host._contacts_store


def _owner_id() -> Optional[str]:
    from colonymind.identity import get_owner_contact_id
    return get_owner_contact_id()


def _ledger() -> Any:
    try:
        from colonymind import get_state_dir
        from colonymind.turns import get_turn_idempotency_ledger
        return get_turn_idempotency_ledger(get_state_dir())
    except Exception as error:  # no ledger: a merge still moves handles; sources wait for the worker
        logger.warning("people: the source ledger is unavailable (%s)", type(error).__name__)
        return None


async def person_sources(contact_id: str) -> List[str]:
    """The person-scoped sources a contact holds in the ledger: what a merge moves (C2)."""
    ledger = _ledger()
    if ledger is None:
        return []
    with closing(ledger._connect()) as conn:
        rows = conn.execute("SELECT turn_id FROM turn_sources WHERE contact_id=? AND scope='person' AND turn_id "
                            "NOT IN (SELECT turn_id FROM source_erasures) ORDER BY turn_id", (contact_id,)).fetchall()
    return [str(row[0]) for row in rows]


def reattribute_hooks(*, reconcile: bool = True) -> List[Any]:
    """``reattribute(old_id, new_id)`` of the stores that key rows by contact (comms, affect,
    commitments), then (``reconcile``) the ledger move of the merge's sources. A row whose contact
    and ledger source disagree is purged as erased, so comms and affect move a sourced row only
    once its source moved: here they move what they can, and the reconciliation of the merge's
    sources (``social_state.reconcile_identity_sources``) moves the rest, now or on the source
    worker's retry. The router reconciles itself, to report what moved."""
    from colonymind.api.routers import host
    hooks = []
    for name in ("_comms_log", "_affect_store", "_commitment_store"):
        hook = getattr(getattr(host, name, None), "reattribute", None)
        if callable(hook):
            hooks.append(hook)
    hooks.append(_reattribute_opinions)
    if reconcile:
        hooks.append(reconcile_merge)
    return hooks


def _reattribute_opinions(drop_id: str, keep_id: str) -> int:
    """The running mind's views about the dropped contact become views about the kept one (integration
    map X14): read when the merge runs, so a mind set up after the store is the one moved."""
    from colonymind.api.routers import mind as mind_router
    store = getattr(getattr(mind_router.get_mind(), "opinions", None), "store", None)
    move = getattr(store, "reattribute_subject", None)
    return int(move(drop_id, keep_id) or 0) if callable(move) else 0


async def reconcile_merge(drop_id: str, keep_id: str) -> None:
    """Move a merge's sources in the ledger now (the source worker retries whatever fails)."""
    from colonymind.api.routers import host
    if host._contacts_store is not None:
        await _reconcile(host._contacts_store, prefix=f"merge:{drop_id}:")


def _speaks_for_owner(request: Optional[Request]) -> bool:
    """The request's credential is granted the owner (an in-process call is): the request's authority,
    never anything its body asserts."""
    from colonymind.api.auth import speaks_for_owner
    return speaks_for_owner(request)


def _is_owner_viewer(request: Optional[Request], contact_id: Optional[str]) -> bool:
    """A request that speaks for the owner, naming no viewer (the key holder, the local owner) or the
    owner as its viewer."""
    if not _speaks_for_owner(request):
        return False
    if contact_id is None:
        return True
    owner = _owner_id()
    return bool(owner) and contact_id == owner


def _operator(request: Optional[Request], contact_id: Optional[str]) -> bool:
    """Whether a change is the operator's own call (the local CLI, a script with the key): a request whose
    credential speaks for the owner and that names no viewer. It cites no owner turn, so it narrows and never
    widens (403 ``approve_required``); naming a viewer (the owner's session) makes a widening the owner's ask
    (409 ``needs_owner_ask``). ``by`` decides nothing."""
    return contact_id is None and _speaks_for_owner(request)


def _require_owner(request: Optional[Request], contact_id: Optional[str], by: str) -> str:
    """7.10: a mutation from a request that speaks for the owner, in the owner's session or naming no
    viewer (the local CLI); returns who performed it. ``by`` only labels the audit."""
    if not _is_owner_viewer(request, contact_id):
        raise HTTPException(status_code=403, detail={
            "code": "not_owner", "message": "only the owner can change who may be contacted, cadences or merges"})
    if contact_id is not None:
        return contact_id
    from colonymind.api.auth import request_authority
    return "cli" if by == "cli" else request_authority(request).principal_id


async def _resolve(reference: str, *, candidates_shown: bool = True) -> Any:
    """The one contact a reference names; otherwise 404 with the people it could mean (who they
    are only), so an ambiguous name is answered, never guessed. A caller that is not the owner
    (``candidates_shown`` False) gets the 404 alone: the contact list is not a guest's to browse."""
    store = _store()
    contact = await store.resolve_reference(reference)
    if contact is None:
        try:
            candidates = [_row(c, [], full=False) for c in await store.search(reference, limit=5)] \
                if candidates_shown else []
        except Exception:
            candidates = []
        raise HTTPException(status_code=404, detail={
            "code": "unknown_contact", "candidates": candidates,
            "message": f"no single contact matches {reference!r}"
                       + (": " + "; ".join(_line(row) for row in candidates) if candidates else "")})
    return contact


# -- rendering ---------------------------------------------------------------------------

def _handles(handles: List[Any]) -> List[Dict[str, Any]]:
    return [{"gateway": h.gateway, "address": h.address, "verified": bool(h.verified)} for h in handles]


def _row(contact: Any, handles: List[Any], *, full: bool) -> Dict[str, Any]:
    if not full:
        return {key: getattr(contact, key) for key in PUBLIC_FIELDS}
    from colonymind.mind.authority import may_contact_of
    return {"contact_id": contact.contact_id, "display_name": contact.display_name, "trust_tier": contact.trust_tier,
            # The owner is ``auto`` by identity, whatever an older row's column says.
            "may_contact": may_contact_of(contact, owner_id=_owner_id()), "cadence_minutes": contact.cadence_minutes,
            "last_interaction_at": contact.last_interaction_at, "interaction_count": contact.interaction_count,
            "handles": _handles(handles)}


def _line(row: Dict[str, Any]) -> str:
    parts = [f"{row['contact_id']}  {row.get('display_name') or '(no name)'}  ({row.get('trust_tier')}"]
    if "may_contact" in row:
        parts[0] += f", may_contact={row['may_contact']}"
        if row.get("cadence_minutes"):
            parts[0] += f", every {row['cadence_minutes']} min"
        if row.get("last_interaction_at"):
            parts[0] += f", last {row['last_interaction_at']}"
    parts[0] += ")"
    handles = ", ".join(f"{h['gateway']}:{h['address']}" for h in row.get("handles") or [])
    return parts[0] + (f"  {handles}" if handles else "")


def _detail(raw: Any) -> Any:
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return raw


# -- reads -------------------------------------------------------------------------------

@router.get("", tags=[PUBLIC])
@router.get("/")
async def who(request: Request, q: str = "", contact_id: Optional[str] = None,
              limit: int = 20) -> Dict[str, Any]:
    """Who is this: by id, handle or name; no query lists the newest contacts. A viewer who is not
    the owner gets only the one person the query names exactly: no listing and no partial matches,
    so a guest learns who someone is without browsing the contact list."""
    store = _store()
    full = _is_owner_viewer(request, contact_id)
    if full:
        found = await store.search(q, limit=max(1, min(int(limit), 100)))
    else:
        exact = await store.resolve_reference(q) if q.strip() else None
        found = [exact] if exact is not None else []
    rows = []
    for contact in found:
        rows.append(_row(contact, await store.get_handles(contact.contact_id) if full else [], full=full))
    return {"contacts": rows, "text": "\n".join(_line(row) for row in rows) or "(nobody matches)"}


@router.get("/proposals", tags=[PUBLIC])
async def proposals(limit: int = 50) -> Dict[str, Any]:
    """Open link proposals: a name suggested that a handle is this person; the owner decides."""
    rows = await _store().list_handle_proposals(limit=max(1, min(int(limit), 100)))
    lines = [f"{row['candidate_id']}: is {row['gateway']}:{row['address']} {row.get('display_name') or row['contact_id']}?"
             for row in rows]
    return {"proposals": rows, "text": "\n".join(lines) or "(no open proposals)"}


@router.get("/{reference}", tags=[PUBLIC])
async def inspect(request: Request, reference: str, contact_id: Optional[str] = None) -> Dict[str, Any]:
    """One person: the record, digest, handles, open proposals and permission history (owner view)."""
    store = _store()
    owner_view = _is_owner_viewer(request, contact_id)
    contact = await _resolve(reference, candidates_shown=owner_view)
    if not owner_view:
        row = _row(contact, [], full=False)
        return {"contact": row, "text": _line(row)}
    row = _row(contact, await store.get_handles(contact.contact_id), full=True)
    # The digest only while every claim it cites stands (``contacts.digest.checked_digest``), as in the section.
    from colonymind.contacts.digest import checked_digest, source_ledger
    digest, _ = await checked_digest(contact, store=store, ledger=source_ledger())
    row.update(digest=digest or None, first_seen_at=contact.first_seen_at, notes=contact.notes, tags=contact.tags,
               introduced_by=contact.introduced_by, import_source=contact.import_source)
    pending = [p for p in await store.list_handle_proposals(limit=100) if p["contact_id"] == contact.contact_id]
    history = [{"action": a["action"], "at": a["created_at"], "by": a["performed_by"], "detail": _detail(a["detail"])}
               for a in await store.get_audit_log(contact.contact_id, limit=50) if a["action"] in PERMISSION_ACTIONS]
    text = [_line(row)]
    if digest:
        text.append(digest)
    text += [f"proposal {p['candidate_id']}: {p['gateway']}:{p['address']}" for p in pending]
    text += [f"{h['at']} {h['action']} by {h['by']}: {h['detail']}" for h in history[:5]]
    return {"contact": row, "proposals": pending, "permission_history": history, "text": "\n".join(text)}


# -- owner mutations -----------------------------------------------------------------------

def _require_people() -> None:
    """Merges, link proposals and cadences are the people faculty's (M5): with it off (the
    ``full-people`` ablation) they are refused, and who, inspect and permission stay."""
    from colonymind.api.routers.mind import faculty_on
    if not faculty_on("people"):
        raise HTTPException(status_code=409, detail={"code": "people_off",
                                                     "message": "the people faculty is off"})


# -- what a change widens, and the owner's ask for it ------------------------------------------

NEEDS_ASK = ("this widens who gets messaged: from the owner's session it is their ask (colonymind_people, "
             "POST /v1/mind/people/changes), which their own answer in their chat applies")
APPROVE_REQUIRED = ("this widens who gets messaged, which only the owner's own answer gives: they say it in their "
                    "own chat, where the agent files it as their question; a request that cites no owner turn (this "
                    "console, a script with the key) may only narrow")


def _label(contact: Any, handles: List[Any]) -> str:
    """A person as the owner reads an ask: the name and one handle (a verified, primary one first)."""
    name = contact.display_name or contact.given_name or contact.contact_id
    ordered = sorted(handles, key=lambda h: (not bool(h.verified), not bool(h.is_primary)))
    return f"{name} ({ordered[0].gateway}:{ordered[0].address})" if ordered else f"{name} ({contact.contact_id})"


class _Plan:
    """One change as decided: ``unchanged``, ``free`` (a narrowing, a dedup, a merge that widens nothing) or
    ``raise`` (the owner's ask), with the ask's question and change, and ``state``: the records as the
    decision read them (``contacts.changes.snapshot``), which the write and the ask's yes are both bound to."""

    def __init__(self, kind: str, *, state: Dict[str, Any], question: str = "",
                 change: Optional[Dict[str, Any]] = None) -> None:
        self.kind, self.state, self.question, self.change = kind, state, question, change or {}

    def expected(self, contact_id: str, key: str) -> Any:
        return (self.state.get(contact_id) or {}).get(key)


async def _plan(operation: str, contact: Any, *, value: Optional[str] = None, minutes: Optional[int] = None,
                drop: Any = None) -> _Plan:
    """Decide one change (``contacts.changes``): what it widens, never whether anything said in the turn
    asks for it. The caller holds ``change_lock`` and passes the records as read under it."""
    store, owner = _store(), _owner_id()
    handles = await store.get_handles(contact.contact_id)
    label = _label(contact, handles)
    if operation == "set_permission":
        current = people_changes.effective_permission(contact, owner_id=owner)
        kind = people_changes.permission_change(current, str(value))
        effect = {"auto": "messages to them would go out without asking you",
                  "ask": "messages to them would be proposed, each asked of you first"}.get(str(value), "")
        question = f"Set {label} to may_contact={value} (now {current}): {effect}"
        change = {"operation": operation, "contact_id": contact.contact_id, "value": value}
    elif operation == "set_cadence":
        current = contact.cadence_minutes
        kind = people_changes.cadence_change(current, minutes)
        permission = people_changes.effective_permission(contact, owner_id=owner)
        effect = {"auto": "each check-in would go out without asking you",
                  "ask": "each check-in would be asked of you first",
                  "never": "nothing is sent while they are set to never"}[permission]
        now = people_changes.every(current) if current else "no cadence"
        question = (f"Check in with {label} {people_changes.every(int(minutes))} (now {now}): {effect}"
                    if minutes else f"Clear the check-ins with {label}")
        change = {"operation": operation, "contact_id": contact.contact_id, "minutes": minutes}
    else:
        drop_handles = await store.get_handles(drop.contact_id)
        # Folded into the owner, a record always widens (``merge_reach``): never a free dedup.
        into_owner = bool(owner) and contact.contact_id == owner
        dedup = not into_owner and people_changes.obvious_dedup(handles, drop_handles)
        reach = [] if dedup else people_changes.merge_reach(contact, drop, handles, drop_handles, owner_id=owner)
        kind = "raise" if reach else "free"
        question = f"Merge {_label(drop, drop_handles)} into {label}: " + "; ".join(reach)
        change = {"operation": operation, "contact_id": contact.contact_id, "drop": drop.contact_id}
    # The records as decided, handles included: the write is made against them, and an ask's yes applies
    # only while they still are (a handle folded in meanwhile is a change the owner never saw).
    state = await people_changes.snapshot(store, [contact] + ([drop] if drop is not None else []))
    change["from"] = state
    if kind != "raise":
        return _Plan("free" if kind == "lower" else kind, state=state, change=change)
    return _Plan("raise", state=state, question=question, change=change)


CHANGED = ("not changed: the contact changed while this was decided (an opt-out, or another change); nothing "
           "was written, so decide it again on the contact as it is now")


def _changed() -> HTTPException:
    return HTTPException(status_code=409, detail={"code": "contact_changed", "message": CHANGED})


async def _fresh(contact: Any) -> Any:
    """``contact`` read again (under ``change_lock``): what the decision and the write start from."""
    current = await _store().get(contact.contact_id)
    if current is None:
        raise HTTPException(status_code=404, detail={"code": "unknown_contact",
                                                     "message": f"{contact.contact_id} is no longer a contact"})
    return current


def _refuse_unasked(plan: _Plan, *, operator: bool) -> None:
    """A widening from a direct route never applies: the operator's call is refused (it cites no owner turn), the
    owner's session is sent to their ask."""
    if plan.kind != "raise":
        return
    if operator:
        raise HTTPException(status_code=403, detail={
            "code": "approve_required", "message": f"not changed ({plan.question}): {APPROVE_REQUIRED}"})
    raise HTTPException(status_code=409, detail={
        "code": "needs_owner_ask", "message": f"not changed ({plan.question}): {NEEDS_ASK}"})


def _cadence_value(minutes: Optional[int]) -> Optional[int]:
    from colonymind.contacts.store import _cadence
    try:
        return _cadence(minutes)
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"code": "bad_cadence", "message": str(error)}) from None


async def _merge_pair(keep_ref: str, drop_ref: str) -> tuple[Any, Any]:
    keep, drop = await _resolve(keep_ref), await _resolve(drop_ref)
    if keep.contact_id == drop.contact_id:
        raise HTTPException(status_code=422, detail={"code": "same_contact", "message": "keep and drop are one contact"})
    if drop.contact_id == _owner_id():
        raise HTTPException(status_code=422, detail={"code": "owner_cannot_be_dropped",
                                                     "message": "merge the other record into the owner instead"})
    return keep, drop


async def _set_permission(contact: Any, value: str, performed_by: str, reason: str = "", *,
                          expected: Any) -> Dict[str, Any]:
    try:
        updated = await _store().set_may_contact(contact.contact_id, value, by=performed_by, reason=reason,
                                                 expected=expected)
    except ContactChanged:
        raise _changed() from None
    return {"ok": True, "contact_id": updated.contact_id, "may_contact": updated.may_contact,
            "text": f"{updated.display_name or updated.contact_id}: may_contact={updated.may_contact}"}


async def _set_cadence(contact: Any, minutes: Optional[int], performed_by: str, *, expected: Any) -> Dict[str, Any]:
    try:
        updated = await _store().set_cadence(contact.contact_id, minutes, by=performed_by, expected=expected)
    except ContactChanged:
        raise _changed() from None
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"code": "bad_cadence", "message": str(error)}) from None
    every = f"every {updated.cadence_minutes} min" if updated.cadence_minutes else "no cadence"
    return {"ok": True, "contact_id": updated.contact_id, "cadence_minutes": updated.cadence_minutes,
            "text": f"{updated.display_name or updated.contact_id}: {every}"}


@router.post("/merge",
             dependencies=[Depends(refuse_hand_session)])
async def merge(body: MergeBody, request: Request) -> Dict[str, Any]:
    """C2: fold ``drop`` into ``keep``; handles, sources, comms and affect follow the person. A merge that widens
    (``contacts.changes.merge_reach``) is refused: 409 ``needs_owner_ask`` from the owner's session, 403
    ``approve_required`` from the operator's call."""
    _require_people()
    performed_by = _require_owner(request, body.contact_id, body.by)
    keep, drop = await _merge_pair(body.keep, body.drop)
    async with people_changes.change_lock(_store()):
        keep, drop = await _fresh(keep), await _fresh(drop)
        plan = await _plan("merge", keep, drop=drop)
        _refuse_unasked(plan, operator=_operator(request, body.contact_id))
        return await _fold(keep, drop, performed_by, expected=plan.state)


async def fold_merge(store: Any, keep_id: str, drop_id: str, *, performed_by: str,
                     expected: Optional[Dict[str, Any]] = None) -> tuple[Any, int, int]:
    """C2 as every merge applies it (this route, and the owner's yes to an asked merge): fold ``drop_id``
    into ``keep_id`` with the other stores' ``reattribute`` hooks and the dropped record's sources, then move
    those sources in the ledger now. Returns the kept record and the sources moved and still pending; a
    refused fold raises ``ValueError`` (``ContactChanged`` when the pair is no longer ``expected``)."""
    merged = await store.merge(keep_id, drop_id, performed_by=performed_by,
                               reattribute=reattribute_hooks(reconcile=False), sources_of=person_sources,
                               expected=expected)
    reconciled, pending = await _reconcile(store, prefix=f"merge:{drop_id}:")
    return merged, reconciled, pending


async def _fold(keep: Any, drop: Any, performed_by: str, *, expected: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    store = _store()
    try:
        merged, reconciled, pending = await fold_merge(store, keep.contact_id, drop.contact_id,
                                                       performed_by=performed_by, expected=expected)
    except ContactChanged:
        raise _changed() from None
    except ValueError as error:
        raise HTTPException(status_code=409, detail={"code": "merge_refused", "message": str(error)}) from None
    row = _row(merged, await store.get_handles(merged.contact_id), full=True)
    return {"ok": True, "contact": row, "dropped": drop.contact_id, "sources_moved": reconciled,
            "sources_pending": pending, "text": f"merged {drop.contact_id} into {_line(row)}"}


@router.post("/changes",
             dependencies=[Depends(refuse_hand_session)])
async def change(body: ChangeBody, request: Request) -> Dict[str, Any]:
    """A change from the owner's session: applied at once when it narrows or widens nothing; otherwise the
    owner's one-tap ask (``Mind.request_people_change``), which their own reply ``yes <code>`` applies. The
    request's credential must speak for the owner, and the session it names must be the owner's."""
    operation = body.operation.strip()
    if operation not in people_changes.OPERATIONS:
        raise HTTPException(status_code=422, detail={"code": "unknown_operation",
                                                     "message": "set_permission, set_cadence or merge"})
    owner = _owner_id()
    if not owner or body.viewer != owner or not _speaks_for_owner(request):
        raise HTTPException(status_code=403, detail={
            "code": "not_owner", "message": "only the owner can change who may be contacted, cadences or merges"})
    if operation != "set_permission":
        _require_people()
    contact = await _resolve(body.contact_id)
    value, minutes, drop = None, None, None
    if operation == "set_permission":
        value = str(body.value or "").strip().lower()
        if value not in MAY_CONTACT:
            raise HTTPException(status_code=422, detail={"code": "unknown_permission", "message": "never, ask or auto"})
        if contact.contact_id == owner:
            raise HTTPException(status_code=422, detail={"code": "owner_is_auto",
                                                         "message": "the owner is always reachable"})
    elif operation == "set_cadence":
        minutes = _cadence_value(body.minutes)
    else:
        if not str(body.drop or "").strip():
            raise HTTPException(status_code=422, detail={"code": "drop_required",
                                                         "message": "drop (folded into contact_id) is required"})
        contact, drop = await _merge_pair(body.contact_id, str(body.drop))
    async with people_changes.change_lock(_store()):
        contact = await _fresh(contact)
        drop = await _fresh(drop) if drop is not None else None
        return await _decide(body, operation, contact, value=value, minutes=minutes, drop=drop, owner=owner)


async def _decide(body: ChangeBody, operation: str, contact: Any, *, value: Optional[str], minutes: Optional[int],
                  drop: Any, owner: str) -> Dict[str, Any]:
    """``/changes`` under ``change_lock``: decide the change on the records as read, then apply it against
    them (compare-and-set) or file the owner's ask bound to them."""
    plan = await _plan(operation, contact, value=value, minutes=minutes, drop=drop)
    if plan.kind == "unchanged":
        return {"ok": True, "status": "unchanged", "contact_id": contact.contact_id,
                "text": f"{contact.display_name or contact.contact_id}: nothing to change"}
    if plan.kind == "raise":
        from colonymind.api.routers.mind import get_mind
        request = getattr(get_mind(), "request_people_change", None)
        row = await request({**plan.change, "question": plan.question}) if callable(request) else None
        if row is None or not getattr(row, "ask_code", None):
            raise HTTPException(status_code=503, detail={
                "code": "needs_owner_ask",
                "message": "not changed: this widens who gets messaged and no ask can be filed (the mind is not "
                           "running)"})
        return {"ok": True, "status": "asked", "code": row.ask_code, "contact_id": contact.contact_id,
                "text": (f"Not changed yet: {plan.question}. Ask {row.ask_code} is open for the owner; only their own "
                         f"reply 'yes {row.ask_code}' applies it, nothing said in this turn.")}
    if operation == "set_permission":
        done = await _set_permission(contact, str(value), owner,
                                     expected=plan.expected(contact.contact_id, "may_contact"))
    elif operation == "set_cadence":
        done = await _set_cadence(contact, minutes, owner,
                                  expected=plan.expected(contact.contact_id, "cadence_minutes"))
    else:
        done = await _fold(contact, drop, owner, expected=plan.state)
    return {**done, "status": "applied"}


@router.post("/link",
             dependencies=[Depends(refuse_hand_session)])
async def link(body: LinkBody, request: Request) -> Dict[str, Any]:
    """Propose that a handle is this person: a candidate the owner confirms, never attribution."""
    _require_people()
    store = _store()
    # The candidates are the owner's to see: a request that speaks for the owner, from the owner's side.
    contact = await _resolve(body.contact_id, candidates_shown=_speaks_for_owner(request) and (
        body.by in {"owner", "cli"} or _is_owner_viewer(request, body.by)))
    holder = await store.resolve_messaging_handle(body.gateway, body.address)
    if holder is not None and holder.contact_id == contact.contact_id:
        return {"ok": True, "status": "already_linked", "contact_id": contact.contact_id,
                "text": f"{body.gateway}:{body.address} is already {contact.display_name or contact.contact_id}"}
    try:
        candidate = await store.propose_handle_link(
            contact.contact_id, body.gateway, body.address, source="proposal",
            evidence_refs=list(body.evidence_refs) or [f"proposed_by:{body.by}"])
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"code": str(error), "message": str(error)}) from None
    await store.record_audit(contact.contact_id, "link_proposed",
                             {"candidate_id": candidate["candidate_id"], "gateway": candidate["gateway"],
                              "held_by": holder.contact_id if holder else None}, performed_by=body.by)
    return {"ok": True, **candidate, "held_by": holder.contact_id if holder else None,
            "text": f"proposed {candidate['gateway']}:{candidate['address']} for "
                    f"{contact.display_name or contact.contact_id}; the owner confirms"}


@router.post("/{reference}/permission",
             dependencies=[Depends(refuse_hand_session)])
async def permission(reference: str, body: PermissionBody, request: Request) -> Dict[str, Any]:
    """7.4: ``may_contact`` lowered or kept; a raise is the owner's own answer, never this route's (409
    ``needs_owner_ask`` from their session, 403 ``approve_required`` from the operator's call)."""
    performed_by = _require_owner(request, body.contact_id, body.by)
    value = body.may_contact.strip().lower()
    if value not in MAY_CONTACT:
        raise HTTPException(status_code=422, detail={"code": "unknown_permission", "message": "never, ask or auto"})
    contact = await _resolve(reference)
    if contact.contact_id == _owner_id():
        raise HTTPException(status_code=422, detail={"code": "owner_is_auto",
                                                     "message": "the owner is always reachable"})
    async with people_changes.change_lock(_store()):
        contact = await _fresh(contact)
        plan = await _plan("set_permission", contact, value=value)
        _refuse_unasked(plan, operator=_operator(request, body.contact_id))
        return await _set_permission(contact, value, performed_by, body.reason,
                                     expected=plan.expected(contact.contact_id, "may_contact"))


@router.post("/{reference}/cadence",
             dependencies=[Depends(refuse_hand_session)])
async def cadence(reference: str, body: CadenceBody, request: Request) -> Dict[str, Any]:
    """The owner's check-in cadence in minutes; null clears it. A slower one or none applies at once; a new or
    faster one is the owner's own answer, never this route's (409 ``needs_owner_ask`` from their session, 403
    ``approve_required`` from the operator's call)."""
    _require_people()
    performed_by = _require_owner(request, body.contact_id, body.by)
    contact = await _resolve(reference)
    async with people_changes.change_lock(_store()):
        contact = await _fresh(contact)
        minutes = _cadence_value(body.minutes)
        plan = await _plan("set_cadence", contact, minutes=minutes)
        _refuse_unasked(plan, operator=_operator(request, body.contact_id))
        return await _set_cadence(contact, body.minutes, performed_by,
                                  expected=plan.expected(contact.contact_id, "cadence_minutes"))


async def _reconcile(store: Any, *, prefix: str) -> tuple[int, int]:
    """Move the merge's sources in the ledger now; whatever fails stays for the source worker."""
    ledger = _ledger()
    operations = [op for op in await store.pending_identity_reconciliations(limit=500)
                  if str(op.get("operation_id", "")).startswith(prefix)]
    if ledger is None:
        return 0, len(operations)
    from colonymind.api.routers.social_state import reconcile_identity_sources
    moved = 0
    for operation in operations:
        try:
            await reconcile_identity_sources(store, ledger, operation)
            moved += len(operation["affected_source_ids"])
        except ValueError as error:
            if str(error) in SOURCE_CONFLICTS:
                await store.mark_sources_conflicted(operation["operation_id"], str(error))
            else:
                logger.warning("people: source reconciliation for %s deferred (%s)", operation["operation_id"], error)
        except Exception as error:
            logger.warning("people: source reconciliation for %s deferred (%s)", operation["operation_id"],
                           type(error).__name__)
    left = [op for op in await store.pending_identity_reconciliations(limit=500)
            if str(op.get("operation_id", "")).startswith(prefix)]
    return moved, sum(len(op["affected_source_ids"]) for op in left)


__all__ = ["fold_merge", "person_sources", "reattribute_hooks", "reconcile_merge", "router"]
