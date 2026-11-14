"""``/v1/secrets``: the sealed store's routes (docs/SECRETS.md).

Every route is the owner's: only a credential granted the owner reaches it (docs/CREDENTIALS.md), whatever a
request names as its lane, contact or session. The Hermes plugin is the caller of ``seal`` and ``resolve``, the
owner's tools and the local CLI of the rest. Conversation values leave through ``resolve``, and only for a purpose
the owner stands behind:

  POST /v1/secrets/seal         {text, owner?, contact_id?, session_id?}
                                 -> {text, sealed, withheld, refs, released: [[start, end], ...]}
                                 a gateway message sealed before Hermes sees it (the plugin's ingress hook), in
                                 the owner's scope when the plugin found the sender to be the owner, else in the
                                 sender's (their contact, or their handle when no contact has it). ``released``:
                                 the spans of the answer that hold a value the owner released (a false positive
                                 left as it came), which the plugin lets through
  POST /v1/secrets/resolve      {refs, purpose: send|fill|use|reveal, intention_id?, lane?, contact_id?,
                                 session_id?, own_refs?, action?, destination?, tool?}
                                 -> {values: {ref: value}, refused: {ref: reason}, asks: {ref: {code, text}},
                                     unavailable: {ref: reason}}
      send    the body, just before it sends an outbox message: the owner's own references in a message to the
              owner, or in the owner's dictated words to someone else; nothing else
      fill    a messaging, terminal or code call in the owner's own session (``action``: the call's tool and its
              complete arguments, references unfilled): a reference sealed from the owner's own words of their
              latest message (``own_refs``, which the plugin saw sealed at the gateway, seen by the vault in
              the last half hour) is filled; any other becomes a one-tap ask bound to the digest of the complete
              action and showing its ``destination`` in full, and the repeated identical call after the
              owner's yes is filled once
      use     ``colonymind_secret use`` in the owner's own session (to Hermes' credentials, a file, an autofill login)
      reveal  ``colonymind_secret reveal`` in the owner's own session
      A contact's value is never resolved, for any purpose. ``unavailable``: a value that cannot be read now (the
      key missing); the caller waits and asks again, it never sends without it.
  GET  /v1/secrets              ?contact_id  the references (labels, scopes, dates; never a value)
  GET  /v1/secrets/{ref}        one reference and its uses
  POST /v1/secrets/{ref}/release  a false positive: the value goes back into memory where the reference stood
  POST /v1/secrets/{ref}/forget   the value is deleted; the reference stays

The private Companion saved-password reveal is separate from conversation references and their tool calls:
``POST /v1/secrets/saved-password/reveal`` (docs/secret-reveal.md), issued owner relay only, never the adapter key.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, APIRouter, HTTPException, Request
from starlette.responses import JSONResponse
from colonymind.hands.sessions import check_hand_sessions, refuse_hand_session
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/secrets", tags=["secrets"])

# The one-tap ask a reference the owner did not name in this turn becomes (an owner question row in the mind).
SECRET_USE = "secret_use"
PURPOSES = ("send", "fill", "use", "reveal")
# How long the owner's own words stand behind a fill without an ask: the vault must have seen the value sealed
# from them that recently.
FRESH = timedelta(minutes=30)
# The largest action an ask can bind to (its complete arguments are kept for the owner to review).
MAX_ACTION_BYTES = 256 * 1024

_vault: Any = None


def set_vault(vault: Any) -> None:
    global _vault
    _vault = vault


def vault() -> Any:
    if _vault is not None:
        return _vault
    from colonymind.secrets.vault import get_vault
    return get_vault()


def _owner_id() -> Optional[str]:
    try:
        from colonymind.identity import get_owner_contact_id
        return get_owner_contact_id() or None
    except Exception:
        return None


def _mind() -> Any:
    from colonymind.api.routers.mind import get_mind
    return get_mind()


class SealRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(max_length=1_000_000)
    owner: bool = False
    contact_id: str = Field(default="", max_length=256)
    session_id: str = Field(default="", max_length=256)


class ResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    refs: List[str] = Field(min_length=1, max_length=32)
    purpose: Literal["send", "fill", "use", "reveal"]
    intention_id: Optional[str] = Field(default=None, max_length=128)
    lane: Literal["owner", "guest", "cron", "worker", "unknown"] = "unknown"
    contact_id: Optional[str] = Field(default=None, max_length=256)
    session_id: Optional[str] = Field(default=None, max_length=256)
    own_refs: List[str] = Field(default_factory=list, max_length=32)
    action: Optional[Dict[str, Any]] = None
    destination: Optional[str] = Field(default=None, max_length=MAX_ACTION_BYTES)
    tool: Optional[str] = Field(default=None, max_length=128)

    @field_validator("action")
    @classmethod
    def _bounded(cls, value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if value is not None and len(_canonical(value).encode("utf-8")) > MAX_ACTION_BYTES:
            raise ValueError("the action is too large to ask the owner about")
        return value


def _canonical(action: Dict[str, Any]) -> str:
    return json.dumps(action, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


@router.post("/seal")
async def seal(body: SealRequest) -> Dict[str, Any]:
    """Seal every detected value in ``text`` in ``contact_id``'s scope. A value the store cannot take is
    withheld (never returned as it came)."""
    from colonymind.secrets.refs import references_in
    from colonymind.secrets.seal import Scope, SealCount, seal_text
    scope = (_owner_id() or body.contact_id) if body.owner else body.contact_id
    if not scope:
        raise HTTPException(status_code=422, detail={"code": "no_scope",
                                                     "message": "name the owner or the sender's contact"})
    from colonymind.secrets.detect import find_secrets
    count = SealCount()
    store = vault()
    sealed = seal_text(body.text, scope=Scope(contact_id=scope, session_id=body.session_id), vault=store,
                       count=count)
    fresh = {item.ref_id: item.label for item in references_in(sealed)}
    released = []
    for hit in find_secrets(sealed):
        try:
            state = store.plan(hit.value, contact_id=scope)[1]
        except Exception:
            state = ""
        if state == "released":
            released.append([hit.start, hit.end])
    return {"text": sealed, "sealed": count.sealed, "withheld": count.withheld,
            "refs": [{"ref": ref, "label": label} for ref, label in fresh.items()], "released": released}


def _entry_or_404(ref: str) -> Dict[str, Any]:
    entry = vault().entry(ref)
    if entry is None:
        raise HTTPException(status_code=404, detail={"code": "unknown_reference"})
    return entry


def _action_digest(action: Optional[Dict[str, Any]]) -> str:
    """The digest an ask binds to: the complete canonical action (its tool and every argument), never a prefix."""
    return hashlib.sha256(_canonical(action or {}).encode("utf-8")).hexdigest()


def _fresh(entry: Dict[str, Any], now: datetime) -> bool:
    """Whether the vault saw this value sealed from a message within ``FRESH``."""
    try:
        seen = datetime.fromisoformat(str(entry.get("last_seen_at") or "").replace("Z", "+00:00"))
    except ValueError:
        return False
    return seen.tzinfo is not None and now - seen <= FRESH


def _context(row: Any) -> Dict[str, Any]:
    return row.context if isinstance(getattr(row, "context", None), dict) else {}


def _send_allowed(intention_id: Optional[str], ref: str, owner: Optional[str]) -> Optional[str]:
    """None when the outbox message ``intention_id`` may carry ``ref``'s value; else why not."""
    from colonymind.mind import audit
    mind = _mind()
    row = mind.store.get(intention_id) if mind is not None and intention_id else None
    if row is None or row.kind != "message" or row.status not in {"approved", "sending"}:
        return "no message of the mind is being sent with this reference"
    if ref not in audit.stored_text(row):
        return "the reference is not in the message's words"
    if audit.to_owner(row, owner):
        return None
    if audit.dictated(row):
        return None  # the owner's own words, which carry their own reference
    return "the mind composed this message itself: a private value is never sent in it"


def _approved_use(ref: str, digest: str) -> Any:
    mind = _mind()
    if mind is None:
        return None
    now = mind.clock()
    for row in mind.store.intentions(status=["approved"], kind=["note"], limit=200):
        context = _context(row)
        if (row.type == SECRET_USE and context.get("ref") == ref and context.get("digest") == digest
                and (row.expires_at is None or row.expires_at > now)):
            return row
    return None


def _ask(ref: str, entry: Dict[str, Any], action: Dict[str, Any], destination: Optional[str],
         digest: str) -> Dict[str, str]:
    """The owner's one-tap question for using ``ref`` in this very call, bound to the digest of its complete
    action; the destination is shown in full and the complete action kept with the ask for the owner to review.
    An open identical one is reused."""
    mind = _mind()
    if mind is None:
        raise LookupError("the mind is not running, so no ask can be formed: give the value in your own "
                          "message to use it")
    from colonymind.mind.authority import ask_expiry, new_ask_code
    store = mind.store
    tool = str(action.get("tool") or "a call")
    where = destination or "this call"
    text = f"Use {entry['label']} in {tool}, to: {where}?"
    for row in store.intentions(status=["asked"], kind=["note"], limit=200):
        context = _context(row)
        if row.type == SECRET_USE and context.get("ref") == ref and context.get("digest") == digest:
            return {"code": row.ask_code, "text": f"{text} yes {row.ask_code}"}
    now = mind.clock()
    code = new_ask_code(store.open_ask_codes())
    row, _ = store.create_intention(
        kind="note", type=SECRET_USE, title=f"Use {entry['label']} in {tool}"[:160], drive="upkeep", cls="owner",
        decision="ask", decision_reason="a sealed value not given in the owner's own words of this message",
        status="asked", dedup_key=None, recipient=getattr(mind, "owner_id", None) or _owner_id(), ask_code=code,
        context={"ref": ref, "label": entry["label"], "tool": tool, "destination": where, "digest": digest,
                 "action": _canonical(action), "notice": False, "concern": text},
        expires_at=ask_expiry(now, mind.policy), created_at=now)
    store.transition(row.id, "asked", action="asked", at=now, details={"code": code, "notice": False})
    return {"code": code, "text": f"{text} yes {code}"}


def _consume(row: Any) -> None:
    mind = _mind()
    now = mind.clock()
    mind.store.transition(row.id, "done", action="permit_used", at=now, outcome="done", verified="owner",
                          completed_at=now, details={"ref": _context(row).get("ref")})


@router.post("/saved-password/reveal", include_in_schema=False)
async def reveal_saved_password(request: Request) -> JSONResponse:
    """Private Companion relay path. Never a model/tool operation (docs/secret-reveal.md)."""
    from colonymind.api.auth import answer_credential, request_authority
    from colonymind.secrets.saved_login import SavedPasswordNotFound, saved_password

    headers = {"Cache-Control": "no-store"}

    def error(status: int, code: str) -> JSONResponse:
        return JSONResponse(status_code=status, content={"detail": {"code": code}}, headers=headers)

    authority = request_authority(request)
    if not answer_credential(authority, "companion") or authority.access != frozenset({"answer"}):
        return error(403, "owner_relay_required")

    def unique_object(pairs):
        # JSON's default last-value-wins rule must not hide a session selector.
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError()
            obj[key] = value
        return obj

    try:
        for key in ("session", "session_id", "requester"):
            if len(request.query_params.getlist(key)) > 1:
                raise ValueError()
        if len(request.headers.getlist("X-ColonyMind-Session")) > 1:
            raise ValueError()
        lengths = request.headers.getlist("content-length")
        if len(lengths) > 1 or (lengths and not 0 <= int(lengths[0]) <= 2048):
            raise ValueError()
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > 2048:
                raise ValueError()
            raw.extend(chunk)
        body = json.loads(raw, object_pairs_hook=unique_object)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        return error(422, "invalid_reveal_request")
    await check_hand_sessions(request, body=body)
    try:
        if not isinstance(body, dict) or set(body) != {"id", "session_id"}:
            raise ValueError()
        handle, session = body["id"], body["session_id"]
        if (not isinstance(handle, str) or not isinstance(session, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}", session)):
            raise ValueError()
        supplied = request.headers.get("X-ColonyMind-Session")
        if supplied is not None and supplied != session:
            raise ValueError()
    except (ValueError, TypeError, UnicodeError):
        # Validation must never echo an unexpected password/token/phone field.
        return error(422, "invalid_reveal_request")
    if not re.fullmatch(r"vault_[0-9a-f]{12}", handle):
        return error(404, "secret_not_found")
    try:
        password = await asyncio.to_thread(saved_password, handle)
        response = JSONResponse({"id": handle, "password": password}, headers=headers)
        # Existing durable vault use ledger, committed before the response leaves.
        # Store only the credential identity and a hash of the relay-bound session.
        detail = json.dumps({"credential": authority.principal_id,
                             "session_sha256": hashlib.sha256(session.encode()).hexdigest()}, separators=(",", ":"))
        vault().note_use(handle, "owner_reveal", detail)
        return response
    except SavedPasswordNotFound:
        return error(404, "secret_not_found")
    except Exception:
        # An unreadable vault or failed audit must never release an unaudited value.
        return error(503, "secret_reveal_unavailable")


@router.post("/resolve")
async def resolve(body: ResolveRequest) -> Dict[str, Any]:
    from colonymind.secrets.vault import SecretForgotten, UnknownReference, VaultUnavailable
    owner = _owner_id()
    store = vault()
    values: Dict[str, str] = {}
    refused: Dict[str, str] = {}
    asks: Dict[str, Dict[str, str]] = {}
    unavailable: Dict[str, str] = {}
    digest = _action_digest(body.action)
    from colonymind.util.temporal import now_utc
    now = now_utc()
    for ref in dict.fromkeys(body.refs):
        entry = store.entry(ref)
        if entry is None:
            refused[ref] = "unknown reference"
            continue
        if not owner or entry["scope_contact"] != owner:
            refused[ref] = "another person's private value is never used"
            continue
        if body.purpose == "send":
            why = _send_allowed(body.intention_id, ref, owner)
            if why:
                refused[ref] = why
                continue
        else:
            if body.lane != "owner" or (body.contact_id and body.contact_id != owner):
                refused[ref] = "only the owner's own conversation uses a private value"
                continue
            if body.purpose == "fill" and body.action is None:
                refused[ref] = "a fill names its complete action"
                continue
            if body.purpose == "fill" and not (ref in body.own_refs and _fresh(entry, now)):
                permit = _approved_use(ref, digest)
                if permit is None:
                    try:
                        asks[ref] = _ask(ref, entry, body.action, body.destination, digest)
                    except LookupError as error:
                        refused[ref] = str(error)
                    continue
                _consume(permit)
        try:
            values[ref] = store.resolve(ref)
        except SecretForgotten:
            refused[ref] = "the owner deleted this value"
            continue
        except UnknownReference:
            refused[ref] = "unknown reference"
            continue
        except VaultUnavailable as error:
            unavailable[ref] = f"the value cannot be read now ({type(error).__name__}); try again"
            continue
        tool = body.tool or str((body.action or {}).get("tool") or "")
        store.note_use(ref, body.purpose, (tool or body.intention_id or "")[:80])
    return {"values": values, "refused": refused, "asks": asks, "unavailable": unavailable}


@router.get("")
async def list_references(contact_id: Optional[str] = None, limit: int = 200) -> Dict[str, Any]:
    return {"entries": vault().entries(contact_id=contact_id, limit=limit), "counts": vault().counts()}


@router.get("/{ref}")
async def show(ref: str) -> Dict[str, Any]:
    entry = _entry_or_404(ref)
    return {**entry, "use_log": vault().uses(ref)}


@router.post("/{ref}/release")
async def release(ref: str) -> Dict[str, Any]:
    from colonymind.secrets.migrate import release as release_everywhere
    from colonymind.secrets.vault import SecretForgotten, VaultUnavailable
    entry = _entry_or_404(ref)
    if entry["scope_contact"] != _owner_id():
        raise HTTPException(status_code=403, detail={"code": "not_the_owners",
                                                     "message": "another person's private value stays sealed"})
    try:
        return {"ok": True, **release_everywhere(ref, vault=vault(), state_dir=vault().state_dir)}
    except SecretForgotten:
        raise HTTPException(status_code=409, detail={"code": "forgotten"}) from None
    except VaultUnavailable:
        raise HTTPException(status_code=503, detail={"code": "vault_unavailable"}) from None


@router.post("/{ref}/forget",
             dependencies=[Depends(refuse_hand_session)])
async def forget(ref: str) -> Dict[str, Any]:
    _entry_or_404(ref)
    return {"ok": True, "forgotten": vault().forget(ref)}


__all__ = ["PURPOSES", "SECRET_USE", "router", "set_vault", "vault"]

