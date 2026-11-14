"""Hands permit asks, read and answered by their id from an owner channel (docs/hands-permit-answer-contract.md).

A hands permit ask is the ``floor_effect`` note a hand's hold forms (``Mind._hold`` with origin ``hand``): the exact
call, its working directory and its bound targets, waiting for the owner's yes. Its RCS copy is answered by the
owner's message (``Mind.read_answer``: a reaction, a quoted reply or the next message). Another owner channel
(``companion``) answers it explicitly instead: by its immutable id and one of its offered choices
(``Mind.answer_explicit``). Both settle the same ask row through the same settle path (``Mind._settle_ask``) under
the same lock, and the settling write is guarded (``SettleGuard``): the first commit wins, and whoever comes second
reads the recorded winner here.

This module is the read side: which rows are permit asks, their full owner-facing terms (never clipped), the choices
they offer, and who settled them (the settling history row, after the ask's last revival). It never writes.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from urllib.parse import urlsplit

from colonymind.api.credentials import ANSWER_PLATFORMS
from colonymind.initiatives.models import StoredInitiative

from . import plain
from .answers import EXPLICIT
from .asks_report import ASK_REVIVED, settler
from .outbox import FLOOR_EFFECT, ask_sentence, floor_class, full_view_state
from .outcomes import owner_answer

#: The owner channels that answer a permit ask by its id, each through its relay's own ``answer`` credential
#: (``colonymind.api.credentials.ANSWER_PLATFORMS``). Messaging channels (RCS, WhatsApp, SMS) answer it by their own
#: message through ``POST /v1/mind/answers``, never here; and ``/answers`` and ``/asks/{id}/shown`` take these
#: platforms from the instance key only (the Hermes adapter's own chat lane), so an ``answer`` credential, or any other
#: credential granted the owner on them, can never bind an answer through the readers.
EXPLICIT_PLATFORMS = frozenset(ANSWER_PLATFORMS)
#: What a hand's permit ask offers, as its own message says ("Reply yes or no."): a hand-origin hold never offers
#: "always" (``allows._grant_offer`` needs a mind run's ``origin_task``).
CHOICES: Tuple[Dict[str, str], ...] = (
    {"value": "yes", "label": "Yes", "means": "the hand may make exactly this call, once"},
    {"value": "no", "label": "No", "means": "the hand does not make this call"},
)
CHOICE_VALUES = tuple(item["value"] for item in CHOICES)
#: A platform as such: what a request names is taken exactly as spelled, never lowercased or trimmed.
_PLATFORM = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,31}$")
#: The answers an explicit answer request ends with, and the HTTP status each answers with (a replay answers the same).
SETTLED, ALREADY_ANSWERED, ASK_CLOSED = "settled", "already_answered", "ask_closed"
STATUS_CODES = {SETTLED: 200, ALREADY_ANSWERED: 409, ASK_CLOSED: 409}

CAPTURE_KINDS = frozenset({"vault_save_login", "vault_code", "vault_login_capture", "vault_fill_confirm"})


def validate_payload(value: Any) -> Dict[str, Any]:
    """Validate a field description, never field values. Errors deliberately do not echo input.

    Fields are names only; arbitrary keys, defaults and values cannot enter the ask ledger.
    A capture answer still uses the existing yes/no choice contract.
    """
    def invalid() -> None:
        raise ValueError("invalid_capture_payload")

    if not isinstance(value, dict):
        invalid()
    fill = value.get("kind") == "vault_fill_confirm"
    required = {"kind", "origin", "ttl_s", "save_toggle_default", "fields"}
    required |= {"label"} if fill else {"site_hint"}
    if not required <= set(value) or set(value) - (required | ({"site_hint"} if fill else set())):
        invalid()
    if not isinstance(value["kind"], str) or value["kind"] not in CAPTURE_KINDS:
        invalid()
    for name, limit in (("origin", 2048), ("site_hint", 256)):
        if name not in value:
            continue
        item = value[name]
        if not isinstance(item, str):
            invalid()
        try:
            if len(item.encode("utf-8")) > limit or any(ord(c) < 32 or ord(c) == 127 for c in item):
                invalid()
        except UnicodeError:
            invalid()
    try:
        origin = urlsplit(value["origin"])
        if (origin.scheme != "https" or not origin.hostname or origin.username is not None
                or origin.password is not None or origin.path or origin.query or origin.fragment
                or value["origin"] != "https://" + origin.netloc
                or re.search(r"[\s\\/%]", origin.netloc)):
            invalid()
        origin.port  # reject malformed/out-of-range ports
    except ValueError:
        invalid()
    if type(value["ttl_s"]) is not int or not 1 <= value["ttl_s"] <= 120:
        invalid()
    if type(value["save_toggle_default"]) is not bool:
        invalid()
    fields = value["fields"]
    expected = [] if fill else ["code"] if value["kind"] == "vault_code" else ["username", "password"]
    if fields != expected or ((fill or value["kind"] == "vault_code") and value["save_toggle_default"]):
        invalid()
    if fill:
        from colonymind_shared.secret_detection import hide_secrets
        label = value["label"]
        if (not isinstance(label, str) or not 1 <= len(label) <= 120
                or any(ord(c) < 32 or ord(c) == 127 for c in label) or hide_secrets(label) != label):
            invalid()
        try:
            label.encode("utf-8")
        except UnicodeError:
            invalid()
    return {**value, "fields": list(fields)}


def create_capture(mind: Any, payload: Any, authorization_id: str) -> StoredInitiative:
    """Use Hands' creation path: a non-executing hand-origin hold, unique per authorization.

    The gateway keeps the session/page binding and all typed values in its own memory. Only
    an opaque random authorization id and this schema travel to the mind.
    """
    import uuid

    payload = validate_payload(payload)
    try:
        if str(uuid.UUID(authorization_id)) != authorization_id:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise ValueError("invalid_capture_authorization") from None
    call = json.dumps({"tool": "vault.capture", "args": {"authorization_id": authorization_id},
                       "targets": []}, sort_keys=True)
    text = (f"Fill your saved login for {payload['label']} on {payload['origin']}?"
            if payload["kind"] == "vault_fill_confirm" else "Enter a credential for " + payload["origin"])
    held = mind._hold(None, {"effect": {"cls": "credential_capture", "tool": "vault.capture",
                        "call": call, "digest": hashlib.sha256(call.encode()).hexdigest(),
                        "text": text},
                        "reason": "the owner supplies the value privately in Companion"},
                      task_id=authorization_id, now=mind.clock(), origin="hand", typed_payload=payload)
    row = mind.store.get(held["ask"])
    if row.context.get("payload") != payload:
        raise ValueError("capture_authorization_conflict")
    return row


class PermitAnswerError(Exception):
    """An explicit answer refused before it settled anything (nothing is recorded under its key)."""

    def __init__(self, code: str, status: int, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.code, self.status, self.message, self.extra = code, status, message, extra

    def detail(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, **self.extra}


def _context(row: Any) -> Dict[str, Any]:
    context = getattr(row, "context", None)
    return context if isinstance(context, dict) else {}


def _iso(value: Any) -> Optional[str]:
    if not isinstance(value, datetime):
        return None
    return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).isoformat()


def canonical(platform: str) -> bool:
    """Whether a platform is spelled as a platform name: lowercase letters, digits, '.', '_' or '-'."""
    return bool(_PLATFORM.match(platform))


def is_permit_ask(row: Optional[StoredInitiative]) -> bool:
    """Whether ``row`` is a hands permit ask: a ``floor_effect`` the hands controller's hold formed (its origin is
    recorded when it forms; nothing a caller labels makes a row one)."""
    return (row is not None and row.type == FLOOR_EFFECT and bool(row.ask_code)
            and _context(row).get("origin") == "hand")


def is_open(row: StoredInitiative, now: datetime) -> bool:
    """Open for an answer: still asked and not past its expiry (the tick's sweep may not have closed it yet)."""
    expires = row.expires_at
    if expires is not None and expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    return row.status == "asked" and (expires is None or expires > now)


def choices(row: StoredInitiative) -> List[Dict[str, str]]:
    """The choices this ask offers, as structured values."""
    return [dict(item) for item in CHOICES] if is_permit_ask(row) else []


def bound_call(row: StoredInitiative) -> Dict[str, Any]:
    """The exact call the ask binds (the hold's canonical binding): tool, arguments, working directory, targets."""
    effect = _context(row).get("effect") if isinstance(_context(row).get("effect"), dict) else {}
    try:
        call = json.loads(str(effect.get("call") or "")) if effect.get("call") else {}
    except ValueError:
        call = {}
    call = call if isinstance(call, dict) else {}
    return {"tool": call.get("tool") or effect.get("tool"), "arguments": call.get("args") or {},
            "workdir": call.get("cwd") or call.get("workdir"), "targets": call.get("targets") or []}


def terms(row: StoredInitiative, store: Any = None) -> Dict[str, Any]:
    """The ask's whole owner-facing terms: the words its own message shows (``ask_sentence``, in full, never
    clipped), its reply line, why it was held, and the exact call those words state. Whether its full text has been
    delivered is read as a yes reads it (``outbox.full_view_state``, which changes nothing)."""
    context = _context(row)
    effect = context.get("effect") if isinstance(context.get("effect"), dict) else {}
    full = context.get("full_view")
    delivered = full_view_state(store, row)[0] if store is not None else (not isinstance(full, dict)
                                                                         or bool(full.get("seen")))
    return {
        "text": ask_sentence(row),
        "reply": plain.reply_line(floor_class(row)),
        "reason": row.decision_reason or "",
        "effect_class": effect.get("cls"),
        "call": bound_call(row),
        "call_digest": context.get("effect_digest"),
        # Too long for one message: a yes counts only once the full text was delivered to the owner in its own
        # message (``Mind._settle_ask``); an explicit yes before that is refused (``full_text_not_delivered``).
        "full_text_required": isinstance(full, dict),
        "full_text_delivered": delivered,
    }


def _ordered(history: Iterable[Any]) -> List[Any]:
    def at(item: Any) -> datetime:
        value = item.timestamp
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return sorted(history, key=lambda item: (at(item), item.id or 0))


def settling_row(store: Any, row: StoredInitiative) -> Optional[Any]:
    """The history row that settled the ask since it last opened: the first approval naming who answered
    (``queued``), outcome or drop after its last revival (``asks_report``'s rule); None while it waits."""
    history = _ordered(store.get_history(row.id, limit=500))
    revivals = [index for index, item in enumerate(history) if item.action == ASK_REVIVED]
    for item in history[revivals[-1] + 1 if revivals else 0:]:
        details = item.details if isinstance(item.details, dict) else {}
        if (item.action == "queued" and details.get("by")) or item.action.startswith("outcome_") \
                or item.action == "dropped":
            return item
    return None


def _channel(answer: Mapping[str, Any]) -> Optional[str]:
    """The channel an owner answer came on: recorded since this release; for an earlier row, the platform its
    message key starts with (``platform:chat:message``; a ``turn:`` key names none)."""
    channel = str(answer.get("channel") or "").strip().lower()
    if channel:
        return channel
    message = str(answer.get("message") or "")
    prefix = message.split(":", 1)[0] if ":" in message else ""
    return prefix.lower() if prefix and prefix != "turn" else None


def settlement(store: Any, row: StoredInitiative) -> Optional[Dict[str, Any]]:
    """Who settled the ask and how: ``by`` (``owner``, ``operator``, ``lapse`` or ``mind``: ``asks_report.settler``),
    the outcome, when, and for the owner's own answer the winner: its channel, choice and path. None while open."""
    item = settling_row(store, row)
    if item is None:
        return None
    details = item.details if isinstance(item.details, dict) else {}
    outcome = "approved" if item.action == "queued" else item.action[len("outcome_"):] \
        if item.action.startswith("outcome_") else item.action
    answer = details.get("answer") if isinstance(details.get("answer"), dict) else {}
    value: Dict[str, Any] = {"by": settler(details.get("by"), outcome, details), "outcome": outcome,
                             "at": _iso(item.timestamp), "winner": None}
    if owner_answer(details.get("by"), details) is True:
        choice = "yes" if item.action == "queued" else "no" if outcome == "denied" else None
        value["winner"] = {"channel": _channel(answer), "choice": choice, "at": _iso(item.timestamp),
                           "path": answer.get("path")}
    return value


def view(store: Any, row: StoredInitiative, *, now: datetime) -> Dict[str, Any]:
    """One permit ask as an owner channel reads it: its immutable id, its state, its full terms, the choices it
    offers and, once settled, who settled it (the winner, when it was the owner's answer)."""
    settled = settlement(store, row) if row.status != "asked" else None
    result = {
        "id": row.id,
        "kind": "hands_permit",
        "code": row.ask_code,
        "hand_id": _context(row).get("held_from"),
        "status": row.status,
        "open": is_open(row, now),
        "outcome": row.outcome,
        "created_at": _iso(row.created_at),
        "expires_at": _iso(row.expires_at),
        "terms": terms(row, store),
        "choices": choices(row),
        "settlement": settled,
        "winner": settled["winner"] if settled else None,
    }
    if "payload" in _context(row):
        result["payload"] = validate_payload(_context(row)["payload"])
    return result


def closed(store: Any, row: StoredInitiative) -> Tuple[str, Dict[str, Any]]:
    """The answer to an explicit answer that came after the ask closed: ``already_answered`` with the recorded
    winner when the owner's own answer (on any channel) settled it, else ``ask_closed`` with how it closed (a lapse,
    the mind's withdrawal, the operator's no)."""
    settled = settlement(store, row)
    result = ALREADY_ANSWERED if settled is not None and settled["winner"] is not None else ASK_CLOSED
    return result, {"result": result, "ask_id": row.id, "status": row.status, "outcome": row.outcome,
                    "winner": settled["winner"] if settled else None, "settlement": settled}


def request_hash(ask_id: str, body: Mapping[str, Any]) -> str:
    """The digest an idempotency key is bound to: the ask the path names and the request body as validated."""
    canonical = json.dumps({"ask_id": ask_id, "body": dict(body)}, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def message_ref(channel: str, scope: str, key: str) -> str:
    """The answer's reference on the ask's settling row (``answer.message``, the permit's ``owner_answer_ref``):
    the channel and a digest of the caller's key, never the key itself."""
    digest = hashlib.sha256(f"{scope}\x00{key}".encode("utf-8")).hexdigest()[:32]
    return f"{channel}:{EXPLICIT}:{digest}"


__all__ = ["ALREADY_ANSWERED", "ASK_CLOSED", "CAPTURE_KINDS", "CHOICES", "CHOICE_VALUES", "EXPLICIT_PLATFORMS", "PermitAnswerError",
           "SETTLED", "STATUS_CODES", "bound_call", "canonical", "choices", "closed", "is_open", "is_permit_ask",
           "message_ref", "request_hash", "settlement", "settling_row", "terms", "view", "create_capture", "validate_payload"]
