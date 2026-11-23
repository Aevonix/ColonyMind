"""The outbox: messages the body sends verbatim, the asks' own messages and the daily digest.

Architecture 6.2 and 6.3. A message intention that authority approved waits
here as ``approved``. The plugin marks it ``sending`` before it sends and
``sent`` after; a message still ``sending`` when the sidecar starts is
``uncertain`` and is never resent. Each open ask goes to the owner as its own
message (``deliver_asks``); the digest goes out once a day.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from colonymind.initiatives.models import StoredInitiative

from . import audit, digest as digest_text, plain
from .audit import NOTICE_TYPES, outgoing_text, to_owner
from colonymind.util.temporal import now_utc

logger = logging.getLogger(__name__)

# A message the body claimed but never settled (it stopped between ``sending``
# and ``sent`` before its own ledger saw the send) is uncertain after this long.
STALE_SENDING = timedelta(minutes=10)


def _context(row: StoredInitiative) -> Dict[str, Any]:
    return row.context if isinstance(row.context, dict) else {}


def message_payload(row: StoredInitiative, *, owner_id: str | None) -> Dict[str, Any]:
    """What the body needs to send one message: its words exactly as they leave (``outgoing_text``: secrets
    hidden in what the mind composed for anyone but the owner, the owner's dictated words as they gave them)."""
    context = _context(row)
    recipient = row.entity_id or owner_id
    return {
        "id": row.id,
        "kind": "notice" if row.type in NOTICE_TYPES or (recipient and recipient == owner_id) else "message",
        "type": row.type,
        "dedup_key": f"mind:{row.id}",
        "recipient": recipient,
        "recipient_is_owner": bool(owner_id) and recipient == owner_id,
        "recipient_handles": list(context.get("recipient_handles") or []),
        "text": outgoing_text(row, owner_id=owner_id),
        "title": row.description,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "expires_at": row.expires_at.isoformat() if row.expires_at else None,
    }


# At most this many held items are listed in one digest; the rest stay held for the next, oldest first.
HELD_LINES = 12
# At most this many open asks and settled actions are listed in one digest.
ASK_LINES = 12
DONE_LINES = 16
# The mind's questions to the owner: answered in words, never with a code (``questions.py``).
QUESTION_TYPES = frozenset({"contradiction"})


# What a yes authorizes is shown in full, never shortened (``ask_parts``); past this many characters it cannot be
# shown in an ask's message, so it goes to the owner in its own message and a yes waits for that delivery.
MANDATORY_CHARS = 1500
ASK_IN_FULL = "ask_in_full"
UNDELIVERABLE = frozenset({"failed", "expired", "cancelled", "dropped", "uncertain"})
# A message that never reached the owner and never will (an uncertain one may have).
NEVER_DELIVERED = frozenset({"failed", "expired", "cancelled", "dropped"})
FLOOR_EFFECT, GRANT_CONFIRM = "floor_effect", "grant_confirm"
# An ask's own message (APPROVALS 8.1, 8.2): its type stays the 1.11 notice's, the owner-facing notice of an ask, which
# every reader already counts as the mind's reporting (``audit.NOTICE_TYPES``) and the owner gate as prompted. Its
# ``asks`` and ``list`` name the asks it shows, in order; ``ask_message`` says which send it is.
ASK_MESSAGE = "ask_notice"
FIRST, REMINDER, LIST = "first", "reminder", "list"
# The owner messages that put an ask in front of the owner or answer their word on one: they go to the chat they are in
# (``Outbox.where``), where their reply binds to them.
ASK_BEARING = frozenset({ASK_MESSAGE, ASK_IN_FULL, "ask_show", "ask_clarify", "ask_ack"})
# The line telling the owner that an ask was answered on an owner channel other than their own message (an explicit
# answer by the ask's id, ``Mind.answer_explicit``), so an answer they did not give is seen where they do read. It
# goes to the hand ask's RCS anchor when it has one (the RCS chat its task came from, as a reply there), else to their
# primary handle; never to the chat they last wrote in, which may be the very channel that answered.
ASK_ANSWERED = "ask_answered"
# The day an ask message counts in (``AskSettings.messages_per_day``), and the most asks one list shows (my choice: a
# list longer than a handful is hard to answer by numbers; the rest wait for the next day's room).
ASK_DAY = timedelta(hours=24)
LIST_ITEMS = 5
# The newest ask messages of each state read for the day's room (``Outbox._day_count``): past the most that can be on
# their way or delivered in a day at the highest ``messages_per_day`` (my choice).
DAY_COUNT_ROWS = 500
# An ask's one reminder goes right after the first digest delivered at least this long after its own message
# (APPROVALS 7.4), so the digest's waiting line and the re-sent message are the one reminder.
REMIND_AFTER_DIGEST = timedelta(hours=12)
# The answer route's record of a "later" on an ask (``asks_report.ASK_LATER``): its reminder moves to after the next
# digest.
ASK_LATER = "ask_later"
# The asks report's record of how long the owner gate held an ask's message for a pause (``asks_report.ASK_PAUSED``).
ASK_PAUSED = "ask_paused"


@dataclass(frozen=True)
class AskSettings:
    """``mind.asks``: how asks reach the owner (docs/MIND.md, Asks). Defaults are the plan's, not measured:
    ``messages_per_day`` ask messages in any 24 hours, then one list for the rest; the one reminder after the first
    digest at least 12 hours on, or ``remind_hours`` after the ask's own message when no digest comes first; and
    ``quiet_minutes``, the pause an ask waits for after the agent's last message (the owner gate; zero, none)."""
    messages_per_day: int = 3
    remind_hours: float = 24.0
    quiet_minutes: float = 10.0

    @classmethod
    def from_config(cls, section: Any) -> "AskSettings":
        """``mind.asks``; a value missing, of the wrong type or out of its range keeps its default (the ranges are
        a builder's choice: 1 to 50 messages a day, 1 to 168 hours, 0 to 240 minutes)."""
        values = dict(section) if isinstance(section, Mapping) else {}
        return cls(messages_per_day=int(_number(values.get("messages_per_day"), cls.messages_per_day, 1, 50)),
                   remind_hours=_number(values.get("remind_hours"), cls.remind_hours, 1, 168),
                   quiet_minutes=_number(values.get("quiet_minutes"), cls.quiet_minutes, 0, 240))


@dataclass(frozen=True)
class DueAsk:
    """One ask whose own message is due: ``purpose`` is which send it is (``first``, ``reminder``); ``own``: it goes
    as its own message whatever the day's count (a weighty ask, never approved from a list; an ask a 1.11 notice
    listed, sent once at the upgrade) and spends none of it."""
    row: StoredInitiative
    purpose: str
    own: bool = False


def _number(value: Any, default: float, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return default
    return float(value) if low <= value <= high else default


# At most this many summaries, within this many seconds in all, are made for one digest; the rest are pointed to.
SUMMARY_CALLS = 6
SUMMARY_SECONDS = 30.0


def _utc(value: Any) -> Optional[datetime]:
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def full_view_state(store: Any, row: StoredInitiative) -> tuple[bool, Optional[StoredInitiative]]:
    """Whether an ask too long to show has reached the owner in full, read from the durable record and changing
    nothing: ``(True, None)`` when it needs no full text or is marked seen; ``(True, notice)`` when a full-ask message
    the body reported ``sent`` counts though the hook that marks it seen never ran (``Outbox.full_view_seen`` marks
    it then); ``(False, None)`` otherwise."""
    current = store.get(row.id) or row
    full = _context(current).get("full_view")
    if not isinstance(full, dict) or full.get("seen"):
        return True, None
    # The notice it records and every attempt's own (a crash may have kept the record from naming it).
    rows = [store.get(str(full["notice"]))] if full.get("notice") else []
    rows += [store.get_by_dedup_key(f"{ASK_IN_FULL}:{current.id}:{attempt}")
             for attempt in range(1, int(full.get("attempts") or 0) + 2)]
    delivered = next((notice for notice in rows if notice is not None and notice.type == ASK_IN_FULL
                      and notice.status == "sent" and _context(notice).get("ask_id") == current.id), None)
    return delivered is not None, delivered


def floor_class(row: StoredInitiative) -> Optional[str]:
    """The floor class a yes to this ask lets happen (a held call's effect), or None."""
    effect = _context(row).get("effect")
    if row.cls != "floor" or not isinstance(effect, dict):
        return None
    return str(effect.get("cls") or "") or None


def _grant_words(spec: Mapping[str, Any]) -> str:
    """A standing grant's terms, every one in full, in words."""
    parts = []
    if spec.get("scope"):
        parts.append(f"for {spec['scope']}")
    if spec.get("class"):
        parts.append(f"covering {str(spec['class']).replace('_', ' ')}")
    if spec.get("tools"):
        parts.append(f"with the tools {', '.join(map(str, spec['tools']))}")
    if spec.get("match"):
        parts.append(f"only for {', '.join(map(str, spec['match']))}")
    if spec.get("max_amount") is not None:
        parts.append(f"at most {spec['max_amount']} {spec.get('currency') or ''} a time".replace("  ", " "))
    if spec.get("per_day") is not None:
        parts.append(f"at most {spec['per_day']} times a day")
    if spec.get("per_month_amount") is not None:
        parts.append(f"at most {spec['per_month_amount']} {spec.get('currency') or ''} a month".replace("  ", " "))
    if spec.get("expires"):
        parts.append(f"until {spec['expires']}")
    known = {"id", "scope", "class", "tools", "match", "max_amount", "currency", "per_day", "per_month_amount",
             "expires", "source"}
    parts += [f"{key}: {value}" for key, value in spec.items() if key not in known and value not in (None, "", [])]
    return "A standing grant " + "; ".join(parts) if parts else "A standing grant"


def ask_parts(row: StoredInitiative) -> Tuple[str, Optional[str]]:
    """An ask as ``(lead, what a yes authorizes)``: the lead may be shortened whole; the second, when the ask
    authorizes a destination or an effect (a private value's use, a held call, a standing grant), is shown in full
    or not at all."""
    context = _context(row)
    if (row.type == FLOOR_EFFECT and isinstance(context.get("payload"), dict)
            and context["payload"].get("kind") == "vault_fill_confirm"):
        from .permit_asks import validate_payload
        payload = validate_payload(context["payload"])
        return f"Fill your saved login for {payload['label']} on {payload['origin']}?", None
    if row.type == "wish" and isinstance(context.get("wish"), dict):
        # A wish: the lead (the mind's own words only under their label, only when the scope check showed them),
        # and everything a yes allows, typed and in full: a yes gives that and nothing its words say.
        from .wishes import ask_parts as wish_ask, wish_of
        return wish_ask(wish_of(row))
    title = plain.ended(row.description or row.type or "")
    if context.get("owner_question"):
        title = f"{title} {plain.ended(context['owner_question'])}"
    if row.type == "secret_use" and context.get("destination"):
        return title, f"It would use a private value in {context.get('tool')}, sent to: {context['destination']}"
    if row.type == FLOOR_EFFECT and isinstance(context.get("effect"), dict) and context["effect"].get("text"):
        return "Let this call run once.", f"The call: {context['effect']['text']}"
    if row.type == GRANT_CONFIRM and isinstance(context.get("grant"), dict):
        return "Allow a standing grant.", _grant_words(context["grant"])
    return title, None


def always_words(row: StoredInitiative, name: str | None = None) -> Optional[str]:
    """What "always" does on this ask (``allows.words``), when the ask offered it as it formed; else None."""
    from .allows import Offer, words
    offer = Offer.from_dict(_context(row).get("always"))
    return words(offer, name=name) if offer is not None else None


def ask_sentence(row: StoredInitiative, *, name: str | None = None) -> str:
    """What a yes to this ask does, in the owner's words, whole: for a message, the exact words it sends (in full:
    a yes approves what the owner saw); otherwise its lead and what a yes authorizes (``ask_parts``)."""
    words = audit.asked_words(row)
    if words is not None:
        hidden = " (private values are hidden in it)" if _context(row).get("hidden") else ""
        return f"Send {name or 'a contact'} this message{hidden}: \"{words}\""
    lead, authorized = ask_parts(row)
    return f"{lead} {plain.ended(authorized)}" if authorized else lead


class Outbox:
    def __init__(self, store: Any, *, owner_id: str | None, clock=None) -> None:
        self.store = store
        self.owner_id = owner_id
        self.clock = clock or (lambda: now_utc())
        self.on_sent = None   # callable(row) set by the tick: a sent message may settle what it was for
        # (text, limit) -> a checked shorter version or None, set by the tick (``plain.summarize`` on its router)
        self.summarize: Optional[Callable[[str, int], Awaitable[Optional[str]]]] = None
        # (row) -> "platform:chat" the owner is in for a message that carries an ask (``ASK_BEARING``), or None for
        # their primary handle; set by the tick (``Mind._owner_chat``), read as each message is offered to the body.
        self.where: Optional[Callable[[StoredInitiative], Optional[str]]] = None
        # A show caused by a recorded owner answer stays in that answer's chat.
        self.answer_ref: Optional[Callable[[StoredInitiative], Optional[Dict[str, str]]]] = None
        self._summary_room: Optional[Dict[str, Any]] = None

    # -- the queue --------------------------------------------------------------

    def ready(self, *, enabled: bool = True, quiet: bool = False) -> List[Dict[str, Any]]:
        """Messages the body may send now; nothing while the mind is off.

        During quiet hours only the digest goes out (it is scheduled outside
        them anyway); everything else waits for the next tick. A message
        unsent past its window is the mind's to expire (``Mind._expire_messages``,
        before every pull and tick), through the same settle path as a task.
        """
        if not enabled:
            return []
        now = self.clock()
        for row in self.store.intentions(status=["sending"], kind=["message"], limit=200):
            claimed = next((item.timestamp for item in self.store.get_history(row.id, limit=50)
                            if item.action == "sending"), None)
            if claimed is not None and claimed.tzinfo is None:
                claimed = claimed.replace(tzinfo=timezone.utc)
            if claimed is None or now - claimed >= STALE_SENDING:
                self.store.transition(row.id, "uncertain", action="uncertain", outcome="uncertain", verified="none",
                                      details={"reason": "no sent report after sending"}, at=now)
        rows = self.store.intentions(status=["approved"], kind=["message"], limit=200)
        ready = []
        for row in sorted(rows, key=lambda item: item.created_at):
            if quiet and row.type != "digest":
                continue
            if self._withdrawn(row):
                continue
            payload = message_payload(row, owner_id=self.owner_id)
            context = _context(row)
            if (payload["recipient_is_owner"] and row.type in {"ask_notice", "ask_show", "ask_clarify"}
                    and context.get("ask_message") != REMINDER):
                waits = []
                for ident in context.get("asks") or ([context["ask_id"]] if context.get("ask_id") else []):
                    ask = self.store.get(str(ident))
                    source = _context(ask) if ask is not None else {}
                    if source.get("origin") == "hand" and source.get("hand_approval_wait"):
                        waits.append(dict(source["hand_approval_wait"]))
                if waits:
                    payload["hand_approval_waits"] = waits
            if payload["recipient_is_owner"] and row.type in ASK_BEARING:
                answer = self.answer_ref(row) if self.answer_ref else None
                if answer:
                    route = {"target": f"{answer['platform']}:{answer['chat']}"}
                    if answer["platform"] == "rcs":
                        route["delivery_ref"] = answer
                    ready.append({**payload, **route})
                    continue
                try:
                    reference = self._ask_delivery_ref(row)
                except ValueError:
                    continue  # Conflicting ask routes never fall back to another chat.
                if reference:
                    ready.append({**payload, "target": f"rcs:{reference['chat']}", "delivery_ref": reference})
                    continue
            if payload["recipient_is_owner"] and row.type == ASK_ANSWERED:
                try:
                    reference = self._ask_delivery_ref(row)
                except ValueError:
                    continue  # Conflicting ask routes never fall back to another chat.
                ready.append({**payload, "target": f"rcs:{reference['chat']}", "delivery_ref": reference}
                             if reference else payload)
                continue
            target = self._where(row) if payload["recipient_is_owner"] and row.type in ASK_BEARING else None
            ready.append({**payload, "target": target} if target else payload)
        return ready

    def _ask_delivery_ref(self, row: StoredInitiative) -> Optional[Dict[str, str]]:
        """Use only controller-owned hand ask origins, never a message's claimed route. The line saying an ask was
        answered in an app names its ask as ``answered_ask``, not ``asks``: it shows no ask (``asks_report`` counts a
        sent message's ``asks`` as sights of them)."""
        context = _context(row)
        if row.type == ASK_ANSWERED:
            ids = [context["answered_ask"]] if context.get("answered_ask") else []
        else:
            ids = context.get("asks") or ([context["ask_id"]] if context.get("ask_id") else [])
        references = []
        for ident in ids:
            ask = self.store.get(str(ident))
            source = _context(ask) if ask is not None else {}
            reference = source.get("delivery_ref") if source.get("origin") == "hand" else None
            if reference is not None:
                if (not isinstance(reference, dict) or reference.get("platform") != "rcs"
                        or any(not isinstance(reference.get(key), str) or not reference[key].strip()
                               for key in ("chat", "message_id"))):
                    raise ValueError("invalid hand ask route")
                references.append({key: reference[key] for key in ("platform", "chat", "message_id")})
        if not references:
            return None
        if len(references) != len(ids) or any(value != references[0] for value in references):
            raise ValueError("conflicting hand ask routes")
        return references[0]

    def _where(self, row: StoredInitiative) -> Optional[str]:
        """The owner chat a message that carries an ask goes to (``where``); None: their primary handle, also when a
        message of this kind to that chat failed in the last day (a platform the body cannot send to is tried once,
        not at every tick)."""
        if not callable(self.where):
            return None
        try:
            target = self.where(row)
        except Exception as error:
            logger.warning("the owner's chat for %s unread (%s); their primary handle", row.id, type(error).__name__)
            return None
        target = str(target) if target and ":" in str(target) else None
        return None if target is None or self._failed_to(target) else target

    def _failed_to(self, target: str) -> bool:
        """Whether a message carrying an ask failed to reach the owner at ``target`` in the last ``ASK_DAY``."""
        for message in self.store.intentions(status=["failed"], kind=["message"], recipient=self.owner_id,
                                             since=self.clock() - ASK_DAY, limit=200):
            if message.type in ASK_BEARING and any(
                    item.action == "sending" and isinstance(item.details, dict) and item.details.get("target") == target
                    for item in self.store.get_history(message.id, limit=20)):
                return True
        return False

    def sending(self, intention_id: str, *, target: str | None = None) -> Optional[StoredInitiative]:
        """The body's claim before it sends: only a ready (``approved``) message, and only for a
        target the body resolved. A claim that names no target is not a claim: the body found no
        handle to send to, so the message stays ready (noted once as ``unroutable``) and is offered
        again at the next pull with the recipient's handles read afresh. Only a delivery the body
        attempted can end ``failed``."""
        row = self.store.get(intention_id)
        if row is None or row.kind != "message" or row.status not in {"approved", "sending"}:
            return None
        if row.type == ASK_ANSWERED:
            try:
                reference = self._ask_delivery_ref(row)
            except ValueError:
                return None
            if reference and target != f"rcs:{reference['chat']}":
                return None
        if row.type in ASK_BEARING:
            answer = self.answer_ref(row) if self.answer_ref else None
            if answer:
                if target != f"{answer['platform']}:{answer['chat']}":
                    return None
            else:
                try:
                    reference = self._ask_delivery_ref(row)
                except ValueError:
                    return None
                if reference and target != f"rcs:{reference['chat']}":
                    return None
        if row.status == "sending":
            return row
        now = self.clock()
        if not str(target or "").strip():
            # Noted once per row, not once per pull: a message that gains a target moves on to
            # ``sending`` and never comes back here.
            if not any(item.action == "unroutable" for item in self.store.get_history(intention_id, limit=100)):
                self.store.transition(intention_id, "approved", action="unroutable", at=now,
                                      details={"reason": "the body has no handle to send this to; it stays ready"})
            return None
        if self._withdrawn(row):
            return None
        if row.type in ASK_BEARING:
            self._note_pause(row, now)
        return self.store.transition(intention_id, "sending", action="sending", hermes_kind="message",
                                     details={"target": target}, at=now)

    def _note_pause(self, message: StoredInitiative, now: datetime) -> None:
        """How long the owner gate held an ask's message for a pause in the conversation (its first ``pause`` wait
        to the claim), recorded on each ask it shows (``ask_paused``, the asks report's)."""
        waits = [_utc(item.timestamp) for item in self.store.get_history(message.id, limit=100)
                 if item.action == "gated" and isinstance(item.details, dict) and item.details.get("stage") == "pause"]
        waits = [at for at in waits if at is not None]
        if not waits:
            return
        seconds = round(max(0.0, (now - min(waits)).total_seconds()), 3)
        context = _context(message)
        for ident in context.get("asks") or ([context["ask_id"]] if context.get("ask_id") else []):
            ask = self.store.get(str(ident))
            if ask is not None:
                self.store.transition(ask.id, ask.status, action=ASK_PAUSED, at=now,
                                      details={"seconds": seconds, "message": message.id})

    def sent(self, intention_id: str, *, result: str = "sent", hermes_ref: str | None = None,
             summary: str = "", error: str | None = None) -> Optional[StoredInitiative]:
        """``sent`` | ``failed`` | ``uncertain`` from the body; anything else is uncertain."""
        row = self.store.get(intention_id)
        if row is None or row.kind != "message":
            return None
        if row.status in {"sent", "uncertain", "failed"}:
            return row
        now = self.clock()
        result = str(result or "uncertain").lower()
        note = summary or error or None
        if result in {"sent", "ok", "done", "delivered"}:
            updated = self.store.transition(intention_id, "sent", action="sent", outcome="done", verified="none",
                                            hermes_kind="message", hermes_ref=hermes_ref or row.hermes_ref,
                                            result=note, completed_at=now, at=now)
            if callable(self.on_sent) and updated is not None:
                try:
                    self.on_sent(updated)
                except Exception as error:
                    logger.warning("sent hook failed for %s (%s)", intention_id, type(error).__name__)
            return updated
        if result in {"failed", "error"}:
            return self.store.transition(intention_id, "failed", action="send_failed", outcome="failed",
                                         verified="hermes_failure", hermes_kind="message",
                                         hermes_ref=hermes_ref or row.hermes_ref, result=note,
                                         failed_at=now, failed_reason=note or "send failed", at=now)
        return self.store.transition(intention_id, "uncertain", action="uncertain", outcome="uncertain",
                                     verified="none", hermes_kind="message", hermes_ref=hermes_ref or row.hermes_ref,
                                     result=note, details={"reason": error} if error else None, at=now)

    def recover(self) -> int:
        """At startup: a message left in ``sending`` is uncertain, never resent."""
        count = 0
        for row in self.store.intentions(status=["sending"], kind=["message"], limit=500):
            self.store.transition(row.id, "uncertain", action="uncertain", outcome="uncertain", verified="none",
                                  details={"reason": "sidecar restarted while sending"}, at=self.clock())
            count += 1
        return count

    def cancel_unsent(self, reason: str, *, owed: Iterable[str] = ()) -> int:
        """The off switch: unsent entries are cancelled at once, except a word owed to the owner (``owed``: a
        task's report, a requested answer), which is its obligation's only word: it goes back to waiting
        (deferred) and nothing sends it until the mind is back on and decides it again."""
        count, owed = 0, set(owed)
        for row in self.store.intentions(status=["approved"], kind=["message"], limit=500):
            if row.type in owed and (not row.entity_id or row.entity_id == self.owner_id):
                self.store.transition(row.id, "proposed", action="deferred", decision="defer",
                                      decision_reason=f"{reason}: owed to the owner, it waits for the mind",
                                      at=self.clock())
                continue
            self.store.transition(row.id, "cancelled", action="cancelled", outcome="cancelled",
                                  cancelled_at=self.clock(), cancelled_reason=reason, at=self.clock())
            count += 1
        return count

    # -- owner notices ----------------------------------------------------------------

    def _owner_message(self, *, type: str, title: str, text: str, dedup_key: str | None,
                       expires_in: timedelta = timedelta(days=2), expires_at: datetime | None = None,
                       extra: Mapping[str, Any] | None = None) -> Optional[StoredInitiative]:
        if not self.owner_id:
            logger.warning("no owner contact configured; %s notice not queued", type)
            return None
        now = self.clock()
        row, outcome = self.store.create_intention(
            kind="message", type=type, title=title, drive="upkeep", cls="owner", decision="act",
            decision_reason="owner notice", status="approved", dedup_key=dedup_key, recipient=self.owner_id,
            context={"text": text, "concern": title, **dict(extra or {})},
            expires_at=expires_at or now + expires_in, hermes_kind="message", created_at=now)
        if outcome == "created":
            self.store.transition(row.id, "approved", action="queued", at=now)
        return row if outcome == "created" else None

    # -- the asks' own messages (APPROVALS 8.1, 8.2) --------------------------------------------------------------

    def _asked_at(self, row: StoredInitiative) -> datetime:
        """When the row last became an ask (its latest ``asked`` history row, newest first), else when it was
        created."""
        at = next((_utc(item.timestamp) for item in self.store.get_history(row.id, limit=200)
                   if item.action == "asked"), None)
        return at or _utc(row.created_at) or self.clock()

    def _ask_messages(self, since: datetime) -> List[StoredInitiative]:
        return self.store.intentions(kind=["message"], types=[ASK_MESSAGE], recipient=self.owner_id, since=since,
                                     limit=2000)

    def _laters(self, row: StoredInitiative) -> List[datetime]:
        """When the owner said "later" to this ask (the answer route's ``ASK_LATER`` records)."""
        return [at for at in (_utc(item.timestamp) for item in self.store.get_history(row.id, limit=200)
                              if item.action == ASK_LATER) if at is not None]

    def _outdated(self, message: StoredInitiative) -> Optional[str]:
        """Why an ask's message not yet claimed must not go, or None: every ask it shows is settled, or the owner
        said "later" to one of them after it was queued. That ask's send moves to after the next digest
        (``asks_due``); a list goes whole or not at all (its numbers are its order), so the others it shows are
        queued again at the next tick, without that ask."""
        if message.type not in {ASK_MESSAGE, "ask_show"} or message.status != "approved":
            return None
        asks = [self.store.get(str(ident)) for ident in _context(message).get("asks") or []]
        if not asks:
            return None
        if all(row is None or row.status != "asked" for row in asks):
            return "the ask it shows is settled"
        queued = _utc(message.created_at)
        if queued is not None and any(row is not None and row.status == "asked"
                                      and any(at > queued for at in self._laters(row)) for row in asks):
            return "the owner said later to an ask it shows: it moves to after the next digest"
        return None

    def _withdrawn(self, message: StoredInitiative) -> bool:
        """Cancel an ask's message that must not go (``_outdated``), before it is offered or claimed."""
        reason = self._outdated(message)
        if reason is None:
            return False
        now = self.clock()
        self.store.transition(message.id, "cancelled", action="cancelled", outcome="cancelled", cancelled_at=now,
                              cancelled_reason=reason, at=now)
        return True

    def withdraw_settled(self) -> int:
        """Every ask's message not yet claimed that must not go (``_outdated``: its asks are settled, or a "later"
        came since it was queued) is cancelled."""
        return sum(1 for message in self.store.intentions(status=["approved"], kind=["message"],
                                                          types=[ASK_MESSAGE], limit=500)
                   if self._withdrawn(message))

    def asks_due(self, asks: Iterable[StoredInitiative], *, digests: Sequence[datetime] = (),
                 weighty: Iterable[str] = (), settings: AskSettings | None = None) -> List[DueAsk]:
        """The open asks whose own message is due now, oldest first (``DueAsk``):

        - ``first``: an ask that never had a message that reached the owner or may still; an ordinary one at once,
          one decided with ``notice: false`` (a suggest-level ask) after the first digest delivered since it was
          asked;
        - ``reminder``: its one re-send (APPROVALS 7.4), right after the first digest delivered at least
          ``REMIND_AFTER_DIGEST`` after its first message, or ``remind_hours`` after that message when no digest
          comes first. A "later" answer (``ASK_LATER``) moves it to after the next digest, once more, and a
          message of it already queued is withdrawn (``withdraw_settled``; ``ready`` and ``sending`` check it too).
          A "later" to an ask whose own message has not gone yet holds that message for the next digest as well.

        ``digests``: when each digest reached the owner. An ask whose message is on its way is not due.
        ``weighty``: the ids of the asks a list may not approve (``Mind._weighty``), each always its own
        message."""
        settings = settings or AskSettings()
        now = self.clock()
        self.withdraw_settled()
        rows = [row for row in asks if row.status == "asked" and row.ask_code and row.type not in QUESTION_TYPES]
        if not rows:
            return []
        weighty = set(weighty)
        asked = {row.id: self._asked_at(row) for row in rows}
        messages = self._ask_messages(min(asked.values()) - timedelta(days=1))
        # The codes a 1.11 notice (no ``asks`` of its own) listed and delivered: those asks go once, each on its own.
        listed_1_11 = {line.split(":", 1)[0].strip().upper() for message in messages
                       if message.status == "sent" and not _context(message).get("asks")
                       for line in digest_text.lines_of(str(_context(message).get("text") or ""),
                                                        digest_text.NEEDS_OK_1_11)}
        due: List[DueAsk] = []
        for row in sorted(rows, key=lambda item: asked[item.id]):
            sends = [message for message in messages if row.id in (_context(message).get("asks") or [])
                     and message.status not in NEVER_DELIVERED]
            if not sends:
                if _context(row).get("notice") is False and not any(at >= asked[row.id] for at in digests):
                    continue
                later = max(self._laters(row), default=None)
                if later is not None and not any(at > later for at in digests):
                    continue            # "later" before its own message went: that message waits for a digest too
                due.append(DueAsk(row, FIRST, own=row.id in weighty or str(row.ask_code).upper() in listed_1_11))
            elif self._reminder_due(row, sends, digests=digests, now=now, settings=settings):
                due.append(DueAsk(row, REMINDER, own=row.id in weighty))
        return due

    def _reminder_due(self, row: StoredInitiative, sends: List[StoredInitiative], *, digests: Sequence[datetime],
                      now: datetime, settings: AskSettings) -> bool:
        """Whether an ask's one reminder is due (``asks_due``): its first message reached the owner (or may have),
        none is on its way, and no message showed it since that first one or since its last "later"."""
        if any(message.status in {"approved", "sending"} for message in sends):
            return False
        delivered = sorted(at for at in (_utc(message.completed_at) or _utc(message.created_at) for message in sends
                                         if message.status in {"sent", "uncertain"}) if at is not None)
        if not delivered:
            return False
        first = delivered[0]
        later = max((at for at in self._laters(row) if at > first), default=None)
        since = later or first
        if any((_utc(message.created_at) or first) > since for message in sends):
            return False
        if later is not None:
            return any(at > later for at in digests)
        return any(at >= first + REMIND_AFTER_DIGEST for at in digests) \
            or now >= first + timedelta(hours=settings.remind_hours)

    def ask_words(self, row: StoredInitiative, *, name: str | None = None) -> str:
        """What a yes to this ask does, as its own message shows it: a message's exact words in full; otherwise its
        lead shortened whole and what a yes authorizes, in full or in its own message (``_with_authorized``)."""
        if audit.asked_words(row) is not None:
            return ask_sentence(row, name=name)
        lead, authorized = ask_parts(row)
        lead = (None if plain.clipped(row.description) else plain.fit(lead, digest_text.ITEM_CHARS)) \
            or plain.pointer("An action I would take")
        return self._with_authorized(row, lead, authorized)

    def deliver_asks(self, due: Iterable[DueAsk], *, names: Mapping[str, str] | None = None,
                     settings: AskSettings | None = None) -> List[StoredInitiative]:
        """Queue the due asks' messages (``asks_due``). Each goes as its own message, its words (``ask_words``) and
        its reply line (``plain.reply_line``) with no code, while the last 24 hours hold fewer than
        ``messages_per_day`` of them (``_day_count``: one still on its way counts whatever its age, one delivered
        by when it reached the owner); the rest go together in one numbered list (``digest.ask_list``, at most
        ``LIST_ITEMS``), at most one list in 24 hours; what does not fit waits for the room (the digest lists it
        meanwhile). A weighty ask, or one a 1.11 notice listed, always goes on its own and is not counted. Each
        message's ``asks`` and ``list`` name what it shows, in order, so a reaction, a quoted reply, the owner's
        next message or a list's numbers bind to it; it expires with its ask."""
        settings = settings or AskSettings()
        names = dict(names or {})
        now = self.clock()
        recent = self._day_count(now)
        singles = sum(1 for message in recent if _context(message).get("ask_message") in {FIRST, REMINDER}
                      and not _context(message).get("own"))
        lists = sum(1 for message in recent if _context(message).get("ask_message") == LIST)
        queued: List[StoredInitiative] = []
        listed: List[DueAsk] = []
        for item in due:
            routed = bool(_context(item.row).get("delivery_ref"))
            if item.own or routed or singles < settings.messages_per_day:
                singles += 0 if item.own else 1
                message = self._queue_ask(item, name=names.get(str(item.row.entity_id)))
                queued += [message] if message is not None else []
            elif not lists and len(listed) < LIST_ITEMS:
                listed.append(item)
        if listed:
            sentences = [self.ask_words(item.row, name=names.get(str(item.row.entity_id))) for item in listed]
            ids = [item.row.id for item in listed]
            message = self._owner_message(
                type=ASK_MESSAGE, title=f"{len(ids)} more ask(s)", text=digest_text.ask_list(sentences, plain.REPLY),
                dedup_key=f"{ASK_MESSAGE}:{LIST}:{now.isoformat()}",
                expires_at=min((item.row.expires_at for item in listed if item.row.expires_at), default=None),
                extra={"asks": ids, "list": ids, "ask_message": LIST})
            queued += [message] if message is not None else []
        return queued

    def _day_count(self, now: datetime) -> List[StoredInitiative]:
        """The ask messages that count in the day's room (``deliver_asks``): each one still on its way whatever its
        age (queued or claimed: it will reach the owner with the rest, so a body away for a day never lets a second
        day's worth pile up behind the first), and each one that reached the owner (or may have) by when it did,
        in the last ``ASK_DAY``."""
        def statuses(*status: str) -> List[StoredInitiative]:
            return self.store.intentions(status=list(status), kind=["message"], types=[ASK_MESSAGE],
                                         recipient=self.owner_id, limit=DAY_COUNT_ROWS)
        return [*statuses("approved", "sending"),
                *(message for message in statuses("sent", "uncertain")
                  if (self._delivered_at(message) or now) > now - ASK_DAY)]

    def _delivered_at(self, message: StoredInitiative) -> Optional[datetime]:
        """When a message reached the owner: its sent report; an uncertain one (it may have) at the body's claim of
        it; else when it was queued."""
        at = _utc(message.completed_at)
        if at is None and message.status == "uncertain":
            at = next((_utc(item.timestamp) for item in self.store.get_history(message.id, limit=50)
                       if item.action == "sending"), None)
        return at or _utc(message.created_at)

    def _queue_ask(self, item: DueAsk, *, name: str | None) -> Optional[StoredInitiative]:
        """One ask's own message: its words and its reply line."""
        row = item.row
        attempt = len([message for message in self._ask_messages(self._asked_at(row) - timedelta(days=1))
                       if row.id in (_context(message).get("asks") or [])]) + 1
        reply = plain.reply_line(floor_class(row), always=always_words(row, name))
        return self._owner_message(
            type=ASK_MESSAGE, title=f"Ask: {plain.clean(row.description or row.type)}"[:160],
            text=f"{self.ask_words(row, name=name)} {reply}",
            dedup_key=f"{ASK_MESSAGE}:{row.id}:{attempt}", expires_at=row.expires_at,
            extra={"asks": [row.id], "list": [row.id], "ask_message": item.purpose, "own": item.own,
                   # A floor ask goes at the owner gate's high urgency (it waits at most the pause); any other normal.
                   "urgency": "high" if row.cls == "floor" else "normal"})

    def _with_authorized(self, row: StoredInitiative, lead: str, authorized: Optional[str]) -> str:
        """An ask's lead and, in full, what a yes authorizes. Too long to show, it goes to the owner in its own
        message (``ask_in_full``) and a yes counts only once that message is delivered (``Mind.answer``)."""
        if not authorized:
            return lead
        if len(authorized) <= MANDATORY_CHARS:
            return f"{lead} {plain.ended(authorized)}"
        self.send_full_view(row, lead, authorized)
        return (f"{lead} What a yes would allow is too long to show here; I am sending it to you in full in its own "
                f"message, and a yes counts only once you have it.")

    def send_full_view(self, row: StoredInitiative, lead: str | None = None, authorized: str | None = None) -> None:
        """The whole ask in its own message, once, and again whenever the last one cannot arrive any more (it
        failed, expired, was cancelled or ended uncertain): until one is delivered a yes does not count."""
        current = self.store.get(row.id) or row
        context = _context(current)
        full = dict(context.get("full_view") or {}) if isinstance(context.get("full_view"), dict) else None
        if full is not None:
            if full.get("seen"):
                return
            notice = self.store.get(str(full.get("notice") or "")) if full.get("notice") else None
            if notice is not None and notice.status not in UNDELIVERABLE:
                return                          # on its way, or delivered (the hook marks it seen)
        if lead is None or authorized is None:
            lead, authorized = ask_parts(current)
        if not authorized:
            return
        attempt = int((full or {}).get("attempts") or 0) + 1
        notice = self._owner_message(type=ASK_IN_FULL, title="The full text of an ask",
                                     text=f"The full text of what I asked you: {lead} {plain.ended(authorized)} "
                                          f"{plain.reply_line(floor_class(current))}",
                                     dedup_key=f"{ASK_IN_FULL}:{current.id}:{attempt}", extra={"ask_id": current.id})
        self.store.update(current.id, context={**context, "full_view": {
            "seen": False, "notice": notice.id if notice else None, "attempts": attempt}})

    def refresh_full_views(self, asks: Iterable[StoredInitiative]) -> int:
        """Each open ask waiting for its full text to arrive gets it again when the last one cannot; one whose
        full text is recorded delivered is marked seen instead (``full_view_seen``)."""
        count = 0
        for row in asks:
            full = _context(row).get("full_view")
            if row.status == "asked" and isinstance(full, dict) and not full.get("seen"):
                if self.full_view_seen(row):
                    continue
                before = full.get("notice")
                self.send_full_view(row)
                count += (_context(self.store.get(row.id) or row).get("full_view") or {}).get("notice") != before
        return count

    def reconcile_full_views(self, asks: Iterable[StoredInitiative]) -> int:
        """At startup: each open ask whose full text is recorded delivered is marked seen (``full_view_seen``)."""
        count = 0
        for row in asks:
            full = _context(row).get("full_view")
            if row.status == "asked" and isinstance(full, dict) and not full.get("seen"):
                count += self.full_view_seen(row)
        return count

    def full_view_delivered(self, notice: StoredInitiative) -> None:
        """An ask's full text reached the owner in its own message: a yes to it counts from that delivery."""
        ask = self.store.get(str(_context(notice).get("ask_id") or ""))
        if ask is None or not isinstance(ask.context, dict):
            return
        full = dict(ask.context.get("full_view") or {})
        if full.get("seen"):
            return
        when = notice.completed_at or self.clock()
        full.update(seen=True, seen_at=when.isoformat(), seen_by=notice.id)
        self.store.update(ask.id, context={**ask.context, "full_view": full})

    def full_view_seen(self, row: StoredInitiative) -> bool:
        """Whether an ask too long to show has reached the owner in full (it needs no full text, or one arrived).
        The durable record decides: a full-ask message the body reported ``sent`` counts even when the hook that
        marks it seen never ran (it failed, or the process stopped between the two), and it is marked seen now."""
        seen, delivered = full_view_state(self.store, row)
        if delivered is not None:
            self.full_view_delivered(delivered)
        return seen

    def notice(self, *, type: str, title: str, text: str, dedup_key: str | None,
               extra: Mapping[str, Any] | None = None) -> Optional[StoredInitiative]:
        return self._owner_message(type=type, title=title, text=text, dedup_key=dedup_key, extra=extra)

    # -- the daily digest ------------------------------------------------------------------

    def digest_sent_today(self, local_date: str) -> bool:
        return self.store.get_by_dedup_key(f"digest:{local_date}") is not None

    async def _shown(self, text: Any, limit: int, *, what: str, cut_before: bool = False) -> str:
        """``text`` as whole sentences in ``limit`` characters, a checked summary, or a pointer (``plain.shown``)."""
        return await plain.shown(text, limit, what=what, cut_before=cut_before,
                                 summarize_with=self._bounded if self.summarize is not None else None)

    async def _bounded(self, text: str, limit: int) -> Optional[str]:
        """One summary inside the digest's allowance: at most ``SUMMARY_CALLS`` calls and ``SUMMARY_SECONDS`` in all
        (``build_digest`` opens it); outside it, none (the item is pointed to)."""
        room = self._summary_room
        if room is None or self.summarize is None or room["calls"] >= SUMMARY_CALLS:
            return None
        left = room["until"] - time.monotonic()
        if left <= 0:
            return None
        room["calls"] += 1
        try:
            return await asyncio.wait_for(self.summarize(text, limit), left)
        except asyncio.TimeoutError:
            return None

    async def _done_line(self, row: StoredInitiative, names: Mapping[str, str]) -> Optional[str]:
        """One settled or started action as a sentence; None for what the owner already received (a message to
        the owner reached the owner: it is never repeated here) and for what came to nothing."""
        title = str(row.description or row.type or "")
        cut = plain.clipped(title)
        room = digest_text.ITEM_CHARS
        if row.kind == "message":
            owner = to_owner(row, self.owner_id)
            name = "you" if owner else (names.get(str(row.entity_id)) or "a contact")
            # A message is shown by the words it carried (whole, or shortened whole), never by a stored title.
            words = outgoing_text(row, owner_id=self.owner_id) or title
            if row.status == "uncertain":
                return (f"I may not have delivered a message to {name}, and did not send it again: "
                        f"{await self._shown(words, room - 70, what='The message')}")
            if owner:
                return None
            if row.status == "sent":
                return f"Sent {name} a message: {await self._shown(words, room - 40, what='The message')}"
            if row.outcome == "failed":
                return f"Could not send {name} a message: {await self._shown(words, room - 40, what='The message')}"
            return None
        if row.kind not in {"task", "goal"} or row.decision != "act":
            return None
        what = "The task" if row.kind == "task" else "The goal"
        if row.outcome == "done":
            return f"Done: {await self._shown(title, room - 6, what=what, cut_before=cut)}"
        if row.outcome == "failed" or row.status == "failed":
            reason = str(row.failed_reason or row.result or "").strip()
            line = f"Could not finish: {await self._shown(title, 120, what=what, cut_before=cut)}"
            if reason:
                line += " " + await self._shown(reason, room - len(line) - 1, what="The reason it stopped")
            return line
        if row.status == "uncertain":
            return f"I cannot tell whether this finished: {await self._shown(title, room - 40, what=what, cut_before=cut)}"
        if row.status in {"approved", "dispatched", "running"}:
            return f"Started: {await self._shown(title, room - 9, what=what, cut_before=cut)}"
        return None

    async def build_digest(self, *, since: datetime, level: str, breaker_states: List[Dict[str, Any]] | None = None,
                           goals: List[str] | None = None, opt_outs: List[str] | None = None,
                           found: List[Tuple[str, str]] | None = None, offers: List[Tuple[str, str]] | None = None,
                           paused: str | None = None, held: List[Tuple[str, str]] | None = None, held_more: int = 0,
                           wishes: List[str] | None = None, drift: List[str] | None = None,
                           standing: List[str] | None = None,
                           names: Mapping[str, str] | None = None, work: str | None = None) -> Optional[str]:
        """The day's digest in two labelled parts (``digest.render``), or None when nothing waits and nothing
        happened. Waiting for your OK: every open ask in words, when it was asked, and no code (a question an earlier
        build raised is never one). Done or found since yesterday: the settled
        actions since ``since`` (never a message that already reached the owner), then owner outreach (what the mind
        ``found`` that bears on the owner's interests, the ``offers`` that went unsent or were put off, a ``paused``
        line), what the owner gate ``held`` for the digest (``(topic, text)``; ``held_more`` counts those left for the
        next one: the caller lists at most ``HELD_LINES`` and marks them listed only once this digest is delivered),
        the wishes' lines, the identity check's runs in which something moved (``drift``), opt-outs, uncertain
        deliveries and tripped breakers. The goals the mind keeps working on and the ``standing`` allows line (once a
        week) ride along, but never make a digest on their own. ``names`` maps contact ids to the names the owner knows them by. Every item is whole sentences:
        long text is shortened whole or summarised (``summarize``), never cut."""
        self._summary_room = {"calls": 0, "until": time.monotonic() + SUMMARY_SECONDS}
        try:
            return await self._build_digest(since=since, breaker_states=breaker_states, goals=goals,
                                            opt_outs=opt_outs, found=found, offers=offers, paused=paused, held=held,
                                            held_more=held_more, wishes=wishes, drift=drift,
                                            standing=standing, names=names, work=work)
        finally:
            self._summary_room = None

    async def _build_digest(self, *, since: datetime, breaker_states, goals, opt_outs, found, offers, paused, held,
                            held_more: int, wishes, drift, names, standing=None, work=None) -> Optional[str]:
        names = dict(names or {})
        room = digest_text.ITEM_CHARS
        needs: List[str] = []
        now = self.clock()
        asks = sorted((row for row in self.store.intentions(status=["asked"], limit=200)
                       if row.ask_code and row.type not in QUESTION_TYPES), key=self._asked_at)
        for row in asks[:ASK_LINES]:
            needs.append(await self._waiting_line(row, names.get(str(row.entity_id)), now))
        done: List[str] = []
        rows = [row for row in self.store.intentions(since=since, limit=500) if row.type not in NOTICE_TYPES]
        for row in rows:
            if len(done) >= DONE_LINES:
                break
            line = await self._done_line(row, names)
            if line:
                done.append(line)
        for topic, text in (found or [])[:6]:
            done.append(f"Found: {await self._shown(f'{topic}: {text}', room - 7, what=f'A finding about {topic}')}")
        if work:
            done.append(work)
        from colonymind.recorder import watch as recorder_watch
        done.extend(recorder_watch.digest_lines(self, since, now))   # the recorder's daily line
        for topic, text in (offers or [])[:6]:
            done.append(f"Offer: {await self._shown(f'{topic}: {text}', room - 7, what=f'An offer about {topic}')}")
        for topic, text in held or []:
            done.append(f"Held for this digest: {await self._shown(text or topic, room - 22, what=f'A note about {topic}')}")
        if held_more > 0:
            done.append(f"{held_more} more held item(s) come in the next digest.")
        for line in (wishes or [])[:4]:
            done.append(await self._shown(line, room, what="A note about a wish"))
        for line in (drift or [])[:4]:
            done.append(await self._shown(line, room, what="An identity check note"))
        for line in (opt_outs or [])[:12]:
            done.append(await self._shown(line, room, what="An opt-out note"))
        for state in breaker_states or []:
            if state.get("tripped"):
                done.append(f"After {state['failures']} failures in a row, I ask you first before any "
                            f"{state['cls']} work until {state['until']}.")
        if paused:
            done.append(plain.ended(paused))
        context = [await self._goal_line(goal) for goal in (goals or [])[:4]]
        context += [await self._shown(line, room, what="A note about what I do without asking")
                    for line in (standing or [])[:1]]
        return digest_text.render(needs=needs, done=done, context=context)

    async def _waiting_line(self, row: StoredInitiative, name: str | None, now: datetime) -> str:
        """One open ask as the digest lists it: what it is, whole, and when it was asked. No reply instruction: the
        ask's own message carries it, and a yes binds to that message, never to this line."""
        asked = digest_text.asked_ago(self._asked_at(row), now)
        room = digest_text.ITEM_CHARS - len(asked) - 1
        words = audit.asked_words(row)
        if words is not None:
            sentence = ask_sentence(row, name=name)
            what = sentence if len(sentence) <= room else \
                f"A message to {name or 'a contact'} (its words are in the ask I sent you)."
        else:
            lead, _ = ask_parts(row)
            what = await self._shown(lead, room, what="An action I would take",
                                     cut_before=plain.clipped(row.description))
        return f"{what} {asked}".strip()

    async def _goal_line(self, goal: Any) -> str:
        """A goal the mind keeps working on: its title whole (a title a store cut is summarised from the goal's full
        description, never shown cut), then its progress as a sentence of its own."""
        if not isinstance(goal, Mapping):
            return f"Still working on: {await self._shown(goal, digest_text.ITEM_CHARS - 18, what='A goal')}"
        title = str(goal.get("title") or "")
        cut = plain.clipped(title)
        name = title[len("Goal: "):] if title.startswith("Goal: ") else title
        source = str(goal.get("description") or "") if cut and goal.get("description") else name
        shown = await self._shown(source, 160, what="A goal", cut_before=cut and not goal.get("description"))
        progress = f"{goal.get('steps_done', 0)} of {goal.get('tasks') or goal.get('steps') or 0} steps done"
        horizon = str(goal.get("horizon") or "")[:10]
        return f"Still working on: {shown} So far {progress}" + (f", due {horizon}." if horizon else ".")

    def queue_digest(self, *, local_date: str, text: str, items: Mapping[str, Any] | None = None,
                     expires_at: datetime | None = None) -> Optional[StoredInitiative]:
        """The day's digest, written in one row with its own manifest (``items``: the questions, held items and
        findings it lists), so what it listed is recovered from it whatever happens after (``Mind._reconcile_digested``)."""
        return self._owner_message(type="digest", title=f"Daily digest {local_date}", text=text,
                                   dedup_key=f"digest:{local_date}", expires_in=timedelta(hours=20),
                                   expires_at=expires_at, extra={"items": dict(items or {})})


__all__ = ["ASK_ANSWERED", "ASK_BEARING", "ASK_DAY", "ASK_LATER", "ASK_LINES", "ASK_MESSAGE", "ASK_PAUSED",
           "AskSettings",
           "DONE_LINES", "DueAsk", "HELD_LINES", "LIST_ITEMS", "NOTICE_TYPES", "QUESTION_TYPES",
           "REMIND_AFTER_DIGEST", "STALE_SENDING", "Outbox", "ask_sentence", "floor_class", "full_view_state",
           "message_payload"]
