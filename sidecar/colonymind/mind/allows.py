"""Standing allows: the owner's "always" on an ask, and the offers to stop asking (docs/MIND.md, Standing grants and
allows).

An ask whose class of work can stop asking offers "always" in its own message (``Offer``, recorded on the ask when it
forms, ``offer_for``), with exactly what it would allow in plain words (``words``). The owner's own "always" (their
yes, read as strictly as any widening answer and bound to that ask by a reaction, a quoted reply or their next message
within ``answers.FOLLOW_UP_HOURS``) settles the ask as a yes and adds one standing allow for exactly that class,
stored beside the standing grants in ``mind_state`` (``allow:<id>``, ``Allows``):

- ``contact``: messages and work for that contact go without asking: its permission becomes ``auto``, written
  compare-and-set against what the ask showed and performed as the owner's answer to that ask;
- ``toolset``: tasks and goals whose toolsets beyond the internal ones are exactly these (none an executor) act at
  ``standard``, and their calls (never a shell, code, browser or delegation call) run as the owner's approved work;
- ``grant``: a held structured floor call (never a shell, code, browser or delegation call) becomes a standing grant
  for exactly its class, its tool, its target and at most its amount, once a day.

A task with an executor toolset offers no "always": a standing permission covers work the owner never saw, and nothing
confines what a shell, code, browser or delegation call reaches to the task's workspace, so each such task keeps its
one ask (``authority.allow_key``).

An allow applies only at ``authority.ALLOW_LEVELS`` and never lifts the off switch, the deny list, the breaker or the
budgets. The console and the API list and revoke allows (a narrowing), and so do the owner's own words: "ask me again
about <thing>" (``answers.ask_again``) revokes every allow the thing names (``named_by``). Nothing but the owner's own
answer adds one. A revocation takes effect at once: a contact whose permission reset the contacts store could not take
is read as asking first until a later tick's retry lands it (``Allows.resetting``).

One line of the daily digest lists them once a week (``digest_line``) and offers an allow that let nothing go for
``UNUSED_DAYS`` for removal; the owner's "ask me again about" it removes it.

**Learned proposals.** After ``mind.asks.propose_after`` of the owner's own yeses in a row within ``PROPOSE_DAYS`` to
asks that offered the same "always", with no no of theirs between (an operator's answer and a lapse neither count
nor break the row; an undo breaks it), the mind asks once, as a widening ask of its own (``PROPOSAL``): stop asking
before it does that again? A yes adds the allow; a no or silence holds the offer back for ``HOLD_DAYS``, and only
yeses after the last offer or the last revocation of that allow count toward the next one.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .authority import (ALLOW_LEVELS, FLOOR_EFFECT, GRANT_CLASSES, INTERNAL_SAFE_TOOLSETS, allow_key, finite_amount,
                        grantable)

CONTACT, TOOLSET, GRANT = "contact", "toolset", "grant"
KEY = "allow:"                  # ``mind_state`` key prefix of an allow in force
RESET = "allow.reset:"          # ``mind_state``: a revoked contact allow whose permission reset has not landed yet
PROPOSAL = "allow_proposal"     # the ask a learned proposal forms
# The history actions an allow leaves: on the ask whose "always" (or proposal) added it, and on the notes that record
# its revocation (``allow_revoked``).
ALLOW_ADDED, ALLOW_REVOKED = "allow_added", "allow_revoked"
PROPOSE_AFTER = 3               # the approvals plan's default (``mind.asks.propose_after``; 0 turns the offers off)
PROPOSE_DAYS = 30               # the yeses in a row counted over this many days (the approvals plan's)
HOLD_DAYS = 30                  # a no or silence to an offer holds the next one back this long (the approvals plan's)
DIGEST_DAYS = 7                 # the digest lists the allows once in this many days (the approvals plan's weekly line)
UNUSED_DAYS = 90                # an allow that let nothing go this long is offered for removal (the plan's assumption)
DIGEST_KEY = "allows.digest_at"  # ``mind_state``: when a digest last listed them
DIGEST_ITEMS = 3                # the allows the line names; the rest are counted
_SLUG = re.compile(r"[^a-z0-9_.-]+")


@dataclass(frozen=True)
class Settings:
    """``mind.asks``: ``propose_after`` owner yeses in a row bring one offer to stop asking (0: never; a value
    missing, of the wrong type or out of 0 to 20 keeps the default, a builder's choice of range)."""
    propose_after: int = PROPOSE_AFTER

    @classmethod
    def from_config(cls, section: Any) -> "Settings":
        values = dict(section) if isinstance(section, Mapping) else {}
        raw = values.get("propose_after")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw) \
                or not 0 <= raw <= 20 or int(raw) != raw:
            return cls()
        return cls(propose_after=int(raw))


@dataclass(frozen=True)
class Offer:
    """What "always" on one ask would allow: its class of work (``key``, ``authority.allow_key``) and, for a held
    floor call, the grant it would add. Recorded on the ask (``context['always']``) when it forms."""
    kind: str
    target: str
    toolsets: Tuple[str, ...] = ()
    grant: Optional[Dict[str, Any]] = None

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.target}"

    @property
    def id(self) -> str:
        """The allow's id: its class in a form a path and a command line can carry."""
        return _SLUG.sub("-", f"{self.kind}-{self.target}".lower()).strip("-")[:80]

    def as_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {"kind": self.kind, "target": self.target, "key": self.key}
        if self.toolsets:
            value["toolsets"] = list(self.toolsets)
        if self.grant:
            value["grant"] = dict(self.grant)
        return value

    @classmethod
    def from_dict(cls, value: Any) -> Optional["Offer"]:
        """An offer as an ask recorded it; None for anything else (an ask formed before offers, a malformed one)."""
        if not isinstance(value, Mapping) or value.get("kind") not in {CONTACT, TOOLSET, GRANT}:
            return None
        target = str(value.get("target") or "").strip()
        if not target:
            return None
        grant = value.get("grant") if isinstance(value.get("grant"), Mapping) else None
        if value["kind"] == GRANT and grant is None:
            return None
        return cls(kind=str(value["kind"]), target=target,
                   toolsets=tuple(str(item) for item in value.get("toolsets") or ()),
                   grant=dict(grant) if grant is not None else None)


def _extra(toolsets: Any) -> Tuple[str, ...]:
    return tuple(sorted({str(name) for name in toolsets or ()} - INTERNAL_SAFE_TOOLSETS))


def offer_for(row: Any, *, cls: str, level: str, owner_id: str | None, toolsets: Any,
              executors: Any) -> Optional[Offer]:
    """The "always" an ask of class ``cls`` offers, or None. Only at a level where allows apply; only where a
    standing permission exists for its class: a message or task to a contact on ``ask`` (never the owner, never a
    contact the owner's own request already allowed, never one held to ask who a matched name is), an external task
    (its extra toolsets; never a task with an executor toolset), and a floor call held from a mind run whose tool a
    grant may name, with an exact target."""
    if level not in ALLOW_LEVELS:
        return None
    context = row.context if isinstance(getattr(row, "context", None), dict) else {}
    kind, recipient = str(getattr(row, "kind", "") or ""), getattr(row, "entity_id", None)
    if getattr(row, "type", None) == FLOOR_EFFECT:
        return _grant_offer(context)
    if cls == "contact":
        # A message held for the owner's word on who a name the store matched is (``ask_owner``) asks about the
        # person, not about the asking: its "always" would widen on the strength of a name.
        if not recipient or recipient == owner_id or context.get("may_contact") != "ask" or context.get("grant") \
                or context.get("ask_owner"):
            return None
        return Offer(CONTACT, str(recipient))
    if allow_key(kind=kind, cls=cls, toolsets=toolsets, recipient=recipient, executors=executors) is None:
        return None
    extra = _extra(toolsets)
    return Offer(TOOLSET, ",".join(extra), toolsets=extra)


def _grant_offer(context: Mapping[str, Any]) -> Optional[Offer]:
    """A held call's "always": a standing grant for exactly its class, tool and targets, at most its amount (money),
    once a day. Only a call a grant can name (``authority.grantable``), from a mind run, that its effect reads
    without a problem and whose targets it knows.

    Its class of work (the offer's key, and the grant's id) is the permission the grant gives: the call's class, tool,
    targets and currency, never the rest of the call (an invoice number, a memo; the complete call's digest binds the
    one-time yes, not the permission), so asks to pay the same payee share one key and their yeses count toward one
    offer to stop asking. The amount is the grant's limit, from the ask that offered it, not part of the key."""
    effect = context.get("effect") if isinstance(context.get("effect"), Mapping) else {}
    cls, tool = str(effect.get("cls") or ""), str(effect.get("tool") or "")
    if not context.get("origin_task") or cls not in GRANT_CLASSES or not grantable(tool) or effect.get("problem"):
        return None
    targets = sorted({str(item) for item in effect.get("recipients") or [] if str(item).strip()}) or \
        ([str(effect["target"]).strip()] if str(effect.get("target") or "").strip() else [])
    if not targets or any(mark in target for target in targets for mark in "*?["):
        return None
    currency = str(effect.get("currency") or "").strip().upper() if cls == "money_movement" else ""
    permission = hashlib.sha256(json.dumps([cls, tool, targets, currency], ensure_ascii=False).encode("utf-8"))
    grant: Dict[str, Any] = {"id": f"always-{permission.hexdigest()[:12]}", "class": cls,
                             "scope": " ".join(str(effect.get("text") or f"{tool} for {targets[0]}").split())[:200],
                             "tools": [tool], "match": targets, "per_day": 1}
    if cls == "money_movement":
        amount = finite_amount(effect.get("amount"))
        if amount is None:
            return None
        grant["max_amount"] = amount
        if currency:
            grant["currency"] = currency
    return Offer(GRANT, grant["id"], grant=grant)


def doing(offer: Offer, *, name: str | None = None) -> str:
    """What the allowed work is, in the owner's words: "contact Sam Rivera", "run tasks that use image_gen", or the
    held call itself."""
    tools = " and ".join(offer.toolsets or offer.target.split(","))
    if offer.kind == CONTACT:
        # The name as the owner knows it (never an internal id); without one, the person the ask is about.
        return f"contact {name if name and name != 'a contact' else 'this person'}"
    if offer.kind == TOOLSET:
        return f"run tasks that use {tools}"
    grant = offer.grant or {}
    scope = str(grant.get("scope") or "this call").rstrip(".")
    limit = " for at most that amount" if grant.get("max_amount") is not None else ""
    return f"make this kind of call again ({scope}){limit}, at most once a day"


def words(offer: Offer, *, name: str | None = None) -> str:
    """What "always" does, as the ask's reply line and the allow's listing say it."""
    return f"let me {doing(offer, name=name)} without asking first"


@dataclass(frozen=True)
class Allow:
    """One standing allow in force: its class of work, the words the owner agreed to, the ask their "always" (or
    their yes to an offer) answered, who answered and when, and how often it let work go without asking."""
    id: str
    kind: str
    target: str
    words: str
    ask: str
    code: str = ""
    by: str = ""
    created_at: str = ""
    toolsets: Tuple[str, ...] = ()
    grant: Optional[str] = None
    before: Optional[str] = None     # a contact's permission before the allow set it to ``auto``
    uses: int = 0
    last_used: Optional[str] = None

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.target}"

    def as_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "target": self.target, "key": self.key, "words": self.words,
                "ask": self.ask, "code": self.code, "by": self.by, "created_at": self.created_at,
                "toolsets": list(self.toolsets), "grant": self.grant, "before": self.before, "uses": self.uses,
                "last_used": self.last_used}

    @classmethod
    def from_dict(cls, value: Any) -> Optional["Allow"]:
        if not isinstance(value, Mapping) or value.get("kind") not in {CONTACT, TOOLSET, GRANT}:
            return None
        try:
            return cls(id=str(value["id"]), kind=str(value["kind"]), target=str(value["target"]),
                       words=str(value.get("words") or ""), ask=str(value.get("ask") or ""),
                       code=str(value.get("code") or ""), by=str(value.get("by") or ""),
                       created_at=str(value.get("created_at") or ""),
                       toolsets=tuple(str(item) for item in value.get("toolsets") or ()),
                       grant=str(value["grant"]) if value.get("grant") else None,
                       before=str(value["before"]) if value.get("before") else None,
                       uses=int(value.get("uses") or 0), last_used=value.get("last_used") or None)
        except (KeyError, TypeError, ValueError):
            return None


class Allows:
    """The allows in force, in ``mind_state`` (``allow:<id>``): what ``Authority`` consults and the console lists."""

    def __init__(self, mind_state: Any, *, clock: Any) -> None:
        self.mind_state = mind_state
        self.clock = clock

    def in_force(self) -> List[Allow]:
        found: List[Allow] = []
        for item in self.mind_state.items(KEY):
            if not str(item.get("key") or "").startswith(KEY):
                continue
            try:
                allow = Allow.from_dict(json.loads(item.get("text") or "{}"))
            except ValueError:
                allow = None
            if allow is not None:
                found.append(allow)
        return found

    def get(self, ident: str) -> Optional[Allow]:
        return next((allow for allow in self.in_force() if allow.id == str(ident or "").strip()), None)

    def for_key(self, key: str) -> Optional[Allow]:
        return next((allow for allow in self.in_force() if allow.key == key), None)

    def add(self, offer: Offer, *, words: str, ask: Any, by: str, now: datetime, grant: str | None = None,
            before: str | None = None) -> Allow:
        allow = Allow(id=offer.id, kind=offer.kind, target=offer.target, words=words, ask=str(ask.id),
                      code=str(ask.ask_code or ""), by=by, created_at=now.isoformat(),
                      toolsets=offer.toolsets, grant=grant, before=before)
        self._write(allow, now)
        return allow

    def used(self, ident: str, now: datetime) -> Optional[Allow]:
        """One piece of work went without asking because of this allow."""
        allow = self.get(ident)
        if allow is None:
            return None
        allow = replace(allow, uses=allow.uses + 1, last_used=now.isoformat())
        self._write(allow, now)
        return allow

    def remove(self, ident: str) -> Optional[Allow]:
        allow = self.get(ident)
        if allow is not None:
            self.mind_state.delete(f"{KEY}{allow.id}")
        return allow

    # A revoked contact allow whose permission reset failed (the contacts store could not take the write) is kept
    # here until the reset lands: while it waits, the mind reads that contact as asking first whatever their stored
    # permission says (``resetting_for``), and each tick tries the reset again.

    def resetting(self) -> List[Allow]:
        found: List[Allow] = []
        for item in self.mind_state.items(RESET):
            if not str(item.get("key") or "").startswith(RESET):
                continue
            try:
                allow = Allow.from_dict(json.loads(item.get("text") or "{}"))
            except ValueError:
                allow = None
            if allow is not None and allow.kind == CONTACT:
                found.append(allow)
        return found

    def resetting_for(self, contact_id: Any) -> Optional[Allow]:
        """The revoked allow whose reset of this contact's permission has not landed yet, if any."""
        return next((allow for allow in self.resetting() if allow.target == str(contact_id or "")), None)

    def hold_reset(self, allow: Allow, now: datetime) -> None:
        self.mind_state.set(f"{RESET}{allow.id}", text=json.dumps(allow.as_dict(), sort_keys=True), now=now)

    def reset_done(self, ident: str) -> None:
        self.mind_state.delete(f"{RESET}{ident}")

    def _write(self, allow: Allow, now: datetime) -> None:
        self.mind_state.set(f"{KEY}{allow.id}", text=json.dumps(allow.as_dict(), sort_keys=True), now=now)


def again(allow: Allow) -> str:
    """What an allow lets go without asking, as "I ask again before I ..." ends: "contact Sam Rivera"."""
    return allow.words.removeprefix("let me ").removesuffix(" without asking first").split(";")[0]


def _when(value: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00")) if value else None
    except ValueError:
        return None
    return parsed if parsed is None or parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def unused(allow: Allow, now: datetime) -> bool:
    """Whether an allow let nothing go without asking for ``UNUSED_DAYS`` (since it was added, or its last use)."""
    last = _when(allow.last_used) or _when(allow.created_at)
    return last is not None and now - last >= timedelta(days=UNUSED_DAYS)


def digest_line(allows: List[Allow], now: datetime) -> Optional[str]:
    """The daily digest's weekly line on the standing allows: what goes without asking (``DIGEST_ITEMS`` of them by
    name, the rest counted), then those unused for ``UNUSED_DAYS``, offered for removal, and how to bring an ask
    back. None without an allow."""
    if not allows:
        return None
    named = [again(allow) for allow in allows[:DIGEST_ITEMS]]
    more = len(allows) - len(named)
    text = (f"Standing allows: I do not ask first before I {'; '.join(named)}"
            f"{f'; and {more} more (colonymind mind allows lists them)' if more else ''}.")
    idle = [again(allow) for allow in allows if unused(allow, now)][:DIGEST_ITEMS]
    if idle:
        which = "it" if len(idle) == 1 else "them"
        return (f"{text} Not used for {UNUSED_DAYS} days, so I offer to remove {which}: {'; '.join(idle)}. "
                f"Say 'ask me again about' {'it' if len(idle) == 1 else 'one'} and I ask first again.")
    return f"{text} Say 'ask me again about' {'it' if len(allows) == 1 else 'one of them'} to bring its ask back."


# -- "ask me again about <thing>": which allows a thing names ----------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+(?:[_'-][a-z0-9]+)*")
# "ask me again about everything": every allow.
EVERYTHING = frozenset({"everything", "anything", "all"})
# Words that name no allow of their own: the frame around the thing ("before you use", "about contacting"), the kinds
# of work every allow shares, and polite words.
_PLAIN = frozenset("""
a an the to for with on in at of about before after when while from into by and or but so then now again first
please pls thanks thank thx you your u me my mine i it its this that these those them they him her his their
ask asking do doing does did make making made run running runs use using used uses let lets allow allowed
task tasks work job jobs call calls thing things stuff kind kinds sort type
contact contacting contacts contacted message messaging messages messaged text texting texts texted write writing
email emailing emails reach reaching out send sending sent computer machine without asking any time
""".split())


def _stem(word: str) -> str:
    """A word without its common English ending, so "paying" meets "pay" and "charts" meets "chart"."""
    for ending, least in (("ing", 6), ("ed", 5), ("es", 5), ("s", 4)):
        if word.endswith(ending) and len(word) >= least and not word.endswith("ss"):
            return word[:-len(ending)]
    return word


def _words(text: str) -> set:
    found = set()
    for word in _WORD.findall(str(text or "").lower()):
        found.add(_stem(word))
        found.update(_stem(part) for part in re.split(r"[_'-]", word) if part)
    return found


def named_by(thing: str, allows: List[Allow], *, names: Mapping[str, List[str]] | None = None) -> List[Allow]:
    """The allows a thing the owner asked to be asked about again names (``answers.ask_again``): each allow whose own
    words, class of work and names (``names``: a contact allow's current names, by allow id) hold every word of the
    thing that names something ("Sam", "image gen", "terminal", "acme"; "before you", "contacting" and "tasks" name
    nothing on their own), and every allow for "everything". None for a thing with no such word. Revoking is a
    narrowing: a thing read too widely brings an ask back, it never removes one."""
    words = [word for word in _WORD.findall(str(thing or "").lower()) if word not in _PLAIN]
    if not words:
        return []
    if set(words) <= EVERYTHING:
        return list(allows)
    wanted = {_stem(word) for word in words}
    found: List[Allow] = []
    for allow in allows:
        vocabulary = _words(" ".join([again(allow), allow.id, allow.target.replace(",", " ").replace(":", " "),
                                      *allow.toolsets, allow.grant or "",
                                      *((names or {}).get(allow.id) or [])]))
        if wanted <= vocabulary:
            found.append(allow)
    return found


def streak(answers: List[Tuple[datetime, Optional[bool]]], *, since: datetime) -> int:
    """The owner's yeses in a row, newest first, among ``answers`` (``(when, yes)``: True for the owner's own yes,
    False for their no or an undo, None for anything that neither counts nor breaks the row) at or after ``since``."""
    count = 0
    for _, yes in sorted((item for item in answers if item[0] >= since), key=lambda item: item[0], reverse=True):
        if yes is None:
            continue
        if not yes:
            break
        count += 1
    return count


def proposal_title(offer: Offer, count: int, *, name: str | None = None) -> str:
    """The offer to stop asking, in the owner's words: the question, then why it comes now."""
    return f"Stop asking before I {doing(offer, name=name)}? You said yes the last {count} times I asked."


def hold_until(settled_at: datetime) -> datetime:
    return settled_at + timedelta(days=HOLD_DAYS)


__all__ = ["ALLOW_ADDED", "ALLOW_REVOKED", "Allow", "Allows", "CONTACT", "DIGEST_DAYS", "DIGEST_ITEMS", "DIGEST_KEY",
           "EVERYTHING", "GRANT", "HOLD_DAYS", "KEY", "Offer", "PROPOSAL", "PROPOSE_AFTER", "PROPOSE_DAYS", "RESET",
           "Settings", "TOOLSET", "UNUSED_DAYS", "again", "digest_line", "doing", "hold_until", "named_by", "offer_for",
           "proposal_title", "streak", "unused", "words"]
