"""The status block: what the agent may do on its own this turn, generated from the running mind.

An owner's turn carries a ``colonymind-status`` section titled "Your status" (docs/MIND.md, The status
block). It states the posture that changes what the agent may do on its own right now, and nothing else:
the autonomy level and whether the mind is on, quiet hours while they hold, a messaging budget that is
used up, outreach that is paused, a tripped breaker, the faculties switched off, and one line for each
problem the sidecar can see in process. Nothing is stored: every assemble renders it again from the
running mind, so it is never older than the turn it rides.

It is built to cost little. Every value is a coarse state, never a count or a time, so an unchanged
posture renders the same bytes and the memory provider sends its one "unchanged since your last turn"
line instead. A line whose source fails says ``<field>: unknown`` rather than a guess, and without a mind
there is no section at all, which the agent reads as "posture unknown", never as "all clear". It never
raises, and reads nothing but the mind's own stores and the health probes it is handed. On a turn
(``collect``) the lines that read a store cost it at most ``DEADLINE_S``: a slow or locked store makes its
own lines unknown, never the turn late.

It rides the per-turn context (the user-message side of the request), never the session prompt, so the
cached prompt prefix is untouched. It is owner-only, never judged or cut by the per-turn selection (it is
one of the pinned sections, ``memory.compass.PINNED_SECTIONS``), and ships off: the configuration's
``context.status`` exports ``COLONYMIND_CONTEXT_STATUS``, which this module only reads.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

from colonymind.util.temporal import now_utc
from colonymind.identity.policy import render as render_text

logger = logging.getLogger(__name__)

SECTION_ID = "colonymind-status"
TITLE = "Your status"
PRIORITY = 79               # above the Mind section (77) and its open questions (78)
MAX_CHARS = 600
REASON_CHARS = 60
# How long a turn waits for the lines that read a store. Their reads take about a millisecond; a store that
# is locked (the capture ledger's busy timeout is longer than the host waits for the whole prefetch) or
# stalled costs the turn this much at most, and those lines say unknown.
DEADLINE_S = 1.0
SWITCH_ENV = "COLONYMIND_CONTEXT_STATUS"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

# What each level lets go ahead unasked, as the authority's decision table has it (``authority.decide_table``,
# which the tests hold these to): a class named here acts at the level, everything else asks.
LEVEL_MEANINGS = {
    "off": "nothing runs on its own",
    "suggest": "internal work and updates the owner asked for go ahead; the rest waits for the owner",
    "standard": "internal work and messages to the owner go ahead, contacts by their permission; the rest asks",
    "trusted": ("internal work, messages to the owner and outside actions go ahead, contacts by their permission;"
                " the floor still asks"),
}
# Flags that choose how a faculty works rather than whether it runs: never listed as switched off.
# ``learning``: how lessons and claims are admitted (the learning rules), not whether the agent learns.
# ``wishes_named``: the form a wish takes (the model's own words beside the typed capability), not whether the agent
# wishes.
MODE_FLAGS = frozenset({"affect_rules", "learning", "wishes_named"})
BREAKER_CLASSES = ("internal", "owner", "contact", "external")
# A problem's fixed line, and the field a failing probe reports as unknown; shown in this order.
PROBLEMS = {
    "semantic_recall": ("Semantic recall", "Degraded: semantic recall unavailable, keyword search only."),
    "tick": ("Mind tick", "Degraded: the mind's tick has stalled, so reminders and your own initiative wait."),
    "body": ("Mind body", "Degraded: the mind's work is not being collected, so reminders and your own initiative wait."),
    "capture": ("Capture", "Degraded: capture is behind, so promises from recent turns may not be recorded yet."),
}
# The store each stored line reads: the budget, outreach and breaker lines read the mind's own store
# (``MIND_STORE``); a problem named here reads a store its probe opens a connection to. Every other line
# reads memory.
MIND_STORE = "mind"
PROBLEM_STORES = {"capture": "capture ledger"}
# Per store, reads abandoned by an earlier status collection or failed until its local deadline.
_store_reads: dict = {}
_store_reads_lock = threading.Lock()

# ``Outbox.ready`` holds every queued message in quiet hours but the digest (which the tests hold this to).
QUIET = ("Quiet hours now: queued messages and reminders wait until they end, except the digest; replies go out"
         " as usual.")
# Charged to the owner budget: reminders and heads-ups; exempt: the digest and its ask, breaker and health
# notices, and the reports owed to the owner (``Authority.budget_check``, which the tests hold this to). It
# says what the budget charges, never that a message goes out: in quiet hours the outbox still holds all but
# the digest (``QUIET``).
OWNER_BUDGET_USED = ("Messages to the owner: today's budget is used, so reminders and heads-ups wait; the digest,"
                     " its notices and owed reports do not count against it.")
OUTREACH_PAUSED = "Outreach to the owner: paused."
OUTREACH_USED = "Outreach to the owner: today's budget is used."
CONTACTS_NONE = "Messages to contacts: none allowed today."


def enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """The switch, read at every assemble so a changed setting holds from the next turn."""
    raw = (os.environ if environ is None else environ).get(SWITCH_ENV, "")
    return str(raw).strip().lower() in _TRUTHY


Problems = Optional[Mapping[str, Callable[[], bool]]]


def build(mind: Any, *, now: Optional[datetime] = None, problems: Problems = None, limit: int = MAX_CHARS) -> str:
    """The section body for ``mind``: "" without one. ``problems`` maps a ``PROBLEMS`` code to a probe that
    says whether that problem exists now; a probe that raises shows its field as unknown. Every line is read
    in the caller's thread, however long its store takes; a turn uses ``collect``."""
    if mind is None:
        return ""
    plan = _plan(mind, _now(mind, now), problems)
    return _render(plan, [_line(field, compute) for field, compute, _store, _part in plan], limit,
                   getattr(mind, "config", {}).get("status_replacements", []))


async def collect(mind: Any, *, problems: Problems = None, deadline: Optional[float] = None,
                  limit: int = MAX_CHARS) -> str:
    """Read each store on the bounded producer pool, within a local status deadline.

    Memory-only lines stay on the loop. Stored lines use worker-local read-only
    connections. An abandoned store is unknown on later turns until its worker
    finishes; a failed store is not retried before its local allowance ends.
    Other stores still supply their lines, and completed reads preserve build's bytes.
    """
    if mind is None:
        return ""
    from colonymind.memory import producers, turn_budget
    if turn_budget.current() is None:
        # Explicit status readers use the same worker facade as context assembly.
        with turn_budget.turn(turn_budget.TurnBudget(wait_s=DEADLINE_S if deadline is None else deadline)):
            return await collect(mind, problems=problems, deadline=deadline, limit=limit)
    ends = time.monotonic() + (DEADLINE_S if deadline is None else deadline)
    plan = _plan(mind, _now(mind, None), problems)
    values = [_line(field, compute) if not store else f"{field}: unknown."
              for field, compute, store, _ in plan]
    stores = {}
    for index, (_, _, store, _) in enumerate(plan):
        if store:
            stores.setdefault(store, []).append(index)
    collected = {}
    lock = threading.Lock()
    running = []

    async def stored(store, indexes, finished):
        entered = threading.Event()
        def read():
            entered.set()
            failed = False
            try:
                producers.limit_deadline(ends)
                for index in indexes:
                    if time.monotonic() >= ends or producers.expired():
                        break
                    field, compute = plan[index][:2]
                    value = _line(field, compute)
                    failed |= value == f"{field}: unknown."
                    if time.monotonic() < ends and not producers.expired():
                        with lock:
                            collected[index] = value
            finally:
                finished.set()
                if failed:
                    with _store_reads_lock:
                        _store_reads.setdefault(store, []).append((finished, ends, mind))
        try:
            await producers.read(read)
        finally:
            # A cancelled waiter cannot release a still-running store.
            if entered.is_set() and not finished.is_set():
                with _store_reads_lock:
                    _store_reads.setdefault(store, []).append((finished, 0, mind))

    for store, indexes in stores.items():
        store = (id(mind), store)
        with _store_reads_lock:
            for key in list(_store_reads):
                retained = [(done, retry_at, owner) for done, retry_at, owner in _store_reads[key]
                            if not done.is_set() or time.monotonic() < retry_at]
                if retained:
                    _store_reads[key] = retained
                else:
                    _store_reads.pop(key)
            outstanding = _store_reads.get(store)
            if outstanding:
                _store_reads[store] = outstanding
                continue
            _store_reads.pop(store, None)
        running.append(asyncio.create_task(stored(store, indexes, threading.Event())))
    try:
        if running:
            await asyncio.wait(running, timeout=max(0.0, ends - time.monotonic()))
    finally:
        for task in running:
            if not task.done():
                task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
    with lock:
        for index, value in collected.items():
            values[index] = value
    return _render(plan, values, limit, getattr(mind, "config", {}).get("status_replacements", []))


# -- the lines -------------------------------------------------------------------------------------------

Plan = List[Tuple[str, Callable[[], Optional[str]], str, str]]


def _now(mind: Any, now: Optional[datetime]) -> datetime:
    if now is not None:
        return now
    try:
        return mind.clock()
    except Exception:
        return now_utc()


def _plan(mind: Any, now: datetime, problems: Problems) -> Plan:
    """Every line in its order as ``(field, compute, store, part)``: the store it reads ("" for memory), and
    its part, head, posture or degraded. Reading the mind switch itself is memory only."""
    plan: Plan = [("Autonomy", lambda: _autonomy(mind), "", "head")]
    try:
        active = bool(mind.enabled)
    except Exception:
        active = True           # unknown: every line that depends on it says so itself
    if active:
        plan += [("Quiet hours", lambda: QUIET if mind.in_quiet_hours(now) else None, "", "posture"),
                 ("Messages to the owner", lambda: _owner_messages(mind, now), MIND_STORE, "posture"),
                 ("Outreach to the owner", lambda: _outreach(mind, now), MIND_STORE, "posture"),
                 ("Messages to contacts", lambda: _contact_messages(mind, now), MIND_STORE, "posture"),
                 ("Breaker", lambda: _breaker(mind, now), MIND_STORE, "posture")]
    plan.append(("Faculties", lambda: _faculties(mind.faculties), "", "posture"))
    for code, (field, text) in PROBLEMS.items():
        probe = (problems or {}).get(code)
        if probe is not None:
            plan.append((field, lambda probe=probe, text=text: text if probe() else None,
                         PROBLEM_STORES.get(code, ""), "degraded"))
    return plan


def _render(plan: Plan, values: Sequence[Optional[str]], limit: int, replacements=()) -> str:
    def part(name: str) -> List[str]:
        return [render_text(value, replacements) for (_field, _compute, _store, where), value in zip(plan, values)
            if where == name and value]
    return _fit(render_text(values[0], replacements), part("posture"), part("degraded"), limit)


def _line(field: str, compute: Callable[[], Optional[str]]) -> Optional[str]:
    try:
        return compute()
    except Exception as exc:
        from colonymind.memory import producers
        producers.read_failed(exc)
        logger.debug("status line %s unavailable", field, exc_info=True)
        return f"{field}: unknown."


def _autonomy(mind: Any) -> str:
    level = str(mind.level)
    meaning = LEVEL_MEANINGS.get(level)
    if level == "off":
        return f"Autonomy: off ({meaning})."
    if mind.enabled:
        return f"Autonomy: {level}{f' ({meaning})' if meaning else ''}. Mind: on."
    reason = _reason(getattr(mind, "off_reason", None))
    return f"Mind: off{f' ({reason})' if reason else ''}: nothing runs on its own. Autonomy when on: {level}."


def _reason(value: Any) -> str:
    text = " ".join("".join(ch if ch.isprintable() else " " for ch in str(value or "")).split())
    return text if len(text) <= REASON_CHARS else text[:REASON_CHARS - 1].rstrip() + "…"


def _owner_messages(mind: Any, now: datetime) -> Optional[str]:
    """The owner budget, which reminders and heads-ups from the mind spend (``Authority.budget_check``)."""
    owner = mind.owner_id
    if not owner:
        return None
    return OWNER_BUDGET_USED if mind.authority.budget_check(kind="message", recipient=owner, now=now) else None


def _outreach(mind: Any, now: datetime) -> Optional[str]:
    """Unprompted outreach to the owner: paused, or its own day's budget used. Off is the faculty line."""
    state = mind._outreach_state(now)
    if not state.get("enabled"):
        return None
    if state.get("paused_until"):
        return OUTREACH_PAUSED
    return OUTREACH_USED if int(state.get("sent_24h") or 0) >= int(state.get("per_day") or 0) else None


def _contact_messages(mind: Any, now: datetime) -> Optional[str]:
    """The day's cap on messages to contacts, used up or set to none (never a per-contact cooldown)."""
    return CONTACTS_NONE if mind.authority.budget_check(kind="message", recipient=None, now=now) else None


def _breaker(mind: Any, now: datetime) -> Optional[str]:
    tripped = [cls for cls in BREAKER_CLASSES if mind.authority.breaker_state(cls, now).get("tripped")]
    return f"Breaker tripped: {_joined(tripped)} acts become asks until it clears." if tripped else None


def _faculties(faculties: Mapping[str, Any]) -> Optional[str]:
    off = [str(name).replace("_", " ") for name, on in dict(faculties).items() if not on and name not in MODE_FLAGS]
    return f"Faculties off: {', '.join(off)}." if off else None


def _joined(names: Sequence[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _fit(head: Optional[str], posture: List[str], degraded: List[str], limit: int) -> str:
    """At most ``limit`` characters: lines go from the bottom of the posture first, then the degradation
    lines; the first line always stays."""
    def text() -> str:
        return "\n".join(line for line in [head, *posture, *degraded] if line)
    while len(text()) > limit and posture:
        posture.pop()
    while len(text()) > limit and degraded:
        degraded.pop()
    body = text()
    return body if len(body) <= limit else body[:max(0, limit - 1)].rstrip() + "…"


__all__ = ["DEADLINE_S", "LEVEL_MEANINGS", "MAX_CHARS", "MIND_STORE", "PRIORITY", "PROBLEMS", "PROBLEM_STORES",
           "SECTION_ID", "SWITCH_ENV", "TITLE", "build", "collect", "enabled"]
