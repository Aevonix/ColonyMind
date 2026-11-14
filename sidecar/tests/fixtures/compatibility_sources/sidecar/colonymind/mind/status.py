"""Frozen compatibility oracle for the renderer behavior under comparison."""

from __future__ import annotations

import asyncio

import logging

import re

import sqlite3

import threading

import time

from contextlib import contextmanager

from datetime import datetime

from typing import Any, Callable, Iterator, List, Mapping, Optional, Sequence, Tuple

from colonymind.util.temporal import now_utc

logger = logging.getLogger(__name__)

MAX_CHARS = 600

REASON_CHARS = 60

DEADLINE_S = 1.0

_DASH = "[\u2010-\u2015\u2e3a\u2e3b\ufe58\ufe63\uff0d]"

_SPACED_DASH = re.compile(rf"\s+{_DASH}+\s+")

_ANY_DASH = re.compile(_DASH)

LEVEL_MEANINGS = {
    "off": "nothing runs on its own",
    "suggest": "internal work and updates the owner asked for go ahead; the rest waits for the owner",
    "standard": "internal work and messages to the owner go ahead, contacts by their permission; the rest asks",
    "trusted": ("internal work, messages to the owner and outside actions go ahead, contacts by their permission;"
                " the floor still asks"),
}

MODE_FLAGS = frozenset({"affect_rules", "learning", "wishes_named"})

BREAKER_CLASSES = ("internal", "owner", "contact", "external")

PROBLEMS = {
    "semantic_recall": ("Semantic recall", "Degraded: semantic recall unavailable, keyword search only."),
    "tick": ("Mind tick", "Degraded: the mind's tick has stalled, so reminders and your own initiative wait."),
    "body": ("Mind body", "Degraded: the mind's work is not being collected, so reminders and your own initiative wait."),
    "capture": ("Capture", "Degraded: capture is behind, so promises from recent turns may not be recorded yet."),
}

MIND_STORE = "mind"

PROBLEM_STORES = {"capture": "capture ledger"}

_stalled: dict = {}

_stalled_lock = threading.Lock()

QUIET = ("Quiet hours now: queued messages and reminders wait until they end, except the digest; replies go out"
         " as usual.")

OWNER_BUDGET_USED = ("Messages to the owner: today's budget is used, so reminders and heads-ups wait; the digest,"
                     " its notices and owed reports do not count against it.")

OUTREACH_PAUSED = "Outreach to the owner: paused."

OUTREACH_USED = "Outreach to the owner: today's budget is used."

CONTACTS_NONE = "Messages to contacts: none allowed today."

Problems = Optional[Mapping[str, Callable[[], bool]]]

def build(mind: Any, *, now: Optional[datetime] = None, problems: Problems = None, limit: int = MAX_CHARS) -> str:
    """The section body for ``mind``: "" without one. ``problems`` maps a ``PROBLEMS`` code to a probe that
    says whether that problem exists now; a probe that raises shows its field as unknown. Every line is read
    in the caller's thread, however long its store takes; a turn uses ``collect``."""
    if mind is None:
        return ""
    plan = _plan(mind, _now(mind, now), problems)
    return _render(plan, [_line(field, compute) for field, compute, _store, _part in plan], limit)

async def collect(mind: Any, *, problems: Problems = None, deadline: Optional[float] = None,
                  limit: int = MAX_CHARS) -> str:
    """``build`` for a turn, bounded: its store reads cost the turn at most ``deadline`` seconds
    (``DEADLINE_S``), and a line not read by then says ``<field>: unknown``. With every read in time, the
    bytes are ``build``'s.

    The mind's own store (``MIND_STORE``) is read where every other use of it runs, on the event loop:
    it has one connection, SQLite serializes each call on it, and the Mind section right after the block
    reads it too, so a read left running on another thread would hold the store, and the turn, until it
    ended. Its reads carry the deadline themselves instead (the progress handler the turn's work sections
    put on theirs): a read still running at the deadline is interrupted, and the store's later lines are
    not read. A lock cannot hold them: WAL readers never wait on a writer, and the connection the sidecar
    keeps open stops any other process from taking the store exclusively.

    A problem in ``PROBLEM_STORES`` opens its own connection, whose busy timeout can outlast the whole
    prefetch: it runs on a worker thread, one per store, while the loop goes on. While an earlier turn's
    read of such a store is still running past its deadline, its lines say unknown at once and no second
    read starts, so a wedged store never holds a thread per turn."""
    if mind is None:
        return ""
    ends = time.monotonic() + (DEADLINE_S if deadline is None else deadline)
    plan = _plan(mind, _now(mind, None), problems)
    stores: dict = {}
    for index, (_field, _compute, store, _part) in enumerate(plan):
        if store:
            stores.setdefault(store, []).append(index)
    read: dict = {}
    lock = threading.Lock()

    def run(indexes: List[int], finished: threading.Event) -> None:
        try:
            for index in indexes:
                value = _line(plan[index][0], plan[index][1])
                with lock:
                    read[index] = value
        finally:
            finished.set()
    loop = asyncio.get_running_loop()
    running = {}
    for store, indexes in stores.items():
        if store != MIND_STORE and not _still_stalled(store):
            finished = threading.Event()
            running[store] = (loop.run_in_executor(None, run, indexes, finished), finished)
    values: List[Optional[str]] = [None if store else _line(field, compute) for field, compute, store, _part in plan]
    with _until(_mind_connections(mind), ends):
        for index in stores.get(MIND_STORE, []):
            field, compute = plan[index][0], plan[index][1]
            values[index] = _line(field, compute) if time.monotonic() < ends else f"{field}: unknown."
    if running:
        await asyncio.wait({future for future, _finished in running.values()},
                           timeout=max(0.0, ends - time.monotonic()))
        with _stalled_lock:
            for store, (_future, finished) in running.items():
                if not finished.is_set():
                    _stalled.setdefault(store, set()).add(finished)
    with lock:
        read = dict(read)
    for store, indexes in stores.items():
        if store != MIND_STORE:
            for index in indexes:
                values[index] = read[index] if index in read else f"{plan[index][0]}: unknown."
    return _render(plan, values, limit)

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

def _render(plan: Plan, values: Sequence[Optional[str]], limit: int) -> str:
    def part(name: str) -> List[str]:
        return [_plain(value) for (_field, _compute, _store, where), value in zip(plan, values)
            if where == name and value]
    return _fit(_plain(values[0]), part("posture"), part("degraded"), limit)

def _mind_connections(mind: Any) -> List[sqlite3.Connection]:
    """The connections the mind-store lines read through: the intentions store's and the mind state's."""
    found: List[sqlite3.Connection] = []
    for conn in (getattr(getattr(mind, "store", None), "_db", None),
                 getattr(getattr(getattr(mind, "authority", None), "store", None), "_db", None),
                 getattr(getattr(mind, "mind_state", None), "conn", None)):
        if isinstance(conn, sqlite3.Connection) and all(conn is not seen for seen in found):
            found.append(conn)
    return found

@contextmanager
def _until(connections: Sequence[sqlite3.Connection], ends: float) -> Iterator[None]:
    """While open, a statement on ``connections`` is interrupted at ``ends`` (``time.monotonic``). Only the
    block's reads run inside it: the stores are used from the event loop alone, and nothing else runs on the
    loop until it closes, when the connections are left as they were."""
    armed = []
    for conn in connections:
        try:
            conn.set_progress_handler(lambda: int(time.monotonic() >= ends), 1000)
            armed.append(conn)
        except Exception:
            logger.debug("status deadline not set on a store", exc_info=True)
    try:
        yield
    finally:
        for conn in armed:
            try:
                conn.set_progress_handler(None, 0)
            except Exception:
                logger.debug("status deadline not cleared from a store", exc_info=True)

def _still_stalled(store: str) -> bool:
    """Whether an earlier turn's reads of ``store`` are still running past their deadline."""
    with _stalled_lock:
        waiting = {finished for finished in _stalled.get(store, ()) if not finished.is_set()}
        if waiting:
            _stalled[store] = waiting
        else:
            _stalled.pop(store, None)
        return bool(waiting)

def _line(field: str, compute: Callable[[], Optional[str]]) -> Optional[str]:
    try:
        return compute()
    except Exception:
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

def _plain(line: Optional[str]) -> Optional[str]:
    """``line`` with no dash glyph, whatever text it carries (an off reason is free text)."""
    return _ANY_DASH.sub("-", _SPACED_DASH.sub(", ", line)) if line else line

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
