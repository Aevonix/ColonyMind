"""The asks report: how the owner's asks went over a window (docs/MIND.md, Asks).

Read-only over the intention store (``InitiativeStore``: the rows and their history), counts and times only, never
an ask's words. Served as ``GET /v1/mind/asks/stats?since_hours`` (the owner's view) and ``colonymind mind
asks-report --since 7d`` (text, or ``--json``). It is the instrument approvals are measured with, before and after
a change: what the owner was asked, when they first saw it, who settled it and how, and what waited on them.

The window holds the approval asks formed in it (never an owner question, which the question protocol follows),
found by that history row whenever the intention itself was created (one deferred for days and asked an hour ago
formed an hour ago). For each ask:

- **formed**: its first ``decided_ask`` or ``asked`` history row;
- **first sight**: the earliest of a sent owner message that shows it (its ``asks`` context lists the ask; an
  ``ask_in_full`` names it), a shown-in-chat row (``ask_shown``), and, for rows before the answer route recorded
  what it showed, a sent ask notice or digest whose "Needs your OK" part lists its code while it was open. None
  when nothing recorded shows it reached the owner;
- **settled**: the first answer, lapse or cancellation after it formed (after its last revival, when it was
  revived): who (``owner``; ``operator``, a credential or the CLI; ``lapse``, no answer in time; ``mind``, the
  mind withdrew it), the answer, and from the settling row's ``answer`` details (recorded by the answer route)
  the path (``reaction``, ``reply``, ``follow_up``, ``list``, ``console``) and the reader (``tables``,
  ``decision``). Earlier rows have neither: their path is ``unknown``, or ``console`` when the CLI settled them;
- **events**: the history actions the answer path records on the ask (``ask_shown``, ``ask_clarified``,
  ``ask_later``, ``ask_undone``, ``ask_grace_cancelled``, ``ask_revived``, ``ask_paused`` with ``seconds``, and
  ``allow_added`` when the owner's "always" turned it into a standing allow), and the lapses (``outcome_expired``);
- **held work**: for a task, goal or held call, the hours from forming to settling (or to now), inside the window.

Two counts are not about one ask: clean yes or no messages that bound to no ask while one was open, and invented
code lines removed from replies. They come from the answer path's own log (``events``: ``{kind, at}``), and are
``None`` ("not recorded") where no such log is given.

Times are medians and 90th percentiles by nearest rank (a choice for small counts: every value reported is one
that happened).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from colonymind.initiatives.models import StoredInitiative

from .allows import ALLOW_ADDED
from .outbox import QUESTION_TYPES
from .outcomes import owner_answer

# History actions the answer path records on an ask.
ASK_SHOWN, ASK_CLARIFIED, ASK_LATER, ASK_UNDONE = "ask_shown", "ask_clarified", "ask_later", "ask_undone"
ASK_GRACE_CANCELLED, ASK_REVIVED, ASK_PAUSED = "ask_grace_cancelled", "ask_revived", "ask_paused"
EVENT_NAMES = {ASK_SHOWN: "shows", ASK_CLARIFIED: "clarifications", ASK_LATER: "later", ASK_UNDONE: "undos",
               ASK_GRACE_CANCELLED: "grace_cancels", ASK_REVIVED: "revivals", "outcome_expired": "lapses",
               ASK_PAUSED: "paused", ALLOW_ADDED: "allows"}
# Message-level events from the answer path's log.
UNBOUND_ANSWER, INVENTED_REFERENCE = "unbound_answer", "invented_reference"

SETTLERS = ("owner", "operator", "lapse", "mind")
MIND_SETTLERS = frozenset({"mind", "body", "model"})
WORK_KINDS = frozenset({"task", "goal"})
FLOOR_EFFECT = "floor_effect"
# How a 1.11 notice and digest listed an ask: its code at the start of a line under this label.
NEEDS_OK_1_11 = "Needs your OK (reply yes or no with the code)"
SIGHT_TYPES_1_11 = ("ask_notice", "digest")
_LISTED = re.compile(r"^([A-Z2-9]{3}|[A-Z2-9]{6}):\s")
_FORMED = ("decided_ask", "asked")
_ROWS = 10000
_PAGE = 2000               # history rows read per page while finding the asks formed in the window (my choice)


def _context(row: StoredInitiative) -> Dict[str, Any]:
    return row.context if isinstance(row.context, dict) else {}


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def spread(values: Sequence[float]) -> Dict[str, Any]:
    """``{count, median_s, p90_s}`` of durations in seconds; the 90th percentile by nearest rank."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "median_s": None, "p90_s": None}
    rank = max(1, math.ceil(0.9 * len(ordered)))
    return {"count": len(ordered), "median_s": median(ordered), "p90_s": ordered[rank - 1]}


def settler(by: Any, outcome: str, details: Mapping[str, Any] | None = None) -> str:
    """Who settled an ask: ``lapse`` for an expiry; ``owner`` for the owner's own answer and ``operator`` for anyone
    else's (the CLI, an API credential, a console), read from the path the answering code recorded in ``details``
    (``outcomes.owner_answer``: ``by`` is the caller's label, which can take the owner's attribution away but never
    give it, so a console answer labelled ``owner`` is the operator's); ``mind`` for the mind's own withdrawal. A
    row with no recorded path (1.11) is read by its label."""
    if outcome == "expired":
        return "lapse"
    answered = owner_answer(by, details)
    if answered is not None:
        return "owner" if answered else "operator"
    value = str(by or "")
    if value in MIND_SETTLERS:
        return "mind"
    if value == "owner" or value.startswith(("owner-turn", "owner:")):
        return "owner"
    return "operator"


def _answer_of(action: str, outcome: str) -> str:
    if action == "queued":
        return "yes"
    return {"done": "yes", "denied": "no", "expired": "lapse"}.get(outcome, outcome or action)


def _is_ask(row: StoredInitiative) -> bool:
    """An ask for the owner's approval: it holds a code or was decided ``ask``. An owner question (a contradiction,
    ``QUESTION_TYPES``) keeps ``decision='ask'`` but lives its own lifecycle, answered in words: moved to the digest
    it is settled "expired" while it stays open there, so it is never one of these asks."""
    return bool(row.ask_code or row.decision == "ask") and row.type not in QUESTION_TYPES


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _formed_ids(store: Any, since: datetime, now: datetime) -> List[str]:
    """The intentions whose ask formed in ``[since, now]``: an ``asked`` or ``decided_ask`` row in the mind's own
    history (``InitiativeStore.MIND_ACTOR``), read newest first until a page lies wholly before the window."""
    actor = getattr(store, "MIND_ACTOR", "mind")
    found: Dict[str, None] = {}
    offset = 0
    while True:
        page = store.get_agent_history(actor, limit=_PAGE, offset=offset)
        for item in page:
            if item.action in _FORMED and since <= _utc(item.timestamp) <= now:
                found.setdefault(item.initiative_id, None)
        if len(page) < _PAGE or all(_utc(item.timestamp) < since for item in page):
            return list(found)
        offset += len(page)


def _history(store: Any, row: StoredInitiative) -> List[Any]:
    return sorted(store.get_history(row.id, limit=500), key=lambda item: (item.timestamp, item.id or 0))


def _settle(history: Sequence[Any], formed_at: datetime) -> Optional[Tuple[Any, str]]:
    """The row that settled the ask and its outcome: the first answer, lapse or withdrawal after it formed, or
    after its last revival."""
    revivals = [index for index, item in enumerate(history) if item.action == ASK_REVIVED]
    for item in history[revivals[-1] + 1 if revivals else 0:]:
        if item.timestamp < formed_at:
            continue
        details = item.details if isinstance(item.details, dict) else {}
        if item.action == "queued" and details.get("by"):
            return item, "approved"
        if item.action.startswith("outcome_"):
            return item, item.action[len("outcome_"):]
        if item.action == "dropped":
            return item, "dropped"
    return None


def _sights(messages: Iterable[StoredInitiative]) -> Tuple[Dict[str, List[datetime]], Dict[str, List[datetime]]]:
    """``(by ask id, by 1.11 code)``: when each sent owner message showed an ask."""
    by_id: Dict[str, List[datetime]] = {}
    by_code: Dict[str, List[datetime]] = {}
    for message in messages:
        if message.kind != "message" or message.status != "sent" or message.completed_at is None:
            continue
        context, at = _context(message), message.completed_at
        for ask_id in context.get("asks") or []:
            by_id.setdefault(str(ask_id), []).append(at)
        if context.get("ask_id"):
            by_id.setdefault(str(context["ask_id"]), []).append(at)
        if message.type in SIGHT_TYPES_1_11:
            for line in _listed_lines(str(context.get("text") or "")):
                match = _LISTED.match(line)
                if match is not None:
                    by_code.setdefault(match.group(1), []).append(at)
    return by_id, by_code


def _listed_lines(text: str) -> List[str]:
    lines: List[str] = []
    inside = False
    for line in text.splitlines():
        if line == f"{NEEDS_OK_1_11}:":
            inside = True
        elif inside and line.startswith("- "):
            lines.append(line[2:])
        elif inside:
            inside = False
    return lines


def _clipped_hours(start: datetime, end: datetime, since: datetime, now: datetime) -> float:
    start, end = max(start, since), min(end, now)
    return max(0.0, (end - start).total_seconds() / 3600.0)


def report(store: Any, *, since: datetime, now: datetime, events: Optional[Iterable[Mapping[str, Any]]] = None
           ) -> Dict[str, Any]:
    """The asks formed in ``[since, now]``, as counts and times (the module docstring says what each is)."""
    rows = [row for row in store.intentions(since=since, limit=_ROWS)]
    messages = [row for row in rows if row.kind == "message" and row.status == "sent"]
    by_id, by_code = _sights(messages)
    # The asks formed in the window, whenever their intention was created: an intention deferred before the window
    # and asked inside it is one of them. Rows created inside keep their fallback (no formation row: created_at).
    candidates = [row for row in rows if _is_ask(row)]
    listed = {row.id for row in rows}
    for ident in _formed_ids(store, since, now):
        if ident not in listed:
            row = store.get(ident)
            if row is not None and row.kind and _is_ask(row):
                candidates.append(row)
    asks: List[Dict[str, Any]] = []
    settled: Counter = Counter({name: 0 for name in SETTLERS})
    paths: Counter = Counter()
    readers: Counter = Counter()
    counted: Counter = Counter({name: 0 for name in EVENT_NAMES.values()})
    paused: List[float] = []
    to_sight: List[float] = []
    to_answer: List[float] = []
    held = {"total": 0.0, "lapsed": 0.0}
    operator_verdicts = 0
    for row in sorted(candidates, key=lambda item: item.created_at or since):
        history = _history(store, row)
        formed_at = next((item.timestamp for item in history if item.action in _FORMED), row.created_at)
        if formed_at is None or formed_at < since or formed_at > now:
            continue
        found = _settle(history, formed_at)
        settled_at = found[0].timestamp if found else None
        details = found[0].details if found and isinstance(found[0].details, dict) else {}
        outcome = found[1] if found else ""
        who = settler(details.get("by"), outcome, details) if found else None
        # When it reached the owner: a message that showed it, a show in chat, or a 1.11 list of its code.
        shown = list(by_id.get(row.id, []))
        shown += [item.timestamp for item in history if item.action == ASK_SHOWN]
        if row.ask_code:
            shown += [at for at in by_code.get(str(row.ask_code).upper(), [])
                      if settled_at is None or at <= settled_at]
        sights = [at for at in shown if at >= formed_at]
        first_sight = min(sights) if sights else None
        answer = details.get("answer") if isinstance(details.get("answer"), dict) else {}
        path = reader = None
        if who in {"owner", "operator"}:
            path = str(answer.get("path") or ("console" if details.get("by") == "cli" else "unknown"))
            paths[path] += 1
            if who == "owner":
                reader = str(answer.get("reader") or "unknown")
                readers[reader] += 1
        if who is not None:
            settled[who] += 1
        if first_sight is not None:
            to_sight.append((first_sight - formed_at).total_seconds())
            if who == "owner" and settled_at is not None and settled_at >= first_sight:
                to_answer.append((settled_at - first_sight).total_seconds())
        for item in history:
            name = EVENT_NAMES.get(item.action)
            if item.action == "outcome_expired" and settled_at is not None and item.timestamp > settled_at:
                name = None        # approved work that expired undispatched later: not the ask's lapse
            if name is not None:
                counted[name] += 1
                if item.action == ASK_PAUSED and isinstance(item.details, dict):
                    try:
                        paused.append(float(item.details.get("seconds")))
                    except (TypeError, ValueError):
                        pass
        if who == "operator" and row.verdict:
            operator_verdicts += 1
        if row.kind in WORK_KINDS or row.type == FLOOR_EFFECT:
            hours = _clipped_hours(formed_at, settled_at or now, since, now)
            held["total"] += hours
            if who == "lapse":
                held["lapsed"] += hours
        asks.append({"id": row.id, "code": row.ask_code, "type": row.type, "cls": row.cls, "kind": row.kind,
                     "status": row.status, "formed_at": _iso(formed_at), "first_sight_at": _iso(first_sight),
                     "settled_at": _iso(settled_at), "settled_by": who,
                     "answer": _answer_of(found[0].action, outcome) if found else None, "path": path,
                     "reader": reader})
    ended = sum(settled.values())
    days = max((now - since).total_seconds() / 86400.0, 1e-9)
    message_level: Dict[str, Optional[int]] = {"unbound_answers": None, "invented_references": None}
    if events is not None:
        kinds = Counter(str(item.get("kind")) for item in events
                        if isinstance(item.get("at"), datetime) and since <= item["at"] <= now)
        message_level = {"unbound_answers": kinds[UNBOUND_ANSWER], "invented_references": kinds[INVENTED_REFERENCE]}
    return {
        "since": _iso(since), "until": _iso(now), "hours": round((now - since).total_seconds() / 3600.0, 3),
        "formed": len(asks), "per_day": len(asks) / days,
        "by_type": dict(Counter(item["type"] for item in asks)), "by_class": dict(Counter(item["cls"] for item in asks)),
        "open": sum(1 for item in asks if item["settled_by"] is None),
        "settled": dict(settled), "owner_answer_rate": settled["owner"] / ended if ended else None,
        "paths": dict(paths), "readers": dict(readers),
        "delivered": sum(1 for item in asks if item["first_sight_at"] is not None),
        "never_delivered": sum(1 for item in asks if item["first_sight_at"] is None),
        "to_first_sight": spread(to_sight), "to_answer": spread(to_answer),
        "events": dict(counted), "paused": spread(paused),
        "operator_verdicts": operator_verdicts,
        "held_work_hours": {key: round(value, 3) for key, value in held.items()},
        **message_level,
        "asks": asks,
    }


def _span(seconds: Optional[float]) -> str:
    from colonymind.reports.text import duration
    return "n/a" if seconds is None else duration(seconds)


def _times(value: Mapping[str, Any]) -> str:
    if not value.get("count"):
        return "none yet"
    return f"median {_span(value.get('median_s'))}, p90 {_span(value.get('p90_s'))} ({value['count']})"


def _counts(values: Mapping[str, int]) -> str:
    return ", ".join(f"{name} {count}" for name, count in sorted(values.items(), key=lambda item: (-item[1], item[0])))


def render(value: Mapping[str, Any]) -> str:
    """The report in plain lines: counts and times, never an ask's words or code."""
    since = str(value.get("since") or "")[:16].replace("T", " ")
    formed = int(value.get("formed") or 0)
    head = f"Asks since {since} UTC ({float(value.get('hours') or 0):g} hours):"
    if not formed:
        return f"{head} No asks formed in the window."
    settled = value.get("settled") or {}
    rate = value.get("owner_answer_rate")
    events = value.get("events") or {}
    held = value.get("held_work_hours") or {}

    def recorded(key: str) -> str:
        return "not recorded yet" if value.get(key) is None else str(value[key])
    lines = [
        head,
        f"- {formed} ask{'s' if formed != 1 else ''} formed ({float(value.get('per_day') or 0):.1f} a day): "
        f"{_counts(value.get('by_type') or {})}; by class: {_counts(value.get('by_class') or {})}.",
        f"- Open: {value.get('open', 0)}. Settled by the owner {settled.get('owner', 0)}, an operator "
        f"{settled.get('operator', 0)}, a lapse {settled.get('lapse', 0)}, the mind {settled.get('mind', 0)}.",
        "- The owner answered " + (f"{rate:.0%} of the settled asks." if rate is not None else "none settled yet."),
        f"- Reached the owner: {value.get('delivered', 0)}; never reached the owner: {value.get('never_delivered', 0)}. "
        f"Formation to first sight: {_times(value.get('to_first_sight') or {})}.",
        f"- First sight to the owner's answer: {_times(value.get('to_answer') or {})}.",
        f"- Answer paths: {_counts(value.get('paths') or {}) or 'none'}. Readers: "
        f"{_counts(value.get('readers') or {}) or 'none'}.",
        f"- Shows {events.get('shows', 0)}, clarifications {events.get('clarifications', 0)}, later "
        f"{events.get('later', 0)}, undos {events.get('undos', 0)}, standing allows added {events.get('allows', 0)}, "
        f"grace cancels {events.get('grace_cancels', 0)}, "
        f"lapses {events.get('lapses', 0)}, revivals {events.get('revivals', 0)}; held by the pause "
        f"{events.get('paused', 0)} ({_times(value.get('paused') or {})}).",
        f"- Operator answers learned as verdicts: {value.get('operator_verdicts', 0)}.",
        f"- Work held waiting on the owner: {float(held.get('total') or 0):.1f} task-hours "
        f"({float(held.get('lapsed') or 0):.1f} of them lapsed).",
        f"- Clean yes or no messages bound to no open ask: {recorded('unbound_answers')}. Invented code lines "
        f"removed: {recorded('invented_references')}.",
    ]
    return "\n".join(lines)


__all__ = ["ASK_CLARIFIED", "ASK_GRACE_CANCELLED", "ASK_LATER", "ASK_PAUSED", "ASK_REVIVED", "ASK_SHOWN",
           "ASK_UNDONE", "EVENT_NAMES", "INVENTED_REFERENCE", "NEEDS_OK_1_11", "SETTLERS", "UNBOUND_ANSWER", "render",
           "report", "settler", "spread"]
