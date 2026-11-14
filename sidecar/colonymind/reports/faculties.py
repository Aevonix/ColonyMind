"""What each faculty of the mind did over a window, in plain sentences (``colonymind report faculties``).

Read-only on the instance's stores (``instance.Instance``): the intention table (the mind's audit log), the
workspace (concerns, the mind's own state and the owner gate's record), the source ledger (lessons, opinions,
the learning log) and the skills directory. Each faculty's section says whether it is on, what it did in the
window, its cost and latency where the stores record them, anything that looks wrong (errors, stalls, on but
silent, off yet active), the latest paired benchmark evidence when ``--bench`` names a results directory, and
the setting that switches it off. People appear by role (the owner, a contact), never by id.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import bench as bench_reading
from .hands import faculty as hands
from .instance import INITIATIVES, LEDGER, MIND, Instance, loads, parse_time
from .plasticity import faculty as plasticity
from .registry import FACULTIES, Faculty
from .senses import read as senses
from .text import Document, duration, listing, plural, sentence
from .work import faculty as work_records
from .working_memory import faculty as working_memory
from .working_memory import benchmark_evidence as working_memory_benchmarks
from colonymind.util.temporal import now_utc

ACTION_KINDS = ("task", "goal", "message")
ACTIVE = ("proposed", "asked", "approved", "dispatched", "sending")
WITH_BODY = frozenset({"approved", "dispatched", "sending"})
STOPPED = frozenset({"cancelled", "expired", "dropped"})
WELCOME, UNWANTED = frozenset({"useful", "actioned"}), frozenset({"not_useful", "dismissed", "wrong"})
OUTREACH = (("outreach_finding", "finding", "findings"), ("outreach_loop", "open loop", "open loops"),
            ("outreach_care", "message of care", "messages of care"),
            ("outreach_answer", "answer the owner asked for", "answers the owner asked for"))
CHECK_INS = frozenset({"check_in", "commitment_check_in"})
PASSED_ON = frozenset({"commitment_notice"})
WHO_IS_WHO = frozenset({"link_proposal", "cadence_confirm", "recipient_unknown"})
NO_RECORD = "It leaves no record in the instance's stores; the paired benchmark is its measure."
SPAN = re.compile(r"^\s*(\d+)\s*([mhdw])\s*$")
UNITS = {"m": ("minute", 60), "h": ("hour", 3600), "d": ("day", 86400), "w": ("week", 604800)}
STALL_AFTER = timedelta(days=1)
UNSENT_AFTER = timedelta(hours=1)


def window(since: str, *, now: datetime) -> Tuple[datetime, datetime, str]:
    """``(start, end, words)`` for ``--since``: a span (``30m``, ``24h``, ``7d``, ``2w``) back from ``now`` or a
    date or time (``2026-09-01``, ISO) up to it."""
    match = SPAN.match(str(since or ""))
    if match:
        count, unit = int(match.group(1)), match.group(2)
        if count <= 0:
            raise ValueError("--since needs a span longer than zero")
        noun, seconds = UNITS[unit]
        return now - timedelta(seconds=count * seconds), now, f"the {plural(count, noun)}"
    start = parse_time(since) if since and not str(since).strip().replace(".", "").isdigit() else None
    if start is None or start >= now:
        raise ValueError("--since takes a span such as 7d, 24h or 2w, or a past date such as 2026-09-01")
    return start, now, f"the time since {start.strftime('%Y-%m-%d %H:%M UTC')}"


def kept_for() -> timedelta:
    """How long the instance keeps the mind's finished work, its history and the owner gate's record: daily
    retention deletes them after this (``tick.RETENTION``, ``gate.RETENTION``), while open work stays. A window
    reaching further back would count the open work and miss the finished, so ``collect`` cuts it here."""
    from colonymind.mind.gate import RETENTION as GATE
    from colonymind.mind.tick import RETENTION as WORK
    return min(GATE, WORK)


def times(count: int) -> str:
    return {1: "once", 2: "twice"}.get(count, f"{count} times")


def number(value: int) -> str:
    return f"{value:,}"


def _was(count: int, singular: str, plural_form: str) -> str:
    return singular if count == 1 else plural_form


@dataclass
class Section:
    faculty: Faculty
    on: bool
    records: bool = True           # whether the faculty leaves a record the stores keep
    lines: List[str] = field(default_factory=list)
    activity: int = 0
    cost: Optional[str] = None
    latency: List[str] = field(default_factory=list)
    wrong: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    reading: Optional[str] = None
    note: Optional[str] = None     # why a faculty whose setting is on is not acting, or acts only in part
    switch_on: Optional[str] = None  # what switches it on when its own setting already is
    off_words: Optional[str] = None  # what appeared while it was off, when the default words would not fit


@dataclass
class Collected:
    home: Any
    start: datetime
    end: datetime
    span: str
    mind_on: bool
    level: str
    found: bool
    sections: List[Section]
    bench: Optional[List[Any]] = None
    evidence: List[Any] = field(default_factory=list)
    configured: bool = True
    skipped: int = 0
    unreadable: List[str] = field(default_factory=list)
    asked: Optional[str] = None    # the window asked for, when it reached past what the instance keeps
    kept: Optional[timedelta] = None


class Reading:
    """Everything the sections read, loaded once."""

    def __init__(self, instance: Instance, start: datetime, end: datetime, now: datetime) -> None:
        self.instance, self.start, self.end, self.now = instance, start, end, now
        self.owner = instance.owner_id
        self.rows = instance.mind_rows(start, end)
        self.open = instance.open_rows(ACTIVE)
        self.rated = instance.history(("rated",), start, end)
        answers = instance.history(("queued",), start - timedelta(days=30), end)
        self.queued = {item["id"]: item["at"] for item in answers}
        self.by_id = {row.id: row for row in self.rows}

    def of(self, *, kind: Iterable[str] = (), types: Iterable[str] = (), where: Optional[Callable] = None) -> List[Any]:
        kind, types = set(kind), set(types)
        return [row for row in self.rows if (not kind or row.kind in kind) and (not types or row.type in types)
                and (where is None or where(row))]

    def to_owner(self, row: Any) -> bool:
        return bool(self.owner) and row.entity_id == self.owner

    def to_contact(self, row: Any) -> bool:
        return bool(row.entity_id) and not self.to_owner(row)

    def in_window(self, value: Any) -> bool:
        at = parse_time(value)
        return at is not None and self.start <= at < self.end


def _is_action(row: Any) -> bool:
    from colonymind.mind.audit import NOTICE_TYPES
    return row.kind in ACTION_KINDS and row.decision in {"act", "ask"} and row.type not in NOTICE_TYPES


def _cost(rows: Iterable[Any]) -> Tuple[int, int]:
    calls = tokens = 0
    for row in rows:
        spent = int(row.cost_tokens or 0)
        context = row.context if isinstance(row.context, dict) else {}
        made = context.get("calls") if row.type == "consolidation" else None
        calls += int(made) if isinstance(made, int) and made >= 0 else (1 if spent > 0 else 0)
        tokens += spent
    return calls, tokens


def _cost_line(calls: int, tokens: int, suffix: str = "") -> Optional[str]:
    if not calls and not tokens:
        return None
    return f"Cost: {plural(calls, 'model call')} and {number(tokens)} tokens{suffix}."


def _median_span(pairs: Iterable[Tuple[Optional[datetime], Optional[datetime]]]) -> Optional[str]:
    spans = [(end - start).total_seconds() for start, end in pairs if start and end and end >= start]
    return duration(median(spans)) if spans else None


def _ago(reading: Reading, at: Optional[datetime]) -> str:
    return duration((reading.now - at).total_seconds()) + " ago" if at else "some time ago"


# -- the sections -----------------------------------------------------------------------------------------------


def initiative(reading: Reading, section: Section) -> None:
    actions = [row for row in reading.rows if _is_action(row)]
    counts = Counter(row.kind for row in actions)
    section.activity = len(actions)
    if actions:
        parts = [plural(counts[kind], kind) for kind in ("task", "message", "goal") if counts[kind]]
        section.lines.append(f"It formed {plural(len(actions), 'intention')} of its own: {listing(parts)}.")
        alone = sum(row.decision == "act" for row in actions)
        section.lines.append(f"It went ahead on its own with {alone} of them and asked first about "
                             f"{len(actions) - alone}.")
    elif section.on:
        section.wrong.append("It is on but formed nothing in this window.")
    asks = [row for row in reading.rows if row.decision == "ask"]
    if asks:
        waiting = [row for row in asks if row.status == "asked"]
        declined = [row for row in asks if row.outcome == "denied" or row.cancelled_reason == "denied"]
        expired = [row for row in asks if row not in declined and (row.outcome == "expired" or row.status == "expired")]
        withdrawn = [row for row in asks if row not in declined and row not in expired and row not in waiting
                     and row.status in STOPPED]
        granted = [row for row in asks if row not in waiting + declined + expired + withdrawn]
        parts = [f"{len(granted)} granted", f"{len(declined)} declined", f"{len(expired)} expired unanswered",
                 f"{len(withdrawn)} withdrawn when no longer needed", f"{len(waiting)} still waiting"]
        parts = [part for part in parts if not part.startswith("0 ")]
        section.lines.append(f"It asked the owner's approval {times(len(asks))}: {listing(parts)}.")
        answered = [(row.created_at, reading.queued.get(row.id)) for row in granted] + \
                   [(row.created_at, row.cancelled_at) for row in declined]
        answer = _median_span(answered)
        if answer:
            section.latency.append(f"The owner answered an ask after a median of {answer}.")
        late = [row for row in reading.open
                if row.status == "asked" and row.expires_at and row.expires_at < reading.now]
        if late:
            section.wrong.append(f"{plural(len(late), 'ask')} {_was(len(late), 'is', 'are')} past its expiry but "
                                 "still open: the mind may not be ticking.")
    tasks = [row for row in actions if row.kind == "task"]
    if tasks:
        # One bucket each, in this order: a blocked task keeps its open status while the body may still settle it.
        finished = [row for row in tasks if row.status == "done" or row.outcome == "done"]
        failed = [row for row in tasks if (row.status == "failed" or row.outcome == "failed") and row not in finished]
        stopped = [row for row in tasks if row.status in STOPPED and row not in finished + failed]
        settled = finished + failed + stopped
        blocked = [row for row in tasks if row.outcome == "blocked" and row not in settled]
        body = [row for row in tasks if row.status in WITH_BODY and row not in settled + blocked]
        waits = [row for row in tasks if row.status in {"asked", "proposed"} and row not in settled + blocked]
        parts = [f"{len(finished)} finished", f"{len(failed)} failed", f"{len(blocked)} blocked",
                 f"{len(stopped)} {_was(len(stopped), 'was', 'were')} stopped before "
                 f"{_was(len(stopped), 'it', 'they')} ran",
                 f"{len(body)} {_was(len(body), 'is', 'are')} with the body with no outcome yet",
                 f"{len(waits)} {_was(len(waits), 'waits', 'wait')} for the owner"]
        section.lines.append(f"Of its {plural(len(tasks), 'task')}, "
                             f"{listing([part for part in parts if not part.startswith('0 ')])}.")
        took = _median_span((row.created_at, row.completed_at or row.failed_at) for row in finished + failed)
        if took:
            section.latency.append(f"Tasks took a median of {took} from forming to an outcome.")
        if failed:
            section.wrong.append(f"{plural(len(failed), 'task')} failed.")
    # A stall: a task with the body for over a day since it was handed over (its approval or dispatch, else its
    # forming), with no outcome. A goal is worked through its steps, never handed over; a blocked task has an
    # outcome and waits for the owner.
    with_body = [row for row in reading.open if row.kind == "task" and row.status in WITH_BODY
                 and row.outcome != "blocked"]
    handed = reading.instance.handed_over([row.id for row in with_body])
    since = {row.id: handed.get(row.id) or row.created_at for row in with_body}
    stalled = [row for row in with_body if since[row.id] and reading.now - since[row.id] > STALL_AFTER]
    if stalled:
        oldest = min(since[row.id] for row in stalled)
        if len(stalled) == 1:
            section.wrong.append(f"1 task was handed to the body {_ago(reading, oldest)} and has no outcome yet.")
        else:
            section.wrong.append(f"{len(stalled)} tasks were handed to the body more than a day ago and have no "
                                 f"outcome yet (the oldest {_ago(reading, oldest)}).")
    messages = [row for row in actions if row.kind == "message"]
    sent = [row for row in messages if row.status == "sent"]
    if messages:
        owner = [row for row in sent if reading.to_owner(row)]
        contact = [row for row in sent if reading.to_contact(row)]
        people = len({row.entity_id for row in contact})
        if contact:
            section.lines.append(f"It sent {plural(len(sent), 'message')}: {len(owner)} to the owner and "
                                 f"{len(contact)} to {plural(people, 'contact')}.")
        elif sent:
            section.lines.append(f"It sent {plural(len(sent), 'message')}, all to the owner.")
        took = _median_span((row.created_at, row.completed_at) for row in owner)
        if took:
            section.latency.append(f"Messages to the owner went out a median of {took} after forming.")
        failed = [row for row in messages if row.status in {"failed", "uncertain"}]
        if failed:
            uncertain = any(row.status == "uncertain" for row in failed)
            section.wrong.append(f"{plural(len(failed), 'message')} could not be sent"
                                 + (" or may not have arrived." if uncertain else "."))
    from colonymind.mind.audit import NOTICE_TYPES
    notices = [row for row in reading.rows if row.kind == "message" and row.type in NOTICE_TYPES
               and row.status == "sent"]
    if notices:
        section.lines.append(f"It also sent the owner {plural(len(notices), 'notice')} about its own work or state.")
    rated = [item for item in reading.rated if (item.get("details") or {}).get("by") == "owner"]
    if rated:
        verdicts = [str((item.get("details") or {}).get("verdict")) for item in rated]
        kinds = {reading.by_id[item["id"]].kind if item["id"] in reading.by_id else "message" for item in rated}
        what = "messages" if kinds == {"message"} else "intentions"
        parts = [f"{sum(v in WELCOME for v in verdicts)} welcome", f"{sum(v in UNWANTED for v in verdicts)} not wanted",
                 f"{sum(v == 'ignored' for v in verdicts)} set aside"]
        section.lines.append(f"The owner rated {len(rated)} of its {what}: "
                             f"{listing([part for part in parts if not part.startswith('0 ')])}.")
    breakers = reading.of(types=("breaker_notice",))
    if breakers:
        section.wrong.append(f"The breaker tripped {times(len(breakers))}: work of one kind failed repeatedly and was "
                             "paused.")
    health = reading.of(types=("health_notice",))
    if health:
        section.wrong.append(f"{plural(len(health), 'health notice')} went to the owner: a check of its own was "
                             "failing.")
    calls, tokens = _cost(reading.rows)
    section.cost = _cost_line(calls, tokens, " of the mind's own thinking (the body's work on its tasks is counted "
                                             "by the body, not here)")


def drives(reading: Reading, section: Section) -> None:
    # Intentions form with the drives on or off (off, priority is flat): they are context here, never its own
    # activity, and no record says which weighting ranked them.
    section.records = False
    actions = [row for row in reading.rows if _is_action(row)]
    counts = Counter(str(row.drive or "unknown") for row in actions)
    if counts:
        parts = [f"{name} ({count})" for name, count in counts.most_common()]
        section.lines.append(f"By drive, its intentions came from {listing(parts)}.")
    if not section.on:
        section.lines.append("With the drives off every drive weighs the same; the weights set in colonymind.yaml "
                             "are not used.")
        return
    weights = reading.instance.config.get("mind.drives") or {}
    if isinstance(weights, dict) and weights:
        section.lines.append("The drive weights are " + listing([f"{name} {float(value):g}"
                                                                 for name, value in weights.items()]) + ".")
        zero = [name for name, value in weights.items() if not float(value or 0)]
        if zero:
            section.wrong.append(f"The {listing(zero)} {_was(len(zero), 'drive is', 'drives are')} set to zero, so "
                                 f"{_was(len(zero), 'its', 'their')} work is off.")


def deliberation(reading: Reading, section: Section) -> None:
    formed = [row for row in reading.rows if row.kind in {"task", "goal"} and int(row.cost_tokens or 0) > 0]
    nothing = reading.of(types=("deliberation",))
    section.activity = len(formed) + len(nothing)
    if section.activity:
        section.lines.append(f"It thought about open-ended concerns with a model call {times(section.activity)}: "
                             f"{len(formed)} became work and {len(nothing)} formed nothing.")
    section.cost = _cost_line(*_cost(formed + nothing))


def goals(reading: Reading, section: Section) -> None:
    adopted = reading.of(kind=("goal",))
    steps = reading.of(types=("goal_step",))
    section.activity = len(adopted) + len(steps)
    if adopted:
        achieved = [row for row in adopted if row.outcome == "done"]
        ran_out = [row for row in adopted if row.outcome == "expired"]
        failed = [row for row in adopted if row.outcome == "failed"]
        dropped = [row for row in adopted if row.outcome == "cancelled"]
        still = [row for row in adopted if row.status in ACTIVE]
        parts = [f"{len(achieved)} {_was(len(achieved), 'was', 'were')} achieved",
                 f"{len(ran_out)} ran out of time", f"{len(failed)} failed",
                 f"{len(dropped)} {_was(len(dropped), 'was', 'were')} dropped",
                 f"{len(still)} {_was(len(still), 'is', 'are')} still open"]
        section.lines.append(f"It adopted {plural(len(adopted), 'goal')}: "
                             f"{listing([part for part in parts if not part.startswith('0 ')])}.")
        took = _median_span((row.created_at, row.completed_at) for row in achieved)
        if took:
            section.latency.append(f"An achieved goal took a median of {took}.")
    if steps:
        done = sum(row.outcome == "done" for row in steps)
        section.lines.append(f"It worked {plural(len(steps), 'step')} toward its goals: {done} finished.")


def broadcast(reading: Reading, section: Section) -> None:
    from colonymind.mind.concerns import BROADCAST
    # The drives raise and settle concerns whatever this switch says: they are context here, never the
    # broadcast's own activity, and it leaves no record of the turns it reached.
    section.records = False
    rows = reading.instance.query(MIND, "SELECT status, salience, created_at, last_touched, drive, kind, "
                                         "thoughts_spent, max_thoughts FROM concerns")
    if reading.instance.db(MIND) is None:
        section.wrong.append("No record of its concerns could be read.")
        return
    # As the mind picks it (Concerns.top, Mind.section): the most salient open concerns with thought left, and of
    # those the Mind section shows all but idle interests.
    open_now = sorted((row for row in rows if row["status"] == "open"),
                      key=lambda row: (-float(row["salience"] or 0), -float(row["last_touched"] or 0)))
    top = [row for row in open_now[:max(BROADCAST * 4, 12)]
           if int(row["thoughts_spent"] or 0) < int(row["max_thoughts"] or 0)][:BROADCAST]
    shown = [row for row in top if not (row["drive"] == "curiosity" and row["kind"] == "interest")]
    text = f"{plural(len(open_now), 'concern')} {_was(len(open_now), 'is', 'are')} open now. "
    if not section.on:
        text += "With the broadcast off, none is carried into the owner's turns."
    elif not top:
        text += "None has thought left, so none is carried into the owner's turns."
    elif len(top) == 1:
        text += ("The top one with thought left widens recall on the owner's turns; it is "
                 + ("shown in the Mind section." if shown else "not shown in the Mind section (idle interests stay "
                                                               "out of it)."))
    else:
        shown_words = ("none of them is" if not shown else "all of them are" if len(shown) == len(top)
                       else f"{len(shown)} of them {_was(len(shown), 'is', 'are')}")
        text += (f"The top {len(top)} with thought left widen recall on the owner's turns; {shown_words} shown in "
                 "the Mind section" + (" (idle interests stay out of it)." if len(shown) < len(top) else "."))
    section.lines.append(text)
    raised = sum(reading.in_window(row["created_at"]) for row in rows)
    resolved = sum(row["status"] == "resolved" and reading.in_window(row["last_touched"]) for row in rows)
    from colonymind.mind.concerns import RESOLVED_RETENTION
    kept = reading.now - RESOLVED_RETENTION
    section.lines.append(f"In this window the drives raised {plural(raised, 'concern')} and {resolved} "
                         f"{_was(resolved, 'was', 'were')} resolved" + (
                             f"; resolved concerns are deleted after {plural(RESOLVED_RETENTION.days, 'day')}, so "
                             f"both counts leave out those settled before {kept.strftime('%Y-%m-%d')}."
                             if reading.start < kept else "."))
    section.lines.append("It leaves no record of each turn it reached; the paired benchmark is its measure.")


def people(reading: Reading, section: Section) -> None:
    messages = reading.of(kind=("message",), where=reading.to_contact)
    sent = [row for row in messages if row.status == "sent"]
    questions = reading.of(types=WHO_IS_WHO)
    section.activity = len(messages) + len(questions)
    if sent:
        check_ins = sum(row.type in CHECK_INS for row in sent)
        passed = sum(row.type in PASSED_ON for row in sent)
        other = len(sent) - check_ins - passed
        parts = [plural(check_ins, "check-in"),
                 f"{passed} {_was(passed, 'message', 'messages')} the owner asked it to pass on",
                 f"{other} other {_was(other, 'message', 'messages')}"]
        section.lines.append(f"It sent {plural(len(sent), 'message')} to "
                             f"{plural(len({row.entity_id for row in sent}), 'contact')}: "
                             f"{listing([part for part in parts if not part.startswith('0 ')])}.")
        replied = sum(row.type in CHECK_INS and row.verdict == "actioned" for row in sent)
        silent = sum(row.type in CHECK_INS and row.verdict == "ignored" for row in sent)
        if replied or silent:
            section.lines.append(f"{plural(replied, 'contact')} replied to a check-in and {silent} did not.")
    unsent = [row for row in messages if row.status in {"expired", "cancelled"} and row.decision == "ask"]
    if unsent:
        section.lines.append(f"{plural(len(unsent), 'message')} to contacts waited for the owner's approval and "
                             f"{_was(len(unsent), 'was', 'were')} never sent.")
    if questions:
        waiting = sum(row.status == "asked" for row in questions)
        section.lines.append(f"It asked the owner {plural(len(questions), 'question')} about who is who"
                             + (f" ({waiting} still waiting)." if waiting else "."))
    failed = [row for row in messages if row.status in {"failed", "uncertain"}]
    if failed:
        section.wrong.append(f"{plural(len(failed), 'message')} to contacts could not be sent.")
    calls, tokens = _cost(messages)
    section.cost = _cost_line(calls, tokens, " to compose its messages")


def outreach(reading: Reading, section: Section) -> None:
    types = {name for name, _, _ in OUTREACH}
    rows = reading.of(kind=("message",), types=types, where=reading.to_owner)
    sent = [row for row in rows if row.status == "sent"]
    followups = reading.of(types=("outreach_followup",))
    section.activity = len(rows) + len(followups)
    if sent:
        counts = Counter(row.type for row in sent)
        parts = [f"{counts[name]} {one if counts[name] == 1 else many}" for name, one, many in OUTREACH if counts[name]]
        section.lines.append(f"It sent the owner {plural(len(sent), 'message')} of its own: {listing(parts)}.")
        welcome = sum(row.verdict in WELCOME for row in sent)
        unwanted = sum(row.verdict in UNWANTED for row in sent)
        unanswered = sum(row.verdict == "ignored" for row in sent)
        if welcome or unwanted or unanswered:
            section.lines.append(f"The owner found {welcome} welcome and {unwanted} not wanted; {unanswered} went "
                                 "unanswered.")
            if unwanted > welcome and unwanted >= 2:
                section.wrong.append("The owner turned more of these down than welcomed.")
        took = _median_span((row.created_at, row.completed_at) for row in sent)
        if took:
            section.latency.append(f"They went out a median of {took} after forming.")
    else:
        section.lines.append("It sent the owner nothing of its own in this window.")
    held = [row for row in rows if row.status in STOPPED]
    if held:
        section.lines.append(f"{plural(len(held), 'more message')} {_was(len(held), 'was', 'were')} formed but not "
                             "sent: held for the digest, over its daily budget or no longer needed.")
    if followups:
        section.lines.append(f"It followed up {plural(len(followups), 'reply', 'replies')} with research.")
    from colonymind.mind.outreach import PAUSE_KEY, pause_until
    pause = reading.instance.query(MIND, "SELECT text, level FROM mind_state WHERE key = ?", (PAUSE_KEY,))
    until = pause_until(dict(pause[0])) if pause else None     # read as the mind reads it; a lapsed pause is over
    if until is not None and until > reading.now:
        section.lines.append("The owner has paused it; it stays quiet until turned on again."
                             if until == datetime.max.replace(tzinfo=timezone.utc) else
                             f"The owner has paused it until {until.astimezone(timezone.utc):%Y-%m-%d %H:%M} UTC.")
    if not sent:
        section.activity = len(rows) + len(followups)


def owner_gate(reading: Reading, section: Section) -> None:
    rows = [row for row in reading.instance.query(MIND, "SELECT * FROM owner_gate") if reading.in_window(row["at"])]
    section.activity = len(rows)
    if rows:
        # One message may be reviewed more than once (made to wait, then let through): each is counted once, by
        # its last verdict in the window. The gate lets a message go or holds it; whether it arrived is the
        # sending path's record (initiative and owner outreach), never counted here.
        messages: Dict[str, List[Any]] = {}
        for row in sorted(rows, key=lambda row: (parse_time(row["at"]), str(row["id"]))):
            messages.setdefault(str(row["ident"] or f"row:{row['id']}"), []).append(row)
        last = [reviews[-1] for reviews in messages.values()]
        counts = Counter(row["decision"] for row in last)
        waited = sum(any(row["decision"] == "wait" for row in reviews[:-1]) and reviews[-1]["decision"] != "wait"
                     for reviews in messages.values())
        parts = [f"{counts['send']} let through", f"{counts['wait']} made to wait",
                 f"{counts['digest']} held for the digest", f"{counts['drop']} dropped"]
        reviews = f" in {plural(len(rows), 'review')}" if len(rows) != len(last) else ""
        section.lines.append(f"It reviewed {plural(len(last), 'message')} to the owner{reviews}: "
                             f"{listing([part for part in parts if not part.startswith('0 ')])}"
                             + (f"; {waited} of them waited first." if waited else "."))
        released = sum(row["decision"] == "digest" and bool(row["digested_at"]) for row in last)
        if counts["digest"]:
            section.lines.append(f"Of those held for the digest, {released} {_was(released, 'has', 'have')} gone "
                                 "out in a digest since.")
        second = sum("second_opinion" in loads(row["scores_json"], {}) for row in rows)
        if second:
            section.lines.append(f"It asked for a second opinion {times(second)}.")
        waited = _median_span((parse_time(row["at"]), parse_time(row["until"])) for row in rows
                              if row["decision"] == "wait")
        if waited:
            section.latency.append(f"It made messages wait a median of {waited}.")
        unrecorded = sum(row["stage"] == "audit" for row in rows)
        if unrecorded:
            section.wrong.append(f"{plural(unrecorded, 'verdict')} could not be recorded, so the message went out "
                                 "unreviewed.")
    elif section.on:
        unprompted = reading.of(kind=("message",), types={name for name, _, _ in OUTREACH} - {"outreach_answer"},
                                where=reading.to_owner)
        if unprompted:
            section.wrong.append(f"It is on, yet it reviewed nothing while {plural(len(unprompted), 'message')} of "
                                 "the mind's own went to the owner.")


def _has_embedder(reading: Reading) -> bool:
    """An embedding endpoint, or the local embedder (``router.embed_provider: local``), which needs none."""
    config = reading.instance.config
    return (config.get("router.embed_provider") == "local"
            or bool(str(config.get("router.embed_url") or "").strip()))


def semantic_recall(reading: Reading, section: Section) -> None:
    section.records = False
    section.lines.append(NO_RECORD)
    config = reading.instance.config
    if section.on and config.get("router.embed_provider") == "local":
        section.lines.append(f"Recall by meaning runs on the local embedder, {config.get('router.embed_model')}, "
                             "which the sidecar loads itself.")
    elif section.on and not _has_embedder(reading):
        section.wrong.append("It is on, but no embedding endpoint is set in the router settings, so recall matches "
                             "words only.")


def no_record(reading: Reading, section: Section) -> None:
    section.records = False
    section.lines.append(NO_RECORD)


NO_EMBEDDER = "It is on, but no embedding endpoint is set in the router settings, so {what}."


def _no_embedder(reading: Reading, section: Section, what: str) -> None:
    if section.on and not _has_embedder(reading):
        section.wrong.append(NO_EMBEDDER.format(what=what))


def priors_by_meaning(reading: Reading, section: Section) -> None:
    no_record(reading, section)
    _no_embedder(reading, section, "the extractor is offered only records that share a message's words")


def contradictions_by_meaning(reading: Reading, section: Section) -> None:
    """What the nights compared by meaning, from each finished night's counts: the names embedded, a failed or cut
    short embedding pass, and whether the embedding model was measured on the labelled pairs."""
    compared = names = failed = deferred = unmeasured = below = 0
    for row in _nights(reading):
        context = row.context if isinstance(row.context, dict) else {}
        counts = context.get("counts") if isinstance(context.get("counts"), dict) else {}
        embedded = int(counts.get("keys_embedded") or 0)
        compared += bool(embedded)
        names += embedded
        failed += bool(counts.get("key_embedding_failed"))
        deferred += int(counts.get("key_embedding_deferred") or 0)
        unmeasured += bool(counts.get("key_bound_unmeasured"))
        below += bool(counts.get("key_bound_below_floor"))
    section.activity = compared
    if compared:
        section.lines.append(f"It compared statements filed under different names by meaning on "
                             f"{plural(compared, 'night')} ({plural(names, 'name')}).")
        section.cost = "Cost: counted in the nightly run (see the nightly consolidation)."
    if deferred:
        section.lines.append(f"The names of {plural(deferred, 'person', 'people')} waited for a later night: the "
                             "night's time for them ran out.")
    if unmeasured:
        section.lines.append("The embedding model was never measured on the labelled pairs that set how close two "
                             "names must be, so the nights used the cautious default.")
    if failed:
        section.wrong.append(f"On {plural(failed, 'night')} the embedding model failed, so "
                             f"{_was(failed, 'that night', 'those nights')} compared statements under one name only.")
    if below:
        section.wrong.append("The embedding model's measurement kept no closeness that finds such pairs reliably, "
                             "so it compares none by meaning.")
    _no_embedder(reading, section, "the nights compare statements under the same name only")


def _nights(reading: Reading) -> List[Any]:
    return reading.of(types=("consolidation",))


def _finished(row: Any) -> bool:
    return not str(row.result or "").startswith("started; did not finish")


def consolidation(reading: Reading, section: Section) -> None:
    from colonymind.mind.questions import is_question, state_of
    nights = _nights(reading)
    questions = [row for row in reading.of(kind=("message",)) if is_question(row)]
    typed = [row for row in reading.of(types=("typed_call",))
             if (row.context or {}).get("task") == "mind_question_answer"]
    section.activity = len(nights) + len(questions)
    if nights:
        finished = [row for row in nights if _finished(row)]
        unfinished = len(nights) - len(finished)
        section.lines.append(f"It ran {plural(len(nights), 'night')}: {len(finished)} finished"
                             + (f" and {unfinished} did not finish." if unfinished else "."))
        counts: Counter = Counter()
        errors = 0
        for row in finished:
            context = row.context if isinstance(row.context, dict) else {}
            counts.update({key: int(value) for key, value in (context.get("counts") or {}).items()
                           if isinstance(value, int)})
            errors += bool(context.get("errors"))
        parts = [f"found {plural(counts['contradictions'], 'contradiction')}" if counts["contradictions"] else "",
                 f"wrote {plural(counts['digests'], 'per-contact digest')}" if counts["digests"] else "",
                 f"summarised {plural(counts['episodes'], 'episode')}" if counts["episodes"] else ""]
        if any(parts):
            section.lines.append(f"The nights {listing([part for part in parts if part])}.")
        took = _median_span((row.created_at, row.completed_at) for row in finished)
        if took:
            section.latency.append(f"A night took a median of {took}.")
        if unfinished:
            section.wrong.append(f"{plural(unfinished, 'night')} did not finish (it runs again at the next due "
                                 "tick).")
        if errors:
            section.wrong.append(f"{plural(errors, 'night')} hit an error.")
    if questions:
        states = [state_of(row) for row in questions]
        answered = sum(item.get("state") == "resolved" and item.get("answer") not in {None, "moot"} for item in states)
        moot = sum(item.get("state") == "resolved" and item.get("answer") == "moot" for item in states)
        expired = sum(item.get("state") == "expired" for item in states)
        waiting = len(states) - answered - moot - expired
        parts = [f"{answered} answered", f"{moot} no longer needed", f"{expired} expired", f"{waiting} still open"]
        section.lines.append(f"It asked the owner {plural(len(questions), 'question')} about statements that "
                             f"disagree: {listing([part for part in parts if not part.startswith('0 ')])}.")
    section.cost = _cost_line(*_cost(nights + typed))


def self_narrative(reading: Reading, section: Section) -> None:
    rewrites = [row for row in _nights(reading) if isinstance(row.context, dict)
                and "narrative" in (row.context.get("counts") or {}) and "narrative" in (row.context.get("done") or [])]
    section.activity = len(rewrites)
    if rewrites:
        section.lines.append(f"It rewrote its account of its recent work on {plural(len(rewrites), 'night')}.")
    current = reading.instance.query(MIND, "SELECT text, updated_at FROM mind_state WHERE key = 'self.recent'")
    if current and current[0]["text"]:
        lines = [line for line in str(current[0]["text"]).splitlines() if line.strip()]
        when = parse_time(current[0]["updated_at"])
        section.lines.append(f"The account holds {plural(len(lines), 'line')}"
                             + (f", last rewritten on {when.strftime('%Y-%m-%d')}." if when else "."))
    section.cost = "Cost: counted in the nightly run (see the nightly consolidation)." if rewrites else None


def opinions(reading: Reading, section: Section) -> None:
    rows = [row for row in reading.instance.query(
        LEDGER, "SELECT status, supersedes, correction_id, created_at, topic FROM self_judgment_revisions")
        if reading.in_window(row["created_at"])]
    formed = sum(row["status"] == "current" and row["supersedes"] is None for row in rows)
    revised = sum(row["status"] == "current" and row["supersedes"] is not None for row in rows)
    withdrawn = [row for row in rows if row["status"] == "withdrawn"]
    reconsider = sum(row["status"] == "reconsidering" for row in rows)
    section.activity = formed + revised + len(withdrawn) + reconsider
    if section.activity:
        by_owner = sum(row["correction_id"] is not None for row in withdrawn)
        text = f"It formed {plural(formed, 'opinion')}, revised {revised} and withdrew {len(withdrawn)}"
        text += f" (the owner asked for {by_owner} of those)." if by_owner else "."
        section.lines.append(text)
        if reconsider:
            section.lines.append(f"The owner asked it to reconsider {reconsider}.")
    # The views the running mind lists for the owner: a stored "current" view whose premise was superseded or
    # whose source is gone is no longer one (the runtime's own check, read-only).
    from colonymind.self_model.judgments import SelfJudgments
    owner = reading.instance.owner_id
    held = reading.instance.read(LEDGER, SelfJudgments.reader(owner).held) if owner else None
    if held:
        section.lines.append(f"It holds {plural(held, 'view')} now.")


def _lesson_entries(reading: Reading) -> List[Tuple[datetime, str, Dict[str, Any]]]:
    entries = []
    for row in reading.instance.query(
            LEDGER, "SELECT turn_id, messages_json, coalesce(occurred_at, ingested_at) AS at FROM turn_sources "
                    "WHERE session_id = 'mind' AND turn_id >= 'mind:lesson:' AND turn_id < 'mind:lesson;' "
                    "ORDER BY at, rowid"):
        messages = loads(row["messages_json"], [])
        metadata = messages[0].get("metadata") if messages and isinstance(messages[0], dict) else None
        at = parse_time(row["at"])
        if isinstance(metadata, dict) and at is not None:
            entries.append((at, str(metadata.get("event") or "").split(":", 1)[0], metadata))
    return entries


def lessons(reading: Reading, section: Section) -> None:
    entries = _lesson_entries(reading)
    status: Dict[str, str] = {}
    for at, event, metadata in entries:
        if event == "admitted" and isinstance(metadata.get("lesson"), dict):
            status[str(metadata["lesson"].get("id"))] = str(metadata["lesson"].get("status") or "candidate")
        elif event in {"activated", "superseded", "retired"}:
            status[str(metadata.get("lesson_id"))] = {"activated": "active"}.get(event, event)
    recent = Counter(event for at, event, _ in entries if reading.start <= at < reading.end)
    uses = reading.of(types=("lesson_use",))
    carried = [row for row in reading.rows if row.kind == "task" and loads(row.lesson_ids, [])]
    section.activity = sum(recent.values()) + len(uses) + len(carried)
    parts = [f"learned {plural(recent['admitted'], 'new lesson')}" if recent["admitted"] else "",
             f"made {recent['activated']} active" if recent["activated"] else "",
             f"replaced {recent['superseded']}" if recent["superseded"] else "",
             f"retired {recent['retired']}" if recent["retired"] else ""]
    if any(parts):
        section.lines.append(f"It {listing([part for part in parts if part])}.")
    if recent["witnessed"]:
        section.lines.append(f"{plural(recent['witnessed'], 'restatement')} counted as further witness of a lesson.")
    kept = Counter(status.values())
    if kept["active"] or kept["candidate"]:
        section.lines.append(f"It keeps {plural(kept['active'], 'active lesson')} and "
                             f"{plural(kept['candidate'], 'candidate')}.")
    if uses:
        section.lines.append(f"Lessons were served in {len(uses)} of the owner's turns.")
    if carried:
        section.lines.append(f"{plural(len(carried), 'task')} carried a lesson into its work.")
    if recent["admitted"] or recent["activated"]:
        section.cost = "Cost: counted in the nightly run (see the nightly consolidation)."


LEARNING = {("claim", "rejected"): "turned away {n} restated {statements} as nothing new",
            ("lesson", "rejected"): "turned away {n} proposed {lessons} as nothing new",
            ("claim", "witnessed"): "counted {n} {restatements} as witnesses of what was already known",
            ("lesson", "witnessed"): "counted {n} {restatements} of lessons as witnesses",
            ("claim", "pending"): "held {n} preference {changes} for the owner's word",
            ("lesson", "pending"): "held {n} lesson {changes} for the owner's word",
            ("claim", "ratified"): "had {n} confirmed by the owner",
            ("lesson", "ratified"): "had {n} lesson {changes} confirmed by the owner",
            ("claim", "declined"): "had {n} declined by the owner",
            ("lesson", "declined"): "had {n} lesson {changes} declined by the owner",
            ("claim", "cooled"): "moved {n} unused {statements} to the cold tier",
            ("lesson", "cooled"): "moved {n} unused {lessons} to the cold tier",
            ("claim", "revived"): "brought {n} back from the cold tier",
            ("lesson", "revived"): "brought {n} {lessons} back from the cold tier",
            ("claim", "reinforced"): "reinforced {n} {statements} that proved useful in recall",
            ("lesson", "reinforced"): "reinforced {n} {lessons} that proved useful"}
LEARNING_ORDER = ("rejected", "witnessed", "pending", "ratified", "declined", "cooled", "revived", "reinforced")


def learning(reading: Reading, section: Section) -> None:
    start, end = reading.start.timestamp(), reading.end.timestamp()
    rows = reading.instance.query(LEDGER, "SELECT store, action FROM learning_log WHERE at >= ? AND at < ?",
                                  (start, end))
    counts = Counter((row["store"], row["action"]) for row in rows)
    # Rejections are written whatever the switch (the novelty gate is mandatory); the rest only with it on.
    section.activity = sum(count for (store, action), count in counts.items() if action != "rejected")
    clauses = []
    for action in LEARNING_ORDER:
        for store in ("claim", "lesson"):
            count = counts.get((store, action), 0)
            template = LEARNING.get((store, action))
            if count and template:
                clauses.append(template.format(
                    n=count, statements=_was(count, "statement", "statements"),
                    lessons=_was(count, "lesson", "lessons"), restatements=_was(count, "restatement", "restatements"),
                    changes=_was(count, "change", "changes")))
    if clauses:
        section.lines.append(sentence(f"It {listing(clauses)}"))
    # The log keeps its newest entries (``learning.prune``), and past them the owner's answers about claims memory
    # still holds: when it has dropped some (only pruning removes a row, so an id is missing) and the oldest of the
    # rest is inside the window, the counts above leave out what came before it.
    first = reading.instance.query(LEDGER, "SELECT count(*) AS n, max(id) AS last, min(CASE WHEN store = 'claim' "
                                           "AND stage = 'owner_answer' THEN NULL ELSE at END) AS at FROM learning_log")
    dropped = bool(first) and (first[0]["last"] or 0) > (first[0]["n"] or 0)
    oldest = parse_time(first[0]["at"]) if dropped else None
    if oldest is not None and oldest > reading.start:
        section.lines.append(f"The learning log keeps only its newest entries; the oldest left is from "
                             f"{oldest.strftime('%Y-%m-%d')}, so these counts leave out what came before it.")
    pending = reading.instance.query(LEDGER, "SELECT count(*) AS n FROM learning_pending WHERE status = 'pending'")
    if pending and pending[0]["n"]:
        section.lines.append(f"{plural(pending[0]['n'], 'change')} {_was(pending[0]['n'], 'waits', 'wait')} for "
                             "the owner's word.")


def skills(reading: Reading, section: Section) -> None:
    from colonymind.mind.skills import MANIFEST, manifest_skills
    manifest = reading.instance.home / "skills" / MANIFEST
    owned = {}
    try:
        # The skills the mind owns, as it reads its manifest (``Skills.owned``), never the file's outer keys.
        owned = manifest_skills(loads(manifest.read_text(encoding="utf-8"), {})) if manifest.is_file() else {}
    except OSError:
        section.wrong.append("The list of skills it keeps could not be read.")
    loaded = reading.instance.query(MIND, "SELECT level, updated_at FROM mind_state WHERE key >= 'skills.loaded:' "
                                          "AND key < 'skills.loaded;'")
    loads_total = int(sum(float(row["level"] or 0) for row in loaded))
    last = max((parse_time(row["updated_at"]) for row in loaded if parse_time(row["updated_at"])), default=None)
    written = removed = 0
    for row in _nights(reading):
        counts = (row.context or {}).get("counts") or {} if isinstance(row.context, dict) else {}
        written += int(counts.get("skills_written", 0) or 0)
        removed += int(counts.get("skills_removed", 0) or 0)
    section.activity = written + removed + int(bool(last and reading.start <= last < reading.end))
    section.lines.append(f"It keeps {plural(len(owned), 'skill')}; skills were loaded {times(loads_total)} in all."
                         + (f" The last load was on {last.strftime('%Y-%m-%d')}." if last and loads_total else ""))
    section.lines.append(f"It wrote {written} and removed {removed} in this window." if written or removed
                         else "No skill was written or removed in this window.")


WISH_GROUPS = (("granted", {"granted", "pursuing", "measuring"}), ("declined", {"denied"}), ("expired", {"expired"}),
               ("withdrawn", {"withdrawn"}), ("measured as helping", {"helped"}),
               ("measured as not helping", {"not_helped"}), ("measured without evidence", {"no_evidence"}),
               ("never pursued", {"not_pursued"}), ("waiting for the owner", {"asked"}))


def _wishes(reading: Reading) -> List[Any]:
    return reading.of(types=("wish",))


def wishes(reading: Reading, section: Section) -> None:
    from colonymind.mind.wishes import state_of
    rows = _wishes(reading)
    section.activity = len(rows)
    if rows:
        states = Counter(state_of(row) for row in rows)
        parts = [f"{sum(states[name] for name in names)} {label}" for label, names in WISH_GROUPS
                 if sum(states[name] for name in names)]
        section.lines.append(f"It made {plural(len(rows), 'wish', 'wishes')}: {listing(parts)}.")


def wishes_named(reading: Reading, section: Section) -> None:
    """The wishes that came with the model's own words for the capability: shown beside the typed capability
    (the scope check found they ask for nothing more) or withheld (unchecked, unsure, or not one phrase)."""
    from colonymind.mind.wishes import WITHHELD, wish_of
    checks = [str(wish_of(row).get("words_check") or "") for row in _wishes(reading)]
    shown, withheld = checks.count("within"), sum(check in WITHHELD for check in checks)
    section.activity = shown + withheld
    if section.activity and section.on:
        parts = [f"{shown} shown to the owner beside the typed capability", f"{withheld} withheld"]
        section.lines.append(f"{plural(section.activity, 'wish', 'wishes')} came with the model's own words for the "
                             f"capability: {listing([part for part in parts if not part.startswith('0 ')])}.")
    section.off_words = f"{plural(section.activity, 'wish', 'wishes')} in the model's own words"


def affect(reading: Reading, section: Section) -> None:
    from colonymind.mind.affect import BANDS, LEVEL_KEYS, RENDER_FLOOR, WORDS
    rows = {row["key"]: row for row in reading.instance.query(
        MIND, "SELECT key, level, updated_at FROM mind_state WHERE key >= 'affect.' AND key < 'affect/'")}
    feelings = []
    for name, key in LEVEL_KEYS.items():
        level = float(rows[key]["level"] or 0) if key in rows else 0.0
        if name in WORDS and level >= RENDER_FLOOR:
            band = next(words for limit, words in BANDS if level <= limit)
            feelings.append((level, f"{band} {WORDS[name]}"))
    frustrated = sum(key.startswith("affect.frustration:") and float(row["level"] or 0) >= RENDER_FLOOR
                     for key, row in rows.items())
    recent = sum(reading.in_window(row["updated_at"]) for row in rows.values())
    section.activity = recent
    words = [text for _, text in sorted(feelings, reverse=True)]
    if words or frustrated:
        text = f"Right now it is {listing(words)}" if words else "Right now it is"
        if frustrated:
            text += f"{',' if words else ''} and frustrated about {plural(frustrated, 'topic')}"
        section.lines.append(sentence(text.replace("it is,", "it is")))
    elif section.on:
        section.lines.append("Right now it feels nothing strongly.")


def drift_check(reading: Reading, section: Section) -> None:
    rows = reading.of(types=("drift_check",))
    section.activity = len(rows)
    if rows:
        verdicts = [str((row.context or {}).get("verdict") or "") if isinstance(row.context, dict) else ""
                    for row in rows]
        moved = sum(value == "drifted" for value in verdicts)
        changed = sum(value == "substrate_changed" for value in verdicts)
        unscored = sum(value in {"unscored", ""} for value in verdicts)
        steady = len(verdicts) - moved - changed - unscored
        parts = [f"{steady} steady", f"{moved} that found something had moved",
                 f"{changed} that found the answering model had changed", f"{unscored} that could not be scored"]
        section.lines.append(f"It ran {plural(len(rows), 'identity check')}: "
                             f"{listing([part for part in parts if not part.startswith('0 ')])}.")
        latest = verdicts[-1]
        if latest == "drifted":
            section.wrong.append("The latest check found something had moved.")
        elif latest == "substrate_changed":
            section.wrong.append("The latest check found the answering model had changed.")
    section.cost = _cost_line(*_cost(rows))


READERS: Dict[str, Callable[[Reading, Section], None]] = {
    "initiative": initiative, "drives": drives, "deliberation": deliberation, "goals": goals,
    "broadcast": broadcast, "people": people, "outreach": outreach, "owner_gate": owner_gate,
    "semantic_recall": semantic_recall, "priors_by_meaning": priors_by_meaning, "blocks": no_record,
    "status": no_record, "consolidation": consolidation, "contradictions_by_meaning": contradictions_by_meaning,
    "self_narrative": self_narrative, "opinions": opinions, "lessons": lessons,
    "learning": learning, "skills": skills, "wishes": wishes, "wishes_named": wishes_named, "affect": affect,
    "affect_rules": no_record, "drift_check": drift_check, "senses": senses, "plasticity": plasticity,
    "working_memory": working_memory, "hands": hands, "work_records": work_records}
OFF_WORDS = {"initiative": "intentions", "goals": "goals or steps", "people": "messages to contacts",
             "outreach": "messages of its own to the owner", "owner_gate": "reviews",
             "consolidation": "nights or questions", "self_narrative": "rewrites of its account",
             "contradictions_by_meaning": "nights that compared names by meaning",
             "opinions": "opinion changes", "lessons": "lesson events", "learning": "applications of the rules",
             "skills": "skill changes or loads", "wishes": "wishes", "drift_check": "identity checks",
             "deliberation": "deliberations", "affect": "changes of feeling"}


# Faculties that run whether or not the mind is on (config.apply_environment exports their switches to the
# host): recall by meaning, the earlier records the claim extractor is offered by meaning, and the claim side of
# the learning rules (the lesson side needs the mind).
WITHOUT_MIND = frozenset({"semantic_recall", "priors_by_meaning", "learning", "working_memory", "work_records"})
# A faculty whose code runs only with others on (mind/tick.py, the embedding model): its setting alone does not
# make it act. The first one off is named.
SEMANTIC = "semantic_recall"
NEEDS = {"goals": (("drives", "Its setting is on, but goals need the drives on, and the drives are off."),),
         "skills": (("lessons", "Its setting is on, but skills are written from lessons, and lessons are off."),),
         "wishes_named": (("wishes", "Its setting is on, but it shapes wishes, and wishes are off."),),
         "contradictions_by_meaning": (
             ("consolidation", "Its setting is on, but it runs in the nightly consolidation, and that is off."),
             (SEMANTIC, "Its setting is on, but it compares names with the embedding model recall uses, and "
                        "semantic recall is off.")),
         "priors_by_meaning": (
             (SEMANTIC, "Its setting is on, but it searches with the embedding model recall uses, and semantic "
                        "recall is off."),)}
MIND_ON = "To switch it on, turn the mind on: set mind.enabled to true in colonymind.yaml, or run colonymind mind on."


def _switch(instance: Instance, faculty: Faculty) -> Tuple[bool, Optional[str], Optional[str]]:
    """``(acts, note, switch_on)``: whether the faculty acts as the running code has it, why not (or only in
    part) when its setting is on, and what switches it on then."""
    if faculty.key == "initiative":
        return instance.mind_enabled, None, None
    if not faculty.setting:
        return True, None, None
    setting = instance.setting(faculty.setting)
    if not setting or not faculty.setting.startswith("mind."):
        return setting, None, None
    needed = next(((name, why) for name, why in NEEDS.get(faculty.key, ())
                   if not instance.setting(f"mind.faculties.{name}")), None)
    if not instance.mind_enabled:
        if faculty.key == "learning":
            return True, ("The mind is off, so only the rules for remembered statements apply; the lesson side "
                          "waits for the mind."), None
        if faculty.key not in WITHOUT_MIND:
            return False, "Its setting is on, but the mind is switched off.", MIND_ON
    if needed is not None:
        name, why = needed
        return False, why, f"To switch it on, set mind.faculties.{name} to true in colonymind.yaml."
    if faculty.key == "affect" and instance.setting("mind.faculties.affect_rules"):
        return (False, "Its setting is on, but the feeling rules stand in for it, so no feeling is kept.",
                "To switch it on, set mind.faculties.affect_rules to false in colonymind.yaml.")
    return True, None, None


def collect(home: Any, *, since: str = "7d", now: Optional[datetime] = None,
            bench: Optional[Sequence[Any]] = None) -> Collected:
    now = (now or now_utc()).astimezone(timezone.utc)
    start, end, span = window(since, now=now)
    kept, asked = kept_for(), None
    if start < now - kept:
        # What retention deleted is gone and what it keeps (open work) stays: the window is cut to what is kept.
        start, asked, span = now - kept, span, f"the {plural(kept.days, 'day')}"
    instance = Instance(home)
    try:
        reading = Reading(instance, start, end, now)
        faculty_bench, working_memory_diagnostics = working_memory_benchmarks(bench or [])
        items = bench_reading.evidence(faculty_bench) if bench is not None else []
        sections = []
        for faculty in FACULTIES:
            on, note, switch_on = _switch(instance, faculty)
            section = Section(faculty=faculty, on=on, note=note, switch_on=switch_on)
            try:
                READERS[faculty.key](reading, section)
            except Exception as error:      # one unreadable record never takes the whole report down
                section.lines, section.activity, section.records = [], 0, False
                section.wrong.append(f"Its record could not be read here ({type(error).__name__}).")
            if not on and section.activity and section.records:
                words = section.off_words or f"{section.activity} {OFF_WORDS.get(faculty.key, 'records')}"
                section.wrong.insert(0, f"It is off, yet {words} appeared in this window (they may have formed "
                                        "before the switch changed, or the sidecar has not reloaded its settings).")
            elif on and section.records and not section.activity and faculty.key != "initiative" and \
                    not section.wrong:
                section.wrong.append("It is on but left no trace in this window.")
            if bench is not None:
                latest = bench_reading.latest(items, faculty.key)
                section.reading = bench_reading.verdict(items, faculty.key)
                section.evidence = [f"On {item.family[0].lower()}{item.family[1:]}, measured "
                                    f"{datetime.fromtimestamp(item.when, timezone.utc).strftime('%Y-%m-%d')}: "
                                    + " ".join(bench_reading.sentences(item)[1:]) for item in latest]
                if faculty.key == "working_memory":
                    section.evidence.extend(working_memory_diagnostics)
            sections.append(section)
        found = instance.db(INITIATIVES) is not None
    finally:
        instance.close()
    return Collected(home=home, start=start, end=end, span=span, mind_on=instance.mind_enabled,
                     level=instance.autonomy, found=found, sections=sections, bench=list(bench) if bench else bench,
                     evidence=items, configured=instance.config.exists, skipped=instance.skipped,
                     unreadable=list(instance.unreadable), asked=asked, kept=kept)


def _summary(collected: Collected) -> List[str]:
    problems = [section.faculty.noun for section in collected.sections if section.wrong]
    if collected.bench is not None:
        return bench_reading.summary(collected.evidence)
    lines = ["What helped: no benchmark evidence was read (add --bench with a results directory).",
             "Unproven: without benchmark evidence every faculty is unproven; by the rule, they stay on."]
    lines.append("What looked harmful: no benchmark evidence was read"
                 + (f"; in this window, problems showed in {listing(problems)} (see their sections)." if problems
                    else "; nothing in this window looked wrong."))
    return lines


def document(collected: Collected) -> Document:
    doc = Document("Faculty report")
    doc.bullets(_summary(collected))
    on = [section for section in collected.sections if section.on]
    off = [section.faculty.noun for section in collected.sections if not section.on]
    intro = [f"This covers {collected.span} to {collected.end.strftime('%Y-%m-%d %H:%M UTC')}."]
    if collected.asked and collected.kept is not None:
        intro.append(f"You asked for {collected.asked}, but the instance deletes the mind's finished work, its "
                     f"history and the owner gate's record after {plural(collected.kept.days, 'day')}, while "
                     "open work stays, so nothing older can be counted whole.")
    intro.append(f"The mind is on at the {collected.level} level." if collected.mind_on
                 else "The mind is switched off, so only what runs without it acts (each section says).")
    intro.append(f"{plural(len(on), 'faculty', 'faculties')} {_was(len(on), 'is', 'are')} on"
                 + (f" and {len(off)} {_was(len(off), 'is', 'are')} off: {listing(off)}." if off else "."))
    if not collected.configured:
        intro.append("No colonymind.yaml was found in this instance, so its switches are the shipped defaults.")
    if not collected.found:
        intro.append("No record of the mind's work was found in this instance.")
    if collected.unreadable:
        intro.append(f"{plural(len(collected.unreadable), 'store')} could not be read, so what "
                     f"{_was(len(collected.unreadable), 'it holds', 'they hold')} is missing below.")
    if collected.skipped:
        intro.append(f"{plural(collected.skipped, 'record')} of the mind's work could not be read and "
                     f"{_was(collected.skipped, 'is', 'are')} left out.")
    problems = [section.faculty.noun for section in collected.sections if section.wrong]
    if problems:
        intro.append(f"Something looks wrong in {plural(len(problems), 'section')}: {listing(problems)}.")
    doc.paragraph(" ".join(intro))
    for section in collected.sections:
        faculty = section.faculty
        doc.heading(f"{faculty.title}: {'on' if section.on else 'off'}")
        doc.paragraph(faculty.what)
        if section.note:
            doc.paragraph(section.note)
        doc.bullets(section.lines)
        if section.cost:
            doc.paragraph(section.cost)
        if section.latency:
            doc.paragraph("Latency: " + " ".join(section.latency))
        if section.wrong:
            doc.paragraph("Looks wrong: " + " ".join(section.wrong))
        if collected.bench is not None:
            if section.evidence:
                verdict = {"helped": "it helped", "harmful": "it looked harmful", "unproven": "unproven",
                           "not measured": "not measured yet"}[section.reading or "not measured"]
                doc.paragraph(f"Benchmark ({verdict}): " + " ".join(section.evidence))
            else:
                doc.paragraph("Benchmark: not measured yet.")
        if section.on:
            doc.paragraph(faculty.off_instruction)
        elif faculty.key == "initiative":
            doc.paragraph("To switch it on, set mind.enabled to true in colonymind.yaml, or run colonymind mind on.")
        else:
            doc.paragraph(section.switch_on or f"To switch it on, set {faculty.setting} to true in colonymind.yaml.")
    return doc


__all__ = ["Collected", "Section", "collect", "document", "window"]
