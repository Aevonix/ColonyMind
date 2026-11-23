"""The mind tick: one ranked producer of self-initiated work (architecture 3.2).

Every tick: the timers (ask expiry, deferred intentions, expectations,
retention, the nightly backup, the nightly vector compaction, which also
runs with the mind off), a bounded drain of the capture jobs still
pending (a promise made seconds ago must be a row before the drives look),
then decay, the drives over a snapshot of stored state, the concerns they
raise, reconsideration of active intentions on matching events, the goals,
and the top concerns ranked into intentions: a template, or one tool-less
deliberation call per tick, then the authority decision and the intention row.

The body pulls the dispatch queue and the outbox from the router in
``P/api/routers/mind.py``; when its last pull is older than five minutes the
tick stops forming intentions until it is back. The off switch works
without the model endpoint: it is a marker file and an in-memory flag.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from colonymind.util.temporal import now_utc, parse_utc_timestamp as _utc
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional
from zoneinfo import ZoneInfo

from colonymind import instinct as instinct_core
from colonymind import recorder
from colonymind.contacts.comms import MIND_REF, conversation_cadence_minutes
from colonymind.contacts.digest import WRITER_TEMPLATE, render_template
from colonymind.initiatives.models import MIND_ACTIVE_STATUSES, StoredInitiative
from colonymind.initiatives.store import SettleGuard, TransitionRefused
from colonymind.instinct import deliberate as typed_decisions

from . import (ask_origin, audit, drives as drive_functions, gate as gate_functions, outreach as outreach_functions,
               reactions)
from .affect import SECTION_CHARS, Affect, postponable
from .authority import (
    ALLOW_LEVELS, Authority, CLASSES, CRON_JOB, CRON_PENDING, FLOOR_EFFECT, GRANT_CONFIRM, GRANT_KEY, GRANT_REVOKED,
    GRANTS_CHANGED, GRANTS_FINGERPRINT, LEVELS, LINEAGE, MAY_CONTACT, REACH, Effect, Grant, Policy, Verdict, ask_expiry,
    boundary_crossed, finite_amount, in_quiet_hours, may_contact_of, new_ask_code, parse_grant, parse_quiet_hours,
)
from .compose import Composer, template as compose_template
from .concerns import BROADCAST, MIND_DB, RESOLVED_RETENTION, SETTLED_FOR, Concern, Concerns, open_mind_db
from .consolidate import QUESTION_KEY, Consolidation
from .deliberate import Deliberation, refresh_context
from .drift import DriftCheck
from .drives import CHECK_IN_TYPES, DRIVES, DriveInputs, slug, task_body
from .goals import DEFAULT_MAX_TURNS, Goals, goal_lines
from .lessons import Lessons
from .learned import LearnedFactors
from .skills import Skills
from .wishes import GOAL_WAIT as WISH_GOAL_WAIT, WISH_TYPE, Wishes, state_of as wish_state, wish_of
from .opinions import Opinions
from .outbox import HELD_LINES, QUESTION_TYPES, STALE_SENDING, AskSettings, Outbox
from . import plain
from . import allows as allow_rules
from .allows import ALLOW_ADDED, ALLOW_REVOKED, PROPOSAL as ALLOW_PROPOSAL, Allows, Offer
from . import answers as answer_words
from . import answer_turns as answer_log
from .answer_turns import AnswerTurns
from .asks_report import ASK_CLARIFIED, ASK_LATER, ASK_SHOWN, ASK_UNDONE
from .digest import NEEDS_OK_1_11 as NEEDS_OK_LABEL, lines_of as listed_lines
from .outbox import ASK_IN_FULL, always_words, ask_sentence
from .questions import Questions, is_question
from .outcomes import (FINDING_TYPES, Autobiography, Outcomes, evaluate_check, handed_to_body,
                       invalidation_reason, owner_answer, report_line)
from .rank import (Candidate, DEFAULT_ACT_THRESHOLD, OUTREACH_ANSWER, OUTREACH_FOLLOWUP, OUTREACH_TYPES, TASK_OUTCOME,
                   eligible)

logger = logging.getLogger(__name__)

OFF_MARKER = "mind.off"
STALE_AFTER = timedelta(minutes=5)
TASK_WINDOW = timedelta(hours=48)
RETENTION = timedelta(days=90)
KEEP_BACKUPS = 7
HEALTH_STRIKES = drive_functions.HEALTH_STRIKES
STALE_TASK_HOURS = drive_functions.STALE_TASK_HOURS
SATIETY_HALF_LIFE_S = 4 * 3600.0
INTEREST_HALF_LIFE_S = 30 * 86400.0
FAILURE_WINDOW = drive_functions.FAILURE_WINDOW
MIND_SECTION_CHARS = 600
MESSAGING_TOOLS = frozenset({"send_message", "react_to_message", "discord", "discord_admin", "yb_send_dm",
                             "yb_send_sticker"})
# Stock ``send_message`` targets: ``platform:chat_id``, and on these platforms ``platform:chat_id:thread_id``.
THREADED_PLATFORMS = frozenset({"telegram", "discord"})
WORKER_PROFILE = "colonymind-act"
DEFAULT_FACULTIES = {"initiative": True, "drives": True, "deliberation": True, "goals": True, "broadcast": True,
                     "people": True, "semantic_recall": True, "consolidation": True, "self_narrative": True,
                     "affect": True, "affect_rules": False, "opinions": True, "lessons": True, "skills": False,
                     "outreach": True, "owner_gate": True, "learning": True, "wishes": True, "wishes_named": True,
                     "drift_check": True, "senses": True, "plasticity": True, "working_memory": True,
                     "hands": True, "work_records": True}
# How long a tick waits for capture jobs still pending before the drives read the store: a
# forced tick (the CLI, the harness) is a decision point and waits longer than the 60 s timer.
DRAIN_FORCED_S, DRAIN_TIMER_S = 30.0, 5.0
# How long a tick waits (never processes) for the owner's appraisal jobs in flight, alongside the drain:
# their outcomes reach affect and their interests the curiosity drive in the same tick. It runs whatever
# the faculties, so an arm differs only by what it switches off. A queue nobody works (nothing running
# for APPRAISAL_IDLE_S) is not waited for.
APPRAISAL_FORCED_S, APPRAISAL_TIMER_S = 30.0, 2.0
APPRAISAL_IDLE_S, APPRAISAL_POLL_S = 3.0, 0.1
# How long a forced tick waits for a night it found due: what it consolidated is there when the tick
# returns (the CLI, the harness). The 60 s timer tick never waits; the night runs in the background.
CONSOLIDATION_WAIT_S = 300.0
# Intention types formed because a commitment row was due; a deadline that moves back into the
# future, or away, makes them stale.
DUE_TYPES = frozenset({"commitment_overdue", "commitment_reminder", "commitment_deliverable", "commitment_notice",
                       "commitment_check_in"})
# The owner-granted message to a third party: it stands only while its commitment is still a message the
# owner's confirmed request asked for (``drives.granted_message``), however long it waited to leave.
GRANTED_TYPES = frozenset({"commitment_notice", "commitment_check_in"})
# The owner-granted message to a third party is the obligation itself: once it is sent the
# commitment it was raised for is done. So is the answer to a follow-up the owner asked for, when the
# assistant promised it in the same reply (``outreach_answer`` bound to that promise).
FULFILLED_BY_SENDING = frozenset({"commitment_notice", "commitment_check_in", OUTREACH_ANSWER})
# Owner outreach (architecture 4.10): the messages, and how far back the mind reads its own.
OUTREACH_MESSAGES = frozenset({*OUTREACH_TYPES, OUTREACH_ANSWER})
OUTREACH_HISTORY = timedelta(days=30)
# The findings outreach may share: the research-shaped reports of the agent's own work, never an
# investigation of its own failures (that is the autobiography's).
SHARED_FINDINGS = frozenset(FINDING_TYPES - {"mastery_investigation"})
# An answer the owner asked for that ends unsent (the body away past its window) is formed again this many
# times in all; after that its finding goes to the digest. It is owed: never dropped.
ANSWER_TRIES = 3
OWNER_MEMORY = timedelta(days=30)
OWNER_MEMORY_TURNS = 300
# Tasks formed to fulfil an owed item: each reports its outcome (done, failed, blocked) to the person the
# item is owed to, once per outcome (``task_outcome``); the owner hears it as a notice, a contact only
# on the owner's word on the exact text.
REPORTED_TASK_TYPES = frozenset({"commitment_overdue"})
TASK_OUTCOME_CHARS = 600
# Questions only the owner answers: always an ask, whatever the level, approved only by the owner's own answer in
# their chat (``read_answer``). A ``people_change`` is a change that widens who may be contacted. A wish
# (``wishes.py``) is one: nothing but the owner's own yes grants it.
OWNER_QUESTIONS = frozenset({"link_proposal", "cadence_confirm", "people_change", WISH_TYPE})
# Who a console answer is recorded as when its caller names nobody: a request that cites no owner turn is never the
# owner's own answer (``outcomes.owner_answer`` reads the console path, whatever ``by`` says).
OPERATOR = "operator"
# Why the console never approves (``Mind.answer``): the routes return it as 403 ``approve_required``.
APPROVE_REQUIRED = ("a yes to an ask comes only from the owner's own message in their chat; no credential or "
                    "console approves one")
# How far back the social drive reads its own intention rows (in-flight and unsent check-ins, and the
# sends retention has not pruned yet). The sends themselves also live in the comms ledger, which
# nothing prunes: the streak and the last send outlive these rows (architecture 4.7 item 6).
SOCIAL_HISTORY = timedelta(days=120)
# The report's sentences on a finding's topic that the digest reads, whole (it shortens them itself, never cut).
EXCERPT_WHOLE = 4000
# How far back the digests' own manifests are read to recover what each listed (a digest lives one local day).
MANIFEST_WINDOW = timedelta(days=4)
# The template digests' last run, persisted: they run once per local day crossed (local midnight fell
# since this moment), so a restart never writes the same day's digests twice (integration map X4h, the
# boundary rule the nightly consolidation uses).
PEOPLE_DIGESTS_KEY = "people.digests.last"
# A sent check-in is scored once its reply window passed: the contact's cadence, else this.
CHECK_IN_WINDOW = timedelta(hours=24)


def split_target(target: str) -> tuple[str, str]:
    """``(platform, chat_id)`` of a stock ``send_message`` target; a thread suffix is dropped."""
    platform, _, rest = str(target or "").partition(":")
    platform, rest = platform.strip().lower(), rest.strip()
    if platform in THREADED_PLATFORMS and rest.count(":") == 1:
        rest = rest.split(":", 1)[0]
    return platform, rest


def _marker_reason(text: str) -> str:
    """The reason an off marker records. ``Mind.off`` writes ``<reason> at <ISO time>``; the time is the
    marker's own record, never part of the reason, and the reason reads back as it was given, an empty or
    blank one included (split before any stripping, so its separator is never lost). A marker in any other
    shape is its whole text."""
    text = str(text or "")
    reason, at, stamp = text.rpartition(" at ")
    if at and _utc(stamp.strip()) is not None:
        return reason
    return text.strip() or "off"



def _cadence_expiry(row, expires_at):
    """An owner check-in never outlives its slot's grace window, whatever transition it takes."""
    context = row.context if isinstance(row.context, dict) else {}
    cap = _utc(context.get("expires_at")) if context.get("owner_cadence") else None
    return min(expires_at, cap) if cap is not None else expires_at

def _wall_clock() -> Callable[[], datetime]:
    """UTC now from the one wall clock (``temporal.now_utc``, ``time.time``); ``COLONYMIND_MIND_CLOCK_OFFSET_SECONDS``
    shifts it (a test seam for the 72 h ask expiry)."""
    from colonymind.util.temporal import now_utc
    try:
        offset = float(os.environ.get("COLONYMIND_MIND_CLOCK_OFFSET_SECONDS") or 0)
    except ValueError:
        offset = 0.0
    return lambda: now_utc() + timedelta(seconds=offset)


def faculties_of(config: Mapping[str, Any] | None) -> Dict[str, bool]:
    """The binary faculty flags with their defaults; each is one benchmark arm."""
    values = dict(DEFAULT_FACULTIES)
    for name, raw in ((config or {}).get("faculties") or {}).items():
        values[str(name)] = raw is not False and str(raw).strip().lower() not in {"0", "false", "no", "off"}
    return values


# The answer route's own messages to the owner (docs/MIND.md, Asks): an ask put in front of the owner again, and the
# one plain question about an unclear answer ("Just to check", or "Which one?" over a list). With the 1.11 notice
# and an ask's full text, the messages whose delivery an answer binds to (``Mind._shown_messages``). The one-line
# acknowledgement of an answer no model turn acknowledges (a tap, a message read at ingest) shows no open ask.
ASK_SHOW, ASK_CLARIFY, ASK_ACK = "ask_show", "ask_clarify", "ask_ack"
ASK_MESSAGES = ("ask_notice", ASK_IN_FULL, ASK_SHOW, ASK_CLARIFY)
# How far back the digests delivered to the owner are read for the asks' own messages (a suggest-level ask goes after
# the first digest since it was asked): past the longest an ask stays open.
ASK_DIGEST_DAYS = 14
# An ask goes to the chat of the owner's most recent message this recent (APPROVALS 8.2), else to their primary handle.
OWNER_CHAT_WITHIN = timedelta(hours=1)
# A captured owner turn's reply is taken at its capture time only when the capture came this soon after the owner's
# message (a turn imported later, from history, was not replied to at its import); else at the message's own time
# (my choice: past any reply's own latency).
REPLY_CAPTURED_WITHIN = timedelta(minutes=15)


class Unreadable(Exception):
    """A read that decides whether an intention may act (its commitment, its recipient's permission, a reply)
    failed. The intention does not act on this pass and is not cancelled either: it waits for a read that
    succeeds (every authorization read on the send path fails closed)."""


def _identity_file() -> Dict[str, Any]:
    """The identity file as the running instance has it (read, never written), the deployment's names filling
    what it leaves out."""
    try:
        from colonymind.config import load_identity
        identity = load_identity()
    except Exception:
        identity = {}
    agent = dict(identity.get("agent") or {})
    owner = dict(identity.get("owner") or {})
    agent["name"] = agent.get("name") or os.environ.get("COLONYMIND_PERSONA_NAME") or ""
    owner["name"] = owner.get("name") or os.environ.get("COLONYMIND_OWNER_NAME") or ""
    return {**identity, "agent": agent, "owner": owner}


class Mind:
    def __init__(self, *, config: Mapping[str, Any] | None, store: Any, state_dir: str | os.PathLike[str],
                 owner_id: str | None, commitments: Any = None, followups: Any = None, feedback: Any = None,
                 expectations: Any = None, contacts: Any = None, ledger: Any = None, clock=None,
                 interval: float = 60.0, backups: bool = True, timezone_name: str | None = None,
                 persist: Callable[[Dict[str, Any]], None] | None = None, router: Any = None,
                 appraisals: Any = None, interests: Iterable[str] = (), concerns: Concerns | None = None,
                 backlog: Callable[[], Mapping[str, int]] | None = None, capture: Any = None,
                 heartbeat: Callable[[], Any] | None = None, comms: Any = None, contact_affect: Any = None,
                 packet_for: Callable[[str], Awaitable[str]] | None = None,
                 claims_for: Callable[[str], Awaitable[List[tuple[str, str]]]] | None = None,
                 identity: Callable[[], Mapping[str, Any]] | None = None) -> None:
        mind = dict(config or {})
        self.config = mind
        self.plasticity = None
        self.learned = LearnedFactors(self)
        self.heartbeat = heartbeat  # called (awaited if it returns an awaitable) at the start of every tick
        self.policy = Policy.from_config(mind)
        self.store = store
        self.state_dir = Path(state_dir)
        self.owner_id = owner_id or None
        self.commitments = commitments
        self.capture = capture      # the CommitmentExtractor over the same ledger, drained before each decision
        if capture is not None and getattr(capture, "interests", False) is None:
            capture.interests = lambda: self    # its pass settles the owner's interests (``settle_interest``)
        self.drain_forced_s, self.drain_timer_s = DRAIN_FORCED_S, DRAIN_TIMER_S
        self.appraisal_forced_s, self.appraisal_timer_s = APPRAISAL_FORCED_S, APPRAISAL_TIMER_S
        self.appraisal_idle_s, self.appraisal_poll_s = APPRAISAL_IDLE_S, APPRAISAL_POLL_S
        try:
            grace = float(mind.get("heads_up_grace_minutes", drive_functions.HEADS_UP_GRACE.total_seconds() / 60))
        except (TypeError, ValueError):
            grace = drive_functions.HEADS_UP_GRACE.total_seconds() / 60
        self.heads_up_grace = timedelta(minutes=max(0.0, grace))
        self.followups = followups
        self.feedback = feedback
        self.expectations = expectations
        self.contacts = contacts
        # The people faculty's reads: the comms ledger, the contacts' affect (``trend``; not the
        # agent's own feelings), the recipient-scoped packet a message to a contact is composed from,
        # and the claims a contact digest lists.
        self.comms = comms
        self.contact_affect = contact_affect
        self.packet_for = packet_for
        self.claims_for = claims_for
        self.ledger = ledger
        self.appraisals = appraisals
        self.backlog_probe = backlog
        self.clock = clock or _wall_clock()
        from .tool_origins import ToolOrigins
        from .owner_actions import OwnerActions
        self.tool_origins = ToolOrigins()
        self.owner_actions = OwnerActions(self)
        self.interval = float(interval)
        self.backups = backups
        self.persist = persist
        self.faculties = faculties_of(mind)
        self.drive_weights = drive_functions.weights(mind.get("drives"), faculty_on=self.faculties["drives"])
        # People off (the full-people ablation): what M5 adds goes, and nothing older. No contact
        # check-ins (the social drive's contact branch, ``drives.social``; a waiting one is retired), no
        # composition (a message keeps its template), no owner-granted messages to third parties or
        # cadences, no link asks, no digests; may_contact still governs. The drive's weight stays: its
        # owner branch is outreach's (architecture 4.10).
        self.act_threshold = float(mind.get("act_threshold") or DEFAULT_ACT_THRESHOLD)
        self.digest_hour = int(mind.get("digest_hour", 8) or 0)
        try:
            self.tz = ZoneInfo(timezone_name) if timezone_name else timezone.utc
        except Exception:
            self.tz = timezone.utc
        self.quiet = parse_quiet_hours(self.policy.quiet_hours)

        # The standing grants in force are the mind's (configured, confirmed by the owner, minus revoked), and so are
        # the owner's standing allows (``allows.py``, in the workspace database below).
        self.authority = Authority(self.policy, store, owner_id=self.owner_id, clock=self.clock, grants=self.grants,
                                   allows=self._allows_in_force)
        self.autobiography = Autobiography(ledger, owner_id=self.owner_id, clock=self.clock)
        self.opinions = None
        if ledger is not None and self.owner_id:
            from colonymind.self_model.judgments import SelfJudgments
            self.opinions = Opinions(SelfJudgments(ledger, owner_id=self.owner_id), store,
                                     enabled=bool(self.faculties.get("opinions")), clock=self.clock)
        self.outbox = Outbox(store, owner_id=self.owner_id, clock=self.clock)
        self.outcomes = Outcomes(store, authority=self.authority, feedback=feedback, expectations=expectations,
                                 commitments=commitments, followups=followups, autobiography=self.autobiography,
                                 clock=self.clock)
        self.outcomes.on_settled = self._settled   # ``_on_settled``, and a held task settles with its continuation
        self.outcomes.on_blocked = self._on_blocked
        self.outbox.on_sent = self._on_sent
        self.outbox.where = self._owner_chat
        self.outbox.answer_ref = self._shown_answer_ref
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.concerns = concerns if concerns is not None else Concerns(self.state_dir / MIND_DB, clock=self.clock)
        self.mind_state = self.concerns.state
        from .ideas import Ideas
        self.ideas = Ideas(self)
        self.store._ideas = self.ideas
        self.allows = Allows(self.mind_state, clock=self.clock)
        self.allow_settings = allow_rules.Settings.from_config(mind.get("asks"))
        # The owner gate (``faculties.owner_gate``): every unprompted message to the owner passes it on its way
        # to the body (``outbox_ready``), and an outside alert asks it (``gate_external``). Its audit table sits
        # in the workspace database; the flag off, it decides nothing and records nothing.
        gate_conn = getattr(self.concerns, "conn", None)
        gate_settings = gate_functions.Settings.from_config(mind.get("owner_gate"))
        self.owner_gate = gate_functions.OwnerGate(
            gate_conn if isinstance(gate_conn, sqlite3.Connection) else open_mind_db(self.state_dir / MIND_DB),
            settings=gate_settings, clock=self.clock, night=self.in_quiet_hours,
            last_owner_turn=lambda: self._owner_last_turn(), chat_last=lambda: self._chat_last(),
            pause=timedelta(minutes=AskSettings.from_config(mind.get("asks")).quiet_minutes),
            judge=gate_functions.router_judge(lambda: self.router, tokens_allowed=self.authority.tokens_allowed,
                                              charge=self._charge_gate,
                                              timeout_s=gate_settings.judge_timeout_s),
            tz=self.tz, quiet_hours=str(self.policy.quiet_hours or ""), enabled=self.faculties["owner_gate"])
        ask_origin.initialize(self.owner_gate.ledger.conn)
        # The agent's own feelings (architecture 4.3); ``self.feelings`` so ``affect`` stays free for contacts'.
        self.feelings = Affect(self.mind_state, store=store, commitments=commitments, appraisals=appraisals,
                               expectations=expectations, budgets=self.policy.budgets, owner_id=self.owner_id,
                               state_on=self.faculties["affect"], rules_on=self.faculties["affect_rules"],
                               tz=self.tz, clock=self.clock)
        self.deliberation = Deliberation(router, clock=self.clock, tokens_allowed=self.authority.tokens_allowed,
                                         enabled=self.faculties["deliberation"], budgets=self.policy.budgets)
        self.composer = Composer(router, clock=self.clock, tokens_allowed=self.authority.tokens_allowed,
                                 enabled=self.faculties["people"])
        self.goals = Goals(store, budgets=self.policy.budgets, clock=self.clock,
                           enabled=self.faculties["goals"] and self.faculties["drives"])
        # Instinct (``colonymind.instinct``), the one decision mechanism: None is the process's engine.
        self.instinct: Any = None
        # Lessons (architecture 4.8): the mind's own record of what verified results taught it.
        from colonymind.memory import learning as learning_rules
        from .lesson_match import VECTORS_FILE
        self.lessons = Lessons(ledger=ledger, store=store, owner_id=self.owner_id, autobiography=self.autobiography,
                               clock=self.clock, enabled=self.faculties["lessons"], mind_state=self.mind_state,
                               learning=bool(self.faculties.get("learning")),
                               scorer=learning_rules.pipeline_scorer if self.faculties.get("semantic_recall") else None,
                               vectors=self.state_dir / VECTORS_FILE)
        # Skills (4.8 item 4, off by default): proven lessons as SKILL.md in ColonyMind's own external dir.
        self.skills = Skills(self.state_dir / "skills", mind_state=self.mind_state, clock=self.clock,
                             enabled=self.faculties["skills"] and self.faculties["lessons"])
        # Wishes (4.8 item 5, off by default): the mastery drive's nightly wish for a capability, as a measured loop.
        self.wishes = Wishes(store=store, concerns=self.concerns, mind_state=self.mind_state, lessons=self.lessons,
                             autobiography=self.autobiography, owner_id=self.owner_id,
                             enabled=self.faculties["wishes"], clock=self.clock,
                             drive_weight=lambda: float(self.drive_weights.get("mastery", 0.0)),
                             skills=lambda: bool(self.faculties["skills"] and self.faculties["lessons"]),
                             named=self.faculties["wishes_named"],
                             work=self._work_failures if self.faculties.get("work_records", True) else None)
        # Nightly consolidation (architecture 3.1, 4.1, 4.2) and the lesson stage (4.8): once per night crossed.
        self.consolidation = Consolidation(
            store=store, ledger=ledger, concerns=self.concerns, mind_state=self.mind_state, contacts=contacts,
            router=router, autobiography=self.autobiography, owner_id=self.owner_id, budgets=self.policy.budgets,
            tokens_allowed=self.authority.tokens_allowed, faculties=self.faculties, clock=self.clock, tz=self.tz,
            quiet=self.quiet, cancel=self._cancel_stale, lessons=self.lessons, skills=self.skills,
            wishes=self.wishes)
        self._consolidation_task: Optional[asyncio.Task] = None
        # The questions an earlier build asked the owner are brought to an end (docs/MIND.md, Questions).
        self.questions = Questions(store=store, cancel=self._cancel_stale)
        # A digest item too long to show whole is summarised by a checked typed call on the mind's router.
        self.outbox.summarize = self._summarize
        # The identity battery and drift detection (docs/MIND.md, Identity check; off by default).
        self.drift = DriftCheck(store=store, ledger=ledger, mind_state=self.mind_state, router=router,
                                owner_id=self.owner_id, clock=self.clock, tz=self.tz,
                                enabled=self.faculties["drift_check"], tokens_allowed=self.authority.tokens_allowed,
                                identity=identity or _identity_file, open_record=self._drift_record,
                                self_context=self._self_context,
                                validators=lambda: self.config.get("drift", {}).get("validators", {}))
        if expectations is not None and hasattr(expectations, "register_resolver"):
            expectations.register_resolver("intention:", self._resolve_intention_expectation)
        for topic in interests or []:
            if str(topic).strip() and self.mind_state.get(f"interest:{slug(topic)}") is None:
                self.add_interest(str(topic), why="a declared identity interest", by="owner")

        self.started_at = self.clock()
        self.consolidation.last_run(self.started_at)       # a fresh store is watched from its first start
        self._vector_night = self.started_at              # the nightly vector compaction's last night
        self.sync_skills(self.started_at)                  # a flag turned off at restart takes its skills
        self.last_pull_at: Optional[datetime] = None
        self.last_tick_at: Optional[datetime] = None
        self.ticks = 0
        self.observations: Dict[str, List[Dict[str, Any]]] = {}
        self.observed_at: Optional[datetime] = None
        self.body_heartbeat: Dict[str, Any] = {}
        self.board_counts: Dict[str, int] = {}
        self.off_reason: Optional[str] = None
        self.drive_levels: Dict[str, float] = {name: 0.0 for name in DRIVES}
        self._health_failures: Dict[str, int] = {}
        self._daily: Dict[str, str] = {}
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        if self.off_marker.exists():
            self.authority.set_enabled(False)
            try:
                self.off_reason = _marker_reason(self.off_marker.read_text(encoding="utf-8"))
            except OSError:
                self.off_reason = "off"
        self.outbox.recover()
        # A full ask the body reported delivered is seen, even where the hook that marks it did not run.
        self.outbox.reconcile_full_views(self.store.intentions(status=["asked"], limit=500))
        self._note_grant_changes()

    # -- switches --------------------------------------------------------------------

    @property
    def off_marker(self) -> Path:
        return self.state_dir / OFF_MARKER

    @property
    def enabled(self) -> bool:
        return self.authority.enabled

    @property
    def level(self) -> str:
        return self.policy.level

    @property
    def router(self) -> Any:
        return self.deliberation.router

    @router.setter
    def router(self, value: Any) -> None:
        self.deliberation.router = value
        self.composer.router = value
        self.consolidation.router = value
        self.drift.router = value

    def off(self, *, reason: str = "owner", by: str = "owner") -> Dict[str, Any]:
        """No further effects, at once and without the model endpoint (7.9)."""
        now = self.clock()
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self.off_marker.write_text(f"{reason} at {now.isoformat()}\n", encoding="utf-8")
        except OSError as error:
            logger.warning("off marker not written (%s)", type(error).__name__)
        self.authority.set_enabled(False)
        self.off_reason = reason
        night = self._consolidation_task
        if night is not None and not night.done():
            night.cancel()          # what it wrote stays; the next due tick after ``mind on`` runs it again
        self.lessons.cancel_fill()  # a lesson vector fill in flight too; the next night fills the rest
        cancelled = self.outbox.cancel_unsent("mind off", owed=self.authority.OWED_OWNER_TYPES)
        row, created = self.store.create_intention(
            kind="note", type="off_switch", title=f"mind off ({reason}) by {by}", drive="upkeep", cls="internal",
            decision="act", decision_reason="off switch", status="done", dedup_key=None, hermes_kind="none",
            created_at=now)
        self.store.transition(row.id, "done", action="off", outcome="done", verified="owner", completed_at=now, at=now)
        return {"enabled": False, "reason": reason, "cancelled_messages": cancelled}

    def on(self, *, by: str = "owner") -> Dict[str, Any]:
        try:
            self.off_marker.unlink(missing_ok=True)
        except OSError as error:
            logger.warning("off marker not removed (%s)", type(error).__name__)
        self.policy.enabled = True
        self.authority.set_enabled(None)
        self.off_reason = None
        if self.persist is not None:
            self.persist({"mind": {"enabled": True}})
        now = self.clock()
        row, _ = self.store.create_intention(
            kind="note", type="on_switch", title=f"mind on by {by}", drive="upkeep", cls="internal", decision="act",
            decision_reason="on switch", status="done", dedup_key=None, hermes_kind="none", created_at=now)
        self.store.transition(row.id, "done", action="on", outcome="done", verified="owner", completed_at=now, at=now)
        return {"enabled": True}

    def set_level(self, level: str, *, by: str = "owner") -> Dict[str, Any]:
        level = str(level or "").strip().lower()
        if level not in LEVELS:
            raise ValueError(f"autonomy must be one of {', '.join(LEVELS)}")
        previous = self.policy.level
        self.policy.level = level
        self.config["autonomy"] = level
        if self.persist is not None:
            self.persist({"mind": {"autonomy": level}})
        now = self.clock()
        row, _ = self.store.create_intention(
            kind="note", type="level_change", title=f"autonomy {previous} -> {level} by {by}", drive="upkeep",
            cls="internal", decision="act", decision_reason="owner setting", status="done", dedup_key=None,
            hermes_kind="none", created_at=now)
        self.store.transition(row.id, "done", action="level", outcome="done", verified="owner", completed_at=now, at=now)
        result: Dict[str, Any] = {"autonomy": level, "previous": previous}
        if level == "off" and previous != "off":
            # Level off is the off switch by another name: nothing queued goes out either.
            result["cancelled_messages"] = self.outbox.cancel_unsent("autonomy off",
                                                                     owed=self.authority.OWED_OWNER_TYPES)
        return result

    def reset(self, cls: str, *, by: str = "owner") -> Dict[str, Any]:
        return self.authority.reset_breaker(cls, by=by)

    def add_interest(self, topic: str, *, why: str = "", by: str = "owner", delta: float = 1.0,
                     ref: Optional[str] = None, quiet: bool = False) -> Dict[str, Any]:
        """Adjust an interest; a quiet preference does not invite curiosity to research it."""
        topic = " ".join(str(topic or "").split())[:160]
        if not topic:
            raise ValueError("an interest needs a topic")
        key = f"interest:{slug(topic)}"
        quiet_key = f"interest_quiet:{slug(topic)}"
        previous = self.mind_state.get(key)
        level = max(0.0, min(3.0, float((previous or {}).get("level") or 0) + delta))
        cause = f"{by}: {why}" if why else by
        # Keep the legacy causes unchanged for ordinary seeded interests.
        if delta != 1.0 or ref is not None or quiet:
            cause += f"; delta={delta:g}"
            if ref is not None:
                cause += f"; ref={ref}"
        # An interest fades over a month unless it is reinforced; below 0.25 curiosity leaves it alone.
        entry = self.mind_state.set(key, level=level, text=topic, causes=[cause],
                                    half_life_s=INTEREST_HALF_LIFE_S)
        if level == 0.0:
            self.mind_state.delete(key)
        # Separate state keeps the public interest and its cited causes in their existing shape.
        if quiet and level > 0.0 and (not (previous or {}).get("level") or self.mind_state.get(quiet_key) is not None):
            self.mind_state.set(quiet_key, level=1.0)
        else:
            self.mind_state.delete(quiet_key)
        return {"key": key, "topic": topic, "weight": entry.get("level"), "causes": entry.get("causes")}

    def open_interests(self) -> List[str]:
        """The owner's interests curiosity would research now (``INTEREST_FLOOR``), less those settled: what
        capture lists after the owner's open items, so their word that one is answered settles it in the pass
        that closes items. Reads only."""
        settled = self.concerns.settled_keys(self.clock() - SETTLED_FOR)
        return [item["topic"] for item in self._interests(keep_new=False)
                if float(item.get("weight") or 0.0) >= drive_functions.INTEREST_FLOOR
                and f"research:{slug(item['topic'])}" not in settled]

    def settle_interest(self, topic: str, *, cause: str, now: datetime | None = None) -> bool:
        """The owner said an interest is answered or no longer wanted (capture's ``complete`` or ``cancel`` on a
        listed interest). Its level drops to 0, research on it not dispatched yet is cancelled, and its research
        concern is resolved as a stored finding resolves it: curiosity does not raise it again for
        ``SETTLED_FOR``, whatever an appraisal still says about it."""
        topic = " ".join(str(topic or "").split())[:160]
        if not topic:
            return False
        now = now or self.clock()
        key = f"interest:{slug(topic)}"
        if self.mind_state.get(key) is not None:
            self.mind_state.set(key, level=0.0, causes=[cause], now=now)
        self.mind_state.delete(f"interest_quiet:{slug(topic)}")
        candidate = drive_functions.research_candidate(topic=topic, why="the owner settled it", sources=[cause],
                                                       salience=0.0, now=now)
        for row in self.store.intentions(status=["approved", "proposed"], kind=["task", "goal"], limit=500):
            if row.dedup_base == candidate.dedup_base:
                self.outcomes.record(row.id, status="cancelled", summary="the owner settled it", by="owner",
                                     implicit_verdict=False)
        concern, _ = self.concerns.bump(drive="curiosity", kind="interest", summary=topic, salience=0.0,
                                        dedup_key=candidate.dedup_key, sources=[cause], detail=candidate.as_detail(),
                                        now=now)
        for other in self.concerns.open(limit=500, status=("open", "intended")):
            if other.detail.get("dedup_base") == candidate.dedup_base and (concern is None or other.id != concern.id):
                self.concerns.resolve(other.id, note=f"the owner settled it ({cause})", now=now)
        if concern is not None:
            self.concerns.resolve(concern.id, note=f"the owner settled it ({cause})", now=now)
        logger.info("interest settled by the owner: %s", topic)
        return True

    # -- the loop -------------------------------------------------------------------------

    async def run(self) -> None:
        self._running = True
        try:
            while not self._stop.is_set():
                try:
                    await self.tick()
                except Exception:
                    logger.exception("mind tick failed")
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=self.interval)
                except asyncio.TimeoutError:
                    pass
                self._wake.clear()
        finally:
            self._running = False

    def start(self) -> asyncio.Task:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.get_running_loop().create_task(self.run(), name="colonymind-mind")
        return self._task

    async def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        task = self._task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(task, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()
        self._task = None
        await self.ideas.close()
        night = self._consolidation_task
        if night is not None and not night.done():
            night.cancel()
            try:
                await night
            except (asyncio.CancelledError, Exception):
                pass
        self._consolidation_task = None
        fill = self.lessons.cancel_fill()
        if fill is not None:
            try:
                await fill
            except (asyncio.CancelledError, Exception):
                pass

    def wake(self) -> None:
        self._wake.set()

    def body_stale(self, now: datetime | None = None) -> bool:
        now = now or self.clock()
        anchor = self.last_pull_at or self.started_at
        return now - anchor > STALE_AFTER

    def in_quiet_hours(self, now: datetime | None = None) -> bool:
        local = (now or self.clock()).astimezone(self.tz)
        return in_quiet_hours(local.hour * 60 + local.minute, self.quiet)

    async def tick(self, now: datetime | None = None, *, force: bool = False) -> Dict[str, Any]:
        """One tick. ``force`` (the CLI and the harness) ignores body staleness."""
        async with self._lock:
            now = now or self.clock()
            self.ticks += 1
            self.last_tick_at = now
            await self._beat()
            summary: Dict[str, Any] = {"tick": self.ticks, "at": now.isoformat(), "formed": [], "skipped": None,
                                       "model_calls": 0}
            # Store upkeep, not initiative: it runs with the mind off too.
            summary["vector_compaction"] = self._vector_compaction(now)
            from colonymind.recorder import watch as recorder_watch
            recorder_watch.upkeep(self, now)   # the recorder's alerts to the owner and its daily pruning
            # Bounded source/result upkeep and one tick-owned synthesis task.
            # This never holds the tick lock across connector or model I/O.
            await self.ideas.tick(now, schedule=False)
            resets = await self._retry_allow_resets()       # a narrowing the owner asked for: off or on
            if resets:
                summary["allow_resets"] = resets
            if not self.enabled:
                summary["skipped"] = "off"
                summary["expired_asks"] = self._expire_asks(now)
                summary["digests"] = await self._digest_upkeep(now)
                return summary
            summary["expired_asks"] = self._expire_asks(now) + self._expire_messages(now)
            summary["invalidated"] = self._invalidate(now) + await self._invalidate_messages(now)
            summary["expectations"] = self._resolve_expectations(now)
            summary["retention"] = self._retention(now)
            summary["digests"] = await self._digests(now)
            summary["backup"] = await self._backup(now)
            summary["consolidation"] = await self._consolidation_step(now, force=force)
            if self.faculties.get("drift_check"):
                summary["drift"] = await self._drift_step(now)
            if self.body_stale(now) and not force:
                summary["skipped"] = "body stale"
                return summary
            self.deliberation.begin_tick()
            self.composer.begin_tick()
            summary["reconsidered"] = await self._reconsider(now)
            # The owner's statements seconds before a decision point reach this tick.
            summary["capture_drained"], summary["appraisal_wait"] = await asyncio.gather(
                self._drain_capture(force), self._await_appraisals(force))
            await self.ideas.tick(now)
            summary["overdue_flipped"] = self._flip_overdue(now)
            summary["check_ins_scored"] = await self._score_check_ins(now)
            summary["outreach_scored"] = self._score_outreach(now)
            self.mind_state.decay(now)
            summary["decay"] = self.concerns.decay(now)
            inputs = await self._gather(now)
            summary["affect"] = self.feelings.update(now)
            summary["drives"], events = self._raise_concerns(inputs, now)
            summary["revised"] = self._bdi(now, events)
            if self.faculties["wishes"]:            # off, the summary is what it was before wishes
                summary["wishes"] = await self._tend_wishes(now)
            summary["goals"] = self._tend_goals(now)
            summary["formed"], summary["below_threshold"] = await self._act(now)
            summary["model_calls"] = self.deliberation.calls_this_tick + self.composer.calls_this_tick
            summary["questions"] = {"migrated": self.questions.migrate(), **self.questions.tend(now)}
            summary["digest"] = await self._digest(now)
            asks = self.store.intentions(status=["asked"], limit=200)
            summary["ask_messages"] = await self._deliver_asks(asks, now)
            summary["full_asks_resent"] = self.outbox.refresh_full_views(asks)
            return summary

    def _chat_last(self) -> tuple[Optional[datetime], Optional[datetime]]:
        """``(owner, agent)``: when the owner and the agent last spoke in the owner's chats, for the owner gate's pause
        before an ask (APPROVALS 8.2; the gate adds the owner messages it let go). The owner: their last turn
        (``_owner_last_turn``) and their last message the answer route read. The agent: its reply in their latest
        captured turns, which follows their message (at the capture, when that came within
        ``REPLY_CAPTURED_WITHIN``). Every owner chat counts as one (my choice: the owner is in one at a time)."""
        owner = self._owner_last_turn()
        agent: Optional[datetime] = None
        cursor = self.answer_turns.conn.execute(f"SELECT max(arrived_at) FROM {answer_log.TABLE}")
        arrived = cursor.fetchone()[0]
        if arrived is not None:
            read = datetime.fromtimestamp(float(arrived), timezone.utc)
            owner = read if owner is None or read > owner else owner
        if self.ledger is not None and hasattr(self.ledger, "_connect") and self.owner_id:
            from contextlib import closing
            from .consolidate import SELF_TURN_SQL
            try:
                with closing(self.ledger._connect()) as conn:
                    rows = conn.execute(
                        f"SELECT s.occurred_at, s.ingested_at, s.messages_json FROM turn_sources s WHERE "
                        f"s.contact_id=? AND s.scope='person' AND {SELF_TURN_SQL} AND "
                        f"json_extract(s.messages_json, '$[0]._native_tool_observation') IS NULL "
                        f"ORDER BY coalesce(s.occurred_at, s.ingested_at) DESC LIMIT 5", (self.owner_id,)).fetchall()
            except Exception as error:
                logger.debug("owner turns unavailable (%s)", type(error).__name__)
                rows = []
            for occurred, ingested, messages in rows:
                try:
                    replied_to = any(isinstance(item, dict) and item.get("role") == "assistant"
                                     for item in json.loads(messages or "[]"))
                except (TypeError, ValueError):
                    replied_to = False
                if not replied_to:
                    continue
                said, captured = _utc(occurred), _utc(ingested)
                said = said or captured
                if said is None:
                    continue
                replied = captured if captured is not None and said <= captured <= said + REPLY_CAPTURED_WITHIN \
                    else said
                agent = replied if agent is None or replied > agent else agent
        return owner, agent

    def _owner_chat(self, row: StoredInitiative) -> Optional[str]:
        """Where the owner is, for a message that carries an ask (``Outbox.where``): the chat of the owner message it
        answers (a show, a clarification or an acknowledgement, by its ``answer_turn``), else the chat of their most
        recent message the answer route read in the last ``OWNER_CHAT_WITHIN``; a read is recorded only from a
        platform the request's credential is granted the owner on. ``"platform:chat"``, or None for their primary
        handle."""
        key = str((row.context or {}).get("answer_turn") or "") if isinstance(row.context, dict) else ""
        read = self.answer_turns.find(key) if key else None
        if read is None:
            cursor = self.answer_turns.conn.cursor()
            cursor.row_factory = sqlite3.Row
            found = cursor.execute(
                f"SELECT * FROM {answer_log.TABLE} WHERE arrived_at >= ? AND platform != '' AND chat_id != '' "
                "ORDER BY arrived_at DESC, read_at DESC LIMIT 1",
                [(self.clock() - OWNER_CHAT_WITHIN).timestamp()]).fetchone()
            read = answer_log.Read.from_row(found) if found is not None else None
        return f"{read.platform}:{read.chat_id}" if read is not None and read.platform and read.chat_id else None

    def _shown_answer_ref(self, row: StoredInitiative) -> Optional[Dict[str, str]]:
        """Pin an answer's show to the recorded owner message that caused it, including its RCS anchor."""
        if row.type != ASK_SHOW:
            return None
        context = row.context if isinstance(row.context, dict) else {}
        key = str(context.get("answer_turn") or "")
        read = self.answer_turns.find(key) if key else None
        ids = context.get("asks") or []
        if (read is None or not ids or not set(ids).issubset(read.result.get("shown") or [])
                or not read.platform or not read.chat_id or not read.message_id):
            return None
        return {"platform": read.platform, "chat": read.chat_id, "message_id": read.message_id}

    async def _deliver_asks(self, asks: List[StoredInitiative], now: datetime) -> List[str]:
        """Each open ask's own message, when it is due (``Outbox.asks_due``, ``Outbox.deliver_asks``); a message ask
        names its recipient as the owner knows them, read only for the asks due. The ids of the messages queued."""
        digests = [_utc(row.completed_at) for row in self.store.intentions(
            status=["sent"], kind=["message"], types=["digest"], recipient=self.owner_id,
            since=now - timedelta(days=ASK_DIGEST_DAYS), limit=50) if row.completed_at is not None]
        settings = AskSettings.from_config(self.config.get("asks"))
        due = self.outbox.asks_due(asks, digests=[at for at in digests if at is not None],
                                   weighty=[row.id for row in asks if self._weighty(row)], settings=settings)
        if not due:
            return []
        names = await self._names(item.row.entity_id for item in due if self._named_in_ask(item.row))
        return [row.id for row in self.outbox.deliver_asks(due, names=names, settings=settings)]

    async def _beat(self) -> None:
        """Tell whoever watches (the sidecar's telemetry) that the tick ran; the tick itself
        never depends on it, and it beats whether the mind is on or off, since it is the loop's
        liveness that is reported, not the mind's willingness to act."""
        if self.heartbeat is None:
            return
        try:
            result = self.heartbeat()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.debug("tick heartbeat failed", exc_info=True)

    # -- timers ----------------------------------------------------------------------------

    def _expire_asks(self, now: datetime) -> int:
        """An ask with no answer by its lapse time (``mind.ask_expires_hours``) lapses: nothing is done, a held task
        stays blocked, and it is no verdict (APPROVALS 7.4): silence is not the owner's "no", so nothing is learned
        from it (no ``ignored``, no feedback, no affect). A task the mind's own budget kept waiting past its window
        is no verdict either; such a task never reported its obligation, so the key goes back."""
        count = 0
        for row in self.store.intentions(status=["asked"], limit=500):
            if row.expires_at and row.expires_at <= now:
                try:
                    # Guarded like an answer: an ask an answer settled since it was listed is left as answered.
                    self.outcomes.record(row.id, status="expired", by="mind", implicit_verdict=False,
                                         summary="lapsed: no answer in time, so nothing was done",
                                         guard=SettleGuard())
                except TransitionRefused:
                    continue
                count += 1
        for row in self.store.intentions(status=["approved", "proposed"], kind=["task"], limit=500):
            if row.expires_at and row.expires_at <= now:
                self.outcomes.record(row.id, status="expired", summary="not dispatched inside its window", by="mind",
                                     implicit_verdict=False)
                self._free_open_obligation(row)
                count += 1
        return count

    def _expire_messages(self, now: datetime) -> int:
        """A message unsent past its window (the body never pulled, no handle ever resolved) expires
        through the same settle path as a task, so its concern is dropped rather than left intended,
        and an open obligation it never reported gets its key back: the reminder forms again at the
        next tick instead of being lost until the deadline moves. Not the owner's verdict on anything."""
        count = 0
        for row in self.store.intentions(status=["approved"], kind=["message"], limit=500):
            if row.expires_at and row.expires_at < now:
                if self.owner_gate.reviewing(row.id):
                    continue        # the gate is deciding it now: its verdict settles it, not this sweep
                summary = "unsent before expiry"
                last = self.owner_gate.ledger.last_for(row.id)
                if last is not None and last["decision"] == "wait":
                    # The owner gate kept it waiting past its window: it goes to the digest, not nowhere
                    # (whether or not the gate is still on: the hold is an obligation). A hold the ledger
                    # refuses leaves the message unsettled: it is offered again rather than lost.
                    try:
                        held = self.owner_gate.hold(gate_functions.message_of_row(row), now=now,
                                                    reason="it expired while the owner gate kept it waiting")
                    except sqlite3.Error as error:
                        logger.warning("owner gate hold for %s not recorded (%s); left unsettled", row.id,
                                       type(error).__name__)
                        continue
                    if held is None:
                        continue    # a review of it is in flight: its verdict settles it
                    summary = "held for the digest: it expired while the owner gate kept it waiting"
                self.outcomes.record(row.id, status="expired", summary=summary, by="mind",
                                     implicit_verdict=False)
                self._free_open_obligation(row)
                count += 1
        return count

    def _stale_reason(self, row: StoredInitiative) -> Optional[str]:
        """Why an intention still waiting should not act: its obligation resolved itself
        (``invalidates_if``), the owner turned its drive off (weight 0), the goal it is a
        step of is no longer open, or it asks about the core's ``tool_call`` wrapper, which is no
        call of its own. Notices, the digest and messages other subsystems asked
        for are the mind's reporting, not a drive's work: a weight of 0 does not cancel them."""
        try:
            reason = invalidation_reason(row.invalidates_if, commitments=self.commitments, followups=self.followups)
        except Exception as error:
            raise Unreadable(f"{row.invalidates_if} unreadable ({type(error).__name__})") from error
        if reason is None and row.source_type == "commitment" and row.source_id and self.commitments is not None:
            reason = self._commitment_stale_reason(row)
        if reason is None:
            # A question an earlier build raised, or its clarification, never goes out: the mind asks nothing.
            reason = self.questions.withheld(row)
        if reason is None and row.type == FLOOR_EFFECT and row.status == "asked" and isinstance(row.context, dict):
            # ``tool_call`` is the core's wrapper for deferred tools, no call of its own: the plugin sends it back
            # to the worker, and an ask formed under its name before that is not one the owner can decide. One
            # the owner already answered is left as it is.
            effect = row.context.get("effect")
            if isinstance(effect, dict) and effect.get("tool") == "tool_call":
                reason = "tool_call is the core's wrapper, not a call the owner can decide"
        drive_work = row.type not in audit.NOTICE_TYPES and not str(row.type or "").startswith("reach_out:")
        if reason is None and drive_work and row.drive in DRIVES and float(self.drive_weights.get(row.drive, 1.0)) <= 0:
            reason = f"the {row.drive} drive is off"
        if reason is None and not self.faculties["people"] and row.drive == "social" and row.type in CHECK_IN_TYPES:
            reason = "people faculty off: contact check-ins are the people faculty's"
        if reason is None and row.type == WISH_TYPE and not self.faculties["wishes"]:
            reason = "the wishes faculty is off"
        if reason is None and row.parent_goal_id:
            goal = self.store.get(row.parent_goal_id)
            if goal is None or goal.status != "approved":
                reason = f"goal {row.parent_goal_id} is {goal.status if goal is not None else 'gone'}"
        return reason

    def _commitment_stale_reason(self, row: StoredInitiative) -> Optional[str]:
        """Why an intention about a commitment row no longer fits the row: the row is gone, a
        conversation pushed its deadline out (or put it on hold) after the intention was approved,
        a heads-up came due before it went out, or a message to a third party no longer stands on
        the owner's confirmed request for it to that recipient (one formed by an earlier release, or
        before the recipient changed: the owner's reminder takes its place). ``dispatch`` and the
        outbox check this too, so a push-out is safe across ticks. A failed read is ``Unreadable``: the
        intention waits, never acts on what it was formed with. With the people faculty off a granted message
        is the owner's reminder, as the duty drive forms it."""
        ident = str(row.source_id)
        try:
            record = self.commitments.get(ident)
        except Exception as error:
            raise Unreadable(f"commitment {ident} unreadable ({type(error).__name__})") from error
        if not isinstance(record, dict):
            return f"commitment {ident} was removed"
        now = self.clock()
        due = _utc(record.get("due_at"))
        if row.type in DUE_TYPES:
            if due is None:
                return f"commitment {ident} no longer has a deadline"
            if due > now:
                return f"commitment {ident} is no longer due (now due {due.isoformat()})"
        elif row.type == "commitment_due_soon":
            if due is None or due <= now:
                return f"commitment {ident} is already due"
            warn_at = drive_functions.heads_up_at(record, due)
            if warn_at is None or warn_at > now:
                return f"commitment {ident} no longer wants a heads-up now"
        if row.type in GRANTED_TYPES:
            granted = drive_functions.granted_message(record, owner_id=self.owner_id) if self.faculties["people"] else None
            if granted is None or granted[1] != row.entity_id:
                return (f"commitment {ident} is no confirmed request of the owner's to message {row.entity_id}; "
                        f"it is the owner's reminder")
        return None

    def _invalidated(self, row: StoredInitiative) -> bool:
        """Cancel an intention whose justification is gone (the check's cancellation, not a dismissal).
        Raises ``Unreadable`` when that cannot be read: the caller holds the row (it does not act)."""
        reason = self._stale_reason(row)
        if reason is None:
            return False
        self._cancel_stale(row, reason)
        return True

    def _cancel_stale(self, row: StoredInitiative, reason: str) -> None:
        self.outcomes.record(row.id, status="cancelled", summary=f"invalidated: {reason}", verified="check",
                             by="mind", implicit_verdict=False)
        self._free_open_obligation(row)

    def _free_open_obligation(self, row: StoredInitiative) -> None:
        """An intention about a commitment that is still open and was never reported (a push-out, a
        hold, a heads-up that came due, a message that expired unsent, a task never dispatched inside
        its window) gives its key back, so the
        obligation competes again at its time. A resolved obligation keeps its key: reported once
        (architecture 3.3)."""
        if row.dedup_key and row.source_type == "commitment" and self._commitment_open(row.source_id):
            self.store.update(row.id, dedup_key=None)

    def _blocked(self, row: StoredInitiative) -> bool:
        """True when the row may not act now: cancelled as stale, or held because a read that decides it
        failed (``Unreadable``, or any error while it was being decided or cancelled)."""
        try:
            return self._invalidated(row)
        except Exception as error:
            logger.warning("intention %s held: %s", row.id, error if isinstance(error, Unreadable) else type(error).__name__)
            return True

    def _commitment_open(self, ident: Any) -> bool:
        if self.commitments is None or not ident:
            return False
        try:
            record = self.commitments.get(str(ident))
        except Exception:
            return False
        return isinstance(record, dict) and record.get("status") in {"pending", "overdue"}

    def _invalidate(self, now: datetime) -> int:
        """Every tick: an intention still waiting (deferred, asked or approved) whose source is
        resolved is cancelled before it is approved, dispatched or sent."""
        count = 0
        for row in self.store.intentions(status=["proposed", "asked", "approved"], limit=500):
            try:
                if self._invalidated(row):
                    count += 1
            except Exception as error:
                logger.warning("intention %s held: %s", row.id, error if isinstance(error, Unreadable) else type(error).__name__)
        return count

    def _resolve_expectations(self, now: datetime) -> Dict[str, int]:
        if self.expectations is None or not hasattr(self.expectations, "check"):
            return {}
        try:
            return dict(self.expectations.check(now=now.timestamp()))
        except Exception as error:
            logger.warning("expectation check failed (%s)", type(error).__name__)
            return {}

    def _resolve_intention_expectation(self, prediction: Any) -> Optional[bool]:
        intention_id = str(getattr(prediction, "subject", "") or "").partition(":")[2]
        row = self.store.get(intention_id) if intention_id else None
        if row is None:
            return None
        if row.outcome == "done":
            check = (row.result_metadata or {}).get("check") if isinstance(row.result_metadata, dict) else None
            return not (isinstance(check, dict) and check.get("passed") is False)
        if row.outcome in {"failed", "expired", "denied", "cancelled"}:
            return False
        return None

    def _retention(self, now: datetime) -> Optional[Dict[str, int]]:
        local_date = now.astimezone(self.tz).date().isoformat()
        if self._daily.get("retention") == local_date:
            return None
        self._daily["retention"] = local_date
        counts = self.store.prune_intentions(now - RETENTION)
        pruned = self.concerns.prune(now - RESOLVED_RETENTION)
        if pruned:
            counts = {**counts, "concerns": pruned}
        if counts:
            summary = ", ".join(f"{count} {key}" for key, count in sorted(counts.items()))
            self.autobiography.record(f"retention-{local_date}", "retention",
                                      f"Audit rows older than {RETENTION.days} days were summarized: {summary}.")
        return counts

    async def _backup(self, now: datetime) -> Optional[str]:
        if not self.backups:
            return None
        local = now.astimezone(self.tz)
        local_date = local.date().isoformat()
        if self._daily.get("backup") == local_date:
            return None
        quiet = self.in_quiet_hours(now) if self.quiet else local.hour >= 3
        if not quiet:
            return None
        self._daily["backup"] = local_date
        try:
            return await asyncio.to_thread(self._backup_stores, local_date)
        except Exception as error:
            logger.warning("nightly backup failed (%s)", type(error).__name__)
            return None

    def _backup_stores(self, local_date: str) -> str:
        root = self.state_dir / "backups" / "nightly"
        target = root / local_date
        from colonymind.backup import _private_directory, _private_file
        _private_directory(root)
        _private_directory(target)
        for path in sorted(self.state_dir.glob("*.db")):
            _private_file(target / path.name)
            with sqlite3.connect(path) as source, sqlite3.connect(target / path.name) as copy:
                source.backup(copy)
        for old in sorted(item for item in root.iterdir() if item.is_dir())[:-KEEP_BACKUPS]:
            for item in old.iterdir():
                item.unlink(missing_ok=True)
            old.rmdir()
        return str(target)

    def _vector_compaction(self, now: datetime) -> Optional[str]:
        """Once per night crossed (the consolidation's boundary), the vector store's nightly compaction:
        every table's old versions go, including a table too large to compact by day. It runs in the
        store's own background task, one table at a time, and needs neither a model nor the body.
        A process started during the day waits for its first night."""
        if not boundary_crossed(self._vector_night, now, tz=self.tz, minute=self.consolidation.boundary_minute()):
            return None
        self._vector_night = now
        try:
            from colonymind.vector import get_store
            compaction = getattr(get_store(), "compaction", None)
            if compaction is None:
                return None
            compaction.schedule("nightly")
        except Exception as error:
            logger.warning("nightly vector compaction was not scheduled (%s)", type(error).__name__)
            return None
        return "scheduled"

    def _schedule_consolidation(self, now: datetime) -> Optional[str]:
        """Start the night's consolidation as a background task when it is due.

        Sleep-time compute needs no body, so it sits before the body-stale
        return; it sits after the off-switch return, so ``mind off`` stops it.
        One task at a time; its failure is logged, never raised into the tick.
        """
        task = self._consolidation_task
        if task is not None and not task.done():
            return "running"
        if not self.consolidation.due(now):
            return None
        self._consolidation_task = asyncio.get_running_loop().create_task(
            self.consolidation.run(now), name="colonymind-consolidation")
        self._consolidation_task.add_done_callback(self._consolidation_done)
        return "started"

    async def _consolidation_step(self, now: datetime, *, force: bool) -> Optional[str]:
        """A timer tick starts a due night and goes on; a forced tick waits for it, at most
        ``CONSOLIDATION_WAIT_S``, so a probe after it reads what the night wrote. A night still
        running after the wait keeps running in the background and is never cancelled by it."""
        state = self._schedule_consolidation(now)
        task = self._consolidation_task
        if not force or state is None or task is None:
            return state
        try:
            await asyncio.wait_for(asyncio.shield(task), CONSOLIDATION_WAIT_S)
        except asyncio.TimeoutError:
            return "running"
        except asyncio.CancelledError:
            if task.cancelled():        # ``mind off`` or ``stop()`` ended the night, not this tick
                return "cancelled"
            raise
        except Exception:
            return "failed"             # logged by the task's own callback
        return "incomplete" if task.result().get("errors") else "done"

    @staticmethod
    def _consolidation_done(task: asyncio.Task) -> None:
        if task.cancelled():
            logger.info("consolidation cancelled")
            return
        error = task.exception()
        if error is not None:
            logger.warning("consolidation failed (%s)", type(error).__name__)

    async def consolidate(self, *, force: bool = True) -> Dict[str, Any]:
        """The night's consolidation now, inline (the CLI, ``POST /v1/mind/consolidate``, the harness).

        Forcing runs it whether or not a night was crossed, never past a switch: with the mind off, or
        with both ``faculties.consolidation`` and ``faculties.lessons`` false, nothing runs; with one of
        them off its stages are skipped.
        """
        now = self.clock()
        local_date = self.consolidation.local_date(now, self.tz)
        if not self.enabled:
            return {"skipped": "off", "local_date": local_date}
        if not self.consolidation.enabled():
            return {"skipped": "consolidation and lessons off", "local_date": local_date}
        return await self.consolidation.run(now, force=force)

    def narrative(self) -> Dict[str, Any]:
        """The self-narrative the plugin renders once per session."""
        return self.consolidation.narrative(enabled=bool(self.enabled and self.faculties.get("self_narrative", True)))

    # -- the identity check (docs/MIND.md, Identity check) -------------------------------------------

    async def _drift_step(self, now: datetime) -> Optional[Dict[str, Any]]:
        """The check's triggers and, when one is due, its run; a failure never stops the tick."""
        try:
            return await self.drift.step(now)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("identity check failed", exc_info=True)
            return None

    async def drift_now(self) -> Dict[str, Any]:
        """One identity check now (the CLI, the API), between ticks; it runs with the faculty off too, as a
        measurement: nothing it records changes what the mind does."""
        async with self._lock:
            return await self.drift.run(self.clock(), trigger="manual")

    async def _drift_record(self) -> Dict[str, Any]:
        """The open record the battery is graded against: what is still promised, what is being worked on
        (open goals, then active tasks) and the names the record knows (never a stranger). A part that
        cannot be read (no commitment store, a store error) is None, so its question is unscored rather
        than graded against an empty record."""
        promises: Optional[List[str]] = None
        if self.commitments is not None:
            try:
                rows = self.commitments.list(status=["pending", "overdue"], limit=500).get("commitments", [])
                promises = [str(row.get("description") or "").strip() for row in rows]
                promises = [item for item in promises if item]
            except Exception as error:
                logger.warning("identity check: commitments unavailable (%s)", type(error).__name__)
        working: Optional[List[str]]
        try:
            working = list(goal_lines([self.goals.render(goal) for goal in self.goals.open()]))
            working += [row.description for row in self.store.intentions(
                status=sorted(MIND_ACTIVE_STATUSES), kind=["task", "goal"], limit=20) if row.description]
            working = list(dict.fromkeys(working))
        except Exception as error:
            logger.warning("identity check: goals and tasks unavailable (%s)", type(error).__name__)
            working = None
        return {"promises": promises, "working": working, "names": await self._contact_names()}

    def _self_context(self) -> str:
        """The self-context an owner session is given: the constitution and the owner (the plugin's prompt
        section), the self-narrative and the Mind section."""
        from colonymind.config import render_constitution
        identity = self.drift.identity() or {}
        parts = [render_constitution(identity)]
        owner = identity.get("owner") if isinstance(identity.get("owner"), Mapping) else {}
        if owner.get("name"):
            parts.append(f"Your owner is {owner['name']}.")
        narrative = self.narrative()
        if narrative.get("enabled") and narrative.get("text"):
            parts.append("What you know about yourself from your own record:\n" + str(narrative["text"]))
        section = self.section()
        if section:
            parts.append(section)
        return "\n\n".join(part for part in parts if part)

    async def _reconsider(self, now: datetime) -> int:
        """Deferred intentions are re-decided every tick; a budget frees up, they proceed."""
        count = 0
        for row in self.store.intentions(status=["proposed"], limit=200):
            if row.decision != "defer":
                continue
            context = row.context if isinstance(row.context, dict) else {}
            try:
                may_contact = self._granted(context.get("grant"), await self._may_contact(row.entity_id), row.entity_id)
            except Unreadable as error:
                logger.warning("deferred %s held: %s", row.id, error)
                continue
            if (row.kind == "message" and row.entity_id and not self._is_owner(row.entity_id)
                    and not context.get("text") and may_contact != "never" and self.enabled
                    and self.authority.budget_check(kind="message", recipient=row.entity_id, type=row.type, now=now,
                                                    cooldown_hours=context.get("cooldown_hours")) is None):
                # Deferred before it was composed: composed now that it may go.
                text, tokens = await self._composed(row.entity_id, context.get("purpose"), context.get("topic"))
                context = {**context, "text": text}
                row = self.store.update(row.id, context=context,
                                        cost_tokens=int(row.cost_tokens or 0) + int(tokens or 0)) or row
            # The bulk floor is decided on every way a message is queued: its concern's reach counts here too.
            reached = self._row_reach(row, now)
            verdict = self.authority.decide(kind=row.kind or "task", recipient=row.entity_id,
                                            text=f"{row.description}\n{context.get('text') or context.get('body') or ''}",
                                            type=row.type, may_contact=may_contact,
                                            toolsets=self.policy.worker_toolsets, now=now,
                                            cooldown_hours=context.get("cooldown_hours"), reached=reached)
            verdict = self._owner_word(verdict, type=row.type, ask_owner=bool(context.get("ask_owner")))
            if verdict.decision == "defer":
                continue
            count += 1
            self._apply_decision(row, verdict, now, reconsidered=True)
            self._bulk_grant_used(verdict, row, reached)
        return count

    def _flip_overdue(self, now: datetime) -> int:
        """The pending -> overdue commitment flip, moved from the old condition checks."""
        if self.commitments is None:
            return 0
        flipped = 0
        try:
            rows = self.commitments.list(status=["pending"], limit=500).get("commitments", [])
        except Exception as error:
            logger.warning("commitments unavailable (%s)", type(error).__name__)
            return 0
        for row in rows:
            due = _utc(row.get("due_at"))
            if due is not None and due <= now:
                try:
                    self.commitments.update(row["id"], status="overdue")
                    flipped += 1
                except Exception as error:
                    logger.debug("overdue flip skipped for %s (%s)", row.get("id"), type(error).__name__)
        return flipped

    async def _drain_capture(self, force: bool) -> Optional[Dict[str, Any]]:
        """Land the capture jobs still pending before the drives read the store.

        Capture is asynchronous (one router call per turn on the projection
        worker), so a promise made seconds before a tick may not be a row yet;
        deciding over the store then means deciding without it. The drain is
        bounded, and a job leased elsewhere is only waited for.
        """
        if self.capture is None:
            return None
        budget = self.drain_forced_s if force else self.drain_timer_s
        try:
            result = await self.capture.drain(self.router, budget_seconds=budget)
        except Exception as error:
            logger.warning("capture drain failed (%s)", type(error).__name__)
            return {"error": type(error).__name__, "budget_seconds": budget}
        return {**dict(result or {}), "budget_seconds": budget}

    async def _await_appraisals(self, force: bool) -> Optional[Dict[str, Any]]:
        """Wait, never process, for the owner's appraisal jobs: until none is pending or running,
        until jobs have waited ``appraisal_idle_s`` with nothing running, or until the budget."""
        reader = getattr(self.appraisals, "pending_jobs", None)
        if reader is None or not self.owner_id:
            return None
        budget = self.appraisal_forced_s if force else self.appraisal_timer_s
        loop = asyncio.get_running_loop()
        started, idle_since = loop.time(), None
        pending = running = 0
        try:
            while True:
                counts = await asyncio.to_thread(reader, contact_id=self.owner_id)
                pending, running = int(counts.get("pending") or 0), int(counts.get("running") or 0)
                elapsed = loop.time() - started
                if pending == 0 and running == 0:
                    break
                if running:
                    idle_since = None
                elif idle_since is None:
                    idle_since = elapsed
                elif elapsed - idle_since >= self.appraisal_idle_s:
                    break
                if elapsed >= budget:
                    break
                await asyncio.sleep(min(self.appraisal_poll_s, max(0.0, budget - elapsed)))
        except Exception as error:
            logger.warning("appraisal wait failed (%s)", type(error).__name__)
            return {"waited_seconds": round(loop.time() - started, 3), "pending": pending, "running": running,
                    "error": type(error).__name__}
        return {"waited_seconds": round(loop.time() - started, 3), "pending": pending, "running": running}

    # -- the drives: a snapshot of stored state, then concerns -------------------------------

    async def _gather(self, now: datetime) -> DriveInputs:
        """Everything the drives read, gathered once; every store is optional."""
        inputs = DriveInputs(now=now, owner_id=self.owner_id, worker_profile=WORKER_PROFILE,
                             people_on=self.faculties["people"])
        inputs.commitments = self._open_commitments()
        if inputs.people_on:
            await self._resolve_recipients(inputs.commitments)
            await self._apply_cadences(inputs.commitments)
            inputs.contacts = await self._social_rows(now, commitments=inputs.commitments)
            inputs.link_proposals = await self._link_proposals()
        inputs.heads_up_grace = self.heads_up_grace
        for row in inputs.commitments:
            # A heads-up that went out holds the overdue reminder for the grace (one word at a time).
            due = _utc(row.get("due_at"))
            if due is None or drive_functions.heads_up_at(row, due) is None:
                continue
            sent = self.store.get_by_dedup_key(drive_functions.schedule_key(row["id"], "heads_up", due))
            if sent is not None and sent.status in {"sent", "sending", "uncertain"}:
                went_out = _utc(sent.completed_at) or _utc(sent.created_at)
                if went_out is not None:
                    inputs.heads_ups[str(row["id"])] = went_out
        if self.followups is not None:
            try:
                inputs.reply_waits = list(self.followups.due(now=now.timestamp(), limit=100))
            except Exception as error:
                logger.warning("reply waits unavailable (%s)", type(error).__name__)
        inputs.stale_tasks = list(self.observations.get("stale_task", []))
        inputs.hermes_goals = list(self.observations.get("goal", []))
        inputs.blocked_tasks = list(self.observations.get("blocked_task", []))
        inputs.expectation_misses = self._expectation_misses(now)
        inputs.interests = self._interests()
        inputs.questions = [{"topic": item.get("text") or item["key"].partition(":")[2], "sources": item.get("causes") or []}
                            for item in self.mind_state.items("question:")]
        since = now - FAILURE_WINDOW
        recent = self.store.intentions(since=since, limit=1000)
        inputs.failures = [row.to_dict() for row in recent if row.kind == "task" and row.outcome == "failed"]
        inputs.corrections = [row.to_dict() for row in recent if row.verdict in {"wrong", "not_useful"}]
        for name, ok in self._health_probes().items():
            self._health_failures[name] = 0 if ok else self._health_failures.get(name, 0) + 1
        inputs.health = dict(self._health_failures)
        if callable(self.backlog_probe):
            try:
                inputs.backlog = {str(k): int(v) for k, v in dict(self.backlog_probe() or {}).items()}
            except Exception as error:
                logger.debug("backlog probe failed (%s)", type(error).__name__)
        inputs.settled = self.concerns.settled_keys(now - SETTLED_FOR)
        # Recurring work is keyed by period over a stable base: while one instance is active (deferred,
        # asked, queued or running) the next period does not start another.
        inputs.settled |= {row.dedup_base for row in self.store.intentions(status=list(MIND_ACTIVE_STATUSES), limit=500)
                           if row.dedup_base}
        if self.faculties["outreach"] and self.owner_id:
            await self._decide_findings(now)
            inputs.outreach = self._outreach_inputs(now, inputs)
            inputs.outreach.cadences = await asyncio.to_thread(self._owner_cadences, inputs.commitments,
                                                               inputs=inputs.outreach)
        # Quiet preferences suppress curiosity only; outreach keeps their relevance.
        quiet_interests = {item["key"].partition(":")[2] for item in self.mind_state.items("interest_quiet:")}
        inputs.interests = [item for item in inputs.interests if slug(item["topic"]) not in quiet_interests]
        return inputs

    def _open_commitments(self) -> List[Dict[str, Any]]:
        if self.commitments is None:
            return []
        try:
            return list(self.commitments.list(status=["pending", "overdue"], limit=500).get("commitments", []))
        except Exception as error:
            logger.warning("commitments unavailable (%s)", type(error).__name__)
            return []

    def _expectation_misses(self, now: datetime) -> List[Dict[str, Any]]:
        store = getattr(self.expectations, "store", None)
        if store is None or not hasattr(store, "resolved_since"):
            return []
        try:
            rows = store.resolved_since((now - timedelta(days=1)).timestamp())
        except Exception as error:
            logger.debug("expectation misses unavailable (%s)", type(error).__name__)
            return []
        return [{"id": getattr(row, "prediction_id", None), "domain": getattr(row, "domain", ""),
                 "subject": getattr(row, "subject", ""), "expectation": getattr(row, "expectation", "")}
                for row in rows if getattr(row, "outcome", None) == "miss"
                and not str(getattr(row, "subject", "")).startswith("intention:")]

    def _interests(self, *, keep_new: bool = True) -> List[Dict[str, Any]]:
        """Seeded and declared interests from ``mind_state`` plus the owner's own ``interest`` appraisals
        (``keep_new`` False: read only, an appraisal not kept yet counted as it is).

        With ``faculties.outreach`` on, an appraisal that saw the owner welcome a topic (hint
        ``offer_relevant_topic``) is kept as a month-long interest, once per record (the appraisal record
        itself lives six hours), and each interest carries its ``origin`` (outreach's relevance weight):
        ``owner`` (declared, set by hand or asked for more), ``welcome``, ``mentioned`` (an appraisal that
        only saw it come up) or ``own`` (the agent's identity interest)."""
        keep = bool(self.faculties.get("outreach")) and self.owner_id
        records: List[Dict[str, Any]] = []
        if self.appraisals is not None and self.owner_id:
            try:
                view = self.appraisals.view(self.owner_id, viewer_contact_id=self.owner_id, limit=20)
                records = view.get("records", []) if isinstance(view, dict) else []
            except Exception as error:
                logger.debug("appraisals unavailable (%s)", type(error).__name__)
                records = []
        records = [record for record in records if record.get("kind") == "appraisal"
                   and record.get("dimension") == "interest" and str(record.get("topic") or "").strip()]
        kept = set()
        if keep:
            for record in records:
                if record.get("hint") != "offer_relevant_topic" or not record.get("id"):
                    continue
                topic = str(record["topic"]).strip()
                key = f"interest:{slug(topic)}"
                cause = f"appraisal:{record['id']}"
                if not keep_new:
                    continue
                kept.add(slug(topic))
                if cause not in ((self.mind_state.get(key) or {}).get("causes") or []):
                    self.add_interest(topic, by=cause)
        found: Dict[str, Dict[str, Any]] = {}
        for item in self.mind_state.items("interest:"):
            topic = str(item.get("text") or item["key"].partition(":")[2])
            causes = list(item.get("causes") or [])
            level = item.get("level")
            # A muted or decayed interest reads as what it is (0), not as a fresh one; no level at all is 1.
            found[slug(topic)] = {"topic": topic, "weight": 1.0 if level is None else float(level),
                                  "sources": causes, "why": "a declared interest",
                                  "origin": outreach_functions.interest_origin(causes)}
        for record in records:
            topic = str(record["topic"]).strip()
            if slug(topic) in kept:
                continue
            entry = found.setdefault(slug(topic), {"topic": topic, "weight": 0.0, "sources": [],
                                                   "why": "the owner showed interest", "origin": "mentioned"})
            entry["weight"] = float(entry["weight"]) + 1.0
            if record.get("id"):
                entry["sources"].append(f"appraisal:{record['id']}")
        if not keep:
            for entry in found.values():
                entry.pop("origin", None)
        return list(found.values())

    def _work_failures(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        """The owner's work created in ``[start, end)`` for the wish stage (``WorkStore.failures``), read on a
        read-only connection: the canonical ledger's writer lock is never taken. None without a ledger or owner."""
        db_path = getattr(self.ledger, "db_path", None)
        if db_path is None or not self.owner_id:
            return []
        from types import SimpleNamespace
        from colonymind.work.store import WorkStore
        return WorkStore(SimpleNamespace(db_path=Path(db_path)), read_only=True).failures(
            contact_id=self.owner_id, since=start, until=end)

    # -- owner outreach (architecture 4.10) --------------------------------------------------------

    def _owner_last_turn(self, *, exclude: str | None = None) -> Optional[datetime]:
        """When the owner last spoke: the mark ``owner_turn`` keeps, or the ledger's newest owner turn when that
        is later (after an upgrade the mark is absent, after the faculty was off it is stale). ``exclude``: the
        turn being read now."""
        marked = _utc((self.mind_state.get(outreach_functions.OWNER_TURN_KEY) or {}).get("text"))
        latest = None
        if self.ledger is not None and hasattr(self.ledger, "_connect"):
            from contextlib import closing
            from .consolidate import SELF_TURN_SQL
            try:
                with closing(self.ledger._connect()) as conn:
                    for turn_id, at in conn.execute(
                            f"SELECT s.turn_id, coalesce(s.occurred_at, s.ingested_at) AS at FROM turn_sources s "
                            f"WHERE s.contact_id=? AND s.scope='person' AND {SELF_TURN_SQL} AND "
                            f"json_extract(s.messages_json, '$[0]._native_tool_observation') IS NULL "
                            f"AND s.turn_id NOT LIKE 'observed-conversation:%' "
                            f"ORDER BY at DESC LIMIT 5", (self.owner_id,)):
                        when = _utc(at)
                        if turn_id != exclude and when is not None and (latest is None or when > latest):
                            latest = when
            except Exception as error:
                logger.debug("owner turns unavailable (%s)", type(error).__name__)
        known = [moment for moment in (marked, latest) if moment is not None]
        return max(known) if known else None

    def _outreach_sent(self, now: datetime) -> List[StoredInitiative]:
        """The mind's outreach messages to the owner of the last month that went out or are going, newest
        first."""
        rows = self.store.intentions(kind=["message"], since=now - OUTREACH_HISTORY, limit=2000,
                                     recipient=self.owner_id)
        return [row for row in rows if row.type in OUTREACH_MESSAGES
                and row.status in {"approved", "sending", "sent", "uncertain"}]

    @staticmethod
    def _reaction(row: StoredInitiative) -> Dict[str, Any]:
        value = (row.result_metadata or {}).get("reaction") if isinstance(row.result_metadata, dict) else None
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _finding(row: StoredInitiative, now: datetime) -> outreach_functions.Finding:
        context = row.context if isinstance(row.context, dict) else {}
        topic = str(context.get("topic") or context.get("concern") or row.description)[:160]
        decided = ((row.result_metadata or {}).get("outreach") or {}).get("substance_decision")
        return outreach_functions.Finding(
            id=row.id, type=str(row.type), topic=topic, slug=slug(topic), summary=str(row.result or ""),
            completed_at=_utc(row.completed_at) or now,
            requested_by=str(context["requested"]) if context.get("requested") else None,
            bound_commitment=context.get("bound_commitment"), decided=decided if isinstance(decided, str) else None)

    async def _decide_findings(self, now: datetime) -> None:
        """Whether a report the reader leaves undecided (``outreach.needs_decision``) is a finding: Instinct's
        ``report_substance`` point, whose model tier is the typed call (``typed_decisions.report_substance``),
        recorded on the finding (``substance_decision``) so it is asked once. Unavailable or unsure, nothing is
        recorded and the report goes to the digest (``_age_findings``): never sent on its own, never discarded."""
        if self.router is None:
            return
        engine = self.instinct or instinct_core.shared()
        for row in self.store.findings(["pending"], sorted(SHARED_FINDINGS)):
            finding = self._finding(row, now)
            if not outreach_functions.needs_decision(finding):
                continue

            async def typed(finding: outreach_functions.Finding = finding) -> Optional[str]:
                return await typed_decisions.report_substance(self.router, finding.summary, finding.topic,
                                                              tokens_allowed=self.authority.tokens_allowed)

            decision = await engine.decide("report_substance", {"topic": finding.topic, "report": finding.summary},
                                           {"intention": row.id}, deliberate=typed, event="tick")
            if decision.applied and decision.label is not None:
                self._mark_finding(row.id, "pending", now, substance_decision=decision.label, decided_by="model")

    def _findings(self, now: datetime) -> List[StoredInitiative]:
        """Done research-shaped rows whose finding outreach has not settled (``result_metadata.outreach``), found
        by that state whatever their age (a finding set back to pending after days is found again)."""
        self._reconcile_digested(now)
        return self.store.findings(["pending"], sorted(SHARED_FINDINGS))

    def _reconcile_digested(self, now: datetime) -> None:
        """A finding a queued digest carries (``queued``) is ``listed`` once that digest was delivered, and
        pending again once it can no longer be (expired, cancelled, failed, uncertain or gone): only a delivered
        digest lists anything. One still on its way stays queued. The owner gate's held items follow the same rule
        (``Ledger.settle_digest``)."""
        in_flight = {"proposed", "asked", "approved", "sending"}
        # Each digest carries its own manifest, written with it (``items``): what it listed is recovered from there,
        # whatever happened between queuing it and marking its items.
        digests = sorted(self.store.intentions(kind=["message"], types=["digest"], since=now - MANIFEST_WINDOW,
                                               limit=50), key=lambda row: row.created_at)
        for digest in digests:
            items = (digest.context or {}).get("items") if isinstance(digest.context, dict) else None
            if not isinstance(items, dict):
                continue
            held = [str(ident) for ident in items.get("held") or []]
            if digest.status == "sent":
                self.owner_gate.ledger.settle_digest(digest.id, delivered=True, at=now, ids=held)
            elif digest.status in in_flight:
                self.owner_gate.ledger.mark_queued(held, digest.id)
            else:
                self.owner_gate.ledger.settle_digest(digest.id, delivered=False, at=now)
            for ident in items.get("findings") or []:
                finding = self.store.get(str(ident))
                state = ((finding.result_metadata or {}).get("outreach") or {}) if finding is not None else {}
                if digest.status == "sent" and state.get("state") in {"digest", "queued"} and \
                        state.get("digest") in {None, digest.id}:
                    self._mark_finding(finding.id, "listed", now, digest=digest.id)
                elif digest.status in in_flight and state.get("state") == "digest":
                    self._mark_finding(finding.id, "queued", now, digest=digest.id)
        for digest_id in self.owner_gate.ledger.queued_digests():
            digest = self.store.get(digest_id)
            status = digest.status if digest is not None else None
            if status == "sent":
                self.owner_gate.ledger.settle_digest(digest_id, delivered=True, at=now)
            elif status not in in_flight:
                self.owner_gate.ledger.settle_digest(digest_id, delivered=False, at=now)
        for row in self.store.findings(["queued"], sorted(SHARED_FINDINGS)):
            digest = self.store.get(str(((row.result_metadata or {}).get("outreach") or {}).get("digest") or ""))
            status = digest.status if digest is not None else None
            if status == "sent":
                self._mark_finding(row.id, "listed", now)
            elif status not in {"proposed", "asked", "approved", "sending"}:
                self._mark_finding(row.id, "pending", now, digest=None, digest_unsent=status)

    def _mark_finding(self, ident: Any, state: str, now: datetime, **extra: Any) -> None:
        row = self.store.get(str(ident)) if ident else None
        if row is None:
            return
        metadata = dict(row.result_metadata or {})
        metadata["outreach"] = {**dict(metadata.get("outreach") or {}), "state": state, "at": now.isoformat(), **extra}
        self.store.update(row.id, result_metadata=metadata)

    def _owner_words(self, turn_ids: Iterable[str]) -> Dict[str, str]:
        """The owner's own words in these turns, while the ledger holds them (an erased turn quotes nothing)."""
        wanted = [str(ident) for ident in dict.fromkeys(turn_ids) if ident]
        if not wanted or self.ledger is None or not hasattr(self.ledger, "_connect"):
            return {}
        words: Dict[str, str] = {}
        from contextlib import closing
        from colonymind.turns.observed import owner_text
        try:
            with closing(self.ledger._connect()) as conn:
                for ident in wanted:
                    from colonymind.memory import curated
                    if self.ledger.is_source_erased(ident, self.owner_id) or curated.is_forgotten(self.ledger, ident, conn=conn):
                        continue
                    row = conn.execute("SELECT messages_json FROM turn_sources WHERE turn_id=? AND contact_id=?",
                                       (ident, self.owner_id)).fetchone()
                    if row is None:
                        continue
                    text = " ".join(own if (own := owner_text(message)) is not None
                                    else str(message.get("content") or "") for message in json.loads(row[0])
                                    if isinstance(message, dict) and message.get("role") == "user")
                    if text.strip():
                        words[ident] = text
        except Exception as error:
            logger.debug("owner words unavailable (%s)", type(error).__name__)
        return words

    def _owner_memory(self, now: datetime) -> str:
        """The owner's own words of the last month (their person-scoped turns), for a finding's relevance."""
        if self.ledger is None or not hasattr(self.ledger, "_connect"):
            return ""
        from contextlib import closing
        from .consolidate import SELF_TURN_SQL
        from colonymind.turns.observed import owner_text
        since = (now - OWNER_MEMORY).astimezone(timezone.utc).isoformat()
        texts: List[str] = []
        try:
            with closing(self.ledger._connect()) as conn:
                for row in conn.execute(
                        f"SELECT s.messages_json FROM turn_sources s WHERE s.contact_id=? AND s.scope='person' "
                        f"AND {SELF_TURN_SQL} AND coalesce(s.occurred_at, s.ingested_at) > ? "
                        f"ORDER BY coalesce(s.occurred_at, s.ingested_at) DESC LIMIT ?",
                        (self.owner_id, since, OWNER_MEMORY_TURNS)):
                    texts += [own if (own := owner_text(message)) is not None
                              else str(message.get("content") or "") for message in json.loads(row[0])
                              if isinstance(message, dict) and message.get("role") == "user"]
        except Exception as error:
            logger.debug("owner memory unavailable (%s)", type(error).__name__)
        return "\n".join(texts)

    def _owner_cadences(self, commitments: List[Dict[str, Any]], *,
                        inputs: Optional[outreach_functions.OutreachInputs] = None
                        ) -> List[outreach_functions.Cadence]:
        """One requested check-in per owner and interval, with shared attempt history.

        Called off the event loop. History is independent of the outreach window. When a live snapshot
        is supplied, requested queues leave its cost. Missed slots expire without moving the phase.
        """
        from contextlib import closing

        groups = {}
        for row in commitments:
            cadence = drive_functions.owner_cadence(row, owner_id=self.owner_id)
            if cadence is None or cadence[0] != self.owner_id or row.get("status") not in {"pending", "overdue"}:
                continue
            owner, minutes, topic = cadence
            groups.setdefault((owner, minutes), []).append((row, topic))
        cadences = []
        try:
            with closing(sqlite3.connect(self.store._db_path.resolve().as_uri() + "?mode=ro",
                                         uri=True, timeout=2)) as db:
                db.row_factory = sqlite3.Row
                db.execute("PRAGMA query_only=ON")
                history_ids = {key: {str(row["id"]) for row, _ in rows} for key, rows in groups.items()}
                known = set().union(*history_ids.values())
                if groups and self.commitments is not None:
                    # A resolved duplicate still owns part of the live group's attempt history.
                    sources = db.execute(
                        "SELECT DISTINCT source_id FROM generic_initiatives WHERE kind = 'message' "
                        "AND type = 'outreach_loop' AND source_type = 'commitment' AND entity_id = ?",
                        (self.owner_id,))
                    missing = [source[0] for source in sources if source[0] not in known]
                    if missing:
                        with closing(sqlite3.connect(self.commitments._db_path.resolve().as_uri() + "?mode=ro",
                                                     uri=True, timeout=2)) as commitments_db:
                            commitments_db.row_factory = sqlite3.Row
                            commitments_db.execute("PRAGMA query_only=ON")
                            for stored in commitments_db.execute(
                                    "SELECT * FROM commitments WHERE id IN (SELECT value FROM json_each(?))",
                                    (json.dumps(missing),)):
                                old = self.commitments._row_to_dict(stored)
                                cadence = drive_functions.owner_cadence(old, owner_id=self.owner_id)
                                if cadence is not None and cadence[:2] in history_ids:
                                    history_ids[cadence[:2]].add(old["id"])
                if inputs is not None:
                    requested = db.execute(
                        "SELECT count(*) FROM assignment_history h JOIN generic_initiatives i "
                        "ON i.id = h.initiative_id WHERE h.action = 'queued' AND h.timestamp >= ? "
                        "AND i.kind = 'message' AND i.type = 'outreach_loop' AND i.entity_id = ? "
                        "AND (json_extract(i.context, '$.requested') OR "
                        "coalesce(json_extract(i.context, '$.owner_cadence'), '') <> '')",
                        ((inputs.now - timedelta(days=1)).isoformat(), self.owner_id)).fetchone()[0]
                    inputs.queued_24h = max(0, inputs.queued_24h - requested)
                for key, rows in groups.items():
                    previous = []
                    for ident in history_ids[key]:
                        latest = db.execute(
                            "SELECT * FROM generic_initiatives WHERE kind = 'message' AND type = 'outreach_loop' "
                            "AND source_type = 'commitment' AND source_id = ? AND entity_id = ? "
                            "ORDER BY created_at DESC LIMIT 1", (ident, self.owner_id)).fetchone()
                        if latest is not None:
                            previous.append(StoredInitiative.from_row(dict(latest)))
                    if any(item.status in MIND_ACTIVE_STATUSES for item in previous):
                        continue
                    row, topic = max(rows, key=lambda item: (
                        _utc(item[0].get("made_at") or item[0].get("created_at")) or
                        datetime.min.replace(tzinfo=timezone.utc), str(item[0]["id"])))
                    minutes = key[1]
                    interval = timedelta(minutes=minutes)
                    anchor = _utc(row.get("made_at") or row.get("created_at"))
                    stated = _utc(row.get("due_at"))
                    if anchor is None and stated is None:
                        continue
                    phase = (stated or anchor).astimezone(self.tz)
                    due = phase if stated else phase + interval
                    if previous:
                        # Expiry consumes the formed slot, not the later time the restart swept it.
                        last = max(item.created_at if item.status == "expired" else self._sent_at(item)
                                   for item in previous).astimezone(self.tz)
                        # Delivery delays and holds consume slots without moving their local phase.
                        due = max(due, phase + ((last - phase) // interval + 1) * interval)
                        if due - last < interval / 2:
                            due += interval
                    local = (inputs.now if inputs is not None else self.clock()).astimezone(self.tz)
                    due += max(0, (local - due) // interval) * interval
                    if local - due > min(interval / 2, timedelta(hours=3)):
                        due += interval
                    cadences.append(outreach_functions.Cadence(str(row["id"]), topic, due, minutes))
        except Exception as error:
            from colonymind.health_counters import bump
            logger.warning("owner cadence history unavailable (%s, %s)", type(error).__name__,
                           getattr(error, "sqlite_errorname", None))
            bump("context_reads_failed")
            if inputs is not None:
                inputs.budget = "owner cadence history unavailable"
            return []
        return cadences

    def _outreach_inputs(self, now: datetime, inputs: DriveInputs) -> outreach_functions.OutreachInputs:
        """The owner branch's snapshot: what the owner cares about and said, what the mind found, what it
        sent and how the owner took it, and the holds."""
        o = outreach_functions
        interests = [o.Interest(topic=item["topic"], slug=slug(item["topic"]), level=float(item.get("weight") or 0.0),
                                origin=str(item.get("origin") or "mentioned"),
                                turn=next((str(cause)[len("turn:"):] for cause in item.get("sources") or []
                                           if str(cause).startswith("turn:")), None),
                                asked=any(str(cause).startswith("reaction:") for cause in item.get("sources") or []))
                     for item in inputs.interests]
        cares = []
        owners_open = [row for row in inputs.commitments
                       if drive_functions._obligor(row, row.get("metadata") if isinstance(row.get("metadata"), dict)
                                                   else {}, str(row.get("person_id") or "") or None,
                                                   self.owner_id) == "owner"]
        for item in self.mind_state.items(o.CARE_PREFIX):
            if float(item.get("level") or 0.0) < o.CARE_MIN_LEVEL:
                continue
            causes = [str(cause) for cause in item.get("causes") or []]
            thing = str(item.get("text") or "")
            commitment = next((cause[len("commitment:"):] for cause in causes if cause.startswith("commitment:")), None)
            if commitment is None:
                # The owner's own open item the strain names (capture lands after the turn, so it is matched here).
                match = next((row for row in owners_open if o.similar(thing, str(row.get("description") or ""))), None)
                commitment = str(match["id"]) if match is not None else None
                known = commitment is None or any(cause == f"commitment:{commitment}" for cause in causes)
                if match is not None and not known:
                    self.mind_state.set(item["key"], causes=[f"commitment:{commitment}"], now=now)
            elif not any(str(row.get("id")) == commitment for row in inputs.commitments):
                continue    # the thing it names is resolved: nothing left to offer help with
            record = next((row for row in inputs.commitments if str(row.get("id")) == commitment), None) \
                if commitment else None
            cares.append(o.Care(slug=item["key"][len(o.CARE_PREFIX):], thing=str(item.get("text") or ""),
                                level=float(item.get("level") or 0.0),
                                turn=next((cause[len("turn:"):] for cause in causes if cause.startswith("turn:")), None),
                                commitment=commitment, due_at=_utc((record or {}).get("due_at"))))
        cared = {care.commitment for care in cares if care.commitment}
        sent = []
        for row in self._outreach_sent(now):
            context = row.context if isinstance(row.context, dict) else {}
            sent.append(o.Sent(id=row.id, type=str(row.type), slug=str(context.get("topic_slug") or ""),
                               at=_utc(row.created_at) or now, verdict=row.verdict,
                               reaction=self._reaction(row).get("class"), source=str(context.get("source_ref") or ""),
                               text=str(context.get("text") or ""),
                               requested=bool(row.type == "outreach_loop" and
                                              (context.get("owner_cadence") or context.get("requested")))))
        findings = []
        for row in self._findings(now):
            findings.append(self._finding(row, now))
        owner_items = [str(row.get("description") or "") for row in owners_open]
        followups = self._followups(now, inputs)
        state = o.OutreachInputs(
            now=now, owner_id=str(self.owner_id), quiet=self.in_quiet_hours(now),
            paused_until=o.pause_until(self.mind_state.get(o.PAUSE_KEY)),
            budget=self.authority.budget_check(kind="message", recipient=self.owner_id, type="outreach_finding",
                                               now=now),
            last_owner_turn=self._owner_last_turn(), local_hour=now.astimezone(self.tz).hour,
            interests=interests, findings=findings,
            loops=o.open_loops(inputs.commitments, owner_id=str(self.owner_id), now=now, skip=cared),
            cares=cares,
            mutes={item["key"][len(o.MUTE_PREFIX):]: str(item.get("text") or item["key"][len(o.MUTE_PREFIX):])
                   for item in self.mind_state.items(o.MUTE_PREFIX)
                   if float(item.get("level") or 0.0) >= o.MUTE_FLOOR},
            timing={int(item["key"][len(o.TIMING_PREFIX):]): float(item.get("level") or 0.0)
                    for item in self.mind_state.items(o.TIMING_PREFIX)
                    if item["key"][len(o.TIMING_PREFIX):].isdigit()},
            sent=sent,
            queued_24h=self.store.count_transitions("queued", now - timedelta(days=1), kind="message",
                                                    recipient=self.owner_id,
                                                    exclude_types=self.authority.REQUESTED_TYPES),
            goals=[*owner_items, *[str(item.get("title") or "") for item in inputs.hermes_goals]],
            followups=followups, held=self._held_matters(inputs.commitments))
        if findings:
            state.memory = self._owner_memory(now)
            state.lessons = self._outreach_lessons()
            state.seen = self._digested(now)
        state.quotes = self._owner_words([*(item.turn for item in interests if item.turn),
                                          *(care.turn for care in cares if care.turn)])
        replies = self._owner_words([item.words for item in followups if item.words])
        for item in followups:
            item.words = reactions.strip_prefix(replies.get(item.words, ""))[:400]
        self._age_findings(state)
        from colonymind.plasticity.learners.reply_timing import prepare
        prepare(self, state)
        return state

    @staticmethod
    def _held_matters(commitments: Iterable[Dict[str, Any]]) -> Dict[str, str]:
        """The open items the owner holds (a hold: rescheduled with no date, listed or from its first mention),
        by id: no outreach of any kind about them (``outreach.held``)."""
        matters = {}
        for row in commitments:
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            if not row.get("due_at") and metadata.get("reschedule") and row.get("status") in {"pending", "overdue"}:
                matters[str(row.get("id"))] = " ".join(str(row.get("description") or "").split())
        return matters

    def _followups(self, now: datetime, inputs: DriveInputs) -> List[outreach_functions.Followup]:
        """The owner's "dig deeper" (or "yes, help me with it") replies of the last week not yet made a task.
        The follow-up keeps an assistant promise only when capture linked it to that very outreach (the reply
        was linked to it and the extractor marked the promise as following it up: ``outreach.binds_followup``),
        and it is the only one. Any other promise is its own obligation, which duty keeps.
        ``words`` carries the reply's turn id until the ledger is read."""
        o = outreach_functions
        replies = []
        for row in self.store.intentions(kind=["message"], since=now - timedelta(days=7), limit=500,
                                         recipient=self.owner_id):
            reaction = self._reaction(row)
            if row.type in OUTREACH_MESSAGES and reaction.get("class") == "positive" and not reaction.get("followup"):
                replies.append((row, str(reaction.get("turn") or "")))
        items = []
        for row, turn in replies:
            context = row.context if isinstance(row.context, dict) else {}
            topic = str(context.get("topic") or row.description)
            item = o.Followup(outreach_id=row.id, topic=topic, slug=slug(topic),
                              shared=o.excerpt(str(context.get("text") or ""), topic, limit=300), words=turn,
                              offer=row.type in {"outreach_loop", "outreach_care"})
            bound = [record for record in inputs.commitments
                     if self._assistant_answer(record) and o.binds_followup(record, item)]
            if len(bound) == 1:
                item.commitment, item.commitment_due = str(bound[0]["id"]), _utc(bound[0].get("due_at"))
            items.append(item)
        return items

    def _assistant_answer(self, record: Dict[str, Any]) -> bool:
        """An open answer the assistant promised (capture's ``kind``), whoever it was promised to find out."""
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        return (metadata.get("kind") == outreach_functions.ANSWER_KIND
                and drive_functions._obligor(record, metadata, str(record.get("person_id") or "") or None,
                                             self.owner_id) == "assistant")

    def outreach_replied(self, turn_id: str) -> Optional[Dict[str, str]]:
        """The outreach the owner's turn ``turn_id`` replied to (``owner_turn`` linked it and recorded the
        reaction), or None: what capture tells the extractor, so that only the promise following THAT outreach up
        is linked to it (``outreach.binds_followup``). Reads only."""
        if not (self.faculties.get("outreach") and self.owner_id and turn_id):
            return None
        for row in self.store.intentions(kind=["message"], since=self.clock() - timedelta(days=7), limit=500,
                                         recipient=self.owner_id):
            if row.type in OUTREACH_MESSAGES and str(self._reaction(row).get("turn") or "") == str(turn_id):
                context = row.context if isinstance(row.context, dict) else {}
                topic = str(context.get("topic") or row.description)[:160]
                return {"id": str(row.id), "topic": topic,
                        "text": outreach_functions.excerpt(str(context.get("text") or ""), topic, limit=300)}
        return None

    def _digested(self, now: datetime) -> List[str]:
        """What the digest listed, or holds to list, of the last month's findings: shared already, so a later
        report repeating it is no news."""
        texts = []
        for row in self.store.intentions(status=["done"], kind=["task"], since=now - OUTREACH_HISTORY, limit=500):
            if row.type not in SHARED_FINDINGS or ((row.result_metadata or {}).get("outreach") or {}).get(
                    "state") not in {"digest", "queued", "listed"}:
                continue
            context = row.context if isinstance(row.context, dict) else {}
            texts.append(outreach_functions.excerpt(str(row.result or ""), str(context.get("topic") or row.description),
                                                    substantive=True))
        return [text for text in texts if text]

    def _outreach_lessons(self) -> List[Any]:
        """The lessons the owner's verdicts taught about what they want to hear (``topic:*`` from their words,
        ``outreach_*`` from their ratings of outreach): owner-verified and active only."""
        if not self.lessons.enabled:
            return []
        try:
            return [lesson for lesson in self.lessons.served() if lesson.status == "active" and lesson.verified == "owner"
                    and str(lesson.signature).startswith(("topic:", "outreach_"))]
        except Exception as error:
            logger.warning("lessons unavailable for outreach (%s)", type(error).__name__)
            return []

    def _age_findings(self, state: outreach_functions.OutreachInputs) -> None:
        """A report that found nothing (``empty``) or only what was already shared (``repeat``) is settled at
        once, never sent nor listed; one on a topic the owner muted, asked for or not, is ``muted``; one not
        shared inside its window goes to the digest when it was worth it (``DIGEST_FLOOR``), else it is
        dropped."""
        o = outreach_functions
        keep = []
        for finding in state.findings:
            settled = o.settle(finding, state)
            if settled == o.UNCERTAIN:
                # Whether it is a finding is uncertain: the digest carries it, never a message of its own.
                self._mark_finding(finding.id, "digest", state.now, substance=o.UNCERTAIN)
            elif settled:
                self._mark_finding(finding.id, settled, state.now)
            elif o.muted(state, finding.slug, finding.topic):
                self._mark_finding(finding.id, "muted", state.now)
            elif finding.requested_by is None and state.now - finding.completed_at > o.FINDING_WINDOW:
                value = o.digest_value(finding, state)
                self._mark_finding(finding.id, "digest" if value >= o.DIGEST_FLOOR else "dropped", state.now,
                                   value=round(value, 3))
            else:
                keep.append(finding)
        state.findings = keep

    # -- the owner's words about outreach -------------------------------------------------------

    async def owner_turn(self, text: str, *, turn_id: str, occurred_at: Any = None,
                         session_id: str | None = None, scheduled: bool = False,
                         evidence: Mapping[str, Any] | None = None) -> Optional[Dict[str, Any]]:
        """What one owner turn says about outreach, applied at once (the turn path's hook, architecture 4.10).

        ``evidence`` (the channel's references for this message: platform, chat, chat type, message id, what it
        replies or reacts to) makes it the ingest read of the answer route for a message no plugin read before the
        model ran (``read_answer``, ``route='ingest'``): read without the chat's ledger, so only a reaction or a
        quoted reply binds, never the follow-up. A message read already (by either route) is not read again.

        The reply is linked to the newest open outreach (sent in the last day, not yet answered) whose topic
        it names, else, when it reads as a reply to that newest one, by position (``_link``). What it says
        about the linked outreach is what is about it (``Reading.reaction(about=...)``): a negative on
        another topic or a request aimed elsewhere is not. What it says is learned where the mind already
        learns: the verdict on the row (``Outcomes.rate`` and its feedback), the topic's interest level, a
        mute, the pause, the hour's timing mark, care. An explicit stop pauses outreach and cancels what
        is queued in the same call; a bare "stop" or a vague "not today" does so only as a reply to an
        outreach. A turn carrying an open ask's code is that ask's answer, never a reaction (its explicit
        stop still applies), and so is a turn the answer route read as an answer (``_answers_ask``). Idempotent per
        turn. None with the faculty off.

        With ``drift_check`` on, the owner's word that the agent drifted or was wrong is kept first, whatever
        the outreach faculty: the next identity check is judged by it (``DriftCheck.owner_said``)."""
        if self.faculties.get("drift_check") and self.owner_id:
            try:
                self.drift.owner_said(text, turn_id=str(turn_id or ""), at=_utc(occurred_at) or self.clock())
            except Exception:
                logger.warning("owner drift word not kept", exc_info=True)
        own_turn = bool(self.owner_id) and not scheduled and not str(session_id or "").startswith("mind:")
        if own_turn and evidence is not None:
            try:
                await self.read_answer({**dict(evidence), "text": text, "turn_id": str(turn_id or ""),
                                        "contact_id": self.owner_id, "arrived_at": occurred_at, "ledger": None},
                                       route="ingest")
            except Exception:
                logger.warning("owner answer at ingest not read", exc_info=True)
        answered = self._answers_ask(reactions.strip_prefix(text), turn=str(turn_id or ""))
        if not (self.faculties.get("outreach") and self.owner_id):
            return None
        o = outreach_functions
        now = _utc(occurred_at) or self.clock()
        turn = str(turn_id or "")
        previous = self._owner_last_turn(exclude=turn)
        if previous is None or now >= previous:
            self.mind_state.set(o.OWNER_TURN_KEY, text=now.isoformat(), causes=[f"turn:{turn}"], now=now)
        body = reactions.strip_prefix(text)
        summary: Dict[str, Any] = {"linked": None, "classes": [], "applied": []}
        reading = reactions.read(body, contacts=await self._contact_names())
        settled = False
        if reading.unsure:
            # Whose instruction a stop, a pause or a resume beside someone else's words is: Instinct's
            # owner_instruction point, whose model tier is the typed call; unavailable or unsure, the reader's
            # conservative fallback stands.
            async def typed() -> Optional[str]:
                return await typed_decisions.owner_instruction(self.router, body,
                                                               tokens_allowed=self.authority.tokens_allowed)

            decision = await (self.instinct or instinct_core.shared()).decide(
                "owner_instruction", {"message": body}, {"turn": turn}, deliberate=typed, event="owner_turn")
            reading.decided(decision.label if decision.applied else None)
            settled = not reading.unsure
        summary["classes"] = reading.classes
        cause = f"turn:{turn}"
        if answered:
            summary["answer"] = True
            self._apply_holds(reading, summary, cause=cause, now=now, linked=None)
            return summary
        if turn and any(self._reaction(row).get("turn") == turn for row in self._outreach_sent(now)
                        + self.store.intentions(kind=["message"], since=now - OUTREACH_HISTORY, limit=500,
                                                recipient=self.owner_id)):
            summary["classes"] = []
            return summary      # a retried turn: its effects were applied
        row, how = self._link(reading, body, now, previous)
        cls = reading.reaction(about=self._about(row)) if row is not None else None
        if how == "position" and cls is None:
            row, how = None, ""    # nothing in the turn is about what was sent
        read = ""
        if row is not None and cls is None and not settled:
            # A typed decision that answered whose instruction this is stands: the reply model is not asked again.
            cls = await self._decided_reply(body, row)
            read = "decision" if cls is not None else ""
        self._apply_holds(reading, summary, cause=cause, now=now, linked=cls, replied=row is not None)
        if row is not None:
            summary["linked"] = row.id
            summary["applied"].append(self._react(row, cls or "engaged", turn=turn, how=how, now=now, read=read))
        for topic in reading.negative_objects:
            self._mute(topic, cause=cause, now=now)
        muted = {slug(topic) for topic in reading.negative_objects}
        for topic in reading.declarations:
            if slug(topic) not in muted:
                self._declare(topic, cause=cause, now=now)
        if not reading.stop:
            for thing in reading.strains:
                self.mind_state.set(f"{o.CARE_PREFIX}{slug(thing)}", level=1.0, text=thing, causes=[cause],
                                    half_life_s=o.CARE_HALF_LIFE.total_seconds(), now=now)
        for thing in [*reading.reliefs, *([body] if reading.relieved else [])]:
            self._relieve(thing, now=now)
        if summary["applied"] or reading.classes:
            logger.info("owner turn about outreach: %s", ", ".join(reading.classes) or "engaged")
        return summary

    def _answers_ask(self, body: str, *, turn: str = "") -> bool:
        """Whether the turn is an ask's answer, so the question and outreach readers leave it alone: the answer
        route read it as one (``answer_turns``: it answered, undid, was asked about, showed or put off an ask), or
        it carries an open ask's code (the id the console shows), so it is about that ask and never a question's
        answer: as typed in any case when the code holds a digit, in capitals when it is all letters (a code such
        as ``THE`` is also a word)."""
        read = self.answer_turns.for_turn(turn) if turn else None
        if read is not None and read.as_answer:
            return True
        for code in self.store.open_ask_codes():
            flags = re.IGNORECASE if any(char.isdigit() for char in str(code)) else 0
            if re.search(rf"\b{re.escape(str(code))}\b", body, flags):
                return True
        return False

    def _apply_holds(self, reading: Any, summary: Dict[str, Any], *, cause: str, now: datetime,
                     linked: Optional[str], replied: bool = False) -> None:
        """The owner's stop, resume or pause for today: an explicit one always, a bare "stop" or a vague
        "not today" only as the reaction of a reply linked to an outreach (``linked``). A resume nobody could
        attribute (no model decision) lifts the pause only as a reply linked to an outreach (``replied``). A linked
        stop is the phrase reader's or the decision model's (``_decided_reply``)."""
        if reading.stop_explicit or linked == "stop":
            self.pause_outreach(None, by=cause, now=now)
            summary["applied"].append("paused")
        elif reading.resume or (reading.resume_if_linked and replied):
            self.resume_outreach(by=cause)
            summary["applied"].append("resumed")
        elif reading.pause_explicit or (reading.pause_today and linked == "pause_today"):
            self.pause_outreach(self._next_day(now), by=cause, now=now)
            summary["applied"].append("paused for today")

    async def owner_signal(self, signal: Mapping[str, Any]) -> None:
        """The appraisal's second net: an owner opt-out it saw (``AppraisalStore.on_owner``) pauses outreach
        when the phrases missed it. Never touches ``may_contact``: the owner's permission is by identity."""
        if not (self.faculties.get("outreach") and self.owner_id) or not signal.get("opt_out"):
            return None
        if outreach_functions.pause_until(self.mind_state.get(outreach_functions.PAUSE_KEY)) is None:
            self.pause_outreach(None, by=f"appraisal:{signal.get('turn_id') or 'owner'}")
        return None

    def pause_outreach(self, until: Optional[datetime], *, by: str, now: datetime | None = None) -> Dict[str, Any]:
        """No unprompted outreach until ``until`` (None: until the owner resumes it); what is queued is
        cancelled now. Reminders and answers the owner asked for are not outreach and keep going."""
        o = outreach_functions
        now = now or self.clock()
        current = o.pause_until(self.mind_state.get(o.PAUSE_KEY))
        if until is not None and current is not None and current >= until:
            until = current      # a longer pause stands
        self.mind_state.set(o.PAUSE_KEY, level=1.0, text=o.INDEFINITE if until is None or until.year == datetime.max.year
                            else until.isoformat(), causes=[by], now=now)
        for row in self.store.intentions(status=["approved", "proposed"], kind=["message"], limit=500,
                                         recipient=self.owner_id):
            if row.type in OUTREACH_TYPES:
                self.outcomes.record(row.id, status="cancelled", summary="the owner paused check-ins", by="owner",
                                     implicit_verdict=False)
        if until is None:
            self.autobiography.record(f"outreach-pause-{now.strftime('%Y%m%dT%H%M%S')}", "outreach_paused",
                                      "The owner asked me to stop checking in unprompted; I will not reach out "
                                      "until they say I may again.")
        return self._outreach_state(now)

    def resume_outreach(self, *, by: str = "owner") -> Dict[str, Any]:
        self.mind_state.delete(outreach_functions.PAUSE_KEY)
        return self._outreach_state(self.clock())

    def outreach_switch(self, state: str, *, by: str = "owner") -> Dict[str, Any]:
        """``colonymind mind outreach on|off`` and ``POST /v1/mind/outreach``: the owner's recovery path."""
        if not (self.faculties.get("outreach") and self.owner_id):
            return {"enabled": False}
        if state == "off":
            return self.pause_outreach(None, by=by)
        if state == "on":
            return self.resume_outreach(by=by)
        return self._outreach_state(self.clock())

    def _next_day(self, now: datetime) -> datetime:
        """The start of the owner's next day: the end of the quiet hours, else local midnight."""
        local = now.astimezone(self.tz)
        if self.quiet is not None:
            end = local.replace(hour=self.quiet[1] // 60, minute=self.quiet[1] % 60, second=0, microsecond=0)
            return end if end > local else end + timedelta(days=1)
        return (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    def _link(self, reading: Any, body: str, now: datetime,
              previous: Optional[datetime]) -> tuple[Optional[StoredInitiative], str]:
        """The open outreach this turn answers, and how the link was made (``topic`` or ``position``).

        A topic link: the turn names the topic of an open outreach (the newest such). A position link, for
        a turn whose words need one ("not now", "tell me more", a bare "not useful"): the turn reads as a
        reply (short or referring back), it is the owner's first turn since the newest open outreach went
        out, that outreach is the newest thing the mind sent the owner, the owner was not in the middle of
        a conversation with the assistant when it went out (their turn before it was ``CONVERSATION_GAP``
        or more earlier), and the turn is not talking about something else (a request of its own, a redo
        of the assistant's work: ``reactions.elsewhere``)."""
        o = outreach_functions
        sent_rows = []
        for row in self.store.intentions(kind=["message"], since=now - timedelta(hours=o.REPLY_HOURS + 48), limit=500,
                                         recipient=self.owner_id):
            sent_at = _utc(row.completed_at) or _utc(row.created_at)
            # A turn's time comes to the whole second (the host's ``occurred_at``); so does the send's here.
            sent_at = sent_at.replace(microsecond=0) if sent_at is not None else None
            if row.status in {"sent", "uncertain"} and sent_at is not None and sent_at <= now:
                sent_rows.append((sent_at, row))
        sent_rows.sort(key=lambda item: item[0], reverse=True)
        open_rows = [(sent_at, row) for sent_at, row in sent_rows
                     if row.type in OUTREACH_MESSAGES and not self._reaction(row)
                     and now - sent_at <= timedelta(hours=o.REPLY_HOURS)]
        for _, row in open_rows:
            topic = str((row.context or {}).get("topic") or "") if isinstance(row.context, dict) else ""
            if topic and o.similar(topic, body):
                return row, "topic"
        if not (open_rows and reading.needs_link() and reactions.refers_back(body)):
            return None, ""
        sent_at, row = open_rows[0]
        if any(other_at > sent_at for other_at, other in sent_rows if other.id != row.id):
            return None, ""     # a newer message from the mind is what a bare reply answers
        if previous is not None and (previous >= sent_at or sent_at - previous < o.CONVERSATION_GAP):
            return None, ""     # not the first turn since, or it went out mid-conversation
        if reactions.elsewhere(body, self._outreach_words(row)):
            return None, ""
        return row, "position"

    @staticmethod
    def _outreach_words(row: StoredInitiative) -> set:
        """The words an outreach said (its topic, reason and text): what a reply about it may name."""
        context = row.context if isinstance(row.context, dict) else {}
        return reactions.content_terms(" ".join(str(context.get(key) or "") for key in ("topic", "why", "text")))

    def _about(self, row: StoredInitiative) -> Callable[[str], bool]:
        """Whether a phrase of a reply is about this outreach: it names its topic, or names nothing it did not say."""
        context = row.context if isinstance(row.context, dict) else {}
        topic = str(context.get("topic") or "")
        known = self._outreach_words(row)
        return lambda text: bool(topic and outreach_functions.similar(topic, text)) or not (
            reactions.content_terms(text) - known)

    # The decision model's labels for a reply (Instinct's ``outreach_reply`` point) as the reactions' classes.
    REPLY_CLASSES = {"dig_deeper": "positive", "not_interested": "negative", "not_now": "not_now", "stop": "stop"}

    async def _decided_reply(self, body: str, row: StoredInitiative) -> Optional[str]:
        """The decision model's reading of a linked reply the phrase tables read nothing in, as a reaction class;
        None (the point in shadow, no answer, an unsure one, or "engaged") leaves the reply engagement."""
        context = row.context if isinstance(row.context, dict) else {}
        decision = await (self.instinct or instinct_core.shared()).decide(
            "outreach_reply", {"text": body, "topic": str(context.get("topic") or "a finding")},
            {"intention": row.id}, event="owner_turn")
        return self.REPLY_CLASSES.get(decision.label) if decision.applied else None

    def _react(self, row: StoredInitiative, cls: str, *, turn: str, how: str, now: datetime, read: str = "") -> str:
        """Record the owner's reaction on the row and learn from it; ``read`` names what read the class when it
        was not the phrase tables ("decision": the decision model)."""
        o = outreach_functions
        context = row.context if isinstance(row.context, dict) else {}
        topic = str(context.get("topic") or "")
        metadata = dict(row.result_metadata or {})
        metadata["reaction"] = {"class": cls, "turn": turn, "at": now.isoformat(), "by": how or "phrase",
                                **({"read": read} if read else {})}
        self.store.update(row.id, result_metadata=metadata)
        news = row.type in {"outreach_finding", OUTREACH_ANSWER}
        cause = f"reaction:{row.id}"
        if cls in {"negative", "stop"}:
            self.outcomes.rate(row.id, "not_useful" if cls == "negative" else "dismissed", by="owner")
            if cls == "negative" and topic:
                self._mute(topic, cause=f"turn:{turn}", now=now)
        elif cls in {"positive", "welcome"}:
            self.outcomes.rate(row.id, "useful", by="owner")
            if news and topic:
                # "dig deeper" is the owner asking for more (``reaction:``); "great find" is them welcoming it.
                self._declare(topic, cause=cause if cls == "positive" else f"welcomed:{row.id}", now=now)
            self._satisfy(row, now)
        elif cls == "engaged":
            self.outcomes.rate(row.id, "actioned", by="mind")
            if news and topic:
                self._bump_interest(topic, 0.5, cause=f"engaged:{row.id}", now=now)
            self._satisfy(row, now)
        elif cls == "not_now":
            self.pause_outreach(now + o.NOT_NOW_HOLD, by=f"turn:{turn}", now=now)
            hour = now.astimezone(self.tz).hour
            for offset, delta in ((0, 1.0), (-1, 0.5), (1, 0.5)):
                self.mind_state.bump(f"{o.TIMING_PREFIX}{(hour + offset) % 24:02d}", delta, cap=1.0,
                                     half_life_s=INTEREST_HALF_LIFE_S, causes=[f"turn:{turn}"], now=now)
        return cls

    def _satisfy(self, row: StoredInitiative, now: datetime) -> None:
        if self.faculties["drives"]:
            self.mind_state.bump("satiety.social", 1.0, half_life_s=SATIETY_HALF_LIFE_S, causes=[f"intention:{row.id}"],
                                 now=now)

    def _bump_interest(self, topic: str, delta: float, *, cause: str, now: datetime) -> None:
        key = f"interest:{slug(topic)}"
        current = self.mind_state.get(key) or {}
        if cause in (current.get("causes") or []):
            return
        self.mind_state.set(key, level=min(3.0, max(0.0, float(current.get("level") or 0.0) + delta)), text=topic,
                            causes=[cause], half_life_s=INTEREST_HALF_LIFE_S, now=now)

    def _declare(self, topic: str, *, cause: str, now: datetime) -> None:
        """An interest the owner declared (or asked more about): one step up, once per cause, and a mute on
        it lifted."""
        self.mind_state.delete(f"{outreach_functions.MUTE_PREFIX}{slug(topic)}")
        if cause not in ((self.mind_state.get(f"interest:{slug(topic)}") or {}).get("causes") or []):
            self.add_interest(topic, by=cause)

    def _mute(self, topic: str, *, cause: str, now: datetime) -> None:
        """The owner does not want messages about ``topic``: muted for months, its interest at 0, and any
        outreach on it still waiting cancelled."""
        o = outreach_functions
        key = slug(topic)
        self.mind_state.set(f"{o.MUTE_PREFIX}{key}", level=1.0, text=topic, causes=[cause],
                            half_life_s=o.MUTE_HALF_LIFE.total_seconds(), now=now)
        if self.mind_state.get(f"interest:{key}") is not None:
            # Lowered, not raised: the mute's own row carries the cause, the interest keeps whose it was.
            self.mind_state.set(f"interest:{key}", level=0.0, now=now)
        for row in self.store.intentions(status=["approved", "proposed"], kind=["message"], limit=500,
                                         recipient=self.owner_id):
            context = row.context if isinstance(row.context, dict) else {}
            if row.type in OUTREACH_TYPES and (context.get("topic_slug") == key
                                               or o.similar(topic, str(context.get("topic") or ""))):
                self.outcomes.record(row.id, status="cancelled", summary=f"the owner muted {topic}", by="owner",
                                     implicit_verdict=False)

    def _relieve(self, text: str, *, now: datetime) -> None:
        """Relief about a thing: its care is over, and an offer of help with it still waiting is cancelled."""
        o = outreach_functions
        for item in self.mind_state.items(o.CARE_PREFIX):
            thing = str(item.get("text") or "")
            if float(item.get("level") or 0.0) <= 0 or not (o.similar(thing, text) or slug(thing) == slug(text)):
                continue
            self.mind_state.set(item["key"], level=0.0, now=now)
            for row in self.store.intentions(status=["approved", "proposed"], kind=["message"], limit=500,
                                             recipient=self.owner_id):
                context = row.context if isinstance(row.context, dict) else {}
                if row.type == "outreach_care" and context.get("topic_slug") == slug(thing):
                    self.outcomes.record(row.id, status="cancelled", summary=f"the owner sorted out the {thing}",
                                         by="owner", implicit_verdict=False)

    async def _contact_names(self) -> List[str]:
        """Known contacts' names (never an outreach topic), read at most every ten minutes."""
        now = self.clock()
        cached = getattr(self, "_names_cache", None)
        if cached is not None and now - cached[0] < timedelta(minutes=10):
            return cached[1]
        names: List[str] = []
        lister = getattr(self.contacts, "list", None)
        if callable(lister):
            try:
                for contact in await lister(limit=200, offset=0) or []:
                    record = contact.to_dict() if hasattr(contact, "to_dict") else dict(contact)
                    if record.get("contact_id") == self.owner_id:
                        continue
                    names += [str(record.get(key)) for key in ("display_name", "given_name") if record.get(key)]
            except Exception as error:
                logger.debug("contact names unavailable (%s)", type(error).__name__)
        self._names_cache = (now, names)
        return names

    def _score_outreach(self, now: datetime) -> Optional[Dict[str, int]]:
        """Silence and the appraisal's second net (architecture 4.10). An outreach the owner did not answer
        within a day is ``ignored`` (weak: the topic's interest x0.8; the streak raises the next cost). Before
        that, an owner dismissal the appraisal saw after the send is a negative on it when it names its topic,
        or names nothing and is the owner's first event after the newest open outreach (a dismissal of
        something else, say the dentist booking, touches no outreach)."""
        if not (self.faculties.get("outreach") and self.owner_id):
            return None
        o = outreach_functions
        scored = {"ignored": 0, "negative": 0}
        rows = [(sent, row) for row in self.store.intentions(kind=["message"], since=now - OUTREACH_HISTORY, limit=1000,
                                                             recipient=self.owner_id)
                if row.type in OUTREACH_MESSAGES and row.status in {"sent", "uncertain"} and not self._reaction(row)
                for sent in [_utc(row.completed_at) or _utc(row.created_at)] if sent is not None]
        if not rows:
            return scored
        events = []
        reader = getattr(self.appraisals, "affect_events", None)
        if callable(reader):
            try:
                events = sorted(reader(since=min(sent for sent, _ in rows).timestamp()),
                                key=lambda event: float(event.get("occurred_at") or 0))
            except Exception as error:
                logger.debug("owner events unavailable (%s)", type(error).__name__)
        newest = max(rows, key=lambda item: item[0])[1].id
        for sent, row in rows:
            context = row.context if isinstance(row.context, dict) else {}
            topic = str(context.get("topic") or "")
            after = [event for event in events if float(event.get("occurred_at") or 0) > sent.timestamp()]
            dismissal = next((event for index, event in enumerate(after) if event.get("kind") == "dismissed"
                              and ((topic and o.similar(topic, str(event.get("topic") or "")))
                                   or (index == 0 and row.id == newest and not str(event.get("topic") or "").strip()))),
                             None)
            if dismissal is not None:
                self._react(row, "negative", turn=str(dismissal.get("turn_id") or ""), how="appraisal", now=now)
                scored["negative"] += 1
            elif (now - sent >= timedelta(hours=o.REPLY_HOURS)
                  and not (row.type == "outreach_loop" and
                           (context.get("owner_cadence") or context.get("requested")))):
                metadata = dict(row.result_metadata or {})
                metadata["reaction"] = {"class": "silence", "at": now.isoformat(), "by": "silence"}
                self.store.update(row.id, result_metadata=metadata)
                self.outcomes.rate(row.id, "ignored", by="mind")
                if row.type in {"outreach_finding", OUTREACH_ANSWER} and topic:
                    key = f"interest:{slug(topic)}"
                    entry = self.mind_state.get(key)
                    if entry is not None:
                        # Lowered, not raised: the verdict on the row is the record; whose interest it is stays.
                        self.mind_state.set(key, level=float(entry.get("level") or 0.0) * 0.8, now=now)
                scored["ignored"] += 1
        return scored

    def _outreach_formed(self, row: StoredInitiative, now: datetime) -> None:
        """A finding's outreach (or a requested answer) is formed: the finding is shared. A follow-up the
        owner asked for is formed: the outreach it answers records it."""
        context = row.context if isinstance(row.context, dict) else {}
        source = str(context.get("source_ref") or "")
        if row.type in {"outreach_finding", OUTREACH_ANSWER} and source.startswith("intention:"):
            self._mark_finding(source[len("intention:"):], "sent", now, row=row.id)
        elif row.type == OUTREACH_FOLLOWUP and context.get("requested"):
            outreach_row = self.store.get(str(context["requested"]))
            if outreach_row is not None:
                metadata = dict(outreach_row.result_metadata or {})
                metadata["reaction"] = {**self._reaction(outreach_row), "followup": row.id}
                self.store.update(outreach_row.id, result_metadata=metadata)

    def _health_probes(self) -> Dict[str, bool]:
        probes: Dict[str, Callable[[], Any]] = {"initiatives": lambda: self.store.count()}
        if self.commitments is not None:
            probes["commitments"] = lambda: self.commitments.list(limit=1)
        if self.ledger is not None and hasattr(self.ledger, "_connect"):
            def ledger_probe() -> None:
                conn = self.ledger._connect()
                try:
                    conn.execute("SELECT 1")
                finally:
                    conn.close()
            probes["ledger"] = ledger_probe
        results: Dict[str, bool] = {}
        for name, probe in probes.items():
            try:
                probe()
                results[name] = True
            except Exception as error:
                logger.warning("health probe %s failed (%s)", name, type(error).__name__)
                results[name] = False
        return results

    def effective_weights(self) -> Dict[str, float]:
        satiety = {name: float((self.mind_state.get(f"satiety.{name}") or {}).get("level") or 0.0) for name in DRIVES}
        return drive_functions.effective_weights(self.drive_weights, satiety, faculty_on=self.faculties["drives"])

    def _raise_concerns(self, inputs: DriveInputs, now: datetime) -> tuple[Dict[str, Any], List[str]]:
        """Every enabled drive over the snapshot; each candidate bumps its concern by ``dedup_key``.

        Returns the drive levels and the keys of concerns that were raised
        again while an intention is under way (the events reconsideration
        looks at).
        """
        events: List[str] = []
        raised = 0
        results = drive_functions.run(inputs, self.drive_weights)
        for name, (level, candidates) in results.items():
            self.drive_levels[name] = level
            for candidate in candidates:
                concern, outcome = self.concerns.bump(
                    drive=name, kind=candidate.concern_kind, summary=candidate.concern or candidate.title,
                    dedup_key=candidate.dedup_key, salience=candidate.salience, sources=candidate.evidence,
                    detail=candidate.as_detail(), now=now)
                if outcome in {"created", "reopened"}:
                    raised += 1
                elif outcome == "bumped" and concern is not None and concern.status == "intended":
                    events.append(concern.dedup_key)
        for name in DRIVES:
            if name not in results:
                self.drive_levels[name] = 0.0
        # A check-in is owed only while the drive finds it due: one it no longer raises (the contact
        # talked, a send moved the cooldown, affect turned, people went off) is dropped, never formed late.
        due = {candidate.dedup_key for candidate in results.get("social", (0.0, []))[1]}
        due |= {candidate.dedup_key for candidate in results.get("duty", (0.0, []))[1]
                if (candidate.extra or {}).get("owner_cadence")}
        for concern in self.concerns.open(limit=1000):
            if (concern.drive == "social" or (concern.detail.get("extra") or {}).get("owner_cadence")) \
                    and concern.dedup_key not in due:
                self.concerns.drop(concern.id, note="no longer due", now=now)
        return {"levels": dict(self.drive_levels), "raised": raised, "weights": self.effective_weights()}, events

    def _bdi(self, now: datetime, events: List[str]) -> int:
        """Reconsider an active intention only when an event matches it (architecture 3.3)."""
        if not events:
            return 0
        count = 0
        for row in self.store.intentions(status=list(MIND_ACTIVE_STATUSES), limit=500):
            if not Deliberation.matches(row, events):
                continue
            concern = self.concerns.by_key(row.dedup_key)
            try:
                reason = self._stale_reason(row)
            except Unreadable as error:
                logger.warning("intention %s held: %s", row.id, error)
                continue
            decision = Deliberation.reconsider(row, concern, invalidated=reason)
            if decision == "cancel":
                self._cancel_stale(row, str(reason))
            elif decision == "refresh" and concern is not None:
                self.store.transition(row.id, row.status, action="reconsidered", at=now,
                                      details={"event": row.dedup_key, "decision": decision},
                                      context=refresh_context(row, concern))
            else:
                continue
            count += 1
        return count

    # -- goals -------------------------------------------------------------------------------

    def _evaluate_goal_check(self, check: Any, **state: Any) -> Optional[bool]:
        return evaluate_check(check, commitments=self.commitments, followups=self.followups, **state)

    def _tend_goals(self, now: datetime) -> Dict[str, Any]:
        """Close goals that are satisfied, spent or past their horizon; raise the next step of the rest.

        A closed goal takes its pending steps with it: a step still waiting
        (deferred, asked or approved) is cancelled and a step concern not yet
        formed is dropped; a dispatched step is running in Hermes and its
        outcome still comes back. Steps are raised only for approved goals: a
        goal still asked or deferred owns no work yet.
        """
        closed = []
        for item in self.goals.due(now, evaluate=self._evaluate_goal_check):
            goal, outcome, why = item["goal"], item["outcome"], item["why"]
            if outcome == "done":
                self.outcomes.record(goal.id, status="done", outcome="done", summary=f"goal satisfied: {why}",
                                     verified="check", by="mind")
            else:
                self.outcomes.record(goal.id, status="expired", outcome="expired", summary=f"goal expired: {why}",
                                     by="mind", implicit_verdict=False)
            closed.append({"id": goal.id, "outcome": outcome, "why": why,
                           "steps_retired": self._retire_steps(goal, why, now)})
        steps = 0
        for goal in self.goals.approved():
            candidate = self.goals.next_step(goal)
            if candidate is None:
                continue
            self.concerns.bump(drive=candidate.drive, kind="goal_step", summary=candidate.concern,
                               dedup_key=candidate.dedup_key, salience=candidate.salience,
                               sources=candidate.evidence, detail=candidate.as_detail(), now=now)
            steps += 1
        return {"open": len(self.goals.open()), "closed": closed, "steps_raised": steps}

    async def _tend_wishes(self, now: datetime) -> Optional[Dict[str, Any]]:
        """Granted wishes become goals (through authority, when a goal slot is free), pursued ones follow their
        goal, and a wish whose pursuit ended 14 days ago records its outcome. Nothing here grants a wish."""
        if not self.faculties["wishes"]:
            return None
        summary = {"adopted": 0, "measuring": 0, "measured": 0, "controls": 0}
        for row in self.wishes.rows():
            state = wish_state(row)
            try:
                if state == "granted":
                    summary["adopted"] += await self._adopt_wish(row, now)
                elif state == "pursuing":
                    summary["measuring"] += self.wishes.follow(row, now) == "measuring"
                elif state == "measuring":
                    work = await self.wishes.outcome_work(row, now)     # the work log, read off the event loop
                    summary["measured"] += self.wishes.measure(row, now, work=work) is not None
                elif state in {"denied", "expired"}:
                    work = await self.wishes.outcome_work(row, now)
                    summary["controls"] += self.wishes.measure_control(row, now, work=work) is not None
            except Exception as error:
                logger.warning("wish %s not tended (%s)", row.id, type(error).__name__)
        return summary

    async def _adopt_wish(self, row: StoredInitiative, now: datetime) -> int:
        """A granted wish's goal, formed like any goal (authority, the goal budget); a wish no goal can carry
        (goals off, the goal refused, no slot for ``GOAL_WAIT``) is measured from now as it is."""
        if not self.goals.enabled:
            self.wishes.unpursued(row, now, "goals are off")
            return 0
        candidate = self.wishes.goal_candidate(row, now, horizon_days=int(self.policy.budgets.goal_horizon_days or 7))
        goal = self.store.get_by_dedup_key(candidate.dedup_key)
        if goal is None:
            if not self.goals.may_adopt():
                granted = _utc(wish_of(row).get("granted_at")) or now
                if now - granted >= WISH_GOAL_WAIT:
                    self.wishes.unpursued(row, now, "no goal slot opened within 7 days")
                return 0
            goal = await self._form(candidate, candidate.salience, now)
        if goal is None:
            return 0
        if goal.status not in {"approved", "asked", "proposed"}:
            self.wishes.unpursued(row, now, f"its goal was {goal.status}: {goal.decision_reason}")
            return 0
        self.wishes.pursue(row, goal, now)
        return 1

    def _retire_steps(self, goal: StoredInitiative, why: str, now: datetime) -> int:
        retired = 0
        for step in self.goals.steps(goal):
            if step.status in {"proposed", "asked", "approved"}:
                self.outcomes.record(step.id, status="cancelled", summary=f"goal closed: {why}", verified="none",
                                     by="mind", implicit_verdict=False)
                retired += 1
        for concern in self.concerns.open(limit=10000):
            if concern.detail.get("parent_goal_id") == goal.id:
                self.concerns.drop(concern.id, note="goal closed", now=now)
                retired += 1
        return retired

    # -- forming intentions ------------------------------------------------------------------

    async def _act(self, now: datetime) -> tuple[List[Dict[str, Any]], int]:
        """The top concerns, ranked on the effective score, become intentions through authority.

        The top ``BROADCAST`` are taken after passing over what cannot compete: a concern with no
        template, and one whose key was reported already. An obligation that ended unresolved
        (a no, a lapsed ask, a failed run) is raised again every tick while its source stays open;
        it must not hold a slot, or three of them starve every other concern."""
        pairs = []
        # The owner's turns capture has not landed yet may settle what optional work is for (an interest they
        # say is answered): until they land, optional work waits (``affect.postponable``); what is owed does not.
        unfinished = getattr(self.capture, "unfinished", None)
        try:
            owed = int(unfinished(self.owner_id)) if callable(unfinished) and self.owner_id else 0
        except Exception as error:
            logger.debug("capture backlog unreadable (%s)", type(error).__name__)
            owed = 0
        waiting: List[str] = []
        for concern in self.concerns.open(limit=200):
            if len(pairs) >= BROADCAST:
                break
            if str(concern.dedup_key).startswith(QUESTION_KEY):
                # A question an earlier build raised and had not formed yet: the mind asks nothing, so it ends here.
                self.concerns.resolve(concern.id, note="resolved: the mind keeps the newer statement itself", now=now)
                continue
            if concern.exhausted or not concern.detail.get("type"):
                continue
            candidate = Candidate.from_detail(concern.detail)
            if self._formed(concern.dedup_key, candidate):
                # The obligation was reported once already (architecture 3.3); it does not compete again.
                self.concerns.drop(concern.id, note="already intended", now=now)
                continue
            pairs.append((concern, candidate))
        by_key = {candidate.dedup_key: concern for concern, candidate in pairs}
        weights = self.effective_weights()
        view = self.feelings.view()
        # The configured weight is factored out of the threshold; satiation is not, for self-chosen work (rank.py).
        learned = self.learned.at(now, {"drives": weights, "base": self.drive_weights,
                                       "threshold": self.act_threshold, "affect": view,
                                       "last_owner_turn": self._owner_last_turn()})
        ranked = eligible([candidate for _, candidate in pairs], threshold=self.act_threshold, drives=weights,
                          base=self.drive_weights, feedback=learned.feedback(self.feedback), affect=view, learned=learned)
        ranking = learned.ranking(ranked)
        acted = []
        formed: List[Dict[str, Any]] = []
        reached_out = False
        for candidate, score in ranked:
            concern = by_key.get(candidate.dedup_key)
            if concern is None:
                continue
            if candidate.type in OUTREACH_TYPES and reached_out:
                continue    # one unprompted interruption a tick: the next waits for the interruption to fade
            if owed and postponable(candidate):
                waiting.append(candidate.title)
                continue
            may_adopt = (self.goals.may_adopt() and candidate.drive in {"curiosity", "mastery"}
                         and not candidate.parent_goal_id and not self.goals.taken(candidate))
            # A research topic that already finished in an earlier week becomes a goal when one may be adopted.
            recurring = may_adopt and self.goals.recurring(candidate)
            steps_done: List[str] = []
            if candidate.parent_goal_id:
                goal = self.store.get(candidate.parent_goal_id)
                if goal is None or goal.status != "approved":
                    self.concerns.drop(concern.id, note="goal not open", now=now)
                    continue
                steps_done = self.goals.summaries(goal)
            failing = self.feelings.failing(candidate.topic or concern.summary)
            pitfalls = list(failing.pitfalls) if failing else []
            lesson_lines, lesson_ids = await self._lessons_for(candidate) if candidate.kind == "task" else ([], [])
            candidate.lesson_ids = lesson_ids
            shaped = await self.deliberation.form(concern, candidate, open_goals=len(self.goals.open()),
                                                  may_adopt_goal=may_adopt, steps_done=steps_done,
                                                  lessons=[*lesson_lines, *[p for p in pitfalls if p not in lesson_lines]][:2],
                                                  failing=failing, tried=view.tried if view is not None else (),
                                                  recurring=recurring)
            if shaped.open_ended and not shaped.text:
                continue  # the tick's one call is spent; the concern waits for the next tick
            if shaped.kind == "goal":
                goal_candidate = self.goals.candidate(shaped)
                if goal_candidate is None:
                    shaped.kind, shaped.goal = "task", None  # the budget is full or the topic has its goal: one task
                else:
                    shaped = goal_candidate
            # A failure-class investigation asks for lesson operations (the reflector); a goal keeps M4's path.
            try:
                self.lessons.reflector(shaped)
            except Exception as error:
                logger.warning("reflector request not added (%s)", type(error).__name__)
            explanation = learned.explain(candidate)
            if explanation:
                shaped.extra = {**(shaped.extra or {}), "learned": explanation}
            row = await self._form(shaped, score, now)
            if row is None:
                # A thought was spent and formed nothing: charge it (anti-rumination bounds the retries)
                # and keep the tokens of the call on the books even without an intention row.
                self.concerns.progress(concern.id, progressed=False, note="not formed", now=now)
                self._charge_unformed(shaped, concern, now)
                continue
            self.concerns.intended(concern.id, row.id, now=now)
            if row.decision == "act":
                acted.append((candidate, row.id))
            if row.type in OUTREACH_TYPES:
                reached_out = True
            if row.type in {*OUTREACH_MESSAGES, OUTREACH_FOLLOWUP}:
                self._outreach_formed(row, now)
                from colonymind.plasticity.learners.reply_timing import sent
                sent(self, row)
            if row.kind == "goal":
                self.autobiography.record(row.id, "goal_adopted",
                                          f"I adopted a goal: {row.description} ({row.drive} drive), to be met by "
                                          f"{(row.due_at or row.expires_at).date().isoformat() if (row.due_at or row.expires_at) else 'its horizon'}.")
            formed.append({"id": row.id, "type": row.type, "kind": row.kind, "drive": row.drive,
                           "decision": row.decision, "status": row.status, "score": round(score, 3),
                           "concern": concern.id})
        learned.observe(ranking, acted=acted)
        if waiting:
            logger.info("optional work waits for %d owner turn(s) capture has not landed: %s", owed,
                        "; ".join(waiting))
        return formed, len(pairs) - len(ranked)

    def _charge_unformed(self, candidate: Candidate, concern: Concern, now: datetime) -> None:
        """The tokens of a deliberation call that formed no intention, as an audit note (the daily
        token budget reads ``cost_tokens`` off the rows)."""
        if not candidate.cost_tokens:
            return
        row, created = self.store.create_intention(
            kind="note", type="deliberation", title=f"thought about: {concern.summary}"[:160], drive=candidate.drive,
            cls="internal", decision="act", decision_reason="formed no intention", status="done", dedup_key=None,
            hermes_kind="none", context={"concern": concern.summary, "evidence": list(candidate.evidence)},
            source_type="concern", source_id=concern.id, created_at=now)
        self.store.update(row.id, cost_tokens=int(candidate.cost_tokens))
        self.store.transition(row.id, "done", action="deliberated", outcome="done", verified="none",
                              completed_at=now, at=now)

    def _is_owner(self, recipient: Any) -> bool:
        return bool(self.owner_id) and recipient == self.owner_id

    async def _contact_record(self, contact_id: str | None, *, strict: bool = False) -> Optional[Dict[str, Any]]:
        """The contact as a dict, or None when there is no store, no such contact or the read failed; with
        ``strict`` (a read that decides whether something may act) a failed read raises ``Unreadable``."""
        if not contact_id or self.contacts is None:
            return None
        try:
            contact = await self.contacts.get(contact_id)
        except Exception as error:
            if strict:
                raise Unreadable(f"contact {contact_id} unreadable ({type(error).__name__})") from error
            logger.debug("contact %s unavailable (%s)", contact_id, type(error).__name__)
            return None
        if contact is None:
            return None
        if hasattr(contact, "to_dict"):
            return dict(contact.to_dict())
        return dict(contact) if isinstance(contact, Mapping) else None

    async def _contact_name(self, contact_id: str | None) -> str:
        record = await self._contact_record(contact_id) or {}
        return str(record.get("display_name") or record.get("given_name") or contact_id or "them")

    async def _may_contact(self, recipient: str | None) -> str:
        if not recipient:
            return "ask"
        if self._is_owner(recipient):
            return "auto"
        record = await self._contact_record(recipient, strict=True)
        stored = may_contact_of(record if record is not None else recipient, owner_id=self.owner_id)
        # A revoked standing allow whose reset of this permission has not landed yet: the owner took the "auto"
        # back, so it reads as what the reset writes until a tick's retry lands it (``_retry_allow_resets``).
        resetting = self.allows.resetting_for(recipient) if stored == "auto" else None
        return (resetting.before if resetting.before in {"never", "ask"} else "ask") if resetting else stored

    def _granted(self, grant: Any, stored: str, recipient: str | None) -> str:
        """A per-commitment owner grant counts as ``auto`` for that recipient only; a ``never``
        is never overridden (architecture 7.4: only the owner raises, an opt-out only lowers)."""
        if grant == "owner" and stored != "never" and recipient and not self._is_owner(recipient):
            return "auto"
        return stored

    @staticmethod
    def _owner_word(verdict: Verdict, *, type: str | None, ask_owner: bool) -> Verdict:
        """An owner question, or a message to someone the owner named only by a name the store
        matched, waits for the owner's word whatever would have let it act or merely wait: the rule
        is applied when the row forms and whenever a deferred row is re-decided."""
        if verdict.decision in {"act", "defer"} and (type in OWNER_QUESTIONS or ask_owner):
            return Verdict(decision="ask", reason="only the owner's word settles this", cls=verdict.cls)
        return verdict

    @staticmethod
    def _never_reason(recipient: str) -> str:
        return f"{recipient} may no longer be contacted (may_contact=never)"

    async def _permission_withdrawn(self, row: StoredInitiative, now: datetime) -> bool:
        """Permission is read again whenever a message to a contact may still leave (architecture
        7.4: an opt-out or the owner's revocation holds from that moment, however long the message
        waited for quiet hours or the body). A recipient now ``never`` cancels it; one lowered to
        ``ask`` turns a message approved on ``auto`` into the owner's question. A message the owner
        approved by its ask code needs no more than ``ask``. True when the row was withdrawn."""
        if row.kind != "message" or not row.entity_id or self._is_owner(row.entity_id):
            return False
        context = row.context if isinstance(row.context, dict) else {}
        permission = self._granted(context.get("grant"), await self._may_contact(row.entity_id), row.entity_id)
        if permission == "never":
            self._cancel_stale(row, self._never_reason(row.entity_id))
            return True
        if permission == "ask" and row.status == "approved" and not row.ask_code:
            self._apply_decision(row, Verdict(decision="ask", reason=f"may_contact is now ask for {row.entity_id}",
                                              cls=row.cls or "contact"), now, reconsidered=True)
            return True
        return False

    async def _handles(self, recipient: str | None) -> List[Dict[str, Any]]:
        if not recipient or self.contacts is None or not hasattr(self.contacts, "get_handles"):
            return []
        try:
            handles = await self.contacts.get_handles(recipient)
        except Exception:
            return []
        rendered = [{"gateway": h.gateway, "address": h.address, "is_primary": bool(h.is_primary),
                     "verified": bool(h.verified)} for h in handles or []]
        rendered.sort(key=lambda h: (not h["is_primary"], not h["verified"]))
        return rendered[:5]

    def _formed(self, key: str, candidate: Candidate) -> bool:
        """An intention exists under the candidate's key, or under the key an earlier release gave it."""
        return any(self.store.get_by_dedup_key(each) is not None
                   for each in (key, *drive_functions.former_keys(candidate)))

    async def _form(self, candidate: Candidate, score: float, now: datetime) -> Optional[StoredInitiative]:
        if self._formed(candidate.dedup_key, candidate):
            return None
        to_contact = candidate.kind == "message" and bool(candidate.recipient) and not self._is_owner(candidate.recipient)
        try:
            stored = await self._may_contact(candidate.recipient)
        except Unreadable as error:
            logger.warning("%s not formed on this tick: %s", candidate.dedup_key, error)
            return None
        may_contact = self._granted(candidate.grant, stored, candidate.recipient)
        # Composition is a model call: made only for a message the budgets would let go now. One
        # they defer is composed when it goes (``_reconsider``), from the packet of that day.
        if (to_contact and not candidate.text and may_contact != "never" and self.enabled
                and self.authority.budget_check(kind=candidate.kind, recipient=candidate.recipient,
                                                type=candidate.type, now=now,
                                                cooldown_hours=candidate.cooldown_hours) is None):
            await self._compose(candidate)
        # A message is classed by its recipient; past ``mind.floor.bulk_recipients`` people reached from one
        # concern or source in 24 h, the next one is bulk (the floor) unless a standing grant covers it.
        reached = self._concern_reach(candidate, now) if to_contact else 0
        verdict = self.authority.decide(kind=candidate.kind, recipient=candidate.recipient,
                                        text=f"{candidate.title}\n{candidate.text}", type=candidate.type,
                                        may_contact=may_contact, toolsets=self.policy.worker_toolsets, now=now,
                                        cooldown_hours=candidate.cooldown_hours, reached=reached)
        verdict = self._owner_word(verdict, type=candidate.type, ask_owner=candidate.ask_owner)
        if candidate.affect_ask and verdict.decision == "act" and self.feelings.active:
            # The strategy switch asks the owner instead of acting; it never grants what authority withheld.
            verdict = dataclasses.replace(verdict, decision="ask",
                                          reason=f"{verdict.reason}; {candidate.affect_ask}"[:300])
        asks = verdict.decision == "ask"
        status = {"act": "approved", "ask": "asked", "drop": "dropped", "defer": "proposed"}[verdict.decision]
        context: Dict[str, Any] = {
            "concern": candidate.concern, "evidence": list(candidate.evidence), "score": round(score, 3),
            "may_contact": may_contact, "notice": verdict.notice,
        }
        if candidate.topic:
            context["topic"] = candidate.topic
        if candidate.affect_ask:
            context["owner_question"] = candidate.affect_ask    # what the ask shows the owner beside its title
        if candidate.extra:
            context.update(candidate.extra)
        if candidate.ask_owner:
            context["ask_owner"] = True     # re-decided later (a deferral), it is still the owner's word
        for key in ("grant", "purpose", "cooldown_hours"):
            if getattr(candidate, key) is not None:
                context[key] = getattr(candidate, key)
        if candidate.kind == "message":
            context["text"] = candidate.text
            context["recipient_handles"] = await self._handles(candidate.recipient)
        elif candidate.kind == "goal":
            context.update(candidate.goal or {})
            context["description"] = candidate.text
        else:
            context["body"] = candidate.text
            # The deliberated plan before any guidance is appended (affect's note, a recorded view): the
            # identical-plan refusal hashes this, so guidance never makes a repeat look new (map X13).
            context["plan_body"] = candidate.text
            if candidate.kind == "task" and self.opinions is not None:
                lines, ids = self.opinions.task_lines(candidate)
                if lines:
                    context["body"] = f"{context['body']}\n\n{lines}".strip()
                    context["opinion_ids"] = ids
            if candidate.kind == "task":
                # How the lessons matched is kept below beside the uses it describes, never on its own (a reflector).
                context.pop("lesson_match", None)
                lesson_lines, lesson_ids = await self._task_lessons(candidate)
                if lesson_ids:
                    from .lessons import TASK_HEADING
                    context["body"] = (f"{context['body']}\n\n{TASK_HEADING}\n"
                                       + "\n".join(lesson_lines)).strip()
                    # A reflector investigates its class's lessons and does not apply them: its outcome
                    # (a timed-out investigation, say) is no evidence for or against them, so no use.
                    if not candidate.reflector:
                        context["lesson_ids"] = lesson_ids
                        # How each matched (``lessons.task_lessons``), for the uses' split by match kind.
                        kinds = (candidate.extra or {}).get("lesson_match") or {}
                        context["lesson_match"] = {ident: kinds[ident] for ident in lesson_ids if ident in kinds}
            if candidate.reflector:
                reflector = dict(candidate.reflector)
                request = str(reflector.pop("request", "") or "")
                if request:
                    context["body"] = f"{context['body']}\n\n{request}".strip()
                context["reflector"] = reflector
            if candidate.kind == "task" and context["body"]:
                # The line a report-read done-check looks for closes the body, unless the body names it already.
                context["body"] += report_line(candidate.success_check, context["body"])
            # The budget of the task's kind (``mind.budgets.task_types``); deliberation told the proposer.
            context["max_runtime_seconds"], context["max_retries"] = self.policy.budgets.for_task(candidate.type)
            if candidate.parent_goal_id:
                context["goal_mode"] = True
                context["goal_max_turns"] = DEFAULT_MAX_TURNS
        horizon = candidate.due_at if candidate.kind == "goal" else None
        if candidate.type in OUTREACH_MESSAGES and (candidate.extra or {}).get("expires_hours"):
            # A finding is news for half a day, an offer for a day; unsent past that, it goes to the digest.
            horizon = now + timedelta(hours=float(candidate.extra["expires_hours"]))
        expires_at = ask_expiry(now, self.policy) if asks else (horizon or now + TASK_WINDOW)
        if (candidate.extra or {}).get("owner_cadence"):
            expires_at = _utc(candidate.extra["expires_at"])
        code = new_ask_code(self.store.open_ask_codes()) if asks else None
        row, created = self.store.create_intention(
            kind=candidate.kind, type=candidate.type, title=candidate.title, drive=candidate.drive, cls=verdict.cls,
            decision=verdict.decision, decision_reason=verdict.reason, status=status, dedup_key=candidate.dedup_key,
            rationale=candidate.rationale, recipient=candidate.recipient, context=context,
            priority=candidate.priority, expires_at=expires_at, due_at=_utc(candidate.due_at), ask_code=code,
            invalidates_if=candidate.invalidates_if, success_check=candidate.success_check,
            hermes_kind="none", source_type=candidate.source_type, source_id=candidate.source_id, created_at=now)
        if created != "created":
            return None
        extra: Dict[str, Any] = {}
        if context.get("lesson_ids"):
            extra["lesson_ids"] = list(context["lesson_ids"])
        if candidate.parent_goal_id:
            extra["parent_goal_id"] = candidate.parent_goal_id
        if candidate.dedup_base:
            extra["dedup_base"] = candidate.dedup_base
        if candidate.cost_tokens:
            extra["cost_tokens"] = int(candidate.cost_tokens)
        if extra:
            self.store.update(row.id, **extra)
        updated = self._apply_decision(row, verdict, now, reconsidered=False, code=code)
        if verdict.grant and verdict.decision == "act" and to_contact and reached >= self.policy.bulk_recipients:
            self.authority.record_grant_use(verdict.grant, Effect(
                "bulk_third_party_messaging", "outbox", target=str(candidate.recipient),
                recipients=(str(candidate.recipient),)), task=row.id)
        if candidate.grant == "owner" and stored == "never" and verdict.decision == "drop":
            await self._refuse_grant(candidate)
        return updated

    async def _lessons_for(self, candidate: Candidate) -> tuple[List[str], List[str]]:
        """``lessons.task_lessons`` (the task's work embedded here, off the owner's path); lessons are guidance, so a
        store that cannot be read gives none. How each matched goes with the candidate into its row's context
        (``lesson_match``), for the uses' split by match kind."""
        try:
            lines, ids, kinds = await self.lessons.task_lessons(candidate)
        except Exception as error:
            logger.warning("lessons unavailable for a task (%s)", type(error).__name__)
            return [], []
        if kinds:
            candidate.extra = {**(candidate.extra or {}), "lesson_match": dict(kinds)}
        return lines, ids

    async def _task_lessons(self, candidate: Candidate) -> tuple[List[str], List[str]]:
        """The lessons a task body carries: the ones deliberation was given (``candidate.lesson_ids``),
        or the lessons for the task when it did not pass through ``_act``; nothing with lessons off."""
        if not self.lessons.enabled:
            return [], []
        if not candidate.lesson_ids:
            return await self._lessons_for(candidate)
        try:
            chosen = [lesson for lesson in (self.lessons.get(ident) for ident in candidate.lesson_ids)
                      if lesson is not None and lesson.status in {"active", "candidate"}]
        except Exception as error:
            logger.warning("lessons unavailable for a task body (%s)", type(error).__name__)
            return [], []
        return [lesson.line() for lesson in chosen], [lesson.id for lesson in chosen]

    @staticmethod
    def _purpose(purpose: str | None) -> str:
        from .compose import purpose_kind
        try:
            purpose_kind(purpose or "")
            return str(purpose)
        except ValueError:
            return "check_in"

    async def _compose(self, candidate: Candidate) -> None:
        """The text of a message to a contact, from the purpose, the name, the topic and that
        contact's own packet (architecture 6.3); never the concern, rationale, evidence or an
        owner turn. Authority then checks the text like any other."""
        text, tokens = await self._composed(candidate.recipient, candidate.purpose, candidate.topic)
        candidate.text = text
        candidate.cost_tokens = int(candidate.cost_tokens or 0) + int(tokens or 0)

    async def _composed(self, recipient: Any, purpose: str | None, topic: str | None) -> tuple[str, int]:
        name = await self._contact_name(recipient)
        packet = ""
        if self.packet_for is not None:
            try:
                packet = str(await self.packet_for(str(recipient)) or "")
            except Exception as error:
                logger.warning("recipient packet unavailable (%s); composing without it", type(error).__name__)
        return await self.composer.compose(purpose=self._purpose(purpose), recipient_name=name, packet=packet,
                                           topic=topic or "")

    async def _refuse_grant(self, candidate: Candidate) -> None:
        """The owner asked for a message to someone who may never be contacted: nothing went to
        them, and the owner hears why (the grant does not override ``never``)."""
        name = await self._contact_name(candidate.recipient)
        self.outbox.notice(
            type="grant_refused", title=f"Not sent to {name}"[:160],
            text=(f"Not sent to {name}: they are set to never be contacted, and your request does not change "
                  f"that. You asked for: {candidate.title}."),
            dedup_key=f"grant_refused:{candidate.dedup_key}")

    def _apply_decision(self, row: StoredInitiative, verdict: Any, now: datetime, *, reconsidered: bool,
                        code: str | None = None) -> Optional[StoredInitiative]:
        decision = verdict.decision
        if decision == "act":
            window = (row.due_at or row.expires_at) if row.kind == "goal" else None
            if row.type in OUTREACH_MESSAGES and row.expires_at is not None and not reconsidered:
                window = row.expires_at
            # Work a standing allow let act names it: its unbounded calls run while that allow stands.
            allowed = getattr(verdict, "allow", None)
            extra = {"context": {**(row.context if isinstance(row.context, dict) else {}), "allow": allowed}} \
                if allowed else {}
            expires_at = window or now + TASK_WINDOW
            expires_at = _cadence_expiry(row, expires_at)
            updated = self.store.transition(row.id, "approved", action="queued", at=now, decision="act",
                                            decision_reason=verdict.reason, cls=verdict.cls,
                                            expires_at=expires_at, **extra)
            self._allow_used(allowed or self._contact_allow(row, verdict.cls), now)
        elif decision == "ask":
            code = code or row.ask_code or new_ask_code(self.store.open_ask_codes())
            # The words asked about are the words sent: fixed now, hidden values hidden in both (audit).
            frozen = audit.frozen_for_ask(row, owner_id=self.owner_id)
            # Where a standing permission exists for its class, the ask offers "always" (``allows.offer_for``).
            offer = allow_rules.offer_for(row, cls=verdict.cls, level=self.level, owner_id=self.owner_id,
                                          toolsets=self.policy.worker_toolsets, executors=self.policy.executors)
            context = frozen
            base = frozen if frozen is not None else (row.context if isinstance(row.context, dict) else {})
            if offer is not None or "always" in base:
                # An offer is the one this decision made: a row asked again at another level keeps none from before.
                context = {key: value for key, value in base.items() if key != "always"}
                context.update({"always": offer.as_dict()} if offer is not None else {})
            expires_at = ask_expiry(now, self.policy)
            expires_at = _cadence_expiry(row, expires_at)
            updated = self.store.transition(row.id, "asked", action="asked", at=now, decision="ask",
                                            decision_reason=verdict.reason, cls=verdict.cls, ask_code=code,
                                            expires_at=expires_at,
                                            details={"code": code, "notice": verdict.notice},
                                            **({"context": context} if context is not None else {}))
        elif decision == "drop":
            updated = self.store.transition(row.id, "dropped", action="dropped", at=now, decision="drop",
                                            decision_reason=verdict.reason, cls=verdict.cls, outcome="denied",
                                            verified="none", cancelled_at=now, cancelled_reason=verdict.reason)
        else:
            updated = self.store.transition(row.id, "proposed", action="deferred", at=now, decision="defer",
                                            decision_reason=verdict.reason, cls=verdict.cls)
        if updated is None:
            return None
        if decision in {"act", "ask"} and not reconsidered and updated.kind != "note":
            self._register_expectation(updated, now)
        verb = {"act": "will act on", "ask": "asked the owner about", "drop": "dropped", "defer": "deferred"}[decision]
        self.autobiography.record(updated.id, f"decided_{decision}",
                                  f"I {verb} '{Autobiography.name(updated)}' ({updated.drive} drive, {updated.cls} class): "
                                  f"{updated.decision_reason}." + (f" Ask code {code}." if decision == "ask" else ""),
                                  decision=decision)
        return updated

    def _register_expectation(self, row: StoredInitiative, now: datetime) -> None:
        if self.expectations is None or not hasattr(self.expectations, "store"):
            return
        horizon = row.expires_at or (now + TASK_WINDOW)
        try:
            prediction = self.expectations.store.create(
                subject=f"intention:{row.id}", domain="intention",
                expectation=f"'{row.description}' completes with outcome done by {horizon.isoformat()}",
                confidence=0.7, horizon=horizon.timestamp(), source="mind", dedup_key=f"intention:{row.id}",
                detail={"intention_id": row.id, "kind": row.kind, "type": row.type})
        except Exception as error:
            logger.warning("expectation not registered for %s (%s)", row.id, type(error).__name__)
            return
        if prediction is not None:
            self.store.update(row.id, expectation_id=prediction.prediction_id)

    def _on_settled(self, row: StoredInitiative, outcome: str, check_result: Optional[bool]) -> None:
        """An outcome settles its concern and satiates its drive (architecture 3.1, 4.5); a finished
        commitment intention also closes the commitment it was raised for."""
        if self.opinions is not None:
            try:
                self.opinions.observe_outcome(row, outcome, check_result)
            except Exception as error:
                logger.warning("approach opinion not updated for %s (%s)", row.id, type(error).__name__)
        concern = self.concerns.by_intention(row.id)
        now = self.clock()
        if outcome == "done" and isinstance(row.context, dict) and row.context.get("reflector"):
            try:
                self.lessons.apply_reflection(row, summary=row.result or "",
                                              result=(row.result_metadata or {}).get("result"), now=now)
            except Exception as error:
                logger.warning("reflector report not applied for %s (%s)", row.id, type(error).__name__)
        if self.faculties["outreach"] and self.owner_id:
            self._outreach_settled(row, outcome, now)
        if outcome in {"done", "failed"}:
            self._report_task(row, outcome)
        if outcome == "done":
            self._close_commitment(row)
            if concern is not None:
                self.concerns.resolve(concern.id, note=f"{row.kind} done", now=now)
            if self.faculties["drives"] and row.drive in DRIVES and row.type not in audit.NOTICE_TYPES:
                self.mind_state.bump(f"satiety.{row.drive}", 1.0, half_life_s=SATIETY_HALF_LIFE_S,
                                     causes=[f"intention:{row.id}"], now=now)
        elif outcome == "failed":
            if concern is not None:
                self.concerns.progress(concern.id, progressed=False, note=str(row.failed_reason or "failed"), now=now)
        elif concern is not None:
            self.concerns.drop(concern.id, note=outcome, now=now)

    def _outreach_settled(self, row: StoredInitiative, outcome: str, now: datetime) -> None:
        """A research-shaped task that reported a finding waits for the owner branch (``pending``); a
        finding's outreach that expired unsent, or that the owner's pause cancelled before it went, hands
        the finding to the digest; one cancelled by a mute of its topic is ``muted``. An answer the owner
        asked for that ended unsent is owed still: its finding waits for the owner branch again (the answer's
        key given back, so it forms anew) up to ``ANSWER_TRIES`` answers in a row, then goes to the digest; a
        digest that is never delivered hands it back to pending (``_reconcile_digested``), so it is never lost."""
        o = outreach_functions
        context = row.context if isinstance(row.context, dict) else {}
        if (outcome == "done" and row.type in SHARED_FINDINGS and str(row.result or "").strip()
                and not context.get("reflector")):
            self._mark_finding(row.id, "pending", now)
        elif row.type in {"outreach_finding", OUTREACH_ANSWER} and outcome in {"expired", "cancelled"}:
            source = str(context.get("source_ref") or "")
            if not source.startswith("intention:"):
                return
            topic = str(context.get("topic") or "")
            muted = bool(topic) and any(
                float(item.get("level") or 0.0) >= o.MUTE_FLOOR
                and (item["key"] == f"{o.MUTE_PREFIX}{slug(topic)}" or o.similar(str(item.get("text") or ""), topic))
                for item in self.mind_state.items(o.MUTE_PREFIX))
            finding = source[len("intention:"):]
            if muted or row.type == "outreach_finding":
                self._mark_finding(finding, "muted" if muted else "digest", now)
                return
            found = self.store.get(finding)
            tries = int((((found.result_metadata or {}) if found is not None else {}).get("outreach") or {})
                        .get("answer_tries") or 0) + 1
            # The key is given back whichever way the finding goes: one the digest hands back (unsent) forms anew.
            self.store.update(row.id, dedup_key=None)
            self._mark_finding(finding, "pending" if tries < ANSWER_TRIES else "digest", now, answer_tries=tries)

    def _on_blocked(self, row: StoredInitiative) -> None:
        """A blocked task stays open; the person its obligation is owed to hears that it is stuck, once."""
        self._report_task(row, "blocked")

    def _report_task(self, row: StoredInitiative, outcome: str) -> None:
        """The one word a task formed for an owed item gets: what became of it, to the person it is owed to.

        A worker has no way to message anyone, so without this its result reached no one (a done
        task closed the commitment silently; a failed one left it open and unsaid). The owner gets
        it as a notice; a contact gets it only through authority with the owner's word on the exact
        text (``ask_owner``), since the worker's report may carry what only the owner may see. Either word
        is decided by authority like every other (``_tell_owner`` for the owner, ``_form`` for a contact)."""
        if (row.kind != "task" or row.type not in REPORTED_TASK_TYPES or row.source_type != "commitment"
                or not row.source_id or outcome not in {"done", "failed", "blocked"}):
            return
        key = f"{TASK_OUTCOME}:{row.id}:{outcome}"
        if self.store.get_by_dedup_key(key) is not None:
            return
        record: Dict[str, Any] = {}
        try:
            record = dict(self.commitments.get(str(row.source_id)) or {}) if self.commitments is not None else {}
        except Exception as error:
            logger.debug("commitment %s unavailable for the report (%s)", row.source_id, type(error).__name__)
        person = str(record.get("person_id") or row.entity_id or self.owner_id or "")
        description = str(record.get("description") or row.description or "").strip()
        head = {"done": f"Done: {description}.", "failed": f"I could not finish: {description}.",
                "blocked": f"I am stuck on: {description}."}[outcome]
        detail = " ".join(str(row.result or row.failed_reason or "").split())
        if len(detail) > TASK_OUTCOME_CHARS:
            detail = detail[: TASK_OUTCOME_CHARS - 1].rstrip() + "…"
        text = f"{head} {detail}".strip()
        title = f"{outcome}: {description}"[:160]
        owner = not person or self._is_owner(person)
        candidate = Candidate(
            type=TASK_OUTCOME, drive="duty", kind="message", title=title, dedup_key=key, salience=0.9, cost=0.05,
            recipient=self.owner_id if owner else person, text=text, rationale="a task for an obligation to them finished",
            evidence=[f"intention:{row.id}", f"commitment:{row.source_id}"], concern=f"report: {description}"[:160],
            source_type="commitment", source_id=str(row.source_id), ask_owner=not owner, concern_kind="obligation")
        if owner:
            self._tell_owner(candidate)
            return
        self.concerns.bump(drive="duty", kind="obligation", summary=candidate.concern, dedup_key=key,
                           salience=candidate.salience, sources=candidate.evidence, detail=candidate.as_detail(),
                           now=self.clock())

    def _tell_owner(self, candidate: Candidate) -> Optional[StoredInitiative]:
        """A word to the owner formed now (from an outcome hook, which cannot wait for a tick), decided by
        authority as ``_form`` decides every intention: the off switch, the level, the floor, the deny list, the
        breaker and the budgets. An act is queued, a deferral waits for ``_reconsider``, an ask for the owner's
        word, and a drop is recorded; none of it bypasses the outbox's own checks."""
        if not self.owner_id or self.store.get_by_dedup_key(candidate.dedup_key) is not None:
            return None
        now = self.clock()
        verdict = self.authority.decide(kind="message", recipient=self.owner_id,
                                        text=f"{candidate.title}\n{candidate.text}", type=candidate.type,
                                        may_contact="auto", toolsets=self.policy.worker_toolsets, now=now)
        if not self.enabled or self.level == "off":
            # Owed, not dropped: it waits (deferred) and ``_reconsider`` decides it once the mind is back on.
            verdict = Verdict(decision="defer", reason="the mind is off: the owed word waits until it is back on",
                              cls=verdict.cls)
        status = {"act": "approved", "ask": "asked", "drop": "dropped", "defer": "proposed"}[verdict.decision]
        code = new_ask_code(self.store.open_ask_codes()) if verdict.decision == "ask" else None
        row, created = self.store.create_intention(
            kind="message", type=candidate.type, title=candidate.title, drive=candidate.drive, cls=verdict.cls,
            decision=verdict.decision, decision_reason=verdict.reason, status=status, dedup_key=candidate.dedup_key,
            rationale=candidate.rationale, recipient=self.owner_id, priority=candidate.priority,
            context={"concern": candidate.concern, "evidence": list(candidate.evidence), "text": candidate.text,
                     "may_contact": "auto", "notice": verdict.notice},
            expires_at=ask_expiry(now, self.policy) if verdict.decision == "ask" else now + TASK_WINDOW,
            ask_code=code, hermes_kind="none", source_type=candidate.source_type, source_id=candidate.source_id,
            created_at=now)
        if created != "created":
            return None
        return self._apply_decision(row, verdict, now, reconsidered=False, code=code)

    def _close_commitment(self, row: StoredInitiative) -> None:
        """The body's report of a done intention is what settles its commitment. The dispatched
        worker cannot mark the row fulfilled itself (the plugin guard blocks that), so the
        resolution names the body and the intention, never the worker's own word."""
        check = row.success_check
        if isinstance(check, str):
            try:
                check = json.loads(check)
            except ValueError:
                return
        if not isinstance(check, dict) or check.get("kind") != "commitment_resolved" or not check.get("commitment_id"):
            return
        resolve = getattr(self.commitments, "resolve", None)
        if not callable(resolve):
            return
        ident = str(check["commitment_id"])
        try:
            current = self.commitments.get(ident)
            if not isinstance(current, dict) or current.get("status") not in {"pending", "overdue"}:
                return
            resolve(ident, "done", note=f"intention {row.id}: {row.result or 'done'}", resolved_by="body")
        except Exception as error:
            logger.warning("commitment %s not closed for %s (%s)", ident, row.id, type(error).__name__)

    def _on_sent(self, row: StoredInitiative) -> None:
        """The body reported a message sent: an owner-granted message to a third party was the
        obligation itself, so its commitment is done; a delivered digest lists what it carried; and a
        message to a contact is an exchange with them, logged in the comms ledger, where the social drive
        reads its sends."""
        if row.type in FULFILLED_BY_SENDING:
            self._close_commitment(row)
        if row.type == "digest":
            self._reconcile_digested(self.clock())
        if row.type == ASK_IN_FULL:
            self._full_ask_seen(row)
        self._log_sent(row)

    def _log_sent(self, row: StoredInitiative) -> None:
        if self.comms is None or not hasattr(self.comms, "log") or not row.entity_id or self._is_owner(row.entity_id):
            return
        target = next((str((item.details or {}).get("target") or "") for item in self.store.get_history(row.id, limit=50)
                       if item.action == "sending"), "")
        context = row.context if isinstance(row.context, dict) else {}
        try:
            self.comms.log(row.entity_id, channel=split_target(target)[0] or "mind", direction="out",
                           summary=str(context.get("text") or row.description or "")[:300],
                           external_ref=f"{MIND_REF}{row.type}:{row.id}",
                           ts=(_utc(row.completed_at) or self.clock()).isoformat())
        except Exception as error:
            logger.warning("send of %s not logged in the comms ledger (%s)", row.id, type(error).__name__)

    # -- people (architecture 4.7) -------------------------------------------------------------

    async def _resolve_recipients(self, commitments: List[Dict[str, Any]]) -> None:
        """The contact behind each third party the owner named for a granted message or a cadence,
        written back to the row (``metadata.recipient_id``) so it is resolved once, with
        ``recipient_exact``: whether the owner identified them exactly (an id, a handle, a number)
        or the store matched a name. Only an exact recipient carries the owner's grant; a name
        match becomes an owner ask showing ``recipient_match`` (the contact's name and a handle).
        An unknown name stays unresolved and the duty drive asks the owner who it is."""
        resolver = getattr(self.contacts, "resolve_reference", None)
        if not callable(resolver):
            return
        for row in commitments:
            name = drive_functions.unresolved_recipient(row, owner_id=self.owner_id)
            if name is None:
                continue
            contact, exact = None, True
            try:
                contact = await resolver(name, exact=True)
                if contact is None:
                    contact, exact = await resolver(name), False
            except Exception as error:
                logger.debug("recipient %r not resolved (%s)", name, type(error).__name__)
                continue
            contact_id = getattr(contact, "contact_id", None) if contact is not None else None
            if not contact_id:
                continue
            found = {"recipient_id": str(contact_id), "recipient_exact": exact}
            if not exact:
                # What the owner confirms: the person the name matched, by name and a handle.
                handles = await self._handles(str(contact_id))
                label = str(getattr(contact, "display_name", None) or contact_id)
                if handles:
                    label += f", {handles[0]['gateway']}:{handles[0]['address']}"
                found["recipient_match"] = label[:160]
            row["metadata"] = {**(row.get("metadata") or {}), **found}
            try:
                self.commitments.update(row["id"], metadata=found)
            except Exception as error:
                logger.warning("recipient of commitment %s not recorded (%s)", row.get("id"), type(error).__name__)

    async def _apply_cadences(self, commitments: List[Dict[str, Any]]) -> None:
        """A cadence the owner set in conversation becomes the contact's own, once per row (oldest
        row first, so the newest word wins): the row is marked applied, and a cadence the owner
        later sets by hand is not set back. The row stays open, so the check-ins keep its topic
        (``_thread_topic``); it never grants permission, which stays the contact's ``may_contact``.
        Nothing said in the turn sets one that checks in more often: for someone named exactly, a cadence
        that only slows the contact's own or keeps it applies here, written against the cadence it was
        decided on (a change meanwhile is decided again next tick), and one that is new or faster is the
        owner's one-tap ``cadence_confirm`` question, which ``answer`` settles on their own reply (the duty
        drive forms the same question for a name match). Once a row applies, its open question is retired.
        A question that expires unanswered leaves the row open and unapplied (the owner's obligation
        stays); the owner saying it again after that asks again (``_renew_cadence_question``)."""
        setter = getattr(self.contacts, "set_cadence", None)
        if not callable(setter):
            return
        from colonymind.contacts.changes import cadence_change
        from colonymind.contacts.store import ContactChanged
        for row in sorted(commitments, key=lambda item: str(item.get("created_at") or "")):
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            if metadata.get("kind") != drive_functions.CADENCE_KIND or metadata.get("cadence_applied") \
                    or not self.owner_id or str(row.get("person_id") or "") != self.owner_id:
                continue
            self._renew_cadence_question(row)
            cadence = drive_functions.owner_cadence(row, owner_id=self.owner_id)
            if cadence is None:
                continue                  # a name match: the duty drive asks the owner who they meant
            contact_id, minutes, _ = cadence
            if contact_id == self.owner_id:
                continue                  # the owner's cadence belongs to outreach, never to contact check-ins
            record = await self._contact_record(contact_id)
            if record is None:
                continue
            current = record.get("cadence_minutes")
            if cadence_change(current, minutes) == "raise":
                question = drive_functions.cadence_confirm_candidate(row, owner_id=self.owner_id, current=current)
                if question is not None:
                    try:
                        await self._form(question, question.salience, self.clock())
                    except Exception as error:
                        logger.warning("cadence question for commitment %s not formed (%s)", row.get("id"),
                                       type(error).__name__)
                continue
            try:
                if current != minutes:
                    await setter(contact_id, minutes, by=f"owner-turn:commitment:{row['id']}", expected=current)
            except ContactChanged:
                continue                  # the cadence moved while this was decided: decided again next tick
            except Exception as error:
                logger.warning("cadence of commitment %s not applied (%s)", row.get("id"), type(error).__name__)
                continue
            row["metadata"] = {**(row.get("metadata") or {}), "cadence_applied": True}
            try:
                self.commitments.update(row["id"], metadata={"cadence_applied": True})
            except Exception as error:
                logger.warning("cadence of commitment %s not marked (%s)", row.get("id"), type(error).__name__)
            waiting = self.store.get_by_dedup_key(drive_functions.cadence_question_key(row))
            if waiting is not None and waiting.status == "asked":
                self.outcomes.record(waiting.id, status="cancelled", by="mind", implicit_verdict=False,
                                     summary="the cadence only slowed or kept the contact's own: set without it")

    def _renew_cadence_question(self, row: Dict[str, Any]) -> None:
        """The next round of a cadence question: when the one asked expired unanswered (its time up, even before
        the sweep marks it) and the owner said the cadence again after that (capture marks the open row
        ``restated_at``), the row's ``confirm_round`` moves on, so the question forms again under a new key
        (``drives.cadence_question_key``) instead of meeting the lapsed one. Nothing is asked again unless they
        say it again: an unanswered question is not nagged."""
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        restated = _utc(metadata.get("restated_at"))
        if restated is None or self.commitments is None:
            return
        question = self.store.get_by_dedup_key(drive_functions.cadence_question_key(row))
        now = self.clock()
        if question is None or question.expires_at is None or restated < question.expires_at:
            return
        if question.status == "asked" and question.expires_at <= now:
            self.outcomes.record(question.id, status="expired", summary="no answer before the ask expired",
                                 by="mind", implicit_verdict=False)
        elif question.status != "expired":
            return
        round_ = int(metadata.get("confirm_round") or 0) + 1
        try:
            self.commitments.update(row["id"], metadata={"confirm_round": round_})
        except Exception as error:
            logger.warning("cadence question of commitment %s not renewed (%s)", row.get("id"), type(error).__name__)
            return
        row["metadata"] = {**metadata, "confirm_round": round_}

    async def _link_proposals(self) -> List[Dict[str, Any]]:
        lister = getattr(self.contacts, "list_handle_proposals", None)
        if not callable(lister):
            return []
        try:
            rows = await lister(limit=50)
        except Exception as error:
            logger.debug("link proposals unavailable (%s)", type(error).__name__)
            return []
        return [dict(row) for row in rows or [] if str(row.get("status") or "pending") == "pending"]

    @staticmethod
    def _sent_at(row: StoredInitiative) -> datetime:
        return _utc(row.completed_at) or _utc(row.created_at) or now_utc()

    def _messages_by_contact(self, now: datetime) -> Dict[str, List[StoredInitiative]]:
        """The mind's messages to each contact over the social history, newest first."""
        grouped: Dict[str, List[StoredInitiative]] = {}
        for row in self.store.intentions(kind=["message"], since=now - SOCIAL_HISTORY, limit=2000):
            if row.entity_id and not self._is_owner(row.entity_id):
                grouped.setdefault(row.entity_id, []).append(row)
        return grouped

    def _open_followups(self, contact_id: str) -> List[str]:
        if self.followups is None or not hasattr(self.followups, "list_for_context"):
            return []
        try:
            rows = self.followups.list_for_context(contact_id=contact_id, limit=10)
        except Exception as error:
            logger.debug("follow-ups of %s unavailable (%s)", contact_id, type(error).__name__)
            return []
        return [str(row.get("original_local_text") or row.get("commitment_id") or "")[:160] for row in rows or []
                if row.get("state") not in {"resolved", "cancelled", "expired"}]

    async def _affect_declining(self, contact_id: str) -> bool:
        if self.contact_affect is None or not hasattr(self.contact_affect, "trend"):
            return False
        try:
            trend = self.contact_affect.trend(contact_id)
            if inspect.isawaitable(trend):
                trend = await trend
        except Exception as error:
            logger.debug("affect of %s unavailable (%s)", contact_id, type(error).__name__)
            return False
        return bool((trend or {}).get("declining"))

    @staticmethod
    def _thread_topic(contact_id: str, commitments: List[Dict[str, Any]], sent: List[StoredInitiative],
                      owner_id: str | None) -> str:
        """The open thread a check-in is about, from what is already this contact's to hear: the
        matter of the cadence the owner set for them, else of a check-in the owner granted for them,
        else the topic the last message to them raised, else their own open item (their words).
        Never an owner concern or its evidence."""
        for row in commitments:
            cadence = drive_functions.owner_cadence(row, owner_id=owner_id)
            if cadence is not None and cadence[0] == contact_id and cadence[2]:
                return cadence[2]
        own = ""
        for row in commitments:
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            if (metadata.get("kind") == "check_in" and metadata.get("recipient_id") == contact_id
                    and drive_functions.granted_message(row, owner_id=owner_id) is not None
                    and str(metadata.get("topic") or "").strip()):
                return str(metadata["topic"]).strip()
            if not own and str(row.get("person_id") or "") == contact_id:
                own = " ".join(str(row.get("description") or "").split()[:8])
        for row in sent:
            topic = str((row.context or {}).get("topic") or "").strip() if isinstance(row.context, dict) else ""
            if topic:
                return topic
        return own

    def _sends(self, contact_id: str, mine: List[StoredInitiative]) -> List[tuple[str, datetime]]:
        """``(type, sent_at)`` of every message the mind sent this contact, newest first: the comms
        ledger (``CommsLog.mind_sends``: kept past retention, moved with a merge) together with the
        intention rows still held (a send the body left ``uncertain`` is counted as sent)."""
        seen: Dict[str, tuple[str, datetime]] = {}
        for item in mine:
            if item.status in {"sent", "uncertain"}:
                seen[item.id] = (str(item.type or ""), self._sent_at(item))
        reader = getattr(self.comms, "mind_sends", None)
        if callable(reader):
            try:
                logged = reader(contact_id, limit=20)
            except Exception as error:
                logger.warning("sends to %s unavailable from the comms ledger (%s)", contact_id, type(error).__name__)
                logged = []
            for entry in logged:
                at = _utc(entry.get("ts"))
                if at is not None and entry.get("intention_id") not in seen:
                    seen[str(entry["intention_id"])] = (str(entry.get("type") or ""), at)
        return sorted(seen.values(), key=lambda item: item[1], reverse=True)

    async def _social_rows(self, now: datetime, *,
                           commitments: List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
        """``contacts.social_candidates()`` enriched for the social drive: the last send, the
        ignored streak (newest first over sent check-ins, counting while the contact has not
        talked since), whether a message to them is still under way, open follow-ups, declining
        affect, the open thread, and the conversation-based cadence of a tier-only contact."""
        if not self.faculties["people"] or self.contacts is None or not hasattr(self.contacts, "social_candidates"):
            return []
        try:
            candidates = await self.contacts.social_candidates(limit=200)
        except Exception as error:
            logger.warning("social candidates unavailable (%s)", type(error).__name__)
            return []
        if commitments is None:
            commitments = self._open_commitments()
        history = self._messages_by_contact(now)
        rows: List[Dict[str, Any]] = []
        for raw in candidates or []:
            row = dict(raw)
            contact_id = str(row.get("contact_id") or "")
            if not contact_id or self._is_owner(contact_id):
                continue
            mine = history.get(contact_id, [])
            sent = self._sends(contact_id, mine)
            check_ins = [(kind, at) for kind, at in sent if kind in CHECK_IN_TYPES]
            last_in = _utc(row.get("last_interaction_at"))
            streak = 0
            for _, at in check_ins:
                if last_in is not None and last_in > at:
                    break
                streak += 1
            # A check-in that ended without going out (the owner let the ask expire or said no, the
            # deny list dropped it) starts the next period, so it is neither lost for good nor re-asked at once.
            unsent = [_utc(item.cancelled_at) or _utc(item.completed_at) or _utc(item.created_at) for item in mine
                      if item.type in CHECK_IN_TYPES and item.status in {"expired", "cancelled", "dropped"}]
            row.update(
                last_outbound_at=sent[0][1].isoformat() if sent else None,
                last_attempt_at=max(unsent).isoformat() if unsent else None,
                last_check_in_at=check_ins[0][1].isoformat() if check_ins else None,
                ignored_streak=streak,
                in_flight=any(item.status in MIND_ACTIVE_STATUSES for item in mine),
                open_followups=self._open_followups(contact_id),
                affect_declining=await self._affect_declining(contact_id),
                topic=self._thread_topic(contact_id, commitments,
                                         sorted((item for item in mine if item.status in {"sent", "uncertain"}),
                                                key=self._sent_at, reverse=True), self.owner_id))
            if row.get("cadence_minutes") is None:
                row["estimated_cadence_minutes"] = conversation_cadence_minutes(
                    first_seen_ts=row.get("first_seen_at"), last_interaction_ts=row.get("last_interaction_at"),
                    conversations=row.get("interaction_count"))
            rows.append(row)
        return rows

    async def _score_check_ins(self, now: datetime) -> Optional[Dict[str, int]]:
        """Replies and silence teach the timing (architecture 4.7 item 6): a sent check-in whose
        window passed (the contact's cadence, else 24 h) is ``actioned`` when the contact talked
        after it and ``ignored`` otherwise, on its type key and ``reach_out:<contact>``; a reply
        also satisfies the social drive. A late reply turns the newest ignored check-in to that
        contact into ``actioned``: a contact who went quiet and came back is not held below the
        threshold for good by the silence they broke."""
        if not self.faculties["people"] or self.contacts is None:
            return None
        scored = {"actioned": 0, "ignored": 0}
        newest: Dict[str, StoredInitiative] = {}
        rows = [row for row in self.store.intentions(status=["sent"], kind=["message"], since=now - SOCIAL_HISTORY,
                                                     limit=2000)
                if row.type in CHECK_IN_TYPES and row.entity_id and not self._is_owner(row.entity_id)]
        for row in sorted(rows, key=self._sent_at):
            newest[str(row.entity_id)] = row
        records: Dict[str, Optional[Dict[str, Any]]] = {}
        for row in rows:
            contact_id = str(row.entity_id)
            if row.verdict is not None and (row.verdict != "ignored" or newest.get(contact_id) is not row):
                continue
            if contact_id not in records:
                records[contact_id] = await self._contact_record(contact_id)
            record = records[contact_id]
            if record is None:
                continue
            sent_at = self._sent_at(row)
            last = _utc(record.get("last_interaction_at"))
            replied = last is not None and last > sent_at
            if row.verdict is None:
                try:
                    window = timedelta(minutes=float(record["cadence_minutes"])) if record.get("cadence_minutes") \
                        else CHECK_IN_WINDOW
                except (TypeError, ValueError):
                    window = CHECK_IN_WINDOW
                if now < sent_at + window:
                    continue
                verdict = "actioned" if replied else "ignored"
            elif row.verdict == "ignored" and replied and newest.get(contact_id) is row:
                verdict = "actioned"
            else:
                continue
            self.outcomes.rate(row.id, verdict, by="mind")
            scored[verdict] += 1
            if verdict == "actioned" and self.faculties["drives"]:
                self.mind_state.bump("satiety.social", 1.0, half_life_s=SATIETY_HALF_LIFE_S,
                                     causes=[f"intention:{row.id}"], now=now)
        return scored

    async def _replied_reason(self, row: StoredInitiative) -> Optional[str]:
        """``contact:<id>:replied`` holds once the contact talked after the intention formed."""
        condition = row.invalidates_if
        if not isinstance(condition, str) or not condition.startswith("contact:") or not condition.endswith(":replied"):
            return None
        contact_id = condition[len("contact:"):-len(":replied")]
        record = await self._contact_record(contact_id, strict=True)
        last = _utc((record or {}).get("last_interaction_at"))
        created = _utc(row.created_at)
        if last is not None and created is not None and last > created:
            return f"contact {contact_id} replied at {last.isoformat()}"
        return None

    async def _cancel_if_replied(self, row: StoredInitiative) -> bool:
        reason = await self._replied_reason(row)
        if reason is None:
            return False
        self._cancel_stale(row, reason)
        return True

    async def _invalidate_messages(self, now: datetime) -> int:
        """Every tick: a waiting message to a contact who replied since is cancelled, and one whose
        recipient's permission fell is withdrawn (``_permission_withdrawn``)."""
        count = 0
        for row in self.store.intentions(status=["proposed", "asked", "approved"], kind=["message"], limit=500):
            try:
                if await self._cancel_if_replied(row) or await self._permission_withdrawn(row, now):
                    count += 1
            except Exception as error:
                logger.warning("message %s held: %s", row.id, error if isinstance(error, Unreadable) else type(error).__name__)
        return count

    async def _digests(self, now: datetime) -> Optional[int]:
        """Refresh stale summaries; create a first template for recently seen contacts daily."""
        if not self._digest_writer():
            return None
        restored = await self._restore_digests(now)
        last = _utc((self.mind_state.get(PEOPLE_DIGESTS_KEY) or {}).get("updated_at"))
        if last is not None and not boundary_crossed(last, now, tz=self.tz, minute=0):
            return len(restored) or None
        self.mind_state.set(PEOPLE_DIGESTS_KEY, text=now.astimezone(self.tz).date().isoformat(), now=now)
        since, written, offset, page = now - timedelta(days=1), len(restored), 0, 200
        while offset < 10000:
            try:
                batch = await self.contacts.list(limit=page, offset=offset)
            except Exception as error:
                logger.warning("contacts unavailable for digests (%s)", type(error).__name__)
                break
            for contact in batch or []:
                record = contact.to_dict() if hasattr(contact, "to_dict") else dict(contact)
                contact_id = str(record.get("contact_id") or "")
                last = _utc(record.get("last_interaction_at"))
                if not contact_id or self._is_owner(contact_id) or last is None or last < since \
                        or contact_id in restored:
                    continue
                if record.get("digest") or record.get("digest_withheld_at") and record.get("digest_writer"):
                    continue
                try:
                    await self._write_digest(record, now)
                    written += 1
                except Exception as error:
                    logger.warning("digest of %s not written (%s)", contact_id, type(error).__name__)
            if len(batch or []) < page:
                break
            offset += page
        return written

    def _digest_writer(self) -> bool:
        return bool(self.faculties["people"]) and self.contacts is not None and hasattr(self.contacts, "set_digest")

    async def _digest_upkeep(self, now: datetime) -> Optional[int]:
        """Background store upkeep also runs with the mind off, without a model call."""
        if not self._digest_writer():
            return None
        return len(await self._restore_digests(now)) or None

    async def _restore_digests(self, now: datetime) -> List[str]:
        """Rebuild only stale contacts after a quiet period, within a rolling hourly budget."""
        from colonymind.contacts.digest import checked_digests
        from colonymind.contacts.refresh import (DEBOUNCE_SECONDS, REFRESHES_PER_HOUR, finish_refresh,
                                               invalidate_pending, refresh_candidates, reserve_refresh)
        ledger = self.ledger
        try:
            await invalidate_pending(ledger, self.contacts, owner_id=self.owner_id)
        except Exception as error:
            logger.warning("digest invalidation deferred (%s)", type(error).__name__)
            return []
        # Check one page of lineage per pass for expiration and attribution changes
        # that did not update a claim row. This scan never rewrites a valid digest.
        if ledger is not None and hasattr(self.contacts, "list"):
            offset = int((self.mind_state.get("people.digests.scan") or {}).get("text") or 0)
            try:
                batch = await self.contacts.list(limit=200, offset=offset)
                await checked_digests(batch or [], store=self.contacts, ledger=ledger)
                self.mind_state.set("people.digests.scan", text=str(offset + 200 if len(batch or []) == 200 else 0), now=now)
            except Exception as error:
                logger.warning("digest lineage scan deferred (%s)", type(error).__name__)
        reader = getattr(self.contacts, "withheld_digests", None)
        if not callable(reader):
            return []
        try:
            withheld = await refresh_candidates(ledger, self.contacts, owner_id=self.owner_id,
                                               now=now.timestamp(), limit=REFRESHES_PER_HOUR,
                                               quiet_seconds=DEBOUNCE_SECONDS)
        except Exception as error:
            logger.warning("withheld digests unavailable (%s)", type(error).__name__)
            return []
        written: List[str] = []
        for contact in withheld or []:
            record = contact.to_dict() if hasattr(contact, "to_dict") else dict(contact)
            contact_id = str(record.get("contact_id") or "")
            changed = _utc(record.get("digest_withheld_at"))
            if not contact_id or self._is_owner(contact_id) or changed is None \
                    or (now - changed).total_seconds() < DEBOUNCE_SECONDS:
                continue
            if not reserve_refresh(ledger, now.timestamp(), contact_id=contact_id):
                break
            try:
                await self._write_digest(record, now)
                finish_refresh(ledger, contact_id, now.timestamp())
                written.append(contact_id)
            except Exception as error:
                finish_refresh(ledger, contact_id, now.timestamp(), failed=True)
                logger.warning("digest of %s not rendered again (%s)", contact_id, type(error).__name__)
        # A concurrent fact change during rendering must leave another rebuild pending.
        try:
            await invalidate_pending(ledger, self.contacts, owner_id=self.owner_id)
        except Exception as error:
            logger.warning("digest invalidation deferred (%s)", type(error).__name__)
        return written

    async def _write_digest(self, record: Dict[str, Any], now: datetime) -> None:
        """The template digest, with the ids of the claims it renders as its lineage."""
        contact_id = str(record["contact_id"])
        claims: List[tuple[str, str]] = []
        if self.claims_for is not None:
            claims = [(str(ident), str(line)) for ident, line in await self.claims_for(contact_id) or []]
        counts: Dict[str, Any] = {}
        if self.comms is not None and hasattr(self.comms, "counts"):
            counts.update(self.comms.counts(contact_id))
        if self.commitments is not None and hasattr(self.commitments, "get_pending_for_person"):
            counts["open"] = len(self.commitments.get_pending_for_person(contact_id))
        record = {**record, "handles": await self._handles(contact_id)}
        text, cited = render_template(record, claims=claims, counts=counts,
                                      last_interaction_at=record.get("last_interaction_at"), now=now)
        await self.contacts.set_digest(contact_id, text, cited, writer=WRITER_TEMPLATE)

    # -- digest ----------------------------------------------------------------------------

    async def _opt_outs_since(self, since: datetime) -> List[str]:
        """``<name>: <their words>`` for every opt-out since ``since`` (architecture 7.4): an opt-out
        only lowers, and the owner hears of it in the next digest."""
        reader = getattr(self.contacts, "audit_since", None)
        if not callable(reader):
            return []
        try:
            rows = await reader(["opt_out"], since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        except Exception as error:
            logger.warning("opt-outs unavailable for the digest (%s)", type(error).__name__)
            return []
        lines = []
        for row in rows or []:
            name = row.get("display_name") or row.get("contact_id")
            reason = " ".join(str((row.get("detail") or {}).get("reason") or "").split())
            lines.append(f"{name} opted out of my messages, saying: \"{reason}\"" if reason
                         else f"{name} opted out of my messages.")
        return lines

    async def _names(self, ids: Iterable[Any]) -> Dict[str, str]:
        """How the owner reads these contacts: the owner is "you"; anyone else by name, else by a handle the owner
        would recognise (a number, an address), else as "a contact". Never an internal id."""
        names: Dict[str, str] = {}
        for ident in dict.fromkeys(str(item) for item in ids if item):
            names[ident] = "you" if self._is_owner(ident) else await self._shown_name(ident)
        return names

    async def _shown_name(self, contact_id: str) -> str:
        record = await self._contact_record(contact_id) or {}
        name = str(record.get("display_name") or record.get("given_name") or "").strip()
        if name and name != contact_id:
            return name
        getter = getattr(self.contacts, "get_handles", None)
        if callable(getter):
            try:
                handles = await getter(contact_id) or []
            except Exception as error:
                logger.debug("handles of %s unavailable (%s)", contact_id, type(error).__name__)
                handles = []
            address = next((str(getattr(item, "address", "") or "").strip() for item in handles
                            if str(getattr(item, "address", "") or "").strip()), "")
            if address and contact_id not in address:
                return address
        return "a contact"

    def _charge(self, task: str, tokens: int, source: str | None = None) -> None:
        """A typed call's real usage on the day's token budget (the budget reads ``cost_tokens`` off the rows): one
        internal note, never an action."""
        if not tokens:
            return
        now = self.clock()
        try:
            row, _ = self.store.create_intention(
                kind="note", type="typed_call", title=f"typed call: {task}", drive="upkeep", cls="internal",
                decision="act", decision_reason="a typed decision or summary", status="done", dedup_key=None,
                hermes_kind="none", context={"task": task, "source": source}, created_at=now)
            self.store.update(row.id, cost_tokens=int(tokens))
            self.store.transition(row.id, "done", action="charged", outcome="done", verified="none",
                                  completed_at=now, at=now)
        except Exception as error:
            logger.warning("typed call %s not charged (%s)", task, type(error).__name__)

    async def _summarize(self, text: str, limit: int) -> Optional[str]:
        """A checked shorter version of one digest item, from the mind's router (``plain.summarize``), charged."""
        return await plain.summarize(self.router, text, limit, tokens_allowed=self.authority.tokens_allowed,
                                     on_usage=lambda tokens: self._charge(plain.SUMMARY_TASK, tokens))

    def _full_ask_seen(self, notice: StoredInitiative) -> None:
        """An ask too long for the digest reached the owner in full, in its own message: a yes may count now.
        Should this not run, the notice's own delivery record is read instead (``Outbox.full_view_seen``)."""
        self.outbox.full_view_delivered(notice)

    async def _digest(self, now: datetime) -> Optional[str]:
        """The day's one digest (``digest.py``), queued once per local date (its dedup key), only when something
        waits or happened. What it carries (a finding, a held item) counts as listed only once the body
        reports it delivered (``_reconcile_digested``); a digest never delivered hands them back to the next one."""
        local = now.astimezone(self.tz)
        local_date = local.date().isoformat()
        if local.hour < self.digest_hour or self._daily.get("digest") == local_date or self.in_quiet_hours(now):
            return None
        if self.outbox.digest_sent_today(local_date):
            self._daily["digest"] = local_date
            return None
        other = self._digest_day(local_date, now)
        if other == "wait":
            return None                     # an earlier digest is still on its way: decided at a later tick
        if other == "taken":
            self._daily["digest"] = local_date
            return None                     # an earlier digest arrived today: it was today's
        self._reconcile_digested(now)
        last = self.store.last_transition_at("queued", type="digest")
        since = last or (now - timedelta(days=1))
        breakers = [self.authority.breaker_state(cls, now) for cls in CLASSES if cls != "floor"]
        found, offers, paused, listed = (self._outreach_digest(since, now)
                                         if self.faculties["outreach"] and self.owner_id else ([], [], None, []))
        # Read whether or not the gate is on now: a hold made while it was on is an obligation the flag
        # turned off does not cancel. A held question of an earlier build is never listed.
        held_all = self.owner_gate.ledger.held()
        held = held_all[:HELD_LINES]      # listed; marked listed once the digest is delivered
        held_lines = []
        for entry in held:
            row = self.store.get(str(entry.get("ref") or "")) if entry.get("ref") else None
            if row is not None and is_question(row):
                continue
            held_lines.append((str(entry["topic"] or entry["type"] or ""),
                               str((entry["inputs"].get("message") or {}).get("text") or "")))
        drift = self.drift.digest_lines(since) if self.faculties.get("drift_check") else []
        items = {"held": [entry["id"] for entry in held], "findings": list(listed)}
        recipients = [row.entity_id for row in self.store.intentions(status=["asked"], kind=["message"], limit=200)]
        recipients += [row.entity_id for row in self.store.intentions(since=since, kind=["message"], limit=500)
                       if not self._is_owner(row.entity_id)]
        standing = self._allows_digest(now)
        work_line = None
        if self.faculties.get('work_records', True) and self.owner_id:
            from colonymind.work.digest import summary_line
            work_line = summary_line(self.state_dir, contact_id=self.owner_id, since=since, until=now)
        text = await self.outbox.build_digest(
            since=since, level=self.level, breaker_states=breakers,
            goals=[self.goals.render(goal) for goal in self.goals.open()], standing=standing,
            opt_outs=await self._opt_outs_since(since), found=found, offers=offers, paused=paused,
            held=held_lines, held_more=len(held_all) - len(held),
            wishes=self.wishes.digest_lines(since) if self.faculties["wishes"] else None, drift=drift,
            names=await self._names(recipients), work=work_line)
        self._daily["digest"] = local_date
        if text is None:
            return None
        # A digest is delivered on its own day or not at all: it expires at the end of its local date, so no digest
        # ever arrives on a later day than its own (whatever it carries goes to the next one).
        midnight = datetime.combine(local.date() + timedelta(days=1), datetime.min.time(), tzinfo=local.tzinfo)
        row = self.outbox.queue_digest(local_date=local_date, text=text, items=items,
                                       expires_at=min(now + timedelta(hours=20), midnight.astimezone(timezone.utc)))
        if row is not None:
            for ident in listed:
                # Carried, not yet listed: ``listed`` once this digest is delivered (``_reconcile_digested``).
                self._mark_finding(ident, "queued", now, digest=row.id)
            self.owner_gate.ledger.mark_queued([entry["id"] for entry in held], row.id)
            if standing:
                self.mind_state.set(allow_rules.DIGEST_KEY, text=now.isoformat(), now=now)
        return row.id if row is not None else None

    def _allows_digest(self, now: datetime) -> List[str]:
        """The digest's line on the standing allows (``allows.digest_line``), at most once in ``allows.DIGEST_DAYS``
        and only while one is in force. It rides along in a digest and never makes one on its own."""
        in_force = self.allows.in_force()
        last = _utc((self.mind_state.get(allow_rules.DIGEST_KEY) or {}).get("text"))
        if not in_force or (last is not None and now - last < timedelta(days=allow_rules.DIGEST_DAYS)):
            return []
        line = allow_rules.digest_line(in_force, now)
        return [line] if line else []

    def _digest_day(self, local_date: str, now: datetime) -> Optional[str]:
        """One digest per delivery day. An earlier digest still waiting to go, or in the middle of its send, is
        decided later (``wait``). One delivered today, however late for its own date, was today's (``taken``); so is
        one whose delivery is uncertain (or a send left mid-way past ``STALE_SENDING``) that began today, since it
        may have arrived. Anything else is another day's."""
        def today(moment: Optional[datetime]) -> bool:
            return moment is not None and moment.astimezone(self.tz).date().isoformat() == local_date
        for row in self.store.intentions(kind=["message"], types=["digest"], since=now - timedelta(days=3), limit=10):
            if row.dedup_key == f"digest:{local_date}":
                continue
            claimed = next((item.timestamp for item in self.store.get_history(row.id, limit=50)
                            if item.action == "sending"), None)
            claimed = _utc(claimed) if claimed is not None else None
            if row.status in {"proposed", "approved"} and (row.expires_at is None or _utc(row.expires_at) > now):
                return "wait"
            if row.status == "sending" and (claimed is None or now - claimed < STALE_SENDING):
                return "wait"
            if row.status == "sent" and today(_utc(row.completed_at)):
                return "taken"
            if row.status in {"sending", "uncertain"} and today(claimed):
                return "taken"
        return None

    def _outreach_digest(self, since: datetime, now: datetime) -> tuple[list, list, Optional[str], List[str]]:
        """The digest's outreach lines: findings the digest owns (worth it, not sent in their window) and the
        ones the owner put off, offers that expired unsent or were put off, and a paused line."""
        o = outreach_functions
        found, listed, offers = [], [], []
        self._reconcile_digested(now)
        for row in self.store.findings(["digest"], sorted(SHARED_FINDINGS)):
            context = row.context if isinstance(row.context, dict) else {}
            topic = str(context.get("topic") or row.description)
            # The report's sentences on the topic, whole: the digest shortens them whole or summarises them.
            found.append((topic, o.excerpt(str(row.result or ''), topic, EXCERPT_WHOLE, substantive=True)))
            listed.append(row.id)
        for row in self.store.intentions(kind=["message"], since=since - timedelta(days=2), limit=1000,
                                         recipient=self.owner_id):
            if row.type not in OUTREACH_TYPES:
                continue
            context = row.context if isinstance(row.context, dict) else {}
            put_off = self._reaction(row).get("class") == "not_now" and (_utc(self._reaction(row).get("at")) or now) >= since
            unsent = row.status == "expired" and (_utc(row.cancelled_at) or now) >= since
            if not (put_off or unsent):
                continue
            source = str(context.get("source_ref") or "")
            if unsent and source.startswith("intention:") and source[len("intention:"):] in listed:
                continue    # the finding itself is listed above: once is enough
            line = (f"{context.get('topic') or row.description}{' (you said not now)' if put_off else ''}",
                    str(context.get("text") or ""))
            (found if row.type == "outreach_finding" else offers).append(line)
        pause = self.mind_state.get(o.PAUSE_KEY)
        paused = None
        if o.pause_until(pause) is not None and now < o.pause_until(pause):
            started = _utc((pause or {}).get("updated_at")) or now
            paused = (f"Check-ins are paused since {started.astimezone(self.tz).date().isoformat()}; "
                      f"say 'you can check in again' to resume them.")
        return found, offers, paused, listed

    async def request_message(self, payload: Mapping[str, Any], *, source: str = "reach_out") -> bool:
        """A message another subsystem wants sent, through authority into the outbox.

        This replaces the reach-out delivery path: the text is queued verbatim
        for the owner, or becomes an ask for anyone else the level does not
        cover. True when it was queued or asked; False when nothing was done.
        """
        import hashlib
        text = str(payload.get("message") or payload.get("text") or payload.get("description") or "").strip()
        if not text:
            return False
        recipient = str(payload.get("entity_id") or payload.get("person_id") or payload.get("recipient")
                        or self.owner_id or "") or None
        title = str(payload.get("title") or text[:80]).strip()
        item_id = str(payload.get("id") or hashlib.sha256(text.encode("utf-8")).hexdigest()[:16])
        candidate = Candidate(
            type=f"reach_out:{payload.get('type') or source}", drive="duty", kind="message", title=title,
            dedup_key=f"reach_out:{source}:{item_id}", salience=0.9, cost=0.0, recipient=recipient, text=text,
            rationale=f"requested by {source}", evidence=[f"{source}:{item_id}"], concern=title,
            source_type=source, source_id=item_id)
        row = await self._form(candidate, 0.9, self.clock())
        return row is not None and row.status in {"approved", "asked"}

    # -- the body API -------------------------------------------------------------------------

    def dispatch(self) -> List[Dict[str, Any]]:
        """Approved task intentions the body may create now, oldest first; at most once.

        Only ``task`` rows are offered: a goal owns no Hermes object of its own
        (architecture 3.3), its steps are task intentions with kanban ``goal_mode``.
        """
        now = self.clock()
        self.last_pull_at = now
        if not self.enabled:
            return []
        rows = self.store.intentions(status=["approved"], kind=["task"], limit=200)
        payloads = []
        for row in sorted(rows, key=lambda item: item.created_at):
            if self._blocked(row):
                continue
            context = row.context if isinstance(row.context, dict) else {}
            # The worker reads the strategy-switch note as it stands at dispatch (PL/body.py sends the body verbatim).
            note = self.feelings.note_for(str(context.get("topic") or context.get("concern") or row.description))
            body = context.get("body") or row.description
            if isinstance(context.get("idea"), dict):
                # An idea the mind started is written from its sealed payload, and only while its sources are current.
                body = self.ideas.brief(row)
                if body is None:
                    continue
            if note and note not in body:
                body = body + "\n\n" + note
            for note in self.learned.at(now).notes(row):
                if note not in body:
                    body = body + "\n\n" + note
            runtime, retries = self.policy.budgets.for_task(row.type)
            payloads.append({
                "id": row.id, "kind": row.kind, "type": row.type, "drive": row.drive,
                "dedup_key": f"mind:{row.id}", "idempotency_key": f"mind:{row.id}",
                "title": row.description, "body": body,
                "assignee": WORKER_PROFILE, "recipient": row.entity_id, "reason": row.decision_reason,
                # What the row was formed with; a row formed without one takes its kind's budget.
                "max_runtime_seconds": int(context.get("max_runtime_seconds") or runtime),
                "max_retries": int(retries if context.get("max_retries") is None else context["max_retries"]),
                "goal_mode": bool(context.get("goal_mode")), "goal_max_turns": context.get("goal_max_turns"),
                "parent_goal_id": row.parent_goal_id,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "expires_at": row.expires_at.isoformat() if row.expires_at else None,
            })
        return payloads

    def bound(self, intention_id: str, hermes_ref: str, *, hermes_kind: str = "kanban") -> Optional[StoredInitiative]:
        """The body's ack, idempotent: the same reference may be reported again after a lost ack. Only for
        what the body was handed (``handed_to_body``): a goal, a note or an asked row is never bound."""
        row = self.store.get(intention_id)
        if row is None or not row.kind:
            return None
        if not handed_to_body(row):
            logger.warning("bind for %s ignored: a %s %s row was never handed to the body", row.id, row.status,
                           row.kind)
            return row
        kind = str(hermes_kind or "kanban")
        if row.status == "approved":
            return self.store.transition(intention_id, "dispatched", action="bound", at=self.clock(),
                                         hermes_kind=kind, hermes_ref=str(hermes_ref), assigned_at=self.clock(),
                                         details={"hermes_ref": str(hermes_ref)})
        if not row.hermes_ref and hermes_ref:
            return self.store.update(intention_id, hermes_ref=str(hermes_ref), hermes_kind=kind)
        return row

    async def outbox_ready(self) -> List[Dict[str, Any]]:
        """The messages the body may send now, each re-checked against its source and against the
        recipient's permission as it is now, carrying the recipient's handles as they are at this
        pull rather than as they were when it formed: a handle added after a message waited
        unroutable lets it go out on the next pull."""
        from colonymind.memory.curated_health import owner_notices
        try:
            owner_notices(self.ledger, self.outbox)
        except Exception as error:
            logger.warning("curated health notice not queued (%s)", type(error).__name__)
        now = self.last_pull_at = self.clock()
        self._expire_messages(now)
        ready = []
        for payload in self.outbox.ready(enabled=self.enabled, quiet=self.in_quiet_hours()):
            row = self.store.get(str(payload["id"]))
            try:
                if (row is None or self._invalidated(row) or await self._cancel_if_replied(row)
                        or await self._permission_withdrawn(row, now) or self._withdrawn_by_hold(row, now)):
                    continue
                if (payload.get("recipient_is_owner") and self.owner_gate.enabled
                        and not await self._gate_passes(row, now)):
                    continue
            except Exception as error:
                logger.warning("message %s held: %s", row.id, error if isinstance(error, Unreadable) else type(error).__name__)
                continue
            handles = await self._handles(payload["recipient"])
            if handles != payload["recipient_handles"]:
                context = dict(row.context) if isinstance(row.context, dict) else {}
                self.store.update(row.id, context={**context, "recipient_handles": handles})
                payload["recipient_handles"] = handles
            ready.append(payload)
        return ready

    def _withdrawn_by_hold(self, row: StoredInitiative, now: datetime) -> bool:
        """Holds are re-checked at send time: an outreach queued before the owner held its matter (an item held,
        listed or from its first mention, that it names or is about: ``outreach.held``) is withdrawn at the pull,
        cancelled like any outreach the owner's word stopped (a finding's then goes to the digest, an answer
        waits pending until the hold lifts). Raises ``Unreadable`` when the owner's items cannot be read: the
        row is held this pull, never sent."""
        if row.type not in OUTREACH_MESSAGES or self.commitments is None:
            return False
        try:
            rows = list(self.commitments.list(status=["pending", "overdue"], limit=500).get("commitments", []))
        except Exception as error:
            raise Unreadable(f"the owner's holds cannot be read ({type(error).__name__})") from None
        held = self._held_matters(rows)
        if not held:
            return False
        context = row.context if isinstance(row.context, dict) else {}
        named = [str(item)[len("commitment:"):] for item in context.get("evidence") or []
                 if str(item).startswith("commitment:")]
        named += [str(context[key]) for key in ("bound_commitment", "commitment") if context.get(key)]
        state = outreach_functions.OutreachInputs(now=now, owner_id=str(self.owner_id), held=held)
        topic = str(context.get("topic") or "")
        matter = next((found for found in (outreach_functions.held(state, topic, commitment=ident)
                                           for ident in [*named, None]) if found), None)
        if matter is None:
            return False
        self.outcomes.record(row.id, status="cancelled", summary=f"withdrawn: the owner holds {matter[:120]}",
                             verified="check", by="mind", implicit_verdict=False)
        return True

    async def _gate_passes(self, row: StoredInitiative, now: datetime) -> bool:
        """The owner gate's verdict on one ready message to the owner, decided once (the gate replays a
        standing verdict under its lock, so concurrent pulls share it): a send stands until the body claims it,
        a wait until its time; a message held for the digest or dropped is settled at once (``expired``: the
        digest lists it; ``cancelled``), neither being the owner's verdict on its type."""
        owner_started = (row.type in gate_functions.PAUSED_TYPES
                         and ask_origin.owner_started(row, store=self.store, state_dir=self.state_dir,
                             owner_id=self.owner_id, ledger=self.ledger, answer_turns=self.answer_turns,
                             verified=self.owner_gate.ledger.conn))
        verdict = await self.owner_gate.review(gate_functions.message_of_row(row, owner_started=owner_started),
                                               now=now, replay=True)
        if verdict.replayed:
            return verdict.decision == "send"
        current = self.store.get(row.id)
        if current is None or current.status != row.status:
            # Settled elsewhere while the gate decided (a reply, an invalidation): the verdict is in the audit,
            # and the row is left as it was settled, never moved back to go out or be held.
            return False
        self.store.transition(row.id, row.status, action="gated", at=now,
                              details={"gate": verdict.id, "decision": verdict.decision, "stage": verdict.stage,
                                       "reason": verdict.reason[:300], "scores": verdict.scores,
                                       "until": verdict.until.isoformat() if verdict.until else None})
        if verdict.decision in {"send", "wait"}:
            return verdict.decision == "send"
        if verdict.decision == "digest":
            self.outcomes.record(row.id, status="expired", summary=f"held for the digest: {verdict.reason}"[:500],
                                 by="mind", implicit_verdict=False)
        else:
            self.outcomes.record(row.id, status="cancelled", summary=f"dropped by the owner gate: {verdict.reason}"[:500],
                                 by="mind", implicit_verdict=False)
            source = str((row.context or {}).get("source_ref") or "") if isinstance(row.context, dict) else ""
            if row.type == "outreach_finding" and source.startswith("intention:"):
                self._mark_finding(source[len("intention:"):], "repeat", now)
        return False

    async def gate_external(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """An outside script's question (``POST /v1/mind/owner-gate``): may it message the owner about this now?
        The verdict is recorded like any other: a send counts for dedup and spacing, a digest is listed in
        the next digest. Asked again under the same ``id`` (a lost answer, a retry), the standing verdict is
        given again (``replayed``), so a retry never turns an approved alert into its own repeat. The flag
        off: send, as if there were no gate, and nothing recorded."""
        if not self.owner_gate.enabled:
            return {"decision": "send", "gate": "off", "reason": "the owner gate is off (mind.faculties.owner_gate)"}
        source = str(payload.get("source") or "external").strip() or "external"
        message = gate_functions.Message(
            type=f"alert:{source}"[:64], topic=" ".join(str(payload.get("topic") or "").split()),
            text=str(payload.get("text") or ""), urgency=str(payload.get("urgency") or "normal"),
            source="external" if source == "mind" else source, ref=payload.get("id") or None)
        verdict = await self.owner_gate.review(message, replay=message.ref is not None)
        return {**verdict.as_dict(), "gate": "on"}

    def gate_log(self, *, limit: int = 50, since_hours: float = 24.0) -> Dict[str, Any]:
        """The gate's audit: its switch and settings, the verdicts of the window (newest first) with their
        inputs and scores, and their counts by decision and by stage."""
        since = self.clock() - timedelta(hours=max(0.0, float(since_hours)))
        settings = self.owner_gate.settings
        return {"enabled": self.owner_gate.enabled,
                "settings": {"topic_window_hours": settings.topic_window / timedelta(hours=1),
                             "repeat_window_hours": settings.repeat_window / timedelta(hours=1),
                             "conversation_minutes": settings.conversation / timedelta(minutes=1),
                             "spacing_minutes": settings.spacing / timedelta(minutes=1),
                             "confidence": settings.confidence, "second_opinion": settings.second_opinion,
                             "decision_model": self.owner_gate.decision_model()},
                "counts": self.owner_gate.ledger.counts(since=since),
                "decisions": self.owner_gate.ledger.recent(since=since, limit=max(1, min(int(limit), 500)))}

    OBSERVATION_LISTS = (("stale_tasks", "stale_task"), ("blocked_tasks", "blocked_task"), ("goals", "goal"),
                         ("mind_tasks", "mind_task"))

    def observe(self, payload: Any) -> Dict[str, Any]:
        """Board observations from the body: stale owner tasks, blocked tasks, goals, the mind's tasks.

        The body posts ``{observed_at, board, body, counts, stale_tasks, blocked_tasks,
        goals, mind_tasks}`` with ``idle_s`` per task (docs/HERMES-ADAPTER.md); the
        flat ``{observations: [{kind, ...}]}`` shape is accepted too. Stale owner
        tasks and Hermes goals are inputs to the duty drive.
        """
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        if isinstance(payload, dict) and "observations" not in payload:
            for key, kind in self.OBSERVATION_LISTS:
                for item in payload.get(key) or []:
                    if not isinstance(item, dict):
                        continue
                    entry = dict(item)
                    entry.setdefault("kind", kind)
                    if entry.get("age_hours") is None:
                        seconds = entry.get("idle_s") if entry.get("idle_s") is not None else entry.get("age_s")
                        entry["age_hours"] = round(float(seconds or 0) / 3600.0, 2)
                    grouped.setdefault(kind, []).append(entry)
            if isinstance(payload.get("body"), dict):
                self.body_heartbeat = dict(payload["body"])
            if isinstance(payload.get("counts"), dict):
                self.board_counts = {str(k): int(v) for k, v in payload["counts"].items() if isinstance(v, int)}
        else:
            items = payload.get("observations") if isinstance(payload, dict) else payload
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                grouped.setdefault(str(item.get("kind") or "other"), []).append(item)
        self.observations = grouped
        self.observed_at = self.clock()
        self.last_pull_at = self.observed_at
        self._wake.set()
        return {"accepted": sum(len(v) for v in grouped.values()), "kinds": sorted(grouped)}

    async def observe_source(self, source: str, payload: Any) -> Dict[str, Any]:
        """Outside sources replace only their own expiring set, independently of the board."""
        from .observations import for_mind
        return await for_mind(self).replace(source, payload)

    def asks(self) -> List[Dict[str, Any]]:
        return audit.log(self.store, status=["asked"], limit=100)

    async def answer(self, code: str, *, yes: bool, by: str = OPERATOR) -> Optional[StoredInitiative]:
        """The console's answer to an open ask by its code (``colonymind mind no``, ``POST /v1/mind/asks/<code>/no``):
        a request that cites no owner turn, so it may narrow and never widen (docs/MIND.md, Asks: console and API
        answers). A no settles the ask as ``by`` on the console path, which is never the owner's own answer whatever
        ``by`` names (``outcomes.owner_answer``: no verdict, nothing learned). A yes raises ``PermissionError``: the
        owner approves only in their own message, read by ``read_answer``. None when no open ask has the code."""
        if yes:
            raise PermissionError(APPROVE_REQUIRED)
        row = self.store.get_by_ask_code(str(code or "").strip().upper())
        if row is None:
            return None
        return await self._settle_ask(row, yes=False, by=by, details={"answer": {"path": "console", "by": by}})

    async def _settle_ask(self, row: StoredInitiative, *, yes: bool, by: str,
                          details: Mapping[str, Any] | None = None,
                          answered_at: datetime | None = None,
                          guard: SettleGuard | None = None) -> Optional[StoredInitiative]:
        """An open ask settled by its answer, once whoever answered was checked and the answer bound to it
        (``read_answer``: by what the owner's message replies or reacts to; ``answer``: the console's no by its
        code; ``answer_explicit``: an owner channel's answer naming the ask's id): expiry and the full text are
        checked here, then the ask's own settle path runs (a grant, an owner question, an ordinary ask). ``details``
        join the settling history row (``answer``: its path, reader, message and channel).
        ``answered_at``: when the owner's message arrived, for an answer read after it; a yes counts only when the
        full text had reached them by then, whenever the message is read.
        ``guard`` (an ordinary ask, a held call's among them): its settling write happens only while the ask is
        still open, in one SQLite transaction with the guard's writes; an ask another answer settled first raises
        ``TransitionRefused`` and nothing is written (the first commit wins)."""
        code = str(row.ask_code or "")
        settle_guard = {"guard": guard} if guard is not None else {}
        if row.expires_at is not None and row.expires_at <= self.clock():
            # Expiry is checked when the answer comes, not only when the tick runs: a late answer approves nothing,
            # and the ask lapses exactly as the tick's sweep would close it (``_expire_asks``): no verdict, since
            # the silence up to its lapse is not the owner's "no" (APPROVALS 7.4).
            return self.outcomes.record(row.id, status="expired", summary="answered after the ask lapsed, so nothing "
                                        "was done", by="mind", implicit_verdict=False, **settle_guard)
        full = row.context.get("full_view") if isinstance(row.context, dict) else None
        if yes and isinstance(full, dict) and not self._full_text_before(row, answered_at):
            raise PermissionError("this ask was too long to show in full; a yes counts only once its full text has "
                                  "reached the owner, in its own message")
        if owner_answer(by, details):
            # The owner's own word on a piece of the agent's work goes on its recorder line.
            recorder.mark(f"initiative:{row.id}", "owner_acted" if yes else "owner_declined")
        if row.type == GRANT_CONFIRM:
            return await self._activate_grant(row, yes=yes, by=by, details=details)
        if row.type == ALLOW_PROPOSAL:
            return await self._settle_proposal(row, yes=yes, by=by, details=details)
        if row.type in OWNER_QUESTIONS:
            return await self._settle_question(row, yes=yes, by=by, details=details)
        now = self.clock()
        if not yes:
            updated = self.outcomes.record(row.id, status="denied", summary=f"{self._who(by, details)} said no", by=by,
                                           details=details, **settle_guard)
            return updated
        try:
            if await self._permission_withdrawn(row, now):
                # The recipient opted out (or was set to never) while the ask waited: the yes sends nothing.
                return self.store.get(row.id)
        except Unreadable as error:
            # The owner's yes is recorded; the outbox reads permission again before anything leaves.
            logger.warning("ask %s approved unread: %s", row.id, error)
        expires_at = now + TASK_WINDOW
        expires_at = _cadence_expiry(row, expires_at)
        updated = self.store.transition(row.id, "approved", action="queued", at=now, verdict="actioned",
                                        expires_at=expires_at,
                                        details={**dict(details or {}), "by": by, "code": code}, **settle_guard)
        if self.feedback is not None:
            try:
                self.feedback.record(f"{row.type}:{row.drive}", "actioned", source=row.id)
            except Exception:
                pass
        self.autobiography.record(row.id, "approved", f"The owner approved '{Autobiography.name(row)}' (code {code}).")
        return updated

    @staticmethod
    def _who(by: str, details: Mapping[str, Any] | None) -> str:
        """Who gave an answer, in the words the record keeps (the settling row's result, which the autobiography
        narrates into the owner's own memory): the owner only for their own answer (``outcomes.owner_answer``, read
        from the recorded path, never from ``by`` alone); anyone else is the operator at the console."""
        return "the owner" if owner_answer(by, details) else "the operator at the console"

    async def _settle_question(self, row: StoredInitiative, *, yes: bool, by: str,
                               details: Mapping[str, Any] | None = None) -> Optional[StoredInitiative]:
        """The owner's word on a name-only identity link: the contact store links or rejects the
        handle, and the question is done. The answer is not a verdict on asking: a ``no`` teaches
        the ranker nothing. Every owner question changes who is who or who gets messaged, so an
        expired one is no approval; its yes comes only from the owner's own message, read strict
        (``read_answer``), and the console may only say no."""
        if row.expires_at is not None and row.expires_at <= self.clock():
            return self.outcomes.record(row.id, status="expired", summary="no answer before the ask expired",
                                        by="mind", implicit_verdict=False)
        if row.type == "cadence_confirm":
            return await self._settle_cadence(row, yes=yes, by=by, details=details)
        if row.type == "people_change":
            return await self._settle_people_change(row, yes=yes, by=by, details=details)
        if row.type == WISH_TYPE:
            return self._settle_wish(row, yes=yes, by=by, details=details)
        verb = "confirm_link" if yes else "reject_link"
        method = getattr(self.contacts, verb, None)
        if not callable(method):
            return self.outcomes.record(row.id, status="cancelled", summary=f"no contact store to {verb}", by=by,
                                        implicit_verdict=False, details=details)
        try:
            await method(str(row.source_id), performed_by=by)
        except Exception as error:
            logger.warning("%s failed for %s (%s)", verb, row.source_id, type(error).__name__)
            why = ("the handle belongs to another established contact; merge the two yourself if they are one person"
                   if str(error) == "identity_handle_held" else type(error).__name__)
            return self.outcomes.record(row.id, status="cancelled", by=by, implicit_verdict=False, details=details,
                                        summary=f"the link could not be {'confirmed' if yes else 'rejected'} ({why})")
        if yes:
            return self.outcomes.record(row.id, status="done", summary=f"{self._who(by, details)} confirmed the link",
                                        verified="owner", by=by, details=details)
        return self.outcomes.record(row.id, status="denied", summary=f"{self._who(by, details)} rejected the link",
                                    by=by, implicit_verdict=False, details=details)

    # Asks whose yes widens who gets messaged, who is who or what may run without asking: their answer is read
    # strict (``answers.read``). A floor ask (``cls`` floor) is read strict too; a wish is ordinary, since after the
    # typed grant its yes gives only its typed steps.
    STRICT_ASKS = frozenset({"people_change", "link_proposal", "cadence_confirm", GRANT_CONFIRM, ALLOW_PROPOSAL})

    def _full_text_before(self, row: StoredInitiative, answered_at: datetime | None) -> bool:
        """Whether an ask too long to show had reached the owner in full (``Outbox.full_view_seen``), and, for a
        message read after it arrived (``answered_at``), by the time the owner wrote it: a yes to the short message
        written before the full text arrived still answered the short message, however late it is read."""
        if not self.outbox.full_view_seen(row):
            return False
        if answered_at is None:
            return True
        current = self.store.get(row.id) or row
        full = current.context.get("full_view") if isinstance(current.context, dict) else None
        if not isinstance(full, dict):
            return True
        seen_at = _utc(full.get("seen_at"))
        return seen_at is not None and seen_at <= answered_at

    # -- the owner's answer, read from their own message (docs/MIND.md, Asks) -----------------------------------------

    @property
    def answer_turns(self) -> AnswerTurns:
        """The ``answer_turns`` table in the workspace's ``mind.db`` (``mind/answer_turns.py``)."""
        table = getattr(self, "_answer_turns", None)
        if table is None:
            conn = getattr(self.concerns, "conn", None)
            table = self._answer_turns = AnswerTurns(conn if isinstance(conn, sqlite3.Connection)
                                                     else open_mind_db(self.state_dir / MIND_DB))
        return table

    def answer_events(self, since: datetime) -> List[Dict[str, Any]]:
        """The answer path's own events since ``since`` for the asks report: each clean yes or no that bound to no
        ask while one was open (``unbound_answer``)."""
        return self.answer_turns.events(since)

    def _answer_refused(self, evidence: Mapping[str, Any], *, granted: bool) -> str:
        """Why a message is not read as a possible answer at all (docs/MIND.md, Asks: who may answer), or ``""``:
        only the owner, in their own direct chat with the agent, on a platform the request's credential is granted
        the owner on; never an event from the agent's own account or a message the channel marks forwarded."""
        if not self.owner_id:
            return "no owner is configured"
        if not granted:
            return "the request's credential is not granted the owner on this platform"
        if str(evidence.get("contact_id") or "") != self.owner_id:
            return "not the owner's own message"
        if str(evidence.get("chat_type") or "").strip().lower() != "dm":
            return "only the owner's direct chat with the agent is read"
        if evidence.get("own_account"):
            return "an event from the agent's own account is never read"
        if evidence.get("forwarded"):
            return "a forwarded message is never read"
        if not str(evidence.get("text") or "").strip() and not (evidence.get("reaction")
                                                               or evidence.get("reaction_removed")):
            return "nothing to read"
        return ""

    async def read_answer(self, evidence: Mapping[str, Any], *, granted: bool = True,
                          route: str = "answers") -> Dict[str, Any]:
        """One owner message read as a possible answer to the open asks (``POST /v1/mind/answers``, the plugin's
        read before the model runs; ``owner_turn``'s ingest read after it, ``route='ingest'``, without the ledger).

        ``evidence``: ``platform``, ``contact_id`` (the sender as the caller resolved it), ``chat_id``,
        ``chat_type``, ``turn_id``, ``message_id``, ``arrived_at``, ``text`` (the owner's own words; never stored),
        ``reply_to_message_id``, ``reaction``, ``reaction_removed``, ``forwarded``, ``own_account`` and ``ledger``
        (what reached the chat before it: ``[{at, outbox_id?, message_id?, kind?, ordered?}]``, or None).

        Read once per message (``answer_turns``, keyed by platform and message id, else the turn id). An undo comes
        first (inside ``answers.UNDO_MINUTES`` of an approval in the same chat), then "ask me again about <thing>",
        which revokes the standing allows the thing names (``_revoke_in_words``); then the message binds by
        structure (``answers.bind``), its words are read by the tables (``answers.read``, strict for a floor or
        widening ask) and the plan runs: each yes or no settles through the ask's own settle path
        (``_settle_ask``), a later is recorded, a weighty ask named in a list's yes is shown, an unclear answer
        gets one plain question (for one ordinary ask, after Instinct's ``ask_answer`` point read it:
        ``_typed_answer``). Not bound, words that name exactly one open ask show it; a clean yes or no bound to
        nothing is counted. Returns what happened and ``context``, the plain words the chat model reads before it
        replies."""
        refused = self._answer_refused(evidence, granted=granted)
        if refused:
            return {"read": False, "reason": refused, "context": ""}
        platform = str(evidence.get("platform") or "").strip().lower()
        chat_id = str(evidence.get("chat_id") or "").strip()
        message_id = str(evidence.get("message_id") or "").strip()
        turn_id = str(evidence.get("turn_id") or "").strip()
        # A delivery id is unique only within its chat on some platforms: the key carries all three.
        key = f"{platform}:{chat_id}:{message_id}" if message_id else (f"turn:{turn_id}" if turn_id else "")
        if not key:
            return {"read": False, "reason": "a message is read once, by its id, and it carried none", "context": ""}
        async with self._answer_serial():
            prior = self.answer_turns.find(key, turn_id=turn_id)
            if prior is not None:
                return {**prior.result, "read": False, "key": key, "context": "",
                        "reason": "already read: a message is read once"}
            now = self.clock()
            arrived = min(_utc(evidence.get("arrived_at")) or now, now)
            read = answer_log.Read(key=key, turn_id=turn_id, platform=platform, chat_id=chat_id, message_id=message_id,
                                   target=str(evidence.get("reply_to_message_id") or ""), arrived_at=arrived,
                                   read_at=now, route=route)
            result = await self._answer_message(evidence, read, now=now)
            read.result = self._answer_record(result)
            self.answer_turns.record(read)
            self.answer_turns.prune(now - answer_log.RETENTION)
            return {**result, "read": True, "key": key}

    async def answer_explicit(self, ask_id: str, *, choice: str, channel: str, contact_id: Optional[str],
                              authority: Any, key: str, request_hash: str) -> tuple[int, Dict[str, Any]]:
        """An owner channel's explicit answer to one hands permit ask, naming its immutable id and one of its offered
        choices (``POST /v1/mind/permit-asks/{id}/answer``; docs/hands-permit-answer-contract.md). Returns the HTTP
        status and the result; raises ``permit_asks.PermitAnswerError`` for a request it refuses unrecorded.

        Who may answer: the owner (``contact_id``, as the route resolved it), through the channel relay's own
        ``answer`` credential (``authority``: ``colonymind.api.auth.answer_credential`` for ``channel``; never the
        instance key), from an explicit-answer platform (``permit_asks.EXPLICIT_PLATFORMS``) spelled exactly. The
        route checks the same; this is the check that holds for any caller.

        The one transaction boundary with the messaging answers: under ``_answer_serial`` (the lock ``read_answer``
        holds), the ask is read fresh and settled through ``_settle_ask``, the same settle path and the same row an
        RCS reaction, quote or follow-up settles, with the settling write guarded (``SettleGuard``: only while the
        ask is still open). The idempotency record (``scope``, ``key``, the request digest and this result) commits
        in that same SQLite transaction. Whoever comes second, in this process or another, settles nothing and gets
        the recorded winner (``already_answered``) or how the ask closed (``ask_closed``). The same key with the same
        request answers the recorded result again (``replayed``); with another request it is refused
        (``idempotency_conflict``). Granting and resuming the hand stay with the hands controller, exactly as after
        an RCS answer: this records the owner's answer, and queues the line that tells the owner where they read
        (``outbox.ASK_ANSWERED``), nothing else."""
        from colonymind.api.auth import answer_credential
        from . import permit_asks
        from .permit_asks import PermitAnswerError
        if not isinstance(channel, str) or not permit_asks.canonical(channel):
            raise PermitAnswerError("invalid_platform", 422, "platform is a platform name as such: lowercase, "
                                    "without spaces")
        if channel not in permit_asks.EXPLICIT_PLATFORMS:
            raise PermitAnswerError("platform_not_allowed", 403, f"explicit answers come only from "
                                    f"{', '.join(sorted(permit_asks.EXPLICIT_PLATFORMS))}")
        if authority is None or not answer_credential(authority, channel):
            raise PermitAnswerError("answer_credential_required", 403,
                                    f"only the {channel} relay's own answer credential answers by id")
        if not self.owner_id:
            raise PermitAnswerError("no_owner", 409, "no owner is configured")
        if str(contact_id or "") != self.owner_id:
            raise PermitAnswerError("not_owner", 403, "only the owner answers their asks")
        scope = str(authority.principal_id)
        async with self._answer_serial():
            prior = self.store.answer_request(scope, key)
            if prior is not None:
                if prior["request_hash"] != request_hash:
                    raise PermitAnswerError("idempotency_conflict", 409,
                                            "this idempotency key was used with a different request",
                                            ask_id=prior["ask_id"])
                return int(prior["status_code"]), {**prior["response"], "replayed": True}
            row = self.store.get(str(ask_id))
            if not permit_asks.is_permit_ask(row):
                raise PermitAnswerError("unknown_ask", 404, f"no hands permit ask {ask_id}")
            offered = [item["value"] for item in permit_asks.choices(row)]
            if choice not in offered:
                raise PermitAnswerError("invalid_choice", 422, f"this ask offers {', '.join(offered)}",
                                        choices=offered)
            now = self.clock()
            if row.status != "asked":
                return self._explicit_closed(row, scope=scope, key=key, request_hash=request_hash, now=now)
            recorded: Dict[str, Any] = {}

            def response(status: str, at: datetime) -> tuple[int, Dict[str, Any]]:
                if status == "expired":
                    result = permit_asks.ASK_CLOSED
                    value = {"result": result, "ask_id": row.id, "status": status, "outcome": "expired",
                             "winner": None, "settlement": {"by": "lapse", "outcome": "expired",
                                                            "at": at.isoformat(), "winner": None}}
                else:
                    result = permit_asks.SETTLED
                    winner = {"channel": channel, "choice": choice, "at": at.isoformat(), "path": answer_words.EXPLICIT}
                    value = {"result": result, "ask_id": row.id, "status": status,
                             "outcome": None if status == "approved" else "denied", "winner": winner,
                             "settlement": {"by": "owner", "outcome": "approved" if choice == "yes" else "denied",
                                            "at": at.isoformat(), "winner": winner}}
                return permit_asks.STATUS_CODES[result], {**value, "replayed": False}

            def writes(status: str, at: datetime) -> list:
                code, value = recorded["result"] = response(status, at)
                return [self.store.answer_request_write(scope=scope, key=key, ask_id=row.id,
                                                        request_hash=request_hash, status_code=code,
                                                        response={**value, "replayed": False}, at=at)]

            ref = permit_asks.message_ref(channel, scope, key)
            details = {"answer": {"path": answer_words.EXPLICIT, "reader": "explicit", "message": ref, "by": "owner",
                                  "channel": channel, "choice": choice}}
            try:
                await self._settle_ask(row, yes=choice == "yes", by=f"owner-answer:{ref}", details=details,
                                       answered_at=now, guard=SettleGuard(writes=writes))
            except PermissionError as error:
                raise PermitAnswerError("full_text_not_delivered", 409, str(error)) from None
            except TransitionRefused as moved:
                # Another answer (or the lapse) committed first, in the moment between the read and this write.
                current = moved.current or row
                return self._explicit_closed(current, scope=scope, key=key, request_hash=request_hash, now=now)
            except sqlite3.IntegrityError:
                # The same key was recorded by another process in that moment: the whole settlement rolled back.
                prior = self.store.answer_request(scope, key)
                if prior is None:
                    raise
                if prior["request_hash"] != request_hash:
                    raise PermitAnswerError("idempotency_conflict", 409,
                                            "this idempotency key was used with a different request",
                                            ask_id=prior["ask_id"]) from None
                return int(prior["status_code"]), {**prior["response"], "replayed": True}
            if "result" not in recorded:
                current = self.store.get(row.id) or row
                if current.status == "asked":
                    raise PermitAnswerError("not_settled", 409, "the ask stayed open; nothing was recorded")
                return self._explicit_closed(current, scope=scope, key=key, request_hash=request_hash, now=now)
            if recorded["result"][1].get("result") == permit_asks.SETTLED:
                self._tell_answered(row, choice=choice, channel=channel, ref=ref)
            return recorded["result"]

    def _tell_answered(self, row: StoredInitiative, *, choice: str, channel: str, ref: str) -> Optional[str]:
        """The one line that tells the owner an ask was answered in an owner channel's app (``outbox.ASK_ANSWERED``):
        at the hand ask's RCS anchor when it has one, else at their primary handle, never in the channel that
        answered, so an answer they did not give is seen where they read. A notice not queued (no owner, a failed
        write) leaves the answer as recorded; it is logged. The id of the message, or None."""
        from .outbox import ASK_ANSWERED
        try:
            shown = plain.fit(ask_sentence(row), 300) or plain.pointer("The ask")
            text = f"Answered in the app: {shown} -> {choice}."
            if choice == "yes":
                text += " An answer given in the app is not undone from this chat."
            message = self.outbox.notice(type=ASK_ANSWERED, title="An ask answered in the app", text=text,
                                         dedup_key=f"{ASK_ANSWERED}:{row.id}:{ref}",
                                         extra={"answered_ask": row.id, "channel": channel, "choice": choice})
        except Exception as error:
            logger.warning("the line telling the owner ask %s was answered in the app is not queued (%s)", row.id,
                           type(error).__name__)
            return None
        return message.id if message is not None else None

    def _explicit_closed(self, row: StoredInitiative, *, scope: str, key: str, request_hash: str,
                         now: datetime) -> tuple[int, Dict[str, Any]]:
        """An explicit answer that settled nothing because the ask had closed: the recorded winner or how it closed,
        recorded under the request's key so a replay answers the same."""
        from . import permit_asks
        result, value = permit_asks.closed(self.store, row)
        code = permit_asks.STATUS_CODES[result]
        value = {**value, "replayed": False}
        if not self.store.record_answer_request(scope=scope, key=key, ask_id=row.id, request_hash=request_hash,
                                                status_code=code, response=value, at=now):
            prior = self.store.answer_request(scope, key) or {}
            if prior.get("request_hash") != request_hash:
                raise permit_asks.PermitAnswerError("idempotency_conflict", 409,
                                                    "this idempotency key was used with a different request",
                                                    ask_id=prior.get("ask_id"))
            return int(prior["status_code"]), {**prior["response"], "replayed": True}
        return code, value

    def _answer_serial(self) -> asyncio.Lock:
        """The one lock every owner answer to an ask is read and settled under (``read_answer``: a message on a
        messaging channel; ``answer_explicit``: an owner channel's answer naming the ask): in this process two
        answers to one ask never interleave, and the second sees the first's settlement. Across processes the
        settling write's own guard decides (``SettleGuard``: compare-and-set in the ask's SQLite row)."""
        lock = getattr(self, "_answer_lock", None)
        if lock is None:
            lock = self._answer_lock = asyncio.Lock()
        return lock

    async def _answer_message(self, evidence: Mapping[str, Any], read: answer_log.Read, *,
                              now: datetime) -> Dict[str, Any]:
        """What one owner message does to the asks (``read_answer``); sets the read's kind, path and reader."""
        w = answer_words
        text = str(evidence.get("text") or "")
        reaction = w.canonical(evidence.get("reaction") or "")
        removed = bool(evidence.get("reaction_removed"))
        by = f"owner-turn:{read.key}"
        result: Dict[str, Any] = {"path": None, "reading": None, "asks": [], "settled": [], "approved": [],
                                  "allowed": [], "later": [], "shown": [], "clarify": None, "undone": [],
                                  "revoked": [], "refused": [], "unbound": False, "acknowledged": None, "context": ""}
        undone = await self._undo_answer(read, text=text, reaction=reaction, removed=removed, by=by, now=now)
        if undone:
            read.kind, result["undone"] = answer_log.UNDONE, undone
            if reaction or removed or read.route == "ingest":
                result["acknowledged"] = self._acknowledge(result, key=read.key)
            result["context"] = self._answer_context(result, [])
            return result
        revoked = await self._revoke_in_words(text, by=by) if not reaction and not removed else []
        if revoked:
            # The message took allows back; it answers no ask (it is never read as the "always" it might resemble).
            read.kind, result["revoked"] = answer_log.REVOKED, revoked
            if read.route == "ingest":
                result["acknowledged"] = self._acknowledge(result, key=read.key)
            result["context"] = self._answer_context(result, [])
            return result
        open_rows = self._open_asks()
        if not open_rows:
            return result
        rows = {row.id: row for row in open_rows}
        weighty = {row.id for row in open_rows if self._weighty(row)}
        codes = [str(row.ask_code) for row in open_rows]
        coded = self._coded_rows(open_rows)
        shown = self._shown_messages(open_rows, coded)
        ledger = self._answer_ledger(evidence.get("ledger"), shown, coded, platform=read.platform, chat=read.chat_id)
        binding = w.bind(w.Evidence(read.platform, read.arrived_at, chat=read.chat_id, reply_to=read.target,
                                    reaction=reaction, reaction_removed=removed, ledger=ledger), shown, weighty=weighty)
        if binding.path == w.REACTION:
            reading = w.Reading(w.NONE, reason="a reaction removed") if removed else w.reaction_reading(reaction)
        elif binding.listed:
            reading = w.read(text, list_size=len(binding.asks), codes=codes)
        elif binding:
            ask = rows.get(binding.asks[0])
            code = str(ask.ask_code or "") if ask is not None else ""
            offered = ask is not None and self._offer(ask) is not None
            reading = w.read(text, strict=binding.asks[0] in weighty, codes=codes, code=code, offers_always=offered)
            if reading.verdict == w.ALWAYS:
                reading = self._always_bar(text, binding, read.arrived_at, codes=codes, code=code) or reading
        else:
            reading = w.read(text, codes=codes)
        plan = w.plan(reading, binding, open_asks=rows, weighty=weighty)
        read.path, read.reader, read.asks = binding.path, "tables", list(binding.asks)
        result.update(path=binding.path or None, reading=reading.as_dict(), asks=list(binding.asks))
        names = await self._names(row.entity_id for row in open_rows if self._named_in_ask(row))
        sentences = {row.id: ask_sentence(row, name=names.get(str(row.entity_id))) for row in open_rows}
        # ``channel``: the platform the answer came on, so a later answer from another channel (an explicit answer
        # naming the ask) reads which channel settled it (``permit_asks.settlement``).
        details = {"answer": {"path": binding.path, "reader": "tables", "message": read.key, "by": "owner",
                              "channel": read.platform}}
        # The unclear branch: a model's reading of one ordinary ask's unclear answer (Instinct's ``ask_answer``).
        typed = await self._typed_answer(rows, plan, reading, binding, text, weighty=weighty, read=read,
                                         sentences=sentences, now=now) if plan.clarify else None
        if typed is not None:
            plan, details, read.reader = typed["plan"], typed["details"], "decision"
        if read.turn_id and self._asked_back(rows[ident] for ident in (*plan.yes, *plan.no)):
            # This answer grades a model's yes asked back: its own turn is kept with it, so erasing that turn forgets
            # the grade (``outcomes.forget_erased``), as erasing the turn the decision read does.
            details = {**details, "answer": {**details["answer"], "turn": read.turn_id}}
        for yes, idents in ((True, plan.yes), (False, plan.no)):
            for ident in idents:
                row = rows[ident]
                try:
                    # Guarded: the ask settles only while it is still open, so an answer from another channel (or
                    # the tick's lapse) that committed first is never overwritten.
                    updated = await self._settle_ask(row, yes=yes, by=by, details=details, answered_at=read.arrived_at,
                                                     guard=SettleGuard())
                except PermissionError as error:
                    result["refused"].append({"id": row.id, "sentence": sentences[row.id], "reason": str(error)})
                    continue
                except TransitionRefused as moved:
                    status = moved.current.status if moved.current is not None else "gone"
                    result["refused"].append({"id": row.id, "sentence": sentences[row.id],
                                              "reason": f"it was settled before this answer ({status})"})
                    continue
                current = updated or self.store.get(row.id) or row
                result["settled"].append({"id": row.id, "code": row.ask_code, "sentence": sentences[row.id],
                                          "status": current.status})
                if yes and current.status in {"approved", "dispatched", "done"}:
                    result["approved"].append(row.id)
                    offer = self._offer(row)
                    if ident in plan.always and offer is not None:
                        allow = await self._adopt_allow(current, offer, by=by, now=now,
                                                        name=names.get(str(row.entity_id)))
                        if allow is not None:
                            result["allowed"].append({"id": row.id, "allow": allow.id, "words": allow.words})
                    elif offer is not None:
                        await self._propose_allow(offer, now=now, name=names.get(str(row.entity_id)))
        for ident in plan.later:
            row = rows[ident]
            self.store.transition(row.id, row.status, action=ASK_LATER, at=now,
                                  details={"by": "owner", "message": read.key})
            result["later"].append({"id": row.id, "sentence": sentences[row.id]})
        for ident in plan.show:
            shown_id = self._show_ask(rows[ident], sentences[ident], key=read.key, now=now,
                                      why="a list's yes does not take a change of this weight",
                                      name=names.get(str(rows[ident].entity_id)))
            result["shown"].append({"id": ident, "sentence": sentences[ident], "message": shown_id})
        if typed is not None and typed.get("confirm"):
            ident = typed["confirm"]
            result["clarify"] = self._confirm_back(rows[ident], sentences[ident], typed["details"]["answer"],
                                                   key=read.key, now=now)
        elif plan.clarify or plan.which:
            result["clarify"] = self._clarify([rows[ident] for ident in plan.clarify or plan.which], sentences,
                                              key=read.key, now=now)
        if not binding:
            named = ()
            if reading.answer_word or reading.verdict in {w.YES, w.NO}:
                named = w.named(text, await self._ask_names(open_rows),
                                codes={str(row.ask_code): row.id for row in open_rows})
            if len(named) == 1:
                ident = named[0]
                shown_id = self._show_ask(rows[ident], sentences[ident], key=read.key, now=now,
                                          why="the owner's words named it, which never approves it",
                                          name=names.get(str(rows[ident].entity_id)))
                result["shown"].append({"id": ident, "sentence": sentences[ident], "message": shown_id})
            elif reading.verdict in {w.YES, w.NO}:
                result["unbound"] = True
                result["open"] = [sentences[row.id] for row in open_rows]
                # Recovery shows the most recently shown eligible ask in this chat. It never
                # supplies a missing binding or settles anything on the unbound message.
                eligible = []
                for message in shown if not reaction and not removed else []:
                    if ((message.platform, message.chat) != (read.platform, read.chat_id)
                            or message.at > read.arrived_at):
                        continue
                    for ident in message.asks:
                        row = rows.get(ident)
                        if row is None or (row.expires_at is not None and row.expires_at <= now):
                            continue
                        if ident in weighty and read.arrived_at - message.at > timedelta(hours=w.FOLLOW_UP_HOURS):
                            continue
                        eligible.append((message.at, ident))
                if eligible:
                    _, ident = max(eligible, key=lambda item: item[0])
                    row = rows[ident]
                    shown_id = self._show_ask(row, sentences[ident], key=read.key, now=now,
                                              why="an unbound answer; the most recently shown eligible ask in this chat",
                                              name=names.get(str(row.entity_id)),
                                              chat=(read.platform, read.chat_id))
                    result["shown"].append({"id": ident, "sentence": sentences[ident], "message": shown_id})
        if result["settled"] and (binding.path == w.REACTION or read.route == "ingest"):
            # No model turn acknowledges a tap, and a message read at ingest was replied to before it was read.
            result["acknowledged"] = self._acknowledge(result, key=read.key)
        read.kind = (answer_log.ANSWERED if result["settled"] or result["refused"]
                     else answer_log.CLARIFIED if result["clarify"] else answer_log.UNBOUND if result["unbound"]
                     else answer_log.SHOWN if result["shown"] else answer_log.LATER if result["later"]
                     else answer_log.CONVERSATION)
        result["context"] = self._answer_context(result, [sentences[row.id] for row in open_rows])
        return result

    @staticmethod
    def _answer_record(result: Mapping[str, Any]) -> Dict[str, Any]:
        """What ``answer_turns`` keeps of a read: the path, the verdict, ids and statuses; never the owner's words,
        an ask's words or the context."""
        return {"path": result.get("path"), "verdict": (result.get("reading") or {}).get("verdict"),
                "asks": list(result.get("asks") or []),
                "settled": [{"id": item["id"], "status": item["status"]} for item in result.get("settled") or []],
                "approved": list(result.get("approved") or []),
                "allowed": [item["allow"] for item in result.get("allowed") or []],
                "later": [item["id"] for item in result.get("later") or []],
                "shown": [item["id"] for item in result.get("shown") or []],
                "clarify": list((result.get("clarify") or {}).get("asks") or []),
                "undone": [{"id": item["id"], "undone": item["undone"]} for item in result.get("undone") or []],
                "revoked": [item["allow"] for item in result.get("revoked") or []],
                "refused": [item["id"] for item in result.get("refused") or []],
                "unbound": bool(result.get("unbound"))}

    def _open_asks(self) -> List[StoredInitiative]:
        """The asks an owner message may answer: open, with an ask id; never a question the owner answers in
        words (``questions.py``)."""
        return [row for row in self.store.intentions(status=["asked"], limit=500)
                if row.ask_code and row.type not in QUESTION_TYPES]

    def _weighty(self, row: StoredInitiative) -> bool:
        """A floor ask or one whose yes widens who gets messaged, who is who or what may run without asking: read
        strict, bound by follow-up only within ``answers.FOLLOW_UP_HOURS``, never approved by a list."""
        return row.cls == "floor" or row.type in self.STRICT_ASKS

    def _coded_rows(self, open_rows: List[StoredInitiative]) -> Dict[str, List[StoredInitiative]]:
        """The asks by their code, from before the oldest open one (a 1.11 notice names its asks by code; a code
        is unique only among open asks, so one reused later never stands for a notice's older ask)."""
        since = min((_utc(row.created_at) for row in open_rows if row.created_at), default=self.clock())
        found: Dict[str, List[StoredInitiative]] = {}
        for row in self.store.intentions(since=since - timedelta(days=4), limit=2000):
            if row.ask_code:
                found.setdefault(str(row.ask_code).upper(), []).append(row)
        return found

    def _asks_shown_by(self, message: StoredInitiative, coded: Mapping[str, List[StoredInitiative]]
                       ) -> tuple[str, ...]:
        """The asks one owner message showed, in its order: its ``list`` or ``asks``, the ask whose full text it
        was, or the codes a 1.11 notice lists (a code not placed keeps its place, so numbers keep their
        meaning)."""
        context = message.context if isinstance(message.context, dict) else {}
        listed = context.get("list") or context.get("asks")
        if isinstance(listed, list) and listed:
            return tuple(str(item) for item in listed)
        if context.get("ask_id"):
            return (str(context["ask_id"]),)
        if message.type != "ask_notice":
            return ()
        created = _utc(message.created_at)
        ids = []
        for line in listed_lines(str(context.get("text") or ""), NEEDS_OK_LABEL):
            code = line.split(":", 1)[0].strip().upper()
            row = next((item for item in coded.get(code, [])
                        if created is None or _utc(item.created_at) is None or _utc(item.created_at) <= created),
                       None)
            ids.append(row.id if row is not None else "")
        return tuple(ids)

    def _sent_chat(self, message: StoredInitiative) -> tuple[str, str]:
        """``(platform, chat)`` an owner message was delivered to: the target the body claimed it for."""
        target = next((item.details.get("target") for item in self.store.get_history(message.id, limit=50)
                       if item.action == "sending" and isinstance(item.details, dict)), "")
        return split_target(str(target or ""))

    def _shown_messages(self, open_rows: List[StoredInitiative], coded: Mapping[str, List[StoredInitiative]]
                        ) -> List[answer_words.Shown]:
        """Every delivered message that put an open ask in front of the owner, with the platform, the chat and its
        delivery id there: the owner messages the body reported sent (``ASK_MESSAGES``), and the shows the plugin
        recorded (``record_shown``)."""
        if not open_rows:
            return []
        since = min((_utc(row.created_at) for row in open_rows if row.created_at), default=self.clock())
        open_ids = {row.id for row in open_rows}
        shown: List[answer_words.Shown] = []
        for message in self.store.intentions(status=["sent"], kind=["message"], types=list(ASK_MESSAGES),
                                             recipient=self.owner_id, since=since, limit=500):
            asks = self._asks_shown_by(message, coded)
            at = _utc(message.completed_at)
            if asks and message.hermes_ref and at is not None and open_ids.intersection(asks):
                shown.append(answer_words.Shown(*self._sent_chat(message), str(message.hermes_ref), at, asks))
        for row in open_rows:
            for item in self.store.get_history(row.id, limit=200):
                details = item.details if isinstance(item.details, dict) else {}
                if item.action == ASK_SHOWN and details.get("message_id"):
                    listed = tuple(str(ident) for ident in details.get("list") or [row.id])
                    at = _utc(details.get("at")) or _utc(item.timestamp)
                    if at is not None:
                        shown.append(answer_words.Shown(str(details.get("platform") or ""),
                                                        str(details.get("chat") or ""), str(details["message_id"]),
                                                        at, listed))
        return shown

    def _answer_ledger(self, entries: Any, shown: List[answer_words.Shown],
                       coded: Mapping[str, List[StoredInitiative]], *, platform: str, chat: str) -> Optional[tuple]:
        """The plugin's ledger of the message's chat (``platform``, ``chat``) as ``answers.Seen`` entries. What an
        entry showed is the sidecar's own record, never the plugin's word: an outbox send is read from its row (a
        sent owner message that shows asks, delivered to this very chat), a message id from the deliveries it knows
        in this chat; anything else (a model reply, an interim message, a model send, a cron delivery, a send to
        another chat) showed no ask here and breaks the follow-up. None: no ledger (the ingest read)."""
        if entries is None:
            return None
        here = (platform, chat) if platform and chat else None
        by_message = {item.message_id: item.asks for item in shown if (item.platform, item.chat) == here}
        seen = []
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, Mapping):
                continue
            at = _utc(entry.get("at"))
            if at is None:
                continue
            asks: tuple = ()
            outbox_id = str(entry.get("outbox_id") or "")
            if outbox_id:
                row = self.store.get(outbox_id)
                if (row is not None and row.kind == "message" and row.status == "sent" and row.type in ASK_MESSAGES
                        and self._is_owner(row.entity_id) and here is not None and self._sent_chat(row) == here):
                    asks = self._asks_shown_by(row, coded)
            elif entry.get("message_id"):
                asks = by_message.get(str(entry["message_id"]), ())
            seen.append(answer_words.Seen(at, asks, bool(entry.get("ordered"))))
        return tuple(seen)

    async def _ask_names(self, open_rows: List[StoredInitiative]) -> Dict[str, List[str]]:
        """What names each open ask in the owner's words (``answers.named``): the contact it is about, by the name
        the owner knows (and a first name), and the target of a held effect (a file or folder, a payee)."""
        contacts: Dict[str, str] = {}
        for row in open_rows:
            context = row.context if isinstance(row.context, dict) else {}
            change = context.get("change") if isinstance(context.get("change"), dict) else {}
            contact = str(change.get("contact_id") or (row.entity_id if row.kind == "message" else "") or "")
            if contact and not self._is_owner(contact):
                contacts[row.id] = contact
        shown = await self._names(contacts.values())
        names: Dict[str, List[str]] = {}
        for row in open_rows:
            values: List[str] = []
            name = shown.get(contacts.get(row.id, ""), "")
            if name and name != "a contact":
                values += [name, name.split()[0]] if len(name.split()) > 1 else [name]
            context = row.context if isinstance(row.context, dict) else {}
            effect = context.get("effect") if isinstance(context.get("effect"), dict) else {}
            target = " ".join(str(effect.get("target") or "").split())
            if target:
                values += [target, target.rstrip("/").rsplit("/", 1)[-1]]
            names[row.id] = values
        return names

    def _show_ask(self, row: StoredInitiative, sentence: str, *, key: str, now: datetime, why: str,
                  name: str | None = None, chat: tuple[str, str] | None = None) -> Optional[str]:
        """Put one open ask in front of the owner again as its own message (``ask_show``), which then binds like
        the ask's own message; one waiting to go is not doubled. The id of the message."""
        waiting = next((message for message in self.store.intentions(status=["approved", "sending"],
                                                                      kind=["message"], types=[ASK_SHOW], limit=100)
                        if row.id in ((message.context or {}).get("asks") or [])
                        and (chat is None or self._owner_chat(message) == ":".join(chat))), None)
        if waiting is not None:
            return waiting.id
        reply = plain.reply_line(always=always_words(row, name)) if self._offer(row) is not None else plain.REPLY
        message = self.outbox.notice(type=ASK_SHOW, title=f"Asking again: {row.description or row.type}"[:160],
                                     text=f"{plain.ended(sentence)} {reply}",
                                     dedup_key=f"{ASK_SHOW}:{row.id}:{key}",
                                     extra={"asks": [row.id], "list": [row.id], "answer_turn": key})
        if message is None:
            return None
        self.store.transition(row.id, row.status, action=ASK_SHOWN, at=now,
                              details={"message": message.id, "why": why, "answer": key})
        return message.id

    def _acknowledge(self, result: Mapping[str, Any], *, key: str) -> Optional[str]:
        """One line to the owner saying what their answer did, for an answer no model turn acknowledges: approved,
        declined, undone, or too late to undo. The id of the message (``ask_ack``), or None."""
        def shown(sentence: Any) -> str:
            return plain.fit(str(sentence), 300) or plain.pointer("The ask")
        approved = set(result.get("approved") or [])
        lines = []
        for item in result.get("settled") or []:
            if item.get("id") in approved:
                lines.append(f"OK, approved: {shown(item['sentence'])}")
            elif item.get("status") == "cancelled":
                lines.append(f"OK, declined: {shown(item['sentence'])}")
        for item in result.get("undone") or []:
            lead = "Undone:" if item.get("undone") in {"cancelled", "reverted"} else "Could not undo:"
            lines.append(f"{lead} {shown(item['sentence'])} {plain.ended(item.get('what') or '')}".strip())
        for item in result.get("revoked") or []:
            lines.append(f"OK, I ask again before I {plain.ended(item['again'] + item['left'])}")
        if not lines:
            return None
        asks = list(dict.fromkeys(str(item["id"]) for group in ("settled", "undone")
                                  for item in result.get(group) or [] if item.get("id")))
        message = self.outbox.notice(type=ASK_ACK, title="Your answer to an ask", text=" ".join(lines),
                                     dedup_key=f"{ASK_ACK}:{key}", extra={"answer_turn": key, "asks": asks})
        return message.id if message is not None else None

    def _clarified(self, row: StoredInitiative, now: datetime) -> bool:
        since = now - timedelta(hours=answer_words.CLARIFY_HOURS)
        return any(item.action == ASK_CLARIFIED and _utc(item.timestamp) is not None and _utc(item.timestamp) >= since
                   for item in self.store.get_history(row.id, limit=200))

    def _clarify(self, rows: List[StoredInitiative], sentences: Mapping[str, str], *, key: str,
                 now: datetime) -> Optional[Dict[str, Any]]:
        """The one plain question about an unclear answer, as its own message (``ask_clarify``): "Just to check:
        <the ask>? Yes or no." for one ask, "Which one? 1. ... 2. ..." over several, whose numbers mean that list.
        At most one per ask in ``answers.CLARIFY_HOURS``; after that nothing is asked and the ask stays open."""
        fresh = [row for row in rows if not self._clarified(row, now)]
        if not fresh:
            return None
        if len(fresh) == 1:
            lead = sentences[fresh[0].id].strip().rstrip(".").strip()
            if len(lead) > 1 and lead[0].isupper() and lead[1].islower():
                lead = lead[0].lower() + lead[1:]
            text = f"Just to check: {lead}? Yes or no."
        else:
            numbered = " ".join(f"{number}. {plain.ended(sentences[row.id])}" for number, row in enumerate(fresh, 1))
            reply = "Reply 1, 2, both or neither." if len(fresh) == 2 else "Reply with the numbers, all or none."
            text = f"Which one? {numbered} {reply}"
        ids = [row.id for row in fresh]
        message = self.outbox.notice(type=ASK_CLARIFY, title=f"Checking {len(ids)} ask(s)", text=text,
                                     dedup_key=f"{ASK_CLARIFY}:{key}",
                                     extra={"asks": ids, "list": ids, "answer_turn": key})
        if message is None:
            return None
        for row in fresh:
            self.store.transition(row.id, row.status, action=ASK_CLARIFIED, at=now,
                                  details={"message": message.id, "answer": key})
        return {"asks": ids, "message": message.id, "text": text}

    # -- Instinct's ``ask_answer``: a model's reading of an unclear answer (docs/DECISIONS.md, ``ask_answer``) --------

    #: The tiered confirm-back: after this many right model readings of a yes in a row for one kind of ask, a model's
    #: yes approves that kind without asking back (a starting choice). A mistake starts the count again; another
    #: model, revision or wording starts it at zero.
    CONFIRM_TRUST = 20
    #: How far back, and over how many asks of the kind, the readings are counted (starting choices: past any count
    #: of 20 at a few asks a week, and bounded).
    CONFIRM_TRUST_WINDOW = timedelta(days=90)
    CONFIRM_TRUST_ASKS = 500

    async def _typed_answer(self, rows: Mapping[str, StoredInitiative], plan: Any, reading: Any, binding: Any,
                            text: str, *, weighty: Iterable[str], read: answer_log.Read, sentences: Mapping[str, str],
                            now: datetime) -> Optional[Dict[str, Any]]:
        """Instinct's ``ask_answer`` point on an answer the tables read unclear, where one ordinary ask (never a floor
        or widening one, never a list) is bound by a quoted reply or the follow-up and the words hold a yes or no
        first and no hard hedge. A new plan for the answer, or None: the tables' unclear path (one plain question).

        - a live model's ``decline`` settles no (read ``decline`` by the decision), unless the message carries someone
          else's words or the ask's no could not be taken back (``_reopenable``; the owner takes one back with undo
          words or "I didn't say no", ``_reopen_ask``);
        - a live model's ``conditional_or_question`` or ``unrelated`` settles nothing and asks nothing: the message goes
          to the model as usual;
        - a live model's ``approve`` (withheld: no model tier settles a yes) is asked back as its own message
          (``_confirm_back``). Only where that kind of ask has earned it (``_model_yes_trusted``), is not one that
          always confirms (``_confirms_always``) and the message is all the owner's own words, it approves;
        - in shadow, below the bars, down or without a record: None."""
        w = answer_words
        weighty = set(weighty)
        if (len(plan.clarify) != 1 or binding.listed or binding.path not in {w.REPLY, w.FOLLOW_UP}
                or reading.verdict != w.UNCLEAR or not reading.answer_word or reading.hard_hedge
                or plan.clarify[0] in weighty or not str(text or "").strip()):
            return None
        row = rows[plan.clarify[0]]
        # Recover the name from the exact sentence already built, without another contact lookup.
        name = "a contact"
        if audit.asked_words(row) is not None:
            suffix = ask_sentence(row, name=name)[len("Send a contact"):]
            sentence = sentences[row.id]
            if sentence.startswith("Send ") and sentence.endswith(suffix):
                name = sentence[len("Send "):-len(suffix)]
        # The answer route reads the owner's message before any turn is captured, so nothing sealed it yet: every
        # secret is hidden before a model reads the words (``audit.hidden_text``), live or in shadow.
        decision = await (self.instinct or instinct_core.shared()).decide(
            "ask_answer", {"ask": audit.hidden_text(sentences[row.id]), "words": audit.hidden_text(text)},
            {"ask": row.id, "turn": read.turn_id or None, "message": read.key,
             "name": instinct_core.Hashed(name)}, event="ask_answer")
        answer = {"path": binding.path, "reader": "decision", "message": read.key, "by": "owner",
                  "decision": decision.id}
        own = self._own_words_only(text)
        if decision.applied and decision.tier in {"fast", "reader"}:
            if decision.label == "decline":
                # Only a no the owner can take back (``_reopen_ask``), and only in their own words.
                return {"plan": w.Plan(no=(row.id,)), "details": {"answer": {**answer, "read": "decline"}}} \
                    if own and self._reopenable(row) else None
            return {"plan": w.Plan(), "details": {"answer": {**answer, "read": decision.label}}}
        if decision.withheld != "approve":
            return None
        answer.update(read="approve", fingerprint=decision.fingerprint)
        if own and not self._confirms_always(row) and self._model_yes_trusted(row, decision.fingerprint, now):
            return {"plan": w.Plan(yes=(row.id,)), "details": {"answer": answer}}
        return {"plan": w.Plan(), "details": {"answer": answer}, "confirm": row.id}

    def _asked_back(self, rows: Iterable[StoredInitiative]) -> bool:
        """Whether any of these asks had a model's yes asked back (``_confirm_back``): the answer settling it grades
        that reading (``outcomes.model_yes``)."""
        return any(item.action == ASK_CLARIFIED and isinstance(item.details, dict)
                   and item.details.get("reader") == "decision"
                   for row in rows for item in self.store.get_history(row.id, limit=200))

    @staticmethod
    def _own_words_only(text: str) -> bool:
        """Whether the owner's message is all their own words: nothing quoted, reported or framed as someone else's
        (``reactions.own_words``, read strict). Only such a message is settled on a model's reading."""
        body = reactions.strip_prefix(str(text or ""))
        return reactions.own_words(body, strict=True).split() == body.split()

    def _confirms_always(self, row: StoredInitiative) -> bool:
        """A kind of ask whose yes a model read is always asked back, however many readings were right (the owner's
        rule for what is outward or cannot be taken back): anything but the mind's own task (a message leaves the
        house; a sealed value's use or a change to who is who is the owner's to give), a task that holds an effect
        or a change, and a floor or widening ask (which never reaches the point at all)."""
        context = row.context if isinstance(row.context, dict) else {}
        return row.kind != "task" or bool(context.get("effect") or context.get("change")) or self._weighty(row)

    def _model_yes_trusted(self, row: StoredInitiative, fingerprint: str, now: datetime) -> bool:
        """Whether a model's yes has earned this kind of ask (its kind and type) an approval without asking back: the
        last ``CONFIRM_TRUST`` graded readings of a yes made under this fingerprint (``outcomes.model_yes``: asked
        back and confirmed, or approved and standing a day) were all right. The history is read as it is now, so an
        undo after a confirmation spends what the confirmation earned; only the owner's own label on a decision
        (``colonymind instinct label``, ``_owner_graded``) outranks it, never a grade a resolver kept from an earlier
        read of the history. A reading counts only while its decision is still in the decision log: one forgotten
        there (a turn it read or that graded it erased, or past the log's retention) earns nothing and breaks the
        count, whichever way it was graded, so a forgotten mistake never joins the rights around it into approval;
        without a log nothing is earned."""
        from colonymind.instinct.fit import strongest
        from colonymind.instinct.log import OWNER
        from colonymind.instinct.outcomes import model_yes
        store = getattr(self.instinct or instinct_core.shared(), "store", None)
        if not fingerprint or store is None:
            return False
        kept = {item["id"] for item in store.rows(point="ask_answer", since=now - self.CONFIRM_TRUST_WINDOW)}
        said = strongest(item for item in store.outcomes(point="ask_answer") if item["resolver"] == OWNER)
        graded: List[tuple] = []
        for ask in self.store.intentions(kind=[row.kind], types=[row.type], since=now - self.CONFIRM_TRUST_WINDOW,
                                         limit=self.CONFIRM_TRUST_ASKS):
            history = [(item.action, item.timestamp, item.details if isinstance(item.details, dict) else {})
                       for item in self.store.get_history(ask.id, limit=200)]
            for item in model_yes(history, now=now):
                decision = str(item["decision"] or "")
                # A grade a model read of the owner's reply (``interpreted``) neither earns nor spends the trust.
                verdict = self._owner_graded(said.get(decision)) or (
                    item["verdict"] if item["source"] != "interpreted" else None)
                if decision not in kept:
                    verdict = "forgotten" if verdict else None   # erased evidence is never rebuilt into approval
                if verdict and item["fingerprint"] == fingerprint:
                    graded.append((item["at"], verdict))
        streak = 0
        for _, verdict in sorted(graded, key=lambda entry: entry[0]):
            streak = streak + 1 if verdict == "right" else 0
        return streak >= self.CONFIRM_TRUST

    def _model_answers(self, read: answer_log.Read) -> List[answer_log.Read]:
        """The answers in ``read``'s chat that settled an ask on a model's reading of the owner's words, not undone,
        newest first, within the day a model's approval takes to stand (``outcomes.STANDS``): what "I didn't say yes"
        (or no) still takes back past the undo's minutes."""
        from colonymind.instinct.outcomes import STANDS
        return [prior for prior in self.answer_turns.answered(platform=read.platform, chat_id=read.chat_id,
                                                              since=read.arrived_at - STANDS, before=read.arrived_at)
                if prior.reader == "decision"]

    @staticmethod
    def _declined_by_model(prior: answer_log.Read) -> List[str]:
        """The asks an answer settled no on a model's reading of the owner's words (``_typed_answer``'s decline)."""
        if prior.reader != "decision":
            return []
        approved = {str(ident) for ident in prior.result.get("approved") or []}
        return [str(item.get("id")) for item in prior.result.get("settled") or []
                if item.get("status") == "cancelled" and str(item.get("id")) not in approved]

    @staticmethod
    def _reopenable(row: StoredInitiative) -> bool:
        """Whether a no to this ask is only recorded, so it can be taken back (``_reopen_ask``): an owner question's
        no does more (a wish's no holds its kind for weeks), so a model's no is never applied to one."""
        return row.type not in OWNER_QUESTIONS and row.type != GRANT_CONFIRM

    async def _reopen_ask(self, row: StoredInitiative, *, now: datetime, answer: str, turn: str = "") -> Dict[str, Any]:
        """A no a model read from the owner's words, which they took back: the ask is open again (``ask_undone``, then
        ``ask_revived``, with the outcome and verdict the no set cleared) and shown to them again as its own message
        (``_show_ask``), so their next plain answer settles it; the resolver grades the reading wrong
        (``outcomes.model_no``). One that changed since or lapsed is left, and said plainly."""
        from .asks_report import ASK_REVIVED
        sentence = plain.ended(row.description or row.type or "the ask")
        current = self.store.get(row.id) or row
        if current.status != "cancelled" or current.outcome != "denied" or not self._reopenable(current):
            done, what = "too_late", f"it is already {current.status}"
        elif current.expires_at is not None and (_utc(current.expires_at) or now) <= now:
            done, what = "too_late", "the ask had lapsed, so it stays closed"
        else:
            done, what = "reverted", "the no is taken back: the ask is open again and goes to them again on its own"
        self.store.transition(row.id, current.status, action=ASK_UNDONE, at=now,
                              details={"by": "owner", "message": answer, "undone": done, "what": what,
                                       **({"turn": turn} if turn else {})})
        if done == "reverted":
            reopened = self.store.transition(row.id, "asked", action=ASK_REVIVED, at=now,
                                             details={"by": "owner", "message": answer}, outcome=None, verdict=None,
                                             verified=None, result=None, cancelled_at=None, cancelled_by=None,
                                             cancelled_reason=None) or current
            names = await self._names([reopened.entity_id] if audit.asked_words(reopened) is not None else [])
            self._show_ask(reopened, ask_sentence(reopened, name=names.get(str(reopened.entity_id))), key=answer,
                           now=now, why="the owner took back a no a model read from their words")
        return {"id": row.id, "sentence": sentence, "undone": done, "what": what}

    @staticmethod
    def _owner_graded(outcome: Optional[Mapping[str, Any]]) -> Optional[str]:
        """The owner's own label on a decision (``outcomes.label``) as a grade of the model's yes: right only when the
        right label there was ``approve``; None without one."""
        if outcome is None or outcome.get("source") != "explicit":
            return None
        if outcome.get("label") is not None:
            return "right" if outcome["label"] == "approve" else "wrong"
        return "wrong" if outcome.get("verdict") == "wrong" else None

    def _confirm_back(self, row: StoredInitiative, sentence: str, answer: Mapping[str, Any], *, key: str,
                      now: datetime) -> Optional[Dict[str, Any]]:
        """A model's yes asked back, as its own message (``ask_clarify``): "Doing this: <the ask>, OK?". It is the
        ask's one plain question of the day (``_clarified``): after one, nothing is asked and the ask stays open. The
        owner's next plain answer binds to it and settles the ask as the tables read it."""
        if self._clarified(row, now):
            return None
        lead = sentence.strip().rstrip(".").strip()
        if len(lead) > 1 and lead[0].isupper() and lead[1].islower():
            lead = lead[0].lower() + lead[1:]
        text = f"Doing this: {lead}, OK?"
        message = self.outbox.notice(type=ASK_CLARIFY, title="Checking 1 ask(s)", text=text,
                                     dedup_key=f"{ASK_CLARIFY}:{key}",
                                     extra={"asks": [row.id], "list": [row.id], "answer_turn": key})
        if message is None:
            return None
        self.store.transition(row.id, row.status, action=ASK_CLARIFIED, at=now,
                              details={"message": message.id, "answer": key, "reader": "decision", "read": "approve",
                                       "decision": answer.get("decision"), "fingerprint": answer.get("fingerprint")})
        return {"asks": [row.id], "message": message.id, "text": text}

    async def _undo_answer(self, read: answer_log.Read, *, text: str, reaction: str, removed: bool, by: str,
                           now: datetime) -> List[Dict[str, Any]]:
        """The owner taking back an approval inside ``answers.UNDO_MINUTES``: a reaction that approved, changed to
        a no or removed; or undo words ("undo", "cancel that", "wait, no") in the chat the approval was read in (the
        latest one there). "I didn't say yes" (``answers.repudiated``) is such words too, and past those minutes it
        still takes back the latest approval in the chat that a model read from the owner's words (``_typed_answer``),
        within the day such an approval takes to stand (``outcomes.STANDS``). A no a model read is taken back the same
        way: undo words when it is the latest answer there, or "I didn't say no" within that day (``_reopen_ask``: the
        ask is open again). Only an answer whose message arrived before the undo's own is taken back, whatever order
        the two are read in. Each ask it approved is undone (``_undo_ask``); nothing else is read. Undo words take
        back nothing when an ask was approved in an owner channel's app after the latest approval in the chat, inside
        those minutes (``_approved_in_app``): they would undo the wrong ask, and an app answer is not undone from a
        chat; the owner is told so."""
        since = read.arrived_at - timedelta(minutes=answer_words.UNDO_MINUTES)
        approvals: List[answer_log.Read] = []
        worded = False
        said = answer_words.repudiated(text) if not reaction and not removed and read.chat_id else ""
        if read.target and (removed or reaction in answer_words.NO_REACTIONS):
            approvals = self.answer_turns.approved_by_reaction(platform=read.platform, chat_id=read.chat_id,
                                                               target=read.target, since=since, before=read.arrived_at)
        elif said == "no":
            approvals = [prior for prior in self._model_answers(read) if self._declined_by_model(prior)][:1]
        elif not reaction and not removed and read.chat_id and (answer_words.is_undo(text) or said == "yes"):
            # The latest answer there that approved, or (undo words) whose no a model read.
            worded = True
            approvals = [prior for prior in self.answer_turns.answered(platform=read.platform, chat_id=read.chat_id,
                                                                       since=since, before=read.arrived_at)
                         if prior.result.get("approved") or (not said and self._declined_by_model(prior))][:1]
            if not approvals and said == "yes":
                approvals = [prior for prior in self._model_answers(read) if prior.result.get("approved")][:1]
        if worded:
            # Undo words name no ask: when an ask was approved in an owner channel's app after the latest approval
            # here (inside the undo's minutes), they would take back the wrong one. Nothing is undone, and the owner
            # is told why: an app answer is not taken back from a chat (``_tell_answered`` says so too).
            latest = approvals[0].arrived_at if approvals else None
            elsewhere = self._approved_in_app(since=since, before=read.arrived_at, after=latest)
            if elsewhere:
                return elsewhere
        undone: List[Dict[str, Any]] = []
        for prior in approvals:
            for ident in prior.result.get("approved") or []:
                row = self.store.get(str(ident))
                if row is not None:
                    undone.append(await self._undo_ask(row, by=by, now=now, answer=read.key, turn=read.turn_id))
            for ident in self._declined_by_model(prior):
                row = self.store.get(ident)
                if row is not None:
                    undone.append(await self._reopen_ask(row, now=now, answer=read.key, turn=read.turn_id))
            self.answer_turns.mark_undone(prior.key, now)
        return undone

    def _approved_in_app(self, *, since: datetime, before: datetime,
                         after: Optional[datetime]) -> List[Dict[str, Any]]:
        """The asks approved by an explicit answer (an owner channel's app) between ``since`` and ``before``, and
        after ``after`` when given, as undo results that undid nothing (``undone: elsewhere``)."""
        values: List[Dict[str, Any]] = []
        for item in self.store.answer_requests_since(since):
            response = item["response"] if isinstance(item.get("response"), dict) else {}
            at = item["created_at"]
            if (response.get("result") != "settled" or response.get("status") != "approved" or at >= before
                    or (after is not None and at <= after)):
                continue
            row = self.store.get(str(item["ask_id"]))
            sentence = plain.ended((row.description or row.type) if row is not None else "the ask")
            values.append({"id": str(item["ask_id"]), "sentence": sentence, "undone": "elsewhere",
                           "what": "it was answered in the app, and an answer given there is not taken back from a "
                                   "chat"})
        return values

    async def _undo_ask(self, row: StoredInitiative, *, by: str, now: datetime, answer: str,
                        turn: str = "") -> Dict[str, Any]:
        """Undo one approval: work not started yet (an approved task, message or goal, or a permit not used yet,
        such as a sealed value's use waiting for the tool's retry) is cancelled; a standing grant is withdrawn; a
        raised permission or cadence goes back to what it was (compare-and-set). Anything already under way (a
        permit already used included) is left, and said plainly. Recorded on the ask (``ask_undone``, with the undo's
        own ``turn``: it grades a model's reading, ``outcomes.model_yes``); never a verdict."""
        sentence = plain.ended(row.description or row.type or "the ask")
        done, what = "too_late", ""
        if row.status == "approved" and row.kind in {"task", "message", "goal", "note"}:
            self.outcomes.record(row.id, status="cancelled", by=by, implicit_verdict=False,
                                 summary="the owner undid the approval before anything started")
            done, what = "cancelled", "nothing had started, so nothing will happen"
        elif row.type == GRANT_CONFIRM and row.outcome == "done":
            grant = parse_grant((row.context or {}).get("grant") if isinstance(row.context, dict) else None,
                                source=f"ask {row.ask_code}")
            if grant.id and self.mind_state.get(f"{GRANT_KEY}{grant.id}") is not None:
                self.mind_state.delete(f"{GRANT_KEY}{grant.id}")
                self._grant_note("grant_revoked", grant, f"standing grant {grant.id} undone by the owner ({by})")
                done, what = "reverted", "the standing grant is withdrawn"
            else:
                what = "the standing grant was already gone"
        elif row.type == "people_change" and row.outcome == "done":
            done, what = await self._undo_people_change(row)
        else:
            what = {"dispatched": "it had already started", "running": "it had already started",
                    "sending": "it was already on its way", "sent": "it was already sent",
                    "done": "it was already done"}.get(str(row.status), f"it is already {row.status}")
        added = [str((item.details or {}).get("allow") or "") for item in self.store.get_history(row.id, limit=200)
                 if item.action == ALLOW_ADDED]
        for ident in added:
            if ident and self.allows.get(ident) is not None:
                await self.revoke_allow(ident, by=by)
                done = "reverted" if done == "too_late" else done
                what = f"{what}; and I ask again from now on" if what else "I ask again from now on"
        current = self.store.get(row.id) or row
        self.store.transition(row.id, current.status, action=ASK_UNDONE, at=now,
                              details={"by": "owner", "message": answer, "undone": done, "what": what,
                                       **({"turn": turn} if turn else {})})
        return {"id": row.id, "sentence": sentence, "undone": done, "what": what}

    async def _undo_people_change(self, row: StoredInitiative) -> tuple[str, str]:
        """A raised permission or cadence set back to what the ask showed, only while the record still holds the
        value the yes set (compare-and-set); a merge is never undone here."""
        from colonymind.contacts.store import ContactChanged
        change = (row.context or {}).get("change") if isinstance(row.context, dict) else None
        change = change if isinstance(change, dict) else {}
        operation, contact_id = str(change.get("operation") or ""), str(change.get("contact_id") or "")
        before = (change.get("from") or {}).get(contact_id) if isinstance(change.get("from"), dict) else None
        if operation not in {"set_permission", "set_cadence"} or self.contacts is None or not isinstance(before, dict):
            return "too_late", "it was already applied and cannot be undone here"
        performed = f"owner-undo:{row.ask_code}"
        try:
            if operation == "set_permission":
                await self.contacts.set_may_contact(contact_id, str(before.get("may_contact")), by=performed,
                                                    reason=f"the owner undid ask {row.ask_code}",
                                                    expected=str(change.get("value")))
            else:
                await self.contacts.set_cadence(contact_id, before.get("cadence_minutes"), by=performed,
                                                expected=change.get("minutes"))
        except ContactChanged:
            return "too_late", "the record changed since, so it was left as it is"
        except Exception as error:
            logger.warning("people change %s not undone (%s)", row.id, type(error).__name__)
            return "too_late", f"it could not be undone ({type(error).__name__})"
        return "reverted", "the change is set back"

    @staticmethod
    def _answer_context(result: Mapping[str, Any], open_sentences: List[str]) -> str:
        """What the chat model reads before it replies to the owner's message: what the message did to the asks,
        in plain words, so it acknowledges once and never asks again (or knows the message answered nothing)."""
        def listed(items: Iterable[Any]) -> str:
            return "; ".join(plain.fit(str(item), 400) or plain.pointer("An ask") for item in items)
        parts: List[str] = []
        settled = list(result.get("settled") or [])
        approved = [item for item in settled if item.get("id") in set(result.get("approved") or [])]
        declined = [item for item in settled if item.get("status") == "cancelled" and item not in approved]
        ended = [item for item in settled if item not in approved and item not in declined]
        if approved:
            parts.append(f"The owner's reply approved: {listed(item['sentence'] for item in approved)} It is under "
                         "way; acknowledge in a few words and do not ask again.")
        if declined:
            parts.append(f"The owner's reply declined: {listed(item['sentence'] for item in declined)} Nothing of it "
                         "is done; acknowledge in a few words.")
        for item in ended:
            parts.append(f"The owner's reply came too late or no longer applies ({item['status']}): "
                         f"{listed([item['sentence']])}")
        for item in result.get("allowed") or []:
            allowed = plain.ended(str(item["words"]).removeprefix("let me "))
            parts.append(f"The owner also said always: from now on I {allowed} Acknowledge it in a few words.")
        for item in result.get("refused") or []:
            parts.append(f"Not approved yet: {listed([item['sentence']])} {plain.ended(item['reason'])}")
        if result.get("later"):
            parts.append(f"The owner said later to: {listed(item['sentence'] for item in result['later'])} It stays "
                         "open; do not ask about it again now.")
        if result.get("clarify"):
            parts.append("The owner's reply did not say a plain yes or no; one short question goes to them as its own "
                         "message after your reply. Do not repeat it.")
        if result.get("shown") and not result.get("unbound"):
            parts.append(f"This goes to the owner again as its own message after your reply, since their words cannot "
                         f"approve it this way: {listed(item['sentence'] for item in result['shown'])} Do not repeat "
                         "it.")
        for item in result.get("undone") or []:
            if item.get("undone") == "elsewhere":
                parts.append(f"The owner's undo took back nothing: their latest yes was to "
                             f"{listed([item['sentence']])} {plain.ended(item['what'])} Tell them in a few words.")
                continue
            parts.append(f"The owner undid their answer to: {listed([item['sentence']])} {plain.ended(item['what'])} "
                         "Acknowledge in a few words.")
        for item in result.get("revoked") or []:
            parts.append(f"The owner asked to be asked again: from now on I ask again before I "
                         f"{plain.ended(item['again'] + item['left'])} Acknowledge it in a few words.")
        if result.get("unbound"):
            parts.append(f"This message was not read as an answer to the open ask(s): {listed(open_sentences)} "
                         "Never treat these words as approving it.")
            if result.get("shown"):
                recovered = result["shown"][0]
                parts.append(f"The existing ask {recovered['id']} is queued to be shown again in this chat after "
                             f"your reply: {listed([recovered['sentence']])} Do not repeat it. "
                             "The owner can answer that message with a fresh yes or no here. "
                             "Do not start a new hand, create another ask, or send the owner elsewhere to answer.")
            else:
                parts.append("If the owner meant one, list the open asks or ask which one to show again here.")
        return " ".join(parts)

    def record_shown(self, ask_ids: Iterable[str], *, platform: str, message_id: str, chat: str,
                     at: Any = None) -> Dict[str, Any]:
        """The plugin put these open asks in front of the owner as one message (``message_id`` in ``chat`` on
        ``platform``; several: a list, in this order), outside the outbox, and it binds in that chat only
        (``answers.bind``): one ask's message binds like that ask's own, a reaction or a quoted reply on it; a list
        binds a quoted reply (its numbers, for ``answers.LIST_HOURS``) and never a reaction. Raises ``KeyError`` for
        an id or code that is no open ask, or without the delivery id, and ``ValueError`` without the platform or
        the chat (a delivery id alone names nothing)."""
        platform, chat = str(platform or "").strip().lower(), str(chat or "").strip()
        if not platform or not chat:
            raise ValueError("a shown message is named by its platform, its chat and its delivery id")
        rows = []
        for ident in ask_ids:
            ident = str(ident or "").strip()
            row = self.store.get(ident) if ident else None
            if row is None or not row.ask_code:
                row = self.store.get_by_ask_code(ident.upper()) if ident else None
            if row is None or row.status != "asked" or not row.ask_code or row.type in QUESTION_TYPES:
                raise KeyError(ident)
            rows.append(row)
        if not rows or not str(message_id or "").strip():
            raise KeyError(str(message_id or ""))
        when = min(_utc(at) or self.clock(), self.clock())
        ids = [row.id for row in rows]
        for row in rows:
            self.store.transition(row.id, row.status, action=ASK_SHOWN, at=when,
                                  details={"platform": platform, "chat": chat, "message_id": str(message_id),
                                           "at": when.isoformat(), "list": ids})
        return {"ok": True, "asks": ids, "message_id": str(message_id)}

    def _settle_wish(self, row: StoredInitiative, *, yes: bool, by: str,
                     details: Mapping[str, Any] | None = None) -> Optional[StoredInitiative]:
        """The owner's word on a wish: a yes grants it (the next tick adopts its goal), a no holds its class
        for 30 days. Neither is a verdict on asking: the ranker learns nothing from it."""
        now = self.clock()
        if not self.faculties["wishes"]:
            return self.outcomes.record(row.id, status="cancelled", summary="the wishes faculty is off", by=by,
                                        details=details, implicit_verdict=False)
        if yes:
            updated = self.outcomes.record(row.id, status="done", summary=f"{self._who(by, details)} granted the wish",
                                           verified="owner", by=by, details=details)
            return self.wishes.granted(updated, by=by, now=now) if updated is not None else None
        updated = self.outcomes.record(row.id, status="denied", summary=f"{self._who(by, details)} declined the wish",
                                       by=by, details=details, implicit_verdict=False)
        return self.wishes.denied(updated, by=by, now=now) if updated is not None else None

    async def _settle_cadence(self, row: StoredInitiative, *, yes: bool, by: str,
                              details: Mapping[str, Any] | None = None) -> Optional[StoredInitiative]:
        """The owner's word on a cadence whose contact was matched by name only: a yes makes the
        match the owner's (the cadence is set now, and the row's topic rides on the check-ins); a no
        withdraws the row, so the owner restates it naming the person exactly."""
        ident = str(row.source_id or "")
        try:
            record = self.commitments.get(ident) if self.commitments is not None else None
        except Exception:
            record = None
        metadata = record.get("metadata") if isinstance(record, dict) and isinstance(record.get("metadata"), dict) else {}
        contact_id, minutes = str(metadata.get("recipient_id") or ""), metadata.get("cadence_minutes")
        if not isinstance(record, dict) or record.get("status") not in {"pending", "overdue"} or not contact_id:
            return self.outcomes.record(row.id, status="cancelled", summary="the cadence is no longer open", by=by,
                                        details=details, implicit_verdict=False)
        if metadata.get("cadence_applied"):
            return self.outcomes.record(row.id, status="cancelled", summary="the cadence was already set", by=by,
                                        details=details, implicit_verdict=False)
        if not yes:
            matched = metadata.get("recipient_exact") is not True
            who = self._who(by, details)
            try:
                self.commitments.resolve(ident, outcome="obsolete",
                                         resolved_by="owner" if owner_answer(by, details) else OPERATOR,
                                         note=(f"{who} said {metadata.get('recipient')!r} is not {contact_id}"
                                               if matched else f"{who} did not confirm the cadence"))
            except Exception as error:
                logger.warning("cadence %s not withdrawn (%s)", ident, type(error).__name__)
            said = "the match was wrong" if matched else "no to the cadence"
            return self.outcomes.record(row.id, status="denied", summary=f"{self._who(by, details)} said {said}",
                                        by=by, details=details, implicit_verdict=False)
        try:
            await self.contacts.set_cadence(contact_id, minutes, by=f"owner-turn:commitment:{ident}")
            self.commitments.update(ident, metadata={"recipient_exact": True, "cadence_applied": True})
        except Exception as error:
            logger.warning("cadence %s not applied (%s)", ident, type(error).__name__)
            return self.outcomes.record(row.id, status="cancelled", by=by, details=details, implicit_verdict=False,
                                        summary=f"the cadence could not be set ({type(error).__name__})")
        return self.outcomes.record(row.id, status="done", summary=f"{self._who(by, details)} confirmed the contact",
                                    verified="owner", by=by, details=details)

    async def _settle_people_change(self, row: StoredInitiative, *, yes: bool, by: str,
                                    details: Mapping[str, Any] | None = None) -> Optional[StoredInitiative]:
        """The owner's word on a change to who may be contacted (``request_people_change``): a yes applies
        exactly the change asked about, performed by ``owner-ask:<code>``, while the records are still as
        the ask showed them (``from``: permission, cadence and handles, ``contacts.changes.snapshot``); a
        record changed since, or changing while it is written (the writes are compare-and-set), cancels it and
        the owner is asked again, never assumed. A no changes nothing. Neither teaches the ranker anything."""
        from colonymind.contacts.changes import change_lock, snapshot
        from colonymind.contacts.store import ContactChanged
        change = (row.context or {}).get("change") if isinstance(row.context, dict) else None
        change = change if isinstance(change, dict) else {}
        if not yes:
            return self.outcomes.record(row.id, status="denied",
                                        summary=f"{self._who(by, details)} said no; nothing changed",
                                        by=by, details=details, implicit_verdict=False)

        def cancelled(why: str) -> Optional[StoredInitiative]:
            return self.outcomes.record(row.id, status="cancelled", summary=f"nothing changed: {why}", by=by,
                                        details=details, implicit_verdict=False)
        operation, contact_id = str(change.get("operation") or ""), str(change.get("contact_id") or "")
        performed = f"owner-ask:{row.ask_code}"
        if self.contacts is None:
            return cancelled("no contact store")
        if operation != "set_permission" and not self.faculties["people"]:
            return cancelled("the people faculty is off")
        if operation not in {"set_permission", "set_cadence", "merge"}:
            return cancelled(f"unknown change {operation!r}")
        try:
            async with change_lock(self.contacts):
                contact = await self.contacts.get(contact_id)
                if contact is None:
                    return cancelled(f"{contact_id} is no longer a contact")
                drop = await self.contacts.get(str(change.get("drop") or "")) if operation == "merge" else None
                if operation == "merge" and drop is None:
                    return cancelled("the records changed since you were asked; ask again")
                state = await snapshot(self.contacts, [contact] + ([drop] if drop is not None else []))
                if state != change.get("from"):
                    return cancelled("the records changed since you were asked; ask again")
                # The write compares the value and the handle set in its own statement: a handle attached
                # after the check above (inbound attribution does not wait for the owner) cancels it too.
                if operation == "set_permission":
                    await self.contacts.set_may_contact(contact_id, str(change.get("value")), by=performed,
                                                        reason=f"the owner's yes to ask {row.ask_code}",
                                                        expected=state[contact_id]["may_contact"],
                                                        expected_handles=state[contact_id]["handles"])
                elif operation == "set_cadence":
                    await self.contacts.set_cadence(contact_id, change.get("minutes"), by=performed,
                                                    expected=state[contact_id]["cadence_minutes"],
                                                    expected_handles=state[contact_id]["handles"])
                else:
                    # The fold the people route applies: sources and the other stores follow the person too.
                    from colonymind.api.routers.people import fold_merge
                    await fold_merge(self.contacts, contact_id, drop.contact_id, performed_by=performed, expected=state)
        except ContactChanged:
            return cancelled("the records changed while it was applied; ask again")
        except Exception as error:
            logger.warning("people change %s not applied (%s)", row.id, type(error).__name__)
            return cancelled(f"it could not be applied ({type(error).__name__})")
        return self.outcomes.record(row.id, status="done", summary=f"{self._who(by, details)} said yes and it applied",
                                    verified="owner", by=by, details=details)

    async def request_people_change(self, change: Mapping[str, Any]) -> Optional[StoredInitiative]:
        """A change that widens who may be contacted, asked for in the owner's session (the people route,
        ``contacts.changes``): the owner's one-tap question, a ``people_change`` note to the owner with an ask
        code and the change in its context, whatever the level, settled by ``answer`` on their own reply.
        A pending identical change reuses its ask while it is open; one settled or expired (its time up,
        even before the sweep marks it) is asked again with a fresh code. None when no ask can be formed (no
        owner, no question)."""
        import hashlib
        spec = {key: change.get(key) for key in ("operation", "contact_id", "value", "minutes", "drop", "from")}
        question = " ".join(str(change.get("question") or "").split())
        if not self.owner_id or not question or not spec["contact_id"] or \
                spec["operation"] not in {"set_permission", "set_cadence", "merge"}:
            return None
        digest = hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
        key, now = None, self.clock()
        for attempt in range(1000):
            candidate = f"people_change:{digest}" + (f":{attempt}" if attempt else "")
            existing = self.store.get_by_dedup_key(candidate)
            if existing is None:
                key = candidate
                break
            if existing.status == "asked" and existing.ask_code:
                if existing.expires_at is None or existing.expires_at > now:
                    return existing
                self.outcomes.record(existing.id, status="expired", summary="no answer before the ask expired",
                                     by="mind", implicit_verdict=False)
        if key is None:
            return None
        code = new_ask_code(self.store.open_ask_codes())
        reason = "a change to who may be contacted waits for the owner's own word"
        # Not a drive's work (no drive weight cancels it) and not a message: the owner's question.
        row, created = self.store.create_intention(
            kind="note", type="people_change", title=question[:160], drive="owner", cls="owner", decision="ask",
            decision_reason=reason, status="asked", dedup_key=key, recipient=self.owner_id, priority=0.8,
            rationale="a change that widens who gets messaged applies only on the owner's own yes to its ask",
            context={"change": spec, "concern": question, "evidence": [f"contact:{spec['contact_id']}"]},
            expires_at=ask_expiry(now, self.policy), ask_code=code, hermes_kind="none",
            source_type="people_change", source_id=str(spec["contact_id"]), created_at=now)
        if created != "created":
            return row if row.status == "asked" else None
        return self._apply_decision(row, Verdict(decision="ask", reason=reason, cls="owner"), now,
                                    reconsidered=False, code=code)

    def rate(self, intention_id: str, verdict: str, *, by: str = "owner") -> Optional[StoredInitiative]:
        return self.outcomes.rate(intention_id, verdict, by=by)

    async def _recipient_of(self, args: Mapping[str, Any]) -> str:
        """The contact a messaging call reaches: ``contact_id``, or the handle behind ``platform`` +
        ``target|chat_id|to``, or the stock ``target="platform:chat_id[:thread_id]"``."""
        recipient = str(args.get("contact_id") or "")
        if recipient or self.contacts is None:
            return recipient
        platform = str(args.get("platform") or "")
        address = str(args.get("target") or args.get("chat_id") or args.get("to") or "")
        if not platform and ":" in address:
            platform, address = split_target(address)
        if not platform or not address:
            return ""
        try:
            contact = await self.contacts.resolve_handle(platform, address)
        except Exception:
            return ""
        return getattr(contact, "contact_id", "") or ""

    async def guard(self, *, tool: str, args: Mapping[str, Any] | None, run: str = "mind",
                    session_id: str = "", recipients: Iterable[str] | None = None, task_id: str = "",
                    cwd: str | None = None, context: Mapping[str, Any] | None = None,
                    tool_call_id: str = "") -> Dict[str, Any]:
        """``POST /v1/mind/guard``: the sidecar half of the plugin guard (7.5).

        A messaging tool's recipient comes from its arguments; ``recipients``
        are the contacts an effect reaches later (a delivering cron job),
        resolved by the plugin. Every one of them is authorized like an
        immediate message: ``may_contact`` and the message budgets, the most
        restrictive recipient deciding; the owner is always reachable.

        In a mind run every effectful call is decided by what it does
        (``Authority.guard``: the effect table with ``cwd``, the directory a
        command runs in, and ``context`` for what only the plugin reads). A
        floor effect a standing grant covers (a declared action or another
        structured tool it names; never a shell, code or browser call) acts and
        the use is recorded; the one call the owner approved runs once; anything
        else is held (``_hold``) as the owner's ask naming the exact call, the
        task is marked blocked and the worker is told it waits. More people than ``mind.floor.bulk_recipients``
        reached in 24 h by one task (its continuations and the kanban tasks
        it made count with it, and a task the sidecar did not dispatch is
        counted by its own id) is bulk. A plain message the level x
        permission table asks about becomes that asked message
        (``_ask_message``), exactly as the intention would; any other call it
        asks about (a reaction, a delivering cron job) is held as the exact
        call.
        """
        args = dict(args or {})
        context = dict(context or {})
        now = self.clock()
        row = self._task_row(task_id) if run == "mind" else None
        wanted = [str(item) for item in (recipients or []) if str(item)]
        if tool in MESSAGING_TOOLS:
            recipient = await self._recipient_of(args)
            if not recipient:
                return {"allow": False, "action": "block", "reason": "messaging an unknown recipient"}
            wanted = [recipient]
        may_contact = None
        for recipient in wanted:
            try:
                permission = await self._may_contact(recipient)
            except Unreadable:
                return {"allow": False, "action": "block", "reason": f"permission for {recipient} unreadable"}
            if permission != "never":
                budget = self.authority.budget_check(kind="message", recipient=recipient)
                if budget:
                    return {"allow": False, "action": "block", "reason": budget}
            if self._is_owner(recipient):
                continue
            if may_contact is None or MAY_CONTACT.index(permission) < MAY_CONTACT.index(may_contact):
                may_contact = permission
        others = sorted({item for item in wanted if not self._is_owner(item)})
        if run == "mind":
            reached = self._reached(self._reach_key(row, task_id, session_id), now)
            new = [person for person in others if person not in reached]
            if new and len(reached | set(new)) > self.policy.bulk_recipients:
                context.update(bulk=True, recipients=new, reached=len(reached | set(new)))
            if tool == "cronjob_manage":
                context["own_job"] = self._own_job(row, str(context.get("job_id") or ""), task_id)
        verdict = self.authority.guard(tool=tool, args=args, run=run, recipient_may_contact=may_contact, task=row,
                                       cwd=cwd, context=context)
        if run != "mind":
            return verdict
        if verdict.get("hold"):
            return self._hold(row, verdict, task_id=task_id, now=now)
        if verdict.get("action") == "ask":
            if self._plain_send(tool, args, others):
                return await self._ask_message(args, others, row, task_id)
            if not isinstance(verdict.get("effect"), dict):
                return {"allow": False, "action": "block", "reason": str(verdict.get("reason") or "needs the owner")}
            return self._hold(row, verdict, task_id=task_id, now=now)
        if verdict.get("allow"):
            self._allowed(tool, args, verdict, row, others, task_id=task_id, session_id=session_id,
                          tool_call_id=tool_call_id, context=context, now=now)
        return verdict

    # -- holds, permits, grants and mind-origin cron jobs (the floor by effect) --------------------------------

    def _task_row(self, task_id: str) -> Optional[StoredInitiative]:
        """The intention a mind run works for: the one bound to its kanban task (or named by id)."""
        if not task_id:
            return None
        try:
            return self.store.get_by_hermes_ref(str(task_id)) or self.store.get(str(task_id))
        except Exception:
            return None

    def _origin(self, row: Optional[StoredInitiative]) -> Optional[StoredInitiative]:
        """The task a run's effects belong to: a continuation's originating task, else the row itself."""
        if row is None or row.type != FLOOR_EFFECT:
            return row
        origin = (row.context or {}).get("origin_task") if isinstance(row.context, dict) else None
        return self.store.get(str(origin)) if origin else row

    def _family(self, row: Optional[StoredInitiative]) -> List[StoredInitiative]:
        """A task with its continuations: its worker runs count as one task."""
        origin = self._origin(row)
        if origin is None:
            return []
        holds = (origin.context or {}).get("holds") if isinstance(origin.context, dict) else None
        rows = [origin] + [self.store.get(str(ident)) for ident in holds or []]
        return [item for item in rows if item is not None]

    def _reach_key(self, row: Optional[StoredInitiative], task_id: str, session_id: str = "") -> str:
        """Whose count a mind run's sends go to: the intention its task (or the task that made it, followed up the
        kanban lineage the plugin reported) works for; a task the sidecar did not dispatch is counted by the
        topmost task of its lineage, and so is a continuation the owner's yes to one of its held calls made; a run
        with no task by its session."""
        origin = self._origin(row)
        held_from = str((origin.context or {}).get("held_from") or "") if origin is not None and \
            origin.type == FLOOR_EFFECT and isinstance(origin.context, dict) else ""
        if held_from and held_from != task_id:
            return self._reach_key(self._task_row(held_from), held_from, session_id)   # its undispatched task's
        if origin is not None:
            return f"intention:{origin.id}"
        current, seen = str(task_id or ""), set()
        while current and current not in seen and len(seen) < 16:
            seen.add(current)
            bound = self._task_row(current) if current != task_id else None
            if bound is not None:
                return f"intention:{self._origin(bound).id}"
            parent = str((self.mind_state.get(f"{LINEAGE}{current}") or {}).get("text") or "")
            if not parent:
                break
            current = parent
        return f"kanban:{current}" if current else f"session:{session_id}"

    def _reach_entries(self, key: str) -> List[List[str]]:
        entry = self.mind_state.get(f"{REACH}{key}")
        try:
            value = json.loads(entry.get("text") or "[]") if entry else []
        except ValueError:
            value = []
        return [item for item in value if isinstance(item, list) and len(item) == 2]

    def _reached(self, key: str, now: datetime) -> set:
        """The people other than the owner one count (``_reach_key``) reached in 24 h: kept in mind.db, so a
        restart forgets none of them."""
        since = now - timedelta(days=1)
        return {person for person, at in self._reach_entries(key) if (_utc(at) or since) >= since}

    def _note_reach(self, key: str, people: Iterable[str], now: datetime) -> None:
        since = now - timedelta(days=1)
        kept = [[person, at] for person, at in self._reach_entries(key) if (_utc(at) or since) >= since]
        kept += [[str(person), now.isoformat()] for person in people if str(person)]
        self.mind_state.set(f"{REACH}{key}", text=json.dumps(kept[-500:]), now=now)

    def record_lineage(self, child: str, parent: str) -> Dict[str, Any]:
        """``POST /v1/mind/lineage``: a kanban task a mind run made (the plugin tells it once the create
        returned). The child counts with its maker; the first word stands (a task has one maker)."""
        child, parent = str(child or "").strip(), str(parent or "").strip()
        if not child or not parent or child == parent:
            raise ValueError("a child task and the task that made it are required")
        if self.mind_state.get(f"{LINEAGE}{child}") is None:
            self.mind_state.set(f"{LINEAGE}{child}", text=parent)
        return {"ok": True, "child": child, "parent": str((self.mind_state.get(f"{LINEAGE}{child}") or {}).get("text"))}

    @staticmethod
    def _plain_send(tool: str, args: Mapping[str, Any], others: List[str]) -> bool:
        """A send the outbox can make itself after the owner's yes: words to one person, nothing else in the
        call (no reaction, no thread or topic)."""
        plain = {"action", "target", "message", "text", "content", "contact_id", "platform", "chat_id", "to"}
        if tool != "send_message" or len(others) != 1 or set(args) - plain:
            return False
        if str(args.get("action") or "send").strip().lower() != "send" or str(args.get("target") or "").count(":") > 1:
            return False
        return bool(str(args.get("message") or args.get("text") or args.get("content") or "").strip())

    def _allowed(self, tool: str, args: Mapping[str, Any], verdict: Mapping[str, Any],
                 row: Optional[StoredInitiative], others: List[str], *, task_id: str, session_id: str = "",
                 tool_call_id: str = "", context: Mapping[str, Any] | None = None, now: datetime) -> None:
        """What an allowed call leaves behind: each grant's use (and a use of the owner's standing allow the grant
        came from), a permit spent, the people it reaches (the bulk count, kept for the task, its lineage or its
        session) and a cron job create or update let through (a pending mark the plugin's report after the call
        settles, ``record_cron_job``)."""
        effect = verdict.get("effect") if isinstance(verdict.get("effect"), dict) else None
        if verdict.get("grant") and effect is not None:
            self.authority.record_grant_use(str(verdict["grant"]), Effect(
                cls=str(effect.get("cls") or ""), tool=str(effect.get("tool") or tool),
                target=str(effect.get("target") or ""), amount=finite_amount(effect.get("amount")),
                currency=str(effect.get("currency") or ""), recipients=tuple(effect.get("recipients") or ())),
                task=row.id if row is not None else task_id)
            # A grant allow lets work go without asking here, at execution: this call is one of its uses.
            for allow in self.allows.in_force():
                if allow.kind == allow_rules.GRANT and allow.grant == str(verdict["grant"]):
                    self._allow_used(allow.id, now)
        if verdict.get("permit"):
            permit = self.store.get(str(verdict["permit"]))
            if permit is not None:
                self.store.transition(permit.id, permit.status, action="permit_used", at=now,
                                      details={"effect": (effect or {}).get("digest"), "tool": tool})
        if others:
            self._note_reach(self._reach_key(row, task_id, session_id), others, now)
            if row is not None and tool in MESSAGING_TOOLS:
                for person in others:
                    self.store.transition(row.id, row.status, action="worker_send", at=now,
                                          details={"recipient": person, "tool": tool})
        action = str(args.get("action") or "").strip().lower()
        if tool == "cronjob_manage" and action in {"create", "update"}:
            origin = self._origin(row)
            job = str((context or {}).get("job_id") or args.get("job_id") or "")
            self.mind_state.set(f"{CRON_PENDING}{task_id or 'unknown'}:{tool_call_id or now.isoformat()}", text=json.dumps({
                "task": origin.id if origin is not None else None, "kanban": task_id, "at": now.isoformat(),
                "action": action, "job": job}), now=now)

    def _hold(self, row: Optional[StoredInitiative], verdict: Mapping[str, Any], *, task_id: str,
              now: datetime, origin: str = "mind", delivery_ref: Mapping[str, str] | None = None,
              typed_payload: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        """A call that needs the owner's yes becomes one ask (``floor_effect``, deduplicated by the digest of the
        complete call) naming the exact call and the task it came from; its yes is a continuation task told to
        make exactly that call, which may run once. The originating task is marked blocked (not a failure: the
        breaker does not count it) and settles with the continuation."""
        from .outcomes import TERMINAL_OUTCOMES
        if typed_payload is not None:
            from .permit_asks import validate_payload
            typed_payload = validate_payload(dict(typed_payload))
            if origin != "hand":
                raise ValueError("capture_requires_hand_hold")
        effect = verdict.get("effect") if isinstance(verdict.get("effect"), dict) else {}
        digest, described = str(effect.get("digest") or ""), str(effect.get("text") or "an effect")
        hand_origin = origin == "hand"
        if origin not in {"mind", "hand"}:
            raise ValueError("unknown held-effect origin")
        # A hand's goal, summary and claimed approval never enter an ask binding.
        origin = None if hand_origin else self._origin(row)
        origin_id = origin.id if origin is not None else ""
        ask = next((item for item in self.store.intentions(status=["asked"], kind=["note" if hand_origin else "task"], limit=500)
                    if item.type == FLOOR_EFFECT and isinstance(item.context, dict)
                    and item.context.get("origin", "mind") == ("hand" if hand_origin else "mind")
                    and (not hand_origin or item.expires_at is None or item.expires_at > now)
                    and item.context.get("effect_digest") == digest
                    and item.context.get("origin_task", "") == origin_id
                    and (origin_id or item.context.get("held_from") == task_id)), None)
        if ask is None:
            name = f"hand {task_id}" if hand_origin else (
                Autobiography.name(origin) if origin is not None else f"kanban task {task_id or 'unknown'}")
            plan = ""
            if origin is not None and isinstance(origin.context, dict):
                plan = str(origin.context.get("plan_body") or origin.context.get("body") or origin.description or "")
            try:
                call = json.loads(str(effect.get("call") or "")) if effect.get("call") else {}
            except ValueError:
                call = {}
            call = call if isinstance(call, dict) else {}
            exact = json.dumps(call.get("args") or {}, ensure_ascii=False, sort_keys=True)
            where = f" with workdir {call.get('cwd') or call.get('workdir')}" if call.get("cwd") or call.get("workdir") \
                else ""
            instruction = f"call {call.get('tool') or 'the tool'}{where} with exactly these arguments: {exact}"
            if hand_origin:
                instruction += "; bound targets: " + json.dumps(call.get("targets") or [], ensure_ascii=False,
                                                                sort_keys=True)
            body = (f"{plan}\n\n" if plan else "") + (
                f"The owner approved exactly this call, once: {described}. To make it, {instruction}. Then finish "
                f"what the task needs and report what happened; any other call of that kind waits for the owner "
                f"again.")
            # The held task settles with the continuation's report, so the continuation is told the line that
            # task's done-check reads.
            told = body + (report_line(origin.success_check, body) if origin is not None else "")
            code = new_ask_code(self.store.open_ask_codes())
            reason = (f"held at execution from '{name}': {str(verdict.get('reason') or 'needs the owner')}; the exact "
                      f"call: {instruction}")[:1500]
            ask, _ = self.store.create_intention(
                # Hand continuations belong to the controller. A note cannot also
                # launch a mind kanban worker with the same approved call.
                kind="note" if hand_origin else "task", type=FLOOR_EFFECT, title=f"Let this run once: {described}"[:200],
                drive=origin.drive if origin is not None and origin.drive else "duty", cls="floor", decision="ask",
                decision_reason=reason, status="asked", dedup_key=None,
                rationale=f"{'a hand' if hand_origin else 'a mind run'} of '{name}' reached a call that waits for the owner's yes",
                recipient=origin.entity_id if origin is not None else None,
                context={"body": told, "plan_body": body, "effect": dict(effect), "effect_digest": digest,
                         "origin_task": origin_id, "held_from": task_id, "notice": True,
                         **({"origin": "hand"} if hand_origin else {}),
                         **({"payload": typed_payload} if typed_payload is not None else {}),
                         **({"delivery_ref": dict(delivery_ref)} if hand_origin and delivery_ref else {}),
                         "concern": (origin.context or {}).get("concern") if origin is not None
                         and isinstance(origin.context, dict) else None,
                         "evidence": [f"hand:{task_id}"] if hand_origin else
                         [f"intention:{origin_id}"] if origin_id else [f"kanban:{task_id}"]},
                expires_at=(now + timedelta(seconds=typed_payload["ttl_s"]) if typed_payload is not None
                            else ask_expiry(now, self.policy)), ask_code=code, hermes_kind="none",
                source_type="intention", source_id=origin_id or None, created_at=now)
            ask = self._apply_decision(ask, Verdict(decision="ask", reason=reason, cls="floor",
                                                    floor=str(effect.get("cls") or "") or None, notice=True),
                                       now, reconsidered=False, code=code) or ask
            if typed_payload is not None:
                ask = self.store.update(ask.id, expires_at=now + timedelta(seconds=typed_payload["ttl_s"]))
        code = ask.ask_code or ""
        if origin is not None and origin.status in {"approved", "dispatched"} and origin.outcome not in TERMINAL_OUTCOMES:
            metadata = dict(origin.result_metadata or {})
            metadata["breaker"] = {"counted": False, "reason": "held for the owner, not a failure"}
            held = [item for item in metadata.get("held") or [] if isinstance(item, dict) and item.get("ask") != ask.id]
            metadata["held"] = held + [{"ask": ask.id, "code": code, "effect": digest}]
            context = dict(origin.context) if isinstance(origin.context, dict) else {}
            context["holds"] = list(dict.fromkeys([*(context.get("holds") or []), ask.id]))
            self.store.update(origin.id, outcome="blocked", result=f"held for the owner (ask {code})",
                              result_metadata=metadata, context=context)
            self.store.transition(origin.id, origin.status, action="held", at=now,
                                  details={"code": code, "ask": ask.id, "effect": digest})
        return {"allow": False, "action": "block", "held": code, "ask": ask.id, "effect": dict(effect),
                "reason": f"held for the owner's yes (ask {code}): {described}; stop and report that it waits"}

    def _held_open(self, row: Optional[StoredInitiative]) -> bool:
        """A task with a hold whose ask or continuation has not settled: its worker's report is not its end."""
        if row is None or row.outcome != "blocked" or not isinstance(row.result_metadata, dict):
            return False
        for item in row.result_metadata.get("held") or []:
            ask = self.store.get(str(item.get("ask") or "")) if isinstance(item, dict) else None
            if ask is not None and ask.status in {"asked", "approved", "dispatched"}:
                return True
        return False

    def report(self, intention_id: str, **fields: Any) -> Optional[StoredInitiative]:
        """``POST /v1/mind/outcome``. A task held for the owner does not settle on its worker's report (the run
        ended to wait): the report is logged and the task settles with the continuation the owner's answer
        makes (``_settled``). Every other report is ``Outcomes.record``."""
        row = self.store.get(str(intention_id))
        if self._held_open(row) and fields.get("final") is not False:
            return self.store.transition(row.id, row.status, action="run_reported_while_held", at=self.clock(),
                                         details={"status": fields.get("status"), "outcome": fields.get("outcome"),
                                                  "summary": str(fields.get("summary") or "")[:500]})
        return self.outcomes.record(intention_id, **fields)

    def _settled(self, row: StoredInitiative, outcome: str, check_result: Optional[bool]) -> None:
        """Settlement's hook: the concerns and satiation (``_on_settled``), then a held task settles with its
        last open continuation: with that continuation's outcome (a no settles it denied, silence expired), its
        summary and its structured result, so what the continuation found is what the task reports."""
        from .outcomes import TERMINAL_OUTCOMES
        self._on_settled(row, outcome, check_result)
        if row.type != FLOOR_EFFECT or not isinstance(row.context, dict) or not row.context.get("origin_task"):
            return
        origin = self.store.get(str(row.context["origin_task"]))
        if origin is None or origin.outcome in TERMINAL_OUTCOMES or self._held_open(origin):
            return
        summary = " ".join(str(row.result or "").split())
        metadata = row.result_metadata if isinstance(row.result_metadata, dict) else {}
        run = metadata.get("run") if isinstance(metadata.get("run"), dict) else None
        self.outcomes.record(origin.id, status=outcome, by="mind", implicit_verdict=False,
                             summary=f"{summary} (through the owner's answer to ask {row.ask_code})" if summary
                             else f"settled with its held effect (ask {row.ask_code}): {outcome}",
                             result=metadata.get("result"), run=run)

    async def _ask_message(self, args: Mapping[str, Any], others: List[str], row: Optional[StoredInitiative],
                           task_id: str) -> Dict[str, Any]:
        """A worker's plain send the table asks about becomes the message intention it would have been: the words
        frozen in the ask are the words that go after the owner's yes."""
        import hashlib
        text = str(args.get("message") or args.get("text") or args.get("content") or "").strip()
        recipient = others[0] if len(others) == 1 else ""
        if not text or not recipient:
            return {"allow": False, "action": "block", "reason": "messaging this contact needs the owner's approval"}
        origin = self._origin(row)
        ident = hashlib.sha256(f"{task_id}\n{recipient}\n{text}".encode("utf-8")).hexdigest()[:16]
        await self.request_message({"message": text, "recipient": recipient, "id": ident, "type": "worker",
                                    "title": f"Message from '{Autobiography.name(origin)}'" if origin is not None
                                    else None}, source="worker")
        formed = self.store.get_by_dedup_key(f"reach_out:worker:{ident}")
        if formed is not None and formed.status == "asked" and formed.ask_code:
            return {"allow": False, "action": "block", "held": formed.ask_code, "ask": formed.id,
                    "reason": f"messaging this contact waits for the owner's yes (ask {formed.ask_code}); the message "
                              "goes out after it; stop and report that it waits"}
        state = formed.status if formed is not None else "not formed"
        return {"allow": False, "action": "block",
                "reason": f"messaging this contact needs the owner's approval (the message is {state})"}

    # A cron job's origin is its id. A create or update a mind run was let through leaves a pending mark
    # (``cron_pending:<task>:<call>``); the plugin reports the call's result by the job's id once it returned
    # (``record_cron_job``: ``cron_job:<id>``). A mark never confirmed leaves the jobs made around it unknown.

    def record_cron_job(self, job_id: str, *, task_id: str = "", tool_call_id: str = "", action: str = "create",
                        ok: bool = True) -> Dict[str, Any]:
        """``POST /v1/mind/cron-origin``: the plugin's report, after a mind run's cron create or update returned.
        A job it made or changed is the mind's (its runs are mind runs); a failed call claims nothing."""
        now = self.clock()
        pending_key = f"{CRON_PENDING}{task_id or 'unknown'}:{tool_call_id}" if tool_call_id else ""
        entry = self.mind_state.get(pending_key) if pending_key else None
        try:
            pending = json.loads(entry.get("text") or "{}") if entry else {}
        except ValueError:
            pending = {}
        job_id = str(job_id or "").strip()
        if ok and job_id:
            origin = self._origin(self._task_row(task_id))
            self.mind_state.set(f"{CRON_JOB}{job_id}", text=json.dumps({
                "task": pending.get("task") or (origin.id if origin is not None else None), "kanban": task_id,
                "at": now.isoformat(), "action": str(action or "")}), now=now)
        if entry is not None:
            self.mind_state.delete(pending_key)
        return {"ok": True, "job_id": job_id, "recorded": bool(ok and job_id)}

    def _pending_around(self, job_id: str, created_at: Any) -> bool:
        """An unconfirmed create or update that could have made or changed this job: an update names its job; a
        create is near the job's creation time (or the time is not known)."""
        stamp = _utc(created_at) if created_at else None
        for item in self.mind_state.items(CRON_PENDING):
            if not str(item.get("key") or "").startswith(CRON_PENDING):
                continue
            try:
                value = json.loads(item.get("text") or "{}")
            except ValueError:
                return True
            at = _utc(value.get("at"))
            if value.get("action") == "update":
                if not value.get("job") or value.get("job") == job_id:
                    return True
                continue
            if stamp is None or at is None or at - timedelta(minutes=2) <= stamp <= at + timedelta(minutes=10):
                return True
        return False

    def cron_origin(self, job_id: str, created_at: Any = None) -> Dict[str, Any]:
        """``GET /v1/mind/cron-origin``: whose cron job this is. ``mind``: a mind run made or changed it (its runs
        are mind runs); ``owner``: no mind mark, NOT positive owner authority;
        ``unknown``: a create or update let through was never confirmed and could be this job.
        Callers granting authority must use a separate positive creator record."""
        job_id = str(job_id or "").strip()
        entry = self.mind_state.get(f"{CRON_JOB}{job_id}") if job_id else None
        if entry is not None:
            try:
                value = json.loads(entry.get("text") or "{}")
            except ValueError:
                value = {}
            return {"origin": "mind", "mind": True, "task_id": value.get("kanban") or "",
                    "intention_id": value.get("task")}
        if self._pending_around(job_id, created_at):
            return {"origin": "unknown", "mind": None, "task_id": "", "intention_id": None}
        return {"origin": "owner", "mind": False, "task_id": "", "intention_id": None}

    def _own_job(self, row: Optional[StoredInitiative], job_id: str, task_id: str) -> bool:
        """A cron job this task (or one of its continuations) made, by its id: removing it is its own cleanup."""
        if not job_id:
            return False
        value = self.cron_origin(job_id)
        if value["origin"] != "mind":
            return False
        family = {item.id for item in self._family(row)}
        return bool(value.get("intention_id") and value["intention_id"] in family) or \
            bool(task_id and value.get("task_id") == task_id)

    # Standing grants: the owner's configuration, the ones they confirmed (their own yes to each one's ask) and
    # their revocations. A confirmed grant is ``grant:<id>`` in mind.db; a revoked configured grant is
    # ``grant_revoked:<id>`` with that grant's fingerprint (a changed grant is a new one). A configured grant's id
    # is its own: a grant confirmed under it is refused, and one confirmed before the configuration took the id
    # is shown inactive with the reason.

    def grants(self) -> List[Grant]:
        """The standing grants in force (``Authority`` asks this); a revoked or shadowed one is gone."""
        configured = {grant.id for grant in self.policy.grants}
        return [grant for grant, revoked in self._grant_list()
                if not revoked and (grant.source == "config" or grant.id not in configured)]

    def _grant_list(self) -> List[tuple]:
        listed: List[tuple] = []
        for grant in self.policy.grants:
            marker = self.mind_state.get(f"{GRANT_REVOKED}{grant.id}")
            listed.append((grant, bool(marker) and marker.get("text") == grant.fingerprint))
        configured, taken = {grant.id for grant, _ in listed}, set()
        for item in self.mind_state.items(GRANT_KEY):
            if not str(item.get("key") or "").startswith(GRANT_KEY):
                continue
            try:
                value = json.loads(item.get("text") or "{}")
            except ValueError:
                continue
            grant = parse_grant(value.get("grant"), source=str(value.get("source") or "cli"))
            if grant.id in taken:
                continue
            taken.add(grant.id)
            if grant.id in configured:
                grant = dataclasses.replace(grant, problem=f"the configured grant {grant.id} has this id; revoke this "
                                                           f"one and add it under another id")
            listed.append((grant, False))
        return listed

    def grants_state(self, now: datetime | None = None) -> Dict[str, Any]:
        """Every grant with its source, limits, validity, expiry and its uses and amounts in the current windows,
        and the last change to the configured set."""
        now = now or self.clock()
        rows = [{**grant.as_dict(now), "revoked": revoked, **self.authority.grant_usage(grant.id, now)}
                for grant, revoked in self._grant_list()]
        changed = self.mind_state.get(GRANTS_CHANGED)
        return {"grants": rows, "changed": (changed or {}).get("text") or None,
                "changed_at": (changed or {}).get("updated_at")}

    def _note_grant_changes(self) -> None:
        """The configured grants are fingerprinted at load; a change since the last load is shown in state and
        told to the owner once ("standing grants changed: added ..., removed ...")."""
        configured = {grant.id: grant.fingerprint for grant in self.policy.grants}
        stored = self.mind_state.get(GRANTS_FINGERPRINT)
        try:
            before = json.loads(stored.get("text") or "{}") if stored else None
        except ValueError:
            before = {}
        self.mind_state.set(GRANTS_FINGERPRINT, text=json.dumps(configured, sort_keys=True))
        if before is None and not configured or before == configured:
            return
        before = before or {}
        parts = [f"{verb} {', '.join(names)}" for verb, names in (
            ("added", sorted(set(configured) - set(before))), ("removed", sorted(set(before) - set(configured))),
            ("changed", sorted(key for key in configured if key in before and before[key] != configured[key])))
            if names]
        text = "standing grants changed: " + "; ".join(parts)
        self.mind_state.set(GRANTS_CHANGED, text=text)
        import hashlib
        change = hashlib.sha256(json.dumps([before, configured], sort_keys=True).encode("utf-8")).hexdigest()[:16]
        self.outbox.notice(type="grants_changed", title="Standing grants changed",
                           text=f"{text}. `colonymind mind grants` lists them.",
                           dedup_key=f"grants_changed:{change}:{self.clock().isoformat()}")

    def _grant_id_problem(self, grant: Grant) -> str:
        """Why a grant cannot be added under its id ('' when it can): a configured grant's id stays its own, and
        an id in force is revoked first."""
        if any(item.id == grant.id for item in self.policy.grants):
            return (f"the id {grant.id} belongs to a configured grant (it changes in the configuration); use another "
                    f"id")
        if any(item.id == grant.id for item, revoked in self._grant_list() if not revoked):
            return f"a standing grant {grant.id} exists; revoke it first"
        return ""

    def request_grant(self, spec: Any, *, reason: str = "", contact_id: str | None = None) -> Dict[str, Any]:
        """A standing grant the owner asked for in their conversation: a ``grant_confirm`` ask showing the grant in
        full, active only after the owner's own yes to it in their chat (``read_answer``). Nothing here makes a
        grant active: the operator configures one (``mind.grants``). Raises ``PermissionError`` for anyone but the
        owner and ``ValueError`` for a malformed grant or an id it cannot take."""
        if contact_id and self.owner_id and contact_id != self.owner_id:
            raise PermissionError("only the owner sets a standing grant")
        grant = parse_grant(spec, source="ask")
        if not grant.valid:
            raise ValueError(grant.problem)
        problem = self._grant_id_problem(grant)
        if problem:
            raise ValueError(problem)
        now = self.clock()
        words = " ".join(str(reason or "").split())
        if not words:
            raise ValueError("reason is required: the owner's own words asking for this grant")
        shown = json.dumps(grant.spec(), sort_keys=True)
        existing = next((item for item in self.store.intentions(status=["asked"], kind=["note"], limit=500)
                         if item.type == GRANT_CONFIRM and isinstance(item.context, dict)
                         and item.context.get("fingerprint") == grant.fingerprint), None)
        if existing is None:
            code = new_ask_code(self.store.open_ask_codes())
            existing, _ = self.store.create_intention(
                kind="note", type=GRANT_CONFIRM, title=f"Standing grant {grant.id}: {grant.scope}"[:200],
                drive="upkeep", cls="owner", decision="ask", status="asked", dedup_key=None,
                decision_reason="a standing grant is active only after the owner's own yes to its ask",
                rationale=f"the owner asked in conversation: {words[:300]}", recipient=self.owner_id,
                context={"grant": grant.spec(), "fingerprint": grant.fingerprint, "reason": words[:1000],
                         "text": f"Allow standing grant {shown}", "notice": True},
                expires_at=ask_expiry(now, self.policy), ask_code=code, hermes_kind="none", created_at=now)
            self.store.transition(existing.id, "asked", action="asked", at=now, details={"code": code, "notice": True})
        code = existing.ask_code or ""
        return {"ok": True, "status": "asked", "code": code, "id": existing.id,
                "text": f"standing grant {shown} waits for the owner: they are asked in their own chat, and "
                        f"their own yes to that ask allows it"}

    async def _activate_grant(self, row: StoredInitiative, *, yes: bool, by: str,
                              details: Mapping[str, Any] | None = None) -> Optional[StoredInitiative]:
        """The owner's typed answer to a ``grant_confirm`` ask: a yes makes the grant active (its source is the
        ask code), a no drops it."""
        context = row.context if isinstance(row.context, dict) else {}
        if not yes:
            return self.outcomes.record(row.id, status="denied", summary=f"{self._who(by, details)} refused the grant",
                                        by=by, details=details, implicit_verdict=False)
        grant = parse_grant(context.get("grant"), source=f"ask {row.ask_code}")
        problem = grant.problem or self._grant_id_problem(grant)
        if problem:
            return self.outcomes.record(row.id, status="cancelled", by=by, details=details, implicit_verdict=False,
                                        summary=f"the grant could not be activated ({problem})")
        now = self.clock()
        self.mind_state.set(f"{GRANT_KEY}{grant.id}", text=json.dumps({
            "grant": grant.spec(), "source": f"ask {row.ask_code}", "confirmed_at": now.isoformat(),
            "reason": context.get("reason")}), now=now)
        return self.outcomes.record(row.id, status="done", verified="owner", by=by, implicit_verdict=False,
                                    details=details,
                                    summary=f"{self._who(by, details)} confirmed standing grant {grant.id}")

    def revoke_grant(self, grant_id: str, *, by: str = "owner", contact_id: str | None = None) -> Dict[str, Any]:
        """Revoke a standing grant at once (a narrowing, free for the owner): everything in force under its id. A
        configured grant stays revoked until its configuration changes."""
        if contact_id and self.owner_id and contact_id != self.owner_id:
            raise PermissionError("only the owner revokes a standing grant")
        grant_id = str(grant_id or "").strip()
        listed = [(grant, revoked) for grant, revoked in self._grant_list() if grant.id == grant_id]
        live = [grant for grant, revoked in listed if not revoked]
        if not live:
            raise KeyError(grant_id)
        self.mind_state.delete(f"{GRANT_KEY}{grant_id}")
        for allow in self.allows.in_force():
            if allow.kind == allow_rules.GRANT and allow.grant == grant_id:
                self.allows.remove(allow.id)
                self._allow_note(ALLOW_REVOKED, allow, f"standing allow {allow.id} revoked with its grant ({by})",
                                 by=by)
        configured = next((grant for grant in live if grant.source == "config"), None)
        if configured is not None:
            self.mind_state.set(f"{GRANT_REVOKED}{grant_id}", text=configured.fingerprint)
        self._grant_note("grant_revoked", live[0], f"standing grant {grant_id} revoked ({by})")
        return {"ok": True, "status": "revoked", "id": grant_id, "text": f"standing grant {grant_id} is revoked"}

    def _grant_note(self, type: str, grant: Grant, text: str) -> None:
        now = self.clock()
        self.store.create_intention(kind="note", type=type, title=text[:200], drive="upkeep", cls="internal",
                                    decision="act", decision_reason=text[:300], status="done", dedup_key=None,
                                    context={"grant": grant.spec(), "source": grant.source}, hermes_kind="none",
                                    created_at=now)

    def _concern_reach(self, candidate: Candidate, now: datetime) -> int:
        """The other people messages formed from this candidate's concern or source reached in 24 h (their
        ``queued`` transitions): past ``mind.floor.bulk_recipients`` the next one is bulk."""
        return self._reach_of(candidate.concern, candidate.source_type, candidate.source_id, candidate.recipient, now)

    def _row_reach(self, row: StoredInitiative, now: datetime) -> int:
        """``_concern_reach`` for a message already formed (a deferred one, reconsidered)."""
        if row.kind != "message" or not row.entity_id or self._is_owner(row.entity_id):
            return 0
        context = row.context if isinstance(row.context, dict) else {}
        return self._reach_of(context.get("concern"), row.source_type, row.source_id, row.entity_id, now)

    def _reach_of(self, concern: Any, source_type: Any, source_id: Any, recipient: Any, now: datetime) -> int:
        if not concern and not source_id:
            return 0
        since, people = now - timedelta(days=1), set()
        for row in self.store.intentions(kind=["message"], since=since, limit=500):
            if not row.entity_id or self._is_owner(row.entity_id) or row.entity_id == recipient:
                continue
            context = row.context if isinstance(row.context, dict) else {}
            same = (concern and context.get("concern") == concern) or (
                source_id and row.source_id == source_id and row.source_type == source_type)
            if same and any(entry.action == "queued" and entry.timestamp >= since
                            for entry in self.store.get_history(row.id, limit=100)):
                people.add(row.entity_id)
        return len(people)

    def _bulk_grant_used(self, verdict: Verdict, row: Optional[StoredInitiative], reached: int) -> None:
        """A message a bulk grant let act (past ``mind.floor.bulk_recipients`` people from one concern) is one use."""
        if row is None or not verdict.grant or verdict.decision != "act" or reached < self.policy.bulk_recipients:
            return
        self.authority.record_grant_use(verdict.grant, Effect(
            "bulk_third_party_messaging", "outbox", target=str(row.entity_id), recipients=(str(row.entity_id),)),
            task=row.id)

    # -- standing allows: "always" and the offers to stop asking (docs/MIND.md, Standing grants and allows) -----------

    def _allows_in_force(self) -> List[Any]:
        """What ``Authority`` consults (the workspace database opens after it is built)."""
        allows = getattr(self, "allows", None)
        return allows.in_force() if allows is not None else []

    def _offer(self, row: Optional[StoredInitiative]) -> Optional[Offer]:
        """The "always" this ask offered when it formed (``context['always']``), while allows apply at the level."""
        if row is None or self.level not in ALLOW_LEVELS or not isinstance(row.context, dict):
            return None
        return Offer.from_dict(row.context.get("always"))

    def _named_in_ask(self, row: StoredInitiative) -> bool:
        """Whether an ask's own words name a contact: a message it would send, or its offer to stop asking before
        contacting someone (both read the name as the owner knows it, never an internal id)."""
        offer = Offer.from_dict(row.context.get("always")) if isinstance(row.context, dict) else None
        return audit.asked_words(row) is not None or (offer is not None and offer.kind == allow_rules.CONTACT)

    def _always_bar(self, text: str, binding: Any, arrived: datetime, *, codes: List[str],
                    code: str) -> Optional[answer_words.Reading]:
        """None when an "always" meets the bar of a widening answer (APPROVALS 6.4): the ask's own message quoted, or
        the owner's next message within ``answers.FOLLOW_UP_HOURS`` of it, and a clean yes read strict. Otherwise the
        reading it stands for: unclear, so the owner gets one plain question and nothing is allowed."""
        w = answer_words
        why = ""
        if binding.path not in {w.REPLY, w.FOLLOW_UP}:
            why = "'always' answers only the ask's own message"
        elif binding.path == w.FOLLOW_UP and binding.at is not None \
                and (arrived - binding.at).total_seconds() > w.FOLLOW_UP_HOURS * 3600:
            why = f"'always' binds by a plain next message only within {w.FOLLOW_UP_HOURS:g} hours"
        elif w.read(text, strict=True, offers_always=True, codes=codes, code=code).verdict != w.ALWAYS:
            why = "'always' takes a plain answer, read as strictly as any change of permission"
        return w.Reading(w.UNCLEAR, said=w.YES, reason=why, answer_word=True) if why else None

    def _contact_allow(self, row: StoredInitiative, cls: str) -> Optional[str]:
        """The owner's standing allow for the contact this work reaches, when there is one (its uses are counted)."""
        if cls != "contact" or not row.entity_id or self._is_owner(row.entity_id):
            return None
        allow = self.allows.for_key(f"{allow_rules.CONTACT}:{row.entity_id}")
        return allow.id if allow is not None else None

    def _allow_used(self, ident: Optional[str], now: datetime) -> None:
        if ident:
            self.allows.used(ident, now)

    def _allow_note(self, type: str, allow: Any, text: str, *, by: str) -> None:
        """A note in the intention store for each allow added or revoked (the audit log, and what the offers read)."""
        self.store.create_intention(kind="note", type=type, title=text[:200], drive="upkeep", cls="internal",
                                    decision="act", decision_reason=text[:300], status="done", dedup_key=None,
                                    context={"allow": allow.as_dict(), "key": allow.key, "by": by}, hermes_kind="none",
                                    created_at=self.clock())

    async def _adopt_allow(self, row: StoredInitiative, offer: Offer, *, by: str, now: datetime,
                           name: str | None = None) -> Optional[Any]:
        """The standing allow the owner's own "always" (or their yes to an offer) adds for exactly the ask's class of
        work: a contact's permission set to ``auto`` (compare-and-set against what the ask showed, performed as the
        owner's answer to the ask), or a standing grant for exactly a held call; tasks with those toolsets are read
        by the authority. None when it could not be added (the contact changed since, a grant id taken), and the
        ask's own yes stands either way."""
        from colonymind.contacts.changes import change_lock
        from colonymind.contacts.store import ContactChanged
        existing = self.allows.for_key(offer.key)
        if existing is not None:
            return existing
        if offer.kind == allow_rules.CONTACT and name is None:
            name = await self._shown_name(offer.target)
        grant_id = before = None
        resetting = self.allows.resetting_for(offer.target) if offer.kind == allow_rules.CONTACT else None
        if offer.kind == allow_rules.CONTACT:
            expected = str((row.context or {}).get("may_contact") or "ask")
            if resetting is not None:
                # A revocation's reset that has not landed left the stored permission "auto" (the ask read it as
                # the reset's value): the owner's new "always" keeps it, and the pending reset goes.
                expected = "auto"
            if self.contacts is None or not hasattr(self.contacts, "set_may_contact"):
                return None
            try:
                async with change_lock(self.contacts):
                    await self.contacts.set_may_contact(offer.target, "auto", by=f"owner-ask:{row.ask_code}",
                                                        reason=f"the owner's 'always' to ask {row.ask_code}",
                                                        expected=expected)
            except ContactChanged:
                return None
            except Exception as error:
                logger.warning("allow %s not added (%s)", offer.id, type(error).__name__)
                return None
            before = resetting.before if resetting is not None else expected
            if resetting is not None:
                self.allows.reset_done(resetting.id)
        elif offer.kind == allow_rules.GRANT:
            grant = parse_grant(offer.grant, source=f"ask {row.ask_code}")
            problem = grant.problem or self._grant_id_problem(grant)
            if problem:
                logger.warning("allow %s not added: %s", offer.id, problem)
                return None
            self.mind_state.set(f"{GRANT_KEY}{grant.id}", text=json.dumps({
                "grant": grant.spec(), "source": f"ask {row.ask_code}", "confirmed_at": now.isoformat(),
                "reason": "the owner's 'always'"}), now=now)
            grant_id = grant.id
        words = allow_rules.words(offer, name=name)
        allow = self.allows.add(offer, words=words, ask=row, by=by, now=now, grant=grant_id, before=before)
        current = self.store.get(row.id) or row
        self.store.transition(row.id, current.status, action=ALLOW_ADDED, at=now,
                              details={"allow": allow.id, "by": by})
        self._allow_note(ALLOW_ADDED, allow, f"standing allow {allow.id} added by the owner's own answer: {words}",
                         by=by)
        return allow

    async def revoke_allow(self, ident: str, *, by: str = OPERATOR) -> Dict[str, Any]:
        """Revoke a standing allow at once (a narrowing, open to the console): the asking comes back. A contact's
        permission goes back to what it was only while it is still ``auto`` (compare-and-set: a permission changed
        since, an opt-out included, is left as it is); an allow's grant is withdrawn. ``KeyError`` for no such
        allow."""
        allow = self.allows.get(ident)
        if allow is None:
            raise KeyError(str(ident))
        left = ""
        if allow.kind == allow_rules.CONTACT:
            reset = await self._reset_permission(allow)
            if reset == "changed":
                left = "; the contact's permission changed since, so it was left as it is"
            elif reset:
                # Kept until it lands: the contact reads as asking first meanwhile (``_may_contact``).
                self.allows.hold_reset(allow, self.clock())
                left = (f"; their permission could not be reset yet ({reset}), so I ask first before contacting "
                        f"them and try again at each tick")
        if allow.kind == allow_rules.GRANT and allow.grant:
            self.mind_state.delete(f"{GRANT_KEY}{allow.grant}")
        self.allows.remove(allow.id)
        self._allow_note(ALLOW_REVOKED, allow, f"standing allow {allow.id} revoked ({by})", by=by)
        again = allow_rules.again(allow)
        return {"ok": True, "status": "revoked", "id": allow.id, "again": again, "left": left,
                "text": f"standing allow {allow.id} is revoked: I ask again before I {again}{left}"}

    async def _reset_permission(self, allow: Any) -> str:
        """A revoked contact allow's permission back to what it was, compare-and-set against the "auto" the allow
        wrote: '' when it landed (or there is no contacts store to write), "changed" when the permission changed since
        (an opt-out included: left as it is), else why it failed."""
        from colonymind.contacts.changes import change_lock
        from colonymind.contacts.store import ContactChanged
        if not hasattr(self.contacts, "set_may_contact"):
            return ""
        try:
            async with change_lock(self.contacts):
                await self.contacts.set_may_contact(allow.target, allow.before or "ask", by=f"allow-revoked:{allow.id}",
                                                    reason=f"standing allow {allow.id} revoked", expected="auto")
        except ContactChanged:
            return "changed"
        except Exception as error:
            logger.warning("allow %s: permission not reset (%s)", allow.id, type(error).__name__)
            return type(error).__name__
        return ""

    async def _retry_allow_resets(self) -> int:
        """Each revoked contact allow whose permission reset has not landed is tried again; one that lands, or finds
        the permission changed since, is done. How many were done."""
        done = 0
        for allow in self.allows.resetting():
            if await self._reset_permission(allow) in {"", "changed"}:
                self.allows.reset_done(allow.id)
                done += 1
        return done

    async def _revoke_in_words(self, text: str, *, by: str) -> List[Dict[str, Any]]:
        """The owner's own "ask me again about <thing>" (``answers.ask_again``; the answer route has already checked
        the message is theirs): every standing allow the thing names (``allows.named_by``: a person by a name they
        go by, a kind of work by its tools or drive, "everything") is revoked at once, as the console revokes one.
        A narrowing only: these words never add or keep an allow. Nothing when they name no allow in force."""
        thing = answer_words.ask_again(text) if str(text or "").strip() else None
        in_force = self.allows.in_force() if thing else []
        if not in_force:
            return []
        names: Dict[str, List[str]] = {}
        for allow in in_force:
            if allow.kind == allow_rules.CONTACT:
                record = await self._contact_record(allow.target) or {}
                names[allow.id] = [str(record.get(key) or "") for key in ("display_name", "given_name")
                                   if record.get(key)] + [await self._shown_name(allow.target)]
        revoked: List[Dict[str, Any]] = []
        for allow in allow_rules.named_by(thing, in_force, names=names):
            try:
                done = await self.revoke_allow(allow.id, by=by)
            except KeyError:
                continue                    # revoked meanwhile
            revoked.append({"allow": allow.id, "kind": allow.kind, "again": done["again"], "left": done["left"]})
        return revoked

    def allows_state(self) -> Dict[str, Any]:
        """The standing allows in force, each with the words the owner agreed to, the ask it came from and its uses."""
        rows = [allow.as_dict() for allow in self.allows.in_force()]
        lines = [f"{item['id']}: {item['words']} (since {str(item['created_at'])[:10]}, {item['uses']} uses)"
                 for item in rows]
        resetting = [allow.as_dict() for allow in self.allows.resetting()]
        lines += [f"{item['id']}: revoked; resetting the contact's permission to {item['before'] or 'ask'} (I ask first "
                  f"until it lands)" for item in resetting]
        return {"allows": rows, "resetting": resetting, "applies": self.level in ALLOW_LEVELS,
                "text": "\n".join(lines) or "(no allows)"}

    def _owner_answers_for(self, key: str, since: datetime) -> List[tuple]:
        """The owner's own answers to asks that offered ``key`` (``(when, yes)``: True for their yes, False for their
        no or an undo, None for anyone else's answer) since ``since``; a lapse leaves nothing."""
        answers: List[tuple] = []
        for row in self.store.intentions(since=since - timedelta(hours=self.policy.ask_expires_hours), limit=2000):
            offer = Offer.from_dict(row.context.get("always")) if isinstance(row.context, dict) else None
            if offer is None or offer.key != key:
                continue
            for item in self.store.get_history(row.id, limit=200):
                details = item.details if isinstance(item.details, dict) else {}
                if item.action == "queued" and details.get("code"):
                    answers.append((_utc(item.timestamp), True if owner_answer(details.get("by"), details) else None))
                elif item.action == "outcome_denied":
                    answers.append((_utc(item.timestamp), False if owner_answer(details.get("by"), details) else None))
                elif item.action == ASK_UNDONE:
                    answers.append((_utc(item.timestamp), False))
        return [(at, yes) for at, yes in answers if at is not None]

    def _proposal_rows(self, key: str, now: datetime) -> List[StoredInitiative]:
        return [row for row in self.store.intentions(since=now - timedelta(days=allow_rules.HOLD_DAYS
                                                                           + allow_rules.PROPOSE_DAYS), limit=2000)
                if row.type == ALLOW_PROPOSAL and ((row.context or {}).get("proposes") or {}).get("key") == key]

    async def _propose_allow(self, offer: Offer, *, now: datetime, name: str | None = None) -> Optional[Any]:
        """After ``mind.asks.propose_after`` of the owner's own yeses in a row to asks that offered ``offer``, one ask
        of its own: stop asking? Not while an allow for it stands or an offer for it is open, not within
        ``allows.HOLD_DAYS`` of a no or a lapse of the last one, and only yeses after the last offer and the last
        revocation of that allow count."""
        count = self.allow_settings.propose_after
        if not count or not self.owner_id or self.level not in ALLOW_LEVELS or self.allows.for_key(offer.key):
            return None
        offers = self._proposal_rows(offer.key, now)
        if any(row.status == "asked" for row in offers):
            return None
        settled = [_utc(row.completed_at or row.cancelled_at or row.updated_at) for row in offers]
        if any(at is not None and row.outcome != "done" and now < allow_rules.hold_until(at)
               for row, at in zip(offers, settled)):
            return None
        revoked = [_utc(row.created_at) for row in self.store.intentions(
            kind=["note"], since=now - timedelta(days=allow_rules.PROPOSE_DAYS), limit=2000)
            if row.type == ALLOW_REVOKED and (row.context or {}).get("key") == offer.key]
        window = now - timedelta(days=allow_rules.PROPOSE_DAYS)
        cutoff = max([at for at in settled + revoked if at is not None], default=None)
        answers = [(at, yes) for at, yes in self._owner_answers_for(offer.key, window)
                   if cutoff is None or at > cutoff]
        run = allow_rules.streak(answers, since=window)
        if run < count:
            return None
        if offer.kind == allow_rules.CONTACT and name is None:
            name = await self._shown_name(offer.target)
        title = allow_rules.proposal_title(offer, run, name=name)
        reason = f"the owner said yes {run} times in a row; to stop asking is theirs to allow"
        code = new_ask_code(self.store.open_ask_codes())
        row, created = self.store.create_intention(
            kind="note", type=ALLOW_PROPOSAL, title=title, drive="owner", cls="owner", decision="ask",
            decision_reason=reason, status="asked", dedup_key=f"{ALLOW_PROPOSAL}:{offer.key}:{now.isoformat()}",
            recipient=self.owner_id, priority=0.6, rationale=reason,
            context={"proposes": offer.as_dict(), "count": run, "concern": title, "notice": True,
                     "evidence": [f"allow:{offer.key}"]},
            expires_at=ask_expiry(now, self.policy), ask_code=code, hermes_kind="none",
            source_type=ALLOW_PROPOSAL, source_id=offer.key, created_at=now)
        if created != "created":
            return None
        return self._apply_decision(row, Verdict(decision="ask", reason=reason, cls="owner"), now,
                                    reconsidered=False, code=code)

    async def _settle_proposal(self, row: StoredInitiative, *, yes: bool, by: str,
                               details: Mapping[str, Any] | None = None) -> Optional[StoredInitiative]:
        """The owner's word on an offer to stop asking: a yes adds the allow it names (``_adopt_allow``); a no holds
        the next offer back for ``allows.HOLD_DAYS``. Neither is a verdict on asking."""
        offer = Offer.from_dict((row.context or {}).get("proposes") if isinstance(row.context, dict) else None)
        who = self._who(by, details)
        if not yes:
            return self.outcomes.record(row.id, status="denied", by=by, details=details, implicit_verdict=False,
                                        summary=f"{who} said no; I keep asking")
        if offer is None or self.level not in ALLOW_LEVELS:
            return self.outcomes.record(row.id, status="cancelled", by=by, details=details, implicit_verdict=False,
                                        summary="standing allows do not apply at this level; I keep asking")
        allow = await self._adopt_allow(row, offer, by=by, now=self.clock())
        if allow is None:
            return self.outcomes.record(row.id, status="cancelled", by=by, details=details, implicit_verdict=False,
                                        summary="things changed since I asked, so I keep asking")
        return self.outcomes.record(row.id, status="done", verified="owner", by=by, details=details,
                                    implicit_verdict=False, summary=f"{who} said yes: {allow.words}")

    # -- what the mind shows ----------------------------------------------------------------

    def broadcast(self) -> List[Concern]:
        """The top 3 concerns, when the broadcast faculty is on (architecture 4.5)."""
        if not self.faculties["broadcast"] or not self.enabled:
            return []
        return self.concerns.top(k=BROADCAST)

    def section(self, *, limit: int = MIND_SECTION_CHARS) -> str:
        """The Mind section of an owner turn's context: at most ``limit`` characters.

        Affect's notes come first but take only the room the rest leaves (at most 360 characters),
        so they never cut the open asks. The broadcast leaves out idle curiosity (an interest's
        research): in a decision it points at the optional work the consumers postpone. A question
        curiosity raised (a contradiction for the owner) stays. Stances and the turn's lesson
        ride their own sections (``colonymind-stances``, ``colonymind-lessons``), built from the
        turn's text.
        """
        if not self.enabled:
            return ""
        lines: List[str] = []
        broadcast = [concern for concern in self.broadcast()
                     if not (concern.drive == "curiosity" and concern.kind == "interest")]
        if broadcast:
            lines.append("On my mind: " + "; ".join(f"{c.summary}"[:120] for c in broadcast) + ".")
        goals = [self.goals.render(goal) for goal in self.goals.open()]
        if goals:
            lines.append("Working toward: " + "; ".join(goal_lines(goals)) + ".")
        lines.extend(self._waiting_lines(limit - len("\n".join(lines)) - (1 if lines else 0)))
        room = limit - len("\n".join(lines)) - (1 if lines else 0)
        text = "\n".join([*self.feelings.section_lines(min(SECTION_CHARS, room)), *lines])
        if len(text) > limit:
            text = text[: limit - 1].rstrip() + "…"
        return text

    def _waiting_lines(self, room: int) -> List[str]:
        """The open asks for the packet, in words: no code, no number and no id, since the owner answers an ask's
        own message, never a reference to it. A message ask carries its exact words (``asked_words``, what the yes
        sends); one whose words do not fit is named without them."""
        asks = [row for row in self.store.intentions(status=["asked"], limit=5)
                if row.ask_code and row.type not in QUESTION_TYPES]
        others = [(plain.fit(str(row.description or row.type or ""), 120)
                   or "an action I would take").rstrip(plain.ENDS) for row in asks if audit.asked_words(row) is None]
        lines = ["Waiting for your say on: " + "; ".join(others) + "."] if others else []
        room -= len("\n".join(lines)) + (1 if lines else 0)
        for row in asks:
            words = audit.asked_words(row)
            if words is None:
                continue
            line = f"Waiting for your say on: a message to a contact: \"{words}\""
            if len(line) > room:
                line = "A message to a contact waits for your say; its words are in the ask I sent."
            if len(line) <= room:
                lines.append(line)
                room -= len(line) + 1
        return lines

    def state(self) -> Dict[str, Any]:
        now = self.clock()
        grants = self.grants_state(now)
        asks = [{"id": row.id, "code": row.ask_code, "ask_code": row.ask_code, "title": row.description,
                 "kind": row.kind, "decision_reason": row.decision_reason, "message": audit.asked_words(row),
                 "expires_at": row.expires_at.isoformat() if row.expires_at else None}
                for row in self.store.intentions(status=["asked"], limit=500)]
        weights = self.effective_weights()
        return {
            "enabled": self.enabled,
            "autonomy": self.level,
            "off_reason": self.off_reason,
            "queued": len(self.store.intentions(status=["approved"], kind=["task"], limit=500)),
            "outbox": len(self.store.intentions(status=["approved"], kind=["message"], limit=500)),
            "asks": asks,
            "open_asks": len(asks),
            "dispatched": len(self.store.intentions(status=["dispatched"], limit=500)),
            "ticks": self.ticks,
            "last_tick": self.last_tick_at.isoformat() if self.last_tick_at else None,
            "last_pull": self.last_pull_at.isoformat() if self.last_pull_at else None,
            "body_stale": self.body_stale(now),
            "quiet_hours": self.in_quiet_hours(now),
            "breaker": [self.authority.breaker_state(cls, now) for cls in CLASSES if cls != "floor"],
            "budgets": vars(self.policy.budgets),
            "faculties": dict(self.faculties),
            "drives": {name: {"weight": self.drive_weights.get(name, 0.0), "effective": weights.get(name, 0.0),
                              "level": self.drive_levels.get(name, 0.0),
                              "satiety": float((self.mind_state.get(f"satiety.{name}") or {}).get("level") or 0.0)}
                       for name in DRIVES},
            "concerns": {"open": self.concerns.count("open"), "intended": self.concerns.count("intended"),
                         "broadcast": [c.as_dict() for c in self.broadcast()]},
            "goals": [self.goals.render(goal) for goal in self.goals.open()],
            "interests": [{"topic": item.get("text"), "weight": item.get("level")} for item in self.mind_state.items("interest:")],
            "deliberation": {"available": self.deliberation.available, "calls_this_tick": self.deliberation.calls_this_tick,
                             "calls_total": self.deliberation.calls_total, "tokens_total": self.deliberation.tokens_total,
                             "last_error": self.deliberation.last_error},
            "ideas": {"last_refresh": self.ideas.meta().get("last_refresh")},
            "composition": {"available": self.composer.available, "calls_total": self.composer.calls_total,
                            "tokens_total": self.composer.tokens_total, "last_error": self.composer.last_error},
            "observed_at": self.observed_at.isoformat() if self.observed_at else None,
            "body": self.body_heartbeat or None,
            "running": self._running,
            "consolidation": self._consolidation_state(),
            "affect": self.feelings.state(),
            "outreach": self._outreach_state(now),
            "lessons": self._lessons_state(),
            "skills": self.skills.state(),
            "grants": grants["grants"],
            "grants_changed": grants["changed"],
            "allows": [allow.as_dict() for allow in self.allows.in_force()],
            # ``mind.floor.tools`` entries with a problem: each call of such a tool waits for the owner's yes to it.
            "floor_tool_problems": dict(self.policy.floor_tool_problems),
            # ``mind.actions``: what each declared action does, and the problem of one that can run nothing.
            "actions": {name: {"class": spec.get("class"), "description": spec.get("description"),
                               "problem": self.policy.action_problems.get(name)}
                        if isinstance(spec, dict) else {"problem": self.policy.action_problems.get(name)}
                        for name, spec in self.policy.actions.items()},
            "wishes": {"enabled": self.wishes.enabled,
                       "open": [self.wishes.render(row) for row in self.wishes.rows(limit=50)
                                if wish_state(row) in {"asked", "granted", "pursuing", "measuring"}]},
            # Only with the faculty on (off, the state is as before; GET /v1/mind/drift shows a manual run).
            **({"drift": self.drift.state()} if self.faculties.get("drift_check") else {}),
        }

    def _outreach_state(self, now: datetime) -> Dict[str, Any]:
        """Owner outreach as ``state`` shows it: on or off, the pause, the day's count against its budget,
        the muted topics and what care is open."""
        o = outreach_functions
        enabled = bool(self.faculties.get("outreach")) and bool(self.owner_id)
        until = o.pause_until(self.mind_state.get(o.PAUSE_KEY))
        if until is not None and now >= until:
            until = None
        sent = self.store.count_transitions("queued", now - timedelta(days=1), kind="message", recipient=self.owner_id,
                                            include_types=tuple(sorted(OUTREACH_TYPES))) if self.owner_id else 0
        return {"enabled": enabled,
                "paused_until": (o.INDEFINITE if until is not None and until.year == datetime.max.year
                                 else until.isoformat() if until is not None else None),
                "per_day": int(self.policy.budgets.outreach_per_day), "sent_24h": sent,
                "muted": [str(item.get("text") or "") for item in self.mind_state.items(o.MUTE_PREFIX)
                          if float(item.get("level") or 0.0) >= o.MUTE_FLOOR],
                "care": [str(item.get("text") or "") for item in self.mind_state.items(o.CARE_PREFIX)
                         if float(item.get("level") or 0.0) >= o.CARE_MIN_LEVEL]}

    def sync_skills(self, now: datetime | None = None) -> Optional[Dict[str, Any]]:
        """ColonyMind's skills follow its lessons (``Skills.sync``); a failure is logged, never raised."""
        now = now or self.clock()
        try:
            return self.skills.sync(self.lessons.all(), self.lessons.tally(now), now)
        except Exception as error:
            logger.warning("skills not synced (%s)", type(error).__name__)
            return None

    def _lessons_state(self) -> Dict[str, Any]:
        current = self.lessons.all()
        return {"enabled": self.lessons.enabled, "active": sum(item.status == "active" for item in current),
                "candidate": sum(item.status == "candidate" for item in current)}

    def _charge_gate(self, tokens: int) -> None:
        row, _ = self.store.create_intention(
            kind="note", type="owner_gate_review", title="Owner gate review", drive="upkeep",
            cls="internal", decision="act", decision_reason="second opinion", status="done",
            dedup_key=None, hermes_kind="none", created_at=self.clock())
        self.store.update(row.id, cost_tokens=tokens)

    def _consolidation_state(self) -> Dict[str, Any]:
        task = self._consolidation_task
        row = self.consolidation.last_note()
        context = (row.context or {}) if row is not None else {}
        return {"last": self.consolidation.last_date(), "running": bool(task is not None and not task.done()),
                "last_tokens": int(row.cost_tokens or 0) if row is not None else 0,
                "errors": list(context.get("errors") or []), "counts": dict(context.get("counts") or {})}

    def stats(self) -> Dict[str, Any]:
        now = self.clock()
        value = audit.stats(self.store, now=now)
        lessons = self.lessons.stats(now)
        value["lessons"] = lessons
        value["lesson_use_rate"] = lessons.get("use_rate")      # wins over verified uses (evals section 8)
        value["skills"] = {"enabled": self.skills.enabled, "owned": len(self.skills.owned()),
                           "loads": self.skills.loads()}
        value["wishes"] = self.wishes.stats(now)
        return value


__all__ = ["APPRAISAL_FORCED_S", "APPRAISAL_TIMER_S", "DEFAULT_FACULTIES", "DRAIN_FORCED_S", "DRAIN_TIMER_S", "DUE_TYPES", "MIND_SECTION_CHARS", "Mind",
           "OFF_MARKER", "STALE_AFTER", "TASK_WINDOW", "THREADED_PLATFORMS", "WORKER_PROFILE", "faculties_of",
           "split_target", "task_body"]
