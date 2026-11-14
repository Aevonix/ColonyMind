"""``/v1/mind``: the body's pull protocol, the guard, the audit log and the switches.

Architecture 6.2 (dispatch and outbox), 7.5 (guard), 7.7 (asks), 7.8 (audit
log) and 7.9 (off switch). Every route sits behind the one API key. The body
(the Hermes plugin) pulls; the sidecar never pushes.

Interface (JSON; the body half is ``plugins/hermes-plugin/body.py``, the same
text in docs/HERMES-ADAPTER.md):
  GET  /dispatch                  -> [ {id, kind: task, dedup_key, idempotency_key, title, body, assignee,
                                         max_runtime_seconds, max_retries, goal_mode, created_at, ...} ]
  POST /dispatch/{id}/bound       {hermes_ref, hermes_kind?, status?, bound_at?}  -> audit entry (idempotent)
  POST /outcome                   {id, hermes_ref, status, outcome?, final?, summary?, error?, verified?,
                                   run?, ...}                                    -> audit entry
  GET  /outbox                    -> [ {id, kind: message|notice, recipient, recipient_is_owner,
                                         recipient_handles, text, ...} ]
  POST /outbox/{id}/sending       {target?, at?} -> audit entry; 409 unless the message was ready and the
                                   claim names a target (no_target: it stays ready, handles re-read per pull)
  POST /outbox/{id}/sent          {result: sent|failed|uncertain, error?, hermes_ref?, summary?} -> audit entry
  POST /observations              {observed_at, board, body, counts, stale_tasks, blocked_tasks, goals,
                                   mind_tasks} (or the flat {observations: [{kind, ...}]})
  GET  /observations/sources/{source}/cap -> {source, cap, basis: runtime|config|default, set_by?, set_at?}
  PUT  /observations/sources/{source}/cap {cap: 0..1000 | null} -> the same: the owner's runtime daily cap for one
                                   source (null clears it), noted in the audit log; the owner's credential, or an
                                   ``observe`` credential for a source it lists (403 ``source_not_granted``)
  POST /guard                     {tool, args, session|session_id, run, task_id, tool_call_id?, recipients?,
                                   cwd?, effect?, context?} -> {allow, action, reason, held?, grant?, permit?}
  POST /tripwire                  {kind: answer_forgery_attempt, tool, named: [api.key|answer_route], run,
                                   session_id?} -> {ok, id, noticed}: a tool call read the instance key or named an
                                   answer route (it ran: a tripwire, not a boundary); noted, and the owner told at
                                   most once a day
  GET  /grants                    the standing grants: source, limits, validity, expiry, uses and amounts
  POST /grants                    {grant, reason, by, contact_id?, session_id?, message?}: from the owner's
                                   conversation, a grant_confirm ask the owner answers in their chat; naming no
                                   conversation, 403 ``approve_required`` for every credential (a standing grant is
                                   the owner's to give, or the operator's to configure: ``mind.grants``)
  POST /grants/{id}/revoke        {by?, contact_id?}              -> revoked at once (a narrowing)
  GET  /allows?viewer             the owner's standing allows: what each lets go without asking (the words the owner
                                   agreed to), the ask whose "always" added it, when, and its uses; a guest gets none.
                                   Nothing here adds one: only the owner's own "always" to an ask, or their yes to an
                                   offer to stop asking, does
  POST /allows/{id}/revoke        {by?} -> revoked at once (a narrowing, open to the console): the asking comes back
                                   (404 unknown)
  GET  /cron-origin?job_id&created_at -> {origin: mind|owner|unknown, mind, task_id, intention_id}: whose cron job
  POST /cron-origin               {job_id, task_id, tool_call_id, action, ok}: a mind run's cron create or update
                                   returned (the plugin's report); a job it made or changed is the mind's
  POST /lineage                   {child, parent}: a kanban task a mind run made; it counts with its maker
  GET  /log?limit&status&kind&since_hours&recipient, /log/{id}, /why/{id}, /asks, /state (/status), /stats
                                  each of /log, /log/{id}, /why/{id}, /asks and /state takes ``viewer``: a
                                  contact id other than the owner's sees only its own messages' bare rows
                                  (no title, concern, evidence or reason), no asks and only the switches.
                                  Only a credential granted the owner sees the owner's view: without the
                                  grant (development mode too) every read is a guest's, and ``viewer`` only
                                  narrows
  GET  /narrative                 the self-narrative {enabled, text, sections, cites, updated_at} (empty
                                   until the mind exposes narrative())
  GET  /blocks                    the resident blocks: the fossil guard's records, the fossilised ids and the last
                                   context fit (blocks rotated or held back, with SHA-256s)
  POST /consolidate               run the nightly consolidation now (501 until the mind exposes it)
  GET  /drift                     the identity check: {enabled, runs, last, pending, owner_word, trend, agreement}
  POST /drift/run                 run one identity check now (a measurement; it runs with the faculty off too;
                                   429 past the day's cap of runs)
  GET  /concerns, /goals          the workspace (open concerns, the broadcast set) and the open goals
  POST /interests                 {topic, why?} -> a seeded interest the curiosity drive researches
  POST /outreach                  {state: on|off} -> the owner's outreach switch (off pauses unprompted outreach
                                   until turned on; reminders and requested answers keep going) and its state
  POST /owner-gate                {topic, text, urgency: low|normal|high|critical, source?, id?} -> {decision:
                                   send|wait|digest|drop, reason, stage, until, scores, facts, id, replayed, gate}:
                                   an outside alert asks the owner gate before it messages the owner; the same id
                                   asked again gets its standing verdict (docs/OWNER-GATE.md)
  GET  /owner-gate?limit&since_hours -> {enabled, settings, counts, decisions}: the gate's audit
  POST /answers                   {platform, sender_id?, contact_id?, chat_id, chat_type, turn_id?, message_id?,
                                   arrived_at, text?, reply_to_message_id?, reaction?, reaction_removed?,
                                   forwarded?, voice?, own_account?, ledger?: [{at, outbox_id?, message_id?, kind?,
                                   ordered?}]} -> {read, reason?, path, reading, asks, settled, later, shown,
                                   clarify, undone, refused, unbound, context}: the owner's own message read as a
                                   possible answer (``Mind.read_answer``), once, from a platform the credential is
                                   granted the owner on (403 ``owner_not_granted`` otherwise); its words are never
                                   stored
  POST /asks/{id}/shown           {platform, chat_id, message_id, at?, asks?} -> {ok, asks, message_id}: the
                                   plugin showed these open asks (an id or a code; ``asks``: a list, in its order)
                                   as one message in the owner's chat, which binds in that chat only: one ask's
                                   like its own message (a reaction or a quoted reply), a list a quoted reply (its
                                   numbers, for a day) and never a reaction (404 no open ask)
  GET  /permit-asks?state&limit   -> {asks: [view]}: the hands permit asks (open by default; ``settled``, ``all``),
                                   each with its immutable id, status, full owner-facing terms, the choices it offers
                                   and, once settled, who settled it and the winner (channel, choice, time); only the
                                   owner channel relay's ``answer`` credential (403 ``answer_credential_required``).
                                   docs/hands-permit-answer-contract.md
  GET  /permit-asks/{id}          -> view: one hands permit ask, whatever its state (404 ``unknown_ask``)
  POST /permit-asks/{id}/answer   {choice, platform: companion, contact_id? | sender_id?} + ``Idempotency-Key`` ->
                                   {result: settled, ask_id, status, outcome, winner, settlement, replayed}: the
                                   owner's explicit answer from an owner channel, settled through the same guarded
                                   settle path and lock as ``/answers`` (the first commit wins); 409
                                   ``already_answered`` (the recorded winner) or ``ask_closed``, 409
                                   ``idempotency_conflict`` or ``full_text_not_delivered``, 403
                                   ``answer_credential_required``, ``platform_not_allowed`` or ``not_owner``, 404
                                   ``unknown_ask``, 422 ``invalid_choice`` or ``invalid_platform``
  POST /asks/{ident}/show         -> {ok, id, message, text}: the console puts an open ask (an id or a code) in
                                   front of the owner again, as its own message in their chat (404 no open ask)
  POST /asks/{code}/yes|no        a yes: 403 ``approve_required`` for every credential (a yes comes only from the
                                   owner's own message); a no: taken, recorded ``by: operator``
  GET  /asks/stats?since_hours&viewer  the asks report over a window (``mind/asks_report.py``): asks formed, first
                                   sight, who settled them and how, how fast, operator verdicts, held work; counts
                                   and times only. The owner's view: a guest gets nothing
  POST /off {reason?}, /on, /tick, /rate {id, verdict}, /level {autonomy}, /reset {cls}
  GET  /lessons?status&uses&viewer the lessons with their verified tallies (and every use, with how it matched,
                                   and lesson_match_unavailable); a guest gets none
  POST /lessons/{id}/retire       {reason, by?} -> the retired lesson (404 unknown, 409 already closed)
  POST /lessons/relayed           {kind, title, when_to_use, content, said_at, note, topic?} -> the lesson and
                                   ``created``: an instruction the owner gave through another channel, recorded
                                   by the operator with that provenance (``relayed``), never as the owner's own
                                   message; a credential granted the owner only (403 ``not_owner``; 409
                                   ``lessons_off`` or ``lesson_closed``; 422 ``invalid_lesson``)
  GET  /learning?store&action&limit&viewer  the learning rules: counts, the log (rejections included) and the
                                   preference changes waiting for the owner; a guest gets none
  POST /learning/pending/{id}/ratify|decline  {reason?, by?} -> the decided row (404 unknown, 409 decided)
  GET  /wishes?viewer             the mastery drive's wishes, each with its cited evidence, its state and its
                                   measured outcome, and their stats; a guest gets none
  POST /skills/used               {skill, session_id?, task_id?} -> {ok, counted, loads}: a load of a
                                   colonymind-* skill (the plugin's on_skill_lifecycle); others are not counted
"""

from __future__ import annotations

from datetime import timedelta
import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from colonymind.hands.sessions import refuse_hand_session
from pydantic import BaseModel, ConfigDict, Field, model_validator

from colonymind.mind import audit
from colonymind.mind.authority import CLASSES, LEVELS
from colonymind.mind.outcomes import VERDICTS
from colonymind.api.public_spec import PUBLIC

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/mind", tags=["mind"])

_mind: Any = None


def set_mind(mind: Any) -> None:
    global _mind
    _mind = mind


def get_mind() -> Any:
    return _mind


@router.get("/ideas/collection", dependencies=[Depends(refuse_hand_session)])
async def ideas_collection(request: Request):
    from colonymind.mind.ideas_build import owner_action
    owner_action(request)
    if _mind is None:
        raise HTTPException(501, "mind_unavailable")
    return _mind.ideas.collection()


class IdeasSources(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observations: List[dict] = Field(max_length=32)


@router.post("/ideas/sources", dependencies=[Depends(refuse_hand_session)])
async def ideas_sources(body: IdeasSources, request: Request):
    from colonymind.mind.ideas_build import owner_action
    owner_action(request)
    if _mind is None:
        raise HTTPException(501, "mind_unavailable")
    try:
        return _mind.ideas.ingest(body.observations)
    except (ValueError, KeyError, TypeError):
        raise HTTPException(422, "invalid_ideas_sources") from None


def faculty_on(name: str) -> bool:
    """Whether the running mind has faculty ``name`` on; without a mind nothing is ablated."""
    faculties = getattr(_mind, "faculties", None)
    return True if not isinstance(faculties, dict) else bool(faculties.get(name, True))


def _require() -> Any:
    if _mind is None:
        raise HTTPException(status_code=503, detail={"code": "mind_not_wired", "message": "the mind is not running"})
    return _mind


def _entry(row: Any) -> Dict[str, Any]:
    from colonymind.initiatives.models import OWNER_KINDS
    if row is None or getattr(row, "kind", None) in OWNER_KINDS:
        raise HTTPException(status_code=404, detail={"code": "unknown_intention"})
    return audit.entry(row)


EMPTY_NARRATIVE: Dict[str, Any] = {"enabled": False, "text": "", "sections": {}, "cites": [], "updated_at": None}


class BoundBody(BaseModel):
    """The body's ack after ``kanban_db.create_task``; the same ref may arrive again."""
    model_config = ConfigDict(extra="ignore")
    hermes_ref: str = Field(min_length=1, max_length=256)
    hermes_kind: str = Field(default="kanban", max_length=32)
    status: Optional[str] = Field(default=None, max_length=32)
    bound_at: Optional[str] = Field(default=None, max_length=64)


class OutcomeBody(BaseModel):
    """One report on a ``mind:*`` task from its kanban state and last run row.

    ``status`` is the kanban status; ``outcome`` (done, blocked, failed,
    cancelled, uncertain) and ``final`` are the body's reading of it. A report
    with ``final: false`` (a failed run that was requeued) is progress, not a
    settlement. ``verified`` may claim only ``hermes_failure``, and only for a
    failure with a reason (``error`` or ``summary``); a claimed ``owner`` or
    ``check`` is ignored and the mind computes the verifier (architecture 4.8).
    A report on a row the mind never handed the body (an owner question, a
    row still deferred or asked, a note, a goal) changes nothing.
    """
    model_config = ConfigDict(extra="ignore")
    id: str = Field(min_length=1, max_length=64)
    status: str = Field(min_length=1, max_length=32)
    outcome: Optional[str] = Field(default=None, max_length=32)
    final: Optional[bool] = None
    hermes_ref: Optional[str] = Field(default=None, max_length=256)
    hermes_kind: Optional[str] = Field(default=None, max_length=32)
    summary: Optional[str] = Field(default=None, max_length=8000)
    error: Optional[str] = Field(default=None, max_length=2000)
    verified: Optional[str] = Field(default=None, max_length=32)
    result: Optional[Dict[str, Any]] = None
    run: Optional[Dict[str, Any]] = None
    block_kind: Optional[str] = Field(default=None, max_length=64)
    consecutive_failures: Optional[int] = None
    completed_at: Optional[str] = Field(default=None, max_length=64)
    observed_at: Optional[str] = Field(default=None, max_length=64)


class SendingBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    target: Optional[str] = Field(default=None, max_length=256)
    at: Optional[str] = Field(default=None, max_length=64)


class SentBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    result: str = Field(default="uncertain", max_length=32)
    hermes_ref: Optional[str] = Field(default=None, max_length=256)
    summary: str = Field(default="", max_length=2000)
    error: Optional[str] = Field(default=None, max_length=2000)
    at: Optional[str] = Field(default=None, max_length=64)


class GuardBody(BaseModel):
    """``recipients``: the contact ids an effect reaches later (a delivering cron job), resolved by
    the plugin; they are authorized like a message's recipient. ``cwd`` is the directory a mind run's
    command runs in (part of the call a permit binds to), ``effect`` the plugin's classification of the
    call by the shared table (the sidecar classifies again) and ``context`` what only the plugin can read
    (a vault item's kind, the id of the cron job a removal or update names). ``tool_call_id`` names the
    call, so the plugin's report of its result (a cron job's id) settles the mark the guard left."""
    model_config = ConfigDict(extra="allow")
    tool: str = Field(min_length=1, max_length=128)
    args: Dict[str, Any] = Field(default_factory=dict)
    session: Optional[str] = Field(default=None, max_length=256)
    session_id: Optional[str] = Field(default=None, max_length=256)
    run: str = Field(default="mind", max_length=32)
    task_id: Optional[str] = Field(default=None, max_length=256)
    tool_call_id: Optional[str] = Field(default=None, max_length=256)
    recipients: Optional[List[str]] = Field(default=None, max_length=32)
    cwd: Optional[str] = Field(default=None, max_length=4096)
    effect: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None


class GrantBody(BaseModel):
    """A standing grant asked for in the owner's conversation (the plugin's ``colonymind_self grant``: the contact,
    the session and the owner's message): it becomes a ``grant_confirm`` ask the owner answers in their chat. A
    request naming none of them cites no owner turn and adds nothing. ``by`` labels the audit."""
    model_config = ConfigDict(extra="ignore")
    grant: Dict[str, Any]
    reason: str = Field(default="", max_length=1500)
    by: str = Field(default="owner", max_length=64)
    contact_id: Optional[str] = Field(default=None, max_length=256)
    session_id: Optional[str] = Field(default=None, max_length=256)
    message: Optional[str] = Field(default=None, max_length=8000)


class RevokeBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    by: str = Field(default="owner", max_length=64)
    contact_id: Optional[str] = Field(default=None, max_length=256)


class AnswerBody(BaseModel):
    """A console answer: it cites no owner turn, so nothing in it speaks for the owner (a ``by`` label, a contact or
    words a client still sends are ignored)."""
    model_config = ConfigDict(extra="ignore")


class LedgerEntry(BaseModel):
    """One thing the plugin saw reach the owner's chat before a message arrived: an outbox send (``outbox_id``), a
    message by its delivery id, or anything else (``kind``: a model reply, an interim message, a model send, a cron
    delivery), which breaks the follow-up. ``ordered``: the plugin fixed its order among its own sends."""
    model_config = ConfigDict(extra="ignore")
    at: str = Field(min_length=1, max_length=64)
    kind: str = Field(default="other", max_length=32)
    outbox_id: Optional[str] = Field(default=None, max_length=64)
    message_id: Optional[str] = Field(default=None, max_length=256)
    ordered: bool = False


class AnswersBody(BaseModel):
    """One owner message as the plugin saw it, before the model ran (docs/MIND.md, Asks). The sidecar checks the
    sender again (``sender_id`` is resolved to a contact; without it ``contact_id`` is the plugin's resolution),
    and nothing here can carry the model's text or a tool result: an unknown field is dropped."""
    model_config = ConfigDict(extra="ignore")
    platform: str = Field(min_length=1, max_length=64)
    sender_id: Optional[str] = Field(default=None, max_length=256)
    contact_id: Optional[str] = Field(default=None, max_length=256)
    chat_id: Optional[str] = Field(default=None, max_length=256)
    chat_type: Optional[str] = Field(default=None, max_length=32)
    turn_id: Optional[str] = Field(default=None, max_length=256)
    message_id: Optional[str] = Field(default=None, max_length=256)
    arrived_at: Optional[str] = Field(default=None, max_length=64)
    text: Optional[str] = Field(default=None, max_length=8000)
    reply_to_message_id: Optional[str] = Field(default=None, max_length=256)
    reaction: Optional[str] = Field(default=None, max_length=32)
    reaction_removed: bool = False
    forwarded: bool = False
    voice: bool = False
    own_account: bool = False
    ledger: Optional[List[LedgerEntry]] = Field(default=None, max_length=200)


class PermitAnswerBody(BaseModel):
    """An owner channel's explicit answer to one hands permit ask: one of the choices the ask offers, the platform it
    comes from (only ``permit_asks.EXPLICIT_PLATFORMS``), and who answers, as ``/answers`` takes it: ``sender_id``
    (the sender's handle on that platform, resolved by the sidecar) or ``contact_id`` (the caller's resolution).
    Naming the owner never supplies authority: the request's credential must be granted the owner on the platform."""
    model_config = ConfigDict(extra="forbid")
    choice: str = Field(min_length=1, max_length=32)
    platform: str = Field(min_length=1, max_length=64)
    contact_id: Optional[str] = Field(default=None, min_length=1, max_length=256)
    sender_id: Optional[str] = Field(default=None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def _names_who(self) -> "PermitAnswerBody":
        if not self.contact_id and not self.sender_id:
            raise ValueError("name who answers: contact_id or sender_id")
        return self


class ShownBody(BaseModel):
    """A message the plugin showed open asks in, named by its platform, its chat and its delivery id there (a
    delivery id alone names nothing: another chat may use it too)."""
    model_config = ConfigDict(extra="ignore")
    platform: str = Field(min_length=1, max_length=64)
    message_id: str = Field(min_length=1, max_length=256)
    chat_id: str = Field(min_length=1, max_length=256)
    at: Optional[str] = Field(default=None, max_length=64)
    asks: Optional[List[str]] = Field(default=None, max_length=32)


class OffBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(default="owner", max_length=200)
    by: str = Field(default="owner", max_length=64)


class RateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=64)
    verdict: str = Field(min_length=1, max_length=32)


class RetireBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(min_length=1, max_length=400)
    by: str = Field(default="owner", max_length=64)


class RelayedLessonBody(BaseModel):
    """An instruction the owner gave through another channel, as the operator records it. Who recorded it is the
    request's credential, never a field of the body."""
    model_config = ConfigDict(extra="forbid")
    kind: str = Field(min_length=1, max_length=16)
    title: str = Field(min_length=1, max_length=400)
    when_to_use: str = Field(min_length=1, max_length=400)
    content: str = Field(min_length=1, max_length=800)
    said_at: str = Field(min_length=1, max_length=64)
    note: str = Field(min_length=1, max_length=800)
    topic: Optional[str] = Field(default=None, max_length=80)


class SkillUsedBody(BaseModel):
    """The plugin's report of Hermes' ``on_skill_lifecycle(action='loaded')`` for a ColonyMind skill."""
    model_config = ConfigDict(extra="ignore")
    skill: str = Field(min_length=1, max_length=128)
    session_id: Optional[str] = Field(default=None, max_length=256)
    task_id: Optional[str] = Field(default=None, max_length=256)


class LevelBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    autonomy: str = Field(min_length=1, max_length=16)


class ResetBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cls: str = Field(min_length=1, max_length=16)


class InterestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic: str = Field(min_length=1, max_length=160)
    why: str = Field(default="", max_length=400)
    by: str = Field(default="owner", max_length=64)
    delta: float = Field(default=1.0, ge=-1.0, le=1.0)
    ref: Optional[str] = Field(default=None, max_length=128)
    quiet: bool = False


# -- who is asking (integration map X7) ----------------------------------------------------

# What a non-owner viewer may read of a row addressed to them: that the message exists and where it
# stands. Its title, concern, evidence and reason can carry the owner's words about that person.
GUEST_FIELDS = ("id", "created_at", "kind", "type", "recipient", "status", "outcome")


def _guest(mind: Any, viewer: Optional[str], request: Optional[Request] = None) -> bool:
    """Whether the request reads the non-owner view. The owner's view is the request credential's to see
    (``speaks_for_owner``, docs/CREDENTIALS.md): without the owner grant (development mode included) every
    read is a guest's, whatever it names. With the grant, a named viewer other than the owner narrows it (no
    owner configured: every named viewer). The record is the owner's; the adapter already refuses a guest
    session, and these routes fail closed the same way."""
    from colonymind.api.auth import speaks_for_owner
    if not speaks_for_owner(request):
        return True
    return bool(viewer) and str(viewer) != str(getattr(mind, "owner_id", None) or "")


def _viewer(mind: Any, viewer: Optional[str], request: Optional[Request] = None) -> Optional[str]:
    """The guest a non-owner read is for: a viewer the request's credential may act as (never the owner), else
    nobody, who is addressed nothing."""
    from colonymind.api.auth import person_granted, request_authority, speaks_for_owner
    viewer = str(viewer or "").strip() or None
    if viewer is None or speaks_for_owner(request):
        return viewer
    owner = str(getattr(mind, "owner_id", None) or "")
    if viewer == owner or not person_granted(request_authority(request), viewer):
        return None
    return viewer


def _guest_entry(value: Dict[str, Any]) -> Dict[str, Any]:
    row = {name: value.get(name) for name in GUEST_FIELDS}
    row["text"] = f"a {row['type'] or row['kind']} message to you, {row['outcome'] or row['status']}"
    return row


# -- state --------------------------------------------------------------------------------

def _state(viewer: Optional[str], request: Optional[Request]) -> Dict[str, Any]:
    mind = _require()
    value = mind.state()
    if _guest(mind, viewer, request):
        return {"enabled": value.get("enabled"), "autonomy": value.get("autonomy")}
    return value


@router.get("/state", tags=[PUBLIC])
async def state(request: Request, viewer: Optional[str] = None) -> Dict[str, Any]:
    return _state(viewer, request)


@router.get("/status")
async def status(request: Request, viewer: Optional[str] = None) -> Dict[str, Any]:
    """The plugin's route probe, status line and process health counters."""
    from colonymind.health_counters import snapshot
    value = _state(viewer, request)
    if not _guest(_require(), viewer, request):
        value = {**value, "health_counters": snapshot()}
    return value


@router.get("/stats", tags=[PUBLIC])
async def stats() -> Dict[str, Any]:
    value = _require().stats()
    return {**value, "text": audit.render_stats(value)}


@router.get("/concerns", tags=[PUBLIC])
async def concerns(limit: int = 24) -> Dict[str, Any]:
    """The workspace: open concerns by salience, and the broadcast set (architecture 4.5)."""
    mind = _require()
    rows = [c.as_dict() for c in mind.concerns.open(limit=max(1, min(int(limit), 200)))]
    broadcast = [c.id for c in mind.broadcast()]
    lines = [f"{'*' if c['id'] in broadcast else ' '} {c['salience']:.2f} {c['drive']}/{c['kind']}: {c['summary']}"
             for c in rows]
    return {"concerns": rows, "broadcast": broadcast, "drives": mind.state()["drives"],
            "text": "\n".join(lines) if lines else "(nothing on my mind)"}


@router.get("/goals", tags=[PUBLIC])
async def goals() -> Dict[str, Any]:
    """The agent-owned goals that are open (at most ``budgets.open_goals``)."""
    mind = _require()
    rows = [mind.goals.render(goal) for goal in mind.goals.open()]
    lines = [f"{g['id'][:8]}  {g['title']} ({g['steps_done']}/{g['tasks']} steps, until {str(g['horizon'])[:10]})"
             for g in rows]
    return {"goals": rows, "open_goals": mind.policy.budgets.open_goals,
            "text": "\n".join(lines) if lines else "(no open goals)"}


@router.post("/interests")
async def interests(body: InterestBody) -> Dict[str, Any]:
    """Seed an interest for the curiosity drive (the CLI's ``colonymind mind interest <topic>``)."""
    try:
        value = _require().add_interest(body.topic, why=body.why, by=body.by,
                                        delta=body.delta, ref=body.ref, quiet=body.quiet)
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"code": "invalid_interest", "message": str(error)}) from None
    return {"ok": True, **value}


class OutreachBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    state: str = Field(pattern="^(on|off)$")


@router.post("/outreach")
async def outreach_switch(body: OutreachBody) -> Dict[str, Any]:
    """Owner outreach on or off (``colonymind mind outreach on|off``), the recovery path of a pause."""
    return _require().outreach_switch(body.state, by="api")


class OwnerGateBody(BaseModel):
    """An outside alert asking whether it may message the owner now (an ops script, a monitor)."""
    model_config = ConfigDict(extra="ignore")
    topic: str = Field(min_length=1, max_length=160)
    text: str = Field(min_length=1, max_length=2000)
    urgency: Literal["low", "normal", "high", "critical"] = "normal"
    source: str = Field(default="external", min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
    id: Optional[str] = Field(default=None, max_length=128)


@router.post("/owner-gate")
async def owner_gate(body: OwnerGateBody) -> Dict[str, Any]:
    """The owner gate's verdict on an outside alert: send, wait (until), digest or drop; recorded."""
    return await _require().gate_external(body.model_dump())


@router.get("/owner-gate", tags=[PUBLIC])
async def owner_gate_log(limit: int = 50, since_hours: float = 24.0) -> Dict[str, Any]:
    """The owner gate's audit: every verdict with its inputs and scores, and their counts."""
    return _require().gate_log(limit=limit, since_hours=since_hours)


# -- the body's pull protocol (6.2) -------------------------------------------------------

@router.get("/dispatch")
async def dispatch() -> List[Dict[str, Any]]:
    return _require().dispatch()


@router.post("/dispatch/{intention_id}/bound")
async def bound(intention_id: str, body: BoundBody) -> Dict[str, Any]:
    return _entry(_require().bound(intention_id, body.hermes_ref, hermes_kind=body.hermes_kind))


@router.post("/outcome")
async def outcome(body: OutcomeBody) -> Dict[str, Any]:
    """The body's report; a task held for the owner settles with its continuation, not its worker's report."""
    mind = _require()
    report = getattr(mind, "report", None) or mind.outcomes.record
    return _entry(report(
        body.id, status=body.status, outcome=body.outcome, final=body.final, hermes_ref=body.hermes_ref,
        summary=body.summary or "", error=body.error, verified=body.verified, result=body.result, run=body.run))


@router.get("/outbox")
async def outbox() -> List[Dict[str, Any]]:
    return await _require().outbox_ready()


@router.post("/outbox/{intention_id}/sending")
async def outbox_sending(intention_id: str, body: SendingBody | None = None) -> Dict[str, Any]:
    """2xx only when the message was ready; 409 tells the body not to send."""
    mind = _require()
    row = mind.store.get(intention_id)
    if row is None or row.kind != "message":
        raise HTTPException(status_code=404, detail={"code": "unknown_message"})
    if row.status != "approved":
        raise HTTPException(status_code=409, detail={"code": "not_ready", "status": row.status})
    body = body or SendingBody()
    claimed = mind.outbox.sending(intention_id, target=body.target)
    if claimed is None:
        raise HTTPException(status_code=409, detail={"code": "no_target" if not body.target else "not_ready",
                                                     "status": row.status})
    return _entry(claimed)


@router.post("/outbox/{intention_id}/sent")
async def outbox_sent(intention_id: str, body: SentBody) -> Dict[str, Any]:
    return _entry(_require().outbox.sent(intention_id, result=body.result, hermes_ref=body.hermes_ref,
                                         summary=body.summary, error=body.error))


@router.post("/observations")
async def observations(body: Dict[str, Any]) -> Dict[str, Any]:
    """The board as the body sees it (docs/HERMES-ADAPTER.md), or the flat ``observations`` list."""
    return _require().observe(body)


def _observer(mind: Any, source: str, request: Request) -> None:
    """The owner's credential, and an ``observe`` credential only for a source it lists (docs/CREDENTIALS.md)."""
    from colonymind.api.auth import observe_source_granted, request_authority, source_not_granted
    if _guest(mind, None, request):
        raise HTTPException(status_code=403, detail={"code": "not_owner"})
    if not observe_source_granted(request, source):
        raise source_not_granted(request_authority(request), source)


@router.post("/observations/sources/{source}", tags=[PUBLIC])
async def source_observations(source: str, body: Dict[str, Any], request: Request) -> Dict[str, Any]:
    """Replace one outside source's set under the owner's credential."""
    mind = _require()
    _observer(mind, source, request)
    try:
        return await mind.observe_source(source, body)
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"code": "invalid_observation", "message": str(error)}) from error


class SourceCapBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cap: Optional[int] = Field(strict=True, ge=0, le=1000)


@router.get("/observations/sources/{source}/cap", tags=[PUBLIC])
async def source_cap(source: str, request: Request) -> Dict[str, Any]:
    """One source's daily cap and its basis: ``runtime`` (the owner's), ``config`` (``source_caps``) or ``default``."""
    mind = _require()
    _observer(mind, source, request)
    from colonymind.mind.observations import for_mind
    try:
        return for_mind(mind).cap(source)
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"code": "invalid_observation", "message": str(error)}) from error


@router.put("/observations/sources/{source}/cap", tags=[PUBLIC])
async def set_source_cap(source: str, body: SourceCapBody, request: Request) -> Dict[str, Any]:
    """The owner sets one source's daily cap at runtime (``{cap: null}`` clears it); it survives a restart."""
    mind = _require()
    _observer(mind, source, request)
    from colonymind.api.auth import request_authority
    from colonymind.mind.observations import for_mind
    try:
        return for_mind(mind).set_cap(source, body.cap, by=request_authority(request).principal_id)
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"code": "invalid_observation", "message": str(error)}) from error


@router.get("/observations", tags=[PUBLIC])
async def read_observations(request: Request, viewer: Optional[str] = None) -> Dict[str, Any]:
    """Only the owner sees active outside observations and their source counts."""
    mind = _require()
    if _guest(mind, viewer, request):
        return {"observations": [], "sources": []}
    from colonymind.mind.observations import for_mind
    from colonymind.reports.instance import Instance
    from colonymind.reports.senses import collect
    now = mind.clock()
    instance = Instance(mind.state_dir)
    try:
        sources = collect(instance, start=now - timedelta(days=1), end=now)
    finally:
        instance.close()
    return {"enabled": bool(mind.faculties.get("senses", True)),
            "observations": for_mind(mind).active(viewer=mind.owner_id), "sources": sources}


# -- the guard (7.5) ---------------------------------------------------------------------

@router.post("/guard")
async def guard(body: GuardBody) -> Dict[str, Any]:
    mind = _require()
    session_id = body.session_id or body.session or ""
    try:
        return await mind.guard(tool=body.tool, args=body.args, run=body.run, session_id=session_id,
                                recipients=body.recipients, task_id=body.task_id or "", cwd=body.cwd,
                                context=body.context, tool_call_id=body.tool_call_id or "")
    except Exception as error:  # the guard fails closed
        logger.warning("guard check failed (%s)", type(error).__name__)
        return {"allow": False, "action": "block", "reason": f"guard error ({type(error).__name__})"}


# -- the answer tripwire (docs/MIND.md, Asks: console and API answers) -------------------------------------------

TRIPWIRE_KINDS = frozenset({"answer_forgery_attempt"})
#: What a report may say a call named: the instance key, or a route that records the owner's answers. A closed set:
#: nothing of the call itself (its command, its code, its path) reaches the sidecar.
TRIPWIRE_SIGNALS = ("api.key", "answer_route")
TRIPWIRE_RUNS = frozenset({"owner", "guest", "mind"})
#: The owner's warning: a notice (``audit.NOTICE_TYPES``), the mind's report and no drive's work.
TRIPWIRE_NOTICE = "answer_forgery_notice"
#: At most one notice to the owner in this many hours (the approvals design's "at most once a day"): the owner's
#: own lane runs code constantly, so a notice per report would be noise.
TRIPWIRE_NOTICE_HOURS = 24
#: A tool's name as a tool registry spells one; anything else is shown as "a tool" (the owner reads it).
_TOOL_NAME = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


class TripwireBody(BaseModel):
    """The plugin's report of one tool call that read the instance key or named an answer route
    (``plugins/hermes-plugin/guard.py``); only the tool's name and the closed set of what it named."""
    model_config = ConfigDict(extra="ignore")
    kind: str = Field(min_length=1, max_length=64)
    tool: str = Field(min_length=1, max_length=128)
    named: List[str] = Field(default_factory=list, max_length=8)
    run: str = Field(default="unknown", max_length=32)
    session_id: Optional[str] = Field(default=None, max_length=256)
    #: The plugin refused the call (a non-owner session or a mind run); in the owner's own lane it ran.
    refused: bool = False


def _tripwire_text(tool: str, named: List[str], run: str = "owner", refused: bool = False) -> str:
    what = " and ".join({"api.key": "read the key that lets a program speak for you here",
                         "answer_route": "called the routes that record your answers to my asks"}[item]
                        for item in named) or "touched the approval routes"
    if refused:
        where = {"guest": "a conversation with someone else", "mind": "work I started on my own"}.get(run, "a session")
        tried = " and ".join({"api.key": "read the key that lets a program speak for you here",
                              "answer_route": "call the routes that record your answers to my asks"}[item]
                             for item in named) or "touch the approval routes"
        return (f"A tool call ({tool}) in {where} tried to {tried}, and I refused it. If you did not expect this, look "
                f"at what happened: colonymind mind log. I tell you this at most once a day.")
    return (f"A tool I ran ({tool}) {what}. Whoever holds that key can write as you and answer my asks, so if you "
            f"did not ask for this, look at what ran: colonymind mind log. Running my shell and code tools where the "
            f"key cannot be read closes this; colonymind doctor says how. I tell you this at most once a day.")


@router.post("/tripwire")
async def tripwire(body: TripwireBody) -> Dict[str, Any]:
    """A tripwire report: in the owner's own lane the call ran; in a non-owner session or a mind run the plugin
    refused it (``refused``). Each report is a note in the audit log (``colonymind mind log``); the owner is told at
    most once in ``TRIPWIRE_NOTICE_HOURS``."""
    if body.kind not in TRIPWIRE_KINDS:
        raise HTTPException(status_code=422, detail={"code": "unknown_kind", "message": ", ".join(TRIPWIRE_KINDS)})
    mind = _require()
    named = [item for item in TRIPWIRE_SIGNALS if item in body.named]
    run = body.run if body.run in TRIPWIRE_RUNS else "unknown"
    tool = body.tool.strip() if _TOOL_NAME.match(body.tool.strip()) else "a tool"
    now = mind.clock()
    logger.warning("%s: a %s call in a %s session named %s", body.kind, tool, run, ", ".join(named) or "nothing known")
    title = f"{tool} in a {run} session named " + (" and ".join(named) or "an approval route")
    note, _ = mind.store.create_intention(
        kind="note", type=body.kind, title=title[:200], drive="upkeep", cls="internal", decision="act",
        decision_reason="the answer tripwire (the plugin refused the call)" if body.refused else
        "the answer tripwire (a report, not a refusal: the call ran)", status="done", dedup_key=None,
        context={"tool": tool, "named": named, "run": run, "session": body.session_id, "refused": body.refused},
        hermes_kind="none", created_at=now)
    last = mind.store.last_transition_at("queued", type=TRIPWIRE_NOTICE)
    noticed = None
    if last is None or now - last >= timedelta(hours=TRIPWIRE_NOTICE_HOURS):
        message = mind.outbox.notice(type=TRIPWIRE_NOTICE, title="A tool touched the approval key",
                                     text=_tripwire_text(tool, named, run, body.refused),
                                     dedup_key=f"{TRIPWIRE_NOTICE}:{now.strftime('%Y%m%d%H%M%S')}")
        noticed = message.id if message is not None else None
    return {"ok": True, "id": note.id, "noticed": noticed}


class CronJobBody(BaseModel):
    """The plugin's report once a mind run's cron create or update returned: the job's id, or ``ok: false``."""
    model_config = ConfigDict(extra="ignore")
    job_id: str = Field(default="", max_length=256)
    task_id: str = Field(default="", max_length=256)
    tool_call_id: str = Field(default="", max_length=256)
    action: str = Field(default="create", max_length=32)
    ok: bool = True


class LineageBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    child: str = Field(min_length=1, max_length=256)
    parent: str = Field(min_length=1, max_length=256)


@router.get("/cron-origin")
async def cron_origin(job_id: str = "", created_at: Optional[str] = None) -> Dict[str, Any]:
    """Whose cron job this is: ``mind`` (its runs are mind runs), ``owner``, or ``unknown`` (gated until known)."""
    return _require().cron_origin(job_id, created_at)


@router.post("/cron-origin")
async def record_cron_job(body: CronJobBody) -> Dict[str, Any]:
    return _require().record_cron_job(body.job_id, task_id=body.task_id, tool_call_id=body.tool_call_id,
                                      action=body.action, ok=body.ok)


@router.post("/lineage")
async def record_lineage(body: LineageBody) -> Dict[str, Any]:
    try:
        return _require().record_lineage(body.child, body.parent)
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"code": "invalid_lineage", "message": str(error)}) from None


# -- standing grants ---------------------------------------------------------------------

@router.get("/grants", tags=[PUBLIC])
async def grants(request: Request, viewer: Optional[str] = None) -> Dict[str, Any]:
    mind = _require()
    if _guest(mind, viewer, request):
        return {"grants": [], "text": "(no grants)"}
    value = mind.grants_state()
    lines = [f"{item['id']} [{item['class']}, {item['source']}"
             + (", revoked" if item.get("revoked") else ", invalid: " + str(item.get("problem"))
                if not item.get("valid") else ", expired" if item.get("expired") else "")
             + f"] {item['scope']}: {item['uses_24h']} uses in 24 h, {item['amount_30d']:g} moved in 30 days"
             for item in value["grants"]]
    return {**value, "text": "\n".join(lines) or "(no grants)"}


GRANT_REQUIRED = ("a standing grant is the owner's to give in their own chat (the agent files it there and they "
                  "answer) or the operator's to configure (mind.grants in colonymind.yaml, then restart the "
                  "sidecar); a request that cites no owner turn cannot add one, whatever its credential")


@router.post("/grants",
             dependencies=[Depends(refuse_hand_session)])
async def add_grant(body: GrantBody) -> Dict[str, Any]:
    # Nothing here makes a grant active: from the owner's conversation it becomes their ask, answered in their chat;
    # a request that names no conversation (the CLI, a script with the key) cites no owner turn and is refused for
    # every credential, since an agent with a shell can hold any of them (docs/MIND.md, Standing grants).
    if body.contact_id is None and body.session_id is None and body.message is None:
        raise approve_required(GRANT_REQUIRED)
    try:
        return _require().request_grant(body.grant, reason=body.reason, contact_id=body.contact_id)
    except PermissionError as error:
        raise HTTPException(status_code=403, detail={"code": "not_owner", "message": str(error)}) from None
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"code": "invalid_grant", "message": str(error)}) from None


@router.post("/grants/{grant_id}/revoke",
             dependencies=[Depends(refuse_hand_session)])
async def revoke_grant(grant_id: str, body: RevokeBody | None = None) -> Dict[str, Any]:
    body = body or RevokeBody()
    try:
        return _require().revoke_grant(grant_id, by=body.by, contact_id=body.contact_id)
    except PermissionError as error:
        raise HTTPException(status_code=403, detail={"code": "not_owner", "message": str(error)}) from None
    except KeyError:
        raise HTTPException(status_code=404, detail={"code": "unknown_grant",
                                                     "message": f"no standing grant {grant_id}"}) from None


# -- standing allows (docs/MIND.md, Standing grants and allows) ---------------------------------------------------

class AllowRevokeBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    by: str = Field(default="operator", max_length=64)


@router.get("/allows", tags=[PUBLIC])
async def allows(request: Request, viewer: Optional[str] = None) -> Dict[str, Any]:
    """The owner's standing allows; the owner's view (a guest gets none)."""
    mind = _require()
    if _guest(mind, viewer, request):
        return {"allows": [], "text": "(no allows)"}
    return mind.allows_state()


@router.post("/allows/{ident}/revoke",
             dependencies=[Depends(refuse_hand_session)])
async def revoke_allow(ident: str, body: AllowRevokeBody | None = None) -> Dict[str, Any]:
    """Revoke a standing allow at once: a narrowing, so the console may; the asking comes back."""
    body = body or AllowRevokeBody()
    try:
        return await _require().revoke_allow(ident, by=body.by)
    except KeyError:
        raise HTTPException(status_code=404, detail={"code": "unknown_allow",
                                                     "message": f"no standing allow {ident}"}) from None


# -- asks (7.7) --------------------------------------------------------------------------

#: Who a console answer is recorded as: a request that cites no owner turn is never the owner's own answer.
OPERATOR = "operator"


def approve_required(message: str) -> HTTPException:
    """403 ``approve_required``: a request that cites no owner turn may narrow, never widen (docs/MIND.md, Asks)."""
    return HTTPException(status_code=403, detail={"code": "approve_required", "message": message})


def yes_required(code: str) -> HTTPException:
    code = str(code or "").strip().upper() or "<code>"
    return approve_required(f"a yes to an ask comes only from the owner's own message in their chat; no credential "
                            f"approves one by itself. 'colonymind mind show {code}' sends the ask to their chat "
                            f"again, and 'colonymind mind no {code}' declines it here")


async def _answer(code: str, answer: str) -> Dict[str, Any]:
    """``yes|no <code>`` from the console. It cites no owner turn (the owner's own message reaches the mind only
    through ``/answers``), so a yes is refused for every credential and a no is the operator's narrowing, recorded
    ``by: operator``."""
    if answer not in {"yes", "no"}:
        raise HTTPException(status_code=422, detail={"code": "unknown_answer", "message": "answer is yes or no"})
    if answer == "yes":
        raise yes_required(code)
    try:
        row = await _require().answer(code, yes=False, by=OPERATOR)
    except PermissionError as error:
        raise HTTPException(status_code=403, detail={"code": "not_owner", "message": str(error)}) from None
    if row is None:
        raise HTTPException(status_code=404, detail={"code": "unknown_ask", "message": f"no open ask {code.upper()}"})
    return {"ok": True, **audit.entry(row), "text": f"{answer}: {row.description} -> {row.status}"}


# -- the self-narrative and consolidation (architecture 4.1, 4.2) ------------------------

@router.get("/narrative", tags=[PUBLIC])
async def narrative() -> Dict[str, Any]:
    """The self-narrative the plugin renders into the session prompt and ``colonymind_self state`` shows:
    ``{enabled, text, sections, cites, updated_at}``, empty until the mind exposes ``narrative()``."""
    render = getattr(_require(), "narrative", None)
    value = render() if callable(render) else None
    return {**EMPTY_NARRATIVE, **value} if isinstance(value, dict) else dict(EMPTY_NARRATIVE)


@router.get("/blocks")
async def resident_blocks() -> Dict[str, Any]:
    """The resident blocks (``colonymind/memory/blocks.py``): the fossil guard's record of every block the night
    persists (``fossilised`` when its sources moved and its content has not for ``context.fossil_days``), the
    ids of the fossilised ones and the last context fit (what a turn rotated or held back, with SHA-256s)."""
    from colonymind.memory import blocks
    ledger = getattr(getattr(_require(), "consolidation", None), "blocks", None)
    records = ledger.records() if ledger is not None else []
    return {"blocks": records, "fossils": [record["id"] for record in records if record["fossilised"]],
            "last_fit": blocks.last_fit(), "settings": blocks.settings()}


@router.post("/consolidate")
async def consolidate() -> Dict[str, Any]:
    """Run the nightly consolidation now (the CLI, the benchmark); 501 until the mind can."""
    run = getattr(_require(), "consolidate", None)
    if not callable(run):
        raise HTTPException(status_code=501, detail={"code": "consolidation_not_available",
                                                     "message": "this sidecar has no consolidation"})
    return await run(force=True)


# -- the identity check (docs/MIND.md, Identity check) ----------------------------------------

@router.get("/drift", tags=[PUBLIC])
async def drift_state() -> Dict[str, Any]:
    """The identity battery and drift detection: the last run, what is pending, the trend and how often
    the score agreed with the owner's word."""
    check = getattr(_require(), "drift", None)
    if check is None:
        raise HTTPException(status_code=501, detail={"code": "drift_check_not_available",
                                                     "message": "this sidecar has no identity check"})
    return check.state()


@router.post("/drift/run")
async def drift_run() -> Dict[str, Any]:
    """Run one identity check now: the battery, the substrate probe and the signals, audited like any run."""
    run = getattr(_require(), "drift_now", None)
    if not callable(run):
        raise HTTPException(status_code=501, detail={"code": "drift_check_not_available",
                                                     "message": "this sidecar has no identity check"})
    from colonymind.mind.drift import DriftCapReached
    try:
        return await run()
    except DriftCapReached as error:
        raise HTTPException(status_code=429, detail={"code": "drift_check_cap", "message": str(error)}) from None


# -- lessons (architecture 4.8) ------------------------------------------------------------

def _lesson_line(row: Dict[str, Any]) -> str:
    tally = row.get("tally") or {}
    uses = f"{tally.get('wins', 0)} wins in {tally.get('uses', 0)} verified uses"
    source = "relayed by the operator" if row.get("relayed") else row["verified"]
    return (f"{row['id']} [{row['status']}, {row['kind']}, {source}] {row['title']}: "
            f"when {row['when_to_use']} ({uses})")


@router.get("/lessons", tags=[PUBLIC])
async def lessons(request: Request, status: Optional[str] = None, uses: bool = False,
                  viewer: Optional[str] = None) -> Dict[str, Any]:
    """What the mind learned, each lesson with its verified tally, and with ``uses`` every use in the
    tally window. ``status`` filters (comma-separated); a guest viewer gets nothing."""
    mind = _require()
    store = getattr(mind, "lessons", None)
    empty = {"enabled": bool(getattr(store, "enabled", False)), "lessons": [], "uses": []}
    if store is None or _guest(mind, viewer, request):
        return {**empty, "text": "(no lessons)"}
    wanted = {item for item in (status or "").split(",") if item}
    tallies = store.tally()
    rows = [{**lesson.as_dict(), "tally": tallies.get(lesson.id, {"uses": 0, "wins": 0, "losses": 0, "applied": 0})}
            for lesson in store.all(include_closed=True) if not wanted or lesson.status in wanted]
    value = {**empty, "lessons": rows, "uses": store.uses() if uses else []}
    if uses:
        # Each use says how it matched (``own_class`` or ``meaning``); the calls that found no work vector since
        # the sidecar started are counted beside them (``lesson_match_unavailable``).
        value.update(store.match_counts())
    lines = [_lesson_line(row) for row in rows]
    skills = getattr(mind, "skills", None)
    if skills is not None:
        value["skills"] = {**skills.state(), "loads": skills.loads()}
        lines += _skill_lines(value["skills"])
    return {**value, "text": "\n".join(lines) or "(no lessons)"}


def _skill_lines(skills: Dict[str, Any]) -> List[str]:
    """The promoted lessons ColonyMind keeps as skills, with how often Hermes loaded each."""
    loads = skills.get("loads") or {}
    return [f"skill {name}: {int(loads.get(name, 0))} loads" for name in skills.get("owned") or []]


@router.get("/wishes", tags=[PUBLIC])
async def wishes(request: Request, viewer: Optional[str] = None) -> Dict[str, Any]:
    """The wishes the mastery drive made (``mind.faculties.wishes``), newest first: what was wished for, the
    evidence it cites, what the owner said, and whether the failure class fell afterwards."""
    mind = _require()
    store = getattr(mind, "wishes", None)
    if store is None or _guest(mind, viewer, request):
        return {"enabled": bool(getattr(store, "enabled", False)), "wishes": [], "stats": {}, "text": "(no wishes)"}
    rows = [store.render(row) for row in store.rows(limit=100)]
    lines = [f"{row['id'][:8]}  [{row['state']}{' ' + row['ask_code'] if row['ask_code'] else ''}] "
             f"{row['capability']} (for {row['class']})"
             + (f": failure rate {row['result']['before_rate']} -> {row['result']['after_rate']}"
                if isinstance(row.get("result"), dict) and row["result"].get("after_rate") is not None else "")
             for row in rows]
    return {"enabled": bool(store.enabled), "wishes": rows, "stats": store.stats(),
            "text": "\n".join(lines) if lines else ("(no wishes)" if store.enabled else "wishes: off")}


@router.post("/lessons/{lesson_id}/retire")
async def retire_lesson(lesson_id: str, body: RetireBody) -> Dict[str, Any]:
    """The owner retires a lesson (the CLI, or the key holder); it stays listed as retired."""
    store = getattr(_require(), "lessons", None)
    lesson = store.get(lesson_id) if store is not None else None
    if lesson is None:
        raise HTTPException(status_code=404, detail={"code": "unknown_lesson", "message": f"no lesson {lesson_id}"})
    if lesson.status not in {"active", "candidate"}:
        raise HTTPException(status_code=409, detail={"code": "lesson_closed", "message": f"{lesson_id} is {lesson.status}"})
    return store.retire(lesson_id, reason=body.reason, by=body.by).as_dict()


@router.post("/lessons/relayed")
async def relay_lesson(body: RelayedLessonBody, request: Request) -> Dict[str, Any]:
    """An instruction the owner gave through another channel (a call, in person), recorded by an operator as a
    lesson with that provenance (``Lessons.relay``): who recorded it (this request's credential), that it relays the
    owner, when the owner said it and the operator's evidence note. It is never the owner's own message or verdict,
    and it is listed, retired and erased like any lesson. Only a credential granted the owner may record the owner's
    instruction; development mode and other credentials are refused."""
    from colonymind.api.auth import request_authority, speaks_for_owner
    from colonymind.mind.lessons import CURRENT
    mind = _require()
    if not speaks_for_owner(request):
        raise HTTPException(status_code=403, detail={"code": "not_owner", "message": "only a credential granted "
                                                     "the owner may record an instruction the owner gave"})
    store = getattr(mind, "lessons", None)
    if store is None or not store.available:
        raise HTTPException(status_code=503, detail={"code": "no_lessons",
                                                     "message": "no lesson record here (no owner or ledger)"})
    if not store.enabled:
        raise HTTPException(status_code=409, detail={"code": "lessons_off", "message": "the lessons faculty is "
                                                     "off (mind.faculties.lessons): nothing new is admitted"})
    authority = request_authority(request)
    try:
        lesson, created = store.relay(body.model_dump(exclude={"said_at", "note"}), said_at=body.said_at,
                                      note=body.note, recorded_by=authority.identity or authority.principal_id)
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"code": "invalid_lesson", "message": str(error)}) from None
    if lesson is None:
        raise HTTPException(status_code=503, detail={"code": "not_written", "message": "the lesson was not "
                                                     "written: the ledger refused it (words once erased are "
                                                     "never recorded again)"})
    if lesson.status not in CURRENT:
        raise HTTPException(status_code=409, detail={"code": "lesson_closed", "message": f"{lesson.id} is "
                                                     f"{lesson.status}: relay a changed instruction instead"})
    return {**lesson.as_dict(), "created": created}


# -- the learning rules (memory/learning.py) ------------------------------------------------------------

def _learning_ledger(mind: Any) -> Any:
    lessons = getattr(mind, "lessons", None)
    ledger = getattr(lessons, "ledger", None) if lessons is not None else None
    if ledger is None:
        raise HTTPException(status_code=503, detail={"code": "no_ledger", "message": "the source ledger is not wired"})
    return ledger


@router.get("/learning", tags=[PUBLIC])
async def learning_state(request: Request, store: Optional[str] = None, action: Optional[str] = None,
                         limit: int = 50, viewer: Optional[str] = None) -> Dict[str, Any]:
    """Counts by store and action, the newest log rows and the preference changes held for the owner."""
    from contextlib import closing
    from colonymind.memory import learning
    mind = _require()
    if _guest(mind, viewer, request):
        return {"enabled": learning.enabled(), "summary": {}, "log": [], "pending": []}
    with closing(_learning_ledger(mind)._connect()) as conn:
        return {"enabled": learning.enabled(), "summary": learning.summary(conn),
                "log": learning.entries(conn, store=store, action=action, limit=limit),
                "pending": learning.pending(conn)}


class PendingVerdictBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(default="", max_length=400)
    by: str = Field(default="owner", max_length=64)


@router.post("/learning/pending/{pending_id}/{verdict}")
async def decide_learning(pending_id: str, verdict: str, body: PendingVerdictBody) -> Dict[str, Any]:
    """The owner's verdict on a held preference change: ``ratify`` retires the old statement, ``decline``
    keeps both."""
    from colonymind.beliefs.source_projection import SourceClaimProjection
    if verdict not in {"ratify", "decline"}:
        raise HTTPException(status_code=404, detail={"code": "unknown_verdict", "message": verdict})
    projection = SourceClaimProjection(_learning_ledger(_require()))
    row = projection.decide(pending_id, ratify=verdict == "ratify", by=body.by, reason=body.reason)
    if row is None:
        raise HTTPException(status_code=404, detail={"code": "unknown_pending", "message": f"no pending {pending_id}"})
    if not row.get("changed"):
        raise HTTPException(status_code=409, detail={"code": "already_decided", "message": f"{pending_id} is {row['status']}"})
    return row


@router.post("/skills/used")
async def skill_used(body: SkillUsedBody) -> Dict[str, Any]:
    """One load of a skill; only ColonyMind's own (``colonymind-*``) are counted."""
    skills = getattr(_require(), "skills", None)
    loads = skills.record_use(skill=body.skill, session_id=body.session_id, task_id=body.task_id) \
        if skills is not None else None
    return {"ok": True, "skill": body.skill, "counted": loads is not None, "loads": loads or 0}


# -- the audit log (7.8) -----------------------------------------------------------------

@router.get("/log", tags=[PUBLIC])
async def log(request: Request, limit: int = 20, status: Optional[str] = None, kind: Optional[str] = None,
              since_hours: Optional[float] = None, recipient: Optional[str] = None,
              viewer: Optional[str] = None, split: bool = False) -> Dict[str, Any]:
    """Newest first. ``since_hours`` and ``recipient`` answer "did I message p-07 yesterday?" exactly.
    A non-owner ``viewer`` gets only the rows addressed to them, bare (``GUEST_FIELDS``). ``split`` is the
    agent's own reading (``colonymind_self log``): its actions (``audit.is_action``) under ``actions`` and the
    rest (the nightly consolidation, notices) under ``notes``, so a note is never read as something it did."""
    mind = _require()
    guest = _guest(mind, viewer, request)
    if guest:
        viewer = _viewer(mind, viewer, request)
        if not viewer or (recipient and recipient != viewer):
            return {"entries": [], "text": "(nothing addressed to you)"}
        recipient = viewer
    since = None
    if since_hours is not None:
        since = mind.clock() - timedelta(hours=max(0.0, float(since_hours)))
    entries = audit.log(mind.store, limit=max(1, min(int(limit), 500)), since=since,
                        status=[s for s in (status or "").split(",") if s] or None,
                        kind=[k for k in (kind or "").split(",") if k] or None, recipient=recipient or None)
    if guest:
        entries = [_guest_entry(item) for item in entries if item.get("recipient") == viewer]
        return {"entries": entries, "text": "\n".join(item["text"] for item in entries) or "(nothing addressed to you)"}
    if split:
        actions, notes = audit.split(entries)
        return {"actions": actions, "notes": notes, "text": audit.render_split(actions, notes)}
    return {"entries": entries, "text": audit.render_log(entries)}


@router.get("/why/{intention_id}", tags=[PUBLIC])
async def why(intention_id: str, request: Request, viewer: Optional[str] = None) -> Dict[str, Any]:
    mind = _require()
    value = audit.why(mind.store, intention_id)
    guest = _guest(mind, viewer, request)
    viewer = _viewer(mind, viewer, request) if guest else viewer
    if value is None or (guest and (not viewer or value.get("recipient") != viewer)):
        # A guest asking about someone else's row hears exactly what an unknown id gets.
        raise HTTPException(status_code=404, detail={
            "code": "unknown_intention", "message": f"no intention {intention_id} exists in the audit log"})
    return _guest_entry(value) if guest else value


@router.get("/log/{intention_id}", tags=[PUBLIC])
async def log_entry(intention_id: str, request: Request, viewer: Optional[str] = None) -> Dict[str, Any]:
    return await why(intention_id, request, viewer=viewer)


@router.get("/asks", tags=[PUBLIC])
async def asks(request: Request, viewer: Optional[str] = None) -> Dict[str, Any]:
    mind = _require()
    if _guest(mind, viewer, request):   # every ask is the owner's question
        return {"asks": [], "text": "(no open asks)"}
    entries = mind.asks()
    lines = [f"[{item['ask_code']}] {item['title']} ({item['decision_reason']}; expires {item['expires_at']})"
             + (f"\n  would send to {item['recipient']}: \"{item['message']}\"" if item.get("message") is not None else "")
             for item in entries]
    return {"asks": entries, "text": "\n".join(lines) if lines else "(no open asks)"}


@router.get("/asks/stats", tags=[PUBLIC])
async def asks_stats(request: Request, since_hours: float = 168.0, viewer: Optional[str] = None) -> Dict[str, Any]:
    """The asks report for the last ``since_hours`` (a week by default, at most a year): the owner's view only.
    The answer path's own log (unbound answers, invented code lines) joins it where the mind keeps one."""
    from colonymind.mind import asks_report
    mind = _require()
    if _guest(mind, viewer, request):
        return {"owner_view": False, "text": "(the asks report is the owner's)"}
    now = mind.clock()
    since = now - timedelta(hours=max(0.0, min(float(since_hours), 24.0 * 366)))
    events = getattr(mind, "answer_events", None)
    value = asks_report.report(mind.store, since=since, now=now, events=events(since) if callable(events) else None)
    return {**value, "text": asks_report.render(value)}


async def _sender_contact(mind: Any, platform: str, sender_id: str) -> Optional[str]:
    """The contact a sender already is (``ParticipantResolver.lookup``: nothing is created or proposed); None when
    unknown or unreadable, which is never the owner."""
    contacts = getattr(mind, "contacts", None)
    if contacts is None:
        return None
    from colonymind.identity.participants import ParticipantResolver
    try:
        found = await ParticipantResolver(contacts).lookup(platform=platform, user_id=sender_id)
    except Exception:
        logger.debug("answer sender unresolved", exc_info=True)
        return None
    return found.contact_id


def _reader_platform(authority: Any, platform: str) -> None:
    """An explicit-answer channel (``permit_asks.EXPLICIT_PLATFORMS``) reaches the answer readers (``/answers``,
    ``/asks/{id}/shown``) from the instance key only: the Hermes adapter's own chat lane on that channel. Any other
    credential granted the owner there (its relay's ``answer`` credential included) could otherwise register its own
    copy of an ask as shown and quote it, approving any kind of ask without the explicit route's checks."""
    from colonymind.api.auth import KEY_PRINCIPAL
    from colonymind.mind.permit_asks import EXPLICIT_PLATFORMS
    if platform in EXPLICIT_PLATFORMS and authority.principal_id != KEY_PRINCIPAL:
        raise HTTPException(status_code=403, detail={
            "code": "platform_not_allowed",
            "message": f"credential {authority.principal_id} may not read answers from {platform}: an answer there is "
                       "the owner's explicit answer by the ask's id (/v1/mind/permit-asks/{id}/answer)"})


@router.post("/answers",
             dependencies=[Depends(refuse_hand_session)])
async def answers(body: AnswersBody, request: Request) -> Dict[str, Any]:
    """The owner's own message read as a possible answer to the open asks (``Mind.read_answer``): only from a
    platform the request's credential is granted the owner on, as for a turn capture (docs/CREDENTIALS.md)."""
    from colonymind.api.auth import owner_not_granted, owner_platform_granted, request_authority
    mind = _require()
    authority = request_authority(request)
    platform = body.platform.strip().lower()
    _reader_platform(authority, platform)
    if not owner_platform_granted(authority, platform):
        raise owner_not_granted(authority, platform)
    contact = body.contact_id
    if body.sender_id:
        contact = await _sender_contact(mind, platform, body.sender_id)
    evidence = body.model_dump(exclude={"sender_id"})
    evidence.update(platform=platform, contact_id=contact)
    result = await mind.read_answer(evidence, granted=True)
    from colonymind.hands.approvals import answer_grants
    try:
        grants = await answer_grants(mind, result.get("approved", []),
                                     store=getattr(request.app.state, "hands_store", None))
    except Exception:
        # The recorded answer remains valid. Runtime polling refreshes it again;
        # an unavailable hand store cannot fabricate a grant for this response.
        logger.warning("hand permits will refresh when the runtime polls", exc_info=True)
        grants = []
    return {**result, "granted_hands": grants}


@router.post("/asks/{ident}/shown",
             dependencies=[Depends(refuse_hand_session)])
async def ask_shown(ident: str, body: ShownBody, request: Request) -> Dict[str, Any]:
    """The plugin showed open asks in the owner's chat as one message (``Mind.record_shown``); the owner's view."""
    from colonymind.api.auth import request_authority
    mind = _require()
    if _guest(mind, None, request):
        raise HTTPException(status_code=403, detail={"code": "not_owner", "message": "asks are the owner's"})
    _reader_platform(request_authority(request), str(body.platform or "").strip().lower())
    try:
        return mind.record_shown(body.asks or [ident], platform=body.platform, message_id=body.message_id,
                                 chat=body.chat_id, at=body.at)
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"code": "unnamed_message", "message": str(error)})
    except KeyError as error:
        raise HTTPException(status_code=404, detail={"code": "unknown_ask",
                                                     "message": f"no open ask {error.args[0] if error.args else ''}"})


# -- hands permit asks, read and answered by id from an owner channel (docs/hands-permit-answer-contract.md) --------

PERMIT_ASK_STATES = ("open", "settled", "all")


def _answer_caller(request: Request) -> Any:
    """The request's authority when it is an ``answer`` credential (the middleware checks the same first), else 403
    ``answer_credential_required``: these routes are the owner channel relay's, never the instance key's."""
    from colonymind.api.auth import answer_credential, answer_credential_required, request_authority
    authority = request_authority(request)
    if not answer_credential(authority):
        raise answer_credential_required(authority)
    return authority


def _capture_authority(request: Request) -> str:
    from colonymind.api.auth import request_authority, speaks_for_owner
    authority = request_authority(request)
    if not speaks_for_owner(request) or "manage" not in authority.access:
        raise HTTPException(403, detail={"code": "not_owner"})
    return authority.principal_id


@router.post("/vault-captures")
async def vault_capture_create(request: Request) -> Dict[str, Any]:
    """Schema-only creation by the body. It grants nothing and accepts no answer or typed value."""
    from colonymind.mind import permit_asks
    principal = _capture_authority(request)
    mind = _require()
    try:
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 4096:
                raise ValueError()
        body = json.loads(raw)
        if not isinstance(body, dict) or set(body) != {"payload", "authorization_id"}:
            raise ValueError()
        row = permit_asks.create_capture(mind, body["payload"], body["authorization_id"])
        if row.context.get("capture_principal", principal) != principal:
            raise ValueError()
    except (ValueError, TypeError, KeyError, UnicodeError):
        raise HTTPException(422, detail={"code": "invalid_capture_payload"}) from None
    # Bind polling to the creating principal; this neither creates nor widens answer access.
    context = {**row.context, "capture_principal": principal}
    mind.store.update(row.id, context=context)
    return {"ask_id": row.id}


@router.get("/vault-captures/{ask_id}")
async def vault_capture_state(ask_id: str, request: Request) -> Dict[str, Any]:
    """The creating body may observe settlement, but never answer with its instance key."""
    from colonymind.mind import permit_asks
    principal = _capture_authority(request)
    mind = _require()
    row = mind.store.get(ask_id)
    if (not permit_asks.is_permit_ask(row) or "payload" not in row.context
            or row.context.get("capture_principal") != principal):
        raise HTTPException(404, detail={"code": "unknown_ask"})
    state = permit_asks.view(mind.store, row, now=mind.clock())
    return {"open": state["open"], "winner": state["winner"]}


async def _origin_body(request: Request) -> Dict[str, Any]:
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > 32768:
            raise ValueError()
    return json.loads(raw)


@router.post("/tool-origins")
async def tool_origin_register(request: Request) -> Dict[str, Any]:
    from colonymind.api.auth import request_authority
    if request_authority(request).credential_id != "api.key":
        raise HTTPException(403, detail={"code": "adapter_credential_required"})
    try:
        _require().tool_origins.register(await _origin_body(request))
        return {"registered": True}
    except (ValueError, TypeError, KeyError, UnicodeError):
        raise HTTPException(422, detail={"code": "invalid_tool_origin"}) from None


def _tool_origin_principal(request: Request) -> str:
    from colonymind.api.auth import request_authority
    from colonymind.api.credentials import TOOL_ORIGIN
    authority = request_authority(request)
    if (not authority.authenticated or authority.credential_id == "api.key" or authority.owner_platforms
            or authority.scopes is not None or authority.access != frozenset({TOOL_ORIGIN})):
        raise HTTPException(403, detail={"code": "tool_origin_credential_required"})
    return authority.principal_id


@router.post("/tool-origins/claim")
async def tool_origin_claim(request: Request) -> Dict[str, Any]:
    _tool_origin_principal(request)
    try:
        result = _require().tool_origins.claim(await _origin_body(request))
    except (ValueError, TypeError, KeyError, UnicodeError):
        raise HTTPException(422, detail={"code": "invalid_tool_origin"}) from None
    if result is None:
        raise HTTPException(404, detail={"code": "unknown_tool_origin"})
    return result


@router.post("/owner-actions", dependencies=[Depends(refuse_hand_session)])
async def owner_action_create(request: Request) -> Dict[str, Any]:
    principal = _tool_origin_principal(request)
    try:
        return _require().owner_actions.request(await _origin_body(request), principal)
    except (ValueError, TypeError, KeyError, UnicodeError):
        raise HTTPException(422, detail={"code": "invalid_owner_action"}) from None


@router.get("/permit-asks")
async def permit_asks_list(request: Request, state: str = "open", limit: int = 50) -> Dict[str, Any]:
    """The hands permit asks, newest first: ``open`` (asked and not past expiry), ``settled`` or ``all``, each in
    full (``permit_asks.view``). The owner's view."""
    from colonymind.mind import permit_asks
    from colonymind.mind.outbox import FLOOR_EFFECT
    _answer_caller(request)
    mind = _require()
    if state not in PERMIT_ASK_STATES:
        raise HTTPException(status_code=422, detail={"code": "invalid_state",
                                                     "message": f"state is one of {', '.join(PERMIT_ASK_STATES)}"})
    limit = max(1, min(int(limit), 200))
    now = mind.clock()
    rows = mind.store.intentions(kind=["note"], types=[FLOOR_EFFECT], status=["asked"] if state == "open" else None,
                                 limit=500)
    rows = [row for row in rows if permit_asks.is_permit_ask(row)
            and (state == "all" or permit_asks.is_open(row, now) == (state == "open"))]
    return {"asks": [permit_asks.view(mind.store, row, now=now) for row in rows[:limit]], "state": state}


@router.get("/permit-asks/{ask_id}")
async def permit_ask(ask_id: str, request: Request) -> Dict[str, Any]:
    """One hands permit ask by its immutable id, whatever its state (``permit_asks.view``). The owner's view."""
    from colonymind.mind import permit_asks
    mind = _require()
    from colonymind.mind import owner_actions
    if not owner_actions.may_read(request, mind):
        _answer_caller(request)
    row = mind.store.get(ask_id)
    if not permit_asks.is_permit_ask(row):
        raise HTTPException(status_code=404, detail={"code": "unknown_ask", "message": f"no hands permit ask {ask_id}"})
    return permit_asks.view(mind.store, row, now=mind.clock())


@router.post("/permit-asks/{ask_id}/answer",
             dependencies=[Depends(refuse_hand_session)])
async def permit_ask_answer(ask_id: str, body: PermitAnswerBody, request: Request,
                            idempotency_key: str = Header(alias="Idempotency-Key", min_length=1, max_length=200)
                            ) -> Any:
    """The owner's explicit answer to one hands permit ask from an owner channel (``Mind.answer_explicit``). Only the
    channel relay's own ``answer`` credential answers (403 ``answer_credential_required``: never the instance key, a
    credential granted the owner on every platform, or development mode), from a platform it names (``platform`` and
    ``choice`` taken exactly as spelled: 422 otherwise), as the owner (``sender_id`` resolved by the contact store,
    else ``contact_id``; 403 ``not_owner``). The hand's grant is then refreshed exactly as after an RCS answer;
    resuming it stays with the hands runtime."""
    from colonymind.mind import permit_asks
    authority = _answer_caller(request)
    mind = _require()
    key = idempotency_key.strip()
    if not key:
        raise HTTPException(status_code=422, detail={"code": "invalid_idempotency_key",
                                                     "message": "Idempotency-Key must not be blank"})
    platform = body.platform
    if not permit_asks.canonical(platform):
        raise HTTPException(status_code=422, detail={
            "code": "invalid_platform", "message": "platform is a platform name as such: lowercase, without spaces"})
    if platform not in permit_asks.EXPLICIT_PLATFORMS:
        raise HTTPException(status_code=403, detail={
            "code": "platform_not_allowed",
            "message": f"explicit answers come only from {', '.join(sorted(permit_asks.EXPLICIT_PLATFORMS))}; a "
                       "messaging channel's answer is the owner's own message, read by /v1/mind/answers"})
    contact = body.contact_id
    if body.sender_id:
        contact = await _sender_contact(mind, platform, body.sender_id)
    try:
        code, result = await mind.answer_explicit(
            ask_id, choice=body.choice, channel=platform, contact_id=contact, authority=authority, key=key,
            request_hash=permit_asks.request_hash(ask_id, body.model_dump()))
    except permit_asks.PermitAnswerError as error:
        raise HTTPException(status_code=error.status, detail=error.detail())
    if result.get("result") == permit_asks.SETTLED:
        from colonymind.hands.approvals import answer_grants
        try:
            await answer_grants(mind, [ask_id] if result.get("status") == "approved" else [],
                                store=getattr(request.app.state, "hands_store", None))
        except Exception:
            # The recorded answer stands; the runtime's poll of held permits refreshes the grant.
            logger.warning("hand permits will refresh when the runtime polls", exc_info=True)
    if code == 200:
        return result
    words = {permit_asks.ALREADY_ANSWERED: "the ask was already answered; this answer settled nothing",
             permit_asks.ASK_CLOSED: "the ask closed before this answer; it settled nothing"}
    return JSONResponse(status_code=code, content={"detail": {
        "code": result["result"], "message": words.get(result["result"], result["result"]), **result}})


def _open_ask(mind: Any, ident: str) -> Any:
    """The open ask an id or a code names (an owner question has no code and is answered in words), or None."""
    from colonymind.mind.outbox import QUESTION_TYPES
    ident = str(ident or "").strip()
    if not ident:
        return None
    row = mind.store.get(ident)
    if row is None or not row.ask_code:
        row = mind.store.get_by_ask_code(ident.upper())
    if row is None or row.status != "asked" or not row.ask_code or row.type in QUESTION_TYPES:
        return None
    return row


@router.post("/asks/{ident}/show",
             dependencies=[Depends(refuse_hand_session)])
async def ask_show(ident: str, request: Request) -> Dict[str, Any]:
    """The console's way to an approval: an open ask goes to the owner's chat again, as its own message in plain
    words (``Mind._show_ask``, the one a show in chat uses), where their own answer approves it; one waiting to go
    is not doubled. The owner's view."""
    from colonymind.mind.outbox import ask_sentence
    mind = _require()
    if _guest(mind, None, request):
        raise HTTPException(status_code=403, detail={"code": "not_owner", "message": "asks are the owner's"})
    row = _open_ask(mind, ident)
    if row is None:
        raise HTTPException(status_code=404, detail={"code": "unknown_ask", "message": f"no open ask {ident}"})
    name = None
    if row.entity_id and mind._named_in_ask(row):
        name = (await mind._names([row.entity_id])).get(str(row.entity_id))
    sentence = ask_sentence(row, name=name)
    now = mind.clock()
    message = mind._show_ask(row, sentence, key=f"console:{row.id}:{now.isoformat()}", now=now,
                             why="the operator asked to show it again", name=name)
    if message is None:
        raise HTTPException(status_code=409, detail={"code": "not_shown",
                                                     "message": "no owner is configured to show it to"})
    return {"ok": True, "id": row.id, "message": message, "text": f"sent to the owner's chat again: {sentence}"}


@router.post("/asks/{code}/{answer}",
             dependencies=[Depends(refuse_hand_session)])
async def answer(code: str, answer: str, body: AnswerBody | None = None) -> Dict[str, Any]:
    """The console's answer (the CLI, a script with the key): it cites no owner turn, so a yes is refused for every
    credential and a no is the operator's."""
    if answer not in {"yes", "no"}:
        raise HTTPException(status_code=404, detail={"code": "unknown_answer"})
    return await _answer(code, answer)


# -- switches and settings (7.9, 7.10) ----------------------------------------------------

@router.post("/off")
async def off(body: OffBody | None = None) -> Dict[str, Any]:
    body = body or OffBody()
    return _require().off(reason=body.reason, by=body.by)


@router.post("/on")
async def on(body: OffBody | None = None) -> Dict[str, Any]:
    body = body or OffBody()
    return _require().on(by=body.by)


@router.post("/tick")
async def tick() -> Dict[str, Any]:
    return await _require().tick(force=True)


@router.post("/rate")
async def rate(body: RateBody) -> Dict[str, Any]:
    if body.verdict.lower() not in VERDICTS:
        raise HTTPException(status_code=422, detail={"code": "unknown_verdict", "message": ", ".join(VERDICTS)})
    return _entry(_require().rate(body.id, body.verdict.lower()))


@router.post("/level")
async def level(body: LevelBody) -> Dict[str, Any]:
    if body.autonomy.lower() not in LEVELS:
        raise HTTPException(status_code=422, detail={"code": "unknown_level", "message": ", ".join(LEVELS)})
    return _require().set_level(body.autonomy.lower())


@router.post("/reset")
async def reset(body: ResetBody) -> Dict[str, Any]:
    if body.cls not in CLASSES:
        raise HTTPException(status_code=422, detail={"code": "unknown_class", "message": ", ".join(CLASSES)})
    return _require().reset(body.cls)


__all__ = ["faculty_on", "get_mind", "router", "set_mind"]
