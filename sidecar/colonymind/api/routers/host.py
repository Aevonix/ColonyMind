"""ColonyMind sidecar host router — ``/v1/host`` API surface.

This is the contract used by external agent harnesses (OpenClaw and any
future shim) to mount ColonyMind's intelligence as a plugin.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import threading
import time
import uuid
import weakref
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional

import anyio
from fastapi import BackgroundTasks, Depends, APIRouter, Body, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from colonymind.hands.sessions import refuse_hand_session
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from colonymind import get_state_dir
from colonymind.memory import recall_exit
from colonymind.events.stream import EventSubscriberBuffer
from colonymind.api.auth import observe_source_granted
from colonymind.api.auth import (
    owner_person_id,
    request_authority,
    resolve_request_person,
    resolve_turn_person,
)

from colonymind.turns.observed import ObservedConversationRequest, ObservedConversationResponse
from colonymind.secrets.vault import VaultUnavailable
from colonymind.util.temporal import now_utc as _temporal_now
from colonymind.api.public_spec import PUBLIC, capture_responses
from colonymind.api.owner_location import OwnerLocationRequest, require_companion_owner
from colonymind.api.schemas.approval_history import ApprovalHistoryDetail, ApprovalHistoryPage
from colonymind.api.schemas.host import (
    HostIdentity,
    HostMessage,
    HostTurnContext,
    HostConfigureRequest,
    HostConfigureResponse,
    ModelInfo,
    ModelListResponse,
    BackfillRequest,
    BackfillResponse,
    ContactCreateRequest,
    ContactIntroRequest,
    ContactIntroResponse,
    ContactListResponse,
    ContactResponse,
    ContactStyleRequest,
    ContactStyleResponse,
    ContactTimezoneRequest,
    ScopeAuthzResponse,
    ScopeCreateRequest,
    ScopeDeactivateRequest,
    ScopePromoteRequest,
    ScopeResponse,
    SourceReference,
    SourceAnnotationCheck,
    SourceInputReference,
    ContextAssembleRequest,
    ContextAssembleResponse,
    ContextSection,
    TemporalConfigRequest,
    TemporalConfigResponse,
    TemporalContact,
    TemporalContactsResponse,
    TimelineEvent,
    TimelineResponse,
    EmbedHealthResponse,
    HostHealthResponse,
    ImageBatchEmbedRequest,
    ImageBatchEmbedResponse,
    ImageEmbedRequest,
    ImageEmbedResponse,
    IndexRequest,
    IndexResponse,
    MemoryEmbedRequest,
    MemoryEmbedResponse,
    MemoryReadRequest,
    MemoryReadResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    VoiceRecallRequest,
    VoiceRecallResponse,
    VoiceAffectRequest,
    VoiceAffectResponse,
    MemoryRecentRequest,
    MemoryRecentResponse,
    RerankRequest,
    RerankResponse,
    RerankResult,
    MigrateRequest,
    MigrateResponse,
    MultimodalSearchRequest,
    MultimodalSearchResponse,
    TurnSyncRequest,
    TurnSyncResponse,
    CommitmentCreateRequest,
    CommitmentListResponse,
    CommitmentResponse,
    CommitmentUpdateRequest,
    ConcernResolveRequest,
    InitiativeCreateRequest,
    InitiativeDetailResponse,
    InitiativeOrigin,
    InitiativeFeedbackRequest,
    InitiativeEngagementRequest,
    InitiativeResponse,
    InitiativeListResponse,
    InitiativeCompleteRequest,
    InitiativeFailRequest,
    InitiativePriorityRequest,
    # Agent Snapshot
    AgentSnapshotInitiative,
    AgentSnapshotResponse,
    RecordOutreachRequest,
    RecordOutreachResponse,
)
from colonymind.util.temporal import now_utc

logger = logging.getLogger(__name__)

# Background task bookkeeping — prevents garbage-collection of fire-and-forget
# asyncio tasks (see https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task)
_background_tasks: set[asyncio.Task] = set()


def _spawn_task(coro) -> asyncio.Task:
    """Create an asyncio task, retain a reference, and auto-discard on completion."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


router = APIRouter(prefix="/v1/host", tags=["host"])
from colonymind.api.routers.activity import register as register_activity_routes
register_activity_routes(router)
v2_router = APIRouter(prefix="/v2/host", tags=["host-v2"])

# ---------------------------------------------------------------------------
# Module-level wiring — subsystems are injected by the server lifespan
# ---------------------------------------------------------------------------

_embedder = None
_event_subscribers: list[EventSubscriberBuffer] = []
_event_broadcast_lock = threading.RLock()


def _event_subscriber_queue_size() -> int:
    """Configured bound for each live event subscriber."""
    raw = os.environ.get("COLONYMIND_EVENT_SUBSCRIBER_QUEUE_SIZE", "256")
    try:
        size = int(raw)
    except ValueError:
        logger.warning(
            "Invalid COLONYMIND_EVENT_SUBSCRIBER_QUEUE_SIZE=%r; using 256", raw
        )
        return 256
    if size < 1:
        logger.warning(
            "COLONYMIND_EVENT_SUBSCRIBER_QUEUE_SIZE must be positive; using 256"
        )
        return 256
    return min(size, 10_000)


def broadcast_event(event: dict) -> Optional[dict]:
    """Persist, canonicalize, then publish an event to live subscribers.

    Called by the autonomy loop and other subsystems
    when state changes that the host should know about (proactive
    messages, briefings, anomalies, etc.).  A journal failure suppresses the
    live frame: clients must never observe an event which cannot be replayed.
    """
    from colonymind.events.journal import append_event_record

    event_type = str(event.get("type", "unknown"))
    payload = event.get("payload") or {}
    occurred_at = str(event.get("occurred_at") or "") or None

    # This lock preserves journal-sequence order in local live buffers when
    # emitters run concurrently on request handlers or worker threads.
    with _event_broadcast_lock:
        record = append_event_record(
            event_type,
            payload,
            occurred_at=occurred_at,
            event_key=event.get("event_key"),
        )
        if record is None:
            logger.error(
                "Suppressing live event %s because journal append failed",
                event_type,
            )
            return None

        if record.get("retained") is False:
            # A replayed publication intent may outlive journal retention.
            # Its durable receipt acknowledges delivery without resurrecting
            # pruned content or offering an unreplayable live frame.
            return record

        frame = {
            **event,
            "type": event_type,
            "payload": record["data"],
            "occurred_at": record["occurredAt"],
            "recordedAt": record["recordedAt"],
            "seq": record["seq"],
            "eventId": record["ulid"],
        }
        for subscriber in tuple(_event_subscribers):
            try:
                subscriber.publish(dict(frame))
            except Exception:
                logger.exception(
                    "Failed to enqueue event seq=%s for one subscriber",
                    record["seq"],
                )
        return frame


#: Why semantic recall is off although it was configured: set by the server when the
#: embedder or the vector store failed to initialise, cleared when they come up.
_embed_failure: Optional[str] = None


def set_embed_failure(reason: Optional[str]) -> None:
    global _embed_failure
    _embed_failure = str(reason).strip() if reason else None


def set_embedder(embedder) -> None:
    global _embedder
    _embedder = embedder


def _mind():
    """The running mind, or None (wired by the server lifespan)."""
    from colonymind.api.routers.mind import get_mind
    return get_mind()


def _mind_section() -> str:
    """The owner's Mind section: broadcast concerns, open goals and open asks, or nothing."""
    mind = _mind()
    if mind is None:
        return ""
    try:
        return str(mind.section() or "")
    except Exception:
        logger.debug("mind section unavailable", exc_info=True)
        return ""


async def _status_section(body: ContextAssembleRequest, *, recall: Optional[str] = None) -> str:
    """The owner's status block (``colonymind.mind.status``), or nothing: switched off, no mind, a kanban
    worker's run (its task body is not the owner's turn), a scheduled run (``metadata.run``: a job's prompt
    is not the owner's turn) or a recipient packet (session ``mind:<contact>``). Rendered from the running
    mind on every call, never stored; never breaks the turn, and its store reads cost the turn at most the
    block's deadline (``status.collect``). ``recall`` is this turn's semantic recall (``_status_problems``)."""
    from colonymind.mind import status as status_block
    context = body.context
    session_id = str(getattr(context, "session_id", "") or "")
    metadata = getattr(context, "metadata", None) or {}
    if (not status_block.enabled() or session_id.startswith("mind:") or metadata.get("kanban_task")
            or metadata.get("run")):
        return ""
    mind = _mind()
    if mind is None:
        return ""
    try:
        return await status_block.collect(mind, problems=_status_problems(mind, recall=recall))
    except Exception:
        logger.debug("status block unavailable", exc_info=True)
        return ""


def _status_problems(mind, *, recall: Optional[str] = None) -> dict:
    """The health route's in-process problems as the status block's probes, and the mind's own body
    staleness (the tick beats before it skips a stale body's work, so a live tick proves nothing about
    the body): no network, no model call.

    Semantic recall is down when it never started (``_embed_failure``), when it is configured (an embedder
    is wired) with no store, pipeline or catalog behind it (the health route's "vector store is unavailable";
    the turn's collection then says ``"unavailable"``), or when this turn's own retrieval fell back to
    keywords (``recall`` is ``"failed"``). Recall switched off by configuration (no embedder) is not a
    problem. ``recall`` is None when the turn ran no recall and ``"unknown"`` when its recall did not
    finish, which shows the field as unknown."""
    def semantic_down() -> bool:
        if _embed_failure or recall == "failed":
            return True
        if _embedder is not None:
            from colonymind.vector import get_pipeline, get_store
            store = get_store()
            if store is None or get_pipeline() is None or getattr(store, "catalog", None) is None:
                return True
        if recall == "unknown":
            raise RuntimeError("this turn's recall did not finish")
        return False

    def tick_stalled() -> bool:
        last = mind.last_tick_at or mind.started_at
        return (mind.clock() - last).total_seconds() / 3600.0 > _tick_stale_hours(mind)

    def capture_behind() -> bool:
        probe = getattr(getattr(mind, "capture", None), "oldest_unfinished_seconds", None)
        age = probe() if probe is not None else None
        return age is not None and float(age) / 3600.0 > _capture_stale_hours()

    return {"semantic_recall": semantic_down, "tick": tick_stalled,
            "body": lambda: bool(mind.body_stale()), "capture": capture_behind}


def _mind_open_questions(query_text: str) -> str:
    """The questions the mind asked the owner that are still open and about this turn's subject (a
    contradiction it asked about), or nothing. Never breaks the turn."""
    mind = _mind()
    if mind is None or not query_text:
        return ""
    try:
        from colonymind.mind.questions import open_questions
        return "\n".join(f"- {line}" for line in open_questions(mind, query_text))
    except Exception:
        logger.debug("open questions unavailable", exc_info=True)
        return ""


def _mind_note_novel(query_text: str) -> None:
    """An owner turn memory recalled nothing for: a new topic, a little curiosity for the agent's own
    affect at the next tick (architecture 4.3). A no-op without a mind; never breaks the turn."""
    feelings = getattr(_mind(), "feelings", None)
    if feelings is None:
        return
    try:
        feelings.note_novel_topic(query_text)
    except Exception:
        logger.debug("novel topic not noted", exc_info=True)


def _mind_stances(query_text: str, *, viewer_contact_id: str, viewer_is_owner: bool, session_id: str) -> str:
    """Recorded views for this turn (architecture 4.4), audience-filtered for the viewer; nothing when the
    mind is off. Who the viewer is comes from ``_assemble_sections`` (the viewer identity every other
    owner-only section uses), so a recipient packet gets only the views meant for that recipient."""
    mind = _mind()
    opinions = getattr(mind, "opinions", None) if mind is not None else None
    if opinions is None or not mind.enabled or not viewer_contact_id:
        return ""
    try:
        return str(opinions.context(query_text, viewer_contact_id=viewer_contact_id,
                                    viewer_is_owner=bool(viewer_is_owner),
                                    session_id=session_id or "") or "")
    except Exception:
        logger.debug("stance section unavailable", exc_info=True)
        return ""


async def _mind_lessons(query_text: str, *, viewer_is_owner: bool, owner_turn: bool, session_id: str,
                        task_run: bool = False) -> str:
    """The one lesson most related by meaning to the owner's own turn (architecture 4.8, ``lessons.turn_lesson``),
    or nothing: never for a guest, a recipient packet (session ``mind:<contact>``), a kanban worker's run
    (``task_run``: its task body carries its own lessons), with the mind off or with ``faculties.lessons`` off. The
    turn's query vector is the recall lane's when it has one for the same text, else one embed call within the
    turn's budget. Rendering it logs the lesson's use for this owner message, which the owner's next message later
    scores; the mind switched off while the turn's words were embedded gets none."""
    mind = _mind()
    lessons = getattr(mind, "lessons", None) if mind is not None else None
    if (lessons is None or not mind.enabled or not lessons.enabled or not viewer_is_owner or not owner_turn
            or task_run or not session_id or str(session_id).startswith("mind:")):
        return ""
    try:
        text, _ = await lessons.turn_lesson(query_text, session_id=str(session_id),
                                            still=lambda: bool(mind.enabled and lessons.enabled))
        return str(text or "")
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("lesson section unavailable", exc_info=True)
        return ""


async def _mind_standing(query_text: str, *, viewer_is_owner: bool, owner_turn: bool, session_id: str,
                         turn_id: str | None = None, task_run: bool = False) -> tuple[str, bool]:
    """Pinned instructions on eligible owner turns; the serving closure owns their lifecycle."""
    try:
        mind = _mind()
        lessons = getattr(mind, "lessons", None) if mind is not None else None
        runtime = getattr(mind, "plasticity", None) if mind is not None else None
        if (lessons is None or runtime is None or not runtime.enabled or not mind.enabled or not lessons.enabled
                or not viewer_is_owner or not owner_turn or task_run or not session_id
                or session_id == "mind" or str(session_id).startswith("mind:")
                or not mind.faculties.get("plasticity", True) or not mind.faculties.get("lessons", True)):
            return "", False
        from colonymind.memory import producers, turn_budget
        from colonymind.mind.lessons import served_key

        def render():
            parts, last, uses = [], False, []
            for lesson, pin in lessons.standing():
                text = lesson.standing_section(pin.level, pin.example)
                if text:
                    parts.append(text)
                    last = last or pin.level >= 3
                    uses.append((lesson.id, pin))
            return "\n\n".join(parts), last, uses

        text, last, uses = await producers.read(render, default=("", False, []))
        if not mind.enabled or not lessons.enabled or not runtime.enabled:
            return "", False
        for ident, pin in uses:
            turn_budget.defer_write(lessons.record_use, session_id=session_id, lesson_ids=[ident],
                served=served_key(query_text), turn_id=turn_id,
                standing={"learner": pin.learner, "level": pin.level, "closure": pin.closure})
        return text, last
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("standing lesson section unavailable", exc_info=True)
        return "", False


def _learning_served(ledger: Any, query_text: str, body: Any, selected: Any) -> None:
    """The learning rules' use log: which claims an owner turn was served, keyed like a lesson use by the
    owner's message, so the night can reinforce the ones the reply used and the owner judged right. Nothing
    with the faculty off, for a recipient packet or a kanban worker's run."""
    # This callback runs after the final host allowance has selected what the owner sees.
    runtime = getattr(_mind(), 'plasticity', None)
    name = runtime.find('query_adapter') if runtime is not None and runtime.enabled else None
    turn_id = getattr(body.context, 'turn_id', None)
    if name is not None and turn_id:
        runtime.loaded[name].driver.record_selection(turn_id, selected)
    from colonymind.memory import learning
    session_id = str(body.context.session_id or "")
    if (not learning.enabled() or not session_id or session_id.startswith("mind:")
            or (body.context.metadata or {}).get("kanban_task")):
        return
    try:
        from contextlib import closing
        from colonymind.mind.lessons import served_key
        ids = learning.claim_ids_of(selected)
        if ids:
            import time
            # The turn it served: the host's turn id, else this serving's time (a question asked again is
            # another use; the night scores two records of one turn once).
            occurrence = str(getattr(body.context, "turn_id", None) or "") or f"at:{time.time_ns()}"
            with closing(ledger._connect()) as conn, conn:
                learning.record_served(conn, session_id=session_id, served=served_key(query_text), store="claim",
                                       item_ids=ids, occurrence=occurrence)
    except Exception as error:
        from colonymind.health_counters import bump
        bump("writes_dropped.learning_use_log")
        logger.warning("learning use log unavailable (%s, sqlite_errorname=%s)",
                       type(error).__name__, getattr(error, "sqlite_errorname", None))


def _mind_recall_query(query_text: str) -> str:
    """The recall query plus the broadcast concerns (the workspace expands recall; broadcast flag)."""
    mind = _mind()
    if mind is None or not query_text:
        return query_text
    try:
        extra = " ".join(concern.summary for concern in mind.broadcast())[:400]
    except Exception:
        logger.debug("mind broadcast unavailable", exc_info=True)
        return query_text
    return f"{query_text} {extra}".strip() if extra else query_text


def _mind_posture() -> tuple[str, bool]:
    """(autonomy level, running) for the status routes that used to read the loop."""
    mind = _mind()
    if mind is None:
        return "unknown", False
    return str(mind.level), bool(mind.enabled)


_llm_router = None


def set_llm_router(router) -> None:
    global _llm_router
    _llm_router = router


_telemetry = None


def set_telemetry(telemetry) -> None:
    global _telemetry
    _telemetry = telemetry


_session_archive_store = None
_archive_engine = None


def set_session_archive_store(store) -> None:
    """Wire an ArchiveStore beside the other host-private stores at startup."""
    global _session_archive_store
    _session_archive_store = store


def set_session_archive_engine(engine) -> None:
    """Inject the deployment's engine; an unwired host has no archive capability."""
    global _archive_engine
    _archive_engine = engine


def _session_archive_engine():
    return _archive_engine


def _session_archive_available():
    engine = _session_archive_engine()
    if _session_archive_store is None or engine is None:
        return False
    try:
        return not hasattr(engine, 'available') or engine.available()
    except Exception:
        return False


def supported_capabilities() -> List[str]:
    """Return the list of capabilities this sidecar advertises."""
    caps: list[str] = ["memory"]
    if _embedder is not None:
        caps.append("embed")
    if _contacts_store is not None:
        caps.append("contacts")
    if _secrets_vault is not None:
        caps.append("secrets")
    if _mind() is not None:
        caps.append("mind")
        if getattr(_mind(), "store", None) is not None:
            caps.append("approval_history")
    caps.append("events")
    if _commitment_store is not None:
        caps.append("commitments")
        readiness = getattr(
            _commitment_store, "resolution_recovery_readiness", None,
        )
        if readiness is not None:
            try:
                status = readiness()
                capability = status.get("capability")
                if status.get("ready") is True and capability:
                    caps.append(str(capability))
            except Exception:
                pass
    if _affect_store is not None:
        caps.append("affect")
    if _reranker is not None:
        caps.append("rerank")
    caps.append("context")
    caps.append("event_journal")
    caps.append("agent_activity_v1")
    from colonymind.api.routers.curated_memory import is_available
    if is_available():
        caps.append("curated_memory_v1")
    if _session_archive_available():
        caps.append("session_archive")
    return caps


# Read-only owner asks. The existing /queue/approvals/* rule supplies
# approvals:read for scoped credentials; decisions keep their existing routes.
def _approval_history_owner(request: Request):
    from colonymind.api.auth import speaks_for_owner
    if not speaks_for_owner(request):
        raise HTTPException(403, detail={"code": "not_owner"}, headers={"Cache-Control": "no-store"})


async def _approval_history_read(request: Request, response: Response, **query):
    from colonymind.mind import approval_history
    _approval_history_owner(request)
    response.headers["Cache-Control"] = "no-store"
    mind = _mind()
    if mind is None or getattr(mind, "store", None) is None:
        raise HTTPException(503, detail={"code": "approval_history_unavailable"},
                            headers={"Cache-Control": "no-store"})
    try:
        value = await asyncio.to_thread(approval_history.read, mind.store, now=mind.clock(), **query)
        model = ApprovalHistoryDetail if "approval_id" in query else ApprovalHistoryPage
        return model.model_validate(value)
    except ValidationError:
        # Response validation errors otherwise carry stored field values into
        # diagnostics. Refuse malformed records without echoing any of them.
        raise HTTPException(503, detail={"code": "approval_history_unavailable"},
                            headers={"Cache-Control": "no-store"}) from None
    except HTTPException as exc:
        exc.headers = {**(exc.headers or {}), "Cache-Control": "no-store"}
        raise


@router.get("/queue/approvals/history", response_model=ApprovalHistoryPage,
            dependencies=[Depends(refuse_hand_session)])
async def approval_history_page(request: Request, response: Response,
                                limit: int = Query(50, ge=1, le=200),
                                state: Literal["all", "pending", "settled"] = "all",
                                session_id: str | None = Query(None, min_length=1, max_length=512),
                                cursor: str | None = Query(None, min_length=1, max_length=4096)):
    return await _approval_history_read(request, response, limit=limit, state=state,
                                        session_id=session_id, cursor=cursor)


@router.get("/queue/approvals/history/{approval_id}", response_model=ApprovalHistoryDetail,
            dependencies=[Depends(refuse_hand_session)])
async def approval_history_detail(approval_id: str, request: Request, response: Response):
    return await _approval_history_read(request, response, approval_id=approval_id)


def _origin_shape(origin) -> bool:
    """An explicit {platform, chat_id, thread_id?} conversation address, nothing else.

    The sidecar has no session inventory, so the caller supplies both origins; see
    the session-archive contract's trust boundary. Shape is checked here, membership is the
    gateway's, and the engine still proves every job's origin by readback.
    """
    # Blank and whitespace-only are refused, not just empty: " " is structurally
    # unusable as a conversation address, and accepting it wrote that exact origin
    # onto a live job. The caller's trust boundary covers WHICH chat, not whether
    # the value is an address at all.
    return (isinstance(origin, dict) and set(origin) <= {'platform', 'chat_id', 'thread_id'}
            and all(isinstance(origin.get(key), str) and origin[key].strip()
                    and len(origin[key]) <= 512 for key in ('platform', 'chat_id'))
            and (origin.get('thread_id') is None
                 or (isinstance(origin['thread_id'], str) and origin['thread_id'].strip()
                     and len(origin['thread_id']) <= 512)))


def _origin_identity(origin) -> tuple:
    return (origin['platform'].lower(), origin['chat_id'], origin.get('thread_id'))


def _session_archive_authority(request: Request, *, write=False):
    from colonymind.api.auth import speaks_for_owner
    from colonymind.session_archive.store import ArchiveError
    authority = request_authority(request)
    scope = 'sessions:write' if write else 'sessions:read'
    granted = ('manage' in authority.access if authority.scopes is None
               else bool({scope, '*'} & authority.scopes))
    if not granted or not speaks_for_owner(request):
        raise ArchiveError('unauthorized')


async def _session_archive_response(request, action, *, write=False):
    from colonymind.session_archive.store import ArchiveError, STATUSES
    import sqlite3
    try:
        _session_archive_authority(request, write=write)
        engine = _session_archive_engine()
        if not await asyncio.to_thread(_session_archive_available):
            raise ArchiveError('unsupported_capability')
        result = await action(_session_archive_store, engine)
        return JSONResponse(result, headers={'Cache-Control': 'no-store'})
    except ArchiveError as exc:
        return JSONResponse(exc.body, status_code=STATUSES[exc.code], headers={'Cache-Control': 'no-store'})
    except sqlite3.OperationalError as exc:
        code = 'busy' if getattr(exc, 'sqlite_errorcode', None) in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED} else 'source_unavailable'
        return JSONResponse({'error': code}, status_code=STATUSES[code], headers={'Cache-Control': 'no-store'})
    except Exception:
        logger.exception('session archive source unavailable')
        return JSONResponse({'error': 'source_unavailable'}, status_code=503, headers={'Cache-Control': 'no-store'})


def _archive_worker_authority(request):
    from colonymind.api.auth import speaks_for_owner, owner_person_id
    from colonymind.session_archive.store import ArchiveError
    authority = request_authority(request)
    granted = ('manage' in authority.access if authority.scopes is None
               else bool({'sessions:archive-worker', '*'} & authority.scopes))
    if not granted or not speaks_for_owner(request):
        raise ArchiveError('unauthorized')
    return (authority.credential_id or authority.principal_id), owner_person_id()


async def _archive_worker_response(action):
    from colonymind.session_archive.store import ArchiveError, STATUSES
    try:
        if _session_archive_store is None:
            raise ArchiveError('source_unavailable')
        return JSONResponse(await action(_session_archive_store), headers={'Cache-Control': 'no-store'})
    except ArchiveError as exc:
        return JSONResponse(exc.body, status_code=STATUSES[exc.code], headers={'Cache-Control': 'no-store'})
    except Exception:
        return JSONResponse({'error': 'source_unavailable'}, status_code=503, headers={'Cache-Control': 'no-store'})


@router.get('/sessions/archive/plans', dependencies=[Depends(refuse_hand_session)])
async def archive_plans(request: Request):
    async def poll(store):
        from colonymind.session_archive.store import ArchiveError
        worker, owner = _archive_worker_authority(request)
        raw = request.query_params.get('wait', '25')
        if not raw.isascii() or not raw.isdecimal() or len(raw) > 2 or int(raw) > 25:
            raise ArchiveError('invalid_request')
        deadline = asyncio.get_running_loop().time() + int(raw)
        await asyncio.to_thread(store.heartbeat)
        while True:
            if await request.is_disconnected():
                return {'plan': None}
            plan = await asyncio.to_thread(store.claim_plan, worker, owner)
            if plan is not None or asyncio.get_running_loop().time() >= deadline:
                return {'plan': plan}
            await asyncio.sleep(.5)
    return await _archive_worker_response(poll)


@router.post('/sessions/archive/plans/{operation_id:path}/receipt', dependencies=[Depends(refuse_hand_session)])
async def archive_plan_receipt(operation_id: str, request: Request):
    async def accept(store):
        from colonymind.session_archive.store import ArchiveError
        worker, _owner = _archive_worker_authority(request)
        try:
            body = await request.json()
        except (ValueError, UnicodeError):
            raise ArchiveError('invalid_request') from None
        if (not isinstance(body, dict) or set(body) != {'claim', 'receipt'}
                or not isinstance(body['claim'], str) or not isinstance(body['receipt'], dict)):
            raise ArchiveError('invalid_request')
        await asyncio.to_thread(store.accept_plan_result, operation_id, worker, body['claim'], body['receipt'])
        return {'accepted': True}
    return await _archive_worker_response(accept)


# Register the collection read before the parameterized session routes.
@router.get('/sessions/archived')
async def archived_sessions(request: Request):
    from colonymind.session_archive.store import ArchiveError
    async def read(store, engine):
        params = request.query_params
        raw = params.get('limit', '50')
        if not raw.isascii() or not raw.isdecimal() or len(raw) > 3 or not 1 <= int(raw) <= 200:
            raise ArchiveError('invalid_request')
        cursor = params.get('cursor')
        if cursor is not None and not 1 <= len(cursor) <= 4096:
            raise ArchiveError('invalid_request')
        return await asyncio.to_thread(store.inventory, int(raw), cursor)
    return await _session_archive_response(request, read)


@router.get('/sessions/{session_id}/archive')
async def session_archive_receipt(session_id: str, request: Request):
    async def read(store, engine):
        return await asyncio.to_thread(store.receipt, session_id, engine)
    return await _session_archive_response(request, read)


@router.post('/sessions/{session_id}/archive')
async def archive_session(session_id: str, request: Request):
    from colonymind.session_archive.store import ArchiveError
    async def write(store, engine):
        try:
            body = await request.json()
        except (ValueError, UnicodeError):
            raise ArchiveError('invalid_request') from None
        if (not isinstance(body, dict)
                or set(body) != {'archived', 'reparent_to_main', 'operation_id',
                                 'main_session_id', 'source_origin', 'main_origin'}
                or type(body['archived']) is not bool or type(body['reparent_to_main']) is not bool
                or not isinstance(body['operation_id'], str) or not 1 <= len(body['operation_id']) <= 128
                or not isinstance(body['main_session_id'], str)
                or not 1 <= len(body['main_session_id']) <= 512
                or body['main_session_id'] == session_id
                or not all(_origin_shape(body[key]) for key in ('source_origin', 'main_origin'))
                or _origin_identity(body['source_origin']) == _origin_identity(body['main_origin'])):
            raise ArchiveError('invalid_request')
        if hasattr(engine, 'archive'):
            from colonymind.api.auth import owner_person_id
            authority = request_authority(request)
            return await engine.archive(session_id, body, {
                'speaks_for_owner': True, 'sessions:write': True,
                'owner_id': owner_person_id(), 'principal_id': authority.principal_id})
        return await asyncio.to_thread(store.archive, session_id, body, engine)
    return await _session_archive_response(request, write, write=True)


# ---------------------------------------------------------------------------
# Host Configuration (LLM from host)
# ---------------------------------------------------------------------------


@router.post("/configure", response_model=HostConfigureResponse)
async def configure_host(body: HostConfigureRequest) -> HostConfigureResponse:
    """Receive LLM configuration from the host.

    The host (OpenClaw, Hermes, etc.) calls this on startup to provide
    its LLM provider credentials and model assignments. ColonyMind does not
    manage its own LLM keys — it inherits them from the host.

    Validate and persist one configuration, then atomically update the shared
    router. Existing consumer references and in-flight snapshots are retained.
    """
    if body.llm is None:
        return HostConfigureResponse(configured=False)

    from colonymind.router.router import LLMRouter
    import os
    import tempfile

    try:
        prepared = LLMRouter(tiers={})
        prepared.configure(body.llm)
        config_path = get_state_dir() / ".colonymind-llm-config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        fd, pending = tempfile.mkstemp(prefix=".llm-config-", dir=config_path.parent)
        try:
            with os.fdopen(fd, "w") as output:
                json.dump(body.llm, output, indent=2)
                output.flush()
                os.fsync(output.fileno())
            os.replace(pending, config_path)
        finally:
            if os.path.exists(pending):
                os.unlink(pending)
        # Keep every extractor/thinker/worker reference live. In-flight calls
        # retain their immutable snapshot; subsequent calls see this revision.
        if _llm_router is not None:
            _llm_router.adopt_configuration(prepared, config_path=config_path)
            new_router = _llm_router
        else:
            prepared.watch_config(config_path)
            new_router = prepared
            set_llm_router(new_router)
        return HostConfigureResponse(
            configured=True, provider=body.llm.get("provider"),
            models=body.llm.get("models", {}), routing=new_router.routing_status())
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Invalid host model configuration") from exc
    except Exception as exc:
        logger.error("configure_host failed: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Model configuration update failed") from exc


@router.get("/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """List available LLM models for the currently configured provider.

    For local providers (Ollama, vLLM, LM Studio, etc.), this queries the
    local server and returns the actual models that are installed.  For
    cloud providers an empty list is returned — the host is expected to
    know which cloud models exist.
    """
    # Load persisted host config to know the current provider/base_url
    from colonymind.router.tiers import discover_local_models

    config_path = get_state_dir() / ".colonymind-llm-config.json"
    provider = ""
    base_url = ""
    api_key = ""
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            provider = cfg.get("provider", "")
            base_url = cfg.get("baseUrl", "")
            api_key = cfg.get("apiKey", "")
        except Exception as exc:
            logger.debug("Could not read persisted LLM config: %s", exc)

    if _llm_router is not None and _llm_router.supports_function_routing:
        observed = await _llm_router.discover_models()
        if observed is not None:
            models = observed['models']
            return ModelListResponse(provider=observed['provider'], base_url=observed['base_url'],
                models=[ModelInfo(id=row['id'], provider=observed['provider'], owned_by=row.get('owned_by')) for row in models],
                discovered=bool(models), routing=observed['routing'],
                error=None if models else 'No fresh model advertisements are available; configured completion aliases remain unchanged.')

    if not provider:
        return ModelListResponse(
            routing=_llm_router.routing_status() if _llm_router is not None else None,
            provider="",
            error="No LLM provider configured. Call POST /v1/host/configure first.",
        )

    if provider not in ("ollama", "local", "custom", "lmstudio", "vllm"):
        return ModelListResponse(
            routing=_llm_router.routing_status() if _llm_router is not None else None,
            provider=provider,
            error="Model listing is only supported for local providers (ollama, local, custom, lmstudio, vllm).",
        )

    discovered = discover_local_models(provider, base_url, api_key)
    if discovered:
        return ModelListResponse(
            routing=_llm_router.routing_status() if _llm_router is not None else None,
            provider=provider,
            base_url=base_url or None,
            models=[
                ModelInfo(
                    id=m.get("name") or m.get("id", ""),
                    provider=provider,
                    size=m.get("size"),
                    owned_by=m.get("owned_by"),
                )
                for m in discovered
                if (m.get("name") or m.get("id"))
            ],
            discovered=True,
        )

    return ModelListResponse(
        routing=_llm_router.routing_status() if _llm_router is not None else None,
        provider=provider,
        base_url=base_url or None,
        error="Could not discover models from the local server. Is it running?",
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

_INDEX_HEALTH_TIMEOUT_SECONDS = 5.0
#: The mind's tick is stale after ten of its intervals, and never sooner than this.
TICK_STALE_FLOOR_HOURS = 0.25
#: A capture job still unfinished after this long means capture is not landing.
CAPTURE_STALE_HOURS = 1.0


def _tick_stale_hours(mind) -> float:
    """How long the mind's tick may be silent: ``COLONYMIND_STALE_TICK_HOURS`` when pinned,
    else ten of the mind's own intervals with a quarter-hour floor."""
    pinned = os.environ.get("COLONYMIND_STALE_TICK_HOURS", "").strip()
    if pinned:
        try:
            return float(pinned)
        except ValueError:
            pass
    try:
        interval = float(getattr(mind, "interval", 60.0) or 60.0)
    except (TypeError, ValueError):
        interval = 60.0
    return max(10 * interval / 3600.0, TICK_STALE_FLOOR_HOURS)


def _capture_stale_hours() -> float:
    pinned = os.environ.get("COLONYMIND_STALE_CAPTURE_HOURS", "").strip()
    if pinned:
        try:
            return float(pinned)
        except ValueError:
            pass
    return CAPTURE_STALE_HOURS


def _capture_backlog_hours(mind) -> Optional[float]:
    """How long the oldest unfinished capture job has waited, or None when nothing waits
    (or no capture queue is wired)."""
    probe = getattr(getattr(mind, "capture", None), "oldest_unfinished_seconds", None)
    if probe is None:
        return None
    try:
        age = probe()
    except Exception as exc:
        logger.warning("capture backlog probe failed: %s", type(exc).__name__)
        return None
    return None if age is None else float(age) / 3600.0


_build_note: Optional[str] = None   # what this process was built from, as found when health first looked
_build_note_lock = threading.Lock()


def _build_stamp_note() -> str:
    """The build stamp in the words of doctor's ``build-stamp`` row, without the branch name: ``health`` answers
    without a key. The stamp is read once per process; every later call returns what was found then."""
    global _build_note
    with _build_note_lock:
        if _build_note is None:
            note = "no readable build stamp: the commit this install was built from is not recorded"
            try:
                from colonymind import build_stamp
                stamp = build_stamp.read()
                if stamp is not None:
                    note = f"commit {stamp['commit'][:12]}, " + (
                        "changed: the tree had uncommitted changes" if stamp["dirty"] else "clean")
            except Exception as exc:
                logger.warning("build stamp not read: %s", type(exc).__name__)
            _build_note = note
        return _build_note


@router.get("/health", response_model=HostHealthResponse, tags=[PUBLIC])
async def health() -> HostHealthResponse:
    caps = await asyncio.to_thread(supported_capabilities)
    notes: dict[str, str] = {}
    problems: list[str] = []   # every reason the status is not "ok", in words
    embed_model = ""

    # the sidecar's own open-file limit (doctor reads this; a low limit makes
    # LanceDB vector recall fail under load — see check_server_fd_limit)
    try:
        import resource
        _fd_soft, _fd_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        notes["fd_limit"] = (
            "unlimited" if _fd_soft == resource.RLIM_INFINITY else str(_fd_soft))
    except Exception:
        pass

    memory_backend_down = False
    try:
        from colonymind.turns import get_turn_idempotency_ledger
        from contextlib import closing
        def probe_source_ledger():
            ledger = get_turn_idempotency_ledger(get_state_dir())
            with closing(ledger._connect()) as conn:
                conn.execute('SELECT 1 FROM turn_sources LIMIT 1').fetchone()
        await asyncio.to_thread(probe_source_ledger)
        notes['memory'] = 'Canonical source ledger readable; semantic projection is optional'
    except Exception as exc:
        memory_backend_down = True
        caps = [c for c in caps if c != 'memory']
        notes['memory'] = 'Canonical source ledger unavailable (' + type(exc).__name__ + ')'
        problems.append(f"the source ledger is unreadable ({type(exc).__name__}: {exc})")
    if _contacts_store is not None:
        notes["contacts"] = "ContactsStore wired"
    embed_degraded = False
    if _embed_failure:
        # Semantic recall was configured and is not running: say so in words, so the
        # keyword fallback is never mistaken for the configured recall.
        embed_degraded = True
        problems.append("semantic recall is off: " + _embed_failure)
        if _embedder is None:
            notes["embed"] = "semantic recall is off: " + _embed_failure
    if _embedder is not None:
        # Get embed model info
        if hasattr(_embedder, "_provider") and hasattr(_embedder._provider, "_config"):
            embed_model = _embedder._provider._config.model_id
        embed_note = f"EmbeddingPipeline wired (model={embed_model})"

        # Managed generation identity governs semantic reads/writes. Verify it
        # and a bounded physical read; per-row legacy metadata discovery is an
        # explicit audit, not a prerequisite for every readiness request.
        try:
            from colonymind.vector import get_store
            store = get_store()
            if store is None:
                raise RuntimeError("Managed vector store is unavailable")
            await asyncio.wait_for(
                store.check_index_health(_embedder.index_identity),
                timeout=_INDEX_HEALTH_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            embed_degraded = True
            embed_note += f" [index-check failed: {type(exc).__name__}: {exc}]"
            problems.append(f"the semantic index check failed: {type(exc).__name__}: {exc}")
            logger.warning("embedding index health probe failed: %s", type(exc).__name__)

        # Check embedder health
        try:
            hc = await _embedder.health_check()
            if hc.get("status") != "ok":
                embed_degraded = True
                embed_note += f" [health: {hc.get('status', 'unknown')}"
                if hc.get("error"):
                    embed_note += f": {hc['error']}"
                embed_note += "]"
                problems.append(f"the embedder is not answering correctly: {hc.get('error') or hc.get('status')}")
        except Exception as exc:
            embed_degraded = True
            embed_note += f" [health probe failed: {exc}]"
            problems.append(f"the embedder health probe failed: {exc}")
            logger.warning("embedder health probe failed: %s", exc)

        notes["embed"] = embed_note
    if _secrets_vault is not None:
        notes["secrets"] = "sealed store wired" + ("" if _secrets_vault.available else " (key unavailable: values withheld)")
    mind = _mind()
    if mind is not None:
        notes["mind"] = f"mind {'on' if mind.enabled else 'off'} (autonomy {mind.level}, ticks={mind.ticks})"
    if _commitment_store is not None:
        if "commitment_resolution_recovery_v1" in caps:
            notes["commitments"] = (
                "CommitmentStore wired (resolution recovery v1 ready)"
            )
        elif hasattr(_commitment_store, "resolution_recovery_readiness"):
            notes["commitments"] = (
                "CommitmentStore wired (resolution recovery unavailable)"
            )
        else:
            notes["commitments"] = "CommitmentStore wired"
    if _affect_store is not None:
        notes["affect"] = "AffectStore wired"

    health_status = "ok"
    if embed_degraded or memory_backend_down:
        health_status = "degraded"
    if (
        _commitment_store is not None
        and hasattr(_commitment_store, "resolution_recovery_readiness")
        and "commitment_resolution_recovery_v1" not in caps
    ):
        health_status = "degraded"
        problems.append("commitment resolution recovery is unavailable")

    # Temporal metrics. What the sidecar runs on its own is tracked for staleness: the
    # mind's tick (it beats whether the mind is on or off) and the capture queue (jobs
    # that sit for hours are not landing). What inbound traffic drives (sync =
    # turns/sync, prefetch = context/assemble) is reported as silence and never flags:
    # a quiet day is not a failure, and a fresh install must be able to be ready.
    temporal = None
    try:
        if _telemetry is not None:
            temporal_data = await _telemetry.to_dict({"tick": _tick_stale_hours(mind)})
            flags = list(temporal_data.get("stale_flags") or [])
            silence = dict(temporal_data.get("silence_hours") or {})
            for flag in flags:
                if flag == "tick:never_ran":
                    problems.append("the mind's tick has not run since the sidecar started")
                elif flag == "tick":
                    problems.append(f"the mind's tick has not run for {silence.get('tick') or 0:.1f} h")
            backlog = await asyncio.to_thread(_capture_backlog_hours, mind)
            if backlog is not None:
                silence["capture"] = backlog
                if backlog > _capture_stale_hours():
                    flags.append("capture")
                    problems.append("capture jobs are not landing: the oldest unfinished job has waited "
                                    f"{backlog:.1f} h")
            temporal_data["stale_flags"], temporal_data["silence_hours"] = flags, silence
            if flags:
                health_status = "degraded"
            from colonymind.api.schemas.host import TemporalMetrics
            temporal = TemporalMetrics(**temporal_data)
    except Exception as exc:
        # If staleness cannot even be computed, the one probe that catches a
        # dead loop is gone — that is degradation, not silent health.
        health_status = "degraded"
        notes["temporal"] = f"staleness computation failed: {exc}"
        problems.append(f"staleness computation failed: {exc}")
        logger.warning("temporal staleness computation failed: %s", exc)

    from colonymind.memory import recall_stats
    recall = recall_stats.STATS.snapshot()
    from colonymind.memory import curated_health
    # Only counts, a temporal stamp and an exception class; never query or memory text.
    curated_status = curated_health.for_path(Path(get_state_dir()) / 'turn-idempotency.db').snapshot()
    recall['curated'] = curated_status
    if curated_status['degraded']:
        health_status = 'degraded'
        problems.append('Curated memory protection has failed open (' + (curated_status['error_class'] or 'unavailable')
                        + '); forgotten items can resurface in plain recall')
    # Person digests withheld because a claim they cited was erased or retracted, waiting for a rebuild.
    counter = getattr(_contacts_store, "count_withheld_digests", None)
    if callable(counter):
        try:
            recall["digests_withheld"] = await counter()
        except Exception as exc:
            logger.warning("withheld digests not counted: %s", exc)
    # Instinct: each decision provider's breaker and each point's states and counts since the start (no text).
    instinct_health = None
    try:
        from colonymind import instinct as instinct_core
        instinct_health = await asyncio.to_thread(instinct_core.shared().health)
    except Exception as exc:
        logger.warning("instinct health unavailable: %s", type(exc).__name__)
    # The last 24 hours' model calls, in counts only (``colonymind.usage``): the instance's running cost. A ledger
    # that cannot be read says so, in words ``colonymind status`` shows as a warning, and never reads as none.
    from colonymind import usage as model_usage
    try:
        notes["model_usage"] = await asyncio.to_thread(model_usage.health_line, get_state_dir())
    except model_usage.Unreadable as exc:
        notes["model_usage"] = f"reading the model usage ledger failed ({exc.reason})"
    except Exception as exc:
        notes["model_usage"] = f"reading the model usage ledger failed ({type(exc).__name__})"
    # The build this process runs, as its stamp says: looked up once, by the first answer, off the event loop.
    notes["build"] = _build_note or await asyncio.to_thread(_build_stamp_note)
    return HostHealthResponse(
        status=health_status,
        capabilities=caps,
        notes=notes,
        temporal=temporal,
        problems=problems,
        recall=recall,
        instinct=instinct_health,
    )


def _require_person_authority(request: Request | None, person_id: str) -> None:
    """Memory reads need an authenticated key (never development mode) and a resolved person."""
    authority = request_authority(request)
    if (not authority.authenticated or authority.anonymous or not authority.principal_id
            or not str(person_id or "").strip()):
        raise HTTPException(status_code=403, detail={
            "code": "person_authority_required",
            "message": "memory reads require the API key and a resolved person",
        })


def _viewer_is_guest(request: Request | None, person_id: Optional[str]) -> bool:
    """A viewer other than the owner, which fails closed.

    A guest's context is contact-scoped: their own canonical sources, the
    commitments their sources prove shared, and their digest; never the
    owner's global or person-store context. Development mode (no key) is
    never the owner (``resolve_request_person`` refuses the owner's lane
    there), so it is a guest for anyone it names and for no one. No person is
    the unscoped owner view only for a credential granted the owner
    (docs/CREDENTIALS.md); for any other it is a guest's.
    """
    from colonymind.api.auth import has_owner_authority
    authority = request_authority(request)
    person = str(person_id or "").strip()
    if authority.anonymous or not authority.authenticated:
        return True
    if not person:
        return not has_owner_authority(authority)
    return person != owner_person_id()


def _commitment_due(row) -> str:
    """A commitment line's due part: the converted time, and beside it the words the person used for it
    (capture's ``metadata.due_text``), so the context never offers only the conversion."""
    if not row.get("due_at"):
        return ""
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    said = " ".join(str(metadata.get("due_text") or "").split()).replace('"', "'")[:120]
    return f" (due: {row['due_at']}" + (f', said as "{said}"' if said else "") + ")"


def _canonical_shared_commitments(rows, contact_id):
    """Expose only descriptions already present in this person's source evidence.

    person_id is a subject selector, not an audience grant. A metadata link
    alone is insufficient: the current person-scoped source must contain the
    displayed description. Unclassified legacy tasks stay owner-private.
    """
    from contextlib import closing
    from colonymind.turns import get_turn_idempotency_ledger
    if not contact_id or not (Path(get_state_dir()) / "turn-idempotency.db").exists():
        return []
    visible = []
    ledger = get_turn_idempotency_ledger(get_state_dir())
    with closing(ledger._connect()) as conn:
        for row in rows:
            metadata = row.get("metadata")
            source_id = metadata.get("source_turn_id") if isinstance(metadata, dict) else None
            description = str(row.get("description") or "").strip()
            if row.get("person_id") != contact_id or not isinstance(source_id, str) or not description:
                continue
            source = conn.execute(
                "SELECT messages_json FROM turn_sources WHERE turn_id=? AND contact_id=? AND scope='person'",
                (source_id, contact_id)).fetchone()
            if source is None:
                continue
            for message in json.loads(source["messages_json"]):
                content = message.get("content")
                texts = ([content] if isinstance(content, str) else
                         [block["text"] for block in content if isinstance(block, dict)
                          and block.get("type") in {"text", "input_text", "output_text"}
                          and isinstance(block.get("text"), str)] if isinstance(content, list) else [])
                if any(description.casefold() in text.casefold() for text in texts):
                    visible.append(row)
                    break
    return visible[:5]


@router.get("/health/llm")
async def llm_health() -> dict:
    """Live-fire the LLM router with one tiny SMALL-tier completion (v0.19.0).

    The cheapest faithful proxy for "can the cognition stack call a
    model at all" — it exercises the exact router path that dies with
    "all tiers exhausted" when the persisted baseUrl/apiKey are wrong.
    Defensive: never raises, always returns {ok, tier, latency_ms, error}.
    """
    if _llm_router is None:
        return {"ok": False, "tier": None, "latency_ms": 0,
                "error": "LLM router not wired"}
    try:
        from colonymind.router.tiers import ModelTier
        resp = await _llm_router.complete(
            [{"role": "user", "content": "Say OK"}],
            force_tier=ModelTier.SMALL,
        )
        tier = resp.tier_used.value if getattr(resp, "tier_used", None) else None
        return {"ok": True, "tier": tier,
                "latency_ms": getattr(resp, "latency_ms", 0), "error": None}
    except Exception as exc:  # noqa: BLE001 — diagnostics must not 500
        return {"ok": False, "tier": None, "latency_ms": 0, "error": str(exc)}


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

_NOT_WIRED = {"error": {"code": "not_wired", "message": "Backend not configured"}}


@router.post("/memory/read", response_model=MemoryReadResponse)
async def memory_read(body: MemoryReadRequest, request: Request = None) -> MemoryReadResponse:
    """Open one exact canonical revision in the authenticated participant scope."""
    person_id = resolve_request_person(request, claimed_person_id=body.person_id)
    _require_person_authority(request, person_id)
    from colonymind.turns import get_turn_idempotency_ledger
    from colonymind.turns.source_read import read, read_video
    try:
        ledger = get_turn_idempotency_ledger(get_state_dir())
        if body.source_view == 'video':
            return MemoryReadResponse(source=await read_video(ledger,
                contact_id=person_id, session_id=body.session_id, source_id=body.source_id,
                source_version=body.source_version, asset_hash=body.asset_hash,
                requested_ms=body.requested_ms, read_revision=body.read_revision))
        return MemoryReadResponse(source=read(ledger,
            contact_id=person_id, session_id=body.session_id, source_id=body.source_id,
            source_version=body.source_version, view=body.source_view, claim_id=body.claim_id,
            offset=body.offset, read_revision=body.read_revision, asset_hash=body.asset_hash, page=body.page))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except Exception as exc:
        logger.warning('Canonical source read failed (%s)', type(exc).__name__)
        raise HTTPException(status_code=503, detail={
            'code': 'memory_backend_unavailable', 'message': 'Canonical source could not be read'}) from None


def _manages(request: Request | None) -> bool:
    """Whether the request's credential reaches the owner's side of the API (``manage``); never development mode,
    never a credential carried from 1.9.0 (its scopes name no trace)."""
    authority = request_authority(request)
    return (authority.authenticated and not authority.anonymous and authority.scopes is None
            and "manage" in authority.access)


def _require_trace_authority(request: Request | None, person: Optional[str] = None) -> None:
    """A recall trace is the owner's diagnostic: a credential that manages the instance may ask for any person's it
    may search; any other only for its own person's (its viewer, or its own identity). ``person`` None: assembly,
    which traces for a managing credential only."""
    if _manages(request):
        return
    authority = request_authority(request)
    own = {value for value in (authority.viewer_person_id, authority.identity) if value}
    if person is not None and not authority.anonymous and authority.authenticated and person in own:
        return
    raise HTTPException(status_code=403, detail={
        "code": "trace_not_granted",
        "message": "a recall trace is for a credential that manages the instance, or for the credential's own "
                   "person in a memory search",
    })


@router.post("/memory/search", response_model=MemorySearchResponse, tags=[PUBLIC])
async def memory_search(body: MemorySearchRequest, request: Request) -> MemorySearchResponse:
    """Search current canonical evidence for an authenticated participant.

    The body names the person (the schema refuses a missing or blank one);
    there is no owner default. Development mode never resolves to the owner
    (``resolve_request_person``) and never passes the authority check. A guest
    searches their own canonical sources only. ``trace`` adds why each
    candidate was kept or left out (``_require_trace_authority``). ``window``
    and ``as_of`` name the search's own time (``MemorySearchRequest``); the
    question's time is read once, here, and handed to the selection. A search
    from a turn in a group chat (``chat_type``, ``chat``) reads that chat's own
    history only, as the turn's recall does (``turns.group_chats``).
    """
    person = resolve_request_person(request, claimed_person_id=body.person_id)
    _require_person_authority(request, person)
    if body.trace:
        _require_trace_authority(request, person)
    canonical_only = _viewer_is_guest(request, person)
    try:
        from colonymind.beliefs.source_time import interpret_time_query
        from colonymind.turns import get_turn_idempotency_ledger
        from colonymind.vector import get_store, get_pipeline
        from colonymind.memory.search import collect_sources, select_memory
        from colonymind.memory.trace import NULL_TRACE, RecallTrace
        from colonymind.util.temporal import now_utc, resolve_communication_timezone
        trace = RecallTrace() if body.trace else NULL_TRACE
        ledger = get_turn_idempotency_ledger(get_state_dir())
        contact_tz = None
        if not canonical_only and _contacts_store is not None:
            try:
                contact = await _contacts_store.get(person)
                contact_tz = getattr(contact, "timezone", None)
            except Exception:
                logger.debug("contact timezone unavailable for memory search", exc_info=True)
        timezone_name = resolve_communication_timezone(contact_tz, body.timezone or ("UTC" if canonical_only else None))
        # The question's time, read once: the search's own window and reference time, else its words.
        now = body.as_of or now_utc()
        time_query = interpret_time_query(body.query, now=now, timezone_name=timezone_name,
                                          window=(body.window.start, body.window.end) if body.window else None)
        from colonymind.memory import curated
        from colonymind.turns import group_chats
        group = group_chats.of_turn({"chat_type": body.chat_type, "chat": body.chat})
        with curated.strict_read(ledger):
            collected = await collect_sources(ledger, query=body.query, contact_id=person,
                session_id=body.session_id or "", vector_store=get_store(), embedding_pipeline=get_pipeline(),
                trace=trace, group=group)
            packet = await select_memory(collected, query=body.query, selector=_memory_context_selector(),
                timezone_name=timezone_name, limit=body.limit, trace=trace, now=now, time_query=time_query)
        return MemorySearchResponse(**packet.public(), **({"trace": trace.report()} if trace else {}))
    except Exception as exc:
        logger.warning("canonical memory search failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail={
            "code": "memory_backend_unavailable",
            "message": "Canonical memory could not be read or selected",
        }) from None


@router.post('/memory/voice-affect', response_model=VoiceAffectResponse)
async def memory_voice_affect(body: VoiceAffectRequest, request: Request) -> VoiceAffectResponse:
    """One optional voice-tail line for the credential's own speaker only."""
    from colonymind.api.auth import owner_platform_granted
    from colonymind.identity import get_owner_contact_id, get_owner_name, get_persona_name
    from colonymind.memory.voice_affect import (
        APPRAISAL_WORDS, DEADLINE_SECONDS, STALE_SECONDS, render_line, speaker_words,
    )
    started = time.perf_counter()
    empty = VoiceAffectResponse()
    authority = request_authority(request)
    if not authority.authenticated or authority.anonymous:
        return empty
    try:
        person = resolve_request_person(request, claimed_person_id=body.person_id)
    except HTTPException:
        return empty
    owner = get_owner_contact_id()
    # No broad recall grant, managing credential or caller-chosen person can
    # inspect another speaker's feelings. Guest requests never reach appraisals.
    if not owner or owner != owner_person_id():
        return empty
    is_owner = person == owner
    if is_owner:
        if not owner_platform_granted(authority, 'voice'):
            return empty
    elif not (authority.contacts in {'own', 'list'} and person == authority.viewer_person_id):
        return empty
    from colonymind.turns import get_turn_idempotency_ledger
    ledger = get_turn_idempotency_ledger(get_state_dir())
    async def read():
        now = time.time()
        speaker, appraisal = '', ''
        speaker_name = get_owner_name(default='Speaker') if is_owner else 'Speaker'
        try:
            speaker = speaker_words(_affect_store, person, now=now)
            if speaker and not is_owner and _contacts_store is not None:
                contact = await _contacts_store.get(person)
                speaker_name = getattr(contact, 'given_name', None) or getattr(contact, 'display_name', None) or 'Speaker'
        except Exception:
            logger.debug('Voice speaker affect unavailable', exc_info=True)
        if is_owner and time.perf_counter() - started < DEADLINE_SECONDS:
            try:
                from colonymind.api.routers.social_state import appraisal_store
                dimension = appraisal_store().voice_appraisal(person, now=now, max_age=STALE_SECONDS)
                appraisal = APPRAISAL_WORDS.get(dimension, '')
            except Exception:
                logger.debug('Voice own appraisal unavailable', exc_info=True)
        if time.perf_counter() - started >= DEADLINE_SECONDS:
            return empty
        return VoiceAffectResponse(line=render_line(speaker, appraisal,
            speaker_name=speaker_name, persona_name=get_persona_name(default='Assistant')))
    from colonymind.memory import curated
    return await curated.strict_withdrawal_fenced_read(ledger, read, empty=empty)


@router.post('/memory/voice-recall', response_model=VoiceRecallResponse)
async def memory_voice_recall(body: VoiceRecallRequest, request: Request) -> VoiceRecallResponse:
    """Token-capped local recall; a newer utterance revision supersedes a partial."""
    started_at = time.monotonic()
    person = resolve_request_person(request, claimed_person_id=body.person_id)
    _require_person_authority(request, person)
    from colonymind.memory.voice_recall import VoiceRequests, VoiceSuperseded, VoiceBusy, read_voice_packet
    from colonymind.router.tokens import estimate_tokens
    from colonymind.turns import get_turn_idempotency_ledger
    from colonymind.util.temporal import resolve_communication_timezone
    authority = request_authority(request)
    coordinator = getattr(request.app.state, 'voice_requests', None)
    if coordinator is None:
        coordinator = request.app.state.voice_requests = VoiceRequests()
    key = (authority.principal_id, authority.credential_id, person, body.session_id or '', body.utterance_id)
    embedding = None

    def prepare():
        nonlocal embedding
        try:
            if body.speculative:
                from colonymind.vector import get_pipeline, get_store
                coordinator.embeddings.start(key, body.query, get_pipeline(), get_store())
            else:
                embedding = coordinator.embeddings.take(key, body.query)
        except Exception:
            # Speculation is optional; it cannot fail canonical keyword recall.
            coordinator.embeddings.entries.pop(key, None)

    # Register before any await (including contact lookup) so a final fences
    # a partial throughout its complete read, rather than only its SQLite phase.
    async def read(cancelled):
        canonical_only = _viewer_is_guest(request, person)
        contact_tz = None
        if not canonical_only and _contacts_store is not None:
            try:
                contact = await _contacts_store.get(person)
                contact_tz = getattr(contact, 'timezone', None)
            except Exception:
                logger.debug('contact timezone unavailable for voice recall', exc_info=True)
        timezone_name = resolve_communication_timezone(contact_tz, body.timezone or ('UTC' if canonical_only else None))
        if cancelled.is_set():
            return None
        def load():
            ledger = get_turn_idempotency_ledger(get_state_dir())
            return read_voice_packet(ledger, query=body.query, contact_id=person,
                session_id=body.session_id or '', k=body.k, max_tokens=body.max_tokens,
                timezone_name=timezone_name, cancelled=cancelled,
                started_at=started_at,
                **({'embedding': embedding} if embedding is not None else {}))
        return await asyncio.to_thread(load)

    try:
        packet = await coordinator.run(key, speculative=body.speculative, read=read, prepare=prepare)
        return VoiceRecallResponse(**packet.public(), token_count=estimate_tokens(packet.content),
            max_tokens=body.max_tokens, utterance_id=body.utterance_id,
            request_id=body.request_id, speculative=body.speculative)
    except VoiceSuperseded:
        raise HTTPException(status_code=409, detail={'code': 'voice_recall_superseded',
            'utterance_id': body.utterance_id, 'request_id': body.request_id}) from None
    except VoiceBusy:
        raise HTTPException(status_code=503, detail={'code': 'voice_recall_busy'}) from None
    except Exception as exc:
        logger.warning('Voice recall unavailable (%s)', type(exc).__name__)
        raise HTTPException(status_code=503, detail={'code': 'memory_backend_unavailable',
            'message': 'Canonical voice memory could not be read or selected'}) from None


@router.post('/memory/recent', response_model=MemoryRecentResponse)
async def memory_recent(body: MemoryRecentRequest, request: Request) -> MemoryRecentResponse:
    """Read the caller's latest recorded conversation without semantic ranking."""
    person = resolve_request_person(request, claimed_person_id=body.person_id)
    _require_person_authority(request, person)
    from colonymind.turns import get_turn_idempotency_ledger
    from colonymind.memory.recent import read_recent
    def load_recent():
        ledger = get_turn_idempotency_ledger(get_state_dir())
        return read_recent(ledger, contact_id=person,
            session_id=body.session_id, platform=body.platform, limit=body.limit,
            comms_log=_comms_log)
    try:
        return MemoryRecentResponse(**await asyncio.to_thread(load_recent))
    except Exception as exc:
        logger.warning('Recent canonical conversation unavailable (%s)', type(exc).__name__)
        raise HTTPException(status_code=503, detail={
            'code': 'memory_backend_unavailable',
            'message': 'Recent canonical conversation could not be read'}) from None


@router.post("/memory/embed", response_model=MemoryEmbedResponse)
async def memory_embed(body: MemoryEmbedRequest) -> MemoryEmbedResponse:
    if _embedder is None:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=_NOT_WIRED)
    try:
        # Support both old `inputs` field and new `texts` field
        texts = body.texts or body.inputs
        if not texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        if len(texts) > 128:
            raise HTTPException(status_code=400, detail=f"Batch size {len(texts)} exceeds limit of 128")
        vectors = await _embedder.embed_batch(texts)
        # Determine model_id from the underlying provider config
        model_id = ""
        if hasattr(_embedder, "_provider") and hasattr(_embedder._provider, "_config"):
            model_id = _embedder._provider._config.model_id
        return MemoryEmbedResponse(model=model_id or body.model or "unknown", vectors=vectors)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("memory_embed failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/memory/rerank", response_model=RerankResponse)
async def memory_rerank(body: RerankRequest) -> RerankResponse:
    """Rerank documents by relevance to a query.

    Requires the reranker to be initialized (see COLONYMIND_RERANKER_MODEL env var).
    """
    if _reranker is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Reranker not initialized. Set COLONYMIND_RERANKER_MODEL to enable.",
        )
    try:
        if not body.documents:
            raise HTTPException(status_code=400, detail="No documents provided")
        if len(body.documents) > 256:
            raise HTTPException(
                status_code=400,
                detail=f"Document count {len(body.documents)} exceeds limit of 256",
            )
        results = await _reranker.rerank(
            query=body.query,
            documents=body.documents,
            top_k=body.top_k or 10,
        )
        return RerankResponse(
            results=[
                RerankResult(index=r.index, score=r.score, text=r.text)
                for r in results
            ],
            model=getattr(_reranker, "_model_id", "unknown"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("memory_rerank failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/embed/health", response_model=EmbedHealthResponse)
async def embed_health() -> EmbedHealthResponse:
    """Check embedder health — verify model is loaded and producing valid output."""
    if _embedder is None:
        return EmbedHealthResponse(status="error", error=_embed_failure or "embedder not initialized")
    try:
        result = await _embedder.health_check()
        # Add multimodal status
        result["modalities"] = _embedder.modalities if hasattr(_embedder, "modalities") else ["text"]
        result["multimodal_enabled"] = _embedder.is_multimodal if hasattr(_embedder, "is_multimodal") else False
        return EmbedHealthResponse(**result)
    except Exception as exc:
        return EmbedHealthResponse(status="error", error=str(exc))


@router.post("/memory/embed/image", response_model=ImageEmbedResponse)
async def memory_embed_image(body: ImageEmbedRequest) -> ImageEmbedResponse:
    """Embed a single image and optionally store it."""
    if _embedder is None:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=_NOT_WIRED)
    if not _embedder.is_multimodal:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Multimodal not enabled")

    try:
        # Determine image source
        source = body.image or body.image_url
        if not source:
            raise HTTPException(status_code=400, detail="No image provided (use image or image_url)")

        vector, meta = await _embedder.embed_image(
            source,
            mime_type=body.mime_type or "",
            caption=body.caption or "",
        )

        # If collection and id provided, also index it
        if body.collection and body.id:
            from colonymind.vector import get_store
            from colonymind.vector.collections import Collection
            from colonymind.vector.query import VectorItem

            store = get_store()
            if store:
                try:
                    col = Collection(body.collection)
                except ValueError:
                    col = Collection.MEMORIES
                vi = VectorItem(
                    id=body.id,
                    text=meta.get("caption", ""),
                    vector=vector,
                    metadata=meta,
                )
                await store.add_batch(col, [vi])

        model_id = meta.get("model_id", "")
        return ImageEmbedResponse(
            model=model_id,
            vector=vector,
            image_hash=meta.get("image_hash", ""),
            image_ref=meta.get("image_ref", ""),
            thumbnail_ref=meta.get("thumbnail_ref", ""),
            caption=meta.get("caption", ""),
            width=meta.get("width", 0),
            height=meta.get("height", 0),
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.warning("image embed failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/memory/embed/image/batch", response_model=ImageBatchEmbedResponse)
async def memory_embed_image_batch(body: ImageBatchEmbedRequest) -> ImageBatchEmbedResponse:
    """Embed multiple images."""
    if _embedder is None:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=_NOT_WIRED)
    if not _embedder.is_multimodal:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Multimodal not enabled")
    if len(body.images) > 32:
        raise HTTPException(status_code=400, detail=f"Batch size {len(body.images)} exceeds limit of 32")

    try:
        results = []
        for img_item in body.images:
            source = img_item.get("image") or img_item.get("image_url")
            if not source:
                continue
            vector, meta = await _embedder.embed_image(
                source,
                mime_type=img_item.get("mime_type", ""),
                caption=img_item.get("caption", ""),
            )
            results.append({"vector": vector, **meta})

        model_id = results[0].get("model_id", "") if results else ""
        return ImageBatchEmbedResponse(model=model_id, results=results)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.warning("image batch embed failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/memory/embed/async")
async def memory_embed_async(body: dict) -> dict:
    """Async embedding for large collections — returns task_id immediately.

    Accepts the same format as /memory/embed, /memory/embed/image/batch,
    or /memory/index but runs in the background.
    Poll GET /memory/embed/async/{task_id} for status.
    """
    if _embedder is None:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=_NOT_WIRED)

    from colonymind.vector import get_store
    store = get_store()

    task_id = str(uuid.uuid4())
    _async_embed_tasks: dict = getattr(router, "_async_embed_tasks", {})
    router._async_embed_tasks = _async_embed_tasks

    embed_type = body.get("type", "texts")  # texts | images | index

    async def _run():
        try:
            _async_embed_tasks[task_id] = {"status": "running", "processed": 0, "failed": 0}

            if embed_type == "texts":
                texts = body.get("texts", body.get("inputs", []))
                if len(texts) > 1024:
                    _async_embed_tasks[task_id] = {"status": "failed", "error": f"Batch size {len(texts)} exceeds 1024"}
                    return
                vectors = await _embedder.embed_batch(texts)
                _async_embed_tasks[task_id] = {"status": "completed", "processed": len(vectors), "failed": 0}

            elif embed_type == "images":
                images = body.get("images", [])
                if len(images) > 128:
                    _async_embed_tasks[task_id] = {"status": "failed", "error": f"Batch size {len(images)} exceeds 128"}
                    return
                results = []
                failed = 0
                for img_item in images:
                    try:
                        source = img_item.get("image") or img_item.get("image_url")
                        if not source:
                            failed += 1
                            continue
                        vector, meta = await _embedder.embed_image(
                            source, mime_type=img_item.get("mime_type", ""),
                            caption=img_item.get("caption", ""),
                        )
                        results.append({"vector": vector, **meta})
                    except Exception:
                        failed += 1
                _async_embed_tasks[task_id] = {"status": "completed", "processed": len(results), "failed": failed}

            elif embed_type == "index":
                if store is None:
                    _async_embed_tasks[task_id] = {"status": "failed", "error": "VectorStore not initialized"}
                    return
                items = body.get("items", [])
                indexed = 0
                failed = 0
                for item in items:
                    try:
                        from colonymind.vector.collections import Collection
                        from colonymind.vector.query import VectorItem

                        if item.get("image") or item.get("image_url"):
                            source = item.get("image") or item.get("image_url")
                            vector, meta = await _embedder.embed_image(
                                source, mime_type=item.get("mime_type", ""),
                                caption=item.get("caption", ""),
                            )
                            col_name = item.get("collection", "memories")
                            try: col = Collection(col_name)
                            except ValueError: col = Collection.MEMORIES
                            vi = VectorItem(id=item.get("id", str(uuid.uuid4())), text=meta.get("caption", ""), vector=vector, metadata=meta)
                        else:
                            text = item.get("text", "")
                            vector = await _embedder.embed(text)
                            col_name = item.get("collection", "memories")
                            try: col = Collection(col_name)
                            except ValueError: col = Collection.MEMORIES
                            meta = item.get("metadata", {})
                            meta["model_id"] = _embedder._provider._config.model_id if hasattr(_embedder, "_provider") else ""
                            vi = VectorItem(id=item.get("id", str(uuid.uuid4())), text=text, vector=vector, metadata=meta)

                        await store.add_batch(col, [vi])
                        indexed += 1
                    except Exception:
                        failed += 1
                _async_embed_tasks[task_id] = {"status": "completed", "indexed": indexed, "failed": failed}

        except Exception as exc:
            _async_embed_tasks[task_id] = {"status": "failed", "error": str(exc)}

    _spawn_task(_run())
    return {"task_id": task_id, "status": "started"}


@router.get("/memory/embed/async/{task_id}")
async def async_embed_status(task_id: str) -> dict:
    """Poll status of an async embed task."""
    _async_embed_tasks: dict = getattr(router, "_async_embed_tasks", {})
    result = _async_embed_tasks.get(task_id)
    if result is None:
        return {"task_id": task_id, "status": "running"}
    return {"task_id": task_id, **result}


@router.post("/memory/search/multimodal", response_model=MultimodalSearchResponse)
async def memory_search_multimodal(body: MultimodalSearchRequest) -> MultimodalSearchResponse:
    """Cross-modal search — text query finds images, image query finds text."""
    if _embedder is None:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=_NOT_WIRED)

    from colonymind.vector import get_store
    store = get_store()
    if store is None:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="VectorStore not initialized")

    try:
        from colonymind.vector.collections import Collection

        col_name = body.collection or "memories"
        try:
            col = Collection(col_name)
        except ValueError:
            col = Collection.MEMORIES

        # Get query vector
        if body.query:
            if _embedder.is_multimodal:
                query_vector = await _embedder._multimodal_provider.embed_text(body.query)
            else:
                query_vector = await _embedder.embed(body.query)
        elif body.query_image:
            if not _embedder.is_multimodal:
                raise HTTPException(status_code=400, detail="Image query requires multimodal to be enabled")
            vector, _ = await _embedder.embed_image(body.query_image)
            query_vector = vector
        else:
            raise HTTPException(status_code=400, detail="No query provided (use query or query_image)")

        results = await store.search_cross_modal(
            col, query_vector,
            limit=body.limit,
            filter_modality=body.filter_modality,
            min_score=body.min_score,
        )

        model_id = ""
        if hasattr(_embedder, "_provider") and hasattr(_embedder._provider, "_config"):
            model_id = _embedder._provider._config.model_id

        return MultimodalSearchResponse(results=results, model=model_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("multimodal search failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/memory/backfill", response_model=BackfillResponse,
             dependencies=[Depends(refuse_hand_session)])
async def memory_backfill(body: BackfillRequest) -> BackfillResponse:
    """Re-embed all vectors using the current embedding pipeline.

    Returns a task_id immediately; backfill runs in the background.
    """
    if _embedder is None:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=_NOT_WIRED)

    from colonymind.vector import get_store
    store = get_store()
    if store is None:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="VectorStore not initialized")

    task_id = str(uuid.uuid4())

    async def _run():
        from colonymind.vector.backfill import backfill
        try:
            result = await backfill(store, _embedder, collection=body.collection, batch_size=body.batch_size)
            # Store result in app state for polling
            _backfill_results[task_id] = result
        except Exception as exc:
            logger.error("Backfill failed: %s", exc)

    _backfill_results: dict = getattr(router, "_backfill_results", {})
    router._backfill_results = _backfill_results

    _spawn_task(_run())
    return BackfillResponse(task_id=task_id, status="started")


@router.get("/memory/backfill/{task_id}", response_model=BackfillResponse)
async def backfill_status(task_id: str) -> BackfillResponse:
    """Check the status of a running backfill task."""
    _backfill_results: dict = getattr(router, "_backfill_results", {})
    result = _backfill_results.get(task_id)
    if result is None:
        return BackfillResponse(task_id=task_id, status="running")
    return BackfillResponse(
        task_id=task_id,
        status="completed",
        total=result.total,
        processed=result.processed,
        failed=result.failed,
        skipped=result.skipped,
        duration_s=round(result.duration_s, 2),
        errors=result.errors,
    )


@router.post("/memory/migrate", response_model=MigrateResponse,
             dependencies=[Depends(refuse_hand_session)])
async def memory_migrate(body: MigrateRequest) -> MigrateResponse:
    """Migrate all vectors from an old model to the current embedding model."""
    if _embedder is None:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=_NOT_WIRED)

    from colonymind.vector import get_store
    store = get_store()
    if store is None:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="VectorStore not initialized")

    if getattr(router, '_migrate_running', False):
        raise HTTPException(status_code=409, detail='A vector rebuild is already running')
    if not 1 <= body.batch_size <= 128:
        raise HTTPException(status_code=422, detail='Rebuild batch size must be between 1 and 128')
    if getattr(store, 'catalog', None) is not None and body.old_model_id:
        raise HTTPException(status_code=422, detail='A generation rebuild must include all retained rows')

    task_id = str(uuid.uuid4())
    if getattr(store, 'catalog', None) is not None:
        task_id = store.catalog.begin(store.identity)['id']
    router._migrate_running = True
    router._migrate_task_id = task_id

    async def _run():
        from colonymind.vector.migrate import migrate_tier
        try:
            result = await migrate_tier(store, _embedder, old_model_id=body.old_model_id,
                                        batch_size=body.batch_size)
            _migrate_results[task_id] = result
        except Exception as exc:
            logger.error("Migration failed: %s", exc)
            from colonymind.vector.migrate import MigrationResult
            _migrate_results[task_id] = MigrationResult(errors=[str(exc)], generation_id=task_id)
            if getattr(store, 'catalog', None) is not None:
                store.catalog.finish(task_id, error=str(exc))
        finally:
            router._migrate_running = False

    _migrate_results: dict = getattr(router, "_migrate_results", {})
    router._migrate_results = _migrate_results
    _migrate_results.pop(task_id, None)

    _spawn_task(_run())
    return MigrateResponse(task_id=task_id, status="started")


@router.get("/memory/migrate/{task_id}", response_model=MigrateResponse)
async def migrate_status(task_id: str) -> MigrateResponse:
    """Check the status of a running migration task."""
    if getattr(router, '_migrate_running', False) and getattr(router, '_migrate_task_id', None) == task_id:
        return MigrateResponse(task_id=task_id, status='running', generation_id=task_id)
    _migrate_results: dict = getattr(router, "_migrate_results", {})
    result = _migrate_results.get(task_id)
    if result is None:
        from colonymind.vector import get_store
        store = get_store()
        if store is not None and getattr(store, 'catalog', None) is not None:
            generation = next((g for g in store.catalog.generations() if g['id'] == task_id), None)
            if generation:
                active = store.catalog.active()
                return MigrateResponse(task_id=task_id, generation_id=task_id,
                    fingerprint=generation['fingerprint'] or '',
                    status='completed' if generation['status'] in {'ready', 'retained'} else 'resumable',
                    errors=[generation['error']] if generation.get('error') else [])
        raise HTTPException(status_code=404, detail='Unknown vector rebuild')
    return MigrateResponse(
        task_id=task_id,
        status="failed" if result.errors else "completed",
        collections_migrated=result.collections_migrated,
        vectors_migrated=result.vectors_migrated,
        vectors_failed=result.vectors_failed,
        duration_s=round(result.duration_s, 2),
        errors=result.errors,
        generation_id=getattr(result, 'generation_id', ''),
        fingerprint=getattr(result, 'fingerprint', ''),
    )


@router.post("/memory/index", response_model=IndexResponse,
             dependencies=[Depends(refuse_hand_session)])
async def memory_index(body: IndexRequest) -> IndexResponse:
    """Embed and store items in one call."""
    if _embedder is None:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=_NOT_WIRED)

    from colonymind.vector import get_store
    store = get_store()
    if store is None:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="VectorStore not initialized")

    if not body.items:
        return IndexResponse(model="unknown", indexed=0, failed=0)
    if len(body.items) > 128:
        raise HTTPException(status_code=400, detail=f"Batch size {len(body.items)} exceeds limit of 128")

    try:
        from colonymind.vector.collections import Collection
        from colonymind.vector.query import VectorItem

        # Determine current model_id
        model_id = ""
        if hasattr(_embedder, "_provider") and hasattr(_embedder._provider, "_config"):
            model_id = _embedder._provider._config.model_id

        # Separate text items from image items
        text_items = []
        image_items = []
        for item in body.items:
            if item.get("image") or item.get("image_url"):
                image_items.append(item)
            else:
                text_items.append(item)

        indexed = 0
        failed = 0

        # Process text items
        if text_items:
            texts = [item.get("text", "") for item in text_items]
            vectors = await _embedder.embed_batch(texts)
            for item, vector in zip(text_items, vectors):
                try:
                    col_name = item.get("collection", "memories")
                    try:
                        col = Collection(col_name)
                    except ValueError:
                        col = Collection.MEMORIES
                    meta = item.get("metadata", {})
                    meta["model_id"] = model_id
                    vi = VectorItem(id=item.get("id", str(uuid.uuid4())), text=item.get("text", ""), vector=vector, metadata=meta)
                    await store.add_batch(col, [vi])
                    indexed += 1
                except Exception as exc:
                    logger.warning("index text item failed: %s", exc)
                    failed += 1

        # Process image items
        for item in image_items:
            try:
                source = item.get("image") or item.get("image_url")
                if not source:
                    failed += 1
                    continue
                vector, meta = await _embedder.embed_image(
                    source,
                    mime_type=item.get("mime_type", ""),
                    caption=item.get("caption", ""),
                )
                col_name = item.get("collection", "memories")
                try:
                    col = Collection(col_name)
                except ValueError:
                    col = Collection.MEMORIES
                vi = VectorItem(id=item.get("id", str(uuid.uuid4())), text=meta.get("caption", ""), vector=vector, metadata=meta)
                await store.add_batch(col, [vi])
                indexed += 1
            except Exception as exc:
                logger.warning("index image item failed: %s", exc)
                failed += 1

        return IndexResponse(model=model_id or "unknown", indexed=indexed, failed=failed)
    except Exception as exc:
        logger.warning("memory_index failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/owner/location", dependencies=[Depends(require_companion_owner)])
async def owner_location(body: OwnerLocationRequest) -> dict:
    """Record only the latest location reported by the owner's authenticated Companion phone."""
    if _contacts_store is None:
        raise HTTPException(503, detail="contacts_unavailable")
    if await _contacts_store.get(owner_person_id()) is None:
        raise HTTPException(404, detail="owner_contact_not_found")
    applied, contact = await _contacts_store.set_location_report(owner_person_id(), **body.model_dump())
    return {"ok": True, "applied": applied, "timezone": contact.timezone,
            "city": contact.location_city, "reported_at": contact.location_reported_at}


async def _build_temporal_section(
    contact_id: Optional[str], override_tz: Optional[str] = None,
    *, include_global_heads_up: bool = True, owner_private_turn: bool = False,
) -> ContextSection:
    """The agent's reference frame for "now" (v0.21.0, factored for per-turn use).

    Every audience gets the agent's home clock. Only an explicitly private owner
    turn also gets the owner's phone report, last exchange and heads-up items.
    Cheap (no memory search): safe per turn.
    """
    from colonymind.memory import producers, turn_budget
    from colonymind.util import temporal as _temporal
    if not (owner_private_turn and contact_id and contact_id == owner_person_id()):
        return ContextSection(id="temporal-context", title="Current Time", priority=100,
                              body=_temporal.describe_agent_now())
    agent_tz = _temporal.agent_timezone()
    contact_tz = None
    contact_label = "the contact"
    contact_obj = None
    if _contacts_store is not None and contact_id:
        try:
            contact_obj = await _contacts_store.get(contact_id)
            if contact_obj is not None:
                contact_tz = getattr(contact_obj, "timezone", None)
                contact_label = (
                    contact_obj.display_name or contact_obj.given_name or "the contact"
                )
        except Exception:
            pass
    # A fallback communication frame is not evidence of a contact's timezone.
    is_owner = bool(contact_id and contact_id == owner_person_id())
    if is_owner and contact_obj is None:
        contact_label = "the owner"
    t_lines = [_temporal.describe_now(agent_tz, contact_tz, contact_label, override_tz=override_tz,
                                     owner=is_owner, location_contact=contact_obj)]
    if contact_obj is not None and getattr(contact_obj, "last_interaction_at", None):
        li = contact_obj.last_interaction_at
        t_lines.append(
            f"Last exchange with {contact_label}: {_temporal.humanize_delta(li)} "
            f"({_temporal.bucket(li, agent_tz)})."
        )
    if include_global_heads_up:
        try:
            from colonymind.util.session_safety import load_last_user_message_at
            last_owner = await producers.read(load_last_user_message_at)
            since = _temporal.humanize_delta(last_owner) if last_owner else ""
            if since and since != "just now":  # in a live exchange this line says nothing
                t_lines.append(f"Owner last messaged {since}.")
        except Exception:
            pass
    # Heads-up: time-sensitive items (overdue commitments + cadence-overdue contacts)
    heads = []
    if include_global_heads_up:
        try:
            if _commitment_store is not None:
                for c in (await producers.read(_commitment_store.get_overdue, default=[]) or [])[:3]:
                    desc = (c.get("description") or "a commitment")[:80]
                    heads.append(
                        f"⚠️ Overdue: {desc} (was due {_temporal.humanize_delta(c.get('due_at'))})"
                    )
        except Exception:
            pass
        try:
            if _contacts_store is not None:
                exclude = {contact_id} if contact_id else set()
                overdue_contacts = await _contacts_store.compute_cadence_overdue(
                    overdue_only=True, limit=3, exclude_ids=exclude,
                )
                for o in overdue_contacts[:2]:
                    heads.append(
                        f"🕰️ Haven't talked to {o['name']} in {int(o['days_since'])}d "
                        f"(usually ~{o['cadence_days']:g}d)."
                    )
        except Exception:
            pass
    if heads:
        t_lines.append("Heads-up:")
        t_lines.extend("  " + h for h in heads)
    return ContextSection(
        id="temporal-context",
        title="Current Time",
        body="\n".join(t_lines),
        priority=100,
    )


async def _owner_continuity(contact_id: str, channel: str | None, override_tz: str | None = None,
                            *, owner_private_turn: bool = False
                            ) -> tuple[ContextSection | None, str | None]:
    """Shared by full recall and the cheap trivial-turn read; no separate store or text cache."""
    from contextlib import nullcontext
    from datetime import timedelta
    from colonymind.contacts.continuity import recent_elsewhere
    from colonymind.memory import producers, turn_budget
    from colonymind.util import temporal
    if not owner_private_turn or not contact_id or contact_id != owner_person_id():
        return None, None
    contact = await _contacts_store.get(contact_id) if _contacts_store is not None else None
    recorded_zone = getattr(contact, "timezone", None)
    recorded_zone = recorded_zone if temporal.is_valid_timezone(recorded_zone) else None
    if _comms_log is None or not channel:
        return None, recorded_zone
    now = temporal.now_utc()
    read_scope = nullcontext() if turn_budget.current() is not None else turn_budget.turn(
        turn_budget.TurnBudget(wait_s=1.5))
    with read_scope:
        rows = await producers.read(_comms_log.elsewhere_history, contact_id, channel,
                                    since=(now - timedelta(hours=72)).isoformat(), until=now.isoformat())
    handles = await _contacts_store.get_handles(contact_id) if _contacts_store is not None else []
    text = recent_elsewhere(rows, channel,
                            temporal.resolve_communication_timezone(recorded_zone, override_tz), now=now,
                            handles=[handle.address for handle in handles])
    section = ContextSection(id="colonymind-owner-continuity", title="Recent conversation elsewhere",
                             body=text, priority=95) if text else None
    return section, recorded_zone


@router.get("/context/temporal")
async def context_temporal(contact_id: Optional[str] = None,
                           tz: Optional[str] = None,
                           channel: Optional[str] = None,
                           owner_private_turn: bool = False,
                           request: Request = None) -> dict:
    """Always-fresh temporal brief for per-turn injection (no caching layer).

    The memory provider calls this every turn so the agent's Current Time
    block can never go stale inside a long-running session (the full
    /context/assemble result is session-cached by design; time must not be).
    The owner's heads-up (their overdue commitments, contacts they have not
    talked to) is owner context: a guest's brief never carries it.
    """
    contact_id = resolve_request_person(request, context_person_id=contact_id)
    owner_private_turn = bool(owner_private_turn and contact_id == owner_person_id()
                              and not _viewer_is_guest(request, contact_id))
    section = await _build_temporal_section(
        contact_id, tz, include_global_heads_up=not channel,
        owner_private_turn=owner_private_turn)
    result = {"id": section.id, "title": section.title, "body": section.body}
    if owner_private_turn:
        continuity, zone = await _owner_continuity(contact_id, channel, tz,
                                               owner_private_turn=owner_private_turn)
        result["owner_timezone"] = zone
        contact = await _contacts_store.get(contact_id) if _contacts_store is not None else None
        result["owner_label"] = (getattr(contact, "display_name", None)
                                 or getattr(contact, "given_name", None) or "the owner")
        if continuity is not None:
            result["continuity"] = continuity.model_dump()
    return result


@router.post("/context/assemble", response_model=ContextAssembleResponse, tags=[PUBLIC])
async def context_assemble(
    body: ContextAssembleRequest,
    request: Request = None,
    background_tasks: BackgroundTasks = None,
) -> ContextAssembleResponse:
    """Assemble within mandatory record, character and elapsed-time caps."""
    from colonymind.memory import blocks, producers, turn_budget
    budget = turn_budget.TurnBudget.for_host(body.wait_ms)
    allowance = int(blocks.settings()["budget_chars"])
    if body.inline_chars is not None:
        body = body.model_copy(update={"inline_chars": min(body.inline_chars, allowance)})
    unavailable = False
    with turn_budget.turn(budget):
        recall_exit.begin(None, bool(body.incoming_message and body.incoming_message.content), body.context.turn_id)
        writes = []
        response = None
        try:
            async with asyncio.timeout(budget.assembly_left()):
                response = await _context_assemble(body, request)
            if budget.assembly_left() > 0:
                response.budget["wait_ms"] = round(budget.wait_s * 1000)
                response.budget["sync_pool_skipped"] = budget.sync_pool_skipped
                response.budget["sync_read_failed"] = budget.sync_read_failed
                writes = turn_budget.take_writes(response.sections)
            else:
                response = None
        except TimeoutError as exc:
            recall_exit.failed(exc, "assembly_left")
            response = None
        except producers.Unavailable as exc:
            recall_exit.failed(exc, "search")
            response = None
            unavailable = True
        finally:
            recall_exit.accepted(response.sections if response is not None else ())
    if response is not None or budget.recall_exit is not None:
        if background_tasks is None:
            turn_budget.schedule_writes(writes, budget.recall_exit)
        else:
            background_tasks.add_task(turn_budget.start_writes, writes, budget.recall_exit)
    if response is not None:
        response.recall_exit = recall_exit.snapshot(budget.recall_exit)
        return response
    # An expired assembly supplies no partial evidence or stale citations.
    return ContextAssembleResponse(sections=[], recall_exit=recall_exit.snapshot(budget.recall_exit), budget={
        "budget": body.inline_chars if body.inline_chars is not None else allowance,
        "chars_after": 0, "recall_items": blocks.settings()["recall_items"],
        "wait_ms": round(budget.wait_s * 1000), "timed_out": not unavailable,
        **({"unavailable": True} if unavailable else {}),
        "sync_pool_skipped": budget.sync_pool_skipped, "sync_read_failed": budget.sync_read_failed})


async def _context_assemble(
    body: ContextAssembleRequest,
    request: Request = None,
) -> ContextAssembleResponse:
    from colonymind.memory import curated
    from types import SimpleNamespace
    path = Path(get_state_dir()) / "turn-idempotency.db"
    ledger = SimpleNamespace(db_path=path) if path.is_file() else None
    captured = []
    async def read():
        if captured:
            from colonymind.turns import get_turn_idempotency_ledger
            current = curated.watermark(get_turn_idempotency_ledger(get_state_dir()), body.context.contact_id)
            if current != captured[0]:
                return ContextAssembleResponse(sections=[], source_erasure_watermark=captured[0],
                                               budget={'chars_after': 0})
        result = await _read_context_assemble(body, request)
        if not captured:
            captured.append(result.source_erasure_watermark)
        return result.model_copy(update={'source_erasure_watermark': captured[0]})
    def withhold(result):
        from colonymind.memory import blocks
        sections = [s for s in result.sections if s.id != 'colonymind-memory']
        return result.model_copy(update={'sections': sections, 'budget': {
            **result.budget, 'unavailable': True,
            'chars_after': sum(blocks.cost(s.title or s.id, s.body) for s in sections)}})
    result = await curated.context_fenced_read(ledger, read, empty=None, withhold=withhold)
    if result is None:
        from colonymind.memory.producers import Unavailable
        raise Unavailable("context erasure during assembly")
    return result


async def _read_context_assemble(
    body: ContextAssembleRequest,
    request: Request = None,
) -> ContextAssembleResponse:
    from colonymind.memory import producers, turn_budget
    body.context.contact_id = resolve_request_person(
        request,
        context_person_id=body.context.contact_id,
        audience=body.audience,
    ) or body.context.contact_id
    authority = request_authority(request)
    if body.trace:
        _require_trace_authority(request)
    # Only the key's own view with no person named is unscoped; every other caller, including a
    # development caller without the key (never the owner), sees the person it names as a viewer.
    viewer = body.context.contact_id if authority.authenticated and not authority.anonymous else (
        body.context.contact_id or "dev-anonymous")
    # Stamp before reading any producer. A concurrent forget makes the entire
    # packet stale at the native request boundary, including derived sections.
    source_erasure_watermark = None
    try:
        from colonymind.turns import get_turn_idempotency_ledger
        source_erasure_watermark = await producers.read(
            lambda: get_turn_idempotency_ledger(get_state_dir()).erasure_watermark(body.context.contact_id)
            if (Path(get_state_dir()) / "turn-idempotency.db").exists() else 0)
    except Exception as exc:
        raise producers.Unavailable("context erasure freshness unavailable") from exc
    superseded: list = []
    notice_rows: list = []
    notice_ledger = None
    served_callbacks: list = []
    from colonymind.memory.trace import NULL_TRACE, RecallTrace
    trace = RecallTrace() if body.trace else NULL_TRACE
    # One time budget for the turn, from how long the host waits for this answer (``memory.turn_budget``).
    from colonymind.memory.selection import turn_packets
    from colonymind.memory import blocks
    from colonymind.turns.source_vectors import turn_query_vectors
    # The turn's query vectors are shared by recall's vector lane and the lesson match, and end with the turn.
    with turn_packets() as recalled, turn_query_vectors():
        sections = await _assemble_sections(body, viewer_person_id=viewer, request=request, superseded=superseded,
                                            **({"trace": trace} if trace else {}))
        # The corrections are owner-critical and pinned: they are fitted with everything else, never appended after.
        sections = _turn_context_corrections(body, sections, superseded, viewer=viewer)
        # Every item kept whole beside the memory now exists, annotated: the memory is fitted in what they leave.
        held: list[ContextSection] = []
        allowance = body.inline_chars if body.inline_chars is not None else int(blocks.settings()["budget_chars"])
        for packet in recalled:
            memory = next((section for section in sections if section.id == "colonymind-memory"), None)
            sections, shown = await _memory_within_allowance(sections, packet, inline_chars=allowance,
                                                             records=superseded, trace=trace)
            if memory is not None and not any(section.id == "colonymind-memory" for section in sections):
                held.append(memory)         # not one record fits: held back whole, and named below
            if packet.served is not None and shown:
                served_callbacks.append((packet.served, shown))
            if body.recall_notice_details:
                notice_rows.extend(shown)
                notice_ledger = packet.ledger
        sections = await _turn_context_selection(body, sections, reserve=_held_note_cost(held, allowance))
    # Level three changes display order only; the section retains its admission rank and full-body fit.
    if getattr(body, "_standing_last", False):
        sections = ([s for s in sections if s.id != "colonymind-standing-lessons"]
                    + [s for s in sections if s.id == "colonymind-standing-lessons"])
    # The block fit shares the selection's budget: the host's inline allowance, or context.budget_chars.
    sections, budget_report = _fit_sections(sections, budget=body.inline_chars, shown=body.shown or (),
                                            where="context/assemble", held=held)
    if trace:
        trace.context_section(next((section.body for section in sections if section.id == "colonymind-memory"), None))

    details = None
    if body.recall_notice_details:
        from colonymind.memory.notice import recall_notice_details
        memory = next((section.body for section in sections if section.id == "colonymind-memory"), "")
        details = await producers.read(recall_notice_details, notice_rows, content=memory,
                                           contact_id=body.context.contact_id, ledger=notice_ledger)

    if _telemetry is not None:
        try:
            await _telemetry.touch("last_prefetch_at")
        except Exception:
            pass

    from colonymind.memory import turn_budget
    if turn_budget.current().assembly_left() > 0:
        turn_budget.defer_write(_remember_served, body, sections, viewer=viewer)
        with turn_budget.producer_writes("colonymind-memory"):
            for callback, shown in served_callbacks:
                turn_budget.defer_write(callback, shown)

    return ContextAssembleResponse(
        sections=sections,
        notices=([_GUEST_CONTEXT_NOTICE] if _viewer_is_guest(request, body.context.contact_id) else None),
        source_erasure_watermark=source_erasure_watermark,
        budget=budget_report,
        recall_notice_details=details,
        **({"trace": trace.report()} if trace else {}),
    )


def _fit_sections(sections: list[ContextSection], *, budget: Optional[int], shown=(),
                  where: str, held: list[ContextSection] = ()) -> tuple[list[ContextSection], Optional[dict]]:
    """The resident-block fit (``colonymind/memory/blocks.py``): every section within its hard limit and all of
    them within ``budget`` (the host's inline allowance, the one the per-turn selection fitted the context to;
    ``context.budget_chars`` when the caller names none), fresh state admitted first,
    anything not shown named in a ``Held back this turn`` section and on the returned report. ``held`` are
    sections already held back whole (a recalled memory with not one record that fits beside the pinned items):
    they are named there too, in room kept for them. The configured ceiling always applies, including when
    the host offers more room or says it will collapse unchanged sections."""
    from colonymind.memory import blocks
    conf = blocks.settings()
    items = [blocks.Block(section.id, section.title or section.id, section.body, index)
             for index, section in enumerate(sections)]
    ceiling = int(conf["budget_chars"])
    total = ceiling if budget is None else max(0, min(int(budget), ceiling))
    reserve = _held_note_cost(held, total)
    # Bound the assembled payload itself; a caller's fingerprints cannot widen its character cap.
    fitted, report = blocks.fit(items, budget=total - reserve, limits=conf.get("limits"))
    report["recall_items"] = conf["recall_items"]
    result = [sections[item.ref].model_copy(update={"body": item.body}) if item.ref >= 0 else
              ContextSection(id=item.id, title=item.title, body=item.body, priority=0) for item in fitted]
    if held:
        report = {**report, "budget": total, "held_back": [*report["held_back"], *(
            {"id": section.id, "chars": len(section.body),
             "sha256": hashlib.sha256(section.body.encode("utf-8", "replace")).hexdigest()} for section in held)]}
    if reserve:
        # Named in the room kept for it (a budget too small for the words still names it on the report).
        note = _held_note(held)
        index = next((i for i, section in enumerate(result) if section.id == blocks.HELD_ID), None)
        if index is None:
            result.append(ContextSection(id=blocks.HELD_ID, title=blocks.HELD_TITLE, body=note, priority=0))
            report["chars_after"] += blocks.cost(blocks.HELD_TITLE, note)
        else:
            result[index] = result[index].model_copy(update={"body": result[index].body + "\n" + note})
            report["chars_after"] += len(note) + 1
    blocks.record_fit(report, where=where)
    return result, report


def _held_note(held: list[ContextSection]) -> str:
    """The words that name sections held back whole before the block fit (``_fit_sections``)."""
    return ("Not shown, not one recalled record fits beside the items kept whole within this turn's context budget: "
            + "; ".join(section.title or section.id for section in held)
            + ". Their records are intact and a memory search reaches them.")


def _held_note_cost(held: list[ContextSection], budget: Optional[int]) -> int:
    """Room for naming ``held``, when the words themselves fit within the budget."""
    from colonymind.memory import blocks
    if not held or budget is None:
        return 0
    cost = blocks.cost(blocks.HELD_TITLE, _held_note(held))
    return cost if cost <= budget else 0


_GUEST_CONTEXT_NOTICE = (
    "Contact-scoped context provides their own source evidence, claims and media, the commitments "
    "proven shared in their source evidence, and their digest. The owner's tasks, graph, "
    "relationship and global context are omitted.")


async def _turn_context_selection(body: ContextAssembleRequest, sections: list[ContextSection], *,
                                  reserve: int = 0) -> list[ContextSection]:
    """The per-turn selection within the budget (``memory.compass``). It runs once the assembly, recall included,
    has finished: recall and the judge share the reranker, and recall's deadline is never spent behind the judge.
    ``reserve`` characters of the host's allowance are kept for the block fit's note of what was held back.

    When the owner-critical items and the memory alone exceed the host's allowance (``over_allowance``) the
    selection cannot fit the context; the assembled context goes on to the block
    fit, which admits fresh state first, keeps a head of every block it can and names each one it holds back,
    rather than keeping the pinned items and the memory alone and letting the host move the rest out of line."""
    from colonymind.memory import compass
    settings = compass.selection_settings()
    if not settings.enabled:
        return sections
    query = body.incoming_message.content if body.incoming_message else ""
    provider = _reranker
    selected, report = await compass.select_context(
        sections, query, provider.rerank if provider is not None else None, settings=settings,
        inline_chars=None if body.inline_chars is None else body.inline_chars - reserve)
    logger.debug("context selection: %s", report)
    from colonymind.memory import recall_stats
    recall_stats.STATS.judged(report)
    if report.get("status") == "over_allowance":
        return sections
    return selected


def _served_key(body: ContextAssembleRequest, viewer: Optional[str]):
    session_id = body.context.session_id if body.context else ""
    return (viewer or "", session_id) if session_id else None


def _turn_context_corrections(body: ContextAssembleRequest, sections: list[ContextSection],
                              superseded: list, *, viewer: Optional[str]) -> list[ContextSection]:
    """The corrections for values this conversation was served earlier that the record has since superseded, as a
    section of this turn's context (pinned by the selection)."""
    from colonymind.memory import compass
    key = _served_key(body, viewer)
    note = compass.SERVED.corrections(key, superseded) if key else ""
    if note:
        sections = [*sections, ContextSection(id=compass.CORRECTIONS, title="Corrections to earlier context",
                                              body=note, priority=99)]
    return sections


def _remember_served(body: ContextAssembleRequest, sections: list[ContextSection], *, viewer: Optional[str]) -> None:
    """What this turn served the conversation, once selected: the corrections of later turns are owed against it."""
    from colonymind.memory import compass
    key = _served_key(body, viewer)
    if key:
        compass.SERVED.remember(key, "\n".join(section.body for section in sections))


def _preference_sections(learner, *, contact_id, session_id, query):
    perspective = getattr(learner, "perspective", None)
    sections = []
    preference_sources = []
    _brief = learner.build_brief(source_ids=preference_sources)
    if _brief:
        sections.append(ContextSection(
            id="colonymind-owner-preferences",
            title="How they want me to communicate",
            body=_brief,
            priority=88,
            citations=(perspective.ledger.source_references(preference_sources,
                contact_id=contact_id, session_id=session_id)
                if perspective is not None else None),
        ))
    if perspective is not None:
        working_sources = []
        working_brief = perspective.brief(query=query, source_ids=working_sources)
        if working_brief:
            sections.append(ContextSection(id='colonymind-self-perspective',
                title='Owner priority corrections', body=working_brief, priority=87,
                citations=perspective.ledger.source_references(working_sources,
                    contact_id=contact_id, session_id=session_id)))
    return sections


async def _assemble_sections(
    body: ContextAssembleRequest,
    *,
    viewer_person_id: Optional[str],
    request: Request | None = None,
    superseded: list | None = None,
    trace=None,
    read_only: bool = False,
) -> list[ContextSection]:
    """The context sections for ``body``'s contact as ``viewer_person_id`` may see them.

    A viewer other than the owner gets the contact-scoped set: their own
    canonical recall, the commitments their sources prove shared, their
    appraisal perspective and their digest; never an owner-only or global
    section. It fails closed: with no owner configured the owner is the
    reserved ``owner`` person, so every named viewer is a guest. ``None`` is
    the key holder asking about no one (the owner's own API), which keeps the
    unscoped context. The route and the mind's recipient packet
    (``assemble_packet``) share this one assembly. A line asserting a value
    the record has superseded carries the current value (``memory.compass``);
    those records are added to ``superseded`` when it is given. A recall
    ``trace`` (``memory.trace``) is filled by the memory's recall. Within context assembly the recalled packet is
    handed over (``selection.turn_packets``), to be fitted once every pinned item exists.

    The memory is recalled while the other lanes are assembled (none of them
    calls the reranker), and the recalled section takes its place in the
    order. An assembly that is cancelled or fails cancels that recall and
    waits for it to end before it propagates.
    """
    from colonymind.memory import curated
    from types import SimpleNamespace
    path = Path(get_state_dir()) / "turn-idempotency.db"
    ledger = SimpleNamespace(db_path=path) if path.is_file() else None
    async def read():
        return await _read_assemble_sections(body, viewer_person_id=viewer_person_id, request=request, superseded=superseded, trace=trace, read_only=read_only)
    return await curated.context_fenced_read(ledger, read, empty=[],
        withhold=lambda sections: [s for s in sections if s.id != 'colonymind-memory'])


async def _read_assemble_sections(
    body: ContextAssembleRequest,
    *,
    viewer_person_id: Optional[str],
    request: Request | None = None,
    superseded: list | None = None,
    trace=None,
    read_only: bool = False,
) -> list[ContextSection]:
    started: list[asyncio.Task] = []
    try:
        return await _assembled_lanes(body, viewer_person_id=viewer_person_id, request=request,
                                      superseded=superseded, started=started, trace=trace, read_only=read_only)
    finally:
        pending = [task for task in started if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


async def _assembled_lanes(
    body: ContextAssembleRequest,
    *,
    viewer_person_id: Optional[str],
    request: Request | None,
    superseded: list | None,
    started: list,
    trace=None,
    read_only: bool = False,
) -> list[ContextSection]:
    """``_assemble_sections``' work; every task it starts is added to ``started``, which the caller ends."""
    from colonymind.memory import producers, turn_budget
    owner_id = owner_person_id()
    _canonical_only = bool(viewer_person_id) and viewer_person_id != owner_id
    _viewer_is_owner = bool(viewer_person_id) and viewer_person_id == owner_id
    # Legacy person stores can contain private owner observations ABOUT a
    # guest. Exact subject identity does not make those observations shareable.
    _legacy_global_allowed = _exact_person_allowed = not _canonical_only
    sections: list[ContextSection] = []
    records: list = []
    query_text = body.incoming_message.content if body.incoming_message else ""

    # Read authenticated work before recall selection. Native requests still
    # refresh this observation at their existing model boundary.
    current_work_available = False
    _owner_turn = False
    # --- Scoped work observations (not a commitment lock) ---
    try:
        from colonymind.api.viewer import authorized_viewer
        from colonymind.api.routers.executions import with_queue_work
        from colonymind.turns.executions import has_work_records, request_work_context
        person, owner = authorized_viewer(request, body.context.contact_id)
        # Public/guest turns do not get cross-session activity. The owner view
        # is sealed from existing exact person grants, never a body owner flag.
        _owner_turn = bool(owner) and not read_only
        if owner:
            work = await with_queue_work({}, owner=True, limit=8)
            current_work_available = True
            if has_work_records(work):  # an all-idle or all-unavailable view is not worth a section
                observed = request_work_context(work, session_id=body.context.session_id,
                                                limit=8, max_chars=4000)
                sections.append(ContextSection(id="colonymind-executions", title="Work observed at turn start", body=observed['text'], priority=73))
    except HTTPException:
        pass
    except Exception:
        logger.debug("execution observation view unavailable", exc_info=True)

    recall_exit.begin(_owner_turn and _viewer_is_owner, bool(query_text), body.context.turn_id)

    # --- Temporal Context: see _build_temporal_section ---
    metadata = body.context.metadata or {}
    # A turn in a group chat is given that chat's own history only (``turns.group_chats``), so it is nobody's own
    # turn for the learners and is told of no work in progress. Its session is noted for the chat once its
    # context is accepted (at once for a caller with no turn budget), off the event loop.
    from colonymind.turns import group_chats
    group = group_chats.of_turn(metadata)
    if group is not None:
        _owner_turn = current_work_available = False
        if group and not read_only and turn_budget.current() is not None:
            turn_budget.defer_write(asyncio.to_thread, group_chats.note, group, body.context.session_id)
        elif group and not read_only:
            await asyncio.to_thread(group_chats.note, group, body.context.session_id)
    owner_private_turn = bool(_viewer_is_owner and metadata.get("owner_private_turn") is True
                              and str(metadata.get("chat_type") or "").strip().lower() in {"", "dm"}
                              and body.audience not in {"shared", "global"}
                              and not metadata.get("kanban_task") and not metadata.get("run"))
    try:
        cid = (
            body.context.contact_id
            if body.context and _exact_person_allowed else None
        )
        override_tz = getattr(body.context, "timezone", None) if body.context else None
        sections.append(await _build_temporal_section(
            cid,
            override_tz,
            include_global_heads_up=_legacy_global_allowed,
            owner_private_turn=owner_private_turn,
        ))
    except Exception as exc:
        logger.debug("context_assemble temporal section failed: %s", exc)

    if _viewer_is_owner:
        try:
            continuity, _ = await _owner_continuity(
                body.context.contact_id, body.context.channel_id, body.context.timezone,
                owner_private_turn=owner_private_turn)
            if continuity is not None:
                sections.append(continuity)
        except Exception:
            logger.debug("context_assemble owner continuity unavailable", exc_info=True)

    # --- Memory: authorized candidates, one selection and one budget ---
    # This turn's semantic recall, for the status block: None without a recall, "unknown" until the
    # collection answers, then its own state (``CollectedSources.semantic``).
    recall_semantic: Optional[str] = "unknown" if query_text else None
    # The status block's place on the owner's own turn, filled once that recall has answered.
    status_slot: Optional[ContextSection] = None

    async def recalled_memory() -> ContextSection | None:
        section = None
        completed = False
        recall_exit.producer_started()
        try:
            from colonymind.memory.live_recall import LIVE
            from colonymind.plasticity.learners.query_adapter import RecallRequest
            from contextlib import nullcontext
            recall_request = RecallRequest.create(getattr(_mind(), 'plasticity', None),
                owner=_owner_turn and _viewer_is_owner, turn_id=getattr(body.context, 'turn_id', None))
            if recall_request is not None:
                selector = _memory_context_selector()
                recall_request.metadata = getattr(selector, '_rerank_calibration_metadata', lambda: {})
            with LIVE.running(), turn_budget.producer_writes("colonymind-memory") as writes:
                with recall_request.use() if recall_request is not None else nullcontext():
                    section = await _recalled_memory(recall_request)
                    if section is None:
                        writes[:] = [write for write in writes if write[0] is None]
                    completed = True
                    return section
        except BaseException as exc:
            recall_exit.failed(exc)
            raise
        finally:
            recall_exit.producer_finished(section, completed)

    async def _recalled_memory(recall_request=None) -> ContextSection | None:
        nonlocal recall_semantic
        from colonymind.memory.search import collect_sources, select_memory
        from colonymind.memory.search import turn_recall_chars
        from colonymind.memory.trace import NULL_TRACE, RecallTrace
        recall_trace = trace if trace is not None else (RecallTrace() if recall_request is not None else NULL_TRACE)
        try:
            from colonymind.turns import get_turn_idempotency_ledger
            from colonymind.vector import get_store, get_pipeline
            source_ledger = (get_turn_idempotency_ledger(get_state_dir())
                if (Path(get_state_dir()) / "turn-idempotency.db").exists() else None)
            # The broadcast set expands the owner's recall query (architecture 4.5); the
            # section itself is rendered below, never for a guest.
            recall_query = (await producers.read(_mind_recall_query, query_text)
                            if _owner_turn else query_text)
            collected = await collect_sources(source_ledger, query=recall_query,
                contact_id=body.context.contact_id, session_id=body.context.session_id,
                vector_store=get_store(), embedding_pipeline=get_pipeline(), trace=recall_trace,
                **({'recall_request': recall_request} if recall_request is not None else {}),
                **({'group': group} if group is not None else {}))
            recall_semantic = collected.semantic
            contact_tz = None
            if not _canonical_only and _contacts_store is not None:
                try:
                    contact = await _contacts_store.get(body.context.contact_id)
                    contact_tz = getattr(contact, "timezone", None)
                except Exception:
                    logger.debug("contact timezone unavailable for memory recall", exc_info=True)
            from colonymind.util.temporal import resolve_communication_timezone
            from types import SimpleNamespace
            from colonymind.memory.selection import hand_over, keep_offered
            # A turn's packet uses context.recall_items even after a verified cutoff. Its offered pool is kept to pack it
            # again within the host's allowance once every item kept whole beside it exists.
            with keep_offered() as offered:
                packet = await select_memory(collected, query=query_text, selector=_memory_context_selector(),
                    timezone_name=resolve_communication_timezone(
                        contact_tz, body.context.timezone or ("UTC" if _canonical_only else None)),
                    current_work_available=current_work_available, limit=None, trace=recall_trace,
                    max_chars=turn_recall_chars(body.inline_chars))
            if recall_request is not None:
                turn_budget.defer_write(recall_request.driver.record_trace, recall_request.turn_id, recall_trace.report())
            if packet.content:
                from functools import partial
                served = (partial(_learning_served, source_ledger, query_text, body)
                          if _owner_turn and _viewer_is_owner and source_ledger is not None else None)
                recalled = SimpleNamespace(packet=packet, offered=offered[-1] if offered else None,
                                           curated_recall=collected.curated_recall,
                                           ledger=source_ledger, contact_id=body.context.contact_id,
                                           session_id=body.context.session_id, served=served)
                # Context assembly fits it and records it as served; the recipient packet is served as recalled.
                if not hand_over(recalled) and served is not None:
                    turn_budget.defer_write(served, packet.selected)
                return ContextSection(
                    id="colonymind-memory", title="Relevant Memories", body=packet.content,
                    priority=90, citations=packet.source_refs or None)
            if _owner_turn and _viewer_is_owner:
                # The owner's own turn, by the viewer identity every other owner-only section uses.
                with turn_budget.producer_writes(None):
                    turn_budget.defer_write(_mind_note_novel, query_text)
        except Exception as exc:
            recall_exit.failed(exc)
            logger.warning("combined memory selection failed (%s)", type(exc).__name__)
        return None

    # Recalled while the other lanes are assembled; the placeholder keeps its place.
    memory_slot = ContextSection(id="colonymind-memory", title="Relevant Memories", body="", priority=90)
    memory_task = asyncio.create_task(recalled_memory(), name="colonymind-recall") if query_text else None
    if memory_task is not None:
        started.append(memory_task)
        sections.append(memory_slot)

    # --- Pending Initiatives (v0.13.0) ---
    if _legacy_global_allowed and body.include_initiatives\
            and _initiative_store is not None:
        try:
            pending = await producers.read(_initiative_store.list, status=["pending"], limit=10, default=[])
            if pending:
                body_text = "\n".join(
                    f"• [{i.type}] {i.description} (priority: {i.priority:.0%})"
                    for i in pending
                )
                sections.append(ContextSection(
                    id="colonymind-initiatives",
                    title="Pending Initiatives",
                    body=body_text,
                    priority=50,
                ))
        except Exception as exc:
            logger.warning("context_assemble initiatives failed: %s", exc)

    # The registry contains internal initiative executors, not instruction
    # skills installed in the requesting runtime. The host owns its actual
    # skill catalog and discovery tools; do not advertise these Python
    # executor names as callable skills in ordinary turn context.

    # --- Pending Commitments ---
    contact_id = body.context.contact_id if body.context else None
    if _commitment_store is not None:
        try:
            commitments = await producers.read(_commitment_store.list,
                person_id=contact_id, status=["pending", "overdue"], limit=50 if _canonical_only else 5, default=[],
            )
            # The exact-person list already includes overdue rows. A sealed
            # guest must not trigger a global get-then-filter query.
            overdue = (
                await producers.read(_commitment_store.get_overdue, default=[])
                if _legacy_global_allowed else []
            )
            if contact_id and overdue:
                overdue = [
                    c for c in overdue
                    if c.get("person_id") == contact_id
                ]
            _listed = (commitments if isinstance(commitments, list)
                       else commitments.get("commitments", []))
            if _canonical_only:
                _listed = await producers.read(_canonical_shared_commitments, _listed, contact_id, default=[])
            _seen_ids = {c.get("id") for c in _listed}
            _overdue_ids = {c.get("id") for c in overdue}
            all_comms = _listed + [c for c in overdue[:5]
                                   if c.get("id") not in _seen_ids]
            from colonymind.memory.compass import commitment_reschedules
            records.extend(commitment_reschedules(all_comms))
            # Most urgent first: a block over its limit keeps its first whole lines (memory/blocks.py), so
            # an overdue obligation is never the line held back behind pending ones.
            all_comms.sort(key=lambda c: 0 if c.get("status") == "overdue" or c.get("id") in _overdue_ids else 1)
            if all_comms:
                from colonymind.commitments.work import CommitmentWork
                reservations = {}
                reservations_available = True
                try:
                    reservations = await producers.read(lambda: CommitmentWork(_commitment_store).for_commitments(
                        [c['id'] for c in all_comms], contact_id=contact_id))
                except Exception:
                    reservations_available = False
                    logger.debug('commitment reservation view unavailable', exc_info=True)
                lines = ["Open commitments (a live reservation held by another session is that session's work):"]
                for c in all_comms:
                    status_tag = "[OVERDUE]" if c.get("status") == "overdue" or c['id'] in _overdue_ids else "[pending]"
                    due = _commitment_due(c)
                    reservation = reservations.get(c['id'])
                    work_tag = ('; work=' + reservation['work_state']
                                + ('' if _canonical_only else '; session=' + reservation.get('session_id', ''))) if reservation else ('; work=unclaimed' if reservations_available else '; work=unknown')
                    lines.append(f"- {status_tag} id={c['id']}; {c.get('description', '')}{due}{work_tag}")
                sections.append(ContextSection(
                    id="colonymind-commitments",
                    title="Pending Commitments",
                    body="\n".join(lines),
                    priority=72,
                ))
        except Exception as exc:
            logger.warning("context_assemble commitments failed: %s", exc)

    # --- Source-backed appraisals and contact-specific decision guidance ---
    if contact_id:
        try:
            from colonymind.api.routers.social_state import appraisal_context
            from colonymind.api.viewer import authorized_viewer
            person, _ = authorized_viewer(request, contact_id)
            social_brief, social_refs = await producers.read(appraisal_context, contact_id=person,
                session_id=body.context.session_id, query=query_text, default=("", []))
            if social_brief:
                sections.append(ContextSection(id='colonymind-appraisals',
                    title='Relevant working perspective', body=social_brief,
                    priority=85, citations=social_refs))
        except HTTPException:
            pass
        except Exception:
            logger.debug('appraisal context unavailable', exc_info=True)

    # --- Recorded views (opinions): any viewer, audience-filtered (architecture 4.4) ---
    stance_text = await producers.read(_mind_stances, query_text, default="", viewer_contact_id=viewer_person_id or contact_id or '',
                                viewer_is_owner=_viewer_is_owner, session_id=body.context.session_id)
    if stance_text:
        sections.append(ContextSection(id='colonymind-stances', title='Your recorded views',
                                       body=stance_text, priority=87))
    # --- What the mind learned (lessons): the owner's own turn only, at most one (architecture 4.8) ---
    lesson_text = await _mind_lessons(query_text, viewer_is_owner=_viewer_is_owner, owner_turn=_owner_turn,
                                      session_id=body.context.session_id,
                                      task_run=bool((body.context.metadata or {}).get("kanban_task")))
    if lesson_text:
        sections.append(ContextSection(id='colonymind-lessons', title='What you learned',
                                       body=lesson_text, priority=86))
    standing_text, standing_last = await _mind_standing(query_text, viewer_is_owner=_viewer_is_owner,
        owner_turn=_owner_turn, session_id=body.context.session_id, turn_id=body.context.turn_id,
        task_run=bool((body.context.metadata or {}).get("kanban_task")))
    if standing_text:
        sections.append(ContextSection(id='colonymind-standing-lessons', title='Standing lessons',
                                       body=standing_text, priority=88))
        object.__setattr__(body, '_standing_last', standing_last)

    if contact_id:
        try:
            from colonymind.api.viewer import authorized_viewer
            person, owner = authorized_viewer(request, contact_id)
            if owner:
                from colonymind.api.routers.social_state import waiting_context
                waiting_brief = await producers.read(waiting_context, person, default="")
                if waiting_brief:
                    sections.append(ContextSection(id='colonymind-waiting', title='Expected replies',
                        body=waiting_brief, priority=74))
                # --- Your status: the posture, from the running mind, on the owner's own turn. Its place is
                # held here; it is rendered once this turn's recall has answered (the recall is one of its
                # probes, and it runs while the other lanes are assembled) ---
                from colonymind.mind import status as status_block
                status_slot = ContextSection(id=status_block.SECTION_ID, title=status_block.TITLE, body="",
                                             priority=status_block.PRIORITY)
                sections.append(status_slot)
                # --- The Mind section (architecture 3.1): at most 600 characters, owner only ---
                mind_text = await producers.read(_mind_section, default="")
                if mind_text:
                    sections.append(ContextSection(id='colonymind-mind', title='Mind', body=mind_text, priority=77))
                # The mind's open questions about this turn's subject: statements that still disagree.
                questions = await producers.read(_mind_open_questions, query_text, default="")
                if questions:
                    sections.append(ContextSection(id='colonymind-open-questions', title='Open questions',
                                                   body=questions, priority=78))
        except HTTPException:
            pass
        except Exception:
            logger.debug('owner context unavailable', exc_info=True)

    # Conversation state uses source-cited appraisals and working judgments.
    # Legacy numeric mood estimates remain inspectable through their explicit API.

    # --- Recorded relationship context ---
    if _exact_person_allowed and _contacts_store is not None and contact_id:
        try:
            _rc = await _contacts_store.get(contact_id)
            if _rc is not None:
                _rt = getattr(_rc, "trust_tier", "") or ""
                _count = getattr(_rc, "interaction_count", None)
                _bits = [f"Recorded interactions: {_count}" if _count is not None
                         else "Interaction count not recorded"]
                if _rt:
                    _bits.append(f"recorded contact tier: {_rt.replace('_', ' ')}")
                _rl = getattr(_rc, "last_interaction_at", None)
                if _rl:
                    _bits.append(f"last recorded interaction: {str(_rl)[:10]}")
                sections.append(ContextSection(
                    id="colonymind-relationship",
                    title="Recorded relationship context",
                    body=" · ".join(_bits),
                    priority=86,
                ))
        except Exception as exc:
            logger.debug("context_assemble relationship failed: %s", exc)

    # --- About this person: their digest, for any viewer but the owner; the people faculty's ---
    # At most the template digest's length on every turn, whichever writer stored it, and only while every
    # claim it cites is still live (one query by id; a digest that cites none needs no read).
    from colonymind.api.routers.mind import faculty_on
    if _contacts_store is not None and contact_id and contact_id != owner_id and faculty_on("people"):
        try:
            from colonymind.contacts.digest import MAX_CHARS as _DIGEST_CHARS, checked_digest, source_ledger
            _record = await _contacts_store.get(contact_id)
            _digest, _ = await checked_digest(_record, store=_contacts_store, ledger=source_ledger()) \
                if _record is not None else ("", "none")
            if _digest:
                sections.append(ContextSection(
                    id="colonymind-person", title="About this person",
                    body=_digest if len(_digest) <= _DIGEST_CHARS else _digest[:_DIGEST_CHARS - 1] + "…",
                    priority=84))
        except Exception as exc:
            logger.debug("context_assemble person digest failed: %s", exc)

    # --- The agent's pages: on this turn's person, and on itself (colonymind.pages) ---
    try:
        from colonymind.pages import turn as page_turn
        sections.extend(ContextSection(**row) for row in await producers.read(
            page_turn.sections, contact_id, people=faculty_on("people"), group=group,
            session_id=body.context.session_id, default=[]))
    except Exception as exc:
        logger.debug("context_assemble pages failed: %s", exc)

    # --- Owner's stated preferences (explicit directives the owner gave me) ---
    if _exact_person_allowed and _preference_learner is not None\
            and contact_id:
        try:
            from colonymind.identity import get_owner_contact_id
            if get_owner_contact_id() == contact_id:
                perspective = getattr(_preference_learner, 'perspective', None)
                sections.extend(await producers.read(_preference_sections, _preference_learner,
                    contact_id=contact_id, session_id=body.context.session_id,
                    query=query_text if _viewer_is_owner else "", default=[]))
        except Exception as exc:
            logger.debug("context_assemble owner preferences failed: %s", exc)

    # --- Communication landscape (cross-channel awareness) ---
    if _exact_person_allowed and _comms_log is not None\
            and _contacts_store is not None and contact_id:
        try:
            _lc = await _contacts_store.get(contact_id)
            if _lc is not None:
                _bits = []
                _per = await producers.read(_comms_log.last_per_channel, contact_id)
                if _per:
                    from colonymind.contacts.continuity import recorded_channels
                    _chs = recorded_channels(_per)
                    _bits.append(f"Recorded channels: {_chs}.")
                _lo = await producers.read(_comms_log.last_outbound, contact_id)
                if _lo:
                    from colonymind.contacts.continuity import platform_name
                    _bits.append(
                        f"Last recorded outgoing message: {platform_name(_lo['channel'])} on {str(_lo['ts'])[:10]}. "
                        "This summary includes conversation replies; it does not establish "
                        "proactive outreach or delivery."
                    )
                if _commitment_store is not None:
                    try:
                        _cm = await producers.read(_commitment_store.list, person_id=contact_id, status=["pending", "overdue"], limit=5, default=[])
                        _cm_items = _cm.get("commitments", []) if isinstance(_cm, dict) else (_cm or [])
                        if _cm_items:
                            _descs = [c.get("description", "") for c in _cm_items if c.get("description")]
                            if _descs:
                                _bits.append("Open follow-ups: " + "; ".join(_descs[:3]) + ".")
                    except Exception:
                        pass
                if _bits:
                    from colonymind.identity import get_owner_contact_id
                    _is_owner = (get_owner_contact_id() == contact_id)
                    if not _is_owner:
                        _bits.append(
                            "Proactively reaching out to them needs owner "
                            "approval first."
                        )
                    sections.append(ContextSection(
                        id="colonymind-comms-landscape",
                        title="Communication landscape",
                        body=" ".join(_bits),
                        priority=83,
                    ))
        except Exception as exc:
            logger.debug("context_assemble comms landscape failed: %s", exc)

    # The other lanes are assembled: a rerank answer that arrives after recall's deadline is no longer free.
    from colonymind.memory import turn_budget
    if turn_budget.current() is not None:
        turn_budget.current().lanes_assembled()
    memory = await memory_task if memory_task is not None else None
    sections = [memory if section is memory_slot else section for section in sections
                if section is not memory_slot or memory is not None]
    if group is not None:
        # The clock, the one line that says recall was narrowed, then what was read from the group's own history.
        return sorted([ContextSection(**group_chats.NOTICE),
                       *(section for section in sections if section.id in group_chats.KEPT)],
                      key=lambda section: -(section.priority or 0))
    if status_slot is not None:
        status_text = await _status_section(body, recall=recall_semantic)
        status_section = (ContextSection(id=status_slot.id, title=status_slot.title, body=status_text,
                                         priority=status_slot.priority) if status_text else None)
        sections = [status_section if section is status_slot else section for section in sections
                    if section is not status_slot or status_section is not None]

    # The superseded-value annotation ships off until its review closes.
    from colonymind.memory.compass import records_annotation_enabled
    annotate = records_annotation_enabled()
    if annotate and contact_id and (Path(get_state_dir()) / "turn-idempotency.db").exists():
        try:
            from colonymind.memory.compass import claim_supersessions
            from colonymind.turns import get_turn_idempotency_ledger
            records.extend(await producers.read(
                claim_supersessions, get_turn_idempotency_ledger(get_state_dir()),
                contact_id=contact_id, session_id=body.context.session_id, default=[]))
        except Exception:
            logger.debug("superseded claim values unavailable", exc_info=True)
    if records and annotate:
        from colonymind.memory.compass import annotate_superseded
        sections = annotate_superseded(sections, records)
        if superseded is not None:
            superseded.extend(records)
    return sections


async def _memory_within_allowance(sections: list[ContextSection], recalled, *, inline_chars: Optional[int],
                                   records=(), trace) -> tuple[list[ContextSection], list]:
    """The recalled section within the host's inline allowance (None: the host names none) less the items kept whole
    beside it, and the records it shows.

    Recall packs within its own budget (``COLONYMIND_RECALL_CONTEXT_MAX_CHARS``) while the other lanes are assembled,
    and a turn's packet has no count limit behind a verified cutoff, so it can outgrow what the host puts inline.
    Once the context is assembled, the corrections to earlier context and the superseded-value annotations
    included, the pinned items the per-turn selection keeps whole (``memory.compass``: the owner-critical sections
    and the commitments due soon) are counted as the selection counts them, and when the memory and they exceed the
    allowance, what recall's packer was offered (``selection.keep_offered``), checked again as recall checks its
    packet, is packed again within what they leave, in rank order (``recall.fit_packet``): a record recall's budget
    left out behind one that now shrinks takes the room it frees, and a record of the packet that no longer fits is
    left out as ``over_host_allowance``. The packet is annotated as the assembled memory was (``records``), and the
    annotated packet is what has to fit. When not one record fits beside them, the section is held back whole: it
    is left out, and the caller names it (``_fit_sections``) and the trace says ``held_back``, so the memory never
    vanishes unmentioned and never passes as a heading without evidence."""
    from colonymind.memory import curated
    async def read():
        return await _read_memory_within_allowance(sections, recalled, inline_chars=inline_chars, records=records, trace=trace)
    return await curated.strict_withdrawal_fenced_read(getattr(recalled, 'ledger', None), read, empty=([s for s in sections if s.id != 'colonymind-memory'], []))


async def _read_memory_within_allowance(sections: list[ContextSection], recalled, *, inline_chars: Optional[int],
                                   records=(), trace) -> tuple[list[ContextSection], list]:
    from colonymind.memory import blocks, compass, producers
    from colonymind.memory.recall import fit_packet, packet_refs
    from colonymind.memory.selection import Offered
    packet = recalled.packet
    index = next((i for i, section in enumerate(sections) if section.id == compass.RECALLED), None)
    if index is None:
        return sections, []
    memory, others = sections[index], sections[:index] + sections[index + 1:]
    offered = recalled.offered or Offered(tuple(packet.selected), int(blocks.settings()["recall_items"]),
                                          len(packet.content))
    pool = list(offered.rows)
    if recalled.ledger is not None:
        # The same check recall makes before it returns its packet: a correction that changed while it ranked.
        from colonymind.turns.source_annotations import current_candidates
        pool = await producers.read(current_candidates, recalled.ledger, pool, contact_id=recalled.contact_id,
                                       session_id=recalled.session_id, default=[])
        from colonymind.memory.curated import strict_current_candidates as curated_current
        from colonymind.memory.curated_health import recalling
        with recalling(getattr(recalled, 'curated_recall', None)):
            pool = await producers.read(curated_current, recalled.ledger, pool,
                                        contact_id=recalled.contact_id, default=[])
        if len(pool) != len(offered.rows):
            trace.rechecked(list(offered.rows), pool)

    # A source tombstone may commit after selection even when no smaller host
    # allowance was requested. Recheck the entire offer before either early return.
    if inline_chars is None:
        room = max(len(memory.body), offered.max_chars)
    else:
        _, counted = await compass.select_context(others, "", None)
        room = inline_chars - int(counted.get("kept_whole") or 0) - blocks.cost(memory.title or memory.id, "")
    if len(pool) == len(offered.rows) and (inline_chars is None or len(memory.body) <= room):
        return sections, packet.selected

    def annotated(content: str) -> str:
        return compass.annotate_superseded([memory.model_copy(update={"body": content})], list(records))[0].body \
            if records and content else content

    # A note grows the lines it annotates: pack again within less until the annotated packet fits (each round
    # packs within fewer characters than the last packet took).
    budget = min(offered.max_chars, room)
    while True:
        kept, content = fit_packet(pool, limit=offered.limit, max_chars=budget, recalled=packet.selected)
        over = len(annotated(content)) - room
        if not content or over <= 0:
            break
        budget = min(budget, len(content)) - over
    if not content:
        # Not one record fits: the section is held back whole, citations included, and the caller names it. Its
        # heading and the packet's first line alone would carry no evidence (the block fit keeps a head of any block).
        return [*sections[:index], *sections[index + 1:]], []
    if trace:
        # The same packing again, told to the trace: what it left out, and the packet the context now carries.
        kept, content = fit_packet(pool, limit=offered.limit, max_chars=budget, recalled=packet.selected, trace=trace)
        trace.packet(kept, content)
    refs = await producers.read(packet_refs, recalled.ledger, kept, contact_id=recalled.contact_id,
                                   session_id=recalled.session_id)
    return [*sections[:index], memory.model_copy(update={"body": annotated(content), "citations": refs or None}),
            *sections[index + 1:]], kept


async def assemble_packet(contact_id: str, *, query: str = "", limit_chars: int = 2000) -> str:
    """The recipient-scoped packet the mind composes a message to ``contact_id`` from.

    It is what /context/assemble gives that contact as its own viewer (for
    anyone but the owner: their own recall, the commitments their sources
    prove shared, their digest; never an owner-only section), rendered the
    way the memory provider renders sections, most important first, and fitted
    to ``limit_chars`` as resident blocks (never cut mid-text; what does not fit
    is named).
    """
    contact_id = str(contact_id or "").strip()
    if not contact_id or limit_chars <= 0:
        return ""
    body = ContextAssembleRequest(
        identity=HostIdentity(host_id="colonymind-mind"),
        context=HostTurnContext(session_id=f"mind:{contact_id}", contact_id=contact_id),
        incoming_message=HostMessage(role="user", content=query or ""),
        include_initiatives=False,
    )
    from colonymind.memory import turn_budget
    budget = turn_budget.TurnBudget.for_host()
    writes = []
    with turn_budget.turn(budget):
        try:
            async with asyncio.timeout(budget.assembly_left()):
                sections = await _assemble_sections(body, viewer_person_id=contact_id)
                sections.sort(key=lambda section: -(section.priority or 0))
                fitted, _ = _fit_sections(sections, budget=limit_chars, where="recipient packet")
                text = "\n\n".join(f"## {section.title or section.id}\n{section.body}" for section in fitted)
            if budget.assembly_left() > 0:
                writes = turn_budget.take_writes(fitted)
            else:
                text = ""
        except TimeoutError:
            text = ""
    turn_budget.schedule_writes(writes)
    return text


async def claims_for(contact_id: str, limit: int = 8) -> list[tuple[str, str]]:
    """The contact's own current source claims, newest first, as ``(claim id, "predicate: value")``.

    The digest's "what they have told me" lines and the ids the template digest cites for them: only
    claims grounded in that contact's person-scoped sources, never superseded, retracted, expired or
    from a source whose projections were erased. A line said by several claims is listed once, under
    the newest.
    """
    contact_id = str(contact_id or "").strip()
    if not contact_id or limit <= 0 or not (Path(get_state_dir()) / "turn-idempotency.db").exists():
        return []
    from contextlib import closing
    from colonymind.beliefs.source_projection import SourceClaimProjection
    from colonymind.turns import get_turn_idempotency_ledger

    def read() -> list[tuple[str, str]]:
        projection = SourceClaimProjection(get_turn_idempotency_ledger(get_state_dir()))
        now = _temporal_now().isoformat()
        with closing(projection.ledger._connect()) as conn:
            rows = projection._rows(conn, contact_id, "", distinct_values=True, limit=limit * 4)
            from colonymind.memory.learning import cold_ids
            cold = cold_ids(conn, "claim", [row["id"] for row in rows])
            erased = {row[0] for row in conn.execute(
                "SELECT turn_id FROM source_projection_erasures WHERE turn_id IN ("
                + ",".join("?" for _ in rows) + ")", [row["turn_id"] for row in rows])} if rows else set()
        lines: dict[str, str] = {}
        for row in rows:
            value = str(row.get("value") or "").strip()
            if (row.get("superseded_by") or row.get("retracted_by") or row["turn_id"] in erased or row["id"] in cold
                    or not value or (row.get("valid_to") and row["valid_to"] <= now)):
                continue
            lines.setdefault(f"{row['predicate']}: {value[:160]}", str(row["id"]))
            if len(lines) >= limit:
                break
        return [(ident, line) for line, ident in lines.items()]

    return await asyncio.to_thread(read)


# ---------------------------------------------------------------------------
# Turns
# ---------------------------------------------------------------------------


class SourceAnnotationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    contact_id: str = Field(min_length=1, max_length=256)
    session_id: str = Field(min_length=1, max_length=256)
    annotation_id: str = Field(min_length=1, max_length=128)
    source_id: str = Field(min_length=1, max_length=256)
    source_version: str = Field(pattern='^[0-9a-f]{64}$')
    excerpt: str = Field(min_length=1, max_length=4096)
    correction: str = Field(min_length=1, max_length=4096)


class SourceDeadlineRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    contact_id: str = Field(min_length=1, max_length=256)
    session_id: str = Field(min_length=1, max_length=256)
    source_id: str = Field(min_length=1, max_length=256)
    source_version: str = Field(pattern='^[0-9a-f]{64}$')
    claim_id: str = Field(min_length=1, max_length=256)
    timezone_name: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode='after')
    def valid_timezone(self):
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
        if self.timezone_name is None:
            return self
        try:
            ZoneInfo(self.timezone_name)
        except (ValueError, ZoneInfoNotFoundError) as exc:
            raise ValueError('Unknown timezone') from exc
        return self


@router.post('/memory/sources/deadline')
async def read_source_deadline(body: SourceDeadlineRequest, request: Request):
    authority = request_authority(request)
    if not authority.authenticated or authority.anonymous:
        raise HTTPException(status_code=403, detail={'code': 'source_deadline_not_authorized'})
    person = resolve_request_person(request, claimed_person_id=body.contact_id)
    from colonymind.turns import get_turn_idempotency_ledger
    from colonymind.beliefs.source_projection import SourceClaimProjection
    from colonymind.util.temporal import resolve_communication_timezone
    contact_tz = None
    if body.timezone_name is None and _contacts_store is not None:
        contact = await _contacts_store.get(person)
        contact_tz = getattr(contact, 'timezone', None)
    zone = resolve_communication_timezone(contact_tz, body.timezone_name)
    projection = SourceClaimProjection(get_turn_idempotency_ledger(get_state_dir()))
    return projection.deadline(**{**body.model_dump(), 'contact_id': person,
        'timezone_name': zone, 'timezone_basis': 'caller_override' if body.timezone_name else 'communication_frame'})


@router.post('/memory/sources/annotations', tags=[PUBLIC],
             dependencies=[Depends(refuse_hand_session)])
def append_source_annotation(body: SourceAnnotationRequest, request: Request):
    authority = request_authority(request)
    if not authority.authenticated or authority.anonymous:
        raise HTTPException(status_code=403, detail={'code': 'source_annotation_not_authorized'})
    person = resolve_request_person(request, claimed_person_id=body.contact_id)
    from colonymind.turns import get_turn_idempotency_ledger
    try:
        return get_turn_idempotency_ledger(get_state_dir()).append_source_annotation(
            **{**body.model_dump(), 'contact_id': person}, author_principal=authority.principal_id)
    except ValueError as exc:
        code = str(exc)
        if code not in {'source_erased', 'source_not_found', 'source_version_mismatch',
                        'source_excerpt_mismatch', 'annotation_id_conflict'}:
            code = 'invalid_source_annotation'
        raise HTTPException(status_code=409 if code.endswith(('conflict', 'mismatch')) else 422,
                            detail={'code': code}) from exc


@router.get("/memory/sources/erasures")
def source_erasure_feed(contact_id: str, after: int = Query(0, ge=0), request: Request = None):
    person = resolve_request_person(request, claimed_person_id=contact_id) or contact_id
    from colonymind.turns import get_turn_idempotency_ledger
    try:
        return get_turn_idempotency_ledger(get_state_dir()).erasure_feed(person, after)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail={"code": "erasure_history_mismatch"}) from exc


class NativeHistoryReference(BaseModel):
    model_config = ConfigDict(extra='forbid')
    session_id: str = Field(min_length=1, max_length=256)
    message_hash: str = Field(pattern=r'^[0-9a-f]{64}$')


class SourceFreshnessRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    contact_id: str = Field(min_length=1, max_length=256)
    session_id: str = Field(min_length=1, max_length=256)
    after: int = Field(default=0, ge=0)
    source_refs: list[SourceReference] = Field(max_length=512)
    unannotated_input_refs: list[SourceInputReference] = Field(default_factory=list, max_length=512)
    native_history_refs: list[NativeHistoryReference] = Field(default_factory=list, max_length=512)
    annotation_checks: list[SourceAnnotationCheck] = Field(default_factory=list, max_length=512)

    @model_validator(mode='after')
    def exact_source_selector(self):
        if not self.source_refs and not self.unannotated_input_refs and not self.native_history_refs:
            raise ValueError('Exact source revisions or captured inputs are required')
        return self


@router.post('/memory/sources/erasures')
def source_freshness_feed(body: SourceFreshnessRequest, request: Request):
    """The existing feed plus current ownership of the exact supplied sources.

    Corrections change attribution without erasing source bytes or changing
    their content digest. Check canonical ownership and descendant validity;
    an erasure watermark alone cannot certify cached recall after a correction.
    Optional exact input membership also rechecks that an operational excerpt
    has no applicable annotation, including a first note added after selection.
    Exact current input revisions are returned separately so a host that only
    retained the original input hash can pin the canonical media revision, then
    repeat the strict supplied-version check. Missing expected versions never
    make sources_current true. POST adds no persisted state.
    """
    person = resolve_request_person(request, claimed_person_id=body.contact_id) or body.contact_id
    from colonymind.turns import get_turn_idempotency_ledger
    ledger = get_turn_idempotency_ledger(get_state_dir())
    from colonymind.memory import curated
    page = curated.fenced_read(ledger, lambda: _source_freshness_feed(body, person, ledger), empty=None)
    if page is None:
        raise HTTPException(status_code=409, detail={'code': 'erasure_during_check'})
    return page


def _source_freshness_feed(body, person, ledger):
    try:
        from colonymind.memory import curated
        page = curated.freshness_feed(ledger, person, body.after)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail={'code': 'erasure_history_mismatch'}) from exc
    expected = {(ref.source_id, ref.source_version) for ref in body.source_refs}
    current = ledger.source_references([ref.source_id for ref in body.source_refs],
                                      contact_id=person, session_id=body.session_id)
    page['sources_current'] = (expected == {(ref['source_id'], ref['source_version']) for ref in current}
                               and not curated.is_forgotten(ledger, current))
    if body.annotation_checks:
        from colonymind.turns.source_annotations import current_candidates
        checks = [check.model_dump() for check in body.annotation_checks]
        candidates = [{'_annotation_source_refs': check['source_refs'],
                       '_annotation_message_hashes': check['message_hashes'],
                       '_annotation_ids': check['annotation_ids']} for check in checks]
        retained = current_candidates(ledger, candidates, contact_id=person, session_id=body.session_id)
        # Keep byte/ownership validity separate from each search's selected
        # correction set. A stale earlier result must not poison a fresh search
        # later in the same native turn.
        page['annotation_checks_current'] = [candidate in retained and
            {(ref['source_id'], ref['source_version']) for ref in check['source_refs']} <= expected
            for check, candidate in zip(checks, candidates)]
    if body.native_history_refs:
        from colonymind.turns.history_references import resolve
        page['native_history_matches'] = resolve(ledger, contact_id=person,
            session_id=body.session_id,
            references=[ref.model_dump() for ref in body.native_history_refs])
    if body.unannotated_input_refs:
        from colonymind.turns.source_annotations import inputs_unannotated
        refs = [ref.model_dump() for ref in body.unannotated_input_refs]
        page['input_source_refs'] = []
        try:
            inputs = ledger.resolve_input_dependencies(contact_id=person,
                session_id=body.session_id, refs=refs)
        except ValueError:
            page['sources_current'] = False
        else:
            versions = {(ref['source_id'], ref['source_version']) for ref in inputs}
            current_inputs = ledger.source_references([ref['source_id'] for ref in inputs],
                contact_id=person, session_id=body.session_id)
            inputs_current = (versions == {(ref['source_id'], ref['source_version']) for ref in current_inputs}
                              and inputs_unannotated(ledger, refs))
            if inputs_current:
                page['input_source_refs'] = current_inputs
            page['sources_current'] &= inputs_current and versions <= expected
    if body.native_history_refs:
        for match in page['native_history_matches']:
            if match.get('erased') or curated.is_forgotten(ledger, match):
                page['sources_current'] = False
    return page


@router.get("/memory/sources/claims/status")
def source_claim_projection_status(contact_id: str, request: Request = None):
    person = resolve_request_person(request, claimed_person_id=contact_id) or contact_id
    from colonymind.turns import get_turn_idempotency_ledger
    from colonymind.beliefs.source_projection import SourceClaimProjection
    from colonymind.turns.media import SourceMedia
    ledger = get_turn_idempotency_ledger(get_state_dir())
    from colonymind.turns.source_vectors import SourceVectors
    from colonymind.vector import get_store, get_pipeline
    return {"sources": SourceClaimProjection(ledger).status(person), "media": SourceMedia(ledger).status(person),
            "semantic": SourceVectors(ledger, get_store(), get_pipeline()).status(person)}


@router.get("/memory/sources/assets/{asset_hash}")
def read_source_asset(asset_hash: str, contact_id: str, session_id: str, request: Request = None):
    person = resolve_request_person(request, claimed_person_id=contact_id) or contact_id
    from colonymind.turns import get_turn_idempotency_ledger
    from colonymind.turns.media import SourceMedia
    try:
        data, mime = SourceMedia(get_turn_idempotency_ledger(get_state_dir())).read(
            asset_hash, contact_id=person, session_id=session_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="unknown source asset") from None
    return Response(content=data, media_type=mime, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})


DERIVED_SESSION_PREFIX = "derived:"


def _derive_missing_session(body: TurnSyncRequest, request: Request | None) -> None:
    """Give a turn sent without a session one derived from its caller and itself.

    A caller that observes work outside any conversation (a voice gateway posting a
    dispatched task's result) sends ``session_id: ""``; its source was refused. The turn
    becomes its own session: a digest of the caller's principal and the envelope, so a
    retry of the same turn lands on the same session, source and reservation (replays
    stay replays, changed content under a turn ID stays a conflict), and two principals
    never share one. A checkpoint is a session's own transcript and an input-linked
    answer resolves its parents in the caller's session: both still need the caller's.
    """
    if body.context.session_id.strip() or body.checkpoint_messages is not None or body.assistant_input_refs:
        return
    from colonymind.api.auth import request_authority
    from colonymind.turns import canonical_turn_digest
    body.context.session_id = ""
    turn = body.model_dump(mode="json")
    if turn.get("work") is None:
        turn.pop("work", None)
    basis = canonical_turn_digest({"principal": request_authority(request).principal_id,
                                   "turn": turn})
    body.context.session_id = DERIVED_SESSION_PREFIX + basis[:32]


async def _secret_scope(body: TurnSyncRequest) -> str:
    """Whose the secrets in a turn are (the vault scope they are sealed in): the sender's contact when the
    envelope names a sender, read and never created, and the sender's handle when no contact has it (never the
    contact the client claimed, which goes stale in a shared session); the envelope's contact otherwise."""
    sender = body.sender
    address = str(getattr(sender, "user_id", "") or "").strip()
    if sender is None or not address:
        return body.context.contact_id
    platform = str(sender.platform or "").strip()
    if _contacts_store is not None and hasattr(_contacts_store, "resolve_handle"):
        try:
            contact = await _contacts_store.resolve_handle(platform, address)
        except Exception:
            contact = None
        if getattr(contact, "contact_id", None):
            return str(contact.contact_id)
    return f"handle:{platform.lower()}:{address}"


def _vault_unavailable() -> HTTPException:
    return HTTPException(status_code=503, headers={"Retry-After": "30"}, detail={
        "code": "secrets_vault_unavailable",
        "message": "an erasure in this session cannot be checked while the sealed store's key is unavailable; "
                   "send the turn again later"})


def _erasure_waits(ledger, body: TurnSyncRequest, turn_seal) -> bool:
    """Whether this turn must wait before anything is reserved: an exact erasure names messages of its session,
    and the form the erasure would compare cannot be worked out now (a value of the turn was withheld because
    the store's key was unavailable, or the turn carries a reference and the key is unavailable)."""
    from colonymind.secrets.detect import REFERENCE
    if not ledger.has_exact_erasures(session_id=body.context.session_id):
        return False
    if turn_seal.stage.withheld:
        return True
    texts = [getattr(message, "content", None) for message in
             [body.user_message, body.assistant_message, *(body.checkpoint_messages or [])] if message is not None]
    carries = any(REFERENCE.search(json.dumps(content, ensure_ascii=False, default=str)) for content in texts)
    return carries and not ledger.vault.available


def _check_turn_ids(*turn_ids: str | None) -> None:
    from colonymind.turns.observed import is_observed_source
    if any(is_observed_source((identifier or '').strip()) for identifier in turn_ids):
        raise HTTPException(422, detail={'code': 'reserved_turn_id',
                            'message': 'observed-conversation: is reserved for observed conversation capture'})


# One gate per event loop: one turn sync at a time runs its storage steps (``_TurnSyncHold``).
_turn_sync_gates: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


async def _to_its_end(awaitable):
    """Wait for ``awaitable`` to end even when the request is cancelled meanwhile.

    A worker thread cannot be stopped, and what follows a step must not run before the step has ended.
    A cancellation that arrives during the wait is kept and lands at the caller's next suspension, where
    one that arrived a moment later would have landed.
    """
    waited = asyncio.ensure_future(awaitable)
    task, kept = asyncio.current_task(), None
    with anyio.CancelScope(shield=True):
        while not waited.done():
            try:
                await asyncio.wait([waited])
            except asyncio.CancelledError as cancelled:
                task.uncancel()
                kept = cancelled
    if kept is not None:
        task.cancel(*kept.args[:1])
    return waited.result()


class _TurnSyncHold:
    """One turn sync's hold on the gate that keeps the storage steps of turn syncs apart.

    The steps between two awaits of a turn sync used to run on the loop with nothing in between. The
    blocking ones now run in a worker thread (``off``) and the loop answers other requests meanwhile, so
    the gate is what keeps another turn sync's steps out: it is held from the first step of such a
    stretch to its last and is free only where a turn sync awaited before (``away``).
    """

    def __init__(self):
        loop = asyncio.get_running_loop()
        gate = _turn_sync_gates.get(loop)
        if gate is None:
            gate = _turn_sync_gates[loop] = asyncio.Lock()
        self._gate = gate
        self.held = False

    async def take(self):
        """Wait for the gate, unless this turn sync holds it."""
        if not self.held:
            await _to_its_end(self._gate.acquire())
            self.held = True

    def give(self):
        if self.held:
            self.held = False
            self._gate.release()

    async def off(self, step, *args, **kwargs):
        """Run one blocking step of a stretch in a worker thread, to its end; the gate stays held."""
        return await _to_its_end(asyncio.to_thread(step, *args, **kwargs))

    async def away(self, awaitable):
        """Await what a turn sync awaited before: the gate is free meanwhile and held after."""
        self.give()
        try:
            return await awaitable
        finally:
            await self.take()


async def _ingest_turn_idempotently(
    body: TurnSyncRequest,
    request: Request | None = None,
    *,
    resolved_sender_contact_id: str | None = None,
    hold: _TurnSyncHold,
) -> tuple[TurnSyncResponse, str]:
    """Run one turn's effects, or replay its durable result.

    Legacy callers without a turn ID retain the v1 behavior. A supplied ID is
    reserved before attribution, memory, cognition, journal, or relationship
    effects run. This is the server's final defense even when a host retries or
    two host integrations accidentally submit the same envelope.

    Detected secrets are sealed first (``colonymind.secrets.seal``), in the
    scope of the person whose message carried them (``_secret_scope``): the
    derived session, the envelope digest, the reservation and every effect
    see references only (a digest over a short value is a way back to it).
    The values are staged and reach the vault only when the turn is admitted
    (an erased, conflicting or replayed turn stores none). The raw forms stay
    in memory for the ledger's erasure check and are never stored.

    ``resolved_sender_contact_id`` is the contact ``_attribute_turn`` found the
    sender to be; the turn is attributed to exactly that contact.

    ``hold`` is the caller's hold on the turn sync gate. It is taken here and is held
    when this returns or raises: the caller's own steps end the last stretch.
    """
    _check_turn_ids(body.context.turn_id)
    if body.work is not None and body.work.root_turn_id != (body.context.turn_id or '').strip():
        raise HTTPException(422, detail={'code': 'work_owner_root_mismatch'})
    from colonymind.secrets.seal import redact_sensitive_turn
    # The first stretch begins after this await: the gate is held from there.
    turn_seal = await hold.off(redact_sensitive_turn, body, vault=_secrets_vault,
                               contact_id=await hold.away(_secret_scope(body)))
    _derive_missing_session(body, request)
    turn_id = (body.context.turn_id or "").strip()
    if body.work is not None and not turn_id:
        raise HTTPException(status_code=422, detail={'code': 'work_turn_id_required'})
    if not turn_id:
        return await _process_turn_sync(body, request=request,
            resolved_sender_contact_id=resolved_sender_contact_id, turn_seal=turn_seal, hold=hold), "unkeyed"
    if len(turn_id) > 256:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_turn_id", "message": "turn_id exceeds 256 characters"},
        )
    body.context.turn_id = turn_id

    from colonymind.turns import (
        ReservationOutcome,
        canonical_turn_digest,
        get_turn_idempotency_ledger,
    )

    digest = canonical_turn_digest(body)
    ledger = get_turn_idempotency_ledger(get_state_dir())
    if await hold.off(ledger.is_source_erased, turn_id, body.context.contact_id):
        return TurnSyncResponse(accepted=False, source_recorded=False, continuity_updated=False, skipped_reason="source_erased", **_work_ack(body, None)), "erased"
    if await hold.off(_erasure_waits, ledger, body, turn_seal):
        # Before anything is reserved: a later retry is processed as new, never answered as ambiguous.
        raise _vault_unavailable()
    if body.assistant_input_refs:
        from colonymind.turns.idempotency import SourceInputPending
        try:
            await hold.off(ledger.resolve_input_dependencies, contact_id=body.context.contact_id,
                session_id=body.context.session_id, turn_id=turn_id,
                refs=[ref.model_dump() for ref in body.assistant_input_refs])
        except SourceInputPending:
            # Ordinary outbox retry, with no reserved or ambiguous effect row.
            return TurnSyncResponse(accepted=False, source_recorded=False, continuity_updated=False,
                                    skipped_reason='source_input_parent_pending'), 'in_progress'
        except ValueError as exc:
            raise HTTPException(422, detail={'code': 'invalid_source_input_dependency'}) from exc
    if body.checkpoint_messages is not None:
        # One atomic source+index commit, no ordinary conversation effects.
        # A retry after an interrupted response can safely repeat this write.
        try:
            created = await hold.off(
                ledger.record_source,
                turn_id, contact_id=body.context.contact_id,
                session_id=body.context.session_id, scope="session",
                messages=[dict(message.model_dump(mode="json"), **(
                    {'_supplied_sources': [ref.model_dump() for ref in body.assistant_source_refs]}
                    if message.role == 'assistant' and body.assistant_source_refs else {}))
                    for message in body.checkpoint_messages],
                occurred_at=(body.context.metadata or {}).get("occurred_at"),
                timezone_name=body.context.timezone,
                channel_id=body.context.channel_id,
                raw_hashes=turn_seal.raw_hashes(body.context.session_id), staged=turn_seal.stage,
            )
        except ValueError as exc:
            from colonymind.turns.idempotency import SourceErased
            if isinstance(exc, SourceErased):
                return TurnSyncResponse(accepted=False, source_recorded=False, continuity_updated=False, skipped_reason="source_erased", **_work_ack(body, None)), "erased"
            raise HTTPException(status_code=409, detail={"code": "checkpoint_source_conflict"}) from exc
        except VaultUnavailable:
            raise _vault_unavailable() from None
        return TurnSyncResponse(
            accepted=True, continuity_updated=False, source_recorded=True,
            skipped_reason="checkpoint_source_only",
        ), "created" if created else "replayed"
    reservation = await hold.off(ledger.reserve, turn_id, digest)
    if reservation.outcome == ReservationOutcome.CONFLICT:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "turn_id_content_conflict",
                "turn_id": turn_id,
                "message": "turn_id is already bound to different canonical content",
            },
        )
    if reservation.outcome == ReservationOutcome.AMBIGUOUS:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "turn_ingestion_ambiguous",
                "turn_id": turn_id,
                "message": "the first attempt was interrupted; inspect before replaying effects",
            },
        )
    if reservation.outcome == ReservationOutcome.REPLAYED:
        if body.work is not None:
            from colonymind.work import schedule_work
            schedule_work(turn_id)
        cached = reservation.response or {
            "accepted": True,
            "continuity_updated": False,
            "skipped_reason": "identical_replay",
        }
        return TurnSyncResponse.model_validate(cached), "replayed"
    if reservation.outcome == ReservationOutcome.IN_PROGRESS:
        return TurnSyncResponse(
            # Truthful pending semantics: the first writer has not committed a
            # result yet and may still fail or become ambiguous. A retry must
            # not look like a completed successful ingestion.
            accepted=False,
            continuity_updated=False,
            skipped_reason="identical_retry_in_progress",
        ), "in_progress"

    # The gate is free until the turn's own steps begin: an identical retry sent meanwhile meets the
    # reservation and is answered as in progress.
    hold.give()
    try:
        result = await _process_turn_sync(body, request=request,
            resolved_sender_contact_id=resolved_sender_contact_id, turn_seal=turn_seal, hold=hold)
    except BaseException as exc:
        await hold.take()
        await hold.off(ledger.mark_ambiguous, turn_id, digest, exc)
        raise
    await hold.take()
    await hold.off(ledger.complete, turn_id, digest, result.model_dump(mode="json"))
    return result, "created"


async def _attribute_turn(body: TurnSyncRequest, request: Request | None) -> str | None:
    """Bind a turn to what its credential may write as, before anything is reserved or stored.

    A credential bound to its own identity writes only as ``credential:<name>`` and names no sender.
    Any other credential may attribute a turn to the owner (the owner, shared or global person, named
    in the body or reached through a sender who is the owner) only from a platform it is granted the
    owner on; a turn that names neither a contact nor a sender is its own. A contact the credential may
    not use is refused with 403 and nothing is stored: never rewritten (docs/CREDENTIALS.md). A turn
    the credential may write that is an operator's test is then kept under the credential's own
    identity, ordinary turn and checkpoint alike. Returns the contact the sender already is, so the turn
    is attributed to exactly what was checked.
    """
    from colonymind.api.auth import (
        _authority_error, contact_not_granted, needs_owner_platform, owner_not_granted, owner_platform_granted,
        person_granted, request_authority, reserved_person_ids,
    )
    from colonymind.api.credentials import INSTANCE_CREDENTIAL, identity_of, is_credential_identity, platform_of
    authority = request_authority(request)
    claimed = (body.context.contact_id or "").strip()
    if authority.contacts == "own":
        if body.sender is not None:
            raise _authority_error("sender_not_granted", f"credential {authority.principal_id} writes only as "
                                   f"{authority.identity}; a sender would make the turn a person's")
        body.context.contact_id = resolve_request_person(request, context_person_id=claimed)
        return None
    # Development mode (no key) names its own default person and never a reserved one
    # (``resolve_request_person``); like every other principal it is checked against its sender below.
    body.context.contact_id = resolve_turn_person(
        request, context_person_id=claimed, has_sender=body.sender is not None) or claimed
    claimed = body.context.contact_id
    if is_credential_identity(claimed) and claimed != authority.identity:
        raise contact_not_granted(authority, claimed)
    if not claimed and body.sender is None and authority.identity:
        body.context.contact_id = authority.identity
        return None
    known = None
    if body.sender is not None and body.checkpoint_messages is None and _contacts_store is not None:
        from colonymind.identity.participants import ParticipantResolver
        known = (await ParticipantResolver(_contacts_store).lookup(
            platform=body.sender.platform, user_id=body.sender.user_id)).contact_id
    platform = platform_of(body.sender.platform if body.sender is not None else None, body.context.channel_id)
    if body.sender is not None and authority.contacts == "list" and body.checkpoint_messages is None and (
            (known is None and not authority.senders_any and _contacts_store is not None)
            or (known is not None and (not authority.senders_any or known in reserved_person_ids())
                and not person_granted(authority, known))):
        # A credential bound to a list of persons writes as a sender's contact only when it lists that
        # contact (or when any sender may be anyone, the owner, shared and global persons still as granted).
        if known is not None and needs_owner_platform(authority, known):
            raise owner_not_granted(authority, platform)
        raise _authority_error("sender_not_granted", f"credential {authority.principal_id} may not write as the "
                               "contact this sender is" + ("" if known else " (a sender it has never met)"))
    if (needs_owner_platform(authority, claimed) or needs_owner_platform(authority, known)) \
            and not owner_platform_granted(authority, platform):
        raise owner_not_granted(authority, platform)
    if _operator_test(body, request):
        # An operator's test through a live channel is never a person's words, even on a platform that
        # speaks for the owner: an ordinary turn and a checkpoint alike are kept under the sending
        # credential's own identity, before either is reserved or stored (docs/CREDENTIALS.md).
        body.context.contact_id = authority.identity or identity_of(INSTANCE_CREDENTIAL)
        body.sender = None      # a sender is a person; a test turn has none
        return None
    return known


def _opening_text(content) -> str:
    """The text a message opens with: a string as it is, else its first text block."""
    if isinstance(content, str):
        return content
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            return block["text"]
    return ""


def _operator_test(body: TurnSyncRequest, request: Request | None) -> bool:
    """Whether a turn is an operator's test by a trusted signal (``is_operator_test``): the request's test
    header, or the exact mark opening its user message (on a checkpoint, any of its user messages)."""
    from colonymind.identity.participants import OPERATOR_TEST_HEADER, is_operator_test
    header = request.headers.get(OPERATOR_TEST_HEADER) if request is not None else None
    if body.checkpoint_messages is not None:
        texts = [_opening_text(message.content) for message in body.checkpoint_messages if message.role == "user"]
    else:
        texts = [_opening_text(body.user_message.content)] if body.user_message is not None else []
    return is_operator_test("", header) or any(is_operator_test(text) for text in texts)


@router.post("/turns/sync", response_model=TurnSyncResponse, tags=[PUBLIC],
             responses=capture_responses(TurnSyncResponse, keyed=False),
             dependencies=[Depends(refuse_hand_session)])
async def turns_sync(
    body: TurnSyncRequest,
    request: Request = None,
    response: Response = None,
) -> TurnSyncResponse:
    _check_turn_ids(body.context.turn_id)
    sender_contact = await _attribute_turn(body, request)
    hold = _TurnSyncHold()
    try:
        result, outcome = await _ingest_turn_idempotently(body, request=request,
                                                          resolved_sender_contact_id=sender_contact, hold=hold)
        from colonymind.api.routers.transport_ingress_api import reconcile_source
        reconcile_source(body, result)
    finally:
        hold.give()
    if response is not None:
        response.headers["Idempotency-Status"] = outcome
        if outcome == "in_progress":
            response.status_code = status.HTTP_202_ACCEPTED
            response.headers["Retry-After"] = "1"
    return result


@v2_router.put('/turns/source-linked/input-parent/{turn_id:path}', response_model=TurnSyncResponse,
             dependencies=[Depends(refuse_hand_session)])
async def source_input_linked_sync(turn_id: str, body: TurnSyncRequest, response: Response, request: Request = None):
    """Older linked-source routes reject this suffix before persisting anything."""
    _check_turn_ids(turn_id, body.context.turn_id)
    if body.context.turn_id == 'source-linked/input-parent/' + turn_id:
        return await turns_sync_v2(body.context.turn_id, body, response, request)
    if not body.assistant_input_refs or body.context.turn_id != turn_id:
        raise HTTPException(422, detail={'code': 'invalid_input_linked_source'})
    return await turns_sync_v2(turn_id, body, response, request)


@v2_router.put('/turns/source-media/audio/{turn_id:path}', response_model=TurnSyncResponse, tags=[PUBLIC],
               responses=capture_responses(TurnSyncResponse, keyed=True),
             dependencies=[Depends(refuse_hand_session)])
async def audio_source_sync(turn_id: str, body: TurnSyncRequest, response: Response, request: Request = None):
    """An older generic turn route must not persist unowned audio bytes."""
    _check_turn_ids(turn_id, body.context.turn_id)
    if body.context.turn_id == 'source-media/audio/' + turn_id:
        return await turns_sync_v2(body.context.turn_id, body, response, request)
    messages = [body.user_message, body.assistant_message, *(body.checkpoint_messages or [])]
    if body.context.turn_id != turn_id or not any(message and isinstance(message.content, list)
            and any(isinstance(block, dict) and block.get('type') == 'input_audio' for block in message.content)
            for message in messages):
        raise HTTPException(422, detail={'code': 'invalid_audio_source'})
    return await turns_sync_v2(turn_id, body, response, request)


@v2_router.put('/turns/source-media/video/{turn_id:path}', response_model=TurnSyncResponse, tags=[PUBLIC],
               responses=capture_responses(TurnSyncResponse, keyed=True),
             dependencies=[Depends(refuse_hand_session)])
async def video_source_sync(turn_id: str, body: TurnSyncRequest, response: Response, request: Request = None):
    """Selected inline MP4 ingestion; predecessors reject unequal path IDs."""
    _check_turn_ids(turn_id, body.context.turn_id)
    if body.context.turn_id == 'source-media/video/' + turn_id:
        return await turns_sync_v2(body.context.turn_id, body, response, request)
    messages = [body.user_message, body.assistant_message, *(body.checkpoint_messages or [])]
    if body.context.turn_id != turn_id or not any(message and isinstance(message.content, list)
            and any(isinstance(block, dict) and block.get('type') == 'input_video' for block in message.content)
            for message in messages):
        raise HTTPException(422, detail={'code': 'invalid_video_source'})
    return await turns_sync_v2(turn_id, body, response, request)


@v2_router.put('/turns/source-media/document/{turn_id:path}', response_model=TurnSyncResponse, tags=[PUBLIC],
               responses=capture_responses(TurnSyncResponse, keyed=True),
             dependencies=[Depends(refuse_hand_session)])
async def document_source_sync(turn_id: str, body: TurnSyncRequest, response: Response, request: Request = None):
    """Explicit inline document ingestion; predecessors reject unequal path IDs."""
    _check_turn_ids(turn_id, body.context.turn_id)
    if body.context.turn_id == 'source-media/document/' + turn_id:
        return await turns_sync_v2(body.context.turn_id, body, response, request)
    messages = [body.user_message, body.assistant_message, *(body.checkpoint_messages or [])]
    if body.context.turn_id != turn_id or not any(message and isinstance(message.content, list)
            and any(isinstance(block, dict) and block.get('type') == 'input_document' for block in message.content)
            for message in messages):
        raise HTTPException(422, detail={'code': 'invalid_document_source'})
    return await turns_sync_v2(turn_id, body, response, request)


@v2_router.put("/turns/source-linked/{turn_id:path}", response_model=TurnSyncResponse,
             dependencies=[Depends(refuse_hand_session)])
async def source_linked_sync(turn_id: str, body: TurnSyncRequest, response: Response, request: Request = None):
    """An old backend must not silently drop a new answer's source references."""
    _check_turn_ids(turn_id, body.context.turn_id)
    if body.context.turn_id == 'source-linked/' + turn_id:
        return await turns_sync_v2(body.context.turn_id, body, response, request)
    if not body.assistant_source_refs or body.context.turn_id != turn_id:
        raise HTTPException(status_code=422, detail={'code': 'invalid_linked_source'})
    return await turns_sync_v2(turn_id, body, response, request)


@v2_router.put("/turns/source-survivors/{turn_id:path}", response_model=TurnSyncResponse,
             dependencies=[Depends(refuse_hand_session)])
async def source_survivor_sync(turn_id: str, body: TurnSyncRequest, response: Response, request: Request = None):
    """Preserve attributed direct survivors without replaying ordinary effects.

    A predecessor matches its generic turn route and rejects the unequal path
    and envelope IDs. Hosts must leave the survivor queued, never fall back.
    """
    _check_turn_ids(turn_id, body.context.turn_id)
    if body.context.turn_id == 'source-survivors/' + turn_id:
        return await turns_sync_v2(body.context.turn_id, body, response, request)
    if body.source_only is not True or body.context.turn_id != turn_id:
        raise HTTPException(status_code=422, detail={'code': 'invalid_source_survivor'})
    return await turns_sync_v2(turn_id, body, response, request)


@v2_router.put("/turns/{turn_id:path}", response_model=TurnSyncResponse, tags=[PUBLIC],
               responses=capture_responses(TurnSyncResponse, keyed=True),
             dependencies=[Depends(refuse_hand_session)])
async def turns_sync_v2(
    turn_id: str,
    body: TurnSyncRequest,
    response: Response,
    request: Request = None,
) -> TurnSyncResponse:
    """Idempotent TurnEnvelopeV2 ingestion compatibility slice.

    The path ID is canonical. Initial acceptance is ``201``; an identical
    replay is ``200``; changed content under the same ID is ``409``.
    """
    _check_turn_ids(turn_id, body.context.turn_id)
    path_turn_id = (turn_id or "").strip()
    if not path_turn_id or len(path_turn_id) > 256:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_turn_id", "message": "turn_id must be 1..256 characters"},
        )
    body_turn_id = (body.context.turn_id or "").strip()
    if body_turn_id and body_turn_id != path_turn_id:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "turn_id_mismatch",
                "path_turn_id": path_turn_id,
                "body_turn_id": body_turn_id,
            },
        )
    body.context.turn_id = path_turn_id
    sender_contact = await _attribute_turn(body, request)
    hold = _TurnSyncHold()
    try:
        result, outcome = await _ingest_turn_idempotently(body, request=request,
                                                          resolved_sender_contact_id=sender_contact, hold=hold)
        from colonymind.api.routers.transport_ingress_api import reconcile_source
        reconcile_source(body, result)
    finally:
        hold.give()
    response.headers["Idempotency-Status"] = outcome
    if outcome == "created":
        response.status_code = status.HTTP_201_CREATED
    elif outcome == "in_progress":
        response.status_code = status.HTTP_202_ACCEPTED
        response.headers["Retry-After"] = "1"
    else:
        response.status_code = status.HTTP_200_OK
    return result


def _observed_refusal(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


def _observed_owner(request: Request | None) -> str:
    """The owner the conversation is filed under; refuses a credential not granted the owner (403 ``not_owner``):
    only the owner's credential may say whose voice is the owner's."""
    from colonymind.api.auth import owner_person_id, speaks_for_owner
    if not speaks_for_owner(request):
        raise _observed_refusal(403, "not_owner", "only a credential granted the owner captures the conversations "
                                "the owner's device heard")
    return owner_person_id()


def _observed_detail(description: str) -> dict:
    from colonymind.api.errors import REFUSAL
    return {"description": description, "content": {"application/json": {"schema": dict(REFUSAL)}}}


@v2_router.put("/conversations/observed/{conversation_id:path}", response_model=ObservedConversationResponse,
               tags=[PUBLIC], dependencies=[Depends(refuse_hand_session)],
               responses={**capture_responses(ObservedConversationResponse, keyed=True),
                          403: _observed_detail("The credential is not granted the owner (not_owner), or not this "
                                                "device source (source_not_granted)."),
                          413: _observed_detail("The conversation is past the size bounds "
                                                "(observed_conversation_too_large).")})
async def observed_conversation_sync(conversation_id: str, body: ObservedConversationRequest, response: Response,
                                     request: Request = None) -> ObservedConversationResponse:
    """A conversation a wearable or room microphone heard, sent once and complete (docs/SOURCE-CONVERSATIONS.md).

    Idempotent by source and id: ``201`` the first time, ``200`` for an identical replay, ``409`` for changed
    content under the same source and id. Only the words labelled ``owner`` are the owner's.
    """
    from colonymind.api.auth import reserved_person_ids
    from colonymind.turns import ReservationOutcome, canonical_turn_digest, get_turn_idempotency_ledger
    from colonymind.turns import observed
    from colonymind.turns.idempotency import SourceErased
    owner = _observed_owner(request)
    try:
        conversation_id = observed.check_conversation_id(conversation_id)
        clocks = observed.validate(body, now=_temporal_now(), reserved_contacts=reserved_person_ids())
    except observed.ObservedRefusal as refusal:
        raise _observed_refusal(refusal.status, refusal.code, refusal.message) from None
    # The one call point of the observe operation's per-source grant (Unit A).
    if not observe_source_granted(request, body.source):
        raise _observed_refusal(403, "source_not_granted", f"the credential may not capture from {body.source}")
    contacts = sorted({speaker.contact_id for speaker in body.speakers if speaker.contact_id})
    if contacts:
        if _contacts_store is None:
            raise HTTPException(status_code=503, headers={"Retry-After": "30"}, detail={
                "code": "contacts_unavailable", "message": "a speaker's contact cannot be checked now"})
        for contact_id in contacts:
            if await _contacts_store.get(contact_id) is None:
                raise _observed_refusal(422, "unknown_contact", "a speaker names a contact that does not exist")
    source = observed.source_id(body.source, conversation_id)
    data = body.model_dump(mode="json")
    digest = canonical_turn_digest({"observed_conversation": conversation_id, "body": data})
    ledger = get_turn_idempotency_ledger(get_state_dir())

    def answer(outcome: str, status_code: int, **fields) -> ObservedConversationResponse:
        response.headers["Idempotency-Status"] = outcome
        response.status_code = status_code
        return ObservedConversationResponse(conversation_id=conversation_id, source_id=source, **fields)

    hold = _TurnSyncHold()
    await hold.take()
    try:
        if await hold.off(ledger.is_source_erased, source):
            return answer("erased", status.HTTP_200_OK, accepted=False, source_recorded=False,
                          skipped_reason="source_erased")
        reservation = await hold.off(ledger.reserve, source, digest)
        if reservation.outcome == ReservationOutcome.CONFLICT:
            raise HTTPException(status_code=409, detail={
                "code": "conversation_id_content_conflict", "conversation_id": conversation_id,
                "message": "conversation_id is already bound to different content"})
        if reservation.outcome == ReservationOutcome.AMBIGUOUS:
            raise HTTPException(status_code=503, detail={
                "code": "conversation_ingestion_ambiguous", "conversation_id": conversation_id,
                "message": "the first attempt was interrupted; inspect before sending it again"})
        if reservation.outcome == ReservationOutcome.IN_PROGRESS:
            response.headers["Retry-After"] = "1"
            return answer("in_progress", status.HTTP_202_ACCEPTED, accepted=False, source_recorded=False,
                          skipped_reason="identical_retry_in_progress")
        if reservation.outcome == ReservationOutcome.REPLAYED:
            cached = dict(reservation.response or {})
            cached.pop("conversation_id", None)
            cached.pop("source_id", None)
            return answer("replayed", status.HTTP_200_OK, **cached)
        try:
            speakers = await hold.away(
                asyncio.to_thread(observed.record, ledger, data, conversation_id, clocks, owner_id=owner))
        except SourceErased:
            await hold.off(ledger.release, source, digest)
            return answer("erased", status.HTTP_200_OK, accepted=False, source_recorded=False,
                          skipped_reason="source_erased")
        except VaultUnavailable:
            # Every write is the ledger's idempotent record: the retry repeats nothing already stored.
            await hold.off(ledger.release, source, digest)
            raise _vault_unavailable() from None
        except ValueError as exc:
            await hold.off(ledger.release, source, digest)
            if "exceeds" in str(exc):
                raise _observed_refusal(413, "observed_conversation_too_large",
                                        "the stored conversation would exceed the source limit") from None
            raise HTTPException(status_code=409, detail={"code": "conversation_source_conflict",
                                                         "conversation_id": conversation_id}) from exc
        except BaseException:
            await hold.off(ledger.release, source, digest)
            raise
        result = answer("created", status.HTTP_201_CREATED, accepted=True, source_recorded=True,
                        speaker_source_ids=speakers)
        await hold.off(ledger.complete, source, digest, result.model_dump(mode="json"))
        return result
    finally:
        hold.give()


@v2_router.get("/conversations/observed/{conversation_id:path}", tags=[PUBLIC],
              dependencies=[Depends(refuse_hand_session)],
              responses={403: _observed_detail("The credential is not granted the owner (not_owner)."),
                         404: _observed_detail("No conversation this credential may read is held under this id: "
                                               "unknown, forgotten, or from a source it is not granted "
                                               "(observed_conversation_not_found).")})
async def observed_conversation_read(conversation_id: str,
                                     source: str = Query(..., pattern=r'^[a-z0-9][a-z0-9_.-]{0,31}$'),
                                     request: Request = None) -> Dict[str, Any]:
    """The conversation as memory holds it now (owner only): its segments rebuilt from the ledger."""
    from colonymind.turns import get_turn_idempotency_ledger
    from colonymind.turns import observed
    _observed_owner(request)
    try:
        conversation_id = observed.check_conversation_id(conversation_id)
    except observed.ObservedRefusal as refusal:
        raise _observed_refusal(refusal.status, refusal.code, refusal.message) from None
    ledger = get_turn_idempotency_ledger(get_state_dir())
    record = await asyncio.to_thread(observed.stored, ledger, source, conversation_id)
    # A conversation from a source the credential is not granted, a forgotten one and an unknown id answer
    # alike: nothing says whether the id was ever used.
    if record is None or not observe_source_granted(request, record["source"]):
        raise _observed_refusal(404, "observed_conversation_not_found",
                                "no conversation this credential may read is held under this id")
    return record


def _occurred_at(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    """When an inbound turn happened: the capture's ``occurred_at``, never later than now (a skewed
    host cannot date a conversation into the future). None when absent or unreadable: the contact
    store then stamps its own now. Both follow ``time.time``, the clock the mind ticks on."""
    from colonymind.util.temporal import parse_iso
    stamp = parse_iso((metadata or {}).get("occurred_at"))
    if stamp is None:
        return None
    now = datetime.fromtimestamp(time.time(), timezone.utc)
    return min(stamp.astimezone(timezone.utc), now).strftime("%Y-%m-%dT%H:%M:%SZ")


def _arrived_at(value: Any, occurred: Optional[datetime]) -> Optional[datetime]:
    """When a captured owner message arrived at the host (``metadata.answer.arrived_at``, the gateway event's own
    time), never later than the capture's ``occurred_at`` or now; None when absent or unreadable."""
    from colonymind.util.temporal import parse_iso
    stamp = parse_iso(value) if isinstance(value, str) and len(value) <= 64 else None
    if stamp is None:
        return None
    now = datetime.fromtimestamp(time.time(), timezone.utc)
    return min(stamp.astimezone(timezone.utc), occurred or now, now)


async def _process_turn_sync(
    body: TurnSyncRequest,
    request: Request | None = None,
    *,
    resolved_sender_contact_id: str | None = None,
    turn_seal=None,
    hold: _TurnSyncHold | None = None,
) -> TurnSyncResponse:
    # A caller's hold on the turn sync gate is held when this returns or raises; a direct caller has
    # none, and the steps run under one of their own.
    own = _TurnSyncHold() if hold is None else None
    try:
        return await _turn_sync_steps(body, request, own or hold, resolved_sender_contact_id, turn_seal)
    finally:
        if own is not None:
            own.give()


async def _turn_sync_steps(body: TurnSyncRequest, request: Request | None, hold: _TurnSyncHold,
                           resolved_sender_contact_id: str | None, turn_seal) -> TurnSyncResponse:
    # Detected secrets are sealed before anything reads the turn (a direct caller included): the source,
    # its summary and topics, the communications log, the preference learner, opt-out, the mind's owner
    # turn and every event see references only. ``turn_seal`` carries what an earlier seal changed.
    from colonymind.secrets.seal import redact_sensitive_turn
    await hold.take()
    sealed_now = await hold.off(redact_sensitive_turn, body, vault=_secrets_vault,
                                contact_id=resolved_sender_contact_id or await hold.away(_secret_scope(body)),
                                stage=None if turn_seal is None else turn_seal.stage)
    turn_seal = sealed_now if turn_seal is None else turn_seal.merge(sealed_now)
    # Keep direct blocks for canonical source storage. All existing text-only
    # cognition consumers receive only explicit text, never repr(base64/URLs).
    from colonymind.turns import canonical_turn_digest
    source_body = body
    source_messages = [{"role": message.role, "content": message.content}
                       for message in (body.user_message, body.assistant_message)
                       if message is not None and (message.content.strip() if isinstance(message.content, str) else message.content)]
    if body.assistant_source_refs:
        for message in source_messages:
            if message['role'] == 'assistant':
                message['_supplied_sources'] = [ref.model_dump() for ref in body.assistant_source_refs]
    if body.assistant_input_refs:
        for message in source_messages:
            if message['role'] == 'assistant':
                message['_supplied_inputs'] = [ref.model_dump() for ref in body.assistant_input_refs]
    body = body.model_copy(deep=True)
    for field in ("user_message", "assistant_message"):
        message = getattr(body, field)
        if message is not None and isinstance(message.content, list):
            message.content = message.text_content()
    # Auto-derive channel_id when the host does not provide one, so
    # context provenance and cross-context leak detection always work.
    body.context.channel_id = await hold.away(_ensure_channel_id(
        body.context, identity=body.identity,
    ))
    # ── Attribution chokepoint (docs/RELATIONSHIPS.md) ──────────────────
    # Resolve WHO said this server-side. A supplied sender overrides the
    # client's contact_id (which goes stale in group sessions); a senderless
    # machine turn (cron/api channel or system-origin text) attributes to
    # the reserved "system" sentinel so it can never pollute a person's
    # affect/facts/psyche/interactions. Rewriting context.contact_id here
    # means every downstream consumer in this handler sees the truth.
    _resolved_human_sender = False
    _contact_grant_attested = False
    from colonymind.api.credentials import is_credential_identity
    # A credential's own identity (docs/CREDENTIALS.md), an operator's test turn included (_attribute_turn
    # classifies it): not a person, so nothing re-attributes it, not a sender and not the machine-turn
    # check (its channel or the words it carries), and its words are kept as session evidence that no
    # projection, appraisal, capture or question reads.
    _credential_turn = is_credential_identity(body.context.contact_id)
    try:
        from colonymind.identity.participants import (
            SYSTEM_CONTACT_ID, ParticipantResolver, is_machine_turn,
        )
        if _credential_turn:
            pass
        elif resolved_sender_contact_id is not None:
            body.context.contact_id = resolved_sender_contact_id
            _resolved_human_sender = True
        elif body.sender is not None and _contacts_store is not None:
            _res = await hold.away(ParticipantResolver(_contacts_store).resolve(
                platform=body.sender.platform,
                user_id=body.sender.user_id,
                display_name=body.sender.display_name,
                group_id=body.sender.group_id,
                channel_id=body.context.channel_id or "",
            ))
            if _res.contact_id:
                if _res.contact_id != body.context.contact_id:
                    logger.info(
                        "turn attribution: %s -> %s (%s%s)",
                        body.context.contact_id, _res.contact_id, _res.method,
                        ", shadow-created" if _res.created else "")
                body.context.contact_id = _res.contact_id
                _resolved_human_sender = True
                # The server resolved the sender itself; that is the attestation.
                _contact_grant_attested = True
        if not (_credential_turn or _resolved_human_sender) and is_machine_turn(
                body.context.channel_id or "",
                (getattr(body.user_message, "content", "") or "")
                if body.user_message else "",
                has_sender=body.sender is not None):
            body.context.contact_id = SYSTEM_CONTACT_ID
    except Exception:
        logger.debug("participant attribution failed; keeping client contact",
                     exc_info=True)
    from colonymind.api.auth import needs_owner_platform, owner_platform_granted, person_granted, request_authority
    # The final attributed contact is what the grant is checked against (docs/CREDENTIALS.md), for every
    # principal, development mode included: a sender who became the owner, or someone a credential bound to a
    # list of persons may not write as, after _attribute_turn looked is not stored.
    _authority = request_authority(request)
    from colonymind.api.credentials import platform_of
    _platform = platform_of(body.sender.platform if body.sender is not None else None,
                            source_body.context.channel_id)
    if needs_owner_platform(_authority, body.context.contact_id):
        if not owner_platform_granted(_authority, _platform):
            logger.warning("turn not stored: it would be the owner's from %r, which principal %s is not granted "
                           "the owner from", _platform, _authority.principal_id)
            return TurnSyncResponse(accepted=False, continuity_updated=False, skipped_reason="owner_not_granted")
    if (_authority.contacts == "list" and not _credential_turn and body.context.contact_id != "system"
            and not person_granted(_authority, body.context.contact_id)
            and not (_resolved_human_sender and _authority.senders_any)):
        logger.warning("turn not stored: principal %s may not write as the contact it resolved to",
                       _authority.principal_id)
        return TurnSyncResponse(accepted=False, continuity_updated=False, skipped_reason="contact_not_granted")
    if not source_body.context.channel_id and _resolved_human_sender and body.sender is not None:
        # A stale claimed contact can have another platform's primary handle.
        # The resolved sender establishes the fallback conversation platform;
        # explicit conversation keys (including derived work) stay unchanged.
        body.context.channel_id = f'{body.sender.platform.strip().lower()}:{body.context.contact_id}'
    # Neither the system sentinel nor a credential's own identity is a person.
    _is_system_turn = body.context.contact_id == "system" or _credential_turn

    # Persist complete attributed messages before derived graph/mining effects.
    # This is the central source of truth even when extractors are disabled.
    source_id = body.context.turn_id or "unkeyed:" + canonical_turn_digest(
        source_body.model_copy(update={"context": body.context}))
    source_recorded = False
    work_result = {}
    work = None
    if source_messages:
        from colonymind.turns import canonical_turn_digest, get_turn_idempotency_ledger
        ledger = get_turn_idempotency_ledger(get_state_dir())
        if await hold.off(ledger.is_source_erased, source_id, body.context.contact_id):
            return TurnSyncResponse(accepted=False, continuity_updated=False, skipped_reason="source_erased", **_work_ack(body, None))
        raw_hashes = turn_seal.raw_hashes(body.context.session_id)
        try:
            retained = await hold.off(ledger.retained_messages, contact_id=body.context.contact_id,
                                      session_id=body.context.session_id, messages=source_messages,
                                      raw_hashes=raw_hashes, staged=turn_seal.stage)
        except VaultUnavailable:
            raise _vault_unavailable() from None
        if not retained:
            return TurnSyncResponse(accepted=False, continuity_updated=False, skipped_reason="source_erased", **_work_ack(body, None))
        from colonymind.turns.idempotency import SourceErased
        try:
            from colonymind.work.ledger import root_candidate
            work = root_candidate(source_body, request=request, contact_id=body.context.contact_id,
                                  resolved_sender=_resolved_human_sender)
            await hold.off(
                ledger.record_source,
                source_id, contact_id=body.context.contact_id,
                session_id=body.context.session_id, messages=source_messages,
                scope="session" if _credential_turn else "person",
                occurred_at=(body.context.metadata or {}).get("occurred_at"),
                timezone_name=body.context.timezone,
                derive_claims=not _credential_turn,
                channel_id=body.context.channel_id,
                raw_hashes=raw_hashes, staged=turn_seal.stage,
                work=work, work_result=work_result,
            )
        except SourceErased:
            return TurnSyncResponse(accepted=False, continuity_updated=False, skipped_reason="source_erased", **_work_ack(body, None))
        except VaultUnavailable:
            raise _vault_unavailable() from None
        except ValueError as exc:
            if str(exc) == 'invalid_source_dependency':
                raise HTTPException(status_code=422, detail={'code': 'invalid_source_dependency'}) from exc
            raise
        source_recorded = True
        if not work_result.get('recorded'):
            work = None
        if work is not None:
            from colonymind.work import schedule_work
            schedule_work(source_id)
        from colonymind.memory import curated
        if (retained != source_messages or await hold.off(ledger.is_projection_erased, source_id)
                or await hold.off(curated.is_forgotten, ledger, source_id)):
            # Preserve unrelated evidence, but never run extractors on an
            # envelope whose summary or tools may repeat erased content.
            return TurnSyncResponse(accepted=False, source_recorded=True, continuity_updated=False, skipped_reason="source_erased", **_work_ack(body, None))
        if body.source_only:
            # record_source still schedules grounded USER claim projection.
            # Only the ordinary summary/tool/relationship effects are skipped.
            return TurnSyncResponse(accepted=True, source_recorded=True, continuity_updated=False,
                skipped_reason='source_survivor_only', **_work_ack(body, work))

    # Admitted: the values only the envelope's other fields carry (summary, topics, a caption) are stored now,
    # before the effects that keep those fields. The source's own were stored with its row.
    try:
        await hold.off(turn_seal.commit)
    except Exception as error:
        logger.warning("a sealed value of this turn's summary could not be stored (%s); its reference stays",
                       type(error).__name__)

    # If structured fields are empty but raw messages are present,
    # extract topics/entities/summary from the raw messages.
    if not body.topics and not body.entities and not body.summary:
        if body.user_message is not None or body.assistant_message is not None:
            user_text = body.user_message.content if body.user_message else ""
            asst_text = body.assistant_message.content if body.assistant_message else ""
            combined = f"User: {user_text}\nAssistant: {asst_text}".strip()
            if combined and combined != "User: \nAssistant:":
                body.summary = combined[:2000]
                # Extract rough topics from user message
                words = user_text.split()
                body.topics = [w.lower().strip(".,!?;:") for w in words if len(w) > 4][:10]

    # Commitment capture runs in the projection worker (commitments/extract.py) on the router.

    # Track last user message for concurrent-session safety (v0.13.0)
    if body.user_message is not None:
        try:
            from colonymind.util.session_safety import save_last_user_message_at
            save_last_user_message_at()
        except Exception:
            pass

    # Owner directive learning: when the owner explicitly states how they want
    # their assistant to communicate ("be concise", "use bullets", "no emoji"),
    # capture it deterministically at high confidence. Owner-only; ordinary
    # conversation never trips this (requires a style keyword + a directive cue).
    if _preference_learner is not None and body.user_message is not None:
        try:
            from colonymind.identity import get_owner_contact_id
            owner_id = get_owner_contact_id()
            if owner_id and body.context.contact_id == owner_id:
                if getattr(_preference_learner, 'perspective', None) is not None:
                    changes = await hold.off(_preference_learner.learn_source, source_id) if source_recorded else []
                    hit = (changes[0][0].split('.', 1)[0], changes[0][0], changes[0][1]) if changes else None
                else:
                    hit = await hold.away(_preference_learner.learn_directive(
                        getattr(body.user_message, "content", "") or ""
                    ))
                if hit is not None:
                    try:
                        from colonymind.events.broadcaster import emit as _emit
                        await hold.off(_emit, "preference.directive_learned",
                                       {"category": hit[0], "key": hit[1], "value": hit[2]})
                    except Exception:
                        pass
        except Exception as error:
            from colonymind.health_counters import bump
            bump("writes_dropped.owner_directive_learning")
            logger.warning("owner directive learning failed (%s, sqlite_errorname=%s)",
                           type(error).__name__, getattr(error, "sqlite_errorname", None))

    if _telemetry is not None:
        try:
            await hold.away(_telemetry.touch("last_sync_at"))
        except Exception:
            pass

    # Timeline spine (v0.21.0): journal the conversation turn so it lands on
    # the unified timeline, and bump the contact's recency (last_interaction_at).
    if body.summary:
        try:
            from colonymind.events.journal import append_event
            turn_event_data = {
                "contact_id": body.context.contact_id,
                "session_id": body.context.session_id,
                "channel_id": body.context.channel_id,   # cross-channel provenance for the timeline / handoff
                "summary": (body.summary or "")[:300],
                "topics": (body.topics or [])[:10],
                "tools_used": (body.tools_used or [])[:20],
            }
            await hold.off(append_event, "conversation.turn", turn_event_data)
        except Exception:
            logger.debug("journal conversation.turn failed", exc_info=True)
    # Opt-out: a contact who asks not to be messaged lowers their own may_contact
    # to never (only the owner ever raises it). Runs on the resolved sender.
    if (_contacts_store is not None and body.context.contact_id and not _is_system_turn
            and body.user_message is not None):
        try:
            from colonymind.contacts.optout import apply_opt_out
            from colonymind.identity import get_owner_contact_id
            await hold.away(apply_opt_out(_contacts_store, body.context.contact_id,
                                          getattr(body.user_message, "content", "") or "",
                                          source_ref=f"turn:{source_id}", owner_id=get_owner_contact_id()))
        except Exception:
            logger.warning("opt-out detection failed", exc_info=True)
    # Owner outreach (architecture 4.10): the owner's words about the mind reaching out (a reply to an
    # outreach, "stop checking in", an interest, a worry) reach the mind in the same turn, and so do their
    # answers to its questions in their own words. Only a turn the request's credential may write as the owner,
    # from a platform it is granted the owner on (docs/CREDENTIALS.md), is theirs. A capture that names its message
    # (``metadata.answer``: its chat, its id, what it replies or reacts to, when it arrived) is also read as a
    # possible answer to an ask the host could not read before its model ran: by explicit references only
    # (docs/MIND.md, Asks), on the platform resolved here; a ledger, words or an owner the capture names are never
    # taken. The owner's turn is then dated when the message arrived, never later than its capture: an ask's full
    # text that reached the owner while the model ran was not there when they answered. Never raises.
    if body.user_message is not None and body.context.contact_id and not _is_system_turn:
        try:
            from colonymind.identity import get_owner_contact_id
            from colonymind.util.temporal import parse_iso
            mind = _mind()
            if (mind is not None and hasattr(mind, "owner_turn")
                    and body.context.contact_id == get_owner_contact_id()
                    and owner_platform_granted(_authority, _platform)):
                metadata = body.context.metadata or {}
                named = metadata.get("answer") if isinstance(metadata.get("answer"), dict) else None
                occurred = parse_iso(_occurred_at(body.context.metadata))
                arrived = None if named is None else _arrived_at(named.get("arrived_at"), occurred)
                evidence = None if named is None else {
                    "platform": _platform,
                    **{key: str(named.get(key) or "")[:256] for key in (
                        "chat_id", "chat_type", "message_id", "reply_to_message_id", "reaction")},
                    **{key: named.get(key) is True for key in (
                        "reaction_removed", "forwarded", "voice", "own_account")},
                    **({"arrived_at": arrived.isoformat()} if arrived is not None else {})}
                await hold.away(mind.owner_turn(
                    getattr(body.user_message, "content", "") or "", turn_id=source_id,
                    occurred_at=arrived or occurred,
                    session_id=body.context.session_id,
                    scheduled=bool(metadata.get("run") or metadata.get("kanban_task")),
                    evidence=evidence))
        except Exception:
            logger.warning("owner outreach reaction failed", exc_info=True)
    try:
        if _contacts_store is not None and body.context.contact_id and not _is_system_turn:
            await hold.away(_contacts_store.record_interaction(body.context.contact_id,
                                                               at_iso=_occurred_at(body.context.metadata)))
        # Cross-channel communication ledger: record this exchange under the
        # CONVERSATION's channel (group vs DM vs voice provenance), never the
        # contact's primary-handle gateway (which collapsed everything to one
        # channel). New prose requires a canonical person source; transport
        # receipts retain their separate metadata-only ingestion path.
        try:
            if _comms_log is not None and body.context.contact_id and source_recorded and not _credential_turn:
                _ch = body.context.channel_id or "direct"
                _sess = body.context.session_id or ""
                _lineage, _ = _comms_log.source_input(source_id, body.context.contact_id)
                _comms_log.log(body.context.contact_id, channel=_ch,
                               direction="in",
                               summary=(body.summary or "")[:300],
                               session_id=_sess, source_lineage=_lineage)
                # Record the assistant's reply as an OUTBOUND exchange on the
                # SAME resolved contact + conversation channel. Without this
                # the ledger sees only half of every conversation, so
                # reciprocity and "when did we last talk each way" (which the
                # reachout recommendation depends on) read as never-replying.
                _asst = (getattr(body.assistant_message, "content", "") or ""
                         ) if body.assistant_message else ""
                if _asst.strip():
                    _comms_log.log(body.context.contact_id, channel=_ch,
                                   direction="out",
                                   summary=_asst[:300], session_id=_sess, source_lineage=_lineage)
        except Exception as error:
            from colonymind.health_counters import bump
            bump("writes_dropped.comms_ledger")
            logger.warning("comms ledger log failed (%s, sqlite_errorname=%s)",
                           type(error).__name__, getattr(error, "sqlite_errorname", None))
    except Exception:
        logger.debug("record_interaction failed", exc_info=True)

    # A turn without messages (the legacy summary-only shape) records nothing:
    # the source ledger is the one memory, and it keeps what was said.
    return TurnSyncResponse(
        accepted=True, continuity_updated=source_recorded, source_recorded=source_recorded,
        skipped_reason=None if source_recorded else "no_source_messages",
        **_work_ack(body, work),
    )


def _work_ack(body, work):
    if body.work is None:
        return {}
    from colonymind.work import work_records_enabled
    return {'work_recorded': work is not None, 'work_skipped_reason': None if work is not None else (
        'ineligible_origin' if work_records_enabled() else 'work_records_disabled')}


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Events (WebSocket)
# ---------------------------------------------------------------------------

@router.websocket("/events")
async def events_ws(ws: WebSocket) -> None:
    await ws.accept()

    # Read auth message. New clients send an exact sequence plus the journal
    # record time; ``lastEventId`` remains accepted as the legacy time cursor.
    last_event_seq: Optional[int] = None
    last_event_time = ""
    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=10)
        import json as _json
        msg = _json.loads(raw)
        if msg.get("type") != "auth":
            await ws.close(code=4001, reason="Expected auth message")
            return
        token = msg.get("token", "")
        from colonymind.api.auth import configured_api_key, token_matches
        expected = configured_api_key()
        if not expected:
            # Fail closed: without a key we cannot authenticate event-stream
            # subscribers, and this socket carries live state changes.
            await ws.close(
                code=4003,
                reason="API authentication not configured on server",
            )
            return
        if not token_matches(str(token), expected):
            await ws.close(code=4003, reason="Invalid API key")
            return

        raw_seq = msg.get("lastEventSeq")
        legacy_cursor = str(msg.get("lastEventId") or "")
        if raw_seq is None and legacy_cursor.isdigit():
            raw_seq = legacy_cursor
        if raw_seq is not None:
            last_event_seq = int(raw_seq)
            if last_event_seq < 0:
                raise ValueError("lastEventSeq must be non-negative")
        last_event_time = str(msg.get("lastEventTime") or "")
        if not last_event_time and legacy_cursor and not legacy_cursor.isdigit():
            last_event_time = legacy_cursor
    except asyncio.TimeoutError:
        await ws.close(code=4001, reason="Auth timeout")
        return
    except Exception:
        await ws.close(code=4001, reason="Invalid auth")
        return

    subscriber = EventSubscriberBuffer(
        _event_subscriber_queue_size(), loop=asyncio.get_running_loop()
    )

    # Subscribe before capturing the durable high-water mark. Events committed
    # during replay are buffered and later filtered by sequence, closing the
    # replay/live race without duplicate delivery.
    from colonymind.events.journal import current_sequence, replay_events
    with _event_broadcast_lock:
        _event_subscribers.append(subscriber)

    disconnect_event = asyncio.Event()
    receive_task: Optional[asyncio.Task] = None

    async def _watch_disconnect() -> None:
        """Consume post-auth client frames so quiet disconnects are observed."""
        try:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    return
                # Application-level pings from legacy clients are deliberately
                # consumed. Modern clients use protocol-level WebSocket ping.
        except (WebSocketDisconnect, RuntimeError):
            return
        except Exception:
            logger.debug("Event WebSocket receive watcher stopped", exc_info=True)
            return
        finally:
            disconnect_event.set()

    try:
        replay_through_seq = await asyncio.to_thread(current_sequence)
        cursor_reset = (
            last_event_seq is not None
            and last_event_seq > replay_through_seq
        )
        await ws.send_json({
            "type": "connected",
            "journalHighWaterSeq": replay_through_seq,
            "queueCapacity": subscriber.queue.maxsize,
            "cursorReset": cursor_reset,
        })
        if cursor_reset:
            await ws.send_json({
                "type": "replay_reset",
                "requestedAfterSeq": last_event_seq,
                "journalHighWaterSeq": replay_through_seq,
                "reason": "cursor_ahead_of_journal",
            })

        # A brand-new subscriber starts at the captured high-water mark rather
        # than receiving an arbitrary retention-window history. Reconnects use
        # the exact processed sequence, with timestamp fallback for old clients.
        replay_after_seq = 0 if cursor_reset else last_event_seq
        if replay_after_seq is None and not last_event_time:
            replay_after_seq = replay_through_seq

        replayed_count = 0
        replay_cursor = replay_after_seq
        replay_since = last_event_time
        first_page = True
        last_replayed_seq = min(replay_after_seq or 0, replay_through_seq)

        while True:
            result = await asyncio.to_thread(
                replay_events,
                replay_since,
                1000,
                None,
                False,
                after_seq=replay_cursor,
                until_seq=replay_through_seq,
            )
            if result.get("replayError"):
                await ws.send_json({
                    "type": "replay_error",
                    "reason": result["replayError"],
                })
                await ws.close(code=1011, reason="Event replay unavailable")
                return

            first_available = int(result.get("firstAvailableSeq") or 0)
            if (
                first_page
                and replay_cursor is not None
                and first_available > 0
                and replay_cursor + 1 < first_available
            ):
                await ws.send_json({
                    "type": "replay_gap",
                    "requestedAfterSeq": replay_cursor,
                    "firstAvailableSeq": first_available,
                    "reason": "cursor_precedes_retention_window",
                })
            corrupt_count = int(result.get("corruptCount") or 0)
            if first_page and corrupt_count:
                await ws.send_json({
                    "type": "replay_integrity_warning",
                    "corruptRecordCount": corrupt_count,
                })

            page = result.get("events", [])
            for event in page:
                frame = {
                    "type": event["type"],
                    "occurred_at": event.get("occurredAt") or event["recordedAt"],
                    "recordedAt": event["recordedAt"],
                    "payload": event.get("data", {}),
                    "seq": event["seq"],
                    "eventId": event.get("ulid", ""),
                }
                await ws.send_json(frame)
                subscriber.mark_delivered(frame)
                replayed_count += 1
                last_replayed_seq = max(last_replayed_seq, int(event["seq"]))

            if not result.get("hasMore") or not page:
                break
            replay_cursor = int(result["lastSeq"])
            replay_since = ""
            first_page = False

        # Always terminate the handshake explicitly. This means an idle stream
        # never requires a client to guess whether replay has completed.
        await ws.send_json({
            "type": "replay_complete",
            "replayedCount": replayed_count,
            "lastSeq": last_replayed_seq,
            "replayThroughSeq": replay_through_seq,
        })
        subscriber.mark_delivered({"seq": replay_through_seq})

        receive_task = asyncio.create_task(_watch_disconnect())
        while True:
            event_task = asyncio.create_task(subscriber.get())
            disconnect_task = asyncio.create_task(disconnect_event.wait())
            done, _ = await asyncio.wait(
                {event_task, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect_task in done:
                event_task.cancel()
                try:
                    await event_task
                except asyncio.CancelledError:
                    pass
                return
            disconnect_task.cancel()
            try:
                await disconnect_task
            except asyncio.CancelledError:
                pass
            event = event_task.result()
            if subscriber.is_overflow(event):
                await ws.send_json(subscriber.overflow_frame())
                await ws.close(
                    code=1013,
                    reason="Event subscriber overflow; reconnect and replay",
                )
                return
            # Frames at or below the replay snapshot were both journaled and
            # buffered while the replay ran. They have already been delivered.
            try:
                seq = int(event.get("seq", 0))
            except (TypeError, ValueError):
                seq = 0
            if seq and seq <= replay_through_seq:
                continue
            await ws.send_json(event)
            subscriber.mark_delivered(event)
    except WebSocketDisconnect:
        pass
    finally:
        if receive_task is not None and not receive_task.done():
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass
        subscriber.close()
        with _event_broadcast_lock:
            try:
                _event_subscribers.remove(subscriber)
            except ValueError:
                pass


@router.get("/events/replay")
async def events_replay(
    since: str = Query("", description="ISO 8601 journal time — replay events after this time"),
    limit: int = Query(500, ge=1, le=1000, description="Max events to return"),
    types: Optional[str] = Query(None, description="Comma-separated event type filter"),
    after_seq: Optional[int] = Query(
        None,
        alias="afterSeq",
        ge=0,
        description="Exact sequence cursor; takes precedence over since",
    ),
    until_seq: Optional[int] = Query(
        None,
        alias="untilSeq",
        ge=0,
        description="Optional inclusive replay high-water sequence",
    ),
) -> dict:
    """Replay journal events for disconnected clients.

    Returns events in sequential order. Prefer ``afterSeq`` from the last
    processed WebSocket frame; ``since`` supports legacy timestamp clients.
    """
    from colonymind.events.journal import replay_events

    type_list = [t.strip() for t in types.split(",")] if types else None
    return await asyncio.to_thread(
        replay_events,
        since=since,
        limit=limit,
        types=type_list,
        after_seq=after_seq,
        until_seq=until_seq,
    )


# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------

_contacts_store = None

def set_contacts_store(store) -> None:
    global _contacts_store
    _contacts_store = store


async def _contact_responses(contacts: List[Any]) -> List[ContactResponse]:
    """Contacts as the contact routes return them: the digest only while every claim it cites is still live
    and the person's own (``contacts.digest.checked_digests``, the check the "About this person" section and
    the owner's inspect make); otherwise neither the digest nor its sources."""
    from colonymind.contacts.digest import checked_digests, source_ledger
    contacts = list(contacts)
    cited = any(contact.digest and contact.digest_sources for contact in contacts)
    checked = await checked_digests(contacts, store=_contacts_store, ledger=source_ledger() if cited else None)
    responses = []
    for contact in contacts:
        row = contact.to_dict()
        if checked.get(contact.contact_id, ("", "none"))[1] != "shown":
            row.update(digest=None, digest_sources=[])
        responses.append(ContactResponse(**row))
    return responses


async def _contact_response(contact: Any) -> ContactResponse:
    return (await _contact_responses([contact]))[0]


@router.get("/contacts", response_model=ContactListResponse)
async def list_contacts(
    source: Optional[str] = None,
    trust_tier: Optional[str] = None,
    include_discovered: bool = True,
    limit: int = 100,
) -> ContactListResponse:
    if _contacts_store is None:
        return ContactListResponse(contacts=[])
    try:
        contacts: List[Any] = []

        # Determine which sources to include
        include_curated = source in (None, "all", "curated")
        include_world = source in (None, "all", "world_model")
        if source is None:
            include_world = include_discovered

        if include_curated:
            curated = await _contacts_store.list(
                trust_tier=trust_tier,
                limit=limit,
            )
            contacts.extend(curated)

        if include_world:
            world = await _contacts_store.list(
                trust_tier=trust_tier or "acquaintance",
                limit=limit,
            )
            # Exclude already-included curated contacts
            curated_ids = {c.contact_id for c in contacts}
            for c in world:
                if c.contact_id not in curated_ids and c.import_source == "world_model":
                    contacts.append(c)

        # Sort: curated first (by trust tier rank desc), then world model
        from colonymind.contacts.models import _TIER_RANK
        def _sort_key(c):
            is_curated = 1 if c.import_source != "world_model" else 0
            tier_rank = _TIER_RANK.get(c.trust_tier, 0)
            last_int = c.last_interaction_at or ""
            return (is_curated, tier_rank, last_int)

        contacts.sort(key=_sort_key, reverse=True)
        contacts = contacts[:limit]

        return ContactListResponse(
            contacts=await _contact_responses(contacts),
            source_filter=source or "all",
            total=len(contacts),
        )
    except Exception as exc:
        logger.warning("list_contacts failed: %s", exc)
        return ContactListResponse(contacts=[])


@router.post("/contacts", response_model=ContactResponse, status_code=201,
             dependencies=[Depends(refuse_hand_session)])
async def create_contact(body: ContactCreateRequest) -> ContactResponse:
    """Create a curated contact (with optional handles) via the API.

    Exists primarily so deployments can bootstrap the OWNER contact; before
    this, contacts could only appear as side effects of message ingestion.
    """
    if _contacts_store is None:
        raise HTTPException(status_code=501, detail="Contact store not initialized")
    try:
        contact = await _contacts_store.create(
            display_name=body.display_name,
            given_name=body.given_name,
            family_name=body.family_name,
            organization=body.organization,
            trust_tier=body.trust_tier,
            may_contact=body.may_contact,
            cadence_minutes=body.cadence_minutes,
            tags=body.tags,
            notes=body.notes,
            import_source="manual",
        )
        for handle in body.handles:
            await _contacts_store.add_handle(
                contact.contact_id,
                gateway=handle.gateway,
                address=handle.address,
                is_primary=handle.is_primary,
                verified=handle.verified,
                source="manual",
            )
        created = await _contacts_store.get(contact.contact_id)
        return await _contact_response(created or contact)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.warning("create_contact failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/contacts/intro", response_model=ContactIntroResponse, status_code=201,
             dependencies=[Depends(refuse_hand_session)])
async def capture_introduction(body: ContactIntroRequest) -> ContactIntroResponse:
    """Capture an organic introduction (social-graph autonomy, generic).

    The agent met or learned of a person; record them as a durable, queryable
    graph node WITH provenance (introduced_by + met_via). If the handle already
    resolves to a known contact, the provenance is recorded on that contact
    instead of duplicating it. A new contact gets may_contact='ask': an intro
    never grants unprompted outreach; only the owner raises it.
    """
    if _contacts_store is None:
        raise HTTPException(status_code=501, detail="Contact store not initialized")
    try:
        # If a handle was given and already resolves to a known person, annotate
        # rather than duplicate (a number is one identity across phone gateways).
        existing = None
        if body.gateway and body.address:
            existing = await _contacts_store.resolve_messaging_handle(
                body.gateway, body.address)
        if existing is not None:
            updated = await _contacts_store.record_introduction(
                existing.contact_id,
                introduced_by=body.introduced_by,
                met_via=body.met_via,
            )
            return ContactIntroResponse(
                contact=await _contact_response(updated or existing),
                created=False,
            )

        contact = await _contacts_store.create(
            display_name=body.name,
            trust_tier=body.trust_tier,
            may_contact="ask",
            import_source="agent_intro",
            notes=body.note,
            introduced_by=body.introduced_by,
            met_via=body.met_via,
        )
        if body.gateway and body.address:
            try:
                # The handle keeps its transport; the store matches phone numbers
                # across gateways on the canonical phone identity.
                await _contacts_store.add_handle(
                    contact.contact_id, gateway=body.gateway,
                    address=body.address, source="agent_intro")
            except ValueError as exc:
                # Handle raced onto another contact between resolve and create.
                logger.info("intro add_handle conflict for %s: %s",
                            contact.contact_id, exc)
        created = await _contacts_store.get(contact.contact_id)
        return ContactIntroResponse(
            contact=await _contact_response(created or contact),
            created=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.warning("capture_introduction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/contacts/resolve", response_model=ContactResponse, tags=[PUBLIC])
async def resolve_contact_by_handle(gateway: str, address: str, request: Request, create: bool = False) -> ContactResponse:
    """Resolve a contact from a messaging handle (v0.21.2).

    Registered BEFORE /contacts/{contact_id} so it isn't shadowed by the
    parameterized route. Lets the host plugin map an inbound sender
    (platform + address) to the real ColonyMind contact, so per-contact
    memory/affect/facts engage instead of pooling everything under 'default'.

    With ``create=true`` an unknown messaging sender becomes a shadow contact
    through the participant ladder, the one shadow creator (trust_tier=unknown,
    may_contact='ask'), so its memory attributes to a real person instead of
    being lost; merge and link proposals reconcile it later.
    """
    if _contacts_store is None:
        raise HTTPException(status_code=404, detail="Contact store not initialized")
    try:
        # Normalized, cross-gateway phone-identity resolution (a number is one contact regardless of
        # the transport it arrived on). find_by_handle stays exact-match for dedup callers.
        contact = await _contacts_store.resolve_messaging_handle(gateway, address)
        if contact is None and create and gateway and address:
            from colonymind.identity.participants import ParticipantResolver
            resolution = await ParticipantResolver(_contacts_store).resolve(
                platform=gateway, user_id=address, channel_id=gateway)
            if resolution.contact_id:
                contact = await _contacts_store.get(resolution.contact_id)
        if contact is None:
            raise HTTPException(status_code=404, detail="No contact for that handle")
        return await _contact_response(contact)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("resolve_contact_by_handle failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/contacts/{contact_id}", response_model=ContactResponse)
async def get_contact(contact_id: str) -> ContactResponse:
    if _contacts_store is None:
        raise HTTPException(status_code=404, detail="Contact not found")
    try:
        contact = await _contacts_store.get(contact_id)
        if contact is None:
            raise HTTPException(status_code=404, detail="Contact not found")
        return await _contact_response(contact)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("get_contact failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/authz/scope", response_model=ScopeAuthzResponse)
async def authz_scope(platform: str, external_id: str, gateway: str, address: str) -> ScopeAuthzResponse:
    """Is this sender authorized WITHIN this group scope? Context-scoped only — it
    says nothing about 1:1 DM rights. Used by the messaging bridge to admit group
    members of an agent-created/joined group without granting them per-user access."""
    if _contacts_store is None:
        return ScopeAuthzResponse(authorized=False)
    try:
        scope = await _contacts_store.get_scope(platform=platform, external_id=external_id)
        if scope is None or not scope.active:
            return ScopeAuthzResponse(
                authorized=False,
                scope_id=(scope.scope_id if scope else None),
                active=bool(scope and scope.active),
            )
        contact = await _contacts_store.resolve_messaging_handle(gateway, address)
        if contact is None:
            return ScopeAuthzResponse(
                authorized=False, scope_id=scope.scope_id,
                granted_tier=scope.granted_tier, active=True,
            )
        ok = await _contacts_store.is_authorized_in_scope(contact.contact_id, scope.scope_id)
        return ScopeAuthzResponse(
            authorized=ok, scope_id=scope.scope_id, granted_tier=scope.granted_tier,
            contact_id=contact.contact_id, active=True,
        )
    except Exception as exc:
        logger.warning("authz_scope failed: %s", exc)
        return ScopeAuthzResponse(authorized=False)


@router.post("/authz/scope", response_model=ScopeResponse, status_code=201,
             dependencies=[Depends(refuse_hand_session)])
async def create_authz_scope(body: ScopeCreateRequest) -> ScopeResponse:
    """Create (idempotent by platform+external_id) a trust scope and add members.
    Unknown member handles are auto-created as shadow contacts on first sight
    (acquaintance tier, no 1:1 interaction) — see social-graph-autonomy spec."""
    if _contacts_store is None:
        raise HTTPException(status_code=501, detail="Contact store not initialized")
    try:
        scope = await _contacts_store.create_scope(
            scope_type=body.scope_type, platform=body.platform, external_id=body.external_id,
            label=body.label, granted_tier=body.granted_tier, created_by=body.created_by,
        )
        member_ids: List[str] = []
        for m in body.members:
            cid = m.contact_id
            if cid is None and m.gateway and m.address:
                contact = await _contacts_store.resolve_messaging_handle(m.gateway, m.address)
                if contact is None:
                    contact = await _contacts_store.create(
                        display_name=(m.name or m.address),
                        trust_tier="acquaintance", import_source="agent_scope",
                    )
                    # rcs is a transport over the phone identity; store the handle under the
                    # canonical phone gateway (sms) so it resolves across phone-bearing channels.
                    store_gw = "sms" if m.gateway == "rcs" else m.gateway
                    await _contacts_store.add_handle(
                        contact.contact_id, gateway=store_gw, address=m.address,
                        source="agent_scope", confidence=0.6,
                    )
                cid = contact.contact_id
            if cid:
                await _contacts_store.add_scope_member(scope.scope_id, cid, role=m.role)
                member_ids.append(cid)
        return ScopeResponse(
            scope_id=scope.scope_id, scope_type=scope.scope_type, platform=scope.platform,
            external_id=scope.external_id, label=scope.label, granted_tier=scope.granted_tier,
            active=scope.active, members=member_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.warning("create_authz_scope failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/authz/scope/deactivate",
             dependencies=[Depends(refuse_hand_session)])
async def deactivate_authz_scope(body: ScopeDeactivateRequest) -> Dict[str, Any]:
    """Revoke group-trust for a whole scope at once (e.g. owner mutes/leaves the group)."""
    if _contacts_store is None:
        raise HTTPException(status_code=501, detail="Contact store not initialized")
    try:
        scope_id = body.scope_id
        if scope_id is None and body.platform and body.external_id:
            scope = await _contacts_store.get_scope(platform=body.platform, external_id=body.external_id)
            scope_id = scope.scope_id if scope else None
        if scope_id is None:
            raise HTTPException(status_code=404, detail="scope not found")
        await _contacts_store.deactivate_scope(scope_id)
        return {"ok": True, "scope_id": scope_id, "active": False}
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("deactivate_authz_scope failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/scopes/promotion-candidates")
async def scope_promotion_candidates() -> Dict[str, Any]:
    """Group-scope members with sustained contact but no 1:1 rights yet: the people the owner
    can promote (group_guest -> regular). They are proposals; only the owner promotes."""
    if _contacts_store is None:
        raise HTTPException(status_code=501, detail="Contact store not initialized")
    cfg = getattr(_contacts_store, "_config", None)
    min_int = int(getattr(cfg, "group_promote_min_interactions", 5))
    cands = await _contacts_store.group_promotion_candidates(min_interactions=min_int)
    return {
        "min_interactions": min_int,
        "candidates": [
            {"contact_id": c.contact_id, "display_name": c.display_name,
             "trust_tier": c.trust_tier, "interaction_count": c.interaction_count}
            for c in cands
        ],
    }


@router.post("/scopes/promote",
             dependencies=[Depends(refuse_hand_session)])
async def scope_promote(body: ScopePromoteRequest) -> Dict[str, Any]:
    """Promote one group-scope member to global 1:1 (tier >= to_tier). Only ever raises the
    tier, never the permission to message them. Called after the owner's approval."""
    if _contacts_store is None:
        raise HTTPException(status_code=501, detail="Contact store not initialized")
    try:
        changed = await _contacts_store.promote_scope_member(body.contact_id, to_tier=body.to_tier)
    except Exception as exc:
        logger.warning("scope_promote failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": True, "contact_id": body.contact_id, "changed": changed}


@router.post("/contacts/{contact_id}/timezone", response_model=ContactResponse,
             dependencies=[Depends(refuse_hand_session)])
async def set_contact_timezone(contact_id: str, body: ContactTimezoneRequest) -> ContactResponse:
    """Set (or clear, with null) a contact's IANA timezone (v0.21.0, editable)."""
    if _contacts_store is None:
        raise HTTPException(status_code=501, detail="Contact store not initialized")
    try:
        await _contacts_store.set_timezone(contact_id, body.timezone)
        contact = await _contacts_store.get(contact_id)
        if contact is None:
            raise HTTPException(status_code=404, detail="Contact not found")
        return await _contact_response(contact)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.warning("set_contact_timezone failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/temporal/config", response_model=TemporalConfigResponse)
async def get_temporal_config() -> TemporalConfigResponse:
    """Current temporal reference frame (agent home tz + defaults). v0.21.0."""
    from colonymind.util import temporal as _temporal
    atz = _temporal.agent_timezone()
    return TemporalConfigResponse(
        agent_timezone=atz,
        default_contact_timezone=_temporal.default_contact_timezone(),
        now_utc=_temporal.now_utc().isoformat(),
        now_agent_local=_temporal.now_in(atz).isoformat(),
        agent_local_clock=_temporal.format_clock(_temporal.now_in(atz)),
    )


@router.post("/temporal/config", response_model=TemporalConfigResponse)
async def set_temporal_config(body: TemporalConfigRequest) -> TemporalConfigResponse:
    """Edit the agent home tz and/or the default contact tz. v0.21.0."""
    from colonymind.util import temporal as _temporal
    try:
        if body.agent_timezone is not None:
            _temporal.set_agent_timezone(body.agent_timezone)
        if body.clear_default_contact_timezone:
            _temporal.set_default_contact_timezone(None)
        elif body.default_contact_timezone is not None:
            _temporal.set_default_contact_timezone(body.default_contact_timezone)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    atz = _temporal.agent_timezone()
    return TemporalConfigResponse(
        agent_timezone=atz,
        default_contact_timezone=_temporal.default_contact_timezone(),
        now_utc=_temporal.now_utc().isoformat(),
        now_agent_local=_temporal.now_in(atz).isoformat(),
        agent_local_clock=_temporal.format_clock(_temporal.now_in(atz)),
    )


_TIMELINE_TYPE_LABELS = {
    "conversation.turn": "💬 talked",
    "outreach.sent": "📤 reached out",
    "initiative.generated": "💡 initiative",
    "initiative.completed": "✅ initiative done",
    "task.created": "📋 task created",
    "task.completed": "✅ task done",
    "commitment.made": "🤝 promised",
    "commitment.fulfilled": "✅ kept promise",
    "memory.written": "🧠 remembered",
}


@router.get("/timeline", response_model=TimelineResponse)
async def get_timeline(
    since: str = Query("24h", description="Relative ('24h','7d') or ISO, or today/yesterday"),
    types: Optional[str] = Query(None, description="Comma-separated event types to include"),
    contact_id: Optional[str] = Query(None, description="Only events involving this contact"),
    limit: int = Query(100, ge=1, le=500),
) -> TimelineResponse:
    """Chronological timeline across all ColonyMind subsystems (v0.21.0).

    Backed by the event journal. Lets the agent answer 'what happened recently',
    'what's been going on with X', 'what changed since I last looked'.
    """
    from colonymind.events.journal import timeline_events
    from colonymind.util import temporal as _t

    since_iso = _t.parse_relative_since(since)
    type_list = [t.strip() for t in types.split(",") if t.strip()] if types else None
    # Walk the journal newest-first so the cap drops the OLD end of a large
    # window — an oldest-first walk here showed stale activity while the
    # endpoint claimed "newest first" whenever the window exceeded the cap.
    raw = await asyncio.to_thread(
        timeline_events, since_iso, limit=limit, types=type_list, contact_id=contact_id,
    )

    all_events: list[TimelineEvent] = []
    for e in raw.get("events", []):
        data = e.get("data", {}) or {}
        cid = data.get("contact_id") or data.get("person_id")
        if contact_id and cid != contact_id:
            continue
        at = e.get("recordedAt", "")
        all_events.append(TimelineEvent(
            seq=e.get("seq", 0),
            type=e.get("type", "unknown"),
            at=at,
            when=_t.humanize_delta(at),
            bucket=_t.bucket(at),
            summary=(data.get("summary") or data.get("title") or data.get("text")
                     or data.get("reason")),
            contact_id=cid,
            data=data,
        ))

    all_events.sort(key=lambda x: x.seq, reverse=True)  # newest first
    events = all_events[:limit]
    has_more = len(all_events) > limit or raw.get("hasMore", False)

    # Resolve contact ids -> names for a readable digest (cached).
    _name_cache: dict = {}
    async def _contact_name(cid):
        if not cid or cid == "default":
            return None
        if cid in _name_cache:
            return _name_cache[cid]
        nm = cid
        if _contacts_store is not None:
            try:
                c = await _contacts_store.get(cid)
                if c is not None:
                    nm = c.display_name or c.given_name or cid
            except Exception:
                pass
        _name_cache[cid] = nm
        return nm

    digest_lines = []
    for ev in events[:40]:
        label = _TIMELINE_TYPE_LABELS.get(ev.type, ev.type)
        snippet = (ev.summary or "").replace("\n", " ").strip()
        if len(snippet) > 120:
            snippet = snippet[:117] + "…"
        nm = await _contact_name(ev.contact_id)
        who = f" with {nm}" if nm else ""
        digest_lines.append(f"• {ev.when} — {label}{who}{': ' + snippet if snippet else ''}")
    digest = ("\n".join(digest_lines)
              if digest_lines else f"No journaled events since {since_iso}.")

    return TimelineResponse(
        since=since_iso,
        count=len(events),
        digest=digest,
        events=events,
        has_more=has_more,
    )


@router.get("/temporal/contacts", response_model=TemporalContactsResponse)
async def temporal_contacts(
    overdue_only: bool = Query(True, description="Only contacts overdue vs their own cadence"),
    limit: int = Query(20, ge=1, le=100),
) -> TemporalContactsResponse:
    """Per-contact cadence + silence (v0.21.0).

    Estimates each contact's typical rhythm and flags those overdue relative to
    *their* cadence. Powers cadence-aware proactive outreach (the worker/agent
    queries this) and the per-turn 'heads-up' line.
    """
    from colonymind.util import temporal as _t
    if _contacts_store is None:
        return TemporalContactsResponse(now=_t.now_utc().isoformat(), count=0, contacts=[])
    rows = await _contacts_store.compute_cadence_overdue(
        overdue_only=overdue_only, limit=limit,
    )
    return TemporalContactsResponse(
        now=_t.now_utc().isoformat(),
        count=len(rows),
        contacts=[TemporalContact(**r) for r in rows],
    )


@router.post("/contacts/{contact_id}/style", response_model=ContactStyleResponse,
             dependencies=[Depends(refuse_hand_session)])
async def get_contact_style(contact_id: str, body: ContactStyleRequest) -> ContactStyleResponse:
    if _contacts_store is None:
        return ContactStyleResponse(person_id=contact_id)
    try:
        style = await _contacts_store.get_style(contact_id)
        return ContactStyleResponse(person_id=contact_id, **style)
    except Exception as exc:
        logger.warning("get_contact_style failed: %s", exc)
        return ContactStyleResponse(person_id=contact_id)


_commitment_store = None


def set_commitment_store(store):
    global _commitment_store
    _commitment_store = store


#: The contacts' affect (``contacts.affect``): the appraisals write a non-owner speaker's valence
#: into it and the mind's social drive reads its trend.
_affect_store = None


def set_affect_store(store):
    global _affect_store
    _affect_store = store


async def _ensure_channel_id(
    context,
    identity=None,
) -> str:
    """Derive a stable channel_id when the host does not provide one.

    Resolution order:
    1. Host-provided channel_id (pass-through)
    2. Contact's primary handle gateway + contact_id
    3. host_id + contact_id
    4. "unknown:" + contact_id
    """
    if context.channel_id:
        return context.channel_id

    contact = context.contact_id or "anonymous"
    gateway = None

    if _contacts_store is not None and context.contact_id:
        try:
            handles = await _contacts_store.get_handles(context.contact_id)
            primary = next(
                (h for h in handles if getattr(h, "is_primary", False)),
                handles[0] if handles else None,
            )
            if primary is not None:
                gateway = getattr(primary, "gateway", None)
        except Exception:
            pass

    if not gateway and identity is not None:
        gateway = getattr(identity, "host_id", None)

    return f"{gateway or 'unknown'}:{contact}"


_comms_log = None


def set_comms_log(store):
    global _comms_log
    _comms_log = store


@router.get("/comms/recent")
async def comms_recent(limit: int = 50, window_days: int = 30) -> dict:
    """Cross-channel communication ledger: the newest exchanges across every
    contact and channel, plus an inbound/outbound rollup per channel over the
    window. Read-only view over the same ledger that ``/turns/sync`` writes;
    contact ids are resolved to display names for the ops view. System turns
    are recorded in the ledger and returned tagged, not hidden."""
    if _comms_log is None:
        return {"available": False}
    try:
        snapshot = await asyncio.to_thread(_comms_log.recent_snapshot, limit=limit,
                                           since_days=window_days)
        entries = snapshot["entries"]
        names: dict = {}
        if _contacts_store is not None:
            for cid in {e.get("contact_id") for e in entries if e.get("contact_id")}:
                try:
                    c = await _contacts_store.get(cid)
                    if c is not None:
                        names[cid] = getattr(c, "display_name", None)
                except Exception:
                    pass
        for e in entries:
            e["display_name"] = names.get(e.get("contact_id"))
        return {"available": True, "window_days": int(window_days),
                "entries": entries,
                "by_channel": snapshot["by_channel"]}
    except Exception as exc:
        return {"available": True, "error": str(exc)}


_preference_learner = None


def set_preference_learner(learner):
    global _preference_learner
    _preference_learner = learner


_feedback_store = None


def set_feedback_store(store) -> None:
    global _feedback_store
    _feedback_store = store


@router.get("/feedback")
def get_type_feedback() -> dict:
    """Per-type outcome feedback + priority multipliers (observability)."""
    if _feedback_store is None:
        return {"available": False, "types": []}
    try:
        return {"available": True, "types": _feedback_store.snapshot()}
    except Exception as exc:
        return {"available": True, "error": str(exc), "types": []}


_expectations = None


def set_expectations(e) -> None:
    global _expectations
    _expectations = e


@router.get("/self/expectations")
async def get_expectations(limit: int = 50) -> dict:
    """Expectation engine: pending predictions with their horizons and
    per-domain calibration (Mind M3a). A prediction that missed became a
    surprise on her mind."""
    if _expectations is None:
        return {"available": False}
    try:
        out = {"available": True}
        out.update(await asyncio.to_thread(_expectations.snapshot, limit=max(1, min(200, limit))))
        return out
    except Exception as exc:
        return {"available": True, "error": str(exc)}


# The action journal remains available to host clients.
_self_model = None


def set_self_model(sm) -> None:
    global _self_model
    _self_model = sm


@router.get("/self/journal")
async def get_action_journal(limit: int = 50, domain: str = "",
                             today: bool = False) -> dict:
    """Unified action journal (what was done, why, with what confidence)."""
    journal = getattr(_self_model, "journal", None) if _self_model else None
    if journal is None:
        return {"available": False, "entries": []}
    try:
        entries = await asyncio.to_thread(lambda: (
            journal.today(domain=domain or None) if today
            else journal.recent(limit=limit, domain=domain or None)))
        return {"available": True, "count": len(entries), "entries": entries}
    except Exception as exc:
        return {"available": True, "error": str(exc), "entries": []}


@router.get("/autonomy/posture")
async def get_autonomy_posture(request: Request) -> dict:
    """Effective autonomy posture: the resolved value of every mode switch as
    the RUNNING process sees it, so unit-pinned env is never invisible."""
    try:
        from colonymind.config import env_bool, env_choice
        posture = {}
        for name, valid, fallback in (
            ("COLONYMIND_INTROSPECT_ENABLED", ("true", "false"), "false"),
            ("COLONYMIND_EXPECTATIONS", ("off", "on", "shadow", "live"), "on"),
        ):
            if valid == ("true", "false"):
                posture[name] = str(env_bool(name, fallback == "true")).lower()
            else:
                posture[name] = env_choice(name, valid, fallback)
        mind = _mind()
        posture["mind.autonomy"] = mind.level if mind is not None else "unknown"
        posture["mind.enabled"] = bool(mind is not None and mind.enabled)
        return {"available": True, "posture": posture}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


@router.get("/preferences")
async def get_owner_preferences(request: Request = None) -> dict:
    """Return the owner's learned communication preferences and rendered brief."""
    if _preference_learner is None:
        return {"available": False, "brief": "", "preferences": []}
    _require_perspective_owner(request)
    prefs = await _preference_learner.get_all_preferences()
    return {
        "available": True,
        "brief": _preference_learner.build_brief(),
        "sourced": _preference_learner.perspective.status() if getattr(_preference_learner, 'perspective', None) is not None else None,
        "preferences": [
            {
                "category": p.category, "key": p.key, "value": p.value,
                "confidence": p.confidence, "learned_from": p.learned_from,
                "last_updated": p.last_updated.isoformat(),
            }
            for p in prefs
        ],
    }


@router.post("/preferences/learn",
             dependencies=[Depends(refuse_hand_session)])
async def learn_owner_preference(body: dict, request: Request = None) -> dict:
    """Teach the owner-preference learner from text.

    Body: ``{"text": "be concise", "explicit": true, "force": false}``
      - explicit (default true): parse as a directive ("be concise"); learned at
        high confidence only if it reads like a communication directive.
      - force (with explicit): if the text isn't a recognized directive, still
        store it as explicit feedback.
      - explicit false: record as an observed behavior signal.
    """
    if _preference_learner is None:
        raise HTTPException(status_code=501, detail=_NOT_WIRED)
    _require_perspective_owner(request)
    if getattr(_preference_learner, 'perspective', None) is not None:
        source_id = body.get('source_id')
        if not isinstance(source_id, str) or not source_id:
            raise HTTPException(status_code=422, detail='Use an attributed owner turn and supply its source_id; free text cannot mint a source-backed correction.')
        changes = _preference_learner.learn_source(source_id)
        return {'learned': changes, 'brief': _preference_learner.build_brief()}
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="text required")
    learned: Any = None
    if body.get("explicit", True):
        hit = await _preference_learner.learn_directive(text)
        if hit is not None:
            learned = {"category": hit[0], "key": hit[1], "value": hit[2]}
        elif body.get("force"):
            await _preference_learner.learn_from_feedback("communication_style", text)
            learned = {"category": "communication_style", "key": "parsed"}
    else:
        await _preference_learner.learn_from_behavior(text)
        learned = {"category": "behavior"}
    return {"learned": learned, "brief": _preference_learner.build_brief()}


def _require_perspective_owner(request):
    if request is not None:
        from colonymind.identity import get_owner_contact_id
        owner = get_owner_contact_id()
        if not owner:
            raise HTTPException(status_code=503, detail='Owner identity is not configured')
        resolve_request_person(request, claimed_person_id=owner)


# ---------------------------------------------------------------------------
# Secrets: the sealed store (docs/SECRETS.md)
# ---------------------------------------------------------------------------

_secrets_vault = None


def set_secrets_vault(vault) -> None:
    """The sealed store turn capture seals into (``colonymind.secrets.vault``); None unwires it."""
    global _secrets_vault
    _secrets_vault = vault


def _memory_context_selector():
    """Keep one selector per active reranker, also usable without graph storage."""
    global _context_recall_selector
    if _context_recall_selector is None or _context_recall_selector[0] is not _reranker:
        from colonymind.memory.selection import RecallSelector
        from colonymind.memory.recall import provider_calibration_metadata
        provider = _reranker
        selector = RecallSelector(
            provider.rerank if provider is not None else None,
            calibration_metadata=(lambda: provider_calibration_metadata(provider))
                if provider is not None else None,
            logger=logger)
        _context_recall_selector = (provider, selector)
    return _context_recall_selector[1]


def set_reranker(reranker) -> None:
    global _reranker
    _reranker = reranker


def _map_initiative_to_schema(i) -> AgentSnapshotInitiative:
    """Map an Initiative model to the AgentSnapshotInitiative schema."""
    from colonymind.initiatives.models import OWNER_KINDS
    if i is None or getattr(i, "kind", None) in OWNER_KINDS:
        raise HTTPException(404, "Initiative not found")
    return AgentSnapshotInitiative(
        id=i.id,
        type=i.type,
        description=i.description,
        priority=i.priority,
        status=i.status,
        rationale=i.rationale,
        action_hint=i.action_hint,
        entity_id=i.entity_id,
        dedup_key=i.dedup_key,
        created_at=i.created_at.isoformat() if i.created_at else "",
        expires_at=i.expires_at.isoformat() if i.expires_at else None,
        assigned_agent_id=i.assigned_agent_id,
        acknowledged_at=i.acknowledged_at.isoformat() if i.acknowledged_at else None,
        completed_at=i.completed_at.isoformat() if i.completed_at else None,
        failed_at=i.failed_at.isoformat() if i.failed_at else None,
        failed_reason=i.failed_reason,
    )


# ---------------------------------------------------------------------------
# Retrieval wiring
# ---------------------------------------------------------------------------

_reranker = None
_context_recall_selector = None
# ---------------------------------------------------------------------------
# Agent Bridge status
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Commitment Tracking
# ---------------------------------------------------------------------------

@router.post("/commitments", status_code=status.HTTP_201_CREATED)
def create_commitment(body: CommitmentCreateRequest) -> CommitmentResponse:
    """Create a new commitment. With `dedupe` set, an open commitment for the
    same person that already says the same thing is returned instead of
    creating a twin (response carries deduped=true)."""
    if _commitment_store is None:
        raise HTTPException(status_code=501, detail="Commitment tracking not initialized")

    try:
        result = _commitment_store.create(
            person_id=body.person_id,
            description=body.description,
            due_at=body.due_at,
            priority=body.priority,
            source_type=body.source_type,
            source_context=body.source_context,
            metadata=body.metadata,
            dedupe=body.dedupe,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if result.get("deduped"):
        return CommitmentResponse(**result)

    try:
        from colonymind.events.broadcaster import emit as _emit
        _emit("commitment.created", {
            "commitment_id": result["id"],
            "person_id": result["person_id"],
            "description": result["description"],
        })
    except Exception:
        pass
    return CommitmentResponse(**result)


@router.get("/commitments", response_model=CommitmentListResponse, tags=[PUBLIC])
async def list_commitments(
    person_id: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None, alias="status"),
    overdue_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> CommitmentListResponse:
    """List commitments with optional filters."""
    async def read():
        return await _read_list_commitments(person_id, status_filter, overdue_only, limit, offset)
    if _commitment_store is None:
        raise HTTPException(status_code=501, detail="Commitment tracking not initialized")
    result = await _commitment_store._fenced_read(read, empty=None)
    return result if result is not None else CommitmentListResponse(commitments=[], total=0, limit=limit, offset=offset)


async def _read_list_commitments(
    person_id: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None, alias="status"),
    overdue_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> CommitmentListResponse:
    if _commitment_store is None:
        raise HTTPException(status_code=501, detail="Commitment tracking not initialized")

    statuses = [s.strip() for s in status_filter.split(",")] if status_filter else None

    # When "overdue" is requested, get commitments that are actually overdue
    # (past due_date + still pending), not just ones already transitioned
    if statuses and "overdue" in statuses:
        try:
            overdue = _commitment_store.get_overdue()
            other_statuses = [s for s in statuses if s != "overdue"]
            if other_statuses:
                result = _commitment_store.list(
                    person_id=person_id,
                    status=other_statuses,
                    overdue_only=False,
                    limit=limit,
                    offset=offset,
                )
                # Merge
                other_items = result if isinstance(result, list) else result.get("commitments", [])
                all_items = overdue + other_items
            else:
                all_items = overdue
            return CommitmentListResponse(
                commitments=all_items, total=len(all_items),
                limit=limit, offset=offset,
            )
        except Exception as exc:
            logger.warning("get_overdue failed: %s", exc)

    result = _commitment_store.list(
        person_id=person_id,
        status=statuses,
        overdue_only=overdue_only,
        limit=limit,
        offset=offset,
    )
    return CommitmentListResponse(**result)


@router.get("/commitments/{commitment_id}", response_model=CommitmentResponse, tags=[PUBLIC])
def get_commitment(commitment_id: str) -> CommitmentResponse:
    """Get a single commitment by ID."""
    if _commitment_store is None:
        raise HTTPException(status_code=501, detail="Commitment tracking not initialized")

    result = _commitment_store.get(commitment_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Commitment not found")
    return CommitmentResponse(**result)


@router.patch("/commitments/{commitment_id}", response_model=CommitmentResponse)
async def update_commitment(commitment_id: str, body: CommitmentUpdateRequest) -> CommitmentResponse:
    """Update a commitment. With `outcome` set this is a resolution: status
    is derived, the reason is recorded in metadata for the learning loop, and
    any workspace concern raised from this commitment is resolved too."""
    if _commitment_store is None:
        raise HTTPException(status_code=501, detail="Commitment tracking not initialized")

    if body.outcome is not None:
        # Resolution path: store.resolve() maps outcome -> status, records
        # {outcome, note, by, at} in metadata and emits the event itself.
        try:
            result = await asyncio.to_thread(
                _commitment_store.resolve,
                commitment_id,
                outcome=body.outcome,
                note=body.reason,
                resolved_by=body.resolved_by or "owner",
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        if result is None:
            raise HTTPException(status_code=404, detail="Commitment not found")
        return CommitmentResponse(**result)

    try:
        result = await asyncio.to_thread(
            _commitment_store.update,
            commitment_id=commitment_id,
            status=body.status,
            fulfilled_at=body.fulfilled_at,
            description=body.description,
            due_at=body.due_at,
            priority=body.priority,
            metadata=body.metadata,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if result is None:
        raise HTTPException(status_code=404, detail="Commitment not found")

    # Emit events for status changes
    if body.status == "fulfilled":
        try:
            from colonymind.events.broadcaster import emit as _emit
            _emit("commitment.fulfilled", {
                "commitment_id": result["id"],
                "person_id": result["person_id"],
            })
        except Exception:
            pass
    elif body.status == "cancelled":
        try:
            from colonymind.events.broadcaster import emit as _emit
            _emit("commitment.cancelled", {
                "commitment_id": result["id"],
                "person_id": result["person_id"],
            })
        except Exception:
            pass

    return CommitmentResponse(**result)


@router.get("/commitments/stats/resolution")
async def commitment_resolution_stats(days: int = Query(30, ge=1, le=365)) -> dict:
    """How commitments created in the window got resolved, per source_type —
    the calibration signal for whatever generates items (introspection,
    cognition, agents): a source whose items keep getting cancelled as
    invalid should get more conservative."""
    if _commitment_store is None:
        raise HTTPException(status_code=501, detail="Commitment tracking not initialized")
    try:
        def read():
            stats = _commitment_store.resolution_stats(days=days)
            stats["recent_rejections"] = _commitment_store.recent_rejections(limit=10)
            return stats
        return _commitment_store._fenced_read(read, empty={"days": days, "by_source": {}, "recent_rejections": []})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/commitments/{commitment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_commitment(commitment_id: str):
    """Delete a commitment. Only allowed for terminal states (fulfilled/cancelled)."""
    if _commitment_store is None:
        raise HTTPException(status_code=501, detail="Commitment tracking not initialized")

    deleted = _commitment_store.delete(commitment_id)
    if not deleted:
        existing = _commitment_store.get(commitment_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Commitment not found")
        else:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot delete commitment in '{existing['status']}' state. Cancel it first.",
            )


# ---------------------------------------------------------------------------
# Cognition Substrate
# ---------------------------------------------------------------------------

# ============================================================================
# Initiatives: the mind's intention rows
# ============================================================================

_initiative_store = None


def set_initiative_store(store) -> None:
    global _initiative_store
    _initiative_store = store


# --- Initiative Management ---

@router.post("/initiatives", response_model=InitiativeResponse)
async def create_initiative(body: InitiativeCreateRequest) -> InitiativeResponse:
    """Create a new initiative."""
    if _initiative_store is None:
        raise HTTPException(status_code=501, detail="Initiative store not initialized")

    try:
        initiative = _initiative_store.create(
            type=body.initiative_type,
            description=body.description,
            # Request priority is 0-100; the store holds 0.0-1.0.
            priority=body.priority / 100.0,
            timeout_seconds=body.timeout_seconds,
            dedup_key=body.dedup_key,
            entity_id=body.entity_id,
            preferred_agent_id=body.target_agent_id,
            context=body.context or None,
            presentation=body.presentation.model_dump(exclude_unset=True) if body.presentation else None,
            items=[item.model_dump(exclude_none=True) for item in body.items],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:  # timeline (v0.21.0)
        from colonymind.events.journal import append_event
        append_event("initiative.generated", {
            "initiative_id": getattr(initiative, "id", None),
            "contact_id": body.entity_id,
            "summary": body.description,
            "initiative_type": body.initiative_type,
        })
    except Exception:
        logger.debug("journal initiative.generated failed", exc_info=True)

    return _initiative_to_response(initiative)


@router.get("/initiatives", response_model=None)
async def list_initiatives(
    status: Optional[str] = Query(None),
    agent_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=1000),
    contract: Optional[int] = Query(None),
    cursor: Optional[str] = Query(None, max_length=4096),
    request: Request = None,
):
    """List initiatives with one optional status; omitted or empty is unfiltered.

    ``contract=2`` asks for the sectioned ideas catalog instead, exactly as ``/initiatives/ideas?contract=2``."""
    if _initiative_store is None:
        raise HTTPException(status_code=501, detail="Initiative store not initialized")
    if contract == 2:
        from colonymind.mind.ideas_build import owner_action
        owner_action(request)
        return _ideas_catalog_v2(limit, cursor)

    initiatives = _initiative_store.list(
        status=[status] if status else None,
        assigned_agent_id=agent_id,
        limit=limit,
    )
    last_responses = _initiative_store.last_responses([i.id for i in initiatives])

    # One malformed row must not fail the whole list.
    responses = []
    for initiative in initiatives:
        try:
            responses.append(_initiative_to_response(initiative, last_response=last_responses.get(initiative.id)))
        except (ValidationError, HTTPException) as exc:
            logger.warning("initiatives list: skipped %s (%s)", initiative.id, type(exc).__name__)
    return InitiativeListResponse(initiatives=responses, total=len(responses))


IDEAS_V2_BUDGET = 800_000


def _ideas_v2_card(row, state: Optional[str]) -> Optional[dict]:
    engine = getattr(_initiative_store, "_ideas", None)
    if engine is None or not engine.valid(row):
        return None
    return engine.card(row)


def _ideas_catalog_v2(limit: int, cursor: Optional[str]) -> dict:
    """A three-card hero followed by category rows, with no duplicate cards."""
    from colonymind.mind.ideas_presentation import KINDS
    for take in range(min(limit, 6), 0, -1):
        try:
            rows, next_cursor, has_built = _initiative_store.ideas_page(take, cursor)
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        cards = [card for row in rows if (card := _ideas_v2_card(row, None)) is not None]
        sections = []
        if cursor is None and len(cards) >= 3:
            sections.append({"id": "for-you", "title": "For you", "subtitle": "Ready for your review",
                             "presentation": {"layout": "hero"}, "cards": cards[:3]})
            cards = cards[3:]
        categories = {}
        for card in cards:
            category = KINDS[card["kind"]][2]
            if category not in categories:
                categories[category] = {"id": category, "title": card["secondaryLabel"], "subtitle": "",
                                        "presentation": {"layout": "rows"}, "cards": []}
            categories[category]["cards"].append(card)
        sections.extend(categories.values())
        value = {"sections": sections, "viewerState": {"hasBuiltIdea": has_built},
                 "pagination": {"nextCursor": next_cursor, "hasNextPage": next_cursor is not None},
                 "capabilities": {"selectedItems": True, "feedback": True, "engagement": True, "sharing": False}}
        if len(json.dumps(value, ensure_ascii=True).encode()) <= IDEAS_V2_BUDGET:
            return value
    raise HTTPException(507, "idea_card_exceeds_relay_budget")


@router.get("/initiatives/ideas")
async def list_ideas(limit: int = Query(50, ge=1, le=1000),
                     cursor: Optional[str] = Query(None, max_length=4096),
                     contract: Optional[int] = Query(None), request: Request = None) -> dict:
    """The bounded owner catalog; legacy callers use the same eligibility."""
    if _initiative_store is None:
        raise HTTPException(status_code=501, detail="Initiative store not initialized")
    from colonymind.mind.ideas_build import owner_action
    owner_action(request)
    if contract == 2:
        return _ideas_catalog_v2(limit, cursor)
    try:
        rows, next_cursor, has_built = _initiative_store.ideas_page(limit, cursor)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    cards = []
    for row in rows:
        card = _ideas_v2_card(row, None)
        if card:
            cards.append({"id": row.id, "initiative_type": row.type, "kind": card["kind"],
                          "title": card["content"]["title"], "summary": card["content"]["summary"],
                          "status": row.status, **({"buildStatus": card["buildStatus"]} if "buildStatus" in card else {}), "presentation": card["presentation"]})
    return {"sections": [{"id": "for-you", "title": "For you", "subtitle": "",
                           "presentation": {"layout": "hero"}, "cards": cards}] if cards else [],
            "viewer_state": {"has_built_idea": has_built},
            "pagination": {"next_cursor": next_cursor, "has_next_page": next_cursor is not None}}


@router.get("/initiatives/media/{handle}")
def idea_media(handle: str, request: Request) -> Response:
    from colonymind.mind.ideas_build import owner_action
    from colonymind.mind.ideas_presentation import media
    owner_action(request)
    try:
        content = media(handle)
    except ValueError as exc:
        raise HTTPException(404, "Idea media not found") from exc
    return Response(content, media_type="image/svg+xml", headers={
        "Cache-Control": "private, max-age=86400", "X-Content-Type-Options": "nosniff",
        "Content-Security-Policy": "default-src 'none'; style-src 'none'; sandbox"})


@router.get("/initiatives/ideas/state")
async def idea_states(ids: str = Query(...), request: Request = None) -> dict:
    """Read authoritative feed-card state for up to 100 initiative ids."""
    if _initiative_store is None:
        raise HTTPException(status_code=501, detail="Initiative store not initialized")
    from colonymind.mind.ideas_build import owner_action
    owner_action(request)
    try:
        return {"states": _initiative_store.idea_states([value.strip() for value in ids.split(",")])}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/initiatives/ideas/associate", dependencies=[Depends(refuse_hand_session)])
async def associate_feed_ideas(request: Request) -> dict:
    """Project grounded feed evidence into the existing idea_action carrier."""
    from colonymind.mind.ideas_build import owner_action
    from colonymind.mind.ideas_association import AssociationRequest, associate
    owner_action(request)
    if _initiative_store is None:
        raise HTTPException(501, "Initiative store not initialized")
    try:
        body = AssociationRequest.model_validate(await request.json())
    except (ValueError, TypeError, UnicodeError):
        # Never reflect untrusted feed inputs (including nested extras) in diagnostics.
        raise HTTPException(422, "invalid_feed_idea_association") from None
    return associate(getattr(_initiative_store, "_ideas", None), body.cards)


def _share_response(share: dict) -> dict:
    from urllib.parse import urlsplit
    base = os.environ.get("COLONYMIND_SHARE_BASE_URL", "").strip().rstrip("/")
    try:
        parsed = urlsplit(base)
        valid = parsed.scheme in {"https", "http"} and parsed.netloc and not (
            parsed.username or parsed.password or parsed.query or parsed.fragment)
    except ValueError:
        valid = False
    return {"share_id": share["share_id"],
            "share_url": f"{base}/v1/host/initiatives/shares/{share['share_id']}" if valid else None,
            "content": {"title": share["title"], "summary": share["summary"]}}


@router.get("/initiatives/shares/{share_id}")
async def resolve_initiative_share(share_id: str) -> dict:
    """Read a minimal share through the same authenticated host API as its source."""
    if _initiative_store is None:
        raise HTTPException(status_code=501, detail="Initiative store not initialized")
    share = _initiative_store.resolve_share(share_id)
    if share is None:
        raise HTTPException(status_code=404, detail="Share not found")
    return _share_response(share)


@router.get("/initiatives/{initiative_id}", response_model=None,
            response_model_exclude_unset=True)
async def get_initiative(initiative_id: str, contract: Optional[int] = Query(None), request: Request = None):
    """Get initiative details."""
    if _initiative_store is None:
        raise HTTPException(status_code=501, detail="Initiative store not initialized")

    initiative = _initiative_store.get(initiative_id)
    if initiative is None or getattr(initiative, "kind", None) in {"owner_goal", "owner_subgoal", "owner_goal_progress"}:
        raise HTTPException(status_code=404, detail="Initiative not found")

    from colonymind.mind.ideas import marked
    if marked(initiative) or contract == 2:
        from colonymind.mind.ideas_build import owner_action
        owner_action(request)
        engine = getattr(_initiative_store, "_ideas", None)
        if engine is None or not engine.valid(initiative, accepted=bool(initiative.context.get("idea", {}).get("execution"))):
            raise HTTPException(404, "Idea not found")
        if contract == 2:
            card = engine.card(initiative)
            if len(json.dumps({"card": card}, ensure_ascii=True).encode()) > IDEAS_V2_BUDGET:
                raise HTTPException(507, "idea_card_exceeds_relay_budget")
            return {"card": card}

    last_responses = _initiative_store.last_responses([initiative_id])
    value = _initiative_to_response(initiative, last_response=last_responses.get(initiative_id)).model_dump()
    # Omitted extensions mean empty on legacy rows, preserving their exact wire shape.
    if initiative.presentation or initiative.items:
        value.update(presentation=initiative.presentation, items=initiative.items)
    origin = _initiative_store.last_response_origin(initiative_id)
    if origin is not None:
        value["origin"] = origin
    return JSONResponse(content=InitiativeDetailResponse(**value).model_dump(mode="json", exclude_unset=True))


@router.post("/initiatives/{initiative_id}/share", dependencies=[Depends(refuse_hand_session)])
async def share_initiative(initiative_id: str) -> dict:
    initiative = _idea_or_404(initiative_id)
    from colonymind.mind.ideas import marked
    if marked(initiative):
        raise HTTPException(409, "private_ideas_cannot_be_shared")
    return _share_response(_initiative_store.share(
        initiative_id, initiative.description[:100], initiative.description))


def _idea_or_404(initiative_id: str):
    if _initiative_store is None:
        raise HTTPException(status_code=501, detail="Initiative store not initialized")
    initiative = _initiative_store.get(initiative_id)
    from colonymind.initiatives.models import OWNER_KINDS
    if initiative is None or getattr(initiative, "kind", None) in OWNER_KINDS:
        raise HTTPException(status_code=404, detail="Initiative not found")
    return initiative


class IdeaExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["full", "selectedItems"]
    itemIds: Optional[List[str]] = Field(default=None, max_length=32)


@router.post("/initiatives/{initiative_id}/execute", dependencies=[Depends(refuse_hand_session)])
async def execute_idea(initiative_id: str, body: IdeaExecuteRequest, request: Request) -> dict:
    from colonymind.mind.ideas import marked
    from colonymind.mind.ideas_build import owner_action
    owner_action(request)
    row = _idea_or_404(initiative_id)
    engine = getattr(_initiative_store, "_ideas", None)
    if not marked(row) or engine is None:
        raise HTTPException(404, "Idea not found")
    result = await engine.build(row, body.mode, body.itemIds, request)
    return {key: result[key] for key in ("cardId", "executionId", "status")}


IDEA_V2_CONTEXT = {"target": "IDEA", "surface": "idea_cards", "lane": "explore"}


def _idea_v2_context(body: dict, allowed: set) -> None:
    """The client's contract-2 card context: exact keys, fixed target/surface/lane and a bounded anchor."""
    anchor = body.get("anchor")
    if (set(body) - allowed or any(body.get(key) != value for key, value in IDEA_V2_CONTEXT.items())
            or not isinstance(anchor, str) or len(anchor) > 128):
        raise HTTPException(status_code=422, detail="invalid idea card context")


@router.post("/initiatives/{initiative_id}/engagement", dependencies=[Depends(refuse_hand_session)])
async def initiative_engagement(initiative_id: str, body: dict = Body(...), request: Request = None) -> dict:
    initiative = _idea_or_404(initiative_id)
    from colonymind.mind.ideas import marked
    if marked(initiative):
        from colonymind.mind.ideas_build import owner_action
        owner_action(request)
    contract_2 = "event" in body
    if contract_2:
        # Contract 2: the client sends no receipt id, so each report is its own receipt.
        _idea_v2_context(body, {"event", "visibleFraction", *IDEA_V2_CONTEXT, "anchor",
                                "value", "anchor_kind", "metadata"})
        fraction = body.get("visibleFraction")
        if body["event"] not in ("visible", "open") or type(fraction) not in (int, float) or not 0 <= fraction <= 1:
            raise HTTPException(status_code=422, detail="invalid idea engagement")
        if "value" in body:
            value = body["value"]
            if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
                raise HTTPException(status_code=422, detail="invalid idea engagement")
        if "anchor_kind" in body:
            anchor_kind = body["anchor_kind"]
            if not isinstance(anchor_kind, str) or len(anchor_kind) > 80:
                raise HTTPException(status_code=422, detail="invalid idea engagement")
        if "metadata" in body:
            metadata = body["metadata"]
            if (not isinstance(metadata, dict) or len(metadata) > 16
                    or any(not isinstance(key, str) or len(key) > 80
                           or type(value) not in (str, bool, int, float, type(None))
                           or (isinstance(value, str) and len(value) > 500)
                           for key, value in metadata.items())):
                raise HTTPException(status_code=422, detail="invalid idea engagement")
            if len(json.dumps(metadata).encode("utf-8")) > 4096:
                raise HTTPException(status_code=422, detail="invalid idea engagement")
        body = {"receipt_id": str(uuid.uuid4()), "kind": "impression" if body["event"] == "visible" else "open",
                "visible_fraction": float(fraction), "surface": body["surface"], "anchor": body["anchor"] or "none"}
    try:
        receipt = InitiativeEngagementRequest.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_url=False, include_input=False)) from None
    try:
        recorded = _initiative_store.record_engagement(initiative_id, receipt.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {**({"ok": True} if contract_2 else {}), "receipt_id": receipt.receipt_id, "recorded": recorded}


@router.post("/initiatives/{initiative_id}/feedback", dependencies=[Depends(refuse_hand_session)])
async def initiative_feedback(initiative_id: str, request: Request, body: dict = Body(...)) -> dict:
    initiative = _idea_or_404(initiative_id)
    from colonymind.mind.ideas import marked
    if marked(initiative):
        from colonymind.mind.ideas_build import owner_action
        owner_action(request)
    ok = {"ok": True} if "signal" in body else {}
    if ok:
        _idea_v2_context(body, {"signal", "reason", *IDEA_V2_CONTEXT, "anchor"})
        reason = body.get("reason")
        if body["signal"] not in ("UP", "DOWN") or (reason is not None and not isinstance(reason, str)):
            raise HTTPException(status_code=422, detail="invalid idea feedback")
        body = {"direction": body["signal"].lower(), "surface": body["surface"],
                **({"reason": reason} if isinstance(reason, str) and len(reason) <= 600 else {})}
    try:
        parsed = InitiativeFeedbackRequest.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_url=False, include_input=False)) from None
    details = parsed.model_dump()
    body = parsed
    if body.direction == "down":
        return {**ok, **(await respond_to_initiative(initiative_id, action="dismissed", details=details,
                                                         request=request))}
    # An explicit topic preference uses the existing type learner, without a lifecycle transition.
    try:
        if _feedback_store is not None:
            from colonymind.mind.outcomes import Outcomes
            for key in Outcomes._feedback_keys(initiative):
                _feedback_store.record(key, "preferred", source=initiative_id)
    except Exception as exc:
        logger.warning("Failed to record initiative preference %s: %s", initiative_id, exc)
    _initiative_store.log_history(initiative_id, action="feedback_up", agent_id="openclaw", details=details)
    return {**ok, "success": True, "initiative_id": initiative_id, "status": initiative.status}


@router.post("/initiatives/{initiative_id}/complete")
async def complete_initiative(
    initiative_id: str,
    body: InitiativeCompleteRequest,
) -> Dict[str, Any]:
    """Mark initiative as completed."""
    if _initiative_store is None:
        raise HTTPException(status_code=501, detail="Initiative store not initialized")

    initiative = _initiative_store.get(initiative_id)
    if initiative is None or getattr(initiative, "kind", None) in {"owner_goal", "owner_subgoal", "owner_goal_progress"}:
        raise HTTPException(status_code=404, detail="Initiative not found")

    if initiative.assigned_agent_id != body.agent_id:
        raise HTTPException(status_code=403, detail="Not assigned to this agent")

    _initiative_store.complete(initiative_id, body.agent_id, body.result.get("result"), body.result)

    return {"status": "completed", "initiative_id": initiative_id}


@router.post("/initiatives/{initiative_id}/fail")
async def fail_initiative(
    initiative_id: str,
    body: InitiativeFailRequest,
) -> Dict[str, Any]:
    """Mark initiative as failed."""
    if _initiative_store is None:
        raise HTTPException(status_code=501, detail="Initiative store not initialized")

    initiative = _initiative_store.get(initiative_id)
    if initiative is None or getattr(initiative, "kind", None) in {"owner_goal", "owner_subgoal", "owner_goal_progress"}:
        raise HTTPException(status_code=404, detail="Initiative not found")

    if initiative.assigned_agent_id != body.agent_id:
        raise HTTPException(status_code=403, detail="Not assigned to this agent")

    _initiative_store.fail(initiative_id, body.agent_id, body.error_message)

    return {"status": "failed", "initiative_id": initiative_id}


@router.patch("/initiatives/{initiative_id}/priority")
async def update_initiative_priority(
    initiative_id: str,
    body: InitiativePriorityRequest,
) -> Dict[str, Any]:
    """Update initiative priority."""
    if _initiative_store is None:
        raise HTTPException(status_code=501, detail="Initiative store not initialized")

    initiative = _initiative_store.get(initiative_id)
    if initiative is None or getattr(initiative, "kind", None) in {"owner_goal", "owner_subgoal", "owner_goal_progress"}:
        raise HTTPException(status_code=404, detail="Initiative not found")

    _initiative_store.update(initiative_id, priority=body.priority)

    return {"status": "updated", "initiative_id": initiative_id, "priority": body.priority}


@router.post("/initiatives/{initiative_id}/retry")
async def retry_initiative(initiative_id: str, request: Request = None) -> Dict[str, Any]:
    """Retry a failed initiative."""
    if _initiative_store is None:
        raise HTTPException(status_code=501, detail="Initiative store not initialized")

    initiative = _initiative_store.get(initiative_id)
    if initiative is None or getattr(initiative, "kind", None) in {"owner_goal", "owner_subgoal", "owner_goal_progress"}:
        raise HTTPException(status_code=404, detail="Initiative not found")

    if initiative.status != "failed":
        raise HTTPException(status_code=400, detail="Can only retry failed initiatives")

    retried = _initiative_store.retry(initiative_id, request_authority(request).principal_id)
    if retried is None:
        raise HTTPException(status_code=400, detail="Can only retry failed initiatives")

    return {"status": "pending", "initiative_id": initiative_id}


@router.post("/initiatives/{initiative_id}/context/refresh", response_model=InitiativeResponse)
async def refresh_initiative_context(initiative_id: str) -> InitiativeResponse:
    """Rebuild the context snapshot for one initiative's subject (v0.16.0).

    Volatile initiative types (calendar, coding, system, agent_action)
    carry context that can go stale while the initiative sits in the
    queue. The agent calls this before acting when the snapshot's
    ``context_captured_at`` is older than the type's freshness TTL.
    Durable types return their stored snapshot unchanged.
    """
    if _initiative_store is None:
        raise HTTPException(status_code=501, detail="Initiative store not initialized")

    initiative = _initiative_store.get(initiative_id)
    if initiative is None or getattr(initiative, "kind", None) in {"owner_goal", "owner_subgoal", "owner_goal_progress"}:
        raise HTTPException(status_code=404, detail="Initiative not found")

    from colonymind.initiatives.context_freshness import DURABLE, durability_for

    # No per-entity context loader remains: the mind's concerns carry their own evidence.
    fresh = None

    if fresh is None:
        if durability_for(initiative.type) == DURABLE:
            # Durable context: the creation-time snapshot is still valid.
            return _initiative_to_response(initiative)
        raise HTTPException(
            status_code=501,
            detail=(
                f"No per-entity context loader registered for volatile "
                f"type '{initiative.type}' — context cannot be refreshed"
            ),
        )

    # Volatile auto-close: if the refreshed snapshot shows the condition
    # has cleared (CI green again, service recovered, meeting over), the
    # initiative retires itself instead of surfacing stale work.
    condition_cleared = bool(fresh.pop("condition_cleared", False))
    if condition_cleared and initiative.is_active:
        _initiative_store.update(
            initiative_id,
            context=fresh,
            status="cancelled",
            cancelled_at=now_utc().isoformat(),
            cancelled_by="context_refresh",
            cancelled_reason="condition_cleared",
            stale_reason="condition_cleared",
        )
        updated = _initiative_store.get(initiative_id)
        return _initiative_to_response(updated or initiative)

    updated = _initiative_store.update(initiative_id, context=fresh)
    return _initiative_to_response(updated or initiative)


@router.delete("/initiatives/{initiative_id}")
async def cancel_initiative(initiative_id: str) -> Dict[str, Any]:
    """Cancel an initiative."""
    if _initiative_store is None:
        raise HTTPException(status_code=501, detail="Initiative store not initialized")

    initiative = _initiative_store.get(initiative_id)
    if initiative is None or getattr(initiative, "kind", None) in {"owner_goal", "owner_subgoal", "owner_goal_progress"}:
        raise HTTPException(status_code=404, detail="Initiative not found")

    _initiative_store.cancel(initiative_id, cancelled_by="api")

    return {"status": "cancelled", "initiative_id": initiative_id}


# --- Initiative Helpers ---

def _initiative_to_response(initiative, *, last_response: Optional[Dict[str, str]] = None) -> InitiativeResponse:
    """Convert StoredInitiative to InitiativeResponse."""
    from colonymind.initiatives.models import OWNER_KINDS
    if initiative is None or getattr(initiative, "kind", None) in OWNER_KINDS:
        raise HTTPException(404, "Initiative not found")
    # Handle result as dict if it's a string or None
    result_dict = None
    if initiative.result:
        if isinstance(initiative.result, dict):
            result_dict = initiative.result
        elif isinstance(initiative.result, str):
            result_dict = {"result": initiative.result}

    from colonymind.initiatives.context_freshness import durability_for

    return InitiativeResponse(
        id=initiative.id,
        initiative_type=initiative.type,
        # Title is the ACTION ("Check in with Jordan"), not the reason.
        # The rationale lives in the context dict.
        title=initiative.description[:100],
        description=initiative.description,
        priority=int(initiative.priority * 100) if initiative.priority else 0,
        status=initiative.status,
        timeout_seconds=initiative.timeout_seconds,
        # NULL for rows created before the v0.16.0 context migration.
        context=initiative.context or {},
        context_durability=durability_for(initiative.type),
        entity_id=initiative.entity_id,
        target_agent_id=initiative.assigned_agent_id or initiative.preferred_agent_id,
        assigned_agent_id=initiative.assigned_agent_id,
        dedup_key=initiative.dedup_key,
        result=result_dict,
        error_message=initiative.failed_reason,
        created_at=initiative.created_at.isoformat() if initiative.created_at else "",
        acknowledged_at=initiative.acknowledged_at.isoformat() if initiative.acknowledged_at else None,
        completed_at=initiative.completed_at.isoformat() if initiative.completed_at else None,
        failed_at=initiative.failed_at.isoformat() if initiative.failed_at else None,
        expires_at=initiative.expires_at.isoformat() if initiative.expires_at else None,
        last_response=last_response,
    )


@router.post("/initiatives/{initiative_id}/respond",
             dependencies=[Depends(refuse_hand_session)])
async def respond_to_initiative(
    initiative_id: str,
    action: Optional[str] = Body(None),
    details: Optional[dict] = Body(None),
    request: Request = None,
    mode: Annotated[Optional[Literal["full", "selected_items", "selectedItems"]], Body()] = None,
    item_ids: Annotated[Optional[List[str]], Body()] = None,
    origin: Annotated[Optional[InitiativeOrigin], Body()] = None,
    itemIds: Annotated[Optional[List[str]], Body()] = None,
) -> Dict[str, Any]:
    """Record the owner's response to an initiative as feedback.

    The client's contract-2 execution body (``action`` approved, ``mode`` full or selectedItems, ``itemIds``)
    answers with an execution receipt: ``cardId``, ``executionId`` and ``status`` accepted, with the
    initiative's own status as ``initiative_status``. Every other body keeps its response."""
    # The contract-2 execution body is exactly {action: approved, mode: full|selectedItems[, itemIds]}.
    contract_2 = (action == "approved" and details is None and origin is None and item_ids is None
                  and (mode == "selectedItems" or (mode == "full" and itemIds is None)))
    if itemIds is not None and not contract_2:
        raise HTTPException(status_code=422, detail="itemIds requires the selectedItems execution body")
    if mode == "selectedItems":
        if not contract_2:
            raise HTTPException(status_code=422, detail="selectedItems requires the contract-2 execution body")
        mode, item_ids = "selected_items", itemIds
    execution_id = str(uuid.uuid4()) if contract_2 else None
    if _initiative_store is None:
        raise HTTPException(status_code=501, detail="Initiative store not initialized")
    initiative = _initiative_store.get(initiative_id)
    if initiative is None or getattr(initiative, "kind", None) in {"owner_goal", "owner_subgoal", "owner_goal_progress"}:
        raise HTTPException(status_code=404, detail="Initiative not found")

    if action is None and mode is None:
        raise HTTPException(status_code=422, detail="action or mode required")
    from colonymind.mind.ideas import marked
    if marked(initiative):
        from colonymind.mind.ideas_build import owner_action
        owner_action(request)
        engine = getattr(_initiative_store, "_ideas", None)
        if engine is None:
            raise HTTPException(501, "ideas_unavailable")
        if action == "dismissed" and mode is None:
            if initiative.context["idea"].get("execution"):
                raise HTTPException(409, "built_idea_requires_hands_controls")
            _initiative_store.transition(initiative_id, "cancelled", action="idea_dismissed")
            from colonymind import recorder
            recorder.mark(f"initiative:{initiative_id}", "owner_declined")
            try:
                if _feedback_store is not None:
                    from colonymind.mind.outcomes import Outcomes
                    for key in Outcomes._feedback_keys(initiative):
                        _feedback_store.record(key, "dismissed", source=initiative_id)
            except Exception as exc:
                logger.warning("Failed to record idea dismissal %s: %s", initiative_id, exc)
            engine.trigger()
            return {"success": True, "initiative_id": initiative_id, "status": "cancelled"}
        if action not in {None, "approved"}:
            raise HTTPException(422, "unsupported_idea_action")
        return await engine.build(initiative, mode, item_ids, request)
    if contract_2:
        raise HTTPException(404, "Idea not found")
    if mode is not None:
        if action is not None and action != "approved":
            raise HTTPException(status_code=422, detail="execution mode requires approved action")
        if mode == "selected_items":
            selectable = {item["id"] for item in initiative.items if item.get("selectable")}
            if not item_ids or len(set(item_ids)) != len(item_ids) or not set(item_ids) <= selectable:
                raise HTTPException(status_code=422, detail="item_ids must be unique selectable items of this initiative")
        elif item_ids is not None:
            raise HTTPException(status_code=422, detail="item_ids requires selected_items mode")
        action = "approved"
        details = {**(details or {}), "mode": mode, "item_ids": item_ids,
                   **({"execution_id": execution_id} if execution_id else {})}
    elif item_ids is not None:
        raise HTTPException(status_code=422, detail="item_ids requires selected_items mode")

    # Update status based on action
    status_map = {
        "acknowledged": "acknowledged",
        "dismissed": "cancelled",
        "snoozed": "pending",
        "approved": "acknowledged",
        "actioned": "completed",
    }
    new_status = status_map.get(action)
    if new_status:
        _initiative_store.update(initiative_id, status=new_status)

    # Close the loop into TypeFeedbackStore. The caller is the model reporting what
    # became of an initiative it was shown. A disposal (dismissed, snoozed,
    # acknowledged) is the outcome this store just recorded and counts, keyed by
    # the initiative so repeating it changes nothing; a claim that the owner
    # approved or acted on it is unverified and is not the owner's verdict, so it
    # does not boost the type. Best-effort: recording never fails the respond.
    _FEEDBACK_OUTCOME_MAP = {
        "dismissed": "dismissed",
        "snoozed": "snoozed",
        "acknowledged": "acknowledged",
    }
    try:
        outcome = _FEEDBACK_OUTCOME_MAP.get(action)
        itype = getattr(initiative, "type", None)
        if _feedback_store is not None and outcome is not None and itype:
            from colonymind.mind.outcomes import Outcomes
            for key in Outcomes._feedback_keys(initiative):
                _feedback_store.record(key, outcome, source=initiative_id)
    except Exception as exc:
        logger.warning(
            "Failed to record type feedback for initiative %s (action=%s): %s",
            initiative_id, action, exc,
        )

    _initiative_store.log_history(
        initiative_id,
        action=f"llm_{action}",
        agent_id="openclaw",
        details=details or {},
        origin=origin.model_dump(exclude_unset=True) if origin is not None else None,
    )
    if execution_id is not None:
        return {"success": True, "initiative_id": initiative_id, "cardId": initiative_id,
                "executionId": execution_id, "status": "accepted",
                "initiative_status": new_status or initiative.status}
    return {
        "success": True,
        "initiative_id": initiative_id,
        "status": new_status or initiative.status,
    }


# --- Agent Snapshot Endpoints ---

@router.get("/agent-snapshot", response_model=AgentSnapshotResponse)
async def agent_snapshot() -> AgentSnapshotResponse:
    """Return a comprehensive snapshot of ColonyMind state for agent evaluation."""
    now = now_utc()

    # Telemetry
    thresholds = {"tick": _tick_stale_hours(_mind())}
    telemetry_dict = await _telemetry.to_dict(thresholds) if _telemetry else {}

    # Pending initiatives (top 20 by priority)
    pending = []
    if _initiative_store is not None:
        pending = _initiative_store.list(status=["pending"], limit=20)

    # Recently completed (top 10 by priority — store orders by priority DESC)
    recent = []
    if _initiative_store is not None:
        recent = _initiative_store.list(status=["completed"], limit=10)

    # Failed initiatives
    failed = []
    if _initiative_store is not None:
        failed = _initiative_store.list(status=["failed"], limit=10)

    # Compute last tick age
    tick_age = None
    if _telemetry is not None and _telemetry.last_tick_at is not None:
        tick_age = (now - _telemetry.last_tick_at).total_seconds() / 60

    # Flags: high-signal items the agent should know about
    flags = []
    if failed:
        flags.append("failed_initiatives")
    if pending and any(i.priority > 0.8 for i in pending):
        flags.append("high_priority_pending")
    if tick_age and tick_age > 30:
        flags.append("stale_autonomy_loop")

    return AgentSnapshotResponse(
        timestamp=now.isoformat(),
        telemetry=telemetry_dict,
        pending_initiatives=[_map_initiative_to_schema(i) for i in pending],
        pending_count=len(pending),
        assigned_count=(
            _initiative_store.count(status=["assigned"]) if _initiative_store else 0
        ),
        failed_count=len(failed),
        recently_completed=[_map_initiative_to_schema(i) for i in recent],
        autonomy_mode=_mind_posture()[0],
        autonomy_running=_mind_posture()[1],
        last_tick_age_minutes=tick_age,
        flags=flags,
    )


@router.post("/agent-snapshot/record-outreach", response_model=RecordOutreachResponse)
async def record_outreach(body: RecordOutreachRequest) -> RecordOutreachResponse:
    """Record that the agent proactively messaged the owner."""
    now = now_utc()
    outreach_at = now.isoformat()
    if _telemetry is not None:
        await _telemetry.touch("last_agent_outreach_at")
        if _telemetry.last_agent_outreach_at is not None:
            outreach_at = _telemetry.last_agent_outreach_at.isoformat()
    logger.info(
        "Agent outreach recorded: agent=%s channel=%s reason=%s",
        body.agent_id, body.channel, body.reason,
    )
    try:  # timeline (v0.21.0)
        from colonymind.events.journal import append_event
        append_event("outreach.sent", {
            "contact_id": getattr(body, "contact_id", None),
            "channel": body.channel,
            "reason": body.reason,
            "summary": body.reason,
        })
    except Exception:
        logger.debug("journal outreach.sent failed", exc_info=True)
    try:
        # Proactive outreach carries an already-resolved target contact; log
        # it outbound so reciprocity accounting stays whole. Skip entirely
        # without a contact (never attribute an outbound to a placeholder).
        _oc = getattr(body, "contact_id", None)
        if _comms_log is not None and _oc and _oc not in ("system", "default"):
            _comms_log.log(_oc, channel=body.channel or "direct",
                           direction="out", summary=(body.reason or "")[:300])
    except Exception as error:
        from colonymind.health_counters import bump
        bump("writes_dropped.comms_ledger_outbound")
        logger.warning("comms ledger outbound log failed (%s, sqlite_errorname=%s)",
                       type(error).__name__, getattr(error, "sqlite_errorname", None))
    return RecordOutreachResponse(
        recorded_at=now.isoformat(),
        last_agent_outreach_at=outreach_at,
    )
