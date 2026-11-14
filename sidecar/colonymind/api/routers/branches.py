"""Owner-scoped branch capture, listing, consults and reaction admission.

The adapter supplies the session's structured sender independently of tool arguments.
The credential must carry that sender's platform and the sender must resolve to the
owner. The local CLI is the sole senderless owner lane, under its credential grant.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager, suppress
import logging
from typing import Annotated, Any

from fastapi import Depends, APIRouter, HTTPException, Query, Request
from colonymind.hands.sessions import refuse_hand_session
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from colonymind.api.auth import (
    owner_person_id,
    owner_platform_granted,
    request_authority,
    resolve_request_person,
    speaks_for_owner,
)
from colonymind.api.routers import host


logger = logging.getLogger(__name__)
# Assumed retention sweep cadence: hourly, including while the faculty is off.
SWEEP_SECONDS = 3600


@asynccontextmanager
async def _lifespan(app):
    def prune():
        _store().prune()

    async def sweep():
        while True:
            try:
                # Pruning leaves durable expiry markers; the hands controller's maintenance drains them.
                await asyncio.to_thread(prune)
            except Exception as exc:
                logger.warning("working-memory retention failed (%s)", type(exc).__name__)
            await asyncio.sleep(SWEEP_SECONDS)

    task = asyncio.create_task(sweep(), name="working-memory-retention")
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


router = APIRouter(prefix="/v1/memory/branches", tags=["working-memory"], lifespan=_lifespan)
# Assumed envelope limits bound parsing before the store's capture and token caps.
Session = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
Handle = Annotated[str, StringConstraints(pattern=r"^br-[0-9a-f]{12}$")]


class Sender(BaseModel):
    model_config = ConfigDict(extra="forbid")
    platform: str = Field(min_length=1, max_length=64)
    user_id: str = Field(default="", max_length=256)


class OwnerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requester: Session
    sender: Sender


class CaptureRequest(OwnerRequest):
    parent_session: Session
    child_session: Session
    messages: list[dict[str, Any]] = Field(min_length=1, max_length=10000)
    brief: str = Field(default="", max_length=200000)
    summary: str = Field(default="", max_length=200000)


class SessionRequest(OwnerRequest):
    session_id: Session


class ConsultRequest(SessionRequest):
    handles: list[Handle] = Field(min_length=1, max_length=256)
    question: str = Field(min_length=1, max_length=12000)


class ReactionRequest(SessionRequest):
    text: str = Field(min_length=1, max_length=12000)
    turn_id: str = Field(default="", max_length=256)
    reference_text: str | None = Field(default=None, max_length=12000)


def _refuse(code: str, message: str, *, status: int = 403) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


async def _owner(request: Request, requester: str, sender: Sender) -> str:
    if not speaks_for_owner(request):
        raise _refuse("owner_not_granted", "branches require an authenticated owner credential")
    person = resolve_request_person(request, claimed_person_id=requester)
    if person != owner_person_id():
        raise _refuse("branch_owner_required", "branches are available only in an owner session")
    platform = sender.platform.strip().lower()
    if not owner_platform_granted(request_authority(request), platform):
        raise _refuse("owner_not_granted", "the credential cannot act for the owner on this platform")
    if platform == "cli" and not sender.user_id.strip():
        return person
    if not sender.user_id.strip() or host._contacts_store is None:
        raise _refuse("branch_sender_unresolved", "the session sender must resolve to the owner")
    from colonymind.identity.participants import ParticipantResolver
    known = await ParticipantResolver(host._contacts_store).lookup(platform=platform, user_id=sender.user_id)
    if known.contact_id != person:
        raise _refuse("branch_sender_not_owner", "the session sender is not the owner")
    return person


def _store():
    from colonymind.config import load_config
    from colonymind.memory.branches import get_store
    return get_store(load_config())


@contextmanager
def _refusals():
    from colonymind.memory.branches import BranchRefused
    try:
        yield
    except BranchRefused as exc:
        status = {"owner_only": 403, "missing_branch": 404, "disabled": 409, "withdrawn": 409,
                  "summary_only": 409, "source_changed": 409, "token_cap": 413, "capture_size": 413,
                  "hourly_budget": 429, "tools_refused": 502}.get(exc.reason, 422)
        raise HTTPException(status_code=status, detail={"code": exc.reason, "handles": exc.handles}) from None


@router.get("")
async def list_branches(request: Request, session_id: Annotated[str, Query(max_length=256)], requester: Session,
                        platform: Annotated[str, Query(min_length=1, max_length=64)],
                        user_id: Annotated[str, Query(max_length=256)] = ""):
    person = await _owner(request, requester, Sender(platform=platform, user_id=user_id))
    store = _store()
    if not store.enabled:
        return {"branches": [], "enabled": False, "available": False, "tool_available": False}
    available = not (store.withdrawn or store.summary_only)
    with _refusals():
        rows = store.list_branches(session_id=session_id, requester=person) if session_id.strip() else []
    return {"branches": rows, "enabled": True, "available": available, "tool_available": available}


@router.post("",
             dependencies=[Depends(refuse_hand_session)])
async def capture_branch(body: CaptureRequest, request: Request):
    person = await _owner(request, body.requester, body.sender)
    store = _store()
    with _refusals():
        return {"branch": store.capture(parent_session=body.parent_session, child_session=body.child_session,
                                         messages=body.messages, brief=body.brief, summary=body.summary,
                                         requester=person)}


@router.post("/session",
             dependencies=[Depends(refuse_hand_session)])
async def session_branch(body: SessionRequest, request: Request):
    person = await _owner(request, body.requester, body.sender)
    store = _store()
    with _refusals():
        return {"branch": store.session_branch(session_id=body.session_id, requester=person)}


@router.post("/consult")
async def consult(body: ConsultRequest, request: Request):
    person = await _owner(request, body.requester, body.sender)
    store = _store()
    if host._llm_router is None and store.enabled and not (store.withdrawn or store.summary_only):
        raise _refuse("consult_unavailable", "no model router is configured for consults", status=503)
    with _refusals():
        return await store.consult(handles=body.handles, question=body.question, requester=person,
                                   session_id=body.session_id, router=host._llm_router)


@router.post("/reaction",
             dependencies=[Depends(refuse_hand_session)])
async def reaction(body: ReactionRequest, request: Request):
    person = await _owner(request, body.requester, body.sender)
    store = _store()
    with _refusals():
        return store.reaction(session_id=body.session_id, text=body.text, requester=person, turn_id=body.turn_id,
                              reference_text=body.reference_text)
