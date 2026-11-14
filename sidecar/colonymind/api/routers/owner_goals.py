"""Contract routes for owner goals. No caller field establishes owner authority."""
from __future__ import annotations

import hashlib
from urllib.parse import unquote_plus

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from colonymind.api.auth import owner_person_id, request_authority, speaks_for_owner
from colonymind.owner_goals.authority import ActionContext, Audience, get_action_verifier
from colonymind.owner_goals.store import OwnerGoalsStore
from colonymind.owner_goals.validation import (GoalError, REQUEST_BYTES, artifact_ref, dumps, identifier,
                                              invalid, loads, operation, pagination)

router = APIRouter(prefix="/v1/mind/owner-goals", tags=["mind"])


def get_owner_goals_store() -> OwnerGoalsStore:
    from .mind import get_mind
    mind = get_mind()
    if mind is None:
        raise HTTPException(503, detail={"code": "owner_goals_unavailable"})
    # The cache belongs to this mind instance and is discarded with it.
    if getattr(mind, "owner_goals", None) is None:
        mind.owner_goals = OwnerGoalsStore(mind.store)
    return mind.owner_goals


async def owner_context(request: Request) -> tuple[str, Audience]:
    if not speaks_for_owner(request):
        raise HTTPException(403, detail={"code": "not_owner"})
    # The owner comes from server configuration, never viewer/sender/body fields.
    owner = owner_person_id()
    # A trusted owner gate can narrow the audience for agent conversation reads.
    bound = getattr(request.state, "owner_goal_audience", None)
    if bound is not None:
        if not isinstance(bound, Audience) or bound.owner_id != owner:
            raise HTTPException(403, detail={"code": "not_owner"})
        return owner, bound
    # A credential alone says nothing about the conversation's participants.
    # The trusted gate must bind the audience for reads and mutations alike.
    raise HTTPException(503, detail={"code": "owner_goals_unavailable"})


def _error(exc: GoalError):
    raise HTTPException(exc.status, detail={"code": exc.code}) from exc


def _query(request, collection):
    raw = request.scope.get("query_string", b"")
    if not collection:
        if raw:
            invalid()
        return {}
    if not raw:
        return pagination([])
    # Check the wire representation too: signs/encoded aliases are not decimal query values.
    try:
        pairs = []
        for part in raw.decode("ascii").split("&"):
            key, sep, value = part.partition("=")
            if not sep or unquote_plus(key) != key or unquote_plus(value) != value:
                invalid()
            pairs.append((key, value))
    except UnicodeError:
        invalid()
    return pagination(pairs)


def _canonical(request):
    raw = request.scope.get("raw_path", request.url.path.encode())
    try:
        canonical = request.url.path.encode("ascii")
    except UnicodeError:
        invalid()
    if raw != canonical:
        invalid()


async def _read(request, store, context, *, id=None, collection=None, proposals=False):
    try:
        _canonical(request)
        if id is not None:
            identifier(id, goal=not proposals)
        args = _query(request, id is None or collection is not None)
        owner, audience = context
        return store.read(owner, audience, id=id, collection=collection, proposals=proposals, **args)
    except GoalError as exc:
        _error(exc)


async def _mutate(request, store, verifier, context, action, *, id=None, type=None, ref=None):
    try:
        _canonical(request)
        _query(request, False)
        if id is not None:
            identifier(id, goal=action != "confirm")
        if type is not None:
            artifact_ref(type, ref)
        keys = request.headers.getlist("idempotency-key")
        if len(keys) != 1:
            invalid()
        key = operation(keys[0])
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > REQUEST_BYTES:
                invalid()
        body = loads(bytes(raw))
        owner, audience = context
        authority = request_authority(request)
        digest = hashlib.sha256(dumps([request.method, request.url.path, body]).encode()).hexdigest()
        ctx = ActionContext(owner, authority.principal_id, audience, request.method, request.url.path,
                            key, body, digest, request)
        status, result = store.mutate(ctx, action, verifier, id=id, type=type, ref=ref)
        return JSONResponse(status_code=status, content=result)
    except GoalError as exc:
        _error(exc)


# Static proposal routes precede every goal-ID route.
@router.get("/proposals")
async def proposals(request: Request, context=Depends(owner_context), store=Depends(get_owner_goals_store)):
    return await _read(request, store, context, proposals=True)


@router.get("/proposals/{id}")
async def proposal(request: Request, id: str, context=Depends(owner_context), store=Depends(get_owner_goals_store)):
    return await _read(request, store, context, id=id, proposals=True)


@router.post("/proposals")
async def propose(request: Request, store=Depends(get_owner_goals_store),
                  verifier=Depends(get_action_verifier), context=Depends(owner_context)):
    return await _mutate(request, store, verifier, context, "propose")


@router.post("/proposals/{id}/confirm")
async def confirm(request: Request, id: str, store=Depends(get_owner_goals_store),
                  verifier=Depends(get_action_verifier), context=Depends(owner_context)):
    return await _mutate(request, store, verifier, context, "confirm", id=id)


@router.get("")
async def goals(request: Request, context=Depends(owner_context), store=Depends(get_owner_goals_store)):
    return await _read(request, store, context)


@router.post("")
async def create(request: Request, store=Depends(get_owner_goals_store),
                 verifier=Depends(get_action_verifier), context=Depends(owner_context)):
    return await _mutate(request, store, verifier, context, "create")


@router.get("/{id}")
async def goal(request: Request, id: str, context=Depends(owner_context), store=Depends(get_owner_goals_store)):
    return await _read(request, store, context, id=id)


@router.patch("/{id}")
async def patch(request: Request, id: str, store=Depends(get_owner_goals_store),
                verifier=Depends(get_action_verifier), context=Depends(owner_context)):
    return await _mutate(request, store, verifier, context, "patch", id=id)


@router.delete("/{id}")
async def delete(request: Request, id: str, store=Depends(get_owner_goals_store),
                 verifier=Depends(get_action_verifier), context=Depends(owner_context)):
    return await _mutate(request, store, verifier, context, "delete", id=id)


@router.get("/{id}/subgoals")
async def subgoals(request: Request, id: str, context=Depends(owner_context), store=Depends(get_owner_goals_store)):
    return await _read(request, store, context, id=id, collection="subgoals")


@router.get("/{id}/progress")
async def progress(request: Request, id: str, context=Depends(owner_context), store=Depends(get_owner_goals_store)):
    return await _read(request, store, context, id=id, collection="progress")


@router.post("/{id}/progress")
async def add_progress(request: Request, id: str, store=Depends(get_owner_goals_store),
                       verifier=Depends(get_action_verifier), context=Depends(owner_context)):
    return await _mutate(request, store, verifier, context, "progress", id=id)


@router.get("/{id}/artifacts")
async def artifacts(request: Request, id: str, context=Depends(owner_context), store=Depends(get_owner_goals_store)):
    return await _read(request, store, context, id=id, collection="artifacts")


@router.post("/{id}/artifacts")
async def attach(request: Request, id: str, store=Depends(get_owner_goals_store),
                 verifier=Depends(get_action_verifier), context=Depends(owner_context)):
    return await _mutate(request, store, verifier, context, "attach", id=id)


@router.delete("/{id}/artifacts/{type}/{ref}")
async def detach(request: Request, id: str, type: str, ref: str, store=Depends(get_owner_goals_store),
                 verifier=Depends(get_action_verifier), context=Depends(owner_context)):
    return await _mutate(request, store, verifier, context, "detach", id=id, type=type, ref=ref)


@router.get("/{id}/activity")
async def activity(request: Request, id: str, context=Depends(owner_context), store=Depends(get_owner_goals_store)):
    return await _read(request, store, context, id=id, collection="activity")
