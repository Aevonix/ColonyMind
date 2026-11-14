"""Owner-only hands API: a fixed route catalog; the behavior is in ``colonymind.hands``."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query, Request
from pydantic import ValidationError

from colonymind.api.auth import KEY_PRINCIPAL, request_authority, speaks_for_owner
from colonymind.hands import approvals, controller, effects, obligations, wire
from colonymind.hands.sessions import refuse_hand_session
from colonymind.hands import digest as digest_ops
from colonymind.hands import notices as notice_ops
from colonymind.hands import definitions
from colonymind.api.credentials import HANDS, hands_route


# Match the route selected by FastAPI, never a caller-controlled path prefix.
COMPANION_ROUTES = frozenset({
    ("GET", "/v1/hands/definitions"),
    ("POST", "/v1/hands/definitions"),
    ("GET", "/v1/hands/definitions/{d}"),
    ("PATCH", "/v1/hands/definitions/{d}"),
    ("DELETE", "/v1/hands/definitions/{d}"),
    ("GET", "/v1/hands/definitions/{d}/tasks"),
    ("POST", "/v1/hands/definitions/{d}/tasks"),
    ("GET", "/v1/hands/"),
    ("GET", "/v1/hands/{id}"),
    ("GET", "/v1/hands/tasks/{tid}"),
    ("GET", "/v1/hands/tasks/{tid}/messages"),
    ("GET", "/v1/hands/digest"),
    ("POST", "/v1/hands/tasks/{tid}/accept"),
    ("POST", "/v1/hands/tasks/{tid}/cancel"),
    ("POST", "/v1/hands/tasks/{tid}/revise"),
    ("POST", "/v1/hands/questions/{qid}/answer"),
})


async def require_owner(request: Request) -> None:
    authority = request_authority(request)
    if (authority.authenticated and not authority.anonymous and authority.scopes is None
            and authority.credential_id != "api.key" and authority.access == frozenset({HANDS})
            and hands_route(request.url.path)):
        route = getattr(request.scope.get("route"), "path", None)
        if "companion" in authority.owner_platforms:
            if (request.method, route) not in COMPANION_ROUTES:
                raise HTTPException(403, "hands_route_not_allowed")
        elif route == "/v1/hands/questions/{qid}/answer":
            raise HTTPException(403, "hands credentials cannot answer asks")
        await refuse_hand_session(request)
        return
    if (authority.principal_id != KEY_PRINCIPAL or authority.scopes is not None
            or not speaks_for_owner(request)):
        raise HTTPException(status_code=403, detail={
            "code": "hands_owner_required", "message": "hands require the owner's instance credential"})


async def idempotency_key(value: Annotated[str, Header(alias="Idempotency-Key", min_length=1)]) -> str:
    if not value.strip():
        raise HTTPException(status_code=422, detail="Idempotency-Key must not be blank")
    return value


async def permit_check_query(hand: wire.HandId, call_digest: wire.Sha256,
                             targets_now: Annotated[str, Query()]) -> wire.PermitCheckRequest:
    # Explicit query fields also work on the package's minimum FastAPI version.
    try:
        return wire.PermitCheckRequest(hand=hand, call_digest=call_digest, targets_now=targets_now)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_context=False)) from None


WriteKey = Annotated[str, Depends(idempotency_key)]
router = APIRouter(prefix="/v1/hands", tags=["hands"], dependencies=[Depends(require_owner)])


# Routes added after the catalog was fixed have their own router; the fixed catalog stays as it is.
runtime_router = APIRouter(prefix="/v1/hands", tags=["hands"], dependencies=[Depends(require_owner)])


# Literal catalog and transcript paths precede the execution-attempt /{id} route.
@runtime_router.get("/definitions", dependencies=[Depends(refuse_hand_session)])
async def list_definitions(enabled_only: bool = False, include_archived: bool = False):
    return await (await controller.get_controller()).list_definitions(
        enabled_only=enabled_only, include_archived=include_archived)


@runtime_router.post("/definitions", dependencies=[Depends(refuse_hand_session)])
async def create_definition(body: definitions.CreateDefinition, key: WriteKey):
    return await (await controller.get_controller()).save_definition(body, key)


async def require_runtime_key(request: Request):
    authority = request_authority(request)
    if authority.credential_id != 'api.key':
        raise HTTPException(403, 'hands_runtime_credential_required')


@runtime_router.post('/definitions/queue/claim', dependencies=[Depends(require_runtime_key), Depends(refuse_hand_session)])
async def claim_definition_tasks(body: definitions.QueueClaim, key: WriteKey):
    return await (await controller.get_controller()).claim_definition_tasks(body, key)


@runtime_router.post('/definitions/queue/release', dependencies=[Depends(require_runtime_key), Depends(refuse_hand_session)])
async def release_definition_claim(body: definitions.QueueRelease, key: WriteKey):
    return await (await controller.get_controller()).release_definition_claim(body, key)


@runtime_router.post('/definitions/queue/bind', dependencies=[Depends(require_runtime_key), Depends(refuse_hand_session)])
async def bind_definition_claim(body: definitions.QueueBind, key: WriteKey):
    return await (await controller.get_controller()).bind_definition_claim(body, key)


@runtime_router.get("/definitions/{d}", dependencies=[Depends(refuse_hand_session)])
async def get_definition(d: str):
    return await (await controller.get_controller()).get_definition(d)


@runtime_router.patch("/definitions/{d}", dependencies=[Depends(refuse_hand_session)])
async def update_definition(d: str, body: definitions.ReplaceDefinition, key: WriteKey):
    return await (await controller.get_controller()).save_definition(body, key, d)


@runtime_router.delete("/definitions/{d}", dependencies=[Depends(refuse_hand_session)])
async def archive_definition(d: str, key: WriteKey):
    return await (await controller.get_controller()).save_definition(None, key, d, archive=True)


@runtime_router.get("/definitions/{d}/tasks", dependencies=[Depends(refuse_hand_session)])
async def definition_tasks(d: str, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    return await (await controller.get_controller()).definition_tasks(d, limit, offset)


@runtime_router.post("/definitions/{d}/tasks", dependencies=[Depends(refuse_hand_session)])
async def queue_task(d: str, body: definitions.TaskBrief, key: WriteKey, request: Request):
    return await (await controller.get_controller()).queue_definition_task(d, body, key, request=request)


@runtime_router.get("/requests/pending", dependencies=[Depends(refuse_hand_session)])
async def pending_definition_tasks():
    return await (await controller.get_controller()).pending_definition_tasks()


@runtime_router.get("/tasks/{tid}/messages", dependencies=[Depends(refuse_hand_session)])
async def task_messages(tid: wire.TaskId):
    return await (await controller.get_controller()).task_messages(tid)


@runtime_router.get("/retention")
async def retention():
    return await (await controller.get_controller()).retention_status()


@runtime_router.get("/permits/held", response_model=wire.HeldPermitsResponse)
async def held_permits(session: wire.Nonempty):
    return await approvals.held_permits(session)


@runtime_router.post("/relay", response_model=wire.RelayResponse, dependencies=[Depends(refuse_hand_session)])
async def relay(body: wire.RelayRequest, key: WriteKey):
    return await (await controller.get_controller()).relay(body, key)


@runtime_router.post("/{id}/probe", response_model=wire.HandResponse)
async def probe(id: wire.HandId, body: wire.ProbeRequest, key: WriteKey):
    return await (await controller.get_controller()).probe(id, body, key)


@router.post("/plan", response_model=wire.PlanResponse)
async def plan(body: wire.PlanRequest, key: WriteKey):
    return await controller.plan(body, key)


@router.post("/owner_message", response_model=wire.OwnerMessageResponse)
async def owner_message(body: wire.MessageRef, key: WriteKey):
    return await (await controller.get_controller()).owner_message(body.model_dump(), key)


@router.post("/{id}/dispatched", response_model=wire.HandResponse)
async def dispatched(id: wire.HandId, body: wire.DispatchedRequest, key: WriteKey):
    return await controller.dispatched(id, body, key)


@router.post("/{id}/release", response_model=wire.HandResponse)
async def release(id: wire.HandId, body: wire.ReleaseRequest, key: WriteKey):
    return await controller.release(id, body, key)


@router.post("/{id}/bind", response_model=wire.HandResponse)
async def bind(id: wire.HandId, body: wire.BindRequest, key: WriteKey):
    return await controller.bind(id, body, key)


@router.post("/{id}/events", response_model=wire.EventsResponse)
async def events(id: wire.HandId, body: wire.EventsRequest, key: WriteKey):
    return await controller.events(id, body, key)


@router.post("/{id}/question", response_model=wire.QuestionResponse)
async def question(id: wire.HandId, body: wire.QuestionRequest, key: WriteKey):
    return await controller.question(id, body, key)


@router.post("/questions/{qid}/answer", response_model=wire.QuestionResponse)
async def answer(qid: str, body: wire.AnswerRequest, key: WriteKey, request: Request):
    authority = request_authority(request)
    actor = (authority.identity or authority.credential_id
             if "companion" in authority.owner_platforms or body.answered_by is None else body.answered_by)
    return await (await controller.get_controller()).answer(qid, body, key, answered_by=actor)


@router.post("/{id}/hold", response_model=wire.HoldResponse)
async def hold(id: wire.HandId, body: wire.HoldRequest, key: WriteKey):
    return await approvals.hold(id, body, key)


@router.get("/permits/check", response_model=wire.PermitCheckResponse)
async def check(body: Annotated[wire.PermitCheckRequest, Depends(permit_check_query)], key: WriteKey):
    return await approvals.check(body, key)


@router.post("/{id}/result", response_model=wire.ResultResponse)
async def result(id: wire.HandId, body: wire.ResultRequest, key: WriteKey):
    return await controller.result(id, body, key)


@router.post("/tasks/{tid}/revise", response_model=wire.TaskResponse)
async def revise(tid: wire.TaskId, body: wire.ReviseRequest, key: WriteKey):
    return await controller.revise(tid, body, key)


@router.post("/tasks/{tid}/cancel", response_model=wire.TaskResponse)
async def cancel(tid: wire.TaskId, body: wire.CancelRequest, key: WriteKey):
    return await controller.cancel(tid, body, key)


@router.post("/tasks/{tid}/accept", response_model=wire.TaskResponse)
async def accept(tid: wire.TaskId, body: wire.AcceptRequest, key: WriteKey):
    return await controller.accept(tid, body, key)


@router.post("/obligations/{oid}/told", response_model=wire.ObligationResponse)
async def told(oid: str, body: wire.ToldRequest, key: WriteKey):
    return await obligations.told(oid, body, key)


@router.post("/obligations/{oid}/review", response_model=wire.ObligationResponse)
async def review(oid: str, body: wire.ReviewRequest, key: WriteKey):
    return await (await controller.get_controller()).review(
        oid, turn_ref=body.turn_ref, successful=body.successful, rendered=body.rendered, key=key)


@router.post("/obligations/{oid}/communicate", response_model=wire.ObligationResponse)
async def communicate(oid: str, body: wire.CommunicateRequest, key: WriteKey):
    return await (await controller.get_controller()).communicate(
        oid, body.turn_ref, silent=body.silent, destination=body.destination.model_dump(), key=key)


@router.post("/obligations/{oid}/delivered", response_model=wire.ObligationResponse)
async def delivered(oid: str, body: wire.DeliveredRequest, key: WriteKey):
    return await obligations.delivered(oid, body, key)


@router.get("/digest", response_model=wire.DigestResponse)
async def digest(body: Annotated[wire.DigestRequest, Depends()]):
    return await digest_ops.digest(body)


@router.get("/", response_model=wire.ListHandsResponse)
async def list_hands(body: Annotated[wire.ListHandsRequest, Depends()]):
    return await controller.list_hands(body)


@router.get("/tasks/{tid}", response_model=wire.TaskResponse)
async def get_task(tid: wire.TaskId):
    return await controller.get_task(tid)


@router.post("/{id}/stop", response_model=wire.HandResponse)
async def stop(id: wire.HandId, body: wire.StopRequest, key: WriteKey):
    return await controller.stop(id, body, key)


@router.post("/{id}/name", response_model=wire.HandResponse)
async def name(id: wire.HandId, body: wire.NameRequest, key: WriteKey):
    return await controller.name(id, body, key)


@router.get("/notices", response_model=wire.NoticesResponse)
async def notices(body: Annotated[wire.NoticesRequest, Depends()]):
    return await notice_ops.notices(body)


@router.post("/notices/{nid}/sent", response_model=wire.NoticeResponse)
async def sent(nid: str, body: wire.NoticeSentRequest, key: WriteKey):
    return await notice_ops.sent(nid, body, key)


@router.post("/reconcile", response_model=wire.ReconcileResponse)
async def reconcile(body: wire.ReconcileRequest, key: WriteKey):
    return await effects.reconcile(body, key)


@router.get("/{id}", response_model=wire.HandResponse)
async def get_hand(id: wire.HandId):
    return await controller.get_hand(id)
