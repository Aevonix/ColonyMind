"""``/v1/instinct``: the Hermes gates' route to Instinct (docs/HERMES-GATES.md).

``POST /v1/instinct/decide`` asks external points only (``colonymind.instinct.external``: the shipped
``inbound_triage`` and ``tool_screen``, and the operator's points), never a point of the mind. Body: ``{point,
fields, refs, unchanged?, event?, deadline_ms?}``, or ``{requests: [{point, fields, refs, unchanged?}], event?,
deadline_ms?}`` for one event's questions together (at most 16). Each decision answers ``{label, apply, tier,
state, decision_id}``: ``state`` is the gate's (``shadow`` or ``live``), and ``apply`` is true only for a label
Instinct applied while the gate is live, so in shadow a gate keeps its existing path whatever the point decided.
``unchanged`` says a cron job's sources are unchanged since its last run (a cron point's rules tier then answers
``no`` and no model is asked). A request that does not finish in ``deadline_ms`` (default 2,000, at most 5,000)
answers 504, and the gate fails open.

``POST /v1/instinct/outcome`` records how a cron job's run ended (``delivered``, ``silent`` or ``no_change``, as the
gate read it from that run's row in Hermes' own execution ledger) as that fire's decisions' outcome: ``{decisions:
[ids], outcome}``.

Both are for the adapter's own credential (the instance key the adapter already uses for capture): never another
credential, never development mode. Every decision is logged by reference (ids and probabilities, the gate's state
among the references), never with the words it read; a reference the route cannot check by its shape (a Hermes id)
is logged only as a keyed hash (``external.references``).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/instinct", tags=["instinct"])

MAX_REQUESTS = 16
MAX_FIELDS = 16
MAX_FIELD_CHARS = 8000
MAX_REF_CHARS = 256
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


class DecideItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    point: str = Field(min_length=1, max_length=64)
    fields: Dict[str, str] = Field(default_factory=dict)
    refs: Dict[str, str] = Field(default_factory=dict)
    unchanged: bool = False


class DecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    point: Optional[str] = Field(default=None, min_length=1, max_length=64)
    fields: Dict[str, str] = Field(default_factory=dict)
    refs: Dict[str, str] = Field(default_factory=dict)
    unchanged: bool = False
    requests: Optional[List[DecideItem]] = Field(default=None, min_length=1, max_length=MAX_REQUESTS)
    event: str = Field(default="", max_length=32)
    deadline_ms: int = Field(default=2000, ge=1, le=5000)


class OutcomeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decisions: List[str] = Field(min_length=1, max_length=64)
    outcome: Literal["delivered", "silent", "no_change"]


def _refused() -> HTTPException:
    return HTTPException(status_code=403, detail={
        "code": "instinct_not_granted",
        "message": "the gates' route is for the adapter's own credential (the instance key)"})


def _require_adapter(request: Request) -> None:
    from colonymind.api.auth import KEY_PRINCIPAL, request_authority
    authority = request_authority(request)
    if not (authority.authenticated and not authority.anonymous and authority.scopes is None
            and authority.principal_id == KEY_PRINCIPAL):
        raise _refused()


def _invalid(code: str, message: str) -> HTTPException:
    return HTTPException(status_code=422, detail={"code": code, "message": message})


def _checked(item: DecideItem) -> DecideItem:
    if len(item.fields) > MAX_FIELDS or len(item.refs) > MAX_FIELDS:
        raise _invalid("too_many_fields", f"at most {MAX_FIELDS} fields and {MAX_FIELDS} references")
    for name, value in item.fields.items():
        if not _NAME.match(name) or len(value) > MAX_FIELD_CHARS:
            raise _invalid("invalid_field", f"a field is a lower-case name with at most {MAX_FIELD_CHARS} "
                                            "characters of text")
    if any(len(value) > MAX_REF_CHARS for value in item.refs.values()):
        raise _invalid("invalid_ref", f"a reference is an id of at most {MAX_REF_CHARS} characters")
    from colonymind.instinct import external
    try:
        external.references(item.refs)
    except external.ExternalError as error:
        raise _invalid("invalid_ref", str(error)) from None
    return item


@router.post("/decide")
async def instinct_decide(body: DecideRequest, request: Request) -> Dict[str, Any]:
    _require_adapter(request)
    from colonymind.instinct import external
    if body.requests is not None:
        if body.point is not None:
            raise _invalid("point_and_requests", "send one point, or requests, not both")
        items = [_checked(item) for item in body.requests]
    elif body.point is None:
        raise _invalid("no_point", "name the external point to ask")
    else:
        items = [_checked(DecideItem(point=body.point, fields=body.fields, refs=body.refs,
                                     unchanged=body.unchanged))]
    engine, table = external.engine()
    for item in items:
        if item.point not in table:
            raise HTTPException(status_code=404, detail={
                "code": "not_an_external_point",
                "message": f"{item.point} is not an external decision point (the gates ask only their own)"})
        if item.unchanged and table[item.point].kind != "cron":
            raise _invalid("unchanged_not_cron", "only a cron point reads unchanged sources")
    states = {item.point: external.gate_state(engine, item.point) for item in items}
    requests = [(item.point, external.fields_for(table[item.point], item.fields, unchanged=item.unchanged),
                 {**external.references(item.refs), "gate": states[item.point]}) for item in items]
    try:
        decisions = await asyncio.wait_for(engine.decide_many(body.event, requests), body.deadline_ms / 1000.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail={
            "code": "deadline", "message": f"no decision within {body.deadline_ms} ms; the gate keeps its "
                                           "existing path"}) from None
    answers = [{"label": decision.label, "apply": bool(decision.applied and states[item.point] == "live"),
                "tier": decision.tier, "state": states[item.point], "decision_id": decision.id}
               for item, decision in zip(items, decisions)]
    return {"decisions": answers} if body.requests is not None else answers[0]


@router.post("/outcome")
async def instinct_outcome(body: OutcomeRequest, request: Request) -> Dict[str, Any]:
    _require_adapter(request)
    from colonymind.instinct import external
    engine, table = external.engine()
    try:
        recorded = external.record_outcome(engine, table, body.decisions, body.outcome)
    except Exception as error:
        logger.warning("a gate's outcome was not recorded (%s)", type(error).__name__)
        raise HTTPException(status_code=503, detail={"code": "outcome_unrecorded",
                                                     "message": "the decision log is unavailable"}) from None
    return {"recorded": recorded}


__all__ = ["router"]
