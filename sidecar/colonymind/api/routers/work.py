"""Owner work reads and immutable late evidence admission."""
from __future__ import annotations

import asyncio
import base64
import binascii
import re
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from pydantic import AwareDatetime, BaseModel, Field

from colonymind.api.auth import KEY_PRINCIPAL, owner_person_id, request_authority, speaks_for_owner
from colonymind.api.errors import REFUSAL
from colonymind.api.public_spec import PUBLIC
from colonymind.util.temporal import now_utc
from colonymind.work import get_work_store
from colonymind.work.models import SourceRef, WorkEvent

RootStatus = Literal["proven", "partial", "unproven", "failed"]
OperationStatus = Literal["verified", "observed", "ran", "failed", "dispatched", "unknown"]
MAX_EFFECTS = 256
MAX_UNKNOWN_EFFECTS = 100
MAX_DAYS = 366


class WorkRoot(BaseModel):
    id: str = Field(max_length=1370, pattern=r"^w:[A-Za-z0-9_-]+$")
    contact_id: str
    session_id: str
    turn_id: str
    status: RootStatus
    created_at: str
    title: str | None = None
    revision: str | int
    overlay: str | None = None


class WorkEffect(BaseModel):
    id: str
    operation_id: str
    tool: str
    action: str
    resource_ref: str | None = None
    kind: str
    status: OperationStatus
    verification: str = Field(default="", max_length=512)
    source_refs: list[SourceRef] = Field(default_factory=list, max_length=64)


class WorkChild(BaseModel):
    turn_id: str
    parent_turn_id: str


class WorkDetail(WorkRoot):
    effects: list[WorkEffect] = Field(default_factory=list, max_length=MAX_EFFECTS)
    children: list[WorkChild] = Field(default_factory=list, max_length=MAX_EFFECTS)
    effects_truncated: bool = False
    children_truncated: bool = False


class WorkList(BaseModel):
    enabled: bool
    since: str
    until: str
    items: list[WorkRoot] = Field(default_factory=list, max_length=100)
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0, le=10000)
    next_offset: int | None = Field(default=None, ge=0, le=10000)


class WorkCounts(BaseModel):
    proven: int = Field(default=0, ge=0)
    partial: int = Field(default=0, ge=0)
    unproven: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)


class WorkRates(BaseModel):
    proven: float = Field(default=0, ge=0, le=1)
    partial: float = Field(default=0, ge=0, le=1)
    unproven: float = Field(default=0, ge=0, le=1)
    failed: float = Field(default=0, ge=0, le=1)


class WorkUnknownEffect(BaseModel):
    tool: str
    action: str
    count: int = Field(ge=0)


class WorkOverlays(BaseModel):
    confirmed: int = Field(default=0, ge=0)
    corrected: int = Field(default=0, ge=0)
    none: int = Field(default=0, ge=0)


class WorkDigestTitles(BaseModel):
    unproven: list[str] = Field(default_factory=list, max_length=5)
    failed: list[str] = Field(default_factory=list, max_length=5)


class WorkVerdictCoverage(BaseModel):
    session_limit: int = 6
    turn_limit: int = 8
    reviewed_at: str | None = None
    sessions_reviewed: int = Field(default=0, ge=0)
    sessions_omitted: int = Field(default=0, ge=0)
    sessions_with_omitted_turns: int = Field(default=0, ge=0)
    vetted: int = Field(default=0, ge=0)
    queued: int = Field(default=0, ge=0)
    unlinked: int = Field(default=0, ge=0)
    verdicts_omitted: int = Field(default=0, ge=0)


class WorkSummary(BaseModel):
    enabled: bool
    since: str
    until: str
    total: int = Field(default=0, ge=0)
    counts: WorkCounts = Field(default_factory=WorkCounts)
    rates: WorkRates = Field(default_factory=WorkRates)
    roots_per_day: float = Field(default=0, ge=0)
    owner_corrected: int = Field(default=0, ge=0)
    owner_corrected_rate: float = Field(default=0, ge=0, le=1)
    unknown_effects: list[WorkUnknownEffect] = Field(default_factory=list, max_length=MAX_UNKNOWN_EFFECTS)
    unknown_effects_truncated: bool = False
    overlays: WorkOverlays = Field(default_factory=WorkOverlays)
    titles: WorkDigestTitles = Field(default_factory=WorkDigestTitles)
    untitled: int = Field(default=0, ge=0)
    verdict_coverage: WorkVerdictCoverage = Field(default_factory=WorkVerdictCoverage)


class WorkEventResult(BaseModel):
    accepted: bool
    duplicate: bool = False
    enabled: bool = True
    skipped_reason: str | None = None


async def require_owner(request: Request) -> None:
    authority = request_authority(request)
    if (authority.principal_id != KEY_PRINCIPAL or authority.scopes is not None
            or not speaks_for_owner(request)):
        raise HTTPException(status_code=403, detail={
            "code": "work_owner_required", "message": "work records require the owner's instance credential"})


def _encode_id(value: str) -> str:
    return "w:" + base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


def _decode_id(value: str) -> str:
    encoded = value[2:] if value.startswith("w:") else ""
    try:
        if not encoded or re.fullmatch(r"[A-Za-z0-9_-]+", encoded) is None:
            raise ValueError
        raw = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True).decode("utf-8")
        if not 1 <= len(raw) <= 256 or _encode_id(raw) != value:
            raise ValueError
        return raw
    except (ValueError, UnicodeError, binascii.Error):
        raise HTTPException(status_code=422, detail={"code": "work_invalid_id"}) from None


def _public_root(row: dict) -> dict:
    return {**row, "id": _encode_id(row["id"])}


def _refusal(description: str) -> dict:
    return {"description": description, "content": {"application/json": {"schema": dict(REFUSAL)}}}


router = APIRouter(prefix="/v1/host/work", tags=["work", PUBLIC], dependencies=[Depends(require_owner)],
                   responses={403: _refusal("The owner instance credential is required.")})


def _window(since: datetime | None, until: datetime | None) -> tuple[datetime, datetime]:
    now = now_utc()
    end = until.astimezone(timezone.utc) if until is not None else now
    start = since.astimezone(timezone.utc) if since is not None else end - timedelta(days=7)
    if end > now or start >= end or end - start > timedelta(days=MAX_DAYS):
        raise HTTPException(status_code=422, detail={
            "code": "work_invalid_window",
            "message": "work bounds must increase, end no later than now, and span at most 366 days"})
    return start, end


Since = Annotated[AwareDatetime | None, Query(description="Inclusive UTC-normalized start; default seven days ago.")]
Until = Annotated[AwareDatetime | None, Query(description="Exclusive UTC-normalized end; default now.")]


@router.get("", response_model=WorkList)
async def work_list(since: Since = None, until: Until = None,
                    status: RootStatus | None = Query(default=None),
                    limit: int = Query(default=50, ge=1, le=100),
                    offset: int = Query(default=0, ge=0, le=10000)):
    start, end = _window(since, until)
    store = get_work_store(read_only=True)
    rows = [] if store is None else await asyncio.to_thread(
        store.list_roots, contact_id=owner_person_id(), since=start.isoformat(), until=end.isoformat(),
        status=status, limit=limit + 1, offset=offset)
    next_offset = offset + limit if len(rows) > limit and offset + limit <= 10000 else None
    return {"enabled": store is not None, "since": start.isoformat(), "until": end.isoformat(),
            "items": [_public_root(row) for row in rows[:limit]], "limit": limit, "offset": offset, "next_offset": next_offset}


@router.get("/summary", response_model=WorkSummary)
async def work_summary(since: Since = None, until: Until = None):
    start, end = _window(since, until)
    store = get_work_store(read_only=True)
    summary = {} if store is None else await asyncio.to_thread(
        store.summary, contact_id=owner_person_id(), since=start.isoformat(), until=end.isoformat())
    counts = WorkCounts.model_validate(summary.get("counts", {})).model_dump()
    total = int(summary.get("total", sum(counts.values())))
    corrected = int(summary.get("owner_corrected", 0))
    unknown = summary.get("unknown_effects", [])
    return {"enabled": store is not None, "since": start.isoformat(), "until": end.isoformat(),
            "total": total, "counts": counts,
            "rates": {key: value / total if total else 0 for key, value in counts.items()},
            "roots_per_day": total / ((end - start).total_seconds() / 86400),
            "owner_corrected": corrected, "owner_corrected_rate": corrected / total if total else 0,
            "overlays": summary.get('overlays', {}), "titles": summary.get('titles', {}),
            "untitled": summary.get('untitled', 0), "verdict_coverage": summary.get('verdict_coverage', {}),
            "unknown_effects": unknown[:MAX_UNKNOWN_EFFECTS],
            "unknown_effects_truncated": len(unknown) > MAX_UNKNOWN_EFFECTS}


@router.post("/events", response_model=WorkEventResult,
             responses={409: _refusal("The event identity conflicts with retained work."),
                        410: _refusal("The root is ineligible or erased, or its parent wait expired."),
                        404: _refusal("The admitted owner root or child lineage is not available yet.")})
async def work_event(body: WorkEvent, request: Request):
    store = get_work_store()
    if store is None:
        return {"accepted": False, "duplicate": False, "enabled": False,
                "skipped_reason": "work_records_disabled"}
    try:
        return await asyncio.to_thread(store.admit_event, body, contact_id=owner_person_id(),
                                       principal_id=request_authority(request).principal_id)
    except ValueError as error:
        reason = str(error)
        conflicts = {"work_event_conflict", "work_operation_conflict", "work_event_identity_conflict"}
        terminal = {"work_root_ineligible", "work_root_erased", "work_parent_expired", "work_source_erased"}
        missing = {"work_root_missing", "work_parent_missing", "work_lineage_missing",
                   "work_source_revision_missing"}
        invalid = {"work_child_parent_mismatch", "work_event_root_mismatch", "work_source_invalid",
                   "work_principal_mismatch", "work_source_erased"}
        status = 410 if reason in terminal else 409 if reason in conflicts else 404 if reason in missing else 422
        code = reason if reason in conflicts | missing | invalid | terminal else "work_event_invalid"
        raise HTTPException(status_code=status, detail={"code": code}) from None


@router.get("/{id}", response_model=WorkDetail,
            responses={404: _refusal("No retained owner work has this identifier.")})
async def work_detail(id: str = Path(min_length=1, max_length=1370)):
    original = _decode_id(id)
    store = get_work_store(read_only=True)
    row = None if store is None else await asyncio.to_thread(store.get_root, original, contact_id=owner_person_id())
    if row is None:
        raise HTTPException(status_code=404, detail={"code": "work_not_found"})
    effects, children = row.get("effects", []), row.get("children", [])
    return {**_public_root(row), "effects": effects[:MAX_EFFECTS], "children": children[:MAX_EFFECTS],
            "effects_truncated": bool(row.get("effects_truncated")) or len(effects) > MAX_EFFECTS,
            "children_truncated": bool(row.get("children_truncated")) or len(children) > MAX_EFFECTS}
