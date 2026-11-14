"""Read-only projection of recorded owner asks, with a closed outcome vocabulary."""
from typing import Any, Literal

from pydantic import BaseModel

from colonymind.hands.wire import BoundTarget

ApprovalOutcome = Literal["allowed", "denied", "auto", "timed_out", "cancelled", "failed", "done", "uncertain"]
ApprovalState = Literal["pending", "settled"]


class ApprovalSummary(BaseModel):
    id: str
    revision: str
    kind: Literal["hands_permit", "ask"]
    type: str
    code: str
    title: str
    hand_id: str | None
    session_id: str | None
    state: ApprovalState
    status: str
    outcome: ApprovalOutcome | None
    uncertainty: Literal["missing_settlement", "unrecognized_settlement"] | None
    open: bool
    created_at: str
    expires_at: str | None
    settled_at: str | None


class ApprovalCall(BaseModel):
    tool: str | None
    arguments: dict[str, Any]
    workdir: str | None
    targets: list[BoundTarget]


class ApprovalTerms(BaseModel):
    text: str
    reply: str
    reason: str
    effect_class: str | None
    call: ApprovalCall
    call_digest: str | None
    full_text_required: bool
    full_text_delivered: bool


class ApprovalWinner(BaseModel):
    channel: str | None
    choice: str | None
    at: str
    path: str | None


class ApprovalSettlement(BaseModel):
    by: Literal["owner", "operator", "lapse", "mind"]
    outcome: str
    at: str
    winner: ApprovalWinner | None


class ApprovalDetail(ApprovalSummary):
    raw_outcome: str | None
    terms: ApprovalTerms
    settlement: ApprovalSettlement | None


class ApprovalCounts(BaseModel):
    pending: int
    settled: int
    total: int


class SessionApprovalCounts(ApprovalCounts):
    session_id: str | None


class ApprovalHistoryPage(BaseModel):
    revision: str
    snapshot_at: str
    items: list[ApprovalSummary]
    counts: ApprovalCounts
    sessions: list[SessionApprovalCounts]
    has_more: bool
    next_cursor: str | None


class ApprovalHistoryDetail(BaseModel):
    revision: str
    snapshot_at: str
    approval: ApprovalDetail
