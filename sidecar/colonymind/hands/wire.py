"""Frozen HTTP contracts for hands (design sections 3.4, 3.6 and 4.2).

Every write carries Idempotency-Key. The controller persists its first response and
replays it without another mutation. The permit check consumes a permit, so its GET also
carries that header. Epochs identify executor processes, not child sessions, which
can rotate. Result text is hash-bound for accepted provenance and is erased with
its parent; branches remain the source of the full executor conversation.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, Json, RootModel, StringConstraints, model_validator

from .types import (
    CheckReceipt, ContractUse, Effort, Evidence, Hand, HandState, Kind, Notice, Obligation,
    Priority, Question, Risk, Task,
)


Nonempty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
TaskId = Annotated[str, StringConstraints(pattern=r"^t-[0-9a-f]{12}$")]
HandId = Annotated[str, StringConstraints(pattern=r"^h-[0-9a-f]{12}$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
# Assumption from section 4.6: names have at most 24 characters.
HandName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9-]{1,24}$")]
Need = Literal["code", "shell", "web", "vision", "audio", "long_context"]
Origin = Literal["owner_turn", "own", "mind", "voice", "harness"]


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class MessageRef(WireModel):
    platform: Nonempty
    chat: Nonempty
    message_id: Nonempty


class ChatRef(WireModel):
    platform: Nonempty
    chat: Nonempty


class OwnerMessageResponse(WireModel):
    recorded: bool


class ExecutorEpoch(WireModel):
    owner_pid: int = Field(gt=0)
    owner_started_at: float = Field(ge=0)


class ErrorDetail(WireModel):
    code: str
    unit: str
    message: str


class ErrorResponse(WireModel):
    detail: ErrorDetail


class AcceptanceCriteria(WireModel):
    """Checks fixed before execution; no result may supply replacement commands."""

    test_command: str | list[str] | None = None
    probe: str | list[str] | None = None
    required_sources: int | None = Field(default=None, ge=1)
    independent_sources: bool = False
    quotes_required: bool = False
    output_type: str | None = None
    requirements: list[str] = Field(default_factory=list)


class PlanTask(WireModel):
    goal: Nonempty
    hand: Annotated[str, StringConstraints(min_length=1, max_length=200)] | None = None
    context: str = ""
    kind: Kind | None = None
    effort: Effort | None = None
    risk: Risk | None = None
    needs: list[Need] = Field(default_factory=list)
    accept: AcceptanceCriteria | None = None
    touches: list[str] = Field(default_factory=list)
    parts: list[PlanTask] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    name: HandName | None = None
    priority: Priority | None = None
    continues: str | None = Field(default=None, alias="continue")
    reconciled: bool = False
    workdir: str | None = None
    queue_claim: str | None = Field(default=None, min_length=1, max_length=100)


class PlanRequest(WireModel):
    parent_session: Nonempty
    turn: Nonempty
    # Completion is a planning context (3.6), never an owner-message task origin.
    origin: Origin | Literal["completion"]
    origin_ref: MessageRef | Nonempty
    tasks: list[PlanTask] = Field(min_length=1)
    decision_ids: list[str] = Field(default_factory=list)


class ContractRef(WireModel):
    provider: Nonempty
    version: Nonempty


class AttemptBudget(WireModel):
    wall_seconds: int = Field(gt=0)
    tool_calls: int = Field(ge=0)
    web_searches: int = Field(ge=0)
    tokens: int = Field(gt=0)


class PlannedHand(WireModel):
    task_id: TaskId
    hand_id: HandId
    name: HandName
    kind: Kind
    effort: Effort
    risk: Risk
    role: Nonempty | None = None
    contract: ContractRef | None = None
    budgets: AttemptBudget
    brief: str
    admit: Literal["start", "queued", "refuse"]
    reason: str | None = None
    lease_token: str | None = None


class PlanResponse(WireModel):
    plans: list[PlannedHand]


class EpochRequest(WireModel):
    epoch: ExecutorEpoch


class DispatchedRequest(EpochRequest):
    delegation_id: Nonempty
    lease_token: Nonempty


class ReleaseRequest(DispatchedRequest):
    reason: str | None = None


class BindRequest(EpochRequest):
    executor_ref: Nonempty
    child_session: Nonempty


class EventInput(WireModel):
    seq: int = Field(ge=0)
    at: str
    type: Nonempty
    payload: dict[str, Any]
    source: Nonempty
    key: Nonempty


class EffectInput(WireModel):
    key: Nonempty
    tool: Nonempty
    call_hash: Sha256
    workdir: str
    intent_at: str
    outcome: Literal["ok", "error", "unknown"] = "unknown"
    exit_code: int | None = None
    result_hash: Sha256 | None = None
    done_at: str | None = None


class EventsRequest(EpochRequest):
    events: list[EventInput] = Field(default_factory=list)
    effects: list[EffectInput] = Field(default_factory=list)


class EventsResponse(WireModel):
    cursor: str


class QuestionRequest(EpochRequest):
    question: str = Field(min_length=1, max_length=500)
    assumption: str = ""
    blocking: bool = False


class AnswerRequest(WireModel):
    answer: Nonempty
    answered_by: Nonempty | None = None


class ExactCall(WireModel):
    tool: Nonempty
    arguments: dict[str, Any]


class BoundTarget(WireModel):
    kind: Literal["file", "git_ref", "destination", "service", "probe"]
    target: Nonempty
    value: Nonempty


class HoldRequest(EpochRequest):
    call: ExactCall
    workdir: Nonempty
    targets: list[BoundTarget]


class HoldResponse(WireModel):
    permit_id: Nonempty
    ask_id: Nonempty


class PermitCheckRequest(WireModel):
    hand: HandId
    call_digest: Sha256
    # JSON query parameter permits repeated target bindings without a GET body.
    targets_now: Json[list[BoundTarget]]


class PermitCheckResponse(WireModel):
    decision: Literal["allow", "hold"]
    reason: str | None = None
    permit_id: str | None = None


class HeldPermit(WireModel):
    permit_id: Nonempty
    ask_id: Nonempty
    hand_id: HandId
    parent_session: Nonempty
    state: Literal["waiting", "granted"]
    call: ExactCall
    workdir: Nonempty
    call_digest: Sha256
    targets: list[BoundTarget]


class HeldPermitsResponse(WireModel):
    permits: list[HeldPermit]
    refreshed: bool = False


class ToolReceipt(WireModel):
    tool: Nonempty
    exit_code: int | None = None
    path: str | None = None
    url: str | None = None
    content_hash: Sha256 | None = None
    result_hash: Sha256 | None = None


class ResultRequest(EpochRequest):
    intent_version: int = Field(ge=1)
    status: Literal["done", "partial", "needs_input", "blocked", "failed", "stopped"]
    summary_ref: Nonempty
    summary_hash: Sha256
    summary_text: str = ""
    tool_digest: list[ToolReceipt]
    receipts: list[CheckReceipt]
    families_used: list[Nonempty]
    contracts_used: list[ContractUse] = Field(default_factory=list)


class ProbeRequest(EpochRequest):
    receipt: CheckReceipt


class RelayRequest(WireModel):
    session: Nonempty
    turn: Nonempty
    hand_ids: list[HandId]


class RelayResponse(WireModel):
    session: Nonempty
    turn: Nonempty
    hand_refs: list[dict]


class ResultResponse(WireModel):
    outcome: HandState
    evidence: Evidence
    # The controller answers "none" for a hand that is done, "wait" for one that was superseded, stopped or
    # failed, and "continue" otherwise. It sends none of the other values.
    next_step: Literal["none", "continue", "escalate", "verify", "integrate", "synthesize", "wait"]


class ReviseRequest(WireModel):
    goal: Nonempty | None = None
    criteria: AcceptanceCriteria | None = None
    stop: bool = False

    @model_validator(mode="after")
    def has_revision(self):
        if self.goal is None and self.criteria is None:
            raise ValueError("a revision requires a goal or criteria")
        return self


class CancelRequest(WireModel):
    reason: Nonempty
    owner_ref: MessageRef | None = None


class AcceptRequest(WireModel):
    hand: HandId | None = None
    reason: Nonempty
    for_version: int | None = Field(default=None, ge=1)


class ToldRequest(WireModel):
    turn_ref: Nonempty


class ReviewRequest(WireModel):
    turn_ref: Nonempty
    successful: bool
    rendered: bool


class CommunicateRequest(WireModel):
    turn_ref: Nonempty
    silent: bool
    destination: ChatRef


class DeliveredRequest(WireModel):
    platform_message_id: Nonempty


class DigestRequest(WireModel):
    session: Nonempty
    cursor: str = ""


class DigestItem(WireModel):
    task_id: TaskId
    hand_id: HandId | None = None
    obligation_id: str | None = None
    question_id: str | None = None
    permit_id: str | None = None
    evidence: Evidence | None = None
    text: str


class DigestResponse(WireModel):
    text: str
    items: list[DigestItem]
    cursor: str


class ListHandsRequest(WireModel):
    session: str | None = None
    task_id: TaskId | None = None
    state: HandState | None = None
    kind: Kind | None = None
    priority: Priority | None = None
    origin: Origin | None = None
    cursor: str = ""


class StopRequest(WireModel):
    # The owner or the front may stop a hand without an executor epoch (4.2: the body is the reason); a plugin
    # stop carries its epoch so a stale executor's stop is refused.
    reason: Nonempty
    epoch: ExecutorEpoch | None = None


class NameRequest(WireModel):
    name: HandName


class NoticesRequest(WireModel):
    cursor: str = ""


class NoticeSentRequest(DeliveredRequest):
    pass


class ExecutorRecord(WireModel):
    hand_id: HandId
    executor_ref: Nonempty
    epoch: ExecutorEpoch
    status: Nonempty
    child_session: str | None = None


class ReconcileRequest(EpochRequest):
    live: list[ExecutorRecord]
    ledger: list[ExecutorRecord]


# Row responses use the same dataclasses as storage rather than parallel field lists.
class HandResponse(RootModel[Hand]):
    pass


@dataclass(kw_only=True)
class TaskQuestion:
    id: str
    text: str | None
    assumption: str | None
    blocking: bool
    asked_at: str
    state: str
    answer: str | None


@dataclass(kw_only=True)
class TaskView(Task):
    status: str | None = None
    status_detail: dict | None = None
    definition_id: str | None = None
    definition_version: int | None = None
    dispatch_state: str | None = None
    questions: list[TaskQuestion] = field(default_factory=list)


class TaskResponse(RootModel[TaskView]):
    pass


class QuestionResponse(RootModel[Question]):
    pass


class ObligationResponse(RootModel[Obligation]):
    pass


class NoticeResponse(RootModel[Notice]):
    pass


class ListHandsResponse(WireModel):
    hands: list[Hand]
    cursor: str


class NoticesResponse(WireModel):
    notices: list[Notice]
    cursor: str


class ReconcileResponse(WireModel):
    hands: list[Hand]
