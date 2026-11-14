"""Shared records for the durable task controller.

Records are snapshots, without transition or execution behavior. JSON records
keep structured references and receipts; brief and summary prose belongs in the
branch store. Timestamps are UTC strings at the persistence boundary. The scope
and contract field layouts are this module's choice where the design leaves a
shape open; scope kinds remain extensible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Literal


JsonObject = dict[str, Any]
Reference = str | dict[str, str]


class TaskState(str, Enum):
    OPEN = "open"
    WAITING = "waiting"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HandState(str, Enum):
    PLANNED = "planned"
    RESERVED = "reserved"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    CHECKING = "checking"
    INTEGRATING = "integrating"
    DONE = "done"
    BELOW_MINIMUM = "below_minimum"
    FAILED = "failed"
    STOPPED = "stopped"
    SUPERSEDED = "superseded"
    LOST = "lost"
    UNKNOWN_EFFECT = "unknown_effect"
    REFUSED = "refused"
    EXPIRED = "expired"


class HandActivity(str, Enum):
    QUEUED = "queued"
    GENERATING = "generating"
    EXECUTING = "executing"
    AWAITING_INPUT = "awaiting_input"
    AWAITING_APPROVAL = "awaiting_approval"


class EvidenceLevel(str, Enum):
    E0 = "E0"
    E1 = "E1"
    E2 = "E2"
    E3 = "E3"
    E4 = "E4"
    E5 = "E5"


class Risk(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class Kind(str, Enum):
    CODING = "coding"
    DEEP = "deep"
    RESEARCH = "research"
    OPS = "ops"
    MEDIA = "media"
    PLAN = "plan"
    SYNTHESIZE = "synthesize"
    VERIFY = "verify"
    INTEGRATE = "integrate"


class Effort(IntEnum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3


class Priority(str, Enum):
    OWNER = "owner"
    OWN = "own"
    AUTO = "auto"
    UPKEEP = "upkeep"


class ObligationState(str, Enum):
    OPEN = "open"
    RESULT_READY = "result_ready"
    REVIEWED = "reviewed"
    ACCEPTED = "accepted"
    COMMUNICATED = "communicated"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


class ReservationState(str, Enum):
    HELD = "held"
    CONFIRMED = "confirmed"
    RELEASED = "released"
    EXPIRED = "expired"


class PermitState(str, Enum):
    WAITING = "waiting"
    GRANTED = "granted"
    CONSUMED = "consumed"
    REFUSED = "refused"
    LAPSED = "lapsed"
    STALE = "stale"


class DerivationState(str, Enum):
    LIVE = "live"
    DOWNGRADED = "downgraded"
    RETRACTED = "retracted"


@dataclass(kw_only=True, slots=True)
class EvidenceArtifact:
    """An exact artifact, commit or state used by a check."""

    kind: str
    reference: str
    sha256: str | None = None


@dataclass(kw_only=True, slots=True)
class EvidenceQuote:
    """A quote bound to the fetched source and, optionally, its review."""

    url: str
    content_hash: str
    quote: str
    start: int | None = None
    end: int | None = None
    judgment: Literal["supports", "does_not_support", "unclear"] | None = None


@dataclass(kw_only=True, slots=True)
class EvidenceScope:
    """What was checked, against which criterion, on which exact artifact.

    The optional fields cover different check kinds without claiming that every
    level has a test command or an independent reviewer. Their presence does not
    itself establish an evidence level.
    """

    kind: str
    reference: str | None = None
    sha256: str | None = None
    dirty_tree_sha256: str | None = None
    command: str | list[str] | None = None
    probe: str | list[str] | None = None
    criterion: str | None = None
    criterion_ref: str | None = None
    artifacts: list[EvidenceArtifact] = field(default_factory=list)
    quotes: list[EvidenceQuote] = field(default_factory=list)
    producer_families: list[str] = field(default_factory=list)
    reviewer_family: str | None = None
    verdict: Literal["pass", "fail", "unclear"] | None = None
    details: JsonObject = field(default_factory=dict)


@dataclass(kw_only=True, slots=True)
class Evidence:
    level: EvidenceLevel
    scope: EvidenceScope


@dataclass(kw_only=True, slots=True)
class CheckReceipt:
    scope: EvidenceScope
    exit_code: int | None = None
    output_hash: str | None = None
    passed: bool | None = None


@dataclass(kw_only=True, slots=True)
class ToolCallQualification:
    qualified: bool
    success_share: float


@dataclass(kw_only=True, slots=True)
class RouteAdmission:
    family: str
    priority: int


@dataclass(kw_only=True, slots=True)
class RouteContract:
    """Versioned provider capabilities; routing validation belongs to admission.

    Assumption: contract versions are opaque strings, JSON qualification is a
    boolean, and effort maps the controller's levels to provider request fields.
    No provider, route endpoint or credential is built into this shape.
    """

    version: str
    family: str
    modalities: list[Literal["text", "image", "audio"]]
    context_tokens: int
    max_output_tokens: int
    tool_calls: ToolCallQualification
    json: bool
    effort: dict[Effort, JsonObject]
    decode_tok_s: float
    admission: RouteAdmission


@dataclass(kw_only=True, slots=True)
class ContractUse:
    """One actual route used by a hand, including any fallback provider."""

    provider: str
    version: str
    family: str
    model: str


@dataclass(kw_only=True, slots=True)
class ContractRef:
    """Planned route and the actual routes observed during execution.

    Assumption: ``used`` stores provider, contract version, family and served
    model together so fallback history stays in the existing contract JSON.
    """

    provider: str
    version: str
    used: list[ContractUse] = field(default_factory=list)


@dataclass(kw_only=True, slots=True)
class ExecutorEpoch:
    owner_pid: int
    owner_started_at: float


@dataclass(kw_only=True, slots=True)
class TaskIntent:
    goal_hash: str
    acceptance: JsonObject
    set_at: str


@dataclass(kw_only=True, slots=True)
class Task:
    id: str
    origin: Literal["owner_turn", "own", "mind", "voice", "harness"]
    origin_ref: Reference
    kind: Kind
    priority: Priority = Priority.OWNER
    risk: Risk = Risk.NORMAL
    needs: list[str] = field(default_factory=list)
    intent_version: int = 1
    intents: dict[str, TaskIntent] = field(default_factory=dict)
    resources: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    state: TaskState = TaskState.OPEN
    budget: JsonObject = field(default_factory=dict)
    usage: JsonObject = field(default_factory=dict)
    failures_in_row: int = 0
    cooldown_until: str | None = None
    obligation_id: str | None = None
    owner_spoke_since: Reference | None = None
    created_at: str | None = None
    closed_at: str | None = None


@dataclass(kw_only=True, slots=True)
class Hand:
    id: str
    task_id: str
    attempt: int
    intent_version: int
    name: str
    kind: Kind
    effort: Effort
    role: str | None = None
    parent_session: str
    executor: Literal["hermes_delegation", "hermes_kanban", "harness"]
    epoch: ExecutorEpoch
    contract: ContractRef | None = None
    families_used: list[str] = field(default_factory=list)
    model: list[str] = field(default_factory=list)
    origin_ref: Reference | None = None
    priority: Priority = Priority.OWNER
    child_session: str | None = None
    executor_ref: str | None = None
    state: HandState = HandState.PLANNED
    activity: HandActivity | None = None
    activity_since: str | None = None
    outcome: str | None = None
    evidence: Evidence | None = None
    check: list[CheckReceipt] = field(default_factory=list)
    branch: str | None = None
    continues: str | None = None
    escalated_from: str | None = None
    budgets: JsonObject = field(default_factory=dict)
    usage: JsonObject = field(default_factory=dict)
    decisions: JsonObject = field(default_factory=dict)
    created_at: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    last_event_at: str | None = None


@dataclass(kw_only=True, slots=True)
class Reservation:
    hand_id: str
    token: str
    family: str
    class_: str = field(metadata={"column": "class"})
    expires_at: str
    state: ReservationState = ReservationState.HELD


@dataclass(kw_only=True, slots=True)
class Lock:
    resource: str
    holder_hand: str
    since: str
    waiters: list[str] = field(default_factory=list)


@dataclass(kw_only=True, slots=True)
class Effect:
    hand_id: str
    key: str
    tool: str
    call_hash: str
    workdir: str
    intent_at: str
    outcome: Literal["ok", "error", "unknown"] = "unknown"
    exit_code: int | None = None
    result_hash: str | None = None
    done_at: str | None = None


@dataclass(kw_only=True, slots=True)
class Obligation:
    id: str
    task_id: str
    owner_ref: Reference
    state: ObligationState = ObligationState.OPEN
    result_ready_at: str | None = None
    reviewed_at: str | None = None
    accepted_at: str | None = None
    communicated_at: str | None = None
    communicated_ref: Reference | None = None
    delivered_at: str | None = None
    delivery_ref: str | None = None
    delivery: Literal["unknown", "receipt"] = "unknown"
    closed_reason: str | None = None


@dataclass(kw_only=True, slots=True)
class Event:
    hand_id: str
    seq: int
    at: str
    type: str
    source: str
    key: str
    payload: JsonObject = field(default_factory=dict)


@dataclass(kw_only=True, slots=True)
class Question:
    id: str
    hand_id: str
    asked_at: str
    text: str | None
    assumption: str | None = None
    blocking: bool = False
    # "open" until the controller records the answer ("answered"); erasing the hand sets "erased".
    state: str = "open"
    answer: str | None = None
    answered_by: str | None = None
    answered_at: str | None = None


@dataclass(kw_only=True, slots=True)
class Notice:
    id: str
    budget: Literal["routine", "actionable"]
    line: str
    queued_at: str
    hand_id: str
    attempt: int
    sent_at: str | None = None
    platform_message_id: str | None = None
