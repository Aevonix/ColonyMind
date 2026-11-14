"""Host API schemas — Pydantic models for the /v1/host surface.

Mirrors the TypeScript types in colonymind's src/types.ts.
The sidecar is the source of truth for these schemas.
"""

from __future__ import annotations

import os as _os
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import (AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, field_validator,
                      model_serializer, model_validator)

from colonymind.memory.recall_stats import RerankMiss
from colonymind.work.models import WorkEnvelope
from colonymind.memory.trace import (
    CANDIDATE_ID as _TRACE_CANDIDATE_ID,
    LANE_NAME as _TRACE_LANE_NAME,
    MAX_ROWS as _TRACE_MAX_ROWS,
    Calibration as _TraceCalibrationStatus,
    Exit as _TraceExit,
    Form as _TraceForm,
    RerankOutcome as _TraceRerankOutcome,
    Section as _TraceSection,
)


MAX_NAME_LEN = 256


# --- Identity ---------------------------------------------------------------

class HostIdentity(BaseModel):
    host_id: str
    host_version: Optional[str] = None
    plugin_version: Optional[str] = None
    instance_id: Optional[str] = None


class HostTurnContext(BaseModel):
    session_id: str
    contact_id: str
    channel_id: Optional[str] = None
    turn_id: Optional[str] = None
    locale: Optional[str] = None
    timezone: Optional[str] = None  # per-communication tz override (v0.21.0)
    metadata: Optional[Dict[str, Any]] = None


# --- Messages ---------------------------------------------------------------

class HostMessage(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# --- Health -----------------------------------------------------------------

class TemporalMetrics(BaseModel):
    started_at: Optional[str] = None
    last_sync_at: Optional[str] = None
    last_tick_at: Optional[str] = None
    last_prefetch_at: Optional[str] = None
    silence_hours: Dict[str, Optional[float]] = Field(default_factory=dict)
    stale_flags: List[str] = Field(default_factory=list)
    #: "ok" normally; "unknown" when persisted telemetry could not be restored
    state: Optional[str] = None


class HostHealthResponse(BaseModel):
    status: Literal["ok", "degraded", "starting", "stopping"]
    api_version: str = "1.0.0"
    capabilities: List[str] = []
    notes: Optional[Dict[str, str]] = None
    temporal: Optional[TemporalMetrics] = None
    #: Every reason the status is not "ok", as sentences an operator can act on.
    problems: List[str] = Field(default_factory=list)
    #: What recall's reranker and the per-turn judge did since the sidecar started (``memory.recall_stats``):
    #: every missed rerank by reason, turns apart from explicit searches, and the fallback records injected.
    recall: Optional[Dict[str, Any]] = None
    #: Instinct (``colonymind.instinct``): each decision provider's breaker and the tier it serves, and per point its
    #: states and counts since the sidecar started. Ids and counts only.
    instinct: Optional[Dict[str, Any]] = None


# --- Memory -----------------------------------------------------------------

class MemoryReadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    identity: HostIdentity
    person_id: str = Field(min_length=1, max_length=256)
    source_id: str = Field(min_length=1, max_length=256)
    source_version: str = Field(pattern='^[0-9a-f]{64}$')
    session_id: str = Field(min_length=1, max_length=256)
    source_view: Literal['source', 'assertions', 'image', 'document', 'video'] = 'source'
    asset_hash: Optional[str] = Field(default=None, pattern='^[0-9a-f]{64}$')
    page: Optional[int] = Field(default=None, ge=1, strict=True)
    requested_ms: Optional[int] = Field(default=None, ge=0, le=30000, strict=True)
    claim_id: Optional[str] = Field(default=None, min_length=1, max_length=256)
    offset: int = Field(default=0, ge=0, le=10000000)
    read_revision: Optional[str] = Field(default=None, pattern='^[0-9a-f]{64}$')

    @model_validator(mode='after')
    def canonical_read_selector(self):
        if (self.source_view == 'assertions') != bool(self.claim_id):
            raise ValueError('assertion history requires its source claim ID')
        if (self.source_view in {'image', 'document', 'video'}) != bool(self.asset_hash):
            raise ValueError('image, document and video reads require an exact asset hash')
        if self.source_view in {'image', 'video'} and self.offset:
            raise ValueError('image and video reads take no page offset')
        if (self.source_view == 'video') != (self.requested_ms is not None):
            raise ValueError('video reads require a clip-relative requested_ms')
        if (self.source_view == 'document') != (self.page is not None):
            raise ValueError('document reads require an original page number')
        if self.offset and not self.read_revision:
            raise ValueError('continuation requires the preceding read revision')
        return self


class MemoryReadResponse(BaseModel):
    source: Dict[str, Any]


MEMORY_SEARCH_MAX_LIMIT = 20


# --- Recall trace (colonymind.memory.trace; docs/RECALL-HYBRID.md, Tracing a recall) --------------------------------

_TraceCandidate = Annotated[str, StringConstraints(pattern=_TRACE_CANDIDATE_ID)]
_TraceLane = Annotated[str, StringConstraints(pattern=_TRACE_LANE_NAME)]


class RecallTraceRow(BaseModel):
    """One candidate of a traced recall, or a message a lane found that no candidate carried to ranking.

    Ids, ranks, a score and closed-set values; never text. ``exit`` is null when it is in the packet, and then
    ``shown_by`` names the candidate that shows its words when that is another one. A row whose source, or a
    message it carries, was erased during the recall has the exit ``erased`` and no ids."""
    model_config = ConfigDict(extra="forbid")
    turn_id: Optional[str] = Field(default=None, min_length=1, max_length=256)
    message_hash: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    candidate: Optional[_TraceCandidate] = None
    form: Optional[_TraceForm] = None
    lanes: Dict[_TraceLane, Annotated[int, Field(ge=1)]] = Field(default_factory=dict)
    fused_rank: Optional[int] = Field(default=None, ge=1)
    submitted: bool = False
    rerank_score: Optional[float] = Field(default=None, allow_inf_nan=False)
    exit: Optional[_TraceExit] = None
    shown_by: Optional[_TraceCandidate] = None


class RecallTraceRerank(BaseModel):
    """What the reranker did: how many candidates it was sent (at most ``submission_bound``, in fused order;
    counted when the call starts, so a rerank that then missed says what it was sent), what came of it, or why
    nothing was sent."""
    model_config = ConfigDict(extra="forbid")
    mode: Literal["off", "shadow", "on"]
    outcome: _TraceRerankOutcome
    missed: Optional[RerankMiss] = None
    late: bool = False
    submission_bound: Optional[int] = Field(default=None, ge=1)
    submitted: int = Field(default=0, ge=0)


class RecallTraceCalibration(BaseModel):
    """The configured cutoff's check. ``cutoff`` is the verified cutoff (null unless verified); it applied when the
    rerank scored."""
    model_config = ConfigDict(extra="forbid")
    status: _TraceCalibrationStatus
    cutoff: Optional[float] = Field(default=None, allow_inf_nan=False)


class RecallTracePacket(BaseModel):
    """The packet's count limit, recall's character budget
    and the records it holds."""
    model_config = ConfigDict(extra="forbid")
    limit: Optional[int] = Field(default=None, ge=0)
    max_chars: int = Field(ge=0)
    count: int = Field(ge=0)


class RecallTraceReport(BaseModel):
    """Why a recall kept or left out each candidate, limited to the caller's scope. Built for this response and
    never stored. ``section`` (context assembly only) says whether the recalled section reached the context."""
    model_config = ConfigDict(extra="forbid")
    lanes: Dict[_TraceLane, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)
    rerank: Optional[RecallTraceRerank] = None
    calibration: RecallTraceCalibration
    packet: Optional[RecallTracePacket] = None
    selected: List[_TraceCandidate] = Field(default_factory=list, max_length=64)
    rows: List[RecallTraceRow] = Field(default_factory=list, max_length=_TRACE_MAX_ROWS)
    rows_omitted: int = Field(default=0, ge=0)
    section: Optional[_TraceSection] = None


def _without_absent_trace(data):
    """A response nobody asked to trace is the response it was: no trace key."""
    if isinstance(data, dict) and data.get("trace") is None:
        data.pop("trace", None)
    return data


class MemorySearchWindow(BaseModel):
    """A span of time a search asks about: timezone-aware instants, ``end`` exclusive and after ``start``."""
    model_config = ConfigDict(extra="forbid")
    start: AwareDatetime
    end: AwareDatetime

    @model_validator(mode='after')
    def ordered(self):
        if self.end <= self.start:
            raise ValueError('a search window ends after it starts')
        return self


class MemorySearchRequest(BaseModel):
    """An explicit search of one person's canonical memory.

    The key selects nobody by itself, so the body names the person: a missing
    or blank ``person_id`` is refused, never read as the owner's search, so a
    caller whose guest resolution failed gets no one's memory (security-6).
    ``session_id`` admits that session's session-scoped evidence; without one
    only person-scoped evidence is read and nothing is excluded. A ``limit``
    above the maximum is clamped, not refused. ``trace`` returns why each
    candidate was kept or left out (``RecallTraceReport``): for a credential
    that manages the instance, or for the credential's own person.

    ``window`` is the span the search asks about, whatever the words say: it
    adds the date-window lane and sets which claims are admitted, and never
    removes a quotation (docs/RECALL-HYBRID.md, Recall and time). ``as_of`` is
    the time relative dates and the current view count from, clamped to no
    later than now: not a view of what was believed then.
    """
    model_config = ConfigDict(extra="forbid")
    identity: HostIdentity
    person_id: str = Field(min_length=1, max_length=256)
    session_id: Optional[str] = Field(default=None, max_length=256)
    query: str = Field(min_length=1, max_length=4096)
    limit: int = Field(default=5, ge=1, strict=True)
    timezone: Optional[str] = Field(default=None, max_length=128)
    trace: bool = Field(default=False, strict=True)
    window: Optional[MemorySearchWindow] = None
    as_of: Optional[AwareDatetime] = None

    @field_validator('limit')
    @classmethod
    def clamp_limit(cls, value: int) -> int:
        return min(value, MEMORY_SEARCH_MAX_LIMIT)

    @field_validator('as_of')
    @classmethod
    def clamp_as_of(cls, value):
        from colonymind.util.temporal import now_utc
        return None if value is None else min(value, now_utc())

    @model_validator(mode='after')
    def named_person(self):
        self.person_id = self.person_id.strip()
        if not self.person_id:
            raise ValueError('memory search names the person it searches (person_id)')
        self.session_id = (self.session_id or '').strip() or None
        return self


class RerankRequest(BaseModel):
    identity: HostIdentity
    query: str
    documents: List[str]
    top_k: Optional[int] = 10


class RerankResult(BaseModel):
    index: int
    score: float
    text: str


class RerankResponse(BaseModel):
    results: List[RerankResult] = []
    model: str = ""


class MemorySearchResponse(BaseModel):
    content: str
    count: int = Field(ge=0, le=20)
    source_refs: List[Dict[str, str]]
    watermark: int = Field(ge=0)
    retrieval: Dict[str, str]
    annotation_checks: List[SourceAnnotationCheck] = Field(default_factory=list, max_length=20)
    # Only when the request asked for it.
    trace: Optional[RecallTraceReport] = None

    @model_serializer(mode="wrap")
    def _trace_only_when_asked(self, handler):
        return _without_absent_trace(handler(self))


class VoiceAffectRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    identity: HostIdentity
    person_id: str = Field(min_length=1, max_length=256, pattern=r'\S')

    @field_validator('person_id')
    @classmethod
    def strip_person(cls, value):
        return value.strip()


class VoiceAffectResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    line: str = Field(default='', max_length=120, pattern=r'^[^\[\]\u2013\u2014\r\n]*$')

    @field_validator('line')
    @classmethod
    def voice_budget(cls, value):
        from colonymind.memory.recall import estimate_voice_tokens
        if estimate_voice_tokens(value) > 25:
            raise ValueError('voice affect exceeds 25 cautious tokens')
        return value


class VoiceRecallRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    identity: HostIdentity
    person_id: str = Field(min_length=1, max_length=256)
    session_id: Optional[str] = Field(default=None, max_length=256)
    query: str = Field(min_length=1, max_length=4096)
    k: int = Field(default=5, ge=3, le=5, strict=True)
    max_tokens: int = Field(default=600, ge=300, le=600, strict=True)
    speculative: bool = Field(default=False, strict=True)
    utterance_id: str = Field(min_length=1, max_length=256)
    request_id: str = Field(min_length=1, max_length=256)
    timezone: Optional[str] = Field(default=None, max_length=128)

    @model_validator(mode='after')
    def named_scope(self):
        for name in ('person_id', 'query', 'utterance_id', 'request_id'):
            value = getattr(self, name).strip()
            if not value:
                raise ValueError(f'{name} must not be blank')
            setattr(self, name, value)
        self.session_id = (self.session_id or '').strip() or None
        return self


class VoiceRecallResponse(BaseModel):
    content: str
    count: int = Field(ge=0, le=5)
    source_refs: List[Dict[str, str]]
    watermark: int = Field(ge=0)
    retrieval: Dict[str, str]
    annotation_checks: List[SourceAnnotationCheck] = Field(default_factory=list, max_length=5)
    token_count: int = Field(ge=0, le=600)
    max_tokens: int = Field(ge=300, le=600)
    utterance_id: str
    request_id: str
    speculative: bool


class MemoryRecentRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    identity: HostIdentity
    person_id: str = Field(min_length=1, max_length=256, pattern=r'\S')
    session_id: str = Field(min_length=1, max_length=256, pattern=r'\S')
    platform: str = Field(min_length=1, max_length=64, pattern=r'^[a-z0-9][a-z0-9_.-]*$')
    limit: int = Field(default=8, ge=1, le=20, strict=True)


class MemoryRecentResponse(BaseModel):
    platform: str
    entries: List[Dict[str, Any]] = Field(max_length=20)
    source_refs: List[Dict[str, str]]
    watermark: int = Field(ge=0)
    annotation_checks: List[SourceAnnotationCheck] = Field(default_factory=list, max_length=20)
    coverage: Dict[str, Any]


# --- Context ----------------------------------------------------------------

class ContextAssembleRequest(BaseModel):
    identity: HostIdentity
    context: HostTurnContext
    audience: Optional[Literal["viewer", "owner", "shared", "global"]] = None
    incoming_message: HostMessage
    available_tools: Optional[List[str]] = None
    citations_mode: Optional[Literal["off", "inline", "appendix"]] = None
    include_initiatives: Optional[bool] = None  # v0.13.0
    # The characters the host puts inline for this context, each section counted as its ``## <title>`` heading
    # and body with a blank line between. Past it a host may move the context out of line (Hermes spills a
    # longer prefetch to a file). The per-turn selection fits the whole context within it, and the resident
    # blocks (colonymind/memory/blocks.py) are fitted to the same number, capped by context.budget_chars.
    inline_chars: Optional[int] = Field(None, ge=1)
    # The fingerprints of the sections the host's session's last turn carried, which it will send as one line
    # each. Assembly caps count the full sections even when the host will collapse them.
    shown: Optional[List[str]] = Field(default=None, max_length=256)
    # How long the host waits for this response before it goes on without it (Hermes drops a later answer). The
    # turn's recall budget is derived from it (``memory.turn_budget``); without it the sidecar assumes
    # COLONYMIND_CONTEXT_HOST_WAIT_MS (default 7,500), which also caps longer requested waits.
    wait_ms: Optional[int] = Field(None, ge=1, le=600_000)
    # Why this turn's recall kept or left out each candidate (``RecallTraceReport``): for a credential that
    # manages the instance only.
    trace: bool = Field(default=False, strict=True)
    # Short, redacted descriptions of the records this turn's final recall contains.
    recall_notice_details: bool = Field(default=False, strict=True)
    # A memory provider from before M5 also sends ``projection_policy``, and one from before the
    # recall fix sends ``session_history``; extra fields are ignored. A guest's context is
    # contact-scoped by construction, and recall keeps the session's own turns, so neither changes
    # anything.


class ContextSection(BaseModel):
    id: str
    title: Optional[str] = None
    body: str
    priority: Optional[int] = None
    citations: Optional[List[Dict[str, Any]]] = None


class ContextProjectionAttestation(BaseModel):
    """Server-sealed viewer/projection posture for one context response."""

    model_config = ConfigDict(populate_by_name=True)
    schema_name: Literal["ContextProjectionAttestationV1"] = Field(
        default="ContextProjectionAttestationV1",
        alias="schema",
    )
    version: Literal[1] = 1
    viewer_person_id: str = ""
    viewer_attested: bool = False
    viewer_is_owner: bool = False
    p8_mode: Literal["off", "shadow", "live"] = "off"
    projection_backend: Literal["unavailable", "p8", "canonical_sources"] = "unavailable"
    scoped_projection_ready: bool = False
    legacy_global_allowed: bool = False


class ContextAssembleResponse(BaseModel):
    recall_exit: Optional[Dict[str, Any]] = None
    sections: List[ContextSection] = []
    notices: Optional[List[str]] = None
    projection_attestation: Optional[ContextProjectionAttestation] = None
    source_erasure_watermark: Optional[int] = None
    # What the resident-block fit did (colonymind/memory/blocks.py): the budget, the sizes before and after,
    # and every block rotated or held back with the SHA-256 of its full text. Fitting is mandatory.
    budget: Optional[Dict[str, Any]] = None
    # Only when the request asked for it.
    trace: Optional[RecallTraceReport] = None
    recall_notice_details: Optional[List[str]] = None

    @model_serializer(mode="wrap")
    def _trace_only_when_asked(self, handler):
        result = _without_absent_trace(handler(self))
        if self.recall_exit is None:
            result.pop("recall_exit", None)
        if self.recall_notice_details is None:
            result.pop("recall_notice_details", None)
        return result


class MemoryEmbedRequest(BaseModel):
    identity: HostIdentity
    inputs: List[str]  # Kept for backward compat — use texts instead
    texts: Optional[List[str]] = None  # Preferred: list of texts to embed
    model: Optional[str] = None


class MemoryEmbedResponse(BaseModel):
    model: str
    vectors: List[List[float]]


class EmbedHealthResponse(BaseModel):
    provider: str = ""
    model: str = ""
    dims: int = 0
    latency_ms: float = 0.0
    status: str = "unknown"
    error: Optional[str] = None
    modalities: List[str] = ["text"]
    multimodal_enabled: bool = False


class BackfillRequest(BaseModel):
    identity: HostIdentity
    collection: Optional[str] = None
    batch_size: int = 64


class BackfillResponse(BaseModel):
    task_id: str = ""
    status: str = "started"  # started | completed | failed
    total: int = 0
    processed: int = 0
    failed: int = 0
    skipped: int = 0
    duration_s: float = 0.0
    errors: List[str] = []


class MigrateRequest(BaseModel):
    identity: HostIdentity
    old_model_id: Optional[str] = None
    batch_size: int = 64


class MigrateResponse(BaseModel):
    task_id: str = ""
    status: str = "started"  # started | completed | failed
    collections_migrated: int = 0
    vectors_migrated: int = 0
    vectors_failed: int = 0
    duration_s: float = 0.0
    errors: List[str] = []
    generation_id: str = ""
    fingerprint: str = ""


class IndexRequest(BaseModel):
    identity: HostIdentity
    items: List[dict]  # [{text, collection, id, metadata?}] or [{image, mime_type, caption, collection, id}]


class IndexResponse(BaseModel):
    indexed: int = 0
    failed: int = 0
    model: str = ""


class ImageEmbedRequest(BaseModel):
    identity: HostIdentity
    image: Optional[str] = None  # Base64-encoded image or data URL
    image_url: Optional[str] = None  # Public http(s) URL to image
    mime_type: Optional[str] = None
    caption: Optional[str] = None
    collection: Optional[str] = None
    id: Optional[str] = None


class ImageEmbedResponse(BaseModel):
    model: str
    vector: List[float]
    image_hash: str = ""
    image_ref: str = ""
    thumbnail_ref: str = ""
    caption: str = ""
    width: int = 0
    height: int = 0
    modality: str = "image"


class ImageBatchEmbedRequest(BaseModel):
    identity: HostIdentity
    images: List[dict]  # [{image, image_url, mime_type, caption}]
    collection: Optional[str] = None


class ImageBatchEmbedResponse(BaseModel):
    model: str
    results: List[dict]  # [{vector, image_hash, caption, ...}]


class MultimodalSearchRequest(BaseModel):
    identity: HostIdentity
    query: Optional[str] = None  # Text query
    query_image: Optional[str] = None  # Base64 image for image-based search
    collection: Optional[str] = None
    filter_modality: Optional[str] = None  # "text" or "image"
    limit: int = 10
    min_score: float = 0.0


class MultimodalSearchResponse(BaseModel):
    results: List[dict]
    model: str = ""


# --- Sender identity ---------------------------------------------------------

class HostSender(BaseModel):
    """WHO produced the user side of this turn (docs/RELATIONSHIPS.md).

    Per-message sender identity so the sidecar attributes group traffic to
    the real speaker server-side, independent of any client-side contact
    caching. Optional and additive: hosts that cannot supply it keep the
    legacy context.contact_id behavior."""
    platform: str = Field(..., max_length=64)
    user_id: str = Field(..., max_length=256)
    display_name: str = Field(default="", max_length=256)
    group_id: str = Field(default="", max_length=256)


# --- Turns ------------------------------------------------------------------

class CheckpointMessage(BaseModel):
    """Direct source content; injected API context is not part of this schema."""
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: Union[str, List[Dict[str, Any]]]

    @model_validator(mode="after")
    def nonempty_content(self):
        if not self.content or isinstance(self.content, str) and not self.content.strip():
            raise ValueError("checkpoint messages must contain source content")
        return self


class TurnMessage(HostMessage):
    """Direct source blocks are retained; ordinary cognition receives text only."""
    content: Union[str, List[Dict[str, Any]]]

    def text_content(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "\n".join(block["text"] for block in self.content
                         if block.get("type") in {"text", "input_text", "output_text"}
                         and isinstance(block.get("text"), str))


class SourceReference(BaseModel):
    """A supplied canonical revision, not a generated citation or truth claim."""
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(min_length=1, max_length=256)
    source_version: str = Field(pattern=r"^[0-9a-f]{64}$")


class SourceAnnotationCheck(BaseModel):
    """Exact candidate membership and correction set already supplied to a model."""
    model_config = ConfigDict(extra='forbid')
    source_refs: List[SourceReference] = Field(min_length=1, max_length=512)
    message_hashes: Dict[str, List[str]]
    annotation_ids: List[str] = Field(max_length=512)

    @model_validator(mode='after')
    def exact_membership(self):
        import re
        identifiers = {ref.source_id for ref in self.source_refs}
        if len(identifiers) != len(self.source_refs) or set(self.message_hashes) != identifiers:
            raise ValueError('annotation checks require exact source membership')
        if any(len(hashes) > 512 or any(not re.fullmatch('[0-9a-f]{64}', value) for value in hashes)
               for hashes in self.message_hashes.values()):
            raise ValueError('invalid annotation message hashes')
        if any(not value or len(value) > 256 for value in self.annotation_ids):
            raise ValueError('invalid annotation source IDs')
        return self


class SourceInputReference(BaseModel):
    """Exact admitted input before canonical media normalization."""
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(min_length=1, max_length=256)
    input_message_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class TurnSyncRequest(BaseModel):
    identity: HostIdentity
    context: HostTurnContext
    sender: Optional[HostSender] = None
    topics: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    pending_tasks: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    # Raw message fields — populated by Hermes provider and MCP tools.
    # When structured fields are empty but raw messages are present,
    # the sidecar runs extraction from the raw messages.
    user_message: Optional[Union[HostMessage, TurnMessage]] = None
    assistant_message: Optional[Union[HostMessage, TurnMessage]] = None
    assistant_source_refs: Optional[List[SourceReference]] = Field(default=None, min_length=1)
    assistant_input_refs: Optional[List[SourceInputReference]] = Field(default=None, min_length=1)
    # Frozen safe work facts, admitted with their owner source.
    work: Optional[WorkEnvelope] = None
    source_only: Optional[Literal[True]] = None
    # Model that produced the assistant side of this turn (optional, additive).
    # Lets the mining layer detect provider escalations / cloud failovers from
    # real per-turn metadata instead of guessing from text.
    # Sent by the adapter; nothing reads it since the escalation miner went (M9). Kept so an older
    # adapter against this sidecar never gets a 422.
    model: Optional[str] = None
    # Evidence-only checkpoints never trigger ordinary turn/relationship effects.
    checkpoint_messages: Optional[List[CheckpointMessage]] = Field(
        default=None, min_length=1, max_length=20_000,
    )

    @model_validator(mode="after")
    def bounded_checkpoint(self):
        if len(self.model_dump_json().encode("utf-8")) > 8 * 1024 * 1024:
            raise ValueError("turn exceeds the 8 MiB source limit")
        if self.checkpoint_messages is not None:
            if self.work is not None:
                raise ValueError('checkpoint cannot create owner work')
            if self.sender is not None or self.user_message is not None or self.assistant_message is not None:
                raise ValueError("checkpoint cannot also represent an ordinary turn")
            if not self.context.turn_id:
                raise ValueError("checkpoint requires an idempotent turn id")
        def nonempty(message):
            return message is not None and bool(message.content.strip() if isinstance(message.content, str) else message.content)
        if self.source_only and (self.checkpoint_messages is not None or not any(
                nonempty(message) for message in (self.user_message, self.assistant_message))):
            raise ValueError('source-only delivery requires nonempty direct source messages')
        if (self.assistant_source_refs or self.assistant_input_refs) and not any(message is not None and message.role == 'assistant' and nonempty(message)
                for message in [self.assistant_message, *(self.checkpoint_messages or [])]):
            raise ValueError('source references require an assistant message')
        if self.assistant_input_refs and (not self.context.turn_id or self.checkpoint_messages is not None):
            raise ValueError('input parents require an identified direct assistant source')
        return self


class TurnSyncResponse(BaseModel):
    accepted: bool
    continuity_updated: bool
    skipped_reason: Optional[str] = None
    errors: Optional[List[str]] = None
    source_recorded: bool = False
    work_recorded: Optional[bool] = None
    work_skipped_reason: Optional[str] = None


# --- Events -----------------------------------------------------------------

# --- Contacts ---------------------------------------------------------------

class ContactResponse(BaseModel):
    contact_id: str
    display_name: Optional[str] = None
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    organization: Optional[str] = None
    relationship_score: float = 0.0
    trust_tier: Optional[str] = None
    may_contact: str = "ask"                    # never | ask | auto (architecture 7.4)
    cadence_minutes: Optional[int] = None       # owner-set check-in cadence; None = none
    digest: Optional[str] = None                # the per-contact digest
    digest_sources: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    privacy_level: Optional[str] = None
    person_node_id: Optional[str] = None
    notes: Optional[str] = None
    import_source: Optional[str] = None
    first_seen_at: Optional[str] = None
    last_interaction_at: Optional[str] = None
    interaction_count: int = 0
    enrichment_source: List[str] = Field(default_factory=list)
    enrichment_last_at: Optional[str] = None
    deleted_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    timezone: Optional[str] = None  # IANA tz for this contact (v0.21.0)
    introduced_by: Optional[str] = None        # contact_id of whoever introduced them
    met_via: Optional[Dict[str, Any]] = None   # {channel, scope_id, ...} provenance


class ContactListResponse(BaseModel):
    contacts: List[ContactResponse] = []
    source_filter: Optional[str] = None
    total: int = 0


# --- Temporal awareness (v0.21.0) -------------------------------------------

class TemporalConfigRequest(BaseModel):
    agent_timezone: Optional[str] = None          # set the agent's home tz
    default_contact_timezone: Optional[str] = None  # fallback tz for contacts
    clear_default_contact_timezone: bool = False    # explicitly clear the fallback


class TemporalConfigResponse(BaseModel):
    agent_timezone: str
    default_contact_timezone: Optional[str] = None
    now_utc: str
    now_agent_local: str
    agent_local_clock: str


class ContactTimezoneRequest(BaseModel):
    timezone: Optional[str] = None  # None clears it


class TimelineEvent(BaseModel):
    seq: int
    type: str
    at: str                       # ISO timestamp the event was recorded
    when: str                     # humanized, e.g. "3h ago"
    bucket: str                   # coarse, e.g. "today"
    summary: Optional[str] = None
    contact_id: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)


class TimelineResponse(BaseModel):
    since: str
    count: int
    digest: str                   # human-readable rollup for the agent
    events: List[TimelineEvent] = Field(default_factory=list)
    has_more: bool = False


class TemporalContact(BaseModel):
    contact_id: str
    name: str
    timezone: Optional[str] = None
    last_interaction_at: Optional[str] = None
    days_since: float
    cadence_days: float
    overdue: bool
    overdue_ratio: float


class TemporalContactsResponse(BaseModel):
    now: str
    count: int
    contacts: List[TemporalContact] = Field(default_factory=list)


class ContactHandleIn(BaseModel):
    gateway: str
    address: str
    is_primary: bool = False
    verified: bool = False


class ContactCreateRequest(BaseModel):
    display_name: str
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    organization: Optional[str] = None
    trust_tier: str = "regular"
    may_contact: str = "ask"                    # a tier implies no permission; only the owner raises it
    cadence_minutes: Optional[int] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    handles: List[ContactHandleIn] = []


class ContactIntroRequest(BaseModel):
    """Capture an organic introduction: the agent met/learned of a new person.

    Creates (or annotates) a PROVISIONAL contact with introduction provenance.
    A provisional contact is inert by design: trust_tier defaults to 'unknown'
    and may_contact starts at 'ask', so capturing an intro never grants anyone
    outreach permission; only the owner raises it.
    """
    name: str
    gateway: Optional[str] = None              # optional handle to attach
    address: Optional[str] = None
    introduced_by: Optional[str] = None        # contact_id of the introducer
    met_via: Optional[Dict[str, Any]] = None   # {channel, scope_id, ...}
    note: Optional[str] = None
    trust_tier: str = "unknown"                # provisional; never grants 1:1 rights


class ContactIntroResponse(BaseModel):
    contact: ContactResponse
    created: bool        # True if a new provisional contact was made; False if annotated existing


class ContactStyleRequest(BaseModel):
    identity: HostIdentity
    person_id: str


class ContactStyleResponse(BaseModel):
    person_id: str
    formality: Optional[str] = None
    tone: Optional[str] = None
    notes: Optional[Dict[str, Any]] = None


# --- Cognition --------------------------------------------------------------


# --- Delivery ---------------------------------------------------------------

# --- Skills -----------------------------------------------------------------


# --- Autonomy ---------------------------------------------------------------

# --- Configure (Host LLM Config) -------------------------------------------

class HostConfigureRequest(BaseModel):
    identity: HostIdentity
    llm: Optional[Dict[str, Any]] = Field(None, description="LLM provider config from host")


class HostConfigureResponse(BaseModel):
    configured: bool = True
    provider: Optional[str] = None
    # Values may be bare model strings or per-tier object specs, so the echo
    # must accept both (a str-only type 500s on multi-endpoint configs).
    models: Optional[Dict[str, Any]] = None
    routing: Optional[Dict[str, Any]] = None


# --- Models (local LLM discovery) -------------------------------------------

class ModelInfo(BaseModel):
    id: str
    provider: Optional[str] = None
    size: Optional[int] = None
    owned_by: Optional[str] = None


class ModelListResponse(BaseModel):
    provider: str = ""
    base_url: Optional[str] = None
    models: List[ModelInfo] = []
    discovered: bool = False
    error: Optional[str] = None
    routing: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Commitment Tracking
# ---------------------------------------------------------------------------

class CommitmentCreateRequest(BaseModel):
    person_id: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1, max_length=1000)
    due_at: Optional[str] = None
    priority: int = Field(default=50, ge=0, le=100)
    # "introspection": a same-interaction owed follow-up surfaced by the per-turn
    # introspection reflex (e.g. "text me the result"), as opposed to a durable
    # future promise ("cognition"). Such a commitment carries a deliverable tag in
    # `metadata`: {"kind": "deliverable", "content": <what to send>,
    # "channel_hint": "sms"|"dm"|..., "delivered": bool}. The autonomy loop turns an
    # undelivered deliverable into an agent_action so the host actually sends it.
    source_type: Literal["manual", "autonomy", "cognition", "introspection"] = "manual"
    source_context: Optional[str] = None
    # When true, an open commitment for the same person that already says the
    # same thing (normalized match) is returned instead of creating a twin.
    # Agent/tool callers should set this; the raw API default stays off.
    dedupe: bool = False
    metadata: Optional[Dict[str, Any]] = None


class CommitmentUpdateRequest(BaseModel):
    status: Optional[Literal["fulfilled", "cancelled"]] = None
    fulfilled_at: Optional[str] = None
    description: Optional[str] = Field(None, min_length=1, max_length=1000)
    due_at: Optional[str] = None
    priority: Optional[int] = Field(None, ge=0, le=100)
    metadata: Optional[Dict[str, Any]] = None
    # Resolution semantics: WHY the item is being settled, so the system can
    # learn from it. When `outcome` is set, `status` may be omitted — it is
    # derived (done -> fulfilled, everything else -> cancelled) and the
    # resolution {outcome, note, by, at} is recorded in metadata.
    outcome: Optional[Literal["done", "invalid", "duplicate", "wont_do", "obsolete"]] = None
    reason: Optional[str] = Field(None, max_length=300)
    resolved_by: Optional[str] = Field(None, max_length=60)


class CommitmentResponse(BaseModel):
    id: str
    person_id: str
    description: str
    made_at: str
    due_at: Optional[str] = None
    fulfilled_at: Optional[str] = None
    status: str
    source_type: str
    source_context: Optional[str] = None
    priority: int
    metadata: Optional[Dict[str, Any]] = None
    # True when a dedupe-create returned an existing open item.
    deduped: bool = False


class CommitmentListResponse(BaseModel):
    commitments: List[CommitmentResponse] = []
    total: int
    limit: int
    offset: int


class ConcernResolveRequest(BaseModel):
    """Owner/agent resolution of a workspace concern. `outcome` says why:
    done means handled, the rest are flavors of "stop tracking this".
    With cascade on (default), sources the concern was raised from (e.g. an
    overdue commitment) are settled too — otherwise the ingest loop re-raises
    the concern from the still-open source and the resolve is cosmetic."""
    note: str = Field("resolved by owner", max_length=300)
    outcome: Literal["done", "invalid", "duplicate", "wont_do", "obsolete"] = "done"
    cascade: bool = True
    resolved_by: str = Field("owner", max_length=60)


# ---------------------------------------------------------------------------
# Cognition Substrate
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Surprise Engine
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Initiatives
# ---------------------------------------------------------------------------

class InitiativeMedia(BaseModel):
    url: Optional[str] = None
    media_id: Optional[str] = None
    mime_type: str

    @model_validator(mode="after")
    def require_reference(self):
        if not self.url and not self.media_id:
            raise ValueError("media needs a url or media_id")
        return self


class InitiativePresentation(BaseModel):
    labels: List[str] = Field(default_factory=list)
    badges: List[str] = Field(default_factory=list)
    action_label: Optional[str] = None
    icon: Optional[InitiativeMedia] = None
    preview: Optional[InitiativeMedia] = None
    fit_reason: Optional[str] = None
    build_summary: Optional[str] = None
    detail_description: Optional[str] = None


class InitiativeItem(BaseModel):
    id: Optional[str] = Field(default=None, min_length=1)
    type: str
    selectable: bool = True
    is_selected: bool = True
    title: str
    summary: str = ""
    build_summary: str = ""


class InitiativeFeedbackRequest(BaseModel):
    direction: Literal["up", "down"]
    reason: Optional[str] = Field(default=None, max_length=600)
    surface: str = Field(min_length=1, max_length=256)


class InitiativeEngagementRequest(BaseModel):
    receipt_id: str = Field(min_length=1, max_length=256)
    kind: Literal["impression", "open"]
    visible_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    surface: str = Field(min_length=1, max_length=256)
    anchor: str = Field(min_length=1, max_length=256)


class InitiativeCreateRequest(BaseModel):
    initiative_type: Literal[
        "PROACTIVE_MESSAGE", "CALENDAR_REMINDER", "TASK_SUGGESTION",
        "RESEARCH_DEEP_DIVE", "SKILL_EVICT", "CODING"
    ]
    title: str
    description: str
    priority: int = Field(default=0, ge=0, le=100)
    timeout_seconds: int = Field(default=3600, ge=60, le=86400)
    context: Dict[str, Any] = Field(default_factory=dict)
    entity_id: Optional[str] = None
    target_agent_id: Optional[str] = None
    dedup_key: Optional[str] = None
    presentation: Optional[InitiativePresentation] = None
    items: List[InitiativeItem] = Field(default_factory=list)


class InitiativeResponse(BaseModel):
    id: str
    initiative_type: str
    title: str
    description: str
    priority: int
    # Host lifecycle plus the mind's intention statuses, which share the table.
    status: Literal[
        "pending", "assigned", "acknowledged", "in_progress",
        "completed", "failed", "cancelled",
        "proposed", "asked", "approved", "dispatched", "sending",
        "done", "expired", "sent", "uncertain", "dropped",
    ]
    timeout_seconds: int
    context: Dict[str, Any]
    # v0.16.0: durability contract for the context snapshot —
    # 'durable' (safe to act on as-is) or 'volatile' (check
    # context_captured_at against the type's freshness TTL first).
    context_durability: Optional[str] = None
    # The SUBJECT of the initiative (a person, a PR, a commitment) —
    # distinct from target/assigned agent (who EXECUTES it).
    entity_id: Optional[str] = None
    target_agent_id: Optional[str]
    assigned_agent_id: Optional[str]
    dedup_key: Optional[str]
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    created_at: str
    acknowledged_at: Optional[str]
    completed_at: Optional[str]
    failed_at: Optional[str]
    expires_at: Optional[str]
    last_response: Optional[Dict[str, str]] = None


class InitiativeOrigin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    surface: Literal["feed", "ideas", "chat"]
    unit_id: Optional[str] = Field(default=None, min_length=1)


class InitiativeDetailResponse(InitiativeResponse):
    presentation: InitiativePresentation = Field(default_factory=InitiativePresentation)
    items: List[InitiativeItem] = Field(default_factory=list)
    origin: Optional[InitiativeOrigin] = None


class InitiativeListResponse(BaseModel):
    initiatives: List[InitiativeResponse]
    total: int


class InitiativeCompleteRequest(BaseModel):
    agent_id: str
    result: Dict[str, Any] = Field(default_factory=dict)


class InitiativeFailRequest(BaseModel):
    agent_id: str
    error_message: str


class InitiativePriorityRequest(BaseModel):
    priority: int = Field(ge=0, le=100)


# --- Agent Snapshot ---------------------------------------------------------

class AgentSnapshotInitiative(BaseModel):
    id: str
    type: str
    description: str
    priority: float
    status: str
    rationale: Optional[str] = None
    action_hint: Optional[str] = None
    entity_id: Optional[str] = None
    dedup_key: Optional[str] = None
    created_at: str
    expires_at: Optional[str] = None
    assigned_agent_id: Optional[str] = None
    acknowledged_at: Optional[str] = None
    completed_at: Optional[str] = None
    failed_at: Optional[str] = None
    failed_reason: Optional[str] = None


class AgentSnapshotResponse(BaseModel):
    timestamp: str
    telemetry: Dict[str, Any]
    pending_initiatives: List[AgentSnapshotInitiative]
    pending_count: int
    assigned_count: int
    failed_count: int
    recently_completed: List[AgentSnapshotInitiative]
    autonomy_mode: str
    autonomy_running: bool
    last_tick_age_minutes: Optional[float] = None
    flags: List[str]


class RecordOutreachRequest(BaseModel):
    # Agent identity is deployment-specific — configure COLONYMIND_AGENT_NAME
    # rather than relying on this generic fallback.
    agent_id: str = Field(
        default_factory=lambda: _os.environ.get("COLONYMIND_AGENT_NAME", "agent")
    )
    channel: str = "whatsapp"
    contact_id: Optional[str] = None
    reason: Optional[str] = None


class RecordOutreachResponse(BaseModel):
    recorded_at: str
    last_agent_outreach_at: str


# ── Trust scopes (context-scoped authorization) ──────────────────────────────

class ScopeAuthzResponse(BaseModel):
    """Whether a sender is authorized WITHIN a scope (group). Says nothing about 1:1."""
    authorized: bool
    scope_id: Optional[str] = None
    granted_tier: Optional[str] = None
    contact_id: Optional[str] = None
    active: bool = False


class ScopeMemberIn(BaseModel):
    gateway: Optional[str] = None      # handle gateway (resolve / auto-create a shadow)
    address: Optional[str] = None      # handle address
    contact_id: Optional[str] = None   # or a known contact directly
    name: Optional[str] = None         # display name for an auto-created shadow
    role: str = "member"


class ScopeCreateRequest(BaseModel):
    platform: str
    external_id: str                   # the group/conversation id on the platform
    label: Optional[str] = None
    scope_type: str = "group"
    granted_tier: str = "group_guest"
    created_by: str = "agent"
    members: List[ScopeMemberIn] = Field(default_factory=list)


class ScopeResponse(BaseModel):
    scope_id: str
    scope_type: str
    platform: Optional[str] = None
    external_id: Optional[str] = None
    label: Optional[str] = None
    granted_tier: str
    active: bool
    members: List[str] = Field(default_factory=list)   # contact_ids


class ScopeDeactivateRequest(BaseModel):
    scope_id: Optional[str] = None
    platform: Optional[str] = None
    external_id: Optional[str] = None


class ScopePromoteRequest(BaseModel):
    contact_id: str
    to_tier: str = "regular"
