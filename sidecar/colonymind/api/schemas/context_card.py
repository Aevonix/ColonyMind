"""The stored card is ready for consumer prewarming, not proof of a warm KV cache."""
from pydantic import BaseModel, ConfigDict


class ContextCardResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    owner_id: str
    version: int
    text: str
    token_count: int
    token_estimator: str
    local_day: str
    built_at: str
    checked_at: str
    stale: bool
    readiness: str = "ready_for_prewarm"
