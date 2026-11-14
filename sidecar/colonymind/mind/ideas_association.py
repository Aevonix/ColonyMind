"""Read-only feed carrier projection from existing, current Ideas evidence."""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, ConfigDict, Field, StrictStr


class SourceBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: StrictStr = Field(min_length=1, max_length=1024)
    revision: StrictStr = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")


class FeedEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unit_id: StrictStr = Field(min_length=1, max_length=256)
    sources: list[SourceBinding] = Field(max_length=64)


class AssociationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cards: list[FeedEvidence] = Field(max_length=100)


def associate(ideas, cards: list[FeedEvidence]) -> dict:
    """Return one carrier per input, without persisting links or reading display copy.

    Accepted historical receipts deliberately do not relax feed freshness: a
    stale/expired recommendation must disappear from the feed's action too.
    """
    candidates = []
    if ideas is not None and any(card.sources for card in cards):
        now = ideas.mind.clock()
        seen = {}

        def current(source, at, *, fresh=True):
            # One freshness read per distinct source per request; candidates share bindings.
            key = (json.dumps(source, sort_keys=True, default=str), fresh)
            if key not in seen:
                seen[key] = ideas.source_current(source, at, fresh=fresh)
            return seen[key]

        for row in ideas.rows():
            try:
                if not ideas.valid(row, now=now, current=current) or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", row.id) is None:
                    continue
                payload = ideas.open(row.context["idea"]["private"])
                bindings = [SourceBinding.model_validate({k: source[k] for k in ("key", "revision")})
                            for source in payload["sources"]]
                pairs = {(binding.key, binding.revision) for binding in bindings}
                if not pairs:
                    continue
                execution = row.context["idea"].get("execution") or {}
                state = execution.get("status") if execution.get("taskId") else "available"
                if execution.get("taskId") and state not in {"accepted", "working", "done", "failed"}:
                    continue
                candidates.append((pairs, row.id, state))
            except Exception:
                # Unreadable/malformed private candidates fail closed. Never
                # log their payload, bindings, exception text or nested values.
                continue

    projected = []
    for card in cards:
        evidence = {(source.key, source.revision) for source in card.sources}
        revisions = {}
        for key, revision in evidence:
            revisions.setdefault(key, set()).add(revision)
        matches = []
        for pairs, ident, state in candidates:
            shared = evidence & pairs
            if shared and not any(key in revisions and revisions[key] != {revision} for key, revision in pairs):
                matches.append((-len(shared), len(pairs - evidence), ident, state))
        carrier = {"unit_id": card.unit_id}
        if matches:
            _, _, ident, state = min(matches)
            carrier["idea_action"] = {"idea_id": ident, "state": state}
        projected.append(carrier)
    return {"cards": projected, "capabilities": {"feed_idea_association": True}, "ready": ideas is not None}
