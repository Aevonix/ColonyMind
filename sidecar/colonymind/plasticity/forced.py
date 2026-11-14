"""Deterministic keyed exploration of would-skip or would-hold items.

The caller records the returned manifest row before acting and trains only on observed
outcomes. Binding this record to Instinct's decision log belongs to the first skip learner.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import asdict, dataclass

DEFAULT_SHARE = 0.05  # Design assumption: run five percent of eligible skips or holds.


@dataclass(frozen=True)
class ForcedRun:
    ref: str
    learner: str
    eligible: bool
    forced: bool
    probability: float

    @property
    def outcome_weight(self) -> float:
        """Inverse observation probability, or zero when the skipped item never ran."""
        return 1.0 / self.probability if self.forced or not self.eligible else 0.0

    def manifest_row(self) -> dict:
        """The by-reference sampling record, containing no item text or key."""
        return asdict(self)


class ForcedRuns:
    def __init__(self, key: bytes, *, learner: str, share: float = DEFAULT_SHARE):
        if not isinstance(key, bytes) or not key or not isinstance(learner, str) or not learner:
            raise ValueError("forced runs require an instance key and a learner name")
        if isinstance(share, bool) or not isinstance(share, (int, float)) or not math.isfinite(share) or not 0 < share <= 1:
            raise ValueError("forced-run share is above zero and at most one")
        self._key, self.learner, self.share = key, learner, float(share)

    def choose(self, ref: str, *, eligible: bool) -> ForcedRun:
        if not isinstance(ref, str) or not ref or not isinstance(eligible, bool):
            raise ValueError("a sampling item has a reference and boolean eligibility")
        payload = json.dumps(["forced-run-1", self.learner, ref], separators=(",", ":")).encode()
        value = int.from_bytes(hmac.new(self._key, payload, hashlib.sha256).digest(), "big")
        chosen = value < int(self.share * (1 << 256))
        return ForcedRun(ref, self.learner, eligible, eligible and chosen, self.share if eligible else 1.0)
