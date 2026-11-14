"""Decision providers: the protocol a model tier asks, and the breaker that skips a provider that keeps failing.

A provider answers typed questions about one state with raw probabilities per question and reports the revision
it serves where it can. Its identity for calibration is its ``kind``, the ``model`` the operator configured and
that served ``revision``. Runtime providers are local: ``local-decision.v1`` (a decision server on the operator's
own network) here; the reader over an LLM the operator already serves is added with its measurement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, FrozenSet, Mapping, Protocol

#: Three failures or timeouts in a row skip a provider for five minutes (chosen values, not measured): a dead
#: endpoint does not cost its time limit on every call, and nothing escalates meanwhile.
BREAKER_FAILURES = 3
BREAKER_COOLDOWN_S = 300.0


@dataclass(frozen=True)
class Answer:
    """Raw probabilities per question (``{question: {label: p}}``) and the revision the server reported ("" for
    none)."""

    probabilities: Mapping[str, Mapping[str, float]]
    revision: str = ""


class Provider(Protocol):
    id: str
    kind: str
    model: str
    private: bool
    max_state_chars: int
    types: FrozenSet[str]
    batch: bool
    timeout_ms: int

    async def answer(self, state: str, questions: Mapping[str, Mapping[str, Any]]) -> Answer: ...


def calibration_kind(provider: Any) -> str:
    """The provider kind, qualified by its declared readout when present."""
    kind = str(getattr(provider, "kind", "") or "")
    readout = getattr(provider, "readout", None)
    return kind + "/" + readout if isinstance(readout, str) and readout else kind


@dataclass
class Breaker:
    """Consecutive failures of one provider; open (skip it) for ``cooldown_s`` once they reach ``failures``."""

    failures: int = BREAKER_FAILURES
    cooldown_s: float = BREAKER_COOLDOWN_S
    monotonic: Callable[[], float] = field(default=time.monotonic)
    consecutive: int = 0
    open_until: float = 0.0

    def is_open(self) -> bool:
        return self.monotonic() < self.open_until

    def succeeded(self) -> None:
        self.consecutive = 0
        self.open_until = 0.0

    def failed(self) -> None:
        self.consecutive += 1
        if self.consecutive >= self.failures:
            self.open_until = self.monotonic() + self.cooldown_s
            self.consecutive = 0

    @property
    def state(self) -> str:
        return "open" if self.is_open() else "closed"


__all__ = ["BREAKER_COOLDOWN_S", "BREAKER_FAILURES", "Answer", "Breaker", "Provider", "calibration_kind"]
