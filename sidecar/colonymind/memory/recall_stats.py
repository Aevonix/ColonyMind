"""What recall's reranker did, counted since the sidecar started; and how much a missed turn may inject.

A missed rerank (deadline, busy reranker, error, unusable answer, no time left in the turn, no reranker) makes recall
fall back to its candidate order, which skips the calibrated abstention cutoff. That used to fill the packet, and
the only sign was a warning at most once per five minutes. Every miss is now counted by its reason, turns
(``/context/assemble``) apart from explicit searches and the mind's packets, and shown in health's ``recall``
block beside the judge's outcomes.

A missed turn injects the fallback's fused order up to the packet's own limit (``recall.MEMORY_PACKET_LIMIT`` for a
turn, an explicit search's own limit), as recall did before this count existed. A smaller bound trades answers the
fallback holds for fewer injections on questions that should get no memory, and only a calibration on a
deployment's positives and negatives (answer retention included) can set that trade: a deployment that has one
sets ``COLONYMIND_RECALL_FALLBACK_LIMIT``, which only ever lowers the packet's limit (0 injects nothing).

"""
from __future__ import annotations

import os
import threading
from collections import Counter
from typing import Literal, get_args

from colonymind.memory.recall import MEMORY_PACKET_LIMIT
from colonymind.util.temporal import now_utc

FALLBACK_ENV = "COLONYMIND_RECALL_FALLBACK_LIMIT"
#: Why a rerank the configuration asks for did not score (``selection.RecallSelector._reranked``): counted here by
#: reason, and named by a recall trace (``memory.trace``) in the same words.
RerankMiss = Literal["no_reranker", "budget", "busy", "timeout", "error", "invalid"]
MISS_REASONS: tuple[str, ...] = get_args(RerankMiss)


def _configured_fallback() -> int | None:
    """The deployment's calibrated bound, or None when it has set none (or an unreadable one)."""
    try:
        return max(int(os.environ.get(FALLBACK_ENV, "").strip()), 0)
    except ValueError:
        return None


def fallback_limit(packet_limit: int = MEMORY_PACKET_LIMIT) -> int:
    """The most records a missed rerank may inject into a packet of ``packet_limit``: the packet's own limit, as
    before, unless a deployment set a calibrated smaller bound."""
    configured = _configured_fallback()
    return packet_limit if configured is None else min(configured, packet_limit)


class RecallStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._since = now_utc().isoformat()
            self._lanes = {lane: {"reranked": 0, "scored": 0, "scored_late": 0, "missed": 0,
                                  "missed_by_reason": Counter(), "fallback_rows": 0}
                           for lane in ("turns", "searches")}
            self._judge = {"selected": 0, "within_budget": 0, "over_allowance": 0, "fallback_by_reason": Counter()}

    def reranked(self, *, turn: bool, missed: str | None = None, late: bool = False) -> None:
        """One recall rerank the configuration asked for: scored (``late`` when used after its deadline) or missed
        for ``missed``."""
        with self._lock:
            lane = self._lanes["turns" if turn else "searches"]
            lane["reranked"] += 1
            if missed:
                lane["missed"] += 1
                lane["missed_by_reason"][missed] += 1
            else:
                lane["scored"] += 1
                lane["scored_late"] += bool(late)

    def fallback(self, *, turn: bool, rows: int) -> None:
        """The records a missed rerank injected."""
        with self._lock:
            self._lanes["turns" if turn else "searches"]["fallback_rows"] += rows

    def judged(self, report: dict) -> None:
        """One per-turn selection's outcome (``compass.select_context``'s report)."""
        with self._lock:
            status = str(report.get("status") or "")
            if status == "fallback":
                self._judge["fallback_by_reason"][str(report.get("reason") or "unknown")] += 1
            elif status in self._judge:
                self._judge[status] += 1

    def snapshot(self) -> dict:
        with self._lock:
            lanes = {name: {**lane, "missed_by_reason": dict(lane["missed_by_reason"])}
                     for name, lane in self._lanes.items()}
            judge = {**self._judge, "fallback_by_reason": dict(self._judge["fallback_by_reason"])}
            # A turn's bound on a miss, and whether a deployment set it (else it is the packet's own limit).
            return {"since": self._since, **lanes, "judge": judge, "fallback_limit": fallback_limit(),
                    "fallback_limit_set": _configured_fallback() is not None}


STATS = RecallStats()
