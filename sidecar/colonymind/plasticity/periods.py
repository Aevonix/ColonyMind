"""The period ledger: per learner, each period's life (open, closed, discarded) and the manifests it used.

A period (one night by default) gathers what a learner may learn from. It opens, receives its manifests, and closes;
a stacked learner's period also records the digest of the delta it trained (``delta``). A failed gate or an erasure
discards it, with the reason; a discarded period never trains again. Each period is one small file, replaced
atomically, so the ledger survives a crash at any point.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from . import _io

PERIOD_STATES: Tuple[str, ...] = ("open", "closed", "discarded")


class PeriodError(ValueError):
    """A period change the ledger refuses, and why."""


@dataclass(frozen=True)
class Period:
    learner: str
    id: str
    seq: int
    state: str
    opened_at: str
    closed_at: str = ""
    manifests: Tuple[str, ...] = ()
    delta: str = ""
    reason: str = ""


class PeriodLedger:
    """One learner's periods, in ``directory``."""

    def __init__(self, directory: Path, learner: str, *, clock: Optional[Callable[[], datetime]] = None):
        self.directory = Path(directory)
        self.learner = learner
        if clock is None:
            from ..util.temporal import now_utc
            clock = now_utc
        self._clock = clock
        _io.sweep(self.directory)

    def _path(self, period_id: str) -> Path:
        return self.directory / f"{period_id}.json"

    def _save(self, period: Period) -> Period:
        record = asdict(period)
        record["manifests"] = list(period.manifests)
        _io.write_json(self._path(period.id), record)
        return period

    def get(self, period_id: str) -> Period:
        try:
            record = _io.read_json(self._path(period_id))
        except FileNotFoundError:
            raise PeriodError(f"unknown period {period_id}") from None
        record["manifests"] = tuple(record.get("manifests", ()))
        return Period(**record)

    def periods(self) -> List[Period]:
        if not self.directory.is_dir():
            return []
        found = [self.get(path.stem) for path in self.directory.glob("p*.json")]
        return sorted(found, key=lambda period: period.seq)

    def retained(self) -> List[Period]:
        """The closed periods that still train, oldest first."""
        return [period for period in self.periods() if period.state == "closed"]

    def current(self) -> Optional[Period]:
        return next((period for period in self.periods() if period.state == "open"), None)

    def open(self) -> Period:
        if self.current() is not None:
            raise PeriodError(f"learner {self.learner}: a period is already open")
        seq = max((period.seq for period in self.periods()), default=0) + 1
        return self._save(Period(learner=self.learner, id=f"p{seq:06d}", seq=seq, state="open",
                                 opened_at=_io.stamp(self._clock())))

    def add_manifest(self, period_id: str, manifest_id: str) -> Period:
        period = self.get(period_id)
        if period.state != "open":
            raise PeriodError(f"period {period_id} is {period.state}: only an open period takes a manifest")
        if manifest_id in period.manifests:
            return period
        return self._save(replace(period, manifests=period.manifests + (manifest_id,)))

    def close(self, period_id: str, *, delta: str = "") -> Period:
        period = self.get(period_id)
        if period.state != "open":
            raise PeriodError(f"period {period_id} is {period.state}: only an open period closes")
        return self._save(replace(period, state="closed", closed_at=_io.stamp(self._clock()), delta=delta))

    def set_delta(self, period_id: str, delta: str) -> Period:
        period = self.get(period_id)
        if period.state != "closed":
            raise PeriodError(f"period {period_id} is {period.state}: a delta belongs to a closed period")
        return self._save(replace(period, delta=delta))

    def discard(self, period_id: str, *, reason: str) -> Period:
        period = self.get(period_id)
        if period.state == "discarded":
            return period
        return self._save(replace(period, state="discarded", reason=reason))

    def replace_manifest(self, period_id: str, old: str, new: Optional[str]) -> Period:
        """Swap manifest ``old`` for ``new`` (an erasure's rewrite), or drop it when ``new`` is None."""
        period = self.get(period_id)
        if old not in period.manifests:
            raise PeriodError(f"period {period_id} does not use manifest {old}")
        manifests = tuple(manifest for manifest in (new if item == old else item for item in period.manifests)
                          if manifest)
        return self._save(replace(period, manifests=manifests))


__all__ = ["PERIOD_STATES", "Period", "PeriodError", "PeriodLedger"]
