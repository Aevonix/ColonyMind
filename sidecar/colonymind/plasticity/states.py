"""Automatic promotion after offline gates, and rollback on measured live regression.

A candidate is a build being checked. Passing gates promote it immediately, including a learner's first revision.
A failed build or a later regression withdraws the closure and discards its new periods. Withdrawn identities never
return. Every mutation shares the root's state lock with erasure, including chain moves and rebuilds.
Rollback publishes its intent before discarding data. Pending failed weights never serve, and the next controller
or comparator request completes the recorded move and report without repeating the rollback.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from . import _io
from .audience import BREADTH, narrowest
from .gates import GateReport
from .lineage import Lineage
from .locking import StateLock
from .revisions import Closure, RevisionError
from .store import LearnerStore

STATES: Tuple[str, ...] = ("candidate", "serving", "superseded", "withdrawn")


class StateError(ValueError):
    """A state change that may not happen, and why. ``code`` names the refusal with a short fixed word, so a caller
    can report which refusal it met without repeating the message."""

    def __init__(self, message: str, *, code: str = "state"):
        super().__init__(message)
        self.code = code


class WithdrawnClosure(StateError):
    """A request to propose or promote a closure that was withdrawn: withdrawn identities never return. The nightly
    reads this one as "already judged" when its own derivation names that identity again (``nightly.Runtime``)."""

    def __init__(self, closure_id: str):
        super().__init__(f"closure {closure_id} is withdrawn and never serves", code="withdrawn")
        self.closure_id = closure_id


class Controller:
    """Moves one learner's revisions between states (the module docstring)."""

    def __init__(self, store: LearnerStore, *, live: bool = True, lineage: Optional[Lineage] = None,
                 clock: Optional[Callable[[], datetime]] = None, lock=None):
        self.store = store
        # ``live`` remains a source-compatible argument; every caller uses the same promotion policy.
        self.lock = lock if lock is not None else StateLock(store.root)
        self.lineage = lineage
        if clock is None:
            from ..util.temporal import now_utc
            clock = now_utc
        self._clock = clock

    @property
    def revisions(self):
        return self.store.revisions

    def events(self) -> List[Dict[str, Any]]:
        """Durable report lines, oldest first; identifiers and reasons only."""
        path = self.store.directory / "transitions.json"
        with self.lock:
            try:
                return _io.read_json(path)
            except FileNotFoundError:
                return []

    def _event(self, event: str, closure: str, *, previous: str = "", reason: str = "") -> None:
        records = self.events()
        records.append({"learner": self.store.name, "event": event, "closure": closure,
                        "previous": previous, "reason": reason, "at": _io.stamp(self._clock())})
        _io.write_json(self.store.directory / "transitions.json", records)

    def previous(self, closure_id: str) -> Closure:
        """The actual closure serving immediately before this promotion, retained for live comparison."""
        try:
            record = _io.read_json(self.store.directory / "promotions" / f"{closure_id}.json")
            previous = self.revisions.get(record["previous"])
            if self.revisions.status(previous.id).state != "withdrawn":
                return previous
        except (FileNotFoundError, RevisionError):
            pass
        return self.revisions.base

    # -- candidates ------------------------------------------------------------------------------------------------

    def propose(self, closure: Closure, state: Mapping[str, np.ndarray]) -> Closure:
        """Write ``closure`` as a candidate (its gates decide what happens next). A closure may not claim a wider
        audience than what it learned from: its manifests' admitted rows, its parent's when it stacks on it, and the
        closures of other learners it learned from."""
        with self.lock:
            self.recover_rollback()
            if self.revisions.status(closure.id).state == "withdrawn":
                raise WithdrawnClosure(closure.id)
            self._check_audience(closure)
            existed = self.revisions.exists(closure.id)
            self.revisions.write(closure, state)
            if not existed:
                self.revisions.set_status(closure.id, "candidate")
            return closure

    def _check_audience(self, closure: Closure) -> None:
        if closure.audience not in BREADTH:
            raise StateError(f"closure {closure.id}: audience {closure.audience!r} is not one of "
                             f"{', '.join(BREADTH)}", code="audience_unknown")
        try:
            learned = self.store.learned_audiences(closure)
        except RevisionError as error:
            raise StateError(str(error), code="lineage_missing") from None
        for learner, upstream_id in closure.upstream:
            store = self.lineage.stores.get(learner) if self.lineage is not None else None
            found = store.find(upstream_id) if store is not None else None
            if found is None:
                raise StateError(f"closure {closure.id} learned from closure {upstream_id} of learner {learner}, "
                                 "which is not known: its audience cannot be shown", code="upstream_unknown")
            learned.append(found.audience)
        if learned and BREADTH[closure.audience] > BREADTH[narrowest(learned)]:
            raise StateError(f"closure {closure.id} claims audience {closure.audience}, wider than what it learned "
                             f"from ({narrowest(learned)})", code="audience_wider")

    def consider(self, closure_id: str, report: GateReport, *, period: Optional[str] = None) -> str:
        """Promote a passing candidate immediately, or withdraw it and discard its introduced periods."""
        with self.lock:
            self.recover_rollback()
            return self._consider(closure_id, report, period=period)

    def _consider(self, closure_id: str, report: GateReport, *, period: Optional[str] = None) -> str:
        if report.closure != closure_id:
            raise StateError(f"the report judges closure {report.closure}, not {closure_id}", code="report_mismatch")
        status = self.revisions.status(closure_id)
        if status.state == "withdrawn":
            raise WithdrawnClosure(closure_id)
        if report.awaiting:
            return "awaiting_evidence"
        if status.state == "serving":
            return "promoted"
        stale = [period_id for period_id in self.revisions.get(closure_id).periods
                 if self.store.periods.get(period_id).state == "discarded"]
        if stale:
            reason = "trained on discarded period(s) " + ", ".join(stale)
            self.revisions.set_status(closure_id, "withdrawn", reason=reason)
            self._event("dropped", closure_id, reason=reason)
            return "dropped"
        if not report.passed:
            reason = "failed gates: " + ", ".join(report.failed)
            self._withdraw(closure_id, reason=reason, also=(period,) if period else ())
            self._event("dropped", closure_id, reason=reason)
            return "dropped"
        self._promote(closure_id)
        return "promoted"

    def _promote(self, closure_id: str) -> Closure:
        previous = self.revisions.serving()
        if previous.id == closure_id:
            # Retrying after the pointer swap must keep the original comparison revision.
            previous = self.previous(closure_id)
        _io.write_json(self.store.directory / "promotions" / f"{closure_id}.json",
                       {"previous": previous.id})
        closure = self.revisions.promote(closure_id, previous=previous.id)
        if previous.id not in (self.revisions.base.id, closure.id):
            self.revisions.set_status(previous.id, "superseded")
        self.revisions.set_status(closure.id, "serving")
        self._event("promoted", closure.id, previous=previous.id)
        if self.lineage is not None:
            parent = previous
            if closure.parent and closure.parent != previous.id:
                try:
                    parent = (self.revisions.base if closure.parent == self.revisions.base.id
                              else self.revisions.get(closure.parent))
                except RevisionError:
                    parent = previous
            self.lineage.carry(parent, closure)
        return closure

    # -- serving ---------------------------------------------------------------------------------------------------

    def fallback(self, closure: Closure) -> Closure:
        """The actual previous revision, else a retained ancestor that served before, else the base."""
        previous = self.previous(closure.id)
        if previous.id != self.revisions.base.id and self.revisions.status(previous.id).state == "superseded":
            return previous
        current = closure
        while current.parent and current.parent != self.revisions.base.id:
            try:
                current = self.revisions.get(current.parent)
            except RevisionError:
                break
            if self.revisions.status(current.id).state == "superseded":
                return current
        return self.revisions.base

    def recheck(self, report: GateReport) -> str:
        """Keep a passing revision or withdraw a failing revision before moving the pointer."""
        with self.lock:
            pending = self.revisions.pending_rollback()
            recovered = self.recover_rollback()
            if pending is not None and pending["closure"] == report.closure:
                return recovered
            serving = self.revisions.serving()
            if report.closure != serving.id:
                raise StateError(f"the report judges closure {report.closure}, not the serving one ({serving.id})",
                                 code="report_mismatch")
            if report.passed or report.awaiting:
                return "kept"
            return self.rollback(reason="failed a later gate: " + ", ".join(report.failed),
                                 expected_closure=serving.id)

    def rollback(self, *, reason: str = "manual rollback", expected_closure: Optional[str] = None) -> str:
        """Withdraw the current revision before falling back; a stale comparator result never moves a newer one."""
        with self.lock:
            if self.revisions.pending_rollback() is not None:
                return self.recover_rollback()
            serving = self.revisions.serving()
            if serving.id == self.revisions.base.id or (expected_closure and serving.id != expected_closure):
                return "kept"
            target = self.fallback(serving)
            self.revisions.begin_rollback(closure=serving.id, target=target.id,
                                          periods=self.revisions.introduced(serving.id), reason=reason)
            return self.recover_rollback()

    def recover_rollback(self) -> str:
        """Finish the recorded withdrawal and pointer move exactly once, including after a process restart."""
        with self.lock:
            pending = self.revisions.pending_rollback()
            if pending is None:
                return "kept"
            closure_id, reason = pending["closure"], pending["reason"]
            for period_id in pending["periods"]:
                self.store.periods.discard(period_id, reason=reason)
            self.revisions.set_status(closure_id, "withdrawn", reason=reason)
            try:
                target = self.revisions.get(pending["target"])
                if self.revisions.status(target.id).state == "withdrawn":
                    target = self.revisions.base
            except RevisionError:
                target = self.revisions.base
            # Report before moving the pointer, and deduplicate retries after that write.
            if not any(event["event"] == "rolled_back" and event["closure"] == closure_id
                       for event in self.events()):
                self._event("rolled_back", closure_id, previous=target.id, reason=reason)
            if target.id == self.revisions.base.id:
                self.revisions.unserve()
            else:
                self.revisions.promote(target.id, previous=self.previous(target.id).id)
                self.revisions.set_status(target.id, "serving")
            self.revisions.end_rollback()
            return "rolled_back"

    def _withdraw(self, closure_id: str, *, reason: str, also: Tuple[str, ...] = ()) -> None:
        """Withdraw ``closure_id`` and discard the periods it introduced (and ``also``). The periods are discarded
        before the status is written: a crash in between leaves a candidate that names a discarded period, which
        ``consider`` drops. Serving rollback records its intent and periods through ``recover_rollback``."""
        periods = tuple(dict.fromkeys(self.revisions.introduced(closure_id) + tuple(also)))
        for period_id in periods:
            self.store.periods.discard(period_id, reason=reason)
        self.revisions.set_status(closure_id, "withdrawn", reason=reason)

    def status(self, closure_id: str):
        return self.revisions.status(closure_id)


__all__ = ["STATES", "Controller", "StateError", "WithdrawnClosure"]
