"""Revisions: the unit that serves, as a closure.

A closure is one id over everything tuned to a learned state: the base identity (weights, quantization, tokenizer,
engine version), the learned state's digest, the objective's identity (its blend included), the calibration record
ids and the declared dependents (a cutoff, baselines, a cache generation: whatever the learner declares its
revisions change). It also records, outside its id, what it came from: its parent closure, the closures of other
learners it learned from (``upstream``), the manifests of its rows, its periods, its audience and its build mode.
Two builds of identical bytes are one closure, and the record written first is kept: the id covers everything that
serves, so the kept record is a derivation of exactly what serves, and its audience and manifests answer for it (a
guest reads bytes guest rows alone produced; an erasure of a row only the second build learned from leaves serving
bytes a derivation without it produced).

- **Promotion swaps one pointer atomically.** A closure's state and record are written whole first (each an atomic
  rename); promotion then replaces the ``SERVING`` pointer in one rename. A crash before or during the swap leaves the
  old revision serving; a reader checks every closure against its id and every state against its digest, so none
  ever serves half of one.
- **Each request reads one closure, for one requester.** ``pin`` loads a closure and its state together, the base
  where the serving closure's audience does not allow the requester; a later promotion does not change what a
  pinned request reads.
- **Undeclared keys are refused.** A closure carries exactly its learner's declared dependents: carrying an
  undeclared key would change it silently, and dropping a declared one would serve without it.
- **The learner's own checks run before anything is written** (``validate``: the learner store checks that the
  closure's manifests are kept and belong to its periods, and that it claims no wider audience than it learned
  from).
- **A new base is a new closure family.** A store opened on another base identity serves its own base, never a
  closure learned on the old one; the old family's states read as orphans, which the next sweep removes.
- **The newest ``KEEP`` closures are kept whole**, the serving one always among them; the oldest others are dropped
  with their state files. A dropped closure leaves a tombstone (``pruned/``): its record without its state, naming
  only ids (its parent, upstream closures, manifests and periods), so lineage stays walkable and an erasure still
  reaches what was built from it. Tombstones never serve.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from . import _io
from .audience import may_serve

#: Whole closures kept per learner, the serving one among them (D assumption).
KEEP = 14
MODES: Tuple[str, ...] = ("base", "rebuild", "stack")
POINTER = "SERVING"
ROLLBACK = "ROLLBACK"


class RevisionError(ValueError):
    """A closure the store refuses to write, read or serve, and why."""


class ForeignClosure(RevisionError):
    """A whole closure of another learner or another base identity (retired by a base swap)."""


@dataclass(frozen=True)
class Closure:
    learner: str
    base: str
    state: str
    objective: str
    calibration: Tuple[str, ...]
    dependents: Mapping[str, str]
    parent: str
    upstream: Tuple[Tuple[str, str], ...]
    manifests: Tuple[str, ...]
    periods: Tuple[str, ...]
    audience: str
    mode: str

    @property
    def id(self) -> str:
        return _io.digest({"learner": self.learner, "base": self.base, "state": self.state,
                           "objective": self.objective, "calibration": list(self.calibration),
                           "dependents": dict(self.dependents)})

    def to_dict(self) -> Dict[str, Any]:
        record = asdict(self)
        record["dependents"] = dict(self.dependents)
        record["upstream"] = [list(pair) for pair in self.upstream]
        for key in ("calibration", "manifests", "periods"):
            record[key] = list(record[key])
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Closure":
        return cls(learner=record["learner"], base=record["base"], state=record["state"],
                   objective=record["objective"], calibration=tuple(record["calibration"]),
                   dependents=dict(record["dependents"]), parent=record["parent"],
                   upstream=tuple((str(name), str(closure)) for name, closure in record["upstream"]),
                   manifests=tuple(record["manifests"]), periods=tuple(record["periods"]),
                   audience=record["audience"], mode=record["mode"])


@dataclass(frozen=True)
class Status:
    """Where a closure stands (``states.STATES``)."""

    state: str = "candidate"
    reason: str = ""


@dataclass(frozen=True)
class Pinned:
    """One whole closure and its learned state, as a request reads it."""

    closure: Closure
    state: Mapping[str, np.ndarray] = field(repr=False)


class RevisionStore:
    """One learner's closures, their states, their statuses and the serving pointer, in ``directory``."""

    def __init__(self, directory: Path, learner: str, *, base: str, declared: Iterable[str] = (), keep: int = KEEP,
                 validate: Optional[Callable[[Closure], None]] = None):
        if keep < 1:
            raise ValueError("a store keeps at least one closure")
        self.directory = Path(directory)
        self.learner = learner
        self.base_identity = base
        self.declared: FrozenSet[str] = frozenset(declared)
        self.keep = keep
        self._validate = validate
        self.base = Closure(learner=learner, base=base, state="", objective="", calibration=(), dependents={},
                            parent="", upstream=(), manifests=(), periods=(), audience="guest", mode="base")
        _io.sweep(self.directory)
        if not self.path_of(self.base.id).exists():
            self._save(self.base, seq=0)

    # -- paths -----------------------------------------------------------------------------------------------------

    def path_of(self, closure_id: str) -> Path:
        return self.directory / "closures" / f"{closure_id}.json"

    def tombstone_path(self, closure_id: str) -> Path:
        return self.directory / "pruned" / f"{closure_id}.json"

    def state_path(self, digest: str) -> Path:
        return self.directory / "states" / f"{digest}.npz"

    def _status_path(self, closure_id: str) -> Path:
        return self.directory / "statuses" / f"{closure_id}.json"

    def _withdrawn_path(self, closure_id: str) -> Path:
        return self.directory / "withdrawn" / f"{closure_id}.json"

    # -- closures --------------------------------------------------------------------------------------------------

    def compose(self, *, state: str, objective: str, calibration: Iterable[str] = (),
                dependents: Optional[Mapping[str, str]] = None, parent: str = "",
                upstream: Iterable[Tuple[str, str]] = (), manifests: Iterable[str] = (), periods: Iterable[str] = (),
                audience: str = "owner", mode: str = "rebuild") -> Closure:
        return Closure(learner=self.learner, base=self.base_identity, state=state, objective=objective,
                       calibration=tuple(calibration), dependents=dict(dependents or {}), parent=parent,
                       upstream=tuple((str(name), str(closure)) for name, closure in upstream),
                       manifests=tuple(manifests), periods=tuple(periods), audience=audience, mode=mode)

    def _save(self, closure: Closure, *, seq: int) -> None:
        _io.write_json(self.path_of(closure.id), {"id": closure.id, "seq": seq, "closure": closure.to_dict()})

    def _record(self, closure_id: str, *, path: Optional[Path] = None) -> Dict[str, Any]:
        try:
            record = json.loads((path or self.path_of(closure_id)).read_text(encoding="utf-8"))
            closure = Closure.from_dict(record["closure"])
        except FileNotFoundError:
            raise RevisionError(f"learner {self.learner}: no closure {closure_id}") from None
        except (ValueError, KeyError, TypeError) as error:
            raise RevisionError(f"learner {self.learner}: closure {closure_id} is not whole ({error})") from None
        if closure.id != closure_id or record.get("id") != closure_id:
            raise RevisionError(f"learner {self.learner}: closure {closure_id}'s content does not match its id")
        if closure.learner != self.learner or closure.base != self.base_identity:
            raise ForeignClosure(f"closure {closure_id} belongs to another learner or base")
        return {"closure": closure, "seq": int(record["seq"])}

    def get(self, closure_id: str) -> Closure:
        return self._record(closure_id)["closure"]

    def seq(self, closure_id: str) -> int:
        return self._record(closure_id)["seq"]

    def exists(self, closure_id: str) -> bool:
        return self.path_of(closure_id).exists()

    def write(self, closure: Closure, state: Mapping[str, np.ndarray]) -> Closure:
        """Write ``closure`` and its learned ``state`` whole, then prune to ``keep``. Nothing serves it yet. A closure
        already written keeps its record (the module docstring)."""
        if self.status(closure.id).state == "withdrawn":
            raise RevisionError(f"closure {closure.id} is withdrawn and never serves")
        if closure.learner != self.learner or closure.base != self.base_identity:
            raise RevisionError(f"closure {closure.id} belongs to another learner or base")
        if closure.mode not in MODES or closure.mode == "base":
            raise RevisionError(f"closure {closure.id}: a learned revision is built by rebuild or stack")
        if _io.state_digest(state) != closure.state:
            raise RevisionError(f"closure {closure.id}: its state does not match the digest it names")
        if self._validate is not None:
            self._validate(closure)
        carried = set(closure.dependents)
        undeclared, missing = sorted(carried - self.declared), sorted(self.declared - carried)
        if undeclared:
            raise RevisionError(f"learner {self.learner}: a revision may not change undeclared key(s) "
                                f"{', '.join(undeclared)}")
        if missing:
            raise RevisionError(f"learner {self.learner}: the revision does not carry declared dependent(s) "
                                f"{', '.join(missing)}")
        state_path = self.state_path(closure.state)
        if not state_path.exists():
            _io.write_atomic(state_path, _io.save_arrays(state))
        if not self.path_of(closure.id).exists():
            newest = max((seq for seq, _ in self._records()), default=0)
            self._save(closure, seq=newest + 1)
            _io.remove(self.tombstone_path(closure.id))
        self._prune(protect=(closure.id,))
        return closure

    def state(self, closure: Closure) -> Dict[str, np.ndarray]:
        """The learned state ``closure`` names, checked against its digest ({} for the base)."""
        if closure.mode == "base":
            return {}
        try:
            state = _io.load_arrays(self.state_path(closure.state).read_bytes())
        except FileNotFoundError:
            raise RevisionError(f"closure {closure.id}: its state {closure.state} is gone") from None
        except (ValueError, OSError) as error:
            raise RevisionError(f"closure {closure.id}: its state is not whole ({error})") from None
        if _io.state_digest(state) != closure.state:
            raise RevisionError(f"closure {closure.id}: its state file does not match its digest")
        return state

    def _listed(self, kind: str) -> List[Tuple[int, Closure]]:
        """The readable learned records under ``kind`` (``closures`` or ``pruned``) with their sequence numbers,
        oldest first."""
        directory = self.directory / kind
        if not directory.is_dir():
            return []
        found = []
        for path in directory.glob("*.json"):
            if path.stem == self.base.id:
                continue
            try:
                record = self._record(path.stem, path=path)
            except RevisionError:
                continue  # not whole: never retained, never served
            found.append((record["seq"], record["closure"]))
        return sorted(found, key=lambda item: item[0])

    def _records(self) -> List[Tuple[int, Closure]]:
        """Every learned closure the store knows, whole or pruned, with its sequence number, oldest first."""
        whole = self._listed("closures")
        ids = {closure.id for _, closure in whole}
        return sorted(whole + [(seq, closure) for seq, closure in self._listed("pruned") if closure.id not in ids],
                      key=lambda item: item[0])

    def retained(self) -> List[Closure]:
        """Every whole learned closure on disk, oldest first (the base is not listed)."""
        return [closure for _, closure in self._listed("closures")]

    def pruned(self) -> List[Closure]:
        """The tombstones of closures retention dropped, oldest first: records without states, never served."""
        return [closure for _, closure in self._listed("pruned")]

    def introduced(self, closure_id: str) -> Tuple[str, ...]:
        """The periods ``closure_id`` is the first to name: those no earlier closure the store knows (whole or
        pruned) that was never withdrawn also names. What a failed gate discards: a rebuilt closure names every
        period it trained on, and only the new ones are its own; a stacked closure names its own period alone."""
        records = self._records()
        seq = next((number for number, closure in records if closure.id == closure_id), None)
        if seq is None:
            raise RevisionError(f"learner {self.learner}: no closure {closure_id}")
        closure = next(closure for number, closure in records if closure.id == closure_id)
        vetted = {period for number, other in records if number < seq and other.id != closure_id
                  and self.status(other.id).state != "withdrawn" for period in other.periods}
        return tuple(period for period in closure.periods if period not in vetted)

    def known(self) -> List[Closure]:
        """Every learned closure the store knows of, whole or pruned (lineage walks them all), each once."""
        found = {closure.id: closure for closure in self.pruned()}
        found.update((closure.id, closure) for closure in self.retained())
        return list(found.values())

    def find(self, closure_id: str) -> Optional[Closure]:
        """The closure ``closure_id``: the base, a whole one or a tombstone; None when the store knows none."""
        if closure_id == self.base.id:
            return self.base
        for path in (self.path_of(closure_id), self.tombstone_path(closure_id)):
            try:
                return self._record(closure_id, path=path)["closure"]
            except RevisionError:
                continue
        return None

    def _drop(self, closure_id: str) -> None:
        """Remove a closure's record, its status, and its state file where no other closure names it."""
        if closure_id == self.base.id:
            raise RevisionError("the base closure is never deleted")
        if closure_id == self.serving().id:
            raise RevisionError(f"closure {closure_id} is serving: unserve it first")
        try:
            closure = self.get(closure_id)
        except RevisionError:
            closure = None
        _io.remove(self.path_of(closure_id))
        _io.remove(self._status_path(closure_id))
        if closure is not None and not any(other.state == closure.state for other in self.retained()):
            _io.remove(self.state_path(closure.state))

    def delete(self, closure_id: str) -> None:
        """Remove a closure whole: its record, status, tombstone, and its state file where no other closure names
        it (an erasure's delete)."""
        self._drop(closure_id)
        _io.remove(self.tombstone_path(closure_id))

    def _prune(self, *, protect: Iterable[str] = ()) -> None:
        serving = self.serving().id
        pending = self.pending_rollback()
        protected = set(protect)
        if pending is not None:
            protected.update((pending["closure"], pending["target"]))
        try:
            previous = _io.read_json(self.directory / POINTER).get("previous", "")
        except FileNotFoundError:
            previous = ""
        closures = self.retained()
        excess = len(closures) - self.keep
        for closure in closures:
            if excess <= 0:
                break
            if closure.id not in (serving, previous) and closure.id not in protected:
                # The tombstone first: a crash between the two leaves both, never neither.
                _io.write_json(self.tombstone_path(closure.id),
                               {"id": closure.id, "seq": self.seq(closure.id), "closure": closure.to_dict()})
                self._drop(closure.id)
                excess -= 1

    def orphans(self) -> List[Path]:
        """State files no whole closure names: what a crash between writing or deleting a closure and its state
        leaves."""
        directory = self.directory / "states"
        if not directory.is_dir():
            return []
        named = {closure.state for closure in self.retained()}
        return sorted(path for path in directory.glob("*.npz") if path.stem not in named)

    # -- serving ---------------------------------------------------------------------------------------------------

    def pending_rollback(self) -> Optional[Dict[str, Any]]:
        """A durable rollback intent, written before discarding any period or changing a status."""
        try:
            return _io.read_json(self.directory / ROLLBACK)
        except FileNotFoundError:
            return None

    def begin_rollback(self, *, closure: str, target: str, periods: Iterable[str], reason: str) -> None:
        if self.pending_rollback() is not None:
            raise RevisionError("finish the pending rollback before starting another")
        _io.write_json(self.directory / ROLLBACK, {"closure": closure, "target": target,
                                                  "periods": list(periods), "reason": reason})

    def end_rollback(self) -> None:
        _io.remove(self.directory / ROLLBACK)

    def serving(self) -> Closure:
        """The closure the pointer names: the base when nothing was promoted, or when the pointer names a closure
        learned on another base. Pending rollback reads the base until the target pointer has been written.
        RevisionError when the closure it names is not whole."""
        try:
            pointer = _io.read_json(self.directory / POINTER)
        except FileNotFoundError:
            return self.base
        try:
            closure_id = pointer["closure"]
            pending = self.pending_rollback()
            if pending is not None and closure_id != pending["target"]:
                return self.base
            if self.status(closure_id).state == "withdrawn":
                previous = pointer.get("previous", self.base.id)
                if self.status(previous).state == "withdrawn":
                    return self.base
                try:
                    return self.get(previous)
                except RevisionError:
                    return self.base
            return self.get(closure_id)
        except ForeignClosure:
            return self.base

    def promote(self, closure_id: str, *, previous: Optional[str] = None) -> Closure:
        """Serve ``closure_id``: checked whole, then one atomic swap of the pointer."""
        if self.status(closure_id).state == "withdrawn":
            raise RevisionError(f"closure {closure_id} is withdrawn and never serves")
        closure = self.get(closure_id)
        self.state(closure)
        pointer = {"closure": closure.id}
        if previous is not None:
            pointer["previous"] = previous
        _io.write_json(self.directory / POINTER, pointer)
        self._prune()
        return closure

    def unserve(self) -> Closure:
        """Serve the base."""
        _io.write_json(self.directory / POINTER, {"closure": self.base.id})
        return self.base

    def pin(self, requester: Optional[str]) -> Pinned:
        """What one request of ``requester`` reads: the serving closure where its audience allows the requester,
        the base otherwise (``audience.may_serve``), loaded together with its state. There is no read of the serving
        stack that does not name its requester; a learner's named exception is applied by
        ``audience.AudiencePolicy.pin``."""
        closure = self.serving()
        if closure.id != self.base.id and not may_serve(closure.audience, requester):
            closure = self.base
        return Pinned(closure=closure, state=self.state(closure))

    # -- statuses --------------------------------------------------------------------------------------------------

    def status(self, closure_id: str) -> Status:
        try:
            record = _io.read_json(self._status_path(closure_id))
            status = Status(state=record["state"], reason=record.get("reason", ""))
        except FileNotFoundError:
            status = Status()
        pending = self.pending_rollback()
        if pending is not None and pending["closure"] == closure_id:
            return Status("withdrawn", pending["reason"])
        if self._withdrawn_path(closure_id).exists():
            return Status("withdrawn", status.reason or "withdrawn identity")
        return status

    def set_status(self, closure_id: str, state: str, *, reason: str = "") -> Status:
        if state != "withdrawn" and self.status(closure_id).state == "withdrawn":
            raise RevisionError(f"closure {closure_id} is withdrawn and never serves")
        status = Status(state=state, reason=reason)
        if state == "withdrawn":
            _io.write_json(self._withdrawn_path(closure_id), {"closure": closure_id})
        _io.write_json(self._status_path(closure_id), asdict(status))
        return status


__all__ = ["KEEP", "MODES", "Closure", "ForeignClosure", "Pinned", "RevisionError", "RevisionStore", "Status"]
