"""Erasure for weights: given erased record ids, remove them from every retained training artifact so that no
served revision was trained with them. It does not undo actions already taken.

In order:

1. **Find** every manifest that names an erased ref, the periods that admitted one, and every closure trained on
   one: a closure naming such a manifest or period (whole, or as a tombstone retention left), for a stacked learner
   every closure of a period that will be discarded, and every descendant of any of them (lineage in the learner and
   upstream across learners).
2. **Unserve at once.** A learner whose serving closure is contaminated falls back to its newest retained closure
   that is clean, served before and was never withdrawn; with none, the base serves. A chain of learners qualified
   together falls back as one, to its newest joint record whose members are all clean, else every member to its
   base, never to a mix.
3. **Delete** the contaminated closures, their states, statuses and budget checks, and the joint records naming
   them. Each closure is deleted after every closure built from it, and the manifests and periods that name a ref
   change only after that, so a crash between two deletions leaves every contaminated closure still on disk
   reachable from them, and a retry finds it. A rebuilt learner's manifests are rewritten without the refs
   (re-admitted from what the source still re-renders, when the caller hands its admission, so a witness of an
   erased row becomes the row); a stacked learner's periods are discarded from the first one that trained on a ref,
   with their deltas and manifests.
4. **Rebuild** each rebuilt learner through the caller's rebuild (a clean retrain on the rewritten manifests,
   gated like any candidate). A stacked learner's data is not kept: its discarded periods are gone.
5. **Sweep and scan.** The temporaries and orphans a crash left (state files no whole closure names, deltas no
   closed period names) are removed, so a retried erasure finishes what an interrupted one began. Then every store
   is scanned for the erased refs, for state hashes no surviving row shares, and for any orphan still on disk:
   nothing may remain. What the erasure did is recorded by count and closure id, never by the erased refs.

A chain's members change one pointer at a time. Its move is recorded before the first changes and cleared after the
last (``JointLedger.begin`` and ``end``); a request reads the chain as one (``JointLedger.pin``: a recorded joint
closure, else every member's base), so a crash between two member swaps never serves a mix, and the next erasure
finishes the move, whatever it erases.

An erasure holds each state root's shared lock through rebuilding, serializing it with builds, promotions and
rollbacks. A rebuild that was requested is recorded in ``erasures/PENDING`` before unserving, by a digest of the
request and the erased state hashes, never the erased refs. Retrying the same refs resumes that rebuild even if
the first attempt had already rewritten every manifest; the marker goes only after rebuilding and a clean scan.
"""

from __future__ import annotations

import json
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import (Any, Callable, Collection, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set,
                    Tuple)

from . import _io
from .admission import Admission
from .audience import AudiencePolicy
from .lineage import Lineage
from .manifests import Manifest, RowSource
from .revisions import Closure, Pinned, RevisionError, RevisionStore
from .store import LearnerStore


@dataclass(frozen=True)
class Chain:
    """Learners qualified together as one joint closure (a query adapter, its score head and its cutoff)."""

    name: str
    members: Tuple[str, ...]


#: The file that names a chain's move while its members change, beside its joint records (not a record).
PENDING = "PENDING"


class JointLedger:
    """The joint closures of each chain: which member closures were promoted together, in order.

    A chain's members change one pointer at a time, so a chain is read as one (``joint``, ``pin``): its members'
    serving closures only when they form a recorded joint record none of whose members was withdrawn, or are all
    their bases, and no move is pending; its bases otherwise. A move records where it is going before any member
    changes (``begin``) and clears it after the last (``end``), so a crash in between is read as the bases and the
    next erasure finishes the move."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def _pending_path(self, chain: str) -> Path:
        return self.directory / chain / PENDING

    def begin(self, chain: str, members: Mapping[str, str], *,
              excluded: Mapping[str, Iterable[str]] = MappingProxyType({}), operation: str = "move",
              reason: str = "") -> None:
        """Record, before any member of ``chain`` moves, the closure each member moves to and the member closures it
        moves away from as contaminated (``excluded``: a later erasure never moves the chain back to one)."""
        _io.write_json(self._pending_path(chain), {"chain": chain, "members": dict(members),
                                                    "operation": operation, "reason": reason,
                                                    "excluded": {member: sorted(ids)
                                                                 for member, ids in excluded.items()}})

    def pending(self, chain: str) -> Optional[Dict[str, Any]]:
        """The move ``chain`` began and did not finish, or None."""
        try:
            return _io.read_json(self._pending_path(chain))
        except FileNotFoundError:
            return None

    def end(self, chain: str) -> None:
        _io.remove(self._pending_path(chain))

    def joint(self, chain: Chain, stores: Mapping[str, RevisionStore]) -> Optional[Dict[str, Closure]]:
        """The closures ``chain`` serves as one (by member): its members' serving closures when they are all their
        bases, or form a recorded joint record none of whose members was withdrawn, and no move is pending. None
        otherwise: a mix, or a member whose store is not given."""
        if any(member not in stores for member in chain.members) or self.pending(chain.name) is not None:
            return None
        serving = {member: stores[member].serving() for member in chain.members}
        if self.pending(chain.name) is not None:
            return None  # a move began while the pointers were read
        ids = {member: closure.id for member, closure in serving.items()}
        if all(closure_id == stores[member].base.id for member, closure_id in ids.items()):
            return serving
        for record in self.records(chain.name):
            if record["members"] == ids:
                withdrawn = any(stores[member].status(closure_id).state == "withdrawn"
                                for member, closure_id in ids.items() if closure_id != stores[member].base.id)
                return None if withdrawn else serving
        return None

    def pin(self, chain: Chain, stores: Mapping[str, RevisionStore], requester: Optional[str], *,
            policy: Optional[AudiencePolicy] = None) -> Dict[str, Pinned]:
        """What one request of ``requester`` reads of ``chain``, member by member: its joint closure (``joint``)
        where every member's audience allows the requester (``audience.AudiencePolicy``, with its named
        exceptions), every member's base otherwise. Never a mix."""
        policy = policy or AudiencePolicy()
        joint = self.joint(chain, stores)
        if joint is not None:
            chosen = {member: policy.select(stores[member].learner, requester, serving=closure,
                                            base=stores[member].base) for member, closure in joint.items()}
            if all(chosen[member].id == closure.id for member, closure in joint.items()):
                return {member: Pinned(closure=closure, state=stores[member].state(closure))
                        for member, closure in chosen.items()}
        return {member: Pinned(closure=stores[member].base, state={}) for member in chain.members}

    def record(self, chain: str, members: Mapping[str, str]) -> Dict[str, Any]:
        seq = max((record["seq"] for record in self.records(chain)), default=0) + 1
        record = {"chain": chain, "seq": seq, "members": dict(members)}
        _io.write_json(self.directory / chain / f"{seq:06d}.json", record)
        return record

    def records(self, chain: str) -> List[Dict[str, Any]]:
        directory = self.directory / chain
        if not directory.is_dir():
            return []
        return sorted((_io.read_json(path) for path in directory.glob("*.json")), key=lambda record: record["seq"])

    def delete(self, chain: str, seq: int) -> None:
        _io.remove(self.directory / chain / f"{seq:06d}.json")


@dataclass(frozen=True)
class ErasureReport:
    refs: int
    events: Tuple[Tuple[str, str], ...] = ()
    contaminated: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    fallbacks: Mapping[str, str] = field(default_factory=dict)
    rewritten: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    discarded: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    left: Tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        return not self.left


def scan(root: Path, needles: Iterable[str]) -> Tuple[str, ...]:
    """Every file under ``root`` that holds one of ``needles`` as a JSON string (every file the store writes is JSON
    or an array file, and a ref such as ``case-3`` must not match inside ``case-30``)."""
    encoded = [json.dumps(needle).encode("utf-8") for needle in needles if needle]
    root = Path(root)
    if not encoded or not root.exists():
        return ()
    found = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            if any(needle in data for needle in encoded):
                found.append(str(path))
    return tuple(found)


class Erasure:
    """Erases record ids from the learners in ``stores`` (the module docstring)."""

    def __init__(self, stores: Mapping[str, LearnerStore], *, chains: Sequence[Chain] = (),
                 joints: Optional[JointLedger] = None, clock: Optional[Callable[[], datetime]] = None,
                 calibrations: Any = None, lock: Any = None):
        self.stores: Dict[str, LearnerStore] = dict(stores)
        self.chains = tuple(chains)
        self.joints = joints
        self.calibrations = calibrations
        from .locking import StateLock
        self._roots = sorted({store.root.resolve() for store in self.stores.values()})
        self.lock = lock if lock is not None else StateLock(self._roots[0]) if self._roots else nullcontext()
        if self.chains and joints is None:
            raise ValueError("chains fall back through their joint records: pass the joint ledger")
        if clock is None:
            from ..util.temporal import now_utc
            clock = now_utc
        self._clock = clock

    def erase(self, refs: Iterable[str], *,
              readmit: Optional[Mapping[str, Tuple[Admission, RowSource]]] = None,
              rebuild: Optional[Mapping[str, Callable[[], Any]]] = None) -> ErasureReport:
        """Serialize erasure with promotion, rollback and rebuilding, including across processes."""
        from .locking import StateLock

        with ExitStack() as locks:
            locks.enter_context(self.lock)
            for root in self._roots:
                locks.enter_context(StateLock(root))
            return self._erase(refs, readmit=readmit, rebuild=rebuild)

    def _erase(self, refs: Iterable[str], *,
               readmit: Optional[Mapping[str, Tuple[Admission, RowSource]]] = None,
               rebuild: Optional[Mapping[str, Callable[[], Any]]] = None) -> ErasureReport:
        refs = frozenset(refs)
        readmit, rebuild = dict(readmit or {}), dict(rebuild or {})
        events: List[Tuple[str, str]] = []
        request = _io.digest(sorted(refs))
        pending_rebuilds: Dict[str, Dict[str, Any]] = {}
        for name, store in self.stores.items():
            path = store.directory / "erasures" / PENDING
            if path.exists():
                pending = _io.read_json(path)
                if pending["request"] == request:
                    if name not in rebuild:
                        raise ValueError(f"retry erasure for {name} with its rebuild callback")
                    pending_rebuilds[name] = pending
                elif refs:
                    raise ValueError(f"retry the pending erasure for {name} with the same refs first")

        # 1. Find.
        named: Dict[str, Dict[str, Manifest]] = {}
        trained: Dict[str, Set[str]] = {}
        erased_hashes: Set[str] = set()
        surviving_hashes: Set[str] = set()
        for name, store in self.stores.items():
            for manifest in store.manifests:
                hit = False
                for entry in manifest.entries:
                    if entry.ref in refs:
                        hit = True
                        erased_hashes.add(entry.state_hash)
                        if entry.verdict == "admitted":
                            trained.setdefault(name, set()).add(manifest.id)
                    else:
                        surviving_hashes.add(entry.state_hash)
                if hit:
                    named.setdefault(name, {})[manifest.id] = manifest
        periods: Dict[str, Set[str]] = {}
        gone: Dict[str, Tuple[str, ...]] = {}
        for name, manifests in trained.items():
            store = self.stores[name]
            periods[name] = {period.id for period in store.periods.periods() if set(period.manifests) & manifests}
            if store.mode == "stack":
                gone[name] = self._to_discard(store, manifests)
        lineage = Lineage({name: store.revisions for name, store in self.stores.items()})
        roots = [(name, closure.id) for name, store in self.stores.items() for closure in self._known(store)
                 if set(closure.manifests) & trained.get(name, set())
                 or set(closure.periods) & (periods.get(name, set()) | set(gone.get(name, ())))]
        reached = lineage.descendants(roots)
        dirty: Dict[str, Dict[str, Closure]] = {}
        for name, closure in reached:
            dirty.setdefault(name, {})[closure.id] = closure
        for name in sorted(set(trained) | set(dirty)):
            store = self.stores[name]
            if store.mode == "rebuild" and name in rebuild:
                pending = pending_rebuilds.setdefault(name, {"request": request, "state_hashes": []})
                pending["state_hashes"] = sorted(set(pending["state_hashes"]) | (erased_hashes - surviving_hashes))
                _io.write_json(store.directory / "erasures" / PENDING, pending)

        # 2. Unserve at once. A chain moves as one: where it goes is recorded before any member moves and cleared
        #    after the last, and a chain found mid-move or serving a mix (what a crash between two member swaps
        #    leaves) moves again, never back to a closure an earlier move left as contaminated.
        fallbacks: Dict[str, str] = {}
        affected = {name for name, store in self.stores.items() if store.revisions.serving().id in dirty.get(name, {})}
        for chain in self.chains:
            members = [member for member in chain.members if member in self.stores]
            pending = self.joints.pending(chain.name)
            if pending is not None and pending.get("operation") == "rollback":
                self._finish_withdrawals(pending)
            mixed = len(members) == len(chain.members) and self.joints.joint(
                chain, {member: self.stores[member].revisions for member in members}) is None
            if not affected & set(members) and pending is None and not mixed:
                continue
            excluded: Dict[str, Set[str]] = {member: set(ids) for member, ids in
                                             (pending or {}).get("excluded", {}).items()}
            for member in members:
                excluded.setdefault(member, set()).update(dirty.get(member, {}))
            target = self._joint_fallback(chain, excluded) or {}
            destination = {member: target.get(member) or self.stores[member].revisions.base.id for member in members}
            self.joints.begin(chain.name, destination, excluded=excluded)
            for member in members:
                fallbacks[member] = self._serve(member, destination[member])
                events.append(("unserve", member))
            self.joints.end(chain.name)
            affected -= set(members)
        for name in sorted(affected):
            fallbacks[name] = self._serve(name, self._fallback(name, dirty.get(name, {})))
            events.append(("unserve", name))

        # 3. Delete, each closure after every closure built from it (``Lineage.leaves_first``): until the manifests
        #    are rewritten below, whatever a crash leaves on disk is still reached from them on a retry.
        contaminated = {name: tuple(sorted(closures)) for name, closures in sorted(dirty.items())}
        if self.calibrations is not None:
            for name, closure in reached:
                self.calibrations.withdraw_revisions({closure.id, closure.state}, records=closure.calibration)
                if ("withdraw_calibration", name) not in events:
                    events.append(("withdraw_calibration", name))
        for name, closure in lineage.leaves_first(reached):
            store = self.stores[name]
            store.revisions.set_status(closure.id, "withdrawn", reason="erasure")
            store.revisions.delete(closure.id)
            if not any(other.state == closure.state for other in store.revisions.retained()):
                store.budgets.forget(name, closure.state)
            if ("delete", name) not in events:
                events.append(("delete", name))
        if self.joints is not None:
            for chain in self.chains:
                for record in self.joints.records(chain.name):
                    if any(closure_id in dirty.get(member, {}) for member, closure_id in record["members"].items()):
                        self.joints.delete(chain.name, record["seq"])
        rewritten: Dict[str, Tuple[str, ...]] = {}
        discarded: Dict[str, Tuple[str, ...]] = {}
        for name, manifests in sorted(named.items()):
            store = self.stores[name]
            if gone.get(name):
                discarded[name] = self._discard(store, gone[name])
            rewritten[name] = self._rewrite(store, manifests, refs, readmit.get(name))
            if name not in dirty:
                events.append(("delete", name))
        for store in self.stores.values():
            store.sweep()

        # 4. Rebuild.
        for name in sorted(pending_rebuilds):
            events.append(("rebuild", name))
            rebuild[name]()

        # 5. Record, then scan.
        needles = set(refs) | (erased_hashes - surviving_hashes)
        for pending in pending_rebuilds.values():
            needles.update(pending["state_hashes"])
        for name in sorted(set(named) | set(dirty) | set(pending_rebuilds)):
            store = self.stores[name]
            directory = store.directory / "erasures"
            seq = len(list(directory.glob("*.json"))) + 1 if directory.is_dir() else 1
            _io.write_json(directory / f"{seq:06d}.json", {
                "at": _io.stamp(self._clock()), "refs": len(refs), "contaminated": list(contaminated.get(name, ())),
                "fallback": fallbacks.get(name, ""), "discarded": list(discarded.get(name, ())),
                "rewritten": len(rewritten.get(name, ()))})
        roots_to_scan = {store.directory for store in self.stores.values()}
        if self.joints is not None:
            roots_to_scan.add(self.joints.directory)
        pending_paths = {str(self.stores[name].directory / "erasures" / PENDING) for name in pending_rebuilds}
        found = {path for root in roots_to_scan for path in scan(root, needles)} - pending_paths
        found |= {str(path) for store in self.stores.values() for path in store.orphans()}
        left = tuple(sorted(found))
        if not left:
            for path in pending_paths:
                _io.remove(Path(path))
        return ErasureReport(refs=len(refs), events=tuple(events), contaminated=contaminated, fallbacks=fallbacks,
                             rewritten=rewritten, discarded=discarded, left=left)

    # -- helpers ---------------------------------------------------------------------------------------------------

    def _finish_withdrawals(self, pending: Mapping[str, Any]) -> None:
        """Finish every member's withdrawal before completing a joint rollback interrupted by a crash."""
        from .states import Controller

        reason = pending.get("reason") or "joint rollback"
        for name, ids in pending.get("excluded", {}).items():
            store = self.stores.get(name)
            if store is None:
                continue
            controller = Controller(store, clock=self._clock, lock=self.lock)
            for closure_id in ids:
                if closure_id == store.revisions.base.id or not store.revisions.exists(closure_id):
                    continue
                controller._withdraw(closure_id, reason=reason)
                if not any(event["event"] == "rolled_back" and event["closure"] == closure_id
                           for event in controller.events()):
                    controller._event("rolled_back", closure_id, previous=pending["members"][name], reason=reason)

    @staticmethod
    def _known(store: LearnerStore) -> List[Closure]:
        """Every learned closure of ``store``: whole, pruned, and the serving one."""
        found = {closure.id: closure for closure in store.revisions.known()}
        serving = store.revisions.serving()
        if serving.id != store.revisions.base.id:
            found.setdefault(serving.id, serving)
        return list(found.values())

    def _fallback(self, name: str, dirty: Mapping[str, Closure]) -> Optional[str]:
        """The newest retained clean closure of ``name`` that served before and was never withdrawn."""
        store = self.stores[name]
        for closure in reversed(store.revisions.retained()):
            if closure.id not in dirty and store.revisions.status(closure.id).state == "superseded":
                return closure.id
        return None

    def _joint_fallback(self, chain: Chain, dirty: Mapping[str, Collection[str]]) -> Optional[Dict[str, str]]:
        for record in reversed(self.joints.records(chain.name) if self.joints else []):
            members = record["members"]
            if set(members) != set(chain.members):
                continue
            usable = True
            for member, closure_id in members.items():
                store = self.stores.get(member)
                if store is None or closure_id in dirty.get(member, {}):
                    usable = False
                    break
                try:
                    store.revisions.get(closure_id)
                except RevisionError:
                    usable = False
                    break
                if store.revisions.status(closure_id).state == "withdrawn":
                    usable = False
                    break
            if usable:
                return dict(members)
        return None

    def _serve(self, name: str, closure_id: Optional[str]) -> str:
        revisions = self.stores[name].revisions
        previous = revisions.serving()
        if closure_id is None or closure_id == revisions.base.id:
            target = revisions.unserve()
        else:
            from .states import Controller
            previous_arm = Controller(self.stores[name], clock=self._clock, lock=self.lock).previous(closure_id)
            target = revisions.promote(closure_id, previous=previous_arm.id)
            revisions.set_status(target.id, "serving")
        if previous.id not in (revisions.base.id, target.id) and revisions.status(previous.id).state == "serving":
            revisions.set_status(previous.id, "superseded")
        return target.id

    @staticmethod
    def _to_discard(store: LearnerStore, trained: Set[str]) -> Tuple[str, ...]:
        """A stacked learner's periods from the first that trained on a ref onward (an open period has trained
        nothing yet): every later state was built on top of it."""
        periods = store.periods.periods()
        first = next((period.seq for period in periods if set(period.manifests) & trained), None)
        if first is None:
            return ()
        return tuple(period.id for period in periods if period.seq >= first and period.state != "open")

    @staticmethod
    def _discard(store: LearnerStore, periods: Sequence[str]) -> Tuple[str, ...]:
        """Discard ``periods`` and delete their deltas."""
        for period_id in periods:
            period = store.periods.discard(period_id, reason="erasure")
            if period.delta:
                _io.remove(store.delta_path(period.delta))
        return tuple(periods)

    @staticmethod
    def _rewrite(store: LearnerStore, manifests: Mapping[str, Manifest], refs: FrozenSet[str],
                 readmit: Optional[Tuple[Admission, RowSource]]) -> Tuple[str, ...]:
        """Each manifest naming a ref, rewritten without it (re-admitted when the admission is at hand), in place of
        the old one in every period that used it."""
        written = []
        for old_id, manifest in manifests.items():
            if readmit is not None:
                admission, source = readmit
                new = admission.readmit(manifest, source.rows(admission.objective))
            else:
                new = replace(manifest, entries=tuple(entry for entry in manifest.entries if entry.ref not in refs))
            if any(entry.ref in refs for entry in new.entries):
                new = replace(new, entries=tuple(entry for entry in new.entries if entry.ref not in refs))
            new_id = store.manifests.put(new)
            for period in store.periods.periods():
                if old_id in period.manifests:
                    store.periods.replace_manifest(period.id, old_id, new_id)
            if new_id != old_id:
                store.manifests.delete(old_id)
            written.append(new_id)
        return tuple(written)


__all__ = ["PENDING", "Chain", "ErasureReport", "Erasure", "JointLedger", "scan"]
