"""Configured learners, serving, and the automatic nightly plasticity lifecycle.

Implementations are code-registered factories, never import paths read from YAML.
The synthetic implementation is the existing CPU drill, not a product learner.
Each driver supplies observe (period), build, propose and gate. The runtime owns
promotion, the joint pointer, compaction, comparator and erasure serialization.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import math
import os
import re
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from . import _io, builds, learners
from .comparator import LiveComparator
from .compaction import compact, discarded_rows
from .erasure import Chain, Erasure, JointLedger
from .gates import Check, GateReport
from .lineage import Lineage
from .locking import StateLock, StateLockBusy, on_busy
from .rows import InstinctRows
from .states import Controller, StateError, WithdrawnClosure
from .store import LearnerStore
from ..util.temporal import now_utc

logger = logging.getLogger(__name__)
POLL_SECONDS = 30  # Builder assumption: check the existing nightly boundary twice a minute.
LIVE_RESTART_SECONDS = 60  # Plan assumption: delay before restarting a failed live loop.
FACTORIES = {}
COMPARATORS: dict[str, type] = {}
from .comparator import BaselineComparator
COMPARATORS["behaviour"] = BaselineComparator


def _erased_hash(root, ref):
    key = (root / "comparator.key").read_bytes()
    return hmac.new(key, _io.canonical(["erased", ref]), hashlib.sha256).hexdigest()


def _erased_refs(root):
    try:
        return set(_io.read_json(root / "erased-refs.json"))
    except FileNotFoundError:
        return set()


def register_factory(name, factory):
    """Register an implementation in code, independently of its learner declarations."""
    if name in FACTORIES:
        raise ValueError(f"implementation already registered: {name}")
    FACTORIES[name] = factory


def _synthetic(*, store, controller, source, declaration, clock):
    from .synthetic import Cycle, correct

    class SyntheticDriver(Cycle):
        def _start(self):
            if declaration.get("accumulation", True):
                return super()._start()
            return self.store.revisions.base, self.learner.base()

        def evidence(self, closure, build, **overrides):
            evidence = super().evidence(closure, build, **overrides)
            if not declaration.get("accumulation", True):
                previous = self.store.revisions.serving()
                state = self.store.revisions.state(previous) or self.learner.base()
                evidence = replace(evidence, reference=correct(self.learner, state, self.suite.items),
                                   parent_calibration=previous.calibration)
            return evidence

    return SyntheticDriver(store, controller, source=source, clock=clock)


class _Rows:
    def __init__(self, source, store, declaration):
        self.source, self.store, self.declaration = source, store, declaration
        self.blocked = set()

    def rows(self, objective, *, since=None):
        omitted = set(self.declaration.get("disabled_teachers", ()))
        discarded = discarded_rows(self.store)
        erased = _erased_refs(self.store.root)
        rows = [dict(row) for row in self.source.rows(objective, since=since)
                if row["ref"] not in self.blocked and row["teacher"] not in omitted
                and (not erased or _erased_hash(self.store.root, row["ref"]) not in erased)
                and _io.digest([row["ref"], row["teacher"]]) not in discarded]
        allowed = self.record_refs(row["ref"] for row in rows)
        yield from (row for row in rows if row["ref"] in allowed)

    def record_refs(self, refs):
        sources = getattr(self.source, "source_ids", None)
        dependencies = {ref: sources(ref) if sources is not None else None for ref in refs}
        try:
            erased = set(_io.read_json(self.store.root / "erased-sources.json"))
        except FileNotFoundError:
            erased = set()
        allowed = {}
        for ref, ids in dependencies.items():
            if ids is not None:
                if isinstance(ids, (str, bytes)):
                    raise ValueError("source_ids must be an iterable of source identifiers")
                ids = tuple(ids)
            if ids is None or not erased.intersection(self.store.source_hash(source) for source in ids):
                allowed[ref] = ids
        self.store.record_source_refs(allowed)
        return allowed.keys()

    def source_of(self, row):
        return self.source.source_of(row)


class _ShuffledLearner:
    """Shuffle admitted targets at fit time, leaving re-rendered manifests unchanged."""

    def __init__(self, learner):
        self.learner = learner

    def __getattr__(self, name):
        return getattr(self.learner, name)

    def fit(self, start, rows, *, objective, seed):
        import numpy as np
        shuffled = list(rows)
        # Builder assumption: a fixed control seed for reproducible paired arms.
        rng = np.random.default_rng(0)
        for teacher in sorted({row.teacher for row in rows}):
            indices = [index for index, row in enumerate(rows) if row.teacher == teacher]
            order = rng.permutation(indices)
            for index, other in zip(indices, order):
                shuffled[index] = replace(rows[index], label=rows[other].label)
        return self.learner.fit(start, shuffled, objective=objective, seed=seed)


@dataclass(frozen=True)
class _WithdrawnBefore:
    """A night's derivation that names a closure already judged and withdrawn (closure ids cover only what
    serves, so the same retained rows and settings give the same id). Nothing is proposed, gated or moved."""

    id: str
    why: str = "withdrawn_before"
    counts: Mapping[str, Mapping[str, int]] | None = None
    capture_present: bool | None = None

    def details(self):
        return {"result": "unchanged", "why": [self.why],
                **({"counts": self.counts} if self.counts is not None else {}),
                **({"capture_present": self.capture_present} if self.capture_present is not None else {})}


#: The short fixed ``why`` of a night whose derivation was judged and withdrawn before.
WITHDRAWN_BEFORE = "withdrawn_before"
#: The short fixed ``why`` of a chain member left unjudged because its chain cannot move tonight.
CHAIN_WAITING = "chain_waiting"
#: No candidate is possible until the learner's declared label minimums are available.
INSUFFICIENT_LABELS = "insufficient_labels"
NO_QUALIFICATION_CAPTURE = "no_gate_suite"


def _error_why(error):
    """An error's ``why``: its type and, for a refused state change, the refusal's fixed code. Never its message,
    which may name inputs."""
    why = [type(error).__name__]
    if isinstance(error, StateError):
        why.append(error.code)
    return why


@dataclass
class Binding:
    name: str
    declaration: Mapping[str, Any]
    registered: Any
    store: LearnerStore
    controller: Controller
    driver: Any
    source: _Rows
    comparator: Any = None
    last_build: Any = None
    pending_closure: Any = None
    pending_period: Any = None
    pending_idle: _WithdrawnBefore | None = None


class Runtime:
    def __init__(self, config, *, data_dir=None, check_learner=None, factories=None, sources=None,
                 instinct_store=None, clock=None):
        self.config = config
        self.data_dir = Path(data_dir or config.home).resolve()
        self.root = (self.data_dir / config.get("plasticity.directory", "plasticity")).resolve()
        if self.root == self.data_dir or not self.root.is_relative_to(self.data_dir):
            raise ValueError("plasticity.directory must stay under the data directory")
        self.enabled = config.get("mind.faculties.plasticity", True) is True
        self._clock = clock or now_utc
        self.lock = StateLock(self.root)
        self.behaviour_pins = ()
        self.loaded = {}
        self.rejected = {}
        self.declarations = config.get("plasticity.learners", {})
        self.joints = JointLedger(self.root / "chains")
        self.chains = ()
        self._unbind = None
        self._unbind_sources = None
        self.instinct_store = instinct_store
        self._last = {}
        if not self.enabled or not self.declarations:
            self._bind_erasure()
            return
        learners.load()
        if self.instinct_store is None:
            from ..instinct.log import Store, DB_FILE
            self.instinct_store = Store(self.data_dir / DB_FILE)
        if check_learner is None:
            from .registration import check_learner
        factories = {"synthetic": _synthetic, **FACTORIES, **(factories or {})}
        sources = sources or {}
        with self.lock:
            key_path = self.root / "comparator.key"
            if not key_path.exists():
                _io.write_atomic(key_path, os.urandom(32))
                key_path.chmod(0o600)
            for name, declaration in self.declarations.items():
                try:
                    if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name):
                        raise ValueError("name: expected a learner identifier")
                    registered = check_learner(name, declaration)
                    if not isinstance(declaration, Mapping):
                        raise ValueError("declaration: expected a mapping")
                    chain = declaration.get("chain")
                    if chain is not None and (not isinstance(chain, str)
                            or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", chain)):
                        raise ValueError("chain: expected an identifier")
                    from .gates import DECLINE_BAR
                    bar = declaration.get("decline_bar", DECLINE_BAR)
                    if isinstance(bar, bool) or not isinstance(bar, (int, float)) or not math.isfinite(bar) or bar < 0:
                        raise ValueError("decline_bar: expected a finite nonnegative number")
                    if declaration.get("enabled", True) is False:
                        continue
                    implementation = declaration.get("implementation", name)
                    # Synthetic fixtures and independent behaviour targets may share an implementation.
                    # Other implementations load once (``find``).
                    existing = None if implementation in {"synthetic", "behaviour"} else self.find(implementation)
                    if existing is not None:
                        raise ValueError(f"implementation: {implementation} already loaded by {existing}")
                    factory = factories.get(implementation)
                    if factory is None:
                        raise ValueError("implementation: no registered factory")
                    base = declaration.get("base", "synthetic-base@1" if implementation == "synthetic" else "")
                    if not base:
                        raise ValueError("base: identity is required")
                    store = LearnerStore(self.root, name, base=base, mode=declaration.get("mode", "rebuild"),
                                         declared=declaration.get("dependents", ()), clock=self._clock)
                    if store.erasure_pending():
                        store.revisions.unserve()
                        if (store.rebuild_required.exists()
                                and _io.read_json(store.rebuild_required).get("observe")):
                            store.clear_derived(calibrations=self.instinct_store)
                    controller = Controller(store, clock=self._clock)
                    controller.behaviour_restarts = config.get("plasticity.behaviour.restarts", [])
                    source = sources.get(name)
                    if source is None:
                        if implementation == "synthetic":
                            from .synthetic import SyntheticSource
                            source = SyntheticSource()
                        else:
                            from ..instinct.outcomes import Sources
                            source = InstinctRows(self.instinct_store, Sources(self.data_dir))
                    source = _Rows(source, store, declaration)
                    driver = factory(store=store, controller=controller, source=source,
                                     declaration=declaration, clock=self._clock)
                    if declaration.get("control") == "shuffled":
                        driver.learner = _ShuffledLearner(driver.learner)
                    self.loaded[name] = Binding(name, declaration, registered, store, controller, driver, source)
                except (ValueError, TypeError, ImportError) as error:
                    self.rejected[str(name)] = str(error)
            self._wire()
            try:
                self._resume_source_erasures()
            except Exception as error:
                logger.warning("plasticity source erasure retry failed (%s)", type(error).__name__)
            self._resume_rebuilds()
            self._publish()

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(config, **kwargs)

    def find(self, implementation: str) -> str | None:
        """Return the loaded learner name for a code-registered implementation."""
        return next((name for name, binding in self.loaded.items()
                     if binding.declaration.get("implementation", name) == implementation), None)

    def _wire(self):
        groups = {}
        for name, binding in self.loaded.items():
            chain = binding.declaration.get("chain")
            if chain:
                groups.setdefault(chain, []).append(name)
        self.chains = tuple(Chain(name, tuple(members)) for name, members in sorted(groups.items()))
        # A partially registered chain cannot qualify or serve alone.
        for chain in self.chains:
            expected = {name for name, declaration in self.declarations.items()
                        if isinstance(declaration, Mapping) and declaration.get("chain") == chain.name}
            if expected != set(chain.members):
                for name in chain.members:
                    self.rejected[name] = "chain: one or more members did not load"
                    self.loaded.pop(name)
        self.chains = tuple(chain for chain in self.chains if all(name in self.loaded for name in chain.members))
        stores = {name: binding.store.revisions for name, binding in self.loaded.items()}
        lineage = Lineage(stores)
        if self.instinct_store is not None:
            lineage.bind_instinct(self.instinct_store, clock=self._clock)
        if self.loaded:
            key_path = self.root / "comparator.key"
            if not key_path.exists():
                _io.write_atomic(key_path, os.urandom(32))
                key_path.chmod(0o600)
            key = key_path.read_bytes()
            from .gates import DECLINE_BAR
            for name, binding in self.loaded.items():
                binding.controller.lineage = lineage
                comparator = COMPARATORS.get(getattr(binding.registered, "kind", None), LiveComparator)
                binding.comparator = comparator(binding.controller, key=key, lock=self.lock,
                    clock=self._clock, bar=binding.declaration.get("decline_bar", DECLINE_BAR),
                    pin=lambda requester, name=name: self.pin(name, requester),
                    rollback=lambda reason, expected_closure=None, name=name: self._rollback(
                        name, reason=reason, expected_closure=expected_closure))
                if hasattr(binding.driver, "comparator"):
                    binding.driver.comparator = binding.comparator
                if hasattr(binding.driver, 'current_records'):
                    binding.comparator.current_records = lambda rows, b=binding: b.driver.current_records(rows, b.comparator)
                if getattr(binding.registered, "kind", None) == "behaviour":
                    binding.driver.runtime = self
        catalog = self._catalog()
        for name, binding in self.loaded.items():
            catalog[name] = {"base": binding.store.revisions.base_identity, "mode": binding.store.mode,
                             "declared": sorted(binding.store.revisions.declared),
                             "kind": getattr(binding.registered, "kind", None),
                             "chain": binding.declaration.get("chain")}
        if self.loaded:
            _io.write_json(self.root / "catalog.json", catalog)
        self._bind_erasure()

    def _bind_erasure(self):
        if not self._catalog():
            return
        if self.instinct_store is None:
            from ..instinct.log import Store, DB_FILE
            if (self.data_dir / DB_FILE).exists():
                self.instinct_store = Store(self.data_dir / DB_FILE)
        if self.instinct_store is not None:
            from ..instinct.log import bind_erasure
            self._unbind = bind_erasure(self.instinct_store.path, self.erase)
        from ..turns.erasure_hooks import bind_source_erasure
        self._unbind_sources = bind_source_erasure(self.erase_sources)

    def _catalog(self):
        try:
            return _io.read_json(self.root / "catalog.json")
        except FileNotFoundError:
            return {}

    def close(self):
        if self._unbind_sources is not None:
            self._unbind_sources()
            self._unbind_sources = None
        if self._unbind is not None:
            self._unbind()
            self._unbind = None

    @on_busy()
    def pin_chain(self, name, requester):
        """Pin the whole chain, or return None when a loop caller cannot acquire it."""
        chain = next(chain for chain in self.chains if chain.name == name)
        with self.lock:
            if any(self.loaded[member].store.erasure_pending() for member in chain.members):
                from .revisions import Pinned
                return {member: Pinned(self.loaded[member].store.revisions.base, {}) for member in chain.members}
            return self.joints.pin(chain, {member: self.loaded[member].store.revisions
                                           for member in chain.members}, requester)

    @on_busy()
    def pin(self, name, requester):
        binding = self.loaded.get(name)
        if not self.enabled or binding is None:
            return None
        with self.lock:
            if binding.store.erasure_pending():
                from .revisions import Pinned
                return Pinned(binding.store.revisions.base, {})
            for chain in self.chains:
                if name in chain.members:
                    return self.pin_chain(chain.name, requester)[name]
            return binding.store.revisions.pin(requester)

    def serve(self, name, ref, inputs, *, predict, requester="owner"):
        if not self.enabled or name not in self.loaded:
            return predict(None, inputs)
        try:
            with self.lock:
                binding = self.loaded[name]
                if binding.store.erasure_pending():
                    return predict(None, inputs)
                # Only a source that declares its dependencies (``source_ids``) is recorded at serve time: an unknown
                # provenance adds nothing an erasure could use, and a per-serve write would grow without bound.
                if (getattr(binding.source.source, "source_ids", None) is not None
                        and ref not in binding.source.record_refs((ref,))):
                    return predict(None, inputs)
                return binding.comparator.serve(ref, inputs, predict, requester=requester)
        except StateLockBusy:
            return predict(None, inputs)

    @on_busy()
    def outcome(self, name, ref, **kwargs):
        with self.lock:
            result = self.loaded[name].comparator.outcome(ref, **kwargs)
            self._publish()
            return result

    def _build(self, binding):
        driver, store = binding.driver, binding.store
        if binding.declaration.get("accumulation", True):
            return driver.build()
        manifests = store.manifests_of(store.periods.retained()[-1:])
        if store.mode == "rebuild":
            return builds.rebuild(driver.learner, driver.objective, manifests, driver.source)
        return builds.stack(driver.learner, driver.objective, driver.learner.base(), manifests, driver.source)

    @staticmethod
    def _minimum_labels(binding):
        # These two established learners are explicitly outside the sparse-idle change.
        # Their proposals, certificates and reports must remain byte-identical.
        if binding.declaration.get("implementation", binding.name) in {"service_health", "reply_timing"}:
            return {}
        return getattr(binding.driver, "minimum_labels", {})

    def _prepare(self, name, *, observe=True):
        binding = self.loaded[name]
        driver = binding.driver
        binding.pending_closure = None
        binding.pending_period = None
        binding.pending_idle = None
        if getattr(driver, "idle_reason", None) in {INSUFFICIENT_LABELS, NO_QUALIFICATION_CAPTURE}:
            driver.idle_reason = None
        if observe:
            current = binding.store.periods.current()
            if current is not None:
                binding.store.periods.discard(current.id, reason="interrupted observation")
            driver.period()
            binding.pending_period = driver.last_period
        build = self._build(binding)
        if getattr(driver, "idle_reason", None):
            binding.last_build = build
            return _WithdrawnBefore(binding.store.revisions.serving().id, driver.idle_reason), None
        minimum = self._minimum_labels(binding)
        if minimum:
            # A qualification capture is operator supplied for this exact build;
            # ordinary training observations cannot accumulate its gate scores.
            capture_present = (driver.qualification_capture_present(build)
                               if hasattr(driver, "qualification_capture_present") else None)
            have = driver.label_counts(build)
            counts = {key: {"have": have[key], "need": need} for key, need in minimum.items()}
            why = INSUFFICIENT_LABELS
            if capture_present is False:
                why = NO_QUALIFICATION_CAPTURE
                # Query-adapter declares no training floor: zero means none required.
                counts = {"training": {"have": build.rows, "need": 0}}
            if capture_present is False or any(row["have"] < row["need"] for row in counts.values()):
                binding.last_build = build
                driver.idle_reason = why
                binding.pending_idle = _WithdrawnBefore(binding.store.revisions.serving().id,
                                                       why, counts, capture_present)
                # Count-only report metadata belongs beside nightly/report.json,
                # outside the learner artifact scan (a ref may itself be "training").
                _io.write_json(self.root / "idle-reports" / f"{name}.json", binding.pending_idle.details())
                return binding.pending_idle, None
            # Persist invalidation before proposing, including erasure rebuilds that
            # do not write nightly.json. A later process must not revive old idling.
            _io.write_json(self.root / "idle-reports" / f"{name}.json", {})
        try:
            closure = driver.propose(build)
        except WithdrawnClosure as withdrawn:
            # The controller refuses before writing anything. The same derivation was gated and withdrawn
            # before, so there is nothing new to judge: ``_finish`` leaves serving alone and keeps the period
            # only where a later build can still learn from it. ``propose`` still refuses any other caller.
            binding.last_build = build
            return _WithdrawnBefore(withdrawn.closure_id), None
        binding.pending_closure = closure
        report = driver.gate(closure, build)
        binding.last_build = build
        return closure, report

    def _finish(self, names, prepared, chain=None):
        # A member without a candidate stops a joint move. Sparse idling does not excuse a sibling's own failures.
        withdrawn_before = {name for name in names if prepared[name][1] is None}
        earlier_derivations = {name for name in withdrawn_before if prepared[name][0].counts is None}
        passed = not withdrawn_before and all(report.passed for _, report in prepared.values())
        current = {name: self.loaded[name].store.revisions.serving().id for name in names}
        failed_serving = [name for name in names if name not in withdrawn_before and not prepared[name][1].passed
                          and not prepared[name][1].awaiting
                          and prepared[name][0].id == current[name]]
        rolled = set()
        if failed_serving:
            name = failed_serving[0]
            self._rollback(name, reason="failed nightly gates: " + ", ".join(prepared[name][1].failed),
                           expected_closure=current[name])
            rolled = {name for name in names if prepared[name][0].id == current[name]
                      and self.loaded[name].store.revisions.serving().id != current[name]}
        if chain is not None and passed:
            self.joints.begin(chain.name, {name: prepared[name][0].id for name in names})
        results = []
        for name in names:
            binding = self.loaded[name]
            closure, report = prepared[name]
            if name in withdrawn_before:
                if closure.counts is None:
                    self._settle_period(binding)
                else:
                    self._settle_idle_period(binding)
                results.append({"learner": name, "closure": closure.id, **closure.details()})
                continue
            if not passed and report.passed:
                report = GateReport(closure.id, report.suite, (*report.checks, Check("chain", False)))
            if name in rolled:
                verdict = "rolled_back"
                if binding.pending_period is not None:
                    binding.store.periods.discard(binding.pending_period, reason="failed nightly gates")
            elif report.awaiting:
                verdict = "awaiting_evidence"
                self._settle_period(binding, waiting=None if closure.id == current[name] else closure.id)
            elif chain is not None and not passed and (earlier_derivations or list(report.failed) == ["chain"]):
                # The chain cannot move tonight: a member has nothing new to judge, or only a sibling failed its
                # own gates. Nothing is judged here: the candidate waits (never withdrawn for the chain alone),
                # and its period stays, so its rows still train when the chain moves.
                self._drop_replaced_candidates(binding, closure.id)
                self._settle_period(binding, waiting=closure.id)
                results.append({"learner": name, "closure": closure.id, "result": "unchanged",
                                "why": [CHAIN_WAITING]})
                continue
            else:
                self._drop_replaced_candidates(binding, closure.id)
                verdict = binding.controller.consider(closure.id, report, period=binding.pending_period)
            if verdict == "promoted" and callable(getattr(binding.driver, "promoted", None)):
                binding.driver.promoted(closure)
            self._publish_pins()
            result = {"learner": name, "closure": closure.id, "result": verdict, "why": list(report.failed)}
            if getattr(binding.registered, "kind", None) == "behaviour":
                result["not_measured_guards"] = binding.comparator.summary().get("not_measured_guards", [])
            results.append(result)
        if chain is not None and passed:
            self.joints.record(chain.name, {name: self.loaded[name].store.revisions.serving().id for name in names})
            self.joints.end(chain.name)
        return results

    @staticmethod
    def _drop_replaced_candidates(binding, keep):
        """Delete (never withdraw) the candidates a member left waiting on earlier nights that tonight's closure
        ``keep`` replaces. Such a candidate never served and nothing was decided about it, so it must not count as
        having vetted the periods it names (``RevisionStore.introduced``) when a later closure is judged; its
        identity may return, and its rows stay in the periods that train ``keep``."""
        revisions = binding.store.revisions
        serving = revisions.serving().id
        for closure in revisions.retained():
            if closure.id not in (keep, serving) and revisions.status(closure.id).state == "candidate":
                revisions.delete(closure.id)

    @staticmethod
    def _settle_idle_period(binding):
        """Only an empty idle period is disposable; every entry remains unjudged."""
        if binding.pending_period is None:
            return
        store = binding.store
        period = store.periods.get(binding.pending_period)
        if not any(manifest.entries for manifest in store.manifests_of([period])):
            store.periods.discard(period.id, reason="idle night: nothing to learn")

    @staticmethod
    def _settle_period(binding, *, waiting=None):
        """Keep tonight's period, on a night that judged nothing for this learner, only where something can still
        learn from it: the waiting candidate names it (discarding it would drop that candidate as stale), it holds
        quarantined rows (offered again once trainable, while a discarded period's rows are excluded for good), or
        it holds admitted rows and every build reads every retained period (rebuild mode with accumulation). Any
        other period is discarded, as on a judged night: a stacked or non-accumulating build reads only the newest
        period, and an empty one holds nothing."""
        period_id = binding.pending_period
        if period_id is None:
            return
        store = binding.store
        if waiting is not None and period_id in store.revisions.get(waiting).periods:
            return
        verdicts = {entry.verdict for manifest in store.manifests_of([store.periods.get(period_id)])
                    for entry in manifest.entries}
        if "quarantine" in verdicts:
            return
        if ("admitted" in verdicts and store.mode == "rebuild"
                and binding.declaration.get("accumulation", True)):
            return
        store.periods.discard(period_id, reason="unchanged night: " + (
            "unused by later builds" if "admitted" in verdicts else "nothing to learn"))

    def run(self):
        if not self.enabled or not self.loaded:
            return []
        with self.lock:
            self._resume_rebuilds()
            if any(binding.store.rebuild_required.exists() for binding in self.loaded.values()):
                raise ValueError("pending source erasure rebuild must finish before learning")
            if any((binding.store.directory / "erasures" / "PENDING").exists()
                   for binding in self.loaded.values()):
                raise ValueError("retry the pending erasure with the same refs before learning")
            # Complete an interrupted pointer move before admitting another period.
            self._recover_moves()
            results = []
            groups = [(chain.members, chain) for chain in self.chains]
            chained = {name for chain in self.chains for name in chain.members}
            groups += [((name,), None) for name in self.loaded if name not in chained]
            for names, chain in groups:
                prepared = {}
                previous = {name: self.loaded[name].store.revisions.serving().id for name in names}
                for name in names:
                    self.loaded[name].pending_closure = None
                    self.loaded[name].pending_period = None
                    self.loaded[name].pending_idle = None
                try:
                    for name in names:
                        prepared[name] = self._prepare(name)
                    results.extend(self._finish(names, prepared, chain))
                except Exception as error:
                    # Preserve the failed check, not an exception's input or model text.
                    for name in names:
                        binding = self.loaded[name]
                        if binding.pending_idle is not None:
                            # Another chain member's error cannot burn this unjudged period.
                            idle = binding.pending_idle
                            self._settle_idle_period(binding)
                            results.append({"learner": name, "closure": idle.id, **idle.details()})
                            continue
                        closure = binding.pending_closure
                        serving = binding.store.revisions.serving()
                        if (serving.mode != "base" and
                                (getattr(binding.registered, "kind", None) == "behaviour" or
                                 (closure is not None and closure.id != previous[name] and serving.id == closure.id))):
                            self._rollback(name, reason="nightly_error", expected_closure=serving.id)
                        elif closure is not None and binding.store.revisions.status(closure.id).state == "candidate":
                            binding.controller.consider(closure.id, GateReport(
                                closure.id, "nightly", (Check("nightly_error", False, type(error).__name__),)),
                                period=binding.pending_period)
                        if binding.pending_period is not None:
                            binding.store.periods.discard(binding.pending_period, reason="nightly_error")
                        current = binding.store.periods.current()
                        if current is not None:
                            binding.store.periods.discard(current.id, reason="nightly_error")
                        results.append({"learner": name, "result": "error", "why": _error_why(error)})
                    self._recover_moves()
                    logger.warning("plasticity nightly failed (%s)", ", ".join(_error_why(error)))
                for name in names:
                    compact(self.loaded[name].store)
            self._last = {"at": _io.stamp(self._clock()), "learners": results}
            _io.write_json(self.root / "nightly.json", self._last)
            for row in results:
                logger.info("plasticity %s: %s (%s)", row["learner"], row["result"], ", ".join(row["why"]))
            self._publish()
            return results

    def _rollback(self, name, *, reason, expected_closure=None):
        with self.lock:
            binding = self.loaded[name]
            if expected_closure and binding.store.revisions.serving().id != expected_closure:
                return "stale"
            chain = next((chain for chain in self.chains if name in chain.members), None)
            if chain is None:
                result = binding.controller.rollback(reason=reason, expected_closure=expected_closure)
                self._publish_pins()
                return result
            if binding.store.revisions.serving().mode == "base":
                return "kept"
            excluded = {member: [self.loaded[member].store.revisions.serving().id]
                        for member in chain.members
                        if self.loaded[member].store.revisions.serving().mode != "base"}
            targets = {member: self.loaded[member].controller.fallback(
                self.loaded[member].store.revisions.serving()).id for member in chain.members}
            self.joints.begin(chain.name, targets, excluded=excluded, operation="rollback", reason=reason)
            for member in chain.members:
                self.loaded[member].controller.rollback(reason=reason)
            self.joints.record(chain.name, {member: self.loaded[member].store.revisions.serving().id
                                           for member in chain.members})
            self.joints.end(chain.name)
            self._publish_pins()
            return "rolled_back"

    def _recover_moves(self):
        """Finish rollback intent and joint moves before subsequent learning or erasure."""
        for binding in self.loaded.values():
            binding.controller.recover_rollback()
        if any(self.joints.pending(chain.name) for chain in self.chains):
            self._eraser().erase(())
        for binding in self.loaded.values():
            binding.comparator.summary()

        self._publish_pins()

    def rollback(self, name):
        result = self._rollback(name, reason="operator rollback")
        self._publish()
        return {"learner": name, "result": result, "serving": self.loaded[name].store.revisions.serving().id}

    def _eraser(self):
        if self.instinct_store is None:
            from ..instinct.log import Store, DB_FILE
            if (self.data_dir / DB_FILE).exists():
                self.instinct_store = Store(self.data_dir / DB_FILE)
        stores = {name: binding.store for name, binding in self.loaded.items()}
        groups = {}
        for name, saved in self._catalog().items():
            if name not in stores:
                stores[name] = LearnerStore(self.root, name, base=saved["base"], mode=saved["mode"],
                                            declared=saved["declared"], clock=self._clock)
            if saved.get("chain"):
                groups.setdefault(saved["chain"], []).append(name)
        chains = tuple(Chain(name, tuple(members)) for name, members in groups.items())
        for store in stores.values():
            Controller(store, clock=self._clock, lock=self.lock).recover_rollback()
        return Erasure(stores, chains=chains, joints=self.joints, clock=self._clock,
                       calibrations=self.instinct_store, lock=self.lock)

    def erase_sources(self, ids):
        """Resolve persisted provenance without loading disabled or missing drivers.

        Older sources expose only a reverse lookup. When that driver is unavailable,
        its derived store is conservatively cleared and a fresh build is required.
        """
        ids = tuple(dict.fromkeys(ids))
        if not ids:
            return
        with self.lock:
            previous = self._resume_source_erasures()
            eraser = self._eraser()
            if not eraser.stores:
                return
            store = next(iter(eraser.stores.values()))
            hashes = {store.source_hash(source) for source in ids}
            pending = self._source_erasures()
            refs = {ref for source in hashes for ref in pending.get(source, ())}
            try:
                erased = set(_io.read_json(self.root / "erased-sources.json"))
            except FileNotFoundError:
                erased = set()
            _io.write_json(self.root / "erased-sources.json", sorted(erased | hashes))
            unresolved = set()
            for name, store in eraser.stores.items():
                saved_refs, complete = store.refs_resting_on(ids)
                refs.update(saved_refs)
                if name not in self.loaded and complete:
                    comparator = self._stored_comparator(name, store)
                    ref_hashes = {comparator._path(ref).stem for ref in store.source_refs()}
                    complete = all(row["ref_hash"] in ref_hashes for row in comparator.records())
                if name not in self.loaded and not complete:
                    unresolved.add(name)
            for binding in self.loaded.values():
                resting = getattr(binding.source.source, "refs_resting_on", None)
                if resting is not None:
                    refs.update(resting(ids))
            if refs:
                for source in hashes:
                    pending[source] = sorted(refs)
                _io.write_json(self.root / "source-erasures.json", pending)
            if unresolved:
                self._reset_unresolved(eraser, unresolved)
            if refs:
                result = self.erase(tuple(sorted(refs)))
                _io.write_json(self.root / "source-erasures.json",
                               {source: refs for source, refs in pending.items() if source not in hashes})
                return result
            self._publish()
            return previous

    def _source_erasures(self):
        try:
            return _io.read_json(self.root / "source-erasures.json")
        except FileNotFoundError:
            return {}

    def _resume_source_erasures(self):
        """Finish the existing transaction before accepting another source deletion."""
        pending = self._source_erasures()
        result = None
        while pending:
            refs = next(iter(pending.values()))
            result = self.erase(refs)
            pending = {source: values for source, values in pending.items() if values != refs}
            _io.write_json(self.root / "source-erasures.json", pending)
        return result

    def _reset_unresolved(self, eraser, names):
        """Invalidate the full lineage when provenance cannot establish a smaller closure."""
        lineage = Lineage({name: store.revisions for name, store in eraser.stores.items()})
        names = set(names)
        while True:
            before = set(names)
            roots = [(name, closure.id) for name in names for closure in eraser.stores[name].revisions.known()]
            names.update(name for name, _ in lineage.descendants(roots))
            # A joint learner must be rebuilt as a whole, even if one member has no rows.
            for chain in eraser.chains:
                if names.intersection(chain.members):
                    names.update(chain.members)
            if names == before:
                break
        for name in names:
            _io.write_json(eraser.stores[name].rebuild_required, {"observe": True})
        for name in names:
            store = eraser.stores[name]
            # All these files are derived learner data. Keep only revision withdrawal
            # identities and erasure intents; fresh admission must not inherit dropped
            # row exclusions, numeric layouts, samples, or comparator evidence.
            store.clear_derived(calibrations=eraser.calibrations)
            if name in self.loaded:
                self.loaded.pop(name)
                self.rejected[name] = "source erasure requires reloading for a clean rebuild"
        self.chains = tuple(chain for chain in self.chains if not names.intersection(chain.members))
        for chain in eraser.chains:
            if names.intersection(chain.members):
                for record in self.joints.records(chain.name):
                    self.joints.delete(chain.name, record["seq"])

    def _resume_rebuilds(self):
        """Finish deferred reconstruction before a reloaded learner can serve."""
        groups = [(chain.members, chain) for chain in self.chains]
        chained = {name for chain in self.chains for name in chain.members}
        groups += [((name,), None) for name in self.loaded if name not in chained]
        for names, chain in groups:
            if not any(self.loaded[name].store.rebuild_required.exists() for name in names):
                continue
            if any((self.loaded[name].store.directory / "erasures" / "PENDING").exists() for name in names):
                continue
            try:
                prepared = {}
                for name in names:
                    binding = self.loaded[name]
                    store = binding.store
                    store.revisions.unserve()
                    marker = (_io.read_json(store.rebuild_required) if store.rebuild_required.exists()
                              else {"observe": False})
                    if not marker["observe"]:
                        Erasure._rewrite(store, {manifest.id: manifest for manifest in store.manifests},
                                         frozenset(), (binding.driver.admission, binding.source))
                    prepared[name] = self._prepare(name, observe=marker["observe"])
                self._finish(names, prepared, chain)
                for name in names:
                    _io.remove(self.loaded[name].store.rebuild_required)
            except Exception as error:
                for name in names:
                    self.loaded[name].store.revisions.unserve()
                logger.warning("plasticity erasure rebuild failed (%s)", ", ".join(_error_why(error)))

    def _stored_comparator(self, name, store):
        kind = self._catalog().get(name, {}).get("kind")
        # Older catalogs predate kind persistence; numeric settings are an unambiguous
        # storage-format witness and do not require the implementation to be loaded.
        if kind == "numeric-online" or (store.directory / "numeric-comparator.json").exists():
            from .numeric_comparator import NumericComparator  # noqa: F401
            kind = "numeric-online"
        comparator = COMPARATORS.get(kind, LiveComparator)
        return comparator(Controller(store, clock=self._clock),
                          key=(self.root / "comparator.key").read_bytes(), lock=self.lock, clock=self._clock)

    def erase(self, refs):
        refs = tuple(dict.fromkeys(refs))
        with self.lock:
            self._recover_moves()
            eraser = self._eraser()
            if refs:
                key_path = self.root / "comparator.key"
                if not key_path.exists():
                    _io.write_atomic(key_path, os.urandom(32))
                    key_path.chmod(0o600)
                tombstones = _erased_refs(self.root)
                tombstones.update(_erased_hash(self.root, ref) for ref in refs)
                _io.write_json(self.root / "erased-refs.json", sorted(tombstones))
            # An off or rejected learner still owns retained comparator records.
            key_path = self.root / "comparator.key"
            if key_path.exists():
                for name, store in eraser.stores.items():
                    if name not in self.loaded:
                        self._stored_comparator(name, store).erase(refs)
            for store in eraser.stores.values():
                store.forget_source_refs(refs)
            for binding in self.loaded.values():
                binding.source.blocked.update(refs)
                binding.comparator.erase(refs)
            readmit = {name: (binding.driver.admission, binding.source) for name, binding in self.loaded.items()}
            # The core persists the rebuild request before deleting any manifests.
            # Chains use one callback for the group; retry replays the same clean build.
            groups = [(chain.members, chain) for chain in self.chains]
            chained = {name for chain in self.chains for name in chain.members}
            groups += [((name,), None) for name in self.loaded if name not in chained]
            callbacks = {}
            rebuilt = set()
            for names, chain in groups:
                if all(self.loaded[name].store.mode == "rebuild" for name in names):
                    def rebuild(names=names, chain=chain):
                        if names in rebuilt:
                            return
                        prepared = {name: self._prepare(name, observe=False) for name in names}
                        self._finish(names, prepared, chain)
                        rebuilt.add(names)
                    callbacks.update({name: rebuild for name in names})
            for name, store in eraser.stores.items():
                if name in self.loaded:
                    continue
                def defer(store=store):
                    # Preserve surviving manifests and their eligibility for re-admission.
                    # Discard + compaction would permanently exclude their clean rows.
                    _io.write_json(store.rebuild_required, {"observe": False})
                    store.revisions.unserve()
                callbacks[name] = defer
            report = eraser.erase(refs, readmit=readmit, rebuild=callbacks)
            if not report.clean:
                raise ValueError("erasure scan incomplete; retry with the same refs")
            for binding in self.loaded.values():
                if hasattr(binding.source.source, "erase"):
                    binding.source.source.erase(refs)
            self._publish()
            return {**asdict(report), "clean": report.clean}

    def learners(self):
        return [{"name": name, "loaded": name in self.loaded,
                 "error": self.rejected.get(name),
                 "serving": self.loaded[name].store.revisions.serving().id if name in self.loaded else None}
                for name in self.declarations]

    def status(self):
        return {"enabled": self.enabled, "learners": self.learners(), "rejected": dict(self.rejected),
                "note": self._note()}

    def _note(self):
        if not self.enabled:
            return "Plasticity is off; requests use the base."
        if not self.declarations:
            return "Plasticity is on with no learners configured."
        return "" if self.loaded else "No configured learners loaded."

    def revisions(self, name):
        store = self.loaded[name].store
        return [{**closure.to_dict(), "id": closure.id, "status": asdict(store.revisions.status(closure.id))}
                for closure in store.revisions.retained()]

    def comparator(self, name):
        return self.loaded[name].comparator.summary()

    def report(self):
        self._recover_moves()
        rows = []
        for name, binding in self.loaded.items():
            teachers = []
            teacher_note = ""
            if binding.last_build is not None:
                try:
                    from .objectives import teacher_lines
                    teachers = teacher_lines(binding.last_build)
                except ImportError:
                    teacher_note = "Teacher split unavailable until the registration and teacher module is installed."
            else:
                try:
                    saved = _io.read_json(binding.store.directory / "teacher-report.json")
                    teachers, teacher_note = saved["teachers"], saved.get("teacher_note", "")
                except FileNotFoundError:
                    pass
            comparison = binding.comparator.summary()
            rows.append({"name": name, "serving": binding.store.revisions.serving().id,
                         "teachers": teachers, "teacher_note": teacher_note, "events": binding.controller.events(),
                         "comparator": comparison,
                         "erasures": len(list((binding.store.directory / "erasures").glob("*.json")))})
            if self._minimum_labels(binding):
                try:
                    rows[-1].update(_io.read_json(self.root / "idle-reports" / f"{name}.json"))
                except FileNotFoundError:
                    pass
        return {"enabled": self.enabled, "note": self._note(), "rejected": dict(self.rejected), "learners": rows}

    def _publish_pins(self):
        if not any(binding.declaration.get("implementation") == "behaviour" for binding in self.loaded.values()):
            self.behaviour_pins = ()
            return
        from .learners.behaviour import publish_pins
        publish_pins(self)

    def _publish(self):
        with self.lock:
            self._publish_pins()
            report = self.report()
            for row in report["learners"]:
                _io.write_json(self.loaded[row["name"]].store.directory / "teacher-report.json",
                               {"teachers": row["teachers"], "teacher_note": row["teacher_note"]})
            _io.write_json(self.root / "report.json", report)


class NightlyTask:
    """Run once when the mind's existing local nightly boundary is crossed."""

    def __init__(self, runtime, *, tz="UTC", boundary_minute=180, started_at=None):
        self.runtime = runtime
        self.tz = ZoneInfo(tz) if isinstance(tz, str) else tz
        self.boundary_minute = boundary_minute
        self.started_at = started_at or runtime._clock()

    def tick(self, now):
        if not self.runtime.enabled or not self.runtime.loaded:
            return []
        from ..mind.authority import boundary_crossed
        with self.runtime.lock:
            path = self.runtime.root / "nightly-boundary.json"
            try:
                previous = _io.parse(_io.read_json(path)["at"])
            except FileNotFoundError:
                previous = self.started_at
                _io.write_json(path, {"at": _io.stamp(previous)})
            if not boundary_crossed(previous, now, tz=self.tz, minute=self.boundary_minute):
                return []
            result = self.runtime.run()
            _io.write_json(path, {"at": _io.stamp(now)})
            return result

    async def run(self, *, sleep=asyncio.sleep):
        while True:
            worker = None
            try:
                worker = asyncio.create_task(asyncio.to_thread(self.tick, self.runtime._clock()))
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                # The thread still owns state and may be rebuilding or erasing. The
                # lifespan must await it before closing the runtime, even if cancelled again.
                if worker is not None:
                    while not worker.done():
                        try:
                            await asyncio.shield(worker)
                        except asyncio.CancelledError:
                            continue
                        except Exception:
                            break
                    if not worker.cancelled():
                        try:
                            worker.result()
                        except Exception as error:
                            logger.warning("plasticity maintenance failed (%s)", type(error).__name__)
                raise
            except Exception as error:
                logger.warning("plasticity maintenance failed (%s)", type(error).__name__)
            await sleep(POLL_SECONDS)


def erase_for_log(path, refs):
    """Standalone Instinct forgets reach retained weights even with the faculty off."""
    from ..config import load_config
    from ..instinct.log import Store
    directory = Path(path).parent
    config = load_config(directory)
    root = directory / config.get("plasticity.directory", "plasticity")
    if not (root / "catalog.json").exists():
        return
    runtime = Runtime.from_config(config, data_dir=directory, instinct_store=Store(path))
    try:
        return runtime.erase(refs)
    finally:
        runtime.close()


def read_report(config):
    """Read the published counts without constructing learners or touching their stores."""
    root = (config.home / config.get("plasticity.directory", "plasticity")).resolve()
    if not root.is_relative_to(config.home.resolve()) or root == config.home.resolve():
        raise ValueError("plasticity.directory must stay under the data directory")
    enabled = config.get("mind.faculties.plasticity", True) is True
    configured = config.get("plasticity.learners", {})
    if not configured:
        return {"enabled": enabled, "learners": [], "rejected": {}, "note":
                "Plasticity is on with no learners configured." if enabled else
                "Plasticity is off; requests use the base."}
    try:
        result = _io.read_json(root / "report.json")
    except FileNotFoundError:
        result = {"learners": [], "rejected": {}, "note": "Plasticity is on with no learners configured."
                  if not config.get("plasticity.learners", {}) else "No nightly learner report yet."}
    result["enabled"] = enabled
    result["learners"] = [row for row in result["learners"] if row["name"] in configured]
    result["rejected"] = {name: reason for name, reason in result["rejected"].items() if name in configured}
    if not enabled:
        result["note"] = "Plasticity is off; requests use the base."
    for row in result["learners"]:
        directory = root / "learners" / row["name"]
        row["erasure_pending"] = (directory / "erasures" / "PENDING").exists()
        try:
            row["serving"] = _io.read_json(directory / "revisions" / "SERVING")["closure"]
        except FileNotFoundError:
            pass
    return result


async def _live(runtime, driver):
    while True:
        try:
            await driver.live(runtime)
            return
        except Exception as error:
            logger.warning("plasticity live failed (%s)", type(error).__name__)
        await asyncio.sleep(LIVE_RESTART_SECONDS)


@asynccontextmanager
async def lifespan(config, *, data_dir, mind=None):
    """Own nightly maintenance and live learners for the server lifespan."""
    from .learners.service_health import off_loop
    runtime = None
    tasks = []

    def initialize():
        nonlocal runtime
        runtime = Runtime.from_config(config, data_dir=data_dir)

    try:
        # Drain initialization even on cancellation, then close its erasure hooks
        # in finally. No runtime may outlive a cancelled startup.
        await off_loop(initialize)
        if mind is not None:
            mind.plasticity = runtime
            from .learners import behaviour
            if getattr(mind, "lessons", None) is not None:
                mind.lessons.standing_pins = lambda: behaviour.pins(runtime)
            for binding in runtime.loaded.values():
                if callable(getattr(getattr(binding, "driver", None), "bind_mind", None)):
                    await off_loop(binding.driver.bind_mind, mind)
        if runtime.enabled and runtime.loaded:
            if mind is not None:
                timezone = mind.tz
                minute = mind.consolidation.boundary_minute()
            else:
                from ..mind.authority import parse_quiet_hours
                from ..mind.consolidate import NIGHT_MINUTE
                quiet = parse_quiet_hours(config.get("mind.quiet_hours", ""))
                minute = quiet[0] if quiet and quiet[0] != quiet[1] else NIGHT_MINUTE
                timezone = os.environ.get("COLONYMIND_TIMEZONE", "UTC")
            tasks.append(asyncio.create_task(NightlyTask(runtime, tz=timezone, boundary_minute=minute).run()))
            for binding in runtime.loaded.values():
                driver = getattr(binding, "driver", None)
                if callable(getattr(driver, "live", None)):
                    tasks.append(asyncio.create_task(_live(runtime, driver)))
        yield runtime
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        if mind is not None:
            if getattr(mind, "lessons", None) is not None:
                mind.lessons.standing_pins = None
            mind.plasticity = None
        if runtime is not None:
            runtime.close()
