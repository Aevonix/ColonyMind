"""One learner's files together: its manifests, periods, revisions, budget checks and stacked deltas.

Layout, under ``root/learners/<name>/``: ``manifests/`` (content-addressed, rows by reference), ``periods/``,
``revisions/`` (closures, tombstones, learned states, statuses and the ``SERVING`` pointer), ``budgets/``,
``deltas/`` (a stacked period's delta) and ``erasures/`` (what each erasure did, by count and closure id, never by
the erased refs).

Every closure written through the store is checked first (``validate``): the manifests it names are kept and each
belongs to one of the periods it names (so an erasure that finds a period finds every closure built on it, even
after a manifest was rewritten under a new id), and it claims no wider audience than what it learned from (its
manifests' admitted rows and, for a stacked period, the state it started from).
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import hmac
from pathlib import Path
from typing import Callable, FrozenSet, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from . import _io
from .audience import BREADTH, narrowest
from .budgets import BudgetLedger
from .builds import MODES
from .certificates import GateRecords
from .manifests import Manifest, ManifestStore
from .periods import Period, PeriodError, PeriodLedger
from .revisions import KEEP, Closure, RevisionError, RevisionStore


class LearnerStore:
    """A learner's store. ``mode``: ``rebuild`` (mode 1) or ``stack`` (mode 2); ``base``: the base identity;
    ``declared``: the dependent keys its revisions carry."""

    def __init__(self, root: Path, name: str, *, base: str, mode: str = "rebuild", declared: Iterable[str] = (),
                 keep: int = KEEP, clock: Optional[Callable[[], datetime]] = None):
        if mode not in MODES:
            raise ValueError(f"a learner builds by {' or '.join(MODES)}, not {mode!r}")
        self.root = Path(root)
        self.name = name
        self.mode = mode
        self.directory = self.root / "learners" / name
        self.manifests = ManifestStore(self.directory / "manifests")
        self.periods = PeriodLedger(self.directory / "periods", name, clock=clock)
        self.revisions = RevisionStore(self.directory / "revisions", name, base=base, declared=declared, keep=keep,
                                       validate=self.validate)
        self.budgets = BudgetLedger(self.directory / "budgets")
        self.certificates = GateRecords(self.directory / "gate-certificates")

    # -- source provenance ----------------------------------------------------------------------------------------

    @property
    def rebuild_required(self) -> Path:
        return self.directory / "erasures" / "REBUILD_REQUIRED"

    def erasure_pending(self) -> bool:
        return self.rebuild_required.exists() or (self.directory / "erasures" / "PENDING").exists()

    def source_hash(self, source: str) -> str:
        key = (self.root / "comparator.key").read_bytes()
        return hmac.new(key, _io.canonical(["source", source]), hashlib.sha256).hexdigest()

    def source_refs(self):
        """Row refs and keyed source ids, never row state. None means unknown provenance."""
        try:
            saved = _io.read_json(self.directory / "source-refs.json")
        except (FileNotFoundError, ValueError):
            return {}
        if not isinstance(saved, dict) or any(not isinstance(ref, str) or not ref
                or (sources is not None and (not isinstance(sources, list)
                    or any(not isinstance(source, str) for source in sources))) for ref, sources in saved.items()):
            return {}
        return saved

    def record_source_refs(self, dependencies):
        """Persist before admission or serving. Explicit empty sources are known independent rows.

        Row sources may implement ``source_ids(ref)`` to declare all label and feature
        dependencies. A reverse-only ``refs_resting_on`` cannot establish this record.
        Keep the union across observations because older revisions still depend on it.
        """
        saved = self.source_refs()
        for ref, sources in dependencies.items():
            if sources is None:
                saved[ref] = None
                continue
            if isinstance(sources, (str, bytes)):
                raise ValueError("source_ids must be an iterable of source identifiers")
            sources = tuple(sources)
            if any(not isinstance(source, str) or not source for source in sources):
                raise ValueError("source_ids must contain nonempty source identifiers")
            if ref in saved and saved[ref] is None:
                continue
            saved[ref] = sorted(set(saved.get(ref, ())) | {self.source_hash(source) for source in sources})
        if saved != self.source_refs():
            _io.write_json(self.directory / "source-refs.json", saved)

    def refs_resting_on(self, ids):
        """Return persisted dependent refs and whether every retained row has known provenance."""
        saved = self.source_refs()
        hashes = {self.source_hash(source) for source in ids}
        refs = {ref for ref, sources in saved.items() if sources is not None and hashes.intersection(sources)}
        retained = {ref for manifest in self.manifests for ref in manifest.refs()}
        complete = bool(saved) and all(sources is not None for sources in saved.values()) and retained <= saved.keys()
        return refs, complete

    def clear_derived(self, *, calibrations=None):
        """Finish a conservative reset without losing withdrawal or erasure intents.

        The runtime withdraws the complete lineage before this step. Keeping the
        marker until reconstruction also makes an interrupted deletion retryable.
        Aggregate gate evidence survives, with its row-set digest removed.
        """
        self.revisions.unserve()
        for closure in self.revisions.known():
            if calibrations is not None:
                calibrations.withdraw_revisions({closure.id, closure.state}, records=closure.calibration)
            self.revisions.set_status(closure.id, "withdrawn", reason="unresolved source erasure")
            self.revisions.delete(closure.id)
        for path in self.certificates.directory.glob("*.json"):
            record = _io.read_json(path)
            if record.get("gate_set") is not None:
                record["gate_set"] = None
                _io.write_json(path, record)
        for path in self.directory.rglob("*"):
            if path.is_file() and path.relative_to(self.directory).parts[0] not in {
                    "revisions", "erasures", "gate-certificates"}:
                _io.remove(path)
        self.sweep()

    def forget_source_refs(self, refs):
        saved = self.source_refs()
        kept = {ref: sources for ref, sources in saved.items() if ref not in refs}
        if saved != kept:
            _io.write_json(self.directory / "source-refs.json", kept)

    # -- closures --------------------------------------------------------------------------------------------------

    def learned_audiences(self, closure: Closure) -> List[str]:
        """The audiences of what ``closure`` learned in this learner: its manifests' admitted rows and, for a stacked
        period, the state it started from. RevisionError when a manifest it names is not kept."""
        found: List[str] = []
        for manifest_id in closure.manifests:
            try:
                manifest = self.manifests.get(manifest_id)
            except KeyError:
                raise RevisionError(f"closure {closure.id} names manifest {manifest_id}, which is not kept") from None
            found += [entry.audience for entry in manifest.admitted()]
        if closure.mode == "stack" and closure.parent and closure.parent != self.revisions.base.id:
            parent = self.revisions.find(closure.parent)
            if parent is None:
                raise RevisionError(f"closure {closure.id} stacks on closure {closure.parent}, which is not known")
            found.append(parent.audience)
        return found

    def validate(self, closure: Closure) -> None:
        """The learner's checks on a closure before it is written (the module docstring); RevisionError on the first
        that fails."""
        if closure.audience not in BREADTH:
            raise RevisionError(f"closure {closure.id}: audience {closure.audience!r} is not one of "
                                f"{', '.join(BREADTH)}")
        learned = self.learned_audiences(closure)
        in_periods = set()
        for period_id in closure.periods:
            try:
                in_periods.update(self.periods.get(period_id).manifests)
            except PeriodError:
                raise RevisionError(f"closure {closure.id} names period {period_id}, which is not in the "
                                    "ledger") from None
        stray = sorted(set(closure.manifests) - in_periods)
        if stray:
            raise RevisionError(f"closure {closure.id}: manifest(s) {', '.join(stray)} belong to none of the periods "
                                "it names")
        if learned and BREADTH[closure.audience] > BREADTH[narrowest(learned)]:
            raise RevisionError(f"closure {closure.id} claims audience {closure.audience}, wider than what it learned "
                                f"from ({narrowest(learned)})")

    def manifests_of(self, periods: Iterable[Period]) -> List[Manifest]:
        return [self.manifests.get(manifest_id) for period in periods for manifest_id in period.manifests]

    def current_manifests(self) -> List[Manifest]:
        """The manifests of the periods that still train."""
        return self.manifests_of(self.periods.retained())

    def decided(self) -> FrozenSet[Tuple[str, str]]:
        """The (ref, teacher) pairs a period has already decided, in any state but quarantine: they are not offered
        again, so a witness stays a witness and a dropped period's rows never train later."""
        found = set()
        for period in self.periods.periods():
            for manifest_id in period.manifests:
                try:
                    manifest = self.manifests.get(manifest_id)
                except KeyError:
                    continue
                found.update((entry.ref, entry.teacher) for entry in manifest.entries
                             if entry.verdict != "quarantine")
        return frozenset(found)

    # -- stacked deltas --------------------------------------------------------------------------------------------

    def delta_path(self, digest: str) -> Path:
        return self.directory / "deltas" / f"{digest}.npz"

    def save_delta(self, period_id: str, delta: Mapping[str, np.ndarray]) -> str:
        """Record a stacked period's delta: its file, and its digest on the period."""
        digest = _io.state_digest(delta)
        path = self.delta_path(digest)
        if not path.exists():
            _io.write_atomic(path, _io.save_arrays(delta))
        self.periods.set_delta(period_id, digest)
        return digest

    def delta(self, digest: str) -> Mapping[str, np.ndarray]:
        state = _io.load_arrays(self.delta_path(digest).read_bytes())
        if _io.state_digest(state) != digest:
            raise ValueError(f"delta {digest}: its file does not match its digest")
        return state

    # -- what a crash leaves ---------------------------------------------------------------------------------------

    def orphans(self) -> List[Path]:
        """Files nothing retained names: state files no whole closure names, and delta files no closed period
        names."""
        found = list(self.revisions.orphans())
        directory = self.directory / "deltas"
        if directory.is_dir():
            named = {period.delta for period in self.periods.retained() if period.delta}
            found += sorted(path for path in directory.glob("*.npz") if path.stem not in named)
        return found

    def sweep(self) -> List[Path]:
        """Remove the temporaries and orphans a crash left; the orphans that could not be removed."""
        _io.sweep(self.directory)
        for path in self.orphans():
            _io.remove(path)
        return [path for path in self.orphans() if path.exists()]


__all__ = ["LearnerStore"]
