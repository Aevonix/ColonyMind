"""A deterministic synthetic learner, its rows and its gate items, for tests and drills.

``SyntheticLearner`` is logistic regression over synthetic features, in NumPy: a state's features are drawn from a
generator seeded by the state's text, and its planted label is the side of a fixed hyperplane it falls on. It
implements the core's ``builds.Learner`` protocol with full-batch gradient descent from the start state, with a
small pull back toward that start (so a rebuild stays near the base and a stacked period near the state it started
from), and a fit is a function of its rows (the erasure drill compares a rebuild with a clean retrain). A pulling
row's term is the cross-entropy toward its label; a push-away row's term is a margin on the disfavoured label only:
it lowers that label's probability while it is above one half and never raises anything past it.

``SyntheticSource`` is a ``RowSource`` over rows held in memory: an erased row no longer re-renders, and a row whose
state no longer matches its logged hash is not yielded. ``Cycle`` drives one learner through a whole night the way a
caller would (admit a period, build, propose, gate, consider), so tests can reach every state.

Every number here is an assumption made for tests: the feature count, the steps and rate, the budget, the suite's
seeds and size, the canaries.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import _io, builds
from .admission import Admission
from .budgets import Budget, BudgetCheck
from .builds import Build, TrainingRow
from .gates import CalibrationRefit, Evidence, GateReport, Item, MIN_POOLED, SetLedger, evaluate, pooled_suite, refit_calibration
from .manifests import Manifest
from .objectives import Objective
from .revisions import Closure
from .states import Controller
from .store import LearnerStore

EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
POINT = "synthetic_yes_no"
LABELS = ("yes", "no")
KEY = b"synthetic learner key, not a secret"
FEATURES = 8
#: The synthetic learner's budget: at most this change in log-odds anywhere in the feature box (an assumption).
BUDGET = 20.0
#: Gate items lie at least this far from the planted hyperplane (an assumption): clear cases, so a revision that
#: learned the rule gets them right and the retention gate reads learning, not noise at the boundary.
SUITE_MARGIN = 0.8
SUITE_SEEDS = tuple(range(1000, 1010))
ITEMS_PER_SEED = 100
CONTROL_SEED = 7
_TRUE = np.random.default_rng(20260101).normal(size=FEATURES)

OBJECTIVE = Objective(
    name="synthetic_outcome", model="synthetic", kind="supervised_refit", target_type="label",
    losses={"verified_outcome": "cross_entropy", "owner_label": "cross_entropy", "synthetic": "cross_entropy",
            "deliberation": "kl", "behavioural_proxy": "margin"},
    blend={"verified_outcome": 1.0, "owner_label": 1.0, "synthetic": 0.5, "deliberation": 0.5,
           "behavioural_proxy": 0.1},
    labels=LABELS, points=(POINT,), what="the synthetic learner's objective (tests only; not registered)")


def state_hash(text: str) -> str:
    """The keyed hash a decision log keeps for a state (here with a fixed test key)."""
    return hmac.new(KEY, text.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def features(states: Sequence[str], dimensions: int = FEATURES) -> np.ndarray:
    """Each state's features in [-1, 1], seeded by its text, with a constant 1 last (the offset is a weight)."""
    rows = []
    for text in states:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        rows.append(np.append(np.random.default_rng(seed).uniform(-1.0, 1.0, dimensions), 1.0))
    return np.array(rows, dtype=np.float64).reshape(len(rows), dimensions + 1)


def planted(text: str) -> str:
    """The planted label of a state: which side of the fixed hyperplane its features fall on."""
    return "yes" if float(features([text])[0, :FEATURES] @ _TRUE) > 0.0 else "no"


def _positive(label: Any) -> bool:
    return label in ("yes", 1, True)


class SyntheticLearner:
    """Logistic regression over ``features`` (the module docstring)."""

    name = "synthetic"
    budget_type = "anchored_output"

    def __init__(self, *, steps: int = 500, rate: float = 2.0, pull: float = 0.01):
        self.steps = steps
        self.rate = rate
        self.pull = pull

    @property
    def identity(self) -> str:
        return f"synthetic-logistic:{FEATURES}:{self.steps}:{self.rate}:{self.pull}"

    def base(self) -> Dict[str, np.ndarray]:
        return {"w": np.zeros(FEATURES + 1)}

    def _weights(self, state: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.asarray(state["w"], dtype=np.float64) if state else self.base()["w"]

    def fit(self, start: Mapping[str, np.ndarray], rows: Sequence[TrainingRow], *, objective: Objective,
            seed: int) -> Dict[str, np.ndarray]:
        anchor = self._weights(start).copy()
        weights = anchor.copy()
        rows = [row for row in rows if objective.blend[row.teacher] > 0]
        if not rows:
            return {"w": weights}
        matrix = features([row.state for row in rows])
        positive = np.array([_positive(row.label) for row in rows], dtype=np.float64)
        push = np.array([row.push_away for row in rows], dtype=bool)
        # Each teacher has its own mean term; row volume cannot change the declared blend.
        counts = {name: sum(row.teacher == name for row in rows) for name in objective.losses}
        scale = np.array([objective.blend[row.teacher] * row.rank_weight / counts[row.teacher]
                          for row in rows], dtype=np.float64)
        if not np.any(scale):
            return {"w": weights}
        for _ in range(self.steps):
            probability = 1.0 / (1.0 + np.exp(-(matrix @ weights)))
            # Pulling rows: cross-entropy toward the label. Pushing rows: -log(1 - p(label)) while p(label) > 1/2.
            disfavoured = np.where(positive == 1.0, probability, 1.0 - probability)
            residual = np.where(push, np.where(disfavoured > 0.5, probability - (1.0 - positive), 0.0),
                                probability - positive)
            weights = weights - self.rate * ((matrix.T @ (scale * residual)) + self.pull * (weights - anchor))
        return {"w": weights}

    def losses(self, state: Mapping[str, np.ndarray], rows: Sequence[TrainingRow], *,
               objective: Objective) -> Dict[str, float]:
        """Per-teacher means at this state, before the objective's blend is applied.

        For hard binary targets KL to a one-hot target equals cross-entropy. The push-only
        term is the continuous capped margin used by fit, zero below one half.
        """
        result = {name: 0.0 for name in objective.losses}
        if not rows:
            return result
        logits = features([row.state for row in rows]) @ self._weights(state)
        positive = np.array([_positive(row.label) for row in rows], dtype=float)
        losses = np.logaddexp(0.0, logits) - positive * logits
        pushing = np.maximum(0.0, np.logaddexp(0.0, (2.0 * positive - 1.0) * logits) - np.log(2.0))
        terms = np.where([row.push_away for row in rows], pushing, losses)
        for name in result:
            selected = [float(terms[index]) * row.rank_weight for index, row in enumerate(rows) if row.teacher == name]
            result[name] = float(np.mean(selected)) if selected else 0.0
        return result

    def predict(self, state: Mapping[str, np.ndarray], states: Sequence[str]) -> np.ndarray:
        """The probability of ``yes`` for each state."""
        return 1.0 / (1.0 + np.exp(-(features(states) @ self._weights(state))))

    def budget_used(self, state: Mapping[str, np.ndarray]) -> float:
        """The largest change in log-odds anywhere in the feature box, against the base: the sum of the weights'
        absolute changes."""
        return float(np.abs(self._weights(state) - self.base()["w"]).sum())

    def exposure(self, state: Mapping[str, np.ndarray], canaries: Sequence[Item]) -> Dict[str, float]:
        """Each canary's exposure: the probability of its secret label."""
        probability = self.predict(state, [canary.state for canary in canaries])
        return {canary.id: float(p if _positive(canary.label) else 1.0 - p)
                for canary, p in zip(canaries, probability)}


def correct(learner: SyntheticLearner, state: Mapping[str, np.ndarray], items: Sequence[Item]) -> List[bool]:
    """Per item, whether ``state`` predicts its label (a probability of exactly one half reads as ``no``)."""
    probability = learner.predict(state, [item.state for item in items])
    return [bool((p > 0.5) == _positive(item.label)) for p, item in zip(probability, items)]


#: A canary lies at least this far from the planted hyperplane (an assumption), so a learner that follows the rule
#: is sure of its side and only one that memorised the canary leans toward its secret label.
CANARY_MARGIN = 1.5


def canaries(count: int = 4) -> Tuple[Item, ...]:
    """Planted canaries: states far from the planted hyperplane whose secret label contradicts the planted rule, so
    only memorising one would expose it. They are never offered as rows."""
    found = []
    index = 0
    while len(found) < count:
        text = f"{POINT} canary {index}"
        index += 1
        if abs(float(features([text])[0, :FEATURES] @ _TRUE)) < CANARY_MARGIN:
            continue
        found.append(Item(id=f"canary-{len(found)}", state=text, label="no" if planted(text) == "yes" else "yes"))
    return tuple(found)


def shuffled(rows: Sequence[TrainingRow], *, seed: int) -> List[TrainingRow]:
    """``rows`` with their labels permuted: the shuffled control's training set."""
    order = np.random.default_rng(seed).permutation(len(rows))
    return [replace(row, label=rows[int(index)].label) for row, index in zip(rows, order)]


class SyntheticTemplate:
    """Gate items at a seed: new states at least ``SUITE_MARGIN`` from the planted hyperplane, labelled by it."""

    name = "synthetic_suite"

    def __init__(self, per_seed: int = ITEMS_PER_SEED):
        self.per_seed = per_seed

    def generate(self, seed: int) -> List[Item]:
        items: List[Item] = []
        index = 0
        while len(items) < self.per_seed:
            text = f"{POINT} suite {seed} {index}"
            index += 1
            if abs(float(features([text])[0, :FEATURES] @ _TRUE)) < SUITE_MARGIN:
                continue
            items.append(Item(id=f"item-{seed}-{len(items)}", state=text, label=planted(text)))
        return items


def make_rows(count: int, *, start: int = 0, prefix: str = "case", refs: Optional[str] = None, point: str = POINT,
              teacher: str = "owner_label", source_kind: str = "explicit", audience: str = "owner",
              at: Optional[str] = None, label: Any = None, push_away: bool = False,
              version: str = "v1") -> List[Dict[str, Any]]:
    """Rows of the contract. The state is ``"<point> <prefix> <i>"`` and the ref ``"<refs or prefix>-<i>"``, so the
    same prefix with other refs gives repeats of the same states; the label is the planted one unless given."""
    rows = []
    for index in range(start, start + count):
        text = f"{point} {prefix} {index}"
        rows.append({"ref": f"{refs or prefix}-{index}", "point": point, "version": version, "state": text,
                     "state_hash": state_hash(text), "label": planted(text) if label is None else label,
                     "teacher": teacher, "push_away": push_away, "source_kind": source_kind, "audience": audience,
                     "at": at or EPOCH.isoformat()})
    return rows


class SyntheticSource:
    """A ``RowSource`` over rows in memory (the module docstring). ``source``: the contact, device or tool that
    supplied a batch, for the per-source cap (None for the owner's own rows)."""

    def __init__(self, rows: Iterable[Mapping[str, Any]] = (), *, source: Optional[str] = None):
        self._rows: List[Dict[str, Any]] = []
        self._sources: Dict[str, str] = {}
        self._erased: set = set()
        self.add(rows, source=source)

    def add(self, rows: Iterable[Mapping[str, Any]], *, source: Optional[str] = None) -> None:
        for row in rows:
            self._rows.append(dict(row))
            if source is not None:
                self._sources[row["ref"]] = source

    def rows(self, objective: Objective, *, since: Any = None) -> Iterator[Dict[str, Any]]:
        moment = _io.parse(since) if since is not None else None
        for row in self._rows:
            if row["ref"] in self._erased:
                continue
            if objective.points and row["point"] not in objective.points:
                continue
            if state_hash(row["state"]) != row["state_hash"]:
                continue  # the source changed since: it cannot re-render the logged state
            if moment is not None and _io.parse(row["at"]) < moment:
                continue
            yield dict(row)

    def erase(self, refs: Iterable[str]) -> None:
        self._erased |= set(refs)

    def alter(self, ref: str, **changes: Any) -> None:
        """Change what a ref re-renders to (its logged hash stays): a source changed after the fact."""
        for row in self._rows:
            if row["ref"] == ref:
                row.update(changes)

    def source_of(self, row: Mapping[str, Any]) -> Optional[str]:
        return self._sources.get(row["ref"])


class Cycle:
    """One learner's nights, driven the way a caller would, for tests and drills."""

    minimum_labels = {"gate": MIN_POOLED}  # The generated suite already supplies this minimum.

    def label_counts(self, build):
        return {"gate": len(self.suite.items)}

    def __init__(self, store: LearnerStore, controller: Controller, *, source: SyntheticSource,
                 clock: Optional[Callable[[], datetime]] = None, learner: Optional[SyntheticLearner] = None,
                 objective: Objective = OBJECTIVE, sets: Optional[SetLedger] = None,
                 refit: Optional[CalibrationRefit] = None):
        self.store, self.controller, self.source = store, controller, source
        self.refit: CalibrationRefit = refit or (lambda state: [f"calibration-{state[:16]}"])
        self.learner = learner or SyntheticLearner()
        self.objective = objective
        self.sets = sets or SetLedger(store.directory / "sets")
        self.suite = pooled_suite([SyntheticTemplate()], seeds=SUITE_SEEDS)
        self.sets.register(self.suite.id, "gate_suite", self.suite.refs())
        if store.budgets.declared(store.name) is None:
            store.budgets.declare(Budget(store.name, self.learner.budget_type, BUDGET))
        self.admission = Admission(objective, learner=store.name, source_of=source.source_of,
                                   protected_refs=self.sets.protected_refs(), clock=clock)
        self.canaries = canaries()
        self.last_period: Optional[str] = None
        self.last_report: Optional[GateReport] = None

    def period(self) -> Manifest:
        """Admit what the source yields into a new period, and close it."""
        period = self.store.periods.open()
        admitted = self.admission.admit(self.source.rows(self.objective), period=period.id,
                                        already=self.store.decided())
        self.store.periods.add_manifest(period.id, self.store.manifests.put(admitted.manifest))
        self.store.periods.close(period.id)
        self.last_period = period.id
        return admitted.manifest

    def _start(self) -> Tuple[Closure, Dict[str, np.ndarray]]:
        serving = self.store.revisions.serving()
        return serving, (self.store.revisions.state(serving) or self.learner.base())

    def build(self) -> Build:
        if self.store.mode == "rebuild":
            return builds.rebuild(self.learner, self.objective, self.store.current_manifests(), self.source)
        serving, start = self._start()
        newest = self.store.periods.retained()[-1:]
        return builds.stack(self.learner, self.objective, start, self.store.manifests_of(newest), self.source,
                            inherited=None if serving.mode == "base" else serving.audience)

    def propose(self, build: Build) -> Closure:
        serving, start = self._start()
        retained = self.store.periods.retained()
        periods = tuple(period.id for period in (retained if build.mode == "rebuild" else retained[-1:]))
        closure = self.store.revisions.compose(
            state=build.digest, objective=self.objective.identity,
            calibration=refit_calibration(self.refit, build.digest),
            dependents={"threshold": f"threshold-{build.digest[:8]}"}, parent=serving.id, manifests=build.manifests,
            periods=periods, audience=build.audience, mode=build.mode)
        self.controller.propose(closure, build.state)
        if build.mode == "stack" and periods:
            self.store.save_delta(periods[-1], builds.delta(build.state, start))
        self.store.budgets.record(BudgetCheck(self.store.name, build.digest, self.learner.budget_type, BUDGET,
                                              self.learner.budget_used(build.state)))
        return closure

    def evidence(self, closure: Closure, build: Build, **overrides: Any) -> Evidence:
        serving, start = self._start()
        items = self.suite.items
        rows = builds.training_rows(self.objective, self.store.current_manifests(), self.source)
        control = self.learner.fit(self.learner.base(), shuffled(rows, seed=CONTROL_SEED), objective=self.objective,
                                   seed=0)
        states = [item.state for item in items]
        fields: Dict[str, Any] = dict(
            reference=correct(self.learner, start, items), base=correct(self.learner, self.learner.base(), items),
            candidate=correct(self.learner, build.state, items),
            control=correct(self.learner, control, items),
            canaries_base=self.learner.exposure(self.learner.base(), self.canaries),
            canaries_candidate=self.learner.exposure(build.state, self.canaries),
            served=self.learner.predict(self.store.revisions.state(closure), states),
            offline=self.learner.predict(build.state, states),
            metrics={"accuracy": float(np.mean(correct(self.learner, build.state, items)))},
            budget=self.store.budgets.result(self.store.name, build.digest), calibration=closure.calibration,
            parent_calibration=serving.calibration, state=build.state)
        fields.update(overrides)
        return Evidence(**fields)

    def gate(self, closure: Closure, build: Build, **overrides: Any) -> GateReport:
        return evaluate(closure.id, self.suite, self.evidence(closure, build, **overrides), sets=self.sets)

    def rebuild(self, *, period: Optional[str] = None, **overrides: Any) -> Tuple[Closure, GateReport, str]:
        """Build, propose, gate and consider, without a new period (an erasure's rebuild)."""
        build = self.build()
        closure = self.propose(build)
        report = self.gate(closure, build, **overrides)
        self.last_report = report
        return closure, report, self.controller.consider(closure.id, report, period=period)

    def step(self, **overrides: Any) -> Tuple[Closure, GateReport, str]:
        """One night: a new period, then build, propose, gate and consider. ``overrides`` replace evidence fields."""
        self.period()
        return self.rebuild(period=self.last_period, **overrides)


__all__ = ["BUDGET", "EPOCH", "FEATURES", "LABELS", "OBJECTIVE", "POINT", "Cycle", "SyntheticLearner",
           "SyntheticSource", "SyntheticTemplate", "canaries", "correct", "features", "make_rows", "planted",
           "shuffled", "state_hash"]
