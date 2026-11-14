"""The two ways to build a revision.

- **Mode 1, rebuild** (``rebuild``): where training data is kept by reference and retraining is cheap, each build is
  a clean retrain from the base on the current manifests. An erased row cannot re-render, so it simply drops out of
  the next build; no period stack is needed.
- **Mode 2, stack** (``stack``): where data is not kept or retraining is costly, each period trains from the state it
  starts from on its own rows only, and its delta is recorded. Erasure then discards back to the last clean period.

A learner is anything with the ``Learner`` protocol; the core never steps a model itself. Rows reach the learner in
one canonical order (by ref, then teacher) whatever order the source yields them in, so a build is a function of its
manifests: the same manifests and source give the same state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from . import _io
from .audience import narrowest
from .admission import recall_weight
from .manifests import Manifest, RowError, RowSource, check_row
from .objectives import Objective

MODES: Tuple[str, ...] = ("rebuild", "stack")
state_digest = _io.state_digest


@dataclass(frozen=True)
class TrainingRow:
    """One admitted row as a learner reads it: its re-rendered state, its target, its teacher's term and weight."""

    ref: str
    point: str
    state: str
    label: Any
    teacher: str
    push_away: bool
    loss: str
    weight: float
    audience: str
    rank_weight: float = 1.0


@runtime_checkable
class Learner(Protocol):
    """What the core needs of a learner: its base state and a fit from a start state on rows. The engine's learners
    implement it; ``synthetic.SyntheticLearner`` implements it for tests."""

    name: str

    def base(self) -> Dict[str, np.ndarray]:
        ...

    def fit(self, start: Mapping[str, np.ndarray], rows: Sequence[TrainingRow], *, objective: Objective,
            seed: int) -> Dict[str, np.ndarray]:
        ...

    def losses(self, state: Mapping[str, np.ndarray], rows: Sequence[TrainingRow], *,
               objective: Objective) -> Mapping[str, float]:
        """One unblended mean loss per declared teacher, zero for a teacher without rows."""
        ...


@dataclass(frozen=True)
class Build:
    """A built learned state and what it came from. ``audience``: the narrowest audience of everything it learned
    (``guest`` when it learned nothing). ``start``: the digest of the state a stacked period started from."""

    mode: str
    state: Mapping[str, np.ndarray]
    digest: str
    refs: FrozenSet[str]
    manifests: Tuple[str, ...]
    audience: str
    rows: int
    start: str = ""
    teacher_losses: Mapping[str, float] = field(default_factory=dict)
    teacher_rows: Mapping[str, int] = field(default_factory=dict)

    @property
    def loss(self) -> float:
        return sum(self.teacher_losses.values())


def check_state(state: Mapping[str, np.ndarray]) -> None:
    """ValueError unless every value of ``state`` is finite."""
    for name, array in state.items():
        if not np.all(np.isfinite(np.asarray(array))):
            raise ValueError(f"learned state {name} holds a value that is not finite")


def _as_admitted(row: Mapping[str, Any], entry: Any) -> bool:
    """Whether ``row`` is still the row its manifest ``entry`` admitted: every key admission decided on (its point and
    version, state hash, label, direction, source kind, audience and time) unchanged."""
    try:
        same_time = _io.parse(row["at"]) == _io.parse(entry.at)
    except ValueError:
        return False
    return (same_time and row["point"] == entry.point and row["version"] == entry.version
            and row["state_hash"] == entry.state_hash and _io.canonical(row["label"]) == _io.canonical(entry.label)
            and row["push_away"] == entry.push_away and row["source_kind"] == entry.source_kind
            and row["audience"] == entry.audience)


def training_rows(objective: Objective, manifests: Sequence[Manifest], source: RowSource) -> List[TrainingRow]:
    """The rows the manifests admitted that ``source`` still re-renders unchanged, in canonical order. A row whose
    source no longer yields it, or yields it with anything admission decided on changed (its state hash, label,
    direction, point, version, source kind, audience or time), is left out until its period is admitted again
    (``admission.Admission.readmit``): a row whose audience narrowed since would otherwise train at the audience it
    was admitted with, and one whose source kind changed would skip the rules its new kind meets. A manifest admitted
    under another objective (another identity: a changed blend, teacher or rule) never trains: its periods are
    admitted again under the current objective first."""
    admitted: Dict[Tuple[str, str], Any] = {}
    for manifest in manifests:
        if manifest.objective_identity != objective.identity:
            raise ValueError(f"manifest {manifest.id} was admitted under objective {manifest.objective} "
                             f"({manifest.objective_identity}), not {objective.name} ({objective.identity})")
        for entry in manifest.admitted():
            admitted[(entry.ref, entry.teacher)] = entry
    found: Dict[Tuple[str, str], TrainingRow] = {}
    for row in source.rows(objective):
        try:
            check_row(row)
        except RowError:
            continue
        key = (row["ref"], row["teacher"])
        entry = admitted.get(key)
        if entry is None or key in found:
            continue
        if not _as_admitted(row, entry):
            continue
        found[key] = TrainingRow(ref=row["ref"], point=row["point"], state=row["state"], label=row["label"],
                                 teacher=row["teacher"], push_away=row["push_away"],
                                 loss=objective.losses[row["teacher"]],
                                 weight=float(objective.blend[row["teacher"]]) * recall_weight(objective, row),
                                 audience=entry.audience, rank_weight=recall_weight(objective, row))
    return [found[key] for key in sorted(found)]


def _audience(rows: Sequence[TrainingRow], inherited: Optional[str]) -> str:
    audiences = [row.audience for row in rows] + ([inherited] if inherited else [])
    return narrowest(audiences) if audiences else "guest"


def _teacher_terms(learner: Learner, objective: Objective, state, rows) -> dict:
    raw = dict(learner.losses(state, rows, objective=objective))
    if set(raw) != set(objective.losses) or any(isinstance(value, bool) or not isinstance(value, (int, float))
                                               or not math.isfinite(value) or value < 0 for value in raw.values()):
        raise ValueError("a build needs a finite nonnegative loss for each declared teacher")
    return {"teacher_losses": {name: float(objective.blend[name]) * value for name, value in raw.items()},
            "teacher_rows": {name: sum(row.teacher == name for row in rows) for name in raw}}


def rebuild(learner: Learner, objective: Objective, manifests: Sequence[Manifest], source: RowSource, *,
            seed: int = 0) -> Build:
    """Mode 1: a clean retrain from the learner's base on ``manifests``."""
    rows = training_rows(objective, manifests, source)
    state = learner.fit(learner.base(), rows, objective=objective, seed=seed)
    return Build(mode="rebuild", state=state, digest=state_digest(state), refs=frozenset(row.ref for row in rows),
                 manifests=tuple(manifest.id for manifest in manifests), audience=_audience(rows, None),
                 rows=len(rows), **_teacher_terms(learner, objective, state, rows))


def stack(learner: Learner, objective: Objective, start: Mapping[str, np.ndarray], manifests: Sequence[Manifest],
          source: RowSource, *, inherited: Optional[str] = None, seed: int = 0) -> Build:
    """Mode 2: one period trained from ``start`` on its own ``manifests``. ``inherited``: the audience of the state it
    starts from (None for the base), since a stacked state carries everything its earlier periods learned."""
    rows = training_rows(objective, manifests, source)
    state = learner.fit(start, rows, objective=objective, seed=seed)
    return Build(mode="stack", state=state, digest=state_digest(state), refs=frozenset(row.ref for row in rows),
                 manifests=tuple(manifest.id for manifest in manifests), audience=_audience(rows, inherited),
                 rows=len(rows), start=state_digest(start), **_teacher_terms(learner, objective, state, rows))


def delta(state: Mapping[str, np.ndarray], start: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """What a stacked period added: ``state`` minus ``start``, array by array."""
    if set(state) != set(start):
        raise ValueError("a delta is taken between states of the same arrays")
    return {name: np.asarray(state[name]) - np.asarray(start[name]) for name in state}


def apply(start: Mapping[str, np.ndarray], *deltas: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """``start`` with ``deltas`` added in order."""
    result = {name: np.array(array, copy=True) for name, array in start.items()}
    for step in deltas:
        if set(step) != set(result):
            raise ValueError("a delta applies to a state of the same arrays")
        for name in result:
            result[name] = result[name] + step[name]
    return result


def relative_difference(first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray]) -> float:
    """The largest relative difference, array by array (the erasure drill's measure)."""
    if set(first) != set(second):
        return math.inf
    worst = 0.0
    for name in first:
        a, b = np.asarray(first[name], dtype=np.float64), np.asarray(second[name], dtype=np.float64)
        if a.shape != b.shape:
            return math.inf
        worst = max(worst, float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-12)))
    return worst


__all__ = ["MODES", "Build", "Learner", "TrainingRow", "apply", "check_state", "delta", "rebuild",
           "relative_difference", "stack", "state_digest", "training_rows"]
