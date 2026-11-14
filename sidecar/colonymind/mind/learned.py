"""Code-registered learned factors for discretionary rank and dispatch guidance.

Each time view keeps the factors it scored so the intention records those same
values even when deliberation reshapes its candidate. Learner faults are neutral;
only their exception types are logged. The ranker owns the combined product bound.
"""

from __future__ import annotations

import logging
import math
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional

from .rank import score

logger = logging.getLogger(__name__)
MAX_NOTES = 2  # Plan assumption: at most two learned notes per dispatch.


@dataclass(frozen=True)
class FactorSpec:
    implementation: str
    low: float
    high: float
    factor: Callable[[Any, str, Any, datetime], float]
    note: Optional[Callable[[Any, str, Any, datetime], Optional[str]]] = None
    wants_context: bool = False
    observe: Optional[Callable[[Any, str, Any, datetime], None]] = None


class FactorValue(float):
    """An opted-in factor may neutralize named factors for baseline exploration.

    Overrides can only be 1.0; they cannot amplify another learner or bypass its
    bounds. Ordinary floats and legacy callbacks keep their existing behavior.
    """

    def __new__(cls, value: float, *, overrides: Mapping[str, float]):
        if any(value != 1.0 for value in overrides.values()):
            raise ValueError("factor overrides must be neutral")
        result = super().__new__(cls, value)
        result.overrides = MappingProxyType(dict(overrides))
        return result


@dataclass(frozen=True)
class Ranking:
    """The served eligible order and its original candidates' formed intentions."""

    eligible: tuple
    context: Mapping[str, Any]
    rescore: Callable[[Any, Mapping[str, float]], float]
    acted: tuple = ()


def _snapshot(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _snapshot(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, set):
        return frozenset(_snapshot(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return replace(value, **{field.name: _snapshot(getattr(value, field.name))
                                     for field in fields(value) if field.init})
        except Exception:   # a dataclass replace cannot rebuild: keep the object as given
            return value
    return value


class _Feedback:
    """Remember the exact feedback reads made by the served ranking pass."""

    def __init__(self, source: Any) -> None:
        self.source = source
        self.values: Dict[str, float] = {}

    def multiplier(self, key: str) -> float:
        try:
            value = float(self.source.multiplier(key)) if self.source is not None else 1.0
        except Exception:
            self.values[key] = 1.0
            raise
        self.values[key] = value
        return value


SPECS: Dict[str, FactorSpec] = {}


def register(spec: FactorSpec) -> None:
    """Register one bounded factor per implementation, when its module imports."""
    if spec.implementation in SPECS:
        raise ValueError(f"factor already registered: {spec.implementation}")
    if not (math.isfinite(spec.low) and math.isfinite(spec.high) and 0 <= spec.low <= spec.high):
        raise ValueError("factor bounds must be finite, nonnegative and ordered")
    SPECS[spec.implementation] = spec


class LearnedFactors:
    def __init__(self, mind: Any) -> None:
        self.mind = mind

    def at(self, now: datetime, context: Mapping[str, Any] | None = None) -> _View:
        """Snapshot the caller's ranking context, empty on legacy dispatch paths.

        The intention-forming pass supplies drives, configured base, threshold,
        affect and last owner turn. Only specs with ``wants_context`` receive the
        read-only mapping, with the same identity through scoring and observation.
        """
        return _View(getattr(self.mind, "plasticity", None), now, context or {})


class _View:
    def __init__(self, runtime: Any, now: datetime, context: Mapping[str, Any]) -> None:
        self.runtime, self.now = runtime, now
        self._values: Dict[int, tuple[Any, Dict[str, float]]] = {}
        self._specs = tuple(self._loaded())
        try:
            self.context = _snapshot({**context,
                                      "implementations": tuple(spec.implementation for spec, _ in self._specs)})
        except Exception as error:   # the ranking pass never fails for a learner's context
            logger.warning("learned context unavailable (%s)", type(error).__name__)
            self.context = MappingProxyType({})
        self._feedback = _Feedback(None)
        self._feedback_values: Dict[int, Dict[str, float]] = {}
        self._observed = False

    def _loaded(self):
        if self.runtime is None:
            return
        for spec in SPECS.values():
            try:
                learner = self.runtime.find(spec.implementation)
            except Exception as error:
                logger.warning("learned factor unavailable (%s)", type(error).__name__)
                continue
            if learner is not None:
                yield spec, learner

    def explain(self, candidate: Any) -> Dict[str, float]:
        """The loaded implementations' clamped factors, empty for owed work."""
        if not candidate.multiplier_keys():
            return {}
        key = id(candidate)
        if key not in self._values:
            values = {}
            overrides = {}
            for spec, learner in self._specs:
                try:
                    kwargs = {"context": self.context} if spec.wants_context else {}
                    result = spec.factor(self.runtime, learner, candidate, self.now, **kwargs)
                    value = float(result)
                    if math.isnan(value):
                        raise ValueError("factor is NaN")
                    value = max(spec.low, min(spec.high, value))
                    if spec.wants_context and isinstance(result, FactorValue):
                        overrides.update(result.overrides)
                except Exception as error:
                    logger.warning("learned factor unavailable (%s)", type(error).__name__)
                    value = 1.0
                values[spec.implementation] = value
            values.update({name: value for name, value in overrides.items() if name in values})
            # Keep the candidate too, so an id cannot be reused within this view.
            self._values[key] = (candidate, values)
        return dict(self._values[key][1])

    def factor(self, candidate: Any) -> float:
        self._feedback_values.setdefault(id(candidate), dict(self._feedback.values))
        return math.prod(self.explain(candidate).values())

    def feedback(self, source: Any) -> _Feedback:
        self._feedback = _Feedback(source)
        return self._feedback

    def ranking(self, ranked: Any) -> Ranking:
        """Freeze score inputs before deliberation may reshape a candidate.

        ``rescore(candidate, overrides)`` replaces named loaded factors within
        their bounds, retaining every other factor and the actual feedback reads.
        It neither reevaluates learners nor changes the served eligible order.
        """
        snapshots = {}
        bounds = {spec.implementation: (spec.low, spec.high) for spec, _ in self._specs}
        served = {id(candidate): value for candidate, value in ranked}
        if not self._specs:
            # The off arm: no learner is loaded, so no copies; the served scores stand.
            return Ranking(tuple(ranked), self.context, lambda candidate, overrides: served[id(candidate)])
        try:
            for candidate, value in ranked:
                snapshots[id(candidate)] = (deepcopy(candidate), value, self.explain(candidate),
                                            dict(self._feedback_values.get(id(candidate), {})))
        except Exception as error:   # an observer loses this pass; the ranking pass goes on
            logger.warning("learned ranking snapshot unavailable (%s)", type(error).__name__)
            self._observed = True
            return Ranking(tuple(ranked), self.context, lambda candidate, overrides: served[id(candidate)])

        def rescore(candidate: Any, overrides: Mapping[str, float]) -> float:
            original, served, values, feedback = snapshots[id(candidate)]
            if not overrides or not original.multiplier_keys():
                return served
            adjusted = dict(values)
            for name, raw in overrides.items():
                if name in adjusted:
                    low, high = bounds[name]
                    value = float(raw)
                    adjusted[name] = 1.0 if math.isnan(value) else max(low, min(high, value))
            return score(original, drives=self.context.get("drives"), affect=self.context.get("affect"),
                         feedback=SimpleNamespace(multiplier=lambda key: feedback.get(key, 1.0)),
                         learned=SimpleNamespace(factor=lambda candidate: math.prod(adjusted.values())))

        return Ranking(tuple(ranked), self.context, rescore)

    def observe(self, ranking: Ranking, *, acted: Any = ()) -> None:
        """Notify each loaded observer once, after the intention-forming loop."""
        if self._observed:
            return
        self._observed = True
        ranking = replace(ranking, acted=tuple(acted))
        for spec, learner in self._specs:
            if spec.observe is None:
                continue
            try:
                spec.observe(self.runtime, learner, ranking, self.now)
            except Exception as error:
                logger.warning("learned observer unavailable (%s)", type(error).__name__)

    def notes(self, row: Any) -> List[str]:
        """At most two distinct non-empty notes from loaded learners."""
        notes = []
        for spec, learner in self._specs:
            if spec.note is None:
                continue
            try:
                kwargs = {"context": self.context} if spec.wants_context else {}
                note = spec.note(self.runtime, learner, row, self.now, **kwargs)
                if isinstance(note, str) and note.strip() and note not in notes:
                    notes.append(note)
            except Exception as error:
                logger.warning("learned note unavailable (%s)", type(error).__name__)
            if len(notes) == MAX_NOTES:
                break
        return notes


__all__ = ["FactorSpec", "FactorValue", "LearnedFactors", "Ranking", "SPECS", "register"]
