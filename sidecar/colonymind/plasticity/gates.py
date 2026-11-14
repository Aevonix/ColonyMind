"""The automatic gates before a revision serves, and the evaluation hygiene they rest on.

**Sets.** Nightly selection runs on gate suites; release claims use sealed evaluation sets that no selection ever
touches, since repeated nightly selection against one set would leak into it. ``SetLedger`` enforces it by set id and
a use ledger: a set used for selection can never be named for evaluation, a sealed set is never used for selection,
the two kinds share no item, a set's role and items never change, and every registered item joins the protected set
that admission refuses. A suite gates only when every one of its items is registered under its id.

**Pooled suites.** A bound of about 1 point needs about 1,000 paired items, so a gate suite pools at least
``MIN_POOLED`` items from templates at new seeds, never the frozen ones. The items are distinct, by id and by state,
wherever a suite comes from: a repeated item would count one observation many times.

**The statistics.** Non-inferiority is judged by the one-sided 95% upper bound of the decline, never by a point
estimate. For paired items the decline is ``(losses - gains) / n`` (a loss: the reference was right and the candidate
wrong), and its bound is the decline plus ``z`` standard errors, with the variance floored so that a set with no
discordant pair still carries the rule of three (``3 / n``): 100 items and no error at all still leave a 3-point
bound, and 1,000 items at 3% disagreement with no net decline give about 0.9 points. The floor is my construction
(an assumption): it reproduces both figures and never tightens as losses grow.

**The gates**, each a named check:

- ``retention``: the decline on the gate suite against the serving revision and against the base, each bounded at
  ``DECLINE_BAR`` (against the base too, so nights that each decline a little within the bar never add up);
- ``specificity``: the candidate beats its shuffled control (the lower bound of its gain is above zero, my
  assumption for "beats");
- ``canaries``: every planted canary's exposure is at the base's level, never above it;
- ``calibration``: the calibration was refit for this revision through the caller's refit hook
  (``refit_calibration``); thresholds are never carried over from other weights;
- ``budget``: the engine's recorded budget check of this very state (its digest) passed;
- ``finite``: every learned value and every metric is finite;
- ``served_offline``: the served outputs agree with the offline ones.

A failed gate drops the period and the previous revision keeps serving (``states.Controller``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from . import _io
from .budgets import BudgetCheck

#: The one-sided 95% normal quantile.
Z95 = 1.6448536269514722
#: A decline of at most 1 point, judged by its one-sided 95% bound (D assumption).
DECLINE_BAR = 0.01
#: Pooled suites hold at least this many items (D 6.4).
MIN_POOLED = 1000
#: Served and offline outputs agree to this absolute difference (my assumption).
SERVED_TOLERANCE = 1e-6
#: A canary's exposure may not exceed the base's by more than this (my assumption: not at all).
CANARY_TOLERANCE = 0.0
ROLES: Tuple[str, ...] = ("gate_suite", "sealed")
USES: Tuple[str, ...] = ("selection", "evaluation")
CHECKS: Tuple[str, ...] = ("retention", "specificity", "canaries", "calibration", "budget", "finite",
                           "served_offline")


class SetError(ValueError):
    """A use of an evaluation set that would break the hygiene, and why."""


# -- the statistics ---------------------------------------------------------------------------------------------------

def rule_of_three(n: int) -> float:
    """The one-sided 95% upper bound on a rate when ``n`` trials showed no event (about ``3 / n``)."""
    if n <= 0:
        raise ValueError("a bound needs at least one item")
    return 3.0 / n


def decline_bound(n: int, *, losses: int, gains: int, z: float = Z95) -> float:
    """The one-sided upper bound of the decline over ``n`` paired items (the module docstring)."""
    if n <= 0:
        raise ValueError("a bound needs at least one item")
    if losses < 0 or gains < 0 or losses + gains > n:
        raise ValueError("losses and gains are counts of distinct items out of n")
    mean = (losses - gains) / n
    variance = (losses + gains) / n - mean * mean
    floor = (rule_of_three(n) / z) ** 2 * n
    return mean + z * math.sqrt(max(variance, floor) / n)


def gain_lower_bound(n: int, *, wins: int, losses: int, z: float = Z95) -> float:
    """The one-sided lower bound of a gain over ``n`` paired items (``wins``: items only the candidate got right)."""
    return -decline_bound(n, losses=losses, gains=wins, z=z)


def wilson(x: float, n: float, z: float = Z95) -> Tuple[float, float]:
    """Wilson score interval, including fractional effective counts."""
    if not all(math.isfinite(v) for v in (x, n, z)) or n <= 0 or not 0 <= x <= n or z <= 0:
        raise ValueError("Wilson bounds need finite counts 0 <= x <= n, n > 0 and z > 0")
    p = x / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0., c - h), min(1., c + h)


def two_proportion_bounds(x1: float, n1: float, x2: float, n2: float,
                          z: float = Z95) -> Tuple[float, float]:
    """Newcombe hybrid score bounds on p1 - p2; each is one-sided 95% at Z95."""
    lo1, hi1 = wilson(x1, n1, z)
    lo2, hi2 = wilson(x2, n2, z)
    p1, p2 = x1 / n1, x2 / n2
    d = p1 - p2
    return (max(-1., d - math.hypot(p1 - lo1, hi2 - p2)),
            min(1., d + math.hypot(hi1 - p1, p2 - lo2)))


def paired(reference: Sequence[bool], candidate: Sequence[bool]) -> Tuple[int, int, int]:
    """``(n, losses, gains)``: items the reference got right and the candidate wrong, and the reverse."""
    if len(reference) != len(candidate):
        raise ValueError("paired results cover the same items")
    losses = sum(1 for before, after in zip(reference, candidate) if before and not after)
    gains = sum(1 for before, after in zip(reference, candidate) if after and not before)
    return len(reference), losses, gains


# -- sets --------------------------------------------------------------------------------------------------------------

class SetLedger:
    """Registered evaluation sets and every use of them, in ``directory``."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def _set_path(self, set_id: str) -> Path:
        return self.directory / "sets" / f"{_io.digest(set_id)}.json"

    def _uses_path(self, set_id: str) -> Path:
        return self.directory / "uses" / f"{_io.digest(set_id)}.json"

    def _sets(self) -> List[Dict[str, Any]]:
        directory = self.directory / "sets"
        return [_io.read_json(path) for path in sorted(directory.glob("*.json"))] if directory.is_dir() else []

    def role(self, set_id: str) -> Optional[str]:
        try:
            return _io.read_json(self._set_path(set_id))["role"]
        except FileNotFoundError:
            return None

    def register(self, set_id: str, role: str, refs: Iterable[str]) -> None:
        if role not in ROLES:
            raise SetError(f"a set's role is one of {', '.join(ROLES)}")
        items = sorted(set(refs))
        existing = self.role(set_id)
        if existing is not None:
            if existing != role:
                raise SetError(f"set {set_id} is registered as {existing}; its role never changes")
            if items != sorted(self.refs(set_id)):
                raise SetError(f"set {set_id} is registered with other items; a set's items never change")
            return
        for other in self._sets():
            if other["role"] != role and not set(other["refs"]).isdisjoint(items):
                raise SetError(f"set {set_id} shares items with {other['role']} set {other['id']}: gate suites and "
                               "sealed sets are kept apart")
        _io.write_json(self._set_path(set_id), {"id": set_id, "role": role, "refs": items})

    def refs(self, set_id: str) -> FrozenSet[str]:
        """The items registered under ``set_id`` (none when it is not registered)."""
        try:
            return frozenset(_io.read_json(self._set_path(set_id))["refs"])
        except FileNotFoundError:
            return frozenset()

    def uses(self, set_id: str) -> List[Dict[str, str]]:
        try:
            return list(_io.read_json(self._uses_path(set_id)))
        except FileNotFoundError:
            return []

    def use(self, set_id: str, use: str, *, revision: str) -> None:
        """Record ``use`` of ``set_id`` for ``revision``; SetError when the hygiene forbids it."""
        if use not in USES:
            raise SetError(f"a set is used for {' or '.join(USES)}")
        role = self.role(set_id)
        if role is None:
            raise SetError(f"unknown set {set_id}: register it before any use")
        uses = self.uses(set_id)
        if use == "selection" and role == "sealed":
            raise SetError(f"set {set_id} is sealed: it is never used for selection")
        if use == "evaluation" and (role == "gate_suite" or any(item["use"] == "selection" for item in uses)):
            raise SetError(f"set {set_id} was used for selection: it can never be named as an evaluation set")
        _io.write_json(self._uses_path(set_id), uses + [{"use": use, "revision": revision}])

    def protected_refs(self) -> FrozenSet[str]:
        """Every item of every registered set: never trained on."""
        return frozenset(ref for record in self._sets() for ref in record["refs"])


# -- pooled suites -----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Item:
    """One gate item: its id, the state a learner reads, and its label."""

    id: str
    state: str
    label: Any


class Template(Protocol):
    """Generates gate items at a seed."""

    name: str

    def generate(self, seed: int) -> Sequence[Item]:
        ...


@dataclass(frozen=True)
class Suite:
    id: str
    items: Tuple[Item, ...]
    seeds: Tuple[int, ...]

    def refs(self) -> FrozenSet[str]:
        return frozenset(item.id for item in self.items)


def check_distinct(items: Sequence[Item]) -> None:
    """ValueError when an item id or an item's state occurs more than once: the statistics count paired items as
    independent observations, and a repeated one would tighten a bound on nothing new."""
    ids, states = set(), set()
    for item in items:
        if item.id in ids:
            raise ValueError(f"item {item.id} occurs more than once (listed or generated twice)")
        if item.state in states:
            raise ValueError(f"the state of item {item.id} occurs more than once (listed or generated twice)")
        ids.add(item.id)
        states.add(item.state)


def pooled_suite(templates: Sequence[Template], seeds: Iterable[int], *, frozen_seeds: Iterable[int] = (),
                 minimum: int = MIN_POOLED) -> Suite:
    """A gate suite pooled from ``templates`` at ``seeds``: ValueError at a frozen seed, a repeated item or state,
    or fewer than ``minimum`` items."""
    seeds = tuple(int(seed) for seed in seeds)
    frozen = set(seeds) & set(frozen_seeds)
    if frozen:
        raise ValueError(f"seed(s) {', '.join(map(str, sorted(frozen)))} are frozen: gate suites use new seeds")
    items: List[Item] = [item for template in templates for seed in seeds for item in template.generate(seed)]
    check_distinct(items)
    if len(items) < minimum:
        raise ValueError(f"a pooled suite holds at least {minimum} items, not {len(items)}")
    identity = _io.digest({"templates": [template.name for template in templates], "seeds": list(seeds),
                           "items": [item.id for item in items]})
    return Suite(id=f"suite-{identity}", items=tuple(items), seeds=seeds)


# -- the gates ---------------------------------------------------------------------------------------------------------

CalibrationRefit = Callable[[str], Optional[Sequence[str]]]


def refit_calibration(refit: CalibrationRefit, state: str) -> Tuple[str, ...]:
    """The calibration record ids ``refit`` fit for the learned state ``state``, or () when it could not requalify
    (then the candidate does not serve). A hook that answers something other than a sequence of ids is a bug."""
    records = refit(state)
    if records is None:
        return ()
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("a calibration refit answers a sequence of record ids, or None")
    records = tuple(records)
    if not records or not all(isinstance(record, str) and record.strip() for record in records):
        return ()
    return records


@dataclass(frozen=True)
class Evidence:
    """What the gates judge a candidate on, measured by the caller over one suite.

    ``reference``, ``base``, ``candidate`` and ``control``: per suite item, whether the serving revision, the base,
    the candidate and its shuffled control were right. ``canaries_base`` and ``canaries_candidate``: each planted
    canary's exposure under the base (never the serving revision, which could ratchet it up) and the candidate.
    ``served`` and ``offline``: the candidate's outputs as served and as computed offline.
    ``budget``: the recorded budget check of the candidate's state. ``calibration`` and ``parent_calibration``: the
    candidate's refit calibration record ids and its parent's. ``state``: the learned state."""

    reference: Sequence[bool]
    base: Sequence[bool]
    candidate: Sequence[bool]
    control: Sequence[bool]
    canaries_base: Mapping[str, float]
    canaries_candidate: Mapping[str, float]
    served: Any
    offline: Any
    metrics: Mapping[str, float]
    budget: Optional[BudgetCheck]
    calibration: Tuple[str, ...]
    parent_calibration: Tuple[str, ...]
    state: Mapping[str, Any]


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str = ""
    awaiting: bool = False


@dataclass(frozen=True)
class GateReport:
    closure: str
    suite: str
    checks: Tuple[Check, ...]

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def failed(self) -> Tuple[str, ...]:
        return tuple(check.name for check in self.checks if not check.passed)

    @property
    def awaiting(self) -> bool:
        return not self.passed and bool(self.checks) and all(c.awaiting for c in self.checks if not c.passed)

    def to_dict(self) -> Dict[str, Any]:
        return {"closure": self.closure, "suite": self.suite, "passed": self.passed, "awaiting": self.awaiting,
                "checks": [{"name": check.name, "passed": check.passed, "detail": check.detail,
                            **({"awaiting": True} if check.awaiting else {})}
                           for check in self.checks]}


def _finite(value: Any) -> bool:
    try:
        return bool(np.all(np.isfinite(np.asarray(value, dtype=np.float64))))
    except (TypeError, ValueError):
        return False


def evaluate(closure: str, suite: Suite, evidence: Evidence, *, sets: SetLedger, bar: float = DECLINE_BAR,
             canary_tolerance: float = CANARY_TOLERANCE, served_tolerance: float = SERVED_TOLERANCE) -> GateReport:
    """Every gate on ``closure`` over ``suite`` (recorded as a selection use of the suite). ValueError, before any
    use is recorded, when an item or its state is repeated or fewer than ``MIN_POOLED`` distinct items remain."""
    check_distinct(suite.items)
    size = len(suite.items)
    if size < MIN_POOLED:
        raise ValueError(f"a gate suite pools at least {MIN_POOLED} items, not {size}")
    for name in ("reference", "base", "candidate", "control"):
        if len(getattr(evidence, name)) != size:
            raise ValueError(f"{name} results cover every suite item ({size}), not {len(getattr(evidence, name))}")
    unregistered = suite.refs() - sets.refs(suite.id)
    if unregistered:
        raise SetError(f"suite {suite.id}: {len(unregistered)} of its items are not registered under its id, so "
                       "they would not be protected from training")
    sets.use(suite.id, "selection", revision=closure)
    checks: List[Check] = []

    held, details = True, []
    for against, results in (("serving", evidence.reference), ("base", evidence.base)):
        n, losses, gains = paired(results, evidence.candidate)
        bound = decline_bound(n, losses=losses, gains=gains)
        held = held and bound <= bar
        details.append(f"against the {against}: decline {(losses - gains) / n:+.4f}, one-sided 95% bound "
                       f"{bound:.4f}")
    checks.append(Check("retention", held, f"{'; '.join(details)}; bar {bar:.4f} over {size} items"))

    n, control_right, wins = paired(evidence.control, evidence.candidate)
    lower = gain_lower_bound(n, wins=wins, losses=control_right)
    checks.append(Check("specificity", lower > 0.0,
                        f"gain over the shuffled control {(wins - control_right) / n:+.4f}, lower bound {lower:+.4f}"))

    base, candidate = dict(evidence.canaries_base), dict(evidence.canaries_candidate)
    exposed = sorted(name for name in base if name not in candidate or not _finite(candidate[name])
                     or not _finite(base[name]) or candidate[name] > base[name] + canary_tolerance)
    exposed += sorted(set(candidate) - set(base))
    checks.append(Check("canaries", bool(base) and not exposed,
                        "no canary above the base's exposure" if base and not exposed
                        else f"exposed above the base: {', '.join(exposed) or 'no canary planted'}"))

    refit = bool(evidence.calibration) and tuple(evidence.calibration) != tuple(evidence.parent_calibration)
    checks.append(Check("calibration", refit,
                        "refit for this revision" if refit else "not refit: thresholds never carry over"))

    budget = evidence.budget
    own = budget is not None and budget.state == _io.state_digest(evidence.state)
    checks.append(Check("budget", own and budget.ok,
                        "no budget check recorded" if budget is None
                        else "the budget check is of another state" if not own
                        else f"{budget.type} used {budget.used} of {budget.limit}"))

    finite = all(_finite(array) for array in evidence.state.values()) and all(
        _finite(value) for value in evidence.metrics.values())
    checks.append(Check("finite", finite, "every value finite" if finite else "a value is not finite"))

    served, offline = np.asarray(evidence.served, dtype=np.float64), np.asarray(evidence.offline, dtype=np.float64)
    agree = (served.shape == offline.shape and _finite(served) and _finite(offline)
             and bool(np.all(np.abs(served - offline) <= served_tolerance)))
    checks.append(Check("served_offline", agree, "served and offline agree" if agree else "served and offline differ"))
    return GateReport(closure=closure, suite=suite.id, checks=tuple(checks))


__all__ = ["CANARY_TOLERANCE", "CHECKS", "DECLINE_BAR", "MIN_POOLED", "ROLES", "SERVED_TOLERANCE", "USES", "Z95",
           "CalibrationRefit", "Check", "Evidence", "GateReport", "Item", "SetError", "SetLedger", "Suite", "Template",
           "check_distinct", "decline_bound", "evaluate", "gain_lower_bound", "paired", "pooled_suite",
           "refit_calibration", "rule_of_three", "two_proportion_bounds", "wilson"]
