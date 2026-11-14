"""Budgets: how far a learner may move from its original, declared per learner, checked by the engine, recorded
and enforced here.

The engine measures a learned state against its learner's budget (it knows the weights; the core does not) and hands
the result to the core as a ``BudgetCheck``. The core records it by the state's digest and refuses what does not
fit: a check for a learner with no declared budget, of another budget type, or against another limit than the one
declared is refused outright; a check whose measure is over the limit or not finite is recorded and fails the gate
(``gates.py``). A recorded check counts only while the declaration it was made against stands: once the budget is
changed, the state needs a new check. The budget types are the engine's: an absolute envelope (the parity target), a
fraction of each site's own norm, factor-norm bounds, an operator-norm bound from the identity, and a bound on the
output where the anchor is a constant or one-hot.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

from . import _io

BUDGET_TYPES: Tuple[str, ...] = ("absolute", "relative_dense", "low_rank", "operator_norm", "anchored_output", "ladder")


class BudgetError(ValueError):
    """A budget declaration or check the core refuses, and why."""


@dataclass(frozen=True)
class Budget:
    """A learner's declared budget: its type and its limit (in the type's own unit)."""

    learner: str
    type: str
    limit: float


@dataclass(frozen=True)
class BudgetCheck:
    """The engine's check of one learned state (``state``, its digest) against its learner's budget."""

    learner: str
    state: str
    type: str
    limit: float
    used: float

    @property
    def ok(self) -> bool:
        return (isinstance(self.used, (int, float)) and not isinstance(self.used, bool) and math.isfinite(self.used)
                and self.used <= self.limit)


def _finite_positive(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


class BudgetLedger:
    """Declared budgets and recorded checks, in ``directory``."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def _declared_path(self, learner: str) -> Path:
        return self.directory / learner / "declared.json"

    def _check_path(self, learner: str, state: str) -> Path:
        return self.directory / learner / "checks" / f"{state}.json"

    def declare(self, budget: Budget) -> Budget:
        if not isinstance(budget.learner, str) or not budget.learner.strip():
            raise BudgetError("a budget names its learner")
        if budget.type not in BUDGET_TYPES:
            raise BudgetError(f"budget type {budget.type!r} is not one of {', '.join(BUDGET_TYPES)}")
        if not _finite_positive(budget.limit):
            raise BudgetError(f"learner {budget.learner}: a budget's limit is a positive finite number")
        _io.write_json(self._declared_path(budget.learner), asdict(budget))
        return budget

    def declared(self, learner: str) -> Optional[Budget]:
        try:
            return Budget(**_io.read_json(self._declared_path(learner)))
        except FileNotFoundError:
            return None

    def record(self, check: BudgetCheck) -> BudgetCheck:
        declared = self.declared(check.learner)
        if declared is None:
            raise BudgetError(f"learner {check.learner} has no declared budget")
        if check.type != declared.type:
            raise BudgetError(f"learner {check.learner}: its budget type is {declared.type}, not {check.type}")
        if check.limit != declared.limit:
            raise BudgetError(f"learner {check.learner}: checked against limit {check.limit}, declared "
                              f"{declared.limit}")
        record = asdict(check)
        used = record["used"]
        record["used"] = used if isinstance(used, (int, float)) and math.isfinite(used) else repr(used)
        _io.write_json(self._check_path(check.learner, check.state), record)
        return check

    def result(self, learner: str, state: str) -> Optional[BudgetCheck]:
        """The recorded check of ``state``, or None when there is none or it was made against another declaration
        than the current one: a changed budget (a tightened limit, another type) needs a new check, since a check
        against the old limit says nothing about the new one."""
        try:
            record = _io.read_json(self._check_path(learner, state))
        except FileNotFoundError:
            return None
        if isinstance(record["used"], str):
            record["used"] = float(record["used"])
        check = BudgetCheck(**record)
        declared = self.declared(learner)
        if declared is None or check.type != declared.type or check.limit != declared.limit:
            return None
        return check

    def forget(self, learner: str, state: str) -> bool:
        return _io.remove(self._check_path(learner, state))


__all__ = ["BUDGET_TYPES", "Budget", "BudgetCheck", "BudgetError", "BudgetLedger"]
