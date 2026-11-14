"""The objective registry: what each learner is trained toward, checked before anything may train on it.

An objective names the model it trains, its learner kind, its target type, a loss per declared teacher (each
teacher keeps its own term; verified outcomes, proxies and model targets are never pooled into one label) and a
declared blend of those terms. The blend is part of the objective's identity, so a changed weight makes a new
revision. Registration refuses, as Instinct's registry refuses a point:

- a teacher that is not in the closed list, or that the objective's kind does not declare;
- a target type the teacher cannot give;
- a loss the teacher's term does not take;
- a blend over teachers the objective does not declare, or a declared teacher without a nonnegative finite weight;
- a model in the protected set (the caller names it: identity models, the owner gate, secrets detection, a live
  model other roles key against).

Safety and consent points learn tighten-only from live data (``SafetyPoint``): toward holding, flagging or honouring
a stop. ``admission.py`` applies the rule to every row at a declared safety point, whatever objective it feeds.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, FrozenSet, Iterable, Mapping, MutableMapping, Optional, Tuple

from . import _io
from .teachers import KINDS, TARGET_TYPES, TEACHERS, canonical_kind

_NAME = re.compile(r"^[a-z][a-z0-9_]{0,47}$")


class RegistrationError(ValueError):
    """An objective that may not be registered, and why."""


@dataclass(frozen=True)
class SafetyPoint:
    """A decision point that learns tighten-only from live data.

    ``labels`` are the point's labels; ``tightening`` the ones that hold, flag or honour a stop.
    ``attacker_reachable``: its input can be written by someone other than the owner (a tool's result, a message
    from others), so even the owner's false-flag marks never become loosening rows there."""

    point: str
    labels: FrozenSet[str]
    tightening: FrozenSet[str]
    attacker_reachable: bool = False

    def __post_init__(self) -> None:
        if not self.tightening or not set(self.tightening) < set(self.labels):
            raise ValueError(f"safety point {self.point}: its tightening labels are some, not all, of its labels")

    def tightens(self, label: object, *, push_away: bool) -> bool:
        """Whether a row with ``label`` moves the point only toward its tightening labels. A pull tightens when its
        label does; a push away from ``label`` tightens only when every other label tightens, since the mass it
        removes goes to all of them."""
        if not isinstance(label, str) or label not in self.labels:
            return False
        if push_away:
            return set(self.labels) - {label} <= set(self.tightening)
        return label in self.tightening


def _safety(point: str, labels: Iterable[str], tightening: Iterable[str], attacker: bool = False) -> SafetyPoint:
    return SafetyPoint(point, frozenset(labels), frozenset(tightening), attacker)


#: The shipped safety and consent points and what tightens each. A screened result flagged (``yes``); a message
#: from others held unanswered (``no``, my reading of holding for a filter whose fallback answers); an unprompted
#: message held for the digest or dropped; a stop honoured; the owner's stop or pause kept.
SAFETY_POINTS: Mapping[str, SafetyPoint] = MappingProxyType({point.point: point for point in (
    _safety("tool_screen", ("yes", "no"), ("yes",), attacker=True),
    _safety("inbound_triage", ("yes", "no"), ("no",), attacker=True),
    _safety("owner_send", ("send_now", "digest", "drop"), ("digest", "drop")),
    _safety("opt_out", ("yes", "no"), ("yes",)),
    _safety("no_reminders", ("yes", "no"), ("yes",)),
    _safety("owner_instruction", ("stop", "pause_today", "resume", "none"), ("stop", "pause_today")),
)})


@dataclass(frozen=True)
class Objective:
    """What a learner trains toward.

    ``losses`` maps each declared teacher to the loss of its own term, ``blend`` each to its weight. ``labels``: the
    closed label set of a ``label`` target (empty: the points' own labels). ``points``: the decision points whose
    rows it trains (empty: whatever its source yields). ``text``: its rows are derived from text, so they wait out
    the quarantine before they train. ``attacker_reachable``: its input can be written by someone other than the
    owner, so a model teacher may only push away."""

    name: str
    model: str
    kind: str
    target_type: str
    losses: Mapping[str, str]
    blend: Mapping[str, float]
    labels: Tuple[str, ...] = ()
    points: Tuple[str, ...] = ()
    text: bool = True
    attacker_reachable: bool = False
    what: str = field(default="", compare=False)

    @property
    def identity(self) -> str:
        """A digest of everything that decides what the objective trains, the blend included."""
        return _io.digest({"name": self.name, "model": self.model, "kind": self.kind, "target": self.target_type,
                           "losses": dict(self.losses), "blend": {name: float(weight) for name, weight in
                                                                 self.blend.items()},
                           "labels": list(self.labels), "points": list(self.points), "text": bool(self.text),
                           "attacker_reachable": bool(self.attacker_reachable)})

    def teachers(self) -> FrozenSet[str]:
        return frozenset(self.losses)


def check(objective: Objective, *, protected: FrozenSet[str] = frozenset()) -> None:
    """Every registration check; RegistrationError on the first that fails."""
    if not isinstance(objective.name, str) or not _NAME.match(objective.name):
        raise RegistrationError(f"objective name {objective.name!r} must be lower-case letters, digits and "
                                "underscores")
    where = f"objective {objective.name}"
    if not isinstance(objective.model, str) or not objective.model.strip():
        raise RegistrationError(f"{where}: it names the model it trains")
    if objective.model in protected:
        raise RegistrationError(f"{where}: model {objective.model} is in the protected set and never learns")
    if canonical_kind(objective.kind) not in KINDS:
        raise RegistrationError(f"{where}: kind {objective.kind!r} is not one of {', '.join(KINDS)}")
    if objective.target_type not in TARGET_TYPES:
        raise RegistrationError(f"{where}: target type {objective.target_type!r} is not one of "
                                f"{', '.join(TARGET_TYPES)}")
    losses = dict(objective.losses or {})
    if not losses:
        raise RegistrationError(f"{where}: it declares at least one teacher")
    for name, loss in losses.items():
        teacher = TEACHERS.get(name)
        if teacher is None:
            raise RegistrationError(f"{where}: unknown teacher {name!r}")
        if objective.target_type not in teacher.target_types:
            raise RegistrationError(f"{where}: teacher {name} gives no {objective.target_type} target (target type "
                                    f"mismatch)")
        if canonical_kind(objective.kind) not in teacher.kinds:
            raise RegistrationError(f"{where}: kind {objective.kind} does not declare teacher {name}")
        if objective.target_type not in teacher.targets_for(objective.kind):
            raise RegistrationError(f"{where}: teacher {name} target type {objective.target_type} does not fit "
                                    f"kind {objective.kind}")
        if loss not in teacher.losses:
            raise RegistrationError(f"{where}: teacher {name}'s term takes {', '.join(teacher.losses)}, not loss "
                                    f"{loss!r}")
    blend = dict(objective.blend or {})
    if set(blend) != set(losses):
        extra, missing = sorted(set(blend) - set(losses)), sorted(set(losses) - set(blend))
        raise RegistrationError(f"{where}: the blend covers exactly its declared teachers (over undeclared: "
                                f"{', '.join(extra) or 'none'}; unweighted: {', '.join(missing) or 'none'})")
    for name, weight in blend.items():
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) \
                or weight < 0:
            raise RegistrationError(f"{where}: teacher {name}'s blend weight must be a nonnegative finite number")
    if objective.target_type == "label" and objective.labels and len(set(objective.labels)) < 2:
        raise RegistrationError(f"{where}: a label objective names at least two labels, or none (the points' own)")
    for point in objective.points:
        if not isinstance(point, str) or not point:
            raise RegistrationError(f"{where}: points are named by their names")


REGISTRY: Dict[str, Objective] = {}


def register(objective: Objective, *, registry: Optional[MutableMapping[str, Objective]] = None,
             protected: FrozenSet[str] = frozenset()) -> Objective:
    """Check ``objective`` and add it to ``registry`` (the process's by default); RegistrationError when it fails a
    check or its name is taken."""
    table = REGISTRY if registry is None else registry
    check(objective, protected=protected)
    if objective.name in table:
        raise RegistrationError(f"objective {objective.name} is already registered")
    table[objective.name] = objective
    return objective


# -- the core objectives (every weight is an assumption; the design's illustrative blends) --------------------------

OUTCOME = register(Objective(
    name="outcome", model="decision_model", kind="supervised_refit", target_type="label",
    losses={"verified_outcome": "cross_entropy", "owner_label": "cross_entropy", "synthetic": "cross_entropy",
            "deliberation": "kl", "behavioural_proxy": "margin"},
    blend={"verified_outcome": 1.0, "owner_label": 1.0, "synthetic": 0.5, "deliberation": 0.5,
           "behavioural_proxy": 0.1},
    what="outcome learning on the fast decision model; the deep path teaches as its own term, never as proof"))

SENSORY_PREDICTION = register(Objective(
    name="sensory_prediction", model="perception_heads", kind="detached_heads", target_type="vector",
    losses={"next_observation": "normalised_squared_error"}, blend={"next_observation": 1.0}, text=False,
    what="detached heads over a frozen perception model predict its next observation"))

RECALL_USE = register(Objective(
    name="recall_use", model="query_adapter", kind="query_adapter", target_type="pairwise",
    losses={"owner_label": "pairwise", "verified_outcome": "pairwise", "synthetic": "pairwise",
            "stronger_model": "listwise_kl", "deliberation": "pairwise", "behavioural_proxy": "margin"},
    blend={"owner_label": 1.0, "verified_outcome": 1.0, "synthetic": 0.5, "stronger_model": 0.3,
           "deliberation": 0.3, "behavioural_proxy": 0.1},
    # Admission checks the provenance of each passage in the pairwise label.
    attacker_reachable=True,
    what="the query side of recall learns which memories are used"))

NEXT_VALUE = register(Objective(
    name="next_value", model="numeric_predictor", kind="online_predictor", target_type="value",
    losses={"realized_value": "normalised_squared_error"}, blend={"realized_value": 1.0}, text=False,
    what="a small online predictor learns a stream's next value"))


__all__ = ["NEXT_VALUE", "OUTCOME", "RECALL_USE", "REGISTRY", "SAFETY_POINTS", "SENSORY_PREDICTION", "Objective",
           "RegistrationError", "SafetyPoint", "check", "register", "teacher_lines"]


def teacher_lines(build) -> list[dict]:
    """Weighted loss terms and counts, with live proof eligibility kept separate from training."""
    return [{"teacher": name, "rows": build.teacher_rows[name], "loss": loss,
             "counts_as_proof": TEACHERS[name].counts_as_proof("live")}
            for name, loss in sorted(build.teacher_losses.items())]
