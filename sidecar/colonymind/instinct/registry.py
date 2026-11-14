"""The registry of decision points: what a point is, and the checks it passes before anything may ask it.

A point is a short typed question with a closed set of labels and one effect from a closed enum (``EFFECTS``).
Its answer can only select among what its own code offers: nothing a model answers approves, grants, widens,
lifts a hold, resumes check-ins the owner stopped, or retracts or supersedes a stored claim. Registration refuses
a point whose model tier may settle such a label (``FORBIDDEN``, and ``EXCLUDED`` per point), so the rule holds in
the registry rather than in each caller.

The tiers, in order: ``rules`` (code, final where it speaks), ``fast`` and ``reader`` (model tiers that answer
probabilities), ``deliberate`` (a typed call on the mind's own model, answering a label) and the ``fallback`` (the
point's conservative answer). A deliberate tier that is the point's existing path (``on_path``: the typed call the
caller made before Instinct, moved unchanged) keeps the labels that call applied; every other model tier is bound
by the exclusions.
"""

from __future__ import annotations

import hashlib
import json
import re
import string
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Mapping, MutableMapping, Optional, Tuple

#: What a point's answer may do: pick a path (``route``), order what is already admitted (``rank``), only remove,
#: hold or delay (``filter``), put one plain question to the owner (``ask``), or label the owner's own words
#: (``read``). Closed: a point with any other effect is refused.
EFFECTS: Tuple[str, ...] = ("route", "rank", "filter", "ask", "read")
TYPES: Tuple[str, ...] = ("yes_no", "choice")
#: The tiers that answer probabilities.
MODEL_TIERS: Tuple[str, ...] = ("fast", "reader")
TIER_NAMES: Tuple[str, ...] = ("fast", "reader", "deliberate")
STATES: Tuple[str, ...] = ("shadow", "live", "demoted")
#: Labels no model tier may settle at any point: each would act for the owner.
FORBIDDEN: FrozenSet[str] = frozenset({"approve", "grant", "widen", "resume", "retract", "supersede", "replaces",
                                       "lift", "unhold", "allow"})
#: Per point, the labels no model tier may settle (a resume lifts the owner's own stop; ``first``, ``second`` and
#: ``neither`` retract stored statements; ``replaces`` supersedes a stored claim; ``approve`` would settle an ask;
#: ``send_now`` would send where the gate's prior held).
EXCLUDED: Dict[str, FrozenSet[str]] = {
    "owner_instruction": frozenset({"resume"}),
    "question_answer": frozenset({"first", "second", "neither"}),
    "claim_relation": frozenset({"replaces"}),
    "ask_answer": frozenset({"approve"}),
    "owner_send": frozenset({"send_now"}),
}
_LABEL = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
MAX_LABELS = 8
MAX_INSTRUCTIONS = 200


class RegistrationError(ValueError):
    """A point that may not be registered, and why."""


@dataclass(frozen=True)
class Tier:
    """One model tier of a point: the labels it may settle, its shipped state and its deadline (milliseconds; None
    is the provider's own time limit). ``on_path``: a deliberate tier that is the point's existing path.

    ``withholds``: labels a model tier reads at a calibrated bar but never settles, each one its point excludes
    (``ask_answer``'s ``approve``). A reading at the bar is handed to the caller as ``withheld``, which may only put
    a question to the owner about it; the fit calibrates these labels as it does the ones the tier settles."""

    name: str
    may_settle: Tuple[str, ...]
    state: str = "shadow"
    deadline_ms: Optional[int] = None
    on_path: bool = False
    withholds: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RulesAnswer:
    """The rules tier's reading: a label (or None), the distribution it came from (the owner gate's prior), and
    whether it speaks (a reading under its bar does not)."""

    label: Optional[str]
    probabilities: Mapping[str, float] = field(default_factory=dict)
    speaks: bool = True


Rules = Callable[[Mapping[str, Any]], Optional[RulesAnswer]]
Fallback = Callable[[Mapping[str, Any], Optional[RulesAnswer]], Optional[str]]
HighStakes = Callable[[Mapping[str, Any], Optional[str]], bool]


@dataclass(frozen=True)
class Point:
    """A decision point, as data.

    ``template`` writes the state a model tier reads from the caller's fields (whitespace folded, never cut);
    ``rules`` reads the fields before any model; ``fallback`` gives the conservative answer when nothing settled;
    ``floors`` are the least a label's calibrated probability must reach at any model tier (configuration and
    records may only raise them); ``literal``: a yes/no about a fact the state states (only such a point ships a
    fast tier live); ``turn_path``: the owner is waiting on it; ``high_stakes(fields, label)``: a case that
    escalates whatever the confidence; ``budget_per_hour``: escalations beyond the existing path, every reader ask
    (live or shadow) among them, allowed per hour; ``caller_rules``: the reader the caller runs first, asking only where it leaves
    the case open."""

    name: str
    type: str
    instructions: str
    labels: Mapping[str, str]
    template: str
    effect: str
    tiers: Tuple[Tier, ...] = ()
    rules: Optional[Rules] = None
    fallback: Optional[Fallback] = None
    floors: Mapping[str, float] = field(default_factory=dict)
    literal: bool = False
    turn_path: bool = False
    high_stakes: Optional[HighStakes] = None
    budget_per_hour: int = 20
    caller_rules: str = ""

    @property
    def version(self) -> str:
        """A hash of what a model reads (type, instructions, labels with their descriptions, template). The tiers
        are left out: a record is keyed by its tier already, and adding a tier must not void a measured record."""
        payload = json.dumps({"type": self.type, "instructions": self.instructions, "labels": dict(self.labels),
                              "template": self.template}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def fields(self) -> Tuple[str, ...]:
        return tuple(name for _, name, _, _ in string.Formatter().parse(self.template) if name)

    def tier(self, name: str) -> Optional[Tier]:
        return next((tier for tier in self.tiers if tier.name == name), None)

    def render(self, fields: Mapping[str, Any]) -> str:
        """The state as a model tier reads it: the template over the fields, each folded to one line."""
        return self.template.format(**{name: " ".join(str(fields.get(name) or "").split())
                                       for name in self.fields})

    def question(self) -> Dict[str, Any]:
        """The typed question, provider-neutral: type, instructions and labels with their descriptions."""
        return {"type": self.type, "instructions": self.instructions, "labels": dict(self.labels)}

    def excluded(self) -> FrozenSet[str]:
        return FORBIDDEN | EXCLUDED.get(self.name, frozenset())


def _check(point: Point) -> None:
    if not isinstance(point.name, str) or not _NAME.match(point.name):
        raise RegistrationError(f"point name {point.name!r} must be lower-case letters, digits and underscores")
    where = f"point {point.name}"
    if point.effect not in EFFECTS:
        raise RegistrationError(f"{where}: effect {point.effect!r} is not one of {', '.join(EFFECTS)}")
    if point.type not in TYPES:
        raise RegistrationError(f"{where}: type {point.type!r} is not one of {', '.join(TYPES)}")
    labels = dict(point.labels or {})
    if point.type == "yes_no" and set(labels) != {"yes", "no"}:
        raise RegistrationError(f"{where}: a yes_no point's labels are yes and no")
    if not 2 <= len(labels) <= MAX_LABELS:
        raise RegistrationError(f"{where}: a point has 2 to {MAX_LABELS} labels")
    for label, description in labels.items():
        if not isinstance(label, str) or not _LABEL.match(label):
            raise RegistrationError(f"{where}: label {label!r} must be lower-case letters, digits and underscores")
        if not isinstance(description, str) or not description.strip():
            raise RegistrationError(f"{where}: label {label} needs a description")
    if not isinstance(point.instructions, str) or not 0 < len(point.instructions.strip()) <= MAX_INSTRUCTIONS:
        raise RegistrationError(f"{where}: instructions of 1 to {MAX_INSTRUCTIONS} characters")
    try:
        point.fields
    except ValueError as error:
        raise RegistrationError(f"{where}: the template does not parse ({error})") from None
    seen = set()
    for tier in point.tiers:
        if tier.name not in TIER_NAMES:
            raise RegistrationError(f"{where}: tier {tier.name!r} is not one of {', '.join(TIER_NAMES)}")
        if tier.name in seen:
            raise RegistrationError(f"{where}: tier {tier.name} is registered twice")
        seen.add(tier.name)
        if tier.state not in STATES:
            raise RegistrationError(f"{where}: tier {tier.name}'s state {tier.state!r} is not one of "
                                    f"{', '.join(STATES)}")
        unknown = set(tier.may_settle) - set(labels)
        if unknown:
            raise RegistrationError(f"{where}: tier {tier.name} may settle an unknown label ({', '.join(unknown)})")
        if tier.on_path and tier.name != "deliberate":
            raise RegistrationError(f"{where}: only a deliberate tier can be the existing path")
        withheld = set(tier.withholds)
        if withheld and tier.name not in MODEL_TIERS:
            raise RegistrationError(f"{where}: only a model tier withholds a label")
        if withheld - set(labels):
            raise RegistrationError(f"{where}: tier {tier.name} withholds an unknown label "
                                    f"({', '.join(sorted(withheld - set(labels)))})")
        if withheld - point.excluded() or withheld & set(tier.may_settle):
            raise RegistrationError(f"{where}: tier {tier.name} withholds only labels its point excludes and it "
                                    "may not settle")
        if not tier.on_path:
            barred = set(tier.may_settle) & point.excluded()
            if barred:
                raise RegistrationError(f"{where}: tier {tier.name} may settle an excluded label "
                                        f"({', '.join(sorted(barred))}); no model tier may act for the owner")
        if tier.name == "fast" and tier.state == "live" and not point.literal:
            raise RegistrationError(f"{where}: a fast tier ships live only on a literal yes/no point")
        if tier.deadline_ms is not None and (isinstance(tier.deadline_ms, bool) or not isinstance(
                tier.deadline_ms, int) or tier.deadline_ms <= 0):
            raise RegistrationError(f"{where}: tier {tier.name}'s deadline is a positive number of milliseconds")
    for label, floor in dict(point.floors or {}).items():
        if label not in labels or isinstance(floor, bool) or not isinstance(floor, (int, float)) \
                or not 0.0 <= float(floor) <= 1.0:
            raise RegistrationError(f"{where}: floor {label!r} must name a label and be from 0 to 1")
    for name in ("rules", "fallback", "high_stakes"):
        value = getattr(point, name)
        if value is not None and not callable(value):
            raise RegistrationError(f"{where}: {name} must be callable")
    if isinstance(point.budget_per_hour, bool) or not isinstance(point.budget_per_hour, int) \
            or point.budget_per_hour < 0:
        raise RegistrationError(f"{where}: budget_per_hour is a whole number, 0 or more")


REGISTRY: Dict[str, Point] = {}


def register(point: Point, *, registry: Optional[MutableMapping[str, Point]] = None) -> Point:
    """Check ``point`` and add it to ``registry`` (the process's by default); RegistrationError when it fails a
    check or its name is taken."""
    table = REGISTRY if registry is None else registry
    _check(point)
    if point.name in table:
        raise RegistrationError(f"point {point.name} is already registered")
    table[point.name] = point
    return point


__all__ = ["EFFECTS", "EXCLUDED", "FORBIDDEN", "MODEL_TIERS", "REGISTRY", "STATES", "TIER_NAMES", "TYPES", "Point",
           "RegistrationError", "RulesAnswer", "Tier", "register"]
