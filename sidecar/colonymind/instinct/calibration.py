"""Calibration records: how one model's raw probabilities become calibrated ones at one point, and the bars a
label must clear there.

A record is keyed by (point version, tier, provider kind, configured model, served revision) and applies only while
that key matches the running provider: another model, another checkpoint of it, or a point whose question changed
gets no record, and its model tier abstains (shadow logging goes on). The key's fingerprint is the recall cutoff's
configuration fingerprint, over these five fields.

Two methods. Temperature scaling: a probability p becomes p^(1/T), renormalised (for yes/no, the log-odds are
divided by T). Isotonic: a non-decreasing step map per label, fitted on labelled answers (``fit``), from the raw
probability of that label to how often the label was right at it; a yes/no has one map (yes), and no is its
complement; a choice's label with no map keeps its raw probability and has no bar, so it never settles. A map is
read only over the raw probabilities it was fitted on (``support``, per label): beyond them it has no evidence, so
each label reads its safe bound under a non-decreasing map, never the nearest step carried outward. Below the
lowest seen, the label reads 0 (and a yes/no's no reads the complement of the first step); above the highest, the
label reads the last step (and a yes/no's no reads 0). Nothing then settles on an answer the map never saw unless
the bound itself clears the bar. A yes/no
answer is yes at or above the yes bar, else no at or above the no bar; a choice's top label counts at or above its
bar. The bar in force is the highest of the point's code floor, the record's bar and a bar the operator configured:
configuration can only raise it.

A record's id is a hash of its content, so every change is a new record (``log.Store`` keeps each one, and the
previous one can be restored). A record may name the revision it was learned from (``parent_revision``): lineage
only, kept with the record and never matched, so a learned revision has no record until one is fitted for it.

Shipped measurements live in ``records/`` as data. Runtime, fitting and record history use that same source.
Each record requires a served revision and matches only its measured key; another checkpoint or readout abstains
until a record is fitted for it.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from colonymind.memory.recall import calibration_fingerprint

NEVER = 1.01
METHODS = ("temperature", "isotonic")
_EPSILON = 1e-6
_MEASURED_REVISION = "typed-decisions@1c5edc17a7acd8701df6fc341c0d179f1c62c982"
#: Revisions used only when carrying operator calibration from 1.11 settings. Shipped data requires a revision.
MEASURED_REVISIONS = (_MEASURED_REVISION, "")


Steps = Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class Record:
    """How one model's raw probabilities are read at one point: ``method`` with its ``temperature`` or its isotonic
    ``maps`` (per label, ``(upper raw probability, calibrated)`` steps), the ``bars``, how many labels it was fitted
    on (``n``), its held-out accuracy with the fallback, where it came from and when. ``parent_revision``: the
    revision this revision was learned from (lineage; never part of the match). ``support``: per mapped label, the
    lowest and highest raw probability its map was fitted on."""

    point: str
    version: str
    tier: str
    provider_kind: str
    model: str = ""
    revision: str = ""
    method: str = "temperature"
    temperature: float = 1.0
    bars: Mapping[str, float] = field(default_factory=dict)
    n: int = 0
    accuracy: Optional[float] = None
    source: str = ""
    fitted_at: str = ""
    maps: Mapping[str, Steps] = field(default_factory=dict)
    parent_revision: str = ""
    support: Mapping[str, Tuple[float, float]] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        """The match: the five fields a running provider is read under (never the parent)."""
        return fingerprint(self.version, self.tier, self.provider_kind, self.model, self.revision)

    @property
    def key(self) -> Dict[str, str]:
        """The calibration key: the five matched fields and the parent revision it admits."""
        return {"version": self.version, "tier": self.tier, "provider_kind": self.provider_kind,
                "model": self.model, "revision": self.revision, "parent_revision": self.parent_revision}

    @property
    def id(self) -> str:
        """A hash of the record's content: the same record has the same id, any change a new one."""
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["bars"] = dict(self.bars)
        value["maps"] = {label: [list(step) for step in steps] for label, steps in dict(self.maps).items()}
        value["support"] = {label: [float(low), float(high)] for label, (low, high) in dict(self.support).items()}
        if not value["support"]:
            value.pop("support")              # a record without one keeps the id it had before support existed
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Record":
        known = {name: value[name] for name in cls.__dataclass_fields__ if name in value}
        known["bars"] = {str(label): float(bar) for label, bar in dict(known.get("bars") or {}).items()}
        known["maps"] = {str(label): tuple((float(step[0]), float(step[1])) for step in steps)
                         for label, steps in dict(known.get("maps") or {}).items()}
        known["support"] = {str(label): (float(bounds[0]), float(bounds[1]))
                            for label, bounds in dict(known.get("support") or {}).items()}
        return cls(**known)


def fingerprint(version: str, tier: str, provider_kind: str, model: str, revision: str) -> str:
    return calibration_fingerprint({"point_version": version, "tier": tier, "provider_kind": provider_kind,
                                    "model": model, "revision": revision})


def find(records: Iterable[Record], *, version: str, tier: str, provider_kind: str, model: str,
         revision: str) -> Optional[Record]:
    """The record for exactly this key, or None."""
    wanted = fingerprint(version, tier, provider_kind, model, revision)
    return next((record for record in records if record.fingerprint == wanted), None)


def lineage(records: Sequence[Record], record: Record) -> List[Record]:
    """The records of ``record``'s ancestors, nearest first: for each parent revision, the record of the same point
    version, tier, provider kind and model at that revision. Lineage only: nothing reads it to match a provider."""
    out: List[Record] = []
    seen = {record.revision}
    current = record
    while current.parent_revision and current.parent_revision not in seen:
        seen.add(current.parent_revision)
        parent = find(records, version=current.version, tier=current.tier, provider_kind=current.provider_kind,
                      model=current.model, revision=current.parent_revision)
        if parent is None:
            break
        out.append(parent)
        current = parent
    return out


def calibrate(probabilities: Mapping[str, float], temperature: float) -> Dict[str, float]:
    """Temperature scaling over a distribution: p^(1/T), renormalised. T above 1 softens, below 1 sharpens."""
    if temperature == 1.0:
        return dict(probabilities)
    weights = {label: float(p) ** (1.0 / temperature) for label, p in probabilities.items()}
    total = sum(weights.values())
    if not total > 0:
        return dict(probabilities)
    return {label: weight / total for label, weight in weights.items()}


def calibrate_yes(p_yes: float, temperature: float) -> float:
    """Temperature scaling of one probability of yes: its log-odds divided by T."""
    if temperature == 1.0:
        return float(p_yes)
    p = min(max(float(p_yes), _EPSILON), 1 - _EPSILON)
    return 1.0 / (1.0 + math.exp(-math.log(p / (1 - p)) / temperature))


def step(steps: Steps, p: float) -> float:
    """An isotonic map read at ``p``: the value of the first step whose upper end is at or above it (above the last,
    the last value)."""
    for upper, value in steps:
        if p <= upper:
            return float(value)
    return float(steps[-1][1]) if steps else float(p)


def _mapped(record: Record, label: str, p: float) -> Tuple[float, int]:
    """A label's map read at ``p`` and where ``p`` lies: -1 below what the map was fitted on, 1 above, 0 within
    (or no support recorded)."""
    steps = record.maps[label]
    bounds = dict(record.support or {}).get(label)
    if bounds is not None and p < bounds[0]:
        return step(steps, bounds[0]), -1
    if bounds is not None and p > bounds[1]:
        return step(steps, bounds[1]), 1
    return step(steps, p), 0


def calibrated(record: Record, type: str, raw: Mapping[str, float]) -> Dict[str, float]:
    """The calibrated probabilities of a raw answer (module docstring). Beyond an isotonic map's support each label
    reads its safe bound, so a yes/no's two readings may sum to less than one there."""
    if record.method not in METHODS:
        raise ValueError(f"no calibration method {record.method!r}")
    if record.method == "isotonic":
        if type == "yes_no":
            if "yes" not in record.maps:
                raise ValueError("an isotonic yes/no record needs its yes map")
            value, side = _mapped(record, "yes", float(raw["yes"]))
            if side < 0:
                return {"yes": 0.0, "no": 1.0 - value}
            if side > 0:
                return {"yes": value, "no": 0.0}
            return {"yes": value, "no": 1.0 - value}
        out: Dict[str, float] = {}
        for label, p in raw.items():
            if label not in record.maps:
                out[label] = float(p)
                continue
            value, side = _mapped(record, label, float(p))
            out[label] = 0.0 if side < 0 else value
        return out
    if type == "yes_no":
        p_yes = calibrate_yes(raw["yes"], record.temperature)
        return {"yes": p_yes, "no": 1.0 - p_yes}
    return calibrate(raw, record.temperature)


def reading(type: str, probabilities: Mapping[str, float], bar: Any) -> Tuple[Optional[str], str, float]:
    """``(label that clears its bar or None, the candidate label, its probability)``: for a yes/no, yes at or above
    its bar, else no at or above its bar; for a choice, the top label at or above its bar."""
    if type == "yes_no":
        if probabilities["yes"] >= bar("yes"):
            return "yes", "yes", probabilities["yes"]
        if probabilities["no"] >= bar("no"):
            return "no", "no", probabilities["no"]
        top = "yes" if probabilities["yes"] >= probabilities["no"] else "no"
        return None, top, probabilities[top]
    top = max(probabilities, key=probabilities.get)
    return (top if probabilities[top] >= bar(top) else None), top, probabilities[top]


def band_bars(labels: Iterable[str], type: str, abstain: Tuple[float, float]) -> Dict[str, float]:
    """A 1.11 abstain band ``[lo, hi]`` as bars: a yes/no is yes at ``hi`` and no at ``1 - lo``; a choice's labels
    count at ``hi``."""
    low, high = float(abstain[0]), float(abstain[1])
    if type == "yes_no":
        return {"yes": high, "no": round(1.0 - low, 10)}
    return {label: high for label in labels}


__all__ = ["MEASURED_REVISIONS", "METHODS", "NEVER", "Record", "Steps", "band_bars", "calibrate",
           "calibrate_yes", "calibrated", "find", "fingerprint", "lineage", "reading", "step"]
