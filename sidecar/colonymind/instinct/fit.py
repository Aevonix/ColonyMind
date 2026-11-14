"""The refit (docs/DECISIONS.md, Calibration and learning): per point, model tier and fingerprint, a new calibration
record from labelled answers, written only where the rules below allow. Nothing here trains a model: it fits how a
model's probabilities are read.

**Labels.** An example is one model tier's raw answer and the label an outcome or a set gave it: ``explicit`` (the
owner's own word or act, ``outcomes``), ``synthetic`` (a labelled set imported with ``colonymind instinct fit
--sets``) or ``behavioural`` (what the owner did, read by a resolver). The deliberate tier's answers and a hosted
service's answers are never labels. Behavioural labels only tighten: they can raise a bar, never lower one or shape
a map.

**The fit.** From ``ISOTONIC_FROM`` explicit or synthetic labels (my choice), an isotonic map per label the tier
reads at a bar (``read_labels``: the ones it may settle and the ones it withholds, ``Tier.withholds``; a yes/no has one
map, for yes); below that a temperature (least negative log-likelihood); below ``MIN_LABELS`` (my choice) no fit. The
bars, per label the tier reads at a bar: the most accurate composition with the fallback (the model's label at its
bar, else the existing path's answer; a withheld label counts as the reading it is, since the owner is asked about
it), ties to the bar that acts least, "never" included, never below the point's code floor. The accuracy a record carries is five-fold held out: the whole fit is redone on four
folds and read on the fifth, the variants of one item kept in one fold.

**The rules** (``refit``). A candidate that would act where the record in force does not, or act otherwise (a first
record included), loosens: it is written only when its held-out accuracy with the fallback is at least the existing
path's and the current record's on the same rows, and the frozen holdout (imported frozen items, never fitted on) still
passes; with no frozen holdout nothing loosens, and a state the frozen holdout holds is never fitted on. A candidate is
a tightening only when it is proved to settle nothing the current record does not settle the same way, over every raw
answer and not only the rows at hand (``contained``): a new temperature or map moves where the bars bite, so higher bars
alone prove nothing. Tightenings must also match the current record's accuracy. A refit that stops all action needs
at least the current record's label count; otherwise the record is kept and the tier is demoted. For a live tier,
a bar may move at most one BARS step per night; larger changes are skipped. With no record in force, one that
never acts is not written: no record already holds the point on its existing path. Every candidate also reads planted canaries: items at a raw
probability of zero for a label, planted with that label, which an honest fit never settles; when it settles every
canary of a label, the fit has seen what it must not (a leak), an audit is opened and no record is written. A map
settles nothing beyond the raw probabilities it was fitted on (it reads a safe bound there), so honest labels bunched at
one end never pass the canaries by extrapolation. Each record written is a new one in the history; the previous one can
be restored.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import calibration, records
from .calibration import NEVER, Record

logger = logging.getLogger(__name__)

FOLDS = 5
SEED = 7
#: From this many explicit or synthetic labels a record is isotonic, below it a temperature (a chosen value).
ISOTONIC_FROM = 50
#: Fewer labels than this fit nothing (my choice: at least two per fold).
MIN_LABELS = 10
#: The least answers a label's isotonic map is fitted on (my choice); a label with fewer has no map and never settles.
MIN_MAP_PAIRS = 5
TEMPERATURES = (0.25, 0.33, 0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0)
BARS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99, NEVER)
SOURCES = ("explicit", "behavioural", "synthetic")
LABELLING = frozenset({"explicit", "synthetic"})
SUBSETS = ("fit", "frozen", "canary")
#: Planted canaries per label (at raw probabilities 0, 0.01 and 0.02 of it).
CANARY_PROBABILITIES = (0.0, 0.01, 0.02)
#: What the existing path answers where it gives no label, at the points whose existing path asks nothing: the
#: caller keeps what its own reader decided (a reply linked with no class is engagement; no opt-out; the reminder
#: kept; the interest left open; the verdict stands). So a live tier's settled answer there has its comparison.
EFFECTIVE: Dict[str, str] = {"outreach_reply": "engaged", "opt_out": "no", "no_reminders": "no",
                             "interest_settled": "no", "owner_verdict": "yes"}
#: The outcome under which the existing path's answer is logged when it was asked beside a live model tier that
#: settled, only to compare (never applied): the comparison drift is measured against where that path asks a model.
COMPARED = "compared"


@dataclass(frozen=True)
class Example:
    """One model tier's raw answer with its label: ``gold`` (the right label, when known) or ``wrong`` (a label
    known wrong, when only that is known), the existing path's answer (``existing``: None when unknown), where the
    label came from (``source``), the subset (``fit``, ``frozen`` never fitted on, ``canary``), tags (``repository``,
    ``novel``, ``adversarial``), the keyed hash of the state and a group kept in one fold."""

    key: str
    point: str
    version: str
    tier: str
    kind: str
    model: str
    revision: str
    raw: Mapping[str, float]
    gold: Optional[str] = None
    wrong: Optional[str] = None
    existing: Optional[str] = None
    source: str = "synthetic"
    subset: str = "fit"
    tags: Tuple[str, ...] = ()
    state_hash: Optional[str] = None
    group: str = ""

    @property
    def fingerprint(self) -> str:
        return calibration.fingerprint(self.version, self.tier, self.kind, self.model, self.revision)


@dataclass
class Result:
    """What a refit did: ``write`` (the record is in force), ``skip`` or ``audit``; why, whether the candidate
    loosened or tightened, and the evidence (counts and accuracies on the same rows)."""

    action: str
    reason: str
    record: Optional[Record] = None
    direction: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)


# -- arithmetic ------------------------------------------------------------------------------------------------------

def usable(point: Any, example: Example) -> bool:
    raw = example.raw
    if not isinstance(raw, Mapping) or set(raw) != set(point.labels):
        return False
    return all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
               and 0.0 <= float(value) <= 1.0 for value in raw.values())


def ece(pairs: Sequence[Tuple[float, bool]], bins: int = 10) -> Optional[float]:
    """Expected calibration error over equal-width bins of a confidence."""
    if not pairs:
        return None
    total = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        chunk = [(c, ok) for c, ok in pairs if (low < c <= high) or (index == 0 and c == 0)]
        if chunk:
            total += len(chunk) * abs(sum(c for c, _ in chunk) / len(chunk) - sum(ok for _, ok in chunk) / len(chunk))
    return total / len(pairs)


def isotonic(pairs: Iterable[Tuple[float, bool]]) -> Tuple[Tuple[float, float], ...]:
    """Pool-adjacent-violators over (raw probability, right): ``((the block's highest probability, its rate), ...)``,
    non-decreasing."""
    tied: Dict[float, List[float]] = {}
    for probability, ok in pairs:
        entry = tied.setdefault(float(probability), [0.0, 0.0])
        entry[0] += float(ok)
        entry[1] += 1.0
    blocks: List[List[float]] = []
    for probability in sorted(tied):
        blocks.append([probability, *tied[probability]])
        while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] >= blocks[-1][1] / blocks[-1][2]:
            last = blocks.pop()
            blocks[-1] = [last[0], blocks[-1][1] + last[1], blocks[-1][2] + last[2]]
    return tuple((round(high, 6), round(right / count, 6)) for high, right, count in blocks)


def _map_pairs(point: Any, examples: Sequence[Example], label: str) -> List[Tuple[float, bool]]:
    out = []
    for example in examples:
        if example.gold is not None:
            out.append((float(example.raw[label]), example.gold == label))
        elif example.wrong == label:
            out.append((float(example.raw[label]), False))
    return out


def _nll(point: Any, examples: Sequence[Example], temperature: float) -> float:
    total = 0.0
    for example in examples:
        read = calibration.calibrated(_shape(point, "", method="temperature", temperature=temperature), point.type,
                                      example.raw)
        total -= math.log(max(float(read[example.gold]), 1e-9))
    return total / len(examples)


def _shape(point: Any, tier: str, **fields: Any) -> Record:
    return Record(point=point.name, version=point.version, tier=tier, provider_kind="", **fields)


def read_labels(tier: Any) -> Tuple[str, ...]:
    """The labels a model tier reads at a calibrated bar: the ones it may settle, then the ones it withholds
    (``Tier.withholds``: read at the bar and only ever asked about). The fit calibrates and bars each of them."""
    return (*tuple(tier.may_settle), *tuple(getattr(tier, "withholds", ()) or ()))


def _calibration(point: Any, tier: Any, labelled: Sequence[Example]) -> Dict[str, Any]:
    """The method and its parameters fitted on ``labelled`` (explicit or synthetic, gold known)."""
    if len(labelled) >= ISOTONIC_FROM:
        labels = ("yes",) if point.type == "yes_no" else read_labels(tier)
        maps, support = {}, {}
        for label in labels:
            pairs = _map_pairs(point, labelled, label)
            if len(pairs) >= MIN_MAP_PAIRS:
                maps[label] = isotonic(pairs)
                # The raw probabilities the map saw: beyond them it reads only a safe bound (calibration).
                support[label] = (min(float(p) for p, _ in pairs), max(float(p) for p, _ in pairs))
        if point.type != "yes_no" or "yes" in maps:
            return {"method": "isotonic", "maps": maps, "temperature": 1.0, "support": support}
    temperature = min(TEMPERATURES, key=lambda t: (_nll(point, labelled, t), abs(math.log(t))))
    return {"method": "temperature", "maps": {}, "temperature": temperature, "support": {}}


def _reader(point: Any, tier: Any, record: Record) -> Callable[[Mapping[str, float]], Optional[str]]:
    """How the runner would read a raw answer under ``record``: the label it settles or withholds, or None."""
    allowed = set(read_labels(tier))

    def bar(label: str) -> float:
        return max(float(point.floors.get(label, 0.0)), float(record.bars.get(label, NEVER)))

    def read(raw: Mapping[str, float]) -> Optional[str]:
        settled, _, _ = calibration.reading(point.type, calibration.calibrated(record, point.type, raw), bar)
        return settled if settled in allowed else None
    return read


def _scored(examples: Iterable[Example]) -> List[Example]:
    return [example for example in examples if example.gold is not None and example.existing is not None]


def accuracy(point: Any, tier: Any, record: Optional[Record], examples: Sequence[Example]) -> Dict[str, Any]:
    """With the fallback, on the rows with a label and a known existing answer: ``{n, right, acts, accuracy}``;
    ``record`` None is the existing path alone."""
    rows = _scored(examples)
    read = _reader(point, tier, record) if record is not None else (lambda raw: None)
    right = acts = 0
    for example in rows:
        settled = read(example.raw)
        acts += settled is not None
        right += (settled if settled is not None else example.existing) == example.gold
    return {"n": len(rows), "right": right, "acts": acts, "accuracy": right / len(rows) if rows else None}


def _bars(point: Any, tier: Any, record: Record, rows: Sequence[Example]) -> Dict[str, float]:
    """Per label the tier reads at a bar (``read_labels``), the most accurate bar with the fallback over ``rows``
    (two passes, one label at a time), ties to the bar that acts least; never below the label's code floor."""
    bars = {label: NEVER for label in read_labels(tier)}
    scored = _scored(rows)
    if not scored:
        return bars
    reads = [(calibration.calibrated(record, point.type, example.raw), example) for example in scored]

    def score(candidate: Mapping[str, float]) -> int:
        def bar(label: str) -> float:
            return float(candidate.get(label, NEVER))
        right = 0
        for probabilities, example in reads:
            settled, _, _ = calibration.reading(point.type, probabilities, bar)
            answer = settled if settled in bars else example.existing
            right += answer == example.gold
        return right

    for _ in range(2):
        for label in read_labels(tier):
            floor = float(point.floors.get(label, 0.0))
            best, best_score = bars[label], score(bars)
            for value in reversed(BARS):                 # from "never" down: a tie keeps the bar that acts least
                if value < floor:
                    break
                trial = {**bars, label: value}
                trial_score = score(trial)
                if trial_score > best_score:
                    best, best_score = value, trial_score
            bars[label] = best
    return bars


def _fitted(point: Any, tier: Any, examples: Sequence[Example], *, key: Example) -> Optional[Record]:
    """A record fitted on every example given (no held-out accuracy)."""
    usable_rows = [example for example in examples if usable(point, example) and example.subset == "fit"]
    labelled = [example for example in usable_rows if example.source in LABELLING and example.gold is not None]
    if len(labelled) < MIN_LABELS:
        return None
    shape = _calibration(point, tier, labelled)
    base = _shape(point, tier.name, method=shape["method"], temperature=shape["temperature"], maps=shape["maps"],
                  support=shape["support"])
    bars = _bars(point, tier, base, [example for example in usable_rows if example.source in LABELLING])
    behavioural = _bars(point, tier, base, usable_rows)
    # Behavioural labels only tighten: per label the higher of the two bars.
    bars = {label: max(bars[label], behavioural[label]) for label in bars}
    return Record(point=point.name, version=point.version, tier=tier.name, provider_kind=key.kind, model=key.model,
                  revision=key.revision, method=shape["method"], temperature=shape["temperature"], bars=bars,
                  maps=shape["maps"], n=len(labelled), support=shape["support"])


def folds(examples: Sequence[Example], count: int = FOLDS, seed: int = SEED) -> List[List[Example]]:
    """Folds stratified by label, a group (``group``, else the key) never split."""
    groups: Dict[str, List[Example]] = defaultdict(list)
    for example in examples:
        groups[example.group or example.key].append(example)
    ordered = sorted(groups.items(), key=lambda item: (str(item[1][0].gold), item[0]))
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=lambda item: str(item[1][0].gold))
    out: List[List[Example]] = [[] for _ in range(count)]
    for index, (_, members) in enumerate(ordered):
        out[index % count].extend(members)
    return out


def _tier(point: Any, name: str) -> Any:
    tier = point.tier(name)
    if tier is None or name not in {"fast", "reader"}:
        raise ValueError(f"{point.name} has no model tier {name!r}")
    return tier


def candidate(point: Any, tier: str, examples: Sequence[Example]) -> Optional[Record]:
    """The record fitted on ``examples`` (one fingerprint's), carrying its five-fold held-out accuracy with the
    fallback on the labelled rows; None with too few labels."""
    spec = _tier(point, tier)
    rows = [example for example in examples if usable(point, example) and example.subset == "fit"]
    if not rows:
        return None
    key = rows[0]
    made = _fitted(point, spec, rows, key=key)
    if made is None:
        return None
    right = total = 0
    parts = folds(rows)
    for index, held in enumerate(parts):
        train = [example for other, part in enumerate(parts) if other != index for example in part]
        fold = _fitted(point, spec, train, key=key)
        scored = [example for example in held if example.source in LABELLING]
        if fold is None:
            measured = accuracy(point, spec, None, scored)
        else:
            measured = accuracy(point, spec, fold, scored)
        right += measured["right"]
        total += measured["n"]
    return dataclasses.replace(made, accuracy=round(right / total, 6) if total else None)


# -- canaries ------------------------------------------------------------------------------------------------------

def canaries(point: Any, tier: str, kind: str, model: str, revision: str) -> List[Example]:
    """Planted items no honest fit settles: per label the tier reads at a bar (``read_labels``), raw answers that
    give that label no probability at all, planted with that label."""
    spec = _tier(point, tier)
    labels = ("yes", "no") if point.type == "yes_no" else read_labels(spec)
    out = []
    for label in labels:
        if label not in read_labels(spec):
            continue
        others = [name for name in point.labels if name != label]
        for index, low in enumerate(CANARY_PROBABILITIES):
            raw = {name: (1.0 - low) / len(others) for name in others}
            raw[label] = low
            out.append(Example(key=f"canary:{point.name}:{label}:{index}", point=point.name, version=point.version,
                               tier=tier, kind=kind, model=model, revision=revision, raw=raw, gold=label,
                               source="synthetic", subset="canary"))
    return out


def canary_leak(point: Any, tier: str, record: Record, planted: Sequence[Example]) -> Optional[str]:
    """The label whose every canary ``record`` settles as planted, or None. A map reads only a safe bound beyond
    the raw probabilities it was fitted on, so an honest fit that never saw answers where the canaries sit cannot
    settle them by carrying its nearest step outward: settling them means the fit saw them (exposure), not that it
    extrapolated."""
    read = _reader(point, _tier(point, tier), record)
    by_label: Dict[str, List[bool]] = defaultdict(list)
    for example in planted:
        by_label[str(example.gold)].append(read(example.raw) == example.gold)
    return next((label for label, passed in sorted(by_label.items()) if passed and all(passed)), None)


# -- the rules ---------------------------------------------------------------------------------------------------

#: How far either side of a computed switching point a raw probability is read again (floating-point slack).
_SLACK = 1e-9


def _switches(point: Any, record: Record) -> List[float]:
    """The raw probabilities of yes at which a yes/no record's reading can change: where its calibrated yes crosses
    the yes bar or the no bar's complement, the ends of its map's steps, and the clamp of a temperature."""
    def bar(label: str) -> float:
        return max(float(point.floors.get(label, 0.0)), float(record.bars.get(label, NEVER)))
    crossings = [value for value in (bar("yes"), 1.0 - bar("no")) if 0.0 < value < 1.0]
    out: List[float] = []
    if record.method == "isotonic":
        for steps in record.maps.values():
            out.extend(float(upper) for upper, _ in steps)
        for low, high in dict(record.support or {}).values():
            out.extend((float(low), float(high)))
    elif record.temperature == 1.0:
        out.extend(crossings)
    else:
        epsilon = calibration._EPSILON
        out.extend((epsilon, 1.0 - epsilon))
        for value in crossings:
            logit = record.temperature * math.log(value / (1.0 - value))
            out.append(1.0 / (1.0 + math.exp(-logit)) if logit > -700 else 0.0)
    return out


def contained(point: Any, tier: Any, new: Record, current: Record) -> bool:
    """Whether every raw answer ``new`` settles, ``current`` settles the same way: containment of what each record
    decides, proved, not sampled from the rows at hand.

    A yes/no reads one number, the raw probability of yes, through a non-decreasing map, so each record's answer is
    constant between its switching points (``_switches``): both are read at every switching point, just either side
    of it and midway between neighbours, which covers every raw answer. A choice is proved only when its reading is
    the same (method, temperature, maps) and no bar fell: then the top label and its probability are the same, and a
    higher bar settles a subset. Any other change of a choice's calibration is not proved, and counts as acting
    more."""
    if point.type != "yes_no":
        if (new.method, new.temperature, dict(new.maps), dict(new.support)) != \
                (current.method, current.temperature, dict(current.maps), dict(current.support)):
            return False
        return all(max(float(point.floors.get(label, 0.0)), float(new.bars.get(label, NEVER)))
                   >= max(float(point.floors.get(label, 0.0)), float(current.bars.get(label, NEVER)))
                   for label in read_labels(tier))
    points = {0.0, 1.0}
    for value in [*_switches(point, new), *_switches(point, current)]:
        points.update(min(1.0, max(0.0, value + shift)) for shift in (-_SLACK, 0.0, _SLACK))
    ordered = sorted(points)
    probes = ordered + [(low + high) / 2.0 for low, high in zip(ordered, ordered[1:])]
    read_new, read_current = _reader(point, tier, new), _reader(point, tier, current)
    for p in probes:
        raw = {"yes": p, "no": 1.0 - p}
        settled = read_new(raw)
        if settled is not None and read_current(raw) != settled:
            return False
    return True


def _loosens(point: Any, tier: Any, new: Record, current: Optional[Record], rows: Sequence[Example]) -> bool:
    """Whether ``new`` acts where ``current`` does not, or acts otherwise: anything not proved contained in what
    the current record decides (``contained``: a new temperature or map changes where bars bite, so higher bars
    alone prove nothing), or a row it settles that the current record leaves or settles otherwise. No current
    record: anything it settles."""
    if current is None:
        return any(value < NEVER for value in new.bars.values())
    if not contained(point, tier, new, current):
        return True
    read_new, read_current = _reader(point, tier, new), _reader(point, tier, current)
    for example in rows:
        settled = read_new(example.raw)
        if settled is not None and read_current(example.raw) != settled:
            return True
    return False


def frozen_passes(point: Any, tier: str, record: Record, frozen: Sequence[Example]) -> Tuple[bool, Dict[str, Any]]:
    """The frozen holdout passes when, with the fallback, ``record`` is at least as accurate as the existing path
    on every frozen row and on the adversarial ones, and reads none of the adversarial rows as a label the tier
    withholds that is not their right one (``false_withheld``). No frozen rows: it cannot pass."""
    spec = _tier(point, tier)
    rows = [example for example in frozen if usable(point, example)]
    evidence: Dict[str, Any] = {}
    if not _scored(rows):
        return False, {"frozen": 0}
    ok = True
    for name, subset in (("all", rows), ("adversarial", [e for e in rows if "adversarial" in e.tags])):
        if not _scored(subset):
            continue
        with_record, existing = accuracy(point, spec, record, subset), accuracy(point, spec, None, subset)
        evidence[name] = {"n": with_record["n"], "record": with_record["accuracy"], "existing": existing["accuracy"]}
        ok = ok and with_record["right"] >= existing["right"]
    withheld = false_withheld(point, spec, record, rows)
    if withheld:
        evidence["withheld"] = withheld
        ok = ok and not any(entry["false"] for entry in withheld.values())
    return ok, evidence


def false_withheld(point: Any, tier: Any, record: Record, frozen: Sequence[Example]) -> Dict[str, Dict[str, int]]:
    """Per label ``tier`` withholds (``Tier.withholds``: a label that only ever puts a question to the owner, such as
    ``ask_answer``'s ``approve``), over the adversarial frozen rows whose right label is another one (someone else's
    yes, a quoted or injected one): how many there are (``n``) and how many ``record`` reads as it (``false``)."""
    read = _reader(point, tier, record)
    pool = [example for example in frozen if "adversarial" in example.tags and example.gold is not None
            and usable(point, example)]
    out: Dict[str, Dict[str, int]] = {}
    for label in getattr(tier, "withholds", ()) or ():
        others = [example for example in pool if example.gold != label]
        out[label] = {"n": len(others), "false": sum(read(example.raw) == label for example in others)}
    return out


def in_force(store: Any, point: Any, tier: str, key: Example) -> Optional[Record]:
    """The record in force for ``key``'s fingerprint: the instance's own, else the shipped one."""
    local = store.records() if store is not None else []
    found = dict(version=point.version, tier=tier, provider_kind=key.kind, model=key.model, revision=key.revision)
    return calibration.find(local, **found) or calibration.find(records.shipped({point.name: point}), **found)


_LOOKUP: Any = object()


def _never_acts(point: Any, tier: Any, record: Record) -> bool:
    """Whether the record reads no label anywhere, including maps that cannot reach their bars."""
    if point.type == "yes_no":
        never = dataclasses.replace(record, bars={label: NEVER for label in point.labels})
        return contained(point, tier, record, never)
    read = _reader(point, tier, record)
    return all(read({name: float(name == label) for name in point.labels}) is None for label in point.labels)


def _bar_step_exceeded(tier: Any, new: Record, current: Record) -> bool:
    """Bars off the grid may move only as far as the nearest BARS value in each direction."""
    for label in read_labels(tier):
        before = float(current.bars.get(label, NEVER))
        after = float(new.bars.get(label, NEVER))
        low = next((bar for bar in reversed(BARS) if bar < before), before)
        high = next((bar for bar in BARS if bar > before), before)
        if not low <= after <= high:
            return True
    return False


def refit(store: Any, point: Any, tier: str, examples: Sequence[Example], *, frozen: Sequence[Example] = (),
          current: Any = _LOOKUP, planted: Optional[Sequence[Example]] = None, now: datetime,
          source: str = "nightly fit") -> Result:
    """One fingerprint's refit under the rules (module docstring); writes the record or the audit to ``store``."""
    from . import state as point_state

    spec = _tier(point, tier)
    # A state the frozen holdout holds is never fitted on, whatever brought it in (a logged decision, a set).
    held_out = {example.state_hash for example in frozen if example.state_hash}
    rows = [example for example in examples if usable(point, example) and example.subset == "fit"
            and not (example.state_hash and example.state_hash in held_out)]
    if not rows:
        return Result("skip", "no labelled answers")
    key = rows[0]
    if current is _LOOKUP:
        current = in_force(store, point, tier, key)
    made = candidate(point, tier, rows)
    if made is None:
        return Result("skip", f"too few labels (fewer than {MIN_LABELS} explicit or synthetic)")
    made = dataclasses.replace(made, source=f"{source}: {made.n} labels, {made.method}",
                               fitted_at=now.isoformat(), parent_revision=getattr(current, "parent_revision", ""))
    planted = canaries(point, tier, key.kind, key.model, key.revision) if planted is None else planted
    leak = canary_leak(point, tier, made, planted)
    if leak is not None:
        evidence = {"label": leak, "canaries": len(planted), "labels": made.n, "fingerprint": key.fingerprint}
        reason = f"the fit settles every planted canary of {leak}: it has seen what it must not; no record written"
        if store is not None:
            store.open_audit(point.name, tier, key.fingerprint, reason, evidence, at=now)
        return Result("audit", reason, None, "", evidence)
    labelled = [example for example in rows if example.source in LABELLING]
    same = {"existing": accuracy(point, spec, None, labelled)["accuracy"], "candidate": made.accuracy,
            "current": accuracy(point, spec, current, labelled)["accuracy"] if current is not None else None,
            "labels": made.n, "fingerprint": key.fingerprint}
    loosens = _loosens(point, spec, made, current, [*rows, *frozen])
    direction = "loosen" if loosens else "tighten"
    if current is not None:
        same.update(current_labels=current.n, current_held_out=current.accuracy)
        if not _never_acts(point, spec, current) and _never_acts(point, spec, made) and made.n < current.n:
            reason = "it would never act with fewer labels than the current record; the current record is kept"
            if store is not None:
                point_state.demote(store, point.name, tier, reason=reason, now=now, by=source, evidence=same,
                                   registry={point.name: point})
            logger.warning("instinct: %s %s refit skipped (%s)", point.name, tier, reason)
            return Result("skip", reason, made, direction, same)
    if loosens:
        held = made.accuracy
        if held is None or same["existing"] is None or held < same["existing"]:
            return Result("skip", "it would loosen, and its held-out accuracy is below the existing path's", made,
                          "loosen", same)
    if current is not None:
        baselines = [value for value in (same["current"], current.accuracy) if value is not None]
        if made.accuracy is None or any(made.accuracy < value for value in baselines):
            return Result("skip", f"it would {direction}, and its held-out accuracy is below the current record's",
                          made, direction, same)
    if loosens:
        passed, frozen_evidence = frozen_passes(point, tier, made, frozen)
        same["frozen"] = frozen_evidence
        if not passed:
            reason = ("no frozen holdout: nothing loosens without one" if not frozen_evidence.get("all")
                      else "it would loosen, and the frozen holdout fails")
            return Result("skip", reason, made, "loosen", same)
    elif current is None:
        # No record holds the point on its existing path; one that never acts would make its answers unsure, and
        # an unsure answer escalates.
        return Result("skip", "it would never act, and no record is in force: none is needed", made, "", same)
    elif _same(made, current):
        return Result("skip", "unchanged", made, "", same)
    if current is not None:
        if point_state.effective(point, tier, store.states() if store is not None else {}) == "live" \
                and _bar_step_exceeded(spec, made, current):
            return Result("skip", "a live tier's bar would move more than one BARS step per night", made,
                          direction, same)
    if store is not None:
        store.put_record(made, reason=f"{source}: {direction}")
    return Result("write", direction, made, direction, same)


# -- the log's labelled answers -------------------------------------------------------------------------------------

def existing_answer(row: Mapping[str, Any]) -> Optional[str]:
    """What the existing path answered at a logged decision: the rules', the deliberate tier's or the fallback's
    label; where a live model tier settled, the existing path's answer asked beside it (``COMPARED``); else, at a
    point whose existing path asks nothing, what the caller keeps (``EFFECTIVE``). None: nothing says what it would
    have answered (a live tier settled where the existing path asks a model, and it was not asked)."""
    if row.get("answered_by") in {"rules", "deliberate", "fallback"} and row.get("label"):
        return str(row["label"])
    compared = (row.get("tiers") or {}).get("deliberate") or {}
    if compared.get("outcome") == COMPARED and compared.get("label"):
        return str(compared["label"])
    return EFFECTIVE.get(str(row.get("point") or ""))


def strongest(outcomes: Iterable[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    """Per decision its strongest outcome (``log.precedence``): the owner's own label, then any other explicit
    outcome, then synthetic, then behavioural; within each, the latest."""
    from .log import precedence
    best: Dict[str, Mapping[str, Any]] = {}
    for outcome in outcomes:
        current = best.get(outcome["decision"])
        if current is None or precedence(outcome) < precedence(current):
            best[outcome["decision"]] = outcome
    return best


def examples_from_log(store: Any, *, registry: Optional[Mapping[str, Any]] = None,
                      since: Optional[datetime] = None, sources: Any = None) -> List[Example]:
    """The model tiers' raw answers at logged decisions that have an outcome, labelled by it. Only an outcome of a
    labelling source is a label (``SOURCES``; never an ``interpreted`` one): a decision with none gives nothing,
    whatever any tier answered. The strongest outcome whose grading sources remain sound supplies the label;
    erased or withheld grading turns never train the refit. A row logged under another version of
    its point, or whose provider identity the log keeps only as a hash, is left out."""
    from . import REGISTRY, outcomes
    table = REGISTRY if registry is None else registry
    sources = sources if sources is not None else outcomes.Sources(store.path.parent)
    graded: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for outcome in store.outcomes():
        graded[outcome["decision"]].append(outcome)
    out: List[Example] = []
    for row in store.rows(since=since):
        outcome = strongest(item for item in graded[row["id"]] if outcomes.sound(item, row, sources)).get(row["id"])
        point = table.get(row["point"])
        if outcome is None or point is None or row["version"] != point.version or outcome["source"] not in SOURCES:
            continue                          # an interpreted outcome (a model's reading of the owner) is no label
        gold, wrong = outcome.get("label"), None
        if gold is None and outcome.get("verdict") == "wrong":
            wrong = outcome.get("judged")
            if point.type == "yes_no" and wrong in {"yes", "no"}:
                gold, wrong = ("no" if wrong == "yes" else "yes"), None
        if gold is None and wrong is None:
            continue
        for tier in ("fast", "reader"):
            entry = (row.get("tiers") or {}).get(tier) or {}
            raw, fingerprint = entry.get("raw"), entry.get("fingerprint")
            kind, model, revision = (str(entry.get(name) or "") for name in ("kind", "model", "revision"))
            if not raw or not fingerprint or calibration.fingerprint(row["version"], tier, kind, model,
                                                                      revision) != fingerprint:
                continue
            out.append(Example(key=row["id"], group=row["id"], point=row["point"], version=row["version"], tier=tier,
                               kind=kind, model=model, revision=revision, raw=dict(raw), gold=gold, wrong=wrong,
                               existing=existing_answer(row), source=str(outcome["source"]),
                               state_hash=row.get("state_hash")))
    return out


# -- the tighten rule ------------------------------------------------------------------------------------------------

#: Explicit wrongs at a point within the window that tighten its model tiers at once (a chosen value).
TIGHTEN_WRONGS = 2
TIGHTEN_WINDOW = timedelta(days=7)
TIGHTEN_STEP = 0.01


def record_by_fingerprint(store: Any, fingerprint: str, *,
                          registry: Optional[Mapping[str, Any]] = None) -> Optional[Record]:
    local = store.records() if store is not None else []
    shipped = records.load() if registry is None else records.shipped(registry)
    return next((record for record in [*local, *shipped] if record.fingerprint == fingerprint), None)


def tighten(store: Any, point_name: str, *, now: datetime,
            registry: Optional[Mapping[str, Any]] = None) -> List[Result]:
    """The tighten rule, at once: where a model tier settled a label, or read at its bar a label it withholds (a
    withheld yes the caller acted on: asked the owner back, or approved on), that ``TIGHTEN_WRONGS`` explicit
    outcomes of the last ``TIGHTEN_WINDOW`` show wrong (live or in shadow), the record it read them under gets that
    label's bar raised just above the highest of those probabilities, as a new record (the previous one can be
    restored). A live tier so tightened is demoted (``state.demote_tightened``)."""
    from . import REGISTRY
    from . import state as point_state
    table = REGISTRY if registry is None else registry
    point = table.get(point_name)
    if point is None or store is None:
        return []
    wrongs: Dict[Tuple[str, str], List[Tuple[str, float, str]]] = defaultdict(list)
    since = (now - TIGHTEN_WINDOW).isoformat()
    # Each decision's strongest outcome only: an explicit wrong the owner then corrected no longer counts.
    for outcome in strongest(store.outcomes(point=point_name)).values():
        if outcome["source"] != "explicit" or str(outcome["at"]) < since:
            continue
        row = store.row(outcome["decision"])
        if row is None:
            continue
        for tier in ("fast", "reader"):
            entry = (row.get("tiers") or {}).get(tier) or {}
            said = point_state.acted(point, tier, entry)       # settled, or withheld at its bar
            if not entry.get("fingerprint") or said is None:
                continue
            if (outcome.get("label") is not None and said != outcome["label"]) or \
                    (outcome.get("label") is None and outcome["verdict"] == "wrong" and outcome.get("judged") == said):
                wrongs[(tier, entry["fingerprint"])].append((said, float(entry.get("probability") or 0.0),
                                                             row["id"]))
    results = []
    for (tier, fingerprint), found in sorted(wrongs.items()):
        if len(found) < TIGHTEN_WRONGS:
            continue
        current = record_by_fingerprint(store, fingerprint, registry=table)
        if current is None:
            continue
        bars = dict(current.bars)
        for label, probability, _ in found:
            above = math.floor(probability / TIGHTEN_STEP + 1e-9) * TIGHTEN_STEP + TIGHTEN_STEP
            bars[label] = min(NEVER, max(float(bars.get(label, NEVER)), round(above, 6)))
        evidence = {"wrongs": len(found), "decisions": [ident for _, _, ident in found], "window_days":
                    TIGHTEN_WINDOW.days, "fingerprint": fingerprint}
        if bars != dict(current.bars):
            made = dataclasses.replace(current, bars=bars, fitted_at=now.isoformat(),
                                       source=f"tightened: {len(found)} explicit wrongs in {TIGHTEN_WINDOW.days} days")
            store.put_record(made, reason=f"tightened: {len(found)} explicit wrongs in {TIGHTEN_WINDOW.days} days")
            results.append(Result("write", "tightened", made, "tighten", evidence))
            evidence = {**evidence, "record": made.id}
        point_state.demote_tightened(store, point, tier, now=now, evidence=evidence, registry=table)
    return results


# -- labelled sets ---------------------------------------------------------------------------------------------------

SET_FORMAT = "colonymind-instinct-sets/1"


class SetError(ValueError):
    """A labelled set that cannot be imported, and why."""


def import_sets(store: Any, document: Mapping[str, Any], *, now: datetime,
                registry: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Keep a labelled set's answers for the refit (``fit``) and the frozen holdout (``frozen``), by reference.

    ``document``: ``{"format": SET_FORMAT, "set": name, "hash": the set's hash, "tier": "fast" | "reader",
    "provider": {"kind", "model", "revision"}, "flip_rates": {point: rate}, "items": [{"key", "point", "raw",
    "gold", "existing", "subset", "tags", "version", "fields", "group"}]}``; items that share a ``group`` (the
    renamed or reordered variants of one item) are kept in one fold. The provider must be a local kind a tier runs
    (a hosted service's answers are never labels); its ``kind`` is the calibration identity (``kind`` or
    ``kind/readout``), kept whole so the records fitted from the set apply only under it. An item's ``fields`` are rendered through its point's template
    only to keep a keyed hash of the state, then dropped: no text is kept. An item asked under another version of
    its point, with labels or probabilities the point does not have, or whose key the other subset already holds,
    is left out and counted."""
    from . import REGISTRY, settings
    table = REGISTRY if registry is None else registry
    if not isinstance(document, Mapping) or document.get("format") != SET_FORMAT:
        raise SetError(f"not a labelled set ({SET_FORMAT})")
    provider = document.get("provider") if isinstance(document.get("provider"), Mapping) else {}
    kind = str(provider.get("kind") or "")   # the calibration identity: a kind, or kind/readout
    if kind.split("/", 1)[0] not in settings.KINDS or provider.get("private") is False:
        raise SetError(f"answers of a {kind or 'unnamed'} provider are never labels: only a local tier's "
                       f"({', '.join(settings.KINDS)})")
    tier = str(document.get("tier") or "")
    if tier not in {"fast", "reader"}:
        raise SetError("tier must be fast or reader")
    name = str(document.get("set") or "").strip()
    if not name:
        raise SetError("the set needs a name")
    model, revision = str(provider.get("model") or ""), str(provider.get("revision") or "")
    rates = {str(point): float(rate) for point, rate in dict(document.get("flip_rates") or {}).items()
             if isinstance(rate, (int, float)) and not isinstance(rate, bool) and 0.0 <= float(rate) <= 1.0}
    held: Dict[Tuple[str, str], str] = {(example.key, example.fingerprint): example.subset
                                        for example in store.examples()}
    kept: List[Example] = []
    skipped: Dict[str, int] = defaultdict(int)
    for item in document.get("items") or []:
        point = table.get(str((item or {}).get("point") or ""))
        if point is None or point.tier(tier) is None or point.tier(tier).on_path:
            skipped["point"] += 1
            continue
        if item.get("version") not in (None, point.version):
            skipped["version"] += 1
            continue
        subset = str(item.get("subset") or "fit")
        gold = item.get("gold")
        if subset not in {"fit", "frozen"} or gold not in point.labels:
            skipped["label"] += 1
            continue
        existing = item.get("existing")
        state_hash = None
        if isinstance(item.get("fields"), Mapping):
            state_hash = store.state_hash(point.render(item["fields"]))
        example = Example(key=str(item.get("key") or ""), point=point.name, version=point.version, tier=tier,
                          kind=kind, model=model, revision=revision, raw=dict(item.get("raw") or {}), gold=gold,
                          existing=existing if existing in point.labels else None, source="synthetic",
                          subset=subset, tags=tuple(str(tag) for tag in item.get("tags") or ()),
                          state_hash=state_hash, group=str(item.get("group") or item.get("key") or ""))
        if not example.key or not usable(point, example):
            skipped["answer"] += 1
            continue
        if held.get((example.key, example.fingerprint), subset) != subset:
            skipped["subset"] += 1            # a frozen item is never fitted on, and a fitted one never frozen
            continue
        held[(example.key, example.fingerprint)] = subset
        kept.append(example)
    count = store.put_examples(kept, set_name=name, set_hash=str(document.get("hash") or ""), flip_rates=rates,
                               at=now)
    return {"set": name, "kept": count, "skipped": dict(skipped)}


# -- the nightly pass --------------------------------------------------------------------------------------------------

def refit_all(store: Any, *, now: datetime, registry: Optional[Mapping[str, Any]] = None,
              sources: Any = None) -> List[Dict[str, Any]]:
    """A refit of every fingerprint with labelled answers (the log's and the imported sets' ``fit`` items), each
    against its own frozen items."""
    from . import REGISTRY
    table = REGISTRY if registry is None else registry
    groups: Dict[Tuple[str, str, str], List[Example]] = defaultdict(list)
    for example in [*examples_from_log(store, registry=table, sources=sources), *store.examples(subset="fit")]:
        groups[(example.point, example.tier, example.fingerprint)].append(example)
    out = []
    for (name, tier, fingerprint), members in sorted(groups.items()):
        point = table.get(name)
        spec = point.tier(tier) if point is not None else None
        if spec is None or spec.on_path or members[0].version != point.version:
            continue
        frozen = store.examples(fingerprint=fingerprint, subset="frozen", point=name)
        try:
            result = refit(store, point, tier, members, frozen=frozen, now=now)
        except Exception as error:            # one group's failure never stops the others
            logger.warning("instinct refit of %s %s failed (%s)", name, tier, type(error).__name__)
            continue
        out.append({"point": name, "tier": tier, "fingerprint": fingerprint, "action": result.action,
                    "reason": result.reason, "record": result.record.id if result.record is not None else None})
    return out


def nightly(state_dir: Any, *, now: datetime, registry: Optional[Mapping[str, Any]] = None,
            sources: Any = None) -> Dict[str, Any]:
    """The nightly pass: forget the decisions whose source was erased, resolve outcomes (explicit wrongs tighten at
    once), refit every fingerprint with labels under the rules, demote on drift or a fingerprint change, and name
    the live tiers whose settled answers had nothing to compare them with (``unmeasured``)."""
    from . import outcomes
    from . import state as point_state
    from .log import DB_FILE, Store
    path = Path(state_dir) / DB_FILE
    if not path.exists():
        return {"skipped": "no decision log"}
    store = Store(path)
    previous = store.transitions(limit=1)
    after_seq = previous[0]["seq"] if previous else 0
    sources = sources if sources is not None else outcomes.Sources(state_dir)
    summary: Dict[str, Any] = {"forgotten": outcomes.forget_erased(store, sources)}
    summary["resolved"] = {point: dict(counts) for point, counts in
                           outcomes.resolve(store, sources, now=now).items()}
    summary["refits"] = refit_all(store, now=now, registry=registry, sources=sources)
    summary["demoted"] = point_state.check(store, now=now, registry=registry)
    summary["carried"] = [change for change in store.transitions(after_seq=after_seq, limit=None)
                          if change["source"].startswith(point_state.CARRIED)]
    summary["unmeasured"] = point_state.unmeasured(store, registry=registry)
    return summary


def _same(new: Record, current: Record) -> bool:
    return (new.method, dict(new.bars), dict(new.maps), new.temperature, dict(new.support)) == \
        (current.method, dict(current.bars), dict(current.maps), current.temperature, dict(current.support))


__all__ = ["BARS", "CANARY_PROBABILITIES", "COMPARED", "EFFECTIVE", "FOLDS", "ISOTONIC_FROM", "LABELLING",
           "MIN_LABELS", "MIN_MAP_PAIRS", "SOURCES", "SUBSETS", "TIGHTEN_WINDOW", "TIGHTEN_WRONGS", "Example", "Result",
           "accuracy", "canaries", "canary_leak", "candidate", "contained", "ece", "examples_from_log",
           "existing_answer", "false_withheld", "folds",
           "frozen_passes", "import_sets", "in_force", "isotonic", "nightly", "read_labels", "record_by_fingerprint",
           "refit", "refit_all", "strongest", "tighten", "usable", "SET_FORMAT", "SetError"]
