"""Training admission: what a period may learn from.

A row is admitted by its structure (its teacher, target, source, audience and time) and never by its words. Per row,
in order, a row is refused when:

1. the objective names its points and the row's point is not among them;
2. its teacher is not in the closed list, or the objective does not declare it;
3. its label does not fit the objective's target type, or, at a safety point, is not one of the point's labels (a
   soft distribution never trains a safety point);
4. its label came from an adversarial source kind (a contact's message, web or feed text, a tool's result): such
   streams are inputs, never labels, so they never label themselves;
5. a live row claims the synthetic teacher, or a synthetic row claims a live one;
6. it belongs to a protected set (a gate suite, a sealed evaluation set, a frozen holdout, a protected source); the
   protected refs may be given as a callable (``gates.SetLedger.protected_refs``), read at each admission, so a set
   registered later is refused too;
7. a push-only teacher (the behavioural proxy) claims to pull toward its label;
8. a model teacher pulls where the input is reachable by someone other than the owner (it may only push away there,
   or tighten a safety point);
9. at a safety point it loosens (anything but holding, flagging or honouring a stop) and its teacher is neither
   ``synthetic`` nor ``owner_label``; where the input is attacker-reachable (the point declares it, or the objective
   the row feeds does), only ``synthetic`` may loosen, so the owner's false-flag marks tune calibration and seed
   synthetic sets but never become loosening rows.

A text-derived row younger than ``QUARANTINE`` is not trainable yet (verdict ``quarantine``), so corrections,
erasures and screening verdicts land first. Across the period's remaining rows:

- repeats are witnesses, not rows: at most one row per state hash, per teacher, per period (the earliest); a state
  one teacher labelled two ways leaves the fit and is counted (``contradiction``);
- where a verified outcome or the owner's label and a model teacher disagree on one state, the outcome wins: the
  model's row leaves the fit and counts toward that teacher's measured error (``overruled``);
- no single contact, device or tool source supplies more than ``SOURCE_SHARE`` of the period's admitted rows; its
  earliest rows are kept, so a later flood displaces nothing. The owner's own rows, synthetic sets and the
  system's resolvers name no source (``source_of`` returns None) and are not capped.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple, Union

from . import _io
from .manifests import Entry, Manifest, RowError, check_row
from .objectives import SAFETY_POINTS, Objective, SafetyPoint
from .teachers import TEACHERS, canonical_kind

#: Text-derived rows become trainable after a day (D assumption).
QUARANTINE = timedelta(hours=24)
#: No single source supplies more than a tenth of a period's rows (D assumption).
SOURCE_SHARE = 0.10
#: The teachers whose rows may loosen a safety point, and the one that may where its input is attacker-reachable.
LOOSENING_TEACHERS: FrozenSet[str] = frozenset({"synthetic", "owner_label"})
ATTACKER_LOOSENING_TEACHERS: FrozenSet[str] = frozenset({"synthetic"})
#: Streams that are inputs, never labels.
ADVERSARIAL_SOURCE_KINDS: FrozenSet[str] = frozenset({"contact", "web", "feed", "tool_result"})
REASONS: Tuple[str, ...] = ("point_not_declared", "unknown_teacher", "undeclared_teacher", "target_mismatch",
                            "adversarial_self_label", "teacher_source_mismatch", "protected",
                            "push_only_teacher_pulls", "model_teacher_on_attacker_input", "loosening_from_live",
                            "loosening_on_attacker_input", "contradiction", "overruled", "source_cap",
                            "recall_provenance", "recall_adversarial_pull", "recall_unshown",
                            "recall_evaluation_overlap")

SourceOf = Callable[[Mapping[str, Any]], Optional[str]]
ProtectedRefs = Union[Iterable[str], Callable[[], Iterable[str]]]


@dataclass(frozen=True)
class Admitted:
    """A period's admission: the manifest (by reference) and the admitted rows, states in memory only."""

    manifest: Manifest
    rows: Tuple[Mapping[str, Any], ...]


def _order(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (_io.parse(row["at"]), row["ref"], row["teacher"])


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _fits(target: str, label: Any, labels: Tuple[str, ...]) -> bool:
    if target == "label":
        return isinstance(label, str) and bool(label) and (not labels or label in labels)
    if target == "binary":
        return isinstance(label, (bool, int)) and label in (0, 1)
    if target == "value":
        return _number(label)
    if target == "vector":
        return isinstance(label, (list, tuple)) and bool(label) and all(_number(value) for value in label)
    if target == "distribution":
        if isinstance(label, str):
            return bool(label) and (not labels or label in labels)
        if not isinstance(label, Mapping) or not label:
            return False
        values = list(label.values())
        return (all(_number(value) and 0.0 <= value <= 1.0 for value in values) and abs(sum(values) - 1.0) <= 1e-6
                and (not labels or set(label) <= set(labels)))
    if target == "pairwise":
        if isinstance(label, Mapping):
            label = (label.get("preferred"), label.get("disfavoured"))
        return (isinstance(label, (list, tuple)) and len(label) == 2 and all(isinstance(item, str) and item
                                                                             for item in label)
                and label[0] != label[1])
    return False


def _key(label: Any) -> bytes:
    return _io.canonical(label)


def _ranked(value: Any) -> bool:
    return isinstance(value, Mapping) and all(isinstance(ref, str) and ref and isinstance(rank, int)
                                             and not isinstance(rank, bool) and rank > 0
                                             for ref, rank in value.items())


def recall_refusal(row: Mapping[str, Any], evaluation_sets: Iterable[Mapping[str, Any]]) -> str:
    """Validate a retrieval target's evidence, retained inside the manifest's label.

    A push-away pair changes only its disfavoured passage, never rewards the preferred one.
    Provenance is supplied by the source collector, not by the teacher's generated text.
    """
    label = row["label"]
    if not isinstance(label, Mapping):
        return "recall_provenance"
    preferred, disfavoured = label.get("preferred"), label.get("disfavoured")
    sources, shown = label.get("passage_sources"), label.get("shown")
    if not isinstance(sources, Mapping) or not _ranked(shown) or any(
            not isinstance(sources.get(ref), str)
            or sources[ref] not in {"owner", "system", "synthetic", *ADVERSARIAL_SOURCE_KINDS}
            for ref in (preferred, disfavoured)):
        return "recall_provenance"
    name = row["teacher"]
    # Even a push row must not encode a reward for an adversarial preferred passage.
    if sources[preferred] in ADVERSARIAL_SOURCE_KINDS and name not in {"owner_label", "verified_outcome"}:
        return "recall_adversarial_pull"
    inside = preferred in shown and disfavoured in shown
    if name in {"deliberation", "behavioural_proxy"} and not inside:
        return "recall_unshown"
    if name == "stronger_model" and not inside:
        wider = label.get("reranked")
        # Twenty is the design's wider-reranker pool, not a tunable teacher assertion.
        if label.get("model_role") != "reranker" or not _ranked(wider) or any(
                ref not in wider or wider[ref] > 20 for ref in (preferred, disfavoured)):
            return "recall_unshown"
    if name == "synthetic":
        question, sessions = label.get("question_id"), label.get("haystack_sessions")
        if not isinstance(question, str) or not question or not isinstance(sessions, (list, tuple)) or not sessions or any(
                not isinstance(session, str) or not session for session in sessions):
            return "recall_provenance"
        for evaluation in evaluation_sets:
            if question == evaluation["question_id"] or set(sessions) & set(evaluation["haystack_sessions"]):
                return "recall_evaluation_overlap"
    return ""


def recall_weight(objective: Objective, row: Mapping[str, Any]) -> float:
    """Inverse preferred shown rank for citations and use signals (G1 weighting assumption)."""
    if canonical_kind(objective.kind) != "query-adapter" or row["teacher"] not in {"deliberation", "behavioural_proxy"}:
        return 1.0
    label = row["label"]
    return 1.0 / label["shown"][label["preferred"]]


class Admission:
    """Admits one objective's rows into a learner's periods (the module docstring)."""

    def __init__(self, objective: Objective, *, learner: str, source_of: SourceOf,
                 protected_refs: ProtectedRefs = frozenset(), protected_sources: Iterable[str] = frozenset(),
                 safety: Mapping[str, SafetyPoint] = SAFETY_POINTS, clock: Optional[Callable[[], datetime]] = None,
                 quarantine: timedelta = QUARANTINE, source_share: float = SOURCE_SHARE,
                 evaluation_sets: Iterable[Mapping[str, Any]] = ()):
        self._evaluation_sets = tuple({"question_id": item["question_id"],
                                       "haystack_sessions": tuple(item["haystack_sessions"])}
                                      for item in evaluation_sets)
        if not callable(source_of):
            raise TypeError("admission needs source_of: which contact, device or tool supplied a row")
        if not 0.0 < source_share <= 1.0:
            raise ValueError("the source share is above 0 and at most 1")
        self.objective = objective
        self.learner = learner
        self._source_of = source_of
        self._protected_refs = protected_refs if callable(protected_refs) else frozenset(protected_refs)
        self._protected_sources = frozenset(protected_sources)
        self._safety = dict(safety)
        if clock is None:
            from ..util.temporal import now_utc
            clock = now_utc
        self._clock = clock
        self._quarantine = quarantine
        self._share = float(source_share)

    # -- per row ---------------------------------------------------------------------------------------------------

    def protected(self) -> FrozenSet[str]:
        """The refs no period may learn from, as they stand now."""
        refs = self._protected_refs
        return frozenset(refs() if callable(refs) else refs)

    def refusal(self, row: Mapping[str, Any], *, protected: Optional[FrozenSet[str]] = None) -> str:
        """The reason a well-formed row is refused, or "" when nothing refuses it. ``protected``: the protected refs,
        read once per admission (read now when not given)."""
        objective = self.objective
        name, label, push = row["teacher"], row["label"], row["push_away"]
        if objective.points and row["point"] not in objective.points:
            return "point_not_declared"
        teacher = TEACHERS.get(name)
        if teacher is None:
            return "unknown_teacher"
        if name not in objective.losses:
            return "undeclared_teacher"
        safety = self._safety.get(row["point"])
        if not _fits(objective.target_type, label, tuple(objective.labels)) or (
                safety is not None and not (isinstance(label, str) and label in safety.labels)):
            return "target_mismatch"
        if row["source_kind"] in ADVERSARIAL_SOURCE_KINDS:
            return "adversarial_self_label"
        if (name == "synthetic") != (row["source_kind"] == "synthetic"):
            return "teacher_source_mismatch"
        if row["ref"] in (self.protected() if protected is None else protected) or \
                row["source_kind"] in self._protected_sources:
            return "protected"
        recall = canonical_kind(objective.kind) == "query-adapter"
        if recall:
            reason = recall_refusal(row, self._evaluation_sets)
            if reason:
                return reason
        if teacher.push_only and not push:
            return "push_only_teacher_pulls"
        attacker = objective.attacker_reachable or (safety is not None and safety.attacker_reachable)
        if teacher.model and attacker and not recall and not push and not (safety is not None and safety.tightens(label,
                                                                                                    push_away=False)):
            return "model_teacher_on_attacker_input"
        if safety is not None and not safety.tightens(label, push_away=push):
            allowed = ATTACKER_LOOSENING_TEACHERS if attacker else LOOSENING_TEACHERS
            if name not in allowed:
                return "loosening_on_attacker_input" if attacker else "loosening_from_live"
        return ""

    def _state_key(self, row: Mapping[str, Any]):
        if canonical_kind(self.objective.kind) == "query-adapter":
            label = row["label"]
            return row["state_hash"], tuple(sorted((label["preferred"], label["disfavoured"])))
        return row["state_hash"]

    def _target_key(self, row: Mapping[str, Any]) -> bytes:
        label = row["label"]
        if canonical_kind(self.objective.kind) == "query-adapter":
            label = label["disfavoured" if row["push_away"] else "preferred"]
        return _key(label)

    # -- a period --------------------------------------------------------------------------------------------------

    def admit(self, rows: Iterable[Mapping[str, Any]], *, period: str,
              already: FrozenSet[Tuple[str, str]] = frozenset(), now: Any = None) -> Admitted:
        """Admit ``rows`` into ``period``. ``already``: (ref, teacher) pairs an earlier period admitted, which are not
        offered again. ``now``: the admission's time (the clock's by default); a re-admission passes the original."""
        moment = _io.parse(now) if now is not None else self._clock()
        protected = self.protected()
        decided: List[Tuple[Mapping[str, Any], str, str]] = []
        candidates: List[Mapping[str, Any]] = []
        malformed = 0
        for row in rows:
            try:
                check_row(row)
            except RowError:
                malformed += 1
                continue
            if (row["ref"], row["teacher"]) in already:
                continue
            reason = self.refusal(row, protected=protected)
            if reason:
                decided.append((row, "refused", reason))
            elif self.objective.text and _io.parse(row["at"]) > moment - self._quarantine:
                decided.append((row, "quarantine", "text_lag"))
            else:
                candidates.append(row)
        candidates.sort(key=_order)

        groups: Dict[Tuple[Any, str], List[Mapping[str, Any]]] = defaultdict(list)
        for row in candidates:
            groups[(self._state_key(row), row["teacher"])].append(row)
        kept: List[Mapping[str, Any]] = []
        witnesses = 0
        for group in groups.values():
            if len({(self._target_key(row), row["push_away"]) for row in group}) > 1:
                decided += [(row, "refused", "contradiction") for row in group]
                continue
            kept.append(group[0])
            decided += [(row, "witness", "repeat") for row in group[1:]]
            witnesses += len(group) - 1
        kept.sort(key=_order)

        outcomes: Dict[Any, set] = defaultdict(set)
        for row in kept:
            if TEACHERS[row["teacher"]].counts_as_proof("live") and not row["push_away"]:
                outcomes[self._state_key(row)].add(self._target_key(row))
        overruled: Dict[str, int] = defaultdict(int)
        standing: List[Mapping[str, Any]] = []
        for row in kept:
            truth = outcomes.get(self._state_key(row))
            if TEACHERS[row["teacher"]].model and truth:
                label = self._target_key(row)
                agrees = (label not in truth) if row["push_away"] else (truth == {label})
                if not agrees:
                    decided.append((row, "refused", "overruled"))
                    overruled[row["teacher"]] += 1
                    continue
            standing.append(row)

        capped = self._capped(standing)
        admitted = [row for row in standing if id(row) not in capped]
        decided += [(row, "refused", "source_cap") for row in standing if id(row) in capped]
        decided += [(row, "admitted", "") for row in admitted]
        decided.sort(key=lambda item: _order(item[0]))
        manifest = Manifest(learner=self.learner, objective=self.objective.name,
                            objective_identity=self.objective.identity, period=period,
                            admitted_at=_io.stamp(moment),
                            entries=tuple(Entry.of(row, verdict, reason) for row, verdict, reason in decided),
                            witnesses=witnesses, overruled=dict(sorted(overruled.items())), malformed=malformed)
        return Admitted(manifest=manifest, rows=tuple(admitted))

    def readmit(self, manifest: Manifest, rows: Iterable[Mapping[str, Any]]) -> Manifest:
        """``manifest``'s period admitted again from what its source still re-renders, at the original time: the
        manifest the period would have had if an erased row had never existed (a witness of an erased row becomes
        the row, a capped source's share is recomputed)."""
        wanted = {(entry.ref, entry.teacher) for entry in manifest.entries}
        offered = [row for row in rows if isinstance(row, Mapping) and (row.get("ref"), row.get("teacher")) in wanted]
        result = self.admit(offered, period=manifest.period, now=manifest.admitted_at).manifest
        return replace(result, malformed=manifest.malformed) if manifest.malformed else result

    def _capped(self, rows: List[Mapping[str, Any]]) -> set:
        """The ids of rows over their source's share of the period (the earliest of each source are kept)."""
        exempt = 0
        by_source: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            source = self._source_of(row)
            if source is None:
                exempt += 1
            else:
                by_source[str(source)].append(row)
        if not by_source:
            return set()
        cap = 0
        for candidate in range(max(len(group) for group in by_source.values()), -1, -1):
            total = exempt + sum(min(len(group), candidate) for group in by_source.values())
            if candidate <= self._share * total + 1e-9:
                cap = candidate
                break
        return {id(row) for group in by_source.values() for row in group[cap:]}


__all__ = ["ADVERSARIAL_SOURCE_KINDS", "ATTACKER_LOOSENING_TEACHERS", "LOOSENING_TEACHERS", "QUARANTINE", "REASONS",
           "SOURCE_SHARE", "Admission", "Admitted"]
