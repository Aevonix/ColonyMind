"""The teachers: every training row carries the id of the teacher that produced its target.

The list is closed, in code, like Instinct's effects: a teacher that is not here cannot teach. Each teacher names
the target types it gives, the learner kinds that may use it, the losses its own term may take, and where it may
count as promotion evidence (``proof``):

- ``always``: the owner's labels and verified outcomes count wherever evidence is weighed;
- ``held_out``: ground truth by construction (the next observation, the realized value) counts on held-out windows
  and frames, never as live evidence;
- ``gate_suite``: synthetic splits count only inside pooled gate suites;
- ``never``: model teachers (deliberation, a stronger model) and behavioural proxies never count. A model teacher
  can make a learner better; it cannot make a learner look better.

A ``push_only`` teacher (the behavioural proxy) may only push away from a label: a margin on the disfavoured label,
never a positive. A ``model`` teacher is another model's judgement: never proof, and where the input is reachable
by someone other than the owner it may only push away (``admission.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import FrozenSet, Mapping, Tuple

#: What a target is: a label from a closed set, a yes-or-no outcome, a number, a vector of site outputs, a soft
#: distribution over labels, or a preference between two items (retrieval).
TARGET_TYPES: Tuple[str, ...] = ("label", "binary", "value", "vector", "distribution", "pairwise")
#: The learner kinds, closed (the engine implements them; the letters are the design's).
KINDS: Tuple[str, ...] = ("installed", "detached-heads", "query-adapter", "numeric-online", "supervised-refit",
                          "served-adapter", "behaviour")
KIND_LETTERS = MappingProxyType({"A": "installed", "B": "detached-heads", "C": "query-adapter",
                                 "D": "numeric-online", "E": "supervised-refit", "F": "served-adapter",
                                 "G": "behaviour"})
LOSSES: Tuple[str, ...] = ("normalised_squared_error", "logistic", "cross_entropy", "pairwise", "kl", "listwise_kl",
                           "margin")
PROOF: Tuple[str, ...] = ("always", "held_out", "gate_suite", "never")
#: Where evidence is weighed: live rows, held-out windows and frozen holdouts, pooled gate suites.
EVIDENCE: Tuple[str, ...] = ("live", "held_out", "gate_suite")
_COUNTS = {"always": frozenset(EVIDENCE), "held_out": frozenset({"held_out", "gate_suite"}),
           "gate_suite": frozenset({"gate_suite"}), "never": frozenset()}


@dataclass(frozen=True)
class Teacher:
    """One teacher: ``id``, what it is, the target types it gives, the kinds that may use it, the losses of its own
    term, where it counts as proof, whether it may only push away, and whether it is a model's judgement."""

    id: str
    what: str
    target_types: FrozenSet[str]
    kinds: FrozenSet[str]
    losses: Tuple[str, ...]
    proof: str
    push_only: bool = False
    model: bool = False

    def targets_for(self, kind: str) -> FrozenSet[str]:
        """Targets are checked for this kind, never against a union of unrelated sites."""
        return KIND_TARGETS.get(canonical_kind(kind), {}).get(self.id, frozenset())

    def counts_as_proof(self, where: str) -> bool:
        """Whether a row from this teacher may count as promotion evidence ``where`` (one of ``EVIDENCE``)."""
        if where not in EVIDENCE:
            raise ValueError(f"evidence is weighed {', '.join(EVIDENCE)}, not {where!r}")
        return where in _COUNTS[self.proof]


def _teacher(id: str, what: str, targets, kinds, losses, proof: str, **flags) -> Teacher:
    return Teacher(id=id, what=what, target_types=frozenset(targets), kinds=frozenset(kinds), losses=tuple(losses),
                   proof=proof, **flags)


_SUPERVISED = ("label", "binary", "pairwise", "distribution")

TEACHERS: Mapping[str, Teacher] = MappingProxyType({teacher.id: teacher for teacher in (
    _teacher("next_observation", "the model's own frozen output on the next admitted observation of the stream",
             ("vector",), ("installed", "detached-heads"), ("normalised_squared_error",), "held_out"),
    _teacher("realized_value", "the measured next value of a stream, or whether the predicted event happened",
             ("value", "binary", "label"), ("numeric-online", "supervised-refit"),
             ("normalised_squared_error", "logistic", "cross_entropy"), "held_out"),
    _teacher("verified_outcome", "an outcome a resolver attributes to the choice the learner made",
             _SUPERVISED, ("query-adapter", "numeric-online", "supervised-refit", "behaviour"),
             ("cross_entropy", "logistic", "pairwise"), "always"),
    _teacher("owner_label", "the owner's explicit label or correction through an owner channel",
             _SUPERVISED, ("query-adapter", "numeric-online", "supervised-refit"),
             ("cross_entropy", "logistic", "pairwise"), "always"),
    _teacher("synthetic", "generated splits at new seeds, never frozen holdouts",
             ("label", "pairwise"), ("query-adapter", "supervised-refit"), ("cross_entropy", "pairwise"),
             "gate_suite"),
    _teacher("deliberation", "the deep path's answer on an escalated case, or a passage the answer cited",
             ("label", "distribution", "pairwise"), ("query-adapter", "supervised-refit"),
             ("kl", "cross_entropy", "pairwise"), "never", model=True),
    _teacher("stronger_model", "a stronger local model's judgement of the same question",
             ("label", "distribution", "pairwise"), ("query-adapter", "supervised-refit"),
             ("listwise_kl", "kl", "cross_entropy", "pairwise"), "never", model=True),
    _teacher("behavioural_proxy", "engagement, silence, an item shown and never used",
             _SUPERVISED, ("query-adapter", "numeric-online", "supervised-refit"), ("margin",), "never",
             push_only=True),
    _teacher("style", "owner-scope style targets", ("label", "distribution"), ("served-adapter",),
             ("cross_entropy", "kl"), "never"),
)})

# Existing objective manifests keep their spelling and digest; new learner declarations use KINDS.
_ALIASES = {"detached_heads": "detached-heads", "query_adapter": "query-adapter",
            "online_predictor": "numeric-online", "supervised_refit": "supervised-refit",
            "served_adapter": "served-adapter"}


def canonical_kind(kind: str) -> str:
    return _ALIASES.get(kind, kind)


_KIND_TARGETS = {
    "behaviour": {"verified_outcome": ("binary",)},
    "installed": {"next_observation": ("vector",)},
    "detached-heads": {"next_observation": ("vector",)},
    "query-adapter": {name: ("pairwise",) for name in
                      ("verified_outcome", "owner_label", "synthetic", "deliberation", "stronger_model",
                       "behavioural_proxy")},
    "numeric-online": {"realized_value": ("value", "binary"), "verified_outcome": ("binary",),
                       "owner_label": ("binary",), "behavioural_proxy": ("binary",)},
    "supervised-refit": {"realized_value": ("label", "binary"),
                         "verified_outcome": ("label", "binary", "distribution"),
                         "owner_label": ("label", "binary", "distribution"), "synthetic": ("label",),
                         "deliberation": ("label", "distribution"),
                         "stronger_model": ("label", "distribution"), "behavioural_proxy": ("label", "binary")},
    "served-adapter": {"style": ("label", "distribution",)},
}
KIND_TARGETS = MappingProxyType({kind: MappingProxyType({name: frozenset(targets) for name, targets in table.items()})
                               for kind, table in _KIND_TARGETS.items()})

#: The teachers whose rows count as live promotion evidence.
PROOF_TEACHERS: FrozenSet[str] = frozenset(name for name, teacher in TEACHERS.items() if teacher.proof == "always")
#: Another model's judgement: never proof.
MODEL_TEACHERS: FrozenSet[str] = frozenset(name for name, teacher in TEACHERS.items() if teacher.model)


def get(teacher_id: str) -> Teacher:
    """The teacher ``teacher_id``; KeyError when the closed list has none by that id."""
    return TEACHERS[teacher_id]


__all__ = ["EVIDENCE", "KINDS", "KIND_LETTERS", "KIND_TARGETS", "canonical_kind", "LOSSES", "MODEL_TEACHERS", "PROOF", "PROOF_TEACHERS",
           "TARGET_TYPES", "TEACHERS", "Teacher", "get"]
