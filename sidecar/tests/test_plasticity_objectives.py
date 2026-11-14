"""Plasticity's closed teacher list and its objective registry (docs/PLASTICITY.md, Teachers and objectives).

The registry refuses an objective with an undeclared teacher, a target type its teacher cannot give, or a blend over
teachers it does not declare, as Instinct's registry refuses a point; and a model teacher never counts as promotion
evidence, wherever the evidence is weighed.
"""

from __future__ import annotations

import math

import pytest

from colonymind.plasticity import objectives, teachers
from colonymind.plasticity.objectives import SAFETY_POINTS, Objective, RegistrationError, SafetyPoint, register


def _objective(**changes):
    fields = dict(name="probe", model="probe_model", kind="supervised_refit", target_type="label",
                  losses={"verified_outcome": "cross_entropy", "owner_label": "cross_entropy"},
                  blend={"verified_outcome": 1.0, "owner_label": 1.0}, labels=("yes", "no"))
    fields.update(changes)
    return Objective(**fields)


def test_the_teacher_list_is_closed():
    assert set(teachers.TEACHERS) == {"verified_outcome", "owner_label", "behavioural_proxy", "synthetic",
                                      "deliberation", "stronger_model", "next_observation", "realized_value", "style"}
    with pytest.raises(TypeError):
        teachers.TEACHERS["style"] = teachers.TEACHERS["synthetic"]  # type: ignore[index]
    for teacher in teachers.TEACHERS.values():
        assert teacher.target_types and teacher.target_types <= set(teachers.TARGET_TYPES)
        assert teacher.kinds and teacher.kinds <= set(teachers.KINDS)
        assert teacher.proof in teachers.PROOF


def test_a_model_teacher_never_counts_as_promotion_evidence():
    for name in ("deliberation", "stronger_model", "behavioural_proxy"):
        for where in teachers.EVIDENCE:
            assert not teachers.TEACHERS[name].counts_as_proof(where), (name, where)
    assert {name for name in teachers.MODEL_TEACHERS} == {"deliberation", "stronger_model"}
    live = {name for name, teacher in teachers.TEACHERS.items() if teacher.counts_as_proof("live")}
    assert live == {"verified_outcome", "owner_label"} == set(teachers.PROOF_TEACHERS)
    # Ground truth by construction counts on held-out windows and frames, never as live evidence.
    for name in ("next_observation", "realized_value"):
        assert teachers.TEACHERS[name].counts_as_proof("held_out")
        assert not teachers.TEACHERS[name].counts_as_proof("live")
    # Synthetic splits count only inside pooled gate suites.
    assert teachers.TEACHERS["synthetic"].counts_as_proof("gate_suite")
    assert not teachers.TEACHERS["synthetic"].counts_as_proof("live")
    assert not teachers.TEACHERS["synthetic"].counts_as_proof("held_out")
    with pytest.raises(ValueError):
        teachers.TEACHERS["owner_label"].counts_as_proof("somewhere")


def test_only_the_behavioural_proxy_is_push_only():
    assert [name for name, teacher in teachers.TEACHERS.items() if teacher.push_only] == ["behavioural_proxy"]


def test_a_well_formed_objective_registers():
    table = {}
    objective = register(_objective(), registry=table)
    assert table == {"probe": objective}
    with pytest.raises(RegistrationError, match="already registered"):
        register(_objective(), registry=table)


def test_the_registry_refuses_an_unknown_teacher():
    with pytest.raises(RegistrationError, match="unknown teacher"):
        register(_objective(losses={"verified_outcome": "cross_entropy", "unknown": "cross_entropy"},
                            blend={"verified_outcome": 1.0, "unknown": 1.0}), registry={})


def test_the_registry_refuses_a_mismatched_target_type():
    # next_observation gives vectors; a label objective cannot take it.
    with pytest.raises(RegistrationError, match="target type"):
        register(_objective(losses={"next_observation": "normalised_squared_error"},
                            blend={"next_observation": 1.0}), registry={})
    with pytest.raises(RegistrationError, match="target type"):
        register(_objective(target_type="colour"), registry={})


def test_the_registry_refuses_a_blend_over_teachers_it_does_not_declare():
    with pytest.raises(RegistrationError, match="blend"):
        register(_objective(blend={"verified_outcome": 1.0, "owner_label": 1.0, "synthetic": 0.5}), registry={})
    with pytest.raises(RegistrationError, match="blend"):
        register(_objective(blend={"verified_outcome": 1.0}), registry={})


@pytest.mark.parametrize("weight", [-0.1, math.inf, math.nan, True])
def test_the_registry_refuses_a_weight_that_is_not_a_positive_number(weight):
    with pytest.raises(RegistrationError, match="weight"):
        register(_objective(blend={"verified_outcome": 1.0, "owner_label": weight}), registry={})


def test_the_registry_refuses_a_teacher_the_kind_does_not_declare():
    # owner_label teaches the query adapter, small predictors and supervised refits, never detached heads.
    with pytest.raises(RegistrationError, match="kind"):
        register(_objective(kind="detached_heads"), registry={})
    with pytest.raises(RegistrationError, match="kind"):
        register(_objective(kind="a_new_kind"), registry={})


def test_the_registry_refuses_a_loss_the_teacher_does_not_take():
    with pytest.raises(RegistrationError, match="loss"):
        register(_objective(losses={"verified_outcome": "kl", "owner_label": "cross_entropy"}), registry={})


def test_the_registry_refuses_a_protected_model():
    with pytest.raises(RegistrationError, match="protected"):
        register(_objective(), registry={}, protected=frozenset({"probe_model"}))


def test_a_label_objective_names_at_least_two_labels_when_it_names_any():
    with pytest.raises(RegistrationError, match="labels"):
        register(_objective(labels=("yes",)), registry={})
    register(_objective(labels=()), registry={})  # the points' own labels


def test_the_blend_is_part_of_the_identity():
    first = _objective()
    assert first.identity == _objective().identity
    assert first.identity != _objective(blend={"verified_outcome": 1.0, "owner_label": 0.5}).identity
    other_loss = _objective(losses={"verified_outcome": "logistic", "owner_label": "cross_entropy"})
    assert first.identity != other_loss.identity


def test_the_core_objectives_are_registered_and_pass_their_checks():
    assert {"outcome", "sensory_prediction", "recall_use", "next_value"} <= set(objectives.REGISTRY)
    for objective in objectives.REGISTRY.values():
        objectives.check(objective)
    outcome = objectives.REGISTRY["outcome"]
    assert "deliberation" in outcome.losses and "behavioural_proxy" in outcome.losses


def test_safety_points_tighten_toward_holding_flagging_or_a_stop():
    assert set(SAFETY_POINTS) == {"tool_screen", "inbound_triage", "owner_send", "opt_out", "no_reminders",
                                  "owner_instruction"}
    assert {name for name, point in SAFETY_POINTS.items() if point.attacker_reachable} == {"tool_screen",
                                                                                          "inbound_triage"}
    stop = SAFETY_POINTS["opt_out"]
    assert stop.tightens("yes", push_away=False) and not stop.tightens("no", push_away=False)
    # Pushing away from "no" leaves only "yes": that tightens. Pushing away from "yes" loosens.
    assert stop.tightens("no", push_away=True) and not stop.tightens("yes", push_away=True)
    instruction = SAFETY_POINTS["owner_instruction"]
    # Away from "none" the mass also reaches "resume", which lifts a stop: that is not tightening.
    assert not instruction.tightens("none", push_away=True)
    assert instruction.tightens("stop", push_away=False) and not instruction.tightens("resume", push_away=False)
    with pytest.raises(ValueError):
        SafetyPoint("probe", labels=frozenset({"yes", "no"}), tightening=frozenset({"maybe"}))
    with pytest.raises(ValueError):
        SafetyPoint("probe", labels=frozenset({"yes", "no"}), tightening=frozenset({"yes", "no"}))


def test_the_safety_points_name_the_labels_instinct_registers_for_them():
    from colonymind.instinct import external, points

    registered = {point.name: set(point.labels) for point in (*points.POINTS, *external.SHIPPED)}
    assert {name: set(point.labels) for name, point in SAFETY_POINTS.items()} == {
        name: registered.get(name) for name in SAFETY_POINTS}


def test_canonical_kinds_and_teacher_targets_are_checked_per_kind():
    assert set(teachers.KINDS) == {"installed", "detached-heads", "query-adapter", "numeric-online",
                                  "supervised-refit", "served-adapter", "behaviour"}
    assert teachers.TEACHERS["owner_label"].targets_for("query-adapter") == frozenset({"pairwise"})
    for kind, target in (("query-adapter", "label"), ("numeric-online", "pairwise")):
        with pytest.raises(RegistrationError, match="target type"):
            objectives.check(_objective(kind=kind, target_type=target))
    objectives.check(_objective(kind="supervised-refit", blend={"verified_outcome": 1.0, "owner_label": 0.0}))
    objectives.check(_objective(kind="served-adapter", losses={"style": "cross_entropy"}, blend={"style": 1.0}))


def test_teacher_allowlists_match_the_six_engine_kinds():
    expected = {
        "installed": {"next_observation"}, "detached-heads": {"next_observation"},
        "query-adapter": {"verified_outcome", "owner_label", "synthetic", "deliberation", "stronger_model",
                          "behavioural_proxy"},
        "numeric-online": {"realized_value", "verified_outcome", "owner_label", "behavioural_proxy"},
        "supervised-refit": {"realized_value", "verified_outcome", "owner_label", "synthetic", "deliberation",
                             "stronger_model", "behavioural_proxy"}, "served-adapter": {"style"},
        "behaviour": {"verified_outcome"},
    }
    assert {kind: {name for name, teacher in teachers.TEACHERS.items() if kind in teacher.kinds}
            for kind in teachers.KINDS} == expected
