"""Fail-closed declarations, with generic roles. Fixture bounds and shares are test assumptions."""
from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from colonymind.plasticity.registration import PROTECTED_ROLES, RegistrationError, check_learner


def declaration():
    return dict(kind="supervised-refit", teachers={"owner_label": "label"},
                model={"roles": ["decision_model"], "change": "head", "own_head": "probe-head",
                       "points": ["report_substance"], "instance_points": ["report_substance"]},
                budget={"type": "relative_dense", "bound": 0.2}, gates=["finite", "budget", "canaries"],
                arms=["frozen", "no_accumulation"], audience="owner", personal_data=True,
                erasure_path="rebuild_from_manifest", dependents=["calibration"], effect="skip",
                forced_run_share=0.05)


def test_a_checked_declaration_is_frozen_and_does_not_keep_mutable_input():
    spec = declaration()
    registered = check_learner("probe", spec)
    spec["dependents"].append("changed")
    assert registered.dependents == ("calibration",)
    assert registered.teachers == (("owner_label", "label"),)
    assert registered.forced_run_share == 0.05
    with pytest.raises(FrozenInstanceError):
        registered.kind = "installed"


@pytest.mark.parametrize("field", list(declaration()))
def test_each_required_field_is_named_when_missing(field):
    spec = declaration()
    del spec[field]
    with pytest.raises(RegistrationError, match=field):
        check_learner("probe", spec)


@pytest.mark.parametrize("field,value,check", [
    ("kind", "unknown", "kind"), ("teachers", {"next_observation": "vector"}, "teacher"),
    ("teachers", {"owner_label": "vector"}, "target"), ("teachers", {}, "teacher"),
    ("budget", {}, "budget"), ("budget", {"type": "relative_dense", "bound": float("nan")}, "budget"),
    ("gates", [], "gates"), ("arms", [], "arms"), ("audience", "everyone", "audience"),
    ("erasure_path", "", "erasure"), ("dependents", None, "dependents"),
    ("personal_data", "false", "personal_data"), ("forced_run_share", 0, "forced_run_share"),
    ("forced_run_share", float("nan"), "forced_run_share"), ("forced_run_share", True, "forced_run_share"),
    ("effect", "unknown", "effect"),
])
def test_invalid_declarations_name_the_failed_check(field, value, check):
    spec = declaration()
    spec[field] = value
    with pytest.raises(RegistrationError, match=check):
        check_learner("probe", spec)


@pytest.mark.parametrize("role", sorted(PROTECTED_ROLES))
def test_protected_roles_cannot_train_weights(role):
    spec = declaration()
    spec["model"]["roles"].append(role)
    with pytest.raises(RegistrationError, match="protected"):
        check_learner("probe", spec)


def test_only_client_adapters_may_use_the_live_embedding_roles():
    spec = declaration()
    spec.update(kind="query-adapter", teachers={"owner_label": "pairwise"})
    spec["model"] = {"roles": ["live_embedder"], "change": "client_adapter"}
    assert check_learner("probe", spec).kind == "query-adapter"
    spec["model"]["roles"].append("identity")
    with pytest.raises(RegistrationError, match="protected"):
        check_learner("probe", spec)


@pytest.mark.parametrize("change", [
    {"own_head": ""}, {"points": []}, {"points": ["tool_screen"]},
    {"change": "encoder", "points": ["report_substance"], "instance_points": ["report_substance", "tool_screen"]},
    {"change": "encoder", "points": ["report_substance"], "instance_points": ["report_substance", "owner_verdict"]},
])
def test_decision_model_heads_and_encoder_coverage_are_required(change):
    spec = declaration()
    spec["model"].update(deepcopy(change))
    with pytest.raises(RegistrationError, match="decision_model"):
        check_learner("probe", spec)


def test_encoder_change_covers_all_nonsafety_points():
    spec = declaration()
    spec["model"].update(change="encoder", points=["owner_verdict", "report_substance"],
                         instance_points=["report_substance", "owner_verdict"])
    assert check_learner("probe", spec).name == "probe"


@pytest.mark.parametrize("field", ["roles", "change"])
def test_missing_model_checks_do_not_pass(field):
    spec = declaration()
    del spec["model"][field]
    with pytest.raises(RegistrationError, match=field):
        check_learner("probe", spec)


def test_style_targets_cannot_train_a_nonowner_audience():
    spec = declaration()
    spec.update(kind="served-adapter", teachers={"style": "label"}, audience="guest")
    spec["model"] = {"roles": ["offline_model"], "change": "offline_adapter"}
    with pytest.raises(RegistrationError, match="audience"):
        check_learner("probe", spec)


def test_unknown_budget_type_is_refused():
    spec = declaration()
    spec["budget"]["type"] = "unbounded"
    with pytest.raises(RegistrationError, match="budget"):
        check_learner("probe", spec)


def test_protected_authority_modules_do_not_import_plasticity():
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    paths = ["sidecar/colonymind/mind/authority.py", "plugins/hermes-plugin/guard.py",
             "plugins/hermes-plugin/capture.py", "plugins/hermes-plugin/secrets.py",
             "sidecar/colonymind/api/routers/secrets.py", "sidecar/colonymind/api/credentials.py",
             "sidecar/colonymind/contacts/identity_links.py", "sidecar/colonymind/contacts/changes.py"]
    files = [root / path for path in paths] + list((root / "sidecar/colonymind/secrets").glob("*.py"))
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert not any("plasticity" in alias.name.split(".") for alias in node.names), path
            elif isinstance(node, ast.ImportFrom):
                assert "plasticity" not in (node.module or "").split("."), path
                assert not any(alias.name == "plasticity" for alias in node.names), path


def test_offline_served_adapter_does_not_train_the_live_brain_online():
    spec = declaration()
    spec.update(kind="served-adapter", teachers={"style": "label"})
    spec["model"] = {"roles": ["live_brain"], "change": "offline_adapter"}
    assert check_learner("probe", spec).kind == "served-adapter"
    spec["model"]["roles"].append("qualified_translator")
    with pytest.raises(RegistrationError, match="protected"):
        check_learner("probe", spec)
