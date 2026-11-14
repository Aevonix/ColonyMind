"""The runtime hands each configured learner to the real check_learner (PLAN 4.5, item 1): no stub."""
from datetime import timedelta

import yaml

from test_plasticity_wiring_runtime import learner_config as load_config
from colonymind.plasticity.synthetic import EPOCH

NOW = EPOCH + timedelta(days=10)


def declaration(**changes):
    """A full registration (G1) plus the runtime's own fields (P1b). Shares and bounds are test assumptions."""
    spec = dict(implementation="synthetic", area="instinct", mode="rebuild",
                kind="supervised-refit", teachers={"owner_label": "label"},
                model={"roles": ["decision_model"], "change": "head", "own_head": "probe-head",
                       "points": ["report_substance"], "instance_points": ["report_substance"]},
                budget={"type": "relative_dense", "bound": 0.2}, gates=["finite", "budget", "canaries"],
                arms=["frozen", "no_accumulation"], audience="owner", personal_data=True,
                erasure_path="rebuild_from_manifest", dependents=["calibration"], effect="skip",
                forced_run_share=0.05)
    spec.update(changes)
    return spec


def test_a_learner_failing_a_real_check_does_not_load_and_the_others_do(tmp_path):
    from colonymind.plasticity.nightly import Runtime
    bad = declaration()
    del bad["erasure_path"]
    data = {"plasticity": {"learners": {"bad": bad, "good": declaration()}}}
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump(data))
    service = Runtime.from_config(load_config(tmp_path, environ={}), clock=lambda: NOW)
    assert list(service.loaded) == ["good"]
    assert service.loaded["good"].registered.kind == "supervised-refit"
    assert "erasure_path" in service.report()["rejected"]["bad"]
