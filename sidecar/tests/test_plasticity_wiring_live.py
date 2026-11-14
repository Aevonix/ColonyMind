"""Configured nightly learners keep a live comparator and recover whole-chain rollback."""

import json
from datetime import timedelta
from types import SimpleNamespace

import pytest

from test_plasticity_wiring_runtime import learner_config as load_config
from colonymind.plasticity.nightly import Runtime
from colonymind.plasticity.synthetic import EPOCH, SyntheticSource, make_rows

NOW = EPOCH + timedelta(days=10)


def _runtime(root, *, chain=False):
    names = ("first", "second") if chain else ("probe",)
    declarations = {name: {"implementation": "synthetic", "kind": "supervised_refit",
                           "area": "instinct", "mode": "rebuild", "dependents": ["threshold"],
                           **({"chain": "pair"} if chain else {})} for name in names}
    (root / "colonymind.yaml").write_text(json.dumps({"plasticity": {"learners": declarations}}))
    sources = {name: SyntheticSource(make_rows(60, prefix=name)) for name in names}
    service = Runtime.from_config(load_config(root, environ={}), sources=sources, clock=lambda: NOW,
                                  check_learner=lambda name, declared: SimpleNamespace(name=name))
    assert all(row["result"] == "promoted" for row in service.run())
    first = {name: binding.store.revisions.serving().id for name, binding in service.loaded.items()}
    for name, source in sources.items():
        source.add(make_rows(30, prefix=f"later-{name}"))
    assert all(row["result"] == "promoted" for row in service.run())
    second = {name: binding.store.revisions.serving().id for name, binding in service.loaded.items()}
    assert all(first[name] != second[name] for name in names)
    return service, first, second


def _regress(service, name, start=0, stop=4):
    result = None
    for index in range(start, stop):
        ref = f"live-{index}"
        service.serve(name, ref, "fixture input", predict=lambda pin, value: pin.closure.id)
        result = service.outcome(name, ref, served_correct=False, reference_correct=True)
    return result


def test_second_night_live_regression_restores_the_first_nights_serving_closure(tmp_path):
    service, first, second = _runtime(tmp_path)
    result = _regress(service, "probe")
    assert result["rolled_back"] and result["reference"] == first["probe"]
    assert service.serve("probe", "after-rollback", "input", predict=lambda pin, value: pin.closure.id) == first["probe"]
    binding = service.loaded["probe"]
    assert binding.store.revisions.status(second["probe"]).state == "withdrawn"
    assert binding.store.periods.periods()[-1].state == "discarded"
    assert service.report()["learners"][0]["events"][-1]["event"] == "rolled_back"


def test_one_chain_members_regression_rolls_back_and_serves_the_previous_whole_joint(tmp_path):
    service, first, second = _runtime(tmp_path, chain=True)
    assert _regress(service, "first")["rolled_back"]
    pins = service.pin_chain("pair", "owner")
    assert {name: pin.closure.id for name, pin in pins.items()} == first
    assert service.joints.pending("pair") is None
    for name, binding in service.loaded.items():
        assert binding.store.revisions.status(second[name]).state == "withdrawn"
        assert service.serve(name, f"after-{name}", "input", predict=lambda pin, value: pin.closure.id) == first[name]
    # The previous joint can itself fail later; then both members really serve their bases.
    assert service.rollback("first")["result"] == "rolled_back"
    assert all(pin.closure.mode == "base" for pin in service.pin_chain("pair", "owner").values())
    assert service.serve("second", "after-base", "input", predict=lambda pin, value: pin.closure.mode) == "base"


def test_interrupted_chain_rollback_withdraws_every_failed_member_before_recovery_can_repromote(tmp_path, monkeypatch):
    service, first, second = _runtime(tmp_path, chain=True)
    _regress(service, "first", stop=3)

    def crash(**kwargs):
        raise OSError("power lost")

    monkeypatch.setattr(service.loaded["second"].controller, "rollback", crash)
    with pytest.raises(OSError, match="power lost"):
        _regress(service, "first", start=3, stop=4)
    assert all(pin.closure.mode == "base" for pin in service.pin_chain("pair", "owner").values())
    monkeypatch.undo()
    service._eraser().erase(())
    assert {name: pin.closure.id for name, pin in service.pin_chain("pair", "owner").items()} == first
    for name, binding in service.loaded.items():
        assert binding.store.revisions.status(second[name]).state == "withdrawn"
        assert binding.store.periods.periods()[-1].state == "discarded"
