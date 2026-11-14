"""The public plasticity controls, report and isolated benchmark interventions.

Fixture counts and losses are synthetic assumptions chosen only to distinguish the report fields.
"""
import argparse
from copy import deepcopy
import json

import pytest


def parser():
    from colonymind.plasticity.cli import add_parser
    value = argparse.ArgumentParser()
    add_parser(value.add_subparsers(dest="command"))
    return value


@pytest.mark.parametrize("command,arguments", [
    ("status", []), ("learners", []), ("revisions", ["sample"]),
    ("comparator", ["sample"]), ("rollback", ["sample"]),
    ("erase", ["--refs", "record-a", "record-b"]),
])
def test_cli_dispatches_each_command_and_returns_json(command, arguments, monkeypatch, capsys):
    from colonymind.plasticity import cli
    calls = []

    class Runtime:
        def __getattr__(self, name):
            def call(*args):
                calls.append((name, args))
                return {"operation": name}
            return call

    monkeypatch.setattr(cli, "_runtime", lambda: Runtime())
    args = parser().parse_args(["plasticity", command, *arguments, "--json"])
    assert cli.run(args) == 0
    assert json.loads(capsys.readouterr().out) == {"operation": command}
    expected = ((["record-a", "record-b"],) if command == "erase" else
                ("sample",) if command in {"revisions", "comparator", "rollback"} else ())
    assert calls == [(command, expected)]


def test_cli_refuses_missing_refs_and_manual_promotion():
    for args in (["erase"], ["erase", "--refs"], ["promote", "sample"]):
        with pytest.raises(SystemExit) as error:
            parser().parse_args(["plasticity", *args])
        assert error.value.code == 2


def test_cli_unknown_learner_is_a_useful_error(monkeypatch, capsys):
    from colonymind.plasticity import cli

    class Runtime:
        def revisions(self, name):
            raise KeyError(name)

    monkeypatch.setattr(cli, "_runtime", lambda: Runtime())
    assert cli.run(parser().parse_args(["plasticity", "revisions", "missing", "--json"])) == 2
    assert "missing" in capsys.readouterr().err


def test_report_keeps_teacher_proof_separate_and_explains_rollbacks():
    from colonymind.reports import plasticity
    payload = {"enabled": True, "rejected": {"other": "budget check failed"}, "learners": [{
        "name": "sample", "serving": "closure-a", "teachers": [
            {"teacher": "owner_label", "rows": 3, "loss": 0.1, "counts_as_proof": True},
            {"teacher": "stronger_model", "rows": 5, "loss": 0.2, "counts_as_proof": False}],
        "events": [{"event": "promoted", "reason": "all gates passed"},
                   {"event": "dropped", "reason": "failed canaries"},
                   {"event": "rolled_back", "reason": "measured regression"}],
        "comparator": {"labelled": 10, "losses": 8, "gains": 0, "rolled_back": True,
                       "reason": "measured regression"}, "erasures": 2}]}
    collected = plasticity.collect(None, runtime=type("Runtime", (), {"report": lambda self: payload})())
    assert json.loads(plasticity.to_json(collected)) == payload
    rendered = plasticity.document(collected).text()
    for expected in ("sample", "3 rows", "5 rows", "counts as proof", "does not count as proof",
                     "1 promoted", "1 refused", "1 rolled back", "failed canaries", "measured regression",
                     "2 erasures", "budget check failed", "Setting: mind.faculties.plasticity"):
        assert expected in rendered


@pytest.mark.parametrize("enabled,expected", [(True, "No learners configured"), (False, "switched off")])
def test_report_explains_idle_and_disabled_faculty(enabled, expected):
    from colonymind.reports import plasticity
    assert expected in plasticity.document({"enabled": enabled, "learners": [], "rejected": {}}).text()


def test_report_reads_snapshot_without_loading_runtime_or_writing(tmp_path, monkeypatch):
    from colonymind.config import Config, DEFAULTS
    from colonymind.plasticity import nightly
    from colonymind.reports import plasticity

    def forbidden(*args, **kwargs):
        raise AssertionError("report loaded a runtime")

    monkeypatch.setattr(nightly.Runtime, "from_config", forbidden)
    config = Config(home=tmp_path, data=deepcopy(DEFAULTS))
    before = sorted(tmp_path.rglob("*"))
    result = plasticity.collect(config)
    assert result["enabled"] is True
    assert result["learners"] == []
    assert sorted(tmp_path.rglob("*")) == before


def test_plasticity_is_registered_immediately_after_learning():
    from colonymind.reports.registry import FACULTIES
    names = [faculty.key for faculty in FACULTIES]
    assert names[names.index("learning") + 1] == "plasticity"


def test_arms_change_only_their_named_learning_dimension():
    from colonymind.qualification.paired import apply_plasticity_arm, plasticity_profiles
    config = {"mind": {"faculties": {"plasticity": True, "learning": True}}, "other": ["kept"],
              "plasticity": {"state_dir": "plasticity", "learners": {
                  "sample": {"area": "recall", "teachers": ["owner_label", "stronger_model"]},
                  "clock": {"area": "timing", "teachers": ["verified_outcome"]}}}}
    before = deepcopy(config)
    profiles = plasticity_profiles(config["plasticity"]["learners"])
    assert {"full-plasticity", "plasticity.sample", "plasticity.recall", "plasticity.instinct",
            "plasticity.perception", "plasticity.telemetry", "plasticity.timing", "plasticity.voice",
            "plasticity.sample.teacher.stronger_model", "shuffled-control", "no_accumulation"} <= set(profiles)
    cases = [("plasticity.sample", {"sample": {"enabled": False}}),
             ("plasticity.recall", {"sample": {"enabled": False}}),
             ("plasticity.sample.teacher.stronger_model", {"sample": {"disabled_teachers": ["stronger_model"]}}),
             ("full-plasticity", {name: {"enabled": False} for name in ("sample", "clock")}),
             ("shuffled-control", {name: {"control": "shuffled"} for name in ("sample", "clock")}),
             ("no_accumulation", {name: {"accumulation": False} for name in ("sample", "clock")})]
    for arm, changes in cases:
        expected = deepcopy(config)
        for name, change in changes.items():
            expected["plasticity"]["learners"][name].update(change)
        if arm == "full-plasticity":   # the base arm also switches off the core default learners it was not given
            from colonymind.config import DEFAULTS
            expected["plasticity"]["learners"].update(
                {name: {"enabled": False} for name in DEFAULTS["plasticity"]["learners"]})
        assert apply_plasticity_arm(config, arm) == expected
    assert config == before
    with pytest.raises(ValueError, match="unknown"):
        apply_plasticity_arm(config, "plasticity.missing")
