"""``colonymind models paired suite``: all-on against every faculty's ablation, on chosen families, in one command,
never more benchmark streams at once than asked for."""
import json
import subprocess
import sys
import threading
import time

import pytest

from colonymind.qualification import paired, paired_suite
from colonymind.reports.registry import FACULTIES


def test_the_suite_measures_every_faculty_within_the_arm_limit(tmp_path):
    families = [tmp_path / "memory-dev", tmp_path / "outreach-dev"]
    for family in families:
        family.mkdir()
    specs = paired_suite.specs(families, label="suite")
    arms = paired_suite.faculty_arms()
    assert {key for key, _ in arms} == {faculty.key for faculty in FACULTIES if faculty.arm}
    assert "blocks" not in dict(arms)  # mandatory since context fitting has no off switch
    assert dict(arms)["initiative"] == "all-on-initiative" and dict(arms)["skills"] == "all-on-skills"
    assert all(arm in paired.PROFILES for _, arm in arms)
    for family in families:
        mine = [spec for spec in specs if spec.dataset_dir == family]
        assert all(spec.arms[0] == "all-on" == spec.reference and 2 <= len(spec.arms) <= 8 for spec in mine)
        covered = [arm for spec in mine for arm in spec.arms[1:]]
        assert sorted(covered) == sorted(arm for _, arm in arms)           # every ablation once per family
        assert max(len(spec.arms) for spec in mine) - min(len(spec.arms) for spec in mine) <= 1   # balanced
    assert len({spec.label for spec in specs}) == len(specs)
    chosen = paired_suite.specs(families[:1], label="suite", faculties=["skills", "status"])
    assert [spec.arms for spec in chosen] == [["all-on", "all-on-skills", "all-on-status"]]
    with pytest.raises(ValueError, match="unknown faculty"):
        paired_suite.specs(families[:1], label="suite", faculties=["rosettacache"])


def test_two_families_with_one_name_are_refused(tmp_path):
    """A plan's label (and its directory) comes from the family's name: two families with the same name would
    share one plan directory, so the second would be skipped as done or collide with the first."""
    first, second = tmp_path / "one" / "memory", tmp_path / "two" / "memory"
    for family in (first, second):
        family.mkdir(parents=True)
    with pytest.raises(ValueError, match="memory"):
        paired_suite.specs([first, second], label="suite", faculties=["skills"])
    with pytest.raises(ValueError, match="memory"):
        paired_suite.specs([first], label="suite", faculties=["skills"], versions=["memory"])
    assert len(paired_suite.specs([first, tmp_path / "outreach"], label="suite", faculties=["skills"])) == 2


def test_the_steps_of_a_plan_are_the_existing_commands(tmp_path):
    family = _family(tmp_path)
    spec = paired_suite.specs([family], label="suite", faculties=["skills"])[0]
    resources = _resources(tmp_path, native_config=tmp_path / "hermes.yaml",
                           container_image="registry.example/agent@sha256:" + "a" * 64)
    steps = paired_suite.steps(spec, tmp_path / "out", resources, program=["colonymind"])
    assert [step[0] for step in steps] == ["plan", "run"]
    plan = steps[0][1]
    assert plan[:4] == ["colonymind", "models", "paired", "plan"]
    assert plan[plan.index("--arms") + 1] == "all-on,all-on-skills"
    assert plan[plan.index("--reference-arm") + 1] == "all-on"
    assert plan[plan.index("--dataset-dir") + 1] == str(family)
    assert plan[plan.index("--output") + 1] == str(tmp_path / "out" / spec.label)
    assert "--resume" not in steps[1][1]
    # Planned and started: the plan is not made again and the run resumes its untouched episodes.
    directory = tmp_path / "out" / spec.label
    _freeze(directory, spec, resources, started=1)
    steps = paired_suite.steps(spec, tmp_path / "out", resources, program=["colonymind"])
    assert [step[0] for step in steps] == ["run"] and "--resume" in steps[0][1]
    # Every arm-episode started and reported: nothing is left to do.
    _freeze(directory, spec, resources)
    for path in ("runs/001-all-on-skills", "runs/002-all-on", "runs/002-all-on-skills"):
        (directory / path).mkdir()
    (directory / "report-001.json").write_text("{}")
    assert paired_suite.steps(spec, tmp_path / "out", resources, program=["colonymind"]) == []
    # A plan the suite cannot read is neither skipped nor resumed.
    (directory / "paired.json").write_text("{}")
    with pytest.raises(ValueError, match="cannot read"):
        paired_suite.steps(spec, tmp_path / "out", resources, program=["colonymind"])


def test_a_plan_with_episodes_left_is_not_done_even_with_a_report(tmp_path):
    """A run writes its report even when it stops early (a container whose state could not be cleaned up ends
    it), so a report alone is not completion: a plan is done when every arm-episode has started, and a run that
    leaves some untouched is reported as not finished, then and on every retry."""
    family = _family(tmp_path)
    spec = paired_suite.specs([family], label="suite", faculties=["skills"])[0]
    resources = _resources(tmp_path, container_image="img@sha256:" + "d" * 64)
    directory = tmp_path / "out" / spec.label
    pairs = _freeze(directory, spec, resources, started=1, reported=True)["pairs"]
    steps = paired_suite.steps(spec, tmp_path / "out", resources, program=["colonymind"])
    assert [step[0] for step in steps] == ["run"] and "--resume" in steps[0][1]
    results = paired_suite.run_suite([spec], tmp_path / "out", resources, streams=1,
                                     execute=lambda argv, log: 0, program=["colonymind"])
    assert results == [{"label": spec.label, "ok": False, "step": "run", "exit": 0, "untouched": 3}]
    for pair in pairs:
        for member in pair["arms"].values():
            (directory / member["path"]).mkdir(parents=True, exist_ok=True)
    assert paired_suite.steps(spec, tmp_path / "out", resources, program=["colonymind"]) == []
    results = paired_suite.run_suite([spec], tmp_path / "out", resources, streams=1,
                                     execute=lambda argv, log: 0, program=["colonymind"])
    assert results == [{"label": spec.label, "ok": True, "step": None, "exit": 0}]


def test_never_more_streams_run_at_once_than_asked(tmp_path):
    families = [tmp_path / f"family-{index}" for index in range(3)]
    for family in families:
        family.mkdir()
    specs = paired_suite.specs(families, label="suite")
    resources = paired_suite.Resources(native_config=tmp_path / "h.yaml", comparison_policy=tmp_path / "p.json",
                                       container_image="img@sha256:" + "b" * 64, docker_host=None,
                                       native_binding="candidate")
    lock, running, seen, calls = threading.Lock(), [0], [0], []

    def execute(argv, log):
        with lock:
            running[0] += 1
            seen[0] = max(seen[0], running[0])
            calls.append(argv)
        time.sleep(0.02)
        with lock:
            running[0] -= 1
        return 1 if "run" in argv and argv[argv.index("--output") + 1].endswith("family-1-2") else 0

    results = paired_suite.run_suite(specs, tmp_path / "out", resources, streams=2, execute=execute,
                                     program=["colonymind"])
    assert seen[0] == 2
    assert len(calls) == 2 * len(specs)                         # a plan and a run for each
    for spec in specs:                                          # each plan's steps in order, in one stream
        mine = [argv[3] for argv in calls if argv[argv.index("--output") + 1].endswith(spec.label)]
        assert mine == ["plan", "run"]
    failed = [item for item in results if not item["ok"]]
    assert [item["label"] for item in failed] == ["suite-family-1-2"] and failed[0]["step"] == "run"
    assert (tmp_path / "out" / "suite.json").is_file()
    manifest = json.loads((tmp_path / "out" / "suite.json").read_text())
    assert manifest["reference"] == "all-on" and len(manifest["plans"]) == len(specs)


def test_the_cli_prints_the_plan_of_the_suite_without_running_it(tmp_path):
    family = tmp_path / "memory-dev"
    family.mkdir()
    command = [sys.executable, "-m", "colonymind", "models", "paired", "suite", "--dataset-dir", str(family),
               "--output", str(tmp_path / "out"), "--native-config", str(tmp_path / "h.yaml"),
               "--comparison-policy", str(tmp_path / "p.json"), "--container-image", "img@sha256:" + "c" * 64,
               "--native-binding", "candidate", "--streams", "2", "--dry-run"]
    done = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert "4 plans" in done.stdout and "at most 2 at once" in done.stdout
    assert done.stdout.count("models paired plan") == 4 and done.stdout.count("models paired run") == 4
    assert not (tmp_path / "out").exists()


def _family(tmp_path, name="memory-dev"):
    family = tmp_path / name
    family.mkdir()
    (family / "manifest.json").write_text('{"dataset_id": "' + name + '"}')
    (family / "scenarios.json").write_text("[]")
    return family


def _freeze(directory, spec, resources, *, started=0, reported=False, **changes):
    """What ``colonymind models paired plan`` freezes for ``spec`` (the fields the suite reads), with ``changes``."""
    import hashlib
    family = spec.dataset_dir
    content = hashlib.sha256(b"manifest\0" + (family / "manifest.json").read_bytes() + b"\0scenarios\0"
                             + (family / "scenarios.json").read_bytes()).hexdigest()
    options = {"native_binding": resources.native_binding, "case_ids": ["case-1", "case-2"], "dataset_version": None,
               "dataset_dir": str(family.resolve()), "arms": list(spec.arms), "reference_arm": spec.reference,
               "profiles": None, "temperature": None, "seeds": [], "embedding": None}
    pairs = [{"arms": {arm: {"path": f"runs/{index:03d}-{arm}"} for arm in spec.arms}} for index in (1, 2)]
    manifest = {"label": spec.label, "evidence_mode": "actual_inference", "options": options, "pairs": pairs,
                "dataset": {"source_sha256": content},
                "recipe": {"container": {"image": resources.container_image, "docker_host": resources.docker_host}},
                "comparison": {"policy": json.loads(resources.comparison_policy.read_text())}}
    for key, value in changes.items():
        (options if key in {*options, "repetitions"} else manifest)[key] = value
    (directory / "runs").mkdir(parents=True, exist_ok=True)
    (directory / "paired.json").write_text(json.dumps(manifest))
    members = [member["path"] for pair in pairs for member in pair["arms"].values()]
    for path in members[:started]:
        (directory / path).mkdir(parents=True)
    if reported:
        (directory / "report-001.json").write_text("{}")
    return manifest


def _resources(tmp_path, **changes):
    policy = tmp_path / "policy.json"
    if not policy.exists():
        policy.write_text(json.dumps({"version": "paired-policy-1", "budget_mode": "matched_work",
                                      "budget_policy": {"turns": 8}, "environment": {"endpoint_usage": "shared"}}))
    return paired_suite.Resources(**{"native_config": tmp_path / "h.yaml", "comparison_policy": policy,
                                     "container_image": "img@sha256:" + "e" * 64, "docker_host": None,
                                     "native_binding": "candidate", **changes})


@pytest.mark.parametrize("reported", [True, False])
def test_a_plan_frozen_for_other_faculties_is_refused_not_skipped_or_resumed(tmp_path, reported):
    """A plan's label comes from its family and position, not its arms: the suite for learning names the plan the
    suite for skills made in the same output. Finished, it was skipped as the learning measurement; unfinished, it
    resumed the skills arms. The suite refuses it, and runs nothing."""
    family = _family(tmp_path)
    resources = _resources(tmp_path)
    skills, = paired_suite.specs([family], label="suite", faculties=["skills"])
    learning, = paired_suite.specs([family], label="suite", faculties=["learning"])
    assert skills.label == learning.label
    directory = tmp_path / "out" / skills.label
    _freeze(directory, skills, resources, started=4 if reported else 1, reported=reported)
    assert paired_suite.steps(skills, tmp_path / "out", resources, program=["colonymind"]) == (
        [] if reported else [("run", paired_suite.steps(skills, tmp_path / "out", resources,
                                                        program=["colonymind"])[0][1])])
    with pytest.raises(ValueError, match="arms"):
        paired_suite.steps(learning, tmp_path / "out", resources, program=["colonymind"])
    calls = []
    with pytest.raises(ValueError, match=learning.label):
        paired_suite.run_suite([learning], tmp_path / "out", resources, streams=1,
                               execute=lambda argv, log: calls.append(argv) or 0, program=["colonymind"])
    assert calls == [] and not (tmp_path / "out" / "suite.json").exists()


@pytest.mark.parametrize("change, expected", [
    ({"dataset_version": "paired-v1", "dataset_dir": None}, "dataset"),
    ({"source_sha256": None}, "dataset"),
    ({"reference_arm": "all-on-skills"}, "reference arm"),
    ({"native_binding": "other"}, "native binding"),
    ({"temperature": 0.2}, "temperature"),
    ({"seeds": [7]}, "seeds"),
    ({"repetitions": 3}, "repetitions"),
    ({"embedding": {"base_url": "http://embed.test", "model": "e", "dimensions": 8}}, "embedding"),
    ({"recipe": {"container": {"image": "img@sha256:" + "f" * 64, "docker_host": None}}}, "container image"),
    ({"comparison": {"policy": {"version": "paired-policy-1"}}}, "comparison policy"),
    ({"label": "other"}, "label"),
])
def test_a_plan_frozen_with_another_dataset_or_setting_is_refused(tmp_path, change, expected):
    family = _family(tmp_path)
    resources = _resources(tmp_path)
    spec, = paired_suite.specs([family], label="suite", faculties=["skills"])
    directory = tmp_path / "out" / spec.label
    if "source_sha256" in change:
        manifest = _freeze(directory, spec, resources, started=4, reported=True)
        manifest["dataset"]["source_sha256"] = "0" * 64
        (directory / "paired.json").write_text(json.dumps(manifest))
    else:
        _freeze(directory, spec, resources, started=4, reported=True, **change)
    with pytest.raises(ValueError, match=expected):
        paired_suite.steps(spec, tmp_path / "out", resources, program=["colonymind"])
    assert paired_suite.frozen_mismatch(spec, directory, _resources(tmp_path)) == [expected]


def test_the_cli_refuses_a_suite_whose_output_holds_another_plan(tmp_path):
    family = _family(tmp_path)
    resources = _resources(tmp_path, native_config=tmp_path / "h.yaml")
    skills, = paired_suite.specs([family], label="faculties", faculties=["skills"])
    _freeze(tmp_path / "out" / skills.label, skills, resources, started=4, reported=True)
    command = [sys.executable, "-m", "colonymind", "models", "paired", "suite", "--dataset-dir", str(family),
               "--output", str(tmp_path / "out"), "--native-config", str(resources.native_config),
               "--comparison-policy", str(resources.comparison_policy), "--container-image",
               resources.container_image, "--native-binding", "candidate", "--faculties", "learning"]
    for extra in (["--dry-run"], []):
        done = subprocess.run(command + extra, capture_output=True, text=True, timeout=120)
        assert done.returncode == 2, done.stderr
        assert "another arms" in done.stdout + done.stderr
    assert not (tmp_path / "out" / "suite.json").exists()
    done = subprocess.run(command[:-1] + ["skills", "--dry-run"], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and "refused" not in done.stdout
