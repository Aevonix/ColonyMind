"""The job that builds the distributions checks the version, ties a release tag to it and writes the build stamp.

The job is ``artifacts`` of ``.github/workflows/ci.yml``: a release runs it and publishes what it built. Its three
steps are read from the workflow, and the script of each one is run here as the runner runs it, in a checkout the
test makes, with the runner's two variables set by hand.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
JOB = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"]["artifacts"]["steps"]
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
SCRIPTS = ("scripts/set_version.py", "scripts/write_build_stamp.py")
STAMP = "sidecar/colonymind/build_stamp.json"
CHANGED = "the build stamp says the tree had uncommitted changes: a tag is not built from such a tree"
# git as the tests and the steps run it: no configuration of this machine, and no variable of a runner but the two
# a test gives.
ENV = {key: value for key, value in os.environ.items() if not key.startswith(("GIT_", "GITHUB_"))} | {
    "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "Example", "GIT_AUTHOR_EMAIL": "example@example.com",
    "GIT_COMMITTER_NAME": "Example", "GIT_COMMITTER_EMAIL": "example@example.com",
}


def places():
    """The places that hold the version, as the script that checks them names them."""
    spec = importlib.util.spec_from_file_location("set_version_script", ROOT / SCRIPTS[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [name for name, _ in module.PLACES]


def steps():
    """The three steps of the job: the check, the tie of a tag to the version, the stamp."""
    scripts = [str(step.get("run", "")) for step in JOB]
    found = []
    for what, wanted in (("checks the version", lambda script: script.strip() == f"python {SCRIPTS[0]} --check"),
                         ("ties a tag to the version", lambda script: "GITHUB_REF_NAME" in script),
                         ("writes the build stamp", lambda script: f"python {SCRIPTS[1]}" in script)):
        matching = [index for index, script in enumerate(scripts) if wanted(script)]
        assert len(matching) == 1, f"the artifacts job has {len(matching)} steps where one {what}"
        found.append(JOB[matching[0]])
    return found


def git(tree, *arguments):
    done = subprocess.run(["git", "-C", str(tree), *arguments], env=ENV, text=True, capture_output=True)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def run(step, tree, kind="", name=""):
    """The script of one step, as the runner runs it (``bash -e``) at the top of the checkout."""
    env = ENV | {"PATH": os.pathsep.join([str(tree.parent / "bin"), os.environ.get("PATH", os.defpath)]),
                 "GITHUB_REF_TYPE": kind, "GITHUB_REF_NAME": name}
    return subprocess.run(["bash", "-e", "-c", step["run"]], cwd=tree, env=env, text=True, capture_output=True)


@pytest.fixture
def checkout(tmp_path):
    """A git checkout on the branch ``work``: the version places and the two scripts of this tree, all committed.

    Beside it stands ``python``, the interpreter of the tests, as the runner's setup puts one first on the path.
    """
    root = tmp_path / "checkout"
    names = [*places(), *SCRIPTS, ".gitignore"]
    for name in names:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, root / name)
    (root / "sidecar/colonymind").mkdir(parents=True)
    (root / "sidecar/colonymind/__init__.py").write_text('"""A package."""\n')
    git(root, "init", "-q")
    git(root, "symbolic-ref", "HEAD", "refs/heads/work")
    git(root, "add", "--", *names, "sidecar/colonymind/__init__.py")
    git(root, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "A tree")
    python = tmp_path / "bin/python"
    python.parent.mkdir()
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    python.chmod(0o755)
    return root


def test_the_three_steps_stand_in_this_order_before_anything_is_installed_or_built():
    check, tag, stamp = (JOB.index(step) for step in steps())
    assert [str(step.get("uses", "")).split("@")[0] for step in JOB[:check]] == ["actions/checkout",
                                                                                 "actions/setup-python"]
    assert (tag, stamp) == (check + 1, check + 2)
    [build] = [index for index, step in enumerate(JOB) if "python -m build" in str(step.get("run", ""))]
    assert stamp < build
    assert JOB[build]["run"].count("python -m build --no-isolation --outdir dist ./sidecar") == 1


def test_the_scripts_of_the_three_steps_hold_no_expression_and_read_the_runners_own_variables():
    for step in steps():
        assert "${{" not in step["run"]
        assert "if" not in step, step["name"]   # each step runs in every run; its script decides what it does
    _, tag, stamp = steps()
    assert "${GITHUB_REF_TYPE}" in tag["run"] and "${GITHUB_REF_NAME}" in tag["run"]
    assert "${GITHUB_REF_TYPE}" in stamp["run"]


def test_a_branch_run_passes_all_three_and_the_stamp_holds_the_branch(checkout):
    check, tag, stamp = (run(step, checkout, "branch", "work") for step in steps())
    assert (check.returncode, check.stdout, check.stderr) == (0, VERSION + "\n", "")
    assert (tag.returncode, tag.stdout, tag.stderr) == (0, "", "")
    commit = git(checkout, "rev-parse", "HEAD")
    assert (stamp.returncode, stamp.stderr) == (0, "")
    assert json.loads((checkout / STAMP).read_text()) == {"commit": commit, "branch": "work", "dirty": False}
    # The facts are printed once, by the script that wrote them.
    [line] = (check.stdout + tag.stdout + stamp.stdout).splitlines()[1:]
    assert line.endswith(f"{STAMP}: commit {commit}, branch work, clean")
    assert git(checkout, "status", "--porcelain", "--untracked-files=normal") == ""


def test_a_tag_run_whose_tag_is_v_plus_the_version_passes_and_its_stamp_names_no_branch(checkout):
    git(checkout, "checkout", "-q", "--detach")   # a tag is checked out so: no branch
    check, tag, stamp = (run(step, checkout, "tag", "v" + VERSION) for step in steps())
    assert (check.returncode, check.stdout, check.stderr) == (0, VERSION + "\n", "")
    assert (tag.returncode, tag.stdout, tag.stderr) == (0, f"tag v{VERSION} fits version {VERSION}\n", "")
    commit = git(checkout, "rev-parse", "HEAD")
    assert (stamp.returncode, stamp.stderr) == (0, "")
    assert json.loads((checkout / STAMP).read_text()) == {"commit": commit, "branch": "", "dirty": False}
    [line] = stamp.stdout.splitlines()
    assert line.endswith(f"{STAMP}: commit {commit}, branch (none: detached), clean")


@pytest.mark.parametrize("name", ["v0.0.1", f"v{VERSION}.1", f"vv{VERSION}", "v", f"version-{VERSION}"])
def test_a_tag_that_starts_with_v_and_is_not_v_plus_the_version_fails_with_a_line_that_names_both(checkout, name):
    tag = run(steps()[1], checkout, "tag", name)
    assert (tag.returncode, tag.stdout) == (1, "")
    assert tag.stderr == f"tag {name} does not fit version {VERSION}: a release tag is v plus the version, v{VERSION}\n"


@pytest.mark.parametrize("kind,name", [("branch", "main"), ("branch", "v0.0.1"), ("tag", "nightly"),
                                       ("tag", f"V{VERSION}"), ("", "")])
def test_the_tie_does_nothing_unless_the_run_is_for_a_tag_that_starts_with_v(checkout, kind, name):
    tag = run(steps()[1], checkout, kind, name)
    assert (tag.returncode, tag.stdout, tag.stderr) == (0, "", "")


def test_the_check_fails_and_no_tag_fits_when_a_place_holds_another_version(checkout):
    place = checkout / "plugins/hermes-plugin/plugin.yaml"
    held = place.read_text(encoding="utf-8")
    assert held.count(f"version: {VERSION}\n") == 1
    place.write_text(held.replace(f"version: {VERSION}\n", "version: 0.0.1\n"), encoding="utf-8")
    check, tag, _ = steps()
    done = run(check, checkout, "tag", "v" + VERSION)
    assert (done.returncode, done.stdout) == (1, "")
    assert done.stderr.startswith("the places do not hold one version:\n")
    for name in ("v" + VERSION, "v0.0.1"):
        done = run(tag, checkout, "tag", name)
        assert (done.returncode, done.stdout) == (1, "")


@pytest.mark.parametrize("name,listed", [("NOTES.md", "?? NOTES.md"), ("pyproject.toml", " M pyproject.toml")])
def test_a_tag_run_in_a_tree_with_an_uncommitted_change_fails_at_the_stamp_and_a_branch_run_does_not(checkout, name,
                                                                                                     listed):
    with open(checkout / name, "a", encoding="utf-8") as handle:
        handle.write("\n# a line no commit holds\n")
    check, tag, stamp = steps()
    commit = git(checkout, "rev-parse", "HEAD")
    said = f"{STAMP}: commit {commit}, branch work, changed"
    for step in (check, tag):
        assert run(step, checkout, "tag", "v" + VERSION).returncode == 0
    done = run(stamp, checkout, "tag", "v" + VERSION)
    assert done.returncode == 1
    assert [line.endswith(said) for line in done.stdout.splitlines()] == [True]
    assert done.stderr.splitlines() == [CHANGED, listed]
    done = run(stamp, checkout, "branch", "work")
    assert (done.returncode, done.stderr) == (0, "")
    assert [line.endswith(said) for line in done.stdout.splitlines()] == [True]
    assert json.loads((checkout / STAMP).read_text()) == {"commit": commit, "branch": "work", "dirty": True}
