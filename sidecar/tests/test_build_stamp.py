"""The build stamp: one script writes it, one module reads it, and ``colonymind doctor`` shows it as one row."""
from __future__ import annotations

import itertools
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from colonymind import doctor
from colonymind.doctor import FAIL, PASS, SKIP, WARN, CheckResult

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/write_build_stamp.py"
STAMP = "sidecar/colonymind/build_stamp.json"
COMMIT = "0a1b2c3d4e" * 4
REFUSED = "refused, nothing written: "
NO_STAMP = "no readable build stamp: the commit this install was built from is not recorded"
# git as the tests and the script run it: no configuration of this machine, no repository but the one it is given.
GIT_ENV = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")} | {
    "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "Example", "GIT_AUTHOR_EMAIL": "example@example.com",
    "GIT_COMMITTER_NAME": "Example", "GIT_COMMITTER_EMAIL": "example@example.com",
}


def git(tree, *arguments):
    done = subprocess.run(["git", "-C", str(tree), *arguments], env=GIT_ENV, text=True, capture_output=True)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def stamp(*arguments, env=GIT_ENV):
    return subprocess.run([sys.executable, "-I", str(SCRIPT), *arguments], env=env, text=True, capture_output=True)


def snapshot(root):
    """Every file under ``root`` with its bytes, a repository's own files among them."""
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def make_copy(root):
    """A tree that is no git checkout: the package directory, this repository's ignore file and one more file."""
    (root / "sidecar/colonymind").mkdir(parents=True)
    (root / "sidecar/colonymind/__init__.py").write_text('"""A package."""\n')
    (root / "README.md").write_text("A tree.\n")
    shutil.copyfile(ROOT / ".gitignore", root / ".gitignore")
    return root


def make_checkout(root):
    """The same tree as a git checkout on the branch ``work``, with everything committed."""
    make_copy(root)
    git(root, "init", "-q")
    git(root, "symbolic-ref", "HEAD", "refs/heads/work")
    git(root, "add", ".gitignore", "README.md", "sidecar/colonymind/__init__.py")
    git(root, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "A tree")
    return root


@pytest.fixture(params=["no checkout", "no git program", "inside another checkout"])
def handmade(request, tmp_path):
    """A tree git gives no facts for, and the environment the script runs in."""
    if request.param == "no checkout":
        return make_copy(tmp_path / "copy"), GIT_ENV
    if request.param == "no git program":
        (tmp_path / "bin").mkdir()
        return make_checkout(tmp_path / "checkout"), {**GIT_ENV, "PATH": str(tmp_path / "bin")}
    return make_copy(make_checkout(tmp_path / "outer") / "copies/tree"), GIT_ENV


def put(monkeypatch, directory, content):
    """Make the reader look under the test's own directory, at a file that holds ``content`` (``None``: no file)."""
    from colonymind import build_stamp
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "build_stamp.json"
    monkeypatch.setattr(build_stamp, "PATH", path)
    if content is not None:
        path.write_bytes(content if isinstance(content, bytes) else json.dumps(content).encode())
    return build_stamp


# --- the script -------------------------------------------------------------------------------------------------

def test_a_clean_checkout_gives_its_commit_its_branch_and_clean_and_no_other_file_changes(tmp_path):
    tree = make_checkout(tmp_path / "checkout")
    commit = git(tree, "rev-parse", "HEAD")
    before = snapshot(tmp_path)
    result = stamp("--root", str(tree))
    assert (result.returncode, result.stderr) == (0, "")
    assert result.stdout == f"{tree / STAMP}: commit {commit}, branch work, clean\n"
    after = snapshot(tmp_path)
    assert json.loads(after.pop(f"checkout/{STAMP}")) == {"commit": commit, "branch": "work", "dirty": False}
    assert len(commit) == 40 and after == before


def test_git_ignores_the_stamp_and_a_second_run_puts_a_whole_new_file_in_its_place(tmp_path):
    tree = make_checkout(tmp_path / "checkout")
    assert stamp("--root", str(tree)).returncode == 0
    assert git(tree, "status", "--porcelain") == ""  # the ignore file names the stamp, so the tree is still clean
    first = (tree / STAMP).read_bytes()
    (tree / "README.md").write_text("A tree, changed.\n")
    with open(tree / STAMP, "rb") as reader:  # a reader that opened the stamp before the second run
        result = stamp("--root", str(tree))
        assert reader.read() == first  # still reads the first stamp, all of it
    assert (result.returncode, result.stderr) == (0, "")
    assert json.loads((tree / STAMP).read_bytes())["dirty"] is True
    assert sorted(path.name for path in (tree / "sidecar/colonymind").iterdir()) == ["__init__.py", "build_stamp.json"]


@pytest.mark.parametrize("name", ["README.md", "sidecar/colonymind/added.py"])
def test_a_checkout_with_a_change_says_so(tmp_path, name):
    """A changed file git knows, and a file it does not know: a build from this tree would hold either."""
    tree = make_checkout(tmp_path / "checkout")
    (tree / name).write_text("changed\n")
    result = stamp("--root", str(tree))
    assert (result.returncode, result.stderr) == (0, "")
    assert result.stdout.endswith(", branch work, changed\n")
    assert json.loads((tree / STAMP).read_bytes()) == {"commit": git(tree, "rev-parse", "HEAD"), "branch": "work",
                                                       "dirty": True}


def test_a_detached_checkout_has_an_empty_branch(tmp_path):
    tree = make_checkout(tmp_path / "checkout")
    git(tree, "checkout", "-q", "--detach")
    result = stamp("--root", str(tree))
    assert (result.returncode, result.stderr) == (0, "")
    assert json.loads((tree / STAMP).read_bytes()) == {"commit": git(tree, "rev-parse", "HEAD"), "branch": "",
                                                       "dirty": False}


def test_without_git_the_facts_come_from_the_arguments_and_no_other_file_changes(handmade, tmp_path):
    tree, env = handmade
    before = snapshot(tmp_path)
    result = stamp("--root", str(tree), "--commit", COMMIT, "--branch", "work", "--clean", env=env)
    assert (result.returncode, result.stderr) == (0, "")
    assert result.stdout == f"{tree / STAMP}: commit {COMMIT}, branch work, clean\n"
    after = snapshot(tmp_path)
    written = after.pop((tree / STAMP).relative_to(tmp_path).as_posix())
    assert json.loads(written) == {"commit": COMMIT, "branch": "work", "dirty": False}
    assert after == before


def test_the_arguments_can_say_detached_and_changed_and_take_a_longer_full_id(tmp_path):
    tree = make_copy(tmp_path / "copy")
    result = stamp("--root", str(tree), "--commit", COMMIT + COMMIT[:24], "--branch", "", "--dirty")
    assert (result.returncode, result.stderr) == (0, "")
    assert json.loads((tree / STAMP).read_bytes()) == {"commit": COMMIT + COMMIT[:24], "branch": "", "dirty": True}


def test_without_git_and_without_arguments_the_script_refuses_and_writes_nothing(handmade, tmp_path):
    tree, env = handmade
    before = snapshot(tmp_path)
    result = stamp("--root", str(tree), env=env)
    assert (result.returncode, result.stdout) == (2, "")
    assert result.stderr == (REFUSED + "this tree is not the top of a git checkout that git can read: "
                             "give --commit ID, --branch NAME and --clean or --dirty\n")
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("arguments,reason", [
    (("--commit", COMMIT), "a tree that is not a git checkout needs every fact: no --branch, no --clean or --dirty"),
    (("--commit", COMMIT, "--branch", "work"), "a tree that is not a git checkout needs every fact: "
                                               "no --clean or --dirty"),
    (("--branch", "", "--clean"), "a tree that is not a git checkout needs every fact: no --commit"),
    (("--commit", COMMIT[:12], "--branch", "work", "--clean"), "--commit takes the full commit id, not "
                                                               f"{COMMIT[:12]!r}\n"),
    (("--commit", COMMIT.upper(), "--branch", "work", "--clean"), "--commit takes the full commit id, not "),
    (("--commit", "unknown", "--branch", "work", "--dirty"), "--commit takes the full commit id, not 'unknown'\n"),
])
def test_a_missing_fact_or_a_commit_that_is_not_a_full_id_is_refused_and_nothing_is_written(tmp_path, arguments,
                                                                                            reason):
    tree = make_copy(tmp_path / "copy")
    before = snapshot(tmp_path)
    result = stamp("--root", str(tree), *arguments)
    assert (result.returncode, result.stdout) == (2, "")
    assert result.stderr.startswith(REFUSED + reason)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("arguments", [("--commit", COMMIT, "--branch", "work", "--clean"), ("--dirty",),
                                       ("--branch", "")])
def test_arguments_are_refused_in_a_checkout_which_gives_the_facts_itself(tmp_path, arguments):
    tree = make_checkout(tmp_path / "checkout")
    before = snapshot(tmp_path)
    result = stamp("--root", str(tree), *arguments)
    assert (result.returncode, result.stdout) == (2, "")
    assert result.stderr == (REFUSED + "this tree is a git checkout, which gives the facts itself: "
                             "give none as arguments\n")
    assert snapshot(tmp_path) == before


def test_a_checkout_that_has_no_commit_yet_is_refused(tmp_path):
    tree = make_copy(tmp_path / "copy")
    git(tree, "init", "-q")
    before = snapshot(tmp_path)
    result = stamp("--root", str(tree))
    assert (result.returncode, result.stdout) == (2, "")
    assert result.stderr == REFUSED + "git names no commit for this tree\n"
    assert snapshot(tmp_path) == before


def test_a_tree_without_the_package_directory_is_refused_and_nothing_is_made(tmp_path):
    result = stamp("--root", str(tmp_path), "--commit", COMMIT, "--branch", "work", "--clean")
    assert (result.returncode, result.stdout) == (2, "")
    assert result.stderr == f"{REFUSED}{tmp_path} has no sidecar/colonymind directory\n"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes into any directory")
def test_a_stamp_that_cannot_be_written_is_refused_and_the_one_before_stays(tmp_path):
    tree = make_copy(tmp_path / "copy")
    arguments = ("--root", str(tree), "--commit", COMMIT, "--branch", "work")
    assert stamp(*arguments, "--clean").returncode == 0
    before = snapshot(tmp_path)
    (tree / "sidecar/colonymind").chmod(0o555)
    try:
        result = stamp(*arguments, "--dirty")
    finally:
        (tree / "sidecar/colonymind").chmod(0o755)
    assert (result.returncode, result.stdout) == (2, "")
    assert result.stderr == f"{REFUSED}{tree / STAMP} cannot be written (PermissionError)\n"
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("name", ["work\nmain", "work\x1b[2J", "work‮main", "w" * 256, b"caf\xe9"],
                         ids=["line-break", "escape", "turned-around", "256-characters", "not-utf-8"])
def test_a_branch_argument_that_does_not_print_or_is_too_long_is_refused_and_nothing_is_written(tmp_path, name):
    tree = make_copy(tmp_path / "copy")
    before = snapshot(tmp_path)
    result = stamp("--root", str(tree), "--commit", COMMIT, "--branch", name, "--clean")
    assert (result.returncode, result.stdout) == (2, "")
    assert result.stderr == REFUSED + "--branch takes a name of at most 255 characters that all print\n"
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("name", ["w" * 255, "tópico/één", "work main"])
def test_a_branch_argument_of_up_to_255_characters_that_all_print_is_written(tmp_path, name):
    tree = make_copy(tmp_path / "copy")
    result = stamp("--root", str(tree), "--commit", COMMIT, "--branch", name, "--clean")
    assert (result.returncode, result.stderr) == (0, "")
    assert json.loads((tree / STAMP).read_bytes()) == {"commit": COMMIT, "branch": name, "dirty": False}


@pytest.mark.parametrize("name", ["work\u0085", "work‮main", "a/" * 150 + "b"],
                         ids=["next-line", "turned-around", "301-characters"])
def test_a_branch_of_the_checkout_that_does_not_print_or_is_too_long_is_refused_and_nothing_is_written(tmp_path,
                                                                                                       name):
    """Names git takes for a branch. None of them is one a stamp holds."""
    tree = make_checkout(tmp_path / "checkout")
    git(tree, "checkout", "-q", "-b", name)
    before = snapshot(tmp_path)
    result = stamp("--root", str(tree))
    assert (result.returncode, result.stdout) == (2, "")
    assert result.stderr == (REFUSED + "the branch name git gives has more than 255 characters or one that does not "
                             "print\n")
    assert snapshot(tmp_path) == before


def test_a_branch_of_the_checkout_that_is_not_utf_8_is_refused_in_one_line_and_nothing_is_written(tmp_path):
    tree = make_checkout(tmp_path / "checkout")
    git(tree, "checkout", "-q", "-b", b"caf\xe9")
    before = snapshot(tmp_path)
    result = stamp("--root", str(tree))
    assert (result.returncode, result.stdout) == (2, "")
    assert result.stderr == REFUSED + "the branch name git gives is not UTF-8\n"
    assert snapshot(tmp_path) == before


def make_other(root):
    """Another repository: on the branch ``elsewhere``, one commit further, with a file the first one lacks."""
    make_checkout(root)
    git(root, "checkout", "-q", "-b", "elsewhere")
    (root / "NOTES.md").write_text("Notes.\n")
    git(root, "add", "NOTES.md")
    git(root, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "Notes")
    return root


@pytest.mark.parametrize("names", [("GIT_DIR",), ("GIT_DIR", "GIT_WORK_TREE"), ("GIT_INDEX_FILE",)])
def test_a_checkout_gives_its_own_facts_whatever_repository_the_environment_names(tmp_path, names):
    """With these variables set, git by itself answers with the other repository's commit, branch or index."""
    tree, other = make_checkout(tmp_path / "checkout"), make_other(tmp_path / "other")
    commit = git(tree, "rev-parse", "HEAD")
    assert git(other, "rev-parse", "HEAD") != commit
    aimed = {"GIT_DIR": str(other / ".git"), "GIT_WORK_TREE": str(other), "GIT_INDEX_FILE": str(other / ".git/index")}
    before = snapshot(tmp_path)
    result = stamp("--root", str(tree), env=GIT_ENV | {name: aimed[name] for name in names})
    assert (result.returncode, result.stderr) == (0, "")
    assert result.stdout == f"{tree / STAMP}: commit {commit}, branch work, clean\n"
    after = snapshot(tmp_path)
    assert json.loads(after.pop(f"checkout/{STAMP}")) == {"commit": commit, "branch": "work", "dirty": False}
    assert after == before


@pytest.mark.parametrize("names", [("GIT_DIR",), ("GIT_DIR", "GIT_WORK_TREE")])
def test_a_tree_that_is_no_checkout_is_refused_whatever_repository_the_environment_names(tmp_path, names):
    tree, other = make_copy(tmp_path / "copy"), make_other(tmp_path / "other")
    aimed = {"GIT_DIR": str(other / ".git"), "GIT_WORK_TREE": str(tree)}
    before = snapshot(tmp_path)
    result = stamp("--root", str(tree), env=GIT_ENV | {name: aimed[name] for name in names})
    assert (result.returncode, result.stdout) == (2, "")
    assert result.stderr == (REFUSED + "this tree is not the top of a git checkout that git can read: "
                             "give --commit ID, --branch NAME and --clean or --dirty\n")
    assert snapshot(tmp_path) == before


def test_git_ignores_the_file_a_stamp_is_first_written_into_and_no_other_name(tmp_path):
    """Left behind by a run that was cut off, that file does not make the next stamp say the tree has changes."""
    tree = make_checkout(tmp_path / "checkout")
    (tree / f"{STAMP}.4242.tmp").write_text("{")
    result = stamp("--root", str(tree))
    assert (result.returncode, result.stderr) == (0, "")
    assert json.loads((tree / STAMP).read_bytes())["dirty"] is False
    assert git(tree, "status", "--porcelain") == ""
    others = [f"{STAMP}.tmp", f"{STAMP}.4242.bak", "sidecar/colonymind/other.json.4242.tmp",
              f"sidecar/{Path(STAMP).name}.4242.tmp"]
    for name in others:
        (tree / name).write_text("{")
    assert sorted(git(tree, "status", "--porcelain").splitlines()) == sorted(f"?? {name}" for name in others)


# The script as the command line runs it, on a standard output that has no character outside ASCII.
ASCII_ONLY = """
import runpy, sys
sys.stdout.reconfigure(encoding="ascii", errors="strict")
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def test_the_last_line_is_printed_on_an_output_that_lacks_a_character_of_the_branch_name(tmp_path):
    tree = make_copy(tmp_path / "copy")
    result = subprocess.run([sys.executable, "-I", "-c", ASCII_ONLY, str(SCRIPT), "--root", str(tree),
                             "--commit", COMMIT, "--branch", "tópico", "--clean"],
                            env=GIT_ENV, text=True, capture_output=True)
    assert (result.returncode, result.stderr) == (0, "")
    assert result.stdout == f"{tree / STAMP}: commit {COMMIT}, branch t\\xf3pico, clean\n"
    assert json.loads((tree / STAMP).read_bytes()) == {"commit": COMMIT, "branch": "tópico", "dirty": False}


# --- the reader -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("held", [
    {"commit": COMMIT, "branch": "work", "dirty": False},
    {"commit": COMMIT, "branch": "", "dirty": True},
    {"commit": COMMIT + COMMIT[:24], "branch": "topic/one", "dirty": False},
])
def test_the_reader_returns_a_stamp_that_is_present(monkeypatch, tmp_path, held):
    assert put(monkeypatch, tmp_path, held).read() == held


def test_the_reader_returns_what_the_script_wrote_and_looks_beside_itself_for_that_file(monkeypatch, tmp_path):
    from colonymind import build_stamp
    assert build_stamp.PATH.parent == Path(build_stamp.__file__).parent == Path(doctor.__file__).parent
    assert build_stamp.PATH.name == Path(STAMP).name
    tree = make_checkout(tmp_path / "checkout")
    assert stamp("--root", str(tree)).returncode == 0
    monkeypatch.setattr(build_stamp, "PATH", tree / STAMP)
    assert build_stamp.read() == {"commit": git(tree, "rev-parse", "HEAD"), "branch": "work", "dirty": False}


def test_the_reader_returns_nothing_when_the_file_is_absent(monkeypatch, tmp_path):
    assert put(monkeypatch, tmp_path, None).read() is None


@pytest.mark.parametrize("content", [b"", b"{", b'{"commit": ', b"not a stamp", b"\xff\xfe\x00{", b"[" * 100_000],
                         ids=["empty", "cut-short", "half-a-pair", "words", "not-text", "nested-too-deep"])
def test_the_reader_returns_nothing_when_the_file_is_damaged(monkeypatch, tmp_path, content):
    assert put(monkeypatch, tmp_path, content).read() is None


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads any file")
def test_the_reader_returns_nothing_when_the_file_may_not_be_read(monkeypatch, tmp_path):
    reader = put(monkeypatch, tmp_path, {"commit": COMMIT, "branch": "work", "dirty": False})
    reader.PATH.chmod(0o000)
    try:
        assert reader.read() is None
    finally:
        reader.PATH.chmod(0o644)


def test_the_reader_returns_nothing_when_a_directory_stands_where_the_file_should_be(monkeypatch, tmp_path):
    reader = put(monkeypatch, tmp_path, None)
    monkeypatch.setattr(reader, "PATH", tmp_path)
    assert reader.read() is None


@pytest.mark.parametrize("held", [
    None, [], "work", 7, {},
    {"commit": COMMIT, "branch": "work"},
    {"commit": COMMIT, "dirty": False},
    {"commit": COMMIT, "branch": "work", "dirty": False, "version": "0.0.1"},
    [COMMIT, "work", False],
])
def test_the_reader_returns_nothing_when_the_file_is_not_the_expected_shape(monkeypatch, tmp_path, held):
    assert put(monkeypatch, tmp_path, held).read() is None


@pytest.mark.parametrize("fact,value", [
    ("commit", None), ("commit", 7), ("commit", [COMMIT]), ("commit", ""), ("commit", COMMIT[:12]),
    ("commit", COMMIT.upper()), ("commit", "g" * 40), ("commit", COMMIT + "\n"),
    ("branch", None), ("branch", 7), ("branch", ["work"]), ("branch", False),
    ("dirty", None), ("dirty", 0), ("dirty", 1), ("dirty", "false"), ("dirty", "yes"),
])
def test_the_reader_returns_nothing_when_a_fact_has_the_wrong_type_or_the_commit_is_not_a_full_id(
        monkeypatch, tmp_path, fact, value):
    held = {"commit": COMMIT, "branch": "work", "dirty": False}
    assert put(monkeypatch, tmp_path, held).read() == held
    assert put(monkeypatch, tmp_path, {**held, fact: value}).read() is None


@pytest.mark.parametrize("branch", ["work\nmain", "work\x1b[2J", "work\u0085", "work‮main", "caf\udce9",
                                    "w" * 256],
                         ids=["line-break", "escape", "next-line", "turned-around", "not-utf-8", "256-characters"])
def test_the_reader_returns_nothing_when_the_branch_does_not_print_or_is_too_long(monkeypatch, tmp_path, branch):
    held = {"commit": COMMIT, "branch": "w" * 255, "dirty": False}
    assert put(monkeypatch, tmp_path, held).read() == held
    assert put(monkeypatch, tmp_path, {**held, "branch": branch}).read() is None


def test_the_reader_is_a_module_of_the_package_that_importing_the_package_does_not_load():
    code = ("import importlib.util, sys, colonymind\n"
            "print(importlib.util.find_spec('colonymind.build_stamp') is not None,"
            " 'colonymind.build_stamp' in sys.modules)")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT / "sidecar", text=True, capture_output=True)
    assert (result.returncode, result.stdout.split()) == (0, ["True", "False"]), result.stderr


# --- the row in ``colonymind doctor`` ----------------------------------------------------------------------------

#: What the stamp file holds (``None``: no file), then the status and the detail of the row.
CASES = {
    "clean": ({"commit": COMMIT, "branch": "work", "dirty": False}, PASS, f"commit {COMMIT[:12]}, branch work, clean"),
    "detached": ({"commit": COMMIT, "branch": "", "dirty": False}, PASS, f"commit {COMMIT[:12]}, detached, clean"),
    "changed": ({"commit": COMMIT, "branch": "work", "dirty": True}, WARN,
                f"commit {COMMIT[:12]}, branch work, changed: the tree had uncommitted changes"),
    "absent": (None, WARN, NO_STAMP),
    "damaged": (b'{"commit": ', WARN, NO_STAMP),
    "wrong types": ({"commit": COMMIT, "branch": "work", "dirty": "no"}, WARN, NO_STAMP),
    "branch that does not print": ({"commit": COMMIT, "branch": "work\x1b[2J", "dirty": False}, WARN, NO_STAMP),
}

#: The remedy of a row without a readable stamp, and of the row of a stamp that says the tree had changes.
WRITE_ONE = ("in a source checkout run 'python scripts/write_build_stamp.py'; an install from a release wheel carries "
             "the stamp")
BUILD_AGAIN = ("commit the changes or take them out of the tree, run 'python scripts/write_build_stamp.py' again, and "
               "build the wheel again if this install came from one")
#: The remedy of each warning row. A passed row has none.
REMEDY = {"changed": BUILD_AGAIN, "absent": WRITE_ONE, "damaged": WRITE_ONE, "wrong types": WRITE_ONE,
          "branch that does not print": WRITE_ONE}


def verdict(results):
    """What a caller of ``colonymind doctor`` gets: the exit status, and ``ok`` of the JSON output."""
    return doctor.exit_code(results), doctor.results_to_json(results)["ok"]


def other_rows():
    """Every mix of up to three passed, four skipped, one failed and one warning row: among them the runs in which
    one more skipped row, or one more passed row, changes ``ok``."""
    for passed, skipped, failed, warned in itertools.product(range(4), range(5), range(2), range(2)):
        yield ([CheckResult("passed", PASS)] * passed + [CheckResult("skipped", SKIP)] * skipped
               + [CheckResult("failed", FAIL)] * failed + [CheckResult("warned", WARN)] * warned)


@pytest.mark.parametrize("case", CASES)
def test_doctor_shows_the_stamp_as_one_row_among_the_local_checks(monkeypatch, tmp_path, case):
    content, status, detail = CASES[case]
    put(monkeypatch, tmp_path, content)
    row = doctor.check_build_stamp()
    assert row == CheckResult("build-stamp", status, detail=detail, remedy=REMEDY.get(case, ""))
    assert [result for name, check in doctor.LOCAL_CHECKS if name == "build-stamp"
            for result in doctor._run(name, check)] == [row]
    assert f"  {doctor._ICONS[status]} build-stamp  {detail}" in doctor.format_report([row], color=False).splitlines()


def shown(row):
    """The lines of the report that are the row's own: its status line, and the remedy line when it has one."""
    lines = doctor.format_report([row], color=False).splitlines()
    first = lines.index(f"  {doctor._ICONS[row.status]} build-stamp  {row.detail}")
    return lines[first:lines.index("", first)]


@pytest.mark.parametrize("case", ["absent", "damaged", "wrong types", "branch that does not print"])
def test_a_row_without_a_stamp_names_the_script_that_writes_one_and_the_wheel_that_carries_one(monkeypatch, tmp_path,
                                                                                               case):
    put(monkeypatch, tmp_path, CASES[case][0])
    row = doctor.check_build_stamp()
    assert (row.status, row.detail, row.remedy) == (WARN, NO_STAMP, WRITE_ONE)
    assert SCRIPT.is_file() and f"'python {SCRIPT.relative_to(ROOT).as_posix()}'" in row.remedy
    assert row.remedy.endswith("an install from a release wheel carries the stamp")
    assert shown(row)[1:] == [f"       -> {WRITE_ONE}"]
    assert doctor.results_to_json([row])["results"] == [
        {"name": "build-stamp", "status": WARN, "detail": NO_STAMP, "remedy": WRITE_ONE}]
    assert all(verdict(rows + [row]) == verdict(rows) for rows in other_rows())


def test_the_row_of_a_changed_tree_says_how_a_stamp_without_changes_is_had(monkeypatch, tmp_path):
    content, status, detail = CASES["changed"]
    put(monkeypatch, tmp_path, content)
    row = doctor.check_build_stamp()
    assert (row.status, row.detail, row.remedy) == (WARN, detail, BUILD_AGAIN)
    assert SCRIPT.is_file() and f"'python {SCRIPT.relative_to(ROOT).as_posix()}'" in row.remedy
    assert shown(row)[1:] == [f"       -> {BUILD_AGAIN}"]
    assert doctor.results_to_json([row])["results"] == [
        {"name": "build-stamp", "status": WARN, "detail": detail, "remedy": BUILD_AGAIN}]
    assert all(verdict(rows + [row]) == verdict(rows) for rows in other_rows())


@pytest.mark.parametrize("case", ["clean", "detached"])
def test_a_passed_row_has_no_remedy_and_shows_one_line(monkeypatch, tmp_path, case):
    put(monkeypatch, tmp_path, CASES[case][0])
    row = doctor.check_build_stamp()
    assert (row.status, row.remedy) == (PASS, "")
    assert len(shown(row)) == 1


def test_the_row_stands_right_after_the_adapter_version():
    names = [name for name, _ in doctor.LOCAL_CHECKS]
    assert names.count("build-stamp") == 1
    assert names.index("build-stamp") == names.index("adapter-version") + 1


@pytest.mark.parametrize("case", ["absent", "damaged", "wrong types", "changed"])
def test_a_missing_unreadable_or_changed_stamp_never_changes_the_exit_status_or_ok(monkeypatch, tmp_path, case):
    """Compared with the same run without the row. Any other status would change one of the two in some run."""
    put(monkeypatch, tmp_path, CASES[case][0])
    row = doctor.check_build_stamp()
    for rows in other_rows():
        assert verdict(rows + [row]) == verdict(rows), rows
    for status in (PASS, SKIP, FAIL):
        assert any(verdict(rows + [CheckResult("build-stamp", status)]) != verdict(rows) for rows in other_rows())


@pytest.mark.parametrize("case", ["clean", "detached"])
def test_a_clean_stamp_is_a_passed_row_and_never_changes_the_exit_status(monkeypatch, tmp_path, case):
    put(monkeypatch, tmp_path, CASES[case][0])
    row = doctor.check_build_stamp()
    assert row.status == PASS
    for rows in other_rows():
        assert doctor.exit_code(rows + [row]) == doctor.exit_code(rows), rows


@pytest.mark.parametrize("other,expected", [(PASS, (0, True)), (FAIL, (1, False))])
def test_a_doctor_run_gives_the_same_exit_status_and_ok_with_and_without_a_stamp(monkeypatch, tmp_path, other,
                                                                                 expected):
    """A whole run: the stamp's own row, beside stand-ins for every other check that all pass or all fail."""
    def stand_in(*arguments):
        return CheckResult("other", other)
    monkeypatch.setattr(doctor, "LOCAL_CHECKS", [(name, check if name == "build-stamp" else stand_in)
                                                 for name, check in doctor.LOCAL_CHECKS])
    monkeypatch.setattr(doctor, "check_sidecar", stand_in)
    monkeypatch.setattr(doctor, "check_semantic_recall", stand_in)
    verdicts = {}
    for case, (content, status, _) in CASES.items():
        put(monkeypatch, tmp_path / case, content)
        results = doctor.run_doctor("http://127.0.0.1:7777", "key", 5)
        assert [result.status for result in results if result.name == "build-stamp"] == [status]
        verdicts[case] = verdict(results)
    assert verdicts == dict.fromkeys(CASES, expected)
