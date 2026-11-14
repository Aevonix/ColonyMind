"""One command checks the version in every tracked place that holds it, and sets it there and nowhere else."""
from pathlib import Path
import errno
import importlib.util
import os
import shutil
import stat
import subprocess
import sys
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/set_version.py"
# Every place, with its version line as the file writes it.
LINES = {
    "pyproject.toml": 'version = "{}"',
    "sidecar/pyproject.toml": 'version = "{}"',
    "plugins/hermes-plugin/__init__.py": '__version__ = "{}"',
    "plugins/instinct-gates/__init__.py": '__version__ = "{}"',
    "plugins/hermes-plugin/plugin.yaml": "version: {}",
    "plugins/instinct-gates/plugin.yaml": "version: {}",
    "plugins/colonymind-hands/plugin.yaml": "version: {}",
    "plugins/colonymind-memory/plugin.yaml": "version: {}",
}
PLACES = tuple(LINES)
BYSTANDER = "plugins/feeds-manage/plugin.yaml"   # a manifest with a version of its own: not a place
CURRENT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
# Versions below any this tree will hold again, so a test never sets the version a place already has.
OTHER, RELEASE, CANDIDATE = "0.0.1", "0.0.2", "0.0.3rc10"


def run(*arguments):
    return subprocess.run([sys.executable, "-I", str(SCRIPT), *arguments], text=True, capture_output=True)


LIMIT = 1 << 16   # longer than any place: a place a test makes longer than this cannot be written whole
# The script as the command line runs it, in a process that cannot make a file longer than its first argument.
LIMITED = """
import resource, runpy, sys
limit = int(sys.argv[1])
resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def run_limited(*arguments):
    return subprocess.run([sys.executable, "-I", "-c", LIMITED, str(LIMIT), str(SCRIPT), *arguments],
                          text=True, capture_output=True)


def load():
    """The script as a module, for a test that stands in for a call the script makes."""
    spec = importlib.util.spec_from_file_location("set_version_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def rewrite(tree, name, line):
    """Put ``line`` where the version line of one place stands."""
    held = (LINES[name].format(CURRENT) + "\n").encode()
    data = (tree / name).read_bytes()
    assert data.count(held) == 1
    (tree / name).write_bytes(data.replace(held, line.encode() + b"\n"))


@pytest.fixture
def tree(tmp_path):
    """A copy of this repository's version places, beside one file that is no place."""
    root = tmp_path / "tree"
    for name in (*PLACES, BYSTANDER):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, root / name)
    return root


def test_the_check_passes_on_this_tree():
    result = run("--check")
    assert (result.returncode, result.stdout, result.stderr) == (0, f"{CURRENT}\n", "")


@pytest.mark.parametrize("version", [RELEASE, CANDIDATE])
def test_set_writes_the_version_into_every_place_and_changes_nothing_else(tree, version):
    before = snapshot(tree)
    result = run("--root", str(tree), version)
    assert (result.returncode, result.stdout, result.stderr) == (0, f"{version}\n", "")
    after = snapshot(tree)
    assert after.keys() == before.keys()
    assert after[BYSTANDER] == before[BYSTANDER]
    for name in PLACES:
        old, new = before[name].splitlines(keepends=True), after[name].splitlines(keepends=True)
        changed = [index for index, (was, now) in enumerate(zip(old, new, strict=True)) if was != now]
        assert len(changed) == 1, name
        assert old[changed[0]] == (LINES[name].format(CURRENT) + "\n").encode(), name
        assert new[changed[0]] == (LINES[name].format(version) + "\n").encode(), name
    assert run("--root", str(tree), "--check").stdout == f"{version}\n"


def test_set_keeps_every_other_byte_of_a_place_and_brings_differing_places_together(tree):
    """Line endings, spacing in other lines and a missing final newline are not the script's to change."""
    place = tree / PLACES[-1]
    place.write_bytes(b"name: example\r\nversion: " + OTHER.encode() + b"\r\nnote:   kept  \r\nlast: line")
    result = run("--root", str(tree), RELEASE)
    assert (result.returncode, result.stdout) == (0, f"{RELEASE}\n")
    assert place.read_bytes() == b"name: example\r\nversion: " + RELEASE.encode() + b"\r\nnote:   kept  \r\nlast: line"


@pytest.mark.parametrize("version", [
    "", "1", "1.13", "1.13.0.0", "v1.13.0", "1.13.0rc", "1.13.0-rc5", "1.13.0.rc5", "1.13.0RC5", "1.13.0rc05",
    "01.13.0", "1.13.0.post1", "1.13.0a1", "1.13.0.dev1", "1.13.0+local", "1.13.0 ", "1.13.0\n", "1.13.x",
])
def test_a_string_that_is_not_a_version_is_refused_and_no_file_changes(tree, version):
    before = snapshot(tree)
    result = run("--root", str(tree), version)
    assert (result.returncode, result.stdout) == (2, "")
    assert f"refused: {version!r} is not a version of the form X.Y.Z or X.Y.ZrcN" in result.stderr
    assert snapshot(tree) == before


@pytest.mark.parametrize("name", PLACES)
def test_a_place_that_is_missing_is_refused_and_no_file_changes(tree, name):
    (tree / name).unlink()
    before = snapshot(tree)
    result = run("--root", str(tree), RELEASE)
    assert (result.returncode, result.stdout) == (2, "")
    assert result.stderr == f"refused, no file changed:\n  {name}: cannot be read (FileNotFoundError)\n"
    assert snapshot(tree) == before


@pytest.mark.parametrize("name,line,reason", [
    ("pyproject.toml", "release = \"{}\"", "has no version line"),
    ("sidecar/pyproject.toml", "version = '{}'", "does not look as expected: \"version = '{}'\""),
    ("plugins/hermes-plugin/__init__.py", "__version__ = \"{}\"  # held", "does not look as expected: "),
    ("plugins/instinct-gates/__init__.py", "__version__ = ('{}')", "does not look as expected: "),
    ("plugins/hermes-plugin/plugin.yaml", "version: \"{}\"", "does not look as expected: 'version: \"{}\"'"),
    ("plugins/instinct-gates/plugin.yaml", "version:  {}", "does not look as expected: "),
    ("plugins/colonymind-hands/plugin.yaml", "version: {} # held", "does not look as expected: "),
    ("plugins/colonymind-memory/plugin.yaml", "version: {}\nversion: {}", "has more than one version line: lines 2, 3"),
])
def test_a_place_whose_line_is_not_as_expected_is_refused_and_no_file_changes(tree, name, line, reason):
    """The last place is among them: every earlier one is still as it was."""
    rewrite(tree, name, line.replace("{}", CURRENT))
    before = snapshot(tree)
    result = run("--root", str(tree), RELEASE)
    assert (result.returncode, result.stdout) == (2, "")
    assert result.stderr.startswith(f"refused, no file changed:\n  {name}: ")
    assert reason.replace("{}", CURRENT) in result.stderr
    assert snapshot(tree) == before


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes into any file")
def test_a_place_that_cannot_be_written_is_refused_and_no_file_changes(tree):
    before = snapshot(tree)
    (tree / PLACES[-1]).chmod(0o444)
    try:
        result = run("--root", str(tree), RELEASE)
    finally:
        (tree / PLACES[-1]).chmod(0o644)
    assert (result.returncode, result.stdout) == (2, "")
    assert result.stderr == f"refused, no file changed:\n  {PLACES[-1]}: cannot be written\n"
    assert snapshot(tree) == before


@pytest.mark.parametrize("name", [PLACES[0], PLACES[len(PLACES) // 2], PLACES[-1]])
def test_a_write_that_fails_leaves_every_file_as_it_was_and_none_beside_them(tree, name):
    """The first, a middle and the last place: each place written before the one that fails is put back."""
    with open(tree / name, "ab") as handle:
        handle.write(b"\n# " + b"-" * LIMIT + b"\n")
    before = snapshot(tree)
    result = run_limited("--root", str(tree), RELEASE)
    assert (result.returncode, result.stdout) == (2, "")
    assert result.stderr == (f"failed: {name} could not be written ({os.strerror(errno.EFBIG)}); "
                             "every place holds what it held before\n")
    assert snapshot(tree) == before


@pytest.mark.parametrize("stuck", [None, PLACES[0], PLACES[3]])
def test_a_place_that_cannot_be_put_back_is_named_with_what_every_place_now_holds(tree, capsys, monkeypatch, stuck):
    """The last place cannot take its new file. With ``stuck``, that place cannot take its old bytes back either."""
    script, real, asked = load(), os.replace, []

    def replace(source, target):
        name = Path(target).relative_to(tree.resolve()).as_posix()
        asked.append(name)
        if name == PLACES[-1] or (name == stuck and asked.count(name) == 2):
            raise OSError(errno.EIO, os.strerror(errno.EIO))
        real(source, target)

    before = snapshot(tree)
    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", replace)
        status = script.main(["--root", str(tree), RELEASE])
    printed, after = capsys.readouterr(), snapshot(tree)
    failure = f"failed: {PLACES[-1]} could not be written ({os.strerror(errno.EIO)})"
    assert (status, printed.out) == (2, "")
    assert after.keys() == before.keys()
    if stuck is None:
        assert printed.err == f"{failure}; every place holds what it held before\n"
        assert after == before
        return
    assert printed.err.splitlines() == [
        f"{failure}, and not every place could be put back; the places now hold:",
        *(f"  {name}: {RELEASE if name == stuck else CURRENT}" for name in PLACES),
    ]
    old, new = (LINES[stuck].format(version).encode() for version in (CURRENT, RELEASE))
    assert after == before | {stuck: before[stuck].replace(old, new)}


def test_set_keeps_the_mode_of_every_place(tree):
    """Modes that a new file does not get by itself."""
    modes = dict(zip(PLACES, (0o600, 0o664, 0o640, 0o755, 0o660, 0o644, 0o700, 0o666), strict=True))
    for name, mode in modes.items():
        (tree / name).chmod(mode)
    assert run("--root", str(tree), RELEASE).returncode == 0
    assert {name: stat.S_IMODE((tree / name).stat().st_mode) for name in PLACES} == modes


def test_a_place_that_is_a_link_is_written_where_it_points_and_stays_a_link(tree, tmp_path):
    aside = tmp_path / "aside"
    aside.mkdir()
    shutil.copyfile(tree / PLACES[-1], aside / "plugin.yaml")
    (tree / PLACES[-1]).unlink()
    (tree / PLACES[-1]).symlink_to(aside / "plugin.yaml")
    result = run("--root", str(tree), RELEASE)
    assert (result.returncode, result.stdout, result.stderr) == (0, f"{RELEASE}\n", "")
    assert (tree / PLACES[-1]).is_symlink()
    assert [path.name for path in aside.iterdir()] == ["plugin.yaml"]
    assert (LINES[PLACES[-1]].format(RELEASE) + "\n").encode() in (aside / "plugin.yaml").read_bytes()


@pytest.mark.parametrize("name", PLACES)
def test_the_check_fails_and_names_a_place_that_holds_another_version(tree, name):
    rewrite(tree, name, LINES[name].format(OTHER))
    before = snapshot(tree)
    result = run("--root", str(tree), "--check")
    assert (result.returncode, result.stdout) == (1, "")
    assert result.stderr.splitlines()[1:] == [f"  {place}: {OTHER if place == name else CURRENT}" for place in PLACES]
    assert snapshot(tree) == before


@pytest.mark.parametrize("held", ["1.13", "1.13.0-rc5", "01.13.0", "1.13.0rc05"])
def test_the_check_fails_when_every_place_holds_one_string_that_is_not_a_version(tree, held):
    """The set command refuses each of these; places that agree on one of them are not a version either."""
    for name in PLACES:
        rewrite(tree, name, LINES[name].format(held))
    before = snapshot(tree)
    result = run("--root", str(tree), "--check")
    assert (result.returncode, result.stdout) == (1, "")
    assert result.stderr == f"every place holds {held!r}, which is not a version of the form X.Y.Z or X.Y.ZrcN\n"
    assert snapshot(tree) == before


def test_the_check_names_a_place_it_cannot_read_a_version_from(tree):
    (tree / PLACES[0]).unlink()
    rewrite(tree, PLACES[-1], "version: [1, 13]")
    before = snapshot(tree)
    result = run("--root", str(tree), "--check")
    assert (result.returncode, result.stdout) == (1, "")
    listed = result.stderr.splitlines()[1:]
    assert listed[0] == f"  {PLACES[0]}: cannot be read (FileNotFoundError)"
    assert listed[-1] == f"  {PLACES[-1]}: line 2 does not look as expected: 'version: [1, 13]'"
    assert listed[1:-1] == [f"  {place}: {CURRENT}" for place in PLACES[1:-1]]
    assert snapshot(tree) == before


@pytest.mark.parametrize("arguments", [(), ("--check", RELEASE)])
def test_neither_or_both_of_check_and_a_version_is_refused_and_no_file_changes(tree, arguments):
    before = snapshot(tree)
    result = run("--root", str(tree), *arguments)
    assert (result.returncode, result.stdout) == (2, "")
    assert "give either --check or NEW_VERSION" in result.stderr
    assert snapshot(tree) == before
