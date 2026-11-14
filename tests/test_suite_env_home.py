"""The shell runner gives a run a home of its own before anything of the tree is imported."""
import json
import os
from pathlib import Path
import pwd

from test_suite_env import runner  # noqa: F401  (the stand-in tree, venvs and suite of the runner's own tests)

NAMES = ("HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME")


def test_a_run_has_its_own_home_from_the_first_import_on(runner, tmp_path):  # noqa: F811
    tree, py, hermes, site, run = runner
    record = tmp_path / "homes.jsonl"
    note = ("import json, os\n"
            f"with open({str(record)!r}, 'a') as stream:\n"
            f"    stream.write(json.dumps({{name: os.environ.get(name) for name in {NAMES!r}}}) + '\\n')\n")
    # The tree's own package is what the import proof imports first; the stand-in pytest is both phases.
    (tree / "colonymind/__init__.py").write_text(note)
    standin = site / "pytest/__main__.py"
    standin.write_text(note + standin.read_text())
    result = run(XDG_CONFIG_HOME=str(tmp_path / "callers-config"), XDG_STATE_HOME=str(tmp_path / "callers-state"))
    assert result.returncode == 0, result.stdout + result.stderr
    out = Path(result.stdout.split("== full output: ", 1)[1].splitlines()[0])
    home = out / "home"
    assert home.is_dir() and home.stat().st_mode & 0o777 == 0o700
    assert home.resolve() != Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    seen = [json.loads(line) for line in record.read_text().splitlines()]
    assert len(seen) == 3, seen   # the import proof, the sidecar phase, the adapter phase
    assert seen == [{**dict.fromkeys(NAMES), "HOME": str(home)}] * 3
