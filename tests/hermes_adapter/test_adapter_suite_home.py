"""A test run has a home of its own: this process and the processes it starts find nothing of the user who runs it."""
import inspect
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys
import tempfile

import pytest


@pytest.mark.parametrize("suite", ["sidecar/tests", "tests/hermes_adapter"])
def test_suite_discards_inherited_home_selectors(tmp_path, suite):
    root = Path(__file__).resolve().parents[2]
    selectors = ("HERMES_HOME", "COLONYMIND_HOME", "COLONYMIND_INSTANCE_SELECTED")
    inherited = tmp_path / "inherited"
    inherited.mkdir()
    script = """
import json, os, runpy, sys
runpy.run_path(sys.argv[1])
print(json.dumps({name: os.environ.get(name) for name in sys.argv[2:]}))
"""
    started = subprocess.run(
        [sys.executable, "-c", script, str(root / suite / "conftest.py"), *selectors],
        env={**os.environ, **dict.fromkeys(selectors, str(inherited))},
        cwd=root, capture_output=True, text=True, timeout=120)
    assert started.returncode == 0, started.stderr[-4000:]
    assert json.loads(started.stdout.splitlines()[-1]) == dict.fromkeys(selectors)
    assert list(inherited.iterdir()) == []

# What selects an instance or a Hermes home, or switches the search for the instance's .env off: without these a
# process is on its defaults.
SELECTORS = ("COLONYMIND_HOME", "COLONYMIND_STATE_DIR", "COLONYMIND_SKIP_DOTENV", "COLONYMIND_INSTANCE_SELECTED",
             "HERMES_HOME")


def _places(search_dotenv):
    """Where a process with nothing selected looks. The started process runs this function from its source."""
    import importlib.util
    import os
    from pathlib import Path
    from hermes_constants import get_hermes_home
    from colonymind_hermes.client import load_settings
    from colonymind_memory.provider import _instance_dir
    settings = load_settings({})
    seen = {"HOME": os.environ.get("HOME", ""), "expanduser": os.path.expanduser("~"), "Path.home": str(Path.home()),
            "Hermes home": str(get_hermes_home()), "plugin: Hermes home": str(settings.hermes_home),
            "plugin: instance": str(settings.home), "plugin: key file": str(settings.key_file),
            "memory provider: instance": str(_instance_dir(None))}
    if importlib.util.find_spec("colonymind") is None:   # an adapter-only environment has no sidecar to ask
        return seen
    import colonymind
    from colonymind import config
    from colonymind.util import instance, session_safety, temporal
    looked, is_file = [], Path.is_file
    if search_dotenv:
        Path.is_file = lambda path: looked.append(str(path)) or is_file(path)
        try:
            instance.load_environment()
        finally:
            Path.is_file = is_file
    seen.update({"sidecar: instance home": str(config.instance_home()),
                 "sidecar: state directory": str(colonymind.get_state_dir()),
                 "sidecar: temporal state": str(temporal._state_dir()),
                 "sidecar: last owner message": str(session_safety._LAST_MSG_PATH),
                 "sidecar: dotenv": sorted({path for path in looked if path.endswith(".env")})})
    return seen


def test_a_test_run_finds_only_its_own_home(tmp_path, monkeypatch):
    real = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()   # from the user database, not from the environment
    home = Path(os.environ["HOME"]).resolve()
    assert home != real and home.is_relative_to(Path(tempfile.gettempdir()).resolve()), home
    for name in SELECTORS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    started = subprocess.run(
        [sys.executable, "-c", inspect.getsource(_places) + "import json\nprint(json.dumps(_places(True)))"],
        capture_output=True, text=True, timeout=120)
    assert started.returncode == 0, started.stderr[-4000:]
    instance = home / ".colonymind"
    expected = {"HOME": home, "expanduser": home, "Path.home": home, "Hermes home": home / ".hermes",
                "plugin: Hermes home": home / ".hermes", "plugin: instance": instance,
                "plugin: key file": instance / "api.key", "memory provider: instance": instance}
    # The sidecar fixes the path of the owner's last message when its module is imported: here it shows that this
    # process had its home before the first product import, and in the started process that the home is inherited.
    sidecar = {"sidecar: instance home": instance, "sidecar: temporal state": instance,
               "sidecar: last owner message": instance / "last_user_message_at.json"}
    for process, seen in (("this process", _places(False)), ("a started process", json.loads(started.stdout))):
        if "sidecar: instance home" in seen:
            dotenv, state = seen.pop("sidecar: dotenv"), Path(seen.pop("sidecar: state directory")).resolve()
            assert {name: Path(path).resolve() for name, path in seen.items()} == {**expected, **sidecar}, process
            # Unconfigured, the state lies in the instance's data directory.
            assert state.is_relative_to(instance), (process, state)
            # The instance's .env is looked for in its home (twice by name) and in the working directory.
            assert {Path(path).resolve() for path in dotenv} <= {instance / ".env", tmp_path.resolve() / ".env"}
            assert process == "this process" or instance / ".env" in {Path(path).resolve() for path in dotenv}
        else:
            assert {name: Path(path).resolve() for name, path in seen.items()} == expected, process
