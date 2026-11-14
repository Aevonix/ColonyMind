"""A test run has a home of its own: this process and the processes it starts find nothing of the user who runs it."""
import inspect
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys
import tempfile

SIDECAR = Path(__file__).resolve().parents[1]
# What selects an instance or switches the search for its .env off: without these a process is on its defaults.
SELECTORS = ("COLONYMIND_HOME", "COLONYMIND_STATE_DIR", "COLONYMIND_SKIP_DOTENV", "COLONYMIND_INSTANCE_SELECTED")


def _places(search_dotenv):
    """Where a process with no instance selected looks. The started process runs this function from its source."""
    import os
    from pathlib import Path
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
    return {"HOME": os.environ.get("HOME", ""), "expanduser": os.path.expanduser("~"), "Path.home": str(Path.home()),
            "instance home": str(config.instance_home()), "state directory": str(colonymind.get_state_dir()),
            "temporal state": str(temporal._state_dir()), "last owner message": str(session_safety._LAST_MSG_PATH),
            "dotenv": sorted({path for path in looked if path.endswith(".env")})}


def test_a_test_run_finds_only_its_own_home(tmp_path, monkeypatch):
    real = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()   # from the user database, not from the environment
    home = Path(os.environ["HOME"]).resolve()
    assert home != real and home.is_relative_to(Path(tempfile.gettempdir()).resolve()), home
    for name in SELECTORS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    started = subprocess.run(
        [sys.executable, "-c", inspect.getsource(_places) + "import json\nprint(json.dumps(_places(True)))"],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (str(SIDECAR), os.environ.get("PYTHONPATH"))))},
        capture_output=True, text=True, timeout=120)
    assert started.returncode == 0, started.stderr[-4000:]
    instance = home / ".colonymind"
    # The path of the owner's last message is fixed when its module is imported: here it shows that this process
    # had its home before the first product import, and in the started process that the home is inherited.
    expected = {"HOME": home, "expanduser": home, "Path.home": home, "instance home": instance,
                "temporal state": instance, "last owner message": instance / "last_user_message_at.json"}
    for process, seen in (("this process", _places(False)), ("a started process", json.loads(started.stdout))):
        dotenv, state = seen.pop("dotenv"), Path(seen.pop("state directory")).resolve()
        assert {name: Path(path).resolve() for name, path in seen.items()} == expected, process
        # Unconfigured, the state lies in the instance's data directory.
        assert state.is_relative_to(instance), (process, state)
        # The instance's .env is looked for in its home (twice by name) and in the working directory.
        assert {Path(path).resolve() for path in dotenv} <= {instance / ".env", tmp_path.resolve() / ".env"}
        assert process == "this process" or instance / ".env" in {Path(path).resolve() for path in dotenv}
