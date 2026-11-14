"""Local first: starting the sidecar makes no network request.

The model library ColonyMind routes requests through downloads its model price list from the internet when it is
imported, unless it is told to use the copy it ships. ColonyMind never reads prices from it, so the sidecar tells it
before the import, and an operator who wants the download can still say so explicitly.
"""
import json
import os
import subprocess
import sys

import pytest

# Runs in a fresh interpreter: every way out to the network raises and is recorded, then the sidecar's app module
# and the model router its startup loads (which imports the model library) are imported.
PROBE = r'''
import json, socket, sys
attempts = []

def refuse(name):
    def blocked(*args, **kwargs):
        attempts.append(name)
        raise OSError("the network is refused in this test")
    return blocked

socket.getaddrinfo = refuse("getaddrinfo")
socket.create_connection = refuse("create_connection")
socket.socket.connect = refuse("connect")
socket.socket.connect_ex = refuse("connect_ex")
import colonymind.server  # noqa: F401
import colonymind.router.router  # noqa: F401  (the model router the sidecar starts, which imports the library)
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map_source_info
source = get_model_cost_map_source_info()
print(json.dumps({"attempts": attempts, "source": source["source"], "forced": source["is_env_forced"],
                  "flag": __import__("os").environ.get("LITELLM_LOCAL_MODEL_COST_MAP")}))
'''


def _import_sidecar(tmp_path, **extra):
    home = tmp_path / "home"
    (home / ".colonymind").mkdir(parents=True, exist_ok=True)
    env = {name: value for name, value in os.environ.items()
           if not name.startswith(("LITELLM_", "COLONYMIND_", "HERMES_"))}
    env.update(HOME=str(home), COLONYMIND_HOME=str(home / ".colonymind"), COLONYMIND_STATE_DIR=str(home / ".colonymind"),
               COLONYMIND_SKIP_DOTENV="1", PYTHONDONTWRITEBYTECODE="1", **extra)
    result = subprocess.run([sys.executable, "-c", PROBE], cwd=tmp_path, env=env, capture_output=True, text=True,
                            timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_importing_the_sidecar_makes_no_network_request(tmp_path):
    seen = _import_sidecar(tmp_path)
    assert seen["attempts"] == []
    assert seen["source"] == "local" and seen["forced"] is True and seen["flag"] == "True"


@pytest.mark.parametrize("value", ["false", "False"])
def test_an_operator_who_asks_for_the_downloaded_price_list_keeps_that_choice(tmp_path, value):
    """The flag is a default, never an override: an explicit ``LITELLM_LOCAL_MODEL_COST_MAP=false`` is left as
    it is (here the download is attempted, and refused by the probe)."""
    seen = _import_sidecar(tmp_path, LITELLM_LOCAL_MODEL_COST_MAP=value)
    assert seen["flag"] == value and seen["forced"] is False and seen["attempts"]
