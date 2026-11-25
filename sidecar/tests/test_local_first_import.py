"""Local first: starting the sidecar makes no network request.

The model library ColonyMind routes requests through downloads its model price list from the internet when it is
imported, unless it is told to use the copy it ships. ColonyMind never reads prices from it, so the sidecar tells it
before the import, and an operator who wants the download can still say so explicitly.
"""
import json
import os
import socket
import subprocess
import sys

import pytest

# Runs in a fresh interpreter: every way out to the network through Python's socket module raises and is recorded,
# then the sidecar's app module and the model router its startup loads (which imports the model library) are
# imported. A library that fetches through its own native code goes past these patches, so the probe cannot see that
# request: the library's own account of where its price list came from is the proof that holds for either transport,
# and the price list address points at a closed local port, so a fetch that does happen never leaves the machine.
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
                  "url": source["url"], "fallback": source["fallback_reason"],
                  "flag": __import__("os").environ.get("LITELLM_LOCAL_MODEL_COST_MAP")}))
'''


def _import_sidecar(tmp_path, **extra):
    home = tmp_path / "home"
    (home / ".colonymind").mkdir(parents=True, exist_ok=True)
    env = {name: value for name, value in os.environ.items()
           if not name.startswith(("LITELLM_", "COLONYMIND_", "HERMES_"))}
    # A port bound and never listened on refuses every connection, and stays taken while the probe runs.
    with socket.socket() as closed_port:
        closed_port.bind(("127.0.0.1", 0))
        address = "http://127.0.0.1:%d/model_prices_and_context_window.json" % closed_port.getsockname()[1]
        env.update(HOME=str(home), COLONYMIND_HOME=str(home / ".colonymind"),
                   COLONYMIND_STATE_DIR=str(home / ".colonymind"), COLONYMIND_SKIP_DOTENV="1",
                   PYTHONDONTWRITEBYTECODE="1", LITELLM_MODEL_COST_MAP_URL=address, **extra)
        result = subprocess.run([sys.executable, "-c", PROBE], cwd=tmp_path, env=env, capture_output=True, text=True,
                                timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1]), address


def test_importing_the_sidecar_makes_no_network_request(tmp_path):
    seen, _ = _import_sidecar(tmp_path)
    assert seen["attempts"] == []
    assert seen["source"] == "local" and seen["forced"] is True and seen["flag"] == "True"


@pytest.mark.parametrize("value", ["false", "False"])
def test_an_operator_who_asks_for_the_downloaded_price_list_keeps_that_choice(tmp_path, value):
    """The flag is a default, never an override: an explicit ``LITELLM_LOCAL_MODEL_COST_MAP=false`` is left as
    it is (here the download is attempted at the closed local port, and the library reports that it failed and fell
    back to the copy it ships)."""
    seen, address = _import_sidecar(tmp_path, LITELLM_LOCAL_MODEL_COST_MAP=value)
    assert seen["flag"] == value and seen["forced"] is False
    assert seen["source"] == "local" and seen["url"] == address and seen["fallback"]
