"""``colonymind doctor`` asks stock Hermes what the owner's lanes run (``doctor.owner_lane_facts``).

The facts are Hermes' own: the terminal backend its tools run on, the terminal, code and file tools each platform's
toolsets resolve to, and whether its WhatsApp bridge is on and in bot mode. Only names and booleans come back, never
a value from Hermes' configuration or its ``.env``.
"""

import sys
from pathlib import Path

import yaml
import pytest

pytestmark = pytest.mark.requires_sidecar


@pytest.fixture
def doctor():
    from colonymind import doctor
    return doctor


SECRET = "sk-canary-" + "hermes-env-5a1f"


@pytest.fixture(autouse=True)
def synthetic_profile_environment(tmp_path, monkeypatch):
    """These probes intentionally read only the fixture's synthetic profile .env."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "0")


def _hermes_home(tmp_path, config, env=""):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    (home / ".env").write_text(env)
    return home


def test_a_default_lane_runs_a_local_terminal_and_code(tmp_path, doctor):
    home = _hermes_home(tmp_path, {"model": {"provider": "custom", "default": "m"}},
                        env=f"OPENAI_API_KEY={SECRET}\nWHATSAPP_ENABLED=true\n")
    found = doctor.owner_lane_facts(Path(sys.executable), home, ["whatsapp", "cli"])
    assert found["backend"] == "local", found
    assert {"terminal", "execute_code"} <= set(found["lanes"]["whatsapp"]), found
    assert "terminal" in found["lanes"]["cli"]
    assert found["whatsapp"] == {"enabled": True, "bot": False}             # the stock default: shared account
    assert SECRET not in repr(found)
    results = {r.name: r for r in doctor.owner_lane_results(found, {"api-key": {"cli", "whatsapp"}})}
    assert results["owner-lane-tools"].status == doctor.WARN
    assert results["cli-owner-platform"].status == doctor.WARN
    assert results["whatsapp-mode"].status == doctor.WARN


def test_a_container_backend_and_a_bot_bridge_are_read_as_hermes_reads_them(tmp_path, doctor):
    home = _hermes_home(tmp_path, {"model": {"provider": "custom", "default": "m"}, "terminal": {"backend": "docker"},
                                   "platform_toolsets": {"whatsapp": ["web", "memory"]}},
                        env="WHATSAPP_ENABLED=true\nWHATSAPP_MODE=bot\n")
    found = doctor.owner_lane_facts(Path(sys.executable), home, ["whatsapp"])
    assert found["backend"] == "docker", found
    assert found["lanes"]["whatsapp"] == []
    assert found["whatsapp"] == {"enabled": True, "bot": True}
    results = {r.name: r for r in doctor.owner_lane_results(found, {"api-key": {"whatsapp"}})}
    assert results["owner-lane-tools"].status == doctor.PASS and results["whatsapp-mode"].status == doctor.PASS


def test_the_lanes_a_contact_reaches_are_the_platforms_hermes_serves(tmp_path, doctor):
    """``doctor.guest_lane_facts``: every platform Hermes serves but its own terminal and cron, with the executor
    tools each resolves to; a plugin platform whose tools only exist once its plugin loads is one Hermes cannot
    say about (None), never a guess."""
    home = _hermes_home(tmp_path, {"model": {"provider": "custom", "default": "m"},
                                   "platforms": {"api_server": {"enabled": True}, "webhook": {"enabled": True},
                                                 "rcs": {"enabled": True}, "signal": {"enabled": False}},
                                   "platform_toolsets": {"telegram": ["web", "memory"]}},
                        env=f"OPENAI_API_KEY={SECRET}\nWHATSAPP_ENABLED=true\nTELEGRAM_BOT_TOKEN=123:{SECRET}\n")
    found = doctor.guest_lane_facts(Path(sys.executable), home)
    assert found["backend"] == "local", found
    lanes = found["lanes"]
    assert {"terminal", "execute_code", "write_file", "patch", "delegate_task"} <= set(lanes["whatsapp"]), found
    assert "terminal" in lanes["api_server"] and "computer_use" not in lanes["api_server"], found
    assert lanes["webhook"] == [] and lanes["telegram"] == [], found
    assert lanes["rcs"] is None, found
    assert not {"cli", "cron", "signal"} & set(lanes), found
    assert SECRET not in repr(found)
    result, = doctor.guest_lane_results(found, [])
    assert result.status == doctor.PASS and "whatsapp" in result.detail and "rcs" in result.detail, result
    opted, = doctor.guest_lane_results(found, ["api_server"])
    assert opted.status == doctor.WARN and "api_server" in opted.detail, opted
