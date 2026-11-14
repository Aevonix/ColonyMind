"""``colonymind doctor``: can the agent approve its own asks on this machine? (docs/MIND.md, Asks: console and API
answers)

No answer format stops an agent that can read the instance key from forging the owner's turn. The doctor says when
that is possible, each time with its remedy: an owner lane whose terminal, code or file tools run on this machine
(where the key can be read); the terminal (``cli``) as an owner platform on such a host (a ``hermes chat`` the agent
starts is written as the owner); and a WhatsApp bridge in shared-account mode (the owner's own account is the
agent's). What Hermes runs is asked of Hermes itself (``owner_lane_facts``); these tests give its answer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from colonymind import doctor
from colonymind.doctor import FAIL, PASS, SKIP, WARN


def facts(backend="local", lanes=None, whatsapp=None, **extra):
    return {"backend": backend, "lanes": {"whatsapp": ["execute_code", "read_file", "terminal"]} if lanes is None
            else lanes, "whatsapp": whatsapp, **extra}


def key_on(platforms):
    """The instance key granted the owner on these platforms, as the check reads the credentials."""
    return {"api-key": set(platforms)}


def by_name(results):
    return {result.name: result for result in results}


def test_an_owner_lane_with_local_tools_warns_and_names_the_remedy():
    results = by_name(doctor.owner_lane_results(facts(), key_on(["whatsapp"])))
    lane = results["owner-lane-tools"]
    assert lane.status == WARN
    assert "whatsapp" in lane.detail and "terminal" in lane.detail and "execute_code" in lane.detail
    assert "instance key" in lane.detail and "forge" in lane.detail
    assert "terminal.backend" in lane.remedy and ("container" in lane.remedy or "docker" in lane.remedy)
    assert chr(0x2014) not in lane.detail + lane.remedy


def test_a_remote_backend_or_no_tools_passes():
    remote = by_name(doctor.owner_lane_results(facts(backend="docker"), key_on(["whatsapp"])))["owner-lane-tools"]
    assert remote.status == PASS and "docker" in remote.detail
    bare = by_name(doctor.owner_lane_results(facts(lanes={"whatsapp": []}), key_on(["whatsapp"])))["owner-lane-tools"]
    assert bare.status == PASS
    # Only the owner's lanes count: a guest-only platform with a terminal is the guard's, not this check's.
    other = by_name(doctor.owner_lane_results(facts(lanes={"whatsapp": [], "discord": ["terminal"]}),
                                              key_on(["whatsapp"])))["owner-lane-tools"]
    assert other.status == PASS


def test_the_terminal_as_an_owner_platform_on_such_a_host_warns():
    results = by_name(doctor.owner_lane_results(facts(lanes={"whatsapp": ["terminal"], "cli": ["terminal"]}),
                                                key_on(["cli", "whatsapp"])))
    cli = results["cli-owner-platform"]
    assert cli.status == WARN and "hermes chat" in cli.detail
    assert "colonymind credentials set api-key --owner-platform whatsapp" in cli.remedy
    # Every platform, ``owner: any``, includes the terminal.
    anywhere = by_name(doctor.owner_lane_results(facts(lanes={"whatsapp": ["terminal"]}), key_on(["*"])))
    assert anywhere["cli-owner-platform"].status == WARN
    # Not an owner platform, or no local shell anywhere the owner is served: nothing to warn about.
    assert "cli-owner-platform" not in by_name(doctor.owner_lane_results(facts(), key_on(["whatsapp"])))
    remote = by_name(doctor.owner_lane_results(facts(backend="ssh", lanes={"cli": ["terminal"]}), key_on(["cli"])))
    assert remote["cli-owner-platform"].status == PASS


def test_the_remedy_names_each_credential_that_grants_the_terminal_and_keeps_its_other_platforms():
    """The fix is per credential: each one that may write the owner from the terminal loses only the terminal, keeping
    every other platform it had. ``owner: any`` becomes the platforms Hermes serves, never ``--no-owner``, and a
    credential that grants only the terminal loses the owner, while one that never granted it is left alone."""
    lanes = {"whatsapp": ["terminal"], "sms": [], "voice": [], "cli": ["terminal"]}
    anywhere = by_name(doctor.owner_lane_results(facts(lanes=lanes), {"api-key": {"*"}}))["cli-owner-platform"]
    assert anywhere.status == WARN
    assert anywhere.remedy.startswith("colonymind credentials set api-key --owner-platform sms --owner-platform voice "
                                      "--owner-platform whatsapp")
    assert "--no-owner" not in anywhere.remedy
    both = by_name(doctor.owner_lane_results(facts(lanes=lanes), {"api-key": {"*"}, "console": {"*"}}))
    assert both["cli-owner-platform"].remedy.count("--owner-platform sms --owner-platform voice --owner-platform "
                                                   "whatsapp") == 2

    secondary = by_name(doctor.owner_lane_results(facts(lanes=lanes), {
        "api-key": {"whatsapp", "sms"}, "console": {"cli"}, "voice-gw": {"voice", "cli"}}))["cli-owner-platform"]
    assert secondary.status == WARN
    assert "colonymind credentials set console --no-owner" in secondary.remedy
    assert "colonymind credentials set voice-gw --owner-platform voice" in secondary.remedy
    assert "api-key" not in secondary.remedy          # it never granted the terminal: nothing of it changes
    assert "console" in secondary.detail and "voice-gw" in secondary.detail


def test_a_shared_account_whatsapp_bridge_warns_and_bot_mode_passes():
    shared = by_name(doctor.owner_lane_results(facts(whatsapp={"enabled": True, "bot": False}), key_on(["whatsapp"])))
    assert shared["whatsapp-mode"].status == WARN
    assert "shared" in shared["whatsapp-mode"].detail and "WHATSAPP_MODE=bot" in shared["whatsapp-mode"].remedy
    bot = by_name(doctor.owner_lane_results(facts(whatsapp={"enabled": True, "bot": True}), key_on(["whatsapp"])))
    assert bot["whatsapp-mode"].status == PASS
    off = by_name(doctor.owner_lane_results(facts(whatsapp={"enabled": False, "bot": False}), key_on(["whatsapp"])))
    assert "whatsapp-mode" not in off
    elsewhere = by_name(doctor.owner_lane_results(facts(whatsapp={"enabled": True, "bot": False}), key_on(["telegram"])))
    assert "whatsapp-mode" not in elsewhere


def test_what_hermes_could_not_say_is_skipped_never_guessed():
    skipped = doctor.owner_lane_results({"error": "ImportError"}, key_on(["whatsapp"]))
    assert [(result.name, result.status) for result in skipped] == [("owner-lanes", SKIP)]
    unknown = by_name(doctor.owner_lane_results(facts(lanes={"whatsapp": None}), key_on(["whatsapp"])))
    assert unknown["owner-lane-tools"].status == SKIP


@pytest.fixture
def instance(tmp_path, monkeypatch):
    home, hermes = tmp_path / "instance", tmp_path / "hermes"
    home.mkdir()
    hermes.mkdir()
    monkeypatch.setenv("COLONYMIND_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    (home / "colonymind.yaml").write_text(yaml.safe_dump({
        "hermes": {"home": str(hermes), "python": "/usr/bin/true"},
        "credentials": {"api-key": {"owner": ["cli", "whatsapp"], "contacts": "all"},
                        "voice-gw": {"sha256": "0" * 64, "owner": ["voice"], "contacts": "all"}}}))
    return home, hermes


def test_the_check_asks_hermes_about_every_owner_platform_of_every_credential(instance, monkeypatch):
    home, hermes = instance
    asked = []

    def probe(python, hermes_home, platforms):
        asked.append((str(python), Path(hermes_home), sorted(platforms)))
        return facts(lanes={"cli": ["terminal"], "whatsapp": ["execute_code"], "voice": []},
                     whatsapp={"enabled": True, "bot": True})

    monkeypatch.setattr(doctor, "_hermes_python", lambda: Path("/usr/bin/true"))
    monkeypatch.setattr(doctor, "owner_lane_facts", probe)
    results = by_name(doctor.check_owner_lanes())
    assert asked == [("/usr/bin/true", hermes, ["cli", "voice", "whatsapp"])]
    assert results["owner-lane-tools"].status == WARN and results["cli-owner-platform"].status == WARN
    assert results["cli-owner-platform"].remedy.startswith("colonymind credentials set api-key --owner-platform "
                                                           "whatsapp (")
    assert "voice-gw" not in results["cli-owner-platform"].remedy        # it never granted the terminal
    assert results["whatsapp-mode"].status == PASS
    assert "owner-lanes" in [name for name, _ in doctor.LOCAL_CHECKS]


def test_without_hermes_or_with_a_broken_configuration_it_skips_and_quotes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "_hermes_python", lambda: None)
    assert [(r.name, r.status) for r in doctor.check_owner_lanes()] == [("owner-lanes", SKIP)]
    home = tmp_path / "broken"
    home.mkdir()
    monkeypatch.setenv("COLONYMIND_HOME", str(home))
    canary = "sk-canary-" + "owner-lanes-0123"
    (home / "colonymind.yaml").write_text(f"credentials:\n  api-key: {canary}: pasted twice\n")
    monkeypatch.setattr(doctor, "_hermes_python", lambda: Path("/usr/bin/true"))
    results = doctor.check_owner_lanes()
    assert [r.status for r in results] == [SKIP] and canary not in json.dumps([r.to_dict() for r in results])
    assert FAIL not in [r.status for r in results]
