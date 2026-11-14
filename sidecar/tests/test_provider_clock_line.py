"""The memory provider's own clock line is the sidecar's line for the same instant and instance.

The provider runs in the host program, without the sidecar's package and possibly without its environment. It reads
the two environment names the sidecar reads and, under them, the instance's ``identity.yaml``, once when it is set up.
Every provider here gets its instance directory under ``tmp_path``.
"""
import importlib.util
import os
import pathlib
import sys
from datetime import datetime, timezone

import pytest

from colonymind.config import apply_environment, load_config, save_identity
from colonymind.util import temporal as T

NOW = datetime(2026, 10, 17, 14, 14, tzinfo=timezone.utc)
NAMES = ("COLONYMIND_PERSONA_NAME", "COLONYMIND_AGENT_TIMEZONE")
_PROVIDER = pathlib.Path(__file__).resolve().parents[2] / "plugins" / "colonymind-memory" / "provider.py"


@pytest.fixture(scope="module")
def provider_mod():
    spec = importlib.util.spec_from_file_location("colonymind_memory_provider_clock_line", _PROVIDER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def home(tmp_path, monkeypatch, provider_mod):
    """An instance directory of the test's own, with nothing configured in it or in the environment."""
    home = tmp_path / "instance"
    home.mkdir()
    monkeypatch.setenv("COLONYMIND_HOME", str(home))
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(home))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    for name in (*NAMES, "COLONYMIND_TIMEZONE", "COLONYMIND_DEFAULT_CONTACT_TIMEZONE", "TZ"):
        monkeypatch.delenv(name, raising=False)
    at(monkeypatch, provider_mod, NOW)
    return home


def at(monkeypatch, provider_mod, instant):
    """Both lines read the one wall clock."""
    monkeypatch.setattr(provider_mod._ttime, "time", instant.timestamp)


def machine_zone(monkeypatch, provider_mod, zone):
    """The zone the machine names, for both readers: ``zone``, or none."""
    monkeypatch.setattr(T, "_system_timezone", lambda: zone or None)
    monkeypatch.setattr(provider_mod, "_machine_zone", lambda: zone)


def provider_lines(provider_mod, **config):
    """The two lines of a provider set up now, in the host program's environment."""
    provider = provider_mod.ColonyMindMemoryProvider(config={"url": "http://sidecar.test", "api_key": "k",
                                                            "contact_id": "cid-owner", **config})
    return provider._clock_line(), provider._owner_clock_line()


def sidecar_lines(home, monkeypatch):
    """The two lines in the sidecar's process: its start exports the instance's identity under its environment."""
    environ = {name: os.environ[name] for name in NAMES if name in os.environ}
    apply_environment(load_config(home, environ={}), environ=environ)
    with monkeypatch.context() as sidecar:
        for name in NAMES:
            if name in environ:
                sidecar.setenv(name, environ[name])
            else:
                sidecar.delenv(name, raising=False)
        return T.describe_agent_now(), T.describe_now(owner=True, contact_label="the owner")


def identity(home, **agent):
    save_identity({"owner": {"name": "Synthetic Owner", "handles": []}, "agent": agent}, home)


def test_a_configured_instance_reads_the_same_line_in_both(home, monkeypatch, provider_mod):
    identity(home, name="Nova", timezone="Europe/Berlin")
    public, owner = provider_lines(provider_mod)
    assert public == "Nova's time: Saturday 4:14 PM CEST (Europe/Berlin)."
    assert owner == public + "\nthe owner's location: no recent report; assume CEST (Europe/Berlin)."
    assert (public, owner) == sidecar_lines(home, monkeypatch)


def test_an_instance_with_nothing_configured_reads_the_same_line_in_both(home, monkeypatch, provider_mod):
    # Whatever zone this machine names, the two agree on it and on the neutral name.
    public, owner = provider_lines(provider_mod)
    assert (public, owner) == sidecar_lines(home, monkeypatch)
    assert public.startswith("the assistant's time: ")
    machine_zone(monkeypatch, provider_mod, "")
    public, owner = provider_lines(provider_mod)
    assert public == "the assistant's time: Saturday 2:14 PM UTC (UTC)."
    assert owner == public + "\nthe owner's location: no recent report; assume UTC (UTC)."
    assert (public, owner) == sidecar_lines(home, monkeypatch)
    machine_zone(monkeypatch, provider_mod, "Asia/Tokyo")
    public, owner = provider_lines(provider_mod)
    assert public == "the assistant's time: Saturday 11:14 PM JST (Asia/Tokyo)."
    assert (public, owner) == sidecar_lines(home, monkeypatch)


def test_the_machine_zone_is_found_by_the_sidecars_rule(home, monkeypatch, provider_mod):
    assert provider_mod._machine_zone() == (T._system_timezone() or "")
    monkeypatch.setattr(provider_mod.os, "readlink", lambda path: "/usr/share/zoneinfo/Asia/Tokyo")
    monkeypatch.setattr(T.Path, "is_symlink", lambda self: True)
    assert provider_mod._machine_zone() == T._system_timezone() == "Asia/Tokyo"
    monkeypatch.setattr(provider_mod.os, "readlink", lambda path: "/usr/share/zoneinfo/Mars/Phobos")
    monkeypatch.setenv("TZ", "Europe/Berlin")
    assert provider_mod._machine_zone() == T._system_timezone() == "Europe/Berlin"
    monkeypatch.setenv("TZ", "Mars/Phobos")
    assert provider_mod._machine_zone() == (T._system_timezone() or "") == ""


@pytest.mark.parametrize("environment, recorded, expected", [
    # The environment alone.
    ({"COLONYMIND_PERSONA_NAME": "Nova", "COLONYMIND_AGENT_TIMEZONE": "Asia/Tokyo"}, None,
     "Nova's time: Saturday 11:14 PM JST (Asia/Tokyo)."),
    # A value set in the environment wins over the instance's file, as in the sidecar's start.
    ({"COLONYMIND_PERSONA_NAME": "Nova", "COLONYMIND_AGENT_TIMEZONE": "Asia/Tokyo"},
     {"name": "Vesper", "timezone": "Pacific/Honolulu"}, "Nova's time: Saturday 11:14 PM JST (Asia/Tokyo)."),
    # A blank value in the environment does not.
    ({"COLONYMIND_PERSONA_NAME": " ", "COLONYMIND_AGENT_TIMEZONE": ""},
     {"name": "Vesper", "timezone": "Pacific/Honolulu"}, "Vesper's time: Saturday 4:14 AM HST (Pacific/Honolulu)."),
    # One of the two from each.
    ({"COLONYMIND_AGENT_TIMEZONE": "America/Sao_Paulo"}, {"name": "Vesper"},
     "Vesper's time: Saturday 11:14 AM -03 (America/Sao_Paulo)."),
    # A zone name that is no zone is no value, in the file or in the environment.
    ({}, {"name": "Vesper", "timezone": "Mars/Phobos"}, "Vesper's time: Saturday 2:14 PM UTC (UTC)."),
    ({"COLONYMIND_AGENT_TIMEZONE": "Mars/Phobos"}, {"name": "Vesper", "timezone": "Pacific/Honolulu"},
     "Vesper's time: Saturday 2:14 PM UTC (UTC)."),
    ({"COLONYMIND_AGENT_TIMEZONE": "utc"}, None, "the assistant's time: Saturday 2:14 PM UTC (utc)."),
])
def test_environment_and_file_take_the_sidecars_precedence(home, monkeypatch, provider_mod, environment, recorded,
                                                           expected):
    machine_zone(monkeypatch, provider_mod, "")
    if recorded is not None:
        identity(home, **recorded)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    public, owner = provider_lines(provider_mod)
    assert public == expected
    assert (public, owner) == sidecar_lines(home, monkeypatch)


@pytest.mark.parametrize("instant, expected", [
    (datetime(2026, 3, 8, 6, 59, tzinfo=timezone.utc), "Sunday 1:59 AM EST (America/New_York)"),
    (datetime(2026, 3, 8, 7, 0, tzinfo=timezone.utc), "Sunday 3:00 AM EDT (America/New_York)"),
    (datetime(2026, 11, 1, 5, 59, tzinfo=timezone.utc), "Sunday 1:59 AM EDT (America/New_York)"),
    (datetime(2026, 11, 1, 6, 0, tzinfo=timezone.utc), "Sunday 1:00 AM EST (America/New_York)"),
])
def test_both_lines_change_abbreviation_at_the_same_instant(home, monkeypatch, provider_mod, instant, expected):
    identity(home, name="Nova", timezone="America/New_York")
    at(monkeypatch, provider_mod, instant)
    public, owner = provider_lines(provider_mod)
    assert public == f"Nova's time: {expected}."
    assert owner.splitlines()[1] == f"the owner's location: no recent report; assume {expected.split(' ', 3)[3]}."
    assert (public, owner) == sidecar_lines(home, monkeypatch)


def _a_directory(path):
    path.mkdir()


@pytest.mark.parametrize("damage", [
    lambda path: None,                                                    # no file
    _a_directory,                                                         # not readable as a file
    lambda path: path.write_bytes(b"\xff\xfe\x00agent"),                  # not text
    lambda path: path.write_text("agent: [Nova\n  timezone: {"),          # not YAML
    lambda path: path.write_text("- Nova\n- Europe/Berlin\n"),            # not a mapping
    lambda path: path.write_text("agent: Nova\n"),                        # no agent mapping
    lambda path: path.write_text("agent:\n  name: ''\n  timezone: 7\n"),  # no name, and a zone that is no zone
], ids=["missing", "directory", "bytes", "malformed", "list", "scalar", "empty"])
def test_a_file_that_cannot_be_read_is_no_value_and_never_an_error(home, monkeypatch, provider_mod, damage):
    machine_zone(monkeypatch, provider_mod, "Asia/Tokyo")
    damage(home / "identity.yaml")
    public, owner = provider_lines(provider_mod)
    assert public == "the assistant's time: Saturday 11:14 PM JST (Asia/Tokyo)."
    assert owner == public + "\nthe owner's location: no recent report; assume JST (Asia/Tokyo)."


def test_the_file_is_found_beside_the_key_file_first(home, tmp_path, monkeypatch, provider_mod):
    identity(home, name="Vesper", timezone="Pacific/Honolulu")
    other = tmp_path / "other"
    other.mkdir()
    identity(other, name="Nova", timezone="Europe/Berlin")
    assert provider_lines(provider_mod, key_file=str(other / "key-file"))[0] == \
        "Nova's time: Saturday 4:14 PM CEST (Europe/Berlin)."
    assert provider_lines(provider_mod)[0] == "Vesper's time: Saturday 4:14 AM HST (Pacific/Honolulu)."


def test_the_instance_is_read_once_at_setup_and_no_turn_asks_the_sidecar(home, monkeypatch, provider_mod):
    identity(home, name="Nova", timezone="Europe/Berlin")
    reads = []
    read = provider_mod._instance_agent
    monkeypatch.setattr(provider_mod, "_instance_agent", lambda key_file: reads.append(key_file) or read(key_file))

    class NoRequests:
        def __getattr__(self, name):
            raise AssertionError("the clock line asked the sidecar")

    monkeypatch.setattr(provider_mod, "httpx", NoRequests())
    provider = provider_mod.ColonyMindMemoryProvider(config={"url": "http://sidecar.test", "api_key": "k",
                                                            "contact_id": "cid-owner"})
    assert reads == [None]
    # What changes after setup, in the file or in the environment, reaches no later turn of this provider.
    identity(home, name="Vesper", timezone="Pacific/Honolulu")
    monkeypatch.setenv("COLONYMIND_PERSONA_NAME", "Vesper")
    monkeypatch.setenv("COLONYMIND_AGENT_TIMEZONE", "Pacific/Honolulu")
    for _ in range(5):
        assert provider._clock_line() == "Nova's time: Saturday 4:14 PM CEST (Europe/Berlin)."
        assert provider._owner_clock_line().splitlines()[1] == \
            "the owner's location: no recent report; assume CEST (Europe/Berlin)."
        assert provider._local_temporal_block() == "## Current Time\n" + provider._clock_line()
    assert reads == [None]


def test_the_timezone_setting_keeps_its_meaning(home, monkeypatch, provider_mod):
    # The provider's own ``timezone`` is the communication zone it sends for a private owner turn, not the home clock.
    identity(home, name="Nova", timezone="Europe/Berlin")
    provider = provider_mod.ColonyMindMemoryProvider(config={"url": "http://sidecar.test", "api_key": "k",
                                                            "contact_id": "cid-owner", "timezone": "Asia/Tokyo"})
    assert provider._timezone == "Asia/Tokyo"
    assert provider._clock_line() == "Nova's time: Saturday 4:14 PM CEST (Europe/Berlin)."
    assert provider_mod.ColonyMindMemoryProvider(config={"url": "http://sidecar.test", "api_key": "k"})._timezone == ""
