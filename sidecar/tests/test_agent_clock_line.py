"""The clock line of every turn names the configured assistant and the agent's configured zone."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from colonymind.api.routers import host
from colonymind.util import temporal as T

NOW = datetime(2026, 10, 17, 14, 14, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def nothing_configured(tmp_path, monkeypatch):
    """No name and no zone in the environment, no stored zone, and a machine that names no zone."""
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    for name in ("COLONYMIND_PERSONA_NAME", "COLONYMIND_AGENT_TIMEZONE", "COLONYMIND_DEFAULT_CONTACT_TIMEZONE", "TZ"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(T, "_system_timezone", lambda: None)
    monkeypatch.setattr(T, "now_utc", lambda: NOW)


def configure(monkeypatch, zone="Europe/Berlin"):
    monkeypatch.setenv("COLONYMIND_PERSONA_NAME", "Nova")
    monkeypatch.setenv("COLONYMIND_AGENT_TIMEZONE", zone)


def test_the_line_names_the_configured_assistant_and_zone(monkeypatch):
    configure(monkeypatch)
    assert T.describe_agent_now() == "Nova's time: Saturday 4:14 PM CEST (Europe/Berlin)."
    assert T.describe_now(owner=True, contact_label="Synthetic  Owner").splitlines() == [
        "Nova's time: Saturday 4:14 PM CEST (Europe/Berlin).",
        "Synthetic Owner's location: no recent report; assume CEST (Europe/Berlin)."]


def test_nothing_configured_reads_the_neutral_name_and_the_fallback_zone(monkeypatch):
    assert T.describe_agent_now() == "the assistant's time: Saturday 2:14 PM UTC (UTC)."
    assert T.describe_owner_now(None, "the owner").splitlines() == [
        "the assistant's time: Saturday 2:14 PM UTC (UTC).",
        "the owner's location: no recent report; assume UTC (UTC)."]
    # Before UTC come the zone the machine names and, before that, a zone stored through the API.
    monkeypatch.setattr(T, "_system_timezone", lambda: "Asia/Tokyo")
    assert T.describe_agent_now() == "the assistant's time: Saturday 11:14 PM JST (Asia/Tokyo)."
    T.set_agent_timezone("Pacific/Honolulu")
    assert T.describe_owner_now(None, "the owner").splitlines() == [
        "the assistant's time: Saturday 4:14 AM HST (Pacific/Honolulu).",
        "the owner's location: no recent report; assume HST (Pacific/Honolulu)."]


def test_a_zone_without_a_lettered_abbreviation_reads_its_offset(monkeypatch):
    configure(monkeypatch, "America/Sao_Paulo")
    assert T.describe_owner_now(None, "the owner").splitlines() == [
        "Nova's time: Saturday 11:14 AM -03 (America/Sao_Paulo).",
        "the owner's location: no recent report; assume -03 (America/Sao_Paulo)."]


@pytest.mark.parametrize("instant,clock,zone", [
    (datetime(2026, 3, 8, 6, 59, tzinfo=timezone.utc), "Sunday 1:59 AM", "EST (America/New_York)"),
    (datetime(2026, 3, 8, 7, 0, tzinfo=timezone.utc), "Sunday 3:00 AM", "EDT (America/New_York)"),
    (datetime(2026, 11, 1, 5, 59, tzinfo=timezone.utc), "Sunday 1:59 AM", "EDT (America/New_York)"),
    (datetime(2026, 11, 1, 6, 0, tzinfo=timezone.utc), "Sunday 1:00 AM", "EST (America/New_York)"),
])
def test_the_abbreviation_is_the_zones_own_at_that_instant(monkeypatch, instant, clock, zone):
    configure(monkeypatch, "America/New_York")
    assert T.describe_agent_now(instant) == f"Nova's time: {clock} {zone}."
    monkeypatch.setattr(T, "now_utc", lambda: instant)
    assert T.describe_owner_now(None, "the owner").splitlines() == [
        f"Nova's time: {clock} {zone}.", f"the owner's location: no recent report; assume {zone}."]


class Contacts:
    def __init__(self, owner):
        self.owner = owner

    async def get(self, contact_id):
        return self.owner if contact_id == "owner-1" else None


@pytest.mark.asyncio
@pytest.mark.parametrize("reported", [False, True])
async def test_no_recorded_reported_or_passed_zone_moves_the_home_clock(monkeypatch, reported):
    configure(monkeypatch)
    owner = SimpleNamespace(
        timezone="Asia/Tokyo", display_name="Synthetic Owner", given_name=None, last_interaction_at=None,
        location_city="Testville", location_region=None, location_country_code="JP",
        location_reported_at=(NOW - timedelta(minutes=14)).isoformat() if reported else None)
    monkeypatch.setattr(host, "_contacts_store", Contacts(owner))
    monkeypatch.setattr(host, "owner_person_id", lambda: "owner-1")
    home = "Nova's time: Saturday 4:14 PM CEST (Europe/Berlin)."
    for contact, private in (("guest-1", True), ("owner-1", False), (None, False)):
        section = await host._build_temporal_section(contact, "Pacific/Honolulu", include_global_heads_up=False,
                                                     owner_private_turn=private)
        assert section.body == home, (contact, private)
    section = await host._build_temporal_section("owner-1", "Pacific/Honolulu", include_global_heads_up=False,
                                                 owner_private_turn=True)
    assert section.body.splitlines() == [home, (
        "Synthetic Owner: Testville (JP), local time 11:14 PM (Asia/Tokyo), reported 14 min ago." if reported
        else "Synthetic Owner's location: no recent report; assume CEST (Europe/Berlin).")]


def test_no_fixed_home_zone_is_left():
    assert not hasattr(T, "OWNER_AGENT_TIMEZONE")
