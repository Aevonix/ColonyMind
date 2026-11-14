"""Synthetic phone reports through real credential middleware and the contact store."""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from colonymind.api.credentials import CredentialTable, secret_digest
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.owner_location import PATH
from colonymind.api.routers import host
from colonymind.contacts.config import ContactsConfig
from colonymind.contacts.store import SQLiteContactStore
from colonymind.util import temporal

pytestmark = pytest.mark.asyncio
NOW = datetime(2026, 10, 17, 14, 14, tzinfo=timezone.utc)


def marker():
    return "marker-" + str(uuid4())


def report(**changes):
    return {"timezone": "Asia/Tokyo", "city": "Testville", "region": "Example Region",
            "country_code": "JP", "reported_at": "2026-10-17T14:00:00Z", **changes}


@pytest.fixture
async def phone(tmp_path, monkeypatch):
    store = SQLiteContactStore(ContactsConfig(sqlite_path=str(tmp_path / "contacts.db")))
    await store.connect()
    owner = await store.create(display_name=marker())
    monkeypatch.setenv("COLONYMIND_OWNER_PERSON_ID", owner.contact_id)
    monkeypatch.setenv("COLONYMIND_AGENT_TIMEZONE", "Pacific/Honolulu")
    monkeypatch.setenv("COLONYMIND_PERSONA_NAME", "Nova")
    monkeypatch.setattr(host, "_contacts_store", store)
    monkeypatch.setattr(time, "time", lambda: NOW.timestamp())
    tokens = {name: marker() for name in ("phone", "multi", "guest", "other", "wildcard", "reader", "instance")}
    platforms = {"phone": ["companion"], "multi": ["whatsapp", "companion"], "guest": [],
                 "other": ["whatsapp"], "wildcard": "any", "reader": ["companion"]}
    table = CredentialTable.from_config({name: {
        "sha256": secret_digest(tokens[name]), "owner": channels, "contacts": "all",
        "access": ["read"] if name == "reader" else ["manage"],
    } for name, channels in platforms.items()})
    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware, api_key=tokens["instance"], credentials=table)
    app.include_router(host.router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield SimpleNamespace(client=client, store=store, owner=owner, tokens=tokens)
    await store.close()


async def send(phone, body=None, credential="phone", headers=None):
    auth = {"Authorization": "Bearer " + phone.tokens.get(credential, credential)} if credential else {}
    return await phone.client.post(PATH, json=report() if body is None else body, headers={**auth, **(headers or {})})


@pytest.mark.parametrize("credential", ["phone", "multi"])
async def test_hub_style_owner_credential_applies_report_and_audits(phone, credential):
    result = await send(phone, credential=credential)
    assert result.status_code == 200, result.text
    assert result.json() == {"ok": True, "applied": True, "timezone": "Asia/Tokyo",
                             "city": "Testville", "reported_at": report()["reported_at"]}
    contact = await phone.store.get(phone.owner.contact_id)
    assert (contact.timezone, contact.location_city, contact.location_region, contact.location_country_code,
            contact.location_reported_at) == ("Asia/Tokyo", "Testville", "Example Region", "JP", report()["reported_at"])
    audits = await phone.store.get_audit_log(contact.contact_id)
    changes = {a["action"]: a for a in audits if a["performed_by"] == "companion-location"}
    assert set(changes) == {"timezone_changed", "location_reported"}
    assert json.loads(changes["timezone_changed"]["detail"])["new_timezone"] == "Asia/Tokyo"
    assert json.loads(changes["location_reported"]["detail"]) == {
        "city": "Testville", "region": "Example Region", "country_code": "JP", "reported_at": report()["reported_at"]}


@pytest.mark.parametrize("credential", ["guest", "other", "wildcard", "reader", "instance", None, "invalid"])
async def test_other_credentials_and_spoofed_headers_are_forbidden(phone, credential):
    before = await phone.store.get_audit_log(phone.owner.contact_id)
    result = await send(phone, credential=credential, headers={
        "X-Owner-Platforms": "companion", "X-Access": "manage", "X-Owner": phone.owner.contact_id,
        "Owner-Goal-Action": "companion-owner-action-v1:fake", "X-Principal-Id": "phone"})
    assert result.status_code == 403, result.text
    assert await phone.store.get_audit_log(phone.owner.contact_id) == before
    assert (await phone.store.get(phone.owner.contact_id)).timezone is None


async def test_no_key_development_mode_is_forbidden(phone):
    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware)
    app.include_router(host.router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
        assert (await client.post(PATH, json=report())).status_code == 403


@pytest.mark.parametrize("changes", [
    {"timezone": "Mars/Phobos"}, {"timezone": "../UTC"}, {"timezone": 3}, {"timezone": "utc"},
    {"city": "x" * 121}, {"city": 1}, {"city": None}, {"region": "x" * 61},
    {"country_code": "USA"}, {"country_code": "ZZ"}, {"country_code": "us"},
    {"reported_at": "2026-10-17T14:00:00"}, {"reported_at": "2026-10-17T14:00:00+01:00"},
    {"reported_at": 1234}, {"reported_at": "2026-02-30T00:00:00Z"},
    {"latitude": 0}, {"longitude": 0}, {"coordinates": [0, 0]}, {"unexpected": "value"},
])
async def test_validation_has_no_coordinate_or_unknown_field_escape(phone, changes):
    result = await send(phone, report(**changes))
    assert result.status_code == 422, result.text
    assert (await phone.store.get(phone.owner.contact_id)).location_reported_at is None


@pytest.mark.parametrize("missing", ["timezone", "reported_at"])
async def test_required_fields(phone, missing):
    body = report()
    del body[missing]
    assert (await send(phone, body)).status_code == 422


async def test_optional_strings_and_utc_offset_form(phone):
    result = await send(phone, {"timezone": "Europe/Paris", "reported_at": "2026-10-17T14:00:00+00:00"})
    assert result.status_code == 200, result.text
    assert result.json()["city"] is None
    assert result.json()["reported_at"] == "2026-10-17T14:00:00Z"
    # Equal instants are idempotent even with a different ISO spelling.
    replay = await send(phone, {"timezone": "Europe/Paris", "reported_at": "2026-10-17T14:00:00Z"})
    assert replay.json()["applied"] is False


async def test_replays_equal_and_older_reports_change_nothing(phone):
    accepted = (await send(phone)).json()
    before = (await phone.store.get(phone.owner.contact_id)).to_dict()
    audits = await phone.store.get_audit_log(phone.owner.contact_id)
    for body in (report(), report(timezone="Europe/Paris", city="Other Testville"),
                 report(timezone="Europe/Paris", reported_at="2026-10-16T14:00:00Z")):
        result = await send(phone, body)
        assert result.status_code == 200, result.text
        assert result.json() == {**accepted, "applied": False}
        assert (await phone.store.get(phone.owner.contact_id)).to_dict() == before
        assert await phone.store.get_audit_log(phone.owner.contact_id) == audits
    # A newer report without optional fields replaces the whole previous report.
    applied = await send(phone, {"timezone": "Europe/Paris", "reported_at": "2026-10-17T14:00:00.000001Z"})
    assert applied.json()["applied"] is True
    latest = await phone.store.get(phone.owner.contact_id)
    assert latest.timezone == "Europe/Paris"
    assert latest.location_city is latest.location_region is latest.location_country_code is None


async def test_concurrent_writers_keep_the_latest_complete_report(phone):
    other = SQLiteContactStore(phone.store._config)
    await other.connect()
    try:
        newer = report(city="New Testville", timezone="Europe/Paris", country_code="FR",
                       reported_at="2026-10-17T14:01:00Z")
        await asyncio.gather(phone.store.set_location_report(phone.owner.contact_id, **report()),
                             other.set_location_report(phone.owner.contact_id, **newer))
        latest = await phone.store.get(phone.owner.contact_id)
        assert latest.timezone == newer["timezone"] and latest.location_city == newer["city"]
        assert latest.location_reported_at == newer["reported_at"]
        assert latest.location_country_code == newer["country_code"]
    finally:
        await other.close()


async def test_failed_report_rolls_back_timezone_report_and_audits(phone, monkeypatch):
    before = (await phone.store.get(phone.owner.contact_id)).to_dict()
    audits = await phone.store.get_audit_log(phone.owner.contact_id)
    record = phone.store.record_audit
    async def fail(contact_id, action, *args, **kwargs):
        if action == "location_reported":
            raise RuntimeError("synthetic write failure")
        return await record(contact_id, action, *args, **kwargs)
    monkeypatch.setattr(phone.store, "record_audit", fail)
    with pytest.raises(RuntimeError, match="synthetic write failure"):
        await phone.store.set_location_report(phone.owner.contact_id, **report())
    assert (await phone.store.get(phone.owner.contact_id)).to_dict() == before
    assert await phone.store.get_audit_log(phone.owner.contact_id) == audits


@pytest.mark.parametrize("age,expected", [
    (None, "no recent report; assume HST (Pacific/Honolulu)."),
    (24 * 3600 + 1, "no recent report; assume HST (Pacific/Honolulu)."),
    (24 * 3600, "reported 24 h ago."), (2 * 3600, "reported 2 h ago."),
    (14 * 60, "reported 14 min ago."), (0, "reported 0 min ago."),
])
async def test_owner_context_freshness_and_fixed_et_clock(phone, age, expected):
    if age is not None:
        stamp = (NOW - timedelta(seconds=age)).isoformat().replace("+00:00", "Z")
        assert (await send(phone, report(reported_at=stamp))).status_code == 200
    # An existing manually recorded timezone alone is never a phone report.
    else:
        await phone.store.set_timezone(phone.owner.contact_id, "Asia/Tokyo")
    result = await phone.client.get("/v1/host/context/temporal", params={
        "contact_id": phone.owner.contact_id, "channel": "companion", "tz": "Europe/London",
        "owner_private_turn": True},
        headers={"Authorization": "Bearer " + phone.tokens["phone"]})
    assert result.status_code == 200, result.text
    lines = result.json()["body"].splitlines()
    assert lines[0] == "Nova's time: Saturday 4:14 AM HST (Pacific/Honolulu)."
    assert lines[1].endswith(expected)
    if age is not None and age <= 24 * 3600:
        assert lines[1].startswith(phone.owner.display_name + ": Testville, Example Region (JP), "
                                   "local time 11:14 PM (Asia/Tokyo), ")
    else:
        assert lines[1].startswith(phone.owner.display_name + "'s location:")
        assert "Testville" not in lines[1] and "Asia/Tokyo" not in lines[1]
    assert result.json()["owner_label"] == phone.owner.display_name


async def test_us_country_is_omitted_and_last_exchange_is_kept(phone):
    await send(phone, report(timezone="America/Los_Angeles", country_code="US"))
    await phone.store.update(phone.owner.contact_id, last_interaction_at="2026-10-17T13:14:00Z")
    section = await host._build_temporal_section(phone.owner.contact_id, include_global_heads_up=False,
                                                 owner_private_turn=True)
    assert f"{phone.owner.display_name}: Testville, Example Region, local time 7:14 AM" in section.body
    assert "(US)" not in section.body
    assert f"Last exchange with {phone.owner.display_name}: 60m ago" in section.body


@pytest.mark.parametrize("zone", [None, "Asia/Tokyo"])
async def test_non_owner_context_has_only_agent_et_clock(phone, zone):
    guest = await phone.store.create(display_name=marker())
    if zone:
        await phone.store.set_timezone(guest.contact_id, zone)
    section = await host._build_temporal_section(guest.contact_id, "Europe/London", include_global_heads_up=False)
    assert section.body == "Nova's time: Saturday 4:14 AM HST (Pacific/Honolulu)."


async def test_additive_migration_preserves_existing_contacts_and_reopens(tmp_path, monkeypatch):
    from colonymind import migrations
    import colonymind.contacts.store as contact_module
    path = tmp_path / "legacy.db"
    discover = migrations._discover
    with sqlite3.connect(path) as db, monkeypatch.context() as patch:
        patch.setattr(migrations, "_discover", lambda directory: [
            item for item in discover(directory) if item[0] != "008"])
        migrations.run_migrations_sync(db, Path(contact_module.__file__).parent / "migrations")
        db.execute("INSERT INTO contacts(contact_id, display_name, timezone, notes) VALUES (?, ?, ?, ?)",
                   ("synthetic-owner", "Synthetic Owner", "Europe/Paris", "synthetic note"))
        assert "location_reported_at" not in {r[1] for r in db.execute("PRAGMA table_info(contacts)")}
    for attempt in range(2):
        store = SQLiteContactStore(ContactsConfig(sqlite_path=str(path)))
        await store.connect()
        try:
            contact = await store.get("synthetic-owner")
            assert contact.display_name == "Synthetic Owner" and contact.notes == "synthetic note"
            if attempt == 0:
                assert contact.timezone == "Europe/Paris" and contact.location_reported_at is None
                await store.set_location_report(contact.contact_id, **report())
            else:
                assert contact.timezone == "Asia/Tokyo" and contact.location_city == "Testville"
                assert contact.location_reported_at == report()["reported_at"]
        finally:
            await store.close()
