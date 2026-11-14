"""A 1.9.0 upgrade carries every keyring principal with exactly the authority it had, neither narrower nor
broader (docs/CREDENTIALS.md): the same persons, the same operations (reading, writing, the rest of the API),
the same owner authority, and, for a principal granted the owner, the platforms it could write the owner
from (every platform when it declared none). The upgrade report states each principal's before and after.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from colonymind import init
from colonymind.api.credentials import CredentialTable, secret_digest
from colonymind.config import load_config
from test_migration_190 import LEGACY_SECRET
from test_upgrade import HERMES_PYTHON

SECRETS = {name: f"{name}-secret-0000000000000000" for name in (
    "voice-gateway", "whatsapp-desk", "owner-cli", "owner-reader", "shared-reader", "alice-desk", "front-door",
    "ops", "idle")}
ALICE, BOB = "cid-alice", "cid-bob"


def principals(owner: str) -> list[dict]:
    """One principal of every kind the 1.9.0 keyring could express, besides the adapter's."""
    def principal(name, **fields):
        return {"principal": name, "status": "active", **fields,
                "credentials": [{"id": "initial", "secret": SECRETS.get(name, LEGACY_SECRET), "status": "active"}]}
    return [
        principal("hermes-local", viewer_person_id=owner, audiences=["viewer"], allow_unscoped_api=False,
                  turn_ingress_platforms=["cli", "whatsapp"],
                  scopes=["context:read", "memory:read", "memory:search", "memory:write", "turns:write",
                          "turns:resolve-sender"]),
        # The owner, from the platform it declared; writes only.
        principal("voice-gateway", viewer_person_id=owner, audiences=["viewer"], turn_ingress_platforms=["voice"],
                  scopes=["turns:write"]),
        # The owner; it declared WhatsApp, which 1.9.0 never made a limit.
        principal("whatsapp-desk", viewer_person_id=owner, turn_ingress_platforms=["whatsapp"],
                  scopes=["turns:write"]),
        # The owner, having declared no platform: from every platform.
        principal("owner-cli", viewer_person_id=owner, scopes=["turns:write", "context:read"]),
        # The owner's context, read only.
        principal("owner-reader", viewer_person_id=owner, audiences=["viewer"], scopes=["context:read",
                                                                                          "memory:search"]),
        # The shared lane only, read only.
        principal("shared-reader", audiences=["shared"], scopes=["context:read"]),
        # One person.
        principal("alice-desk", viewer_person_id=ALICE, person_ids=[ALICE],
                  scopes=["turns:write", "context:read", "memory:write"]),
        # A transport that resolves its senders.
        principal("front-door", viewer_person_id="cid-lobby", scopes=["turns:write", "turns:resolve-sender"],
                  turn_ingress_platforms=["sms"], attested_contact_grants={"platforms": ["sms"]}),
        # Every scope, no person.
        principal("ops", scopes=["*"]),
        # Nothing at all.
        principal("idle", scopes=[]),
    ]


@pytest.fixture
def migrated(tmp_path, monkeypatch):
    from test_migration_190 import _write_190_instance
    home, hermes_home = tmp_path / "colonymind", tmp_path / "hermes"
    owner = _write_190_instance(home, hermes_home, hermes_python=HERMES_PYTHON)
    monkeypatch.setenv("COLONYMIND_HOME", str(home))
    keyring = principals(owner)
    (home / "api-keyring.json").write_text(json.dumps({"version": 1, "principals": keyring}))
    report = "\n".join(init.migrate_legacy_instance(home))
    credentials = load_config(home, environ={}).get("credentials")
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", owner)
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path / "state"))
    return type("Migrated", (), {"owner": owner, "keyring": keyring, "report": report, "credentials": credentials,
                                 "table": CredentialTable.from_config(credentials), "tmp": tmp_path})


def call(table, secret: str, method: str, path: str, body: dict | None = None):
    """One request with ``secret`` to the host routes behind ``table`` (the adapter's secret is api.key)."""
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from colonymind.api.errors import install_exception_handlers
    from colonymind.api.middleware import ApiKeyMiddleware
    from colonymind.api.routers import host

    async def run():
        app = FastAPI()
        install_exception_handlers(app)
        app.include_router(host.router)
        app.include_router(host.v2_router)
        app.add_middleware(ApiKeyMiddleware, api_key=LEGACY_SECRET, credentials=table)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.request(method, path, json=body, headers={"Authorization": "Bearer " + secret})
    return asyncio.run(run())


@pytest.fixture(autouse=True)
def quiet_host(monkeypatch):
    from colonymind.api.routers import host
    for name in ("_contacts_store", "_telemetry", "_reranker", "_context_recall_selector", "_comms_log",
                 "_preference_learner", "_goals_store", "_initiative_store", "_facts_store", "_commitment_store"):
        monkeypatch.setattr(host, name, None, raising=False)


def turn(contact: str, turn_id: str, *, channel: str = "cli", sender: dict | None = None) -> dict:
    body = {"identity": {"host_id": "client"},
            "context": {"session_id": "s-1", "contact_id": contact, "turn_id": turn_id, "channel_id": channel},
            "user_message": {"role": "user", "content": "The gate code changed."},
            "assistant_message": {"role": "assistant", "content": "Noted."}}
    if sender:
        body["sender"] = sender
    return body


def assemble(contact: str = "", audience: str | None = None) -> dict:
    body = {"identity": {"host_id": "client"}, "context": {"session_id": "s-1", "contact_id": contact},
            "incoming_message": {"role": "user", "content": "What changed?"}}
    if audience:
        body["audience"] = audience
    return body


def status(migrated, name: str, method: str, path: str, body: dict | None = None) -> int:
    return call(migrated.table, SECRETS[name], method, path, body).status_code


def test_a_read_only_owner_principal_still_only_reads(migrated):
    owner = migrated.owner
    assert status(migrated, "owner-reader", "POST", "/v1/host/context/assemble", assemble(owner)) == 200
    assert status(migrated, "owner-reader", "POST", "/v1/host/turns/sync", turn(owner, "r-1")) == 403
    assert status(migrated, "owner-reader", "GET", "/v1/host/goals") == 403


def test_a_shared_lane_reader_reads_the_shared_lane_and_never_the_owners(migrated):
    assert status(migrated, "shared-reader", "POST", "/v1/host/context/assemble", assemble(audience="shared")) == 200
    for audience in ("owner", "global"):
        assert status(migrated, "shared-reader", "POST", "/v1/host/context/assemble",
                      assemble(audience=audience)) == 403, audience
    assert status(migrated, "shared-reader", "POST", "/v1/host/context/assemble", assemble(migrated.owner)) == 403
    assert status(migrated, "shared-reader", "POST", "/v1/host/turns/sync", turn("shared", "s-1")) == 403


def test_a_principal_bound_to_one_person_writes_only_as_that_person(migrated):
    assert status(migrated, "alice-desk", "POST", "/v1/host/turns/sync", turn(ALICE, "a-1")) == 200
    assert status(migrated, "alice-desk", "POST", "/v1/host/turns/sync", turn("", "a-2")) == 200   # its viewer
    for contact in (BOB, migrated.owner, "credential:alice-desk"):
        assert status(migrated, "alice-desk", "POST", "/v1/host/turns/sync", turn(contact, f"a-{contact}")) == 403
    assert status(migrated, "alice-desk", "POST", "/v1/host/context/assemble", assemble(BOB)) == 403


def test_owner_principals_keep_the_owner_from_every_platform_they_could(migrated):
    """1.9.0 never limited a principal granted the owner to the platforms it declared: each still writes the
    owner's turns from any platform, the declared ones and the rest alike."""
    owner = migrated.owner
    voice = turn(owner, "v-1", channel="voice", sender={"platform": "voice", "user_id": owner})
    response = call(migrated.table, SECRETS["voice-gateway"], "POST", "/v1/host/turns/sync", voice)
    assert response.status_code == 200, response.text
    for name in ("voice-gateway", "whatsapp-desk", "owner-cli"):
        for channel in ("cli", "whatsapp:+15550000001", "sms:+15550000001", "rcs:+15550000001", "telegram:1001"):
            assert status(migrated, name, "POST", "/v1/host/turns/sync",
                          turn(owner, f"{name}-{channel}", channel=channel)) == 200, (name, channel)
    assert status(migrated, "voice-gateway", "POST", "/v1/host/context/assemble", assemble(owner)) == 403
    assert status(migrated, "owner-cli", "GET", "/v1/host/goals") == 403          # no generic API access


def test_each_principal_reaches_exactly_the_routes_its_scopes_reached(migrated):
    """The 1.9.0 context:read route rules carry over to the temporal brief."""
    temporal = f"/v1/host/context/temporal?contact_id={migrated.owner}"
    for name in ("owner-cli", "owner-reader"):                       # context:read
        response = call(migrated.table, SECRETS[name], "GET", temporal)
        assert response.status_code == 200, (name, response.text)
    assert status(migrated, "voice-gateway", "GET", temporal) == 403


def test_everything_and_nothing(migrated):
    assert status(migrated, "ops", "GET", "/v1/host/goals") != 403
    assert status(migrated, "ops", "POST", "/v1/host/turns/sync", turn(migrated.owner, "o-1")) == 403
    for method, path, body in (("POST", "/v1/host/turns/sync", turn("", "i-1")),
                               ("POST", "/v1/host/context/assemble", assemble()),
                               ("GET", "/v1/host/goals", None)):
        assert status(migrated, "idle", method, path, body) == 403, path


def test_every_principal_has_exactly_the_authority_it_had(migrated):
    """Before (the 1.9.0 principal) equals after (its credential), for every kind; the report says both."""
    from colonymind.api.credentials import authority_of
    for raw in migrated.keyring[1:]:
        name = raw["principal"]
        before, _unmapped = init.legacy_authority(raw, migrated.owner)
        after = authority_of(migrated.table.get(name), migrated.owner)
        assert before == after, (name, before, after)
        line = next(line for line in migrated.report.splitlines() if f"(1.9.0 principal {name})" in line)
        said_before = line.split("before: ", 1)[1].split("; after: ", 1)[0]
        said_after = line.split("; after: ", 1)[1].split("; its 1.9.0 scopes", 1)[0]
        assert said_before == said_after, line
    assert secret_digest(SECRETS["ops"]) in json.dumps(migrated.credentials)
    for secret in [LEGACY_SECRET, *SECRETS.values()]:
        assert secret not in migrated.report
    # Every scope is carried, the ones without a named operation here too (memory:write).
    assert migrated.credentials["alice-desk"]["scopes"] == ["context:read", "memory:write", "turns:write"]
    desk = next(line for line in migrated.report.splitlines() if "(1.9.0 principal alice-desk)" in line)
    assert "not carried" not in desk
    adapter = next(line for line in migrated.report.splitlines() if "from 1.9.0 principal hermes-local)" in line)
    assert "after: may write and read as the owner from any platform and anyone the sidecar resolves" in adapter



def test_senders_resolve_only_to_whom_the_principal_could_write_as(migrated, monkeypatch):
    """With the contact store running: a principal bound to one person may not reach another through a sender;
    one that resolved senders still writes as the guest a sender is, and never as the owner."""
    from colonymind.api.routers import host
    from colonymind.contacts.config import ContactsConfig
    from colonymind.contacts.store import SQLiteContactStore

    async def seeded():
        store = SQLiteContactStore(ContactsConfig(sqlite_path=str(migrated.tmp / "contacts.db")))
        await store.connect()
        await store.add_handle(migrated.owner, "sms", "+15550000001", verified=True) if await store.get(
            migrated.owner) else None
        guest = (await store.create(display_name="Guest")).contact_id
        await store.add_handle(guest, "sms", "+15550000002", verified=True)
        return store, guest
    store, guest = asyncio.run(seeded())
    try:
        senders_are_checked(migrated, monkeypatch, store, guest)
    finally:
        asyncio.run(store.close())


def senders_are_checked(migrated, monkeypatch, store, guest):
    from colonymind.api.routers import host
    owner_handle = {"platform": "sms", "user_id": "+15550000001"}
    guest_handle = {"platform": "sms", "user_id": "+15550000002"}
    monkeypatch.setattr(host, "_contacts_store", store)
    if asyncio.run(store.get(migrated.owner)) is None:       # the fixture's owner lives in another store
        owner_id = asyncio.run(store.create(display_name="Owner")).contact_id
        asyncio.run(store.add_handle(owner_id, "sms", "+15550000001", verified=True))
        monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", owner_id)
    assert status(migrated, "alice-desk", "POST", "/v1/host/turns/sync",
                  turn(ALICE, "s-1", channel="sms", sender=guest_handle)) == 403
    assert status(migrated, "front-door", "POST", "/v1/host/turns/sync",
                  turn("", "s-2", channel="sms", sender=guest_handle)) == 200
    # 1.9.0 let a principal that resolved its senders name the resolved guest in the body too.
    assert status(migrated, "front-door", "POST", "/v1/host/turns/sync",
                  turn(guest, "s-4", channel="sms", sender=guest_handle)) == 200
    assert status(migrated, "front-door", "POST", "/v1/host/turns/sync",
                  turn("", "s-3", channel="sms", sender=owner_handle)) == 403
