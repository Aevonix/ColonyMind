"""Deployment-local SMS identity: no inferred owner phone and no durable environment alias."""

from types import SimpleNamespace

import pytest
import yaml
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from colonymind.api.errors import install_exception_handlers
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import host
from colonymind.config import ConfigError, apply_environment, load_config, owner_phone_handle
from colonymind.contacts.config import ContactsConfig
from colonymind.contacts.store import SQLiteContactStore
from colonymind.identity.participants import ParticipantResolver
from test_credential_attribution import GAME_SECRET, KEY, RELAY_SECRET, bearer, credentials, sources, turn

# Deliberately fictional placeholder numbers.
SMS = "+15550000011"
OTHER = "+15550000012"


@pytest.fixture(params=["sms", "rcs"])
def platform(request):
    return request.param


@pytest.fixture
def home(tmp_path, monkeypatch, platform):
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("COLONYMIND_OWNER_SMS_HANDLE", raising=False)
    monkeypatch.delenv("COLONYMIND_OWNER_RCS_HANDLE", raising=False)
    return tmp_path


def configure(home, value, platform):
    (home / "colonymind.yaml").write_text(yaml.safe_dump({"owner": {f"{platform}_handle": value}}))


def test_sms_setting_is_optional_private_and_environment_wins(home, platform):
    assert owner_phone_handle(platform, home, environ={}) == ""
    configure(home, SMS, platform)
    cfg = load_config(home, environ={})
    assert cfg.get(f"owner.{platform}_handle") == owner_phone_handle(platform, home, environ={}) == SMS
    environment = {}
    apply_environment(cfg, environ=environment)
    assert f"COLONYMIND_OWNER_{platform.upper()}_HANDLE" not in environment
    assert load_config(home, environ={f"COLONYMIND_OWNER_{platform.upper()}_HANDLE": OTHER}).get(f"owner.{platform}_handle") == OTHER
    assert owner_phone_handle(platform, home, environ={f"COLONYMIND_OWNER_{platform.upper()}_HANDLE": OTHER}) == OTHER
    assert owner_phone_handle(platform, home, environ={f"COLONYMIND_OWNER_{platform.upper()}_HANDLE": ""}) == ""
    assert load_config(home, environ={f"COLONYMIND_OWNER_{platform.upper()}_HANDLE": ""}).get(f"owner.{platform}_handle") == ""
    assert yaml.safe_load((home / "colonymind.yaml").read_text())["owner"][f"{platform}_handle"] == SMS


@pytest.mark.parametrize("value", [123, True, [SMS], "5550000011", "phone:" + SMS, "not-a-phone", SMS + "@example.test"])
def test_bad_sms_configuration_fails_without_disclosing_the_value(home, value, platform):
    configure(home, value, platform)
    with pytest.raises(ConfigError, match=f"owner.{platform}_handle") as error:
        load_config(home, environ={})
    assert str(value) not in str(error.value)
    with pytest.raises(ConfigError, match=f"owner.{platform}_handle"):
        owner_phone_handle(platform, home, environ={})


@pytest.fixture
async def world(home, monkeypatch, platform):
    store = SQLiteContactStore(ContactsConfig(sqlite_path=str(home / "contacts.db")))
    await store.connect()
    owner = (await store.create(display_name="Owner")).contact_id
    guest = (await store.create(display_name="Guest")).contact_id
    await store.add_handle(owner, "whatsapp", SMS, verified=True)
    await store.add_handle(guest, platform, OTHER, verified=True)
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", owner)
    monkeypatch.setattr(host, "_contacts_store", store)
    for name in ("_telemetry", "_reranker", "_context_recall_selector", "_comms_log", "_preference_learner"):
        monkeypatch.setattr(host, name, None, raising=False)
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(host.router)
    app.include_router(host.v2_router)
    app.add_middleware(ApiKeyMiddleware, api_key=KEY, credentials=credentials())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield SimpleNamespace(store=store, client=client, owner=owner, guest=guest, state=home)
    await store.close()


async def test_sms_does_not_infer_owner_from_whatsapp_or_a_contact_id(world, home, platform):
    resolver = ParticipantResolver(world.store)
    assert (await resolver.lookup(platform=platform, user_id=SMS)).contact_id is None
    assert (await resolver.lookup(platform=platform, user_id=world.owner)).contact_id is None
    assert (await resolver.lookup(platform="whatsapp", user_id=SMS)).contact_id == world.owner
    configure(home, SMS, platform)
    assert (await resolver.lookup(platform=platform, user_id=SMS)).contact_id == world.owner
    assert (await resolver.lookup(platform=platform, user_id=OTHER)).contact_id == world.guest


async def test_local_config_works_after_environment_setup_and_does_not_bind_other_platforms(world, home, monkeypatch, platform):
    (home / "colonymind.yaml").write_text(yaml.safe_dump({
        "owner": {"contact_id": world.owner, f"{platform}_handle": OTHER}}))
    monkeypatch.delenv("COLONYMIND_OWNER_CONTACT_ID")
    exported = {}
    apply_environment(load_config(home, environ={}), environ=exported)
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", exported["COLONYMIND_OWNER_CONTACT_ID"])
    assert (await world.store.resolve_messaging_handle(platform, OTHER)).contact_id == world.owner
    assert (await world.store.resolve_messaging_handle("sms" if platform == "rcs" else "rcs", OTHER)).contact_id == world.guest
    configure(home, "invalid", platform)
    assert (await world.store.resolve_messaging_handle(platform, OTHER)).contact_id == world.guest
    assert await world.store.resolve_messaging_handle(platform, SMS) is None


async def test_legacy_explicit_owner_sms_handles_are_independent_of_the_new_binding(world, home, platform):
    await world.store.add_handle(world.owner, platform, SMS, verified=True)
    configure(home, "", platform)
    assert (await world.store.resolve_messaging_handle(platform, SMS)).contact_id == world.owner


async def test_changing_and_removing_sms_binding_revokes_it_without_persisting_handles(world, home, monkeypatch, platform):
    before = await world.store.get_handles(world.owner)
    configure(home, SMS, platform)
    assert (await world.store.resolve_messaging_handle(platform, SMS)).contact_id == world.owner
    monkeypatch.setenv(f"COLONYMIND_OWNER_{platform.upper()}_HANDLE", OTHER)
    assert await world.store.resolve_messaging_handle(platform, SMS) is None
    assert (await world.store.resolve_messaging_handle(platform, OTHER)).contact_id == world.owner
    monkeypatch.setenv(f"COLONYMIND_OWNER_{platform.upper()}_HANDLE", "")
    assert (await world.store.resolve_messaging_handle(platform, OTHER)).contact_id == world.guest
    assert await world.store.resolve_messaging_handle(platform, SMS) is None
    monkeypatch.delenv(f"COLONYMIND_OWNER_{platform.upper()}_HANDLE")
    (home / "colonymind.yaml").unlink()
    assert await world.store.resolve_messaging_handle(platform, SMS) is None
    assert await world.store.get_handles(world.owner) == before


@pytest.mark.parametrize("version", [1, 2])
async def test_configured_sms_owner_is_attributed_by_server_and_integrations_cannot_impersonate(world, home, version, platform):
    configure(home, SMS, platform)
    body = turn("", turn_id="sms-1", channel=platform + ":" + SMS, sender={"platform": platform, "user_id": SMS})
    path = "/v1/host/turns/sync" if version == 1 else "/v2/host/turns/sms-1"
    method = "POST" if version == 1 else "PUT"
    for secret in (GAME_SECRET, RELAY_SECRET):
        denied = await world.client.request(method, path, headers=bearer(secret), json=body)
        assert denied.status_code == 403
    assert sources(world) == []
    response = await world.client.request(method, path, headers=bearer(KEY), json=body)
    assert response.status_code == (200 if version == 1 else 201), response.text
    assert sources(world) == [("sms-1", world.owner, "person")]
    resolved = await world.client.get("/v1/host/contacts/resolve", headers=bearer(KEY),
                                      params={"gateway": platform, "address": SMS})
    assert resolved.status_code == 200 and resolved.json()["contact_id"] == world.owner
