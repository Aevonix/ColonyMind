"""The credential grants in ``colonymind.yaml``: validation, init, upgrade and ``colonymind credentials``."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
from types import SimpleNamespace

import pytest
import yaml

from colonymind import init
from colonymind.api.credentials import (
    DEFAULT_OWNER_PLATFORMS,
    CredentialTable,
    describe,
    secret_digest,
)
from colonymind.config import ConfigError, load_config, validate
from colonymind import config as configuration
from test_init import STOCK_CONFIG, _args
from test_migration_190 import LEGACY_SECRET
from test_upgrade import HERMES_PYTHON, _upgrade_args

DIGEST = "a" * 64


def _validated(credentials):
    data = configuration._merge(configuration.DEFAULTS, {"credentials": credentials})
    return validate(data)["credentials"]


def test_the_default_grants_the_instance_key_the_owner_on_messaging_and_voice():
    assert DEFAULT_OWNER_PLATFORMS == ("whatsapp", "sms", "rcs", "voice")
    table = CredentialTable.from_config(None)
    key = table.instance
    assert key.name == "api-key" and key.identity == "credential:api-key"
    assert key.owner_platforms == frozenset(DEFAULT_OWNER_PLATFORMS) and key.contacts == "all"
    assert describe(key) == ("may write as the owner from whatsapp, sms, rcs and voice, and as anyone else "
                             "the sidecar resolves; any other turn only as credential:api-key")


@pytest.mark.parametrize("credentials,message", [
    ({"Bad Name": {"sha256": DIGEST}}, "credentials.Bad Name"),
    ({"api-key": {"sha256": DIGEST}}, "api-key is the instance key"),
    ({"bot": {"owner": [], "contacts": "own"}}, "credentials.bot.sha256"),
    ({"bot": {"sha256": "short", "contacts": "own"}}, "credentials.bot.sha256"),
    ({"bot": {"sha256": DIGEST, "owner": ["whatsapp"], "contacts": "own"}}, "contacts: own"),
    ({"bot": {"sha256": DIGEST, "owner": "any", "contacts": "own"}}, "contacts: own"),
    ({"bot": {"sha256": DIGEST, "owner": ["What's App"]}}, "credentials.bot.owner"),
    ({"bot": {"sha256": DIGEST, "contacts": "some"}}, "credentials.bot.contacts"),
    ({"bot": {"sha256": DIGEST, "mode": "x"}}, "unknown key mode"),
    ({"a": {"sha256": DIGEST}, "b": {"sha256": DIGEST}}, "same secret"),
    ({"anonymous-dev": {"sha256": DIGEST}}, "development mode"),
])
def test_invalid_grants_are_refused_with_the_key_named(credentials, message):
    with pytest.raises(ConfigError, match=message):
        _validated(credentials)


def test_valid_grants_are_normalized():
    checked = _validated({"api-key": {"owner": ["WhatsApp", "voice", "voice"]},
                          "bot": {"sha256": DIGEST.upper()}})
    assert checked["api-key"] == {"owner": ["whatsapp", "voice"], "contacts": "all"}
    assert checked["bot"] == {"sha256": DIGEST, "owner": [], "contacts": "own"}


def test_a_credential_is_matched_by_its_secret_and_nothing_else():
    table = CredentialTable.from_config({"bot": {"sha256": secret_digest("s3cret-token")}})
    assert table.match("s3cret-token").name == "bot"
    assert table.match("s3cret-token ") is None and table.match("") is None and table.match("other") is None
    assert secret_digest("s3cret-token") == hashlib.sha256(b"s3cret-token").hexdigest()


@pytest.fixture
def homes(tmp_path, monkeypatch):
    home = tmp_path / "colonymind"
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(STOCK_CONFIG, sort_keys=False))
    for name, value in (("COLONYMIND_HOME", home), ("COLONYMIND_STATE_DIR", home), ("HERMES_HOME", hermes_home)):
        monkeypatch.setenv(name, str(value))
    monkeypatch.delenv("COLONYMIND_API_KEY", raising=False)
    monkeypatch.delenv("COLONYMIND_OWNER_CONTACT_ID", raising=False)
    return home, hermes_home


def _raw(home):
    return yaml.safe_load((home / "colonymind.yaml").read_text())


def test_init_records_the_instance_keys_grants_and_says_what_it_may_write_as(homes, capsys):
    """A fresh instance's key writes the owner from the terminal and the platforms of the owner's handles."""
    home, hermes_home = homes
    assert init.run_init(_args(home, hermes_home)) == 0
    assert _raw(home)["credentials"] == {"api-key": {"owner": ["cli", "telegram"], "contacts": "all"}}
    out = capsys.readouterr().out
    assert ("credentials: api-key (api.key: the Hermes adapter, and any client that shares its key) may write "
            "as the owner from cli and telegram") in out
    assert "--owner-platform" not in out
    # A handle added later on a platform the key may not write the owner from: said, not granted.
    assert init.run_init(_args(home, hermes_home, owner_handle=["telegram=1001", "discord=42"])) == 0
    out = capsys.readouterr().out
    assert _raw(home)["credentials"]["api-key"]["owner"] == ["cli", "telegram"]
    assert "discord" in out and "--owner-platform discord" in out
    assert init.run_init(_args(home, hermes_home, owner_platform=["cli", "telegram", "discord"])) == 0
    assert _raw(home)["credentials"]["api-key"]["owner"] == ["cli", "telegram", "discord"]


def test_upgrade_writes_explicit_grants_for_an_instance_without_them(homes, capsys):
    home, hermes_home = homes
    assert init.run_init(_args(home, hermes_home)) == 0
    raw = _raw(home)
    del raw["credentials"]                                   # an instance from the one-key line
    (home / "colonymind.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    capsys.readouterr()
    assert init.run_upgrade(_upgrade_args(home)) == 0
    out = capsys.readouterr().out
    assert _raw(home)["credentials"] == {"api-key": {"owner": list(DEFAULT_OWNER_PLATFORMS), "contacts": "all"}}
    assert "credential grants recorded" in out
    assert ("api-key (api.key: the Hermes adapter, and any client that shares its key) may write as the owner "
            "from whatsapp, sms, rcs and voice") in out
    assert "credential:api-key" in out
    assert init.run_upgrade(_upgrade_args(home)) == 0
    assert "nothing to do" in capsys.readouterr().out


def test_the_190_migration_carries_other_live_principals_with_exactly_their_authority(tmp_path, monkeypatch):
    from test_migration_190 import _write_190_instance
    home, hermes_home = tmp_path / "colonymind", tmp_path / "hermes"
    _write_190_instance(home, hermes_home, hermes_python=HERMES_PYTHON)
    monkeypatch.setenv("COLONYMIND_HOME", str(home))
    (home / "api-keyring.json").write_text(json.dumps({"version": 1, "principals": [
        {"principal": "hermes-local", "status": "active",
         "credentials": [{"id": "initial", "secret": LEGACY_SECRET, "status": "active"}]},
        {"principal": "Game_Bot", "status": "active",
         "credentials": [{"id": "old", "secret": "game-old-secret-000000000", "status": "revoked"},
                         {"id": "initial", "secret": "game-secret-0000000000000", "status": "active"}]},
        {"principal": "gone", "status": "revoked",
         "credentials": [{"id": "initial", "secret": "gone-secret-00000000000000", "status": "active"}]},
    ]}))
    notes = init.migrate_legacy_instance(home)
    credentials = load_config(home, environ={}).get("credentials")
    assert credentials["api-key"] == {"owner": list(DEFAULT_OWNER_PLATFORMS), "contacts": "all"}
    # Granted no person and no scope in 1.9.0, it may do nothing now either.
    assert credentials["game_bot"] == {"sha256": secret_digest("game-secret-0000000000000"), "owner": [],
                                       "contacts": [], "scopes": []}
    assert set(credentials) == {"api-key", "game_bot"}
    text = "\n".join(notes)
    assert "api-key (api.key, the Hermes adapter's, from 1.9.0 principal hermes-local): before: " in text
    nothing = ("may neither write nor read (its persons: no one); a request that names no one is refused; it "
               "reaches the routes its 1.9.0 scopes reached (none)")
    assert f"game_bot (1.9.0 principal Game_Bot): before: {nothing}; after: {nothing}" in text
    assert "1.9.0 principal gone (revoked) is not carried: none of its secrets is accepted" in text
    assert "game-secret" not in text and LEGACY_SECRET not in text


VOICE_SECRET = "voice-gateway-secret-000000000"
DESK_SECRET = "front-desk-secret-00000000000"
GAME_SECRET = "game-bot-secret-0000000000000"


def _migrate(tmp_path, monkeypatch, principals):
    """A 1.9.0 instance whose keyring holds ``principals(owner_id)``, migrated; returns the instance, the owner,
    the credentials written and the upgrade report."""
    from test_migration_190 import _write_190_instance
    home, hermes_home = tmp_path / "colonymind", tmp_path / "hermes"
    owner = _write_190_instance(home, hermes_home, hermes_python=HERMES_PYTHON)
    monkeypatch.setenv("COLONYMIND_HOME", str(home))
    (home / "api-keyring.json").write_text(json.dumps({"version": 1, "principals": principals(owner)}))
    notes = init.migrate_legacy_instance(home)
    return home, owner, load_config(home, environ={}).get("credentials"), "\n".join(notes)


def test_the_report_says_why_each_principal_is_not_carried(tmp_path, monkeypatch):
    """A live principal is left behind only for want of a live secret; any other for its status or its time."""
    home, owner, credentials, report = _migrate(tmp_path, monkeypatch, lambda owner: [
        {"principal": "hermes-local", "status": "active", "viewer_person_id": owner,
         "credentials": [{"id": "initial", "secret": LEGACY_SECRET, "status": "active"}]},
        {"principal": "keyless", "status": "active",
         "credentials": [{"id": "old", "secret": "keyless-secret-000000000000", "status": "revoked"}]},
        {"principal": "lapsed", "status": "retiring", "accept_until": "2020-01-01T00:00:00Z",
         "credentials": [{"id": "c", "secret": "lapsed-secret-0000000000000", "status": "active"}]},
        {"principal": "switched-off", "status": "disabled",
         "credentials": [{"id": "c", "secret": "disabled-secret-000000000000", "status": "active"}]},
    ])
    assert set(credentials) == {"api-key"}
    for name, reason in (("keyless", "no live secret"), ("lapsed", "expired"), ("switched-off", "disabled")):
        assert f"1.9.0 principal {name} ({reason}) is not carried: none of its secrets is accepted" in report


def _voice_turn(owner: str) -> dict:
    return {"identity": {"host_id": "voice"},
            "context": {"session_id": "", "contact_id": owner, "channel_id": "voice"},
            "sender": {"platform": "voice", "user_id": owner, "display_name": "Owner"},
            "user_message": {"role": "user", "content": "Remind me to water the ferns."},
            "assistant_message": {"role": "assistant", "content": "I will."}}


async def _post(table, secret: str, body: dict, monkeypatch, state, *, api_key: str = "the-instance-key-0000000000"):
    """``body`` posted to the turn route with ``secret``, behind ``table`` (``api_key`` is the instance key)."""
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from colonymind.api.errors import install_exception_handlers
    from colonymind.api.middleware import ApiKeyMiddleware
    from colonymind.api.routers import host
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(state))
    for name in ("_contacts_store", "_telemetry", "_reranker", "_context_recall_selector", "_comms_log",
                 "_preference_learner"):
        monkeypatch.setattr(host, name, None, raising=False)
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(host.router)
    app.add_middleware(ApiKeyMiddleware, api_key=api_key, credentials=table)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post("/v1/host/turns/sync", json=body, headers={"Authorization": "Bearer " + secret})


def _cli_turn(owner: str, turn_id: str, channel: str = "cli") -> dict:
    return {"identity": {"host_id": "hermes"},
            "context": {"session_id": "s-cli", "contact_id": owner, "turn_id": turn_id, "channel_id": channel},
            "user_message": {"role": "user", "content": "Remember the boiler service is on Friday."},
            "assistant_message": {"role": "assistant", "content": "Noted."}}


def test_an_owner_principal_that_declared_no_platform_keeps_the_owner_from_every_platform(tmp_path, monkeypatch):
    """1.9.0 did not limit a principal granted the owner to its declared platforms, so one that declared none
    wrote the owner's turns from any platform, the CLI included. It still does after the upgrade."""
    home, owner, credentials, report = _migrate(tmp_path, monkeypatch, lambda owner: [
        {"principal": "hermes-local", "status": "active", "viewer_person_id": owner, "scopes": ["turns:write"],
         "credentials": [{"id": "initial", "secret": LEGACY_SECRET, "status": "active"}]},
        {"principal": "desk-cli", "status": "active", "viewer_person_id": owner, "scopes": ["turns:write"],
         "credentials": [{"id": "initial", "secret": DESK_SECRET, "status": "active"}]}])
    assert credentials["api-key"]["owner"] == "any" and credentials["desk-cli"]["owner"] == "any"
    table = CredentialTable.from_config(credentials)
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", owner)
    for secret in (LEGACY_SECRET, DESK_SECRET):
        for channel in ("cli", "telegram:1001"):
            response = asyncio.run(_post(table, secret, _cli_turn(owner, f"{secret[:4]}-{channel}", channel),
                                         monkeypatch, tmp_path / "state", api_key=LEGACY_SECRET))
            assert response.status_code == 200 and response.json()["source_recorded"] is True, response.text
    assert "desk-cli (1.9.0 principal desk-cli): before: may write as the owner from any platform" in report


def test_the_190_migration_keeps_every_live_secret_with_its_expiry(tmp_path, monkeypatch):
    """A rotating principal keeps every secret still accepted, each until its own expiry, and the principal's
    own expiry bounds them all; the sidecar refuses each one once its time is up."""
    from datetime import timedelta
    from colonymind.util.temporal import now_utc
    now = now_utc().replace(microsecond=0)
    soon, later = now + timedelta(days=2), now + timedelta(days=30)
    home, owner, credentials, report = _migrate(tmp_path, monkeypatch, lambda owner: [
        {"principal": "hermes-local", "status": "active", "viewer_person_id": owner, "credentials": [
            {"id": "old", "secret": "adapter-old-secret-0000000000", "status": "retiring",
             "accept_until": soon.isoformat()},
            {"id": "initial", "secret": LEGACY_SECRET, "status": "active"}]},
        {"principal": "bot", "status": "retiring", "accept_until": later.isoformat(), "credentials": [
            {"id": "old", "secret": "bot-old-secret-00000000000000", "status": "retiring",
             "accept_until": soon.isoformat()},
            {"id": "current", "secret": "bot-new-secret-00000000000000", "status": "active"},
            {"id": "dead", "secret": "bot-dead-secret-0000000000000", "status": "revoked"}]},
    ])
    assert init.read_api_key(home, environ={}) == LEGACY_SECRET
    assert credentials["bot"] == {
        "secrets": [{"sha256": secret_digest("bot-old-secret-00000000000000"), "accept_until": soon.isoformat()},
                    {"sha256": secret_digest("bot-new-secret-00000000000000")}],
        "accept_until": later.isoformat(), "owner": [], "contacts": [], "scopes": []}
    assert credentials["api-key"]["secrets"] == [
        {"sha256": secret_digest("adapter-old-secret-0000000000"), "accept_until": soon.isoformat()}]
    table = CredentialTable.from_config(credentials)
    assert table.match("bot-old-secret-00000000000000", now=now).name == "bot"
    assert table.match("bot-new-secret-00000000000000", now=now).name == "bot"
    assert table.match("adapter-old-secret-0000000000", now=now).instance
    assert table.match("bot-dead-secret-0000000000000", now=now) is None
    after_soon = now + timedelta(days=3)
    assert table.match("bot-old-secret-00000000000000", now=after_soon) is None      # its own expiry
    assert table.match("adapter-old-secret-0000000000", now=after_soon) is None
    assert table.match("bot-new-secret-00000000000000", now=after_soon).name == "bot"
    assert table.match("bot-new-secret-00000000000000", now=later) is None            # the principal's
    bot = next(line for line in report.splitlines() if "(1.9.0 principal bot)" in line)
    assert f"; accepted until {later.isoformat()}; 2 secrets (until {soon.isoformat()}, no expiry)" in bot
    adapter = next(line for line in report.splitlines() if "from 1.9.0 principal hermes-local)" in line)
    assert f"; also accepts 1 other secret (until {soon.isoformat()})" in adapter


def test_a_live_legacy_bearer_stays_accepted_as_the_instance_key(tmp_path, monkeypatch):
    """1.9.0 accepted COLONYMIND_API_KEY beside the keyring: it stays accepted, as the instance key."""
    from test_migration_190 import _write_190_instance
    home, hermes_home = tmp_path / "colonymind", tmp_path / "hermes"
    _write_190_instance(home, hermes_home, hermes_python=HERMES_PYTHON)
    monkeypatch.setenv("COLONYMIND_HOME", str(home))
    env = (home / ".env").read_text().replace("COLONYMIND_API_KEY=\n", "COLONYMIND_API_KEY=legacy-bearer-0000000000\n")
    (home / ".env").write_text(env)
    report = "\n".join(init.migrate_legacy_instance(home))
    credentials = load_config(home, environ={}).get("credentials")
    assert init.read_api_key(home, environ={}) == LEGACY_SECRET
    assert credentials["api-key"]["secrets"] == [{"sha256": secret_digest("legacy-bearer-0000000000")}]
    assert CredentialTable.from_config(credentials).match("legacy-bearer-0000000000").instance
    assert "the 1.9.0 COLONYMIND_API_KEY is accepted as the instance key (api-key)" in report
    assert "legacy-bearer" not in report


def test_a_retiring_configured_key_is_not_made_permanent(tmp_path, monkeypatch):
    """The 1.9.0 client key the adapter used is retiring: api.key is a new secret, and the old one stays
    accepted as the instance key only until its expiry."""
    from datetime import timedelta
    from colonymind.util.temporal import now_utc
    soon = now_utc().replace(microsecond=0) + timedelta(days=2)
    home, owner, credentials, report = _migrate(tmp_path, monkeypatch, lambda owner: [
        {"principal": "hermes-local", "status": "active", "viewer_person_id": owner, "credentials": [
            {"id": "initial", "secret": LEGACY_SECRET, "status": "retiring", "accept_until": soon.isoformat()}]}])
    key = init.read_api_key(home, environ={})
    assert key and key != LEGACY_SECRET
    assert credentials["api-key"]["secrets"] == [{"sha256": secret_digest(LEGACY_SECRET),
                                                  "accept_until": soon.isoformat()}]
    assert "api.key generated" in report


def test_a_credential_accepts_each_of_its_secrets_until_its_own_expiry():
    from datetime import datetime, timedelta, timezone
    now = datetime(2030, 1, 1, tzinfo=timezone.utc)
    table = CredentialTable.from_config({
        "api-key": {"secrets": [{"sha256": secret_digest(DESK_SECRET), "accept_until": "2030-01-02T00:00:00Z"}]},
        "bot": {"secrets": [{"sha256": secret_digest(GAME_SECRET), "accept_until": "2030-01-02T00:00:00+00:00"},
                            {"sha256": secret_digest(VOICE_SECRET)}],
                "accept_until": "2030-02-01T00:00:00Z"}})
    assert table.match(GAME_SECRET, now=now).name == "bot" and table.match(VOICE_SECRET, now=now).name == "bot"
    assert table.match(DESK_SECRET, now=now).instance
    later = now + timedelta(days=2)
    assert table.match(GAME_SECRET, now=later) is None and table.match(DESK_SECRET, now=later) is None
    assert table.match(VOICE_SECRET, now=later).name == "bot"
    assert table.match(VOICE_SECRET, now=now + timedelta(days=31)) is None
    assert describe(table.get("bot"), now) == (
        "may write only as credential:bot; accepted until 2030-02-01T00:00:00+00:00; 2 secrets "
        "(until 2030-01-02T00:00:00+00:00, no expiry)")


@pytest.mark.parametrize("credentials,message", [
    ({"bot": {"sha256": DIGEST, "secrets": [{"sha256": "b" * 64}]}}, "not both"),
    ({"bot": {"secrets": []}}, "credentials.bot.secrets"),
    ({"bot": {"secrets": [{"sha256": DIGEST, "until": "2030-01-01T00:00:00Z"}]}}, "unknown key until"),
    ({"bot": {"sha256": DIGEST, "accept_until": "soon"}}, "credentials.bot.accept_until"),
    ({"bot": {"secrets": [{"sha256": DIGEST}, {"sha256": DIGEST}]}}, "same secret as itself"),
    ({"api-key": {"accept_until": "2030-01-01T00:00:00Z"}}, "api.key never expires"),
])
def test_invalid_lifetimes_are_refused_with_the_key_named(credentials, message):
    with pytest.raises(ConfigError, match=message):
        _validated(credentials)


async def test_an_expired_credential_is_refused_at_authentication(tmp_path, monkeypatch):
    table = CredentialTable.from_config({
        "bot": {"sha256": secret_digest(GAME_SECRET), "accept_until": "2020-01-01T00:00:00+00:00"},
        "rotating": {"secrets": [{"sha256": secret_digest(DESK_SECRET), "accept_until": "2020-01-01T00:00:00Z"},
                                 {"sha256": secret_digest(VOICE_SECRET)}]}})
    body = {"identity": {"host_id": "bot"}, "context": {"session_id": "s", "contact_id": ""},
            "user_message": {"role": "user", "content": "hello"}}
    assert (await _post(table, GAME_SECRET, body, monkeypatch, tmp_path)).status_code == 401
    assert (await _post(table, DESK_SECRET, body, monkeypatch, tmp_path)).status_code == 401
    assert (await _post(table, VOICE_SECRET, body, monkeypatch, tmp_path)).status_code == 200
    assert "expired at 2020-01-01T00:00:00+00:00" in describe(table.get("bot"))


def _cli(home, *argv):
    from colonymind.credentials_cli import run
    parser_args = SimpleNamespace(home=str(home), **dict(argv))
    return run(parser_args)


def test_credentials_add_list_set_and_remove(homes, capsys):
    home, hermes_home = homes
    assert init.run_init(_args(home, hermes_home)) == 0
    capsys.readouterr()
    key_file = home / "game-bot.key"
    assert _cli(home, ("command", "add"), ("name", "game-bot"), ("owner_platform", None), ("contacts", None),
                ("key_file", str(key_file))) == 0
    secret = key_file.read_text().strip()
    assert stat.S_IMODE(os.stat(key_file).st_mode) == 0o600 and len(secret) >= 32
    raw = _raw(home)["credentials"]["game-bot"]
    assert raw == {"sha256": secret_digest(secret), "owner": [], "contacts": "own"}
    out = capsys.readouterr().out
    assert secret not in out and "game-bot may write only as credential:game-bot" in out
    assert "restart the sidecar" in out

    assert _cli(home, ("command", "add"), ("name", "voice"), ("owner_platform", ["voice"]), ("contacts", "all"),
                ("key_file", None)) == 0
    printed = capsys.readouterr().out
    voice_secret = next(line.split()[-1] for line in printed.splitlines() if line.strip().startswith("secret:"))
    assert _raw(home)["credentials"]["voice"]["sha256"] == secret_digest(voice_secret)

    assert _cli(home, ("command", "set"), ("name", "api-key"), ("owner_platform", ["whatsapp", "cli"]),
                ("contacts", None)) == 0
    assert _raw(home)["credentials"]["api-key"] == {"owner": ["whatsapp", "cli"], "contacts": "all"}
    assert _cli(home, ("command", "list")) == 0
    listing = capsys.readouterr().out
    assert "api-key may write as the owner from whatsapp and cli" in listing
    assert "voice may write as the owner from voice" in listing
    assert voice_secret not in listing and secret_digest(voice_secret) not in listing

    assert _cli(home, ("command", "remove"), ("name", "game-bot")) == 0
    assert "game-bot" not in _raw(home)["credentials"]
    assert _cli(home, ("command", "remove"), ("name", "api-key")) == 1       # the instance key stays
    assert _cli(home, ("command", "add"), ("name", "voice"), ("owner_platform", None), ("contacts", None),
                ("key_file", None)) == 1                                         # names are unique
