"""Upgrading an instance that already runs 1.10 carries nothing from a 1.9.0 keyring left on disk.

1.10 authenticated only the instance key (``api.key``); every client used it. A 1.9.0 keyring still on disk
from before the 1.10 upgrade was not honoured by 1.10, so upgrading from 1.10 records only the instance key,
granted the owner on the default platforms and the whole API as 1.10 gave it, names the keyring as retired
and not carried, and leaves the file for the owner to remove. Every keyring secret stays refused.

The signal that an instance runs 1.10 or later is ``colonymind.yaml``: 1.10 introduced it as the instance's
one configuration file, every 1.10 instance has one (its ``init`` writes it, and so does its own 1.9.0
upgrade), and 1.9.0 never wrote or read it (docs/CREDENTIALS.md).
"""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from types import SimpleNamespace

import yaml

from colonymind import init
from colonymind.api.credentials import DEFAULT_OWNER_PLATFORMS, CredentialTable
from colonymind.config import load_config, read_api_key
from colonymind.util.temporal import now_utc
from test_credential_grants import _raw, homes  # noqa: F401  (homes: fixture)
from test_init import _args
from test_upgrade import _upgrade_args

LATER = (now_utc() + timedelta(days=90)).replace(microsecond=0).isoformat()
KEYRING_SECRETS = [f"stale-keyring-secret-{index:02d}-000000000000" for index in range(12)]


def stale_keyring(owner: str) -> dict:
    """About a dozen 1.9.0 principals, as a keyring left on disk: unexpired secrets, owner audiences, a
    person-bound integration, every scope."""
    def principal(index, **fields):
        return {"principal": f"p-{index:02d}", "status": "active", "accept_until": LATER, **fields,
                "credentials": [{"id": "c", "secret": KEYRING_SECRETS[index], "status": "active",
                                 "accept_until": LATER}]}
    return {"version": 1, "principals": [
        principal(0, viewer_person_id=owner, audiences=["viewer", "owner"], scopes=["*"]),
        principal(1, viewer_person_id=owner, audiences=["owner"], scopes=["turns:write", "context:read"]),
        principal(2, viewer_person_id="cid-integration", person_ids=["cid-integration"],
                  scopes=["turns:write", "context:read"]),                       # wrote under its own person
        principal(3, audiences=["shared", "global"], scopes=["context:read"]),
        principal(4, viewer_person_id=owner, scopes=["turns:write", "turns:resolve-sender"],
                  turn_ingress_platforms=["voice"]),
        *[principal(index, viewer_person_id=owner, audiences=["owner"], scopes=["api:access", "turns:write"])
          for index in range(5, 12)],
    ]}


def upgraded_110(home, hermes_home, capsys):
    """A 1.10 instance (no credentials in colonymind.yaml, the 1.9.0 manifest and keyring still on disk),
    upgraded; returns its owner and the upgrade's report."""
    assert init.run_init(_args(home, hermes_home)) == 0
    raw = _raw(home)
    del raw["credentials"]                                   # 1.10 had one key and no grants
    (home / "colonymind.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    owner = str(raw["owner"]["contact_id"])
    (home / "api-keyring.json").write_text(json.dumps(stale_keyring(owner)))
    (home / "instance.json").write_text(json.dumps({"version": 1, "owner_id": owner}))
    capsys.readouterr()
    assert init.run_upgrade(_upgrade_args(home)) == 0
    return owner, capsys.readouterr().out


def call(home, secret: str, method: str, path: str, body: dict | None = None):
    """One request with ``secret`` to the sidecar as the upgraded instance configures it."""
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from colonymind.api.errors import install_exception_handlers
    from colonymind.api.middleware import ApiKeyMiddleware
    from colonymind.api.routers import executions, host
    from colonymind.api.routers import mind as mind_router

    async def run():
        app = FastAPI()
        install_exception_handlers(app)
        for router in (host.router, host.v2_router, executions.router, mind_router.router):
            app.include_router(router)
        app.add_middleware(ApiKeyMiddleware, api_key=read_api_key(home, environ={}),
                           credentials=CredentialTable.load(home))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.request(method, path, json=body, headers={"Authorization": "Bearer " + secret})
    return asyncio.run(run())


def owner_turn(owner: str, platform: str) -> dict:
    return {"identity": {"host_id": "hermes"},
            "context": {"session_id": f"s-{platform}", "contact_id": owner, "turn_id": f"t-{platform}",
                        "channel_id": f"{platform}:+15550000001"},
            "user_message": {"role": "user", "content": "The boiler service moved to Friday."},
            "assistant_message": {"role": "assistant", "content": "Noted."}}


def test_a_110_upgrade_carries_nothing_from_a_stale_190_keyring(homes, capsys, monkeypatch):
    home, hermes_home = homes
    owner, report = upgraded_110(home, hermes_home, capsys)
    # Only the instance key, with the owner on the default platforms and the whole API, as 1.10 gave it.
    assert load_config(home).get("credentials") == {"api-key": {"owner": list(DEFAULT_OWNER_PLATFORMS),
                                                                "contacts": "all"}}
    assert "api-keyring.json is a 1.9.0 keyring this instance no longer honoured" in report
    assert "retired: none of its principals or secrets is carried" in report
    assert (home / "api-keyring.json").is_file()                            # the owner removes it
    assert not any(secret in report for secret in KEYRING_SECRETS)

    from colonymind.api.routers import host
    from colonymind.api.routers import mind as mind_router
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", owner)
    for name in ("_contacts_store", "_telemetry", "_reranker", "_context_recall_selector", "_comms_log",
                 "_preference_learner"):
        monkeypatch.setattr(host, name, None, raising=False)
    mind_router.set_mind(SimpleNamespace(owner_id=owner, state=lambda: {"enabled": True, "autonomy": "standard"}))
    try:
        key = read_api_key(home, environ={})
        for platform in DEFAULT_OWNER_PLATFORMS:
            response = call(home, key, "POST", "/v1/host/turns/sync", owner_turn(owner, platform))
            assert response.status_code == 200 and response.json()["source_recorded"] is True, response.text
        assert call(home, key, "GET", f"/v1/host/context/temporal?contact_id={owner}").status_code == 200
        assert call(home, key, "GET", "/v1/mind/state").status_code == 200
        for secret in KEYRING_SECRETS:
            assert call(home, secret, "POST", "/v1/host/turns/sync",
                        owner_turn(owner, "whatsapp")).status_code == 401, secret
    finally:
        mind_router.set_mind(None)


def test_the_keyring_of_an_instance_that_runs_110_is_never_read(homes, capsys):
    """Whatever calls it, the 1.9.0 keyring reader reads nothing once the instance runs 1.10 or later."""
    home, hermes_home = homes
    owner, _report = upgraded_110(home, hermes_home, capsys)
    assert init.legacy_principals(home, {}, owner) == []
    assert init.migrate_legacy_credentials(home, {}, owner).credentials == {
        "api-key": {"owner": list(DEFAULT_OWNER_PLATFORMS), "contacts": "all"}}


def test_the_kept_self_routes_answer_the_instance_key_after_a_110_upgrade(homes, capsys, monkeypatch):
    """The instance key still reads the journal after upgrade; stale keys are refused."""
    home, hermes_home = homes
    owner, _report = upgraded_110(home, hermes_home, capsys)
    from colonymind.api.routers import host
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", owner)
    journal = SimpleNamespace(recent=lambda limit, domain=None: [{"action": "renewed the certificate"}],
                              today=lambda domain=None: [])
    monkeypatch.setattr(host, "_self_model", SimpleNamespace(
        status=lambda: {"domains": {"coding": 0.8}}, brief=lambda: "steady", journal=journal))
    key = read_api_key(home, environ={})
    entries = call(home, key, "GET", "/v1/host/self/journal?limit=5")
    assert entries.status_code == 200 and entries.json()["entries"] == [{"action": "renewed the certificate"}]
    for path in ("/v1/host/self/journal",):
        assert call(home, KEYRING_SECRETS[0], "GET", path).status_code == 401, path
