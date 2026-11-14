"""The ``answer`` credential (docs/CREDENTIALS.md; docs/hands-permit-answer-contract.md).

Only an owner channel relay's own credential reads and answers the hands permit asks by id: ``answer`` stands alone in
its access, its owner platforms are explicit-answer channels named one by one, and the instance key never has it, so
nothing that holds ``api.key`` approves a held call that way. It reaches no other route.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from colonymind import config as configuration
from colonymind.api.credentials import (ANSWER, CredentialTable, answers_asks, describe, route_granted,
                                       route_operation, secret_digest)
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import mind as mind_routes
from colonymind.config import ConfigError, DEFAULTS, save_config

KEY = "instance-key"
HUB, ROTATED, LEGACY = "hub-secret", "rotated-instance-secret", "legacy-secret"


def _validate(credentials):
    return configuration.validate(configuration._merge(configuration.DEFAULTS, {"credentials": credentials}))


def test_an_answer_credential_is_the_relays_own_and_does_nothing_else():
    table = CredentialTable.from_config({"companion-hub": {"sha256": "a" * 64, "owner": ["companion"],
                                                           "access": ["answer"]}})
    hub = table.get("companion-hub")
    assert hub.access == frozenset({ANSWER}) and hub.owner == ("companion",) and hub.contacts == "own"
    assert answers_asks(hub) and not answers_asks(table.instance)
    assert describe(hub).startswith("may only read and answer the owner's hands permit asks, as the owner from "
                                    "companion")


@pytest.mark.parametrize("entry,match", [
    ({"api-key": {"owner": ["companion"], "contacts": "all", "access": ["answer"]}}, "instance key never answers"),
    ({"api-key": {"owner": ["companion"], "contacts": "all", "access": ["read", "write", "manage", "answer"]}},
     "instance key never answers"),
    ({"hub": {"sha256": "b" * 64, "owner": "any", "access": ["answer"]}}, "never any"),
    ({"hub": {"sha256": "b" * 64, "owner": ["companion"], "access": ["answer", "manage"]}}, "stands alone"),
    ({"hub": {"sha256": "b" * 64, "owner": ["companion"], "access": ["answer", "note"]}}, "stands alone"),
    ({"hub": {"sha256": "b" * 64, "owner": ["rcs"], "access": ["answer"]}}, "of companion"),
    ({"hub": {"sha256": "b" * 64, "owner": ["companion", "rcs"], "access": ["answer"]}}, "of companion"),
    ({"hub": {"sha256": "b" * 64, "owner": [], "access": ["answer"]}}, "of companion"),
    ({"hub": {"sha256": "b" * 64, "access": ["answer"]}}, "of companion"),
    ({"hub": {"sha256": "b" * 64, "owner": ["companion"], "contacts": "all", "access": ["answer"]}}, "writes as no one"),
    ({"hub": {"sha256": "b" * 64, "owner": ["companion"], "scopes": ["*"], "access": ["answer"]}}, "not both"),
])
def test_validation_refuses_every_other_shape(entry, match):
    with pytest.raises(ConfigError, match=match):
        _validate(entry)


def test_no_default_or_carried_grant_answers():
    assert ANSWER not in CredentialTable.from_config(None).instance.access
    table = CredentialTable.from_config({
        "api-key": {"owner": "any", "contacts": "all"},
        "desk": {"sha256": "c" * 64, "owner": ["companion"], "contacts": "all"},
        "old": {"sha256": "d" * 64, "owner": "any", "contacts": "all", "scopes": ["*"]}})
    for grant in table.grants():
        assert ANSWER not in grant.access and not answers_asks(grant), grant.name


def test_the_permit_ask_routes_are_the_answer_operation_and_nothing_else_is():
    for method, path in (("GET", "/v1/mind/permit-asks"), ("GET", "/v1/mind/permit-asks/a-1"),
                         ("POST", "/v1/mind/permit-asks/a-1/answer"), ("DELETE", "/v1/mind/permit-asks/a-1")):
        assert route_operation(method, path) == ANSWER and not route_granted(method, path)
    for method, path in (("POST", "/v1/mind/answers"), ("GET", "/v1/mind/asks"), ("GET", "/v1/mind/permit-asksx"),
                         ("POST", "/v1/mind/asks/a-1/shown")):
        assert route_operation(method, path) == "manage"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    save_config({**DEFAULTS, "owner": {"contact_id": "p-01"}}, tmp_path)
    return tmp_path


def _cli(home, **argv):
    from colonymind.credentials_cli import run
    return run(SimpleNamespace(home=str(home), **argv))


def test_the_cli_issues_the_relay_its_answer_credential_and_never_the_instance_key(home, capsys):
    assert _cli(home, command="add", name="companion-hub", owner_platform=["companion"], contacts=None,
                access=["answer"], key_file=None) == 0
    out = capsys.readouterr().out
    assert "may only read and answer the owner's hands permit asks" in out and "secret:" in out
    entry = yaml.safe_load((home / "colonymind.yaml").read_text())["credentials"]["companion-hub"]
    assert (entry["owner"], entry["contacts"], entry["access"]) == (["companion"], "own", ["answer"])
    before = (home / "colonymind.yaml").read_text()
    assert _cli(home, command="set", name="api-key", owner_platform=["rcs", "companion"], contacts=None,
                access=["read", "write", "manage", "answer"]) != 0
    assert "instance key never answers" in "".join(capsys.readouterr())
    assert _cli(home, command="add", name="desk", owner_platform=["companion"], contacts=None,
                access=["answer", "manage"], key_file=None) != 0
    assert "stands alone" in "".join(capsys.readouterr())
    assert (home / "colonymind.yaml").read_text() == before


# -- the middleware ---------------------------------------------------------------------------------------------

def _app(api_key):
    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware, api_key=api_key, credentials=CredentialTable.from_config({
        "api-key": {"owner": ["rcs", "companion"], "contacts": "all",
                    "secrets": [{"sha256": secret_digest(ROTATED)}]},
        "companion-hub": {"sha256": secret_digest(HUB), "owner": ["companion"], "access": ["answer"]},
        "old": {"sha256": secret_digest(LEGACY), "owner": "any", "contacts": "all", "scopes": ["*"]}}))
    app.include_router(mind_routes.router)

    @app.post("/v1/host/turns/sync")
    async def turns():
        return {"ok": True}
    return app


ANSWER_ROUTES = (("GET", "/v1/mind/permit-asks"), ("GET", "/v1/mind/permit-asks/a-1"),
                 ("POST", "/v1/mind/permit-asks/a-1/answer"))


async def test_only_the_answer_credential_reaches_the_answer_routes(monkeypatch):
    """The instance key (by api.key or a rotation secret), a 1.9.0 principal with every scope and development mode get
    403 ``answer_credential_required`` before any handler runs; the relay's credential gets through (here to a
    sidecar with no mind: 503)."""
    # Mounting the router does not reset its process-wide mind from earlier apps.
    monkeypatch.setattr(mind_routes, "_mind", None)
    async with AsyncClient(transport=ASGITransport(app=_app(KEY)), base_url="http://localhost") as client:
        for method, path in ANSWER_ROUTES:
            for secret in (KEY, ROTATED, LEGACY):
                refused = await client.request(method, path, headers={"Authorization": f"Bearer {secret}"})
                assert refused.status_code == 403, (method, path, secret)
                assert refused.json()["detail"]["code"] == "answer_credential_required"
            passed = await client.request(method, path, headers={"Authorization": f"Bearer {HUB}"})
            assert passed.status_code != 403, (method, path, passed.text)
    async with AsyncClient(transport=ASGITransport(app=_app(None)), base_url="http://localhost") as client:
        for method, path in ANSWER_ROUTES:
            refused = await client.request(method, path)
            assert refused.status_code == 403 and refused.json()["detail"]["code"] == "answer_credential_required"


async def test_the_answer_credential_reaches_no_other_route():
    async with AsyncClient(transport=ASGITransport(app=_app(KEY)), base_url="http://localhost",
                           headers={"Authorization": f"Bearer {HUB}"}) as client:
        for method, path in (("POST", "/v1/mind/answers"), ("POST", "/v1/mind/asks/a-1/shown"),
                             ("GET", "/v1/mind/asks"), ("POST", "/v1/mind/asks/K7F/no"), ("GET", "/v1/mind/state"),
                             ("POST", "/v1/host/turns/sync"), ("GET", "/v1/secrets")):
            refused = await client.request(method, path, json={})
            assert refused.status_code == 403, (method, path, refused.text)
            assert refused.json()["detail"]["code"] == "route_not_granted", (method, path)
        mcp = await client.post("/v1/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert mcp.status_code == 403 and mcp.json()["detail"]["code"] == "mcp_not_granted"


def test_the_instance_key_never_answers_even_with_a_grant_validation_would_refuse():
    """Defense in depth: a table built around ``validate_credentials`` (or an in-process call) still never makes the
    instance key, its rotation secrets or a credential granted every platform an answer credential."""
    from colonymind.api.auth import answer_credential, credential_authority, key_authority
    from colonymind.api.credentials import Grant
    forged = Grant(name="api-key", owner=("companion",), contacts="all", access=frozenset({ANSWER}))
    assert not answer_credential(key_authority(forged)) and not answer_credential(key_authority())
    everywhere = Grant(name="desk", owner=("*",), contacts="all", secrets=(("e" * 64, None),),
                       access=frozenset({ANSWER}))
    assert not answer_credential(credential_authority(everywhere))
    hub = Grant(name="companion-hub", owner=("companion",), contacts="own", secrets=(("f" * 64, None),),
                access=frozenset({ANSWER}))
    assert answer_credential(credential_authority(hub), "companion")
    assert not answer_credential(credential_authority(hub), "rcs")
