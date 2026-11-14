"""The ``observe`` credential and the owner's runtime cap per outside source (docs/CREDENTIALS.md, docs/API.md).

A hub's ``observe`` credential posts the owner's outside observations for the sources it lists, reads and sets their
daily caps and records the conversations observed from them; it stands alone in its access, is granted the owner on
platforms named one by one and reaches no other route. ``manage`` with the owner reaches the same routes for every
source. A source's cap is the owner's runtime cap, else ``source_caps``, else ``daily_source_cap``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from colonymind import config as configuration
from colonymind.api.credentials import (OBSERVE, CredentialTable, answers_asks, authority_of, authority_text,
                                       describe, observed_source, observes, route_granted, route_operation,
                                       secret_digest)
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import mind as mind_routes
from colonymind.config import ConfigError, DEFAULTS, save_config
from test_mind_loop import OWNER, pinned_clock  # noqa: F401
from test_mind_observations_sources import fx, observation  # noqa: F401

KEY = "instance-key"
HUB, WIDE, GUEST = "hub-secret", "desk-secret", "guest-secret"
HUB_ENTRY = {"sha256": secret_digest(HUB), "owner": ["companion"], "access": ["observe"],
             "observe_sources": ["pendant"]}


def _validate(credentials):
    return configuration.validate(configuration._merge(configuration.DEFAULTS, {"credentials": credentials}))


# -- the grant ------------------------------------------------------------------------------------------------------

def test_an_observe_credential_is_the_hubs_own_and_does_nothing_else():
    table = CredentialTable.from_config({"pendant-hub": HUB_ENTRY})
    hub = table.get("pendant-hub")
    assert hub.access == frozenset({OBSERVE}) and hub.owner == ("companion",) and hub.contacts == "own"
    assert hub.observe_sources == ("pendant",)
    assert observes(hub) and not answers_asks(hub) and not observes(table.instance)
    text = describe(hub)
    assert text.startswith("may only post the owner's outside observations from pendant") and "nothing else" in text
    assert "terminal" not in text
    assert "posts outside observations for the sources it lists" in authority_text(authority_of(hub, OWNER), OWNER)


@pytest.mark.parametrize("entry,match", [
    ({"api-key": {"owner": ["companion"], "contacts": "all", "access": ["observe"], "observe_sources": ["pendant"]}},
     "instance key never has observe"),
    ({"hub": {**HUB_ENTRY, "access": ["observe", "manage"]}}, "stands alone"),
    ({"hub": {**HUB_ENTRY, "access": ["observe", "read"]}}, "stands alone"),
    ({"hub": {**HUB_ENTRY, "access": ["observe", "answer"]}}, "stands alone"),
    ({"hub": {**HUB_ENTRY, "owner": "any"}}, "never any"),
    ({"hub": {**HUB_ENTRY, "owner": []}}, "granted the owner"),
    ({"hub": {key: value for key, value in HUB_ENTRY.items() if key != "owner"}}, "granted the owner"),
    ({"hub": {**HUB_ENTRY, "contacts": "all"}}, "writes as no one"),
    ({"hub": {key: value for key, value in HUB_ENTRY.items() if key != "observe_sources"}}, "must list the sources"),
    ({"hub": {**HUB_ENTRY, "observe_sources": []}}, "must list the sources"),
    ({"hub": {**HUB_ENTRY, "observe_sources": "pendant"}}, "must list the sources"),
    ({"hub": {**HUB_ENTRY, "observe_sources": ["bad source"]}}, "not a source identifier"),
    ({"hub": {**HUB_ENTRY, "observe_sources": ["-lead"]}}, "not a source identifier"),
    ({"hub": {**HUB_ENTRY, "observe_sources": [7]}}, "not a source identifier"),
    ({"hub": {**HUB_ENTRY, "observe_sources": ["x" * 129]}}, "not a source identifier"),
    ({"desk": {"sha256": "b" * 64, "owner": ["companion"], "contacts": "all", "observe_sources": ["pendant"]}},
     "takes access: \\[observe\\]"),
    ({"hub": {**HUB_ENTRY, "scopes": ["*"]}}, "not both"),
])
def test_validation_refuses_every_other_shape(entry, match):
    with pytest.raises(ConfigError, match=match):
        _validate(entry)


def test_sources_are_normalized_and_no_default_grant_observes():
    checked = _validate({"hub": {**HUB_ENTRY, "observe_sources": [" pendant", "pendant", "cam:1"]}})
    assert checked["credentials"]["hub"]["observe_sources"] == ["pendant", "cam:1"]
    assert checked["credentials"]["hub"]["contacts"] == "own"
    for grant in CredentialTable.from_config(None).grants():
        assert OBSERVE not in grant.access and not observes(grant)


OBSERVE_PATHS = (("POST", "/v1/mind/observations/sources/pendant"),
                 ("GET", "/v1/mind/observations/sources/pendant/cap"),
                 ("PUT", "/v1/mind/observations/sources/pendant/cap"),
                 ("GET", "/v2/host/conversations/observed/conv-1"),
                 ("PUT", "/v2/host/conversations/observed/conv-1"))


def test_the_observe_routes_are_their_own_operation_and_nothing_else_is():
    for method, path in OBSERVE_PATHS:
        assert route_operation(method, path) == OBSERVE and not route_granted(method, path), (method, path)
    assert observed_source("POST", "/v1/mind/observations/sources/pendant") == "pendant"
    assert observed_source("PUT", "/v1/mind/observations/sources/pendant/cap") == "pendant"
    assert observed_source("PUT", "/v2/host/conversations/observed/conv-1") is None
    for method, path in (("GET", "/v1/mind/observations"), ("POST", "/v1/mind/observations"),
                         ("GET", "/v1/mind/observations/sources/pendant"),
                         ("DELETE", "/v1/mind/observations/sources/pendant/cap"),
                         ("POST", "/v1/mind/observations/sources/pendant/cap"),
                         ("POST", "/v1/mind/observations/sources/pendant/extra"),
                         ("DELETE", "/v2/host/conversations/observed/conv-1"),
                         ("GET", "/v2/host/conversations/observed"),
                         ("GET", "/v2/host/conversations/observed/conv-1/turns")):
        assert route_operation(method, path) == "manage", (method, path)
    assert route_operation("POST", "/v1/host/turns/sync") == "write"
    assert route_operation("POST", "/v1/host/memory/search") == "read"


@pytest.mark.parametrize("identifier", ["conv-1", "a.b_c", "device:7:123", "Z9"])
def test_valid_conversation_ids_stay_in_the_observe_operation(identifier):
    from colonymind.turns.observed import check_conversation_id, ObservedRefusal
    assert check_conversation_id(identifier) == identifier
    for method in ("GET", "PUT"):
        assert route_operation(method, "/v2/host/conversations/observed/" + identifier) == OBSERVE
    with pytest.raises(ObservedRefusal) as refused:
        check_conversation_id(identifier + "/part")
    assert refused.value.status == 422 and refused.value.code == "invalid_conversation_id"


# -- the CLI --------------------------------------------------------------------------------------------------------

@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    save_config({**DEFAULTS, "owner": {"contact_id": "p-01"}}, tmp_path)
    return tmp_path


def _cli(home, **argv):
    from colonymind.credentials_cli import run
    return run(SimpleNamespace(home=str(home), **argv))


def test_the_cli_issues_and_changes_an_observe_credential(home, capsys):
    assert _cli(home, command="add", name="pendant-hub", owner_platform=["companion"], contacts=None,
                access=["observe"], observe_source=["pendant"], key_file=None) == 0
    out = capsys.readouterr().out
    assert "may only post the owner's outside observations from pendant" in out and "secret:" in out
    entry = yaml.safe_load((home / "colonymind.yaml").read_text())["credentials"]["pendant-hub"]
    assert (entry["owner"], entry["contacts"], entry["access"], entry["observe_sources"]) == (
        ["companion"], "own", ["observe"], ["pendant"])
    assert len(entry["sha256"]) == 64 and HUB not in (home / "colonymind.yaml").read_text()
    assert _cli(home, command="set", name="pendant-hub", owner_platform=None, contacts=None, access=None,
                observe_source=["pendant", "cam"]) == 0
    entry = yaml.safe_load((home / "colonymind.yaml").read_text())["credentials"]["pendant-hub"]
    assert entry["observe_sources"] == ["pendant", "cam"]
    before = (home / "colonymind.yaml").read_text()
    assert _cli(home, command="add", name="other", owner_platform=["companion"], contacts=None,
                access=["observe"], observe_source=None, key_file=None) != 0
    assert "must list the sources" in "".join(capsys.readouterr())
    assert _cli(home, command="add", name="other", owner_platform=None, contacts=None,
                access=["observe"], observe_source=["pendant"], key_file=None) != 0
    assert "granted the owner" in "".join(capsys.readouterr())
    assert _cli(home, command="set", name="api-key", owner_platform=None, contacts=None,
                access=["observe"], observe_source=["pendant"]) != 0
    assert "instance key never has observe" in "".join(capsys.readouterr())
    assert (home / "colonymind.yaml").read_text() == before
    # Dropping observe from its access drops its sources with it.
    assert _cli(home, command="set", name="pendant-hub", owner_platform=None, no_owner=True, contacts=None,
                access=["read"], observe_source=None) == 0
    entry = yaml.safe_load((home / "colonymind.yaml").read_text())["credentials"]["pendant-hub"]
    assert "observe_sources" not in entry and entry["access"] == ["read"]


# -- the middleware -------------------------------------------------------------------------------------------------

def _table():
    return CredentialTable.from_config({
        "api-key": {"owner": ["rcs"], "contacts": "all"},
        "pendant-hub": HUB_ENTRY,
        "desk": {"sha256": secret_digest(WIDE), "owner": ["companion"], "contacts": "all"},
        "guest": {"sha256": secret_digest(GUEST), "owner": [], "contacts": "all",
                  "access": ["read", "write", "manage"]}})


def _app(api_key=KEY, mind=None):
    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware, api_key=api_key, credentials=_table())
    app.include_router(mind_routes.router)
    mind_routes.set_mind(mind)

    @app.api_route("/v2/host/conversations/observed/{conversation_id}", methods=["GET", "PUT"])
    async def observed(conversation_id: str):
        return {"ok": True}

    @app.post("/v1/host/turns/sync")
    async def turns():
        return {"ok": True}
    return app


def _client(app, secret=None):
    headers = {"Authorization": f"Bearer {secret}"} if secret else {}
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost", headers=headers)


async def test_the_observe_credential_reaches_only_its_routes_and_sources():
    """Before any mind exists the observe routes pass the middleware (here to 503: no mind) for the granted source;
    another source is ``source_not_granted``; every other route is ``route_not_granted``."""
    try:
        async with _client(_app(), HUB) as client:
            for method, path in OBSERVE_PATHS:
                passed = await client.request(method, path, json={"cap": 1})
                assert passed.status_code not in {401, 403}, (method, path, passed.text)
            for method, path in (("POST", "/v1/mind/observations/sources/other"),
                                 ("GET", "/v1/mind/observations/sources/other/cap"),
                                 ("PUT", "/v1/mind/observations/sources/other/cap")):
                refused = await client.request(method, path, json={"cap": 1})
                assert refused.status_code == 403, (method, path)
                assert refused.json()["detail"]["code"] == "source_not_granted"
                assert "pendant" in refused.json()["detail"]["message"]
            for method, path in (("POST", "/v1/host/turns/sync"), ("PUT", "/v2/host/turns/t-1"),
                                 ("POST", "/v1/host/memory/read"), ("POST", "/v1/host/memory/search"),
                                 ("POST", "/v1/host/context/assemble"), ("GET", "/v1/host/memory/inventory"),
                                 ("GET", "/v1/mind/outbox"), ("POST", "/v1/mind/outbox/i-1/sent"),
                                 ("GET", "/v1/mind/observations"), ("POST", "/v1/mind/observations"),
                                 ("GET", "/v1/mind/state"), ("GET", "/v1/mind/asks"),
                                 ("GET", "/v1/mind/permit-asks"), ("POST", "/v1/host/configure"),
                                 ("GET", "/v1/host/credentials"), ("GET", "/v1/secrets"),
                                 ("DELETE", "/v2/host/conversations/observed/conv-1")):
                refused = await client.request(method, path, json={})
                assert refused.status_code == 403, (method, path, refused.text)
                code = refused.json()["detail"]["code"]
                assert code in {"route_not_granted", "answer_credential_required"}, (method, path, code)
            mcp = await client.post("/v1/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
            assert mcp.status_code == 403
    finally:
        mind_routes.set_mind(None)


async def test_manage_with_the_owner_reaches_every_source_and_without_it_none():
    try:
        app = _app()
        for secret in (KEY, WIDE):
            async with _client(app, secret) as client:
                for path in ("/v1/mind/observations/sources/other", "/v1/mind/observations/sources/pendant"):
                    passed = await client.post(path, json={})
                    assert passed.status_code not in {401, 403}, (secret, path, passed.text)
                passed = await client.put("/v2/host/conversations/observed/conv-1", json={})
                assert passed.status_code == 200
        async with _client(app, GUEST) as client:
            for method, path in OBSERVE_PATHS:
                refused = await client.request(method, path, json={"cap": 1})
                assert refused.status_code == 403 and refused.json()["detail"]["code"] == "not_owner", (method, path)
        async with _client(_app(None)) as client:
            for method, path in OBSERVE_PATHS:
                refused = await client.request(method, path, json={"cap": 1})
                assert refused.status_code == 403, (method, path)
    finally:
        mind_routes.set_mind(None)


def test_observe_source_granted_is_the_body_sourced_routes_check():
    from colonymind.api.auth import (anonymous_authority, credential_authority, key_authority,
                                    observe_source_granted)
    table = _table()

    def request(authority):
        return SimpleNamespace(state=SimpleNamespace(colonymind_authority=authority))
    hub = request(credential_authority(table.get("pendant-hub")))
    assert observe_source_granted(hub, "pendant") and not observe_source_granted(hub, "other")
    assert not observe_source_granted(hub, "")
    for wide in (request(credential_authority(table.get("desk"))), request(key_authority(table.instance)), None):
        assert observe_source_granted(wide, "other")
    for refused in (request(credential_authority(table.get("guest"))), request(anonymous_authority())):
        assert not observe_source_granted(refused, "pendant")


# -- the cap --------------------------------------------------------------------------------------------------------

@pytest.fixture
def capped(tmp_path, monkeypatch):
    from test_mind_loop import Fixture
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fixture = Fixture(tmp_path, config={"faculties": {"outreach": False},
                                        "observations": {"daily_source_cap": 6, "source_caps": {"pendant": 3}}})
    yield fixture
    fixture.store.close()
    mind_routes.set_mind(None)


def _outside(fixture):
    from colonymind.mind.observations import for_mind
    return for_mind(fixture.mind)


async def _post(fixture, source, count, prefix="k"):
    return await fixture.mind.observe_source(source, {"source": source, "observations": [
        observation(fixture, key=f"{prefix}{index}") for index in range(count)]})


async def test_the_effective_cap_is_runtime_then_config_then_default_and_survives_a_restart(capped):
    assert _outside(capped).cap("pendant") == {"source": "pendant", "cap": 3, "basis": "config"}
    assert _outside(capped).cap("sensor") == {"source": "sensor", "cap": 6, "basis": "default"}
    assert (await _post(capped, "pendant", 5))["accepted"] == 3
    assert (await _post(capped, "sensor", 8))["accepted"] == 6
    value = _outside(capped).set_cap("pendant", 4, by="pendant-hub")
    assert (value["cap"], value["basis"], value["set_by"]) == (4, "runtime", "pendant-hub") and value["set_at"]
    capped.restart()
    assert _outside(capped).cap("pendant")["cap"] == 4
    assert (await _post(capped, "pendant", 5, prefix="r"))["accepted"] == 1
    zero = _outside(capped).set_cap("sensor", 0, by="owner")
    assert zero["basis"] == "runtime" and zero["cap"] == 0
    assert _outside(capped).set_cap("pendant", None, by="owner") == {"source": "pendant", "cap": 3, "basis": "config"}
    capped.restart()
    assert _outside(capped).cap("pendant")["basis"] == "config" and _outside(capped).cap("sensor")["cap"] == 0
    notes = [row for row in capped.store.intentions(types=["observation_cap"], limit=100)]
    assert len(notes) == 3 and all(row.kind == "note" and row.status == "done" for row in notes)
    assert any("pendant: 3 (config) -> 4 (runtime) by pendant-hub" in row.description for row in notes)
    # Repeating a setting, or clearing one that is not set, writes nothing and notes nothing.
    assert _outside(capped).set_cap("sensor", 0, by="pendant-hub")["set_by"] == "owner"
    assert _outside(capped).set_cap("pendant", None, by="owner")["basis"] == "config"
    assert len([row for row in capped.store.intentions(types=["observation_cap"], limit=100)]) == 3
    # A zero cap takes nothing in.
    assert (await _post(capped, "sensor", 2, prefix="z"))["accepted"] == 0
    for bad in (-1, 1001, True, 2.5, "7"):
        with pytest.raises(ValueError):
            _outside(capped).set_cap("pendant", bad, by="owner")
    with pytest.raises(ValueError):
        _outside(capped).cap("bad source")


async def test_the_cap_routes_under_the_observe_credential_and_the_instance_key(capped):
    app = _app(mind=capped.mind)
    async with _client(app, HUB) as client:
        read = await client.get("/v1/mind/observations/sources/pendant/cap")
        assert read.status_code == 200 and read.json() == {"source": "pendant", "cap": 3, "basis": "config"}
        put = await client.put("/v1/mind/observations/sources/pendant/cap", json={"cap": 40})
        assert put.status_code == 200 and put.json()["basis"] == "runtime" and put.json()["set_by"] == "pendant-hub"
        for body in ({"cap": 1001}, {"cap": -1}, {"cap": "5"}, {"cap": True}, {}, {"cap": 3, "extra": 1}):
            assert (await client.put("/v1/mind/observations/sources/pendant/cap", json=body)).status_code == 422, body
        posted = await client.post("/v1/mind/observations/sources/pendant",
                                   json={"source": "pendant", "observations": [observation(capped)]})
        assert posted.status_code == 200 and posted.json()["accepted"] == 1
        refused = await client.put("/v1/mind/observations/sources/sensor/cap", json={"cap": 1})
        assert refused.status_code == 403 and refused.json()["detail"]["code"] == "source_not_granted"
        assert (await client.get("/v1/mind/observations")).status_code == 403
    async with _client(app, KEY) as client:
        assert (await client.get("/v1/mind/observations/sources/pendant/cap")).json()["cap"] == 40
        cleared = await client.put("/v1/mind/observations/sources/pendant/cap", json={"cap": None})
        assert cleared.status_code == 200 and cleared.json() == {"source": "pendant", "cap": 3, "basis": "config"}
        other = await client.put("/v1/mind/observations/sources/sensor/cap", json={"cap": 2})
        assert other.status_code == 200 and other.json()["set_by"] == "api-key"


async def test_the_handlers_refuse_an_ungranted_source_without_the_middleware(capped):
    """Defense in depth: the handlers ask ``observe_source_granted`` themselves."""
    from colonymind.api.auth import credential_authority
    from starlette.requests import Request
    mind_routes.set_mind(capped.mind)
    request = Request({"type": "http", "method": "PUT", "path": "/", "headers": []})
    request.state.colonymind_authority = credential_authority(_table().get("pendant-hub"))
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as refused:
        await mind_routes.set_source_cap("sensor", mind_routes.SourceCapBody(cap=1), request)
    assert refused.value.status_code == 403 and refused.value.detail["code"] == "source_not_granted"
    assert (await mind_routes.source_cap("pendant", request))["cap"] == 3


def test_source_caps_validate_like_the_daily_cap():
    def check(caps):
        return configuration.validate(configuration._merge(configuration.DEFAULTS,
                                                           {"mind": {"observations": {"source_caps": caps}}}))
    assert DEFAULTS["mind"]["observations"]["source_caps"] == {"omi": 40}
    assert check({"pendant": 0})["mind"]["observations"]["source_caps"] == {"omi": 40, "pendant": 0}
    for caps, match in (({"pendant": -1}, "non-negative whole number"), ({"pendant": True}, "non-negative"),
                        ({"pendant": 2.5}, "non-negative"), ({"bad source": 1}, "source identifiers"),
                        (["pendant"], "must be a mapping")):
        with pytest.raises(ConfigError, match=match):
            check(caps)
