"""Private owner reveals; all credential/password fixtures are synthetic and ephemeral."""
from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import sys
from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from colonymind.api.credentials import CredentialTable, Grant, SECRET_REVEAL_PATH, secret_digest
from colonymind.api.middleware import ApiKeyMiddleware, BodySizeLimitMiddleware
from colonymind.api.routers import secrets as routes
from colonymind.hands import sessions
from colonymind.hands.store import HandsStore
from colonymind.secrets import saved_login
from colonymind.secrets.vault import Vault
from test_hands_permits import EPOCH, seed

HANDLE = "vault_0123456789ab"
MISSING = "vault_abcdef012345"
BODY = {"id": HANDLE, "session_id": "owner-session"}


@pytest.fixture
async def rig(tmp_path, monkeypatch):
    keys = {name: secrets.token_hex(24) for name in
            ("instance", "rotated", "relay", "manager", "guest", "legacy", "tool", "observer")}
    table = CredentialTable.from_config({
        "api-key": {"owner": ["companion"], "contacts": "all",
                    "secrets": [{"sha256": secret_digest(keys["rotated"])}]},
        "relay": {"sha256": secret_digest(keys["relay"]), "owner": ["companion"], "access": ["answer"]},
        "manager": {"sha256": secret_digest(keys["manager"]), "owner": ["companion"], "contacts": "all"},
        "guest": {"sha256": secret_digest(keys["guest"]), "owner": [], "access": ["manage"]},
        "legacy": {"sha256": secret_digest(keys["legacy"]), "owner": "any", "contacts": "all", "scopes": ["*"]},
        "observer": {"sha256": secret_digest(keys["observer"]), "owner": ["companion"],
                     "access": ["observe"], "observe_sources": ["sensor"]},
    })
    # A tool-origin grant, built directly, also checks that this operation stays isolated from the route.
    table = CredentialTable({**{grant.name: grant for grant in table.grants()},
                             "tool": Grant(name="tool", owner=(), contacts="own", access=frozenset({"tool-origin"}),
                                           secrets=((secret_digest(keys["tool"]), None),))})
    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware, api_key=keys["instance"], credentials=table)
    app.include_router(routes.router)
    store = HandsStore(tmp_path / "hands.db")
    await store.connect()
    await seed(store)
    await sessions.register(store, "child-session", "h-000000000001", parent_session="owner-session", epoch=EPOCH)
    await sessions.retire(store, "child-session")
    app.state.hands_store = store
    vault = Vault(tmp_path / "audit")
    monkeypatch.setattr(routes, "_vault", vault)
    password = secrets.token_hex(24)
    calls = []

    def read(handle):
        calls.append(handle)
        if handle != HANDLE:
            raise saved_login.SavedPasswordNotFound()
        return password

    monkeypatch.setattr(saved_login, "saved_password", read)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
        yield SimpleNamespace(app=app, client=client, keys=keys, password=password, calls=calls, vault=vault)
    await store.close()


async def post(rig, body=None, *, caller="relay", headers=None, query=""):
    auth = {"Authorization": "Bearer " + rig.keys[caller]} if caller else {}
    return await rig.client.post(SECRET_REVEAL_PATH + query, json=BODY if body is None else body,
                                 headers={**auth, **(headers or {})})


def audit_rows(rig):
    if not rig.vault.db_path.exists():
        return []
    # Inspection is read-only. Fixture stores are never production databases.
    with sqlite3.connect(rig.vault.db_path.as_uri() + "?mode=ro", uri=True) as db:
        return db.execute("SELECT ref, at, purpose, detail FROM sealed_uses").fetchall()


async def test_owner_gets_one_password_and_each_reveal_commits_one_audit(rig, caplog):
    for _ in range(2):
        response = await post(rig)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        value = response.json()
        assert set(value) == {"id", "password"} and value["id"] == HANDLE
        matches = secrets.compare_digest(value["password"], rig.password)
        assert matches
    assert rig.calls == [HANDLE, HANDLE]
    rows = audit_rows(rig)
    assert len(rows) == 2
    for handle, at, purpose, detail in rows:
        assert (handle, purpose) == (HANDLE, "owner_reveal") and at
        assert json.loads(detail) == {"credential": "relay",
                                      "session_sha256": hashlib.sha256(b"owner-session").hexdigest()}
    safe = all(rig.password not in str(row) for row in rows) and rig.password not in caplog.text
    assert safe


@pytest.mark.parametrize("caller", [None, "instance", "rotated", "manager", "guest", "legacy", "tool", "observer"])
async def test_non_owner_and_model_tool_credentials_have_no_existence_oracle(rig, caller):
    responses = [await post(rig, {**BODY, "id": handle}, caller=caller) for handle in (HANDLE, MISSING)]
    assert all(r.status_code == (401 if caller is None else 403) for r in responses)
    assert responses[0].content == responses[1].content
    assert all(r.headers["cache-control"] == "no-store" for r in responses)
    assert not rig.calls and not audit_rows(rig)


@pytest.mark.parametrize("identity", ["body", "header", "query", "nested"])
async def test_hand_sessions_including_retired_aliases_are_refused(rig, identity):
    body, headers, query = dict(BODY), {}, ""
    if identity == "body":
        body["session_id"] = "child-session"
    elif identity == "header":
        headers["X-ColonyMind-Session"] = "child-session"
    elif identity == "query":
        query = "?session_id=child-session"
    else:
        body["context"] = {"session_id": "child-session"}
    response = await post(rig, body, headers=headers, query=query)
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "hand_session_refused"
    assert response.headers["cache-control"] == "no-store"
    assert not rig.calls and not audit_rows(rig)


@pytest.mark.parametrize("source,selector", [
    ("query", "session"), ("query", "session_id"), ("query", "requester"),
    ("header", "X-ColonyMind-Session"), ("body", "session_id"),
    ("context", "session_id"), ("metadata", "requester"), ("turn", "session"),
])
@pytest.mark.parametrize("values", [
    ("owner-session", "child-session"), ("child-session", "owner-session"),
    ("owner-session", "owner-session"),
])
async def test_repeated_session_selectors_are_invalid_in_every_order(rig, source, selector, values):
    headers = [("Authorization", "Bearer " + rig.keys["relay"])]
    content, params = json.dumps(BODY), []
    pairs = [(selector, value) for value in values]
    if source == "query":
        params = pairs
    elif source == "header":
        headers += [(selector.lower(), values[0]), (selector.upper(), values[1])]
    else:
        repeated = ",".join(json.dumps(key) + ":" + json.dumps(value) for key, value in pairs)
        if source == "body":
            content = '{"id":' + json.dumps(HANDLE) + "," + repeated + "}"
        else:
            content = content[:-1] + "," + json.dumps(source) + ":{" + repeated + "}}"
    response = await rig.client.post(SECRET_REVEAL_PATH, content=content, params=params, headers=headers)
    assert response.status_code == 422
    assert response.json() == {"detail": {"code": "invalid_reveal_request"}}
    assert response.headers["cache-control"] == "no-store"
    assert not rig.calls and not audit_rows(rig)


@pytest.mark.parametrize("source", ["query", "context", "metadata", "turn"])
@pytest.mark.parametrize("selector", ["session", "session_id", "requester"])
async def test_every_session_alias_is_checked_against_retired_hands(rig, source, selector):
    body, params = dict(BODY), []
    if source == "query":
        params = [(selector, "child-session")]
    else:
        body[source] = {selector: "child-session"}
    response = await rig.client.post(SECRET_REVEAL_PATH, json=body, params=params,
                                     headers={"Authorization": "Bearer " + rig.keys["relay"]})
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "hand_session_refused"
    assert response.headers["cache-control"] == "no-store"
    assert not rig.calls and not audit_rows(rig)


@pytest.mark.parametrize("content", [b"{", b"\xff", b"null", b"[" * 1000 + b"0" + b"]" * 1000,
                                     b"[" * 20000 + b"0" + b"]" * 20000])
async def test_malformed_or_deep_json_is_a_private_validation_error(rig, content, caplog):
    async with AsyncClient(transport=ASGITransport(app=rig.app, raise_app_exceptions=False),
                           base_url="http://localhost") as client:
        response = await client.post(SECRET_REVEAL_PATH, content=content,
                                     headers={"Authorization": "Bearer " + rig.keys["relay"]})
    assert response.status_code == 422
    assert response.json() == {"detail": {"code": "invalid_reveal_request"}}
    assert response.headers["cache-control"] == "no-store"
    assert rig.password not in response.text + caplog.text
    assert not rig.calls and not audit_rows(rig)


@pytest.mark.parametrize("declared", [None, "1", "2048"])
async def test_stream_limit_precedes_decoding_and_stops_reading(rig, monkeypatch, declared):
    chunks_read = []

    async def chunks():
        for chunk in (b" " * 1024, b" " * 1024, b"x", b"unread"):
            chunks_read.append(len(chunk))
            yield chunk

    def must_not_decode(*_args, **_kwargs):
        pytest.fail("oversized body reached the JSON decoder")

    headers = {"Authorization": "Bearer " + rig.keys["relay"]}
    if declared is not None:
        headers["Content-Length"] = declared
    with monkeypatch.context() as patch:
        patch.setattr(routes.json, "loads", must_not_decode)
        response = await rig.client.post(SECRET_REVEAL_PATH, content=chunks(), headers=headers)
    assert response.status_code == 422
    assert response.json() == {"detail": {"code": "invalid_reveal_request"}}
    assert response.headers["cache-control"] == "no-store"
    assert chunks_read == [1024, 1024, 1]
    assert not rig.calls and not audit_rows(rig)


@pytest.mark.parametrize("declared", ["2049", "40001", "-1", "invalid"])
async def test_content_length_is_checked_before_reading_or_decoding(rig, monkeypatch, declared):
    async def unread():
        pytest.fail("invalid declared length reached the body stream")
        yield b""

    def must_not_decode(*_args, **_kwargs):
        pytest.fail("invalid declared length reached the JSON decoder")

    with monkeypatch.context() as patch:
        patch.setattr(routes.json, "loads", must_not_decode)
        response = await rig.client.post(SECRET_REVEAL_PATH, content=unread(),
                                         headers={"Authorization": "Bearer " + rig.keys["relay"],
                                                  "Content-Length": declared})
    assert response.status_code == 422
    assert response.json() == {"detail": {"code": "invalid_reveal_request"}}
    assert response.headers["cache-control"] == "no-store"
    assert not rig.calls and not audit_rows(rig)


@pytest.mark.parametrize("size", [2048, 2049])
async def test_actual_body_size_boundary_with_valid_json(rig, size):
    content = json.dumps(BODY).encode()
    content += b" " * (size - len(content))
    response = await rig.client.post(SECRET_REVEAL_PATH, content=content,
                                     headers={"Authorization": "Bearer " + rig.keys["relay"]})
    assert response.status_code == (200 if size == 2048 else 422)
    assert response.headers["cache-control"] == "no-store"
    assert len(rig.calls) == len(audit_rows(rig)) == (1 if size == 2048 else 0)


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("failure", ["authentication", "handler"])
async def test_unexpected_server_errors_are_sanitized_and_never_cached(rig, monkeypatch, caplog, debug, failure):
    def fail(*_args, **_kwargs):
        raise RuntimeError(rig.password)

    if failure == "authentication":
        monkeypatch.setattr(CredentialTable, "match", fail)
    else:
        from colonymind.api import auth
        monkeypatch.setattr(auth, "request_authority", fail)
    rig.app.debug = debug
    async with AsyncClient(transport=ASGITransport(app=rig.app, raise_app_exceptions=False),
                           base_url="http://localhost") as client:
        response = await client.post(SECRET_REVEAL_PATH, json=BODY,
                                     headers={"Authorization": "Bearer " + rig.keys["relay"]})
    assert response.status_code == 500
    assert response.json() == {"detail": {"code": "secret_reveal_unavailable"}}
    assert response.headers["cache-control"] == "no-store"
    assert rig.password not in response.text + caplog.text
    assert not rig.calls and not audit_rows(rig)


async def test_body_middleware_refusals_and_route_redirects_are_never_cached(rig):
    # Same ordering as create_app: auth wraps the general body-size middleware.
    rig.app.add_middleware(BodySizeLimitMiddleware, max_bytes=4096)
    # Adding middleware inserts it outermost; restore the production order.
    rig.app.user_middleware.reverse()
    headers = {"Authorization": "Bearer " + rig.keys["relay"]}
    response = await rig.client.post(SECRET_REVEAL_PATH, content=b" " * 4097, headers=headers)
    assert response.status_code == 413
    assert response.headers["cache-control"] == "no-store"
    response = await rig.client.post(SECRET_REVEAL_PATH + "/", json=BODY, headers=headers)
    assert response.status_code == 307
    assert response.headers["cache-control"] == "no-store"
    assert not rig.calls and not audit_rows(rig)


async def test_unknown_and_non_login_ids_are_a_finite_404_without_audit(rig):
    for handle in (MISSING, "arbitrary-unknown-id"):
        response = await post(rig, {**BODY, "id": handle})
        assert response.status_code == 404
        assert response.json() == {"detail": {"code": "secret_not_found"}}
        assert response.headers["cache-control"] == "no-store"
    assert not audit_rows(rig)


@pytest.mark.parametrize("shape", ["list", "ids", "id_list", "extra", "no_session", "session_mismatch", "oversized"])
async def test_bulk_and_untrusted_origin_fields_are_rejected_without_echo(rig, shape, caplog):
    body, headers = dict(BODY), {}
    if shape == "list":
        body = [BODY, BODY]
    elif shape == "ids":
        body = {"ids": [HANDLE, MISSING], "session_id": "owner-session"}
    elif shape == "id_list":
        body["id"] = [HANDLE, MISSING]
    elif shape == "extra":
        body.update(origin="tool", password=rig.password)
    elif shape == "no_session":
        body.pop("session_id")
    elif shape == "session_mismatch":
        headers["X-ColonyMind-Session"] = "another-session"
    else:
        body["session_id"] = "x" * 4096
    response = await post(rig, body, headers=headers)
    assert response.status_code == 422
    assert response.json() == {"detail": {"code": "invalid_reveal_request"}}
    assert response.headers["cache-control"] == "no-store"
    safe = rig.password not in response.text + caplog.text
    assert safe and not rig.calls and not audit_rows(rig)


@pytest.mark.parametrize("failure", ["vault", "audit"])
async def test_failures_never_release_or_log_a_value(rig, monkeypatch, caplog, failure):
    def fail(*_args):
        raise RuntimeError(rig.password)

    if failure == "vault":
        monkeypatch.setattr(saved_login, "saved_password", fail)
    else:
        monkeypatch.setattr(rig.vault, "note_use", fail)
    response = await post(rig)
    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "secret_reveal_unavailable"}}
    assert response.headers["cache-control"] == "no-store"
    safe = rig.password not in response.text + caplog.text
    assert safe and not audit_rows(rig)


async def test_missing_middleware_and_development_mode_never_become_owner(rig):
    for configured in (False, True):
        app = FastAPI()
        if configured:
            app.add_middleware(ApiKeyMiddleware, api_key=None, credentials=CredentialTable.from_config(None))
        app.include_router(routes.router)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            response = await client.post(SECRET_REVEAL_PATH, json=BODY)
        assert response.status_code == 403
        assert response.headers["cache-control"] == "no-store"
    assert not rig.calls


async def test_no_get_or_list_reveal_and_no_schema_or_mcp_tool(rig):
    from colonymind.api.mcp import TOOLS

    assert SECRET_REVEAL_PATH not in rig.app.openapi()["paths"]
    assert all("saved-password" not in json.dumps(tool.listing()) for tool in TOOLS)
    response = await rig.client.get(SECRET_REVEAL_PATH, headers={"Authorization": "Bearer " + rig.keys["relay"]})
    assert response.status_code == 405 and response.headers["cache-control"] == "no-store"
    assert not rig.calls


def test_real_hermes_store_bridge_reads_only_login_password(tmp_path, monkeypatch, caplog):
    # Hermes is optional in the sidecar venv; qualification runs this with its stock source on PYTHONPATH.
    module = pytest.importorskip("agent.vault_store")
    from colonymind import config
    from colonymind.secrets import cli

    monkeypatch.setattr(config, "load_config", lambda: SimpleNamespace(hermes_home=tmp_path))
    monkeypatch.setattr(cli, "_hermes_python", lambda: sys.executable)
    store = module.VaultStore(base_dir=tmp_path / "vault")
    password = secrets.token_hex(24)
    meta = store.add_item("login", "Synthetic fixture", {"identifier_type": "username", "identifier": "fixture",
                                                        "password": password}, origin="https://example.invalid")
    value = saved_login.saved_password(meta.id)
    matches = secrets.compare_digest(value, password)
    assert matches
    with pytest.raises(saved_login.SavedPasswordNotFound):
        saved_login.saved_password(MISSING)
    address = store.add_item("address", "Synthetic address", {"address_line1": "fixture", "city": "fixture",
                                                               "postal_code": "fixture", "country": "fixture"})
    with pytest.raises(saved_login.SavedPasswordNotFound):
        saved_login.saved_password(address.id)
    (tmp_path / "vault" / "vault.json.enc").write_bytes(b"invalid-ciphertext")
    with pytest.raises(saved_login.SavedPasswordUnavailable):
        saved_login.saved_password(meta.id)
    safe = password not in caplog.text
    assert safe
