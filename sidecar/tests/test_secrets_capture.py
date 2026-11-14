"""A secret in a turn is sealed before anything stores it: turn capture, the source ledger and its every writer.

The value moves into the vault and a reference stands where it was, in the ledger, its lexical index, every queue
derived from a turn (claims, opinions, appraisals, commitments, vectors), the communications log, memory search
and the context packet. The same value in the same scope gets the same reference, so replays store the same bytes.
Erasure on both forms of a message is the ledger's own test (test_turn_idempotency_ledger.py).

secret-scan: synthetic fixtures (every value below is made up and assembled at run time)
"""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from colonymind.api.routers import host
from colonymind.api.schemas.host import TurnSyncRequest
from colonymind.contacts.comms import CommsLog
from colonymind.secrets.detect import REFERENCE
from colonymind.secrets.refs import references_in
from colonymind.secrets.vault import Vault, reset_vaults
from colonymind.turns import TurnIdempotencyLedger, canonical_turn_digest
from test_turn_source_evidence import recalled, source_app  # noqa: F401 (fixture)

KEY = "sk-" + "proj" + "Qz7Wx3Er9Ty1Ui5Op2As8Df4"
WIFI = "Hunter" + "22!" + "wifi"


@pytest.fixture(autouse=True)
def fresh_vaults():
    reset_vaults()
    yield
    reset_vaults()


def dump(path) -> str:
    """Every row of a database as SQL text (what a copy of the file would give away)."""
    with sqlite3.connect(path) as conn:
        return "\n".join(conn.iterdump())


def stored_messages(path, turn_id):
    with sqlite3.connect(path) as conn:
        return json.loads(conn.execute("SELECT messages_json FROM turn_sources WHERE turn_id=?",
                                       (turn_id,)).fetchone()[0])


# -- the ledger: every writer ---------------------------------------------------------------------------------

def test_record_source_seals_before_the_digest_the_row_the_index_and_every_queue(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    assert ledger.record_source("turn-1", contact_id="p-owner", session_id="s-1", messages=[
        {"role": "user", "content": f"my key is {KEY} and the wifi password is {WIFI}"},
        {"role": "assistant", "content": "Noted."}])
    text = dump(tmp_path / "turn-idempotency.db")
    assert KEY not in text and WIFI not in text
    content = stored_messages(tmp_path / "turn-idempotency.db", "turn-1")[0]["content"]
    refs = references_in(content)
    assert [ref.label for ref in refs] == ["API key", "wifi password"]
    vault = Vault(tmp_path)
    assert {vault.resolve(ref.ref_id) for ref in refs} == {KEY, WIFI}
    assert all(vault.entry(ref.ref_id)["scope_contact"] == "p-owner" for ref in refs)
    # Memory search finds the turn by what the value was, never by the value.
    hits = ledger.search_sources("the wifi password I gave you", contact_id="p-owner", session_id="later")
    assert hits and refs[1].text in hits[0]["content"] and WIFI not in json.dumps(hits)


def test_the_queues_a_turn_feeds_carry_the_reference_only(tmp_path):
    """Claims, opinions, appraisals, commitments and vectors are all queued from the stored turn."""
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    ledger.record_source("turn-q", contact_id="p-owner", session_id="s-1", messages=[
        {"role": "user", "content": f"Remember the wifi password is {WIFI}; I will send the deck by Friday."}])
    with sqlite3.connect(tmp_path / "turn-idempotency.db") as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        queued = {table: conn.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
                  for table in ("source_claim_jobs", "opinion_jobs", "appraisal_runs", "commitment_runs",
                                "source_vector_jobs") if table in tables}
    assert queued and all(count >= 1 for count in queued.values()), queued
    assert WIFI not in dump(tmp_path / "turn-idempotency.db")


@pytest.mark.parametrize("writer", ["history_import", "tool_observation", "checkpoint"])
def test_every_writer_is_sealed(tmp_path, writer):
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    message = {"role": "tool" if writer == "tool_observation" else "user",
               "content": [{"type": "text", "text": f"export OPENAI_API_KEY={KEY}"}]}
    ledger.record_source(f"turn-{writer}", contact_id="p-owner", session_id="s-1", messages=[message],
                         scope="session" if writer == "checkpoint" else "person",
                         derive_claims=writer != "history_import")
    assert KEY not in dump(tmp_path / "turn-idempotency.db")
    assert message["content"][0]["text"].endswith(KEY)       # the caller's own input is never changed


def test_every_text_a_message_carries_is_sealed_not_only_its_content(tmp_path):
    """A tool observation's executed arguments and the reason it was kept, a message's metadata: all stored."""
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    ledger.record_source("turn-meta", contact_id="p-owner", session_id="mind", messages=[
        {"role": "assistant", "content": "Noted the vendor setup.",
         "metadata": {"origin": "mind", "subject": f"vendor key {KEY}"}}], derive_claims=False)
    ledger.record_source("turn-prov", contact_id="p-owner", session_id="s-1", messages=[
        {"role": "tool", "content": "ok", "provenance": {
            "kind": "native-tool-observation-v1", "selection": {"author": "model", "reason": f"used {KEY}"},
            "input": {"arguments": {"command": f"curl -H 'Authorization: Bearer {KEY}' https://api.vendor.io"}}}}],
        derive_claims=False)
    text = dump(tmp_path / "turn-idempotency.db")
    assert KEY not in text and REFERENCE.search(text)


def test_structure_is_left_alone_only_where_it_is_structure(tmp_path):
    """A field named like structure deeper in a message (``content``, ``native``, ``role`` under metadata) is
    words like any other; only a message's own role, source references and a native observation's identity are
    left as they are."""
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    ledger.record_source("turn-deep", contact_id="p-owner", session_id="s-1", messages=[
        {"role": "assistant", "content": "Noted.",
         "metadata": {"content": f"vendor key {KEY}", "native": {"note": f"key {KEY}"},
                      "role": f"the wifi password is {WIFI}", "credentials": {"password": "SyntheticPass8!Demo"}}}],
        derive_claims=False)
    text = dump(tmp_path / "turn-idempotency.db")
    assert KEY not in text and WIFI not in text and "SyntheticPass8!Demo" not in text
    stored = stored_messages(tmp_path / "turn-idempotency.db", "turn-deep")[0]
    assert stored["role"] == "assistant" and stored["content"] == "Noted."
    assert REFERENCE.fullmatch(stored["metadata"]["credentials"]["password"])


def test_an_annotation_is_sealed_like_any_source(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    ledger.record_source("turn-a", contact_id="p-owner", session_id="s-1",
                         messages=[{"role": "user", "content": f"the router is in the hall, my key is {KEY}"}])
    version = canonical_turn_digest(stored_messages(tmp_path / "turn-idempotency.db", "turn-a"))
    note = dict(contact_id="p-owner", session_id="s-1", annotation_id="a-1", source_id="turn-a",
                source_version=version, excerpt=f"my key is {KEY}",
                correction=f"the router admin password is {WIFI}", author_principal="owner")
    first = ledger.append_source_annotation(**note)
    assert first["created"] is True
    assert ledger.append_source_annotation(**note)["created"] is False   # the same request, the same note
    text = dump(tmp_path / "turn-idempotency.db")
    assert WIFI not in text and KEY not in text
    stored = stored_messages(tmp_path / "turn-idempotency.db", first["source_id"])[0]["content"]
    assert [ref.label for ref in references_in(stored)] == ["API key", "password"]


def test_a_replay_stores_identical_bytes_and_two_scopes_two_references(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    messages = [{"role": "user", "content": f"key {KEY}"}]
    assert ledger.record_source("turn-r", contact_id="p-owner", session_id="s-1", messages=messages) is True
    first = stored_messages(tmp_path / "turn-idempotency.db", "turn-r")
    assert ledger.record_source("turn-r", contact_id="p-owner", session_id="s-1", messages=messages) is False
    assert stored_messages(tmp_path / "turn-idempotency.db", "turn-r") == first
    ledger.record_source("turn-g", contact_id="p-guest", session_id="s-2", messages=messages)
    other = stored_messages(tmp_path / "turn-idempotency.db", "turn-g")
    assert references_in(first[0]["content"])[0].ref_id != references_in(other[0]["content"])[0].ref_id


def test_what_memory_needs_is_stored_as_it_came(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    words = ("Order 1234567890123 ships with tracking 1Z999AA10123456784, fixed in a3f5c9e "
             "(commit " + "9c1f" * 10 + "), call +14155552671, booking ABC123XYZ789.")
    ledger.record_source("turn-plain", contact_id="p-owner", session_id="s-1",
                         messages=[{"role": "user", "content": words}])
    assert stored_messages(tmp_path / "turn-idempotency.db", "turn-plain")[0]["content"] == words
    assert not (tmp_path / "secrets-vault.db").exists() or Vault(tmp_path).entries() == []


def test_the_vault_unavailable_withholds_the_value_and_never_stores_it(tmp_path):
    (tmp_path / "secrets-vault.key").write_text("not a key\n")
    (tmp_path / "secrets-vault.key").chmod(0o600)
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    ledger.record_source("turn-w", contact_id="p-owner", session_id="s-1",
                         messages=[{"role": "user", "content": f"key {KEY}"}])
    content = stored_messages(tmp_path / "turn-idempotency.db", "turn-w")[0]["content"]
    assert KEY not in content and "withheld" in content


# -- turn capture -----------------------------------------------------------------------------------------------

def turn(turn_id, text, *, contact="p-owner", sender=None):
    body = {"identity": {"host_id": "fixture"},
            "context": {"session_id": "session-a", "contact_id": contact, "channel_id": "test:thread-a",
                        **({"turn_id": turn_id} if turn_id else {})},
            "user_message": {"role": "user", "content": text},
            "assistant_message": {"role": "assistant", "content": f"Saved it: {text}"},
            "summary": f"The owner shared: {text}", "topics": [f"credentials {KEY}"]}
    if sender:
        body["sender"] = sender
    return body


@pytest.mark.asyncio
@pytest.mark.parametrize("keyed", [True, False], ids=["idempotent", "unkeyed"])
async def test_turn_capture_seals_everything_it_keeps(source_app, tmp_path, monkeypatch, keyed):
    log = CommsLog(str(tmp_path / "colonymind-comms.db"),
                   source_ledger=TurnIdempotencyLedger(tmp_path / "turn-idempotency.db"))
    monkeypatch.setattr(host, "_comms_log", log)
    text = f"the wifi password is {WIFI} and my key is {KEY}"
    async with AsyncClient(transport=ASGITransport(app=source_app), base_url="http://test") as client:
        if keyed:
            response = await client.put("/v2/host/turns/turn-cap", json=turn("turn-cap", text))
            assert response.status_code == 201, response.text
            again = await client.put("/v2/host/turns/turn-cap", json=turn("turn-cap", text))
            assert again.status_code == 200 and again.json() == response.json()   # a replay, never a conflict
        else:
            response = await client.post("/v1/host/turns/sync", json=turn(None, text))
            assert response.status_code == 200, response.text
        assert response.json()["source_recorded"] is True
        packet = await recalled(client, contact="p-owner", session="session-b", query="what is the wifi password")
    assert WIFI not in packet and KEY not in packet and REFERENCE.search(packet), packet
    for path in tmp_path.glob("*.db"):
        if path.name != "secrets-vault.db":
            assert WIFI not in dump(path) and KEY not in dump(path), path.name
    assert any(REFERENCE.search(row["summary"] or "") for row in log.history("p-owner"))


@pytest.mark.asyncio
async def test_a_direct_caller_of_turn_processing_is_sealed_too(source_app, tmp_path):
    body = TurnSyncRequest.model_validate(turn("turn-direct", f"my key is {KEY}"))
    result = await host._process_turn_sync(body)
    assert result.source_recorded is True
    assert KEY not in dump(tmp_path / "turn-idempotency.db")
    assert KEY not in body.user_message.content and REFERENCE.search(body.user_message.content)


class Contacts:
    def __init__(self, known):
        self.known = known

    async def resolve_handle(self, gateway, address):
        contact = self.known.get((gateway, address))
        return SimpleNamespace(contact_id=contact) if contact else None


@pytest.mark.asyncio
async def test_a_senders_value_is_theirs_never_the_contact_the_client_claimed(tmp_path, monkeypatch):
    """In a shared session the client's contact can be stale: the value is scoped to the sender (their contact,
    or their handle when no contact has it), so it never lands in the owner's scope."""
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(host, "_contacts_store", Contacts({("telegram", "42"): "p-friend"}))
    for sender, whose in (({"platform": "telegram", "user_id": "42"}, "p-friend"),
                          ({"platform": "telegram", "user_id": "77"}, "handle:telegram:77")):
        body = TurnSyncRequest.model_validate(turn(None, f"key {KEY}", contact="p-owner", sender=sender))
        assert await host._secret_scope(body) == whose
    assert await host._secret_scope(TurnSyncRequest.model_validate(turn(None, "hi"))) == "p-owner"


# -- the seal and resolve routes --------------------------------------------------------------------------------

@pytest.fixture
def secrets_app(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from colonymind.api.routers import secrets as secrets_router
    vault = Vault(tmp_path)
    monkeypatch.setattr(secrets_router, "_vault", vault)
    monkeypatch.setattr(secrets_router, "_owner_id", lambda: "p-owner")
    monkeypatch.setattr(secrets_router, "_mind", lambda: None)
    app = FastAPI()
    app.include_router(secrets_router.router)
    return app, vault


async def _post(app, path, body):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post(path, json=body)


@pytest.mark.asyncio
async def test_the_gateway_seal_route_scopes_the_owner_and_everyone_else(secrets_app):
    app, vault = secrets_app
    owner = await _post(app, "/v1/secrets/seal", {"text": f"key {KEY}", "owner": True})
    guest = await _post(app, "/v1/secrets/seal", {"text": f"key {KEY}", "contact_id": "handle:telegram:77"})
    assert owner.status_code == guest.status_code == 200
    assert KEY not in owner.text + guest.text
    owner_ref, guest_ref = owner.json()["refs"][0]["ref"], guest.json()["refs"][0]["ref"]
    assert vault.entry(owner_ref)["scope_contact"] == "p-owner"
    assert vault.entry(guest_ref)["scope_contact"] == "handle:telegram:77"
    assert (await _post(app, "/v1/secrets/seal", {"text": "no scope"})).status_code == 422


@pytest.mark.asyncio
async def test_a_contacts_value_is_never_resolved_and_the_owners_only_in_his_own_lane(secrets_app):
    app, vault = secrets_app
    mine = vault.seal(KEY, contact_id="p-owner", kind="prefix", label="API key")
    theirs = vault.seal(WIFI, contact_id="p-friend", kind="labeled", label="wifi password")
    for purpose in ("use", "reveal", "fill", "send"):
        answer = (await _post(app, "/v1/secrets/resolve", {
            "refs": [theirs], "purpose": purpose, "lane": "owner", "own_refs": [theirs],
            "action": {"tool": "send_message", "args": {"message": f"send [secret {theirs}: wifi password]"}}})).json()
        assert answer["values"] == {} and theirs in answer["refused"], purpose
    for lane in ("guest", "cron", "worker", "unknown"):
        answer = (await _post(app, "/v1/secrets/resolve", {"refs": [mine], "purpose": "use", "lane": lane})).json()
        assert answer["values"] == {}, lane
    used = (await _post(app, "/v1/secrets/resolve", {"refs": [mine], "purpose": "use", "lane": "owner"})).json()
    assert used["values"] == {mine: KEY}
    filled = (await _post(app, "/v1/secrets/resolve", {
        "refs": [mine], "purpose": "fill", "lane": "owner", "own_refs": [mine],
        "action": {"tool": "send_message", "args": {"message": f"[secret {mine}: API key]", "target": "telegram:9"}}}
    )).json()
    assert filled["values"] == {mine: KEY}
    assert vault.entry(mine)["uses"] == 2


@pytest.mark.asyncio
async def test_the_sealed_store_is_the_owners_whatever_a_request_names(secrets_app):
    """Only a credential granted the owner reaches the sealed store (docs/CREDENTIALS.md): one that manages the
    rest of the API without the owner grant, or a 1.9.0 principal carried with the unscoped API, is refused on
    every route, whatever lane, contact or session its body names; the instance key still reaches it."""
    from colonymind.api.credentials import CredentialTable, secret_digest
    from colonymind.api.middleware import ApiKeyMiddleware
    app, vault = secrets_app
    instance, manager, carried = "instance-key-0123456789abcdef", "manager-secret-0123456789", "carried-0123456789ab"
    app.add_middleware(ApiKeyMiddleware, api_key=instance, credentials=CredentialTable.from_config({
        "api-key": {"owner": ["whatsapp", "sms", "rcs", "voice"], "contacts": "all"},
        "manager": {"sha256": secret_digest(manager), "owner": [], "contacts": "all",
                    "access": ["read", "write", "manage"]},
        "carried": {"sha256": secret_digest(carried), "owner": [], "contacts": "all", "scopes": ["api:access"]}}))
    mine = vault.seal(KEY, contact_id="p-owner", kind="prefix", label="API key")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for secret in (manager, carried):
            headers = {"Authorization": "Bearer " + secret}
            for method, path, body in (
                    ("GET", "/v1/secrets", None), ("GET", f"/v1/secrets/{mine}", None),
                    ("POST", "/v1/secrets/resolve", {"refs": [mine], "purpose": "reveal", "lane": "owner",
                                                     "contact_id": "p-owner", "session_id": "chat-1"}),
                    ("POST", "/v1/secrets/seal", {"text": f"key {WIFI}", "owner": True}),
                    ("POST", f"/v1/secrets/{mine}/release", None), ("POST", f"/v1/secrets/{mine}/forget", None)):
                refused = await client.request(method, path, headers=headers, json=body)
                assert refused.status_code == 403 and refused.json()["detail"]["code"] == "not_owner", (secret, path)
                assert KEY not in refused.text
        used = await client.post("/v1/secrets/resolve", headers={"Authorization": "Bearer " + instance},
                                 json={"refs": [mine], "purpose": "use", "lane": "owner"})
    assert used.status_code == 200 and used.json()["values"] == {mine: KEY}
    assert vault.entry(mine)["state"] != "forgotten" and vault.entry(mine)["uses"] == 1
    assert [entry["ref"] for entry in vault.entries()] == [mine]


@pytest.mark.asyncio
async def test_forget_and_the_listing_never_show_a_value(secrets_app):
    app, vault = secrets_app
    ref = vault.seal(KEY, contact_id="p-owner", kind="prefix", label="API key")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        listing = await client.get("/v1/secrets")
        shown = await client.get(f"/v1/secrets/{ref}")
        forgotten = await client.post(f"/v1/secrets/{ref}/forget")
    assert KEY not in listing.text + shown.text + forgotten.text
    assert listing.json()["entries"][0]["label"] == "API key"
    assert forgotten.json()["forgotten"] is True and vault.entry(ref)["state"] == "forgotten"


def test_without_a_key_no_value_ever_leaves(monkeypatch):
    """Development mode (no key) serves loopback callers, but never a route that hands out or restores a value."""
    from fastapi.testclient import TestClient
    from colonymind.server import create_app
    monkeypatch.delenv("COLONYMIND_API_KEY", raising=False)
    monkeypatch.delenv("COLONYMIND_API_KEYRING_PATH", raising=False)
    client = TestClient(create_app(), base_url="http://localhost:7777", client=("127.0.0.1", 50000),
                        raise_server_exceptions=False)
    resolve = client.post("/v1/secrets/resolve", json={"refs": ["S-AAAAAAAA"], "purpose": "reveal", "lane": "owner"})
    assert resolve.status_code == 503
    assert client.post("/v1/secrets/S-AAAAAAAA/release").status_code == 503
    listing = client.get("/v1/secrets")                          # the owner's labels: development mode is never them
    assert listing.status_code == 403 and listing.json()["detail"]["code"] == "route_not_granted"
