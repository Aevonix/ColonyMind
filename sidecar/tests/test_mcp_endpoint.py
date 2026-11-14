"""The MCP endpoint: read-first, and bound to the credential that calls it (docs/MCP.md).

``POST /v1/mcp`` speaks the Model Context Protocol over streamable HTTP, answering with JSON. Every request carries
an issued credential: no credential (development mode included) and the instance key get nothing. Who the tools act
for comes from that credential alone, never from an argument: a credential granted the owner reads the owner's
memory and the mind's owner view, one bound to a person reads that person's, one bound to its own identity reads its
own. The tools read memory, the open commitments, the mind's status, why it acted and its open asks; the one write,
``note``, is kept under the credential's own identity as session evidence, which never becomes a claim, an
appraisal, a commitment or a question to the owner. Nothing answers asks, changes grants or the level, or erases.
"""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from onekey import KEY
from colonymind.api.credentials import CredentialTable, secret_digest
from colonymind.api.errors import install_exception_handlers
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import host
from colonymind.api.routers import mind as mind_router
from colonymind.commitments.store import CommitmentStore
from colonymind.initiatives.store import InitiativeStore
from colonymind.turns import get_turn_idempotency_ledger

PATH = "/v1/mcp"
OWNER = "cid-owner"
GUEST = "cid-guest"
NOW = datetime(2027, 3, 1, 12, 0, tzinfo=timezone.utc)
PRIVATE = "the owner's private plan"
OWNER_SAID = "My passport renewal appointment is on Tuesday at the consulate."
GUEST_SAID = "Please send the passport photos in the blue folder."
GUEST_PROMISE = "send the passport photos"

SECRETS = {name: f"{name}-credential-secret-0123456789" for name in
           ("desk", "reader", "front", "game-bot", "writer", "ops", "expired", "legacy")}
READ_TOOLS = {"recall", "open_commitments", "mind_status", "why", "asks"}
ALL_TOOLS = READ_TOOLS | {"note"}
EM_DASH = chr(0x2014)          # the tools' own words carry none
PERSON_SELECTORS = {"contact_id", "person_id", "viewer", "viewer_person_id", "sender", "sender_id", "by", "owner",
                    "principal", "credential", "identity", "audience"}


def table() -> CredentialTable:
    def digest(name):
        return secret_digest(SECRETS[name])
    return CredentialTable.from_config({
        "api-key": {"owner": ["whatsapp", "sms", "rcs", "voice"], "contacts": "all"},
        # The owner's own reader for a coding agent: the owner's view, reading and leaving notes, writing no turn.
        "desk": {"sha256": digest("desk"), "owner": ["mcp"], "contacts": "all", "access": ["read", "note"]},
        "reader": {"sha256": digest("reader"), "owner": ["mcp"], "contacts": "all", "access": ["read"]},
        # Bound to one guest.
        "front": {"sha256": digest("front"), "owner": [], "contacts": [GUEST], "viewer": GUEST,
                  "access": ["read", "note"]},
        # Its own identity only, with the default access (read and write).
        "game-bot": {"sha256": digest("game-bot"), "owner": [], "contacts": "own"},
        "writer": {"sha256": digest("writer"), "owner": [], "contacts": "own", "access": ["write"]},
        "ops": {"sha256": digest("ops"), "owner": ["mcp"], "contacts": "all", "access": ["manage"]},
        "expired": {"sha256": digest("expired"), "owner": ["mcp"], "contacts": "all", "access": ["read"],
                    "accept_until": "2020-01-01T00:00:00+00:00"},
        # Carried from 1.9.0: its scopes name no MCP tool.
        "legacy": {"sha256": digest("legacy"), "owner": "any", "contacts": "all", "scopes": ["*"]},
    })


def bearer(name: str) -> dict:
    return {"Authorization": "Bearer " + (KEY if name == "api-key" else SECRETS[name])}


def message(method: str, params: dict | None = None, *, ident: int | str | None = 1) -> dict:
    body = {"jsonrpc": "2.0", "method": method}
    if ident is not None:
        body["id"] = ident
    if params is not None:
        body["params"] = params
    return body


def turn(contact_id: str, text: str, *, turn_id: str, channel: str) -> dict:
    return {"identity": {"host_id": "hermes"},
            "context": {"session_id": "session-1", "contact_id": contact_id, "turn_id": turn_id,
                        "channel_id": channel},
            "user_message": {"role": "user", "content": text},
            "assistant_message": {"role": "assistant", "content": "Noted."}}


def rows(state, sql: str) -> list:
    with closing(get_turn_idempotency_ledger(state)._connect()) as conn:
        return [tuple(row) for row in conn.execute(sql)]


def made(store: CommitmentStore, commitment_id: str, when: datetime) -> None:
    """Set when a commitment was made: the store lists the newest first, so the test decides that order."""
    with closing(store._connect()) as conn:
        conn.execute("UPDATE commitments SET made_at=? WHERE id=?", (when.isoformat(), commitment_id))
        conn.commit()


def queued(state) -> dict:
    """The derived work a stored turn schedules: claims, appraisals, commitments and opinions."""
    return {name: rows(state, f"SELECT count(*) FROM {name}")[0][0]  # noqa: S608
            for name in ("source_claim_jobs", "appraisal_runs", "commitment_runs", "opinion_jobs")}


def app_with(api_key: str | None) -> FastAPI:
    from colonymind.api.mcp import router as mcp_router
    app = FastAPI()
    install_exception_handlers(app)
    for router in (host.router, host.v2_router, mind_router.router, mcp_router):
        app.include_router(router)
    app.add_middleware(ApiKeyMiddleware, api_key=api_key, credentials=table())
    return app


@pytest.fixture
async def world(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    for name in ("_telemetry", "_reranker", "_context_recall_selector", "_comms_log", "_preference_learner",
                 "_contacts_store", "_commitment_store"):
        monkeypatch.setattr(host, name, None, raising=False)
    initiatives = InitiativeStore(state_dir=tmp_path)
    row, _ = initiatives.create_intention(kind="task", type="research", title=PRIVATE, drive="curiosity",
                                          cls="internal", decision="act", decision_reason="test",
                                          status="dispatched", dedup_key=None, recipient=None, created_at=NOW)
    asked = []
    stand_in = SimpleNamespace(
        store=initiatives, clock=lambda: NOW, enabled=True, owner_id=OWNER,
        state=lambda: {"enabled": True, "autonomy": "standard", "working_on": PRIVATE},
        asks=lambda: asked + [{"ask_code": "AB12", "title": PRIVATE, "decision_reason": "r", "expires_at": "later"}])
    mind_router.set_mind(stand_in)
    commitments = CommitmentStore(tmp_path / "commitments.db")
    monkeypatch.setattr(host, "_commitment_store", commitments)
    app = app_with(KEY)
    from colonymind.api.routers import curated_memory
    app.include_router(curated_memory.router)
    from colonymind.turns import TurnIdempotencyLedger
    curated_memory.bind_ledger(app, TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db'))
    async with AsyncClient(transport=ASGITransport(app=app, client=("127.0.0.1", 50000)),
                           base_url="http://127.0.0.1:7777") as client:
        for body in (turn(OWNER, OWNER_SAID, turn_id="owner-1", channel="whatsapp:+15550000001"),
                     turn(GUEST, GUEST_SAID, turn_id="guest-1", channel="whatsapp:+15550000002")):
            stored = await client.post("/v1/host/turns/sync", json=body, headers=bearer("api-key"))
            assert stored.status_code == 200 and stored.json()["source_recorded"] is True, stored.text
        due = (datetime.now(timezone.utc) + timedelta(days=2)).replace(microsecond=0).isoformat()
        commitments.create(OWNER, "renew the passport at the consulate", due_at=due)
        commitments.create(GUEST, GUEST_PROMISE, due_at=due, metadata={"source_turn_id": "guest-1"})
        commitments.create(GUEST, "a private note about the guest", due_at=due)
        done = commitments.create(OWNER, "a finished errand", due_at=due)
        commitments.update(done["id"], status="fulfilled")
        try:
            yield SimpleNamespace(client=client, state=tmp_path, row=row.id, initiatives=initiatives,
                                  commitments=commitments)
        finally:
            mind_router.set_mind(None)
            initiatives.close()


async def rpc(world, name: str, method: str, params: dict | None = None, *, headers: dict | None = None):
    return await world.client.post(PATH, json=message(method, params),
                                   headers={**bearer(name), "Accept": "application/json, text/event-stream",
                                            **(headers or {})})


async def call(world, name: str, tool: str, arguments: dict | None = None) -> dict:
    response = await rpc(world, name, "tools/call", {"name": tool, "arguments": arguments or {}})
    assert response.status_code == 200, response.text
    body = response.json()
    assert "error" not in body, body
    return body["result"]


def text_of(result: dict) -> str:
    return "\n".join(block["text"] for block in result["content"] if block.get("type") == "text")


async def tool_names(world, name: str) -> set[str]:
    response = await rpc(world, name, "tools/list")
    assert response.status_code == 200, response.text
    return {tool["name"] for tool in response.json()["result"]["tools"]}


# --- No credential, no access ---------------------------------------------------------------------------------------

async def test_no_credential_no_access(world):
    for headers in ({}, {"Authorization": "Bearer not-a-credential"}, bearer("expired"),
                    {"Authorization": "Bearer "}):
        for method in ("POST", "GET", "DELETE"):
            response = await world.client.request(method, PATH, json=message("tools/list") if method == "POST"
                                                  else None, headers=headers)
            assert response.status_code == 401, (method, headers, response.text)
            assert response.headers.get("www-authenticate", "").lower().startswith("bearer")
            assert "recall" not in response.text and PRIVATE not in response.text


async def test_the_instance_key_is_never_an_mcp_credential(world):
    """The instance key is the Hermes adapter's: an MCP client gets a credential of its own, which the owner can
    bind, narrow and retire on its own."""
    response = await rpc(world, "api-key", "tools/list")
    assert response.status_code == 403, response.text
    assert response.json()["detail"]["code"] == "instance_key_refused"
    assert "colonymind credentials add" in response.json()["detail"]["message"]


async def test_development_mode_serves_no_mcp(tmp_path, monkeypatch):
    """Without a key the sidecar serves loopback callers as development mode; the MCP endpoint serves no one."""
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    app = app_with(None)
    async with AsyncClient(transport=ASGITransport(app=app, client=("127.0.0.1", 50000)),
                           base_url="http://127.0.0.1:7777") as client:
        for headers in ({}, bearer("desk")):
            response = await client.post(PATH, json=message("tools/list"), headers=headers)
            assert response.status_code == 401, response.text
            assert response.json()["detail"]["code"] == "credential_required"


async def test_the_endpoint_serves_no_one_without_the_auth_middleware(tmp_path, monkeypatch):
    """Mounted without the middleware (another app that includes the routers), the endpoint still has no credential to
    act for, so it serves nothing rather than acting as development mode's caller."""
    from colonymind.api.mcp import router as mcp_router
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    app = FastAPI()
    app.include_router(mcp_router)
    async with AsyncClient(transport=ASGITransport(app=app, client=("127.0.0.1", 50000)),
                           base_url="http://127.0.0.1:7777") as client:
        for headers in ({}, bearer("desk")):
            response = await client.post(PATH, json=message("tools/list"), headers=headers)
            assert response.status_code == 401, response.text
            assert response.json()["detail"]["code"] == "credential_required" and "recall" not in response.text


@pytest.mark.parametrize("name", ["ops", "legacy"])
async def test_a_credential_granted_neither_read_nor_note_gets_no_tools(world, name):
    response = await rpc(world, name, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                                     "clientInfo": {"name": "fixture", "version": "1"}})
    assert response.status_code == 403, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "mcp_not_granted" and name in detail["message"]


async def test_a_request_from_a_foreign_origin_is_refused(world):
    """A browser page that rebinds a name to this machine sends its own origin: only a loopback origin passes."""
    for origin in ("http://attacker.example", "null", "http://192.0.2.10:7777"):
        response = await rpc(world, "desk", "tools/list", headers={"Origin": origin})
        assert response.status_code == 403, (origin, response.text)
        assert response.json()["detail"]["code"] == "origin_not_allowed"
    for origin in ("http://localhost:3000", "http://127.0.0.1:7777", "http://[::1]:8080"):
        assert (await rpc(world, "desk", "tools/list", headers={"Origin": origin})).status_code == 200, origin


# --- The protocol ---------------------------------------------------------------------------------------------------

async def test_initialize_negotiates_the_version_and_names_the_server(world):
    response = await rpc(world, "desk", "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                                       "clientInfo": {"name": "fixture", "version": "1"}})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert "mcp-session-id" not in response.headers        # stateless: every request carries its credential
    result = response.json()["result"]
    assert response.json()["id"] == 1 and response.json()["jsonrpc"] == "2.0"
    assert result["protocolVersion"] == "2025-06-18"
    assert result["serverInfo"]["name"] == "colonymind" and result["serverInfo"]["version"]
    assert result["capabilities"] == {"tools": {"listChanged": False}}
    assert "credential" in result["instructions"]
    # A version this server does not speak: it answers with the newest one it does.
    newer = await rpc(world, "desk", "initialize", {"protocolVersion": "2099-01-01", "capabilities": {},
                                                    "clientInfo": {"name": "fixture", "version": "1"}})
    assert newer.json()["result"]["protocolVersion"] == "2025-11-25"


async def test_notifications_and_responses_are_accepted_without_a_body(world):
    for body in (message("notifications/initialized", ident=None),
                 message("notifications/cancelled", {"requestId": 3}, ident=None),
                 {"jsonrpc": "2.0", "id": 9, "result": {}}):
        response = await world.client.post(PATH, json=body, headers=bearer("desk"))
        assert response.status_code == 202 and response.content == b"", response.text


async def test_an_unsupported_protocol_version_header_is_refused(world):
    response = await rpc(world, "desk", "tools/list", headers={"MCP-Protocol-Version": "1999-01-01"})
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == -32600
    ok = await rpc(world, "desk", "tools/list", headers={"MCP-Protocol-Version": "2025-03-26"})
    assert ok.status_code == 200


async def test_json_rpc_errors(world):
    broken = await world.client.post(PATH, content=b"{not json", headers={**bearer("desk"),
                                                                          "Content-Type": "application/json"})
    assert broken.status_code == 400 and broken.json()["error"]["code"] == -32700
    assert broken.json()["id"] is None
    unknown = await rpc(world, "desk", "resources/list")
    assert unknown.status_code == 200 and unknown.json()["error"]["code"] == -32601
    invalid = await world.client.post(PATH, json={"id": 1, "method": "ping"}, headers=bearer("desk"))
    assert invalid.json()["error"]["code"] == -32600
    ping = await rpc(world, "desk", "ping")
    assert ping.json() == {"jsonrpc": "2.0", "id": 1, "result": {}}
    # A batch (the 2025-03-26 transport): one response per request, in order; notifications get none.
    batch = await world.client.post(PATH, json=[message("ping", ident="a"),
                                                message("notifications/initialized", ident=None),
                                                message("tools/list", ident="b")], headers=bearer("desk"))
    assert batch.status_code == 200 and [item["id"] for item in batch.json()] == ["a", "b"]


async def test_get_opens_no_stream_and_delete_ends_no_session(world):
    for method in ("GET", "DELETE"):
        response = await world.client.request(method, PATH, headers=bearer("desk"))
        assert response.status_code == 405, response.text
        assert response.headers["allow"] == "POST"


# --- Read-first: no write tool beyond note --------------------------------------------------------------------------

async def test_the_tools_are_the_read_first_set_and_the_grant_decides_which(world):
    assert await tool_names(world, "desk") == ALL_TOOLS
    assert await tool_names(world, "front") == ALL_TOOLS
    assert await tool_names(world, "game-bot") == ALL_TOOLS          # the default access writes, so it notes
    assert await tool_names(world, "reader") == READ_TOOLS           # read only: no note
    assert await tool_names(world, "writer") == {"note"}             # write only: a note, no read
    tools = (await rpc(world, "desk", "tools/list")).json()["result"]["tools"]
    for tool in tools:
        schema = tool["inputSchema"]
        assert schema["type"] == "object" and schema["additionalProperties"] is False, tool["name"]
        # Identity comes from the credential: no tool takes a person.
        assert not set(schema["properties"]) & PERSON_SELECTORS, tool["name"]
        assert tool["annotations"]["readOnlyHint"] is (tool["name"] != "note"), tool["name"]
        assert tool["annotations"]["openWorldHint"] is False
        assert tool["description"] and EM_DASH not in tool["description"]


@pytest.mark.parametrize("tool", ["answer", "approve", "grant", "set_level", "forget", "erase", "off", "chat",
                                  "complete"])
async def test_nothing_answers_grants_changes_the_level_erases_or_proxies_a_model(world, tool):
    before = world.initiatives.intentions(limit=500)
    response = await rpc(world, "desk", "tools/call", {"name": tool, "arguments": {"code": "AB12"}})
    assert response.status_code == 200 and response.json()["error"]["code"] == -32602, response.text
    assert world.initiatives.intentions(limit=500) == before


async def test_a_credential_that_may_not_note_writes_nothing(world):
    before = rows(world.state, "SELECT count(*) FROM turn_sources")
    result = await call(world, "reader", "note", {"text": "The owner prefers aisle seats."})
    assert result["isError"] is True and "note" in text_of(result)
    assert rows(world.state, "SELECT count(*) FROM turn_sources") == before


async def test_a_note_is_kept_under_the_credentials_own_identity_and_never_becomes_the_owners_claim(world):
    before = queued(world.state)
    result = await call(world, "desk", "note", {"text": "The owner prefers aisle seats on long flights."})
    assert result["isError"] is False, result
    notes = rows(world.state, "SELECT contact_id, scope FROM turn_sources WHERE contact_id LIKE 'credential:%'")
    # The desk speaks for the owner, and still its note is its own session evidence.
    assert notes == [("credential:desk", "session")]
    assert queued(world.state) == before                 # no claim, appraisal, commitment or opinion job
    owner_view = await call(world, "desk", "recall", {"query": "aisle seats"})
    assert "aisle" not in text_of(owner_view)             # not the owner's memory
    # A credential bound to its own identity reads its own notes back.
    await call(world, "game-bot", "note", {"text": "Checkpoint: level four cleared with the harbour key."})
    own = await call(world, "game-bot", "recall", {"query": "harbour key"})
    assert "harbour key" in text_of(own)
    assert rows(world.state, "SELECT contact_id, scope FROM turn_sources WHERE contact_id='credential:game-bot'") \
        == [("credential:game-bot", "session")]


# Words a machine's turn carries (a system note, a system-generated line, a system tag, an interrupted turn): a note
# that says them is still the credential's own, recallable and erasable by the identity and source the tool returned.
SYSTEM_WORDING = ("System note: ready for next task, the harbour key is in the lighthouse.",
                  "This harbour key summary was system-generated from the level log.",
                  "<system>checkpoint</system> the harbour key opens the lighthouse.",
                  "The previous turn was interrupted; the harbour key is still unused.")


async def test_an_optional_argument_sent_as_null_is_left_out(world):
    """A model often sends null for an optional argument: it reads as absent, and a required one stays required."""
    result = await call(world, "desk", "recall", {"query": "passport", "limit": None, "explain": None})
    assert result["isError"] is False and "renewal appointment" in text_of(result)
    refused = await call(world, "desk", "recall", {"query": None})
    assert refused["isError"] is True and "needs query" in text_of(refused)


async def test_note_arguments_are_checked(world):
    for arguments in ({}, {"text": ""}, {"text": 7}, {"text": "x" * 8001}, {"text": "ok", "contact_id": OWNER}):
        result = await call(world, "desk", "note", arguments)
        assert result["isError"] is True, arguments
    assert rows(world.state, "SELECT count(*) FROM turn_sources WHERE contact_id LIKE 'credential:%'") == [(0,)]


# --- Identity comes from the credential -----------------------------------------------------------------------------

async def test_recall_reads_the_credentials_person_and_no_argument_changes_it(world):
    owner = text_of(await call(world, "desk", "recall", {"query": "passport"}))
    assert "renewal appointment" in owner and "blue folder" not in owner
    guest = text_of(await call(world, "front", "recall", {"query": "passport"}))
    assert "blue folder" in guest and "renewal appointment" not in guest
    own = text_of(await call(world, "game-bot", "recall", {"query": "passport"}))
    assert "renewal appointment" not in own and "blue folder" not in own
    for selector in ({"person_id": OWNER}, {"contact_id": OWNER}, {"viewer": OWNER}):
        refused = await call(world, "front", "recall", {"query": "passport", **selector})
        assert refused["isError"] is True and "renewal appointment" not in text_of(refused), selector


async def test_recall_returns_the_packet_and_its_trace_on_request(world):
    result = await call(world, "desk", "recall", {"query": "passport", "limit": 3})
    assert result["isError"] is False
    packet = result["structuredContent"]
    assert packet["count"] >= 1 and "renewal appointment" in packet["content"] and "trace" not in packet
    explained = await call(world, "desk", "recall", {"query": "passport", "explain": True})
    assert explained["structuredContent"]["trace"]["rows"]


async def test_the_minds_views_follow_the_credential(world):
    from colonymind.health_counters import snapshot
    owner = await call(world, "desk", "mind_status")
    assert owner["structuredContent"]["working_on"] == PRIVATE
    assert owner["structuredContent"]["health_counters"] == snapshot()
    guest = await call(world, "front", "mind_status")
    assert guest["structuredContent"] == {"enabled": True, "autonomy": "standard"}
    assert PRIVATE not in text_of(guest)
    asks = await call(world, "desk", "asks")
    assert [item["ask_code"] for item in asks["structuredContent"]["asks"]] == ["AB12"]
    for name in ("front", "game-bot"):
        none = await call(world, name, "asks")
        assert none["structuredContent"]["asks"] == [] and PRIVATE not in text_of(none)
    why = await call(world, "desk", "why", {"intention_id": world.row})
    assert why["isError"] is False and why["structuredContent"]["title"] == PRIVATE
    hidden = await call(world, "front", "why", {"intention_id": world.row})
    assert hidden["isError"] is True and PRIVATE not in text_of(hidden)


async def test_a_mind_that_is_not_running_is_a_tool_error(world):
    mind_router.set_mind(None)
    result = await call(world, "desk", "mind_status")
    assert result["isError"] is True and "mind" in text_of(result)


async def test_open_commitments_follow_the_credential(world):
    owner = await call(world, "desk", "open_commitments")
    listed = {row["description"] for row in owner["structuredContent"]["commitments"]}
    assert listed == {"renew the passport at the consulate", GUEST_PROMISE, "a private note about the guest"}
    guest = await call(world, "front", "open_commitments")
    # A guest sees only what their own words already say.
    assert [row["description"] for row in guest["structuredContent"]["commitments"]] == [GUEST_PROMISE]
    own = await call(world, "game-bot", "open_commitments")
    assert own["structuredContent"]["commitments"] == []


def when(days: int) -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0) + timedelta(days=days)


async def test_open_commitments_are_overdue_first_before_the_limit(world):
    """Overdue first holds before the limit: two overdue commitments made long before the pending ones are what a
    list of two shows, and a longer list shows them first."""
    for n in (1, 2):
        late = world.commitments.create(OWNER, f"overdue errand {n}", due_at=when(-3).isoformat(), allow_overdue=True)
        made(world.commitments, late["id"], when(-30) + timedelta(minutes=n))

    async def listed(limit: int | None = None) -> list:
        result = await call(world, "desk", "open_commitments", {} if limit is None else {"limit": limit})
        return [(row["description"], row["status"]) for row in result["structuredContent"]["commitments"]]
    assert sorted(await listed(2)) == [("overdue errand 1", "overdue"), ("overdue errand 2", "overdue")]
    assert [status for _, status in await listed(3)] == ["overdue", "overdue", "pending"]
    assert [status for _, status in await listed()] == ["overdue"] * 2 + ["pending"] * 3
    text = text_of(await call(world, "desk", "open_commitments", {"limit": 1}))
    assert text.startswith("- overdue errand") and "pending" not in text


async def test_a_guests_open_commitments_are_overdue_first_before_the_limit(world, monkeypatch):
    said = turn(GUEST, "Could you book the ferry tickets?", turn_id="guest-2", channel="whatsapp:+15550000002")
    assert (await world.client.post("/v1/host/turns/sync", json=said, headers=bearer("api-key"))).status_code == 200
    late = world.commitments.create(GUEST, "book the ferry tickets", due_at=when(-3).isoformat(), allow_overdue=True,
                                    metadata={"source_turn_id": "guest-2"})
    made(world.commitments, late["id"], when(-30))

    async def listed(limit: int | None = None) -> list:
        result = await call(world, "front", "open_commitments", {} if limit is None else {"limit": limit})
        return [(row["description"], row["status"]) for row in result["structuredContent"]["commitments"]]
    assert await listed(1) == [("book the ferry tickets", "overdue")]
    assert await listed() == [("book the ferry tickets", "overdue"), (GUEST_PROMISE, "pending")]
    # Read a row at a time, the newest pending row (not the guest's to see) is left out and the list still fills.
    from colonymind.api import mcp
    monkeypatch.setattr(mcp, "COMMITMENTS_MAX", 1)
    assert await listed(2) == [("book the ferry tickets", "overdue"), (GUEST_PROMISE, "pending")]


async def test_a_guests_open_commitments_are_not_cut_at_their_contexts_five(world):
    """A guest's conversation context lists at most five commitments; the tool lists up to its limit."""
    errands = ("book the ferry tickets", "water the tomato plants", "return the library books", "fix the garden gate",
               "call the plumber about the leak", "pick up the dry cleaning", "renew the parking permit",
               "order the birthday cake")
    said = turn(GUEST, "Could you " + ", ".join(errands) + "?", turn_id="guest-2", channel="whatsapp:+15550000002")
    assert (await world.client.post("/v1/host/turns/sync", json=said, headers=bearer("api-key"))).status_code == 200
    for errand in errands:
        world.commitments.create(GUEST, errand, due_at=when(2).isoformat(), metadata={"source_turn_id": "guest-2"})
    listed = (await call(world, "front", "open_commitments", {"limit": 20}))["structuredContent"]["commitments"]
    # All nine the guest's own words show, and still never the one they do not.
    assert sorted(row["description"] for row in listed) == sorted([*errands, GUEST_PROMISE])
    assert len((await call(world, "front", "open_commitments", {"limit": 7}))["structuredContent"]["commitments"]) == 7


# --- The CLI issues a credential with its access --------------------------------------------------------------------

def test_credentials_add_and_set_take_the_access(tmp_path, capsys):
    import argparse

    from colonymind import credentials_cli
    from colonymind.config import DEFAULTS, save_config
    save_config(dict(DEFAULTS), tmp_path)
    parser = argparse.ArgumentParser()
    credentials_cli.add_parser(parser.add_subparsers(dest="command"))
    added = parser.parse_args(["credentials", "--home", str(tmp_path), "add", "desk", "--owner-platform", "mcp",
                               "--access", "read", "--access", "note"])
    assert credentials_cli.run(added) == 0
    grant = CredentialTable.load(tmp_path).get("desk")
    assert grant.access == frozenset({"read", "note"}) and grant.owner == ("mcp",)
    assert "notes" in capsys.readouterr().out
    changed = parser.parse_args(["credentials", "--home", str(tmp_path), "set", "desk", "--access", "read"])
    assert credentials_cli.run(changed) == 0
    assert CredentialTable.load(tmp_path).get("desk").access == frozenset({"read"})


def test_note_is_an_access_a_credential_may_be_granted():
    from colonymind.api.credentials import describe, validate_credentials
    checked = validate_credentials({"desk": {"sha256": "0" * 64, "owner": ["mcp"], "contacts": "all",
                                             "access": ["note", "read"]}})
    assert checked["desk"]["access"] == ["read", "note"]
    assert "leaves notes" in describe(CredentialTable.from_config(checked).get("desk"))


# --- The sidecar mounts it ------------------------------------------------------------------------------------------

def test_the_sidecar_serves_the_endpoint_behind_its_credentials(monkeypatch):
    from fastapi.testclient import TestClient

    from colonymind.server import create_app
    monkeypatch.setenv("COLONYMIND_API_KEY", "test-secret-key")
    client = TestClient(create_app(), raise_server_exceptions=False)
    assert client.post(PATH, json=message("tools/list")).status_code == 401
    refused = client.post(PATH, json=message("tools/list"), headers={"Authorization": "Bearer test-secret-key"})
    assert refused.status_code == 403 and refused.json()["detail"]["code"] == "instance_key_refused"
    # The retired server's path stays unserved.
    assert client.post("/mcp", json=message("tools/list"),
                       headers={"Authorization": "Bearer test-secret-key"}).status_code == 404


from colonymind.memory import curated
async def test_a_note_names_its_source_and_the_owner_erases_it_like_any_other(world):
    kept = await call(world, 'game-bot', 'note', {'text': 'Checkpoint: the harbour key opens the lighthouse.'})
    source = kept['structuredContent']['source_id']
    assert rows(world.state, "SELECT turn_id FROM turn_sources WHERE contact_id='credential:game-bot'") == [(source,)]
    assert 'harbour key' in text_of(await call(world, 'game-bot', 'recall', {'query': 'harbour key'}))
    erased = await world.client.post('/v1/host/memory/curated/sources/forget', headers=bearer('api-key'), json={'person_id': 'credential:game-bot', 'source_ids': [source]})
    assert erased.status_code == 200, erased.text
    assert 'harbour key' not in text_of(await call(world, 'game-bot', 'recall', {'query': 'harbour key'}))

@pytest.mark.parametrize('text', SYSTEM_WORDING)
async def test_a_note_in_a_machines_words_stays_the_credentials_own(world, text):
    kept = (await call(world, 'game-bot', 'note', {'text': text}))['structuredContent']
    assert kept['identity'] == 'credential:game-bot'
    assert rows(world.state, "SELECT contact_id, scope FROM turn_sources WHERE turn_id LIKE 'mcp-note-%'") == [('credential:game-bot', 'session')]
    assert 'harbour key' in text_of(await call(world, 'game-bot', 'recall', {'query': 'harbour key'}))
    erased = await world.client.post('/v1/host/memory/curated/sources/forget', headers=bearer('api-key'), json={'person_id': kept['identity'], 'source_ids': [kept['source_id']]})
    assert erased.status_code == 200, erased.text
    assert 'harbour key' not in text_of(await call(world, 'game-bot', 'recall', {'query': 'harbour key'}))
