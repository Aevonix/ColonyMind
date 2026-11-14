"""The memory inventory: what memory holds about one person, as claim cards.

``GET /v1/host/memory/inventory?person_id=`` lists a person's source claims, newest first, each with its state
(current, in conflict, expired, superseded, retracted), its quotation, and the exact source revision that the
annotate and forget routes act on. The owner (a credential that manages the instance) may list anyone; a credential
bound to a person may list only that person. Pages hold still under writes: a listing covers the claims that existed
when it began, and an erasure or an attribution change since then makes the next page ask for a fresh listing.
It is a read of existing records: listing writes nothing.

secret-scan: synthetic fixtures (every value below is made up and assembled at run time)
"""
from __future__ import annotations

from contextlib import closing
import importlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from onekey import KEY
from colonymind.api.credentials import CredentialTable, secret_digest
from colonymind.api.errors import install_exception_handlers
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import host
from colonymind.beliefs.source_projection import SourceClaimProjection
from colonymind.secrets.vault import reset_vaults
from colonymind.turns import get_turn_idempotency_ledger
from test_source_claim_projection import claim, prepared
from test_source_quoted_preferences import TEXT as PREFERENCE, add as add_preference, preference

GUEST = "guest-a-secret-0123456789abcdef"     # bound to contact-a
RELAY = "relay-secret-0123456789abcdef0"      # writes as anyone but the owner, bound to no one
WIFI = "Hunter" + "22!" + "wifi"
INVENTORY = "/v1/host/memory/inventory"


class Scripted:
    """Extraction proposes the scripted claims for each marker the message holds, quoting the message as it was
    stored; ``choose`` names the value of the offered prior record a proposal corrects or changes. Review keeps
    every proposal."""

    def __init__(self, outputs):
        self.outputs = outputs

    def tier_config(self, tier):
        return SimpleNamespace(base_url="http://127.0.0.1:8080/v1")

    async def complete(self, messages, **kwargs):
        payload = json.loads(messages[-1]["content"])
        if kwargs.get("context", {}).get("task") == "source_claim_review":
            return SimpleNamespace(content=json.dumps({str(row["index"]): {
                "keep": True, "reason": "Scripted review of this fixture."} for row in payload["proposals"]}),
                model_id="fixture-model")
        text, proposals = payload["message"], []
        for marker, items in self.outputs.items():
            if marker not in text:
                continue
            for item in items:
                item = dict(item)
                wanted = item.pop("choose", None)
                proposal = claim(text, **item)
                if wanted:
                    proposal["prior_claim_id"] = next(row["id"] for row in payload["prior_assertions"]
                                                      if row["value"] == wanted)
                proposals.append(proposal)
        return SimpleNamespace(content=json.dumps(proposals), model_id="fixture-model")


MODEL = Scripted({
    "office is in River": [{"value": "River"}],
    "office is now in Lake": [{"value": "Lake", "operation": "change", "valid_from_text": "2026-03-12",
                               "choose": "River"}],
    "office is in Hill": [{"value": "Hill", "operation": "correct", "choose": "Lake"}],
    "office moves to Lake": [{"value": "Lake", "operation": "change", "valid_from_text": "2026-10-01",
                              "choose": "River"}],
    "office might be in Hill": [{"value": "Hill"}],
    "office was never in River": [{"value": "Hill", "operation": "correct", "choose": "River"}],
    "starts at nine": [{"value": "nine", "subject": "workshop", "predicate": "start_time"}],
    "starts at ten": [{"value": "ten", "subject": "workshop", "predicate": "start_time", "choose": "nine"}],
    "gym pass works": [{"value": "works", "predicate": "gym_pass_status", "valid_to_text": "2026-03-05"}],
    "tea is green": [{"value": "green", "predicate": "tea_preference"}],
    "wifi password": [{"value": "wifi password", "predicate": "home_network"}],
    **{f"hobby number {n}": [{"value": f"hobby number {n}", "predicate": f"hobby_{n}"}] for n in range(10)},
})


def inventory_module():
    return importlib.import_module("colonymind.api.routers.inventory")


def make_app(*, api_key=KEY):
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(host.router)
    app.include_router(inventory_module().router)
    app.add_middleware(ApiKeyMiddleware, api_key=api_key, credentials=CredentialTable.from_config({
        "api-key": {"owner": ["whatsapp"], "contacts": "all"},
        "guest-a": {"sha256": secret_digest(GUEST), "owner": [], "contacts": ["contact-a"], "viewer": "contact-a"},
        "relay": {"sha256": secret_digest(RELAY), "owner": [], "contacts": "all"}}))
    return app


@pytest.fixture
def state(monkeypatch, tmp_path):
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_OWNER_PERSON_ID", "owner-1")
    for name in ("_contacts_store", "_telemetry", "_reranker", "_context_recall_selector", "_comms_log"):
        monkeypatch.setattr(host, name, None)
    reset_vaults()
    yield tmp_path
    reset_vaults()


def ledger_of(state):
    return get_turn_idempotency_ledger(Path(state))


async def say(state, turn, text, *, contact="contact-a", at="2026-03-01T09:00:00+00:00", session="s1"):
    ledger = ledger_of(state)
    ledger.record_source(turn, contact_id=contact, session_id=session, occurred_at=at,
                         messages=[{"role": "user", "content": text}])
    assert await SourceClaimProjection(ledger).process_one(MODEL)


def claim_ids(state, *, contact="contact-a"):
    with closing(sqlite3.connect(Path(state) / "turn-idempotency.db")) as conn:
        return {row[0]: row[1] for row in conn.execute(
            "SELECT c.id, json_extract(c.data_json,'$.value') FROM source_claims c JOIN turn_sources s "
            "ON s.turn_id=c.turn_id WHERE s.contact_id=?", (contact,))}


def id_of(state, value, *, contact="contact-a"):
    return next(ident for ident, found in claim_ids(state, contact=contact).items() if found == value)


async def get(app, path, *, key=KEY, params=None, base_url="http://test"):
    headers = {"Authorization": "Bearer " + key} if key else {}
    async with AsyncClient(transport=ASGITransport(app=app), base_url=base_url, headers=headers) as client:
        return await client.get(path, params=params)


async def listing(app, person, *, key=KEY, **params):
    return await get(app, INVENTORY, key=key, params={"person_id": person, **params})


def dump(path) -> str:
    with closing(sqlite3.connect(path)) as conn:
        return "\n".join(conn.iterdump())


# --- 1. scope --------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_owner_lists_anyone_and_a_guest_only_its_own_person(state):
    await say(state, "a-1", "My office is in River.")
    await say(state, "b-1", "My tea is green.", contact="contact-b")
    app = make_app()

    owner_a = await listing(app, "contact-a")
    assert owner_a.status_code == 200, owner_a.text
    assert [card["value"] for card in owner_a.json()["claims"]] == ["River"]
    owner_b = await listing(app, "contact-b")
    assert [card["value"] for card in owner_b.json()["claims"]] == ["green"]

    own = await listing(app, "contact-a", key=GUEST)
    assert own.status_code == 200, own.text
    assert [card["value"] for card in own.json()["claims"]] == ["River"]

    other = await listing(app, "contact-b", key=GUEST)
    assert other.status_code == 403 and "green" not in other.text
    owner_lane = await listing(app, "owner-1", key=GUEST)
    assert owner_lane.status_code == 403
    # A credential that may write as anyone but is bound to no person is not the owner: it lists no one else.
    relay = await listing(app, "contact-a", key=RELAY)
    assert relay.status_code == 403 and "River" not in relay.text
    assert relay.json()["detail"]["code"] == "inventory_not_granted"


@pytest.mark.asyncio
async def test_development_mode_lists_no_one(state):
    await say(state, "a-1", "My office is in River.")
    keyless = await listing(make_app(api_key=None), "contact-a", key=None, base_url="http://localhost")
    assert keyless.status_code == 403 and "River" not in keyless.text


LEGACY_READER = "legacy-reader-secret-0123456789a"   # a 1.9.0 principal with memory:read, bound to contact-a
LEGACY_WRITER = "legacy-writer-secret-0123456789a"   # a 1.9.0 principal with the unscoped API but not memory:read


@pytest.mark.asyncio
async def test_a_credential_carried_from_1_9_0_lists_only_with_memory_read(state):
    """A credential an upgrade carried from 1.9.0 keeps its scopes, and 1.9.0's rules gave every memory read to
    ``memory:read``: the inventory and a claim's history need it, and the unscoped API (``api:access``) is not it."""
    await say(state, "a-1", "My office is in River.")
    river = id_of(state, "River")
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(host.router)
    app.include_router(inventory_module().router)
    bound = {"owner": [], "contacts": ["contact-a"], "viewer": "contact-a"}
    app.add_middleware(ApiKeyMiddleware, api_key=KEY, credentials=CredentialTable.from_config({
        "api-key": {"owner": ["whatsapp"], "contacts": "all"},
        "legacy-reader": {"sha256": secret_digest(LEGACY_READER), **bound, "scopes": ["memory:read"]},
        "legacy-writer": {"sha256": secret_digest(LEGACY_WRITER), **bound, "scopes": ["api:access", "turns:write"]}}))

    allowed = await listing(app, "contact-a", key=LEGACY_READER)
    assert allowed.status_code == 200, allowed.text
    assert [card["value"] for card in allowed.json()["claims"]] == ["River"]
    walked = await get(app, f"{INVENTORY}/claims/{river}/history", key=LEGACY_READER,
                       params={"person_id": "contact-a"})
    assert walked.status_code == 200, walked.text

    refused = await listing(app, "contact-a", key=LEGACY_WRITER)
    assert refused.status_code == 403 and "River" not in refused.text
    assert "memory:read" in refused.text
    unwalked = await get(app, f"{INVENTORY}/claims/{river}/history", key=LEGACY_WRITER,
                         params={"person_id": "contact-a"})
    assert unwalked.status_code == 403 and "River" not in unwalked.text


def test_the_routes_are_served_and_are_memory_reads(monkeypatch):
    monkeypatch.setenv("COLONYMIND_API_KEY", "test-secret-key")
    from colonymind.api.credentials import route_operation
    from colonymind.server import create_app
    paths = create_app().openapi()["paths"]
    assert "get" in paths[INVENTORY]
    assert "get" in paths[INVENTORY + "/claims/{claim_id}/history"]
    # A read of memory, like search: a credential bound to a person reaches it with its default access.
    assert route_operation("GET", INVENTORY) == "read"
    assert route_operation("GET", INVENTORY + "/claims/claim:abc/history") == "read"
    assert route_operation("POST", INVENTORY) == "manage"


# --- 2. erasure --------------------------------------------------------------------------------------------------


# --- 3. states -----------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_each_card_carries_its_state_and_machine_judged_claims_are_unverified(state):
    await say(state, "o-1", "From 2026-03-01, my office is in River.")
    await say(state, "o-2", "Starting 2026-03-12, my office is now in Lake.", at="2026-03-02T09:00:00+00:00")
    await say(state, "o-3", "Correction: my office is in Hill, not Lake.", at="2026-03-03T09:00:00+00:00")
    await say(state, "w-1", "Nora says the workshop starts at nine.", at="2026-03-04T09:00:00+00:00")
    await say(state, "w-2", "Owen says the workshop starts at ten.", at="2026-03-05T09:00:00+00:00")
    await say(state, "g-1", "My gym pass works until 2026-03-05.", at="2026-03-01T10:00:00+00:00")
    response = await listing(make_app(), "contact-a")
    assert response.status_code == 200, response.text
    cards = {card["value"]: card for card in response.json()["claims"]}
    assert {value: card["state"] for value, card in cards.items()} == {
        "River": "superseded", "Lake": "retracted", "Hill": "current",
        "nine": "in_conflict", "ten": "in_conflict", "works": "expired"}
    assert cards["River"]["superseded_by"] == cards["Lake"]["claim_id"]
    assert cards["Lake"]["retracted_by"] == cards["Hill"]["claim_id"]
    assert cards["Hill"]["quote"] == "Correction: my office is in Hill, not Lake."
    # Every claim here was formed and reviewed by a model: none is verified.
    assert {card["verification"] for card in cards.values()} == {"unverified"}
    # Newest first, in the order the claims were formed.
    assert [card["value"] for card in response.json()["claims"]] == ["works", "ten", "nine", "Hill", "Lake", "River"]


@pytest.mark.asyncio
async def test_a_change_from_a_later_date_takes_effect_on_that_date(state):
    """A change dated ahead supersedes its prior only from that date: until then the prior is what recall reads
    (and it can be in conflict), and the change has not begun. States follow the dates, as recall does."""
    await say(state, "o-1", "My office is in River.")
    await say(state, "o-2", "Starting 2026-10-01, my office moves to Lake.", at="2026-09-02T09:00:00+00:00")
    await say(state, "o-3", "Nora thinks my office might be in Hill.", at="2026-09-03T09:00:00+00:00")
    river, lake, hill = id_of(state, "River"), id_of(state, "Lake"), id_of(state, "Hill")
    from datetime import datetime, timezone
    from colonymind.memory import inventory
    ledger = ledger_of(state)

    def states(now):
        body = inventory.page(ledger, "contact-a", now=now)
        return {card["claim_id"]: card["state"] for card in body["claims"]}

    def recalled(now):
        packet, = prepared(SourceClaimProjection(ledger), now=now.isoformat())
        return packet["status"], {row["value"] for row in packet["assertions"]}

    september = datetime(2026, 9, 15, 12, tzinfo=timezone.utc)
    assert recalled(september) == ("unresolved_conflict", {"River", "Hill"})
    assert states(september) == {river: "in_conflict", hill: "in_conflict", lake: "upcoming"}
    october = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
    assert recalled(october) == ("unresolved_conflict", {"Lake", "Hill"})
    assert states(october) == {river: "superseded", hill: "in_conflict", lake: "in_conflict"}


@pytest.mark.asyncio
async def test_many_repeats_of_one_value_never_hide_a_different_one(state, monkeypatch):
    """In a store that does not fold restatements (the learning rules off, or one upgraded before them), one value
    said many times is many rows. Recall compares distinct values, so an older different value is still beside
    them: every card of both values is in conflict, however many rows repeat one of them."""
    from colonymind.memory import inventory, learning
    monkeypatch.setenv(learning.FLAG, "0")
    await say(state, "o-0", "Nora thinks my office might be in Hill.")
    repeats = 257
    for n in range(repeats):
        await say(state, f"o-{n + 1}", "My office is in River.", at=f"2026-03-02T09:{n // 60:02d}:{n % 60:02d}+00:00")
    ledger = ledger_of(state)
    packet, = prepared(SourceClaimProjection(ledger))
    assert packet["status"] == "unresolved_conflict"
    found, token = [], None
    while True:
        body = inventory.page(ledger, "contact-a", token=token, limit=inventory.PAGE_MAX)
        found += body["claims"]
        token = body["next_page"]
        if not token:
            break
    assert len(found) == repeats + 1
    assert {(card["value"], card["state"]) for card in found} == {("River", "in_conflict"), ("Hill", "in_conflict")}


@pytest.mark.asyncio
async def test_a_claim_the_owner_kept_in_an_answer_is_labelled_owner_confirmed(state):
    await say(state, "w-1", "Nora says the workshop starts at nine.")
    await say(state, "w-2", "Owen says the workshop starts at ten.", at="2026-03-02T09:00:00+00:00")
    nine, ten = id_of(state, "nine"), id_of(state, "ten")
    from colonymind.memory import learning
    # What the owner's answer "the first" does to a contradiction question's two statements.
    with closing(ledger_of(state)._connect()) as conn, conn:
        conn.execute("UPDATE source_claims SET retracted_by=? WHERE id=?", ("intention:7", ten))
        learning.log(conn, store="claim", action="declined", stage="owner_answer",
                     reason="the owner answered first", item_id=ten, contact_id="contact-a", refs=["intention:7"])
        learning.log(conn, store="claim", action="ratified", stage="owner_answer",
                     reason="the owner answered first", item_id=nine, contact_id="contact-a", refs=["intention:7"])
    cards = {card["value"]: card for card in (await listing(make_app(), "contact-a")).json()["claims"]}
    assert (cards["nine"]["state"], cards["nine"]["verification"]) == ("current", "owner_confirmed")
    assert (cards["ten"]["state"], cards["ten"]["retracted_by"]) == ("retracted", "intention:7")


@pytest.mark.asyncio
async def test_a_sealed_value_never_reaches_the_inventory(state):
    """A message is sealed before it is stored, and a claim quoting a sealed reference is refused at formation:
    the inventory, which reads claims as stored and resolves nothing, has nothing of it to show."""
    await say(state, "n-1", f"Remember the wifi password is {WIFI} at home.")
    response = await listing(make_app(), "contact-a")
    assert response.status_code == 200
    assert response.json()["claims"] == []
    assert WIFI not in response.text


# --- 4. paging ------------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pages_are_stable_under_concurrent_writes(state):
    for n in range(7):
        await say(state, f"h-{n}", f"My hobby number {n} is fun.", at=f"2026-03-0{n + 1}T09:00:00+00:00")
    app = make_app()
    first = await listing(app, "contact-a", limit=3)
    assert first.status_code == 200, first.text
    body = first.json()
    assert len(body["claims"]) == 3 and body["next_page"] and body["complete"] is False
    seen = [card["claim_id"] for card in body["claims"]]
    revision = body["read_revision"]

    # Writes between pages: new claims, and a retraction of one the next page lists.
    await say(state, "h-new-1", "My hobby number 8 is fun.", at="2026-03-10T09:00:00+00:00")
    await say(state, "h-new-2", "My hobby number 9 is fun.", at="2026-03-11T09:00:00+00:00")
    later = id_of(state, "hobby number 2")
    with closing(ledger_of(state)._connect()) as conn, conn:
        conn.execute("UPDATE source_claims SET retracted_by=? WHERE id=?", ("intention:9", later))

    pages, token = [body], body["next_page"]
    while token:
        page = await listing(app, "contact-a", limit=3, page=token)
        assert page.status_code == 200, page.text
        assert page.json()["read_revision"] == revision
        pages.append(page.json())
        seen += [card["claim_id"] for card in page.json()["claims"]]
        token = page.json()["next_page"]
    assert [len(page["claims"]) for page in pages] == [3, 3, 1]
    assert pages[-1]["complete"] is True
    original = {ident for ident, value in claim_ids(state).items() if value not in {"hobby number 8", "hobby number 9"}}
    assert len(seen) == len(set(seen)) == 7 and set(seen) == original
    states = {card["claim_id"]: card["state"] for page in pages for card in page["claims"]}
    assert states[later] == "retracted"          # each page shows the state as it is when it is read

    fresh = await listing(app, "contact-a", limit=50)
    assert len(fresh.json()["claims"]) == 9


@pytest.mark.asyncio
async def test_a_source_moved_to_another_person_between_pages_restarts_the_listing(state, monkeypatch):
    """A claim said twice is one claim with a witness, anchored at the newest. When its anchor message is moved to
    another person, the witness still in this person's scope becomes its anchor: the claim stays, with no removed
    claim and no invalidated reply, but a page already read showed it quoting a message now another person's."""
    from colonymind.memory import learning
    monkeypatch.setenv(learning.FLAG, "1")
    await say(state, "h-0", "My hobby number 0 is fun.")
    await say(state, "o-1", "My office is in River.", at="2026-03-02T09:00:00+00:00")
    await say(state, "o-2", "My office is in River.", at="2026-03-03T09:00:00+00:00")
    river = id_of(state, "River")
    with closing(ledger_of(state)._connect()) as conn:
        assert conn.execute("SELECT count(*) FROM source_claim_witnesses WHERE claim_id=?", (river,)).fetchone()[0]
    app = make_app()
    first = (await listing(app, "contact-a", limit=1)).json()
    assert [(card["claim_id"], card["source"]["source_id"]) for card in first["claims"]] == [(river, "o-2")]

    from colonymind.turns.source_attribution import correct
    correct(ledger_of(state), operation_id="moved-1", performed_by="fixture-admin", old_contact_id="contact-a",
            contact_id="contact-b", source_ids=["o-2"], evidence_refs=["reviewed-fixture"])
    fresh = (await listing(app, "contact-a", limit=1)).json()
    assert [(card["claim_id"], card["source"]["source_id"]) for card in fresh["claims"]] == [(river, "o-1")]
    stale = await listing(app, "contact-a", limit=1, page=first["next_page"])
    assert stale.status_code == 409 and stale.json()["detail"]["code"] == "inventory_changed_restart"


# --- 5. the source revision annotate and forget act on -----------------------------------------------------------

@pytest.mark.asyncio
async def test_every_card_carries_a_source_version_the_owners_annotate_api_accepts(state):
    await say(state, "a-1", "My office is in River.", session="s-first")
    await say(state, "a-2", "My tea is green.", at="2026-03-02T09:00:00+00:00", session="s-second")
    app = make_app()
    cards = (await listing(app, "contact-a")).json()["claims"]
    assert {card["source"]["source_id"] for card in cards} == {"a-1", "a-2"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test",
                           headers={"Authorization": "Bearer " + KEY}) as client:
        for number, card in enumerate(cards):
            source = card["source"]
            response = await client.post("/v1/host/memory/sources/annotations", json={
                "contact_id": "contact-a", "session_id": source["session_id"], "annotation_id": f"review-{number}",
                "source_id": source["source_id"], "source_version": source["source_version"],
                "excerpt": card["quote"], "correction": "Reported, not checked."})
            assert response.status_code == 200, response.text
            assert response.json()["created"] is True
            assert response.json()["target"] == {"source_id": source["source_id"],
                                                 "source_version": source["source_version"]}
    again = {card["claim_id"]: card for card in (await listing(app, "contact-a")).json()["claims"]}
    assert {card["corrections"] for card in again.values()} == {1}
    assert {card["claim_id"]: card["source"] for card in cards} == {
        ident: card["source"] for ident, card in again.items()}


@pytest.mark.asyncio
async def test_a_card_listed_by_a_credential_that_may_not_annotate_is_still_refused_by_annotate(state):
    """Listing never widens annotation. The native annotate tool left with 1.10.0's thin adapter, and no native tool
    takes a source version: the annotate route is the one way to annotate, and it takes the owner's side of the API
    (or a 1.9.0 credential's ``memory:write``). A credential that may list a person's cards is refused there with the
    exact source version its own listing gave it, and nothing is recorded."""
    await say(state, "a-1", "My office is in River.")
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(host.router)
    app.include_router(inventory_module().router)
    bound = {"owner": [], "contacts": ["contact-a"], "viewer": "contact-a"}
    app.add_middleware(ApiKeyMiddleware, api_key=KEY, credentials=CredentialTable.from_config({
        "api-key": {"owner": ["whatsapp"], "contacts": "all"},
        "guest-a": {"sha256": secret_digest(GUEST), **bound},
        "legacy-reader": {"sha256": secret_digest(LEGACY_READER), **bound, "scopes": ["memory:read"]}}))
    before = dump(Path(state) / "turn-idempotency.db")
    for key in (GUEST, LEGACY_READER):
        listed = await listing(app, "contact-a", key=key)
        assert listed.status_code == 200, listed.text
        card = listed.json()["claims"][0]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test",
                               headers={"Authorization": "Bearer " + key}) as client:
            response = await client.post("/v1/host/memory/sources/annotations", json={
                "contact_id": "contact-a", "session_id": card["source"]["session_id"], "annotation_id": "listed-1",
                "source_id": card["source"]["source_id"], "source_version": card["source"]["source_version"],
                "excerpt": card["quote"], "correction": "Listed, so annotated."})
        assert response.status_code == 403, response.text
    assert dump(Path(state) / "turn-idempotency.db") == before


@pytest.mark.asyncio
async def test_listing_writes_nothing_so_no_card_becomes_a_supplied_source(state):
    """The inventory never widens what a turn was supplied: it records no read, no receipt and no source, so a
    listed card is not in any turn's recall snapshot."""
    await say(state, "a-1", "My office is in River.")
    await say(state, "a-2", "My tea is green.", at="2026-03-02T09:00:00+00:00")
    app = make_app()
    before = dump(Path(state) / "turn-idempotency.db")
    cards = (await listing(app, "contact-a", limit=1)).json()
    await listing(app, "contact-a", limit=1, page=cards["next_page"])
    await get(app, f"{INVENTORY}/claims/{cards['claims'][0]['claim_id']}/history",
              params={"person_id": "contact-a"})
    assert dump(Path(state) / "turn-idempotency.db") == before


# --- 6. a claim's own history ---------------------------------------------------------------------------------------

async def history(app, claim_id, *, key=KEY, person=None):
    params = {"person_id": person} if person else None
    return await get(app, f"{INVENTORY}/claims/{claim_id}/history", key=key, params=params)


def links(body):
    """The replacements in a history, in order: (event, the claim it ended, the claim that ended it)."""
    return [(step["event"], step["claim_id"], step["by"].get("claim_id")) for step in body["steps"]
            if step["event"] in ("superseded", "retracted")]


async def office_chain(state, *, correction="Correction: my office is in Hill."):
    await say(state, "o-1", "My office is in River.")
    await say(state, "o-2", "Starting 2026-03-12, my office is now in Lake.", at="2026-03-02T09:00:00+00:00")
    await say(state, "o-3", correction, at="2026-03-03T09:00:00+00:00")
    return id_of(state, "River"), id_of(state, "Lake"), id_of(state, "Hill")


@pytest.mark.asyncio
async def test_a_superseded_then_retracted_chain_lists_both_steps_in_order(state):
    river, lake, hill = await office_chain(state)
    app = make_app()
    for start in (river, lake, hill):
        response = await history(app, start, person="contact-a")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["claim_id"] == start and body["complete"] is True
        assert body["steps"][0]["event"] == "stated" and body["steps"][0]["claim"]["value"] == "River"
        assert links(body) == [("superseded", river, lake), ("retracted", lake, hill)]
        replaced = [step for step in body["steps"] if step["event"] in ("superseded", "retracted")]
        # Each step carries the source reference of the statement that made it.
        assert [step["by"]["source"]["source_id"] for step in replaced] == ["o-2", "o-3"]
        assert [step["by"]["state"] for step in replaced] == ["retracted", "current"]
    # The owner may name no person: the claim's own person is the one checked.
    unnamed = await history(app, river)
    assert unnamed.status_code == 200 and links(unnamed.json()) == links(body)


@pytest.mark.asyncio
async def test_a_history_is_the_whole_chain_from_any_of_its_claims(state, monkeypatch):
    """A correction retracts the value, not one row: a store that did not fold restatements holds the value twice,
    and both rows are retracted by the one correction. The chain then joins two claims into one, and a history
    from either of them holds both, as the history from the correction does."""
    from colonymind.memory import learning
    monkeypatch.setenv(learning.FLAG, "0")
    await say(state, "o-1", "My office is in River.")
    await say(state, "o-2", "My office is in River.", at="2026-03-02T09:00:00+00:00")
    first, second = sorted(ident for ident, value in claim_ids(state).items() if value == "River")
    monkeypatch.setenv(learning.FLAG, "1")
    await say(state, "o-3", "Correction: my office was never in River, it is in Hill.", at="2026-03-03T09:00:00+00:00")
    hill = id_of(state, "Hill")
    with closing(ledger_of(state)._connect()) as conn:
        assert {row[0]: row[1] for row in conn.execute(
            "SELECT id, retracted_by FROM source_claims WHERE id IN (?,?)", (first, second))} == {
            first: hill, second: hill}
    app = make_app()
    walks = {}
    for start in (first, second, hill):
        response = await history(app, start, person="contact-a")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["complete"] is True
        walks[start] = body["steps"]
        assert sorted(links(body)) == sorted([("retracted", first, hill), ("retracted", second, hill)])
        assert [step["event"] for step in body["steps"]].count("stated") == 2
    assert walks[first] == walks[second] == walks[hill]
    # The walk's one bound covers both ways: a chain it cannot read whole says so.
    from colonymind.memory import inventory
    monkeypatch.setattr(inventory, "HISTORY_LIMIT", 2)
    for start in (first, second, hill):
        cut = (await history(app, start, person="contact-a")).json()
        assert cut["complete"] is False and len(links(cut)) == 1
        assert cut["limit_reached"] is True and cut["gaps"] == []


@pytest.mark.asyncio
async def test_a_repeat_that_moves_the_anchor_keeps_the_corrections_of_the_earlier_statement(state, monkeypatch):
    """With the learning rules a repeat folds into the live claim and becomes its anchor; the statement it repeats
    stays a witness, and recall still reads the claim through it with the correction recorded against it. The card
    keeps counting that correction and the history keeps showing its words."""
    from colonymind.memory import learning
    monkeypatch.setenv(learning.FLAG, "1")
    await say(state, "o-1", "My office is in River.")
    app = make_app()
    card = (await listing(app, "contact-a")).json()["claims"][0]
    note = ledger_of(state).append_source_annotation(
        contact_id="contact-a", session_id="review", annotation_id="review-1",
        source_id=card["source"]["source_id"], source_version=card["source"]["source_version"],
        excerpt="office is in River", correction="That was the old lease.", author_principal="api-key")
    assert (await listing(app, "contact-a")).json()["claims"][0]["corrections"] == 1
    await say(state, "o-2", "My office is in River.", at="2026-03-02T09:00:00+00:00")
    [folded] = (await listing(app, "contact-a")).json()["claims"]
    assert (folded["claim_id"], folded["source"]["source_id"], folded["witnesses"]) == (card["claim_id"], "o-2", 2)
    assert folded["corrections"] == 1
    body = (await history(app, folded["claim_id"], person="contact-a")).json()
    assert [step["event"] for step in body["steps"]] == ["stated", "corrected"]
    assert body["steps"][1]["by"]["correction"] == "That was the old lease."
    assert body["steps"][1]["by"]["source"]["source_id"] == note["source_id"]


@pytest.mark.asyncio
async def test_the_owners_answers_outlast_the_nightly_bound_on_the_learning_log(state):
    """The learning log is audit and keeps only its newest entries (``learning.prune``, every night), but an
    owner's answer about a claim memory still holds is its only record that the owner kept or dropped it: the card
    stays ``owner_confirmed`` and the history keeps both answers after the log has moved on."""
    await say(state, "w-1", "Nora says the workshop starts at nine.")
    await say(state, "w-2", "Owen says the workshop starts at ten.", at="2026-03-02T09:00:00+00:00")
    nine, ten = id_of(state, "nine"), id_of(state, "ten")
    from colonymind.memory import learning
    with closing(ledger_of(state)._connect()) as conn, conn:
        conn.execute("UPDATE source_claims SET retracted_by=? WHERE id=?", ("intention:7", ten))
        learning.log(conn, store="claim", action="declined", stage="owner_answer",
                     reason="the owner answered first", item_id=ten, contact_id="contact-a", refs=["intention:7"])
        learning.log(conn, store="claim", action="ratified", stage="owner_answer",
                     reason="the owner answered first", item_id=nine, contact_id="contact-a", refs=["intention:7"])
        # An answer about a claim memory no longer holds is audit like the rest.
        learning.log(conn, store="claim", action="ratified", stage="owner_answer",
                     reason="the owner answered first", item_id="claim:" + "e" * 64, contact_id="contact-a",
                     refs=["intention:3"])
        for n in range(12):
            learning.log(conn, store="claim", action="rejected", reason=f"filler {n}")
        assert learning.prune(conn, keep=10) >= 3
        left = {row[0] for row in conn.execute("SELECT item_id FROM learning_log WHERE stage='owner_answer'")}
    assert left == {nine, ten}
    app = make_app()
    cards = {card["value"]: card for card in (await listing(app, "contact-a")).json()["claims"]}
    assert (cards["nine"]["state"], cards["nine"]["verification"]) == ("current", "owner_confirmed")
    kept = (await history(app, nine, person="contact-a")).json()
    assert [step["event"] for step in kept["steps"]] == ["stated", "kept_by_owner"]
    dropped = (await history(app, ten, person="contact-a")).json()
    assert dropped["steps"][-1]["event"] == "retracted"
    assert dropped["steps"][-1]["by"] == {"ref": "intention:7", "owner_answer": True}


MISLINKED = "Correction: I prefer quiet rooms."     # a cue, but nothing of the preference it names


async def held_preference(state, monkeypatch):
    """A quoted preference and a correction of it that its words do not name: the change is held for the owner
    (``learning_pending``), and both statements stand until the owner decides."""
    from colonymind.memory import learning
    monkeypatch.setenv(learning.FLAG, "1")
    projection = SourceClaimProjection(ledger_of(state))
    await add_preference(projection, "original", PREFERENCE, preference())
    original = id_of(state, PREFERENCE, contact="owner")
    await add_preference(projection, "changed", MISLINKED,
                         preference(MISLINKED, operation="correct", prior_claim_id=original))
    with closing(ledger_of(state)._connect()) as conn:
        [held] = learning.pending(conn, contact_id="owner")
    return projection, original, id_of(state, MISLINKED, contact="owner"), held["id"]


@pytest.mark.asyncio
async def test_a_held_preference_change_is_the_same_history_from_either_statement(state, monkeypatch):
    projection, original, changed, held = await held_preference(state, monkeypatch)
    app = make_app()
    for status in ("pending", "declined"):
        if status == "declined":
            projection.decide(held, ratify=False, reason="both hold")
        walks = [(await history(app, start, person="owner")).json() for start in (original, changed)]
        assert walks[0]["steps"] == walks[1]["steps"] and walks[1]["complete"] is True
        steps = walks[1]["steps"]
        assert [(step["event"], step["claim_id"]) for step in steps] == [
            ("stated", original), ("preference_change", original), ("stated", changed)]
        assert (steps[1]["status"], steps[1]["by"]["claim_id"]) == (status, changed)
        assert not links(walks[1])


@pytest.mark.asyncio
async def test_a_history_is_scoped_like_the_inventory(state):
    await say(state, "a-1", "My office is in River.")
    await say(state, "b-1", "My tea is green.", contact="contact-b")
    river, green = id_of(state, "River"), id_of(state, "green", contact="contact-b")
    app = make_app()
    own = await history(app, river, key=GUEST)
    assert own.status_code == 200 and own.json()["steps"][0]["claim"]["value"] == "River"
    # Another person's claim is not found for a guest, whether it names that person or not.
    unnamed = await history(app, green, key=GUEST)
    assert unnamed.status_code == 404 and "green" not in unnamed.text
    named = await history(app, green, key=GUEST, person="contact-b")
    assert named.status_code == 403 and "green" not in named.text
    wrong_person = await history(app, green, person="contact-a")
    assert wrong_person.status_code == 404
    relay = await history(app, river, key=RELAY)
    assert relay.status_code == 404 and "River" not in relay.text
    assert (await history(app, "claim:" + "0" * 64)).status_code == 404
    keyless = await history(make_app(api_key=None), river, key=None, person="contact-a")
    assert keyless.status_code == 403 and "River" not in keyless.text


from colonymind.memory import curated
@pytest.mark.asyncio
async def test_erased_claims_are_absent(state):
    await say(state, 'a-1', 'My office is in River.')
    await say(state, 'a-2', 'My tea is green.', at='2026-03-02T09:00:00+00:00')
    app = make_app()
    assert {card['value'] for card in (await listing(app, 'contact-a')).json()['claims']} == {'River', 'green'}
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger_of(state))
    curated.forget_sources(ledger_of(state), 'contact-a', source_ids=['a-1'])
    assert historical_bytes(ledger_of(state)) == _native_before_withdrawal
    after = await listing(app, 'contact-a')
    assert after.status_code == 200
    assert [card['value'] for card in after.json()['claims']] == ['green']
    assert 'River' not in after.text and 'a-1' not in after.text

@pytest.mark.asyncio
async def test_an_erasure_between_pages_restarts_the_listing(state):
    for n in range(4):
        await say(state, f'h-{n}', f'My hobby number {n} is fun.', at=f'2026-03-0{n + 1}T09:00:00+00:00')
    app = make_app()
    first = (await listing(app, 'contact-a', limit=2)).json()
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger_of(state))
    curated.forget_sources(ledger_of(state), 'contact-a', source_ids=['h-0'])
    assert historical_bytes(ledger_of(state)) == _native_before_withdrawal
    stale = await listing(app, 'contact-a', limit=2, page=first['next_page'])
    assert stale.status_code == 409 and stale.json()['detail']['code'] == 'inventory_changed_restart'
    assert 'hobby' not in stale.text
    await say(state, 'b-1', 'My tea is green.', contact='contact-b')
    fresh = (await listing(app, 'contact-a', limit=2)).json()
    crossed = await listing(app, 'contact-b', limit=2, page=fresh['next_page'])
    assert crossed.status_code == 409
    garbled = await listing(app, 'contact-a', page='not-a-page')
    assert garbled.status_code == 422 and garbled.json()['detail']['code'] == 'invalid_page'
    too_many = await listing(app, 'contact-a', limit=100000)
    assert too_many.status_code == 422

@pytest.mark.asyncio
async def test_an_erased_step_shows_a_marker_and_never_its_words(state):
    river, lake, hill = await office_chain(state)
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger_of(state))
    curated.forget_sources(ledger_of(state), 'contact-a', source_ids=['o-2'])
    assert historical_bytes(ledger_of(state)) == _native_before_withdrawal
    app = make_app()
    for start in (river, hill):
        response = await history(app, start, person='contact-a')
        assert response.status_code == 200, response.text
        assert links(response.json()) == [('superseded', river, lake), ('retracted', lake, hill)]
        assert (response.json()['complete'], response.json()['gaps']) == (True, [])
        assert 'Lake' not in response.text and 'Starting' not in response.text
        marker = [step['by'] for step in response.json()['steps'] if step['event'] == 'superseded'][0]
        assert marker == {'claim_id': lake, 'erased': True}
    assert (await history(app, lake, person='contact-a')).status_code == 404
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger_of(state))
    curated.forget_sources(ledger_of(state), 'contact-a', source_ids=['o-3'])
    assert historical_bytes(ledger_of(state)) == _native_before_withdrawal
    last = await history(app, river, person='contact-a')
    assert 'Hill' not in last.text and 'Lake' not in last.text
    assert [step['event'] for step in last.json()['steps']][:2] == ['stated', 'superseded']
    assert last.json()['steps'][0]['claim']['state'] == 'superseded'
    assert last.json()['complete'] is False and last.json()['limit_reached'] is False
    assert last.json()['gaps'] == [{'claim_id': lake, 'side': 'after'}]

@pytest.mark.asyncio
async def test_a_chain_broken_by_erasures_says_it_is_incomplete_on_both_sides(state, monkeypatch):
    """In River, Lake, Hill, Pier (each replacing the one before), erasing Lake and Hill leaves no retained claim
    that joins them: River names Lake as what superseded it, Pier names Hill as what it corrected, and nothing names
    what came after Lake or before Hill. Each side shows what it can establish and says the chain may go on; the
    two are never joined, and an erased claim is never taken for the chain's first or last statement."""
    monkeypatch.setitem(MODEL.outputs, 'office is at Pier', [{'value': 'Pier', 'operation': 'correct', 'choose': 'Hill'}])
    river, lake, hill = await office_chain(state)
    await say(state, 'o-4', 'Correction: my office is at Pier.', at='2026-03-04T09:00:00+00:00')
    pier = id_of(state, 'Pier')
    app = make_app()
    whole = [('superseded', river, lake), ('retracted', lake, hill), ('retracted', hill, pier)]
    for start in (river, lake, hill, pier):
        body = (await history(app, start, person='contact-a')).json()
        assert (links(body), body['complete'], body['gaps']) == (whole, True, [])
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger_of(state))
    curated.forget_sources(ledger_of(state), 'contact-a', source_ids=['o-2', 'o-3'])
    assert historical_bytes(ledger_of(state)) == _native_before_withdrawal
    first = await history(app, river, person='contact-a')
    assert links(first.json()) == [('superseded', river, lake)]
    assert (first.json()['complete'], first.json()['gaps']) == (False, [{'claim_id': lake, 'side': 'after'}])
    last = await history(app, pier, person='contact-a')
    assert links(last.json()) == [('retracted', hill, pier)]
    assert [step['event'] for step in last.json()['steps']] == ['stated', 'retracted']
    assert last.json()['steps'][0] == {'event': 'stated', 'claim_id': hill, 'erased': True}
    assert (last.json()['complete'], last.json()['gaps']) == (False, [{'claim_id': hill, 'side': 'before'}])
    for response in (first, last):
        assert 'Lake' not in response.text and 'Hill' not in response.text
        assert response.json()['limit_reached'] is False

@pytest.mark.asyncio
async def test_the_owners_answers_and_corrections_are_steps_of_the_history(state):
    await say(state, 'w-1', 'Nora says the workshop starts at nine.')
    await say(state, 'w-2', 'Owen says the workshop starts at ten.', at='2026-03-02T09:00:00+00:00')
    nine, ten = (id_of(state, 'nine'), id_of(state, 'ten'))
    app = make_app()
    card = next((card for card in (await listing(app, 'contact-a')).json()['claims'] if card['claim_id'] == ten))
    note = ledger_of(state).append_source_annotation(contact_id='contact-a', session_id='review', annotation_id='review-1', source_id=card['source']['source_id'], source_version=card['source']['source_version'], excerpt='starts at ten', correction="Owen was reading last year's leaflet.", author_principal='api-key')
    from colonymind.memory import learning
    with closing(ledger_of(state)._connect()) as conn, conn:
        conn.execute('UPDATE source_claims SET retracted_by=? WHERE id=?', ('intention:7', ten))
        learning.log(conn, store='claim', action='declined', stage='owner_answer', reason='the owner answered first', item_id=ten, contact_id='contact-a', refs=['intention:7'])
        learning.log(conn, store='claim', action='ratified', stage='owner_answer', reason='the owner answered first', item_id=nine, contact_id='contact-a', refs=['intention:7'])
    body = (await history(app, ten, person='contact-a')).json()
    assert [step['event'] for step in body['steps']] == ['stated', 'corrected', 'retracted']
    corrected, retracted = (body['steps'][1], body['steps'][2])
    assert corrected['by']['correction'] == "Owen was reading last year's leaflet."
    assert corrected['by']['source']['source_id'] == note['source_id']
    assert retracted['by']['ref'] == 'intention:7' and retracted['by']['owner_answer'] is True
    kept = (await history(app, nine, person='contact-a')).json()
    assert [step['event'] for step in kept['steps']] == ['stated', 'kept_by_owner']
    assert kept['steps'][1]['by']['ref'] == 'intention:7'
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger_of(state))
    curated.forget_sources(ledger_of(state), 'contact-a', source_ids=[note['source_id']])
    assert historical_bytes(ledger_of(state)) == _native_before_withdrawal
    after = await history(app, ten, person='contact-a')
    assert 'leaflet' not in after.text
    assert after.json()['steps'][1] == {'event': 'corrected', 'claim_id': ten, 'corrects': 'w-2', 'by': {'source_id': note['source_id'], 'erased': True}}

@pytest.mark.asyncio
async def test_a_correction_recorded_against_a_correction_is_a_step_of_the_history(state):
    """Recall expands a correction as it does any source, so a correction recorded against a correction (one that
    withdraws it, say) comes with the claim too. The history shows both, each naming what it was recorded against,
    and the card counts both; an erased correction is a marker, and nothing recorded against it is read."""
    await say(state, 'w-1', 'Nora says the workshop starts at nine.')
    await say(state, 'w-2', 'Owen says the workshop starts at ten.', at='2026-03-02T09:00:00+00:00')
    ten = id_of(state, 'ten')
    app = make_app()
    card = next((card for card in (await listing(app, 'contact-a')).json()['claims'] if card['claim_id'] == ten))
    first = ledger_of(state).append_source_annotation(contact_id='contact-a', session_id='review', annotation_id='review-1', source_id=card['source']['source_id'], source_version=card['source']['source_version'], excerpt='starts at ten', correction="Owen was reading last year's leaflet.", author_principal='api-key')
    second = ledger_of(state).append_source_annotation(contact_id='contact-a', session_id='review', annotation_id='review-2', source_id=first['source_id'], source_version=first['source_version'], excerpt="last year's leaflet", correction="Withdrawn: the leaflet is this year's.", author_principal='api-key')
    body = (await history(app, ten, person='contact-a')).json()
    corrected = [step for step in body['steps'] if step['event'] == 'corrected']
    assert [(step['corrects'], step['by']['source']['source_id']) for step in corrected] == [('w-2', first['source_id']), (first['source_id'], second['source_id'])]
    assert corrected[1]['by']['correction'] == "Withdrawn: the leaflet is this year's."
    cards = {card['claim_id']: card for card in (await listing(app, 'contact-a')).json()['claims']}
    assert cards[ten]['corrections'] == 2
    from colonymind.turns.source_annotations import expand
    [recalled] = expand(ledger_of(state), [{'id': 'quote', 'kind': 'source_quote', 'source_turn_id': 'w-2', 'role': 'user', 'content': 'starts at ten'}], contact_id='contact-a', session_id='review')
    assert set(recalled['_annotation_ids']) == {step['by']['source']['source_id'] for step in corrected}
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger_of(state))
    curated.forget_sources(ledger_of(state), 'contact-a', source_ids=[first['source_id']])
    assert historical_bytes(ledger_of(state)) == _native_before_withdrawal
    after = await history(app, ten, person='contact-a')
    assert 'leaflet' not in after.text and 'Withdrawn' not in after.text
    assert [step for step in after.json()['steps'] if step['event'] == 'corrected'] == [{'event': 'corrected', 'claim_id': ten, 'corrects': 'w-2', 'by': {'source_id': first['source_id'], 'erased': True}}]
    assert {card['claim_id']: card for card in (await listing(app, 'contact-a')).json()['claims']}[ten]['corrections'] == 0

@pytest.mark.asyncio
@pytest.mark.parametrize('decision', ['pending', 'declined', 'ratified'])
async def test_an_erasure_never_turns_a_change_that_did_not_take_effect_into_one_that_did(state, monkeypatch, decision):
    """Once the earlier statement is erased, the proposing statement's pointer to it is all that is left, and a
    pointer only names what was proposed. A held change the owner did not ratify retired nothing, so its history
    shows the proposal as it stands and no retraction; a ratified one did retire it."""
    projection, original, changed, held = await held_preference(state, monkeypatch)
    if decision != 'pending':
        projection.decide(held, ratify=decision == 'ratified', reason="the owner's answer")
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger_of(state))
    curated.forget_sources(ledger_of(state), 'owner', source_ids=['original'])
    assert historical_bytes(ledger_of(state)) == _native_before_withdrawal
    body = (await history(make_app(), changed, person='owner')).json()
    assert links(body) == ([('retracted', original, changed)] if decision == 'ratified' else [])
    assert body['gaps'] == [{'claim_id': original, 'side': 'before'}] + ([] if decision == 'ratified' else [{'claim_id': original, 'side': 'after'}])
    [proposal] = [step for step in body['steps'] if step['event'] == 'preference_change']
    assert (proposal['claim_id'], proposal['status'], proposal['by']['claim_id']) == (original, decision, changed)
    assert PREFERENCE not in json.dumps(body)

@pytest.mark.asyncio
async def test_an_erasure_never_turns_an_undated_change_into_a_replacement(state, monkeypatch):
    """A change to a quoted preference with no date it can resolve replaces nothing: both statements stay live.
    Erasing the earlier one leaves the later one's pointer, which does not make it a replacement."""
    from colonymind.memory import learning
    monkeypatch.setenv(learning.FLAG, '0')
    projection = SourceClaimProjection(ledger_of(state))
    await add_preference(projection, 'original', PREFERENCE, preference())
    original = id_of(state, PREFERENCE, contact='owner')
    words = 'I now prefer a long handoff with a purchasing table.'
    await add_preference(projection, 'changed', words, preference(words, operation='change', prior_claim_id=original))
    changed = id_of(state, words, contact='owner')
    with closing(ledger_of(state)._connect()) as conn:
        assert tuple(conn.execute("SELECT json_extract(data_json,'$.operation'), valid_from FROM source_claims WHERE id=?", (changed,)).fetchone()) == ('change', None)
        assert tuple(conn.execute('SELECT superseded_by, retracted_by FROM source_claims WHERE id=?', (original,)).fetchone()) == (None, None)
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger_of(state))
    curated.forget_sources(ledger_of(state), 'owner', source_ids=['original'])
    assert historical_bytes(ledger_of(state)) == _native_before_withdrawal
    body = (await history(make_app(), changed, person='owner')).json()
    assert links(body) == [] and [step['claim_id'] for step in body['steps']] == [changed]
    assert (body['complete'], body['gaps']) == (True, [])
