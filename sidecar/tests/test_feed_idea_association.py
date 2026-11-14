"""Offline FEED19: exact evidence associations reuse live Ideas and owner authority."""
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from colonymind.api.credentials import CredentialTable, default_credentials, secret_digest
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import host
from colonymind.initiatives.store import InitiativeStore
from colonymind.mind.ideas import Ideas
from colonymind.mind.ideas_association import AssociationRequest, associate
from test_grounded_ideas import KEY, app, env, generate, mailbox  # noqa: F401

pytestmark = pytest.mark.asyncio
ROOT = "/v1/host/initiatives"
ROUTE = ROOT + "/ideas/associate"


@pytest.fixture
async def client(env):
    async with AsyncClient(transport=ASGITransport(app=app()), base_url="http://test",
                           headers={"Authorization": "Bearer " + KEY}) as value:
        yield value


def evidence(env, row, unit="feed-unit-1"):
    return {"unit_id": unit, "sources": [{k: s[k] for k in ("key", "revision")}
            for s in env.ideas.open(row.context["idea"]["private"])["sources"]]}


async def project(client, *cards):
    response = await client.post(ROUTE, json={"cards": list(cards)})
    assert response.status_code == 200, response.text
    assert response.json()["capabilities"] == {"feed_idea_association": True}
    return response.json()


def action(row, state="available"):
    return {"idea_id": row.id, "state": state}


async def test_evidence_link_detail_handoff_and_side_effect_free_reads(env, client, monkeypatch):
    row = (await generate(env, mailbox()[:1]))[0]
    card = evidence(env, row)
    calls = env.model.calls
    before = env.store.get(row.id), env.store.get_history(row.id), env.ideas.meta()
    # A display/result read would disclose more than an association needs.
    with monkeypatch.context() as patch:
        patch.setattr(env.ideas, "card", lambda *_: pytest.fail("association read display/result payload"))
        for _ in range(2):
            result = await project(client, card, {"unit_id": "empty", "sources": []})
            assert result == {"cards": [{"unit_id": card["unit_id"], "idea_action": action(row)}, {"unit_id": "empty"}],
                              "capabilities": {"feed_idea_association": True}, "ready": True}
        assert before == (env.store.get(row.id), env.store.get_history(row.id), env.ideas.meta())
        assert env.model.calls == calls and env.queue.calls == []
    detail = await client.get(f"{ROOT}/{row.id}?contract=2")
    assert detail.status_code == 200 and detail.json()["card"]["id"] == row.id


@pytest.mark.parametrize("terminal", ["done", "failed"])
async def test_execute_and_respond_handoffs_track_real_lifecycle(env, client, terminal):
    row = (await generate(env, mailbox()[:1]))[0]
    card = evidence(env, row)
    receipt = await client.post(f"{ROOT}/{row.id}/execute", json={"mode": "full"})
    assert receipt.status_code == 200, receipt.text
    response = await client.post(f"{ROOT}/{row.id}/respond", json={
        "action": "approved", "mode": "full", "origin": {"surface": "feed", "unit_id": card["unit_id"]}})
    assert response.status_code == 200 and response.json()["executionId"] == receipt.json()["executionId"]
    for state in ("accepted", "working", terminal):
        env.queue.state = state
        await env.ideas.reconcile()
        assert (await project(client, card))["cards"][0]["idea_action"] == action(row, state)
    assert env.ideas.catalog() == []
    assert len(env.queue.tasks) == 1 and len(env.queue.calls) == 1
    assert env.store.get(row.id).context["idea"]["execution"]["taskId"] == response.json()["taskId"]


async def test_exact_pairs_no_provider_or_mixed_revision_matching(env, client):
    row = (await generate(env, mailbox()[:1]))[0]
    original = evidence(env, row)
    key, revision = (original["sources"][0][k] for k in ("key", "revision"))
    bad_revision = "f" * 64 if revision != "f" * 64 else "e" * 64
    for sources in ([{"key": key, "revision": bad_revision}],
                    [{"key": "provider-id", "revision": revision}],
                    [original["sources"][0], {"key": key, "revision": bad_revision}]):
        assert (await project(client, {**original, "sources": sources}))["cards"] == [{"unit_id": original["unit_id"]}]
    duplicate = {**original, "sources": original["sources"] * 2}
    assert (await project(client, duplicate))["cards"][0]["idea_action"] == action(row)


async def test_calendar_cancellation_unlinks(env, client):
    row = (await generate(env, mailbox()[-1:]))[0]
    card = evidence(env, row)
    assert (await project(client, card))["cards"][0]["idea_action"] == action(row)
    cancelled = deepcopy(mailbox()[-1:])
    cancelled[0]["result"]["data"]["items"][0]["status"] = "cancelled"
    env.ideas.ingest(cancelled)
    assert (await project(client, card))["cards"] == [{"unit_id": card["unit_id"]}]


async def test_commitment_revision_and_fulfillment_unlink(env, client, monkeypatch):
    from colonymind.commitments import store as commitments
    monkeypatch.setattr(commitments, "now_utc", lambda: env.clock[0])
    env.mind.commitments = commitments.CommitmentStore(env.tmp / "commitments.db")
    item = env.mind.commitments.create(env.mind.owner_id, "Review invoice documents")
    await env.ideas.refresh(env.clock[0], 0)
    row, = env.ideas.catalog()
    card = evidence(env, row)
    assert card["sources"][0]["key"] == "commitment:" + item["id"]
    assert (await project(client, card))["cards"][0]["idea_action"] == action(row)
    env.mind.commitments.update(item["id"], description="Review corrected invoice documents")
    assert (await project(client, card))["cards"] == [{"unit_id": card["unit_id"]}]
    env.mind.commitments.update(item["id"], status="fulfilled")
    assert (await project(client, card))["cards"] == [{"unit_id": card["unit_id"]}]


async def test_deterministic_overlap_extra_evidence_and_id_tiebreaks(env):
    # Synthetic metadata isolates ranking from publication's dedupe policy.
    a, b, c = ({"key": key, "revision": key * 64} for key in ("a", "b", "c"))
    rows = [SimpleNamespace(id="a-id", context={"idea": {"private": "extra"}}),
            SimpleNamespace(id="z-id", context={"idea": {"private": "exact"}}),
            SimpleNamespace(id="b-id", context={"idea": {"private": "exact"}})]
    payloads = {"extra": {"sources": [a, b, c]}, "exact": {"sources": [a, b]}}
    engine = SimpleNamespace(mind=env.mind, rows=lambda: rows, valid=lambda *a, **kw: True,
                             open=lambda ref: payloads[ref])
    request = AssociationRequest.model_validate({"cards": [{"unit_id": "u", "sources": [b, a, b]}]})
    first = associate(engine, request.cards)
    assert first["cards"][0]["idea_action"] == {"idea_id": "b-id", "state": "available"}
    rows.reverse()
    assert associate(engine, request.cards) == first
    payloads["exact"] = {"sources": [a]}
    assert associate(engine, request.cards)["cards"][0]["idea_action"]["idea_id"] == "a-id"


async def test_empty_engine_legacy_exclusion_and_batch_edges(env, client):
    assert await project(client) == {"cards": [], "capabilities": {"feed_idea_association": True}, "ready": True}
    env.store.create(type="TASK_SUGGESTION", description="Legacy offer")
    card = {"unit_id": "unlinked", "sources": [{"key": "x", "revision": "a" * 64}]}
    assert (await project(client, card))["cards"] == [{"unit_id": "unlinked"}]
    row = (await generate(env, mailbox()[:1]))[0]
    card = evidence(env, row)
    result = await project(client, *([card] * 100))
    assert result["cards"] == [{"unit_id": card["unit_id"], "idea_action": action(row)}] * 100
    env.store._ideas = None
    assert (await project(client, card)) == {"cards": [{"unit_id": card["unit_id"]}],
                                           "capabilities": {"feed_idea_association": True}, "ready": False}
    host.set_initiative_store(None)
    response = await client.post(ROUTE, json={"cards": []})
    assert response.status_code == 501 and response.json()["detail"] == "Initiative store not initialized"


@pytest.mark.parametrize("body", [None, [], {}, {"cards": [], "owner": "private-canary"},
    {"cards": [{"unit_id": 17, "sources": []}]}, {"cards": [{"unit_id": "", "sources": []}]},
    {"cards": [{"unit_id": "x" * 257, "sources": []}]}, {"cards": [{"unit_id": "u"}]},
    {"cards": [{"unit_id": "u", "sources": [], "text": "private-canary"}]},
    {"cards": [{"unit_id": "u", "sources": [{"key": "k", "revision": "A" * 64}]}]},
    {"cards": [{"unit_id": "u", "sources": [{"key": "k", "revision": "a" * 64 + "\n"}]}]},
    {"cards": [{"unit_id": "u", "sources": [{"key": "", "revision": "a" * 64}]}]},
    {"cards": [{"unit_id": "u", "sources": [{"key": "k", "revision": "a" * 64, "nested": {"text": "private-canary"}}]}]},
    {"cards": [{"unit_id": "u", "sources": []}] * 101},
    {"cards": [{"unit_id": "u", "sources": [{"key": "k", "revision": "a" * 64}] * 65}]}])
async def test_invalid_requests_have_redacted_refusals(env, client, body, caplog):
    response = await client.post(ROUTE, json=body)
    assert response.status_code == 422, response.text
    assert response.json() == {"detail": "invalid_feed_idea_association"}
    assert "private-canary" not in caplog.text
    response = await client.post(ROUTE, content=b'{invalid-private-canary', headers={"Content-Type": "application/json"})
    assert response.json() == {"detail": "invalid_feed_idea_association"}


async def test_restart_recomputes_persisted_build_without_new_association_store(env, client, monkeypatch):
    row = (await generate(env, mailbox()[:1]))[0]
    card = evidence(env, row)
    await env.ideas.build(row, "full", None, None)
    before = await project(client, card)
    env.store.close()
    # The shared fixture owns finalization of the original, now closed store.
    monkeypatch.setattr(env.store, "close", lambda: None)
    env.store = InitiativeStore(env.tmp)
    env.mind.store = env.store
    env.ideas = Ideas(env.mind, vault=env.ideas.vault, queue=env.queue)
    env.store._ideas = env.mind.ideas = env.ideas
    host.set_initiative_store(env.store)
    assert await project(client, card) == before
    assert (await env.ideas.build(env.store.get(row.id), "full", None, None))["taskId"] == "t-offline-1"
    assert len(env.queue.tasks) == 1 and len(env.queue.calls) == 1
    env.store.close()


async def test_owner_relay_allowed_generic_agent_scope_and_hand_refused(env):
    config = default_credentials()
    for name, owner, access, extra in (
        ("phone", ["companion"], ["read", "manage"], {}),
        ("agent", "any", ["read", "write", "manage"], {}),
        ("guest", [], ["read", "manage"], {}),
        ("reader", ["companion"], ["read"], {}),
        ("scoped", ["companion"], ["read", "manage"], {"scopes": ["context:read"]}),
        ("hand", [], ["hands"], {}),
    ):
        config[name] = {"owner": owner, "contacts": "all" if owner else "own", "access": access,
                        "secrets": [{"sha256": secret_digest(name)}], **extra}
        if "scopes" in extra:
            del config[name]["access"]
    application = FastAPI()
    application.add_middleware(ApiKeyMiddleware, api_key=KEY, credentials=CredentialTable.from_config(config))
    application.include_router(host.router)
    async with AsyncClient(transport=ASGITransport(app=application), base_url="http://test") as client:
        for token in (None, "invalid", "phone", "agent", "guest", "reader", "scoped", "hand"):
            response = await client.post(ROUTE, json={"cards": []}, headers={"Authorization": "Bearer " + token} if token else {})
            assert response.status_code == (401 if token in {None, "invalid"} else 200 if token == "phone" else 403), response.text
    development = FastAPI()
    development.add_middleware(ApiKeyMiddleware, api_key=None, credentials=CredentialTable.from_config(default_credentials()))
    development.include_router(host.router)
    async with AsyncClient(transport=ASGITransport(app=development, client=("127.0.0.1", 1)), base_url="http://localhost") as client:
        assert (await client.post(ROUTE, json={"cards": []})).status_code == 403


async def test_one_freshness_read_per_distinct_source_per_request(env, client, monkeypatch):
    rows = await generate(env, mailbox())
    assert len(rows) > 1
    sources = {(s["key"], s["revision"]) for row in rows for s in env.ideas.open(row.context["idea"]["private"])["sources"]}
    original, calls = env.ideas.source_current, []
    def counted(source, now, *, fresh=True):
        calls.append(source["key"])
        return original(source, now, fresh=fresh)
    monkeypatch.setattr(env.ideas, "source_current", counted)
    cards = [evidence(env, row, unit=f"u{i}") for i, row in enumerate(rows)] * 25  # 100 cards
    result = await project(client, *cards)
    assert len(calls) == len(sources) and len(set(calls)) == len(calls)
    assert [c["idea_action"]["idea_id"] for c in result["cards"][:len(rows)]] == [r.id for r in rows]


from colonymind.memory import curated
@pytest.mark.parametrize('invalidate', ['revision', 'erasure', 'thread_reply', 'expiry', 'cancelled', 'dropped', 'expired', 'wrong_owner', 'unreadable', 'unknown_state'])
async def test_stale_expired_private_and_unsupported_candidates_unlink(env, client, invalidate):
    row = (await generate(env, mailbox()[:1]))[0]
    card = evidence(env, row)
    assert 'idea_action' in (await project(client, card))['cards'][0]
    if invalidate == 'revision':
        changed = deepcopy(mailbox()[:1])
        changed[0]['result']['data']['snippet'] = 'Invoice resolved.'
        env.ideas.ingest(changed)
    elif invalidate == 'erasure':
        source = env.ideas.open(row.context['idea']['private'])['sources'][0]
        from test_curated_memory import historical_bytes
        _native_before_withdrawal = historical_bytes(env.mind.ledger)
        curated.forget_sources(env.mind.ledger, env.mind.owner_id, source_ids=[source['ref']['source_id']])
        assert historical_bytes(env.mind.ledger) == _native_before_withdrawal
    elif invalidate == 'thread_reply':
        reply = deepcopy(mailbox()[:1])
        reply[0]['result']['data']['id'] = 'reply-in-same-thread'
        env.ideas.ingest(reply)
    elif invalidate == 'expiry':
        await env.ideas.build(row, 'full', None, None)
        env.clock[0] += timedelta(hours=24)
        assert env.ideas.valid(env.store.get(row.id), accepted=True)
    elif invalidate in {'cancelled', 'expired', 'dropped'}:
        env.store.update(row.id, status=invalidate)
    else:
        idea = deepcopy(row.context['idea'])
        if invalidate == 'wrong_owner':
            idea['owner'] = 'another-owner'
        elif invalidate == 'unreadable':
            idea['private'] = 'missing-private-payload'
        else:
            idea['execution'] = {'taskId': 'durable-task', 'status': 'uncertain'}
        env.store.update(row.id, context={'idea': idea})
    assert (await project(client, card))['cards'] == [{'unit_id': card['unit_id']}]
