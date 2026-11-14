"""An integration's game log never becomes the owner's words.

An integration that holds the instance key and names the owner's contact in its turns writes its whole
game log as the owner: every line becomes the owner's claim, and the night's consolidation asks the
owner which of two things "you told me" is right. With credentials bound to who they may write as, the
same batch from an integration's own credential is refused as the owner and, sent under its own
identity, is kept as session evidence that no projection, appraisal, commitment capture or contradiction
check reads. The same two lines sent by the owner over WhatsApp still raise the question, so the
check would see a regression.
"""

from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from colonymind.api.credentials import CredentialTable, secret_digest
from colonymind.api.errors import install_exception_handlers
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import host
from colonymind.api.routers import mind as mind_router
from colonymind.beliefs.source_projection import SourceClaimProjection
from colonymind.mind.consolidate import TASK_DIGEST
from onekey import KEY
from test_mind_consolidate import OWNER, Fixture
from test_source_claim_projection import Model, claim

GAME_SECRET = "game-integration-secret-0123456789"
CREATURES = ("Ember", "Tide", "Moss", "Gale")


def game_log(index: int) -> str:
    return f"Turn {index}: my lead creature is {CREATURES[index % len(CREATURES)]}."


def extractor() -> Model:
    """A model that states one scalar claim for every game-log line it is shown."""
    return Model({game_log(index): claim(game_log(index), CREATURES[index % len(CREATURES)],
                                         predicate="lead_creature") for index in range(40)})


def turn(contact_id: str, index: int, *, channel: str = "game:stadium") -> dict:
    return {"identity": {"host_id": "hermes"},
            "context": {"session_id": f"match-{index // 10}", "contact_id": contact_id,
                        "turn_id": f"game-turn-{index}-{contact_id or 'own'}", "channel_id": channel},
            "user_message": {"role": "user", "content": game_log(index)},
            "assistant_message": {"role": "assistant", "content": "Your move."}}


@pytest.fixture
async def arm(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    for name in ("_contacts_store", "_telemetry", "_reranker", "_context_recall_selector", "_comms_log",
                 "_preference_learner"):
        monkeypatch.setattr(host, name, None, raising=False)
    fx = Fixture(tmp_path)
    fx.router.answers[TASK_DIGEST] = None
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(host.router)
    app.add_middleware(ApiKeyMiddleware, api_key=KEY, credentials=CredentialTable.from_config({
        "api-key": {"owner": ["whatsapp", "sms", "rcs", "voice"], "contacts": "all"},
        "game-bot": {"sha256": secret_digest(GAME_SECRET), "owner": [], "contacts": "own"}}))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield fx, client
    fx.store.close()


async def project_everything(fx) -> int:
    """Drain the claim projection with a model that would state a claim for every line."""
    model, drained = extractor(), 0
    while await SourceClaimProjection(fx.ledger).process_one(model):
        drained += 1
    return drained


async def night(fx) -> list:
    await fx.mind.consolidate()
    await fx.mind.tick(force=True)
    return fx.messages("contradiction")


async def test_a_batch_of_game_log_turns_from_an_integration_never_reaches_the_owner(arm):
    fx, client = arm
    headers = {"Authorization": "Bearer " + GAME_SECRET}
    for index in range(40):
        refused = await client.post("/v1/host/turns/sync", headers=headers, json=turn(OWNER, index))
        assert refused.status_code == 403 and refused.json()["detail"]["code"] == "owner_not_granted"
        kept = await client.post("/v1/host/turns/sync", headers=headers, json=turn("", index))
        assert kept.status_code == 200 and kept.json()["source_recorded"] is True
    with closing(fx.ledger._connect()) as conn:
        stored = conn.execute("SELECT contact_id, scope, count(*) FROM turn_sources GROUP BY 1, 2").fetchall()
        assert [tuple(row) for row in stored] == [("credential:game-bot", "session", 40)]
        for table in ("source_claim_jobs", "appraisal_runs", "commitment_runs", "opinion_jobs"):
            assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0, table  # noqa: S608

    assert await project_everything(fx) == 0
    assert fx.claims() == []
    assert await night(fx) == []
    assert [c for c in fx.mind.concerns.open(limit=100, status=("open", "intended")) if c.kind == "question"] == []


async def test_the_same_lines_from_the_owner_on_whatsapp_still_raise_the_question(arm):
    """The check above is not vacuous: two of the lines told by the owner become two claims and one question."""
    fx, client = arm
    headers = {"Authorization": "Bearer " + KEY}
    for index in (0, 1):
        response = await client.post("/v1/host/turns/sync", headers=headers,
                                     json=turn(OWNER, index, channel="whatsapp:+15550000001"))
        assert response.status_code == 200, response.text
        fx.shift(hours=1)
    assert await project_everything(fx) == 2
    assert len(fx.claims()) == 2
    question, = await night(fx)
    assert question.entity_id == OWNER and "You told me two things" in question.context["text"]


# ---------------------------------------------------------------------------
# The owner's answers in their own words come only from turns the credential may write as the owner
# ---------------------------------------------------------------------------

RELAY_SECRET = "relay-integration-secret-0123456789"


@pytest.mark.parametrize("alias", ["canonical", "deprecated"])
async def test_only_the_owners_own_turns_answer_his_question(tmp_path, monkeypatch, alias):
    """A delivered question is answered in the owner's words (digest-ux), read from their turns. A reply that an
    integration sends as the owner, the owner's lane from a platform the key is not granted the owner on, and an
    operator's test through the owner's channel are never their answer; their own reply over WhatsApp is. With the
    owner named only by the deprecated alias the sidecar still accepts, the answer is still the credential's to
    grant: a relay that may never write as the owner does not answer for them."""
    from test_digest_ux import TurnRouter
    from test_digest_ux_r2 import ANSWER, calls, live, office
    from test_mind_consolidate import Fixture as NightFixture
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    if alias == "canonical":
        monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    else:
        monkeypatch.delenv("COLONYMIND_OWNER_CONTACT_ID", raising=False)
        monkeypatch.delenv("COLONYMIND_OWNER_PERSON_ID", raising=False)
        monkeypatch.setenv("COLONYMIND_HOST_CONTACT_ID", OWNER)
    for name in ("_contacts_store", "_telemetry", "_reranker", "_context_recall_selector", "_comms_log",
                 "_preference_learner"):
        monkeypatch.setattr(host, name, None, raising=False)
    # A day in the past: the replies are dated by the fixture's clock, never later than the real one.
    fx = NightFixture(tmp_path, router=TurnRouter({"mind_claim_relation": {"relation": "replaces"},
                                                   ANSWER: {"answer": "second"}}),
                      at=datetime.now(UTC).replace(microsecond=0) - timedelta(days=1))
    fx.router.answers[TASK_DIGEST] = None
    mind_router.set_mind(fx.mind)
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(host.router)
    app.add_middleware(ApiKeyMiddleware, api_key=KEY, credentials=CredentialTable.from_config({
        "api-key": {"owner": ["whatsapp", "sms", "rcs", "voice"], "contacts": "all"},
        "game-bot": {"sha256": secret_digest(GAME_SECRET), "owner": [], "contacts": "own"},
        "relay": {"sha256": secret_digest(RELAY_SECRET), "owner": [], "contacts": "all"}}))
    try:
        _question, first, second = await office(fx)

        async def reply(secret, text, turn_id, *, contact=OWNER, channel="whatsapp:+15550000001", headers=None):
            fx.shift(minutes=1)
            response = await client.post("/v1/host/turns/sync", headers={
                "Authorization": "Bearer " + secret, **(headers or {})}, json={
                "identity": {"host_id": "hermes"},
                "context": {"session_id": "chat-1", "contact_id": contact, "turn_id": turn_id,
                            "channel_id": channel, "metadata": {"occurred_at": fx.now.isoformat()}},
                "user_message": {"role": "user", "content": text},
                "assistant_message": {"role": "assistant", "content": "Noted."}})
            await fx.mind.questions.drain()
            return response

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            refused = [
                await reply(GAME_SECRET, "The second one.", "r-game"),
                await reply(GAME_SECRET, "The second one.", "r-game-own", contact=""),
                await reply(RELAY_SECRET, "The second one.", "r-relay"),
                await reply(KEY, "The second one.", "r-cli", channel="cli"),
                await reply(KEY, "The second one.", "r-header", headers={"X-ColonyMind-Test": "1"}),
                await reply(KEY, "[operator-test] The second one.", "r-marked"),
            ]
            if alias == "canonical":
                assert [response.status_code for response in refused] == [403, 200, 403, 403, 200, 200]
            assert calls(fx) == [] and live(fx, first, second) == {first, second}
            owners = await reply(KEY, "The second one.", "r-owner")
            assert owners.status_code == 200 and owners.json()["source_recorded"] is True
        assert len(calls(fx)) == 1 and live(fx, first, second) == {second}
    finally:
        mind_router.set_mind(None)
        fx.store.close()
