"""Pages in a turn: the assembled context carries the page on the turn's person, whole, and the agent's own
page, whose lines are chosen against the turn's message."""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from colonymind.api.errors import install_exception_handlers
from colonymind.api.routers import host
from colonymind.api.schemas.host import ContextSection
from colonymind.memory import blocks, compass
from colonymind.pages import store as pages
from colonymind.pages import turn
from colonymind.turns import get_turn_idempotency_ledger


@pytest.fixture
def world(monkeypatch, tmp_path):
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "off")
    for name in ("_contacts_store", "_telemetry", "_reranker", "_context_recall_selector", "_comms_log"):
        monkeypatch.setattr(host, name, None)
    from colonymind import vector
    monkeypatch.setattr(vector, "get_store", lambda: None)
    monkeypatch.setattr(vector, "get_pipeline", lambda: None)
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(host.router)
    ledger = get_turn_idempotency_ledger(tmp_path)
    for turn_id, contact, said in (("t-casey", "casey", "Mornings suit me best."),
                                   ("t-sam", "sam", "Please give me time to answer.")):
        ledger.record_source(turn_id, contact_id=contact, session_id="s-" + contact, derive_claims=False, messages=[
            {"role": "user", "content": said}, {"role": "assistant", "content": f"Noted for {contact}."}])
    return SimpleNamespace(app=app, ledger=ledger)


async def assembled(app, contact, message="When shall we talk?"):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/host/context/assemble", json={
            "identity": {"host_id": "test-host"}, "context": {"contact_id": contact, "session_id": "s-now"},
            "incoming_message": {"role": "user", "content": message}})
    assert response.status_code == 200, response.text
    return {section["id"]: section["body"] for section in response.json()["sections"]}


async def test_a_turn_with_a_person_carries_their_page_and_the_agents_own(world):
    mark, at = pages.begin(world.ledger)
    pages.rewrite(world.ledger, "casey", ("about",), [
        {"kind": "about", "text": "Prefers mornings.", "sources": ["t-casey"]}], mark=mark, at=at)
    pages.add(world.ledger, "casey", "boundary", "Keep replies short.", ["t-casey"])
    pages.add(world.ledger, "sam", "boundary", "Be patient.", ["t-sam"])
    pages.add(world.ledger, pages.SELF, "like", "Long walks.", ["t-casey"])
    got = await assembled(world.app, "casey")
    assert got[turn.PERSON_ID] == "- Your boundary with them: Keep replies short.\n- About them: Prefers mornings."
    assert got[turn.SELF_ID] == "- You like: Long walks."
    assert "Be patient" not in "\n".join(got.values())
    # Someone with no page gets the agent's own page alone.
    assert turn.PERSON_ID not in await assembled(world.app, "robin")
    # A line stops being carried when the turn it cites is forgotten.
    from colonymind.memory import curated
    curated.forget_sources(world.ledger, "casey", source_ids=["t-casey"])
    gone = await assembled(world.app, "casey")
    assert turn.PERSON_ID not in gone and turn.SELF_ID not in gone


async def test_the_persons_page_stays_whole_and_the_agents_own_lines_are_chosen_for_the_turn():
    person = ContextSection(id=turn.PERSON_ID, title=turn.PERSON_TITLE, priority=85, body="\n".join(
        f"- Your boundary with them: rule {n} " + "x" * 90 for n in range(8)))
    own = ContextSection(id=turn.SELF_ID, title=turn.SELF_TITLE, priority=82, body="\n".join(
        [f"- You like: thing {n} " + "y" * 90 for n in range(6)] + ["- You dislike: loud rooms."]))

    async def judge(query, documents, top_k=10):
        return [SimpleNamespace(index=index, score=float("loud rooms" in text)) for index, text in enumerate(documents)]

    settings = compass.SelectionSettings(enabled=True, budget=200, timeout_s=1.0, candidates=48,
                                         due_soon=compass.timedelta(hours=24), min_score=0.5)
    selected, report = await compass.select_context([person, own], "Is the room loud?", judge, settings=settings)
    assert report["status"] == "selected"
    assert [section.body for section in selected] == [person.body, "- You dislike: loud rooms."]
    # Under a short budget the person's page is admitted before the recalled memory, within its own limit.
    assert blocks.RANK.index(turn.PERSON_ID) < blocks.RANK.index("colonymind-memory") < blocks.RANK.index(turn.SELF_ID)
    assert blocks.LIMITS[turn.PERSON_ID] == 2000
