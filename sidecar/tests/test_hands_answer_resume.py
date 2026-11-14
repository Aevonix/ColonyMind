"""Answer handoffs expose only current grants from the trusted answer reader."""

from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.requests import Request

from colonymind.api.auth import key_authority
from colonymind.api.routers import mind as mind_routes
from colonymind.hands import approvals, sessions
from owner_answer import owner_answers
from test_hands_permits import fx, pinned_clock  # noqa: F401


@pytest.fixture
def answer_route(fx, monkeypatch):
    mind, store, service, _, _ = fx
    monkeypatch.setattr(mind_routes, "_mind", mind.mind)

    @asynccontextmanager
    async def opened():
        yield store

    monkeypatch.setattr(sessions, "open_store", opened)
    app = FastAPI()
    app.state.hands_store = store
    request = Request({"type": "http", "method": "POST", "path": "/v1/mind/answers", "headers": [],
                       "app": app, "state": {"colonymind_authority": key_authority()}})
    shown = set()

    async def answer(ident, text="yes", *, platform="whatsapp", message="owner-answer"):
        if (ident, platform) not in shown:
            mind.mind.record_shown([ident], platform=platform, message_id="ask-message", chat="owner-chat",
                                   at=mind.now - timedelta(seconds=30))
            shown.add((ident, platform))
        body = mind_routes.AnswersBody(platform=platform, contact_id=mind.mind.owner_id,
            chat_id="owner-chat", chat_type="dm", message_id=message, turn_id=message,
            arrived_at=mind.now.isoformat(), text=text, reply_to_message_id="ask-message")
        # Invoke the handler directly: grants must be synchronized before its
        # response, not later in a request dependency's teardown.
        return await mind_routes.answers(body, request)

    return answer


@pytest.mark.parametrize("platform", ["rcs", "whatsapp", "sms"])
async def test_answer_returns_exact_hand_grant_before_the_handler_returns(fx, answer_route, platform):
    _, store, service, body, _ = fx
    held = await service.hold("h-000000000001", body, "hold")
    result = await answer_route(held.ask_id, platform=platform)
    expected = {"permit_id": held.permit_id, "ask_id": held.ask_id, "hand_id": "h-000000000001",
                "parent_session": "owner-session", "call": body.call.model_dump(), "workdir": body.workdir}
    assert result["approved"] == [held.ask_id]
    assert result["granted_hands"] == [expected]
    row = dict((await store.db.execute_fetchall("SELECT * FROM permits"))[0])
    assert row["state"] == "granted" and row["owner_answer_ref"]
    assert (await answer_route(held.ask_id, platform=platform))["granted_hands"] == [expected]
    assert (await store.db.execute_fetchall("SELECT COUNT(*) FROM permits"))[0][0] == 1


@pytest.mark.parametrize("verdict", ["no", "lapse"])
async def test_no_and_lapse_publish_no_hand_grant(fx, answer_route, verdict):
    mind, store, service, body, _ = fx
    held = await service.hold("h-000000000001", body, "hold")
    if verdict == "lapse":
        mind.shift(days=4)
    result = await answer_route(held.ask_id, "no" if verdict == "no" else "yes")
    assert result["granted_hands"] == []
    assert (await store.db.execute_fetchall("SELECT state FROM permits"))[0][0] in {"refused", "lapsed"}


async def test_console_yes_stays_refused_and_has_no_grant(fx, answer_route):
    mind, _, service, body, _ = fx
    held = await service.hold("h-000000000001", body, "hold")
    with pytest.raises(PermissionError):
        await mind.mind.answer(mind.store.get(held.ask_id).ask_code, yes=True)
    assert (await answer_route(held.ask_id, "maybe"))["granted_hands"] == []
    rows = await approvals.held_permits("owner-session")
    assert rows.permits[0].state == "waiting"


@pytest.mark.parametrize("verdict", ["yes", "no", "lapse", "forged"])
async def test_held_api_refreshes_ingested_answers_and_refusals(fx, answer_route, verdict):
    mind, store, service, body, _ = fx
    held = await service.hold("h-000000000001", body, "hold")
    if verdict in {"yes", "no"}:
        await owner_answers(mind.mind, held.ask_id, verdict)
    elif verdict == "lapse":
        mind.shift(days=4)
    else:
        mind.store.transition(held.ask_id, "approved", action="queued", at=mind.now,
                              details={"by": "owner", "answer": {"message": "untrusted"}})
    result = await approvals.held_permits("owner-session")
    assert result.refreshed is True
    if verdict == "yes":
        assert len(result.permits) == 1
        permit = result.permits[0]
        assert permit.state == "granted" and permit.permit_id == held.permit_id
        assert permit.ask_id == held.ask_id and permit.call == body.call and permit.workdir == body.workdir
    else:
        assert result.permits == []
    assert (await store.db.execute_fetchall("SELECT state FROM permits"))[0][0] != "consumed"


async def test_unavailable_mind_keeps_bindings_readable_but_cannot_authorize_a_resume(fx, answer_route, monkeypatch):
    mind, _, service, body, _ = fx
    held = await service.hold("h-000000000001", body, "hold")
    await answer_route(held.ask_id)
    mind.shift(days=4)
    monkeypatch.setattr(mind_routes, "_mind", None)
    result = await approvals.held_permits("owner-session")
    assert result.permits[0].state == "granted"
    assert result.refreshed is False
    monkeypatch.setattr(mind_routes, "_mind", mind.mind)
    result = await approvals.held_permits("owner-session")
    assert result.refreshed is True and result.permits == []


@pytest.mark.parametrize("name", ["HANDS.md", "APPROVALS.md"])
def test_docs_describe_automatic_resume_and_the_answer_turn_boundary(name):
    text = (Path(__file__).resolve().parents[2] / "docs" / name).read_text()
    flat = " ".join(text.split())
    assert "runtime automatically resumes" in flat
    assert "waits for the current attempt to end" in flat
    assert "must not continue it again or perform the action itself" in flat
    assert "duplicate continue is a no-op" in flat
    assert "a no, a lapse or a refusal resumes nothing" in flat
