"""Front HTTP writes preserve controller authority, lifecycle and idempotency."""

from dataclasses import replace

import httpx
import pytest
from fastapi import FastAPI

from colonymind.api.auth import key_authority
from colonymind.api.routers.hands import router
from colonymind.hands import controller
from test_hands_controller import REF, c, finish, running


ROUTES = [
    ("/owner_message", REF),
    ("/obligations/obligation-one/review", {"turn_ref": "turn-two", "successful": True, "rendered": True}),
    ("/obligations/obligation-one/communicate", {
        "turn_ref": "turn-two", "silent": False, "destination": {"platform": "test", "chat": "owner-chat"}}),
]


def client_for(authority):
    app = FastAPI()
    app.include_router(router)

    async def authenticated(scope, receive, send):
        scope.setdefault("state", {})["colonymind_authority"] = authority
        await app(scope, receive, send)

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=authenticated), base_url="http://test")


@pytest.fixture
async def route_controller(c, monkeypatch):
    async def current():
        return c
    monkeypatch.setattr(controller, "get_controller", current)
    return c


async def post(client, path, body, key):
    return await client.post("/v1/hands" + path, json=body, headers={"Idempotency-Key": key})


async def test_owner_message_route_records_identity_and_replays_the_original_write(route_controller):
    c = route_controller
    plan = await running(c)
    reference = {**REF, "message_id": "owner-message-two"}
    async with client_for(key_authority()) as client:
        response = await post(client, "/owner_message", reference, "owner-message")
        assert response.status_code == 200, response.text
        assert response.json() == {"recorded": True}
        assert (await c.get_task(plan["task_id"]))["owner_spoke_since"] == reference
        replay = await post(client, "/owner_message", reference, "owner-message")
        assert replay.json() == response.json()
        changed = await post(client, "/owner_message", {**REF, "message_id": "another-message"}, "owner-message")
        assert changed.status_code == 200 and changed.json() == response.json()
        assert (await c.get_task(plan["task_id"]))["owner_spoke_since"] == reference


async def test_review_and_communication_routes_keep_controller_state_rules(route_controller):
    c = route_controller
    plan = await running(c)
    await finish(c, plan)
    oid = (await c.get_task(plan["task_id"]))["obligation_id"]
    prefix = f"/obligations/{oid}"
    async with client_for(key_authority()) as client:
        for successful, rendered in ((False, True), (True, False)):
            body = {"turn_ref": "review-turn", "successful": successful, "rendered": rendered}
            response = await post(client, prefix + "/review", body, f"ignored:{successful}:{rendered}")
            assert response.status_code == 200, response.text
            assert response.json()["state"] == "result_ready"
        reviewed = {"turn_ref": "review-turn", "successful": True, "rendered": True}
        response = await post(client, prefix + "/review", reviewed, "reviewed")
        assert response.status_code == 200, response.text
        assert response.json()["state"] == "reviewed"
        replay = await post(client, prefix + "/review", reviewed, "reviewed")
        assert replay.json() == response.json()
        communication = {"turn_ref": "review-turn", "silent": False,
                         "destination": {"platform": REF["platform"], "chat": REF["chat"]}}
        reused = await post(client, prefix + "/communicate", communication, "reviewed")
        assert reused.status_code == 409
        silent = await post(client, prefix + "/communicate", {**communication, "silent": True}, "silent")
        assert silent.status_code == 200 and silent.json()["state"] == "reviewed"
        wrong_turn = await post(client, prefix + "/communicate", {**communication, "turn_ref": "other"}, "wrong-turn")
        assert wrong_turn.status_code == 409
        wrong_chat = await post(client, prefix + "/communicate", {
            **communication, "destination": {"platform": REF["platform"], "chat": "other"}}, "wrong-chat")
        assert wrong_chat.status_code == 409
        response = await post(client, prefix + "/communicate", communication, "communicated")
        assert response.status_code == 200, response.text
        assert response.json()["state"] == "communicated"
        assert response.json()["delivery"] == "unknown"
        assert response.json()["communicated_ref"] == "review-turn"
        replay = await post(client, prefix + "/communicate", communication, "communicated")
        assert replay.json() == response.json()


@pytest.mark.parametrize("path,body", ROUTES)
@pytest.mark.parametrize("authority", [None, replace(key_authority(), owner_platforms=frozenset()),
                                      replace(key_authority(), principal_id="credential:guest")])
async def test_front_writes_require_the_owner_credential(path, body, authority):
    async with client_for(authority) as client:
        response = await post(client, path, body, "not-authorized")
    assert response.status_code == 403, response.text


@pytest.mark.parametrize("path,body", ROUTES)
@pytest.mark.parametrize("headers", [{}, {"Idempotency-Key": ""}, {"Idempotency-Key": "  "}])
async def test_front_writes_require_a_nonblank_idempotency_key(path, body, headers):
    async with client_for(key_authority()) as client:
        response = await client.post("/v1/hands" + path, json=body, headers=headers)
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("path,body", [
    ("/owner_message", {"platform": "test", "chat": "owner-chat"}),
    ("/obligations/obligation-one/review", {"turn_ref": "turn-two", "successful": True}),
    ("/obligations/obligation-one/communicate", {"turn_ref": "turn-two", "silent": False,
        "destination": {"platform": "test"}}),
    ("/obligations/obligation-one/communicate", {"turn_ref": "turn-two", "silent": False, "destination": REF}),
])
async def test_front_routes_reject_incomplete_or_unexpected_fields(path, body):
    async with client_for(key_authority()) as client:
        response = await post(client, path, body, "invalid-body")
    assert response.status_code == 422, response.text
