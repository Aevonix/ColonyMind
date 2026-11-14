"""External-write asks share hands creation and settlement without recording bodies."""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import json

import pytest

from test_hands_permit_answers import rig, app_answer, ORIGIN_SECRET, ORIGIN2_SECRET  # noqa: F401
from test_hands_permits import fx  # noqa: F401
from test_mind_loop import pinned_clock  # noqa: F401

AUTH = {"Authorization": "Bearer " + ORIGIN_SECRET}


def terms():
    return {"tool": "google_send", "args_hash": "a" * 64, "reason": "recipient came from tool content",
            "summary": {"recipients": ["friend@example.test"], "subject_or_title": "Hello"}}


async def request(rig, body=None, headers=AUTH):
    return await rig.client.post("/v1/mind/owner-actions", json=body or terms(), headers=headers)


@pytest.mark.parametrize("choice", ["yes", "no"])
async def test_create_dedup_poll_and_settle_through_hands_path(rig, choice):
    created = await request(rig)
    assert created.status_code == 200, created.text
    asked = created.json()
    ident = asked["ask_id"]
    assert asked["state"] == "asked"
    assert (await request(rig)).json() == asked
    row = rig.mind.store.get(ident)
    assert row.kind == "note" and row.context["origin"] == "hand" and row.hermes_kind == "none"
    view = (await rig.hub.get("/v1/mind/permit-asks/" + ident)).json()
    assert view["open"] is True and view["terms"]["call_digest"] == "a" * 64
    assert view["terms"]["call"]["arguments"] == terms()["summary"]
    assert "recipient came from tool content" not in json.dumps(row.context)
    assert (await app_answer(rig, ident, choice)).status_code == 200
    result = (await request(rig)).json()
    assert result["state"] == ("approved" if choice == "yes" else "denied")
    if choice == "yes":
        assert result["owner_action"] == {"ask_id": ident, "args_hash": "a" * 64,
            "settled_at": (await rig.hub.get("/v1/mind/permit-asks/" + ident)).json()["winner"]["at"]}
    assert (await request(rig)).json() == {"state": "denied", "ask_id": ident}


async def test_schema_redacts_secrets_and_rejects_body_and_answer_fields(rig):
    body = terms()
    body["summary"]["subject_or_title"] = "password: synthetic-password"
    ident = (await request(rig, body)).json()["ask_id"]
    row = rig.mind.store.get(ident)
    assert "synthetic-password" not in json.dumps(row.context)
    assert "synthetic-password" not in (await rig.hub.get("/v1/mind/permit-asks/" + ident)).text
    for change in ({"body": "PRIVATE-BODY"}, {"headers": {"Authorization": "PRIVATE-KEY"}},
                   {"winner": "yes"}, {"args_hash": "invalid"},
                   {"summary": {"recipients": ["bad"], "subject_or_title": ""}}):
        denied = await request(rig, terms() | change)
        assert denied.status_code == 422
        assert "PRIVATE" not in denied.text


async def test_same_call_from_another_creator_gets_its_own_ask(rig):
    first = (await request(rig)).json()["ask_id"]
    second = (await request(rig, headers={"Authorization": "Bearer " + ORIGIN2_SECRET})).json()["ask_id"]
    assert first != second
    assert (await app_answer(rig, first)).status_code == 200
    assert (await request(rig, headers={"Authorization": "Bearer " + ORIGIN2_SECRET})).json()["state"] == "asked"


@pytest.mark.parametrize("age,expected", [(599, "approved"), (600, "denied"), (601, "denied"), (-1, "denied")])
async def test_settlement_ten_minute_bound(rig, monkeypatch, age, expected):
    ident = (await request(rig)).json()["ask_id"]
    assert (await app_answer(rig, ident)).status_code == 200
    settled = rig.mind.clock()
    monkeypatch.setattr(rig.mind, "clock", lambda: settled + timedelta(seconds=age))
    assert (await request(rig)).json()["state"] == expected


async def test_binding_and_changed_args(rig):
    first = (await request(rig)).json()["ask_id"]
    second = (await request(rig, terms() | {"args_hash": "b" * 64})).json()["ask_id"]
    assert first != second
    assert (await app_answer(rig, first)).status_code == 200
    row = rig.mind.store.get(first)
    rig.mind.store.update(first, context={**row.context, "effect_digest": "0" * 64})
    assert (await request(rig)).json()["state"] == "denied"


async def test_concurrent_repeats_create_once_and_consume_once(rig):
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: rig.mind.owner_actions.request(terms(), "google-tools"), range(8)))
    assert len({r["ask_id"] for r in results}) == 1
    ident = results[0]["ask_id"]
    assert (await app_answer(rig, ident)).status_code == 200
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: rig.mind.owner_actions.request(terms(), "google-tools"), range(8)))
    assert sum(r["state"] == "approved" for r in results) == 1


async def test_no_new_ask_after_record_cap(rig):
    asked = (await request(rig)).json()
    rig.mind.owner_actions.records.update({("filler", i): {} for i in range(1024)})
    assert (await request(rig)).json() == asked
    assert (await request(rig, terms() | {"args_hash": "b" * 64})).json()["state"] == "denied"
