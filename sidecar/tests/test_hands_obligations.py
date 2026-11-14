"""Obligations require successful review, explicit communication and real receipts."""

import pytest
from fastapi import HTTPException

from colonymind.hands import wire
from test_hands_controller import c, finish, running, REF


async def test_obligation_walks_all_states_and_silence_never_closes(c):
    p = await running(c)
    task = await c.get_task(p["task_id"])
    oid = task["obligation_id"]
    assert (await c.obligation(oid))["state"] == "open"
    await finish(c, p)
    assert (await c.obligation(oid))["state"] == "result_ready"
    await c.review(oid, turn_ref="review", successful=False, key="failed-review")
    assert (await c.obligation(oid))["state"] == "result_ready"
    await c.review(oid, turn_ref="review", successful=True, key="review")
    assert (await c.obligation(oid))["state"] == "reviewed"
    await c.accept(p["task_id"], wire.AcceptRequest(hand=p["hand_id"], reason="Checked", for_version=1), "accept")
    assert (await c.obligation(oid))["state"] == "accepted"
    await c.communicate(oid, "review", silent=True, destination=REF, key="silent")
    assert (await c.obligation(oid))["state"] == "accepted"
    digest = await c.digest(wire.DigestRequest(session="new-owner-session"))
    assert any(item["obligation_id"] == oid for item in digest["items"])
    await c.told(oid, wire.ToldRequest(turn_ref="reply"), "told")
    obligation = await c.obligation(oid)
    assert obligation["state"] == "communicated" and obligation["delivery"] == "unknown"
    await c.delivered(oid, wire.DeliveredRequest(platform_message_id="receipt-1"), "delivered")
    assert (await c.obligation(oid))["delivery"] == "receipt"
    assert (await c.obligation(oid))["state"] == "delivered"


async def test_delivery_without_communication_refused_and_revision_supersedes(c):
    p = await running(c)
    oid = (await c.get_task(p["task_id"]))["obligation_id"]
    with pytest.raises(HTTPException):
        await c.delivered(oid, wire.DeliveredRequest(platform_message_id="unrelated"), "receipt")
    await c.revise(p["task_id"], wire.ReviseRequest(goal="New goal"), "revise")
    assert (await c.obligation(oid))["state"] == "superseded"
    task = await c.get_task(p["task_id"])
    assert task["obligation_id"] != oid
    await c.cancel(p["task_id"], wire.CancelRequest(reason="Stop", owner_ref=REF), "cancel")
    assert (await c.obligation(task["obligation_id"]))["state"] == "cancelled"


async def test_digest_twelve_active_hands_with_evidence_within_900(c):
    c.config["max_running"] = 12
    for n in range(12):
        p = await running(c, f"hand-{n}")
        from test_hands_controller import EPOCH
        await c.question(p["hand_id"], wire.QuestionRequest(epoch=EPOCH,
                         question="Which artifact should be checked? " * 10), f"question-{n}")
    result = await c.digest(wire.DigestRequest(session="new-session"))
    assert len(result["text"]) <= 900
    assert "12" in result["text"] and "owed" in result["text"]
    assert len(result["items"]) == 12
    assert "more" in result["text"]
