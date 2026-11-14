"""Repeated delegation replaces one exact hold without moving its authority."""
from __future__ import annotations

import asyncio
import json

import pytest

from colonymind.hands import approvals, wire
from colonymind.hands.store import HandsStore
from owner_answer import owner_answers
from test_hands_permits import fx, query, seed  # noqa: F401
from test_mind_loop import pinned_clock  # noqa: F401


FIRST = "h-000000000001"
SECOND = "h-000000000002"


async def second_hand(store, *, parent="owner-session"):
    await seed(store, hand=SECOND, task="t-000000000002")
    await store.db.execute("UPDATE hands SET parent_session=? WHERE id=?", (parent, SECOND))
    await store.db.commit()


async def permit(store, ident):
    return dict((await store.db.execute_fetchall("SELECT * FROM permits WHERE id=?", (ident,)))[0])


def open_asks(mind):
    return [row.id for row in mind.store.intentions(status=["asked"], limit=100)]


@pytest.mark.asyncio
@pytest.mark.parametrize("answered", [False, True])
async def test_new_hand_replaces_same_parent_hold_and_never_inherits_yes(fx, answered):
    mind, store, service, body, _ = fx
    first = await service.hold(FIRST, body, "first")
    if answered:
        await owner_answers(mind.mind, first.ask_id)
        await service.refresh_pending()
        assert (await permit(store, first.permit_id))["state"] == "granted"
    await second_hand(store)
    second = await service.hold(SECOND, body, "second")

    assert open_asks(mind) == [second.ask_id]
    assert mind.store.get(first.ask_id).status == "cancelled"
    assert (await permit(store, first.permit_id))["state"] == "stale"
    assert (await service.check(query(body, FIRST), "old-check")).decision == "hold"
    assert (await service.check(query(body, SECOND), "new-before-yes")).decision == "hold"
    assert await service.hold(FIRST, body, "first") == first
    assert open_asks(mind) == [second.ask_id]

    # A late reply to the superseded ask cannot settle the replacement.
    result = await owner_answers(mind.mind, first.ask_id)
    assert result is None
    assert (await service.check(query(body, SECOND), "late-answer")).decision == "hold"
    await owner_answers(mind.mind, second.ask_id)
    assert (await service.check(query(body, FIRST), "old-after-yes")).decision == "hold"
    assert (await service.check(query(body, SECOND), "new-after-yes")).decision == "allow"
    assert (await service.check(query(body, SECOND), "new-repeat")).decision == "hold"


@pytest.mark.asyncio
async def test_replacement_also_revokes_old_continuation_and_withdraws_unsent_notice(fx):
    mind, store, service, body, _ = fx
    first = await service.hold(FIRST, body, "first")
    await mind.mind._deliver_asks([mind.store.get(first.ask_id)], mind.mind.clock())
    assert mind.mind.outbox.ready()
    await seed(store, hand="h-000000000003", continues=FIRST)
    await second_hand(store)
    second = await service.hold(SECOND, body, "second")

    assert mind.mind.outbox.ready() == []
    await owner_answers(mind.mind, second.ask_id)
    assert (await service.check(query(body, "h-000000000003"), "old-continuation")).decision == "hold"
    assert (await service.check(query(body, SECOND), "new-hand")).decision == "allow"


@pytest.mark.asyncio
@pytest.mark.parametrize("difference", ["parent", "call", "workdir", "target", "value"])
async def test_different_binding_keeps_both_asks(fx, difference):
    mind, store, service, body, _ = fx
    first = await service.hold(FIRST, body, "first")
    await second_hand(store, parent="another-parent" if difference == "parent" else "owner-session")
    changed = body.model_dump()
    if difference == "call":
        changed["call"]["arguments"]["command"] += " --verbose"
    elif difference == "workdir":
        changed["workdir"] += "/another-directory"
    elif difference == "target":
        changed["targets"][0]["target"] += ".other"
    elif difference == "value":
        changed["targets"][0]["value"] = "another-content-digest"
    second = await service.hold(SECOND, wire.HoldRequest(**changed), "second")

    assert set(open_asks(mind)) == {first.ask_id, second.ask_id}
    assert (await permit(store, first.permit_id))["state"] == "waiting"


@pytest.mark.asyncio
async def test_concurrent_holds_across_connections_leave_one_ask_and_one_permit(fx):
    mind, store, service, body, _ = fx
    await second_hand(store)
    other = HandsStore(store.path)
    await other.connect()
    try:
        held = await asyncio.gather(service.hold(FIRST, body, "first"),
            approvals.Approvals(other, mind.mind).hold(SECOND, body, "second"))
        rows = [await permit(store, result.permit_id) for result in held]
        current = next(row for row in rows if row["state"] == "waiting")
        stale = next(row for row in rows if row["state"] == "stale")
        assert open_asks(mind) == [current["ask_id"]]
        assert mind.store.get(stale["ask_id"]).status == "cancelled"
        await owner_answers(mind.mind, current["ask_id"])
        results = await asyncio.gather(service.check(query(body, FIRST), "check-first"),
            approvals.Approvals(other, mind.mind).check(query(body, SECOND), "check-second"))
        assert sorted(result.decision for result in results) == ["allow", "hold"]
    finally:
        await other.close()


@pytest.mark.asyncio
async def test_target_order_is_not_a_new_binding(fx):
    mind, store, service, body, _ = fx
    targets = [item.model_dump() for item in body.targets]
    targets.append({"kind": "file", "target": "/tmp/another-file", "value": "another-digest"})
    body = wire.HoldRequest(**{**body.model_dump(), "targets": targets})
    first = await service.hold(FIRST, body, "first")
    await second_hand(store)
    body = wire.HoldRequest(**{**body.model_dump(), "targets": list(reversed(targets))})
    second = await service.hold(SECOND, body, "second")

    assert open_asks(mind) == [second.ask_id]
    assert (await permit(store, first.permit_id))["state"] == "stale"
    assert json.loads((await permit(store, second.permit_id))["targets"]) == approvals.target_values(body.targets)
