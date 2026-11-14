"""Repeated pulls reuse ask decisions until conversation evidence changes or a deadline arrives."""

from datetime import timedelta

import pytest

from test_ask_pause import asked, conversation, fx, its_message, last_verdict, owner_writes, ready_ids
from test_mind_loop import pinned_clock  # noqa: F401


def decisions(fixture, message):
    return [row for row in fixture.mind.owner_gate.ledger.recent(
        since=message.created_at - timedelta(minutes=1), limit=500) if row["ref"] == message.id]


async def test_unchanged_pulls_replay_a_pause_and_an_unclaimed_offer_across_restart(fx):
    conversation(fx, "initial-turn")
    fx.shift(seconds=10)
    message = await its_message(fx, asked(fx, "K7F"))
    assert message.id not in await ready_ids(fx)
    held = last_verdict(fx, message)
    assert held["stage"] == "pause"
    for index in range(100):
        fx.shift(seconds=0.1)
        if index == 50:
            fx.restart()
        assert message.id not in await ready_ids(fx)
        assert last_verdict(fx, message)["id"] == held["id"]
    assert len(decisions(fx, message)) == 1

    await owner_writes(fx)
    assert message.id in await ready_ids(fx)
    offered = last_verdict(fx, message)
    assert offered["decision"] == "send"
    for index in range(100):
        fx.shift(seconds=0.1)
        if index == 50:
            fx.restart()
        assert message.id in await ready_ids(fx)
        assert last_verdict(fx, message)["id"] == offered["id"]
    assert len(decisions(fx, message)) == 2


@pytest.mark.parametrize("cls,wait_minutes", [("floor", 10), ("self", 60)])
async def test_changing_conversation_rechecks_are_bounded_by_the_first_pause(fx, cls, wait_minutes):
    conversation(fx, "initial-turn")
    fx.shift(seconds=10)
    row = asked(fx, "K7F", cls=cls,
                type_="floor_effect" if cls == "floor" else "research")
    message = await its_message(fx, row)
    assert message.id not in await ready_ids(fx)
    first_pause = fx.now
    cap = first_pause + timedelta(minutes=wait_minutes)

    for minute in range(1, wait_minutes):
        fx.now = first_pause + timedelta(minutes=minute)
        await owner_writes(fx, message_id=f"owner-{minute}")
        assert message.id in await ready_ids(fx)
        conversation(fx, f"reply-{minute}")
        fx.shift(seconds=6)
        assert message.id not in await ready_ids(fx)
        held = last_verdict(fx, message)
        assert held["until"] <= cap.isoformat()
        for _ in range(5):
            fx.shift(seconds=0.1)
            assert message.id not in await ready_ids(fx)
            assert last_verdict(fx, message)["id"] == held["id"]
    assert len(decisions(fx, message)) == 1 + 2 * (wait_minutes - 1)

    fx.now = cap
    conversation(fx, "reply-at-cap")
    assert message.id in await ready_ids(fx)
    offered = last_verdict(fx, message)
    fx.restart()
    for index in range(10):
        fx.shift(seconds=1)
        conversation(fx, f"reply-after-cap-{index}")
        assert message.id in await ready_ids(fx)
        assert last_verdict(fx, message)["id"] == offered["id"]
