"""The mind's own corrections and the owner's asks: every copy retracted, and delivery receipts.

Keeping the newer statement retracts every live copy of the older value; and a full ask whose delivery was
recorded can be approved whatever the hook that marks it seen did.
"""

from __future__ import annotations

import pytest

from test_digest_ux import deliver, night
from test_digest_ux_r2 import live, secret_use
from test_mind_consolidate import OWNER
from owner_answer import owner_answers


# ---------------------------------------------------------------------------
# Keeping the newer statement retracts every live copy of the older value
# ---------------------------------------------------------------------------

async def twice_room_4(fx):
    """Room 4 said twice (two claims: a store that does not fold restatements), then room 7."""
    old = await fx.fact("o-0", OWNER, "s-0", "My office is room 4.", "room 4")
    fx.shift(hours=1)
    a = await fx.fact("o-1", OWNER, "s-1", "My office is room 4.", "room 4")
    fx.shift(hours=1)
    b = await fx.fact("o-2", OWNER, "s-2", "My office is room 7.", "room 7")
    assert old != a
    return old, a, b


async def test_keeping_the_newer_statement_retracts_every_copy_of_the_older(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch)
    old, a, b = await twice_room_4(fx)
    assert (await fx.mind.consolidate())["counts"]["contradictions"] == 1
    assert live(fx, old, a, b) == {b}
    shown = fx.assertions("office")                                # what recall shows a new session
    assert [item["value"] for bundle in shown for item in bundle["assertions"]] == ["room 7"]
    assert not any(bundle["status"] == "unresolved_conflict" for bundle in shown)


# ---------------------------------------------------------------------------
# A full ask recorded delivered can be approved, whatever its hook did
# ---------------------------------------------------------------------------

def failing_hook(row):
    raise RuntimeError("the hook failed")


def queue_ask_messages(fx):
    """The ask's own message, as the tick queues it: an ask too long to show sends its full text in its own."""
    return fx.mind.outbox.deliver_asks(fx.mind.outbox.asks_due(fx.store.intentions(status=["asked"], limit=10)))


async def full_ask_delivered_without_its_hook(fx, hook):
    destination = "https://files.example.org/" + "c" * 3000
    secret_use(fx, "K7F", destination)
    queue_ask_messages(fx)
    kept, fx.mind.outbox.on_sent = fx.mind.outbox.on_sent, hook   # the process stops, or the hook fails
    full = [p for p in await deliver(fx) if destination in p["text"]]
    assert len(full) == 1 and fx.store.get(full[0]["id"]).status == "sent"
    fx.mind.outbox.on_sent = kept
    return full[0]["id"]


@pytest.mark.parametrize("hook", [None, failing_hook], ids=["stopped", "failed"])
async def test_a_full_ask_recorded_delivered_can_be_approved_without_its_hook(tmp_path, monkeypatch, hook):
    fx = night(tmp_path, monkeypatch)
    await full_ask_delivered_without_its_hook(fx, hook)
    assert (await owner_answers(fx.mind, "K7F")).status == "approved"


async def test_a_full_ask_recorded_delivered_is_seen_after_a_restart(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch)
    notice = await full_ask_delivered_without_its_hook(fx, None)
    fx.mind.outbox.sent(notice)                                     # a repeated report changes nothing either
    fx.restart()
    ask, = fx.store.intentions(status=["asked"], limit=10)
    full = ask.context["full_view"]
    assert full["seen"] is True and full["seen_at"] == fx.store.get(notice).completed_at.isoformat()
    fx.shift(minutes=5)
    await fx.mind.tick(force=True)
    assert [p for p in await deliver(fx) if p["type"] == "ask_in_full"] == []   # never sent again
    assert (await owner_answers(fx.mind, "K7F")).status == "approved"


async def test_a_full_ask_not_delivered_still_blocks_the_yes(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch)
    destination = "https://files.example.org/" + "d" * 3000
    secret_use(fx, "K7F", destination)
    queue_ask_messages(fx)
    fx.restart()
    with pytest.raises(PermissionError):
        await owner_answers(fx.mind, "K7F")
