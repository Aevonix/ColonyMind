"""Retained owner origins do not override invalidation or unreadable evidence."""

from contextlib import closing

import pytest

from colonymind.turns.source_attribution import initialize
from test_ask_pause import fx, last_verdict, ready_ids, request_hand_ask
from test_hand_ask_origin_retention import retire_controller
from test_mind_loop import pinned_clock  # noqa: F401


@pytest.mark.parametrize("platform", ["rcs", "whatsapp", "sms", "api"])
async def test_invalidated_source_revokes_a_retained_owner_exemption(fx, platform):
    message, _ = await request_hand_ask(fx, platform=platform)
    assert message.id in await ready_ids(fx)
    first = last_verdict(fx, message)
    retire_controller(fx)
    with closing(fx.ledger._connect()) as db:
        initialize(db)
        db.execute("INSERT INTO source_attribution_invalidations VALUES (?,?,?)",
                   ("owner-request", "corrected-source", "correction"))
        db.commit()
    fx.restart()
    assert message.id not in await ready_ids(fx)
    held = last_verdict(fx, message)
    assert held["id"] != first["id"]
    assert (held["decision"], held["stage"]) == ("wait", "pause")
    for _ in range(3):
        fx.shift(seconds=1)
        assert message.id not in await ready_ids(fx)
        assert last_verdict(fx, message)["id"] == held["id"]


@pytest.mark.parametrize("restart", [False, True])
async def test_corrupt_controller_holds_even_with_retained_origin_and_send_verdict(fx, restart):
    message, _ = await request_hand_ask(fx)
    assert message.id in await ready_ids(fx)
    first = last_verdict(fx, message)
    (fx.state / "hands.db").write_bytes(b"unreadable controller fixture")
    if restart:
        fx.restart()
    for _ in range(3):
        fx.shift(seconds=1)
        assert message.id not in await ready_ids(fx)
        assert last_verdict(fx, message) == first


@pytest.mark.parametrize("already_verified", [False, True])
async def test_rejected_origin_write_holds_without_recording_a_gate_verdict(fx, already_verified):
    message, _ = await request_hand_ask(fx)
    if already_verified:
        assert message.id in await ready_ids(fx)
    first = last_verdict(fx, message)
    db = fx.mind.owner_gate.ledger.conn
    db.execute("CREATE TRIGGER reject_origin BEFORE INSERT ON hand_ask_origins "
               "BEGIN SELECT RAISE(ABORT, 'origin fixture write rejected'); END")
    db.commit()
    for _ in range(3):
        fx.shift(seconds=1)
        assert message.id not in await ready_ids(fx)
        assert last_verdict(fx, message) == first
    db.execute("DROP TRIGGER reject_origin")
    db.commit()
    assert message.id in await ready_ids(fx), "a recovered origin store releases the ask"
