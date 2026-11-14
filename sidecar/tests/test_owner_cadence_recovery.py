"""Missed slots, resolved history and cancellation of an owner/interval group."""

import asyncio
import sqlite3
from datetime import timedelta

import pytest

from colonymind.mind import Mind, outreach
from test_mind_outreach_loop import H, OWNER, T0, make  # noqa: F401
from test_mind_owner_cadence import DAY, cadence_row, create_cadence, snapshot


@pytest.mark.parametrize("days", [1, 7])
async def test_restart_before_slot_skips_missed_slots(make, days):
    fx = make()
    interval = days * DAY
    row = create_cadence(fx, created=T0 - interval, minutes=days * 1440)
    await fx.tick()
    fx.now = T0 + 3 * interval - timedelta(minutes=1)
    fx.mind = Mind(config=fx.mind.config, store=fx.store, state_dir=fx.store._state_dir,
                   owner_id=OWNER, commitments=fx.commitments, feedback=fx.feedback, ledger=fx.ledger,
                   clock=lambda: fx.now, backups=False)
    fx.mind.digest_hour = 25
    assert (await fx.tick())["formed"] == []
    assert snapshot(fx, [row]).cadences[0].due_at == T0 + 3 * interval
    fx.shift(timedelta(minutes=1))
    assert len((await fx.tick())["formed"]) == 1
    fx.shift(outreach.MIN_GAP)
    assert (await fx.tick())["formed"] == []
    assert len(fx.outreach_rows()) == 2


@pytest.mark.parametrize("minutes", [240, 1440, 7 * 1440])
@pytest.mark.parametrize("late", [False, True])
async def test_first_slot_grace_uses_half_interval_capped_at_three_hours(make, minutes, late):
    fx = make()
    interval = timedelta(minutes=minutes)
    row = create_cadence(fx, created=T0 - 3 * interval, minutes=minutes)
    grace = min(interval / 2, 3 * H)
    fx.now = T0 + grace + (timedelta(seconds=1) if late else timedelta(0))
    assert len((await fx.tick())["formed"]) == int(not late)
    assert snapshot(fx, [row]).cadences[0].due_at == T0 + interval
    fx.now = T0 + interval
    assert len((await fx.tick())["formed"]) == 1


async def test_min_gap_hold_keeps_slot_and_later_days_keep_phase(make):
    fx = make()
    row = create_cadence(fx, created=T0 - DAY)
    earlier = T0 - timedelta(minutes=1)
    fx.store.create_intention(kind="message", type="outreach_finding", title="A finding", drive="curiosity",
        cls="internal", decision="act", decision_reason="finding", status="sent", dedup_key=None,
        recipient=OWNER, created_at=earlier)
    assert (await fx.tick())["formed"] == []
    fx.now = earlier + outreach.MIN_GAP
    assert len((await fx.tick())["formed"]) == 1
    assert snapshot(fx, [row]).cadences[0].due_at == T0 + DAY
    fx.now = T0 + DAY
    assert len((await fx.tick())["formed"]) == 1


async def test_legacy_late_attempt_keeps_half_interval_between_attempts(make):
    fx = make()
    row = create_cadence(fx, created=T0 - DAY)
    fx.store.create_intention(kind="message", type="outreach_loop", title="Requested check-in", drive="duty",
        cls="internal", decision="act", decision_reason="requested", status="sent", dedup_key=None,
        recipient=OWNER, source_type="commitment", source_id=row["id"], created_at=T0 + 14 * H)
    fx.now = T0 + DAY
    assert (await fx.tick())["formed"] == []
    assert snapshot(fx, [row]).cadences[0].due_at == T0 + 2 * DAY
    fx.shift(DAY)
    assert len((await fx.tick())["formed"]) == 1


async def test_resolved_cadence_history_reader_is_read_only_and_batched(make, monkeypatch):
    fx = make()
    first = create_cadence(fx, created=T0 - 10 * DAY)
    for index in range(3):
        old = fx.owner_commitment("Requested check-in", created=T0 - (5 + index) * DAY,
            **{**cadence_row()["metadata"], "topic": f"topic {index}"})
        fx.store.create_intention(kind="message", type="outreach_loop", title="Requested check-in", drive="duty",
            cls="internal", decision="act", decision_reason="requested", status="sent", dedup_key=None,
            recipient=OWNER, source_type="commitment", source_id=old["id"], created_at=T0)
        fx.commitments.resolve(old["id"], outcome="duplicate")
    opened, queries = [], []
    connect = sqlite3.connect

    def observed(path, *args, **kwargs):
        db = connect(path, *args, **kwargs)
        if fx.commitments._db_path.name in str(path):
            assert str(path).endswith("?mode=ro") and kwargs["uri"]
            opened.append(db)

            def trace(statement):
                if statement.startswith("SELECT * FROM commitments"):
                    queries.append(db.execute("PRAGMA query_only").fetchone()[0])

            db.set_trace_callback(trace)
        return db

    monkeypatch.setattr(sqlite3, "connect", observed)
    cadences = await asyncio.to_thread(fx.mind._owner_cadences, [first])
    assert cadences == [outreach.Cadence(first["id"], "the garden", T0 + DAY)]
    assert len(opened) == len(queries) == 1
    assert queries == [1]


async def test_cancelling_daily_preserves_weekly_and_unrelated_work(make, monkeypatch):
    fx = make()
    monkeypatch.setattr("colonymind.commitments.store.now_utc", lambda: fx.now)
    first = create_cadence(fx, created=T0 - DAY)
    newer = fx.owner_commitment("Requested daily check-in", created=T0 - DAY + timedelta(minutes=2),
                                **{**cadence_row()["metadata"], "topic": "the workshop"})
    weekly = create_cadence(fx, created=T0 - 7 * DAY, minutes=7 * 1440)
    unrelated = fx.owner_commitment("Prepare the report")
    fx.commitments.resolve(newer["id"], outcome="wont_do")
    assert fx.commitments.get(first["id"])["status"] == "cancelled"
    assert fx.commitments.get(unrelated["id"])["status"] == "pending"
    await fx.tick()
    sent, = fx.outreach_rows()
    assert sent.source_id == weekly["id"]
    for days in (1, 2, 7):
        fx.now = T0 + days * DAY + timedelta(minutes=2)
        await fx.tick()
    requested = [row for row in fx.outreach_rows() if row.context.get("owner_cadence")]
    assert [row.source_id for row in requested] == [weekly["id"], weekly["id"]]
    assert fx.commitments.get(unrelated["id"])["status"] == "pending"


@pytest.mark.parametrize("restart_after", [11 * H, DAY - timedelta(minutes=1), DAY])
async def test_queued_check_in_expires_on_restart_and_next_slot_keeps_its_time(make, restart_after):
    fx = make()
    row = create_cadence(fx, created=T0 - DAY)
    await fx.mind.tick(force=True)
    queued, = fx.outreach_rows()
    assert queued.source_id == row["id"] and queued.status == "approved"
    assert queued.expires_at == T0 + 3 * H
    fx.now = T0 + restart_after
    fx.mind = Mind(config=fx.mind.config, store=fx.store, state_dir=fx.store._state_dir,
                   owner_id=OWNER, commitments=fx.commitments, feedback=fx.feedback, ledger=fx.ledger,
                   clock=lambda: fx.now, backups=False)
    fx.mind.digest_hour = 25
    await fx.tick()
    assert fx.store.get(queued.id).status == "expired"
    if restart_after < DAY:
        assert fx.sent == []
        fx.now = T0 + DAY
        await fx.tick()
    sent, = [item for item in fx.outreach_rows() if item.status == "sent"]
    assert sent.id != queued.id and sent.source_id == row["id"]
    assert sent.completed_at == T0 + DAY
    assert fx.store.get(queued.id).verdict is None


async def test_queued_check_in_still_delivers_two_hours_after_slot(make):
    fx = make()
    create_cadence(fx, created=T0 - DAY)
    await fx.mind.tick(force=True)
    queued, = fx.outreach_rows()
    fx.shift(2 * H)
    await fx.tick()
    assert [item["id"] for item in fx.sent] == [queued.id]
    assert fx.store.get(queued.id).status == "sent"


@pytest.mark.parametrize("minutes", [240, 1440, 7 * 1440])
async def test_late_formation_does_not_extend_slot_expiry(make, minutes):
    fx = make()
    interval = timedelta(minutes=minutes)
    create_cadence(fx, created=T0 - interval, minutes=minutes)
    fx.shift(H)
    await fx.mind.tick(force=True)
    queued, = fx.outreach_rows()
    assert queued.expires_at == T0 + min(interval / 2, 3 * H)
    fx.now = queued.expires_at + timedelta(seconds=1)
    assert await fx.mind.outbox_ready() == []
    await fx.tick()
    assert fx.store.get(queued.id).status == "expired"
    assert fx.sent == []
    fx.now = T0 + interval
    await fx.tick()
    assert len(fx.sent) == 1


@pytest.mark.parametrize("create_first", [False, True])
@pytest.mark.parametrize("outcome", ["obsolete", "wont_do", "cancelled"])
async def test_replacement_survives_both_orders_and_uses_evening_slot(make, monkeypatch, create_first, outcome):
    fx = make()
    monkeypatch.setattr("colonymind.commitments.store.now_utc", lambda: fx.now)
    old = create_cadence(fx, created=T0 - 21 * DAY)
    await fx.mind.tick(force=True)
    fx.shift(H)

    def create():
        return fx.owner_commitment("Check in every evening instead", created=fx.now, due=T0 + 6 * H,
                                   **{**cadence_row()["metadata"], "topic": "the evening"})

    if create_first:
        new = create()
    if outcome == "cancelled":
        fx.commitments.update(old["id"], status="cancelled")
    else:
        fx.commitments.resolve(old["id"], outcome=outcome)
    if not create_first:
        new = create()
    assert fx.commitments.get(new["id"])["status"] == "pending"
    assert await fx.mind.outbox_ready() == []
    # The cancelled attempt still consumes its slot and the half-interval gap.
    fx.now = T0 + 6 * H
    assert (await fx.tick())["formed"] == []
    evening = T0 + DAY + 6 * H
    fx.now = evening - timedelta(seconds=1)
    assert (await fx.tick())["formed"] == []
    fx.shift(timedelta(seconds=1))
    await fx.tick()
    sent, = [item for item in fx.outreach_rows() if item.status == "sent"]
    assert sent.source_id == new["id"] and sent.completed_at == evening
    fx.now = T0 + 2 * DAY
    assert (await fx.tick())["formed"] == []
    fx.now += 6 * H
    await fx.tick()
    assert len(fx.sent) == 2
    assert all(item.source_id == new["id"] for item in fx.outreach_rows() if item.status == "sent")
