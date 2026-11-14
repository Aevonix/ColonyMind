"""The questions' waiting, digest and expiry steps at every time of day, whatever the wall clock says.

The wall clock (``time.time``, which ``temporal.now_utc`` reads) is pinned to the mind's own clock, or pinned
eleven hours away from it: the mind reads only the clock it was given, so both run the same. Each run puts the
digest step at one local time of day, including just before and just after midnight, where the digest's own day
ends (it expires at the end of its local date) and the next day's digest begins.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from colonymind.mind import digest as digest_text
from colonymind.mind.questions import state_of
from test_digest_ux import TurnRouter, deliver, the_digest
from test_digest_ux_r2 import calls, desk, live, office, said


def state(fx, row):
    """The question's state as the mind reads it (recorded, or derived from its row)."""
    return state_of(fx.store.get(row.id))
from test_mind_consolidate import OWNER, Fixture as NightFixture

TIMES = ["00:00", "00:05", "07:59", "12:00", "20:16", "23:30", "23:59"]
WALL = [0, 11]          # hours the wall clock is pinned away from the mind's own


def world(tmp_path, monkeypatch, wall_hours, *, autonomy="standard"):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    start = datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)
    holder = {}
    monkeypatch.setattr(time, "time", lambda: (holder["fx"].now if "fx" in holder else start).timestamp()
                        + wall_hours * 3600)
    fx = NightFixture(tmp_path, at=start, autonomy=autonomy, router=TurnRouter({"mind_claim_relation": {"relation": "replaces"},
                                                             "mind_question_answer": {"answer": "second"}}))
    holder["fx"] = fx
    fx.router.answers["mind_consolidate_digest"] = None
    return fx


def at_next(fx, hhmm):
    """Move the mind's clock forward to the next local ``hhmm`` (at least an hour ahead)."""
    hour, minute = (int(part) for part in hhmm.split(":"))
    target = fx.now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    while target <= fx.now + timedelta(hours=1):
        target += timedelta(days=1)
    fx.now = target


def digests_carrying(fx, words):
    return [row for row in fx.messages("digest") if words in (row.context or {}).get("text", "")]


@pytest.mark.parametrize("wall_hours", WALL)
@pytest.mark.parametrize("hhmm", TIMES)
async def test_waiting_digest_and_expiry_steps_at_any_time_of_day(tmp_path, monkeypatch, hhmm, wall_hours):
    fx = world(tmp_path, monkeypatch, wall_hours)
    first, a, b = await office(fx, deliver_it=False)                 # the body is away
    fx.shift(hours=1)
    second, c, d = await desk(fx)
    assert state(fx, second)["state"] == "waiting"
    for _ in range(8):                                               # waiting never counts toward the expiry
        fx.shift(days=1)
        await fx.mind.tick(force=True)
    assert state(fx, second)["state"] == "waiting" and state(fx, first)["state"] == "digest"

    at_next(fx, hhmm)                                                # the digest step at this time of day
    listed = await the_digest(fx)
    assert "room 7" in " ".join(digest_text.lines_of(listed.context["text"], digest_text.QUESTION))
    assert [p["id"] for p in await deliver(fx)] == [listed.id]
    await said(fx, "The second one.", "t-1")                         # a minute later, perhaps the next day
    assert live(fx, a, b) == {b}

    fx.shift(hours=1)
    await fx.mind.tick(force=True)                                   # the next question moves on to the owner
    assert state(fx, second)["state"] in {"digest", "queued"}
    if not digests_carrying(fx, "number 30"):
        assert await the_digest(fx) is None                          # today already had its digest
        fx.shift(days=1)
        await fx.mind.tick(force=True)                               # so it goes in tomorrow's
    carrying, = digests_carrying(fx, "number 30")
    assert [p["id"] for p in await deliver(fx) if p["type"] == "digest"] == [carrying.id]
    assert state(fx, second)["state"] == "listed"

    fx.shift(hours=24, minutes=1)                                    # its window closes a day after delivery
    await fx.mind.tick(force=True)
    assert state(fx, second)["state"] == "expired"
    await said(fx, "The first one.", "t-2")
    assert len(calls(fx)) == 1 and live(fx, c, d) == {c, d}
    days = [row.completed_at.date() for row in fx.messages("digest") if row.status == "sent"]
    assert len(days) == len(set(days))                              # never two digests on one delivery day


@pytest.mark.parametrize("wall_hours", WALL)
async def test_a_digest_not_delivered_by_midnight_hands_its_question_to_the_next_day(tmp_path, monkeypatch,
                                                                                     wall_hours):
    fx = world(tmp_path, monkeypatch, wall_hours, autonomy="suggest")     # the question goes to the digest
    first, a, b = await office(fx, deliver_it=False)
    assert state(fx, first)["state"] == "digest"
    at_next(fx, "23:59")
    late = await the_digest(fx)
    assert "room 7" in late.context["text"] and state(fx, first)["state"] == "queued"
    fx.shift(minutes=2)                                              # 00:01: the body comes back
    assert [p for p in await deliver(fx) if p["type"] == "digest"] == []
    assert fx.store.get(late.id).status == "expired"
    await fx.mind.tick(force=True)                                   # the new day's digest (the tick's, or the next)
    today = next((row for row in digests_carrying(fx, "room 7") if row.id != late.id), None) or await the_digest(fx)
    assert today is not None and today.id != late.id and "room 7" in today.context["text"]
    assert [p["id"] for p in await deliver(fx) if p["type"] == "digest"] == [today.id]
    assert state(fx, first)["state"] == "listed"
