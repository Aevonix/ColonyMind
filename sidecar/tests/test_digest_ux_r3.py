"""The owner's asks and digest: redelivery and the delivery day.

A full ask that did not arrive is sent again; and no second digest arrives on a day a late one already did.
"""

from __future__ import annotations

from datetime import datetime, timezone

from test_digest_ux import TurnRouter, asked, deliver, night, the_digest
from test_digest_ux_r2 import secret_use
from test_mind_consolidate import OWNER, Fixture as NightFixture
from owner_answer import owner_answers


# ---------------------------------------------------------------------------
# A full ask that did not arrive is sent again
# ---------------------------------------------------------------------------

async def test_a_full_ask_that_did_not_arrive_is_sent_again(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch)
    destination = "https://files.example.org/" + "b" * 3000
    secret_use(fx, "K7F", destination)
    await the_digest(fx)
    for payload in await fx.mind.outbox_ready():
        fx.mind.outbox.sending(payload["id"], target="telegram:owner-handle")
        fx.mind.outbox.sent(payload["id"], result="failed" if destination in payload["text"] else "sent")
    fx.shift(minutes=5)
    await fx.mind.tick(force=True)
    again = [p for p in await deliver(fx) if destination in p["text"]]
    assert len(again) == 1                                          # a new full ask, delivered this time
    assert (await owner_answers(fx.mind, "K7F")).status == "approved"


# ---------------------------------------------------------------------------
# One digest per delivery day: a late one for yesterday counts as today's
# ---------------------------------------------------------------------------

async def test_a_digest_that_finished_after_midnight_counts_for_that_day(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    start = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
    fx = NightFixture(tmp_path, at=start, router=TurnRouter({"mind_claim_relation": {"relation": "replaces"}}))
    asked(fx, "K7F", "Re-run the archive export for the board figures")
    first = await the_digest(fx)
    fx.shift(hours=11, minutes=59)                                   # 23:59: the body claims it
    assert fx.mind.outbox.sending(first.id, target="telegram:owner-handle") is not None
    fx.shift(minutes=6)                                              # 00:05 the next day: it arrives
    fx.mind.outbox.sent(first.id)
    fx.shift(hours=12)
    assert await the_digest(fx) is None                              # today already had its digest
    days = [row.completed_at.date() for row in fx.messages("digest") if row.status == "sent"]
    assert len(days) == len(set(days)) == 1
