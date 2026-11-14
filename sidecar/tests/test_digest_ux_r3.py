"""The owner's questions and digest: recovery, redelivery and expiry.

A resolution claim left by a crash is recovered; a question delivered under the earlier state format stays
answerable; a full ask that did not arrive is sent again; a recovered delivery keeps its real time; the expiry
counts only from delivery; and no second digest arrives on a day a late one already did.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from colonymind.mind import digest as digest_text
from colonymind.mind.questions import asked_of
from test_digest_ux import TurnRouter, asked, deliver, night, the_digest
from test_digest_ux_r2 import calls, desk, live, office, said, secret_use, state
from test_mind_consolidate import OWNER, Fixture as NightFixture
from owner_answer import owner_answers


# ---------------------------------------------------------------------------
# A claim a crash left behind is recovered at the next tick, idempotently
# ---------------------------------------------------------------------------

async def test_a_claim_left_before_the_memory_change_is_reapplied(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch, mind_question_answer={"answer": "second"})
    question, a, b = await office(fx)
    assert fx.mind.questions.ledger.claim(question.id, "t-crash", "second", fx.now)   # then the process died
    await said(fx, "Room 7, since last week.", "t-2")
    assert live(fx, a, b) == {a, b}                                  # the fresh claim still holds it
    fx.shift(minutes=5)
    await fx.mind.tick(force=True)
    assert state(fx, question)["state"] == "resolved" and live(fx, a, b) == {b}


async def test_a_claim_left_after_the_memory_change_is_finished_not_repeated(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch)
    question, a, b = await office(fx)
    assert fx.mind.questions.ledger.claim(question.id, "t-crash", "first", fx.now)
    assert fx.mind.consolidation.settle_question(asked_of(question), "first", ref=f"intention:{question.id}",
                                                 now=fx.now)           # changed, then the process died
    fx.shift(minutes=5)
    await fx.mind.tick(force=True)
    current = state(fx, question)
    assert current["state"] == "resolved" and current["answer"] == "first" and live(fx, a, b) == {a}


# ---------------------------------------------------------------------------
# A question delivered under the earlier state format (sent_at, listed_at) stays answerable
# ---------------------------------------------------------------------------

async def test_a_question_delivered_before_the_upgrade_stays_answerable(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch, mind_question_answer={"answer": "second"})
    question, a, b = await office(fx)
    row = fx.store.get(question.id)
    older = {"state": "sent", "sent_at": fx.now.isoformat(), "asked_at": fx.now.isoformat()}
    fx.store.update(row.id, result_metadata={**(row.result_metadata or {}), "question": older})
    await fx.mind.tick(force=True)
    await said(fx, "The second one.", "t-1")
    assert live(fx, a, b) == {b}


async def test_a_question_listed_before_the_upgrade_moves_on_after_its_window(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch)
    question, _, _ = await office(fx)
    row = fx.store.get(question.id)
    older = {"state": "sent", "sent_at": fx.now.isoformat(), "asked_at": fx.now.isoformat()}
    fx.store.update(row.id, result_metadata={**(row.result_metadata or {}), "question": older})
    fx.shift(days=1, hours=1)
    await fx.mind.tick(force=True)
    assert state(fx, question)["state"] == "digest"                  # the daily step still runs


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
# A delivery recovered later keeps its real time
# ---------------------------------------------------------------------------

async def test_a_recovered_digest_delivery_keeps_its_real_time(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch, autonomy="suggest", mind_question_answer={"answer": "second"})
    question, a, b = await office(fx, deliver_it=False)
    listed = await the_digest(fx)
    hook, fx.mind.outbox.on_sent = fx.mind.outbox.on_sent, None      # delivered, then the process died
    assert [p["id"] for p in await deliver(fx)] == [listed.id]
    fx.mind.outbox.on_sent = hook
    fx.shift(hours=25)
    await fx.mind.tick(force=True)
    await fx.mind.tick(force=True)
    await said(fx, "The second one.", "t-late")
    assert calls(fx) == [] and live(fx, a, b) == {a, b}            # its window closed a day after the delivery


# ---------------------------------------------------------------------------
# The seven days count from the delivery, never from time spent waiting
# ---------------------------------------------------------------------------

async def test_a_question_waiting_behind_another_never_expires_before_it_is_asked(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch, mind_question_answer={"answer": "second"})
    first, a, b = await office(fx, deliver_it=False)                 # the body is away
    fx.shift(hours=1)
    second, c, d = await desk(fx)
    assert state(fx, second)["state"] == "waiting"
    for _ in range(8):
        fx.shift(days=1)
        await fx.mind.tick(force=True)
    assert state(fx, second)["state"] == "waiting"                   # still to be asked
    listed = await the_digest(fx)                                    # the body is back
    assert "room 7" in " ".join(digest_text.lines_of(listed.context["text"], digest_text.QUESTION))
    await deliver(fx)
    await said(fx, "The second one.", "t-1")
    assert live(fx, a, b) == {b}
    fx.shift(hours=1)
    await fx.mind.tick(force=True)
    # It moves on to the owner: waiting for the digest, or already in the next day's when this hour crossed local
    # midnight (the tick then builds that day's digest with it). Which one depends on the time of day the run
    # starts at, never on the wall clock (tests/test_digest_clock.py runs these steps at set times of day).
    assert state(fx, second)["state"] in {"digest", "queued"}
    carrying = [row for row in fx.messages("digest") if "number 30" in (row.context or {}).get("text", "")]
    if not carrying:
        fx.shift(days=1)
        carrying = [await the_digest(fx)]
    assert "number 30" in carrying[-1].context["text"]


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
