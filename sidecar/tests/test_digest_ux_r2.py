"""The owner's questions and digest: delivery, answer validation and recovery.

One question is answerable at a time, and only after its confirmed delivery; a turn is read once, globally; an
answer is claimed atomically and revalidated against the exact live pair; a memory change that fails leaves the
question open and is retried; classification runs off the turn's critical path and is charged to the day's token
budget. Every overlapping pair of statements is checked. An ask that authorizes a destination shows it in full or
cannot be approved. A digest's items are recovered from its own manifest, and no digest arrives on a later day
than its own. Older contradiction asks move to the question protocol. No cut word and no internal id is shown.
"""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import closing
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from colonymind.mind import digest as digest_text
from colonymind.mind.consolidate import QUESTION_KEY, TASK_DIGEST
from test_digest_ux import (CODE, TurnRouter, acted, asked, deliver, items_are_whole, night, no_field_names,
                            the_digest)
from test_mind_consolidate import CONTACT, OWNER, Fixture as NightFixture
from owner_answer import owner_answers

ANSWER = "mind_question_answer"


def calls(fx, task=ANSWER):
    return [(messages, context) for messages, context in fx.router.calls if context.get("task") == task]


def shown_question(messages):
    return json.loads(messages[-1]["content"])


async def said(fx, text, turn, *, minutes=1, session="chat-1"):
    """An owner turn a minute later, and its reading (off the turn's path) awaited."""
    fx.shift(minutes=minutes)
    await fx.mind.owner_turn(text, turn_id=turn, session_id=session)
    await fx.mind.questions.drain()


async def office(fx, *, deliver_it=True):
    a = await fx.fact("o-1", OWNER, "s-1", "My office is room 4.", "room 4")
    fx.shift(hours=1)
    b = await fx.fact("o-2", OWNER, "s-2", "My office is room 7.", "room 7")
    await fx.mind.consolidate()
    await fx.mind.tick(force=True)
    question, = fx.messages("contradiction")
    if deliver_it:
        assert [p["id"] for p in await deliver(fx)] == [question.id]
    return question, a, b


async def desk(fx):
    c = await fx.fact("d-1", OWNER, "s-3", "My desk is number 12.", "number 12", predicate="desk_number")
    fx.shift(hours=1)
    d = await fx.fact("d-2", OWNER, "s-4", "My desk is number 30.", "number 30", predicate="desk_number")
    await fx.mind.consolidate()
    await fx.mind.tick(force=True)
    row = next(row for row in fx.messages("contradiction") if "desk" in row.description)
    return row, c, d


def live(fx, *ids):
    rows = {row["id"]: row for row in fx.claims()}
    return {ident for ident in ids if rows[ident]["retracted_by"] is None}


def state(fx, row):
    return fx.store.get(row.id).result_metadata["question"]


# ---------------------------------------------------------------------------
# #5 and #9: only a confirmed delivery makes a question answerable; an unsent one goes to the digest
# ---------------------------------------------------------------------------

async def test_an_uncertain_send_is_not_answerable_until_a_digest_delivers_it(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch, mind_question_answer={"answer": "second"})
    question, a, b = await office(fx, deliver_it=False)
    fx.mind.outbox.sending(question.id, target="telegram:owner-handle")
    fx.restart()                                                   # a crash between the claim and the send
    assert fx.store.get(question.id).status == "uncertain"
    await said(fx, "The second one.", "t-1")
    assert calls(fx) == [] and live(fx, a, b) == {a, b}            # never seen: nothing is read
    fx.shift(hours=1)
    await fx.mind.tick(force=True)
    listed = await the_digest(fx)
    assert "room 7" in " ".join(digest_text.lines_of(listed.context["text"], digest_text.QUESTION))
    assert [p["id"] for p in await deliver(fx)] == [listed.id]
    await said(fx, "The second one.", "t-2")
    assert len(calls(fx)) == 1 and live(fx, a, b) == {b}


async def test_a_failed_send_goes_to_the_digest_instead_of_closing(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch)
    question, _, _ = await office(fx, deliver_it=False)
    fx.mind.outbox.sending(question.id, target="telegram:owner-handle")
    fx.mind.outbox.sent(question.id, result="failed", error="the platform refused it")
    fx.shift(hours=1)
    await fx.mind.tick(force=True)
    listed = await the_digest(fx)
    assert "room 7" in " ".join(digest_text.lines_of(listed.context["text"], digest_text.QUESTION))


# ---------------------------------------------------------------------------
# #1: one question is answerable at a time; the reply is read against it with its conversation
# ---------------------------------------------------------------------------

async def test_one_question_at_a_time_and_a_reply_binds_only_to_the_delivered_one(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch, mind_question_answer={"answer": "second"})
    first, a, b = await office(fx)
    fx.shift(hours=1)
    second, c, d = await desk(fx)
    assert state(fx, second)["state"] == "waiting" and second.status != "approved"   # never sent alongside
    assert await deliver(fx) == []
    fx.ledger.record_source("chat-0", contact_id=OWNER, session_id="chat-1", occurred_at=fx.now.isoformat(),
                            messages=[{"role": "user", "content": "Which colour do you like for the banner?"},
                                      {"role": "assistant", "content": "Blue or green, you choose."}])
    await said(fx, "The second one.", "t-1")
    (messages, _), = calls(fx)
    data = shown_question(messages)
    assert "room 7" in data["question"] and "number 30" not in data["question"]
    assert any("banner" in line for line in data["earlier"])        # the conversation it was said in
    assert live(fx, a, b) == {b} and live(fx, c, d) == {c, d}
    fx.shift(hours=1)
    await fx.mind.tick(force=True)                                  # the next question waits for the digest
    assert state(fx, second)["state"] == "digest"
    listed = await the_digest(fx)
    q, = digest_text.lines_of(listed.context["text"], digest_text.QUESTION)
    assert "number 30" in q and "room" not in q


async def test_a_reply_to_something_else_changes_nothing(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch, mind_question_answer={"answer": "none"})
    question, a, b = await office(fx)
    await said(fx, "The second one, the green banner.", "t-1")
    assert len(calls(fx)) == 1 and live(fx, a, b) == {a, b}
    assert state(fx, question)["state"] == "sent"


async def test_a_turn_before_the_delivery_is_never_read(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch, mind_question_answer={"answer": "first"})
    question, a, b = await office(fx, deliver_it=False)
    await fx.mind.owner_turn("The first one.", turn_id="t-0", session_id="chat-1",
                             occurred_at=(fx.now - timedelta(minutes=5)).isoformat())
    await deliver(fx)
    await fx.mind.owner_turn("The first one.", turn_id="t-0b", session_id="chat-1",
                             occurred_at=(fx.now - timedelta(minutes=5)).isoformat())
    await fx.mind.questions.drain()
    assert calls(fx) == [] and live(fx, a, b) == {a, b}


# ---------------------------------------------------------------------------
# #2: a turn is read once, globally
# ---------------------------------------------------------------------------

async def test_a_retried_turn_never_answers_the_next_question(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch, mind_question_answer={"answer": "second"})
    first, a, b = await office(fx)
    fx.shift(hours=1)
    second, c, d = await desk(fx)
    await said(fx, "The second one.", "t-1")
    assert live(fx, a, b) == {b}
    fx.shift(hours=1)
    await fx.mind.tick(force=True)
    listed = await the_digest(fx)
    assert [p["id"] for p in await deliver(fx)] == [listed.id]
    await said(fx, "The second one.", "t-1")                        # the same turn again
    assert len(calls(fx)) == 1 and live(fx, c, d) == {c, d}
    await said(fx, "The first one.", "t-2")
    assert len(calls(fx)) == 2 and live(fx, c, d) == {d}            # (the model answers "second")


# ---------------------------------------------------------------------------
# #3: concurrent answers: one resolution, revalidated against the exact live pair
# ---------------------------------------------------------------------------

class RacingRouter(TurnRouter):
    """Holds every answer call until two are in flight, then answers each from its own reply."""

    def __init__(self, answers):
        super().__init__(answers)
        self.waiting, self.both = 0, asyncio.Event()

    async def complete(self, messages, *, context=None, **kwargs):
        if (context or {}).get("task") == ANSWER:
            self.calls.append((messages, context))
            self.waiting += 1
            if self.waiting >= 2:
                self.both.set()
            await asyncio.wait_for(self.both.wait(), 5)
            reply = shown_question(messages)["reply"]
            answer = "first" if "first" in reply else "second"
            return SimpleNamespace(content=json.dumps({"answer": answer}), usage={"total_tokens": 10})
        return await super().complete(messages, context=context, **kwargs)


async def test_concurrent_answers_resolve_once_and_never_retract_both(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fx = NightFixture(tmp_path, router=RacingRouter({"mind_claim_relation": {"relation": "replaces"}}))
    fx.router.answers[TASK_DIGEST] = None
    question, a, b = await office(fx)
    fx.shift(minutes=1)
    await fx.mind.owner_turn("Keep the first.", turn_id="t-1", session_id="chat-1")
    await fx.mind.owner_turn("Keep the second.", turn_id="t-2", session_id="chat-2")
    await fx.mind.questions.drain()
    assert len(calls(fx)) == 2
    kept = live(fx, a, b)
    assert len(kept) == 1
    answer = state(fx, question)["answer"]
    assert kept == ({a} if answer == "first" else {b})


# ---------------------------------------------------------------------------
# #4: the question closes only once the memory change succeeded; a failed one is retried
# ---------------------------------------------------------------------------

async def test_a_failed_memory_change_keeps_the_question_open_and_is_retried(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch, mind_question_answer={"answer": "second"})
    question, a, b = await office(fx)
    settle = fx.mind.consolidation.settle_question

    def broken(*args, **kwargs):
        raise RuntimeError("the ledger is locked")
    monkeypatch.setattr(fx.mind.consolidation, "settle_question", broken)
    await said(fx, "The second one.", "t-1")
    assert state(fx, question)["state"] != "resolved" and live(fx, a, b) == {a, b}
    assert fx.mind.concerns.by_key(question.dedup_key).status != "resolved"
    monkeypatch.setattr(fx.mind.consolidation, "settle_question", settle)
    fx.shift(minutes=5)
    await fx.mind.tick(force=True)                                  # the owner's answer is applied now
    assert state(fx, question)["state"] == "resolved" and live(fx, a, b) == {b}


# ---------------------------------------------------------------------------
# The reading is off the turn's path and charged to the day's budget
# ---------------------------------------------------------------------------

class HeldRouter(TurnRouter):
    def __init__(self, answers):
        super().__init__(answers)
        self.release = asyncio.Event()

    async def complete(self, messages, *, context=None, **kwargs):
        if (context or {}).get("task") == ANSWER:
            await asyncio.wait_for(self.release.wait(), 5)
        return await super().complete(messages, context=context, **kwargs)


async def test_the_reading_never_delays_the_turn_and_its_tokens_are_charged(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fx = NightFixture(tmp_path, router=HeldRouter({"mind_claim_relation": {"relation": "replaces"},
                                                   ANSWER: {"answer": "second"}}))
    fx.router.answers[TASK_DIGEST] = None
    question, a, b = await office(fx)
    before = fx.store.tokens_since(fx.now - timedelta(days=1))
    fx.shift(minutes=1)
    await asyncio.wait_for(fx.mind.owner_turn("The second one.", turn_id="t-1", session_id="chat-1"), 1)
    assert live(fx, a, b) == {a, b}                                  # returned before the decision
    fx.router.release.set()
    await fx.mind.questions.drain()
    assert live(fx, a, b) == {b}
    assert fx.store.tokens_since(fx.now - timedelta(days=1)) >= before + 10


# ---------------------------------------------------------------------------
# Digest summaries are bounded in number and time, and charged
# ---------------------------------------------------------------------------

async def test_digest_summaries_are_bounded_in_number_and_charged(tmp_path, monkeypatch):
    reason = ("the archive service refused every request after the first two hundred rows and kept refusing until "
              "the retry budget ran out, well before the service recovered, so none of the remaining figures were "
              "exported and the board pack for the quarter still lacks its appendix tables")
    fx = night(tmp_path, monkeypatch,
               mind_plain_summary={"sentence": "The export stopped when the archive service refused requests."})
    for index in range(10):
        acted(fx, f"f{index}", f"Export the board figures {index + 1}", outcome="failed", status="failed",
              reason=reason)
    before = fx.store.tokens_since(fx.now - timedelta(days=1))
    text = (await the_digest(fx)).context["text"]
    from colonymind.mind.outbox import SUMMARY_CALLS
    assert len(calls(fx, "mind_plain_summary")) == SUMMARY_CALLS < 10
    assert text.count("is too long to show here") == 10 - SUMMARY_CALLS
    assert fx.store.tokens_since(fx.now - timedelta(days=1)) >= before + 100 * SUMMARY_CALLS
    items_are_whole(text)


# ---------------------------------------------------------------------------
# #6: every overlapping pair is checked; compatibility is not transitive
# ---------------------------------------------------------------------------

def by_units(messages, context):
    data = json.loads(messages[-1]["content"])
    pair = {data["older"]["said"], data["newer"]["said"]}
    units = {"metric" in text for text in pair} | {"imperial" in text for text in pair}
    return {"relation": "replaces" if any("metric" in t for t in pair) and any("imperial" in t for t in pair)
            else "adds"} if units else {"relation": "adds"}


async def test_an_incompatible_pair_is_found_past_compatible_neighbours(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch, mind_claim_relation=by_units)
    await fx.fact("r-1", OWNER, "s-1", "My reports use metric units.", "metric units", predicate="report_format")
    fx.shift(hours=1)
    await fx.fact("r-2", OWNER, "s-2", "My reports have captions under charts.", "captions under charts",
                  predicate="report_format")
    fx.shift(hours=1)
    await fx.fact("r-3", OWNER, "s-3", "My reports use imperial units.", "imperial units", predicate="report_format")
    result = await fx.mind.consolidate()
    assert result["counts"].get("contradictions") == 1
    concern, = [c for c in fx.mind.concerns.open(limit=50) if c.kind == "question"]
    text = concern.detail["text"]
    assert "metric units" in text and "imperial units" in text and "captions" not in text


# ---------------------------------------------------------------------------
# #7: an ask that authorizes a destination shows it in full, or cannot be approved yet
# ---------------------------------------------------------------------------

def secret_use(fx, code, destination):
    return asked(fx, code, "Use the archive credential in the upload tool", type_="secret_use", kind="note",
                 context={"tool": "upload_file", "destination": destination, "notice": True})


async def test_a_secret_use_ask_shows_its_tool_and_destination_in_full(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch)
    destination = ("https://files.example.org/upload/archive/quarterly/board-pack/appendix-tables/"
                   + "section-" * 30 + "final")
    secret_use(fx, "K7F", destination)
    text = (await the_digest(fx)).context["text"]
    line, = digest_text.lines_of(text, digest_text.WAITING)            # what waits, in words; its message says more
    assert line == "Use the archive credential in the upload tool. Asked less than an hour ago."
    notice, = fx.mind.outbox.deliver_asks(fx.mind.outbox.asks_due(fx.store.intentions(status=["asked"], limit=10)))
    assert destination in notice.context["text"] and "upload_file" in notice.context["text"]


async def test_a_destination_too_long_to_show_blocks_the_yes_until_the_owner_has_it_in_full(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch)
    destination = "https://files.example.org/" + "a" * 3000
    ask = secret_use(fx, "K7F", destination)
    notice, = fx.mind.outbox.deliver_asks(fx.mind.outbox.asks_due([ask]))
    assert destination not in notice.context["text"] and "too long to show here" in notice.context["text"]
    with pytest.raises(PermissionError):
        await owner_answers(fx.mind, "K7F")
    full = [p for p in await deliver(fx) if destination in p["text"]]
    assert len(full) == 1                                           # the whole ask, in its own message
    assert (await owner_answers(fx.mind, "K7F")).status == "approved"
    assert fx.store.get(ask.id).status == "approved"


# ---------------------------------------------------------------------------
# #8 and the delivery date: membership from the digest's own manifest; one digest per delivery day
# ---------------------------------------------------------------------------

async def test_a_crash_after_queuing_keeps_the_digests_items_with_it(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch, autonomy="suggest", mind_question_answer={"answer": "second"})
    question, a, b = await office(fx, deliver_it=False)
    monkeypatch.setattr(fx.mind.questions, "mark_queued", lambda *args, **kwargs: None)
    monkeypatch.setattr(fx.mind.owner_gate.ledger, "mark_queued", lambda *args, **kwargs: None)
    listed = await the_digest(fx)
    assert [p["id"] for p in await deliver(fx)] == [listed.id]
    await said(fx, "The second one.", "t-1")
    assert live(fx, a, b) == {b}                                     # delivered with the digest: answerable
    fx.shift(days=1)
    await fx.mind.tick(force=True)
    later = await the_digest(fx)
    assert later is None or "room" not in later.context["text"]


async def test_no_digest_arrives_on_a_later_day_than_its_own(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    start = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
    fx = NightFixture(tmp_path, at=start, router=TurnRouter({"mind_claim_relation": {"relation": "replaces"}}))
    asked(fx, "K7F", "Re-run the archive export for the board figures")
    first = await the_digest(fx)                                     # queued, the body away
    fx.shift(hours=13)                                               # 01:00 the next day
    assert [p for p in await deliver(fx) if p["type"] == "digest"] == []
    fx.shift(hours=11)
    second = await the_digest(fx)
    delivered = [p for p in await deliver(fx) if p["type"] == "digest"]
    assert [p["id"] for p in delivered] == [second.id] and fx.store.get(first.id).status == "expired"
    days = [row.completed_at.date() for row in fx.messages("digest") if row.status == "sent"]
    assert len(days) == len(set(days)) == 1


# ---------------------------------------------------------------------------
# An older contradiction ask moves to the question protocol and loses its code
# ---------------------------------------------------------------------------

async def test_an_older_contradiction_ask_moves_to_the_question_protocol(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch, autonomy="suggest")
    a = await fx.fact("o-1", OWNER, "s-1", "My office is room 4.", "room 4")
    fx.shift(hours=1)
    b = await fx.fact("o-2", OWNER, "s-2", "My office is room 7.", "room 7")
    legacy, _ = fx.store.create_intention(
        kind="message", type="contradiction", title="Which is right for your office location: 'room 4' or 'room 7'?",
        drive="curiosity", cls="owner", decision="ask", decision_reason="suggest: owner -> ask", status="asked",
        dedup_key=f"{QUESTION_KEY}legacy0000000001", ask_code="Q4R", recipient=OWNER,
        context={"text": "You told me two things about your office location. Which is right?", "evidence": [a, b],
                 "notice": True}, hermes_kind="none", created_at=fx.now, expires_at=fx.now + timedelta(days=3))
    await fx.mind.tick(force=True)
    row = fx.store.get(legacy.id)
    assert row.ask_code is None and fx.mind.outbox.asks_due([row]) == []
    text = (await the_digest(fx)).context["text"]
    assert digest_text.lines_of(text, digest_text.WAITING) == []
    q, = digest_text.lines_of(text, digest_text.QUESTION)
    assert '"My office is room 4."' in q and '"My office is room 7."' in q and "Q4R" not in text


# ---------------------------------------------------------------------------
# No cut word from a capped title; never an internal contact id
# ---------------------------------------------------------------------------

async def test_a_goal_whose_stored_title_was_cut_is_never_shown_cut(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch)
    title = ("Goal: " + "work out which invoices from the spring and summer quarters still lack the associated "
             "reconciliation entries and prepare the corrections for the finance review")[:160]
    assert len(title) == 160 and not title.endswith(" ")
    goal, _ = fx.store.create_intention(
        kind="goal", type="goal", title=title, drive="curiosity", cls="self", decision="act",
        decision_reason="standard: self -> act", status="approved", dedup_key="test:goal", hermes_kind="none",
        context={"description": "Reconcile the spring invoices.", "tasks": 2}, created_at=fx.now,
        expires_at=fx.now + timedelta(days=7))
    asked(fx, "K7F", "Re-run the archive export for the board figures")
    text = (await the_digest(fx)).context["text"]
    assert title.split()[-1] not in text.split()                     # the word the cap cut in half
    goal_line, = [line for line in digest_text.lines_of(text, digest_text.DONE) if line.startswith("Still working")]
    assert re.search(r"\b\w+[.?!]", goal_line) and "(0/2" not in goal_line
    items_are_whole(text)


async def test_a_contact_without_a_name_is_named_by_a_handle_or_as_a_contact(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch)

    async def handles(contact_id):
        return [SimpleNamespace(gateway="sms", address="+15550100", is_primary=True, verified=True)]
    monkeypatch.setattr(fx.contacts, "get_handles", handles)
    asked(fx, "M4P", "Tell the contact the delivery date", kind="message", type_="commitment_notice",
          context={"text": "Delivery is Friday."}, recipient=CONTACT)
    text = (await the_digest(fx)).context["text"]
    assert CONTACT not in text and "+15550100" in text

    async def none(contact_id):
        return []
    monkeypatch.setattr(fx.contacts, "get_handles", none)
    fx.shift(days=1)
    text = (await the_digest(fx)).context["text"]
    assert CONTACT not in text and "Send a contact this message" in text
    no_field_names(text)
