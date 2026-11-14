"""The owner's questions and asks: erasure, correction and delivery receipts.

A clarification never carries statements that were forgotten or a question that is over; an answer retracts every
live copy of the value it rejects; and a full ask whose delivery was recorded can be approved whatever the hook
that marks it seen did.
"""

from __future__ import annotations

import pytest

from test_digest_ux import deliver, night, the_digest
from test_digest_ux_r2 import live, office, said, secret_use, state
from test_mind_consolidate import OWNER
from owner_answer import owner_answers


# ---------------------------------------------------------------------------
# A clarification follows its question and the statements it quotes
# ---------------------------------------------------------------------------


async def test_a_clarification_is_withdrawn_once_its_question_is_answered(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch, mind_question_answer={"answer": "unclear"})
    question, a, b = await office(fx)
    await said(fx, "I need more details to answer this", "t-1")
    again, = fx.messages("question_again")
    assert again.context["question_id"] == question.id and set(again.context["claims"]) == {a, b}
    fx.router.answers["mind_question_answer"] = {"answer": "second"}
    await said(fx, "Room 7, the other one is old.", "t-2")          # answered before the body pulled it
    assert state(fx, question)["state"] == "resolved"
    assert fx.store.get(again.id).status == "cancelled"
    assert await deliver(fx) == []


# ---------------------------------------------------------------------------
# An answer retracts every live copy of the value it rejects
# ---------------------------------------------------------------------------

async def twice_room_4(fx):
    """Room 4 said twice (two claims: a store that does not fold restatements), then room 7."""
    old = await fx.fact("o-0", OWNER, "s-0", "My office is room 4.", "room 4")
    fx.shift(hours=1)
    question, a, b = await office(fx)
    assert old != a and set(question.context["question"]["claims"]) == {a, b}   # the newest room 4 is asked
    return question, old, a, b


async def test_keeping_the_second_retracts_every_copy_of_the_first(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch, mind_question_answer={"answer": "second"})
    question, old, a, b = await twice_room_4(fx)
    await said(fx, "Room 7, the other one is old.", "t-1")
    assert state(fx, question)["state"] == "resolved"
    assert live(fx, old, a, b) == {b}
    shown = fx.assertions("office")                                # what recall shows a new session
    assert [item["value"] for bundle in shown for item in bundle["assertions"]] == ["room 7"]
    assert not any(bundle["status"] == "unresolved_conflict" for bundle in shown)


async def test_neither_retracts_every_copy_of_both(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch, mind_question_answer={"answer": "neither"})
    question, old, a, b = await twice_room_4(fx)
    await said(fx, "Neither, I work from home now.", "t-1")
    assert state(fx, question)["state"] == "resolved" and live(fx, old, a, b) == set()


async def test_a_crash_after_the_whole_change_is_finished_not_repeated(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch)
    question, old, a, b = await twice_room_4(fx)
    assert fx.mind.questions.ledger.claim(question.id, "t-crash", "second", fx.now)
    from colonymind.mind.questions import asked_of
    assert fx.mind.consolidation.settle_question(asked_of(question), "second", ref=f"intention:{question.id}",
                                                 now=fx.now)
    fx.shift(minutes=5)
    await fx.mind.tick(force=True)
    current = state(fx, question)
    assert current["state"] == "resolved" and current["answer"] == "second" and live(fx, old, a, b) == {b}


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


from colonymind.memory import curated
async def test_a_clarification_of_forgotten_statements_is_never_sent(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch, mind_question_answer={'answer': 'unclear'})
    question, a, b = await office(fx)
    await said(fx, 'I need more details to answer this', 't-1')
    again, = fx.messages('question_again')
    assert again.status == 'approved'
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(fx.ledger)
    curated.forget_sources(fx.ledger, OWNER, source_ids=['o-1', 'o-2'])
    assert historical_bytes(fx.ledger) == _native_before_withdrawal
    fx.shift(minutes=1)
    await fx.mind.tick(force=True)
    assert state(fx, question)['state'] == 'resolved'
    sent = await deliver(fx)
    assert not any(('room' in payload['text'] for payload in sent)), sent
    assert fx.store.get(again.id).status == 'cancelled'

async def test_a_clarification_is_checked_again_when_the_body_pulls_it(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch, mind_question_answer={'answer': 'unclear'})
    await office(fx)
    await said(fx, 'I need more details to answer this', 't-1')
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(fx.ledger)
    curated.forget_sources(fx.ledger, OWNER, source_ids=['o-1'])
    assert historical_bytes(fx.ledger) == _native_before_withdrawal
    sent = await deliver(fx)
    assert not any(('room' in payload['text'] for payload in sent)), sent

async def test_a_question_whose_statements_were_forgotten_is_never_sent(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch)
    question, _, _ = await office(fx, deliver_it=False)
    assert fx.store.get(question.id).status == 'approved'
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(fx.ledger)
    curated.forget_sources(fx.ledger, OWNER, source_ids=['o-2'])
    assert historical_bytes(fx.ledger) == _native_before_withdrawal
    sent = await deliver(fx)
    assert not any(('room' in payload['text'] for payload in sent)), sent
