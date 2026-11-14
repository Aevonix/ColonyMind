"""The decision points the fast tier reads (``colonymind.instinct``): where the model's typed answer may change what
the existing path decided, and where its silence (shadow, failed, unsure) leaves it as it was."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from colonymind.commitments.extract import CommitmentExtractor
from colonymind.commitments.store import CommitmentStore
from colonymind.contacts.config import ContactsConfig
from colonymind.contacts.optout import apply_opt_out
from colonymind.contacts.store import SQLiteContactStore
from colonymind.instinct import FAST_POINTS, REGISTRY
from colonymind.instinct.calibration import Record
from colonymind.instinct.providers import Answer
from colonymind.instinct.providers.local_decision import LocalDecision
from colonymind.instinct.tiers import Instinct
from colonymind.mind import outreach
from colonymind.turns import TurnIdempotencyLedger
from test_commitment_first_mention_hold import _Router, _item
from test_mind_interest_settled import CaptureRouter, fx  # noqa: F401  (fx is a pytest fixture)
from test_mind_lessons_night import OWN_RULE, add, judged, label, make as lesson_fixture, night, two_requests
from test_mind_loop import OWNER, pinned_clock  # noqa: F401  (pinned_clock: autouse pytest fixture)
from test_mind_outreach_loop import make  # noqa: F401  (make is a pytest fixture)
from test_mind_outreach_reactions import reaction, say, shared

MEASURED = "typed-decisions@1c5edc17a7acd8701df6fc341c0d179f1c62c982"


class Provider:
    """A decision server that answers each point surely with ``answers[point]`` (a label), or not at all (None:
    a malformed answer), and keeps every question: ``calls`` is ``[(point, state)]``."""

    kind, model, private, max_state_chars, batch, timeout_ms = "local-decision.v1", "", True, 1200, False, 250
    types = frozenset({"yes_no", "choice"})

    def __init__(self, answers, *, delay=0.0):
        self.id, self.answers, self.delay, self.calls = "local", answers, delay, []

    async def answer(self, state, questions):
        self.calls += [(name, state) for name in questions]
        if self.delay:
            await asyncio.sleep(self.delay)
        out = {}
        for name in questions:
            label = self.answers.get(name)
            if label is None:
                raise ValueError("malformed")
            labels = FAST_POINTS[name].labels if FAST_POINTS[name].kind == "choice" else {"yes": 1, "no": 1}
            out[name] = {item: (0.99 if item == label else 0.01 / (len(labels) - 1)) for item in labels}
        return Answer(out, revision=MEASURED)


def Decider(**answers):
    """Instinct with explicit synthetic records, over a server that answers as told."""
    records = [Record(point=name, version=REGISTRY[name].version, tier="fast",
                      provider_kind="local-decision.v1", revision=MEASURED,
                      bars={label: 0.8 for label in REGISTRY[name].labels}, source="synthetic wiring")
               for name in answers]
    engine = Instinct(providers={"local": Provider(answers)}, fast="local", records=records)
    for point in answers:
        engine.set_state(point, "fast", "live", source="test")
    return engine


def calls(engine):
    return engine.provider_for("fast").calls


def served(answer, *, point, truncated=False):
    """The real ``local-decision.v1`` client, with ``point`` live, over an endpoint that always gives ``answer``."""
    body = {"protocol": "local-decision.v1", "backend": {"name": "fake", "checkpoint": "typed-decisions",
                                                          "model_revision": MEASURED.split("@")[1]},
            "elapsed_ms": 3.0, "usage": {"input_tokens": 40, "output_tokens": 0}, "input_truncated": truncated,
            "answers": {"decision": answer}}
    provider = LocalDecision("local", url="http://decide.test",
                             transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)))
    engine = Instinct(providers={"local": provider}, fast="local")
    engine.set_state(point, "fast", "live", source="test")
    return engine


def failed(engine, point):
    return engine.health()["points"][point]["counts"].get("fast", {}).get("failed", 0)


# -- the owner's reply to an outreach -----------------------------------------------------------------------

NO_CUE = "Ah, the tidal energy item you sent reminds me of a trip I took."


async def test_a_linked_reply_the_phrases_do_not_read_takes_the_decision_models_reading(make):
    fx = make()
    row = await shared(fx)
    fx.shift(timedelta(minutes=5))
    fx.mind.instinct = Decider(outreach_reply="not_interested")
    await say(fx, NO_CUE, "t-1", "owner-2")
    assert reaction(fx, row)["class"] == "negative" and reaction(fx, row)["read"] == "decision"
    assert fx.store.get(row.id).verdict == "not_useful"
    assert fx.mind.mind_state.get("outreach.mute:tidal-energy")["level"] == 1.0
    [(point, state)] = calls(fx.mind.instinct)
    assert point == "outreach_reply" and NO_CUE in state and "tidal energy" in state


async def test_no_answer_from_the_decision_model_leaves_the_reply_as_engagement(make):
    fx = make()
    row = await shared(fx)
    fx.shift(timedelta(minutes=5))
    fx.mind.instinct = Decider(outreach_reply=None)
    await say(fx, NO_CUE, "t-1", "owner-2")
    assert reaction(fx, row)["class"] == "engaged" and "read" not in reaction(fx, row)
    assert len(calls(fx.mind.instinct)) == 1


async def test_a_reply_the_phrases_read_is_never_sent_to_the_decision_model(make):
    fx = make()
    row = await shared(fx)
    fx.shift(timedelta(minutes=5))
    fx.mind.instinct = Decider(outreach_reply="stop")
    await say(fx, "That tidal energy item you sent was not useful to me.", "t-1", "owner-2")
    assert reaction(fx, row)["class"] == "negative" and calls(fx.mind.instinct) == []


async def test_a_stop_the_decision_model_reads_in_a_linked_reply_pauses_outreach(make):
    fx = make()
    row = await shared(fx)
    fx.shift(timedelta(minutes=5))
    fx.mind.instinct = Decider(outreach_reply="stop")
    summary = await say(fx, NO_CUE, "t-1", "owner-2")
    assert reaction(fx, row)["class"] == "stop" and "paused" in summary["applied"]
    assert outreach.pause_until(fx.mind.mind_state.get(outreach.PAUSE_KEY)) is not None


async def test_the_decision_models_stop_never_overrides_the_typed_decision_that_the_stop_is_someone_elses(make):
    """The instruction question is the typed decision's once it has answered: the reply model is not asked again."""
    from test_mind_typed_decisions import DecisionRouter
    fx = make()
    row = await shared(fx)
    fx.shift(timedelta(minutes=5))
    fx.mind.router = DecisionRouter(instruction="none")
    fx.mind.instinct = Decider(outreach_reply="stop")
    summary = await say(fx, 'About the tidal energy item: my boss wrote "stop checking in on the team". How do I answer?',
                        "t-1", "owner-2")
    assert "paused" not in summary["applied"] and outreach.pause_until(
        fx.mind.mind_state.get(outreach.PAUSE_KEY)) is None
    assert calls(fx.mind.instinct) == [] and reaction(fx, row)["class"] != "stop"


async def test_a_stop_read_from_a_malformed_choice_leaves_the_reply_as_engagement(make):
    fx = make()
    row = await shared(fx)
    fx.shift(timedelta(minutes=5))
    faint_stop = {"engaged": 0, "dig_deeper": 0, "not_interested": 0, "not_now": 0, "stop": 0.05}
    fx.mind.instinct = served({"type": "choice", "choice": "e",
                              "probabilities": dict(zip("abcde", faint_stop.values())), "confidence": 0.05},
                              point="outreach_reply")
    summary = await say(fx, NO_CUE, "t-1", "owner-2")
    assert reaction(fx, row)["class"] == "engaged" and "paused" not in summary["applied"]
    assert outreach.pause_until(fx.mind.mind_state.get(outreach.PAUSE_KEY)) is None
    assert failed(fx.mind.instinct, "outreach_reply") == 1


# -- a contact's opt-out ------------------------------------------------------------------------------------

@pytest.fixture
async def contacts():
    value = SQLiteContactStore(ContactsConfig(sqlite_path=":memory:"))
    await value.connect()
    yield value
    await value.close()


async def test_an_opt_out_the_phrases_miss_is_one_when_the_decision_model_says_so(contacts):
    owner = await contacts.create(display_name="Owner", may_contact="auto")
    contact = await contacts.create(display_name="Contact", may_contact="auto")
    decider = Decider(opt_out="yes")
    lowered = await apply_opt_out(contacts, contact.contact_id, "Lose my number.", source_ref="turn:1",
                                  owner_id=owner.contact_id, instinct=decider)
    assert lowered is not None and lowered.may_contact == "never"
    [row] = [row for row in await contacts.get_audit_log(contact.contact_id) if row["action"] == "opt_out"]
    assert "decision model" in row["detail"] and "Lose my number" not in row["detail"]
    assert calls(decider) == [("opt_out", 'Message to the assistant: "Lose my number."')]


async def test_no_answer_or_a_no_leaves_the_contact_as_the_phrases_did(contacts):
    owner = await contacts.create(display_name="Owner", may_contact="auto")
    contact = await contacts.create(display_name="Contact", may_contact="auto")
    for answer in (None, "no"):
        assert await apply_opt_out(contacts, contact.contact_id, "Lose my number.", source_ref="turn:1",
                                   owner_id=owner.contact_id, instinct=Decider(opt_out=answer)) is None
    assert (await contacts.get(contact.contact_id)).may_contact == "auto"


async def test_a_phrase_opt_out_and_the_owner_never_reach_the_decision_model(contacts):
    owner = await contacts.create(display_name="Owner", may_contact="auto")
    contact = await contacts.create(display_name="Contact", may_contact="auto")
    decider = Decider(opt_out="yes")
    assert await apply_opt_out(contacts, owner.contact_id, "Lose my number.", source_ref="turn:1",
                               owner_id=owner.contact_id, instinct=decider) is None
    lowered = await apply_opt_out(contacts, contact.contact_id, "Please remove me.", source_ref="turn:2",
                                  owner_id=owner.contact_id, instinct=decider)
    assert lowered.may_contact == "never" and calls(decider) == []


# -- a new item the person wants no reminders about ---------------------------------------------------------

def capture(tmp_path, monkeypatch, text):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    commitments = CommitmentStore(tmp_path / "commitments.db")
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    extractor = CommitmentExtractor(ledger, lambda: commitments)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    ledger.record_source("t-1", contact_id=OWNER, session_id="owner-1", occurred_at=now.isoformat(), messages=[
        {"role": "user", "content": text}, {"role": "assistant", "content": "Understood."}])
    return extractor, commitments, now


HANDLED = "The tax form for p-05 is due in two hours. I've got it, no pings about it please."


async def test_a_reminder_the_decision_model_reads_as_unwanted_is_created_held(tmp_path, monkeypatch):
    extractor, commitments, now = capture(tmp_path, monkeypatch, HANDLED)
    extractor.instinct = Decider(no_reminders="yes")
    due = (now + timedelta(hours=2)).isoformat()
    item = {**_item("Send p-05 the tax form", due_at=due), "metadata": {"heads_up_at": due}}
    assert await extractor.process_one(_Router([item])) is True
    [row] = commitments.get_pending_for_person(OWNER)
    assert row["due_at"] is None and "heads_up_at" not in (row["metadata"] or {})
    [(point, state)] = calls(extractor.instinct)
    assert point == "no_reminders" and state == f'Item: Send p-05 the tax form\nThe person said: "{HANDLED}"'


async def test_each_reminder_decision_is_logged_with_its_turn_and_its_own_item_reference(tmp_path, monkeypatch):
    """Two proposed reminders in one turn: each logged decision names the originating turn (so erasing that turn
    leaves its rows out) and its own item (its place in the capture call's answer, and a keyed hash of its words,
    which never appear in the clear, even as one word that could pass for an id)."""
    from colonymind.instinct.log import Store
    extractor, commitments, now = capture(tmp_path, monkeypatch, HANDLED)
    store = Store(tmp_path / "instinct.db")
    engine = Instinct(providers={"local": Provider({"no_reminders": "no"})}, fast="local", store=store)
    engine.set_state("no_reminders", "fast", "live", source="test")
    extractor.instinct = engine
    due = (now + timedelta(hours=2)).isoformat()
    items = [_item("Send p-05 the tax form", due_at=due), _item("Groceries", due_at=due)]
    assert await extractor.process_one(_Router(items)) is True
    await engine.drain()
    rows = store.rows(point="no_reminders")
    assert len(rows) == 2 and {row["refs"]["turn"] for row in rows} == {"t-1"}
    assert sorted(row["refs"]["item"] for row in rows) == ["0", "1"]
    hashes = {row["refs"]["about"] for row in rows}
    assert len(hashes) == 2 and all(value.startswith("h:") for value in hashes)
    dump = json.dumps(rows)
    assert "Groceries" not in dump and "groceries" not in dump and "tax form" not in dump


async def test_no_answer_keeps_the_reminder_the_capture_call_set(tmp_path, monkeypatch):
    extractor, commitments, now = capture(tmp_path, monkeypatch, HANDLED)
    extractor.instinct = Decider(no_reminders=None)
    due = (now + timedelta(hours=2)).isoformat()
    assert await extractor.process_one(_Router([_item("Send p-05 the tax form", due_at=due)])) is True
    [row] = commitments.get_pending_for_person(OWNER)
    assert row["due_at"] is not None


async def test_a_held_item_and_a_message_to_a_contact_are_never_asked_about(tmp_path, monkeypatch):
    extractor, commitments, now = capture(tmp_path, monkeypatch, "Tell p-05 at five that the form is in. "
                                                                 "The receipt I will handle, no reminders.")
    extractor.instinct = Decider(no_reminders="yes")
    due = (now + timedelta(hours=2)).isoformat()
    notice = {**_item("Tell p-05 the form is in", due_at=due),
              "metadata": {"kind": "notice", "recipient": "p-05", "content": "The form is in.",
                           "asked": "Tell p-05 at five that the form is in", "grant": "owner"},
              "obligor": "assistant"}
    assert await extractor.process_one(_Router([notice, _item("Send p-05 the receipt")])) is True
    assert calls(extractor.instinct) == []


def Unanswered():
    """A decision server that takes every question and answers only after its time limit."""
    engine = Instinct(providers={"local": Provider({"no_reminders": "yes"}, delay=30)}, fast="local")
    engine.set_state("no_reminders", "fast", "live", source="test")
    return engine


def lease(extractor):
    from contextlib import closing
    with closing(extractor.ledger._connect()) as conn:
        return dict(conn.execute("SELECT status, lease_until, attempts FROM commitment_runs").fetchone())


async def test_a_drain_that_runs_out_during_a_decision_hands_the_job_back_at_once(tmp_path, monkeypatch):
    """The capture call answered; the drain's budget runs out while the decision model is asked. The job goes back
    pending and uncharged, claimable at once, and lands on the next pass."""
    extractor, commitments, now = capture(tmp_path, monkeypatch, HANDLED)
    extractor.instinct = Unanswered()
    due = (now + timedelta(hours=2)).isoformat()
    router = _Router([_item("Send p-05 the tax form", due_at=due)], [_item("Send p-05 the tax form", due_at=due)])
    await extractor.drain(router, budget_seconds=0.2)
    assert len(calls(extractor.instinct)) == 1
    assert lease(extractor) == {"status": "pending", "lease_until": 0, "attempts": 0}
    extractor.instinct = Decider(no_reminders=None)
    assert await extractor.process_one(router) is True
    [row] = commitments.get_pending_for_person(OWNER)
    assert row["due_at"] is not None and lease(extractor)["status"] != "running"


async def test_a_drain_that_runs_out_after_the_capture_call_anywhere_hands_the_job_back(tmp_path, monkeypatch):
    """Not only a decision: any wait between the capture call's answer and the write (the owner's names here)."""
    extractor, commitments, now = capture(tmp_path, monkeypatch, "Remind me to send p-05 the tax form at five.")
    looked_up = []

    async def names(contact_id):
        looked_up.append(contact_id)
        if len(looked_up) > 1:                        # the owner's names, looked up after the capture call
            await asyncio.Event().wait()
        return []

    extractor.aliases = names
    due = (now + timedelta(hours=2)).isoformat()
    await extractor.drain(_Router([_item("Send p-05 the tax form", due_at=due)]), budget_seconds=0.2)
    assert len(looked_up) == 2
    assert lease(extractor) == {"status": "pending", "lease_until": 0, "attempts": 0}


# -- the owner settling an interest -------------------------------------------------------------------------

SATISFIED = "Found a good video on tide tables, so I'm sorted there."


async def test_an_interest_the_owner_settles_in_words_the_capture_call_missed_is_settled(fx):
    fx.mind.add_interest("tide tables")
    fx.mind.add_interest("moss lawns")
    fx.mind.capture.instinct = Decider(interest_settled="yes")
    fx.turn("t-settle", OWNER, SATISFIED, "Good to hear.")
    assert await fx.mind.capture.process_one(CaptureRouter(lambda prompt: [])) is True
    assert fx.mind.open_interests() == ["moss lawns"]
    assert calls(fx.mind.capture.instinct) == [("interest_settled", f'Topic: tide tables\nThe owner said: "{SATISFIED}"')]


async def test_an_interest_stays_open_without_an_answer_or_when_the_turn_does_not_name_it(fx):
    fx.mind.add_interest("tide tables")
    fx.mind.capture.instinct = Decider(interest_settled=None)
    fx.turn("t-1", OWNER, SATISFIED, "Good to hear.")
    assert await fx.mind.capture.process_one(CaptureRouter(lambda prompt: [])) is True
    assert fx.mind.open_interests() == ["tide tables"]
    fx.mind.capture.instinct = Decider(interest_settled="yes")
    fx.turn("t-2", OWNER, "The lease is signed, so I'm sorted there.", "Good.")
    assert await fx.mind.capture.process_one(CaptureRouter(lambda prompt: [])) is True
    assert fx.mind.open_interests() == ["tide tables"] and calls(fx.mind.capture.instinct) == []


# -- whether an owner message is a verdict on the agent's work ----------------------------------------------

def reported_request(prompt):
    """The lesson call reporting the owner's second request (it follows the agent's reply) as a verdict."""
    second = label(prompt, "Order 5522 came in")
    return {"verdicts": [judged(prompt, "Order 5522 came in", "What is its code?", work_was="right")],
            "ops": [add([second], quote="What is its code?", content=OWN_RULE)]}


async def test_a_reported_verdict_the_decision_model_reads_as_a_request_is_withdrawn(tmp_path, monkeypatch):
    fx = lesson_fixture(tmp_path, monkeypatch, reported_request)
    fx.mind.lessons.instinct = Decider(owner_verdict="no")
    two_requests(fx)
    result = await night(fx)
    assert result["counts"]["lesson_verdicts_vetoed"] == 1 and result["counts"]["lesson_ops_rejected"] == 1
    assert fx.mind.lessons.all(include_closed=True) == []
    [(point, state)] = calls(fx.mind.lessons.instinct)
    assert point == "owner_verdict" and state.startswith('The assistant replied: "The order code is C4411."')
    assert 'The owner then said: "Order 5522 came in' in state
    fx.store.close()


async def test_a_withdrawn_verdict_is_a_rejection_in_the_learning_log(tmp_path, monkeypatch):
    """The learning log keeps every rejection (with the rules on or off): a verdict the decision model withdraws is
    logged as a rejected verdict with the owner turn it came from, as one the quote check refused is."""
    from contextlib import closing
    from colonymind.memory import learning
    fx = lesson_fixture(tmp_path, monkeypatch, reported_request)
    fx.mind.lessons.instinct = Decider(owner_verdict="no")
    two_requests(fx)
    await night(fx)
    with closing(fx.mind.lessons.ledger._connect()) as conn:
        rows = learning.entries(conn, store="lesson", action="rejected")
    withdrawn = [row for row in rows if row["stage"] == "verdict" and "decision model" in row["reason"]]
    assert len(withdrawn) == 1 and withdrawn[0]["refs"] and withdrawn[0]["refs"][0].startswith("turn:")
    fx.store.close()


async def test_no_answer_leaves_the_reported_verdict_standing(tmp_path, monkeypatch):
    fx = lesson_fixture(tmp_path, monkeypatch, reported_request)
    fx.mind.lessons.instinct = Decider(owner_verdict=None)
    two_requests(fx)
    result = await night(fx)
    assert "lesson_verdicts_vetoed" not in result["counts"] and result["counts"].get("lessons_admitted") == 1
    fx.store.close()


async def test_a_veto_read_from_a_cut_input_leaves_the_reported_verdict_standing(tmp_path, monkeypatch):
    fx = lesson_fixture(tmp_path, monkeypatch, reported_request)
    fx.mind.lessons.instinct = served({"type": "choice", "probabilities": {"a": 0.01, "b": 0.99}, "confidence": 0.99}, point="owner_verdict",
                                      truncated=True)
    two_requests(fx)
    result = await night(fx)
    assert "lesson_verdicts_vetoed" not in result["counts"] and result["counts"].get("lessons_admitted") == 1
    assert failed(fx.mind.lessons.instinct, "owner_verdict") >= 1
    fx.store.close()


def test_the_fast_points_are_the_ones_wired():
    assert set(FAST_POINTS) == {"outreach_reply", "opt_out", "no_reminders", "interest_settled", "owner_verdict"}
    assert json.dumps(sorted(FAST_POINTS))


async def test_a_shadow_point_is_asked_where_1_11_would_have_asked_and_never_applied(make):
    """Shipped in shadow, ``outreach_reply`` is asked exactly where the phrases read nothing, and its answer is
    logged, never applied."""
    fx = make()
    row = await shared(fx)
    fx.shift(timedelta(minutes=5))
    engine = Instinct(providers={"local": Provider({"outreach_reply": "stop"})}, fast="local")
    fx.mind.instinct = engine
    summary = await say(fx, NO_CUE, "t-1", "owner-2")
    await engine.drain()
    assert reaction(fx, row)["class"] == "engaged" and "paused" not in summary["applied"]
    assert [point for point, _ in calls(engine)] == ["outreach_reply"]
