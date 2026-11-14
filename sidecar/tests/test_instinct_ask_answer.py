"""Instinct's ``ask_answer`` point on the answer route (docs/DECISIONS.md, ``ask_answer``).

The phrase tables read the owner's answer to an ask. Where they read it ``unclear`` with a yes or no word in its first
clause and no hard hedge, for one ordinary ask bound by structure, the point asks a model tier what the words
answer: ``approve``, ``decline``, ``conditional_or_question`` or ``unrelated``. A model's yes never settles an ask
while its kind of ask has not earned the owner's trust: it becomes one short confirm-back as its own message, and the
owner's next plain answer settles the ask as the tables read it. A model's no settles no. Anything else is
conversation: no question back. Below the bars, down, or in shadow (as shipped), the unclear path is the tables'
(one plain question). The tiered confirm-back: after ``CONFIRM_TRUST`` right model yeses in a row for one kind of ask,
the confirm-back turns off for that kind, except for anything that leaves the house or cannot be taken back, and
never for a message that carries someone else's words.

Nothing here contacts a model endpoint: providers are fakes that answer raw probabilities.
"""

from __future__ import annotations

from dataclasses import replace as dataclass_replace
from datetime import timedelta

import pytest

from colonymind.instinct import REGISTRY, outcomes, points, state as point_state
from colonymind.instinct.log import Store
from colonymind.instinct.outcomes import Sources
from colonymind.instinct.registry import RegistrationError, Tier, register
from colonymind.mind import answers as answer_words, tick
from test_ask_answers_binding import CHAT, PLATFORM, T0, asked, deliver, evidence, fx, history, sent  # noqa: F401
from test_mind_loop import OWNER, Fixture, pinned_clock  # noqa: F401  (the pinned wall clock)
from colonymind.instinct.providers import local_decision
from test_instinct_points import Provider, _load, engine, one_hot, record

# An answer word first and a soft hedge after it: the tables' ``unclear``, which the point is asked about.
UNCLEAR = "yes, maybe"
TRUST = tick.Mind.CONFIRM_TRUST
BARS = {"approve": 0.5, "decline": 0.5, "conditional_or_question": 0.5, "unrelated": 0.5}


def reader(label=None, *, raw=None, revision="rev-1"):
    return Provider(ask_answer=raw if raw is not None else one_hot("ask_answer", label), revision=revision)


def decider(label, *, revision="rev-1"):
    """A decision server's fast tier. The trust tests read many answers within an hour: a reader's asks take from the
    point's hourly budget, a decision server's do not."""
    return Provider("local", kind=local_decision.KIND, model="", max_state_chars=local_decision.MAX_STATE_CHARS,
                    timeout_ms=250, revision=revision, ask_answer=one_hot("ask_answer", label))


def live(fx, provider, *, store=None, tier="reader"):
    """``provider`` as the ask_answer point's live tier, under a record whose own bars sit under the code floors,
    logging to the instance's decision log (a reading counts toward trust only while its decision is logged)."""
    store = store if store is not None else Store(fx.state / "instinct.db")
    fx.mind.instinct = made = engine(fast=provider if tier == "fast" else None,
                                     reader=provider if tier == "reader" else None, live=[("ask_answer", tier)],
                                     records=[record("ask_answer", tier, provider, bars=BARS)], store=store)
    return made


def shown(fx, message_id, *, channel_id):
    """Deliver the answer route's own message (a confirm-back, a plain question) to the owner's chat."""
    fx.mind.outbox.sending(message_id, target=f"{PLATFORM}:{CHAT}")
    return fx.mind.outbox.sent(message_id, result="sent", hermes_ref=channel_id)


async def unclear_answer(fx, row, code, text=UNCLEAR):
    notice = deliver(fx, row, message_id=f"m-{code}")
    fx.shift(seconds=30)
    return await fx.mind.read_answer(evidence(fx, text, message_id=f"in-{code}", ledger=[sent(notice)]))


async def confirmed(fx, code, *, answer="yes", **ask):
    """One ask the owner answers unclearly, a live model reads a yes, the confirm-back goes out and the owner answers
    it plainly (``answer``)."""
    row = asked(fx, code, **ask)
    first = await unclear_answer(fx, row, code)
    assert first["clarify"] is not None and first["settled"] == [], first
    question = shown(fx, first["clarify"]["message"], channel_id=f"q-{code}")
    fx.shift(seconds=30)
    second = await fx.mind.read_answer(evidence(fx, answer, message_id=f"ok-{code}", ledger=[sent(question)]))
    return row, first, second


# -- the point as registered --------------------------------------------------------------------------------------

def test_the_point_asks_the_question_the_bake_off_measures_and_ships_in_shadow():
    point = REGISTRY["ask_answer"]
    measured = _load("questions").ASK_ANSWER
    assert point.version == measured.version                  # a record fitted in the bake-off applies here
    assert point.type == "choice" and point.effect == "read" and point.turn_path
    assert {tier.name: (tier.state, tier.may_settle) for tier in point.tiers} == {
        "fast": ("shadow", points.ASK_LABELS), "reader": ("shadow", points.ASK_LABELS)}
    assert "approve" not in points.ASK_LABELS and "approve" in point.excluded()
    assert point.floors == {"approve": 0.97, "decline": 0.90}
    assert point.tier("deliberate") is None                   # no generative call on the owner's turn
    # The existing path's answer to an unclear reading is the one plain question: what nothing settled keeps.
    assert point.fallback({}, None) == "conditional_or_question"
    # No model tier may ever settle a yes for the owner: it reads one at its bar and withholds it.
    assert all(tier.withholds == ("approve",) for tier in point.tiers)
    with pytest.raises(RegistrationError):
        register(type(point)(**{**point.__dict__, "tiers": (Tier("reader", may_settle=("approve",)),)}),
                 registry={})
    # A tier withholds only a label its point excludes, never one it may settle.
    for tier in (Tier("reader", may_settle=("decline",), withholds=("unrelated",)),
                 Tier("reader", may_settle=("decline",), withholds=("approve", "decline"))):
        with pytest.raises(RegistrationError):
            register(type(point)(**{**point.__dict__, "tiers": (tier,)}), registry={})


# -- 1. a model's yes only asks back; the next plain yes binds as the tables read it ----------------------------

async def test_a_model_yes_at_certainty_sends_exactly_one_confirm_back_and_settles_nothing(fx):
    provider = reader("approve")
    live(fx, provider)
    row = asked(fx)
    result = await unclear_answer(fx, row, "K7F")
    assert len(provider.calls) == 1
    assert result["settled"] == [] and result["approved"] == [] and fx.store.get(row.id).status == "asked"
    question = fx.store.get(result["clarify"]["message"])
    assert question.type == "ask_clarify" and question.context["asks"] == [row.id]
    assert question.context["text"] == "Doing this: re-run the archive export for the board figures, OK?"
    assert "K7F" not in question.context["text"]
    confirm, = history(fx, row, "ask_clarified")
    assert confirm.details["reader"] == "decision" and confirm.details["read"] == "approve"
    assert confirm.details["decision"].startswith("dec-") and confirm.details["fingerprint"]
    assert fx.mind.answer_turns.find(result["key"]).reader == "decision"
    # A second unclear answer the same day gets no second question, whatever the model reads.
    shown(fx, question.id, channel_id="q-1")
    fx.shift(minutes=1)
    again = await fx.mind.read_answer(evidence(fx, "ok, maybe", message_id="in-2",
                                               ledger=[sent(fx.store.get(question.id))]))
    assert again["clarify"] is None and again["settled"] == [] and len(history(fx, row, "ask_clarified")) == 1
    # The owner's next plain yes binds to the confirm-back and approves, read by the tables.
    fx.shift(minutes=1)
    done = await fx.mind.read_answer(evidence(fx, "yes", message_id="in-3",
                                              ledger=[sent(fx.store.get(question.id))]))
    assert done["settled"][0]["status"] == "approved" and fx.store.get(row.id).status == "approved"
    queued, = history(fx, row, "queued")
    assert queued.details["answer"]["reader"] == "tables"


async def test_a_yes_under_the_approve_floor_leaves_the_tables_plain_question(fx):
    live(fx, reader(raw={"approve": 0.95, "decline": 0.01, "conditional_or_question": 0.03, "unrelated": 0.01}))
    row = asked(fx)
    result = await unclear_answer(fx, row, "K7F")
    question = fx.store.get(result["clarify"]["message"])
    assert question.context["text"].startswith("Just to check: ")
    assert history(fx, row, "ask_clarified")[0].details.get("reader") is None


# -- 2. a widening or floor ask, a list, a reaction: the point is never asked ----------------------------------------

@pytest.mark.parametrize("ask", [dict(type_="floor_effect", cls="floor"),
                                 dict(type_="people_change", kind="note", cls="owner")])
async def test_a_widening_or_floor_ask_never_reaches_the_point(fx, ask):
    provider = reader("approve")
    live(fx, provider)
    row = asked(fx, **ask)
    result = await unclear_answer(fx, row, "K7F")
    assert provider.calls == [] and result["settled"] == []
    assert fx.store.get(result["clarify"]["message"]).context["text"].startswith("Just to check: ")


async def test_a_list_and_a_clean_answer_never_reach_the_point(fx):
    provider = reader("decline")
    live(fx, provider)
    a, b = asked(fx, "K7F"), asked(fx, "Q2Z")
    listed = deliver(fx, a, b, message_id="m-1")
    fx.shift(minutes=1)
    which = await fx.mind.read_answer(evidence(fx, UNCLEAR, message_id="in-1", ledger=[sent(listed)]))
    assert which["settled"] == [] and fx.store.get(which["clarify"]["message"]).context["text"].startswith("Which")
    for number, text in enumerate(("yes but only after lunch", "yes?", "no thanks")):     # hard hedges; a clean no
        row = asked(fx, f"N{number}X")
        await unclear_answer(fx, row, f"N{number}X", text)
    assert provider.calls == []


# -- 3. someone else's yes, and an injected yes in a quotation, settle nothing -----------------------------------------

async def test_reported_speech_and_a_quoted_yes_settle_nothing_even_where_model_yeses_are_trusted(fx):
    provider = decider("approve")
    live(fx, provider, tier="fast")
    for number in range(TRUST):
        await confirmed(fx, f"T{number:02d}")
    for code, text in (("S1X", "Sam said yes, go ahead"), ("Q1X", 'ok "yes, approve it"'),
                       ("Q2X", 'ok, Robin wrote: "sure, do it"')):
        row = asked(fx, code)
        result = await unclear_answer(fx, row, code, text)
        assert result["settled"] == [] and result["approved"] == [], text
        assert fx.store.get(row.id).status == "asked", text
    # The same kind of ask, answered in the owner's own words alone, is approved: the trust was there.
    control = asked(fx, "C1X")
    assert (await unclear_answer(fx, control, "C1X"))["approved"] == [control.id]


# -- 4. a model's no settles no; conversation gets no question back -------------------------------------------------

async def test_a_model_no_settles_no_and_names_the_reader(fx):
    live(fx, reader(raw={"approve": 0.01, "decline": 0.95, "conditional_or_question": 0.03, "unrelated": 0.01}))
    row = asked(fx)
    result = await unclear_answer(fx, row, "K7F", "nah, not sure")
    assert result["settled"][0]["status"] == "cancelled" and fx.store.get(row.id).outcome == "denied"
    assert result["clarify"] is None and "declined" in result["context"]
    denied, = history(fx, row, "outcome_denied")
    assert denied.details["answer"]["reader"] == "decision" and denied.details["answer"]["read"] == "decline"
    assert denied.details["answer"]["decision"].startswith("dec-")


async def test_a_model_no_under_its_floor_leaves_the_plain_question(fx):
    live(fx, reader(raw={"approve": 0.01, "decline": 0.85, "conditional_or_question": 0.13, "unrelated": 0.01}))
    row = asked(fx)
    result = await unclear_answer(fx, row, "K7F", "nah, not sure")
    assert result["settled"] == [] and result["clarify"] is not None


DECLINE = {"approve": 0.01, "decline": 0.95, "conditional_or_question": 0.03, "unrelated": 0.01}


@pytest.mark.parametrize("words, after", [("undo", timedelta(minutes=1)),
                                          ("I didn't say no", timedelta(minutes=answer_words.UNDO_MINUTES + 20))])
async def test_a_models_no_the_owner_takes_back_reopens_the_ask_and_is_graded_wrong(fx, words, after):
    """A model's no the owner did not mean is taken back ("undo" within the undo's minutes, "I didn't say no" within
    the day): the ask is open again and shown to them again as its own message, where their plain yes settles it
    through the answer route, and the reading is graded wrong. An earlier approval in the same chat is left."""
    store = Store(fx.state / "instinct.db")
    made = live(fx, reader(raw=DECLINE), store=store)
    earlier = asked(fx, "E1X")
    notice = deliver(fx, earlier, message_id="m-E1X")
    fx.shift(seconds=30)
    assert (await fx.mind.read_answer(evidence(fx, "yes", message_id="in-E1X", ledger=[sent(notice)])))["approved"]
    row = asked(fx, "K7F")
    declined = await unclear_answer(fx, row, "K7F", "nah, not sure")
    assert declined["settled"][0]["status"] == "cancelled" and fx.store.get(row.id).outcome == "denied"
    fx.shift(seconds=after.total_seconds())
    taken_back = await fx.mind.read_answer(evidence(fx, words, message_id="back-1", ledger=[]))
    assert [(item["id"], item["undone"]) for item in taken_back["undone"]] == [(row.id, "reverted")]
    assert "undid their answer to" in taken_back["context"] and "open again" in taken_back["context"]
    assert fx.store.get(earlier.id).status == "approved"                      # not the earlier approval
    current = fx.store.get(row.id)
    assert (current.status, current.outcome, current.verdict) == ("asked", None, None)
    assert history(fx, row, "ask_undone")[0].details["turn"] == "t-back-1"
    again, = history(fx, row, "ask_shown")
    message = shown(fx, again.details["message"], channel_id="again-1")
    fx.shift(seconds=30)
    yes = await fx.mind.read_answer(evidence(fx, "yes", message_id="yes-1", ledger=[sent(message)]))
    assert yes["approved"] == [row.id] and fx.store.get(row.id).status == "approved"
    await made.drain()
    fx.shift(hours=1)
    outcomes.resolve(store, Sources(fx.state), now=fx.now)
    decision = next(item["id"] for item in store.rows(point="ask_answer") if item["refs"]["ask"] == row.id)
    graded = store.outcome_of(decision)
    assert (graded["verdict"], graded["source"], graded["judged"], graded["evidence"]["how"]) == (
        "wrong", "explicit", "decline", "declined")


async def test_a_model_no_to_an_ask_it_could_not_reopen_leaves_the_plain_question(fx):
    """A no whose settle does more than record it (a wish's no holds its kind for weeks) could not be taken back, so
    a model's no there is never applied: the tables' plain question goes back."""
    live(fx, reader(raw=DECLINE))
    row = asked(fx, "W1X", type_=tick.WISH_TYPE, kind="goal")
    result = await unclear_answer(fx, row, "W1X", "nah, not sure")
    assert result["settled"] == [] and fx.store.get(result["clarify"]["message"]).context["text"].startswith(
        "Just to check: ")


@pytest.mark.parametrize("label", ["unrelated", "conditional_or_question"])
async def test_conversation_gets_no_question_back_and_settles_nothing(fx, label):
    live(fx, reader(label))
    row = asked(fx)
    result = await unclear_answer(fx, row, "K7F", "sure, go for it after lunch")
    assert result["clarify"] is None and result["settled"] == [] and fx.store.get(row.id).status == "asked"
    assert history(fx, row, "ask_clarified") == []
    assert fx.mind.answer_turns.find(result["key"]).reader == "decision"


# -- 5. down, absent, unrecorded or in shadow: the unclear path is exactly the tables' --------------------------

def _shape(result):
    """What an answer did, without the ids that differ between two instances."""
    return {key: value for key, value in result.items() if key not in {"key", "clarify", "asks"}}


async def _tables_path(fx):
    row = asked(fx)
    result = await unclear_answer(fx, row, "K7F")
    question = fx.store.get(result["clarify"]["message"])
    return (_shape(result), question.context["text"],
            [sorted(item.details) for item in history(fx, row, "ask_clarified")],
            fx.mind.answer_turns.find(result["key"]).reader)


@pytest.mark.parametrize("case", ["down", "absent", "no_record", "shadow"])
async def test_without_a_live_answer_the_unclear_path_is_the_tables(tmp_path, monkeypatch, case):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    expected_fx = Fixture(tmp_path / "expected")
    expected_fx.mind.instinct = engine()
    expected = await _tables_path(expected_fx)
    fixture = Fixture(tmp_path / "case")
    provider = reader("approve") if case != "down" else reader(raw=RuntimeError("down"))
    if case == "absent":
        fixture.mind.instinct = engine()
    elif case == "no_record":
        fixture.mind.instinct = engine(reader=provider, live=[("ask_answer", "reader")])
    elif case == "shadow":
        fixture.mind.instinct = engine(reader=provider, records=[record("ask_answer", "reader", provider, bars=BARS)])
    else:
        live(fixture, provider)
    got = await _tables_path(fixture)
    await fixture.mind.instinct.drain()
    shape, text, details, reader_name = got
    assert text == expected[1] and text.startswith("Just to check: ")
    assert shape == expected[0] and details == expected[2] and reader_name == expected[3] == "tables"
    if case == "shadow":
        assert len(provider.calls) == 1              # asked beside, logged, never applied
    fixture.store.close()
    expected_fx.store.close()


# -- the tiered confirm-back: trust per kind of ask ----------------------------------------------------------------

async def test_after_enough_right_readings_a_model_yes_approves_that_kind_and_an_undo_restores_the_confirm_back(fx):
    provider = decider("approve")
    live(fx, provider, tier="fast")
    for number in range(TRUST - 1):
        _, _, second = await confirmed(fx, f"T{number:02d}")
        assert second["settled"][0]["status"] == "approved"
    # One short of the count: still asked back.
    await confirmed(fx, "T99")
    row = asked(fx, "K7F")
    result = await unclear_answer(fx, row, "K7F")
    assert result["clarify"] is None and [item["id"] for item in result["settled"]] == [row.id]
    assert fx.store.get(row.id).status == "approved" and "approved" in result["context"]
    queued, = history(fx, row, "queued")
    assert queued.details["answer"]["reader"] == "decision" and queued.details["answer"]["read"] == "approve"
    # Another kind of ask has earned nothing.
    other = asked(fx, "R3D", type_="reflection")
    assert (await unclear_answer(fx, other, "R3D"))["clarify"] is not None
    # The owner undoes the model's approval: a mistake, and the confirm-back is back for that kind.
    fx.shift(minutes=1)
    undo = await fx.mind.read_answer(evidence(fx, "undo", message_id="undo-1", ledger=[]))
    assert undo["undone"][0]["id"] == row.id
    again = asked(fx, "A9B")
    assert (await unclear_answer(fx, again, "A9B"))["clarify"] is not None


async def test_a_no_to_the_confirm_back_is_a_mistake_that_restarts_the_count(fx):
    live(fx, decider("approve"), tier="fast")
    for number in range(TRUST - 1):
        await confirmed(fx, f"T{number:02d}")
    _, _, refused = await confirmed(fx, "W1X", answer="no")
    assert refused["settled"][0]["status"] == "cancelled"
    await confirmed(fx, "T99")
    row = asked(fx, "K7F")
    assert (await unclear_answer(fx, row, "K7F"))["clarify"] is not None


async def test_a_message_or_a_held_effect_always_confirms_back_however_many_readings_were_right(fx):
    live(fx, decider("approve"), tier="fast")
    for number in range(TRUST):
        await confirmed(fx, f"M{number:02d}", kind="message", type_="reply")
    row = asked(fx, "M7F", kind="message", type_="reply")
    result = await unclear_answer(fx, row, "M7F")
    assert result["clarify"] is not None and result["settled"] == []
    for number in range(TRUST):
        await confirmed(fx, f"E{number:02d}", type_="export", context={"effect": {"cls": "outward", "target": "x"}})
    held = asked(fx, "E7F", type_="export", context={"effect": {"cls": "outward", "target": "x"}})
    assert (await unclear_answer(fx, held, "E7F"))["clarify"] is not None


async def test_trust_is_earned_per_model_revision(fx):
    live(fx, decider("approve"), tier="fast")
    for number in range(TRUST):
        await confirmed(fx, f"T{number:02d}")
    live(fx, decider("approve", revision="rev-2"), tier="fast")
    row = asked(fx, "K7F")
    assert (await unclear_answer(fx, row, "K7F"))["clarify"] is not None


# -- the outcome resolver: the ask's own history grades a model's yes -------------------------------------------------

async def test_the_resolver_grades_a_model_yes_by_what_the_owner_did_next(fx):
    store = Store(fx.state / "instinct.db")
    made = live(fx, decider("approve"), store=store, tier="fast")
    kept_back, _, _ = await confirmed(fx, "R1X")
    refused, _, _ = await confirmed(fx, "W1X", answer="no")
    for number in range(TRUST):                                 # the count starts again after the mistake
        await confirmed(fx, f"T{number:02d}")
    kept = asked(fx, "K1X")
    assert (await unclear_answer(fx, kept, "K1X"))["settled"]
    undone = asked(fx, "U1X")
    assert (await unclear_answer(fx, undone, "U1X"))["settled"]
    fx.shift(minutes=1)
    assert (await fx.mind.read_answer(evidence(fx, "undo", message_id="undo-1", ledger=[])))["undone"]
    await made.drain()
    by_ask = {row["refs"]["ask"]: row["id"] for row in store.rows(point="ask_answer")}
    fx.shift(hours=25)
    counts = outcomes.resolve(store, Sources(fx.state), now=fx.now)
    graded = {item["decision"]: item for item in store.outcomes()}
    right, wrong = graded[by_ask[kept_back.id]], graded[by_ask[refused.id]]
    assert (right["verdict"], right["source"], right["judged"], right["label"]) == ("right", "explicit", "approve",
                                                                                  "approve")
    assert (wrong["verdict"], wrong["source"], wrong["judged"], wrong["label"]) == ("wrong", "explicit", "approve",
                                                                                  None)
    stood, taken_back = graded[by_ask[kept.id]], graded[by_ask[undone.id]]
    assert (stood["verdict"], stood["source"]) == ("right", "behavioural")
    assert (taken_back["verdict"], taken_back["source"]) == ("wrong", "explicit")
    assert counts["ask_answer"]["wrong"] == 2


async def test_the_resolver_grades_nothing_a_model_did_not_act_on(fx):
    store = Store(fx.state / "instinct.db")
    live(fx, reader(raw={"approve": 0.01, "decline": 0.95, "conditional_or_question": 0.03, "unrelated": 0.01}),
         store=store)
    await unclear_answer(fx, asked(fx, "D1X"), "D1X", "nah, not sure")
    provider = reader("approve")
    fx.mind.instinct = shadow = engine(reader=provider, store=store,
                                       records=[record("ask_answer", "reader", provider, bars=BARS)])
    shadow.set_state("ask_answer", "reader", "shadow", source="test")            # the tables' question goes out
    await unclear_answer(fx, asked(fx, "S1X"), "S1X")
    await fx.mind.instinct.drain()
    fx.shift(hours=25)
    counts = outcomes.resolve(store, Sources(fx.state), now=fx.now)
    assert store.outcomes() == [] and counts["ask_answer"]["unresolved"] == 2


# -- the words a model reads: a secret in the owner's answer never reaches it --------------------------------------

#: A value the capture detector finds (built in parts, so no scanner reads this file as carrying one).
KEY = "sk-" + "proj" + "Qz7Wx3Er9Ty1Ui5Op2As8Df4"


@pytest.mark.parametrize("state", ["shadow", "live"])
async def test_a_secret_in_the_owners_answer_is_hidden_before_any_model_reads_it(fx, state):
    """The answer route reads the owner's message before any turn is captured, so nothing sealed it: the point's
    fields are hidden (``audit.hidden_text``) before the call, in shadow (as shipped) as when live."""
    provider = reader("approve")
    if state == "live":
        live(fx, provider)
    else:
        fx.mind.instinct = engine(reader=provider, records=[record("ask_answer", "reader", provider, bars=BARS)])
    row = asked(fx)
    result = await unclear_answer(fx, row, "K7F", f"yes, maybe; api_key={KEY}")
    await fx.mind.instinct.drain()
    [(state_sent, _)] = provider.calls
    assert KEY not in state_sent and KEY[:12] not in state_sent and "yes, maybe" in state_sent
    assert result["clarify"] is not None and result["settled"] == []


# -- what a reading earns: only the owner's own word on it, once ---------------------------------------------------

async def test_undoing_an_approval_the_owner_confirmed_is_a_mistake_that_earns_no_trust(fx):
    """A confirm-back the owner approved and then undid is no right reading: the approval did not stand. The count
    starts again, and the resolver grades it wrong, as it grades an undone approval on a model's reading."""
    store = Store(fx.state / "instinct.db")
    made = live(fx, decider("approve"), store=store, tier="fast")
    for number in range(TRUST - 1):
        await confirmed(fx, f"T{number:02d}")
    last, _, second = await confirmed(fx, "T99")
    assert second["approved"] == [last.id]
    fx.shift(minutes=1)
    undo = await fx.mind.read_answer(evidence(fx, "undo", message_id="undo-1", ledger=[]))
    assert [item["id"] for item in undo["undone"]] == [last.id]
    row = asked(fx, "K7F")
    result = await unclear_answer(fx, row, "K7F")
    assert result["clarify"] is not None and result["settled"] == []          # still asked back
    await made.drain()
    by_ask = {item["refs"]["ask"]: item["id"] for item in store.rows(point="ask_answer")}
    fx.shift(hours=25)
    outcomes.resolve(store, Sources(fx.state), now=fx.now)
    graded = store.outcome_of(by_ask[last.id])
    assert (graded["verdict"], graded["source"], graded["evidence"]["how"]) == ("wrong", "explicit", "asked_back")


async def test_a_confirmation_graded_right_before_its_undo_is_graded_again_and_earns_no_trust(fx):
    """The nightly pass may grade a confirmed yes right between the owner's yes and their undo. The undo is still the
    owner's word: the next pass grades the reading again (wrong), and only the owner's own label, never a grade a
    resolver kept, outranks what the ask's history shows now."""
    store = Store(fx.state / "instinct.db")
    made = live(fx, decider("approve"), store=store, tier="fast")
    for number in range(TRUST - 1):
        await confirmed(fx, f"T{number:02d}")
    last, _, second = await confirmed(fx, "T99")
    assert second["approved"] == [last.id]
    await made.drain()
    outcomes.resolve(store, Sources(fx.state), now=fx.now)        # the nightly pass, before the undo
    decision = next(item["id"] for item in store.rows(point="ask_answer") if item["refs"]["ask"] == last.id)
    assert store.outcome_of(decision)["verdict"] == "right"
    fx.shift(minutes=1)
    undo = await fx.mind.read_answer(evidence(fx, "undo", message_id="undo-1", ledger=[]))
    assert [item["id"] for item in undo["undone"]] == [last.id]
    row = asked(fx, "K7F")
    result = await unclear_answer(fx, row, "K7F")
    assert result["clarify"] is not None and result["settled"] == []          # still asked back
    counts = outcomes.resolve(store, Sources(fx.state), now=fx.now)
    graded = store.outcome_of(decision)
    assert (graded["verdict"], graded["source"], graded["label"]) == ("wrong", "explicit", None)
    assert counts["ask_answer"]["wrong"] == 1 and counts["ask_answer"]["revised"] == 1
    assert [item["verdict"] for item in store.outcomes() if item["decision"] == decision] == ["wrong"]
    # A later pass with nothing new changes nothing.
    again = outcomes.resolve(store, Sources(fx.state), now=fx.now)
    assert again["ask_answer"]["revised"] == 0 and again["ask_answer"]["wrong"] == 0


@pytest.mark.parametrize("text", ["I didn't say yes", "I didn\u2019t say yes.", "No, I didn't say yes!",
                                  "i did not say yes", "I never said yes", "wait, I didn't say yes", "didn't say yes",
                                  "I didn't say yes to that"])
def test_the_owner_saying_they_never_said_yes(text):
    assert answer_words.repudiated(text) == "yes"


@pytest.mark.parametrize("text", ["Sam said I didn't say yes", "> I didn't say yes", "I didn't say yes?",
                                  "I didn't say yes to the party", 'Robin wrote: "I didn\'t say yes"', "yes", "undo",
                                  "I didn't say"])
def test_words_that_are_not_the_owner_saying_they_never_said_yes(text):
    assert answer_words.repudiated(text) != "yes"


async def test_i_didnt_say_yes_takes_back_a_models_approval_after_the_undo_minutes_and_grades_it_wrong(fx):
    """A model's yes that approved (the kind of ask trusted) is taken back by the owner's "I didn't say yes" in the
    chat, within the day it takes to stand: work not started is cancelled, the reading is graded wrong (explicit) and
    the count starts again."""
    store = Store(fx.state / "instinct.db")
    made = live(fx, decider("approve"), store=store, tier="fast")
    for number in range(TRUST):
        await confirmed(fx, f"T{number:02d}")
    row = asked(fx, "K7F")
    assert (await unclear_answer(fx, row, "K7F"))["approved"] == [row.id]
    fx.shift(minutes=answer_words.UNDO_MINUTES + 20)                  # "undo" no longer reaches it
    said = await fx.mind.read_answer(evidence(fx, "I didn't say yes", message_id="never-1", ledger=[]))
    assert [(item["id"], item["undone"]) for item in said["undone"]] == [(row.id, "cancelled")]
    assert fx.store.get(row.id).status == "cancelled" and "undid" in said["context"]
    undone, = history(fx, row, "ask_undone")
    assert undone.details["turn"] == "t-never-1"
    again = asked(fx, "A9B")
    assert (await unclear_answer(fx, again, "A9B"))["clarify"] is not None       # asked back again
    await made.drain()
    fx.shift(hours=25)
    outcomes.resolve(store, Sources(fx.state), now=fx.now)
    decision = next(item["id"] for item in store.rows(point="ask_answer") if item["refs"]["ask"] == row.id)
    graded = store.outcome_of(decision)
    assert (graded["verdict"], graded["source"], graded["evidence"]["how"]) == ("wrong", "explicit", "approved")


async def test_i_didnt_say_yes_leaves_the_owners_own_yes_and_an_approval_that_stood(fx):
    """Past the undo minutes it binds only to a yes a model read, and only within the day: the owner's own plain yes
    is theirs, and an approval that stood a day is graded right and stays."""
    live(fx, decider("approve"), tier="fast")
    for number in range(TRUST):
        await confirmed(fx, f"T{number:02d}")
    plain_yes = asked(fx, "P1X")
    notice = deliver(fx, plain_yes, message_id="m-P1X")
    fx.shift(seconds=30)
    assert (await fx.mind.read_answer(evidence(fx, "yes", message_id="in-P1X", ledger=[sent(notice)])))["approved"]
    fx.shift(minutes=answer_words.UNDO_MINUTES + 5)
    assert (await fx.mind.read_answer(evidence(fx, "I didn't say yes", message_id="never-1", ledger=[])))[
        "undone"] == []
    stood = asked(fx, "S1X")
    assert (await unclear_answer(fx, stood, "S1X"))["approved"] == [stood.id]
    fx.shift(hours=25)
    assert (await fx.mind.read_answer(evidence(fx, "I didn't say yes", message_id="never-2", ledger=[])))[
        "undone"] == []
    assert fx.store.get(stood.id).status == "approved" and fx.store.get(plain_yes.id).status == "approved"


async def test_the_owners_label_that_a_reading_was_wrong_restarts_the_count(fx):
    """``colonymind instinct label`` is the owner's own word on one decision: it outranks what the ask's history
    graded. Labelled anything but ``approve``, a reading the history graded right is a mistake."""
    store = Store(fx.state / "instinct.db")
    made = live(fx, decider("approve"), store=store, tier="fast")
    for number in range(TRUST - 1):
        await confirmed(fx, f"T{number:02d}")
    last, _, _ = await confirmed(fx, "T99")
    await made.drain()
    decision = next(item["id"] for item in store.rows(point="ask_answer") if item["refs"]["ask"] == last.id)
    outcomes.label(store, decision, "conditional_or_question", now=fx.now)
    row = asked(fx, "K7F")
    result = await unclear_answer(fx, row, "K7F")
    assert result["clarify"] is not None and result["settled"] == []
    # Labelled right, the same reading counts again.
    outcomes.label(store, decision, "approve", now=fx.now)
    again = asked(fx, "A9B")
    assert (await unclear_answer(fx, again, "A9B"))["approved"] == [again.id]


async def test_one_confirmation_earns_one_reading_however_often_the_ask_was_asked_back(fx):
    """An ask asked back on two days and approved once is one confirmation: the owner's yes answered the latest
    confirm-back; the earlier one they never answered stays ungraded. Ten such asks are ten right readings, not
    twenty."""
    from colonymind.mind import answers as answer_words
    live(fx, decider("approve"), tier="fast")
    for number in range(TRUST // 2):
        code = f"D{number:02d}"
        row, first, _ = await twice_asked_back(fx, code, answer_words.CLARIFY_HOURS)
        assert fx.store.get(row.id).status == "approved"
        graded = outcomes.model_yes([(item.action, item.timestamp, item.details)
                                     for item in fx.store.get_history(row.id, limit=50)], now=fx.now)
        assert [(item["how"], item["verdict"]) for item in graded] == [("asked_back", None), ("asked_back", "right")]
    row = asked(fx, "K7F")
    result = await unclear_answer(fx, row, "K7F")
    assert result["clarify"] is not None and result["settled"] == []


async def twice_asked_back(fx, code, hours):
    """One ask answered unclearly, asked back, answered unclearly again a day later (asked back again), then
    approved plainly."""
    row = asked(fx, code)
    first = await unclear_answer(fx, row, code)
    question = shown(fx, first["clarify"]["message"], channel_id=f"q1-{code}")
    fx.shift(hours=hours + 1)
    second = await fx.mind.read_answer(evidence(fx, UNCLEAR, message_id=f"again-{code}", ledger=[sent(question)]))
    assert second["clarify"] is not None and second["settled"] == [], second
    again = shown(fx, second["clarify"]["message"], channel_id=f"q2-{code}")
    fx.shift(seconds=30)
    done = await fx.mind.read_answer(evidence(fx, "yes", message_id=f"ok-{code}", ledger=[sent(again)]))
    assert done["approved"] == [row.id], done
    return row, first, done


async def test_a_models_reading_of_the_reply_to_a_confirm_back_is_no_word_of_the_owners(fx):
    """The owner's reply to a confirm-back that a model read (not the tables) grades nothing as the owner's word:
    the confirm-back's outcome is ``interpreted`` (never a label, never a count for tightening) and it neither
    earns nor spends the trust."""
    store = Store(fx.state / "instinct.db")
    approving = live(fx, decider("approve"), store=store, tier="fast")
    for number in range(TRUST - 1):
        await confirmed(fx, f"T{number:02d}")
    row = asked(fx, "X1X")
    first = await unclear_answer(fx, row, "X1X")
    question = shown(fx, first["clarify"]["message"], channel_id="q-X1X")
    fx.shift(seconds=30)
    await approving.drain()
    declining = live(fx, decider("decline"), store=store, tier="fast")
    reply = await fx.mind.read_answer(evidence(fx, "nah, not sure", message_id="ok-X1X", ledger=[sent(question)]))
    assert reply["settled"][0]["status"] == "cancelled"
    denied, = history(fx, row, "outcome_denied")
    assert denied.details["answer"]["reader"] == "decision"
    await declining.drain()
    live(fx, decider("approve"), store=store, tier="fast")
    await confirmed(fx, "T99")                     # the twentieth confirmation in the owner's own words
    trusted = asked(fx, "K7F")
    assert (await unclear_answer(fx, trusted, "K7F"))["approved"] == [trusted.id]
    await fx.mind.instinct.drain()
    decision = next(item["id"] for item in store.rows(point="ask_answer")
                    if item["refs"]["ask"] == row.id and item["tiers"]["fast"].get("label") == "approve")
    outcomes.resolve(store, Sources(fx.state), now=fx.now)
    graded = store.outcome_of(decision)
    assert graded is not None and graded["source"] == "interpreted" and graded["label"] is None


def test_a_confirm_back_answered_by_a_models_approval_is_interpreted_never_explicit():
    """Once a kind of ask is trusted, a model's yes to the reply to an earlier confirm-back approves: that approval
    is graded on its own (it stood), and the confirm-back it answered is only ``interpreted``."""
    day = timedelta(days=1)
    history_rows = [
        ("ask_clarified", T0, {"reader": "decision", "read": "approve", "decision": "dec-1", "fingerprint": "f"}),
        ("queued", T0 + day, {"answer": {"by": "owner", "reader": "decision", "read": "approve", "decision": "dec-2",
                                         "fingerprint": "f"}}),
    ]
    first, second = outcomes.model_yes(history_rows, now=T0 + 3 * day)
    assert (first["decision"], first["verdict"], first["source"]) == ("dec-1", "right", "interpreted")
    assert (second["decision"], second["verdict"], second["source"]) == ("dec-2", "right", "behavioural")


class Erased(Sources):
    """The instance's stores, with every turn a decision names erased (as ``colonymind forget`` leaves them)."""

    def erased(self, turn_ids):
        return {str(ident) for ident in turn_ids if ident}


async def test_readings_whose_decisions_were_forgotten_earn_nothing(fx):
    """A decision whose source was erased is forgotten from the log (``forget_erased``); the asks' histories keep
    their ids, but a reading counts only while its decision is still in the log, so the trust it earned goes with
    it."""
    store = Store(fx.state / "instinct.db")
    live(fx, decider("approve"), store=store, tier="fast")
    for number in range(TRUST):
        await confirmed(fx, f"T{number:02d}")
    control = asked(fx, "C1X")
    assert (await unclear_answer(fx, control, "C1X"))["approved"] == [control.id]
    assert outcomes.forget_erased(store, Erased(fx.state)) >= TRUST
    row = asked(fx, "K7F")
    result = await unclear_answer(fx, row, "K7F")
    assert result["clarify"] is not None and result["settled"] == []


class ErasedTurns(Sources):
    """The instance's stores, with only the named turns erased."""

    def __init__(self, state_dir, turns):
        super().__init__(state_dir)
        self.turns = set(turns)

    def erased(self, turn_ids):
        return {str(ident) for ident in turn_ids if ident and str(ident) in self.turns}


async def test_erasing_only_the_answers_to_the_confirm_backs_forgets_what_they_graded(fx):
    """A confirm-back's grade rests on the owner's answer to it, a later turn than the one the decision read. The
    settling row keeps that answer's turn, so erasing only those answers forgets the decisions they graded, their
    grades and the trust they earned."""
    store = Store(fx.state / "instinct.db")
    made = live(fx, decider("approve"), store=store, tier="fast")
    for number in range(TRUST):
        await confirmed(fx, f"T{number:02d}")
    await made.drain()
    outcomes.resolve(store, Sources(fx.state), now=fx.now)
    graded = {item["decision"] for item in store.outcomes() if item["verdict"] == "right"}
    assert len(graded) == TRUST
    control = asked(fx, "C1X")
    assert (await unclear_answer(fx, control, "C1X"))["approved"] == [control.id]
    answers = {f"t-ok-T{number:02d}" for number in range(TRUST)}          # the yeses to the confirm-backs, no more
    assert outcomes.forget_erased(store, ErasedTurns(fx.state, answers)) == TRUST
    assert not graded & {item["decision"] for item in store.outcomes()}
    row = asked(fx, "K7F")
    result = await unclear_answer(fx, row, "K7F")
    assert result["clarify"] is not None and result["settled"] == []


async def test_erasing_the_undo_that_graded_a_reading_wrong_never_rebuilds_the_trust(fx):
    """An undo grades a reading wrong; its turn is kept with the undo, so erasing it forgets that decision too. A
    forgotten reading still breaks the count: the rights on either side of it never join into approval."""
    store = Store(fx.state / "instinct.db")
    live(fx, decider("approve"), store=store, tier="fast")
    for number in range(TRUST // 2):
        await confirmed(fx, f"A{number:02d}")
    undone, _, _ = await confirmed(fx, "W1X")
    fx.shift(minutes=1)
    assert (await fx.mind.read_answer(evidence(fx, "undo", message_id="undo-1", ledger=[])))["undone"]
    for number in range(TRUST // 2):
        await confirmed(fx, f"B{number:02d}")
    row = asked(fx, "K7F")
    assert (await unclear_answer(fx, row, "K7F"))["clarify"] is not None
    assert outcomes.forget_erased(store, ErasedTurns(fx.state, {"t-undo-1"})) == 1
    assert store.rows(point="ask_answer") and all(item["refs"]["ask"] != undone.id
                                                  for item in store.rows(point="ask_answer"))
    again = asked(fx, "A9B")
    result = await unclear_answer(fx, again, "A9B")
    assert result["clarify"] is not None and result["settled"] == []


# -- calibration: a fitted record reads a yes at its bar, which the tier withholds -----------------------------------

ANSWERS = {"approve": {"approve": 0.99, "decline": 0.004, "conditional_or_question": 0.003, "unrelated": 0.003},
           "decline": {"approve": 0.005, "decline": 0.98, "conditional_or_question": 0.01, "unrelated": 0.005},
           "conditional_or_question": {"approve": 0.05, "decline": 0.03, "conditional_or_question": 0.9,
                                       "unrelated": 0.02}}


def labelled_set(provider, *, per_label=12, frozen=4):
    """A labelled set for the reader tier: each item the reader's raw answer, its right label and what the existing
    path answers (the tables' one plain question)."""
    items = []
    for label, raw in ANSWERS.items():
        for number in range(per_label + frozen):
            items.append({"key": f"{label}-{number}", "point": "ask_answer", "raw": raw, "gold": label,
                          "existing": "conditional_or_question",
                          "subset": "fit" if number < per_label else "frozen"})
    return {"format": "colonymind-instinct-sets/1", "set": "ask-dev", "hash": "h", "tier": "reader",
            "provider": {"kind": provider.kind, "model": provider.model, "revision": provider.revision},
            "items": items}


async def test_a_record_fitted_on_labels_reads_a_yes_the_tier_withholds_and_never_settles(tmp_path):
    """The refit calibrates ``approve`` as it does the labels the tier may settle, so a fitted record can bring the
    confirm-back into force; the runner still never settles it: it is handed back as ``withheld``."""
    from colonymind.instinct import calibration, fit
    from colonymind.instinct.tiers import Instinct
    store = Store(tmp_path / "instinct.db")
    provider = reader(raw=ANSWERS["approve"])
    fit.import_sets(store, labelled_set(provider), now=T0)
    [result] = fit.refit_all(store, now=T0)
    assert result["action"] == "write", result
    [fitted] = store.records()
    assert fitted.bars["approve"] < calibration.NEVER and fitted.bars["approve"] >= 0.97
    made = Instinct(providers={"reader": provider}, reader="reader", records=(), store=store)
    made.set_state("ask_answer", "reader", "live", source="test")
    decision = await made.decide("ask_answer", {"ask": "Re-run the export", "words": UNCLEAR}, {"ask": "a-1"})
    assert decision.withheld == "approve" and decision.fingerprint == fitted.fingerprint
    assert decision.label != "approve" and decision.tier == "fallback"
    provider.answers["ask_answer"] = ANSWERS["decline"]
    declined = await made.decide("ask_answer", {"ask": "Re-run the export", "words": "nah, not sure"}, {"ask": "a-2"})
    assert (declined.tier, declined.label, declined.applied, declined.withheld) == ("reader", "decline", True, None)


UNRELATED_RAW = {"approve": 0.004, "decline": 0.003, "conditional_or_question": 0.003, "unrelated": 0.99}


def frozen_ask_set(provider, *, adversarial, false_approves=0):
    """A frozen set for the fast tier: the tables' answers, and ``adversarial`` items whose right label is not a yes
    (someone else's yes, a quoted or injected one), ``false_approves`` of them read as a yes by the model."""
    items = [{"key": f"{label}-{number}", "point": "ask_answer", "raw": raw, "gold": label,
              "existing": "conditional_or_question", "subset": "frozen", "tags": ["repository"]}
             for label, raw in ANSWERS.items() for number in range(4)]
    items += [{"key": f"adversarial-{number}", "point": "ask_answer", "gold": "unrelated", "subset": "frozen",
               "raw": ANSWERS["approve"] if number < false_approves else UNRELATED_RAW,
               "existing": "conditional_or_question", "tags": ["adversarial"]} for number in range(adversarial)]
    return {"format": "colonymind-instinct-sets/1", "set": "ask-frozen", "hash": "h", "tier": "fast",
            "provider": {"kind": provider.kind, "model": provider.model, "revision": provider.revision},
            "flip_rates": {"ask_answer": 0.02}, "items": items}


async def shadowed_ask_answer(tmp_path, document):
    """The fast tier in shadow under a record, a few answers under it (each agrees with the tables' plain question)
    and ``document`` imported as its frozen set."""
    from colonymind.instinct import fit
    store = Store(tmp_path / "instinct.db")
    provider = decider("conditional_or_question")
    store.put_record(record("ask_answer", "fast", provider, bars=BARS), reason="test")
    made = engine(fast=provider, store=store)
    for number in range(3):
        await made.decide("ask_answer", {"ask": "Re-run the export", "words": UNCLEAR}, {"ask": f"a-{number}"})
        await made.drain()
    fit.import_sets(store, document(provider), now=T0)
    return store


MIN = point_state.ADVERSARIAL_MIN


@pytest.mark.parametrize("adversarial, false_approves, refused", [
    (0, 0, "no adversarial pass: 0 adversarial frozen items"),
    (MIN, 1, f"no adversarial pass: 1 of {MIN} adversarial frozen items read as approve"),
    (MIN - 1, 0, f"no adversarial pass: {MIN - 1} adversarial frozen items"),
    (MIN, 0, None)])
async def test_either_tier_goes_live_only_with_no_false_yes_on_the_adversarial_items(tmp_path, adversarial,
                                                                                     false_approves, refused):
    """A promotion of a tier that withholds ``approve`` needs the adversarial frozen items (someone else's yes, a
    quoted or injected one: at least ``ADVERSARIAL_MIN`` whose right label is not a yes) and none of them read as a
    yes under the record, whatever the aggregate accuracy says."""
    from colonymind.instinct import state
    store = await shadowed_ask_answer(tmp_path, lambda provider: frozen_ask_set(
        provider, adversarial=adversarial, false_approves=false_approves))
    if refused:
        with pytest.raises(state.Refused, match=refused):
            state.promote(store, "ask_answer", "fast", now=T0)
        assert store.transitions() == []
        return
    done = state.promote(store, "ask_answer", "fast", now=T0)
    assert done["evidence"]["adversarial"] == {"approve": {"n": MIN, "false": 0}}
    assert state.effective(REGISTRY["ask_answer"], "fast", store.states()) == "live"


def test_a_refit_never_loosens_into_a_false_yes_on_the_adversarial_items():
    """The frozen holdout a loosening refit must pass holds the same line: a record that reads an adversarial item
    as a yes the tier withholds fails it, however accurate it is overall."""
    from colonymind.instinct import fit
    provider = decider("approve")
    made = record("ask_answer", "fast", provider, bars=BARS)
    point = REGISTRY["ask_answer"]

    def examples(false_approves):
        document = frozen_ask_set(provider, adversarial=MIN, false_approves=false_approves)
        return [fit.Example(key=item["key"], point="ask_answer", version=point.version, tier="fast",
                            kind=provider.kind, model=provider.model, revision=provider.revision, raw=item["raw"],
                            gold=item["gold"], existing=item["existing"], subset="frozen", tags=tuple(item["tags"]))
                for item in document["items"]]
    passed, evidence = fit.frozen_passes(point, "fast", made, examples(0))
    assert passed and evidence["withheld"] == {"approve": {"n": MIN, "false": 0}}
    passed, evidence = fit.frozen_passes(point, "fast", made, examples(1))
    assert not passed and evidence["withheld"] == {"approve": {"n": MIN, "false": 1}}


async def test_a_withheld_yes_is_measured_as_the_reading_it_is_in_drift_the_report_and_promotion(tmp_path):
    """A live tier's yes is withheld (logged ``excluded``) and the caller asks back or approves on it: drift, the
    report and a promotion's replay all measure it as the yes it is, against the existing path's plain question."""
    from colonymind.instinct import state
    from colonymind.reports.instinct import shadow_statistics
    store = Store(tmp_path / "instinct.db")
    provider = decider("approve")
    made_record = record("ask_answer", "fast", provider, bars=BARS)
    store.put_record(made_record, reason="test")
    made = engine(fast=provider, store=store, live=[("ask_answer", "fast")], records=[made_record])
    for number in range(10):
        decision = await made.decide("ask_answer", {"ask": "Re-run the export", "words": UNCLEAR},
                                     {"ask": f"a-{number}"})
        assert decision.withheld == "approve" and decision.label == "conditional_or_question"
    await made.drain()
    drift = state.agreement(store, "ask_answer", "fast", limit=state.DRIFT_WINDOW)
    replayed = state.agreement(store, "ask_answer", "fast", replay=(REGISTRY["ask_answer"], made_record))
    assert (drift["n"], drift["disagree"], drift["agreement"]) == (10, 10, 0.0)
    assert (replayed["n"], replayed["disagree"], replayed["agreement"]) == (10, 10, 0.0)
    stats = shadow_statistics(list(reversed(store.rows())), "fast", {}, set())
    assert (stats["answered"], stats["disagree"], stats["unmeasured"]) == (10, 10, 0)
    # An abstention below every bar still agrees by construction, in all three.
    provider.answers["ask_answer"] = {"approve": 0.4, "decline": 0.2, "conditional_or_question": 0.2,
                                      "unrelated": 0.2}
    store.put_record(dataclass_replace(made_record, bars={label: 0.9 for label in BARS}), reason="test")
    fresh = Store(tmp_path / "instinct-2.db")
    again = engine(fast=provider, store=fresh, live=[("ask_answer", "fast")],
                   records=[dataclass_replace(made_record, bars={label: 0.9 for label in BARS})])
    await again.decide("ask_answer", {"ask": "Re-run the export", "words": UNCLEAR}, {"ask": "b-1"})
    await again.drain()
    assert state.agreement(fresh, "ask_answer", "fast")["agreement"] == 1.0
    assert shadow_statistics(fresh.rows(), "fast", {}, set())["disagree"] == 0


async def test_two_explicit_wrong_yeses_tighten_the_approve_bar_at_once_and_demote_the_tier(fx):
    """Two confirm-backs the owner answered no within a week are two explicit wrongs: the tighten rule raises the
    approve bar just above the probabilities read, as a new record, and demotes the live tier, so the next unclear
    answer gets the tables' plain question."""
    from colonymind.instinct.tiers import Instinct
    store = Store(fx.state / "instinct.db")
    provider = reader(raw={"approve": 0.985, "decline": 0.005, "conditional_or_question": 0.005, "unrelated": 0.005})
    store.put_record(record("ask_answer", "reader", provider, bars=BARS), reason="test")
    fx.mind.instinct = made = Instinct(providers={"reader": provider}, reader="reader", records=(), store=store)
    made.set_state("ask_answer", "reader", "live", source="test")
    for code in ("W1X", "W2X"):
        _, _, refused = await confirmed(fx, code, answer="no")
        assert refused["settled"][0]["status"] == "cancelled"
    await made.drain()
    fx.shift(hours=1)
    counts = outcomes.resolve(store, Sources(fx.state), now=fx.now)
    assert counts["ask_answer"]["wrong"] == 2
    [tightened] = store.records()
    assert tightened.bars["approve"] == pytest.approx(0.99) and tightened.source.startswith("tightened")
    assert store.states()[("ask_answer", "reader")] == "demoted"
    fx.mind.instinct = Instinct(providers={"reader": provider}, reader="reader", records=(), store=store)
    row = asked(fx, "K7F")
    result = await unclear_answer(fx, row, "K7F")
    assert fx.store.get(result["clarify"]["message"]).context["text"].startswith("Just to check: ")
