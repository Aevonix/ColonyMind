"""A cadence set from conversation is a change like any other: one that checks in more often waits for the
owner's yes.

Capture records the rhythm and the matter (case 4) and nothing of the owner's words: no words stand for a
yes. The tick applies a captured cadence that only slows the contact's own or keeps it (compare-and-set
against the cadence it decided on); a new or faster one is the owner's one-tap ``cadence_confirm`` question,
and their own reply ``yes <code>`` sets it. A question that expires unanswered leaves the obligation open and
unapplied; the owner saying it again after that asks again.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from colonymind.commitments import extract
from colonymind.commitments.extract import CommitmentExtractor, record_items
from colonymind.commitments.store import CommitmentStore
from colonymind.mind import drives
from test_mind_cadence import CADENCE_ITEM, ITEM, CadenceRouter, cadence_log
from test_mind_social import CADENCE, CONTACT, OWNER, PAST, C, contact, make  # noqa: F401
from owner_answer import owner_answers

SAID = f"Please check in with {CONTACT} every {CADENCE} minutes about the {ITEM}; they asked for it."
UNWORDED = f"{CONTACT} would like updates every {CADENCE} minutes about the {ITEM}."


def item(asked=None, **metadata):
    extra = {"asked": asked} if asked is not None else {}
    return {**CADENCE_ITEM, "metadata": {**CADENCE_ITEM["metadata"], **extra, **metadata}}


def recorded(tmp_path, asked, said=SAID):
    tmp_path.mkdir()
    store = CommitmentStore(tmp_path / "colonymind-commitments.db")
    result = record_items([item(asked)], person_id=OWNER, commitment_store=store, existing=[], rejections=[],
                          turn_id="turn-1", owner_id=OWNER, owner_text=said)
    return store.get(result["created"][0])["metadata"]


def resolved(**extra):
    metadata = {**CADENCE_ITEM["metadata"], "recipient_id": CONTACT, "recipient_exact": True, **extra}
    return {"id": "c-1", "person_id": OWNER, "status": "pending", "metadata": metadata}


def owner_says(fx, text, turn="turn-1"):
    fx.ledger.record_source(turn, contact_id=OWNER, session_id="owner-1", occurred_at=fx.now.isoformat(),
                            messages=[{"role": "user", "content": text}, {"role": "assistant", "content": "Noted."}])


def questions(fx):
    return [row for row in fx.store.intentions(status=["asked"], limit=20) if row.type == "cadence_confirm"]


def test_the_extractor_contract_asks_for_no_words_on_a_cadence():
    case = extract.SYSTEM.split("4. A RECURRING CHECK-IN", 1)[1].split("\n\n", 1)[0]
    assert '"asked"' not in case and '"cadence_minutes":<N in minutes>}' in case


def test_capture_keeps_no_words_for_a_cadence_whatever_the_extractor_quoted(tmp_path):
    for name, quote in (("a", f"check in with {CONTACT} every {CADENCE} minutes"), ("b", None)):
        metadata = recorded(tmp_path / name, quote)
        assert "asked" not in metadata and metadata["cadence_minutes"] == CADENCE and metadata["recipient"] == CONTACT


def test_owner_cadence_is_the_exact_recipients_rhythm():
    assert drives.owner_cadence(resolved(), owner_id=OWNER) == (CONTACT, CADENCE, f"the {ITEM}")
    assert drives.owner_cadence(resolved(recipient_exact=False), owner_id=OWNER) is None
    assert drives.owner_cadence(resolved(), owner_id="p-someone-else") is None


def test_the_question_is_asked_for_a_name_match_and_for_an_exact_recipient_only_when_it_raises():
    assert drives.cadence_confirm_candidate(resolved(), owner_id=OWNER) is None       # the tick decides that one
    raised = drives.cadence_confirm_candidate(resolved(), owner_id=OWNER, current=None)
    assert raised.type == "cadence_confirm" and raised.recipient == OWNER and raised.kind == "note"
    assert CONTACT in raised.text and f"every {CADENCE} minutes" in raised.text and "none now" in raised.text
    faster = drives.cadence_confirm_candidate(resolved(), owner_id=OWNER, current=60)
    assert "now every hour" in faster.text
    named = drives.cadence_confirm_candidate(resolved(recipient_exact=False, recipient="Anna"), owner_id=OWNER)
    assert "Is that who you meant?" in named.text and "'Anna'" in named.text
    assert drives.cadence_confirm_candidate(resolved(cadence_applied=True), owner_id=OWNER, current=None) is None


def test_the_question_states_the_interval_it_would_set_exactly():
    question = drives.cadence_confirm_candidate(resolved(cadence_minutes=90), owner_id=OWNER, current=None)
    assert "every 90 minutes" in question.text and "1 h" not in question.text
    named = drives.cadence_confirm_candidate(resolved(cadence_minutes=90, recipient_exact=False), owner_id=OWNER)
    assert "every 90 minutes" in named.text


def test_each_round_of_the_question_has_its_own_key():
    assert drives.cadence_question_key(resolved()) == "commitment:c-1:cadence"
    assert drives.cadence_question_key(resolved(confirm_round=2)) == "commitment:c-1:cadence:2"
    again = drives.cadence_confirm_candidate(resolved(confirm_round=2), owner_id=OWNER, current=None)
    assert again.dedup_key == "commitment:c-1:cadence:2"


async def test_a_cadence_the_owner_asked_for_in_so_many_words_still_waits_for_his_yes(make):
    fx = make([contact(CONTACT, may_contact="auto")], router=CadenceRouter(item(SAID)))
    fx.mind.capture = CommitmentExtractor(fx.ledger, lambda: fx.commitments)
    owner_says(fx, SAID)
    await fx.tick()
    assert fx.contacts.records[CONTACT]["cadence_minutes"] is None and cadence_log(fx) == []
    question, = questions(fx)
    assert question.ask_code and CONTACT in question.description
    await fx.tick()
    assert len(questions(fx)) == 1                                              # asked once
    await owner_answers(fx.mind, question.ask_code)
    assert fx.contacts.records[CONTACT]["cadence_minutes"] == CADENCE
    fx.shift(C + PAST)
    formed = [entry for tick in [await fx.tick(), await fx.tick()] for entry in tick["formed"]]
    assert [entry["type"] for entry in formed] == ["check_in"]
    payload, = [p for p in await fx.send_all() if p["recipient"] == CONTACT]
    assert ITEM in payload["text"]


async def test_a_no_to_a_cadence_withdraws_it(make):
    fx = make([contact(CONTACT, may_contact="auto")], router=CadenceRouter(item()))
    fx.mind.capture = CommitmentExtractor(fx.ledger, lambda: fx.commitments)
    owner_says(fx, UNWORDED)
    await fx.tick()
    question, = questions(fx)
    await owner_answers(fx.mind, question.ask_code, "no")
    row, = fx.commitments.list(limit=10)["commitments"]
    assert row["status"] != "pending" and fx.contacts.records[CONTACT]["cadence_minutes"] is None
    fx.shift(timedelta(minutes=CADENCE) + PAST)
    assert (await fx.tick())["formed"] == [] and fx.messages_to(CONTACT) == []


async def test_a_reply_that_is_not_the_owners_own_answer_settles_nothing(make):
    """A reply that only asks about the question, or makes its yes a condition, neither sets the cadence nor
    withdraws the owner's ask; neither does the console's yes."""
    fx = make([contact(CONTACT, may_contact="auto")], router=CadenceRouter(item()))
    fx.mind.capture = CommitmentExtractor(fx.ledger, lambda: fx.commitments)
    owner_says(fx, UNWORDED)
    await fx.tick()
    question, = questions(fx)
    code = question.ask_code
    for message in ("what is it?", "what is it? do not approve it", "If I say yes, go ahead"):
        await owner_answers(fx.mind, code, message)
    with pytest.raises(PermissionError):
        await fx.mind.answer(code, yes=True)
    row, = fx.commitments.list(limit=10)["commitments"]
    assert row["status"] == "pending" and fx.store.get(question.id).status == "asked"
    assert fx.contacts.records[CONTACT]["cadence_minutes"] is None


async def test_the_cadence_question_forms_even_with_the_duty_drive_off(make):
    """The question is the owner's, not duty work: the tick forms it where it would have applied the cadence,
    so a duty weight of 0 neither applies the cadence silently nor loses it."""
    fx = make([contact(CONTACT, may_contact="auto")], router=CadenceRouter(item()), config={"drives": {"duty": 0.0}})
    fx.mind.capture = CommitmentExtractor(fx.ledger, lambda: fx.commitments)
    owner_says(fx, UNWORDED)
    await fx.tick()
    assert fx.contacts.records[CONTACT]["cadence_minutes"] is None
    question, = fx.store.intentions(status=["asked"], limit=10)
    assert question.type == "cadence_confirm" and question.ask_code


async def test_a_cadence_answered_after_its_ask_expired_sets_nothing(make):
    fx = make([contact(CONTACT, may_contact="auto")], router=CadenceRouter(item()))
    fx.mind.capture = CommitmentExtractor(fx.ledger, lambda: fx.commitments)
    owner_says(fx, UNWORDED)
    await fx.tick()
    question, = questions(fx)
    fx.mind.clock = lambda: question.expires_at
    settled = await owner_answers(fx.mind, question.ask_code)
    assert settled.status == "expired" and fx.contacts.records[CONTACT]["cadence_minutes"] is None


async def test_an_expired_question_keeps_the_obligation_and_saying_it_again_asks_again(make):
    fx = make([contact(CONTACT, may_contact="auto")], router=CadenceRouter(item()))
    fx.mind.capture = CommitmentExtractor(fx.ledger, lambda: fx.commitments)
    owner_says(fx, UNWORDED)
    await fx.tick()
    first, = questions(fx)
    fx.shift(timedelta(hours=80))
    await fx.tick()
    await fx.tick()
    assert fx.store.get(first.id).status == "expired" and questions(fx) == []   # no nagging on its own
    row, = fx.commitments.list(limit=10)["commitments"]
    assert row["status"] == "pending" and not row["metadata"].get("cadence_applied")   # the obligation stays
    owner_says(fx, UNWORDED, turn="turn-2")
    await fx.tick()
    again, = questions(fx)
    assert again.id != first.id and again.ask_code != first.ask_code
    kept, = fx.commitments.list(limit=10)["commitments"]
    assert kept["id"] == row["id"]                                              # the same obligation, asked again
    await owner_answers(fx.mind, again.ask_code)
    assert fx.contacts.records[CONTACT]["cadence_minutes"] == CADENCE


async def test_an_expired_name_match_question_keeps_the_obligation_too(make):
    named = item(recipient="Anna")
    fx = make([contact(CONTACT, may_contact="auto", name="Anna")], router=CadenceRouter(named))
    fx.mind.capture = CommitmentExtractor(fx.ledger, lambda: fx.commitments)
    owner_says(fx, f"Please check in with Anna every {CADENCE} minutes about the {ITEM}")
    await fx.tick()
    first, = questions(fx)
    assert "Is that who you meant?" in first.description
    fx.shift(timedelta(hours=80))
    await fx.tick()
    await fx.tick()
    row, = fx.commitments.list(limit=10)["commitments"]
    assert row["status"] == "pending" and questions(fx) == []
    owner_says(fx, f"Please check in with Anna every {CADENCE} minutes about the {ITEM}", turn="turn-2")
    await fx.tick()
    again, = questions(fx)
    assert again.id != first.id


async def test_a_captured_slower_cadence_applies_without_an_ask(make):
    fx = make([contact(CONTACT, may_contact="auto", cadence=5)], router=CadenceRouter(item()))
    fx.mind.capture = CommitmentExtractor(fx.ledger, lambda: fx.commitments)
    owner_says(fx, UNWORDED)
    await fx.tick()
    assert fx.contacts.records[CONTACT]["cadence_minutes"] == CADENCE
    assert questions(fx) == []


async def test_a_captured_slower_cadence_never_lands_over_a_clear_made_meanwhile(make):
    """Decided as a slowdown of 5 minutes, the write is made against 5: the owner clearing the cadence while it
    was decided wins, and the next tick decides again on no cadence (a raise: the owner's question)."""
    fx = make([contact(CONTACT, may_contact="auto", cadence=5)], router=CadenceRouter(item()))
    fx.mind.capture = CommitmentExtractor(fx.ledger, lambda: fx.commitments)
    owner_says(fx, UNWORDED)
    real, reads = fx.mind._contact_record, []

    async def racing(contact_id, **kwargs):
        record = await real(contact_id, **kwargs)
        reads.append(contact_id)
        if len(reads) == 1:
            fx.contacts.records[CONTACT]["cadence_minutes"] = None               # the owner clears it meanwhile
        return record

    fx.mind._contact_record = racing
    await fx.tick()
    assert fx.contacts.records[CONTACT]["cadence_minutes"] is None
    fx.mind._contact_record = real
    await fx.tick()
    assert fx.contacts.records[CONTACT]["cadence_minutes"] is None and len(questions(fx)) == 1
