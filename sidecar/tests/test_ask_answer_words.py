"""The owner's words to an ask, read by the answer reader (``colonymind.mind.answers``) wherever the answer route
reads a reply to an ask's own message, for every ask type.

What ties a message to an ask is structure (here a quoted reply to the ask's own message, ``owner_answer``); what the
owner's words say is read by the fixed tables: a plain yes or no in their own words answers; a hedge, a question, a
later or someone else's words do not, and the ask stays open. The earlier chat path, the ask's code typed in the
owner's words (``Mind.answer`` with a message, the strict ``yes <code>`` shape and its readers), is gone.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from colonymind.contacts import changes
from colonymind.mind import answers, authority
from colonymind.mind.tick import Mind
from owner_answer import owner_answers
from test_mind_loop import OWNER, Fixture, pinned_clock  # noqa: F401  (the pinned wall clock)

K7F = "K7F"


@pytest.fixture
def fx(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fixture = Fixture(tmp_path)
    yield fixture
    fixture.store.close()


def asked(fx, code=K7F, *, type_="research", kind="task", cls="self", context=None):
    row, _ = fx.store.create_intention(
        kind=kind, type=type_, title="Re-run the archive export for the board figures", drive="curiosity", cls=cls,
        decision="ask", decision_reason="suggest: self -> ask", status="asked", dedup_key=f"test:{code}",
        ask_code=code, context={"notice": False, **(context or {})}, hermes_kind="none", created_at=fx.now,
        expires_at=fx.now + timedelta(days=3))
    return row


async def answer(fx, message, *, code=K7F):
    return await owner_answers(fx.mind, code, message)


def test_the_code_readers_are_retired():
    assert not hasattr(authority, "answers_ask") and "answers_ask" not in authority.__all__
    assert not hasattr(changes, "answer") and "answer" not in changes.__all__
    assert not hasattr(changes, "SPOKEN_DIGITS")
    assert not hasattr(answers, "references") and "references" not in answers.__all__
    assert not hasattr(Mind, "_bound_by_code") and not hasattr(Mind, "_read_answer_words")


@pytest.mark.parametrize("message", ["sure, go ahead", "👍", "yes please", "Yes, thanks!", "ok sounds good",
                                     "go for it", "please yes"])
async def test_their_plain_yes_in_their_own_words_approves(fx, message):
    row = asked(fx)
    approved = await answer(fx, message)
    assert approved.status == "approved" and fx.store.get(row.id).status == "approved"


@pytest.mark.parametrize("message", ["no thanks", "nope", "👎", "Don't", "no, never mind"])
async def test_their_plain_no_in_their_own_words_declines(fx, message):
    row = asked(fx)
    declined = await answer(fx, message)
    assert declined.status == "cancelled" and fx.store.get(row.id).outcome == "denied"


@pytest.mark.parametrize("message", ["yes but not the payment", "sure?", "yes, if it is cheap", "maybe", "later",
                                     "not now", "yes and no"])
async def test_words_that_do_not_say_a_plain_yes_approve_nothing(fx, message):
    row = asked(fx)
    assert (await answer(fx, message)).status == "asked"
    assert fx.store.get(row.id).status == "asked"


@pytest.mark.parametrize("message", ["Sam said yes", "Here is what Sam wrote: yes, go ahead",
                                     '[Replying to: "yes"]\n\nhm', '"yes" was what Sam wrote'])
async def test_someone_elses_yes_approves_nothing(fx, message):
    """Words the owner passes on (a quotation, a quoted reply, a line naming its speaker) are not their answer."""
    row = asked(fx)
    assert (await answer(fx, message)).status == "asked"
    assert fx.store.get(row.id).status == "asked"


@pytest.mark.parametrize("message", ["k", "go", "Yes. How long will it take?"])
async def test_a_floor_ask_takes_only_a_clean_yes(fx, message):
    """A floor ask is read strict: "k" or "go" alone do not count, and a question anywhere voids the yes. An
    ordinary ask takes the same words."""
    floor = asked(fx, cls="floor", type_="floor_effect")
    assert (await answer(fx, message)).status == "asked"
    assert (await answer(fx, "okay")).status == "approved"
    ordinary = asked(fx, "Q2Z")
    assert (await answer(fx, message, code="Q2Z")).status == "approved"
    assert fx.store.get(ordinary.id).status == "approved" and fx.store.get(floor.id).status == "approved"


@pytest.mark.parametrize("type_", ["people_change", "link_proposal", "cadence_confirm", "grant_confirm"])
async def test_a_widening_ask_is_read_strict_and_takes_a_plain_yes(fx, type_):
    row = asked(fx, type_=type_, kind="note", cls="owner")
    for message in ("k", "sure. why?", "yes, only for today", "yes. Only for today.", "yes, and only for Robin"):
        assert (await answer(fx, message)).status == "asked", message
    await answer(fx, "yes thanks")
    assert fx.store.get(row.id).status != "asked"


async def test_a_wish_is_an_ordinary_ask(fx):
    """After the typed grant a yes to a wish gives only its typed steps: it is read like any ordinary ask."""
    row = asked(fx, type_="wish", kind="note", cls="self")
    await answer(fx, "go")
    assert fx.store.get(row.id).status != "asked"


async def test_an_expired_question_is_expired_whatever_their_words(fx):
    row = asked(fx, type_="people_change", kind="note", cls="owner")
    fx.shift(days=4)
    settled = await answer(fx, "yes")
    assert settled.status == "expired"
    assert fx.store.get(row.id).status == "expired"
