"""Learned proposals: the mind offers to stop asking once the owner keeps saying yes (docs/MIND.md, Standing grants and
allows).

After ``mind.asks.propose_after`` (3) of the owner's own yeses in a row within 30 days to asks that offer the same
"always", with no no of theirs between, the mind asks once: stop asking before it does that again? The question is a
widening ask (its own message, read strict). A yes adds the standing allow; a no or silence holds the offer back for
30 days, and only yeses after that count toward the next one. An operator's answer and a lapse neither count nor
break the row. Nothing is ever allowed without the owner's own yes.
"""

from __future__ import annotations

from owner_answer import owner_answers
from test_mind_allows import DECLARED, Contacts, _dispatched, allows, formed, make  # noqa: F401  (make: the fixture)
from test_mind_loop import CONTACT, pinned_clock  # noqa: F401  (the pinned wall clock)

PROPOSAL = "allow_proposal"


async def answered(fx, count, *, start=0, answer="yes"):
    """``count`` asks to reach the contact, each answered by the owner's own reply; each commitment is then settled,
    so none forms again while a test moves the clock by days."""
    rows = []
    for number in range(start, start + count):
        row = await formed(fx, CONTACT, f"send the contact the notes, part {number}")
        assert row.status == "asked", row.decision_reason
        rows.append(await owner_answers(fx.mind, row.ask_code, answer))
        fx.commitments.resolve(str(row.source_id), outcome="done", resolved_by="owner")
    return rows


def proposals(fx, status=None):
    return [row for row in fx.store.intentions(limit=500) if row.type == PROPOSAL
            and (status is None or row.status == status)]


async def test_three_yeses_in_a_row_bring_one_offer_to_stop_asking(make):
    fx = make()
    await answered(fx, 2)
    assert proposals(fx) == []
    await answered(fx, 1, start=2)
    proposal, = proposals(fx)
    assert proposal.status == "asked" and proposal.ask_code and proposal.cls == "owner"
    assert "3" in proposal.description and "Stop asking" in proposal.description
    assert (proposal.context["proposes"]["kind"], proposal.context["proposes"]["target"]) == ("contact", CONTACT)
    assert "always" not in proposal.context                                # the offer is the question itself
    assert fx.mind._weighty(proposal)                                      # a widening ask: read strict
    assert (await owner_answers(fx.mind, proposal.ask_code, "k")).status == "asked"
    assert allows(fx) == []                                                 # nothing without the owner's yes
    await owner_answers(fx.mind, proposal.ask_code, "yes")
    allow, = allows(fx)
    assert (allow["kind"], allow["target"], allow["ask"]) == ("contact", CONTACT, proposal.id)
    assert fx.contacts.records[CONTACT]["may_contact"] == "auto"
    assert (await formed(fx, CONTACT, "send the contact the summary")).decision != "ask"   # acts, or waits its budget
    assert len(proposals(fx)) == 1


async def test_a_no_breaks_the_row_and_operator_answers_and_lapses_do_not_count(make):
    fx = make()
    await answered(fx, 2)
    await answered(fx, 1, start=2, answer="no")
    await answered(fx, 2, start=3)
    console = await formed(fx, CONTACT, "send the contact the draft")
    await fx.mind.answer(console.ask_code, yes=False, by="cli")
    await formed(fx, CONTACT, "send the contact the photos")
    fx.shift(hours=73)
    await fx.mind.tick(force=True)                                         # that one lapses: no verdict
    assert proposals(fx) == []
    await answered(fx, 1, start=5)
    assert len(proposals(fx, "asked")) == 1


async def test_a_no_or_silence_holds_the_offer_back_for_thirty_days(make):
    fx = make()
    await answered(fx, 3)
    first, = proposals(fx)
    await owner_answers(fx.mind, first.ask_code, "no")
    await answered(fx, 3, start=3)
    assert len(proposals(fx)) == 1                                         # held back
    fx.shift(days=31)
    await answered(fx, 3, start=6)
    second, = proposals(fx, "asked")
    fx.shift(hours=73)
    await fx.mind.tick(force=True)
    assert fx.store.get(second.id).status == "expired" and allows(fx) == []
    await answered(fx, 3, start=9)
    assert proposals(fx, "asked") == []                                    # silence holds it back too


async def test_a_revoked_allow_is_offered_again_only_after_new_yeses(make):
    fx = make()
    await answered(fx, 3)
    proposal, = proposals(fx)
    await owner_answers(fx.mind, proposal.ask_code, "yes")
    allow, = allows(fx)
    await fx.mind.revoke_allow(allow["id"], by="cli")
    await answered(fx, 2, start=3)                                         # the yeses before it no longer count
    assert proposals(fx, "asked") == []
    await answered(fx, 1, start=5)
    assert len(proposals(fx, "asked")) == 1


async def test_proposals_follow_the_setting_and_never_come_at_suggest(make):
    fx = make(config={"asks": {"propose_after": 0}})
    await answered(fx, 4)
    assert proposals(fx) == []
    fx = make(config={"asks": {"propose_after": 2}})
    await answered(fx, 2)
    assert len(proposals(fx)) == 1
    fx = make(autonomy="suggest")
    await answered(fx, 3)
    assert proposals(fx) == []


async def test_payments_to_one_payee_count_toward_one_offer_whatever_else_the_call_carries(make):
    """The offer counts yeses per permission (a held call's class, tool, payee and currency), never per complete call:
    three payments to the same payee for different invoices bring the offer, and its yes adds one grant for that payee
    at most the amount last asked, which covers the next invoice and no other payee."""
    fx = make(config={"floor": {"tools": DECLARED}})
    keys = set()
    for number in range(3):
        task = _dispatched(fx, f"pay invoice {number}")
        pay = {"amount": 40, "currency": "usd", "payee": "acme", "invoice": f"INV-{number}"}
        held = await fx.mind.guard(tool="pay_invoice", args=pay, run="mind", task_id=task.hermes_ref)
        ask = fx.store.get_by_ask_code(held["held"])
        keys.add(ask.context["always"]["key"])
        assert (await owner_answers(fx.mind, ask.ask_code, "yes")).status in {"approved", "dispatched"}
        fx.shift(days=1)                                   # the permit's one call made or not, a day apart
    assert len(keys) == 1
    proposal, = proposals(fx, "asked")
    assert proposal.context["proposes"]["key"] in keys and "acme" in proposal.description
    await owner_answers(fx.mind, proposal.ask_code, "yes")
    allow, = allows(fx)
    grant, = [item for item in fx.mind.grants_state()["grants"] if item["id"] == allow["grant"]]
    assert (grant["tools"], grant["match"], grant["max_amount"], grant["per_day"]) == (["pay_invoice"], ["acme"], 40, 1)
    fx.shift(days=1)
    for payee, runs in (("acme", True), ("someone-else", False)):
        task = _dispatched(fx, f"pay {payee}")
        pay = {"amount": 40, "currency": "usd", "payee": payee, "invoice": "INV-9"}
        assert (await fx.mind.guard(tool="pay_invoice", args=pay, run="mind", task_id=task.hermes_ref))["allow"] is runs
