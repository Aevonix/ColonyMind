"""An ask shown again is shown whole, in the sidecar's own words (docs/MIND.md, Asks).

The Hermes plugin never words an ask: ``colonymind_self show`` asks the sidecar to put it in front of the owner again,
and the sidecar sends its own ``ask_show`` message with the ask's whole sentence (``ask_sentence``). What the model
lists of an ask is a title; a title shown to the owner could stand for terms they never saw (a grant's tools, amounts
and frequency, where a private value goes, what a wish lets the agent do), and the owner's yes to that message would
grant them. These pin that the sidecar's show carries every one of those terms in full.
"""

from __future__ import annotations

import pytest

from colonymind.mind.outbox import ask_parts, message_payload
from test_ask_answers_binding import asked, fx  # noqa: F401  (the answer route's mind fixture)
from test_mind_loop import OWNER, pinned_clock  # noqa: F401  (the pinned wall clock)

GRANT = {"id": "g-bills", "class": "money", "scope": "the utility bills", "tools": ["pay_invoice"], "max_amount": 50,
         "currency": "USD", "per_day": 2}
CASES = {
    "grant_confirm": ({"grant": GRANT}, ["the utility bills", "pay_invoice", "at most 50 USD a time",
                                         "at most 2 times a day"]),
    "secret_use": ({"tool": "pay_invoice", "destination": "https://billing.example/pay/inv-17"},
                   ["pay_invoice", "https://billing.example/pay/inv-17"]),
    "wish": ({"wish": {"topic": "tide tables", "steps": ["research"]}}, ["tide tables"]),
}


@pytest.mark.parametrize("kind", sorted(CASES))
def test_a_shown_ask_carries_every_term_a_yes_grants(fx, kind):  # noqa: F811
    from colonymind.mind.outbox import ask_sentence
    context, terms = CASES[kind]
    row = asked(fx, "G4H", type_=kind, context=context, title="Pay the bills")
    _, authorized = ask_parts(row)
    assert authorized, "this kind of ask authorizes something beyond its lead"
    ident = fx.mind._show_ask(row, ask_sentence(row), key="show:test", now=fx.now, why="the model showed it")
    text = message_payload(fx.store.get(ident), owner_id=OWNER)["text"]
    assert authorized.rstrip(".") in text, text                 # what a yes allows, whole
    for term in terms:
        assert term in text, (term, text)
    assert "G4H" not in text and text.endswith("Reply yes or no.")
