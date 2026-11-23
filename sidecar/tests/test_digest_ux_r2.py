"""The owner's digest and asks: bounded summaries, pair checking and delivery.

Digest summaries are bounded in number and charged to the day's token budget. Every overlapping pair of statements
is checked. An ask that authorizes a destination shows it in full or cannot be approved. No digest arrives on a
later day than its own. No cut word and no internal id is shown.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from colonymind.mind import digest as digest_text
from test_digest_ux import TurnRouter, acted, asked, deliver, items_are_whole, night, no_field_names, the_digest
from test_mind_consolidate import CONTACT, OWNER, Fixture as NightFixture
from owner_answer import owner_answers


def calls(fx, task):
    return [(messages, context) for messages, context in fx.router.calls if context.get("task") == task]


def live(fx, *ids):
    rows = {row["id"]: row for row in fx.claims()}
    return {ident for ident in ids if rows[ident]["retracted_by"] is None}


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
    metric = await fx.fact("r-1", OWNER, "s-1", "My reports use metric units.", "metric units",
                           predicate="report_format")
    fx.shift(hours=1)
    captions = await fx.fact("r-2", OWNER, "s-2", "My reports have captions under charts.", "captions under charts",
                             predicate="report_format")
    fx.shift(hours=1)
    imperial = await fx.fact("r-3", OWNER, "s-3", "My reports use imperial units.", "imperial units",
                             predicate="report_format")
    result = await fx.mind.consolidate()
    assert result["counts"].get("contradictions") == 1
    assert live(fx, metric, captions, imperial) == {captions, imperial}   # only the replaced statement goes


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
# The delivery date: one digest per delivery day
# ---------------------------------------------------------------------------

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
