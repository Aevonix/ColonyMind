"""What an ask looks like and how it reaches the owner (docs/MIND.md, Asks; the daily digest).

Each open ask goes to the owner as its own message, in at most three plain sentences with no code, no internal id and
no expiry line: what the agent wants to do with its object named, what a yes cannot undo where that applies, and the
reply line ("Reply yes or no."). Its delivery id is what a reaction, a quoted reply or the owner's next message binds
to (``Mind.read_answer``), so no code is ever needed; the code stays the ask's id for the console and the API. The
digest's first part lists what waits for the owner's OK in words, with no reply instruction: the ask's own message
carries it.
"""

from __future__ import annotations

import re
from datetime import timedelta

import pytest

from colonymind.mind import digest as digest_text
from colonymind.mind import outbox as outbox_module
from colonymind.mind.outbox import ASK_MESSAGE
from test_mind_loop import CONTACT, OWNER, Fixture, pinned_clock  # noqa: F401  (the pinned wall clock)

REPLY = "Reply yes or no."


@pytest.fixture
def fx(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fixture = Fixture(tmp_path)
    fixture.mind.digest_hour = 0
    fixture.mind.quiet = None
    yield fixture
    fixture.store.close()


def asked(fx, code, title="Re-run the archive export for the board figures", *, type_="research", kind="task",
          cls="self", notice=True, context=None, recipient=None):
    row, _ = fx.store.create_intention(
        kind=kind, type=type_, title=title, drive="curiosity", cls=cls, decision="ask",
        decision_reason="standard: self -> ask", status="asked", dedup_key=f"test:{code}", ask_code=code,
        context={"notice": notice, **(context or {})}, hermes_kind="none", created_at=fx.now,
        expires_at=fx.now + timedelta(days=3), recipient=recipient)
    fx.store.transition(row.id, "asked", action="asked", at=fx.now, details={"code": code, "notice": notice})
    return row


def ask_messages(fx, *, status=None):
    rows = fx.store.intentions(kind=["message"], types=[ASK_MESSAGE], limit=500)
    return [row for row in rows if status is None or row.status in status]


def deliver(fx, message, *, chat="owner-chat", ref=None):
    """The body sends one queued owner message and reports it sent."""
    fx.mind.outbox.sending(message.id, target=f"telegram:{chat}")
    return fx.mind.outbox.sent(message.id, result="sent", hermes_ref=ref or f"out-{message.id}")


def owner_texts(fx):
    return [row.context.get("text") or "" for row in fx.store.intentions(kind=["message"], limit=500)
            if row.entity_id == OWNER]


def no_code_in(text, codes):
    for code in codes:
        assert not re.search(rf"\b{code}\b", text), (code, text)
    assert "code" not in text.lower(), text
    assert "expire" not in text.lower(), text


def listed_1_11(fx, rows):
    """A 1.11 notice that listed ``rows`` by code, delivered."""
    text = "\n".join([f"{digest_text.NEEDS_OK_1_11}:", *(f"- {row.ask_code}: {row.description}." for row in rows)])
    notice = fx.mind.outbox.notice(type=ASK_MESSAGE, title=f"{len(rows)} open ask(s)", text=text,
                                   dedup_key="ask_notice:1.11")
    return deliver(fx, notice)


# -- one message per ask, with its reply line --------------------------------------------------------------------------

async def test_each_open_ask_goes_as_its_own_message_with_its_reply_line_and_no_code(fx):
    first = asked(fx, "K7F")
    second = asked(fx, "Q2Z", "Check the venue hours before Friday")
    text, _ = fx.store.create_intention(
        kind="message", type="reach_out:check_in", title="Check in with Sam", drive="social", cls="contact",
        decision="ask", decision_reason="ask", status="asked", dedup_key="test:M4P", ask_code="M4P",
        context={"notice": True, "text": "Hi Sam, how did the move go?"}, hermes_kind="none", created_at=fx.now,
        expires_at=fx.now + timedelta(days=3), recipient=CONTACT)
    summary = await fx.mind.tick(force=True)
    messages = ask_messages(fx)
    assert sorted(summary["ask_messages"]) == sorted(row.id for row in messages) and len(messages) == 3
    by_ask = {row.context["asks"][0]: row for row in messages}
    assert set(by_ask) == {first.id, second.id, text.id}
    for ask_id, row in by_ask.items():
        assert row.context["asks"] == [ask_id] and row.context["list"] == [ask_id]
        assert row.status == "approved" and row.entity_id == OWNER
        assert row.context["text"].endswith(REPLY)
        no_code_in(row.context["text"], ("K7F", "Q2Z", "M4P"))
        assert "K7F" not in row.description and "Q2Z" not in row.description
    assert by_ask[first.id].context["text"] == f"Re-run the archive export for the board figures. {REPLY}"
    assert by_ask[text.id].context["text"] == f"Send a contact this message: \"Hi Sam, how did the move go?\" {REPLY}"
    # Once each ask has its message, a later tick queues nothing more.
    assert (await fx.mind.tick(force=True))["ask_messages"] == [] and len(ask_messages(fx)) == 3


async def test_a_suggest_level_ask_waits_for_the_digest_then_goes_as_its_own_message(fx):
    row = asked(fx, "K7F", notice=False)
    await fx.mind.tick(force=True)
    assert ask_messages(fx) == []
    digest, = fx.store.intentions(kind=["message"], types=["digest"], limit=5)
    assert "Re-run the archive export" in digest.context["text"]
    await fx.mind.tick(force=True)
    assert ask_messages(fx) == []                       # the digest is on its way, not delivered yet
    deliver(fx, digest)
    fx.shift(minutes=1)
    await fx.mind.tick(force=True)
    message, = ask_messages(fx)
    assert message.context["asks"] == [row.id] and message.context["text"].endswith(REPLY)


async def test_a_floor_ask_says_what_its_yes_cannot_undo(fx):
    effect = {"cls": "irreversible_deletion", "tool": "terminal", "target": "old-builds",
              "text": "irreversible deletion: terminal 'rm -r old-builds'"}
    asked(fx, "K7F", "Let this run once: delete old-builds", type_="floor_effect", cls="floor",
          context={"effect": effect})
    await fx.mind.tick(force=True)
    message, = ask_messages(fx)
    assert message.context["text"] == (
        "Let this call run once. The call: irreversible deletion: terminal 'rm -r old-builds'. "
        "What is deleted cannot be recovered. Reply yes or no.")


async def test_no_text_bound_for_the_owner_carries_a_code_or_an_expiry(fx):
    codes = ("K7F", "Q2Z", "M4P")
    asked(fx, "K7F")
    asked(fx, "Q2Z", "Allow a standing grant.", type_="grant_confirm", kind="note",
          context={"grant": {"id": "g1", "scope": "the weekly report " * 120, "class": "bulk_third_party_messaging"}})
    asked(fx, "M4P", "Check the venue hours before Friday", notice=False)
    await fx.mind.tick(force=True)
    for message in fx.store.intentions(kind=["message"], limit=50):
        if message.status == "approved":
            deliver(fx, message)
    fx.shift(hours=1)
    await fx.mind._digest(fx.now)
    texts = owner_texts(fx)
    assert any(text.startswith("Daily digest") for text in texts)
    assert any("in full" in text for text in texts)            # the grant too long to show went on its own
    for text in texts:
        no_code_in(text, codes)
    for line in fx.mind._waiting_lines(2000):
        no_code_in(line, codes)


async def test_the_digest_lists_what_waits_in_words_with_no_reply_instruction(fx):
    asked(fx, "K7F")
    fx.shift(hours=5)
    asked(fx, "Q2Z", "Check the venue hours before Friday")
    fx.shift(minutes=30)
    await fx.mind._digest(fx.now)
    digest, = [row for row in fx.store.intentions(kind=["message"], types=["digest"], limit=5)]
    text = digest.context["text"]
    assert text.split("\n\n")[1].splitlines()[0] == f"{digest_text.WAITING}:"
    assert digest_text.lines_of(text, digest_text.WAITING) == [
        "Re-run the archive export for the board figures. Asked 5 hours ago.",
        "Check the venue hours before Friday. Asked less than an hour ago.",
    ]
    assert "reply" not in text.lower() and "code" not in text.lower() and "expire" not in text.lower()


async def test_an_ask_message_not_yet_sent_is_withdrawn_once_its_ask_is_settled(fx):
    row = asked(fx, "K7F")
    await fx.mind.tick(force=True)
    message, = ask_messages(fx)
    await fx.mind.answer("K7F", yes=False, by="cli")
    await fx.mind.tick(force=True)
    assert fx.store.get(message.id).status == "cancelled" and fx.store.get(row.id).outcome == "denied"
    assert [item for item in await fx.mind.outbox_ready() if item["type"] == ASK_MESSAGE] == []


def test_the_waiting_lines_of_the_turns_mind_section_name_no_code(fx):
    asked(fx, "K7F")
    fx.store.create_intention(
        kind="message", type="reach_out:check_in", title="Check in with Sam", drive="social", cls="contact",
        decision="ask", decision_reason="ask", status="asked", dedup_key="test:M4P", ask_code="M4P",
        context={"notice": True, "text": "Hi Sam, how did the move go?"}, hermes_kind="none", created_at=fx.now,
        expires_at=fx.now + timedelta(days=3), recipient=CONTACT)
    lines = fx.mind._waiting_lines(2000)
    assert lines == ["Waiting for your say on: Re-run the archive export for the board figures.",
                     "Waiting for your say on: a message to a contact: \"Hi Sam, how did the move go?\""]
    assert fx.mind._waiting_lines(60) == ["Waiting for your say on: Re-run the archive export for the board figures."]


# -- at most three a day; the rest in one list -----------------------------------------------------------------------

async def test_asks_beyond_the_days_three_go_in_one_list_whose_numbers_bind(fx):
    fx.mind.config["asks"] = {"remind_hours": 168}               # no reminder competes for the next day's room
    rows = [asked(fx, code, f"Check the venue hours for event {number}") for number, code in
            enumerate(("K7F", "Q2Z", "M4P", "R8T", "W3X"), 1)]
    await fx.mind.tick(force=True)
    singles = [row for row in ask_messages(fx) if row.context["ask_message"] == "first"]
    listed, = [row for row in ask_messages(fx) if row.context["ask_message"] == "list"]
    assert sorted(row.context["asks"][0] for row in singles) == sorted(row.id for row in rows[:3])
    assert listed.context["asks"] == listed.context["list"] == [rows[3].id, rows[4].id]
    assert listed.context["text"] == ("2 more things wait for your OK: 1. Check the venue hours for event 4. "
                                      "2. Check the venue hours for event 5. Reply 1, 2, both or neither.")
    for row in singles:
        deliver(fx, row)
    # A list's numbers bind to it: "2" in reply approves the second.
    deliver(fx, listed, ref="list-1")
    fx.shift(minutes=1)
    result = await fx.mind.read_answer({"platform": "telegram", "contact_id": OWNER, "chat_id": "owner-chat",
                                        "chat_type": "dm", "turn_id": "t-in-1", "message_id": "in-1",
                                        "arrived_at": fx.now.isoformat(), "text": "2",
                                        "reply_to_message_id": "list-1", "ledger": None})
    assert [item["id"] for item in result["settled"]] == [rows[4].id]
    # The day's room is spent: an ask formed later that day waits (the digest lists it), and goes the next day.
    later = asked(fx, "Z9Y", "Check the venue hours for event 6")
    fx.shift(hours=2)
    await fx.mind.tick(force=True)
    assert not [row for row in ask_messages(fx) if later.id in row.context["asks"]]
    fx.shift(hours=23)
    await fx.mind.tick(force=True)
    own, = [row for row in ask_messages(fx) if later.id in row.context["asks"]]
    assert own.context["ask_message"] == "first" and own.context["list"] == [later.id]


async def test_messages_still_queued_hold_the_days_room_however_long_the_body_is_away(fx):
    """The day's three count every ask message still on its way whatever its age, and each one delivered by when
    it reached the owner: after a day offline, the three queued before still hold the room, the asks formed
    meanwhile go in one list, and when the body comes back the owner gets three messages and a list, not six."""
    fx.mind.config["asks"] = {"remind_hours": 168}               # no reminder competes for the next day's room
    early = [asked(fx, code, f"Check the venue hours for event {number}") for number, code in
             enumerate(("K7F", "Q2Z", "M4P"), 1)]
    await fx.mind.tick(force=True)
    assert len(ask_messages(fx, status={"approved"})) == 3
    fx.shift(hours=25)                                            # the body is away; nothing was claimed
    late = [asked(fx, code, f"Check the venue hours for event {number}") for number, code in
            enumerate(("R8T", "W3X", "Z9Y"), 4)]
    await fx.mind.tick(force=True)
    queued = ask_messages(fx, status={"approved"})
    singles = [row for row in queued if row.context["ask_message"] in {"first", "reminder"}]
    listed, = [row for row in queued if row.context["ask_message"] == "list"]
    assert sorted(row.context["asks"][0] for row in singles) == sorted(row.id for row in early)
    assert listed.context["asks"] == [row.id for row in late]
    for row in queued:                                            # the body is back: everything goes at once
        deliver(fx, row)
    fx.shift(hours=2)
    another = asked(fx, "V5N", "Check the venue hours for event 7")
    await fx.mind.tick(force=True)
    assert sends_of(fx, another) == []                           # the room counts from delivery, not from queueing
    fx.shift(hours=22, minutes=1)
    await fx.mind.tick(force=True)
    own, = sends_of(fx, another)
    assert own.context["ask_message"] == "first"


async def test_the_day_holds_as_many_ask_messages_as_the_owner_sets(fx):
    fx.mind.config["asks"] = {"messages_per_day": 1}             # mind.asks.messages_per_day
    rows = [asked(fx, code, f"Check the venue hours for event {number}") for number, code in
            enumerate(("K7F", "Q2Z", "M4P", "R8T"), 1)]
    await fx.mind.tick(force=True)
    single, = [row for row in ask_messages(fx) if row.context["ask_message"] == "first"]
    listed, = [row for row in ask_messages(fx) if row.context["ask_message"] == "list"]
    assert single.context["asks"] == [rows[0].id] and listed.context["list"] == [row.id for row in rows[1:]]
    assert listed.context["text"].startswith("3 more things wait for your OK: 1. ")
    assert listed.context["text"].endswith(" Reply with the numbers you approve, all or none.")


async def test_a_weighty_ask_always_goes_on_its_own_outside_the_days_count(fx):
    """A floor ask, or one that widens who gets messaged or who is who, is never approved from a list, so it always
    goes as its own message, and it spends none of the day's three."""
    ordinary = [asked(fx, code, f"Check the venue hours for event {number}") for number, code in
                enumerate(("K7F", "Q2Z", "M4P"), 1)]
    floor = asked(fx, "R8T", "Let this run once: delete old-builds", type_="floor_effect", cls="floor",
                  context={"effect": {"cls": "irreversible_deletion", "text": "irreversible deletion: rm old-builds"}})
    change = asked(fx, "W3X", "Let Sam be messaged any time", type_="people_change", kind="note")
    await fx.mind.tick(force=True)
    messages = ask_messages(fx)
    assert all(row.context["ask_message"] == "first" for row in messages) and len(messages) == 5
    assert {row.context["asks"][0] for row in messages} == {row.id for row in (*ordinary, floor, change)}


async def test_asks_a_1_11_notice_listed_each_go_once_on_their_own(fx):
    """At the upgrade, each ask still open from a 1.11 notice gets its own plain message once, outside the day's
    count, and binds like any new ask from then on."""
    rows = [asked(fx, code, f"Check the venue hours for event {number}") for number, code in
            enumerate(("K7F", "Q2Z", "M4P", "R8T", "W3X"), 1)]
    listed_1_11(fx, rows)
    fx.shift(minutes=1)
    await fx.mind.tick(force=True)
    messages = ask_messages(fx, status={"approved"})
    assert len(messages) == 5 and all(row.context["ask_message"] == "first" for row in messages)
    assert {row.context["asks"][0] for row in messages} == {row.id for row in rows}


# -- one reminder, aligned with the digest ---------------------------------------------------------------------------

def a_digest_delivered(fx, day):
    row = fx.mind.outbox.queue_digest(local_date=f"test-{day}", text=f"Daily digest\n\nDone or found since yesterday:"
                                                                   f"\n- Done: the backup check {day}.")
    return deliver(fx, row)


def sends_of(fx, row):
    return sorted((message for message in ask_messages(fx) if row.id in message.context["asks"]),
                  key=lambda message: message.created_at)


async def delivered_ask(fx, code="K7F", **fields):
    row = asked(fx, code, **fields)
    await fx.mind.tick(force=True)
    first, = sends_of(fx, row)
    return row, deliver(fx, first)


async def test_the_one_reminder_goes_right_after_the_first_digest_twelve_hours_on(fx):
    row, first = await delivered_ask(fx)
    fx.shift(hours=6)
    a_digest_delivered(fx, 1)                          # too soon after the ask's own message: not the reminder
    await fx.mind.tick(force=True)
    assert len(sends_of(fx, row)) == 1
    fx.shift(hours=7)
    a_digest_delivered(fx, 2)
    await fx.mind.tick(force=True)
    _, reminder = sends_of(fx, row)
    assert reminder.context["ask_message"] == "reminder" and reminder.context["asks"] == [row.id]
    assert reminder.context["text"] == first.context["text"]
    deliver(fx, reminder)
    # Never a second: not after another digest, not a day later.
    for day in (3, 4):
        fx.shift(hours=24)
        a_digest_delivered(fx, day)
        await fx.mind.tick(force=True)
    assert len(sends_of(fx, row)) == 2


async def test_without_a_digest_the_one_reminder_goes_a_day_after_the_ask(fx):
    row, _ = await delivered_ask(fx)
    fx.shift(hours=23)
    await fx.mind.tick(force=True)
    assert len(sends_of(fx, row)) == 1
    fx.shift(hours=1, minutes=1)
    await fx.mind.tick(force=True)
    _, reminder = sends_of(fx, row)
    assert reminder.context["ask_message"] == "reminder"
    deliver(fx, reminder)
    fx.shift(hours=30)
    await fx.mind.tick(force=True)
    assert len(sends_of(fx, row)) == 2


async def test_the_owner_sets_when_the_timed_reminder_goes(fx):
    fx.mind.config["asks"] = {"remind_hours": 6}                 # mind.asks.remind_hours
    row, _ = await delivered_ask(fx)
    fx.shift(hours=6, minutes=1)
    await fx.mind.tick(force=True)
    assert [message.context["ask_message"] for message in sends_of(fx, row)] == ["first", "reminder"]


async def test_later_moves_the_reminder_to_after_the_next_digest(fx):
    row, first = await delivered_ask(fx)
    fx.shift(hours=1)
    result = await fx.mind.read_answer({"platform": "telegram", "contact_id": OWNER, "chat_id": "owner-chat",
                                        "chat_type": "dm", "turn_id": "t-later", "message_id": "in-later",
                                        "arrived_at": fx.now.isoformat(), "text": "not now",
                                        "reply_to_message_id": first.hermes_ref, "ledger": None})
    assert [item["id"] for item in result["later"]] == [row.id]
    fx.shift(hours=25)                                          # past the day: the later answer holds it
    await fx.mind.tick(force=True)
    assert len(sends_of(fx, row)) == 1
    a_digest_delivered(fx, 1)
    await fx.mind.tick(force=True)
    _, reminder = sends_of(fx, row)
    assert reminder.context["ask_message"] == "reminder" and fx.store.get(row.id).status == "asked"


def owner_later(fx, reply_to, message_id="in-later"):
    return fx.mind.read_answer({"platform": "telegram", "contact_id": OWNER, "chat_id": "owner-chat",
                                "chat_type": "dm", "turn_id": f"t-{message_id}", "message_id": message_id,
                                "arrived_at": fx.now.isoformat(), "text": "not now",
                                "reply_to_message_id": reply_to, "ledger": None})


async def test_later_also_moves_a_reminder_already_queued_but_not_yet_sent(fx):
    """A reminder queued while delivery waited (the body offline, a pause) and a "later" that came before it went:
    it is not offered, a claim of it is refused, and the one reminder goes after the next digest instead."""
    row, first = await delivered_ask(fx)
    fx.shift(hours=24, minutes=1)
    await fx.mind.tick(force=True)
    _, reminder = sends_of(fx, row)
    assert reminder.status == "approved" and reminder.context["ask_message"] == "reminder"
    fx.shift(minutes=1)
    result = await owner_later(fx, first.hermes_ref)
    assert [item["id"] for item in result["later"]] == [row.id]
    assert reminder.id not in [item["id"] for item in await fx.mind.outbox_ready()]
    assert fx.mind.outbox.sending(reminder.id, target="telegram:owner-chat") is None
    assert fx.store.get(reminder.id).status == "cancelled"
    fx.shift(hours=25)
    await fx.mind.tick(force=True)
    assert [message.status for message in sends_of(fx, row)] == ["sent", "cancelled"]
    a_digest_delivered(fx, 1)
    await fx.mind.tick(force=True)
    *_, again = sends_of(fx, row)
    assert again.status == "approved" and again.context["ask_message"] == "reminder"
    assert fx.store.get(row.id).status == "asked"


async def test_a_claim_of_a_queued_reminder_is_refused_once_the_owner_said_later(fx):
    row, first = await delivered_ask(fx)
    fx.shift(hours=24, minutes=1)
    await fx.mind.tick(force=True)
    _, reminder = sends_of(fx, row)
    payload, = [item for item in await fx.mind.outbox_ready() if item["id"] == reminder.id]
    fx.shift(minutes=1)
    await owner_later(fx, first.hermes_ref)                     # between the body's pull and its claim
    assert fx.mind.outbox.sending(payload["id"], target="telegram:owner-chat") is None
    assert fx.store.get(reminder.id).status == "cancelled"


async def test_later_on_a_listed_ask_withdraws_its_queued_list_and_the_others_go_without_it(fx):
    """A list not yet sent that shows an ask the owner has since said "later" to does not go: the others in it are
    queued again at the next tick without that ask, which waits for the next digest."""
    fx.mind.config["asks"] = {"messages_per_day": 1}
    first_ask = asked(fx, "K7F", "Check the venue hours for event 1")
    second = asked(fx, "Q2Z", "Check the venue hours for event 2")
    await fx.mind.tick(force=True)
    single, = sends_of(fx, first_ask)
    alone, = sends_of(fx, second)
    assert alone.context["ask_message"] == "list" and alone.context["asks"] == [second.id]
    deliver(fx, single, ref="k7f-1")
    deliver(fx, alone, ref="q2z-1")
    fx.shift(hours=12)
    third = asked(fx, "M4P", "Check the venue hours for event 3")
    await fx.mind.tick(force=True)
    assert sends_of(fx, third) == []                             # the day's single and list are spent
    fx.shift(hours=12, minutes=1)
    await fx.mind.tick(force=True)
    pending, = [message for message in ask_messages(fx, status={"approved"})
                if message.context["ask_message"] == "list"]
    assert pending.context["asks"] == [second.id, third.id]      # the second's reminder and the third's first
    fx.shift(minutes=1)
    result = await owner_later(fx, "q2z-1")
    assert [item["id"] for item in result["later"]] == [second.id]
    assert pending.id not in [item["id"] for item in await fx.mind.outbox_ready()]
    assert fx.store.get(pending.id).status == "cancelled"
    fx.shift(minutes=1)
    await fx.mind.tick(force=True)
    again, = [message for message in ask_messages(fx, status={"approved"}) if message.context["ask_message"] == "list"]
    assert again.context["asks"] == [third.id]


async def test_later_on_an_ask_whose_first_message_is_still_queued_holds_it_for_the_next_digest(fx):
    """The owner said "later" to an ask they saw elsewhere (the digest's waiting line, a show) before its own
    message went: that message waits for the next digest too, instead of going at the next tick."""
    row = asked(fx, "K7F")
    await fx.mind.tick(force=True)
    queued, = sends_of(fx, row)
    fx.shift(minutes=1)
    fx.store.transition(row.id, "asked", action=outbox_module.ASK_LATER, at=fx.now, details={"by": "owner"})
    assert queued.id not in [item["id"] for item in await fx.mind.outbox_ready()]
    fx.shift(minutes=1)
    await fx.mind.tick(force=True)
    assert [message.status for message in sends_of(fx, row)] == ["cancelled"]
    a_digest_delivered(fx, 1)
    await fx.mind.tick(force=True)
    _, first = sends_of(fx, row)
    assert first.status == "approved" and first.context["ask_message"] == "first"


def test_the_outbox_reads_the_answer_routes_later_record():
    from colonymind.mind import asks_report, outbox
    assert outbox.ASK_LATER == asks_report.ASK_LATER


# -- a lapse is no verdict -----------------------------------------------------------------------------------------

async def test_an_ask_that_lapses_records_no_verdict_and_nothing_is_done(fx):
    """At the lapse time nothing is done and nothing is learned: no "ignored" on the ask, nothing in the type's
    feedback, and its reminder never goes."""
    row, _ = await delivered_ask(fx)
    before = fx.feedback.multiplier("research:curiosity")
    fx.shift(hours=73)
    summary = await fx.mind.tick(force=True)
    lapsed = fx.store.get(row.id)
    assert summary["expired_asks"] >= 1 and lapsed.status == "expired" and lapsed.outcome == "expired"
    assert lapsed.verdict is None and fx.feedback.multiplier("research:curiosity") == before
    assert len(sends_of(fx, row)) == 1 and fx.mind.dispatch() == []


@pytest.mark.parametrize("words", ["yes", "no"])
async def test_an_answer_after_the_lapse_before_the_next_tick_records_no_verdict_either(fx, words):
    """The owner answers once the ask has lapsed but before a tick closed it: nothing is done, and the silence up to
    its lapse still teaches nothing (no "ignored", nothing in the type's feedback), whichever path settles it."""
    row, first = await delivered_ask(fx)
    before = fx.feedback.multiplier("research:curiosity")
    fx.shift(hours=73)
    result = await fx.mind.read_answer({"platform": "telegram", "contact_id": OWNER, "chat_id": "owner-chat",
                                        "chat_type": "dm", "turn_id": "t-late", "message_id": "in-late",
                                        "arrived_at": fx.now.isoformat(), "text": words,
                                        "reply_to_message_id": first.hermes_ref, "ledger": None})
    lapsed = fx.store.get(row.id)
    assert [item["id"] for item in result["settled"]] == [row.id] and result["approved"] == []
    assert lapsed.status == "expired" and lapsed.outcome == "expired"
    assert lapsed.verdict is None and fx.feedback.multiplier("research:curiosity") == before
    assert fx.mind.dispatch() == []


async def test_a_console_answer_after_the_lapse_records_no_verdict(fx):
    row, _ = await delivered_ask(fx)
    before = fx.feedback.multiplier("research:curiosity")
    fx.shift(hours=73)
    await fx.mind.answer("K7F", yes=False, by="cli")
    lapsed = fx.store.get(row.id)
    assert lapsed.status == "expired" and lapsed.verdict is None
    assert fx.feedback.multiplier("research:curiosity") == before and fx.mind.dispatch() == []


# -- where the owner is ---------------------------------------------------------------------------------------------

def owner_says(fx, text, *, platform="whatsapp", chat="chat-2", message_id="w-1", reply_to=None):
    return {"platform": platform, "contact_id": OWNER, "chat_id": chat, "chat_type": "dm", "turn_id": f"t-{message_id}",
            "message_id": message_id, "arrived_at": fx.now.isoformat(), "text": text,
            "reply_to_message_id": reply_to, "ledger": None}


async def test_an_ask_goes_to_the_chat_the_owner_wrote_in_within_the_hour_else_to_his_primary_handle(fx):
    await fx.mind.read_answer(owner_says(fx, "Morning, anything for me today?"))
    fx.shift(minutes=20)
    row = asked(fx, "K7F")
    await fx.mind.tick(force=True)
    payload, = [item for item in await fx.mind.outbox_ready() if item["type"] == ASK_MESSAGE]
    assert fx.store.get(payload["id"]).context["asks"] == [row.id] and payload["target"] == "whatsapp:chat-2"
    fx.shift(hours=2)
    other = asked(fx, "Q2Z", "Check the venue hours before Friday")
    await fx.mind.tick(force=True)
    later, = [item for item in await fx.mind.outbox_ready() if item["type"] == ASK_MESSAGE
              and fx.store.get(item["id"]).context["asks"] == [other.id]]
    assert "target" not in later                                 # the body sends it to their primary handle


async def test_the_question_about_an_unclear_answer_goes_back_to_the_chat_it_came_from(fx):
    row = asked(fx, "K7F")
    await fx.mind.tick(force=True)
    message, = sends_of(fx, row)
    fx.mind.outbox.sending(message.id, target="whatsapp:chat-2")
    message = fx.mind.outbox.sent(message.id, result="sent", hermes_ref="ask-1")
    fx.shift(minutes=2)
    result = await fx.mind.read_answer(owner_says(fx, "yes?", reply_to="ask-1"))
    clarify = [item for item in await fx.mind.outbox_ready() if item["id"] == result["clarify"]["message"]]
    assert clarify and clarify[0]["target"] == "whatsapp:chat-2"


async def test_a_chat_the_body_could_not_deliver_to_is_not_tried_again(fx):
    """The owner last wrote on a platform the body cannot send to (a terminal chat, say): the ask's message there
    fails, and the ask goes to their primary handle instead of failing at every tick."""
    await fx.mind.read_answer(owner_says(fx, "Anything for me?", platform="cli", chat="terminal-1"))
    row = asked(fx, "K7F")
    await fx.mind.tick(force=True)
    payload, = [item for item in await fx.mind.outbox_ready() if item["type"] == ASK_MESSAGE]
    assert payload["target"] == "cli:terminal-1"
    fx.mind.outbox.sending(payload["id"], target=payload["target"])
    fx.mind.outbox.sent(payload["id"], result="failed", error="no such platform")
    fx.shift(minutes=1)
    await fx.mind.tick(force=True)
    again, = [item for item in await fx.mind.outbox_ready() if item["type"] == ASK_MESSAGE]
    assert again["id"] != payload["id"] and "target" not in again
    assert fx.store.get(again["id"]).context["asks"] == [row.id]
