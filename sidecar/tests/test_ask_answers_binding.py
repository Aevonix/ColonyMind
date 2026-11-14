"""Which ask the owner's message answers, and what it does: the answer route (docs/MIND.md, Asks).

The owner answers an ask the way they would answer a person, with no code: a reaction on the ask's own message, a
quoted reply to it, or their next message when that ask's own message was the last thing in their chat when it
arrived. The binding is structural (``answers.bind``: what the message replies or reacts to, what the chat showed
and when), judged at the message's arrival; only then are the words read (``answers.read``). Words that name an
open ask only put it in front of the owner again, and nothing else binds. Each message is read once
(``answer_turns``), settles through the settle paths ``Mind.answer`` uses, and never stores its text.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from colonymind.mind import answers
from colonymind.mind.digest import NEEDS_OK_1_11
from colonymind.mind.answers import Evidence, Seen, Shown
from test_mind_loop import CONTACT, OWNER, Fixture, pinned_clock  # noqa: F401  (the pinned wall clock)
from owner_answer import owner_answers

T0 = datetime(2030, 1, 7, 12, 0, tzinfo=timezone.utc)
PLATFORM = "telegram"
CHAT = "owner-chat"


def at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


# -- the binding, pure ---------------------------------------------------------------------------------------------

def test_a_reaction_binds_only_to_an_asks_own_message():
    own, listed = Shown(PLATFORM, CHAT, "m-1", at(0), ("a",)), Shown(PLATFORM, CHAT, "m-2", at(1), ("a", "b"))
    bound = answers.bind(Evidence(PLATFORM, at(60), chat=CHAT, reply_to="m-1", reaction="👍"), [own, listed])
    assert bound.asks == ("a",) and bound.path == answers.REACTION
    # On a list, on another message, on another platform or in another chat (the same id there is another
    # message), with no chat, or on a message delivered after it: nothing.
    for evidence in (Evidence(PLATFORM, at(60), chat=CHAT, reply_to="m-2", reaction="👍"),
                     Evidence(PLATFORM, at(60), chat=CHAT, reply_to="m-9", reaction="👍"),
                     Evidence("whatsapp", at(60), chat=CHAT, reply_to="m-1", reaction="👍"),
                     Evidence(PLATFORM, at(60), chat="another-chat", reply_to="m-1", reaction="👍"),
                     Evidence(PLATFORM, at(60), reply_to="m-1", reaction="👍"),
                     Evidence(PLATFORM, at(-5), chat=CHAT, reply_to="m-1", reaction="👍")):
        assert not answers.bind(evidence, [own, listed]), evidence
    # A delivery whose chat or platform the sidecar does not know names nothing.
    for unknown in (Shown(PLATFORM, "", "m-1", at(0), ("a",)), Shown("", CHAT, "m-1", at(0), ("a",))):
        assert not answers.bind(Evidence(PLATFORM, at(60), chat=CHAT, reply_to="m-1", reaction="👍"), [unknown])


def test_a_quoted_reply_binds_to_the_ask_or_the_list_it_quotes_and_beats_the_follow_up():
    own, listed = Shown(PLATFORM, CHAT, "m-1", at(0), ("a",)), Shown(PLATFORM, CHAT, "m-2", at(1), ("a", "b"))
    ledger = (Seen(at(1), ("a", "b")), Seen(at(30)))          # a model reply came last
    bound = answers.bind(Evidence(PLATFORM, at(60), chat=CHAT, reply_to="m-1", ledger=ledger), [own, listed])
    assert bound.asks == ("a",) and bound.path == answers.REPLY
    bound = answers.bind(Evidence(PLATFORM, at(60), chat=CHAT, reply_to="m-2", ledger=ledger), [own, listed])
    assert bound.asks == ("a", "b") and bound.path == answers.LIST and bound.listed
    # A list's numbers mean that list for a day only.
    late = Evidence(PLATFORM, at(1 + 25 * 3600), chat=CHAT, reply_to="m-2")
    assert not answers.bind(late, [own, listed])


def test_a_quoted_reply_to_anything_but_an_open_asks_message_binds_nothing():
    """The owner quoted something else (the model's earlier message, a closed ask's): that is what they answered,
    even while an ask was the last thing the chat showed. An explicit target that shows no open ask binds nothing."""
    shown = [Shown(PLATFORM, CHAT, "m-1", at(0), ("a",))]
    ledger = (Seen(at(0), ("a",)),)
    assert answers.bind(Evidence(PLATFORM, at(60), ledger=ledger), shown).path == answers.FOLLOW_UP
    for target in ("model-7", "closed-ask-message"):
        assert not answers.bind(Evidence(PLATFORM, at(60), chat=CHAT, reply_to=target, ledger=ledger), shown, weighty={"a"})
        assert not answers.bind(Evidence(PLATFORM, at(60), chat=CHAT, reply_to=target, ledger=ledger), shown)


def test_the_follow_up_binds_only_when_the_ask_was_the_last_thing_the_chat_showed_at_arrival():
    shown = [Shown(PLATFORM, CHAT, "m-1", at(0), ("a",))]
    bound = answers.bind(Evidence(PLATFORM, at(60), ledger=(Seen(at(-600)), Seen(at(0), ("a",)))), shown)
    assert bound.asks == ("a",) and bound.path == answers.FOLLOW_UP
    cases = {
        "a model reply after the ask": (Seen(at(0), ("a",)), Seen(at(20))),
        "an ask delivered after the message arrived": (Seen(at(-600)), Seen(at(61), ("a",))),
        "an ask delivered under 3 s before it arrived": (Seen(at(-600)), Seen(at(58), ("a",))),
        "two sends within 5 s in no fixed order": (Seen(at(0), ("a",)), Seen(at(-3))),
        "nothing in the ledger": (),
    }
    for why, ledger in cases.items():
        assert not answers.bind(Evidence(PLATFORM, at(60), ledger=ledger), shown), why
    # Two sends the plugin ordered itself bind; with no ledger at all (the ingest read) the follow-up never does.
    assert answers.bind(Evidence(PLATFORM, at(60), ledger=(Seen(at(-3), ordered=True),
                                                          Seen(at(0), ("a",), ordered=True))), shown)
    assert not answers.bind(Evidence(PLATFORM, at(60), ledger=None), shown)


def test_a_weighty_ask_binds_by_follow_up_only_within_twelve_hours():
    ledger = (Seen(at(0), ("a",)),)
    within = Evidence(PLATFORM, at(11 * 3600), ledger=ledger)
    beyond = Evidence(PLATFORM, at(13 * 3600), ledger=ledger)
    assert answers.bind(within, [], weighty={"a"}) and answers.bind(beyond, [])
    assert not answers.bind(beyond, [], weighty={"a"})


def test_the_plan_for_one_ask_and_for_a_list():
    one = answers.Binding(("a",), answers.FOLLOW_UP)
    yes, no = answers.read("yes"), answers.read("no")
    assert answers.plan(yes, one, open_asks={"a"}).yes == ("a",)
    assert answers.plan(no, one, open_asks={"a"}).no == ("a",)
    assert answers.plan(answers.read("later"), one, open_asks={"a"}).later == ("a",)
    assert answers.plan(answers.read("yes but only after lunch"), one, open_asks={"a"}).clarify == ("a",)
    assert not answers.plan(answers.read("thanks for the update"), one, open_asks={"a"})
    assert not answers.plan(yes, one, open_asks=set())                        # no longer open
    listed = answers.Binding(("a", "b", "c"), answers.LIST)

    def on_list(text, weighty=()):
        return answers.plan(answers.read(text, list_size=3), listed, open_asks={"a", "b", "c"}, weighty=weighty)
    assert on_list("2").yes == ("b",) and on_list("no 1").no == ("a",) and on_list("1 and 3").yes == ("a", "c")
    assert on_list("all").yes == ("a", "b", "c") and on_list("none").no == ("a", "b", "c")
    assert on_list("yes").which == ("a", "b", "c") and on_list("no").which == ("a", "b", "c")
    # A weighty ask never takes a list's yes: it is shown instead; a list's no still declines it.
    assert on_list("all", weighty={"b"}).yes == ("a", "c") and on_list("all", weighty={"b"}).show == ("b",)
    assert on_list("2", weighty={"b"}).show == ("b",) and on_list("no 2", weighty={"b"}).no == ("b",)


def test_named_words_find_the_asks_they_name_and_nothing_else():
    names = {"a": ["Sam Rivera", "Sam"], "b": ["old-builds"]}
    assert answers.named("yes, text Sam the recap", names) == ("a",)
    assert answers.named("sure, clean up old-builds", names) == ("b",)
    assert answers.named("yes K7F", names, codes={"K7F": "a"}) == ("a",)
    assert answers.named("yes to Samantha", names) == ()
    assert answers.named("Sam said yes", {"b": ["old-builds"]}) == ()
    assert set(answers.named("Sam and old-builds, go", names)) == {"a", "b"}


@pytest.mark.parametrize("text", ["undo", "Undo that", "cancel that", "wait, no", "I didn't mean that",
                                  "undo please", "No, undo it!"])
def test_undo_words(text):
    assert answers.is_undo(text)


@pytest.mark.parametrize("text", ["undo the last edit to the file", "can you undo the rename?", "yes", "no",
                                  "cancel the meeting", "wait",
                                  # someone else's words the owner passes on, and a question, are not their undo
                                  "> undo", "> cancel that", "Sam said: cancel that", "undo?",
                                  "cancel that?", "wait, no?"])
def test_words_that_are_not_an_undo(text):
    assert not answers.is_undo(text)


# -- the answer route on a mind ------------------------------------------------------------------------------------

@pytest.fixture
def fx(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fixture = Fixture(tmp_path)
    yield fixture
    fixture.store.close()


def asked(fx, code="K7F", *, type_="research", kind="task", cls="self", context=None, title=None):
    row, _ = fx.store.create_intention(
        kind=kind, type=type_, title=title or "Re-run the archive export for the board figures", drive="curiosity", cls=cls,
        decision="ask", decision_reason="suggest: self -> ask", status="asked", dedup_key=f"test:{code}",
        ask_code=code, context={"notice": True, **(context or {})}, hermes_kind="none", created_at=fx.now,
        expires_at=fx.now + timedelta(days=3))
    return row


def deliver(fx, *rows, message_id, type_="ask_show"):
    """One message showing ``rows`` delivered to the owner's chat now, as the body reports it."""
    ids = [row.id for row in rows]
    notice = fx.mind.outbox.notice(type=type_, title="ask", text="the ask in words", dedup_key=f"test:{message_id}",
                                   extra={"asks": ids, "list": ids})
    fx.mind.outbox.sending(notice.id, target=f"{PLATFORM}:{CHAT}")
    return fx.mind.outbox.sent(notice.id, result="sent", hermes_ref=message_id)


def evidence(fx, text="", *, message_id, ledger=(), **extra):
    return {"platform": PLATFORM, "contact_id": OWNER, "chat_id": CHAT, "chat_type": "dm", "turn_id": f"t-{message_id}",
            "message_id": message_id, "arrived_at": fx.now.isoformat(), "text": text,
            "ledger": None if ledger is None else list(ledger), **extra}


def sent(notice):
    return {"at": notice.completed_at.isoformat(), "outbox_id": notice.id}


def reply(when):
    return {"at": when.isoformat(), "kind": "reply"}


def history(fx, row, action):
    return [item for item in fx.store.get_history(row.id, limit=50) if item.action == action]


async def test_a_plain_yes_right_after_the_asks_own_message_approves_it_with_no_code(fx):
    row = asked(fx)
    notice = deliver(fx, row, message_id="m-1")
    fx.shift(minutes=2)
    result = await fx.mind.read_answer(evidence(fx, "sure, go ahead", message_id="in-1", ledger=[sent(notice)]))
    assert result["read"] is True and result["path"] == "follow_up"
    assert [(item["id"], item["status"]) for item in result["settled"]] == [(row.id, "approved")]
    assert fx.store.get(row.id).status == "approved"
    settled, = history(fx, row, "queued")
    assert settled.details["answer"] == {"path": "follow_up", "reader": "tables", "message": f"{PLATFORM}:{CHAT}:in-1",
                                         "by": "owner", "channel": PLATFORM}
    assert "approved" in result["context"] and "do not ask again" in result["context"]
    # The text is never stored: not in the settling row, not in the answer's record.
    assert "go ahead" not in repr(settled.details) and "go ahead" not in repr(fx.mind.answer_turns.find(result["key"]))


async def test_a_plain_no_declines_and_the_settling_row_names_the_path(fx):
    row = asked(fx)
    notice = deliver(fx, row, message_id="m-1")
    fx.shift(minutes=1)
    result = await fx.mind.read_answer(evidence(fx, "no thanks", message_id="in-1", ledger=[sent(notice)]))
    assert result["settled"][0]["status"] == "cancelled" and fx.store.get(row.id).outcome == "denied"
    denied, = history(fx, row, "outcome_denied")
    assert denied.details["answer"]["path"] == "follow_up" and denied.details["by"].startswith("owner-turn:")


async def test_a_reaction_on_the_asks_own_message_answers_it(fx):
    first, second, third = asked(fx, "K7F"), asked(fx, "Q2Z"), asked(fx, "M4P")
    for row, message in ((first, "m-1"), (second, "m-2"), (third, "m-3")):
        deliver(fx, row, message_id=message)
    fx.shift(minutes=5)
    assert (await fx.mind.read_answer(evidence(fx, reaction="👍", reply_to_message_id="m-1", message_id="r-1",
                                                ledger=None)))["path"] == "reaction"
    await fx.mind.read_answer(evidence(fx, reaction="👎", reply_to_message_id="m-2", message_id="r-2", ledger=None))
    heart = await fx.mind.read_answer(evidence(fx, reaction="❤️", reply_to_message_id="m-3", message_id="r-3",
                                               ledger=None))
    assert fx.store.get(first.id).status == "approved" and fx.store.get(second.id).outcome == "denied"
    assert fx.store.get(third.id).status == "asked" and heart["settled"] == []


async def test_a_quoted_reply_to_the_ask_binds_even_after_a_model_reply(fx):
    row = asked(fx)
    notice = deliver(fx, row, message_id="m-1")
    fx.shift(minutes=3)
    ledger = [sent(notice), reply(fx.now - timedelta(minutes=1))]
    result = await fx.mind.read_answer(evidence(fx, "go ahead", message_id="in-1", reply_to_message_id="m-1",
                                                ledger=ledger))
    assert result["path"] == "reply" and fx.store.get(row.id).status == "approved"


async def test_a_yes_quoting_another_message_never_approves_the_newest_ask(fx):
    """The owner quotes the model's earlier message, or an ask that is closed, and says yes: the newest ask was the
    last thing the chat showed, but the quote says what the yes answers. Nothing is approved, a floor ask included."""
    closed = asked(fx, "Q2Z")
    deliver(fx, closed, message_id="m-0")
    await fx.mind.answer("Q2Z", yes=False, by="cli")
    floor = asked(fx, "K7F", type_="floor_effect", cls="floor")
    own = deliver(fx, floor, message_id="m-1")
    fx.shift(minutes=2)
    for number, target in enumerate(("model-7", "m-0")):
        result = await fx.mind.read_answer(evidence(fx, "yes", message_id=f"in-{number}", reply_to_message_id=target,
                                                    ledger=[sent(own)]))
        assert result["settled"] == [] and result["path"] is None, target
    assert fx.store.get(floor.id).status == "asked"


async def test_a_bare_yes_after_a_model_reply_is_conversation_and_counted_unbound(fx):
    row = asked(fx)
    notice = deliver(fx, row, message_id="m-1")
    fx.shift(minutes=3)
    result = await fx.mind.read_answer(evidence(fx, "yes", message_id="in-1",
                                                ledger=[sent(notice), reply(fx.now - timedelta(minutes=1))]))
    assert result["read"] is True and result["settled"] == [] and result["unbound"] is True
    assert fx.store.get(row.id).status == "asked"
    assert "not read as an answer" in result["context"] and row.description in result["context"]
    assert [event["kind"] for event in fx.mind.answer_events(fx.now - timedelta(hours=1))] == ["unbound_answer"]


async def test_words_that_name_an_open_ask_only_show_it(fx):
    row = asked(fx, title="Clean up the old-builds folder", type_="floor_effect", cls="floor",
                context={"effect": {"cls": "irreversible_deletion", "target": "old-builds", "text": "delete old-builds"}})
    notice = deliver(fx, row, message_id="m-1")
    fx.shift(minutes=3)
    result = await fx.mind.read_answer(evidence(fx, "yes, clean up old-builds", message_id="in-1",
                                                ledger=[sent(notice), reply(fx.now - timedelta(minutes=1))]))
    assert result["settled"] == [] and [item["id"] for item in result["shown"]] == [row.id]
    assert fx.store.get(row.id).status == "asked"
    show = fx.store.get(result["shown"][0]["message"])
    assert show.type == "ask_show" and show.context["asks"] == [row.id] and show.status == "approved"
    assert "yes or no" in show.context["text"].lower() and "K7F" not in show.context["text"]
    assert len(history(fx, row, "ask_shown")) == 1
    # The show is the ask's own message now: a yes right after it binds.
    fx.mind.outbox.sending(show.id, target=f"{PLATFORM}:{CHAT}")
    fx.mind.outbox.sent(show.id, result="sent", hermes_ref="m-2")
    fx.shift(minutes=1)
    done = await fx.mind.read_answer(evidence(fx, "yes", message_id="in-2", ledger=[sent(fx.store.get(show.id))]))
    assert done["settled"][0]["id"] == row.id and fx.store.get(row.id).status == "approved"


async def test_an_unclear_answer_gets_one_plain_question_and_never_a_second(fx):
    row = asked(fx)
    notice = deliver(fx, row, message_id="m-1")
    fx.shift(minutes=1)
    first = await fx.mind.read_answer(evidence(fx, "yes but only after lunch", message_id="in-1",
                                               ledger=[sent(notice)]))
    assert first["settled"] == [] and first["clarify"]["asks"] == [row.id]
    question = fx.store.get(first["clarify"]["message"])
    assert question.type == "ask_clarify" and question.context["text"].startswith("Just to check: ")
    assert question.context["text"].endswith("? Yes or no.") and question.context["asks"] == [row.id]
    fx.mind.outbox.sending(question.id, target=f"{PLATFORM}:{CHAT}")
    fx.mind.outbox.sent(question.id, result="sent", hermes_ref="m-2")
    fx.shift(minutes=1)
    second = await fx.mind.read_answer(evidence(fx, "yes, but not today", message_id="in-2",
                                                ledger=[sent(fx.store.get(question.id))]))
    assert second["clarify"] is None and fx.store.get(row.id).status == "asked"
    assert len(history(fx, row, "ask_clarified")) == 1
    fx.shift(minutes=1)
    third = await fx.mind.read_answer(evidence(fx, "yes", message_id="in-3",
                                               ledger=[sent(fx.store.get(question.id))]))
    assert third["settled"][0]["status"] == "approved"


async def test_later_keeps_the_ask_open_with_no_verdict(fx):
    row = asked(fx)
    notice = deliver(fx, row, message_id="m-1")
    fx.shift(minutes=1)
    result = await fx.mind.read_answer(evidence(fx, "not now", message_id="in-1", ledger=[sent(notice)]))
    current = fx.store.get(row.id)
    assert result["later"][0]["id"] == row.id and current.status == "asked" and current.verdict is None
    assert len(history(fx, row, "ask_later")) == 1 and "stays open" in result["context"]


async def test_a_list_takes_numbers_all_and_none_and_asks_which_one_for_a_bare_yes(fx):
    a, b = asked(fx, "K7F"), asked(fx, "Q2Z")
    listed = deliver(fx, a, b, message_id="m-1")
    fx.shift(minutes=1)
    which = await fx.mind.read_answer(evidence(fx, "yes", message_id="in-1", ledger=[sent(listed)]))
    assert which["settled"] == [] and which["clarify"]["asks"] == [a.id, b.id]
    text = fx.store.get(which["clarify"]["message"]).context["text"]
    assert text.startswith("Which one? 1. ") and " 2. " in text and text.endswith("Reply 1, 2, both or neither.")
    fx.shift(minutes=1)
    picked = await fx.mind.read_answer(evidence(fx, "2", message_id="in-2", reply_to_message_id="m-1", ledger=None))
    assert picked["path"] == "list" and [item["id"] for item in picked["settled"]] == [b.id]
    assert fx.store.get(a.id).status == "asked" and fx.store.get(b.id).status == "approved"
    fx.shift(minutes=1)
    await fx.mind.read_answer(evidence(fx, "none", message_id="in-3", reply_to_message_id="m-1", ledger=None))
    assert fx.store.get(a.id).outcome == "denied"


async def test_a_1_11_notice_is_a_list_of_the_asks_it_names(fx):
    a, b = asked(fx, "K7F"), asked(fx, "Q2Z")
    listed = "\n".join([f"{NEEDS_OK_1_11}:", *(f"- {row.ask_code}: {row.description}." for row in (a, b)),
                        "Without an answer they expire and nothing is done."])
    notice = fx.mind.outbox.notice(type="ask_notice", title="2 open ask(s)", text=listed, dedup_key="notice:1.11")
    fx.mind.outbox.sending(notice.id, target=f"{PLATFORM}:{CHAT}")
    fx.mind.outbox.sent(notice.id, result="sent", hermes_ref="m-1")
    fx.shift(minutes=1)
    result = await fx.mind.read_answer(evidence(fx, "both", message_id="in-1", ledger=[sent(fx.store.get(notice.id))]))
    assert result["path"] == "list" and {item["id"] for item in result["settled"]} == {a.id, b.id}


async def test_a_weighty_ask_reads_strict_and_never_takes_a_lists_yes(fx):
    floor = asked(fx, "K7F", type_="floor_effect", cls="floor")
    ordinary = asked(fx, "Q2Z")
    own = deliver(fx, floor, message_id="m-1")
    fx.shift(minutes=1)
    weak = await fx.mind.read_answer(evidence(fx, "k", message_id="in-1", ledger=[sent(own)]))
    assert weak["settled"] == [] and weak["clarify"]["asks"] == [floor.id]
    listed = deliver(fx, floor, ordinary, message_id="m-2")
    fx.shift(minutes=1)
    both = await fx.mind.read_answer(evidence(fx, "both", message_id="in-2", ledger=[sent(listed)]))
    assert [item["id"] for item in both["settled"]] == [ordinary.id] and [item["id"] for item in both["shown"]] == [floor.id]
    assert fx.store.get(floor.id).status == "asked"


async def test_a_yes_still_waits_for_the_full_text_of_a_long_ask(fx):
    row = asked(fx, context={"full_view": {"seen": False, "notice": None, "attempts": 1}})
    own = deliver(fx, row, message_id="m-1")
    fx.shift(minutes=1)
    result = await fx.mind.read_answer(evidence(fx, "yes", message_id="in-1", ledger=[sent(own)]))
    assert result["settled"] == [] and fx.store.get(row.id).status == "asked"
    assert "full" in result["refused"][0]["reason"]


async def test_a_yes_written_before_the_full_text_arrived_does_not_count_when_it_is_read_later(fx):
    """The owner answers the short message; the full text arrives after they wrote; their answer is read only
    then. It still answered the short message, so it approves nothing. A yes written after the full text does."""
    row = asked(fx, context={"full_view": {"seen": False, "notice": None, "attempts": 0}})
    own = deliver(fx, row, message_id="m-1")
    fx.shift(minutes=1)
    written = fx.now
    fx.shift(seconds=30)
    fx.mind.outbox.send_full_view(row, "Re-run the archive export.", "What a yes allows, in full. " * 40)
    full = fx.store.get(fx.store.get(row.id).context["full_view"]["notice"])
    fx.mind.outbox.sending(full.id, target=f"{PLATFORM}:{CHAT}")
    full = fx.mind.outbox.sent(full.id, result="sent", hermes_ref="m-2")
    fx.shift(seconds=30)
    early = await fx.mind.read_answer({**evidence(fx, "yes", message_id="in-1", ledger=[sent(own)]),
                                       "arrived_at": written.isoformat()})
    assert early["settled"] == [] and "full" in early["refused"][0]["reason"]
    assert fx.store.get(row.id).status == "asked"
    fx.shift(minutes=1)
    after = await fx.mind.read_answer(evidence(fx, "yes", message_id="in-2", ledger=[sent(own), sent(full)]))
    assert after["settled"][0]["status"] == "approved" and fx.store.get(row.id).status == "approved"


async def test_a_message_is_read_once_whatever_route_brings_it(fx):
    row = asked(fx, "K7F")
    other = asked(fx, "Q2Z")
    own = deliver(fx, row, message_id="m-1")
    fx.shift(minutes=1)
    first = await fx.mind.read_answer(evidence(fx, "yes", message_id="in-1", ledger=[sent(own)]))
    assert first["read"] is True
    again = await fx.mind.read_answer(evidence(fx, "yes", message_id="in-1", ledger=[sent(own)]))
    assert again["read"] is False and "once" in again["reason"]
    # The same turn arriving at ingest (by its turn id, with no message id) is read already too.
    ingest = await fx.mind.read_answer({**evidence(fx, "no", message_id="", ledger=None), "turn_id": "t-in-1"})
    assert ingest["read"] is False and fx.store.get(row.id).status == "approved"
    assert fx.store.get(other.id).status == "asked"


async def test_without_the_ledger_only_explicit_references_bind(fx):
    row = asked(fx)
    deliver(fx, row, message_id="m-1")
    fx.shift(minutes=1)
    bare = await fx.mind.read_answer(evidence(fx, "yes", message_id="in-1", ledger=None))
    assert bare["settled"] == [] and fx.store.get(row.id).status == "asked"
    quoted = await fx.mind.read_answer(evidence(fx, "yes", message_id="in-2", reply_to_message_id="m-1", ledger=None))
    assert quoted["settled"][0]["id"] == row.id


async def test_undo_inside_ten_minutes_cancels_work_that_has_not_started(fx):
    row = asked(fx)
    own = deliver(fx, row, message_id="m-1")
    fx.shift(minutes=1)
    await fx.mind.read_answer(evidence(fx, "yes", message_id="in-1", ledger=[sent(own)]))
    assert fx.store.get(row.id).status == "approved"
    fx.shift(minutes=4)
    result = await fx.mind.read_answer(evidence(fx, "undo that", message_id="in-2",
                                                ledger=[sent(own), reply(fx.now - timedelta(minutes=3))]))
    assert result["undone"][0]["id"] == row.id and result["undone"][0]["undone"] == "cancelled"
    current = fx.store.get(row.id)
    assert current.status == "cancelled" and current.verdict != "dismissed" and len(history(fx, row, "ask_undone")) == 1
    assert "undid" in result["context"]
    assert fx.mind.dispatch() == []


@pytest.mark.parametrize("by", ["words", "reaction"])
async def test_an_undo_read_late_takes_back_only_what_was_approved_before_it_arrived(fx, by):
    """The owner's undo arrives, is read three minutes later, and meanwhile they approved an ask: the undo was about
    something before it, never the approval that came after, whichever the order of reading."""
    row = asked(fx)
    own = deliver(fx, row, message_id="m-1")
    fx.shift(minutes=1)
    written = fx.now
    fx.shift(minutes=1)
    if by == "words":
        await fx.mind.read_answer(evidence(fx, "yes", message_id="in-1", ledger=[sent(own)]))
        undo = evidence(fx, "undo", message_id="in-2", ledger=[])
    else:
        await fx.mind.read_answer(evidence(fx, reaction="👍", reply_to_message_id="m-1", message_id="r-1",
                                           ledger=None))
        undo = evidence(fx, reaction="👍", reaction_removed=True, reply_to_message_id="m-1", message_id="r-2",
                        ledger=None)
    assert fx.store.get(row.id).status == "approved"
    fx.shift(minutes=1)
    result = await fx.mind.read_answer({**undo, "arrived_at": written.isoformat()})
    assert result["undone"] == [] and fx.store.get(row.id).status == "approved"


@pytest.mark.parametrize("text", ["> undo", "> cancel that", "undo?"])
async def test_a_quoted_or_asked_undo_leaves_the_approval(fx, text):
    row = asked(fx)
    own = deliver(fx, row, message_id="m-1")
    fx.shift(minutes=1)
    await fx.mind.read_answer(evidence(fx, "yes", message_id="in-1", ledger=[sent(own)]))
    fx.shift(minutes=1)
    result = await fx.mind.read_answer(evidence(fx, text, message_id="in-2", ledger=[]))
    assert result["undone"] == [] and fx.store.get(row.id).status == "approved"


async def test_undo_withdraws_a_secret_permit_not_used_yet(fx, monkeypatch):
    """A yes to a sealed value's use is a permit for that one call, waiting for the tool's retry. Undone before
    the retry, the permit is withdrawn: the retry gets no value, and asks again."""
    from fastapi import FastAPI
    from colonymind.api.routers import secrets as secrets_router
    from colonymind.secrets.vault import Vault
    vault = Vault(fx.state)
    monkeypatch.setattr(secrets_router, "_vault", vault)
    monkeypatch.setattr(secrets_router, "_owner_id", lambda: OWNER)
    monkeypatch.setattr(secrets_router, "_mind", lambda: fx.mind)
    app = FastAPI()
    app.include_router(secrets_router.router)
    value = "sk-" + "synthetic" + "0123456789abcdefghijklmnop"          # secret-scan: synthetic
    ref = vault.seal(value, contact_id=OWNER, kind="prefix", label="API key")
    action = {"tool": "send_message", "args": {"target": "telegram:p-02-handle", "message": f"[secret {ref}: API key]"}}

    async def retry():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/v1/secrets/resolve", json={
                "refs": [ref], "purpose": "fill", "lane": "owner", "contact_id": OWNER, "action": action,
                "destination": "telegram:p-02-handle", "tool": "send_message"})
        assert response.status_code == 200, response.text
        return response.json()
    permit = fx.store.get_by_ask_code((await retry())["asks"][ref]["code"])
    own = deliver(fx, permit, message_id="m-1")
    fx.shift(minutes=1)
    approved = await fx.mind.read_answer(evidence(fx, "yes", message_id="in-1", ledger=[sent(own)]))
    assert approved["settled"][0]["status"] == "approved"
    fx.shift(minutes=1)
    undone = await fx.mind.read_answer(evidence(fx, "undo", message_id="in-2", ledger=[]))
    assert undone["undone"][0]["undone"] == "cancelled" and fx.store.get(permit.id).status == "cancelled"
    after = await retry()
    assert after["values"] == {} and value not in repr(after) and after["asks"][ref]["code"] != permit.ask_code


async def test_undo_after_ten_minutes_or_in_another_chat_does_nothing(fx):
    row = asked(fx)
    own = deliver(fx, row, message_id="m-1")
    fx.shift(minutes=1)
    await fx.mind.read_answer(evidence(fx, "yes", message_id="in-1", ledger=[sent(own)]))
    elsewhere = await fx.mind.read_answer({**evidence(fx, "undo", message_id="in-2", ledger=[]),
                                           "chat_id": "another-chat"})
    assert elsewhere["undone"] == [] and fx.store.get(row.id).status == "approved"
    fx.shift(minutes=11)
    late = await fx.mind.read_answer(evidence(fx, "undo", message_id="in-3", ledger=[]))
    assert late["undone"] == [] and fx.store.get(row.id).status == "approved"


async def test_a_removed_reaction_inside_the_window_undoes_and_started_work_is_said_plainly(fx):
    first, second = asked(fx, "K7F"), asked(fx, "Q2Z")
    deliver(fx, first, message_id="m-1")
    deliver(fx, second, message_id="m-2")
    fx.shift(minutes=1)
    for row, message in ((first, "m-1"), (second, "m-2")):
        await fx.mind.read_answer(evidence(fx, reaction="👍", reply_to_message_id=message, message_id=f"r-{message}",
                                           ledger=None))
    fx.mind.bound(second.id, "kanban:t-1")                           # the second one already started
    fx.shift(minutes=2)
    removed = await fx.mind.read_answer(evidence(fx, reaction="👍", reaction_removed=True, reply_to_message_id="m-1",
                                                 message_id="r-3", ledger=None))
    assert removed["undone"][0]["undone"] == "cancelled" and fx.store.get(first.id).status == "cancelled"
    changed = await fx.mind.read_answer(evidence(fx, reaction="👎", reply_to_message_id="m-2", message_id="r-4",
                                                 ledger=None))
    assert changed["undone"][0]["undone"] == "too_late" and "already" in changed["undone"][0]["what"]
    assert fx.store.get(second.id).status == "dispatched"


SECOND = {"chat_id": "second-dm"}
ELSEWHERE = {
    "a quoted reply from another chat": {**SECOND, "reply_to_message_id": "m-1"},
    "a reaction from another chat": {**SECOND, "reaction": "👍", "reply_to_message_id": "m-1", "text": ""},
    "a reaction from another chat on a show the plugin recorded": {**SECOND, "reaction": "👍",
                                                                  "reply_to_message_id": "m-7", "text": ""},
    "a quoted reply on another platform": {"platform": "whatsapp", "reply_to_message_id": "m-1"},
    "the ledger names the id from another chat": {**SECOND, "ledger": "m-1"},
    "the ledger names a send to another chat": {"ledger": "other"},
}


@pytest.mark.parametrize("why", list(ELSEWHERE))
async def test_a_delivery_id_binds_only_in_the_chat_and_on_the_platform_it_was_delivered_in(fx, why):
    """Message ids are unique only within a chat on some platforms: the owner's message in another chat (another
    direct chat of theirs, or the same id on another platform) that carries the same id binds nothing, whether it
    replies, reacts, or names it in its ledger. An outbox send the ledger names counts only in the chat it went to."""
    row = asked(fx)
    own = deliver(fx, row, message_id="m-1")                         # to telegram:owner-chat
    other = fx.mind.outbox.notice(type="ask_show", title="ask", text="the ask", dedup_key="test:other",
                                  extra={"asks": [row.id], "list": [row.id]})
    fx.mind.outbox.sending(other.id, target="sms:owner-number")
    other = fx.mind.outbox.sent(other.id, result="sent", hermes_ref="s-1")
    fx.mind.record_shown([row.id], platform=PLATFORM, chat=CHAT, at=fx.now, message_id="m-7")
    fx.shift(minutes=2)
    change = dict(ELSEWHERE[why])
    ledger = {"m-1": [{"at": own.completed_at.isoformat(), "message_id": "m-1"}], "other": [sent(other)]}
    change["ledger"] = ledger.get(change.get("ledger"), [])
    result = await fx.mind.read_answer({**evidence(fx, "yes", message_id="in-1"), **change})
    assert result["read"] is True and result["settled"] == [] and result["path"] is None, why
    assert fx.store.get(row.id).status == "asked"
    # In its own chat the same reply binds.
    done = await fx.mind.read_answer(evidence(fx, "yes", message_id="in-2", reply_to_message_id="m-1", ledger=[]))
    assert done["settled"][0]["id"] == row.id


async def test_a_reaction_removed_in_another_chat_undoes_nothing(fx):
    row = asked(fx)
    deliver(fx, row, message_id="m-1")
    fx.shift(minutes=1)
    await fx.mind.read_answer(evidence(fx, reaction="👍", reply_to_message_id="m-1", message_id="r-1", ledger=None))
    assert fx.store.get(row.id).status == "approved"
    fx.shift(minutes=1)
    removed = await fx.mind.read_answer({**evidence(fx, reaction="👍", reaction_removed=True, reply_to_message_id="m-1",
                                                    message_id="r-2", ledger=None), "chat_id": "second-dm"})
    assert removed["undone"] == [] and fx.store.get(row.id).status == "approved"


async def test_the_gate_reads_only_the_owner_in_their_direct_chat(fx):
    row = asked(fx)
    own = deliver(fx, row, message_id="m-1")
    fx.shift(minutes=1)
    base = evidence(fx, "yes", message_id="in-x", ledger=[sent(own)])
    refused = {
        "a guest": {"contact_id": CONTACT},
        "a group": {"chat_type": "group"},
        "the agent's own account": {"own_account": True},
        "a forwarded message": {"forwarded": True},
        "an integration's identity": {"contact_id": "credential:importer"},
    }
    for number, (why, change) in enumerate(refused.items()):
        result = await fx.mind.read_answer({**base, **change, "message_id": f"in-{number}"})
        assert result["read"] is False, why
    not_granted = await fx.mind.read_answer({**base, "message_id": "in-9"}, granted=False)
    assert not_granted["read"] is False
    assert fx.store.get(row.id).status == "asked"


async def test_the_console_takes_only_a_no_and_every_answer_records_its_path(fx):
    """The earlier chat path (the ask's code in the owner's words, ``Mind.answer`` with a message) is gone: in process
    the console may only say no, recorded on the console path; the owner's yes is their own reply, recorded as one."""
    row = asked(fx)
    with pytest.raises(PermissionError):
        await fx.mind.answer("K7F", yes=True, by="owner")
    assert fx.store.get(row.id).status == "asked"
    approved = await owner_answers(fx.mind, "K7F")
    assert approved.status == "approved"
    settled, = history(fx, row, "queued")
    assert settled.details["answer"]["path"] == answers.REPLY and settled.details["by"].startswith("owner-turn:")
    console = asked(fx, "Q2Z")
    await fx.mind.answer("Q2Z", yes=False, by="cli")
    denied, = history(fx, console, "outcome_denied")
    assert denied.details["answer"]["path"] == "console"


async def test_a_show_recorded_by_the_plugin_is_the_asks_own_message(fx):
    row = asked(fx)
    fx.mind.record_shown([row.id], platform=PLATFORM, chat=CHAT, at=fx.now, message_id="m-7")
    assert len(history(fx, row, "ask_shown")) == 1
    fx.shift(minutes=1)
    result = await fx.mind.read_answer(evidence(fx, reaction="👍", reply_to_message_id="m-7", message_id="r-1",
                                                ledger=None))
    assert result["settled"][0]["id"] == row.id


# -- the owner's turn at ingest ----------------------------------------------------------------------------------

async def test_a_turn_read_as_an_answer_is_skipped_by_the_question_and_outreach_readers(fx):
    row = asked(fx)
    own = deliver(fx, row, message_id="m-1")
    fx.shift(minutes=1)
    await fx.mind.read_answer(evidence(fx, "yes, and stop checking in with me", message_id="in-1",
                                       ledger=[sent(own)]))
    summary = await fx.mind.owner_turn("yes, and stop checking in with me", turn_id="t-in-1",
                                       occurred_at=fx.now.isoformat(), session_id="chat-1")
    assert summary["answer"] is True and summary["linked"] is None
    assert fx.mind.state()["outreach"]["paused_until"] is not None     # the explicit stop still applies
    # A turn nobody read as an answer is conversation.
    other = await fx.mind.owner_turn("thanks", turn_id="t-other", occurred_at=fx.now.isoformat(), session_id="chat-1")
    assert not other.get("answer")


async def test_the_ingest_read_takes_explicit_references_only(fx):
    row = asked(fx, "K7F")
    other = asked(fx, "Q2Z")
    deliver(fx, row, message_id="m-1")
    own = deliver(fx, other, message_id="m-2")
    fx.shift(minutes=1)
    await fx.mind.owner_turn("yes", turn_id="t-1", occurred_at=fx.now.isoformat(), session_id="chat-1",
                             evidence={"platform": PLATFORM, "chat_id": CHAT, "chat_type": "dm",
                                       "message_id": "in-1", "reply_to_message_id": "m-1"})
    assert fx.store.get(row.id).status == "approved"
    # No reference: the follow-up never binds without the plugin's ledger, even right after the ask.
    await fx.mind.owner_turn("yes", turn_id="t-2", occurred_at=fx.now.isoformat(), session_id="chat-1",
                             evidence={"platform": PLATFORM, "chat_id": CHAT, "chat_type": "dm", "message_id": "in-2",
                                       "ledger": [sent(own)]})
    assert fx.store.get(other.id).status == "asked"


async def test_an_answer_settled_with_no_model_turn_is_acknowledged_in_one_line(fx):
    """A tap has no model turn to acknowledge it, and a message read at ingest was replied to before it was read: the
    owner gets one line saying what their answer did. An answer the model reads first is acknowledged by the model."""
    tapped, typed, ingested = asked(fx, "K7F"), asked(fx, "Q2Z"), asked(fx, "M4P")
    deliver(fx, tapped, message_id="m-1")
    own = deliver(fx, typed, message_id="m-2")
    deliver(fx, ingested, message_id="m-3")
    fx.shift(minutes=1)
    await fx.mind.read_answer(evidence(fx, reaction="👍", reply_to_message_id="m-1", message_id="r-1", ledger=None))
    await fx.mind.read_answer(evidence(fx, "yes", message_id="in-2", ledger=[sent(own)]))
    await fx.mind.owner_turn("no", turn_id="t-3", occurred_at=fx.now.isoformat(), session_id="chat-1",
                             evidence={"platform": PLATFORM, "chat_id": CHAT, "chat_type": "dm", "message_id": "in-3",
                                       "reply_to_message_id": "m-3"})
    acks = [row for row in fx.store.intentions(kind=["message"], types=["ask_ack"], limit=10)]
    texts = sorted(row.context["text"] for row in acks)
    assert len(acks) == 2 and texts[0].startswith("OK, approved: ") and texts[1].startswith("OK, declined: ")
    assert all(row.status == "approved" and row.entity_id == OWNER for row in acks)
    assert not any("K7F" in text or "M4P" in text for text in texts)


async def test_the_answers_record_keeps_ids_and_statuses_never_words(fx):
    row = asked(fx, title="Send the draft that says the launch slips to March")
    own = deliver(fx, row, message_id="m-1")
    fx.shift(minutes=1)
    result = await fx.mind.read_answer(evidence(fx, "yes, go ahead please", message_id="in-1", ledger=[sent(own)]))
    kept = fx.mind.answer_turns.find(result["key"])
    assert kept.result["settled"] == [{"id": row.id, "status": "approved"}] and kept.result["approved"] == [row.id]
    assert "slips to March" not in repr(kept) and "go ahead" not in repr(kept)
    again = await fx.mind.read_answer(evidence(fx, "yes, go ahead please", message_id="in-1", ledger=[sent(own)]))
    assert again["read"] is False and again["context"] == "" and "slips" not in repr(again)
