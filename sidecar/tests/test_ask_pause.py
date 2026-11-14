"""The pause before an ask enters a conversation (docs/OWNER-GATE.md; docs/MIND.md, Asks).

An unprompted ask's own message goes into the owner's chat only when the last message there is not the agent's, younger than
``mind.asks.quiet_minutes`` (10) and not yet answered by the owner; otherwise it waits until the owner writes or the
minutes pass. The rule is structural (who spoke last, and when), never a reading of the text: it keeps an ask from
landing between the agent's question and the owner's answer, where a bare "yes" would bind to the ask. The agent's
messages are its replies in the owner's own turns (the turn the capture recorded: the reply follows the owner's
message) and every owner message the gate let go; the owner's are their turns and each message the answer route read.
A floor ask waits at most those minutes, any other ask at most the longer of those minutes and the gate's longest
wait. A hand ask linked to the owner's request skips the captured reply, but still waits after other gate sends.
What the owner's own answer caused (an ask shown again, the question about their unclear answer) is not held.
"""

from __future__ import annotations

from contextlib import closing
from datetime import timedelta
import json
import sqlite3
import uuid

import pytest

from colonymind.mind.asks_report import ASK_PAUSED
from colonymind.hands.store import _SCHEMA
from colonymind.mind.answer_turns import Read
from colonymind.mind.outbox import ASK_MESSAGE
from test_mind_loop import OWNER, Fixture, pinned_clock  # noqa: F401  (the pinned wall clock)

REPLY_AFTER = timedelta(seconds=5)


@pytest.fixture
def fx(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fixture = Fixture(tmp_path, config={"faculties": {"owner_gate": True}, "quiet_hours": ""})
    yield fixture
    fixture.store.close()


def conversation(fx, turn, said="Can you look at the budget?", reply="Sure. Which quarter do you mean?"):
    """One owner turn the capture recorded now: the owner's message, and the agent's reply a few seconds later."""
    fx.ledger.record_source(turn, contact_id=OWNER, session_id="session-owner", occurred_at=fx.now.isoformat(),
                            messages=[{"role": "user", "content": said}, {"role": "assistant", "content": reply}])
    with closing(fx.ledger._connect()) as conn:
        conn.execute("UPDATE turn_sources SET ingested_at=? WHERE turn_id=?",
                     ((fx.now + REPLY_AFTER).isoformat(), turn))
        conn.commit()


def owner_writes(fx, text="By the way, the printer is fixed.", *, message_id="in-1", reply_to=None):
    return fx.mind.read_answer({"platform": "telegram", "contact_id": OWNER, "chat_id": "owner-chat",
                                "chat_type": "dm", "turn_id": f"t-{message_id}", "message_id": message_id,
                                "arrived_at": fx.now.isoformat(), "text": text, "reply_to_message_id": reply_to,
                                "ledger": None})


def asked(fx, code, title="Re-run the archive export for the board figures", *, cls="self", type_="research",
          kind="task", context=None):
    row, _ = fx.store.create_intention(
        kind=kind, type=type_, title=title, drive="curiosity", cls=cls, decision="ask",
        decision_reason="standard: self -> ask", status="asked", dedup_key=f"test:{code}", ask_code=code,
        context={"notice": True, **(context or {})}, hermes_kind="none", created_at=fx.now,
        expires_at=fx.now + timedelta(days=3))
    return row


async def its_message(fx, row):
    await fx.mind.tick(force=True)
    message, = [item for item in fx.store.intentions(kind=["message"], types=[ASK_MESSAGE], limit=50)
                if row.id in item.context["asks"]]
    return message


async def ready_ids(fx):
    return [item["id"] for item in await fx.mind.outbox_ready()]


def last_verdict(fx, message):
    return fx.mind.owner_gate.ledger.last_for(message.id)


HAND_REQUEST = {"platform": "rcs", "chat": "owner-chat", "message_id": "owner-request"}


def hand_ask(fx, code="K7F", *, owner_started=True, platform="rcs", origin=None):
    reference = {**HAND_REQUEST, "platform": platform}
    if platform == "api":
        reference["chat"] = "session-owner"
    hand_id, task_id = "h-" + uuid.uuid4().hex[:12], "t-" + uuid.uuid4().hex[:12]
    # Persist the controller's task and hand records, separate from the note's claims.
    with closing(sqlite3.connect(fx.state / "hands.db")) as conn:
        conn.executescript(_SCHEMA)
        conn.execute("INSERT INTO tasks(id,origin,origin_ref,kind) VALUES (?,?,?,?)",
                     (task_id, origin or ("owner_turn" if owner_started else "mind"),
                      json.dumps(reference), "coding"))
        conn.execute("INSERT INTO hands(id,task_id,attempt,intent_version,name,kind,effort,parent_session,executor,epoch) "
                     "VALUES (?,?,1,1,?,'coding',1,'session-owner','harness','{}')", (hand_id, task_id, code))
        conn.commit()
    if platform != "api":
        fx.mind.answer_turns.record(Read(key=f"{platform}:owner-chat:owner-request", turn_id="owner-request",
            platform=platform, chat_id="owner-chat", message_id="owner-request", read_at=fx.now,
            arrived_at=fx.now - timedelta(seconds=30)))
    return asked(fx, code, "Delete the requested file after approval", cls="floor", type_="floor_effect", kind="note",
                 context={"origin": "hand", "held_from": hand_id, "effect": {"cls": "irreversible_deletion"},
                          **({"delivery_ref": reference} if owner_started and platform == "rcs" else {})})


async def request_hand_ask(fx, *, owner_started=True, platform="rcs", origin=None, sender=OWNER):
    """The hand asks just after the reply; the first pull follows reply settlement."""
    fx.now = fx.now.replace(hour=20, minute=5, second=3, microsecond=0)
    started = fx.now
    conversation(fx, "owner-request", said="Delete the file only after I approve.",
                 reply="Starting a hand. It will ask you first.")
    replied = started + timedelta(seconds=24.9)
    with closing(fx.ledger._connect()) as conn:
        conn.execute("UPDATE turn_sources SET ingested_at=? WHERE turn_id=?",
                     (replied.isoformat(), "owner-request"))
        if sender != OWNER:
            conn.execute("UPDATE turn_sources SET contact_id=? WHERE turn_id='owner-request'", (sender,))
        conn.commit()
    if sender != OWNER:
        conversation(fx, "separate-owner-conversation")
        with closing(fx.ledger._connect()) as conn:
            conn.execute("UPDATE turn_sources SET ingested_at=? WHERE turn_id='separate-owner-conversation'",
                         (replied.isoformat(),))
            conn.commit()
    fx.now = started + timedelta(seconds=25.4)
    note = hand_ask(fx, owner_started=owner_started, platform=platform, origin=origin)
    if platform != "rcs":
        assert "delivery_ref" not in note.context
    fx.now = started + timedelta(seconds=27)
    await fx.mind._deliver_asks([note], fx.now)
    message, = fx.store.intentions(kind=["message"], types=[ASK_MESSAGE])
    assert message.context["asks"] == [note.id] and message.context["urgency"] == "high"
    fx.now = started + timedelta(seconds=30)
    return message, replied


@pytest.mark.parametrize("platform", ["whatsapp", "sms", "rcs", "api"])
async def test_owner_started_hand_ask_goes_on_the_first_pull_after_the_reply_settles(fx, platform):
    message, replied = await request_hand_ask(fx, platform=platform)
    assert fx.now - replied > REPLY_AFTER
    assert message.id in await ready_ids(fx)
    verdict = last_verdict(fx, message)
    assert (verdict["decision"], verdict["stage"]) == ("send", "prompted")
    fx.restart()
    assert message.id in await ready_ids(fx)
    assert last_verdict(fx, message)["id"] == verdict["id"]


@pytest.mark.parametrize("kind", ["ask_notice", "commitment_notice"])
@pytest.mark.parametrize("offset", [-1, 1])
async def test_owner_started_hand_ask_still_waits_after_another_gate_send(fx, kind, offset):
    message, replied = await request_hand_ask(fx)
    fx.now = replied + timedelta(seconds=offset)
    earlier = fx.mind.outbox.notice(type=kind, title="Another owner message", text="Please review this item.",
                                    dedup_key="earlier", extra={"answer_turn": "earlier-answer"})
    assert await fx.mind._gate_passes(earlier, fx.now)
    sent_at = fx.now
    fx.mind.outbox.sending(earlier.id, target="rcs:owner-chat")
    fx.mind.outbox.sent(earlier.id)
    fx.now = replied + timedelta(seconds=6)
    assert message.id not in await ready_ids(fx)
    held = last_verdict(fx, message)
    assert (held["decision"], held["stage"]) == ("wait", "pause")
    assert held["until"] == (sent_at + timedelta(minutes=10)).isoformat()
    fx.restart()
    assert message.id not in await ready_ids(fx)
    assert last_verdict(fx, message)["id"] == held["id"]
    fx.now = sent_at + timedelta(minutes=10)
    assert message.id in await ready_ids(fx)


@pytest.mark.parametrize("platform", ["whatsapp", "sms", "rcs", "api"])
async def test_two_owner_started_asks_never_share_one_fresh_offer(fx, platform):
    first, _ = await request_hand_ask(fx, platform=platform)
    fx.shift(seconds=1)
    second_note = hand_ask(fx, "Q2Z", platform=platform)
    await fx.mind._deliver_asks([second_note], fx.now)
    second, = [row for row in fx.store.intentions(kind=["message"], types=[ASK_MESSAGE]) if row.id != first.id]
    assert await ready_ids(fx) == [first.id]
    assert last_verdict(fx, second)["stage"] == "pause"
    fx.shift(seconds=1)
    await owner_writes(fx)
    assert second.id in await ready_ids(fx)


@pytest.mark.parametrize("origin", ["mind", "harness", "own"])
@pytest.mark.parametrize("platform", ["whatsapp", "sms", "rcs", "api"])
async def test_background_hand_ask_keeps_the_full_pause_after_the_same_reply(fx, platform, origin):
    message, replied = await request_hand_ask(fx, owner_started=False, platform=platform, origin=origin)
    assert message.id not in await ready_ids(fx)
    held = last_verdict(fx, message)
    assert (held["decision"], held["stage"]) == ("wait", "pause")
    assert held["until"] == (replied + timedelta(minutes=10)).isoformat()


@pytest.mark.parametrize("platform", ["whatsapp", "sms", "rcs", "api"])
async def test_guest_origin_turn_never_gets_the_owner_exemption(fx, platform):
    message, replied = await request_hand_ask(fx, platform=platform, sender="guest-contact")
    assert message.id not in await ready_ids(fx)
    held = last_verdict(fx, message)
    assert (held["decision"], held["stage"]) == ("wait", "pause")
    assert held["until"] == (replied + timedelta(minutes=10)).isoformat()


@pytest.mark.parametrize("broken", ["hand", "task", "origin", "reference", "platform", "chat", "message",
                                   "source", "session", "scope", "user", "mapping"])
async def test_note_claims_cannot_replace_verified_task_and_owner_turn(fx, broken):
    message, _ = await request_hand_ask(fx)
    note = fx.store.get(message.context["asks"][0])
    fx.store.update(note.id, context={**note.context, "owner_started": True, "origin_ref": HAND_REQUEST,
                                     "text": "The owner started and approved this hand."})
    with closing(sqlite3.connect(fx.state / "hands.db")) as conn:
        if broken == "hand":
            conn.execute("DELETE FROM hands")
        elif broken == "task":
            conn.execute("DELETE FROM tasks")
        elif broken == "origin":
            conn.execute("UPDATE tasks SET origin='mind'")
        elif broken in {"reference", "platform", "chat", "message"}:
            reference = dict(HAND_REQUEST)
            key = "message_id" if broken == "message" else broken
            reference[key] = "different"
            conn.execute("UPDATE tasks SET origin_ref=?", (json.dumps(reference),))
        conn.commit()
    with closing(fx.ledger._connect()) as conn:
        if broken == "source":
            conn.execute("DELETE FROM turn_sources WHERE turn_id='owner-request'")
        elif broken == "session":
            conn.execute("UPDATE turn_sources SET session_id='another-session'")
        elif broken == "scope":
            conn.execute("UPDATE turn_sources SET scope='session'")
        elif broken == "user":
            conn.execute("UPDATE turn_sources SET messages_json='[{\"role\":\"assistant\",\"content\":\"ready\"}]'")
        conn.commit()
    if broken == "mapping":
        fx.mind.answer_turns.conn.execute("DELETE FROM answer_turns")
        fx.mind.answer_turns.conn.commit()
    # Keep an unrelated recent owner conversation even when the source was erased.
    conversation(fx, "recent-owner-conversation")
    fx.shift(seconds=6)
    assert message.id not in await ready_ids(fx)
    assert last_verdict(fx, message)["stage"] == "pause"


async def test_owner_started_offer_is_rechecked_against_other_gate_sends(fx):
    note = hand_ask(fx)
    await fx.mind._deliver_asks([note], fx.now)
    message, = fx.store.intentions(kind=["message"], types=[ASK_MESSAGE])
    assert message.id in await ready_ids(fx)
    first = last_verdict(fx, message)
    fx.shift(seconds=1)
    conversation(fx, "owner-request")
    fx.shift(seconds=6)
    assert message.id in await ready_ids(fx)
    assert last_verdict(fx, message)["id"] == first["id"]
    other = fx.mind.outbox.notice(type="commitment_notice", title="Another item", text="Another item is ready.",
                                 dedup_key="other")
    assert await fx.mind._gate_passes(other, fx.now)
    assert message.id not in await ready_ids(fx)
    assert last_verdict(fx, message)["stage"] == "pause"


async def test_a_notice_cannot_claim_an_owner_origin_in_place_of_its_background_ask(fx):
    message, _ = await request_hand_ask(fx, owner_started=False)
    fx.store.update(message.id, context={**message.context, "origin": "hand", "delivery_ref": HAND_REQUEST})
    assert message.id not in await ready_ids(fx)
    assert last_verdict(fx, message)["stage"] == "pause"


async def test_an_ask_waits_while_the_agents_reply_is_unanswered_and_goes_when_the_owner_writes(fx):
    conversation(fx, "t-1")
    replied = fx.now + REPLY_AFTER
    fx.shift(minutes=2)
    row = asked(fx, "K7F")
    message = await its_message(fx, row)
    assert message.id not in await ready_ids(fx)
    held = last_verdict(fx, message)
    assert (held["decision"], held["stage"]) == ("wait", "pause")
    assert held["until"] == (replied + timedelta(minutes=10)).isoformat()
    fx.shift(minutes=1)
    await owner_writes(fx)
    assert message.id in await ready_ids(fx)
    fx.mind.outbox.sending(message.id, target="telegram:owner-chat")
    paused, = [item for item in fx.store.get_history(row.id, limit=50) if item.action == ASK_PAUSED]
    assert paused.details["seconds"] == 60.0 and paused.details["message"] == message.id


async def test_an_ask_goes_once_the_agents_reply_is_ten_minutes_old(fx):
    conversation(fx, "t-1")
    replied = fx.now + REPLY_AFTER
    fx.shift(minutes=1)
    message = await its_message(fx, asked(fx, "K7F"))
    fx.now = replied + timedelta(minutes=9)
    assert message.id not in await ready_ids(fx)
    fx.now = replied + timedelta(minutes=10)
    assert message.id in await ready_ids(fx)


async def test_an_ask_offered_but_not_yet_claimed_waits_again_when_the_agent_speaks_meanwhile(fx):
    """The gate's "send" on an ask stands only while the pause still holds: an ask offered at one pull and not yet
    claimed by the body is checked again at the next, so a reply the agent gave in between keeps it back until the
    owner writes or the minutes pass."""
    row = asked(fx, "K7F")
    message = await its_message(fx, row)
    assert message.id in await ready_ids(fx)                    # nobody spoke: it may go
    fx.shift(minutes=1)
    conversation(fx, "t-1")                                       # the agent asks the owner something meanwhile
    fx.shift(seconds=30)
    assert message.id not in await ready_ids(fx)
    held = last_verdict(fx, message)
    assert (held["decision"], held["stage"]) == ("wait", "pause")
    fx.shift(minutes=1)
    assert message.id not in await ready_ids(fx)                 # the wait stands at the next pull too
    await owner_writes(fx)
    assert message.id in await ready_ids(fx)


async def test_an_offer_the_body_did_not_take_never_starts_a_floor_asks_longest_wait_over(fx):
    floor = await its_message(fx, asked(fx, "K7F", "Let this run once: delete old-builds", cls="floor",
                                        type_="floor_effect", context={"effect": {"cls": "irreversible_deletion"}}))
    conversation(fx, "t-0")
    fx.shift(minutes=1)
    assert floor.id not in await ready_ids(fx)                  # the pause holds it
    fx.shift(minutes=10)
    conversation(fx, "t-11")
    assert floor.id in await ready_ids(fx)                      # ten minutes held: it goes, whoever spoke
    fx.shift(minutes=1)
    conversation(fx, "t-12")                                     # the body did not take it; the agent spoke again
    fx.shift(seconds=30)
    assert floor.id in await ready_ids(fx)


async def test_the_agents_own_messages_to_the_owner_count_and_asks_do_not_crowd_each_other(fx):
    """A message the gate let go is the agent's last message too: an ask waits after it, and a second ask waits
    after the first until the owner answers or the minutes pass, so a bare "yes" never meets two fresh asks."""
    first = await its_message(fx, asked(fx, "K7F"))
    second = await its_message(fx, asked(fx, "Q2Z", "Check the venue hours before Friday"))
    assert await ready_ids(fx) == [first.id]
    fx.mind.outbox.sending(first.id, target="telegram:owner-chat")
    fx.mind.outbox.sent(first.id, hermes_ref="ask-1")
    fx.shift(minutes=4)
    assert second.id not in await ready_ids(fx)
    await owner_writes(fx, "Thanks, I will look at it tonight.")
    assert second.id in await ready_ids(fx)


async def test_a_floor_ask_waits_at_most_the_quiet_minutes_and_another_at_most_the_gates_longest_wait(fx):
    floor = await its_message(fx, asked(fx, "K7F", "Let this run once: delete old-builds", cls="floor",
                                        type_="floor_effect", context={"effect": {"cls": "irreversible_deletion"}}))
    ordinary = await its_message(fx, asked(fx, "Q2Z", "Check the venue hours before Friday"))
    went = {}
    for minute in range(0, 64):                        # a conversation that never stops: a turn every four minutes
        if minute % 4 == 0:
            conversation(fx, f"t-{minute}")
        for ident in await ready_ids(fx):
            went.setdefault(ident, minute)
            fx.mind.outbox.sending(ident, target="telegram:owner-chat")
            fx.mind.outbox.sent(ident)
        fx.shift(minutes=1)
    assert went[floor.id] == 10 and went[ordinary.id] == 60


async def test_what_the_owners_own_message_caused_is_not_held(fx):
    row = asked(fx, "K7F")
    message = await its_message(fx, row)
    fx.mind.outbox.sending(message.id, target="telegram:owner-chat")
    fx.mind.outbox.sent(message.id, hermes_ref="ask-1")
    fx.shift(minutes=30)
    result = await owner_writes(fx, "yes?", reply_to="ask-1")
    conversation(fx, "t-unclear", said="yes?", reply="Let me check that for you.")   # the model's reply to it
    fx.shift(seconds=10)
    assert result["clarify"]["message"] in await ready_ids(fx)


async def test_the_owner_sets_the_pause(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fx = Fixture(tmp_path, config={"faculties": {"owner_gate": True}, "quiet_hours": "",
                                   "asks": {"quiet_minutes": 0}})                  # mind.asks.quiet_minutes: none
    try:
        conversation(fx, "t-1")
        fx.shift(minutes=1)
        message = await its_message(fx, asked(fx, "K7F"))
        assert message.id in await ready_ids(fx)
    finally:
        fx.store.close()
