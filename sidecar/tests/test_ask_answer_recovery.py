"""An unbound answer shows one recent ask again without granting anything."""

from datetime import timedelta

import pytest

from colonymind.mind import answers
from test_ask_answers_binding import asked, evidence, fx, reply, sent
from test_mind_loop import pinned_clock


def shown(fx, *rows, platform="rcs", chat="owner-chat", message="ask-first"):
    note = fx.mind.outbox.notice(type="ask_show", title="Held call", text="Let this call run once?",
                                dedup_key=message, extra={"asks": [row.id for row in rows]})
    fx.mind.outbox.sending(note.id, target=f"{platform}:{chat}")
    return fx.mind.outbox.sent(note.id, result="sent", hermes_ref=message)


@pytest.mark.parametrize("platform", ["rcs", "whatsapp", "sms"])
@pytest.mark.parametrize("word", ["yes", "no"])
@pytest.mark.parametrize("stale_count", [0, 2])
async def test_completion_after_ask_recovers_in_same_chat_then_fresh_answer_binds(fx, platform, word, stale_count):
    stale = [asked(fx, code=f"S{index}Z", cls="floor", title=f"Earlier call {index}")
             for index in range(stale_count)]
    for index, old in enumerate(stale):
        shown(fx, old, platform=platform, message=f"earlier-ask-{index}")
        fx.shift(minutes=1)
    row = asked(fx, type_="floor_effect", cls="floor", title="Remove the current fixture")
    notice = shown(fx, row, platform=platform)
    fx.shift(seconds=6)
    completion = reply(fx.now)
    fx.shift(seconds=26)
    incoming = evidence(fx, word, message_id="answer-first", platform=platform,
                        ledger=[sent(notice), completion])
    first = await fx.mind.read_answer(incoming)
    assert first["unbound"] and not first["settled"] and not first["approved"]
    assert fx.store.get(row.id).status == "asked"
    assert [item["id"] for item in first["shown"]] == [row.id]
    assert all(fx.store.get(old.id).status == "asked" for old in stale)
    assert row.id in first["context"]
    assert "this chat" in first["context"] and "new hand" in first["context"]
    assert "not read as an answer" in first["context"]
    assert [event["kind"] for event in fx.mind.answer_events(fx.now - timedelta(hours=1))] == ["unbound_answer"]
    again = await fx.mind.read_answer(incoming)
    assert not again["read"]
    assert len(fx.store.intentions(kind=["message"], types=["ask_show"], status=["approved"])) == 1
    message = first["shown"][0]["message"]
    ready = next(item for item in fx.mind.outbox.ready() if item["id"] == message)
    assert ready["target"] == f"{platform}:owner-chat"
    if platform == "rcs":
        assert ready["delivery_ref"] == {"platform": platform, "chat": "owner-chat",
                                         "message_id": "answer-first"}
    fx.shift(seconds=10)
    fx.mind.outbox.sending(message, target=ready["target"])
    fresh = fx.mind.outbox.sent(message, result="sent", hermes_ref="ask-again")
    fx.shift(seconds=10)
    second = await fx.mind.read_answer(evidence(fx, word, message_id="answer-fresh", platform=platform,
                                                ledger=[sent(notice), completion, sent(fresh)]))
    assert second["path"] == "follow_up"
    assert [item["id"] for item in second["settled"]] == [row.id]
    assert fx.store.get(row.id).status == ("approved" if word == "yes" else "cancelled")
    assert all(fx.store.get(old.id).status == "asked" for old in stale)


@pytest.mark.parametrize("case", ["unseen", "other_chat", "other_platform", "future", "too_old", "expired",
                                  "unclear", "reaction"])
async def test_recovery_never_guesses_a_reference_or_reopens_a_stale_ask(fx, case):
    row = asked(fx, type_="floor_effect", cls="floor")
    if case != "unseen":
        shown(fx, row, platform="sms" if case == "other_platform" else "rcs",
              chat="different-chat" if case == "other_chat" else "owner-chat")
    arrived = fx.now - timedelta(seconds=1) if case == "future" else None
    fx.shift(hours=answers.FOLLOW_UP_HOURS + 1 if case == "too_old" else 1)
    if case == "expired":
        fx.store.update(row.id, expires_at=fx.now - timedelta(seconds=1))
    extra = {"arrived_at": arrived.isoformat()} if arrived else {}
    if case == "reaction":
        extra.update(reaction="👍", reply_to_message_id="unrelated-message")
    result = await fx.mind.read_answer(evidence(fx, "maybe" if case == "unclear" else "yes",
                                                message_id="unbound", platform="rcs", ledger=[], **extra))
    assert not result["shown"] and not result["settled"]


async def test_one_ask_here_is_not_confused_with_an_open_ask_in_another_chat(fx):
    here = asked(fx, cls="floor")
    away = asked(fx, code="Q2Z", cls="floor")
    shown(fx, here)
    shown(fx, away, chat="other-chat", message="ask-away")
    fx.shift(minutes=1)
    result = await fx.mind.read_answer(evidence(fx, "yes", message_id="unbound", platform="rcs", ledger=[]))
    assert [item["id"] for item in result["shown"]] == [here.id]
    assert not result["approved"]


async def test_old_floor_ask_does_not_block_recovery_of_a_recent_ask(fx):
    old = asked(fx, cls="floor")
    shown(fx, old)
    fx.shift(hours=answers.FOLLOW_UP_HOURS + 1)
    recent = asked(fx, code="Q2Z", cls="floor")
    shown(fx, recent, message="recent-ask")
    fx.shift(minutes=1)
    result = await fx.mind.read_answer(evidence(fx, "yes", message_id="unbound", platform="rcs", ledger=[]))
    assert result["unbound"] and not result["settled"]
    assert [item["id"] for item in result["shown"]] == [recent.id]
    assert fx.store.get(old.id).status == "asked"


async def test_recovery_uses_the_latest_show_not_the_latest_created_ask(fx):
    earlier = asked(fx, cls="floor")
    shown(fx, earlier)
    fx.shift(minutes=1)
    newer = asked(fx, code="Q2Z", cls="floor")
    shown(fx, newer, message="newer-ask")
    fx.shift(minutes=1)
    fx.mind.record_shown([earlier.id], platform="rcs", chat="owner-chat", message_id="earlier-again", at=fx.now)
    fx.shift(minutes=1)
    result = await fx.mind.read_answer(evidence(fx, "yes", message_id="unbound", platform="rcs", ledger=[]))
    assert result["unbound"] and not result["settled"]
    assert [item["id"] for item in result["shown"]] == [earlier.id]


@pytest.mark.parametrize("case", ["expired", "cancelled", "other_chat", "other_platform", "future"])
async def test_recovery_selects_the_latest_eligible_show_in_this_chat(fx, case):
    eligible = asked(fx, cls="floor")
    shown(fx, eligible)
    fx.shift(minutes=1)
    latest = asked(fx, code="Q2Z", cls="floor")
    arrived = fx.now - timedelta(seconds=1) if case == "future" else fx.now
    shown(fx, latest, platform="sms" if case == "other_platform" else "rcs",
          chat="other-chat" if case == "other_chat" else "owner-chat", message="latest-show")
    if case == "expired":
        fx.store.update(latest.id, expires_at=fx.now - timedelta(seconds=1))
    elif case == "cancelled":
        fx.mind.outcomes.record(latest.id, status="cancelled", summary="No longer needed", by="test")
    fx.shift(minutes=1)
    result = await fx.mind.read_answer(evidence(fx, "yes", message_id="unbound", platform="rcs", ledger=[],
                                                arrived_at=arrived.isoformat()))
    assert result["unbound"] and not result["settled"] and not result["approved"]
    assert [item["id"] for item in result["shown"]] == [eligible.id]


async def test_recovery_stays_with_its_answer_even_if_original_hand_route_or_latest_chat_differs(fx):
    row = asked(fx, cls="floor", context={"origin": "hand", "delivery_ref": {
        "platform": "rcs", "chat": "original-chat", "message_id": "original-request"}})
    # An existing explicit show reached another owner chat through the plugin.
    fx.mind.record_shown([row.id], platform="whatsapp", chat="owner-chat", message_id="shown-here", at=fx.now)
    fx.shift(minutes=1)
    result = await fx.mind.read_answer(evidence(fx, "yes", message_id="unbound", platform="whatsapp", ledger=[]))
    fx.shift(minutes=1)
    await fx.mind.read_answer(evidence(fx, "hello", message_id="elsewhere", platform="sms", chat_id="latest-chat"))
    note = result["shown"][0]["message"]
    ready = next(item for item in fx.mind.outbox.ready() if item["id"] == note)
    assert ready["target"] == "whatsapp:owner-chat" and "delivery_ref" not in ready
    assert fx.mind.outbox.sending(note, target="sms:latest-chat") is None
    assert fx.mind.outbox.sending(note, target="whatsapp:owner-chat") is not None


async def test_pending_show_in_another_chat_does_not_steal_recovery(fx):
    row = asked(fx, cls="floor")
    shown(fx, row, chat="first-chat")
    shown(fx, row, chat="second-chat", message="second-show")
    fx.shift(minutes=1)
    replies = []
    for index, chat in enumerate(("first-chat", "second-chat", "second-chat")):
        result = await fx.mind.read_answer(evidence(fx, "yes", message_id=f"unbound-{index}",
                                                    platform="rcs", chat_id=chat, ledger=[]))
        replies.append(result["shown"][0]["message"])
        message = next(item for item in fx.mind.outbox.ready() if item["id"] == replies[-1])
        assert message["target"] == f"rcs:{chat}"
    assert replies[0] != replies[1] == replies[2]
    assert fx.store.get(row.id).status == "asked"


async def test_replaced_ask_cannot_deliver_a_queued_recovery_show(fx):
    row = asked(fx, cls="floor")
    shown(fx, row)
    fx.shift(minutes=1)
    result = await fx.mind.read_answer(evidence(fx, "yes", message_id="unbound", platform="rcs", ledger=[]))
    note = result["shown"][0]["message"]
    fx.mind.outcomes.record(row.id, status="cancelled", summary="replaced by a newer identical hold", by="hands")
    assert note not in {item["id"] for item in fx.mind.outbox.ready()}
    assert fx.mind.outbox.sending(note, target="rcs:owner-chat") is None
