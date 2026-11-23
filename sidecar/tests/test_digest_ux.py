"""The owner's digest in plain words (docs/MIND.md, The daily digest; Asks).

The digest is one message a day, only when something waits, in labelled parts: actions waiting for the owner's OK
(in words, with no code and no reply instruction: each ask's own message carries its reply line) and what was done or
found. No item is ever cut mid-word or mid-sentence, and no internal field name reaches the owner. A contradiction goes
through a typed decision: an addition is compatible, and when the newer statement replaces the older the mind keeps the
newer one itself and asks the owner nothing. A digest arrives once a day whatever the restarts, retries and bodies.
"""

from __future__ import annotations

import json
import re
from contextlib import closing
from datetime import timedelta
from types import SimpleNamespace

from colonymind.mind import plain
from colonymind.mind import digest as digest_text
from colonymind.mind.consolidate import TASK_DIGEST
from owner_answer import owner_answers
from test_mind_consolidate import OWNER, Fixture as NightFixture, NightRouter

CODE = re.compile(r"\b[A-Z2-9]{3}\b")
FIELD_NAMES = ("office_location", "subject_key", "predicate", "speaker", "p1a", "claim:", "cid-", "intention:")


def items_are_whole(text: str) -> None:
    """Every item line of a digest or message ends as a sentence ends and carries no ellipsis."""
    for line in text.splitlines():
        if not line.startswith("- "):
            continue
        assert "…" not in line and "..." not in line, line
        assert line.rstrip("\"')]")[-1] in ".?!", line


def no_field_names(text: str) -> None:
    for name in FIELD_NAMES:
        assert name not in text, (name, text)


# ---------------------------------------------------------------------------
# Plain words: whole sentences, a checked summary, or a pointer; never a cut
# ---------------------------------------------------------------------------

def test_fit_keeps_whole_sentences_and_never_cuts_a_word_or_a_sentence():
    assert plain.fit("Research the venue hours", 80) == "Research the venue hours."
    assert plain.fit("It failed. The run took too long. Try again tomorrow.", 40) == "It failed. The run took too long."
    assert plain.fit("One sentence that is much too long to fit in the room it has been given here.", 30) is None
    assert plain.fit("A report that was already cut…", 200) is None
    assert plain.whole("The venue opens at nine.", "The venue opens at nine on weekdays and ten on Sundays.", 60) is None
    assert plain.whole("The venue opens at nine on weekdays.", "The venue opens at nine on weekdays. It closes at six.",
                       60) == "The venue opens at nine on weekdays."
    assert plain.whole("It opens at nine on weekdays and ten on Sundays.",
                       "The venue opens at nine on weekdays and ten on Sundays.", 60)
    assert plain.whole("Opens at nine…", "Opens at nine on weekdays.", 60) is None
    assert plain.pointer("a report about kites").startswith("A report about kites is too long to show here")


async def test_shown_prefers_whole_sentences_then_a_checked_summary_then_a_pointer():
    long = ("The overnight export of the board figures stopped because the archive service refused every request "
            "after the first two hundred rows, and the retry budget ran out before the service recovered at dawn")

    async def good(text, limit):
        return "The overnight export stopped when the archive service refused requests."

    async def cheat(text, limit):
        return long[:limit - 1] + "…"

    assert await plain.shown(long, 120, what="a failure report", summarize_with=good) == (
        "The overnight export stopped when the archive service refused requests.")
    shown = await plain.shown(long, 120, what="a failure report", summarize_with=cheat)
    assert shown.startswith("A failure report is too long to show here") and "…" not in shown
    assert (await plain.shown(long, 120, what="a failure report")).startswith("A failure report is too long")


def test_render_has_two_labelled_parts_and_nothing_when_nothing_waits():
    assert digest_text.render() is None
    assert digest_text.render(context=["Still working on the garden plan."]) is None
    text = digest_text.render(needs=["Re-run the archive export. Asked 2 hours ago."],
                              done=["Done: the weekly backup check."])
    assert text.split("\n\n") == [
        "Daily digest",
        "Waiting for your OK:\n- Re-run the archive export. Asked 2 hours ago.",
        "Done or found since yesterday:\n- Done: the weekly backup check.",
    ]


# ---------------------------------------------------------------------------
# The served digest: labelled parts, no code, whole sentences, no field names
# ---------------------------------------------------------------------------

class TurnRouter(NightRouter):
    """The night's router, which also answers the owner-turn typed decisions (not background work) and the
    plain summaries; ``answers`` may map any task to a payload or a callable."""

    async def complete(self, messages, *, context=None, **kwargs):
        task = (context or {}).get("task")
        if task in {"mind_question_answer", "mind_owner_instruction", "mind_report_substance"}:
            self.calls.append((messages, context))
            answer = self.answers.get(task)
            payload = answer(messages, context) if callable(answer) else (answer or {"answer": "unsure"})
            return SimpleNamespace(content=json.dumps(payload), usage={"total_tokens": 10})
        return await super().complete(messages, context=context, **kwargs)


def night(tmp_path, monkeypatch, *, autonomy="standard", relation="replaces", **answers):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    router = TurnRouter({"mind_claim_relation": {"relation": relation}, **answers})
    fx = NightFixture(tmp_path, autonomy=autonomy, router=router)
    fx.router.answers[TASK_DIGEST] = None
    return fx


def asked(fx, code, title, *, type_="research", kind="task", **fields):
    row, _ = fx.store.create_intention(
        kind=kind, type=type_, title=title, drive="curiosity", cls="self", decision="ask",
        decision_reason="suggest: self -> ask", status="asked", dedup_key=f"test:{code}", ask_code=code,
        context={"notice": False, **fields.pop("context", {})}, hermes_kind="none", created_at=fx.now,
        expires_at=fx.now + timedelta(days=3), **fields)
    return row


def acted(fx, key, title, *, outcome, status, reason=None):
    row, _ = fx.store.create_intention(
        kind="task", type="research", title=title, drive="curiosity", cls="self", decision="act",
        decision_reason="standard: self -> act", status="approved", dedup_key=f"test:{key}", hermes_kind="task",
        created_at=fx.now)
    return fx.store.update(row.id, status=status, outcome=outcome, failed_reason=reason)


async def the_digest(fx):
    fx.mind.digest_hour = 0
    fx.mind.quiet = None
    ident = await fx.mind._digest(fx.now)
    return fx.store.get(ident) if ident else None


async def test_the_digest_puts_actions_and_news_in_labelled_parts_with_no_code(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch, autonomy="suggest")
    asked(fx, "K7F", "Re-run the archive export for the board figures")
    acted(fx, "done", "Check the venue hours", outcome="done", status="done")
    row = await the_digest(fx)
    text = row.context["text"]
    parts = text.split("\n\n")
    assert parts[0] == "Daily digest"
    assert [part.splitlines()[0] for part in parts[1:]] == ["Waiting for your OK:", "Done or found since yesterday:"]
    assert digest_text.lines_of(text, digest_text.WAITING) == [
        "Re-run the archive export for the board figures. Asked less than an hour ago."]
    assert "K7F" not in text and "code" not in text.lower() and "reply" not in text.lower()
    assert "Done: Check the venue hours." in digest_text.lines_of(text, digest_text.DONE)
    items_are_whole(text)
    no_field_names(text)
    assert "yes <code>" not in text


async def test_a_long_item_is_summarised_whole_and_an_unchecked_summary_is_never_shown(tmp_path, monkeypatch):
    reason = ("the archive service refused every request after the first two hundred rows and kept refusing until "
              "the retry budget ran out, well before the service recovered, so none of the remaining figures were "
              "exported and the board pack for the quarter still lacks its appendix tables")
    summaries = iter(["The export stopped when the archive service refused requests.", reason[:200] + "…"])
    fx = night(tmp_path, monkeypatch,
               mind_plain_summary=lambda messages, context: {"sentence": next(summaries)})
    for index in range(2):
        acted(fx, f"f{index}", f"Export the board figures {index + 1}", outcome="failed", status="failed",
              reason=reason)
    text = (await the_digest(fx)).context["text"]
    done = digest_text.lines_of(text, digest_text.DONE)
    assert any("The export stopped when the archive service refused requests." in line for line in done)
    assert any("is too long to show here; ask me and I will send it in full." in line for line in done)
    assert reason[:120] not in text
    items_are_whole(text)


async def test_nothing_waiting_means_no_digest(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch)
    assert await the_digest(fx) is None
    assert fx.messages("digest") == []


async def test_an_asks_message_sent_on_its_own_is_plain_words(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch)
    ask = asked(fx, "M4P", "Re-run the archive export for the board figures", context={"notice": True})
    notice, = fx.mind.outbox.deliver_asks(fx.mind.outbox.asks_due([ask]))
    assert notice.context["text"] == "Re-run the archive export for the board figures. Reply yes or no."
    assert notice.context["asks"] == [ask.id]


# ---------------------------------------------------------------------------
# Contradictions: a typed decision first; an addition is compatible, a replacement is settled by the mind
# ---------------------------------------------------------------------------

async def note_statements(fx):
    a = await fx.fact("turn-a", OWNER, "s-1", "My weekly note should end with a short table.",
                      "end with a short table", predicate="weekly_note_format")
    fx.shift(hours=2)
    b = await fx.fact("turn-b", OWNER, "s-2", "My weekly note should also name the next step in the retry row.",
                      "name the next step in the retry row", predicate="weekly_note_format")
    return a, b


async def test_an_addition_is_recorded_compatible_and_never_asked(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch, relation="adds")
    a, b = await note_statements(fx)
    first = await fx.mind.consolidate()
    assert first["counts"].get("compatible") == 1 and not first["counts"].get("contradictions")
    relation_calls = [(m, c) for m, c in fx.router.calls if c["task"] == "mind_claim_relation"]
    assert len(relation_calls) == 1
    shown = relation_calls[0][0][-1]["content"]
    assert "end with a short table" in shown and "name the next step in the retry row" in shown
    assert fx.mind.consolidation.relation(a, b) == "adds"
    assert [c for c in fx.mind.concerns.open(limit=50) if c.kind == "question"] == []
    await fx.mind.tick(force=True)
    assert fx.messages("contradiction") == []
    fx.shift(days=1)
    await fx.mind.consolidate()                                   # decided once: never asked again
    assert fx.router.tasks().count("mind_claim_relation") == 1
    with closing(fx.ledger._connect()) as conn:
        logged = conn.execute("SELECT action, stage, reason FROM learning_log WHERE stage='contradiction_check'"
                              ).fetchall()
    assert [tuple(row) for row in logged] == [("labelled", "contradiction_check", "adds")]


async def test_an_unsure_decision_waits_for_another_night_and_a_replacement_is_settled_by_the_mind(tmp_path,
                                                                                                   monkeypatch):
    fx = night(tmp_path, monkeypatch, relation="unsure")
    a = await fx.fact("turn-a", OWNER, "telegram-1", "My office is room 4.", "room 4")
    fx.shift(days=1)
    b = await fx.fact("turn-b", OWNER, "discord-2", "My office is room 7.", "room 7")
    unsure = await fx.mind.consolidate()
    assert not unsure["counts"].get("contradictions") and fx.mind.consolidation.relation(a, b) is None
    assert [c for c in fx.mind.concerns.open(limit=50) if c.kind == "question"] == []
    fx.router.answers["mind_claim_relation"] = {"relation": "replaces"}
    fx.shift(days=1)
    replaced = await fx.mind.consolidate()
    assert replaced["counts"]["contradictions"] == 1 and fx.mind.consolidation.relation(a, b) == "replaces"
    kept = {row["id"]: row["retracted_by"] for row in fx.claims()}
    assert kept[a].startswith("note:") and kept[b] is None             # the newer one is kept, nothing is asked
    assert [c for c in fx.mind.concerns.open(limit=50) if c.kind == "question"] == []


# ---------------------------------------------------------------------------
# Free words never grant a wish or widen permissions
# ---------------------------------------------------------------------------

async def deliver(fx):
    sent = []
    for payload in await fx.mind.outbox_ready():
        fx.mind.outbox.sending(payload["id"], target=f"telegram:{payload['recipient']}-handle")
        fx.mind.outbox.sent(payload["id"])
        sent.append(payload)
    return sent


async def test_words_never_grant_a_wish_or_widen_permissions(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch)
    wish = asked(fx, "W9K", "A wish: reading scanned tables", type_="wish", kind="note")
    change = asked(fx, "P3Q", "Let Sam be messaged without asking first", type_="people_change", kind="note")
    fx.shift(minutes=1)
    await fx.mind.owner_turn("Yes, and yes to everything else too.", turn_id="t-1")
    assert fx.store.get(wish.id).status == "asked" and fx.store.get(change.id).status == "asked"
    await owner_answers(fx.mind, "W9K", "sure, if it is free")
    assert fx.store.get(wish.id).status == "asked"
    # Their own plain yes to the ask's own message settles it; after the typed grant a wish's yes gives only its
    # steps.
    await owner_answers(fx.mind, "W9K", "sure, sounds good")
    assert fx.store.get(wish.id).status != "asked" and fx.store.get(change.id).status == "asked"


# ---------------------------------------------------------------------------
# Delivery: once a day, held items listed only when delivered
# ---------------------------------------------------------------------------

async def test_one_digest_a_day_across_restarts_retries_and_two_bodies(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from colonymind.api.routers import mind as mind_router

    fx = night(tmp_path, monkeypatch)
    asked(fx, "K7F", "Re-run the archive export for the board figures")
    first = await the_digest(fx)
    assert first is not None
    fx.restart()
    fx.mind.digest_hour, fx.mind.quiet = 0, None
    assert await fx.mind._digest(fx.now) is None                    # a restart builds no second one
    app = FastAPI()
    app.include_router(mind_router.router)
    mind_router.set_mind(fx.mind)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            ready = (await client.get("/v1/mind/outbox")).json()
            assert [item["id"] for item in ready if item["type"] == "digest"] == [first.id]
            one = await client.post(f"/v1/mind/outbox/{first.id}/sending", json={"target": "telegram:owner-handle"})
            other = await client.post(f"/v1/mind/outbox/{first.id}/sending", json={"target": "sms:owner"})
            assert one.status_code == 200 and other.status_code == 409   # a second body or channel never sends it
            fx.restart()                                             # the body died mid-send
            mind_router.set_mind(fx.mind)
            assert fx.store.get(first.id).status == "uncertain"
            again = await client.post(f"/v1/mind/outbox/{first.id}/sending", json={"target": "telegram:owner-handle"})
            assert again.status_code == 409                          # never resent after a restart
            fx.mind.digest_hour, fx.mind.quiet = 0, None
            assert await fx.mind._digest(fx.now) is None
    finally:
        mind_router.set_mind(None)
    fx.shift(days=1)
    nxt = await the_digest(fx)
    assert nxt is not None and nxt.id != first.id
    assert len(fx.messages("digest")) == 2


async def test_held_items_are_listed_only_once_the_digest_is_delivered(tmp_path, monkeypatch):
    from test_owner_gate_mind import Fx, M
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fx = Fx(tmp_path)
    try:
        fx.notice("task_outcome", "done: file the tax return", "Done: file the tax return.", "k1")
        await fx.pull()
        fx.now += 60 * M
        fx.notice("stale_task", "Stale task: file the tax return", "The tax return task is stale.", "k2")
        assert await fx.pull() == []                                   # held for the digest
        fx.mind.digest_hour = 8
        fx.now += 60 * M
        first = fx.store.get(await fx.mind._digest(fx.now))
        assert "The tax return task is stale." in first.context["text"]
        fx.now += 24 * 60 * M                                          # never delivered: it expired unsent
        assert await fx.pull() == [] and fx.store.get(first.id).status == "expired"
        second = fx.store.get(await fx.mind._digest(fx.now))
        assert "The tax return task is stale." in second.context["text"]
        assert [p["id"] for p in await fx.pull()] == [second.id]
        fx.now += 24 * 60 * M
        third = await fx.mind._digest(fx.now)
        assert third is None or "The tax return task is stale." not in fx.store.get(third).context["text"]
    finally:
        fx.store.close()


# ---------------------------------------------------------------------------
# Asks: the owner's plain words answer; a hedge, a question or a later does not
# ---------------------------------------------------------------------------

async def test_an_ask_is_answered_in_the_owners_own_plain_words(tmp_path, monkeypatch):
    fx = night(tmp_path, monkeypatch)
    ask = asked(fx, "K7F", "Re-run the archive export for the board figures")
    for message in ("ok do it?", "not sure", "yes, but only the first part", "later"):
        await owner_answers(fx.mind, "K7F", message)
    assert fx.store.get(ask.id).status == "asked"
    assert (await owner_answers(fx.mind, "K7F", "please yes")).status == "approved"


async def test_a_title_the_store_cut_is_never_shown_as_it_is(tmp_path, monkeypatch):
    """A title stored cut at a store's cap (here 160 characters) is summarised or pointed to, never shown cut."""
    long = ("Overdue: send Dana the signed lease, the updated insurance certificate and the landlord's letter about "
            "the parking space before the end of the month, as promised")
    fx = night(tmp_path, monkeypatch,
               mind_plain_summary={"sentence": "Send Dana the signed lease and the insurance papers."})
    acted(fx, "cut", long[:160], outcome="done", status="done")
    assert plain.clipped(long[:160]) and not plain.clipped("Check the venue hours")
    text = (await the_digest(fx)).context["text"]
    assert "Done: Send Dana the signed lease and the insurance papers." in text
    assert long[:150] not in text
    items_are_whole(text)
