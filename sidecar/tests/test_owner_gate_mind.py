"""The owner gate in the served mind (``mind.faculties.owner_gate``): every owner message the body pulls passes
it, a held message reaches the digest once, an outside alert asks through ``POST /v1/mind/owner-gate``, the
audit reads back through ``GET /v1/mind/owner-gate``, and with the flag off nothing changes. It ships on."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from colonymind.api.routers import mind as mind_router
from colonymind.initiatives.store import InitiativeStore
from colonymind.mind import Mind, outreach
from colonymind.mind.tick import DEFAULT_FACULTIES

OWNER = "p-01"
T0 = (datetime.now(timezone.utc) + timedelta(days=2)).replace(hour=12, minute=0, second=0, microsecond=0)
M = timedelta(minutes=1)


class Fx:
    def __init__(self, tmp_path, *, gate=True, config=None):
        self.now = T0
        self.store = InitiativeStore(state_dir=tmp_path)
        config = {"autonomy": "standard", "quiet_hours": "", **(config or {})}
        config["faculties"] = {"consolidation": False, "owner_gate": gate, **(config.get("faculties") or {})}
        self.mind = Mind(config=config, store=self.store, state_dir=tmp_path, owner_id=OWNER,
                         clock=lambda: self.now, backups=False)
        self.mind.digest_hour = 25
        self.sent = []

    async def pull(self):
        ready = await self.mind.outbox_ready()
        for payload in ready:
            self.mind.outbox.sending(payload["id"], target=f"capture:{payload['recipient']}")
            self.mind.outbox.sent(payload["id"])
            self.sent.append(payload)
        return ready

    def notice(self, type, title, text, key):
        return self.mind.outbox.notice(type=type, title=title, text=text, dedup_key=key)

    def owner_spoke(self, at):
        self.mind.mind_state.set(outreach.OWNER_TURN_KEY, text=at.isoformat(), now=self.now)


@pytest.fixture
def make(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    made = []

    def build(**kwargs):
        directory = tmp_path / f"fx-{len(made)}"
        directory.mkdir()
        made.append(Fx(directory, **kwargs))
        return made[-1]

    yield build
    for fx in made:
        fx.store.close()


def test_the_flag_ships_on():
    """On unless shown harmful: the owner gate ships on (switched off, the outbox is what it was, below)."""
    assert DEFAULT_FACULTIES["owner_gate"] is True
    from colonymind.config import DEFAULTS
    assert DEFAULTS["mind"]["faculties"]["owner_gate"] is True


async def test_flag_off_the_outbox_is_what_it_was(make):
    fx = make(gate=False)
    fx.notice("task_outcome", "done: file the tax return", "Done: file the tax return.", "k1")
    fx.notice("health_notice", "Health: file the tax return", "The tax return check is failing.", "k2")
    fx.owner_spoke(T0 - M)
    assert len(await fx.pull()) == 2
    assert fx.mind.owner_gate.ledger.recent(since=T0 - M) == []


async def test_an_ask_for_the_owners_yes_is_never_held(make):
    """An ask waits on the owner's yes, so holding the notice that carries it would hold the agent back: it is a
    question only the owner can answer and passes as prompted, mid-conversation and on a topic the owner was just
    sent. Anything else the mind raises unprompted still waits there."""
    fx = make()
    first = fx.notice("task_outcome", "done: clean the build folders", "Done: clean the build folders.", "k1")
    assert [p["id"] for p in await fx.pull()] == [first.id]
    fx.now += M
    fx.owner_spoke(fx.now - M)
    ask = fx.notice("ask_notice", "1 open ask(s)", "Needs your OK: clean the build folders (terminal).", "k2")
    later = fx.notice("stale_task", "Stale task: clean the build folders", "The build folder task is stale.", "k3")
    assert [p["id"] for p in await fx.pull()] == [ask.id]
    verdicts = {row["type"]: (row["decision"], row["stage"]) for row in fx.mind.owner_gate.ledger.recent(since=T0)}
    assert verdicts["ask_notice"] == ("send", "prompted") and verdicts["stale_task"][0] != "send"
    assert fx.store.get(later.id).status != "sent"


async def test_a_second_message_on_a_topic_waits_for_the_digest_across_kinds(make):
    fx = make()
    first = fx.notice("task_outcome", "done: file the tax return", "Done: file the tax return.", "k1")
    assert [p["id"] for p in await fx.pull()] == [first.id]
    fx.now += 60 * M
    second = fx.notice("stale_task", "Stale task: file the tax return", "The tax return task is stale.", "k2")
    assert await fx.pull() == []
    row = fx.store.get(second.id)
    assert row.status == "expired"
    history = [item.action for item in fx.store.get_history(second.id, limit=20)]
    assert "gated" in history
    assert await fx.pull() == []            # settled once: never offered again
    fx.mind.digest_hour = 8
    fx.now += 60 * M
    digest_id = await fx.mind._digest(fx.now)
    text = fx.store.get(digest_id).context["text"]
    assert "- Held for this digest: The tax return task is stale." in text
    assert [p["id"] for p in await fx.pull()] == [digest_id]         # delivered: listed
    fx.now += 24 * 60 * M
    later = await fx.mind._digest(fx.now)
    assert later is None or "Held for this digest" not in fx.store.get(later).context["text"]


async def test_a_review_that_fails_holds_its_message_and_the_pull_goes_on(make, monkeypatch):
    """Every check that decides whether a message may leave fails closed for that message alone: a gate review
    that raises holds its message on this pull (not sent, not settled) and never takes the rest of the pull
    down; the next pull asks the gate again."""
    fx = make()
    first = fx.notice("task_outcome", "done: file the tax return", "Done: file the tax return.", "k1")
    second = fx.notice("health_notice", "Health: backups", "The backup check is failing.", "k2")
    review = fx.mind.owner_gate.review

    async def failing(message, **kwargs):
        if message.ref == first.id:
            raise RuntimeError("the gate's store is unavailable")
        return await review(message, **kwargs)
    monkeypatch.setattr(fx.mind.owner_gate, "review", failing)
    assert [p["id"] for p in await fx.pull()] == [second.id]
    assert fx.store.get(first.id).status == "approved"
    monkeypatch.setattr(fx.mind.owner_gate, "review", review)
    fx.now += 11 * M                        # past the spacing after the message that went
    assert [p["id"] for p in await fx.pull()] == [first.id]


async def test_mid_conversation_a_notice_waits_then_goes_and_a_send_is_decided_once(make):
    fx = make()
    fx.owner_spoke(T0 - 2 * M)
    row = fx.notice("health_notice", "Health: backups", "The backup check is failing.", "k1")
    assert await fx.pull() == [] and await fx.pull() == []
    waits = [e for e in fx.mind.owner_gate.ledger.recent(since=T0 - M) if e["decision"] == "wait"]
    assert len(waits) == 1                 # the second pull inside the wait decides nothing new
    fx.now += 9 * M
    assert [p["id"] for p in await fx.pull()] == [row.id]
    assert fx.store.get(row.id).status == "sent"
    decided = [e["decision"] for e in fx.mind.owner_gate.ledger.recent(since=T0 - M)]
    assert decided.count("send") == 1


async def test_a_requested_answer_is_never_held(make):
    fx = make()
    fx.owner_spoke(T0 - M)
    row, _ = fx.store.create_intention(
        kind="message", type="outreach_answer", title="the detail you asked for", drive="social", cls="owner",
        decision="act", decision_reason="requested", status="approved", dedup_key="ans-1", recipient=OWNER,
        context={"text": "Here is the detail.", "topic": "kites"}, hermes_kind="message", created_at=fx.now,
        expires_at=fx.now + 60 * M)
    assert [p["id"] for p in await fx.pull()] == [row.id]


async def test_the_route_answers_an_outside_alert_and_the_audit_reads_back(make):
    fx = make()
    app = FastAPI()
    app.include_router(mind_router.router)
    mind_router.set_mind(fx.mind)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            alert = {"topic": "nas disk", "text": "NAS disk 3 reports errors.", "urgency": "high", "source": "ops"}
            decisions = []
            for index in range(5):
                response = await client.post("/v1/mind/owner-gate", json={**alert, "id": f"a{index}"})
                assert response.status_code == 200, response.text
                decisions.append(response.json()["decision"])
                fx.now += 4 * M
            assert decisions == ["send", "drop", "drop", "drop", "drop"]
            low = await client.post("/v1/mind/owner-gate", json={"topic": "scrub", "text": "Scrub done.",
                                                                  "urgency": "low", "source": "ops"})
            assert low.json()["decision"] == "digest"
            bad = await client.post("/v1/mind/owner-gate", json={"topic": "x", "text": "y", "urgency": "panic"})
            assert bad.status_code == 422
            audit = (await client.get("/v1/mind/owner-gate", params={"since_hours": 2})).json()
            assert audit["enabled"] is True and audit["counts"]["decision"] == {"send": 1, "drop": 4, "digest": 1}
            assert audit["decisions"][0]["inputs"]["message"]["text"] == "Scrub done."
            assert audit["decisions"][0]["scores"]["fast_source"] == "prior"
    finally:
        mind_router.set_mind(None)


async def test_the_route_with_the_flag_off_says_send_and_records_nothing(make):
    fx = make(gate=False)
    app = FastAPI()
    app.include_router(mind_router.router)
    mind_router.set_mind(fx.mind)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            response = await client.post("/v1/mind/owner-gate", json={"topic": "t", "text": "x", "urgency": "low"})
            assert response.json()["decision"] == "send" and response.json()["gate"] == "off"
            audit = (await client.get("/v1/mind/owner-gate")).json()
            assert audit["enabled"] is False and audit["decisions"] == []
    finally:
        mind_router.set_mind(None)


async def test_concurrent_pulls_reuse_one_approval_while_an_outside_alert_awaits_its_judge(make):
    import asyncio

    fx = make()
    release = asyncio.Event()

    async def slow_judge(system, prompt):
        await release.wait()
        return {"choice": "digest", "reason": "it can wait"}

    fx.mind.owner_gate.judge = slow_judge
    alert = asyncio.create_task(fx.mind.gate_external({"topic": "printer", "text": "Printer offline.",
                                                       "urgency": "normal", "source": "ops"}))
    await asyncio.sleep(0)
    row = fx.notice("health_notice", "Health: backups", "The backup check is failing.", "k1")
    first, second = (asyncio.create_task(fx.mind._gate_passes(fx.store.get(row.id), fx.now)) for _ in range(2))
    for _ in range(5):
        await asyncio.sleep(0)
    release.set()
    assert (await alert)["decision"] == "digest"
    assert [await first, await second] == [True, True]
    decided = [e["decision"] for e in fx.mind.owner_gate.ledger.recent(since=T0 - M) if e["ref"] == row.id]
    assert decided == ["send"] and fx.store.get(row.id).status == "approved"


async def test_an_outside_alert_whose_send_answer_was_lost_is_sent_on_retry_under_its_id(make):
    fx = make()
    alert = {"topic": "ups", "text": "UPS on battery.", "urgency": "high", "source": "ops", "id": "ups-1"}
    lost = await fx.mind.gate_external(alert)
    fx.now += 2 * M
    retry = await fx.mind.gate_external(alert)
    assert (lost["decision"], retry["decision"]) == ("send", "send")
    assert retry["replayed"] is True and retry["id"] == lost["id"]
    assert [e["decision"] for e in fx.mind.owner_gate.ledger.recent(since=T0 - M)] == ["send"]


async def test_held_items_past_what_one_digest_lists_stay_held_for_the_next(make):
    fx = make()
    for index in range(13):
        verdict = await fx.mind.gate_external({"topic": f"job {index}", "text": f"Job {index} finished.",
                                               "urgency": "low", "source": "ops", "id": f"job-{index}"})
        assert verdict["decision"] == "digest"
        fx.now += M
    fx.mind.digest_hour = 8
    fx.now += 60 * M
    text = fx.store.get(await fx.mind._digest(fx.now)).context["text"]
    listed = [line for line in text.splitlines() if line.startswith("- Held for this digest: Job ")]
    assert len(listed) == 12 and "Job 12 finished." not in text
    assert "- 1 more held item(s) come in the next digest." in text
    [left] = fx.mind.owner_gate.ledger.held()
    assert left["topic"] == "job 12"
    await fx.pull()                                                   # the first digest is delivered
    fx.now += 24 * 60 * M
    later = fx.store.get(await fx.mind._digest(fx.now)).context["text"]
    assert "Job 12 finished." in later and fx.mind.owner_gate.ledger.held() == []


async def test_a_hold_made_while_the_gate_was_on_reaches_the_digest_after_it_is_turned_off(make):
    fx = make()
    held = await fx.mind.gate_external({"topic": "scrub", "text": "The weekly scrub finished.", "urgency": "low",
                                        "source": "ops", "id": "scrub-1"})
    assert held["decision"] == "digest"
    off = Fx(fx.mind.state_dir, gate=False)      # the same workspace, restarted with the flag off
    try:
        off.now = fx.now + 60 * M
        off.mind.digest_hour = 8
        text = off.store.get(await off.mind._digest(off.now)).context["text"]
        assert "- Held for this digest: The weekly scrub finished." in text
        assert off.mind.owner_gate.ledger.held() == []
    finally:
        off.store.close()


def _owner_row(fx, type, title, text, key, *, expires_in):
    row, _ = fx.store.create_intention(
        kind="message", type=type, title=title, drive="upkeep", cls="owner", decision="act",
        decision_reason="standard", status="approved", dedup_key=key, recipient=OWNER, context={"text": text},
        hermes_kind="message", created_at=fx.now, expires_at=fx.now + expires_in)
    return row


async def test_a_notice_that_expires_while_the_gate_keeps_it_waiting_is_held_for_the_digest(make):
    fx = make()
    fx.owner_spoke(T0 - M)
    row = _owner_row(fx, "health_notice", "Health: backups", "The backup check is failing.", "k1", expires_in=5 * M)
    assert await fx.pull() == []
    fx.now += 9 * M
    assert await fx.pull() == []
    assert fx.store.get(row.id).status == "expired"
    [held] = fx.mind.owner_gate.ledger.held()
    assert held["ref"] == row.id and held["stage"] == "expiry"
    fx.mind.digest_hour = 8
    fx.now += 60 * M
    assert "The backup check is failing." in fx.store.get(await fx.mind._digest(fx.now)).context["text"]


async def test_a_chatty_owner_defers_a_notice_only_up_to_the_wait_bound(make):
    fx = make()
    row = _owner_row(fx, "health_notice", "Health: backups", "The backup check is failing.", "k1",
                     expires_in=48 * 60 * M)
    for _ in range(40):                 # the owner keeps talking, a turn every five minutes
        fx.owner_spoke(fx.now - M)
        if await fx.pull():
            break
        fx.now += 5 * M
    assert [p["id"] for p in fx.sent] == [row.id]
    assert fx.now - T0 <= fx.mind.owner_gate.settings.max_wait + 5 * M


async def test_a_notice_whose_digest_hold_cannot_be_recorded_goes_out_instead_of_expiring(make):
    fx = make()
    first = fx.notice("task_outcome", "done: file the tax return", "Done: file the tax return.", "k1")
    assert [p["id"] for p in await fx.pull()] == [first.id]
    fx.now += 60 * M
    fx.mind.owner_gate.ledger.conn.execute(
        "CREATE TRIGGER refuse BEFORE INSERT ON owner_gate BEGIN SELECT RAISE(ABORT, 'disk full'); END")
    second = fx.notice("stale_task", "Stale task: file the tax return", "The tax return task is stale.", "k2")
    assert [p["id"] for p in await fx.pull()] == [second.id]
    assert fx.store.get(second.id).status == "sent"


async def test_an_expiring_wait_whose_hold_cannot_be_recorded_is_not_expired_into_nothing(make):
    fx = make()
    fx.owner_spoke(T0 - M)
    row = _owner_row(fx, "health_notice", "Health: backups", "The backup check is failing.", "k1", expires_in=5 * M)
    assert await fx.pull() == []
    fx.mind.owner_gate.ledger.conn.execute(
        "CREATE TRIGGER refuse BEFORE INSERT ON owner_gate BEGIN SELECT RAISE(ABORT, 'disk full'); END")
    fx.now += 9 * M
    await fx.pull()
    assert fx.store.get(row.id).status != "expired"


async def _expiry_during_review(fx, answer, *, waited, meanwhile=None):
    """A review of an owner notice blocked on its second opinion while the notice's window closes and the
    expiry sweep runs (or ``meanwhile(row)`` settles it); then the model answers ``answer``. Returns the
    review's pass and the row."""
    import asyncio
    from dataclasses import replace

    release = asyncio.Event()

    async def slow_judge(system, prompt):
        await release.wait()
        return {"choice": answer, "reason": f"judged {answer}"}

    gate = fx.mind.owner_gate
    gate.judge, gate.settings = slow_judge, replace(gate.settings, confidence=0.95)   # the judge is asked
    if waited:
        fx.owner_spoke(T0 - M)                  # mid-conversation: it waits until T0+9
    row = _owner_row(fx, "health_notice", "Health: backups", "The backup check is failing.", "k1",
                     expires_in=12 * M if waited else 5 * M)
    if waited:
        assert await fx.pull() == []
        fx.now += 10 * M
    review = asyncio.create_task(fx.mind._gate_passes(fx.store.get(row.id), fx.now))
    for _ in range(5):
        await asyncio.sleep(0)
    fx.now += 6 * M                             # past the window while the model thinks
    if meanwhile is None:
        fx.mind._expire_messages(fx.now)
    else:
        meanwhile(row)
    release.set()
    return await review, fx.store.get(row.id)


@pytest.mark.parametrize("waited", [True, False])
async def test_expiry_during_a_review_that_holds_the_notice_leaves_one_hold(make, waited):
    fx = make()
    passed, row = await _expiry_during_review(fx, "digest", waited=waited)
    assert passed is False and row.status == "expired"
    held = [entry for entry in fx.mind.owner_gate.ledger.held() if entry["ref"] == row.id]
    assert len(held) == 1 and held[0]["stage"] == "second_opinion"
    assert sum(1 for e in fx.mind.owner_gate.ledger.recent(since=T0 - M) if e["decision"] == "digest") == 1


@pytest.mark.parametrize("waited", [True, False])
async def test_expiry_during_a_review_that_sends_the_notice_neither_holds_nor_expires_it(make, waited):
    fx = make()
    passed, row = await _expiry_during_review(fx, "send_now", waited=waited)
    assert passed is True and row.status == "approved"
    assert fx.mind.owner_gate.ledger.held() == []


@pytest.mark.parametrize("answer", ["send_now", "digest"])
async def test_a_message_settled_elsewhere_while_the_gate_decides_is_not_brought_back(make, answer):
    fx = make()
    passed, row = await _expiry_during_review(
        fx, answer, waited=False, meanwhile=lambda row: fx.mind.outcomes.record(
            row.id, status="cancelled", summary="the owner answered it", by="mind", implicit_verdict=False))
    assert passed is False and row.status == "cancelled"
