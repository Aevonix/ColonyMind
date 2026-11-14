"""The owner gate's review, stage by stage: prompted, dedup across kinds, reachability, the decision (Instinct's
``owner_send`` point: the prior as its rules tier, the fast tier in shadow, the second opinion as its deliberate
tier on low confidence or at night) and the audit of every verdict."""

from __future__ import annotations

from secrets import token_hex

import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from colonymind.instinct import REGISTRY
from colonymind.instinct.calibration import Record
from colonymind.instinct.log import Store
from colonymind.instinct.providers import Answer
from colonymind.instinct.tiers import Instinct
from colonymind.mind import gate
from colonymind.mind.gate import Message, OwnerGate, Settings

T0 = datetime(2031, 3, 4, 12, 0, tzinfo=timezone.utc)
M = timedelta(minutes=1)


class Fx:
    def __init__(self, *, instinct=None, judge=None, night=False, settings=None):
        self.now = T0
        self.turn = None
        self.is_night = night
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.gate = OwnerGate(self.conn, settings=settings or Settings(), clock=lambda: self.now,
                              night=lambda now: self.is_night, last_owner_turn=lambda: self.turn,
                              instinct=instinct or Instinct(), judge=judge, quiet_hours="22:00-07:00")

    async def review(self, **kwargs):
        fields = {"type": "health_notice", "topic": "backup", "text": "The backup check is failing.", **kwargs}
        return await self.gate.review(Message(**fields))


class Provider:
    """A decision server answering the ``owner_send`` choice with ``probabilities`` (None: a failure)."""

    kind, model, private, max_state_chars, batch, timeout_ms = "local-decision.v1", "", True, 1200, False, 250
    types = frozenset({"yes_no", "choice"})

    def __init__(self, probabilities=None):
        self.id, self.probabilities, self.calls = "local", probabilities, []

    async def answer(self, state, questions):
        self.calls.append((state, dict(questions)))
        if self.probabilities is None:
            raise RuntimeError("boom")
        return Answer({name: dict(self.probabilities) for name in questions}, revision="rev-1")


def Decider(probabilities=None, *, live=True, store=None):
    """Instinct with the ``owner_send`` fast tier ``live`` (or shadow) on a record for this provider."""
    point = REGISTRY[gate.OWNER_SEND]
    record = Record(point=point.name, version=point.version, tier="fast", provider_kind="local-decision.v1",
                    revision="rev-1", temperature=1.0, bars={"send_now": 0.8, "digest": 0.8, "drop": 0.95},
                    source="test")
    engine = Instinct(providers={"local": Provider(probabilities)}, fast="local", records=(record,), store=store)
    engine.set_state(gate.OWNER_SEND, "fast", "live" if live else "shadow", source="test")
    return engine


class Judge:
    def __init__(self, choice="digest"):
        self.choice, self.prompts = choice, []

    async def __call__(self, system, prompt):
        self.prompts.append(prompt)
        return None if self.choice is None else {"choice": self.choice, "reason": f"judged {self.choice}"}


async def test_a_mind_message_keeps_the_existing_path_and_every_verdict_is_audited_with_inputs_and_scores():
    fx = Fx()
    verdict = await fx.review(ref="row-1")
    assert (verdict.decision, verdict.stage) == ("send", "fast_decision")
    assert verdict.scores["fast_source"] == "prior" and verdict.scores["fast"]["send_now"] == 0.9
    [entry] = fx.gate.ledger.recent(since=T0 - M)
    assert entry["decision"] == "send" and entry["ref"] == "row-1" and entry["scores"]["confidence"] == 0.9
    assert entry["inputs"]["message"]["text"] == "The backup check is failing."
    assert entry["inputs"]["facts"]["sent_24h"] == 0 and entry["inputs"]["facts"]["night"] is False


async def test_a_prompted_message_always_passes_and_later_messages_see_it():
    fx = Fx()
    fx.turn = T0 - M      # mid-conversation holds nothing the owner asked for
    first = await fx.review(type="commitment_due_soon", topic="tax return", text="Due soon: tax return", ref="r1")
    assert (first.decision, first.stage) == ("send", "prompted")
    fx.turn, fx.now = None, T0 + 30 * M
    second = await fx.review(type="outreach_loop", topic="tax return", text="Want a hand with the tax return?",
                             ref="r2")
    assert (second.decision, second.stage) == ("digest", "dedup")


async def test_dedup_is_per_topic_across_kinds_and_a_more_urgent_message_passes():
    fx = Fx()
    assert (await fx.review(type="outreach_finding", topic="solar panels", text="A finding on solar panels",
                            ref="a")).decision == "send"
    fx.now += 60 * M
    held = await fx.review(type="stale_task", topic="solar panels", text="Stale task: solar panels", urgency="low",
                           ref="b")
    assert (held.decision, held.stage) == ("digest", "dedup")
    fx.now += 60 * M
    urgent = await fx.review(type="task_outcome", topic="solar panels", text="I could not finish: solar panels",
                             urgency="high", ref="c")
    assert urgent.decision == "send"
    fx.now += 13 * 60 * M       # outside the topic window
    assert (await fx.review(type="stale_task", topic="solar panels", text="Stale again", urgency="low",
                            ref="d")).decision == "send"


async def test_the_same_words_on_the_same_topic_are_dropped_inside_the_repeat_window():
    fx = Fx()
    await fx.review(ref="a", topic="disk", text="Disk usage is at 91%.")
    fx.now += 13 * 60 * M
    again = await fx.review(ref="b", topic="Disk", text="Disk usage is  at 91%.")
    assert (again.decision, again.stage) == ("drop", "dedup")


async def test_the_same_words_on_another_topic_or_more_urgent_are_not_a_repeat():
    fx = Fx()
    lock = await fx.review(source="ops", ref="a", topic="door lock", text="Battery empty", urgency="high")
    fx.now += 30 * M
    pump = await fx.review(source="medical", ref="b", topic="infusion pump", text="Battery empty",
                           urgency="critical")
    assert (lock.decision, pump.decision) == ("send", "send")
    fx.now += 30 * M
    worse = await fx.review(source="ops", ref="c", topic="door lock", text="Battery empty", urgency="critical")
    assert worse.decision == "send" and worse.stage != "dedup"
    fx.now += 30 * M
    same = await fx.review(source="ops", ref="d", topic="door lock", text="Battery empty", urgency="critical")
    assert (same.decision, same.stage) == ("drop", "dedup")


async def test_mid_conversation_and_just_messaged_wait_but_critical_does_not():
    fx = Fx()
    fx.turn = T0 - 3 * M
    verdict = await fx.review(ref="a")
    assert (verdict.decision, verdict.stage) == ("wait", "reachability") and verdict.until == T0 + 7 * M
    assert (await fx.review(ref="c", topic="fire", text="Smoke alarm", urgency="critical",
                            source="ops")).decision == "send"
    fx.turn, fx.now = None, T0 + 4 * M
    spaced = await fx.review(ref="d", topic="printer", text="Printer offline")
    assert spaced.decision == "wait" and spaced.reason == "the owner was just messaged"
    assert spaced.until == T0 + 10 * M


async def test_an_outside_alert_uses_its_urgency_day_and_night():
    day = Fx()
    assert (await day.review(source="ops", ref="1", urgency="low", topic="a", text="x")).decision == "digest"
    assert (await day.review(source="ops", ref="2", urgency="high", topic="b", text="y")).decision == "send"
    night = Fx(night=True)
    assert (await night.review(source="ops", ref="1", urgency="high", topic="a", text="x")).decision == "digest"
    unconfirmed = await night.review(source="ops", ref="2", urgency="critical", topic="b", text="y")
    assert unconfirmed.decision == "send" and "unconfirmed" in unconfirmed.reason
    assert unconfirmed.scores["second_opinion"]["reason"] == "unavailable"


async def test_a_night_send_is_always_reconfirmed_by_the_second_opinion():
    judge = Judge("digest")
    fx = Fx(night=True, judge=judge)
    verdict = await fx.review(source="ops", ref="1", urgency="critical", topic="freezer", text="Freezer warm")
    assert (verdict.decision, verdict.stage) == ("digest", "second_opinion") and len(judge.prompts) == 1
    assert "quiet hours: the owner is asleep" in judge.prompts[0] and "Freezer warm" in judge.prompts[0]
    judge.choice = "send_now"
    fx.now += 60 * M
    assert (await fx.review(source="ops", ref="2", urgency="critical", topic="flood", text="Water on the floor")
            ).decision == "send"


async def test_the_second_opinion_is_asked_only_when_confidence_is_low():
    judge = Judge("drop")
    fx = Fx(judge=judge)
    assert (await fx.review(ref="1")).decision == "send" and judge.prompts == []     # 0.9: confident
    fx.now += 60 * M
    verdict = await fx.review(source="ops", ref="2", urgency="normal", topic="nas", text="NAS scrub finished")
    assert (verdict.decision, verdict.stage) == ("drop", "second_opinion") and len(judge.prompts) == 1
    judge.choice = None           # no answer: the fast decision stands in the day
    fx.now += 60 * M
    kept = await fx.review(source="ops", ref="3", urgency="normal", topic="ups", text="UPS on battery")
    assert kept.decision == "send" and kept.scores["second_opinion"]["reason"] == "unavailable"


ALERT = {"source": "ops", "ref": "1", "urgency": "normal", "topic": "nas", "text": "NAS scrub finished"}


async def test_where_the_prior_is_confident_the_fast_tier_is_never_asked():
    engine = Decider({"send_now": 0.03, "digest": 0.95, "drop": 0.02})
    fx = Fx(instinct=engine)
    verdict = await fx.review(ref="1", type="outreach_finding", topic="kites", text="Kite news")
    assert (verdict.decision, verdict.scores["fast_source"]) == ("send", "prior")
    assert engine.provider_for("fast").calls == []


async def test_a_live_fast_tier_may_hold_where_the_prior_is_unsure_and_never_sends():
    engine = Decider({"send_now": 0.03, "digest": 0.95, "drop": 0.02})
    judge = Judge("send_now")
    fx = Fx(instinct=engine, judge=judge)
    verdict = await fx.review(**ALERT)
    assert (verdict.decision, verdict.stage, verdict.scores["fast_source"]) == ("digest", "fast_decision", "fast")
    # The second opinion the prior would have asked is asked beside the hold, inside the review, only to compare
    # (test_instinct_compare.py): its send is never applied.
    assert len(judge.prompts) == 1
    [(state, questions)] = engine.provider_for("fast").calls
    assert "messages to the owner" in state and "A normal external about nas" in state
    # A send it may not settle leaves the prior's choice: the live tier stands in for the second opinion by day.
    sends = Decider({"send_now": 0.97, "digest": 0.02, "drop": 0.01})
    judge = Judge("send_now")
    fx = Fx(instinct=sends, judge=judge)
    verdict = await fx.review(**ALERT)
    assert (verdict.decision, verdict.stage) == ("send", "fast_decision") and judge.prompts == []


async def test_an_unsure_or_failing_live_fast_tier_leaves_the_priors_choice_and_asks_no_second_opinion_by_day():
    """The live tier stands in for the second opinion by day: whatever it leaves unsettled keeps the prior's
    choice, so promoting it never adds a judge call. At night a wake is still put to the second opinion."""
    for engine in (Decider({"send_now": 0.4, "digest": 0.35, "drop": 0.25}), Decider(None)):
        judge = Judge("drop")
        fx = Fx(instinct=engine, judge=judge)
        verdict = await fx.review(**ALERT)
        assert (verdict.decision, verdict.stage, verdict.scores["fast_source"]) == ("send", "fast_decision", "prior")
        assert judge.prompts == []
        night = Fx(instinct=engine, judge=judge, night=True)
        woke = await night.review(source="ops", ref="n1", urgency="critical", topic="flood", text="Water")
        assert (woke.decision, woke.stage) == ("drop", "second_opinion") and len(judge.prompts) == 1


async def test_the_shipped_shadow_fast_tier_is_asked_and_logged_and_never_decides(tmp_path):
    store = Store(tmp_path / "instinct.db")
    engine = Decider({"send_now": 0.01, "digest": 0.01, "drop": 0.98}, live=False, store=store)
    judge = Judge("send_now")
    fx = Fx(instinct=engine, judge=judge)
    verdict = await fx.review(**ALERT)
    await engine.drain()
    assert (verdict.decision, verdict.stage) == ("send", "second_opinion") and len(judge.prompts) == 1
    [row] = store.rows()
    assert row["point"] == "owner_send" and row["tiers"]["fast"]["state"] == "shadow"
    assert row["tiers"]["fast"]["label"] == "drop" and row["answered_by"] == "deliberate"
    assert verdict.scores["instinct"] == row["id"]


async def test_a_malformed_or_failing_provider_or_judge_never_reaches_the_caller():
    async def bad_judge(system, prompt):
        raise RuntimeError("boom")

    fx = Fx(instinct=Decider(None), judge=bad_judge)
    verdict = await fx.review(**ALERT)
    assert verdict.decision == "send" and verdict.scores["fast_source"] == "prior"
    assert verdict.scores["second_opinion"]["reason"] == "unavailable"
    weird = Fx(instinct=Decider({"send_now": 2.0, "digest": 0.0, "drop": 0.0}))
    assert (await weird.review(**ALERT)).decision == "send"
    assert (await Fx(instinct=Decider(None)).review(ref="1")).scores["fast_source"] == "prior"


async def test_router_judge_reads_only_the_typed_choice():
    class Router:
        supports_function_routing = True

        def __init__(self, content):
            self.content, self.contexts = content, []

        async def complete(self, messages, context=None, **_):
            self.contexts.append(context)
            return SimpleNamespace(content=self.content, usage={})

    router = Router('{"choice": "digest", "reason": "can wait"}')
    judge = gate.router_judge(router)
    assert await judge("s", "p") == {"choice": "digest", "reason": "can wait"}
    assert router.contexts[0]["task"] == gate.JUDGE_TASK and router.contexts[0]["response_schema"]
    assert await gate.router_judge(Router('{"choice": "wake"}'))("s", "p") is None
    assert await gate.router_judge(Router("not json"))("s", "p") is None
    assert await gate.router_judge(router, tokens_allowed=lambda: False)("s", "p") is None
    assert await gate.router_judge(None)("s", "p") is None


def test_settings_and_topics():
    settings = Settings.from_config({"topic_window_hours": 2, "confidence": 1.5, "second_opinion": False})
    assert settings.topic_window == timedelta(hours=2) and settings.confidence == 0.75
    assert settings.second_opinion is False
    assert gate.topic_of("Due soon: file the tax return") == "file the tax return"
    assert gate.topic_of("done: Finish the report", {"topic": "report"}) == "report"
    assert gate.topic_of("Who is Sam?") == "Who is Sam?"
    assert gate.urgency_of("task_outcome", "failed: x") == "high" and gate.urgency_of("stale_task") == "low"
    assert gate.same_topic("tidal energy", "Tidal Energy") and not gate.same_topic("tidal energy", "solar energy")
    assert gate.Message(type="digest", topic="", text="x").kind == "prompted"
    assert gate.Message(type="x", topic="", text="x", source="ops").kind == "external"


async def test_a_decision_the_engine_could_not_make_still_holds_a_night_wake():
    from colonymind.instinct.tiers import Decision

    class Broken:
        async def decide(self, point, fields, refs=None, *, deliberate=None, event=""):
            return Decision("dec-x", point, None, False, "fallback", reason="error")

    fx = Fx(instinct=Broken(), night=True)
    held = await fx.review(ref="1")
    assert (held.decision, held.reason) == ("digest", "night, and no second opinion confirmed a wake")
    critical = await fx.review(source="ops", ref="2", urgency="critical", topic="flood", text="Water")
    assert critical.decision == "send" and "unconfirmed" in critical.reason


def test_a_decision_model_decides_only_with_a_fast_provider_and_the_tier_live():
    assert Fx().gate.decision_model() is False
    assert Fx(instinct=Decider({"send_now": 0.1, "digest": 0.8, "drop": 0.1}, live=False)).gate.decision_model() is False
    assert Fx(instinct=Decider({"send_now": 0.1, "digest": 0.8, "drop": 0.1})).gate.decision_model() is True


@pytest.mark.parametrize("urgency", gate.URGENCIES)
def test_the_prior_is_a_distribution(urgency):
    for night in (False, True):
        values = gate.prior(Message(type="x", topic="t", text="x", urgency=urgency, source="ops"), night=night)
        assert set(values) == set(gate.CHOICES) and abs(sum(values.values()) - 1.0) < 1e-9


async def test_an_alert_asked_again_under_its_own_id_gets_its_verdict_again_and_another_id_is_a_repeat():
    fx = Fx()
    alert = {"source": "ops", "urgency": "high", "topic": "ups", "text": "UPS on battery"}
    first = await fx.gate.review(Message(type="alert:ops", ref="alert-7", **alert), replay=True)
    fx.now += 30 * M        # the answer was lost; the script asks again under the same id
    again = await fx.gate.review(Message(type="alert:ops", ref="alert-7", **alert), replay=True)
    assert (first.decision, again.decision) == ("send", "send") and again.replayed and again.id == first.id
    other = await fx.gate.review(Message(type="alert:ops", ref="alert-8", **alert), replay=True)
    assert (other.decision, other.stage) == ("drop", "dedup")
    fx.now += 25 * 60 * M   # past the repeat window the id is asked afresh, and is no repeat of itself
    fresh = await fx.gate.review(Message(type="alert:ops", ref="alert-7", **alert), replay=True)
    assert fresh.decision == "send" and not fresh.replayed and fresh.id != first.id


def _row(ident, type, title, text=None, context=None):
    return SimpleNamespace(id=ident, type=type, description=title,
                           context={"text": text or title, **(context or {})})


async def test_a_subject_before_a_colon_is_part_of_the_topic_unless_it_is_one_of_the_minds_own_labels():
    assert gate.topic_of("Alice: approved") == "Alice: approved"
    assert gate.topic_of("Bob: approved") != gate.topic_of("Alice: approved")
    for title in ("done: tax return", "failed: tax return", "Stale task: tax return", "Due soon: tax return",
                  "Overdue: tax return", "Goal stalled: tax return"):
        assert gate.topic_of(title) == "tax return", title
    fx = Fx()
    first = await fx.gate.review(gate.message_of_row(_row("r1", "task_outcome", "Alice: approved")))
    fx.now += 60 * M
    second = await fx.gate.review(gate.message_of_row(_row("r2", "task_outcome", "Bob: approved")))
    assert (first.decision, second.decision) == ("send", "send")


async def test_a_distinct_critical_event_on_a_topic_already_raised_critically_is_judged_not_digested():
    fx = Fx()
    smoke = await fx.review(source="alarm", ref="a", topic="house alarm", text="Smoke detected in garage",
                            urgency="critical")
    fx.now += 20 * M
    gas = await fx.review(source="alarm", ref="b", topic="house alarm", text="Carbon monoxide in bedroom",
                          urgency="critical")
    assert (smoke.decision, gas.decision) == ("send", "send") and gas.stage != "dedup"
    assert gas.scores["same_topic_as"] == smoke.id
    judge = Judge("digest")
    judged = Fx(judge=judge)
    await judged.review(source="alarm", ref="a", topic="house alarm", text="Smoke detected in garage",
                        urgency="critical")
    judged.now += 20 * M
    second = await judged.review(source="alarm", ref="b", topic="house alarm", text="Smoke still in garage",
                                 urgency="critical")
    assert (second.decision, second.stage) == ("digest", "second_opinion") and len(judge.prompts) == 1


async def test_no_secret_survives_into_the_audit_or_the_verdict_from_any_field():
    import json

    secret = "sk-" + token_hex(24)  # Generated detector fixture, never a provisioned key.

    class QuotingJudge:
        async def __call__(self, system, prompt):
            return {"choice": "digest", "reason": f"it quotes api_key={secret} so it can wait"}

    fx = Fx(judge=QuotingJudge())
    verdict = await fx.gate.review(Message(
        type="alert:ops", topic=f"leaked api_key={secret}", text=f"Found api_key={secret} in a log.",
        urgency="normal", source="ops", ref="leak-1", evidence=(f"log line api_key={secret}",)))
    assert verdict.decision == "digest" and verdict.stage == "second_opinion"
    sent = await fx.gate.review(Message(type="alert:ops", topic=f"rotate api_key={secret}", text="Rotate the key.",
                                        urgency="high", source="ops", ref="leak-2"))
    assert sent.decision == "send"
    fx.now += 30 * M
    later = await fx.gate.review(Message(type="alert:ops", topic="printer", text="Printer offline.", source="ops",
                                         urgency="high", ref="p-1"))
    assert secret not in json.dumps(verdict.as_dict(), default=str)
    assert secret not in json.dumps(later.as_dict(), default=str) and "rotate api_key=***" in str(later.facts)
    rows = [tuple(row) for row in fx.conn.execute("SELECT * FROM owner_gate")]
    assert rows and secret not in json.dumps(rows, default=str) and secret.lower() not in json.dumps(rows).lower()


def _refuse_writes(conn):
    conn.execute("CREATE TRIGGER refuse BEFORE INSERT ON owner_gate BEGIN SELECT RAISE(ABORT, 'disk full'); END")


async def test_a_hold_that_cannot_be_recorded_is_never_acknowledged_as_a_digest():
    fx = Fx()
    _refuse_writes(fx.conn)
    verdict = await fx.review(source="ops", ref="scrub-1", topic="scrub", text="Scrub done.", urgency="low")
    assert verdict.decision == "send" and verdict.stage == "audit" and verdict.scores["unrecorded"] is True
    assert fx.gate.ledger.held() == []
    with pytest.raises(sqlite3.Error):
        fx.gate.hold(Message(type="health_notice", topic="t", text="x", ref="r"), reason="expired")


def _alert(ref, urgency, *, topic="freezer", text="The freezer is warm."):
    return Message(type="alert:ops", source="ops", ref=ref, topic=topic, text=text, urgency=urgency)


async def test_an_escalation_under_its_own_id_is_reviewed_afresh_not_replayed():
    fx = Fx()
    fx.turn = T0 - M                          # mid-conversation: a high alert waits until 12:09
    waiting = await fx.gate.review(_alert("f-1", "high"), replay=True)
    assert waiting.decision == "wait" and waiting.until == T0 + 9 * M
    fx.now += M
    retry = await fx.gate.review(_alert("f-1", "high"), replay=True)
    assert retry.replayed and retry.id == waiting.id
    calmer = await fx.gate.review(_alert("f-1", "normal"), replay=True)
    assert calmer.replayed and calmer.id == waiting.id
    raised = await fx.gate.review(_alert("f-1", "critical"), replay=True)
    assert not raised.replayed and raised.decision == "send" and raised.stage == "fast_decision"
    # a standing send is not taken back by a raised urgency: the retry that lost its answer still goes
    again = await fx.gate.review(_alert("f-1", "critical"), replay=True)
    assert again.replayed and again.id == raised.id


async def test_an_escalated_hold_or_drop_is_reviewed_afresh_and_the_hold_it_replaces_is_listed_no_more():
    fx = Fx()
    held = await fx.gate.review(_alert("s-1", "low", topic="scrub", text="Scrub found 2 errors."), replay=True)
    assert held.decision == "digest"
    raised = await fx.gate.review(_alert("s-1", "high", topic="scrub", text="Scrub found 2 errors."), replay=True)
    assert not raised.replayed and raised.decision == "send"
    assert fx.gate.ledger.held() == []        # the owner was sent it: the hold is not listed as well
    # a hold replaced by a newer hold is listed once, with the newer words
    await fx.gate.review(_alert("d-1", "low", topic="disk", text="Disk 60% full."), replay=True)
    fx.now += 11 * 60 * M
    await fx.gate.review(_alert("d-1", "low", topic="disk", text="Disk 60% full."), replay=True)
    fx.gate.ledger.conn.execute("UPDATE owner_gate SET at=? WHERE decision='digest'", ((T0 - 25 * 60 * M).isoformat(),))
    await fx.gate.review(_alert("d-1", "low", topic="disk", text="Disk 70% full."), replay=True)
    assert [entry["inputs"]["message"]["text"] for entry in fx.gate.ledger.held()] == ["Disk 70% full."]
    # a dropped repeat raised to critical is news
    await fx.gate.review(_alert("u-1", "high", topic="ups", text="UPS on battery."), replay=True)
    dropped = await fx.gate.review(_alert("u-2", "high", topic="ups", text="UPS on battery."), replay=True)
    assert dropped.decision == "drop"
    critical = await fx.gate.review(_alert("u-2", "critical", topic="ups", text="UPS on battery."), replay=True)
    assert not critical.replayed and critical.decision == "send"


async def test_a_secret_in_a_callers_id_source_topic_or_text_is_never_stored_and_the_id_still_decides_once():
    import json

    secret = "sk-" + token_hex(24)  # Generated detector fixture, never a provisioned key.
    fx = Fx()
    for ref in (secret, f"alert-{secret}", f"x_{secret}"):
        message = Message(type="alert:ops", source="ops", ref=ref, topic=f"key api_key={secret}",
                          text=f"Found api_key={secret} in a log.", urgency="low")
        first = await fx.gate.review(message, replay=True)
        again = await fx.gate.review(message, replay=True)
        assert again.replayed and again.id == first.id and secret not in json.dumps(first.as_dict(), default=str)
    rows = [tuple(row) for row in fx.conn.execute("SELECT * FROM owner_gate")]
    assert len(rows) == 3 and secret not in json.dumps(rows, default=str)
    assert "SENTINEL0123456789abcdef" not in json.dumps(rows, default=str)
    # the mind's own intention ids are its own, and read back as given
    await fx.gate.review(Message(type="health_notice", topic="backup", text="Failing.", ref="row-9"))
    assert fx.gate.ledger.last_for("row-9")["ref"] == "row-9"


def _fail_commit(conn):
    """A deferred constraint: the INSERT succeeds and the COMMIT fails."""
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("CREATE TABLE anchor(id TEXT PRIMARY KEY);"
                       "CREATE TABLE pin(ref TEXT REFERENCES anchor(id) DEFERRABLE INITIALLY DEFERRED);"
                       "CREATE TRIGGER pin_it AFTER INSERT ON owner_gate BEGIN INSERT INTO pin VALUES ('none'); END;")


async def test_a_hold_whose_commit_fails_is_never_acknowledged_on_a_retry_and_nothing_is_left_pending():
    fx = Fx()
    _fail_commit(fx.conn)
    lost = await fx.gate.review(_alert("c-1", "low", topic="scrub", text="Scrub done."), replay=True)
    assert lost.decision == "send" and lost.scores["unrecorded"] is True
    retry = await fx.gate.review(_alert("c-1", "low", topic="scrub", text="Scrub done."), replay=True)
    assert retry.decision == "send" and not retry.replayed
    assert fx.gate.ledger.held() == [] and fx.gate.ledger.last_for("c-1", "ops") is None
    with pytest.raises(sqlite3.Error):
        fx.gate.hold(Message(type="health_notice", topic="t", text="x", ref="r"), reason="expired")
    assert fx.gate.ledger.last_for("r") is None and not fx.conn.in_transaction
    fx.conn.rollback()
    assert fx.gate.ledger.held() == []


async def test_a_gate_write_never_rolls_back_another_writers_pending_work_on_a_shared_connection():
    fx = Fx()
    fx.conn.executescript("CREATE TABLE other(v TEXT);")
    fx.conn.execute("INSERT INTO other VALUES ('kept')")          # pending, not the gate's
    await fx.review(ref="row-1")
    fx.conn.rollback()
    assert [row[0] for row in fx.conn.execute("SELECT v FROM other")] == ["kept"]
    # another writer's pending work that cannot commit: the gate records nothing and leaves it pending
    fx.conn.execute("PRAGMA foreign_keys=ON")
    fx.conn.executescript("CREATE TABLE anchor(id TEXT PRIMARY KEY);"
                          "CREATE TABLE pin(ref TEXT REFERENCES anchor(id) DEFERRABLE INITIALLY DEFERRED);")
    fx.conn.execute("INSERT INTO pin VALUES ('none')")
    fx.now += 30 * M
    verdict = await fx.gate.review(_alert("c-2", "low", topic="scrub", text="Scrub done."), replay=True)
    assert verdict.decision == "send" and verdict.scores["unrecorded"] is True
    assert fx.conn.in_transaction and fx.conn.execute("SELECT count(*) FROM pin").fetchone()[0] == 1
    assert fx.gate.ledger.last_for("c-2", "ops") is None       # nothing of the gate's is left pending
