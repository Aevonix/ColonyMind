"""Instinct's outcomes (docs/DECISIONS.md, Calibration and learning): each resolver labels a decision right, wrong or
nothing from what the instance's own stores record after it (never from the answer a tier gave), the owner's
explicit label, the tighten rule at once, and erasure: a decision whose source was erased leaves the fit, and what
it counted stays."""

from __future__ import annotations

import json
import pathlib
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone

import pytest


from colonymind.contacts.config import ContactsConfig
from colonymind.contacts.optout import apply_opt_out
from colonymind.contacts.store import SQLiteContactStore
from colonymind.initiatives.store import InitiativeStore
from colonymind.instinct import REGISTRY, calibration, fit, outcomes
from colonymind.instinct.calibration import Record
from colonymind.instinct.log import Store
from colonymind.instinct.outcomes import Sources
from colonymind.instinct.providers import Answer, calibration_kind
from colonymind.instinct.registry import Point, Tier, register
from colonymind.instinct.tiers import Instinct
from colonymind.mind.consolidate import QUESTION_KEY
from colonymind.turns.idempotency import TurnIdempotencyLedger
from test_mind_outreach_loop import OWNER, make  # noqa: F401  (make is a pytest fixture)
from test_mind_outreach_reactions import say, shared
from test_mind_typed_decisions import DecisionRouter
from test_opinion_pass import world  # noqa: F401  (world is a pytest fixture)

MEASURED = calibration.MEASURED_REVISIONS[0]
QUOTE = 'My boss wrote "stop checking in on the team every hour". How do I answer that?'


class Provider:
    """A fast decision provider whose answer can change between calls."""

    kind, model, private, max_state_chars, batch, timeout_ms = "local-decision.v1", "", True, 1200, False, 250
    readout = "letters-r1"
    types = frozenset({"yes_no", "choice"})

    def __init__(self, p_yes=0.99, *, revision=MEASURED):
        self.id, self.p_yes, self.revision = "local", p_yes, revision

    async def answer(self, state, questions):
        return Answer({name: {"yes": self.p_yes, "no": 1 - self.p_yes} for name in questions},
                      revision=self.revision)


def owner_verdict_record(provider):
    """Fixture assumption: identity calibration and 0.8 bars let both scripted verdict labels settle."""
    point = REGISTRY["owner_verdict"]
    return Record(point=point.name, version=point.version, tier="fast", provider_kind=calibration_kind(provider),
                  model=provider.model, revision=provider.revision, bars={"yes": 0.8, "no": 0.8}, source="test fixture")


def wired(fx):
    """The Fx mind with its own Instinct engine logging to the instance's instinct.db."""
    engine = Instinct(store=Store(fx.mind.state_dir / "instinct.db"), clock=lambda: fx.now)
    fx.mind.instinct = fx.mind.owner_gate.instinct = fx.mind.lessons.instinct = engine
    return engine


def decisions(engine, point):
    return [row for row in engine.store.rows() if row["point"] == point]


def answered(label):
    async def deliberate():
        return label
    return deliberate


# -- owner_send ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("reply, verdict, source", [
    ("Great find, thanks.", "right", "behavioural"),
    ("STOP", "wrong", "explicit"),
    ("That tidal energy item you sent was not useful to me.", "wrong", "behavioural"),
    (None, None, None),
])
async def test_owner_send_is_graded_by_how_the_owner_took_the_message_the_gate_sent(make, reply, verdict, source):
    fx = make(config={"faculties": {"owner_gate": True}})
    engine = wired(fx)
    row = await shared(fx)
    await engine.drain()
    sources = Sources(fx.mind.state_dir)
    [decision] = [item for item in decisions(engine, "owner_send")
                  if any(v["ref"] == row.id for v in sources.gate_verdicts(item["id"]))]
    assert decision["applied"] and decision["label"] == "send_now" and decision["answered_by"] == "rules"
    if reply is not None:
        fx.shift(timedelta(minutes=20))
        await say(fx, reply, "t-reply", "owner-2")
    counts = outcomes.resolve(engine.store, sources, now=fx.now)
    outcome = engine.store.outcome_of(decision["id"])
    if verdict is None:
        assert outcome is None and counts["owner_send"]["unresolved"] >= 1
        return
    assert (outcome["verdict"], outcome["source"], outcome["judged"]) == (verdict, source, "send_now")
    assert outcome["evidence"]["intention"] == row.id
    assert engine.store.row(decision["id"])["outcome_source"] == source


def sent_with(reaction, store=None):
    """owner_send's view of one message the gate sent, with the reaction the mind recorded on it."""
    from types import SimpleNamespace
    message = SimpleNamespace(id="m-1", result_metadata={"reaction": reaction})
    sources = SimpleNamespace(gate_verdicts=lambda ident: [{"decision": "send", "source": "mind", "ref": "m-1"}],
                              intention=lambda ident: message if ident == "m-1" else None)
    return outcomes.Context(store=store, sources=sources, now=datetime.now(timezone.utc))


SENT = {"id": "dec-1", "point": "owner_send", "applied": True, "label": "send_now", "refs": {"intention": "m-1"},
        "at": datetime.now(timezone.utc).isoformat()}


@pytest.mark.parametrize("reaction, verdict, source", [
    ({"class": "stop", "turn": "t-3", "by": "position"}, "wrong", "explicit"),
    ({"class": "stop", "turn": "t-3", "by": "position", "read": "decision"}, "wrong", "interpreted"),
    ({"class": "negative", "turn": "t-3", "by": "phrase"}, "wrong", "behavioural"),
    ({"class": "negative", "turn": "t-3", "by": "appraisal"}, "wrong", "interpreted"),
    ({"class": "positive", "turn": "t-3", "by": "topic", "read": "decision"}, "right", "interpreted"),
    ({"class": "welcome", "turn": "t-3", "by": "phrase"}, "right", "behavioural"),
])
def test_a_reaction_a_model_read_is_an_interpreted_outcome_never_the_owners_word(reaction, verdict, source):
    outcome = outcomes.owner_send(SENT, sent_with(reaction))
    assert (outcome["verdict"], outcome["source"]) == (verdict, source)


async def test_a_stop_whose_speaker_a_model_decided_is_interpreted_and_never_a_label(make):
    """"My boss wrote 'stop...'" leaves the reader unsure whose stop it is; the owner_instruction point settles it
    (a model, or its fallback). The stop recorded on the message is that reading, not the owner's plain word: an
    interpreted outcome, kept apart, never a label for the fit and never a count for the tighten rule."""
    fx = make(config={"faculties": {"owner_gate": True}})
    engine = wired(fx)
    fx.mind.router = DecisionRouter(instruction="stop")
    row = await shared(fx)
    await engine.drain()
    sources = Sources(fx.mind.state_dir)
    [decision] = [item for item in decisions(engine, "owner_send")
                  if any(v["ref"] == row.id for v in sources.gate_verdicts(item["id"]))]
    fx.shift(timedelta(minutes=20))
    await say(fx, 'My boss wrote "stop checking in on the team" about that tidal energy item. How do I answer that?',
              "t-reply", "owner-2")
    reaction = (fx.store.get(row.id).result_metadata or {})["reaction"]
    assert (reaction["class"], reaction["by"], reaction.get("read")) == ("stop", "topic", None)
    assert any(item["refs"].get("turn") == "t-reply" for item in decisions(engine, "owner_instruction"))
    outcomes.resolve(engine.store, sources, now=fx.now)
    outcome = engine.store.outcome_of(decision["id"])
    assert (outcome["verdict"], outcome["source"]) == ("wrong", "interpreted")
    assert fit.examples_from_log(engine.store) == []


# -- owner_instruction -------------------------------------------------------------------------------------------------

async def test_owner_instruction_a_stop_the_owner_lifts_within_the_day_is_wrong(make):
    fx = make()
    engine = wired(fx)
    fx.mind.router = DecisionRouter(instruction="stop")
    await say(fx, QUOTE, "t-1")
    [decision] = decisions(engine, "owner_instruction")
    assert decision["label"] == "stop" and decision["answered_by"] == "deliberate"
    sources = Sources(fx.mind.state_dir)
    # Standing, and too early to call: the deliberate answer is never its own grade.
    assert outcomes.resolve(engine.store, sources, now=fx.now + timedelta(hours=1))["owner_instruction"] == \
        {"unresolved": 1}
    fx.shift(timedelta(hours=2))
    await say(fx, "You can check in again.", "t-2")
    outcomes.resolve(engine.store, sources, now=fx.now)
    outcome = engine.store.outcome_of(decision["id"])
    assert (outcome["verdict"], outcome["source"], outcome["judged"]) == ("wrong", "behavioural", "stop")


async def test_owner_instruction_a_stop_still_standing_a_day_on_is_right(make):
    fx = make()
    engine = wired(fx)
    fx.mind.router = DecisionRouter(instruction="stop")
    await say(fx, QUOTE, "t-1")
    [decision] = decisions(engine, "owner_instruction")
    outcomes.resolve(engine.store, Sources(fx.mind.state_dir), now=fx.now + timedelta(hours=30))
    outcome = engine.store.outcome_of(decision["id"])
    assert (outcome["verdict"], outcome["label"]) == ("right", "stop")


async def test_owner_instruction_none_is_never_graded(make):
    fx = make()
    engine = wired(fx)
    fx.mind.router = DecisionRouter(instruction="none")
    await say(fx, QUOTE, "t-1")
    outcomes.resolve(engine.store, Sources(fx.mind.state_dir), now=fx.now + timedelta(hours=30))
    assert engine.store.outcomes() == []


# -- opt_out --------------------------------------------------------------------------------------------------------

@pytest.fixture
async def contacts(tmp_path):
    value = SQLiteContactStore(ContactsConfig(sqlite_path=str(tmp_path / "contacts.db")))
    await value.connect()
    yield value
    await value.close()


async def test_opt_out_a_model_read_opt_out_the_owner_sets_back_is_an_explicit_wrong(tmp_path, contacts):
    owner = await contacts.create(display_name="Owner", may_contact="auto")
    sam = await contacts.create(display_name="Sam", may_contact="auto")
    store = Store(tmp_path / "instinct.db")
    engine = Instinct(providers={"local": Provider(0.99)}, fast="local", store=store)
    engine.set_state("opt_out", "fast", "live", source="test")
    lowered = await apply_opt_out(contacts, sam.contact_id, "Lose my number.", source_ref="turn:c-1",
                                  owner_id=owner.contact_id, instinct=engine)
    assert lowered.may_contact == "never"
    [decision] = store.rows()
    sources = Sources(tmp_path, contacts_db=tmp_path / "contacts.db")
    assert outcomes.resolve(store, sources, now=datetime.now(timezone.utc))["opt_out"] == {"unresolved": 1}
    await contacts.set_may_contact(sam.contact_id, "auto", by="owner", reason="Sam wants the reminders")
    outcomes.resolve(store, sources, now=datetime.now(timezone.utc))
    outcome = store.outcome_of(decision["id"])
    assert (outcome["verdict"], outcome["source"], outcome["judged"], outcome["label"]) == \
        ("wrong", "explicit", "yes", "no")


# -- question_answer and report_substance, from the mind's rows ------------------------------------------------

def intention(initiatives, **fields):
    fields.setdefault("drive", "curiosity")
    row, _ = initiatives.create_intention(cls="internal", decision="act", decision_reason="test", hermes_kind="none",
                                          **fields)
    return row


async def test_question_answer_an_applied_answer_whose_statements_are_disputed_again_is_wrong(tmp_path):
    initiatives = InitiativeStore(state_dir=tmp_path)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    asked = {"kind": "contradiction", "text": "Which city?", "first": "Lives in Lyon", "second": "Lives in Porto",
             "claims": ["c-1", "c-2"], "contact_id": OWNER}
    question = intention(initiatives, kind="message", type="contradiction", title="Which statement",
                         status="sent", dedup_key=f"{QUESTION_KEY}q-1", context={"question": asked},
                         created_at=now - timedelta(hours=1))
    initiatives.update(question.id, result_metadata={"question": {"state": "resolved", "answer": "first",
                                                                  "answered_turn": "t-9"}})
    store = Store(tmp_path / "instinct.db")
    engine = Instinct(store=store)
    fields = {"question": asked["text"], "first": asked["first"], "second": asked["second"], "earlier": "",
              "reply": "Lyon."}
    await engine.decide("question_answer", fields, {"question": question.id, "turn": "t-9"},
                        deliberate=answered("first"))
    [decision] = store.rows()
    sources = Sources(tmp_path)
    outcomes.resolve(store, sources, now=now)
    assert store.outcome_of(decision["id"]) is None
    intention(initiatives, kind="message", type="contradiction", title="Which statement again", status="sent",
              dedup_key=f"{QUESTION_KEY}q-2", created_at=now + timedelta(hours=2),
              context={"question": {**asked, "claims": ["c-1", "c-7"]}})
    outcomes.resolve(store, sources, now=now + timedelta(hours=3))
    outcome = store.outcome_of(decision["id"])
    assert (outcome["verdict"], outcome["source"], outcome["judged"]) == ("wrong", "behavioural", "first")
    initiatives.close()


@pytest.mark.parametrize("decided, reaction, verdict", [
    ("finding", "welcome", "right"), ("finding", "engaged", "right"), ("finding", "negative", "wrong"),
    ("finding", None, None), ("empty", None, None),
])
async def test_report_substance_a_finding_sent_is_graded_by_how_the_owner_took_it(tmp_path, decided, reaction,
                                                                                   verdict):
    initiatives = InitiativeStore(state_dir=tmp_path)
    finding = intention(initiatives, kind="task", type="research", title="Research: moss lawns", status="done",
                        dedup_key="research:moss", context={"topic": "moss lawns"})
    message = intention(initiatives, kind="message", type="outreach_finding", title="Moss lawns", status="sent",
                        dedup_key="outreach:moss", context={"topic": "moss lawns"})
    if reaction:
        initiatives.update(message.id, result_metadata={"reaction": {"class": reaction, "turn": "t-3"}})
    initiatives.update(finding.id, result_metadata={"outreach": {"state": "sent", "row": message.id,
                                                                 "substance_decision": decided}})
    store = Store(tmp_path / "instinct.db")
    await Instinct(store=store).decide("report_substance", {"topic": "moss lawns", "report": "a report"},
                                       {"intention": finding.id}, deliberate=answered(decided))
    [decision] = store.rows()
    outcomes.resolve(store, Sources(tmp_path), now=datetime.now(timezone.utc))
    outcome = store.outcome_of(decision["id"])
    assert (outcome and outcome["verdict"]) == verdict
    initiatives.close()


# -- owner_verdict, from the turns ----------------------------------------------------------------------------------

def turn(ledger, turn_id, at, text, session="owner-1"):
    # Each reply names its turn: an identical message in two turns of a session is one message to an erasure.
    ledger.record_source(turn_id, contact_id=OWNER, session_id=session, scope="person", occurred_at=at.isoformat(),
                         messages=[{"role": "user", "content": text},
                                   {"role": "assistant", "content": f"Noted ({turn_id})."}])


@pytest.mark.parametrize("session, verdict", [("owner-1", "wrong"), ("owner-2", None)])
async def test_owner_verdict_a_withdrawn_verdict_the_owner_restates_is_wrong(tmp_path, session, verdict):
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    now = datetime.now(timezone.utc).replace(microsecond=0)
    turn(ledger, "t-1", now - timedelta(minutes=30), "That was wrong, you missed the deadline.")
    turn(ledger, "t-2", now - timedelta(minutes=10), "I said it was wrong: the deadline was Friday.", session)
    store = Store(tmp_path / "instinct.db")
    provider = Provider(0.001)
    engine = Instinct(providers={"local": provider}, fast="local", store=store,
                      records=[owner_verdict_record(provider)])
    withdrawn = await engine.decide("owner_verdict", {"reply": "Done.", "text": "That was wrong"}, {"turn": "t-1"})
    assert withdrawn.applied and withdrawn.label == "no"
    provider.p_yes = 0.999
    stood = await engine.decide("owner_verdict", {"reply": "Noted.", "text": "I said it was wrong"}, {"turn": "t-2"})
    assert stood.label == "yes"
    outcomes.resolve(store, Sources(tmp_path), now=now)
    outcome = store.outcome_of(withdrawn.id)
    assert (outcome and (outcome["verdict"], outcome["label"])) == ((verdict, "yes") if verdict else None)
    assert store.outcome_of(stood.id) is None


# -- never from a tier's answer ----------------------------------------------------------------------------------

async def test_a_decision_with_no_outcome_is_no_label_whatever_its_tiers_answered(tmp_path):
    store = Store(tmp_path / "instinct.db")
    engine = Instinct(providers={"local": Provider(0.99)}, fast="local", store=store)
    for n in range(5):
        await engine.decide("opt_out", {"text": f"words {n}"}, {"contact": "p-02", "source": f"turn:c-{n}"})
    await engine.drain()
    assert all("raw" in row["tiers"]["fast"] for row in store.rows())
    assert fit.examples_from_log(store) == []
    await engine.decide("report_substance", {"topic": "t", "report": "r"}, {"intention": "missing"},
                        deliberate=answered("empty"))
    outcomes.resolve(store, Sources(tmp_path), now=datetime.now(timezone.utc))
    assert store.outcomes() == []


# -- the owner's label -------------------------------------------------------------------------------------------

async def test_the_owners_label_is_explicit_checked_and_replaces_their_earlier_one(tmp_path):
    store = Store(tmp_path / "instinct.db")
    engine = Instinct(providers={"local": Provider(0.99)}, fast="local", store=store)
    await engine.decide("opt_out", {"text": "words"}, {"contact": "p-02", "source": "turn:c-1"})
    await engine.drain()
    [row] = store.rows()
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError):
        outcomes.label(store, row["id"], "maybe", now=now)
    with pytest.raises(KeyError):
        outcomes.label(store, "dec-missing", "no", now=now)
    first = outcomes.label(store, row["id"], "yes", now=now)
    # Nothing was applied: what was decided is what the caller kept (no opt-out), and the owner says otherwise.
    assert (first["verdict"], first["judged"], first["source"]) == ("wrong", "no", "explicit")
    outcomes.label(store, row["id"], "no", now=now)
    [kept] = store.outcomes()
    assert kept["label"] == "no" and store.row(row["id"])["outcome_label"] == "no"
    [example] = fit.examples_from_log(store)
    assert (example.gold, example.existing, example.source, example.tier) == ("no", "no", "explicit", "fast")


def resolved(store, decision, *, label, at, point="probe"):
    """An explicit outcome a resolver recorded (as opt_out's does when the owner sets a permission back)."""
    store.put_outcome({"decision": decision, "point": point, "resolver": point, "source": "explicit",
                       "verdict": "wrong", "judged": "yes", "label": label, "at": at.isoformat(), "evidence": {}})


async def test_the_owners_later_label_overrides_a_resolvers_explicit_outcome_everywhere(tmp_path):
    point = probe()
    table = {}
    register(point, registry=table)
    store = Store(tmp_path / "instinct.db")
    provider = Provider(0.9, revision="rev-1")
    store.put_record(Record(point="probe", version=point.version, tier="fast", provider_kind=calibration_kind(provider),
                            revision="rev-1", bars={"yes": 0.8, "no": 0.8}, source="fitted"))
    engine = Instinct(providers={"local": provider}, fast="local", store=store, registry=table)
    first = await engine.decide("probe", {"text": "one"}, {"turn": "t-1"})
    second = await engine.decide("probe", {"text": "two"}, {"turn": "t-2"})
    now = datetime.now(timezone.utc)
    resolved(store, first.id, label="no", at=now)
    corrected = outcomes.label(store, first.id, "yes", now=now + timedelta(minutes=5), registry=table)
    assert corrected["verdict"] == "right"
    best = store.outcome_of(first.id)
    assert (best["resolver"], best["label"]) == ("owner", "yes") and store.row(first.id)["outcome_label"] == "yes"
    assert fit.strongest(store.outcomes())[first.id]["label"] == "yes"
    assert [(example.key, example.gold) for example in fit.examples_from_log(store, registry=table)] == \
        [(first.id, "yes")]
    # A second resolver wrong within the week: the first was corrected, so there is one wrong, and nothing tightens.
    resolved(store, second.id, label="no", at=now + timedelta(minutes=10))
    assert fit.tighten(store, "probe", now=now + timedelta(minutes=10), registry=table) == []
    assert [item.bars for item in store.records()] == [{"yes": 0.8, "no": 0.8}]


# -- 3. two explicit wrongs within seven days tighten at once ----------------------------------------------------

def probe():
    return Point(name="probe", type="yes_no", instructions="Is it so?", labels={"yes": "it is so", "no": "it is not"},
                 template='Words: "{text}"', effect="filter", literal=True,
                 tiers=(Tier("fast", may_settle=("yes", "no"), state="live"),))


async def test_two_explicit_wrongs_within_seven_days_tighten_at_once(tmp_path):
    point = probe()
    table = {}
    register(point, registry=table)
    store = Store(tmp_path / "instinct.db")
    provider = Provider(0.9, revision="rev-1")
    store.put_record(Record(point="probe", version=point.version, tier="fast", provider_kind=calibration_kind(provider),
                            revision="rev-1", bars={"yes": 0.8, "no": 0.8}, source="fitted"))
    ticks = [0.0]
    engine = Instinct(providers={"local": provider}, fast="local", store=store, registry=table,
                      monotonic=lambda: ticks[0])
    made = []
    for p in (0.9, 0.95, 0.93):
        provider.p_yes = p
        made.append(await engine.decide("probe", {"text": f"at {p}"}, {"turn": f"t-{p}"}))
    assert all(item.applied and item.label == "yes" for item in made)
    now = datetime.now(timezone.utc)
    outcomes.label(store, made[0].id, "no", now=now, registry=table)
    assert [item.bars for item in store.records()] == [{"yes": 0.8, "no": 0.8}]      # one wrong: nothing yet
    outcomes.label(store, made[2].id, "no", now=now + timedelta(days=8), registry=table)
    assert [item.bars for item in store.records()] == [{"yes": 0.8, "no": 0.8}]      # the first is a week old
    outcomes.label(store, made[1].id, "no", now=now + timedelta(days=8), registry=table)
    [tightened] = store.records()
    assert tightened.bars["yes"] == pytest.approx(0.96) and tightened.bars["no"] == 0.8
    assert store.record_history()[0]["reason"].startswith("tightened: 2 explicit wrongs")
    # The live tier is demoted by the same rule, with its evidence.
    assert store.states() == {("probe", "fast"): "demoted"}
    [transition] = store.transitions()
    assert transition["source"].startswith("demoted by the tighten rule")
    assert set(transition["evidence"]["decisions"]) == {made[1].id, made[2].id}
    assert transition["evidence"]["record"] == tightened.id
    ticks[0] += 3600                                                   # the engine reads the store again
    provider.p_yes = 0.95
    again = await engine.decide("probe", {"text": "again"}, {"turn": "t-again"})
    assert not (again.applied and again.label == "yes")


# -- 7. an erased source's decisions leave the fit, and the counts remain ---------------------------------------


# -- Q1. decision rows re-render from their references and match their keyed hash, and the share that do --------


async def test_a_contacts_opt_out_row_re_renders_from_its_turn(tmp_path, contacts):
    owner = await contacts.create(display_name="Owner", may_contact="auto")
    sam = await contacts.create(display_name="Sam", may_contact="auto")
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    ledger.record_source("c-1", contact_id=sam.contact_id, session_id="sam-1", scope="person",
                         occurred_at=datetime.now(timezone.utc).isoformat(),
                         messages=[{"role": "user", "content": "Lose my number."}])
    store = Store(tmp_path / "instinct.db")
    await apply_opt_out(contacts, sam.contact_id, "Lose my number.", source_ref="turn:c-1", owner_id=owner.contact_id,
                        instinct=Instinct(store=store))
    assert dict(outcomes.replay_share(store, Sources(tmp_path))["opt_out"]) == {"matched": 1}


def test_the_share_needs_the_instances_secret_and_never_makes_one(tmp_path):
    store = Store(tmp_path / "instinct.db")
    store.write({"id": "dec-1", "at": datetime.now(timezone.utc).isoformat(), "event": "", "point": "opt_out",
                 "version": "v", "tiers": {}, "label": None, "applied": False, "effect": "filter",
                 "answered_by": "fallback", "reason": "fallback", "refs": {}, "state_hash": "ab"})
    assert outcomes.replay_share(store, Sources(tmp_path)) == {}
    assert not (tmp_path / "instinct.key").exists()


# -- Q1 (1.13). more points re-render; each point without a loader says why; the training row reader ------------

HANDLED = "The tax form for p-05 is due in two hours. I've got it, no pings about it please."


def rewrite(ledger, turn_id, messages):
    """A stored turn made to read otherwise than when it was decided on (as a decision that read other words)."""
    with closing(ledger._connect()) as conn, conn:
        conn.execute("UPDATE turn_sources SET messages_json=? WHERE turn_id=?", (json.dumps(messages), turn_id))


def test_every_point_without_a_loader_has_a_reason_code_the_report_shows(tmp_path):
    from colonymind.instinct import REGISTRY
    from colonymind.reports.instinct import collect, document
    from colonymind.reports.text import markdown
    unloaded = sorted(name for name in REGISTRY if name not in outcomes.LOADERS)
    assert unloaded == sorted(outcomes.NO_LOADER) and not set(outcomes.NO_LOADER) & set(outcomes.LOADERS)
    assert {outcomes.NO_LOADER[name] for name in unloaded} <= set(outcomes.NO_LOADER_REASONS)
    for name in ("cron_supplier_reply", "inbound_triage", "tool_screen"):
        assert outcomes.no_loader_reason(name) == "external"
    assert {outcomes.no_loader_reason(name) for name in outcomes.LOADERS} == {None}
    store = Store(tmp_path / "instinct.db")
    store.write({"id": "dec-1", "at": datetime.now(timezone.utc).isoformat(), "event": "", "point": "tool_screen",
                 "version": "v", "tiers": {}, "label": None, "applied": False, "effect": "filter",
                 "answered_by": "fallback", "reason": "fallback", "refs": {}, "state_hash": "ab"})
    store.secret()
    text = markdown(document(collect(tmp_path)))
    sections = {part.split("\n", 1)[0]: part for part in text.split("\n## ")[1:]}
    for name in unloaded:
        assert f"(no loader, {outcomes.NO_LOADER[name]}: " in sections[name], name
    assert "(no loader, external: " in sections["tool_screen"] and "none of 1 re-render" in sections["tool_screen"]


MARKER = "Kestrel-7731"


def files_holding(root, marker):
    return {str(path) for path in root.rglob("*") if path.is_file() and marker.encode() in path.read_bytes()}


def cites(ledger, source):
    """An answer's dependency on a recorded source (``_supplied_sources``), as the host records one."""
    from colonymind.turns.idempotency import canonical_turn_digest
    with closing(ledger._connect()) as conn:
        messages = json.loads(conn.execute("SELECT messages_json FROM turn_sources WHERE turn_id=?",
                                           (source,)).fetchone()[0])
    return [{"source_id": source, "source_version": canonical_turn_digest(messages)}]


async def test_a_turn_withheld_from_projections_leaves_the_rows_that_read_it(tmp_path, contacts):
    """A correction of who said something invalidates the answers that drew on it: every projection of the ledger
    leaves those turns out, and so does re-rendering, whether a decision read one as its own turn or as the reply
    before it in its session. Their rows no longer re-render, so no training row reads them."""
    from colonymind.turns.source_attribution import correct
    owner = await contacts.create(display_name="Owner", may_contact="auto")
    sam = await contacts.create(display_name="Sam", may_contact="auto")
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    now = datetime.now(timezone.utc).replace(microsecond=0)

    def said(turn_id, minutes, session, *messages):
        ledger.record_source(turn_id, contact_id=owner.contact_id, session_id=session, scope="person",
                             occurred_at=(now - timedelta(minutes=minutes)).isoformat(), messages=list(messages))

    said("src", 50, "owner-0", {"role": "user", "content": "The bicycle is turquoise."})
    # The reply before a verdict drew on the source (context); another verdict's own turn holds such a reply.
    said("t-1", 40, "owner-1", {"role": "user", "content": "What colour is my bicycle?"},
         {"role": "assistant", "content": "Your bicycle is turquoise.", "_supplied_sources": cites(ledger, "src")})
    said("t-2", 39, "owner-1", {"role": "user", "content": f"That was wrong, {MARKER} context."})
    said("t-3", 30, "owner-2", {"role": "user", "content": "Draft the note."},
         {"role": "assistant", "content": "Here is the note."})
    said("t-4", 29, "owner-2", {"role": "user", "content": f"That was wrong, {MARKER} own."},
         {"role": "assistant", "content": "Turquoise it is.", "_supplied_sources": cites(ledger, "src")})
    said("t-5", 20, "owner-3", {"role": "user", "content": "Draft the other note."},
         {"role": "assistant", "content": "Here is the other note."})
    said("t-6", 19, "owner-3", {"role": "user", "content": f"That was wrong, {MARKER} kept."})
    store = Store(tmp_path / "instinct.db")
    engine = Instinct(providers={"local": Provider(0.99)}, fast="local", store=store)
    made = {}
    for case, turn_id, reply in (("context", "t-2", "Your bicycle is turquoise."), ("own", "t-4", "Here is the note."),
                                 ("kept", "t-6", "Here is the other note.")):
        decision = await engine.decide("owner_verdict", {"reply": reply, "text": f"That was wrong, {MARKER} {case}."},
                                       {"turn": turn_id})
        outcomes.label(store, decision.id, "yes", now=now)
        made[case] = decision.id
    await engine.drain()
    sources = Sources(tmp_path, contacts_db=tmp_path / "contacts.db", owner_id=owner.contact_id)
    secret = store.secret()
    assert {case: outcomes.rerender(store.row(ident), sources, secret) for case, ident in made.items()} == {
        "context": "matched", "own": "matched", "kept": "matched"}
    assert {row["ref"] for row in outcomes.replay_rows(store, sources)} == set(made.values())
    result = correct(ledger, operation_id="who-1", performed_by="owner", old_contact_id=owner.contact_id,
                     contact_id=sam.contact_id, source_ids=["src"], evidence_refs=["source:owner-confirmation"])
    assert result["invalidated_source_ids"] == ["t-1", "t-4"]
    assert {case: outcomes.rerender(store.row(ident), sources, secret) for case, ident in made.items()} == {
        "context": "changed", "own": "missing", "kept": "matched"}
    assert [row["ref"] for row in outcomes.replay_rows(store, sources)] == [made["kept"]]


def test_forget_runs_erasure_before_deleting_rows_and_retries_after_failure(tmp_path):
    from colonymind.instinct.log import Store, bind_erasure

    store = Store(tmp_path / "instinct.db")
    store.write({"id": "decision-erase", "at": "2026-10-05T03:00:00+00:00", "point": "probe",
                 "version": "v1", "refs": {"turn": "turn-erased"}})
    calls = []

    def erase(refs):
        calls.append(tuple(refs))
        assert store.row("decision-erase") is not None
        if len(calls) == 1:
            raise RuntimeError("interrupted erasure")

    class Gone:
        def erased(self, refs):
            return {"turn-erased"} & set(refs)

    unbind = bind_erasure(store.path, erase)
    try:
        with pytest.raises(RuntimeError, match="interrupted erasure"):
            outcomes.forget_erased(Store(store.path), Gone())
        assert store.row("decision-erase") is not None
        assert outcomes.forget_erased(Store(store.path), Gone()) == 1
        assert store.row("decision-erase") is None
        assert calls == [("decision-erase",), ("decision-erase",)]
    finally:
        unbind()


@pytest.mark.parametrize("mapping", [False, True])
def test_forget_keeps_retry_references_when_the_erasure_scan_is_not_clean(tmp_path, mapping):
    from types import SimpleNamespace

    store = Store(tmp_path / "instinct.db")
    store.write({"id": "decision-erase", "at": "2026-10-05T03:00:00+00:00", "point": "probe", "version": "v1"})
    result = {"clean": False} if mapping else SimpleNamespace(clean=False)
    with pytest.raises(ValueError, match="erasure"):
        store.forget(["decision-erase"], erasure=lambda refs: result)
    assert store.row("decision-erase") is not None


@pytest.mark.parametrize("unlabelled, expected", [
    (None, [(5, "nothing to compare them with")]),
    (0, [(5, "nothing to compare them with")]),
    (5, [(5, "no outcome to say who was right")]),
    (2, [(3, "nothing to compare them with"), (2, "no outcome to say who was right")]),
])
def test_cli_fit_explains_each_reason_drift_is_unmeasured(tmp_path, monkeypatch, capsys, unlabelled, expected):
    """Unlabelled learned disagreements and missing comparisons keep their own counts and explanations."""
    import argparse

    from colonymind.instinct import cli

    entry = {"point": "owner_verdict", "tier": "fast", "unmeasured": 5, "compared": 7}
    if unlabelled is not None:
        entry["unlabelled"] = unlabelled
    monkeypatch.setattr(fit, "nightly", lambda *args, **kwargs: {"unmeasured": [entry]})
    assert cli._learning("fit", argparse.Namespace(sets=[], json=False), tmp_path) == 0
    assert capsys.readouterr().out.splitlines() == ["forgotten (erased sources): 0"] + [
        f"drift not measured on {count} settled answers of owner_verdict fast: {reason} (7 compared)"
        for count, reason in expected
    ]


@pytest.mark.parametrize("gone", ['withheld'])
async def test_a_grade_whose_restatement_is_gone_teaches_nothing(tmp_path, gone):
    """``owner_verdict`` grades a withdrawn verdict wrong from a later turn, the owner's restatement: the grade rests
    on that turn as much as on the one the decision read. Once the restatement is erased, or withheld from every
    projection, the grade is no label for training, before and after the nightly pass forgets what an erasure
    reached (the withdrawn decision with the restatement's own)."""
    from colonymind.turns.source_attribution import correct
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    now = datetime.now(timezone.utc).replace(microsecond=0)
    ledger.record_source("src", contact_id=OWNER, session_id="owner-0", scope="person",
                         occurred_at=(now - timedelta(minutes=50)).isoformat(),
                         messages=[{"role": "user", "content": "The deadline is Friday."}])
    ledger.record_source("t-0", contact_id=OWNER, session_id="owner-1", scope="person",
                         occurred_at=(now - timedelta(minutes=40)).isoformat(),
                         messages=[{"role": "user", "content": "Draft the note."},
                                   {"role": "assistant", "content": "Here is the note."}])
    turn(ledger, "t-1", now - timedelta(minutes=30), "That was wrong, you missed the deadline.")
    ledger.record_source("t-2", contact_id=OWNER, session_id="owner-1", scope="person",
                         occurred_at=(now - timedelta(minutes=10)).isoformat(),
                         messages=[{"role": "user", "content": "I said it was wrong: the deadline was Friday."},
                                   {"role": "assistant", "content": "Friday, noted.",
                                    "_supplied_sources": cites(ledger, "src")}])
    store = Store(tmp_path / "instinct.db")
    provider = Provider(0.001)
    engine = Instinct(providers={"local": provider}, fast="local", store=store,
                      records=[owner_verdict_record(provider)])
    withdrawn = await engine.decide("owner_verdict", {"reply": "Here is the note.",
                                                      "text": "That was wrong, you missed the deadline."},
                                    {"turn": "t-1"})
    provider.p_yes = 0.999
    await engine.decide("owner_verdict", {"reply": "Noted (t-1).",
                                          "text": "I said it was wrong: the deadline was Friday."}, {"turn": "t-2"})
    await engine.drain()
    sources = Sources(tmp_path, owner_id=OWNER)
    outcomes.resolve(store, sources, now=now)
    graded = store.outcome_of(withdrawn.id)
    assert (graded["verdict"], graded["source"], graded["evidence"]["restated"]) == ("wrong", "behavioural", "t-2")
    [row] = outcomes.replay_rows(store, sources)
    assert (row["ref"], row["label"], row["teacher"], row["push_away"]) == (withdrawn.id, "no", "behavioural_proxy",
                                                                            True)
    correct(ledger, operation_id='who-1', performed_by='owner', old_contact_id=OWNER, contact_id='cid-other', source_ids=['src'], evidence_refs=['source:owner-confirmation'])
    assert outcomes.rerender(store.row(withdrawn.id), sources, store.secret()) == "matched"   # its own turn stands
    assert list(outcomes.replay_rows(store, sources)) == []
    assert outcomes.forget_erased(store, sources) == (2 if gone == "erased" else 0)
    assert list(outcomes.replay_rows(store, sources)) == []

from colonymind.memory import curated

async def test_a_decision_whose_turn_lost_a_message_to_an_erasure_leaves_the_fit_too(tmp_path):
    """Erasing one turn removes its exact copies from the other turns of the session (a partial erasure, recorded
    under the surviving turn's id): a decision that read one of them is forgotten like one whose turn went whole."""
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    now = datetime.now(timezone.utc).replace(microsecond=0)
    ledger.record_source("t-1", contact_id=OWNER, session_id="s-1", scope="person",
                         occurred_at=(now - timedelta(minutes=6)).isoformat(),
                         messages=[{"role": "user", "content": "the private words"}])
    ledger.record_source("t-2", contact_id=OWNER, session_id="s-1", scope="person",
                         occurred_at=(now - timedelta(minutes=5)).isoformat(),
                         messages=[{"role": "user", "content": "the private words"},
                                   {"role": "assistant", "content": "Noted."}])
    store = Store(tmp_path / "instinct.db")
    engine = Instinct(providers={"local": Provider(0.99)}, fast="local", store=store)
    decision = await engine.decide("owner_verdict", {"reply": "Noted.", "text": "the private words"}, {"turn": "t-2"})
    outcomes.label(store, decision.id, "yes", now=now)
    await engine.drain()
    assert [example.key for example in fit.examples_from_log(store)] == [decision.id]
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(ledger)
    _source_forget_receipt = curated.forget_sources(ledger, OWNER, source_ids=['t-1'])
    assert historical_bytes(ledger) == _native_before_forget
    assert all(curated.is_forgotten(ledger, sid) for sid in _source_forget_receipt['source_ids'])
    assert ledger.is_source_erased("t-2") is False                     # t-2 survives, less the erased copy
    assert Sources(tmp_path).erased(["t-2"]) == {"t-2"}
    assert outcomes.forget_erased(store, Sources(tmp_path)) == 1
    assert store.rows() == [] and fit.examples_from_log(store) == [] and store.outcomes() == []
    assert store.kept_counts()["owner_verdict"]["log"] == {"erased": 1}


async def test_a_reminder_decision_re_renders_from_the_item_its_turn_made(tmp_path, monkeypatch):
    """``no_reminders`` through capture itself: the item is found again by the keyed hash the decision logged as its
    ``about`` reference, among the items the turn made, beside the turn's words."""
    from colonymind.commitments.extract import CommitmentExtractor
    from colonymind.commitments.store import CommitmentStore
    from test_commitment_first_mention_hold import _Router, _item
    from test_decisions_wiring import Provider as Labels
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    commitments = CommitmentStore(tmp_path / outcomes.COMMITMENTS_DB)
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    now = datetime.now(timezone.utc).replace(microsecond=0)
    ledger.record_source("t-1", contact_id=OWNER, session_id="owner-1", occurred_at=now.isoformat(), messages=[
        {"role": "user", "content": HANDLED}, {"role": "assistant", "content": "Understood."}])
    extractor = CommitmentExtractor(ledger, lambda: commitments)
    store = Store(tmp_path / "instinct.db")
    extractor.instinct = engine = Instinct(providers={"local": Labels({"no_reminders": "no"})}, fast="local",
                                           store=store)
    due = (now + timedelta(hours=2)).isoformat()
    assert await extractor.process_one(_Router([_item("Send p-05 the tax form", due_at=due),
                                                _item("Groceries", due_at=due)])) is True
    await engine.drain()
    rows = store.rows(point="no_reminders")
    sources, secret = Sources(tmp_path, owner_id=OWNER), store.secret()
    assert len(rows) == 2 and [outcomes.rerender(row, sources, secret) for row in rows] == ["matched", "matched"]
    # An item whose words changed since no longer re-renders; an erased turn's rows find nothing.
    with closing(sqlite3.connect(tmp_path / outcomes.COMMITMENTS_DB)) as conn, conn:
        conn.execute("UPDATE commitments SET description='Send p-05 the new form' WHERE description LIKE 'Send%'")
    assert sorted(outcomes.rerender(row, sources, secret) for row in rows) == ["changed", "matched"]
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(ledger)
    _source_forget_receipt = curated.forget_sources(ledger, OWNER, source_ids=['t-1'])
    assert historical_bytes(ledger) == _native_before_forget
    assert all(curated.is_forgotten(ledger, sid) for sid in _source_forget_receipt['source_ids'])
    assert {outcomes.rerender(row, sources, secret) for row in rows} == {"missing"}


async def test_the_opinion_pre_screen_re_renders_from_its_turn(world):
    from test_instinct_points import NOTHING, Provider as Reader, engine as points_engine, opinions_run
    from test_instinct_points import record as points_record
    from test_opinion_pass import ASK, RECORD, REPLY, Router, form, turn as opinion_turn
    from test_opinion_pass import OWNER as OPINION_OWNER
    opinion_turn(world, "ask-1", OPINION_OWNER, ASK, REPLY, admitted=RECORD)
    reader = Reader(opinion_none=NOTHING)
    store = Store(world.path.parent / "instinct.db")
    world.store.instinct = made = points_engine(reader=reader, store=store,
                                                records=[points_record("opinion_none", "reader", reader)])
    assert await opinions_run(world.store, Router(lambda packet: form()))
    await made.drain()
    [row] = store.rows(point="opinion_none")
    sources, secret = Sources(world.path.parent, owner_id=OPINION_OWNER), store.secret()
    assert outcomes.rerender(row, sources, secret) == "matched"
    rewrite(world.ledger, "ask-1", [{"role": "user", "content": ASK}, {"role": "assistant", "content": "Other."}])
    assert outcomes.rerender(row, sources, secret) == "changed"
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(world.ledger)
    _source_forget_receipt = curated.forget_sources(world.ledger, OPINION_OWNER, source_ids=['ask-1'])
    assert historical_bytes(world.ledger) == _native_before_forget
    assert all(curated.is_forgotten(world.ledger, sid) for sid in _source_forget_receipt['source_ids'])
    assert outcomes.rerender(row, sources, secret) == "missing"


async def test_a_first_appraisals_pre_screen_re_renders_from_its_turn(tmp_path):
    from types import SimpleNamespace

    from colonymind.self_model.appraisals import AppraisalStore
    from test_instinct_points import NOTHING, Provider as Reader, engine as points_engine
    from test_instinct_points import record as points_record
    from test_source_appraisals import Processor, source
    clock = SimpleNamespace(value=1800000000.0)
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    appraisals = AppraisalStore(ledger, owner_id="owner", clock=lambda: clock.value)
    appraisals.test_clock = clock
    source(appraisals, "t1", "The export failed again and I am fed up with it.")
    reader = Reader(appraisal_empty=NOTHING)
    store = Store(tmp_path / "instinct.db")
    appraisals.instinct = made = points_engine(reader=reader, store=store,
                                               records=[points_record("appraisal_empty", "reader", reader)])
    assert await appraisals.process_one(Processor())
    await made.drain()
    [row] = store.rows(point="appraisal_empty")
    sources, secret = Sources(tmp_path, owner_id="owner"), store.secret()
    assert outcomes.rerender(row, sources, secret) == "matched"
    rewrite(ledger, "t1", [{"role": "user", "content": "The export failed again."}])
    assert outcomes.rerender(row, sources, secret) == "changed"
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(ledger)
    _source_forget_receipt = curated.forget_sources(ledger, 'person', source_ids=['t1'])
    assert historical_bytes(ledger) == _native_before_forget
    assert all(curated.is_forgotten(ledger, sid) for sid in _source_forget_receipt['source_ids'])
    assert outcomes.rerender(row, sources, secret) == "missing"


async def test_an_erased_sources_decisions_leave_the_fit_and_their_counts_remain(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    now = datetime.now(timezone.utc).replace(microsecond=0)
    for name in ('t-1', 't-2'):
        turn(ledger, name, now - timedelta(minutes=5), f'words of {name}')
    store = Store(tmp_path / 'instinct.db')
    provider = Provider(0.99)
    engine = Instinct(providers={'local': provider}, fast='local', store=store, records=[owner_verdict_record(provider)])
    kept = []
    for name in ('t-1', 't-2'):
        decision = await engine.decide('owner_verdict', {'reply': 'Done.', 'text': f'words of {name}'}, {'turn': name})
        outcomes.label(store, decision.id, 'yes', now=now)
        kept.append(decision.id)
    await engine.drain()
    assert {example.key for example in fit.examples_from_log(store)} == set(kept)
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger)
    curated.forget_sources(ledger, OWNER, source_ids=['t-1'])
    assert historical_bytes(ledger) == _native_before_withdrawal
    assert outcomes.forget_erased(store, Sources(tmp_path)) == 1
    assert [row['id'] for row in store.rows()] == [kept[1]]
    assert {example.key for example in fit.examples_from_log(store)} == {kept[1]}
    assert [item['decision'] for item in store.outcomes()] == [kept[1]]
    counts = store.kept_counts()
    assert counts['owner_verdict']['log'] == {'erased': 1} and counts['owner_verdict']['fast'] == {'settled': 1}
    assert outcomes.forget_erased(store, Sources(tmp_path)) == 0

async def test_the_minds_decision_rows_re_render_from_their_references_and_match_their_keyed_hash(make):
    from types import SimpleNamespace
    from colonymind.reports.instinct import collect, document
    from colonymind.reports.text import markdown
    fx = make()
    engine = wired(fx)
    fx.mind.router = DecisionRouter(instruction='none', verdict='finding')
    await shared(fx)
    fx.shift(timedelta(minutes=5))
    await say(fx, 'Ah, the tidal energy item you sent reminds me of a trip I took.', 't-2', 'owner-2')
    fx.shift(timedelta(minutes=30))
    await say(fx, QUOTE, 't-3', 'owner-3')
    fx.mind.add_interest('moss lawns', by='turn:t-4')
    finding = fx.found('moss lawns', 'finding: Moss lawns: nothing groundbreaking, sadly')
    fx.shift(timedelta(minutes=30))
    await fx.tick()
    fx.ledger.record_source('t-5', contact_id=OWNER, session_id='owner-5', scope='person', occurred_at=fx.now.isoformat(), messages=[{'role': 'user', 'content': 'Draft the note to the landlord.'}, {'role': 'assistant', 'content': 'Here is the draft:   the rent will follow.'}])
    fx.shift(timedelta(minutes=1))
    fx.ledger.record_source('t-6', contact_id=OWNER, session_id='owner-5', scope='person', occurred_at=fx.now.isoformat(), messages=[{'role': 'user', 'content': 'That was wrong, the rent is due on Friday.'}])
    packet = fx.mind.lessons._packet(fx.now)
    label = next((name for name, owner in packet['owner'].items() if owner['turn_id'] == 't-6'))
    await fx.mind.lessons._vetted({label: 'wrong'}, packet, SimpleNamespace(count=lambda *args, **kwargs: None))
    await engine.decide('interest_settled', {'topic': 'moss lawns', 'text': 'never mind'}, {'turn': 't-7'})
    await engine.drain()
    sources = Sources(fx.mind.state_dir)
    share = outcomes.replay_share(engine.store, sources)
    assert {point: dict(counts) for point, counts in share.items()} == {'outreach_reply': {'matched': 1}, 'owner_instruction': {'matched': 1}, 'report_substance': {'matched': 1}, 'owner_verdict': {'matched': 1}, 'interest_settled': {'no_loader': 1}}
    text = markdown(document(collect(fx.mind.state_dir)))
    assert 'Re-rendered from its references: 1 of 1 match their keyed hash' in text
    assert 'Re-rendered from its references: 0 of 1 match their keyed hash' in text
    [old_interest] = engine.store.rows(point='interest_settled')
    assert outcomes.no_loader_reason('interest_settled', row=old_interest) == 'before_references'
    fx.store.update(finding.id, result='finding: Moss lawns: a new study compares two mixes.')
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(fx.ledger)
    curated.forget_sources(fx.ledger, OWNER, source_ids=['t-3'])
    assert historical_bytes(fx.ledger) == _native_before_withdrawal
    share = outcomes.replay_share(engine.store, sources)
    assert dict(share['report_substance']) == {'changed': 1} and dict(share['owner_instruction']) == {'missing': 1}
    assert dict(share['outreach_reply']) == {'matched': 1}

async def test_the_row_reader_yields_matched_rows_with_an_outcome_their_teacher_and_audience(tmp_path, contacts, caplog):
    """Per decision that re-renders and has an outcome that is a label: its teacher follows the outcome's source,
    a behavioural one only pushes away (and a behavioural right is no row), an interpreted one is no label; the
    audience is the narrowest its sources show. The rendered text lives only in the rows: no file under the
    instance and no log line gains it, and an erased source's rows are gone."""
    import logging
    caplog.set_level(logging.DEBUG)
    owner = await contacts.create(display_name='Owner', may_contact='auto')
    sam = await contacts.create(display_name='Sam', may_contact='auto')
    robin = await contacts.create(display_name='Robin', may_contact='auto', import_source='auto:chat')
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    now = datetime.now(timezone.utc).replace(microsecond=0)
    ledger.record_source('t-0', contact_id=owner.contact_id, session_id='owner-1', scope='person', occurred_at=(now - timedelta(minutes=30)).isoformat(), messages=[{'role': 'user', 'content': 'Draft the note.'}, {'role': 'assistant', 'content': 'Here is the note.'}])
    store = Store(tmp_path / 'instinct.db')
    engine = Instinct(providers={'local': Provider(0.99)}, fast='local', store=store)
    made = {}
    for number, case in enumerate(('owner', 'resolver', 'pushed', 'proxy_right', 'interpreted', 'none', 'erased')):
        turn_id = f't-{number + 1}'
        text = f'That was wrong, {MARKER} {case}.'
        ledger.record_source(turn_id, contact_id=owner.contact_id, session_id='owner-1', scope='person', occurred_at=(now - timedelta(minutes=20 - number)).isoformat(), messages=[{'role': 'user', 'content': text}])
        decision = await engine.decide('owner_verdict', {'reply': 'Here is the note.', 'text': text}, {'turn': turn_id})
        made[case] = decision.id
    for who, name in ((sam, 'c-1'), (robin, 'c-2')):
        text = f'{MARKER}: lose my number.'
        ledger.record_source(name, contact_id=who.contact_id, session_id=f'{name}-s', scope='person', occurred_at=now.isoformat(), messages=[{'role': 'user', 'content': text}])
        decision = await engine.decide('opt_out', {'text': text}, {'contact': who.contact_id, 'source': f'turn:{name}'})
        made[name] = decision.id
    await engine.drain()

    def grade(case, source, verdict, label, *, judged='yes', resolver='owner_verdict'):
        store.put_outcome({'decision': made[case], 'point': store.row(made[case])['point'], 'resolver': resolver, 'source': source, 'verdict': verdict, 'judged': judged, 'label': label, 'at': now.isoformat(), 'evidence': {}})
    outcomes.label(store, made['owner'], 'yes', now=now)
    grade('resolver', 'explicit', 'wrong', 'no')
    grade('pushed', 'behavioural', 'wrong', None)
    grade('proxy_right', 'behavioural', 'right', 'yes')
    grade('interpreted', 'interpreted', 'wrong', None)
    outcomes.label(store, made['erased'], 'no', now=now)
    grade('c-1', 'explicit', 'wrong', 'no', resolver='opt_out')
    grade('c-2', 'explicit', 'wrong', 'no', resolver='opt_out')
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger)
    curated.forget_sources(ledger, owner.contact_id, source_ids=['t-7'])
    assert historical_bytes(ledger) == _native_before_withdrawal
    before = {str(path) for path in tmp_path.rglob('*') if path.is_file()}
    sources = Sources(tmp_path, contacts_db=tmp_path / 'contacts.db', owner_id=owner.contact_id)
    rows = {row['ref']: row for row in outcomes.replay_rows(store, sources)}
    expected = {made['owner']: ('yes', 'owner_label', False, 'explicit', 'owner'), made['resolver']: ('no', 'verified_outcome', False, 'resolver', 'owner'), made['pushed']: ('yes', 'behavioural_proxy', True, 'behavioural', 'owner'), made['c-1']: ('no', 'verified_outcome', False, 'resolver', 'contact'), made['c-2']: ('no', 'verified_outcome', False, 'resolver', 'guest')}
    assert {ref: (row['label'], row['teacher'], row['push_away'], row['source_kind'], row['audience']) for ref, row in rows.items()} == expected
    for ref, row in rows.items():
        assert set(row) == {'ref', 'point', 'version', 'state', 'state_hash', 'label', 'teacher', 'push_away', 'source_kind', 'audience', 'at'}
        assert MARKER in row['state'] and row['state_hash'] == store.row(ref)['state_hash']
        assert row['at'].endswith('+00:00') and row['version'] == store.row(ref)['version']
    assert [row['ref'] for row in outcomes.replay_rows(store, sources, point='opt_out')] == [made['c-1'], made['c-2']]
    after = {str(path) for path in tmp_path.rglob('*') if path.is_file()}
    assert {path for path in after - before if MARKER.encode() in open(path, 'rb').read()} == set()
    assert all((pathlib.Path(path).name.startswith('turn-idempotency.db') for path in files_holding(tmp_path, MARKER)))
    assert MARKER not in caplog.text
    assert list(outcomes.replay_rows(store, Sources(tmp_path, contacts_db=tmp_path / 'contacts.db', owner_id=''))) == []

async def test_a_grade_whose_reaction_turn_was_erased_teaches_nothing(tmp_path):
    """A finding reported is graded by how the owner took the message (the reaction the mind recorded, in their
    turn): erasing that turn leaves no training row, and the nightly pass forgets the decision it graded."""
    initiatives = InitiativeStore(state_dir=tmp_path)
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    ledger.record_source('t-3', contact_id=OWNER, session_id='owner-2', scope='person', occurred_at=datetime.now(timezone.utc).isoformat(), messages=[{'role': 'user', 'content': 'That moss lawns item was not useful to me.'}])
    finding = intention(initiatives, kind='task', type='research', title='Research: moss lawns', status='done', dedup_key='research:moss', context={'topic': 'moss lawns'})
    message = intention(initiatives, kind='message', type='outreach_finding', title='Moss lawns', status='sent', dedup_key='outreach:moss', context={'topic': 'moss lawns'})
    initiatives.update(message.id, result_metadata={'reaction': {'class': 'negative', 'turn': 't-3', 'by': 'phrase'}})
    initiatives.update(finding.id, result_metadata={'outreach': {'state': 'sent', 'row': message.id, 'substance_decision': 'finding'}})
    store = Store(tmp_path / 'instinct.db')
    await Instinct(store=store).decide('report_substance', {'topic': 'moss lawns', 'report': ''}, {'intention': finding.id}, deliberate=answered('finding'))
    [decision] = store.rows()
    sources = Sources(tmp_path, owner_id=OWNER)
    outcomes.resolve(store, sources, now=datetime.now(timezone.utc))
    assert store.outcome_of(decision['id'])['source'] == 'behavioural'
    [row] = outcomes.replay_rows(store, sources)
    assert (row['label'], row['teacher'], row['push_away'], row['audience']) == ('finding', 'behavioural_proxy', True, 'owner')
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger)
    curated.forget_sources(ledger, OWNER, source_ids=['t-3'])
    assert historical_bytes(ledger) == _native_before_withdrawal
    assert outcomes.rerender(decision, sources, store.secret()) == 'matched'
    assert list(outcomes.replay_rows(store, sources)) == []
    assert outcomes.forget_erased(store, sources) == 1 and store.rows() == []
    initiatives.close()
