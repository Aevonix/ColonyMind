"""Instinct's points (I2a): reader tiers in shadow on the ranked points, the night's relation read and the two
pre-screens as points, the owner gate's hold-only reader, and calibration records shipped as data.

Every new model tier ships in shadow; a model tier never applies a label its point excludes; a shipped record
applies only to the model and revision it was measured on. Nothing here contacts a model endpoint: providers are
fakes that answer raw probabilities."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import pytest
import yaml

from colonymind.instinct import REGISTRY, calibration, points
from colonymind.instinct.calibration import Record
from colonymind.instinct.providers import Answer, local_decision, logits
from colonymind.instinct.registry import MODEL_TIERS
from colonymind.instinct.tiers import Instinct

HERE = Path(__file__).resolve().parents[2] / "benchmarks" / "decisions"


class Provider:
    """A model tier's provider: answers ``answers[point]`` (raw probabilities; an exception raises it, None is a
    malformed answer) and keeps every call."""

    private, batch = True, False
    types = frozenset({"yes_no", "choice"})

    def __init__(self, id="reader", *, kind=logits.KIND, model="served-model", revision="rev-1",
                 max_state_chars=logits.MAX_STATE_CHARS, timeout_ms=800, **answers):
        self.id, self.kind, self.model, self.revision = id, kind, model, revision
        self.max_state_chars, self.timeout_ms = max_state_chars, timeout_ms
        self.answers, self.calls = answers, []

    async def answer(self, state, questions):
        self.calls.append((state, dict(questions)))
        out = {}
        for name in questions:
            value = self.answers.get(name)
            if isinstance(value, BaseException):
                raise value
            if value is None:
                raise ValueError("malformed")
            out[name] = dict(value)
        return Answer(out, revision=self.revision)


def fast_provider(**answers):
    return Provider("local", kind=local_decision.KIND, model="", max_state_chars=local_decision.MAX_STATE_CHARS,
                    timeout_ms=250, **answers)


def record(point, tier, provider, *, bars=None):
    spec = REGISTRY[point]
    return Record(point=point, version=spec.version, tier=tier, provider_kind=provider.kind, model=provider.model,
                  revision=provider.revision, temperature=1.0,
                  bars=bars or {label: 0.8 for label in spec.labels}, source="test")


def engine(*, fast=None, reader=None, live=(), records=(), store=None):
    providers = {}
    if fast is not None:
        providers["local"] = fast
    if reader is not None:
        providers["reader"] = reader
    made = Instinct(providers=providers, fast="local" if fast is not None else "",
                    reader="reader" if reader is not None else "", records=tuple(records), store=store)
    for point, tier in live:
        made.set_state(point, tier, "live", source="test")
    return made


def one_hot(point, label):
    return {name: (1.0 if name == label else 0.0) for name in REGISTRY[point].labels}


def counts(made, point, tier):
    return made.health()["points"][point]["counts"].get(tier, {})


def _load(name):
    sys.path.insert(0, str(HERE))
    try:
        spec = importlib.util.spec_from_file_location(f"decisions_{name}", HERE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(HERE))


# -- what ships: every new model tier in shadow, and only the labels its point allows -----------------------------

FIVE = {"outreach_reply": ("engaged", "dig_deeper", "not_interested", "not_now", "stop"),
        "opt_out": ("yes", "no"), "no_reminders": ("yes", "no"), "interest_settled": ("yes", "no"),
        "owner_verdict": ("yes", "no")}
SHIPPED = {
    "owner_instruction": {"reader": ("stop", "pause_today", "none")},
    "report_substance": {"reader": ("finding", "empty")},
    "question_answer": {"reader": ("both", "unclear", "none")},
    "claim_relation": {"fast": ("adds",), "reader": ("adds",)},
    "opinion_none": {"reader": ("yes", "no")},
    "appraisal_empty": {"reader": ("yes", "no")},
    "owner_send": {"fast": ("digest", "drop"), "reader": ("digest", "drop")},
    **{name: {"fast": labels, "reader": labels} for name, labels in FIVE.items()},
    # The reply to an unprompted message keeps its measured fast tier only (see the test below).
    "outreach_reply": {"fast": FIVE["outreach_reply"]},
}


def test_every_ranked_point_ships_its_model_tiers_in_shadow_and_the_verdict_stays_live_on_the_fast_tier():
    for name, tiers in SHIPPED.items():
        point = REGISTRY[name]
        assert {tier.name: tier.may_settle for tier in point.tiers if tier.name in MODEL_TIERS} == tiers, name
        for tier in point.tiers:
            if tier.name in MODEL_TIERS:
                assert tier.state == ("live" if (name, tier.name) == ("owner_verdict", "fast") else "shadow")
                assert not set(tier.may_settle) & point.excluded(), name
    # The moved typed calls stay the existing path; the relation read's is the night's own call.
    for name in ("owner_instruction", "report_substance", "question_answer", "claim_relation", "owner_send"):
        assert REGISTRY[name].tier("deliberate").on_path, name
    assert REGISTRY["claim_relation"].tier("deliberate").may_settle == ("replaces", "adds")
    # A pre-screen skips a call only at a very high calibrated "nothing" (the code floor, my choice).
    for name in ("opinion_none", "appraisal_empty"):
        assert REGISTRY[name].effect == "route" and REGISTRY[name].floors == {"no": points.PRESCREEN_FLOOR}
    assert points.PRESCREEN_FLOOR >= 0.99


async def test_no_reader_reads_a_reply_while_its_question_cannot_say_the_reply_is_about_something_else():
    """A topic-linked reply the phrases read nothing in may be a new request, not an answer to the message. A question
    with only the five reactions has no label for that and forces it into one of them, and a promoted tier would then
    pause, mute or satisfy on it. So no reader tier is registered on ``outreach_reply`` until its question has an
    out-of-scope label, measured with it; its fast tier stays the measured 1.11 one."""
    point = REGISTRY["outreach_reply"]
    assert point.tier("reader") is None or "none" in point.labels
    reader = Provider(outreach_reply={"engaged": 0.0, "dig_deeper": 0.0, "not_interested": 0.0, "not_now": 0.0,
                                      "stop": 1.0})
    made = engine(reader=reader, records=[record("outreach_reply", "reader", reader)])
    decision = await made.decide("outreach_reply", {"topic": "tidal energy",
                                                    "text": "Please rewrite the tidal energy paragraph in my "
                                                            "application."})
    await made.drain()
    assert reader.calls == [] and not decision.applied


def test_the_relation_point_asks_the_question_the_bake_off_measures():
    """A record the bake-off measures for the relation read applies only if the registered question is the one it
    asked: the same version (type, instructions, labels and template)."""
    questions = _load("questions")
    assert REGISTRY["claim_relation"].version == questions.CLAIM_RELATION.version
    assert REGISTRY["claim_relation"].effect == "filter"         # an addition only holds a question back


# -- 1. an excluded label, answered at certainty, changes nothing ------------------------------------------------

DAY_ALERT = {"facts": "local time Wed 12:00; the owner last spoke 90 minutes ago", "urgency": "normal",
             "kind": "external", "topic": "nas", "text": "NAS scrub finished", "source": "ops", "night": False,
             "raised": False, "confidence": 0.75}
FIELDS = {
    "owner_instruction": {"message": "My sister wrote: stop the check-ins, resume them next week."},
    "question_answer": {"question": "Which office do I keep?", "first": "room 4", "second": "room 7",
                        "earlier": "", "reply": "the first one, I moved back"},
    "claim_relation": {"about": "your office location", "older": "My office is room 4.", "older_on": "May 1",
                       "newer": "My office is room 7.", "newer_on": "May 2"},
    "owner_send": DAY_ALERT,
}
NEXT = {"owner_instruction": "none", "question_answer": "unclear", "claim_relation": "adds"}
EXCLUSIONS = [("owner_instruction", "reader", "resume"), ("question_answer", "reader", "first"),
              ("question_answer", "reader", "second"), ("question_answer", "reader", "neither"),
              ("claim_relation", "fast", "replaces"), ("claim_relation", "reader", "replaces"),
              ("owner_send", "fast", "send_now"), ("owner_send", "reader", "send_now")]


@pytest.mark.parametrize("point, tier, label", EXCLUSIONS)
async def test_a_model_tier_answering_an_excluded_label_at_certainty_changes_nothing(point, tier, label):
    provider = (fast_provider if tier == "fast" else Provider)(**{point: one_hot(point, label)})
    made = engine(fast=provider if tier == "fast" else None, reader=provider if tier == "reader" else None,
                  live=[(point, tier)], records=[record(point, tier, provider, bars={
                      name: 0.5 for name in REGISTRY[point].labels})])
    asked = []

    async def deliberate():
        asked.append(point)
        return {"choice": "drop", "label": "drop", "reason": "judged"} if point == "owner_send" else NEXT[point]

    decision = await made.decide(point, FIELDS[point], deliberate=deliberate)
    assert counts(made, point, tier) == {"excluded": 1}             # it answered, and its answer was refused
    assert decision.tier != tier and decision.applied
    if point == "owner_send":
        # Hold-only, and by day a live tier that settled nothing leaves the prior's choice: no second opinion.
        assert (decision.tier, decision.label, asked) == ("fallback", "send_now", [])
    else:
        assert (decision.tier, decision.label, asked) == ("deliberate", NEXT[point], [point])

    async def silent():
        return None

    kept = await made.decide(point, FIELDS[point], deliberate=silent)
    assert kept.tier == "fallback" and kept.label == ("send_now" if point == "owner_send" else None)


# -- 4. owner_send: hold-only, and a live tier's silence leaves the prior with no judge call by day -------------

class Judge:
    def __init__(self, choice="drop"):
        self.choice, self.asked = choice, 0

    async def __call__(self):
        self.asked += 1
        return {"choice": self.choice, "label": self.choice, "reason": "judged"}


@pytest.mark.parametrize("tier", ["fast", "reader"])
async def test_owner_send_a_model_tier_never_sends_where_the_prior_held(tier):
    """A prior ``digest`` the prior did not settle (a low alert by day, the gate's bar raised): a model tier
    answering ``send_now`` at certainty never sends, and nothing asks the judge."""
    fields = {**DAY_ALERT, "urgency": "low", "confidence": 0.99}           # the prior's digest, under its bar
    provider = (fast_provider if tier == "fast" else Provider)(owner_send=one_hot("owner_send", "send_now"))
    made = engine(fast=provider if tier == "fast" else None, reader=provider if tier == "reader" else None,
                  live=[("owner_send", tier)], records=[record("owner_send", tier, provider)])
    judge = Judge("send_now")
    decision = await made.decide("owner_send", fields, deliberate=judge)
    assert (decision.tier, decision.label, judge.asked) == ("fallback", "digest", 0)
    assert counts(made, "owner_send", tier) == {"excluded": 1}


@pytest.mark.parametrize("tier", ["fast", "reader"])
@pytest.mark.parametrize("answer", ["unsure", "failure", "malformed"])
async def test_owner_send_a_live_tier_that_abstains_or_fails_leaves_the_prior_and_asks_no_judge_by_day(tier, answer):
    raw = {"unsure": {"send_now": 0.4, "digest": 0.35, "drop": 0.25}, "failure": RuntimeError("down"),
           "malformed": None}[answer]
    provider = (fast_provider if tier == "fast" else Provider)(owner_send=raw)
    made = engine(fast=provider if tier == "fast" else None, reader=provider if tier == "reader" else None,
                  live=[("owner_send", tier)], records=[record("owner_send", tier, provider)])
    judge = Judge("drop")
    decision = await made.decide("owner_send", DAY_ALERT, deliberate=judge)
    assert (decision.tier, decision.label, judge.asked) == ("fallback", "send_now", 0)
    assert len(provider.calls) == 1


async def test_owner_send_a_live_tier_that_is_sure_holds_and_a_high_stakes_case_still_asks_the_judge():
    provider = Provider(owner_send={"send_now": 0.02, "digest": 0.95, "drop": 0.03})
    made = engine(reader=provider, live=[("owner_send", "reader")], records=[record("owner_send", "reader", provider)])
    judge = Judge("drop")
    held = await made.decide("owner_send", DAY_ALERT, deliberate=judge)
    # The second opinion is asked beside the settle only to compare, never applied (test_instinct_compare.py).
    assert (held.tier, held.label, judge.asked) == ("reader", "digest", 1)
    judge = Judge("drop")
    # A night-time wake: the prior's send is high stakes, so an unsure model leaves it to the second opinion.
    unsure = Provider(owner_send={"send_now": 0.5, "digest": 0.3, "drop": 0.2})
    made = engine(reader=unsure, live=[("owner_send", "reader")], records=[record("owner_send", "reader", unsure)])
    night = {**DAY_ALERT, "urgency": "critical", "night": True}
    woke = await made.decide("owner_send", night, deliberate=judge)
    assert (woke.tier, woke.label, judge.asked) == ("deliberate", "drop", 1)
    # A critical alert on a topic already sent is high stakes by day too.
    raised = await made.decide("owner_send", {**DAY_ALERT, "urgency": "critical", "raised": True}, deliberate=judge)
    assert raised.tier == "deliberate" and judge.asked == 2


async def test_owner_send_as_shipped_asks_the_reader_in_shadow_and_keeps_the_existing_path():
    """In shadow the reader is asked beside the existing path: the prior unsure by day, the second opinion
    decides as before."""
    provider = Provider(owner_send={"send_now": 0.01, "digest": 0.01, "drop": 0.98})
    made = engine(reader=provider, records=[record("owner_send", "reader", provider)])
    judge = Judge("send_now")
    decision = await made.decide("owner_send", DAY_ALERT, deliberate=judge)
    await made.drain()
    assert (decision.tier, decision.label, judge.asked) == ("deliberate", "send_now", 1)
    assert len(provider.calls) == 1 and counts(made, "owner_send", "reader") == {"settled": 1}


# -- the configured reader: its own time limit where the operator leaves it out ---------------------------------

def test_a_reader_left_without_a_time_limit_gets_the_readers_own_default_through_the_config_the_sidecar_reads(
        tmp_path):
    """A reader entry without ``timeout_ms`` keeps the reader's default (a completions call with rotations), not the
    decision server's: through ``colonymind.yaml``, the environment it exports and the engine the sidecar builds. A
    decision server left without one keeps its own."""
    from colonymind.config import apply_environment, load_config
    from colonymind.instinct import settings
    section = {"providers": {"local": {"kind": local_decision.KIND, "url": "http://decide.test"},
                             "reader": {"kind": logits.KIND, "url": "http://reader.test/v1", "model": "served-model"}},
               "fast": "local", "reader": "reader"}
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump({"owner": {"contact_id": "owner-1"},
                                                             "instinct": section}))
    exported = apply_environment(load_config(tmp_path, environ={}), environ={})
    for made in (Instinct.from_environment(exported), Instinct.from_settings(section),
                 Instinct.from_settings(settings.validate(section))):
        assert made.provider_for("reader").timeout_ms == logits.DEFAULT_TIMEOUT_MS
        assert made.provider_for("fast").timeout_ms == local_decision.DEFAULT_TIMEOUT_MS
    for normalised in (settings.validate(section), settings.parse(section)):
        assert normalised["providers"]["reader"]["timeout_ms"] == logits.DEFAULT_TIMEOUT_MS
        assert normalised["providers"]["local"]["timeout_ms"] == local_decision.DEFAULT_TIMEOUT_MS
    # A time limit the operator sets is kept, whatever the kind.
    section["providers"]["reader"]["timeout_ms"] = 250
    assert Instinct.from_settings(settings.validate(section)).provider_for("reader").timeout_ms == 250


# -- rank 1: the typed calls gain a reader in shadow -----------------------------------------------------------

@pytest.mark.parametrize("point, fields, answer, settled", [
    ("owner_instruction", {"message": "stop the check-ins please"}, "stop", "stop"),
    ("report_substance", {"topic": "tidal energy", "report": "Nothing new this week."}, "empty", "empty"),
    ("question_answer", FIELDS["question_answer"], "both", "both"),
])
async def test_a_ranked_typed_call_gains_a_reader_asked_in_shadow_that_settles_only_once_live(point, fields,
                                                                                              answer, settled):
    provider = Provider(**{point: one_hot(point, answer)})
    typed = []

    async def deliberate():
        typed.append(point)
        return "none" if point == "owner_instruction" else "finding" if point == "report_substance" else "unclear"

    shadow = engine(reader=provider, records=[record(point, "reader", provider)])
    before = await shadow.decide(point, fields, deliberate=deliberate)
    await shadow.drain()
    assert before.tier == "deliberate" and typed == [point] and len(provider.calls) == 1
    assert counts(shadow, point, "reader") == {"settled": 1}
    live = engine(reader=provider, live=[(point, "reader")], records=[record(point, "reader", provider)])
    after = await live.decide(point, fields, deliberate=deliberate)
    await live.drain()
    # The typed call is asked beside the settle only to compare (test_instinct_compare.py): never applied.
    assert (after.tier, after.label) == ("reader", settled) and typed == [point, point]


# -- 2. the night's relation read -----------------------------------------------------------------------------

@pytest.fixture
def night(tmp_path, monkeypatch):
    import colonymind.vector as vector
    from colonymind.mind.consolidate import TASK_DIGEST
    from test_mind_consolidate import OWNER, Fixture
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    monkeypatch.setattr(vector, "_pipeline", None)
    made = []

    def build(answer="adds", *, instinct=None, sub="n"):
        fixture = Fixture(tmp_path / sub)
        fixture.router.answers[TASK_DIGEST] = None
        from colonymind.mind.consolidate import RELATION_TASK
        fixture.router.answers[RELATION_TASK] = {"relation": answer}
        fixture.mind.consolidation.instinct = instinct if instinct is not None else Instinct()
        made.append(fixture)
        return fixture

    yield build
    for fixture in made:
        fixture.store.close()


async def pair(fx):
    from test_mind_consolidate import OWNER
    first = await fx.fact("turn-a", OWNER, "telegram-1", "My office is room 4.", "room 4")
    fx.shift(days=1)
    second = await fx.fact("turn-b", OWNER, "discord-2", "My office is room 7.", "room 7")
    return first, second


def relation_calls(fx):
    from colonymind.mind.consolidate import RELATION_TASK
    return [context for _, context in fx.router.calls if context["task"] == RELATION_TASK]


async def test_the_relation_read_in_shadow_asks_the_reader_and_charges_the_night_exactly_as_before(night):
    base = night("adds", sub="base")
    first, second = await pair(base)
    before = await base.mind.consolidate()
    reader = Provider(claim_relation=one_hot("claim_relation", "replaces"))
    made = engine(reader=reader, records=[record("claim_relation", "reader", reader)])
    fx = night("adds", instinct=made, sub="shadow")
    a, b = await pair(fx)
    after = await fx.mind.consolidate()
    await made.drain()
    assert (after["calls"], after["tokens"]) == (before["calls"], before["tokens"])
    assert len(relation_calls(fx)) == len(relation_calls(base)) == 1
    assert fx.mind.consolidation.relation(a, b) == base.mind.consolidation.relation(first, second) == "adds"
    [(state, questions)] = reader.calls
    assert set(questions) == {"claim_relation"} and "room 4" in state and "room 7" in state


@pytest.mark.parametrize("typed, recorded", [("adds", "adds"), ("replaces", "replaces"), ("unsure", None)])
async def test_a_model_replaces_is_never_recorded_the_nights_own_call_decides(night, typed, recorded):
    reader = Provider(claim_relation=one_hot("claim_relation", "replaces"))
    made = engine(reader=reader, live=[("claim_relation", "reader")],
                  records=[record("claim_relation", "reader", reader)])
    fx = night(typed, instinct=made)
    a, b = await pair(fx)
    result = await fx.mind.consolidate()
    assert len(relation_calls(fx)) == 1 and result["calls"] >= 1
    assert fx.mind.consolidation.relation(a, b) == recorded
    assert counts(made, "claim_relation", "reader") == {"excluded": 1}


async def test_a_live_reader_that_is_sure_of_an_addition_records_it_whatever_the_nights_call_says(night):
    """The night's call is asked beside the settle only to compare, inside the night and charged to it: its
    ``replaces`` is never applied."""
    reader = Provider(claim_relation={"adds": 0.97, "replaces": 0.03})
    made = engine(reader=reader, live=[("claim_relation", "reader")],
                  records=[record("claim_relation", "reader", reader)])
    fx = night("replaces", instinct=made)
    a, b = await pair(fx)
    result = await fx.mind.consolidate()
    assert len(relation_calls(fx)) == 1 and result["calls"] >= 1 and not made._tasks
    assert fx.mind.consolidation.relation(a, b) == "adds"
    assert result["counts"].get("compatible") == 1 and not result["counts"].get("contradictions")


async def test_the_nights_compares_are_charged_to_the_night_before_it_ends_and_stop_at_its_budget(tmp_path,
                                                                                                   monkeypatch):
    """Three pairs a live tier settles, under a night's share that fits two relation calls: each compare is the
    night's own call, made in turn and charged before the next is checked, so the third is refused by the night's
    budget and nothing is left running (or charged) after the night's summary."""
    import colonymind.vector as vector
    from colonymind.mind.consolidate import RELATION_TASK, TASK_DIGEST
    from test_mind_consolidate import OWNER, Fixture, NightRouter
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    monkeypatch.setattr(vector, "_pipeline", None)
    fx = Fixture(tmp_path, config={"budgets": {"learn_share": 0.0004}}, router=NightRouter(tokens=40))
    fx.router.answers[TASK_DIGEST] = None
    fx.router.answers[RELATION_TASK] = {"relation": "adds"}
    provider = fast_provider(claim_relation={"adds": 0.99, "replaces": 0.01})
    made = engine(fast=provider, live=[("claim_relation", "fast")], records=[record("claim_relation", "fast", provider)])
    fx.mind.consolidation.instinct = made
    kinds = ("office_location", "home_city", "favorite_color")
    for index, predicate in enumerate(kinds):
        await fx.fact(f"a-{index}", OWNER, "telegram-1", f"My {predicate} is A{index}.", f"a{index}",
                      predicate=predicate)
    fx.shift(days=1)
    for index, predicate in enumerate(kinds):
        await fx.fact(f"b-{index}", OWNER, "discord-2", f"My {predicate} is B{index}.", f"b{index}",
                      predicate=predicate)
    result = await fx.mind.consolidate()
    assert (result["budget"], result["tokens"], result["calls"]) == (80, 80, 2) and not made._tasks
    assert result["counts"]["compatible"] == 3
    await made.drain()
    assert fx.store.get(result["id"]).cost_tokens == 80 and len(relation_calls(fx)) == 2
    fx.store.close()


# -- 3. the pre-screens: in shadow the call always runs; an opt-out turn is never eligible -----------------------

from test_opinion_pass import ASK, RECORD, REPLY, Router, form, job, turn, world  # noqa: E402,F401
from test_opinion_pass import OWNER as OPINION_OWNER  # noqa: E402
from test_source_appraisals import Processor, source, state  # noqa: E402,F401

NOTHING = {"yes": 0.001, "no": 0.999}


async def test_the_opinion_pre_screen_is_asked_in_shadow_and_the_call_always_runs(world):
    turn(world, "ask-1", OPINION_OWNER, ASK, REPLY, admitted=RECORD)
    reader = Provider(opinion_none=NOTHING)
    world.store.instinct = made = engine(reader=reader, records=[record("opinion_none", "reader", reader)])
    router = Router(lambda packet: form())
    assert await opinions_run(world.store, router)
    await made.drain()
    assert len(router.packets) == 1 and job(world, "ask-1")["disposition"] == "formed"
    [(state_text, questions)] = reader.calls
    assert set(questions) == {"opinion_none"} and "Plan Ash or Plan Birch" in state_text and REPLY in state_text
    assert counts(made, "opinion_none", "reader") == {"settled": 1}


async def test_a_live_opinion_pre_screen_skips_the_call_only_at_its_floor(world):
    turn(world, "ask-1", OPINION_OWNER, ASK, REPLY, admitted=RECORD)
    reader = Provider(opinion_none=NOTHING)
    world.store.instinct = engine(reader=reader, live=[("opinion_none", "reader")],
                                  records=[record("opinion_none", "reader", reader, bars={"yes": 0.5, "no": 0.5})])
    router = Router(lambda packet: form())
    assert await opinions_run(world.store, router)
    assert router.packets == [] and job(world, "ask-1")["disposition"] == "screened_none"
    # Under the floor (a record's own bar cannot lower it) the call runs.
    turn(world, "ask-2", OPINION_OWNER, ASK.replace("archive", "backup"), REPLY, admitted=RECORD)
    reader.answers["opinion_none"] = {"yes": 0.02, "no": 0.98}
    assert await opinions_run(world.store, router)
    assert len(router.packets) == 1


async def test_an_opt_out_turn_is_never_put_to_the_opinion_pre_screen(world):
    words = f"Do not text me anymore. {ASK}"
    turn(world, "ask-1", OPINION_OWNER, words, REPLY, admitted=RECORD)
    reader = Provider(opinion_none=NOTHING)
    world.store.instinct = engine(reader=reader, live=[("opinion_none", "reader")],
                                  records=[record("opinion_none", "reader", reader)])
    router = Router(lambda packet: form())
    assert await opinions_run(world.store, router)
    assert reader.calls == [] and len(router.packets) == 1


def many_messages(world, turn_id, users, reply):
    """One captured turn with several user messages; the first carries the admitted record."""
    from test_opinion_pass import admit
    messages = [{"role": "user", "content": text} for text in users] + [{"role": "assistant", "content": reply}]
    world.ledger.record_source(turn_id, contact_id=OPINION_OWNER, session_id=f"session-{OPINION_OWNER}",
                               messages=messages)
    admit(world, turn_id, RECORD)
    with world.ledger._connect() as conn:
        conn.execute("UPDATE source_claim_jobs SET status='complete' WHERE turn_id=?", (turn_id,))


@pytest.mark.parametrize("users", [
    # The opt-out comes after the four messages the opinion call reads.
    [ASK, "One more thing.", "And the logs.", "Also the dates.", "Please stop texting me."],
    # The opt-out comes after the part of a long message the opinion call reads.
    [f"{ASK} {'The rollout notes run long. ' * 200}Please stop texting me."],
], ids=["after-the-fourth-message", "past-the-message-cut"])
async def test_an_opt_out_anywhere_in_the_turn_keeps_it_from_the_opinion_pre_screen(world, users):
    """Eligibility reads every message of the turn in full, not only the part the opinion call is shown."""
    many_messages(world, "ask-1", users, REPLY)
    reader = Provider(opinion_none=NOTHING, max_state_chars=50_000)     # a window wide enough for any state
    world.store.instinct = engine(reader=reader, live=[("opinion_none", "reader")],
                                  records=[record("opinion_none", "reader", reader, bars={"yes": 0.5, "no": 0.5})])
    router = Router(lambda packet: form())
    assert await opinions_run(world.store, router)
    assert reader.calls == [] and len(router.packets) == 1


async def opinions_run(store, router):
    from colonymind.mind.opinions import run_one
    return await run_one(store, router, enabled=True)


def run_row(store, turn_id):
    with closing(store.ledger._connect()) as conn:
        return dict(conn.execute("SELECT status, disposition FROM appraisal_runs WHERE turn_id=?",
                                 (turn_id,)).fetchone())


async def test_the_appraisal_pre_screen_is_asked_in_shadow_and_the_call_always_runs(state):
    source(state, "t1", "The export failed again and I am fed up with it.")
    reader = Provider(appraisal_empty=NOTHING)
    state.instinct = made = engine(reader=reader, records=[record("appraisal_empty", "reader", reader)])
    processor = Processor()
    assert await state.process_one(processor)
    await made.drain()
    assert len(processor.requests) == 1
    [(state_text, questions)] = reader.calls
    assert set(questions) == {"appraisal_empty"} and "fed up" in state_text


async def test_a_live_appraisal_pre_screen_skips_the_call_and_an_opt_out_turn_is_never_eligible(state):
    reader = Provider(appraisal_empty=NOTHING)
    state.instinct = engine(reader=reader, live=[("appraisal_empty", "reader")],
                            records=[record("appraisal_empty", "reader", reader, bars={"yes": 0.5, "no": 0.5})])
    processor = Processor()
    source(state, "t1", "Thanks, see you at three.")
    assert await state.process_one(processor)
    assert processor.requests == [] and run_row(state, "t1") == {"status": "complete",
                                                                 "disposition": "screened_empty"}
    source(state, "t2", "Please stop texting me.")
    assert await state.process_one(processor)
    assert len(reader.calls) == 1 and len(processor.requests) == 1      # the appraisal is the second net


async def test_an_opt_out_in_a_message_the_appraisal_leaves_out_still_keeps_the_turn_from_its_pre_screen(state):
    """Eligibility reads every message of the turn, including one too long for the appraisal's evidence."""
    from test_source_appraisals import module
    reader = Provider(appraisal_empty=NOTHING, max_state_chars=50_000)
    state.instinct = engine(reader=reader, live=[("appraisal_empty", "reader")],
                            records=[record("appraisal_empty", "reader", reader, bars={"yes": 0.5, "no": 0.5})])
    messages = [{"role": "user", "content": "Thanks, see you at three."},
                {"role": "user", "content": f"{'The rollout notes run long. ' * 500}Please stop texting me."}]
    state.ledger.record_source("t1", contact_id="person", session_id="session-t1", scope="person",
                               messages=messages)
    with state.ledger._connect() as conn, conn:
        module.enqueue(conn, "t1", "person", messages, scope="person")
    assert await state.process_one(Processor())
    assert reader.calls == [] and run_row(state, "t1")["disposition"] != "screened_empty"


# -- 5. each template fits its tiers on the longest item of its set --------------------------------------------

LIMITS = {"fast": local_decision.MAX_STATE_CHARS, "reader": logits.MAX_STATE_CHARS}


def test_each_template_fits_its_tiers_on_the_longest_item_or_the_tier_reads_only_the_short_states():
    catalog = _load("catalog")
    lengths = defaultdict(list)
    for items in catalog.frozen().values():
        for item in items:
            if item["point"] in REGISTRY:
                lengths[item["point"]].append(len(REGISTRY[item["point"]].render(item["fields"])))
    checked = set()
    for name, found in lengths.items():
        for tier in REGISTRY[name].tiers:
            if tier.name not in LIMITS:
                continue
            checked.add((name, tier.name))
            if tier.name in points.SHORT_STATES.get(name, ()):
                # Declared: the tier skips the longer states (never cut) and still reads the short ones.
                assert max(found) > LIMITS[tier.name] >= min(found), (name, tier.name)
            else:
                assert max(found) <= LIMITS[tier.name], (name, tier.name, max(found))
    assert {("claim_relation", "fast"), ("claim_relation", "reader"), ("owner_send", "fast"),
            ("owner_send", "reader"), ("question_answer", "reader"),
            *((name, "reader") for name in FIVE if name != "outreach_reply")} <= checked
    assert points.SHORT_STATES == {"owner_send": ("fast",)}


# -- 6. calibration records shipped as data ---------------------------------------------------------------------

def _entry(point="report_substance", **changes):
    spec = REGISTRY["report_substance"]
    entry = {"point": point, "version": spec.version, "tier": "reader", "provider_kind": logits.KIND,
             "model": "served-model", "revision": "rev-1", "method": "temperature", "temperature": 1.0,
             "bars": {label: 0.8 for label in spec.labels}, "n": 120, "accuracy": 0.95,
             "source": "measured on the frozen sets (test)", "fitted_at": "2026-09-28"}
    entry.update(changes)
    entry.setdefault("fingerprint", calibration.fingerprint(entry["version"], entry["tier"], entry["provider_kind"],
                                                            entry["model"], entry["revision"]))
    return entry


def _write(directory, *entries, name="reader.json"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps({"records": list(entries)}))
    return directory


async def test_a_shipped_record_carries_the_fingerprint_it_was_measured_on_and_applies_to_nothing_else(
        tmp_path, monkeypatch):
    from colonymind.instinct import records
    entry = _entry()
    directory = _write(tmp_path / "records", entry)
    [shipped] = records.shipped(REGISTRY, directory=directory)
    assert shipped.fingerprint == entry["fingerprint"] and shipped.model == "served-model"
    fields = {"topic": "tidal energy", "report": "Nothing new this week."}
    for model, revision, applies in (("served-model", "rev-1", True), ("served-model", "rev-2", False),
                                     ("another-model", "rev-1", False)):
        reader = Provider(model=model, revision=revision, report_substance={"finding": 0.05, "empty": 0.95})
        made = Instinct(providers={"reader": reader}, reader="reader", records=(shipped,))
        made.set_state("report_substance", "reader", "live", source="test")
        decision = await made.decide("report_substance", fields)
        assert (decision.tier == "reader") is applies, (model, revision)
        assert counts(made, "report_substance", "reader") == ({"settled": 1} if applies else {"no_record": 1})
    # The engine the sidecar builds reads the same shipped data source.
    monkeypatch.setattr(records, "DIRECTORY", directory)
    made = Instinct()
    assert made.shipped_records == (shipped,)


@pytest.mark.parametrize("revision", ["", "   ", None, 1])
def test_a_shipped_record_requires_a_served_revision(tmp_path, revision):
    from colonymind.instinct import records
    directory = _write(tmp_path / "records", _entry(revision=revision))
    with pytest.raises(records.RecordError, match="served revision"):
        records.load(directory, strict=True)
    assert records.shipped(REGISTRY, directory=directory) == ()


def test_a_record_whose_key_no_longer_matches_its_fingerprint_or_its_question_is_not_shipped(tmp_path):
    from colonymind.instinct import records
    edited = _entry(revision="rev-2", fingerprint=_entry()["fingerprint"])        # the key changed by hand
    stale = _entry(version="0123456789abcdef")                                     # measured on another question
    unknown = _entry(point="no_such_point", version="0123456789abcdef")
    excluded_tier = _entry(tier="deliberate")
    directory = _write(tmp_path / "records", edited, stale, unknown, excluded_tier, _entry(model="ok-model"))
    assert [item.model for item in records.shipped(REGISTRY, directory=directory)] == ["ok-model"]
    with pytest.raises(records.RecordError):
        records.load(directory, strict=True)
    (directory / "broken.json").write_text("{not json")
    assert [item.model for item in records.shipped(REGISTRY, directory=directory)] == ["ok-model"]


def test_a_shipped_isotonic_record_keeps_its_readout_maps_support_and_never_bar(tmp_path):
    from colonymind.instinct import records
    entry = _entry(provider_kind=local_decision.KIND + "/letters-r1", method="isotonic",
                   maps={"finding": [[0.4, 0.0], [0.8, 0.9]]}, support={"finding": [0.2, 0.8]},
                   bars={"finding": 0.8, "empty": calibration.NEVER})
    [loaded] = records.load(_write(tmp_path / "records", entry), strict=True)
    assert loaded.provider_kind == entry["provider_kind"]
    assert loaded.maps == {"finding": ((0.4, 0.0), (0.8, 0.9))}
    assert loaded.support == {"finding": (0.2, 0.8)}
    assert loaded.as_dict()["maps"] == entry["maps"] and loaded.as_dict()["support"] == entry["support"]
    assert calibration.calibrated(loaded, "choice", {"finding": 0.1, "empty": 0.9})["finding"] == 0.0
    assert calibration.reading("choice", {"finding": 0.0, "empty": 1.0}, loaded.bars.__getitem__)[0] is None
    assert calibration.find([loaded], version=loaded.version, tier=loaded.tier, provider_kind=local_decision.KIND,
                            model=loaded.model, revision=loaded.revision) is None


@pytest.mark.parametrize("bar", [-0.01, 1.001, 1.02, float("inf"), float("nan"), True, "0.8"])
def test_a_shipped_record_refuses_invalid_bars_besides_the_exact_never_sentinel(tmp_path, bar):
    from colonymind.instinct import records
    with pytest.raises(records.RecordError, match="bars"):
        records.load(_write(tmp_path / "records", _entry(bars={"finding": bar})), strict=True)


def test_a_readout_does_not_make_an_unknown_provider_kind_valid(tmp_path):
    from colonymind.instinct import records
    with pytest.raises(records.RecordError, match="provider kind"):
        records.load(_write(tmp_path / "records", _entry(provider_kind="hosted-decision.v1/letters-r1")), strict=True)


#: The model tiers of these points that ship with no measured record, so they abstain (shadow logging goes on) and
#: cannot be promoted until the bake-off's campaign measures them on the typed checkpoint and the benchmark reader
#: and their records ship here as data. A record that lands, or a tier added without one, changes this on purpose.
AWAITING_MEASUREMENT = {
    ("percept_attend", "reader"),
    ("claim_relation", "fast"), ("owner_send", "fast"), ("ask_answer", "fast"), ("ask_answer", "reader"),
    *((name, "fast") for name in ("outreach_reply", "no_reminders", "interest_settled", "owner_verdict")),
    *((name, "reader") for name in ("opt_out", "no_reminders", "interest_settled", "owner_verdict",
                                    "owner_instruction", "report_substance", "question_answer", "claim_relation",
                                    "opinion_none", "appraisal_empty", "owner_send")),
}


def test_the_records_that_ship_are_valid_and_each_matches_a_registered_question():
    from colonymind.instinct import records
    loaded = records.load(strict=True)
    assert all(item.version == REGISTRY[item.point].version for item in loaded)
    assert records.shipped(REGISTRY) == loaded


def test_every_model_tier_has_a_shipped_record_for_its_question_or_is_declared_awaiting_its_measurement():
    """Record coverage, asserted rather than assumed: each model tier of these points either has a record for the
    question it asks (data in ``records/``) or is declared awaiting one."""
    shipped = Instinct().shipped_records
    covered = {(item.point, item.tier) for item in shipped
               if item.point in REGISTRY and item.version == REGISTRY[item.point].version}
    tiers = {(point.name, tier.name) for point in points.POINTS for tier in point.tiers if tier.name in MODEL_TIERS}
    assert tiers - covered == AWAITING_MEASUREMENT
    assert not AWAITING_MEASUREMENT & covered              # a record that lands is taken off the list
    assert covered == {("opt_out", "fast")}
