"""The Instinct core (``colonymind.instinct``): the registry and its checks, the tiers in order, shadow beside the
deciding path, the breaker, calibration records by fingerprint, the escalation budget and the decision log."""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from colonymind.instinct import calibration, registry
from colonymind.instinct.calibration import Record, calibrate, calibrate_yes
from colonymind.instinct.log import Store
from colonymind.instinct.providers import Answer
from colonymind.instinct.providers.local_decision import (DEFAULT_TIMEOUT_MS, MAX_STATE_CHARS, TIMEOUT_MS_BOUNDS,
                                                       LocalDecision, answer_probabilities)
from colonymind.instinct.registry import EFFECTS, Point, RegistrationError, RulesAnswer, Tier, register
from colonymind.instinct.tiers import Instinct

KIND = "local-decision.v1"
REVISION = "rev-1"


class Provider:
    """A decision provider that answers ``answers[point]`` (raw probabilities, an exception, or None for a
    malformed answer) and keeps every call."""

    kind, model, private, max_state_chars, batch, timeout_ms = KIND, "", True, 1200, False, 250
    types = frozenset({"yes_no", "choice"})

    def __init__(self, id="fake", *, delay=0.0, revision=REVISION, **answers):
        self.id, self.delay, self.revision, self.answers, self.calls = id, delay, revision, answers, []

    async def answer(self, state, questions):
        self.calls.append((state, dict(questions)))
        if self.delay:
            await asyncio.sleep(self.delay)
        out = {}
        for name in questions:
            value = self.answers.get(name)
            if isinstance(value, BaseException):
                raise value
            if value is None:
                raise ValueError("malformed")
            out[name] = dict(value)
        return Answer(out, revision=self.revision)


def yes_no(name="probe", *, tiers=(), rules=None, fallback=None, literal=True, high_stakes=None, floors=None,
           effect="filter", budget=20):
    return Point(name=name, type="yes_no", instructions="Is it so?",
                 labels={"yes": "it is so", "no": "it is not so"}, template='Words: "{text}"', effect=effect,
                 tiers=tuple(tiers), rules=rules, fallback=fallback, literal=literal, high_stakes=high_stakes,
                 floors=floors or {}, budget_per_hour=budget)


def choice(name="pick", *, tiers=(), labels=None, **kwargs):
    labels = labels or {"a": "the first kind", "b": "the second kind", "none": "neither kind"}
    return Point(name=name, type="choice", instructions="Which kind is it?", labels=labels,
                 template='Words: "{text}"', effect=kwargs.pop("effect", "route"), tiers=tuple(tiers), **kwargs)


def record(point, tier="fast", *, temperature=1.0, bars=None, revision=REVISION, model="", kind=KIND):
    return Record(point=point.name, version=point.version, tier=tier, provider_kind=kind, model=model,
                  revision=revision, temperature=temperature,
                  bars=bars or {label: 0.8 for label in point.labels}, source="test")


def engine(points, *, fast=None, reader=None, records=(), store=None, states=None, **kwargs):
    table = {}
    for point in points:
        register(point, registry=table)
    providers = {}
    if fast is not None:
        providers["local"] = fast
    if reader is not None:
        providers["reader"] = reader
    made = Instinct(providers=providers, fast="local" if fast is not None else "",
                    reader="reader" if reader is not None else "", records=tuple(records), store=store,
                    registry=table, **kwargs)
    for (point, tier), state in (states or {}).items():
        made.set_state(point, tier, state, source="test")
    return made


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "instinct.db")


# -- 1. the registry: a closed effect enum, the exclusions --------------------------------------------------------

def test_the_effect_enum_is_closed():
    assert EFFECTS == ("route", "rank", "filter", "ask", "read")
    for effect in ("grant", "approve", "send", "widen", ""):
        with pytest.raises(RegistrationError, match="effect"):
            register(yes_no(effect=effect), registry={})


@pytest.mark.parametrize("point", [
    choice("owner_instruction", labels={"stop": "s", "pause_today": "p", "resume": "r", "none": "n"},
           tiers=(Tier("fast", may_settle=("stop", "resume")),)),
    choice("question_answer", labels={"first": "f", "second": "s", "both": "b", "neither": "n", "unclear": "u",
                                      "none": "o"}, tiers=(Tier("reader", may_settle=("both", "first")),)),
    choice("claim_relation", labels={"adds": "a", "replaces": "r", "none": "n"},
           tiers=(Tier("fast", may_settle=("replaces",)),)),
    choice("ask_answer", labels={"approve": "a", "decline": "d", "none": "n"},
           tiers=(Tier("fast", may_settle=("approve",)),)),
    choice("owner_send", labels={"send_now": "s", "digest": "d", "drop": "x"},
           tiers=(Tier("fast", may_settle=("send_now", "digest")),)),
    choice("anything", labels={"grant": "g", "none": "n"}, tiers=(Tier("fast", may_settle=("grant",)),)),
    choice("anything", labels={"widen": "w", "none": "n"}, tiers=(Tier("reader", may_settle=("widen",)),)),
    choice("anything", labels={"retract": "r", "none": "n"}, tiers=(Tier("fast", may_settle=("retract",)),)),
    choice("anything", labels={"supersede": "s", "none": "n"}, tiers=(Tier("fast", may_settle=("supersede",)),)),
    choice("anything", labels={"approve": "a", "none": "n"},
           tiers=(Tier("deliberate", may_settle=("approve",)),)),        # a new deliberate tier is a model tier
])
def test_a_model_tier_that_may_apply_an_excluded_label_is_refused(point):
    with pytest.raises(RegistrationError, match="exclu"):
        register(point, registry={})


def test_the_deliberate_tier_that_is_the_existing_path_keeps_the_labels_it_applied():
    """The moved typed calls answered these labels before Instinct: behaviour identity keeps them. The exclusions
    bind every new model tier of the same point."""
    labels = {"stop": "s", "pause_today": "p", "resume": "r", "none": "n"}
    register(choice("owner_instruction", labels=labels,
                    tiers=(Tier("deliberate", may_settle=tuple(labels), on_path=True),)), registry={})
    with pytest.raises(RegistrationError):
        register(choice("owner_instruction", labels=labels,
                        tiers=(Tier("deliberate", may_settle=tuple(labels), on_path=True),
                               Tier("fast", may_settle=("resume",)))), registry={})


def test_registration_checks_labels_tiers_states_and_floors():
    with pytest.raises(RegistrationError, match="label"):
        register(choice(tiers=(Tier("fast", may_settle=("c",)),)), registry={})
    with pytest.raises(RegistrationError, match="state"):
        register(yes_no(tiers=(Tier("fast", may_settle=("yes",), state="on"),)), registry={})
    with pytest.raises(RegistrationError, match="tier"):
        register(yes_no(tiers=(Tier("oracle", may_settle=("yes",)),)), registry={})
    with pytest.raises(RegistrationError, match="floor"):
        register(yes_no(floors={"maybe": 0.5}), registry={})
    with pytest.raises(RegistrationError, match="yes and no"):
        register(Point(name="x", type="yes_no", instructions="?", labels={"true": "t", "false": "f"},
                       template="{text}", effect="filter"), registry={})
    with pytest.raises(RegistrationError, match="literal"):
        register(choice(tiers=(Tier("fast", may_settle=("a",), state="live"),)), registry={})
    table = {}
    register(yes_no(), registry=table)
    with pytest.raises(RegistrationError, match="already"):
        register(yes_no(), registry=table)


def test_a_points_version_is_a_hash_of_what_the_model_reads_and_not_of_its_tiers():
    first = yes_no(tiers=(Tier("fast", may_settle=("yes",)),))
    assert first.version == yes_no().version and len(first.version) == 16
    assert yes_no().version != Point(name="probe", type="yes_no", instructions="Is it so now?",
                                     labels={"yes": "it is so", "no": "it is not so"}, template='Words: "{text}"',
                                     effect="filter").version


def test_every_registered_point_of_the_mind_keeps_the_exclusions():
    from colonymind.instinct import REGISTRY
    assert {"owner_verdict", "outreach_reply", "opt_out", "no_reminders", "interest_settled", "owner_instruction",
            "report_substance", "question_answer", "mind_wish_scope", "owner_send"} <= set(REGISTRY)
    for point in REGISTRY.values():
        assert point.effect in EFFECTS
        for tier in point.tiers:
            if tier.name in registry.MODEL_TIERS or not tier.on_path:
                assert not set(tier.may_settle) & (registry.FORBIDDEN | registry.EXCLUDED.get(point.name, set()))
    assert REGISTRY["owner_send"].tier("fast").may_settle == ("digest", "drop")


# -- 3. rules first --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("state", ["live", "shadow"])
async def test_where_the_rules_speak_no_model_tier_is_asked(state, store):
    fast = Provider(probe={"yes": 0.99, "no": 0.01})
    asked = []

    async def deliberate():
        asked.append(1)
        return "yes"

    point = yes_no(tiers=(Tier("fast", may_settle=("yes", "no"), state=state), Tier("deliberate",
                                                                                     may_settle=("yes", "no"))),
                   rules=lambda fields: RulesAnswer("no") if fields["text"] == "plain" else None)
    made = engine([point], fast=fast, records=[record(point)], store=store)
    decision = await made.decide("probe", {"text": "plain"}, {"turn": "t-1"}, deliberate=deliberate)
    await made.drain()
    assert (decision.label, decision.applied, decision.tier) == ("no", True, "rules")
    assert fast.calls == [] and asked == []
    await made.decide("probe", {"text": "other"}, {"turn": "t-2"})
    await made.drain()
    assert len(fast.calls) == 1


# -- the tiers: fast, calibrated and barred; the fallback --------------------------------------------------------

async def test_a_live_fast_tier_settles_only_above_its_bar_and_otherwise_the_fallback_answers(store):
    point = yes_no(tiers=(Tier("fast", may_settle=("yes", "no"), state="live"),), fallback=lambda fields, rules: None)
    sure = engine([point], fast=Provider(probe={"yes": 0.95, "no": 0.05}), records=[record(point)], store=store)
    decision = await sure.decide("probe", {"text": "x"})
    assert (decision.label, decision.applied, decision.tier, decision.provider) == ("yes", True, "fast", "fake")
    assert decision.probability == pytest.approx(0.95)
    unsure = engine([point], fast=Provider(probe={"yes": 0.6, "no": 0.4}), records=[record(point)], store=store)
    decision = await unsure.decide("probe", {"text": "x"})
    assert (decision.label, decision.applied, decision.tier) == (None, False, "fallback")


async def test_a_yes_no_record_reads_yes_then_no_as_the_band_did():
    """Temperature scaling first; a yes at or above the yes bar; else a no at or above the no bar (the 1.11 band
    [lo, hi] is bars yes = hi, no = 1 - lo)."""
    point = yes_no(tiers=(Tier("fast", may_settle=("yes", "no"), state="live"),))
    band = record(point, temperature=0.25, bars={"yes": 0.97, "no": 0.97})
    for p_yes, expected in ((0.01, "no"), (0.5, None), (0.3, None), (0.99, "yes")):
        made = engine([point], fast=Provider(probe={"yes": p_yes, "no": 1 - p_yes}), records=[band])
        decision = await made.decide("probe", {"text": "x"})
        low, high = 0.03, 0.97
        calibrated = calibrate_yes(p_yes, 0.25)
        oracle = "yes" if calibrated >= high else "no" if calibrated <= low else None
        assert decision.label == oracle == expected, p_yes


async def test_configured_bars_only_raise_the_code_floor_and_the_record():
    point = yes_no(tiers=(Tier("fast", may_settle=("yes", "no"), state="live"),), floors={"yes": 0.9})
    provider = Provider(probe={"yes": 0.92, "no": 0.08})
    lowered = engine([point], fast=provider, records=[record(point, bars={"yes": 0.5, "no": 0.5})],
                     points_config={"probe": {"bars": {"yes": 0.1}}})
    assert (await lowered.decide("probe", {"text": "x"})).label == "yes"          # the floor 0.9 holds
    raised = engine([point], fast=provider, records=[record(point, bars={"yes": 0.5, "no": 0.5})],
                    points_config={"probe": {"bars": {"yes": 0.95}}})
    assert (await raised.decide("probe", {"text": "x"})).label is None
    floored = engine([point], fast=Provider(probe={"yes": 0.85, "no": 0.15}),
                     records=[record(point, bars={"yes": 0.5, "no": 0.5})])
    assert (await floored.decide("probe", {"text": "x"})).label is None


async def test_a_label_outside_may_settle_or_a_high_stakes_case_escalates_instead_of_settling(store):
    point = choice(tiers=(Tier("fast", may_settle=("b", "none"), state="live"),
                          Tier("deliberate", may_settle=("a", "b", "none"), state="live")),
                   high_stakes=lambda fields, label: fields.get("night") and label == "b", literal=True)
    asked = []

    async def deliberate():
        asked.append(1)
        return {"label": "none", "reason": "judged"}

    made = engine([point], fast=Provider(pick={"a": 0.97, "b": 0.02, "none": 0.01}), records=[record(point)],
                  store=store)
    decision = await made.decide("pick", {"text": "x"}, deliberate=deliberate)
    assert (decision.label, decision.tier, decision.detail) == ("none", "deliberate", {"label": "none",
                                                                                        "reason": "judged"})
    night = engine([point], fast=Provider(pick={"a": 0.01, "b": 0.98, "none": 0.01}), records=[record(point)])
    decision = await night.decide("pick", {"text": "x", "night": True}, deliberate=deliberate)
    assert decision.tier == "deliberate" and len(asked) == 2


# -- 5. shadow: logged, never applied, never on the deciding path's time ---------------------------------------

async def test_a_shadow_tier_is_logged_and_never_applied_at_any_probability(store):
    point = yes_no(tiers=(Tier("fast", may_settle=("yes", "no"), state="shadow"),), fallback=lambda f, r: None)
    made = engine([point], fast=Provider(probe={"yes": 0.9999, "no": 0.0001}), records=[record(point)], store=store)
    decision = await made.decide("probe", {"text": "x"}, {"turn": "t-1"}, event="owner_turn")
    assert (decision.label, decision.applied, decision.tier) == (None, False, "fallback")
    await made.drain()
    [row] = store.rows()
    fast = row["tiers"]["fast"]
    assert fast["state"] == "shadow" and fast["outcome"] == "settled" and fast["label"] == "yes"
    assert fast["raw"]["yes"] == pytest.approx(0.9999) and row["applied"] is False and row["label"] is None
    assert row["refs"] == {"turn": "t-1"} and row["event"] == "owner_turn"


async def test_a_shadow_ask_never_delays_the_applied_answer(store):
    point = yes_no(tiers=(Tier("fast", may_settle=("yes", "no"), state="shadow"),))
    slow = Provider(delay=2.0, probe={"yes": 0.99, "no": 0.01})
    made = engine([point], fast=slow, records=[record(point)], store=store)
    loop = asyncio.get_running_loop()
    started = loop.time()
    decision = await made.decide("probe", {"text": "x"})
    assert loop.time() - started < 0.2 and decision.applied is False
    await made.drain()
    [row] = store.rows()
    assert row["tiers"]["fast"]["outcome"] == "failed"          # past the provider's deadline: a timeout


async def test_decide_many_sends_one_events_questions_together():
    one = yes_no("one", tiers=(Tier("fast", may_settle=("yes", "no"), state="live"),))
    two = yes_no("two", tiers=(Tier("fast", may_settle=("yes", "no"), state="live"),))
    provider = Provider(delay=0.5, one={"yes": 0.9, "no": 0.1}, two={"yes": 0.1, "no": 0.9})
    provider.timeout_ms = 3000
    made = engine([one, two], fast=provider, records=[record(one), record(two)])
    loop = asyncio.get_running_loop()
    started = loop.time()
    first, second = await made.decide_many("owner_turn", [("one", {"text": "x"}, {}), ("two", {"text": "y"}, {})])
    assert loop.time() - started < 0.95                          # a parallel burst, not one after the other
    assert (first.label, second.label) == ("yes", "no")


# -- 6. outage: the breaker, and no escalation on a failure ------------------------------------------------------

async def test_the_breaker_opens_after_three_failures_and_skips_the_provider_for_five_minutes(store):
    now = [1000.0]
    point = yes_no(tiers=(Tier("fast", may_settle=("yes", "no"), state="live"),))
    down = Provider(probe=httpx.ConnectError("refused"))
    made = engine([point], fast=down, records=[record(point)], store=store, monotonic=lambda: now[0])
    for _ in range(3):
        assert (await made.decide("probe", {"text": "x"})).applied is False
    assert len(down.calls) == 3 and made.health()["providers"]["local"]["breaker"] == "open"
    await made.decide("probe", {"text": "x"})
    assert len(down.calls) == 3
    assert store.rows(limit=1)[0]["tiers"]["fast"]["outcome"] == "provider_down"
    now[0] += 299
    await made.decide("probe", {"text": "x"})
    assert len(down.calls) == 3
    now[0] += 2
    await made.decide("probe", {"text": "x"})
    assert len(down.calls) == 4


async def test_a_failed_provider_never_escalates_to_the_deliberate_tier():
    point = yes_no(tiers=(Tier("fast", may_settle=("yes", "no"), state="live"),
                          Tier("deliberate", may_settle=("yes", "no"), state="live")))
    asked = []

    async def deliberate():
        asked.append(1)
        return "yes"

    for provider in (Provider(probe=httpx.ConnectError("refused")), Provider(probe=None), None):
        made = engine([point], fast=provider, records=[record(point)])
        decision = await made.decide("probe", {"text": "x"}, deliberate=deliberate)
        assert decision.tier == "fallback"
    assert asked == []
    unsure = engine([point], fast=Provider(probe={"yes": 0.5, "no": 0.5}), records=[record(point)])
    assert (await unsure.decide("probe", {"text": "x"}, deliberate=deliberate)).tier == "deliberate"
    assert asked == [1]


def _reader(**answers):
    reader = Provider("reader-1", **answers)
    reader.kind = "logits"
    return reader


@pytest.mark.parametrize("reader_state", ["live", "shadow"])
async def test_a_failed_down_or_unmatched_provider_never_escalates_to_the_reader(reader_state, store):
    """Only an unsure answer escalates: a fast provider that fails, is skipped by its open breaker or has no record
    for the checkpoint it serves leaves the point on its existing path, and the reader (an LLM call) is not asked,
    live or in shadow."""
    now = [1000.0]
    point = yes_no(tiers=(Tier("fast", may_settle=("yes", "no"), state="live"),
                          Tier("reader", may_settle=("yes", "no"), state=reader_state)))
    records = [record(point), record(point, "reader", kind="logits")]
    for fast in (Provider(probe=httpx.ConnectError("refused")), Provider(probe=None),
                 Provider(probe={"yes": 0.99, "no": 0.01}, revision="unmeasured")):
        reader = _reader(probe={"yes": 0.99, "no": 0.01})
        made = engine([point], fast=fast, reader=reader, records=records, store=store, monotonic=lambda: now[0])
        for n in range(4):                              # the fourth finds the breaker open after three failures
            decision = await made.decide("probe", {"text": str(n)})
            await made.drain()
            assert (decision.tier, decision.applied) == ("fallback", False)
        assert reader.calls == []
        assert {row["tiers"]["reader"]["outcome"] for row in store.rows(limit=4)} == {"not_asked"}
    down = store.rows(limit=4)
    assert down[0]["tiers"]["fast"]["outcome"] == "no_record"
    unsure = _reader(probe={"yes": 0.99, "no": 0.01})
    made = engine([point], fast=Provider(probe={"yes": 0.5, "no": 0.5}), reader=unsure, records=records)
    decision = await made.decide("probe", {"text": "x"})
    await made.drain()
    assert len(unsure.calls) == 1
    assert (decision.tier, decision.label) == (("reader", "yes") if reader_state == "live" else ("fallback", None))


async def test_an_open_breaker_never_escalates_to_the_reader(store):
    now = [1000.0]
    point = yes_no(tiers=(Tier("fast", may_settle=("yes", "no"), state="live"),
                          Tier("reader", may_settle=("yes", "no"), state="live")))
    fast = Provider(probe=httpx.ConnectError("refused"))
    reader = _reader(probe={"yes": 0.99, "no": 0.01})
    made = engine([point], fast=fast, reader=reader, records=[record(point), record(point, "reader", kind="logits")],
                  store=store, monotonic=lambda: now[0])
    for _ in range(3):
        await made.decide("probe", {"text": "x"})
    assert made.health()["providers"]["local"]["breaker"] == "open"
    decision = await made.decide("probe", {"text": "x"})
    assert store.rows(limit=1)[0]["tiers"]["fast"]["outcome"] == "provider_down"
    assert reader.calls == [] and decision.tier == "fallback"


async def test_with_no_fast_provider_or_a_state_too_long_for_it_the_reader_is_the_first_model_tier(store):
    point = yes_no(tiers=(Tier("fast", may_settle=("yes", "no"), state="live"),
                          Tier("reader", may_settle=("yes", "no"), state="live")))
    records = [record(point), record(point, "reader", kind="logits")]
    reader = _reader(probe={"yes": 0.99, "no": 0.01})
    made = engine([point], reader=reader, records=records, store=store)
    assert (await made.decide("probe", {"text": "x"})).tier == "reader"
    small = Provider(probe={"yes": 0.99, "no": 0.01})
    small.max_state_chars = 5
    made = engine([point], fast=small, reader=reader, records=records, store=store)
    assert (await made.decide("probe", {"text": "a longer state"})).tier == "reader"
    assert small.calls == [] and len(reader.calls) == 2


async def test_the_existing_paths_deliberate_call_is_made_exactly_as_without_a_provider():
    point = choice(tiers=(Tier("fast", may_settle=("a", "b", "none"), state="live"),
                          Tier("deliberate", may_settle=("a", "b", "none"), on_path=True)), literal=True)
    asked = []

    async def deliberate():
        asked.append(1)
        return "b"

    for provider in (Provider(pick=httpx.ConnectError("refused")), None, Provider(pick={"a": 0.4, "b": 0.3,
                                                                                      "none": 0.3})):
        made = engine([point], fast=provider, records=[record(point)])
        decision = await made.decide("pick", {"text": "x"}, deliberate=deliberate)
        assert (decision.label, decision.tier) == ("b", "deliberate")
    assert len(asked) == 3


# -- 7. a state too long for a tier is never cut ------------------------------------------------------------------

async def test_a_state_over_the_tiers_limit_is_never_sent_and_never_cut(store):
    point = yes_no(tiers=(Tier("fast", may_settle=("yes", "no"), state="live"),))
    provider = Provider(probe={"yes": 0.99, "no": 0.01})
    made = engine([point], fast=provider, records=[record(point)], store=store)
    decision = await made.decide("probe", {"text": "word " * 400})
    assert provider.calls == [] and decision.applied is False
    assert store.rows()[0]["tiers"]["fast"]["outcome"] == "skipped"


# -- 8. a record applies only on a matching fingerprint ---------------------------------------------------------

@pytest.mark.parametrize("change", [{"version": "0" * 16}, {"tier": "reader"}, {"provider_kind": "logits"},
                                    {"model": "another"}, {"revision": "rev-2"}, {"revision": "*"}, {"revision": ""}])
async def test_a_record_applies_only_on_a_matching_fingerprint(change, store):
    point = yes_no(tiers=(Tier("fast", may_settle=("yes", "no"), state="live"),))
    base = record(point)
    mismatched = Record(**{**base.__dict__, **change})
    made = engine([point], fast=Provider(probe={"yes": 0.99, "no": 0.01}), records=[mismatched], store=store)
    decision = await made.decide("probe", {"text": "x"})
    assert decision.applied is False
    fast = store.rows()[0]["tiers"]["fast"]
    assert fast["outcome"] == "no_record" and fast["raw"]["yes"] == pytest.approx(0.99)
    matched = engine([point], fast=Provider(probe={"yes": 0.99, "no": 0.01}), records=[base])
    assert (await matched.decide("probe", {"text": "x"})).label == "yes"


async def test_without_a_record_a_shadow_tier_still_logs_its_raw_answer(store):
    point = yes_no(tiers=(Tier("fast", may_settle=("yes", "no")),))
    made = engine([point], fast=Provider(probe={"yes": 0.7, "no": 0.3}), store=store)
    await made.decide("probe", {"text": "x"})
    await made.drain()
    fast = store.rows()[0]["tiers"]["fast"]
    assert fast["outcome"] == "no_record" and fast["raw"] == {"yes": pytest.approx(0.7), "no": pytest.approx(0.3)}


async def test_every_answer_logs_the_calibration_identity_it_was_read_under(store):
    """Kind, configured model, served revision and the key's fingerprint go with every answer, matched or not, so
    two checkpoints behind one model name leave raw probabilities that can be told apart and refit per key."""
    point = yes_no(tiers=(Tier("fast", may_settle=("yes", "no"), state="live"),))
    matched = record(point, model="decider-large")
    for revision in (REVISION, "rev-2", ""):
        provider = Provider(probe={"yes": 0.99, "no": 0.01}, revision=revision)
        provider.model = "decider-large"
        made = engine([point], fast=provider, records=[matched], store=store)
        await made.decide("probe", {"text": "x"})
    rows = store.rows()[::-1]
    fast = [row["tiers"]["fast"] for row in rows]
    assert [entry["outcome"] for entry in fast] == ["settled", "no_record", "no_record"]
    assert [entry["revision"] for entry in fast] == [REVISION, "rev-2", ""]
    assert all(entry["kind"] == KIND and entry["model"] == "decider-large" for entry in fast)
    assert fast[0]["fingerprint"] == matched.fingerprint == fast[0]["record"]["fingerprint"]
    assert fast[1]["fingerprint"] == calibration.fingerprint(point.version, "fast", KIND, "decider-large", "rev-2")
    assert len({entry["fingerprint"] for entry in fast}) == 3
    worded = Provider(probe={"yes": 0.99, "no": 0.01}, revision="a checkpoint with words in it")
    await engine([point], fast=worded, store=store).decide("probe", {"text": "x"})
    assert store.rows(limit=1)[0]["tiers"]["fast"]["revision"].startswith("h:")   # never words in the log
    down = Provider(probe=httpx.ConnectError("refused"))
    down.model = "decider-large"
    await engine([point], fast=down, store=store).decide("probe", {"text": "x"})
    failed = store.rows(limit=1)[0]["tiers"]["fast"]
    assert failed["outcome"] == "failed" and (failed["kind"], failed["model"]) == (KIND, "decider-large")


def test_a_records_fingerprint_is_the_recall_cutoffs_helper():
    from colonymind.memory.recall import calibration_fingerprint
    point = yes_no()
    made = record(point)
    assert made.fingerprint == calibration_fingerprint({"point_version": point.version, "tier": "fast",
                                                        "provider_kind": KIND, "model": "", "revision": REVISION})


# -- 9. the log: by reference, no text ----------------------------------------------------------------------------

TEXT_COLUMNS = {"text", "state", "message", "content", "words", "body", "reply", "prompt", "reason_text"}


def test_the_decision_log_has_no_text_field(store):
    with sqlite3.connect(store.path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(instinct_decisions)")}
    assert columns and not columns & TEXT_COLUMNS
    assert "state_hash" in columns and "refs_json" in columns and "tiers_json" in columns


async def test_a_row_carries_raw_probabilities_and_references_and_never_the_words(store):
    point = yes_no(tiers=(Tier("fast", may_settle=("yes", "no"), state="live"),))
    made = engine([point], fast=Provider(probe={"yes": 0.99, "no": 0.01}), records=[record(point)], store=store)
    words = "My door code is 4417 and the spare key is under the mat"
    await made.decide("probe", {"text": words}, {"turn": "t-9", "intention": "row-1", "note": words})
    with sqlite3.connect(store.path) as conn:
        dump = "\n".join(str(value) for row in conn.execute("SELECT * FROM instinct_decisions") for value in row)
    assert "4417" not in dump and "spare key" not in dump
    [row] = store.rows()
    assert row["refs"]["turn"] == "t-9" and row["refs"]["intention"] == "row-1"
    assert row["refs"]["note"].startswith("h:")                   # not an id: kept as a keyed hash
    assert row["tiers"]["fast"]["raw"]["yes"] == pytest.approx(0.99) and row["version"] == point.version


def test_the_state_hash_is_keyed_by_the_instance_secret(tmp_path):
    first, second = Store(tmp_path / "a" / "instinct.db"), Store(tmp_path / "b" / "instinct.db")
    assert first.state_hash("Words: \"yes\"") == first.state_hash("Words: \"yes\"")
    assert first.state_hash("Words: \"yes\"") != second.state_hash("Words: \"yes\"")
    import hashlib
    assert first.state_hash("yes") != hashlib.sha256(b"yes").hexdigest()[:32]
    reopened = Store(tmp_path / "a" / "instinct.db")
    assert reopened.state_hash("Words: \"yes\"") == first.state_hash("Words: \"yes\"")


def test_the_upgrade_backup_carries_the_instance_secret_so_a_restore_keeps_the_hashes(tmp_path):
    """The upgrade's backup copies instinct.db and its secret, private: an instance restored from it into a fresh
    directory reads the same keyed hashes (the state hashes and the hashed references stay matchable)."""
    import os
    import shutil
    from colonymind.init import backup_instance
    from colonymind.instinct.log import KEY_FILE, references
    home = tmp_path / "instance"
    original = Store(home / "instinct.db")
    before = original.state_hash('Words: "yes"'), references({"about": "tax form"}, original.secret())
    backup = backup_instance(home)
    assert (backup / KEY_FILE).is_file() and (backup / KEY_FILE).stat().st_mode & 0o777 == 0o600
    fresh = tmp_path / "restored"
    fresh.mkdir()
    for path in backup.iterdir():
        shutil.copy2(path, fresh / path.name)
    restored = Store(fresh / "instinct.db")
    assert (restored.state_hash('Words: "yes"'), references({"about": "tax form"}, restored.secret())) == before
    assert os.stat(fresh / KEY_FILE).st_mode & 0o777 == 0o600


def test_rows_older_than_the_retention_are_dropped_and_their_counts_kept(store):
    old = datetime.now(timezone.utc) - timedelta(days=181)
    store.write({"id": "dec-old", "at": old.isoformat(), "event": "tick", "point": "probe", "version": "v",
                 "tiers": {"fast": {"state": "shadow", "outcome": "settled"}}, "label": None, "applied": False,
                 "effect": "filter", "answered_by": "fallback", "reason": "fallback", "refs": {}, "state_hash": None})
    store.write({"id": "dec-new", "at": datetime.now(timezone.utc).isoformat(), "event": "tick", "point": "probe",
                 "version": "v", "tiers": {}, "label": "yes", "applied": True, "effect": "filter",
                 "answered_by": "rules", "reason": "rules", "refs": {}, "state_hash": None})
    assert store.prune(datetime.now(timezone.utc)) == 1
    assert [row["id"] for row in store.rows()] == ["dec-new"]
    assert store.kept_counts()["probe"]["fast"]["settled"] == 1


# -- 11. the escalation budget ----------------------------------------------------------------------------------

async def test_the_budget_counts_only_escalations_beyond_the_existing_path(store):
    point = yes_no(tiers=(Tier("fast", may_settle=("yes", "no"), state="live"),
                          Tier("deliberate", may_settle=("yes", "no"), state="live")), budget=2)
    asked = []

    async def deliberate():
        asked.append(1)
        return "no"

    made = engine([point], fast=Provider(probe={"yes": 0.5, "no": 0.5}), records=[record(point)], store=store)
    tiers = [(await made.decide("probe", {"text": str(n)}, deliberate=deliberate)).tier for n in range(3)]
    assert tiers == ["deliberate", "deliberate", "fallback"] and len(asked) == 2
    assert store.rows(limit=1)[0]["tiers"]["deliberate"]["outcome"] == "budget"
    on_path = yes_no("path", tiers=(Tier("fast", may_settle=("yes", "no"), state="live"),
                                    Tier("deliberate", may_settle=("yes", "no"), on_path=True)), budget=1)
    walked = engine([on_path], fast=Provider(path={"yes": 0.5, "no": 0.5}), records=[record(on_path)])
    tiers = [(await walked.decide("path", {"text": str(n)}, deliberate=deliberate)).tier for n in range(3)]
    assert tiers == ["deliberate"] * 3


async def test_live_reader_asks_count_against_the_budget_and_at_budget_the_point_falls_back(store):
    """A live reader is an LLM call the existing path does not make: each ask takes one of the point's hourly
    escalations, whether an unsure fast answer sent it there or it is the first model tier, and at budget the point
    keeps its existing path (here, the fallback)."""
    point = yes_no(tiers=(Tier("fast", may_settle=("yes", "no"), state="live"),
                          Tier("reader", may_settle=("yes", "no"), state="live")), budget=1)
    records = [record(point), record(point, "reader", kind="logits")]
    reader = _reader(probe={"yes": 0.99, "no": 0.01})
    made = engine([point], fast=Provider(probe={"yes": 0.5, "no": 0.5}), reader=reader, records=records,
                  store=store)
    decisions = [await made.decide("probe", {"text": str(n)}) for n in range(3)]
    assert [(item.tier, item.applied) for item in decisions] == [("reader", True), ("fallback", False),
                                                                 ("fallback", False)]
    assert len(reader.calls) == 1
    assert [row["tiers"]["reader"]["outcome"] for row in store.rows()] == ["budget", "budget", "settled"]
    assert [bool(row["tiers"]["reader"].get("escalation")) for row in store.rows()] == [False, False, True]
    first = _reader(probe={"yes": 0.99, "no": 0.01})
    alone = engine([point], reader=first, records=records)
    assert [(await alone.decide("probe", {"text": str(n)})).tier for n in range(2)] == ["reader", "fallback"]
    assert len(first.calls) == 1


async def test_the_readers_shadow_asks_count_against_the_budget(store):
    point = yes_no(tiers=(Tier("reader", may_settle=("yes", "no")),), literal=False, budget=2)
    reader = Provider("reader-1", probe={"yes": 0.9, "no": 0.1})
    reader.kind = "logits"
    made = engine([point], reader=reader, store=store)
    for n in range(3):
        await made.decide("probe", {"text": str(n)})
    await made.drain()
    assert len(reader.calls) == 2
    assert [row["tiers"]["reader"]["outcome"] for row in store.rows()].count("budget") == 1


# -- the provider: local-decision.v1 -----------------------------------------------------------------------------

URL = "http://decide.test"


def body(answer, *, truncated=False, backend=None):
    return {"protocol": "local-decision.v1", "backend": backend or {"name": "fake"}, "elapsed_ms": 3.0,
            "input_truncated": truncated, "answers": {"decision": answer}}


def served(respond, **kwargs):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        result = respond(requests[-1])
        return result if isinstance(result, httpx.Response) else httpx.Response(200, json=result)
    return LocalDecision("local", url=URL, transport=httpx.MockTransport(handler), **kwargs), requests


QUESTION = {"type": "choice", "instructions": "Which?", "labels": {"a": "first", "b": "second"}}


async def test_the_local_provider_asks_one_typed_question_per_request_and_reads_the_revision():
    provider, requests = served(lambda request: body({"type": "choice", "probabilities": {"a": 0.8, "b": 0.2}},
                                                     backend={"name": "x", "checkpoint": "c1",
                                                              "model_revision": "abc"}))
    answer = await provider.answer("Words", {"pick": QUESTION})
    assert answer.probabilities == {"pick": {"a": 0.8, "b": 0.2}} and answer.revision == "c1@abc"
    [request] = requests
    assert set(request) == {"state", "questions"} and set(request["questions"]) == {"decision"}
    assert request["questions"]["decision"] == {"type": "choice", "instructions": "Which?",
                                                "criteria": {"a": "first", "b": "second"}}
    yes_no_question = {"type": "yes_no", "instructions": "?", "labels": {"yes": "so", "no": "not so"}}
    provider, requests = served(lambda request: body({"type": "choice", "probabilities": {"a": 0.7, "b": 0.3}}))
    answer = await provider.answer("Words", {"probe": yes_no_question})
    assert answer.probabilities["probe"] == {"yes": pytest.approx(0.7), "no": pytest.approx(0.3)}
    assert answer.revision == ""
    assert requests[0]["questions"]["decision"]["criteria"] == {"a": "so", "b": "not so"}


@pytest.mark.parametrize("response", [
    httpx.Response(429, json={"detail": "busy"}), httpx.Response(503, json={}), httpx.Response(200, text="no json"),
    httpx.Response(200, json={"protocol": "other", "answers": {}}),
    httpx.Response(200, json=body({"type": "choice", "probabilities": {"a": 0.9, "c": 0.1}})),
    httpx.Response(200, json=body({"type": "choice", "probabilities": {"a": 0.9, "b": 1.7}})),
    httpx.Response(200, json=body({"type": "choice", "probabilities": {"a": 0.4, "b": 0.4}})),     # sums to 0.8
    httpx.Response(200, json=body({"type": "choice", "probabilities": {"yes": 0.9, "no": 0.1}})),
    httpx.Response(200, json=body({"type": "choice", "probabilities": {"a": 0.9, "b": 0.1}}, truncated=True)),
    httpx.Response(200, json={**body({"type": "choice", "probabilities": {"a": 0.9, "b": 0.1}}),
                              "input_truncated": None}),
])
async def test_a_failed_or_malformed_answer_raises_for_the_breaker(response):
    provider, _ = served(lambda request: response)
    with pytest.raises(Exception):
        await provider.answer("Words", {"pick": QUESTION})


def test_answer_probabilities_is_the_one_reader_of_the_contract():
    point = type("Spec", (), {"kind": "choice", "labels": {"a": "", "b": ""}})
    assert answer_probabilities(point, body({"type": "choice", "probabilities": {"a": 0.5, "b": 0.495}})) == {
        "a": 0.5, "b": 0.495}


@pytest.mark.parametrize("value", [math.inf, math.nan, 0, -5, 0.5, 5001, "soon", None, True])
def test_the_provider_refuses_an_unusable_time_limit(value):
    provider = LocalDecision("local", url=URL, timeout_ms=value)
    assert provider.timeout_ms == DEFAULT_TIMEOUT_MS == 250
    assert TIMEOUT_MS_BOUNDS == (1, 5000) and MAX_STATE_CHARS == 1200


async def test_a_hanging_endpoint_ends_at_the_time_limit():
    class Hanging(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            await asyncio.Event().wait()

    point = yes_no(tiers=(Tier("fast", may_settle=("yes", "no"), state="live"),))
    provider = LocalDecision("local", url=URL, timeout_ms=50, transport=Hanging())
    made = engine([point], fast=provider, records=[record(point)])
    loop = asyncio.get_running_loop()
    started = loop.time()
    assert (await asyncio.wait_for(made.decide("probe", {"text": "x"}), 2.0)).applied is False
    assert loop.time() - started < 0.5


# -- calibration moved unchanged -----------------------------------------------------------------------------------

def test_temperature_scaling_is_calibration_and_keeps_the_ranking():
    flat = calibrate({"a": 0.8, "b": 0.2}, 2.0)
    assert flat["a"] == pytest.approx(math.sqrt(0.8) / (math.sqrt(0.8) + math.sqrt(0.2)))
    assert calibrate({"a": 0.8, "b": 0.2}, 1.0) == pytest.approx({"a": 0.8, "b": 0.2})
    sharp = calibrate({"a": 0.6, "b": 0.3, "c": 0.1}, 0.5)
    assert sharp["a"] > 0.6 and sharp["a"] > sharp["b"] > sharp["c"] and sum(sharp.values()) == pytest.approx(1)
    assert calibrate_yes(0.9, 0.5) == pytest.approx(0.81 / 0.82)


def test_the_measured_records_ship_for_the_measured_points_only():
    from colonymind.instinct import REGISTRY
    records = Instinct().shipped_records
    assert {(item.point, item.tier) for item in records} == {("opt_out", "fast")}
    for item in records:
        assert item.version == REGISTRY[item.point].version, item.point      # a changed question voids its record
        assert item.provider_kind == KIND + "/letters-r1" and item.model == ""


# -- surfaces: health ----------------------------------------------------------------------------------------------

async def test_health_names_each_providers_breaker_and_counts_per_point():
    point = yes_no(tiers=(Tier("fast", may_settle=("yes", "no"), state="live"),))
    made = engine([point], fast=Provider(probe={"yes": 0.99, "no": 0.01}), records=[record(point)])
    await made.decide("probe", {"text": "x"})
    health = made.health()
    assert health["providers"]["local"] == {"kind": KIND, "tier": "fast", "breaker": "closed", "failures": 0}
    assert health["points"]["probe"]["states"] == {"fast": "live"}
    assert health["points"]["probe"]["counts"]["fast"]["settled"] == 1
    assert json.dumps(health)
