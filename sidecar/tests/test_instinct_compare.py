"""A live model tier's settle, compared with the existing path it stands in for (docs/DECISIONS.md, Calibration and
learning).

Where a point's existing path asks a model (its deliberate tier is the path: the owner gate's second opinion, the
typed calls), a live model tier that settles leaves that call unasked, so drift could not be measured on the settles
themselves. The call is asked beside the settle, within the point's hourly budget: its answer is logged under the
deliberate tier as ``compared`` and never applied. It is the caller's own call on the caller's budget, so off the
owner's turn it runs inside the caller's run (before the settled answer returns, in turn with the caller's other
calls); on the owner's turn it runs in the background, one at a time per point. Nothing here contacts a model
endpoint: providers are fakes that answer raw probabilities."""

from __future__ import annotations

import asyncio

import pytest

from colonymind.instinct import fit
from colonymind.instinct import state as point_state
from colonymind.instinct.log import Store
from test_instinct_points import DAY_ALERT, Provider, engine, fast_provider, one_hot, record

SURE_DIGEST = {"send_now": 0.02, "digest": 0.95, "drop": 0.03}


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "instinct.db")


class Judge:
    """The owner gate's second opinion as a caller passes it: counts its calls, may wait on a gate, fail, or answer
    a label the point does not have."""

    def __init__(self, choice="drop", *, gate=None, error=None):
        self.choice, self.gate, self.error = choice, gate, error
        self.asked = self.answered = 0
        self.started = asyncio.Event()

    async def __call__(self):
        self.asked += 1
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        self.answered += 1
        return {"choice": self.choice, "label": self.choice, "reason": "judged"}


def live(store, tier="reader", answer=None):
    provider = (fast_provider if tier == "fast" else Provider)(owner_send=answer or SURE_DIGEST)
    made = engine(fast=provider if tier == "fast" else None, reader=provider if tier == "reader" else None,
                  live=[("owner_send", tier)], records=[record("owner_send", tier, provider)], store=store)
    return made, provider


@pytest.mark.parametrize("tier", ["fast", "reader"])
async def test_a_live_settle_asks_the_existing_paths_model_beside_it_and_logs_it_compared_never_applied(store, tier):
    made, _ = live(store, tier)
    judge = Judge("drop")
    decision = await made.decide("owner_send", DAY_ALERT, deliberate=judge)
    # Off the owner's turn the second opinion is asked inside the caller's run, and its answer is never applied.
    assert (decision.tier, decision.label, decision.applied) == (tier, "digest", True)
    assert (judge.asked, judge.answered) == (1, 1) and not made._tasks
    await made.drain()
    row, = store.rows(point="owner_send")
    assert (row["answered_by"], row["label"], row["applied"]) == (tier, "digest", True)
    assert row["tiers"]["deliberate"]["outcome"] == fit.COMPARED == "compared"
    assert row["tiers"]["deliberate"]["label"] == "drop" and row["tiers"]["deliberate"]["state"] == "live"
    # Drift is measured on the settle: the existing path's answer is known, and it disagreed.
    assert fit.existing_answer(row) == "drop"
    measured = point_state.agreement(store, "owner_send", tier)
    assert (measured["n"], measured["disagree"], measured["unmeasured"]) == (1, 1, 0)


async def test_an_agreeing_compare_counts_as_agreement(store):
    made, _ = live(store)
    await made.decide("owner_send", DAY_ALERT, deliberate=Judge("digest"))
    await made.drain()
    measured = point_state.agreement(store, "owner_send", "reader")
    assert (measured["n"], measured["agree"], measured["unmeasured"]) == (1, 1, 0)


async def test_the_compare_takes_from_the_hourly_budget_and_at_budget_the_settle_stays_unmeasured(store):
    made, _ = live(store, "fast")
    judge = Judge("digest")
    budget = made.registry["owner_send"].budget_per_hour
    for _ in range(budget + 1):
        decision = await made.decide("owner_send", DAY_ALERT, deliberate=judge)
        assert (decision.tier, decision.label) == ("fast", "digest")
    await made.drain()
    assert judge.asked == budget
    newest = store.rows(point="owner_send", limit=1)[0]
    assert newest["tiers"]["deliberate"]["outcome"] == "budget" and fit.existing_answer(newest) is None
    measured = point_state.agreement(store, "owner_send", "fast")
    assert (measured["n"], measured["unmeasured"]) == (budget, 1)


@pytest.mark.parametrize("judge, outcome", [(Judge(error=RuntimeError("down")), "failed"),
                                            (Judge("maybe"), "no_answer")])
async def test_a_compare_that_fails_or_answers_nothing_usable_leaves_the_settle_unmeasured(store, judge, outcome):
    made, _ = live(store)
    decision = await made.decide("owner_send", DAY_ALERT, deliberate=judge)
    await made.drain()
    assert (decision.tier, decision.label) == ("reader", "digest")
    row, = store.rows(point="owner_send")
    assert row["tiers"]["deliberate"]["outcome"] == outcome and fit.existing_answer(row) is None


async def test_no_compare_without_a_deliberate_call_or_at_a_point_whose_existing_path_asks_no_model(store):
    made, _ = live(store)
    decision = await made.decide("owner_send", DAY_ALERT)
    await made.drain()
    assert decision.tier == "reader"
    assert (store.rows(point="owner_send")[0]["tiers"].get("deliberate") or {}).get("outcome") != "compared"
    # opt_out's existing path is the opt-out phrases: nothing there asks a model, so nothing is asked beside a
    # settle, whatever the caller passes.
    provider = fast_provider(opt_out=one_hot("opt_out", "no"))
    opted = engine(fast=provider, live=[("opt_out", "fast")], records=[record("opt_out", "fast", provider)],
                   store=store)
    judge = Judge("yes")
    settled = await opted.decide("opt_out", {"text": "see you at five"}, deliberate=judge)
    await opted.drain()
    assert (settled.tier, settled.label, judge.asked) == ("fast", "no", 0)


async def test_a_shadow_tier_never_asks_a_compare_beyond_the_existing_path(store):
    """In shadow the existing path decides and is asked exactly once, as before: nothing more is asked beside."""
    provider = Provider(owner_send=SURE_DIGEST)
    made = engine(reader=provider, records=[record("owner_send", "reader", provider)], store=store)
    judge = Judge("send_now")
    decision = await made.decide("owner_send", DAY_ALERT, deliberate=judge)
    await made.drain()
    assert (decision.tier, decision.label, judge.asked) == ("deliberate", "send_now", 1)
    assert store.rows(point="owner_send")[0]["tiers"]["deliberate"]["outcome"] == "answered"


INSTRUCTION = {"message": "stop the check-ins please"}


def live_instruction(store):
    """The owner_instruction point (on the owner's turn) with a live reader that settles ``stop``."""
    provider = Provider(owner_instruction=one_hot("owner_instruction", "stop"))
    return engine(reader=provider, live=[("owner_instruction", "reader")],
                  records=[record("owner_instruction", "reader", provider)], store=store)


async def test_on_the_owners_turn_the_compare_runs_in_the_background_one_at_a_time(store):
    """The owner waits on the turn, so the settled answer never waits for the compare; a second settle while one
    is still running is not compared (``busy``: unmeasured), so at most one such call is in flight per point."""
    made = live_instruction(store)
    gate = asyncio.Event()
    judge = Judge("none", gate=gate)
    first = await made.decide("owner_instruction", INSTRUCTION, deliberate=judge)
    second = await made.decide("owner_instruction", INSTRUCTION, deliberate=judge)
    assert (first.tier, first.label, second.tier, second.label) == ("reader", "stop", "reader", "stop")
    await asyncio.sleep(0)
    assert (judge.asked, judge.answered) == (1, 0)
    gate.set()
    await made.drain()
    outcomes = sorted(row["tiers"]["deliberate"]["outcome"] for row in store.rows(point="owner_instruction"))
    assert outcomes == ["busy", "compared"] and judge.asked == 1
    third = await made.decide("owner_instruction", INSTRUCTION, deliberate=judge)
    await made.drain()
    assert third.label == "stop" and judge.asked == 2         # free again once the first finished


async def test_closing_the_engine_cancels_a_compare_in_flight(store):
    made = live_instruction(store)
    judge = Judge("none", gate=asyncio.Event())
    await made.decide("owner_instruction", INSTRUCTION, deliberate=judge)
    await asyncio.sleep(0)
    await made.aclose()
    assert judge.asked == 1 and judge.answered == 0
    row, = store.rows(point="owner_instruction")
    assert row["tiers"]["deliberate"]["outcome"] != "compared"


async def test_cancelling_the_caller_cancels_its_compare_and_leaves_nothing_running(store):
    """Off the owner's turn the compare is the caller's own call: a caller cut short (the night's deadline) takes
    it down with it, and nothing runs on after."""
    made, _ = live(store)
    judge = Judge("drop", gate=asyncio.Event())
    caller = asyncio.get_running_loop().create_task(made.decide("owner_send", DAY_ALERT, deliberate=judge))
    try:
        await asyncio.wait_for(judge.started.wait(), 5)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(caller, 5)
    finally:
        judge.gate.set()
        caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)
    assert not made._tasks and judge.answered == 0


async def test_cancellation_as_provider_finishes_cannot_enter_an_unbounded_compare(store, monkeypatch):
    made, provider = live(store)
    original = provider.answer
    judge = Judge("drop", gate=asyncio.Event())
    caller = None

    async def answer(*args, **kwargs):
        result = await original(*args, **kwargs)
        # With wait_for this cancels its parent just before the completed child's
        # waiter wakes. The cancelled parent must never continue into the judge.
        asyncio.get_running_loop().call_soon(caller.cancel)
        return result

    monkeypatch.setattr(provider, "answer", answer)
    caller = asyncio.create_task(made.decide("owner_send", DAY_ALERT, deliberate=judge))
    try:
        done, _ = await asyncio.wait({caller}, timeout=1)
        assert done, "caller cancellation was lost; compare is still waiting"
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert not made._tasks
    finally:
        judge.gate.set()
        caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)


class Night:
    """The night's accounting as its own call does it: a call must fit in what is left (checked before it is
    made), and its use is charged after it answers."""

    def __init__(self, budget, cost):
        self.budget, self.cost, self.tokens, self.calls = budget, cost, 0, 0

    async def call(self):
        if self.tokens + self.cost > self.budget:
            return None
        await asyncio.sleep(0)                   # the model call
        self.tokens += self.cost
        self.calls += 1
        return {"label": "adds"}


async def test_the_nights_compares_stay_inside_its_budget_and_its_run(store):
    """Three settles in a row, each compared with the night's call, under a budget for two: each compare runs in
    turn and is charged before the next is checked, so the third is refused by the night's own check, and nothing
    is left running when the night is done."""
    provider = fast_provider(claim_relation={"adds": 0.99, "replaces": 0.01})
    made = engine(fast=provider, live=[("claim_relation", "fast")],
                  records=[record("claim_relation", "fast", provider)], store=store)
    night = Night(budget=80, cost=40)
    fields = {"about": "your office", "older": "room 4", "older_on": "", "newer": "room 7", "newer_on": ""}
    for _ in range(3):
        decision = await made.decide("claim_relation", fields, deliberate=night.call)
        assert (decision.tier, decision.label) == ("fast", "adds")
    assert (night.tokens, night.calls) == (80, 2) and not made._tasks
    await made.drain()
    assert night.tokens == 80
    outcomes = sorted(row["tiers"]["deliberate"]["outcome"] for row in store.rows(point="claim_relation"))
    assert outcomes == ["compared", "compared", "no_answer"]
