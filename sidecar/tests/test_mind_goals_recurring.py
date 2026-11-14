"""Goals form from recurring topics.

A research topic whose research task already finished in an earlier week becomes a goal when a goal slot is free,
whatever the number of steps the proposal lists. Every other limit stays: the goal budget, one goal per topic, the
drives that may own a goal, and a goal's own steps are never goals. The proposer's prompt is unchanged, and so is
the curiosity weight.
"""

from __future__ import annotations

from datetime import timedelta

from colonymind.mind.authority import Budgets
from colonymind.mind.concerns import Concern
from colonymind.mind.deliberate import GOAL_TASKS, apply_proposal
from colonymind.mind.drives import period
from colonymind.mind.rank import Candidate
from test_mind_drives_loop import DeliberationRouter, fx, idle, research  # noqa: F401  (fixture)
from test_mind_loop import pinned_clock  # noqa: F401  (autouse pytest fixture: fx builds a Fixture)

TOPIC = "local history"
PROMPT_RULE = ("You may propose a goal only if the concern warrants work over several days with a checkable end.")


async def researched(fx, topic=TOPIC, *, outcome="done"):
    """One week's research of the topic, settled: done with a finding, or failed."""
    first, = await idle(fx)
    [formed] = [item for item in first["formed"] if item["type"] == "research"]
    assert formed["kind"] == "task"                                        # a first week's topic is a task
    fx.mind.bound(formed["id"], f"kanban:{formed['id'][:8]}")
    summary = "finding: founded in 1851." if outcome == "done" else "the archive was down"
    fx.mind.outcomes.record(formed["id"], status=outcome, hermes_ref=f"kanban:{formed['id'][:8]}", summary=summary,
                            **({"error": summary} if outcome == "failed" else {}))
    return fx.store.get(formed["id"])


def goal_row(fx, key, status="approved"):
    row, created = fx.store.create_intention(
        kind="goal", type="goal", title=f"Goal: {key}", drive="curiosity", cls="internal", decision="act",
        decision_reason="test", status=status, dedup_key=key, context={"topic": key, "tasks": 2},
        hermes_kind="none", created_at=fx.now)
    assert created == "created"
    return row


async def test_a_topic_researched_in_an_earlier_week_becomes_a_goal_when_a_slot_is_free(fx):
    router = DeliberationRouter({TOPIC: research(TOPIC)})                  # one run, no steps: never a goal before
    fx.mind.router = router
    fx.mind.add_interest(TOPIC)
    first = await researched(fx)
    assert fx.mind.goals.recurring(_candidate(fx)) is False               # this week's own research is no history
    fx.shift(days=8)
    assert fx.mind.goals.recurring(_candidate(fx)) is True
    second, = await idle(fx)
    [formed] = second["formed"]
    goal = fx.store.get(formed["id"])
    assert formed["kind"] == "goal" and goal.dedup_key == "goal:local-history" and goal.drive == "curiosity"
    assert goal.context["tasks"] == GOAL_TASKS and goal.context["topic"] == TOPIC
    assert "Find out and report finding" in goal.context["description"]
    assert all(PROMPT_RULE in prompt for prompt in router.prompts)       # the proposer is told nothing new
    assert len(router.prompts) == 2 and first.dedup_key == f"research:local-history:{period(fx.now - timedelta(days=8))}"
    # The goal's steps run as steps: one task each, never a goal inside the goal.
    third, = await idle(fx)
    assert [(item["type"], item["kind"]) for item in third["formed"]] == [("goal_step", "task")]


async def test_a_topic_whose_earlier_research_did_not_finish_stays_a_task(fx):
    fx.mind.router = DeliberationRouter({TOPIC: research(TOPIC)})
    fx.mind.add_interest(TOPIC)
    await researched(fx, outcome="failed")
    fx.shift(days=8)
    assert fx.mind.goals.recurring(_candidate(fx)) is False
    second, = await idle(fx)
    assert [(item["type"], item["kind"]) for item in second["formed"]] == [("research", "task")]


async def test_a_full_goal_budget_keeps_a_recurring_topic_a_task(fx):
    router = DeliberationRouter({TOPIC: research(TOPIC)})
    fx.mind.router = router
    fx.mind.add_interest(TOPIC)
    await researched(fx)
    goal_row(fx, "goal:bees", status="asked")
    goal_row(fx, "goal:tides", status="asked")              # open (no steps while asked); open_goals: 2, the default
    fx.shift(days=8)
    second, = await idle(fx)
    assert [(item["type"], item["kind"]) for item in second["formed"]] == [("research", "task")]
    assert "Do not propose a goal" in router.prompts[-1]


async def test_a_topic_that_already_had_a_goal_keeps_a_recurring_topic_a_task(fx):
    fx.mind.router = DeliberationRouter({TOPIC: research(TOPIC)})
    fx.mind.add_interest(TOPIC)
    await researched(fx)
    goal_row(fx, "goal:local-history", status="expired")                  # one goal per topic, whatever became of it
    fx.shift(days=8)
    second, = await idle(fx)
    assert [(item["type"], item["kind"]) for item in second["formed"]] == [("research", "task")]


def _candidate(fx, **fields):
    from colonymind.mind.drives import research_candidate
    candidate = research_candidate(topic=TOPIC, why="a declared interest", sources=[], salience=0.8, now=fx.now)
    for key, value in fields.items():
        setattr(candidate, key, value)
    return candidate


def test_only_research_recurs_and_only_outside_a_goal():
    concern = Concern(id="c-1", drive="curiosity", kind="interest", summary=TOPIC, dedup_key="interest:x",
                      salience=0.8)
    proposal = {"kind": "task", "title": "Mill brief", "body": "Collect the records and report finding: <list>."}
    candidate = Candidate(type="research", drive="curiosity", kind="task", title="Research", open_ended=True,
                          dedup_key="research:mill:2026w40", topic="the mill")
    shaped = apply_proposal(candidate, concern, dict(proposal), budgets=Budgets(), may_adopt_goal=True, recurring=True)
    assert shaped.kind == "goal" and shaped.goal["tasks"] == GOAL_TASKS
    assert shaped.goal["success_check"] == {"kind": "steps_done", "count": 2}
    for may_adopt_goal, parent in ((False, None), (True, "g-1")):          # no slot, or a goal's own step
        task = apply_proposal(Candidate(type="research", drive="curiosity", kind="task", title="Research",
                                        open_ended=True, dedup_key="research:mill:2026w40", topic="the mill",
                                        parent_goal_id=parent),
                              concern, dict(proposal), budgets=Budgets(), may_adopt_goal=may_adopt_goal,
                              recurring=True)
        assert task.kind == "task" and task.goal is None
    plain = apply_proposal(Candidate(type="research", drive="curiosity", kind="task", title="Research",
                                     open_ended=True, dedup_key="research:mill:2026w40", topic="the mill"),
                           concern, dict(proposal), budgets=Budgets(), may_adopt_goal=True)
    assert plain.kind == "task"                                            # one run, not recurring: a task as before


async def test_questions_and_goal_steps_never_recur(fx):
    fx.mind.add_interest(TOPIC)
    await researched(fx)
    fx.shift(days=8)
    goals = fx.mind.goals
    assert goals.recurring(_candidate(fx)) is True
    assert goals.recurring(_candidate(fx, type="question")) is False
    assert goals.recurring(_candidate(fx, parent_goal_id="g-1")) is False
    assert goals.recurring(_candidate(fx, dedup_base=None)) is False
