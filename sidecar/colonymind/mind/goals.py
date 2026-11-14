"""Agent-owned goals: a desire the agent pursues across days (architecture 4.5).

A goal is one ``kind='goal'`` row in the intentions table: a description, a
``success_check`` the mind can evaluate over its own state, a task budget
and a horizon. Curiosity and mastery may adopt one; at most
``budgets.open_goals`` are open at once (a third is not adopted while two
are open). Each tick may form the goal's next step as an ordinary ``task``
intention with kanban ``goal_mode``. The goal closes when its check passes
(satisfied), when its task budget is spent, or when its horizon passes
(expired). Authority applies to every step as usual.

A goal may carry a ``plan`` (a granted wish's goal does, ``wishes.py``): its steps are then formed from
the plan in order, each exactly as planned and without a deliberation call. A planned step that did not
finish is tried once more inside the task budget; one the owner said no to, or authority dropped, is
never formed again, and the plan stops there (a dropped step would otherwise keep its key and hold the
goal's slot until the horizon). When every planned step is done or spent, or one was refused, the goal closes (its
check, or ``expired``).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

from colonymind.initiatives.models import MIND_ACTIVE_STATUSES, StoredInitiative

from .drives import slug
from .rank import Candidate
from colonymind.util.temporal import now_utc

logger = logging.getLogger(__name__)

GOAL_TYPE = "goal"
STEP_TYPE = "goal_step"
OPEN_STATUSES = ("approved", "asked", "proposed")   # a goal owns no Hermes object; approved means open
DEFAULT_HORIZON_DAYS = 7
DEFAULT_TASKS = 4
DEFAULT_MAX_TURNS = 12
PLAN_TRIES = 2          # a planned step that did not finish is tried once more, inside the task budget
# Recurring self-work a goal may grow from: a topic researched again in a later week (``Goals.recurring``).
RECURRING_TYPES = frozenset({"research"})


def goal_context(row: StoredInitiative) -> Dict[str, Any]:
    return row.context if isinstance(row.context, dict) else {}


def plan_of(goal: StoredInitiative) -> List[Dict[str, Any]]:
    """The goal's planned steps, or [] for a goal whose steps are deliberated one at a time."""
    plan = goal_context(goal).get("plan")
    if not isinstance(plan, list) or not all(isinstance(item, dict) and item.get("body") for item in plan):
        return []
    return plan


def _plan_index(row: StoredInitiative) -> Optional[int]:
    index = goal_context(row).get("plan_step")
    return None if isinstance(index, bool) or not isinstance(index, int) else index


def refused(row: StoredInitiative) -> Optional[str]:
    """Who refused this step, or None: authority dropped it when it was formed (the deny list, the off
    switch: the same step would be dropped again), or the owner said no to it."""
    if row.status == "dropped":
        return "authority"
    return "owner" if row.outcome == "denied" else None


class Goals:
    def __init__(self, store: Any, *, budgets: Any, clock: Callable[[], datetime] | None = None,
                 enabled: bool = True) -> None:
        self.store = store
        self.budgets = budgets
        self.clock = clock or (lambda: now_utc())
        self.enabled = enabled

    # -- reads -----------------------------------------------------------------------

    def open(self) -> List[StoredInitiative]:
        rows = self.store.intentions(status=list(OPEN_STATUSES), kind=[GOAL_TYPE], limit=200)
        return sorted(rows, key=lambda row: row.created_at)

    def approved(self) -> List[StoredInitiative]:
        """The open goals that may own work: one still asked or deferred has no steps yet."""
        return [goal for goal in self.open() if goal.status == "approved"]

    def may_adopt(self) -> bool:
        return bool(self.enabled) and len(self.open()) < int(getattr(self.budgets, "open_goals", 2) or 0)

    def steps(self, goal: StoredInitiative) -> List[StoredInitiative]:
        rows = [row for row in self.store.intentions(kind=["task"], limit=500)
                if row.parent_goal_id == goal.id]
        return sorted(rows, key=lambda row: row.created_at)

    def running_step(self, goal: StoredInitiative) -> Optional[StoredInitiative]:
        for row in self.steps(goal):
            if row.status in MIND_ACTIVE_STATUSES:
                return row
        return None

    def steps_done(self, goal: StoredInitiative) -> List[StoredInitiative]:
        return [row for row in self.steps(goal) if row.outcome == "done"]

    def summaries(self, goal: StoredInitiative) -> List[str]:
        return [str(row.result or row.description)[:300] for row in self.steps_done(goal)]

    def planned_steps(self, goal: StoredInitiative) -> List[StoredInitiative]:
        """The goal's step rows formed from its plan (each carries its ``plan_step`` index)."""
        return [row for row in self.steps(goal) if _plan_index(row) is not None]

    def plan_refused(self, goal: StoredInitiative) -> Optional[str]:
        """Who refused one of its planned steps (``refused``), or None: the plan does not go on without it."""
        return next((who for who in map(refused, self.planned_steps(goal)) if who), None)

    def next_planned(self, goal: StoredInitiative) -> Optional[int]:
        """The index of the next planned step: the first not done and tried fewer than ``PLAN_TRIES`` times;
        None once a planned step was refused (a repeat of it is the same question, and the steps after it
        build on it)."""
        plan = plan_of(goal)
        done, tries = set(), {}
        for row in self.planned_steps(goal):
            if refused(row):
                return None
            index = _plan_index(row)
            tries[index] = tries.get(index, 0) + 1
            if row.outcome == "done":
                done.add(index)
        return next((index for index in range(len(plan)) if index not in done and tries.get(index, 0) < PLAN_TRIES),
                    None)

    def plan_spent(self, goal: StoredInitiative) -> bool:
        return bool(plan_of(goal)) and self.next_planned(goal) is None

    def budget_spent(self, goal: StoredInitiative) -> bool:
        limit = int(goal_context(goal).get("tasks") or DEFAULT_TASKS)
        used = [row for row in self.steps(goal) if row.status != "dropped"]
        return len(used) >= limit

    # -- adoption ---------------------------------------------------------------------

    @staticmethod
    def key_for(candidate: Candidate) -> str:
        """The goal key of a concern's topic: one goal per topic, whatever became of an earlier one."""
        return f"goal:{slug(candidate.topic or candidate.concern or candidate.title)}"

    def taken(self, candidate: Candidate) -> bool:
        return self.store.get_by_dedup_key(self.key_for(candidate)) is not None

    def recurring(self, candidate: Candidate) -> bool:
        """A research topic whose research task already finished in an earlier week: the interest came back
        after its work was done, so the work on it is a goal when a slot is free (``deliberate.apply_proposal``).
        It decides only that the topic warrants a goal: the goal budget, one goal per topic and the drive still
        decide whether one is adopted."""
        if candidate.type not in RECURRING_TYPES or not candidate.dedup_base or candidate.parent_goal_id:
            return False
        return self.store.finished_under_base(candidate.dedup_base, other_than=candidate.dedup_key)

    def candidate(self, candidate: Candidate) -> Optional[Candidate]:
        """Turn a deliberated goal proposal into the goal intention's candidate, or None when the
        budget is full or a goal for the topic already exists (the tick does the one task instead)."""
        if (candidate.kind != "goal" or not isinstance(candidate.goal, dict) or not self.may_adopt()
                or self.taken(candidate)):
            return None
        proposal = candidate.goal
        horizon_days = int(proposal.get("horizon_days") or DEFAULT_HORIZON_DAYS)
        tasks = int(proposal.get("tasks") or DEFAULT_TASKS)
        topic = candidate.topic or candidate.concern or candidate.title
        return Candidate(
            type=GOAL_TYPE, drive=candidate.drive, kind="goal", title=f"Goal: {candidate.title}"[:160],
            dedup_key=self.key_for(candidate), salience=candidate.salience, cost=candidate.cost,
            text=str(proposal.get("description") or candidate.text)[:2000], rationale=candidate.rationale,
            evidence=list(candidate.evidence), concern=candidate.concern, topic=topic,
            success_check=proposal.get("success_check") or {"kind": "steps_done", "count": 2},
            source_type=candidate.source_type, source_id=candidate.source_id, concern_kind="goal",
            goal={"description": str(proposal.get("description") or candidate.text)[:2000],
                  "horizon_days": horizon_days, "tasks": tasks, "topic": topic},
            due_at=self.clock() + timedelta(days=horizon_days), cost_tokens=candidate.cost_tokens)

    # -- steps ------------------------------------------------------------------------

    def next_step(self, goal: StoredInitiative) -> Optional[Candidate]:
        """The next step as an open-ended task candidate, or None while one runs or the budget is spent."""
        if self.running_step(goal) is not None or self.budget_spent(goal):
            return None
        context = goal_context(goal)
        done = self.steps_done(goal)
        ordinal = len([row for row in self.steps(goal) if row.status != "dropped"]) + 1
        topic = str(context.get("topic") or goal.description)
        if plan_of(goal):
            return self._planned_step(goal, ordinal, topic)
        return Candidate(
            type=STEP_TYPE, drive=goal.drive or "curiosity", kind="task",
            title=f"Step {ordinal} of goal: {topic}"[:160], dedup_key=f"goal:{goal.id}:step:{ordinal}",
            salience=max(0.8, float(goal.priority or 0.5)), cost=0.2, rationale=f"step {ordinal} of an adopted goal",
            evidence=[f"goal:{goal.id}", *[f"step done: {row.result or row.description}"[:200] for row in done[-3:]]],
            concern=str(context.get("description") or goal.description)[:300], topic=topic, open_ended=True,
            concern_kind="goal_step", parent_goal_id=goal.id, source_type="goal", source_id=goal.id)

    def _planned_step(self, goal: StoredInitiative, ordinal: int, topic: str) -> Optional[Candidate]:
        """The next planned step, as planned: a closed-form task (no deliberation call) of the plan's type."""
        index = self.next_planned(goal)
        if index is None:
            return None
        plan = plan_of(goal)
        item = plan[index]
        return Candidate(
            type=str(item.get("type") or STEP_TYPE), drive=goal.drive or "curiosity", kind="task",
            title=f"Step {ordinal} of goal: {item.get('title') or topic}"[:160],
            dedup_key=f"goal:{goal.id}:step:{ordinal}", salience=max(0.8, float(goal.priority or 0.5)), cost=0.2,
            rationale=f"step {ordinal} of an adopted goal: planned step {index + 1} of {len(plan)}",
            evidence=[f"goal:{goal.id}", *[str(ref) for ref in item.get("evidence") or []][:8]],
            concern=str(goal_context(goal).get("description") or goal.description)[:300], topic=topic,
            text=str(item["body"]), open_ended=False, success_check=item.get("success_check"),
            concern_kind="goal_step", parent_goal_id=goal.id,
            source_type=str(item.get("source_type") or "goal"), source_id=str(item.get("source_id") or goal.id),
            extra={"plan_step": index, "plan_kind": str(item.get("kind") or "step")})

    # -- closing ----------------------------------------------------------------------

    def check(self, goal: StoredInitiative, *, evaluate: Callable[..., Optional[bool]]) -> Optional[bool]:
        """The goal's success check over mind-observable state; None when it cannot run."""
        steps_done = self.steps_done(goal)
        latest = steps_done[-1] if steps_done else None
        return evaluate(goal.success_check, summary=str(latest.result or "") if latest else "",
                        result=(latest.result_metadata or {}).get("result") if latest else None,
                        steps_done=len(steps_done))

    def due(self, now: datetime, *, evaluate: Callable[..., Optional[bool]]) -> List[Dict[str, Any]]:
        """Goals to close now: ``satisfied`` (check passed), ``expired`` (horizon or budget)."""
        closing: List[Dict[str, Any]] = []
        for goal in self.open():
            passed = self.check(goal, evaluate=evaluate)
            if passed is True:
                closing.append({"goal": goal, "outcome": "done", "why": "the success check passed"})
                continue
            horizon = goal.due_at or goal.expires_at
            if horizon is not None and horizon <= now:
                closing.append({"goal": goal, "outcome": "expired", "why": "the horizon passed"})
            elif self.budget_spent(goal) and self.running_step(goal) is None:
                closing.append({"goal": goal, "outcome": "expired", "why": "the task budget is spent"})
            elif self.plan_spent(goal) and self.running_step(goal) is None:
                why = {"owner": "the owner declined one of its planned steps",
                       "authority": "authority refused one of its planned steps"}.get(
                    self.plan_refused(goal) or "", "its planned steps are spent")
                closing.append({"goal": goal, "outcome": "expired", "why": why})
        return closing

    def render(self, goal: StoredInitiative) -> Dict[str, Any]:
        context = goal_context(goal)
        steps = self.steps(goal)
        return {
            "id": goal.id, "title": goal.description, "drive": goal.drive, "status": goal.status,
            "topic": context.get("topic"), "description": context.get("description"),
            "steps_done": len([row for row in steps if row.outcome == "done"]),
            "steps": len([row for row in steps if row.status != "dropped"]),
            "tasks": int(context.get("tasks") or DEFAULT_TASKS),
            "horizon": (goal.due_at or goal.expires_at).isoformat() if (goal.due_at or goal.expires_at) else None,
            "success_check": goal.success_check, "outcome": goal.outcome, "created_at": goal.created_at.isoformat(),
        }


def goal_lines(goals: Iterable[Dict[str, Any]], *, limit: int = 2) -> List[str]:
    lines = []
    for goal in list(goals)[:limit]:
        lines.append(f"{goal['title']} ({goal['steps_done']}/{goal['tasks']} steps, until {str(goal['horizon'])[:10]})")
    return lines


__all__ = ["DEFAULT_HORIZON_DAYS", "DEFAULT_MAX_TURNS", "DEFAULT_TASKS", "GOAL_TYPE", "Goals", "OPEN_STATUSES",
           "PLAN_TRIES", "STEP_TYPE", "goal_context", "goal_lines", "plan_of"]
