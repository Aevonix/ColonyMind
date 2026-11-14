"""Attempt, task and initiative budgets share the task's durable counters."""

from datetime import timedelta


def attempt(config, kind, effort=2):
    values = dict(config["budgets"]["attempt"][kind])
    if effort == 3:
        for key in ("tokens", "wall_seconds"):
            values[key] *= config["budgets"]["effort_3_multiplier"]
    return values


def task_budget(config, first):
    limits = config["budgets"]["task"]
    return {"tokens": first["tokens"] * limits["token_multiplier"],
            "wall_seconds": limits["wall_seconds"], "hands": limits["max_hands"]}


async def reason(c, task, family):
    usage, limit = task["usage"], task["budget"]
    if any(usage.get(key, 0) >= limit[key] for key in ("tokens", "wall_seconds", "hands")):
        return "task_budget"
    cooldown = c.config["cooldowns"]
    if task["failures_in_row"] >= cooldown["task_park_failures"]:
        return "parked"
    if task["cooldown_until"] and task["cooldown_until"] > c.now():
        return "cooldown"
    since = (c.clock() - timedelta(seconds=cooldown["resource_window_seconds"])).isoformat()
    failures = await c.all("SELECT DISTINCT t.id, t.resources, h.ended_at FROM hands h JOIN tasks t ON t.id=h.task_id "
                           "WHERE h.state='failed' AND h.ended_at>=?", (since,))
    for resource in task["resources"]:
        affected = {row["id"] for row in failures if resource in row["resources"]}
        recent = [row["ended_at"] for row in failures if resource in row["resources"]]
        if len(affected) >= cooldown["resource_failures"] and recent:
            if c.parse_time(max(recent)) + timedelta(seconds=cooldown["resource_seconds"]) > c.clock():
                return "resource_cooldown"
    if task["priority"] == "owner":
        return None
    day = c.day()
    rows = await c.all("SELECT usage FROM hands WHERE priority!='owner'")
    count = sum(row["usage"].get("admitted_day") == day for row in rows)
    limits = c.config["budgets"]["day"]
    tokens = sum(row["usage"].get("daily_tokens", {}).get(day, {}).get(family, 0) for row in rows)
    token_cap = limits["tokens_per_family"].get(family, limits["default_tokens_per_family"])
    if count >= limits["max_hands"] or tokens >= token_cap:
        return "daily_budget"
    return None


async def charge_reservation(c, hand, task):
    usage = dict(task["usage"])
    usage["hands"] = usage.get("hands", 0) + 1
    await c.update("tasks", "id", task["id"], usage=usage, state="open")
    spent = dict(hand["usage"])
    spent["admitted_day"] = c.day()
    await c.update("hands", "id", hand["id"], usage=spent)


async def outcome(c, hand, task, failed):
    failures = task["failures_in_row"] + 1 if failed else 0
    cooldown_until = None
    if failures >= c.config["cooldowns"]["task_failures"]:
        cooldown_until = (c.clock() + timedelta(seconds=c.config["cooldowns"]["task_seconds"])).isoformat()
    await c.update("tasks", "id", task["id"], failures_in_row=failures, cooldown_until=cooldown_until,
                   state="waiting" if cooldown_until else "open")
