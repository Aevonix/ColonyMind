"""A continuation is one cascade, charged once when its plan is admitted."""

from datetime import timedelta, timezone


async def used(c, task):
    if "cascades" in task["usage"]:
        return task["usage"]["cascades"]
    # Existing attempts count too. This seeds the durable counter on its first new admission.
    row = await c.one("SELECT count(*) AS n FROM hands WHERE task_id=? AND continues IS NOT NULL "
                      "AND state!='refused' AND json_extract(decisions,'$.cascade_admitted_at') IS NULL",
                      (task["id"],))
    return row["n"]


async def reason(c, task):
    limits = c.config["cascade"]
    if await used(c, task) >= limits["max_per_task"]:
        return "cascade_task_limit"
    since = (c.clock().astimezone(timezone.utc) - timedelta(hours=1)).isoformat()
    row = await c.one("SELECT count(*) AS n FROM hands WHERE continues IS NOT NULL AND state!='refused' "
                      "AND julianday(coalesce(json_extract(decisions,'$.cascade_admitted_at'),created_at)) "
                      ">julianday(?)", (since,))
    if row["n"] >= limits["max_per_hour"]:
        return "cascade_hour_limit"
    return None


async def charge(c, hand, task, previous_count):
    current = await c.get_task(task["id"])
    await c.update("tasks", "id", task["id"], usage={**current["usage"], "cascades": previous_count + 1})
    current_hand = await c.get_hand(hand["id"])
    await c.update("hands", "id", hand["id"], decisions={**current_hand["decisions"], "cascade_admitted_at": c.now()})
    await c.log(hand["id"], "cascade_admitted", {"previous": hand["continues"], "task_count": previous_count + 1})
