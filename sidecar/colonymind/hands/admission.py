"""The sidecar half of admission. Executor slots stay with the plugin."""

from datetime import timedelta
import secrets

from . import budgets, locks


ACTIVE = ("reserved", "dispatched", "running", "checking", "integrating")


async def reserve(c, hand, task, family, *, charge=True):
    reason = await budgets.reason(c, task, family)
    if reason:
        return await defer(c, hand, task, reason)
    for dependency in task["depends_on"]:
        row = await c.one("SELECT state FROM tasks WHERE id=?", (dependency,))
        if row is None or row["state"] in {"cancelled", "failed"}:
            result = await defer(c, hand, task, "dependency_failed", refuse=True)
            await c.update("tasks", "id", task["id"], state="failed", closed_at=c.now())
            await c._ready(task["id"])
            return result
        if row["state"] != "done":
            return await defer(c, hand, task, "dependency")
    lock_reason = await locks.check(c, hand["id"], task["resources"])
    if lock_reason:
        return await defer(c, hand, task, lock_reason, refuse=lock_reason == "lock_queue_full")
    rows = await c.all("SELECT h.id, r.family FROM hands h JOIN reservations r ON r.hand_id=h.id "
                       "WHERE h.state IN ('reserved','dispatched','running','checking','integrating')")
    occupied = sum(row["family"] == family for row in rows)
    cap = c.config["health"]["family_caps"].get(family, c.config["health"]["default_family_cap"])
    reason = ("capacity" if len(rows) >= c.config["max_running"] else
              "family_capacity" if occupied >= cap else c.health.admit(family, hand["priority"], occupied))
    if reason:
        return await defer(c, hand, task, reason)
    token = secrets.token_hex(24)
    expires = (c.clock() + timedelta(seconds=c.config["lease_seconds"])).isoformat()
    await locks.acquire(c, hand["id"], task["resources"])
    await c.execute("DELETE FROM reservations WHERE hand_id=?", (hand["id"],))
    await c.insert("reservations", hand_id=hand["id"], token=token, family=family,
                   **{"class": hand["priority"]}, expires_at=expires, state="held")
    await c.update("hands", "id", hand["id"], state="reserved", activity=None, outcome=None,
                   decisions={**hand["decisions"], "admission": {"lease_token": token, "expires_at": expires, "family": family}})
    if charge:
        await budgets.charge_reservation(c, hand, task)
    return "start", None, token


async def defer(c, hand, task, reason, refuse=False):
    limits = c.config["queue"]
    rows = await c.all("SELECT priority FROM hands WHERE state='planned' AND activity='queued' AND id!=?", (hand["id"],))
    priority = hand["priority"]
    cap = limits["max_owner"] if priority == "owner" else limits["max_own"]
    full = len(rows) >= limits["max_queued"] or sum(r["priority"] == priority for r in rows) >= cap
    refuse = refuse or reason in {"task_budget", "unreconciled_effect", "parked", "contract"} or full or priority in {"auto", "upkeep"}
    await c.update("hands", "id", hand["id"], state="refused" if refuse else "planned",
                   activity=None if refuse else "queued", activity_since=c.now(), outcome=reason,
                   ended_at=c.now() if refuse else None)
    await c.update("tasks", "id", task["id"], state="waiting")
    if refuse:
        await locks.release(c, hand["id"])
        await c._ready(task["id"])
    if hand["outcome"] != reason or hand["state"] != ("refused" if refuse else "planned"):
        await c.log(hand["id"], "refused" if refuse else "queued", {"reason": reason})
        if (task["priority"] == "owner" and refuse) or reason in {"task_budget", "daily_budget", "parked"}:
            await c._notice(f"ntc:{hand['id']}:admission", hand["id"], "actionable",
                            f"{hand['name']} waiting: {reason}")
    return "refuse" if refuse else "queued", reason, None
