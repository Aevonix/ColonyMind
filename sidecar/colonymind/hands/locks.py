"""Exclusive resources. Caller holds the controller write transaction."""


async def check(c, hand, resources):
    blocked = []
    # A third writer is refused by default: one waiter, within the schema's
    # hard maximum of two; hosts may explicitly choose two.
    maximum = max(0, min(2, c.config["locks"]["max_waiters"], c.config["locks"].get("admission_waiters", 1)))
    for resource in sorted(set(resources)):
        row = await c.one("SELECT * FROM locks WHERE resource=?", (resource,))
        if row and row["holder_hand"] != hand:
            if hand not in row["waiters"] and len(row["waiters"]) >= maximum:
                return "lock_queue_full"
            blocked.append(row)
    if blocked:
        for row in blocked:
            if hand not in row["waiters"]:
                await c.update("locks", "resource", row["resource"], waiters=row["waiters"] + [hand])
        return "lock"
    return None


async def acquire(c, hand, resources):
    for resource in sorted(set(resources)):
        row = await c.one("SELECT * FROM locks WHERE resource=?", (resource,))
        if row and row["holder_hand"] != hand:
            raise RuntimeError("resource was not checked in the reservation transaction")
        if row is None:
            await c.insert("locks", resource=resource, holder_hand=hand, since=c.now(), waiters=[])


async def release(c, hand):
    for row in await c.all("SELECT * FROM locks"):
        if row["holder_hand"] == hand:
            # The sweeper admits waiters in creation order through all checks.
            await c.execute("DELETE FROM locks WHERE resource=?", (row["resource"],))
        elif hand in row["waiters"]:
            await c.update("locks", "resource", row["resource"], waiters=[h for h in row["waiters"] if h != hand])
