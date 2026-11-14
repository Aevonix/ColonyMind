"""Owner obligations outlive sessions; silence and queue acknowledgments close none."""

import secrets

from fastapi import HTTPException


class ObligationOperations:
    def obligation_overdue(self, obligation, task):
        if obligation["state"] not in {"open", "result_ready", "reviewed", "accepted"}:
            return False
        # A revision opens a new debt; refusals and repeated results do not reset its age.
        since = task["intents"][str(task["intent_version"])]["set_at"]
        return (self.clock() - self.parse_time(since)).total_seconds() > self.config["obligations"]["overdue_seconds"]

    async def _overdue_obligations(self):
        overdue = []
        for obligation in await self.all("SELECT * FROM obligations WHERE state IN ('open','result_ready','reviewed','accepted')"):
            task = await self.get_task(obligation["task_id"])
            if task["obligation_id"] != obligation["id"] or not self.obligation_overdue(obligation, task):
                continue
            key = f"ctl:obligation:{obligation['id']}:overdue"
            if await self.one("SELECT key FROM events WHERE key=?", (key,)):
                continue
            hand = await self.one("SELECT * FROM hands WHERE task_id=? ORDER BY attempt DESC LIMIT 1", (task["id"],))
            if hand:
                await self.log(hand["id"], "obligation_overdue", {"obligation_id": obligation["id"],
                               "overdue_seconds": self.config["obligations"]["overdue_seconds"]}, key=key)
                await self._notice(f"ntc:{obligation['id']}:overdue", hand["id"], "actionable",
                                   f"{hand['name']}: answer overdue; /hands."[:self.config["notices"]["max_chars"]])
                overdue.append(obligation["id"])
        return overdue

    async def _create_obligation(self, task):
        oid = "o-" + secrets.token_hex(6)
        await self.insert("obligations", id=oid, task_id=task["id"], owner_ref=task["origin_ref"])
        await self.update("tasks", "id", task["id"], obligation_id=oid)
        return oid

    async def obligation(self, oid):
        return await self.require("obligations", oid)

    async def _ready(self, tid, accepted=False):
        task = await self.get_task(tid)
        if not task["obligation_id"]:
            return
        obligation = await self.obligation(task["obligation_id"])
        if obligation["state"] in {"open", "result_ready", "reviewed", "accepted"}:
            await self.update("obligations", "id", obligation["id"], state="result_ready",
                              result_ready_at=self.now(), reviewed_at=None,
                              accepted_at=self.now() if accepted else None)

    async def review(self, oid, *, turn_ref, successful, key, rendered=True):
        """Called from a successful front outcome, never a settled-hook alone."""
        async def apply():
            obligation = await self.obligation(oid)
            if not successful or not rendered:
                return obligation
            if obligation["state"] != "result_ready":
                raise HTTPException(status_code=409, detail="obligation_not_ready")
            await self.update("obligations", "id", oid, state="reviewed", reviewed_at=self.now())
            hand = await self.one("SELECT id FROM hands WHERE task_id=? ORDER BY attempt DESC LIMIT 1", (obligation["task_id"],))
            await self.log(hand["id"], "reviewed", {"obligation_id": oid, "turn_ref": turn_ref})
            return await self.obligation(oid)
        return await self.write(key, f"/obligations/{oid}/review", {"turn_ref": turn_ref, "successful": successful, "rendered": rendered}, apply)

    async def communicate(self, oid, turn_ref, *, silent, destination, key):
        """Automatic communication is bound to the successful review and origin chat."""
        async def apply():
            obligation = await self.obligation(oid)
            if silent:
                return obligation
            origin = obligation["owner_ref"]
            if not isinstance(origin, dict) or not all(origin.get(k) == destination.get(k) for k in ("platform", "chat")):
                raise HTTPException(status_code=409, detail="obligation_destination")
            reviews = await self.all("SELECT payload FROM events WHERE type='reviewed' AND hand_id IN "
                                    "(SELECT id FROM hands WHERE task_id=?)", (obligation["task_id"],))
            if not any(row["payload"].get("obligation_id") == oid and row["payload"].get("turn_ref") == turn_ref for row in reviews):
                raise HTTPException(status_code=409, detail="successful_review_required")
            return await self._told(oid, turn_ref)
        return await self.write(key, f"/obligations/{oid}/communicate", {"turn_ref": turn_ref, "silent": silent, "destination": destination}, apply)

    async def _told(self, oid, turn_ref):
        obligation = await self.obligation(oid)
        if obligation["state"] not in {"result_ready", "reviewed", "accepted"}:
            raise HTTPException(status_code=409, detail="obligation_not_ready")
        await self.update("obligations", "id", oid, state="communicated", communicated_at=self.now(),
                          communicated_ref=turn_ref, delivery="unknown", closed_reason="communicated")
        return await self.obligation(oid)

    async def told(self, oid, body, key):
        async def apply():
            return await self._told(oid, body.turn_ref)
        return await self.write(key, f"/obligations/{oid}/told", body, apply)

    async def delivered(self, oid, body, key):
        async def apply():
            obligation = await self.obligation(oid)
            if obligation["state"] != "communicated" or not obligation["communicated_ref"]:
                raise HTTPException(status_code=409, detail="communication_required")
            await self.update("obligations", "id", oid, state="delivered", delivery="receipt",
                              delivered_at=self.now(), delivery_ref=body.platform_message_id, closed_reason="delivered")
            return await self.obligation(oid)
        return await self.write(key, f"/obligations/{oid}/delivered", body, apply)


async def told(oid, body, idempotency_key):
    from .controller import get_controller
    return await (await get_controller()).told(oid, body, idempotency_key)


async def delivered(oid, body, idempotency_key):
    from .controller import get_controller
    return await (await get_controller()).delivered(oid, body, idempotency_key)
