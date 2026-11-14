"""Expire settled controller history while retaining authority and replay tombstones."""

import json
import logging
from datetime import timedelta, timezone


logger = logging.getLogger(__name__)
RETENTION_BATCH_SIZE = 256
# Unfinished work, debt, effects and provenance must keep their supporting rows.
SETTLED_TASK = """
    t.state IN ('done','cancelled','failed') AND t.closed_at IS NOT NULL
    AND NOT EXISTS (SELECT 1 FROM task_definitions d WHERE d.task_id=t.id)
    AND NOT EXISTS (SELECT 1 FROM obligations o WHERE o.task_id=t.id
                    AND o.state IN ('open','result_ready','reviewed','accepted'))
    AND NOT EXISTS (SELECT 1 FROM hands h WHERE h.task_id=t.id AND
                    (h.state NOT IN ('done','below_minimum','failed','stopped','superseded','lost','unknown_effect','refused','expired')
                     OR (h.state='unknown_effect' AND coalesce(json_extract(h.decisions,'$.effect_reconciled'),0)!=1)))
    AND NOT EXISTS (SELECT 1 FROM effects e JOIN hands h ON h.id=e.hand_id
                    WHERE h.task_id=t.id AND e.done_at IS NULL)
    AND NOT EXISTS (SELECT 1 FROM questions q JOIN hands h ON h.id=q.hand_id
                    WHERE h.task_id=t.id AND q.state='open')
    AND NOT EXISTS (SELECT 1 FROM permits p WHERE p.task_id=t.id AND p.state IN ('waiting','granted'))
    AND NOT EXISTS (SELECT 1 FROM notices n JOIN hands h ON h.id=n.hand_id
                    WHERE h.task_id=t.id AND n.sent_at IS NULL)
    AND NOT EXISTS (SELECT 1 FROM derivations d JOIN hands h ON h.id=d.hand_id WHERE h.task_id=t.id)
"""


class RetentionOperations:
    async def retention_status(self):
        row = await self.one("SELECT at,payload FROM maintenance WHERE id='retention'")
        return row or {"at": None, "payload": {"events": 0, "tasks": 0, "hands": 0, "replays": 0}}

    async def _prune_retained(self):
        """Prune one bounded batch per hour inside an isolated sweep savepoint."""
        now = self.clock()
        prior_run = await self.retention_status()
        result = {"events": 0, "tasks": 0, "hands": 0, "replays": 0}
        if prior_run["at"] and (now - self.parse_time(prior_run["at"])).total_seconds() < 3600:
            return result
        limits = self.config["retention"]
        event_cutoff = (now - timedelta(days=limits["events_days"])).astimezone(timezone.utc).isoformat()
        row_cutoff = now.timestamp() - limits["rows_days"] * 86400
        await self.execute("DELETE FROM events WHERE rowid IN (SELECT rowid FROM events WHERE at < ? "
                           f"AND hand_id IN (SELECT h.id FROM hands h JOIN tasks t ON t.id=h.task_id WHERE {SETTLED_TASK}) "
                           "ORDER BY at LIMIT ?)", (event_cutoff, RETENTION_BATCH_SIZE))
        result["events"] = (await self.one("SELECT changes() AS n"))["n"]

        cooldown = self.config["cooldowns"]
        resource_cutoff = self.clock().timestamp() - cooldown["resource_window_seconds"] - cooldown["resource_seconds"]
        rows = await self.all(f"SELECT t.id,t.origin,t.origin_ref FROM tasks t WHERE {SETTLED_TASK} "
            "AND julianday(t.closed_at)<julianday(?,'unixepoch') "
            "AND NOT EXISTS (SELECT 1 FROM hands h WHERE h.task_id=t.id AND "
            "(julianday(coalesce(h.last_event_at,h.ended_at,h.created_at))>=julianday(?,'unixepoch') "
            "OR julianday(h.created_at)>julianday(?,'unixepoch') "
            "OR (h.state='failed' AND julianday(h.ended_at)>=julianday(?,'unixepoch')) "
            "OR json_extract(h.usage,'$.admitted_day')=? "
            "OR json_extract(h.usage,'$.daily_tokens.' || ?) IS NOT NULL)) "
            "AND NOT EXISTS (SELECT 1 FROM tasks child,json_each(child.depends_on) dep WHERE dep.value=t.id) "
            "LIMIT ?",
            (row_cutoff, row_cutoff, now.timestamp() - 3600, resource_cutoff, self.day(), self.day(), RETENTION_BATCH_SIZE))
        identifiers = set()
        for row in rows:
            tid = row["id"]
            identifiers.add(tid)
            hands = await self.all("SELECT id,child_session,parent_session,epoch FROM hands WHERE task_id=?", (tid,))
            for hand in hands:
                identifiers.add(hand["id"])
                # Every old alias remains a hand identity even after its attempt expires.
                await self.execute("INSERT OR IGNORE INTO hand_session_tombstones "
                    "SELECT session_id,hand_id,parent_session,epoch FROM hand_sessions WHERE hand_id=?", (hand["id"],))
                if hand["child_session"]:
                    await self.execute("INSERT OR IGNORE INTO hand_session_tombstones VALUES (?,?,?,?)",
                        (hand["child_session"], hand["id"], hand["parent_session"], json.dumps(hand["epoch"])))
                for table in ("questions", "permits", "notices"):
                    identifiers.update(r["id"] for r in await self.all(f"SELECT id FROM {table} WHERE hand_id=?", (hand["id"],)))
            identifiers.update(r["id"] for r in await self.all("SELECT id FROM obligations WHERE task_id=?", (tid,)))
            result["hands"] += len(hands)
            if row["origin"] == "owner_turn":
                await self.execute("INSERT INTO work_refresh_origins(origin_ref) VALUES (?) "
                                   "ON CONFLICT(origin_ref) DO UPDATE SET generation=generation+1", (json.dumps(row["origin_ref"]),))
            await self.execute("DELETE FROM tasks WHERE id=?", (tid,))
            await self.execute("DELETE FROM work_refresh_tasks WHERE task_id=?", (tid,))
            result["tasks"] += 1

        # Keep each key and scope, but expire old answers and every answer referring to a deleted record.
        # A tombstone can never replay an old lease or create the task again.
        from .controller import encode
        query = "SELECT key,path,response,created_at FROM idempotency WHERE status_code!=410"
        replays = await self.all(query) if identifiers else await self.all(
            query + " AND julianday(created_at)<julianday(?,'unixepoch')", (row_cutoff,))
        for row in replays:
            if row["response"] in ({"expired": True}, {"erased": True}):
                continue
            serialized = row["path"] + encode(row["response"])
            if (self.parse_time(row["created_at"]).timestamp() < row_cutoff
                    or any(ident in serialized for ident in identifiers)):
                await self.update("idempotency", "key", row["key"], response={"expired": True}, status_code=410)
                result["replays"] += 1

        prior = prior_run["payload"]
        totals = {key: prior.get("totals", {}).get(key, 0) + count for key, count in result.items()}
        payload = {**result, "totals": totals, "events_days": limits["events_days"], "rows_days": limits["rows_days"]}
        await self.execute("INSERT INTO maintenance(id,at,payload) VALUES ('retention',?,?) "
                           "ON CONFLICT(id) DO UPDATE SET at=excluded.at,payload=excluded.payload",
                           (self.now(), encode(payload)))
        if any(result.values()):
            logger.info("Hands retention: %s", result)
        return result
