"""Bounded home-channel lines and immutable delivery identifiers.

Lines in the same budget and merge window share queued_at. The sink groups
those lines into one message, then acknowledges every line with its receipt.
The sink owns external delivery deduplication; this module never sends messages.
"""

from datetime import datetime, timezone

from fastapi import HTTPException

from colonymind.redact import redact_sensitive_text
from colonymind.identity.policy import patterns


class NoticeOperations:
    def _notice_text_allowed(self, line):
        return not any(pattern.search(line) for pattern in
                       patterns(self.config["notices"].get("forbidden_patterns", [])))

    async def _progress(self, hand, event_key, text):
        if not isinstance(text, str) or not text:
            return
        text = redact_sensitive_text(text)
        previous = hand["decisions"].get("notice_progress", {})
        if previous.get("text") == text or (previous.get("at") and
                (self.clock() - self.parse_time(previous["at"])).total_seconds() < self.config["notices"]["progress_seconds"]):
            return
        line = f"{hand['name']}: {text}"[:self.config["notices"]["max_chars"]]
        if not self._notice_text_allowed(line):
            return
        await self._notice(f"ntc:{hand['id']}:progress:{event_key}", hand["id"], "routine", line)
        await self.update("hands", "id", hand["id"], decisions={**hand["decisions"], "notice_progress": {"text": text, "at": self.now()}})

    async def _notice(self, ident, hand_id, budget, line):
        existing = await self.one("SELECT * FROM notices WHERE id=?", (ident,))
        if existing:
            if existing["hand_id"] != hand_id or existing["budget"] != budget:
                raise HTTPException(status_code=409, detail="notice_identity_changed")
            return existing
        config = self.config["notices"]
        if budget not in {"routine", "actionable"}:
            raise ValueError("unknown notice budget")
        if (not isinstance(line, str) or not line.strip() or len(line) > config["max_chars"]
                or "\r" in line or "\n" in line or not self._notice_text_allowed(line)):
            raise ValueError("notice line exceeds limits")
        hand = await self.get_hand(hand_id)
        interval = config["merge_seconds"]
        window = datetime.fromtimestamp(int(self.clock().timestamp() // interval) * interval, timezone.utc).isoformat()
        from zoneinfo import ZoneInfo
        zone = ZoneInfo(self.config["timezone"]) if self.config.get("timezone") else None
        rows = await self.all("SELECT queued_at FROM notices WHERE budget=?", (budget,))
        windows = {row["queued_at"] for row in rows
                   if self.parse_time(row["queued_at"]).astimezone(zone).date().isoformat() == self.day()}
        cap = config["routine_per_day"] if budget == "routine" else config["actionable_per_day"]
        if window not in windows and len(windows) >= cap:
            await self.log(hand_id, "notice_capped", {"notice_id": ident, "budget": budget, "line": line})
            return None
        await self.insert("notices", id=ident, budget=budget, line=redact_sensitive_text(line), queued_at=window,
                          hand_id=hand_id, attempt=hand["attempt"])
        return await self.require("notices", ident)

    async def notice(self, ident, hand_id, budget, line):
        async def apply():
            return await self._notice(ident, hand_id, budget, line)
        return await self.write(ident, "/notice", {"hand_id": hand_id, "budget": budget, "line": line}, apply)

    async def notices(self, body):
        # Cursor is informational: unacknowledged lines are always returned, so
        # a delivery timeout or a backdated merge window cannot lose a notice.
        from zoneinfo import ZoneInfo
        zone = ZoneInfo(self.config["timezone"]) if self.config.get("timezone") else None
        rows = await self.all("SELECT * FROM notices ORDER BY queued_at, id")
        selected = []
        for budget in ("routine", "actionable"):
            sent = {r["queued_at"] for r in rows if r["budget"] == budget and r["sent_at"]
                    and self.parse_time(r["sent_at"]).astimezone(zone).date().isoformat() == self.day()}
            cap = self.config["notices"][budget + "_per_day"]
            groups = set(sent)
            for row in rows:
                if row["budget"] != budget or row["sent_at"]:
                    continue
                if (self.clock() - self.parse_time(row["queued_at"])).total_seconds() < self.config["notices"]["merge_seconds"]:
                    continue
                if row["queued_at"] not in groups and len(groups) >= cap:
                    continue
                groups.add(row["queued_at"])
                selected.append(row)
        return {"notices": selected, "cursor": selected[-1]["id"] if selected else body.cursor}

    async def sent(self, nid, body, key):
        async def apply():
            notice = await self.require("notices", nid)
            if notice["sent_at"]:
                if notice["platform_message_id"] != body.platform_message_id:
                    raise HTTPException(status_code=409, detail="notice_already_delivered")
                return notice
            await self.update("notices", "id", nid, sent_at=self.now(), platform_message_id=body.platform_message_id)
            return await self.require("notices", nid)
        return await self.write(key, f"/notices/{nid}/sent", body, apply)


async def notices(body):
    from .controller import get_controller
    return await (await get_controller()).notices(body)


async def sent(nid, body, idempotency_key):
    from .controller import get_controller
    return await (await get_controller()).sent(nid, body, idempotency_key)
