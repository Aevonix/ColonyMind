"""Frozen compatibility oracle for the renderer behavior under comparison."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import HTTPException

from colonymind.redact import redact_sensitive_text

class NoticeOperations:

    async def _notice(self, ident, hand_id, budget, line):
        existing = await self.one("SELECT * FROM notices WHERE id=?", (ident,))
        if existing:
            if existing["hand_id"] != hand_id or existing["budget"] != budget:
                raise HTTPException(status_code=409, detail="notice_identity_changed")
            return existing
        config = self.config["notices"]
        if budget not in {"routine", "actionable"}:
            raise ValueError("unknown notice budget")
        if len(line) > config["max_chars"] or any(chr(n) in line for n in (0x2013, 0x2014)):
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
