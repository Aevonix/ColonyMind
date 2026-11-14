"""Read-only Google collection on the existing singleton body pass.

No Google client, credentials, worker thread or model lives here. The host's
capability-gated call_mcp uses its existing native connector connections.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import time

READS = frozenset({"gmail_search", "gmail_get", "calendar_events"})


class GoogleIdeas:
    def __init__(self, client, call_mcp, *, clock=None):
        self.client, self.call_mcp = client, call_mcp
        self.clock = clock or (lambda: datetime.fromtimestamp(time.time(), timezone.utc))

    def read(self, server, account, tool, arguments):
        if tool not in READS:
            raise ValueError("read_only_ideas_connector")
        response = self.call_mcp(server, tool, {**arguments, "account": account}, timeout=3)
        if (not isinstance(response, dict) or response.get("ok") is not True
                or response.get("truncated") or response.get("error")):
            raise ValueError("connector_unavailable")
        result = response.get("result")
        if isinstance(result, str):
            result = json.loads(result)
        # Native MCP may return its standard text-content carrier.
        if isinstance(result, dict) and "content" in result and "account" not in result:
            content = result["content"]
            if len(content) != 1 or content[0].get("type") != "text":
                raise ValueError("unsupported_connector_result")
            result = json.loads(content[0]["text"])
        if (not isinstance(result, dict) or result.get("account") != account
                or not isinstance(result.get("data"), dict) or result.get("isError") or result.get("error")):
            raise ValueError("account_read_unavailable")
        return result

    def collect(self):
        response = self.client.get("/v1/mind/ideas/collection", timeout=3)
        if not response.is_success:
            return False
        spec = response.json()
        if not spec.get("due"):
            return False
        observations = []
        started = time.monotonic()
        now = self.clock()
        for account in spec.get("accounts", [])[:2]:
            server, ident = account["server"], account["account"]
            try:
                found = self.read(server, ident, "gmail_search", {"query": "newer_than:14d", "page_size": 8})
                for item in found["data"].get("messages", [])[:8]:
                    if time.monotonic() - started > 15:
                        break
                    try:
                        message = self.read(server, ident, "gmail_get", {"message_id": item["id"], "format": "full"})
                        observations.append({"server": server, "account": ident, "tool": "gmail_get", "result": message})
                    except (ValueError, KeyError, PermissionError, TimeoutError):
                        continue
            except (ValueError, KeyError, PermissionError, TimeoutError):
                pass  # no substitution of another account or claim of no mail
            if time.monotonic() - started > 17:
                break
            calendar = (account.get("calendars") or ["primary"])[0]
            try:
                events = self.read(server, ident, "calendar_events", {
                    "calendar_id": calendar, "time_min": now.isoformat(),
                    "time_max": (now + timedelta(days=14)).isoformat(), "single_events": True, "page_size": 16})
                observations.append({"server": server, "account": ident, "tool": "calendar_events",
                                     "calendar_id": calendar, "result": events})
            except (ValueError, KeyError, PermissionError, TimeoutError):
                pass
        response = self.client.post("/v1/mind/ideas/sources", timeout=5, json={"observations": observations})
        return response.is_success
