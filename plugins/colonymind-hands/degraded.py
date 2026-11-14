"""Controller outage admission and local journal failure enforcement."""

from __future__ import annotations

from collections import deque
import hashlib
import threading
import time
from typing import Any


class AdoptionRefused(RuntimeError):
    def __init__(self, hand_id: str):
        self.hand_id = hand_id
        super().__init__("controller refused adoption of an outage hand")


def remap_request(spool, path: str, body: dict, key: str) -> tuple[str, dict, str]:
    """Adopt local outage attempts using the durable controller plan response.

    The local ids are deterministic hashes of plan keys. HTTP keys, journal
    keys and local records remain unchanged across recovery and later replays.
    """
    aliases, denied = {}, set()
    for row in spool.records():
        if row["path"] != "/v1/hands/plan":
            continue
        response = spool.response(row["key"])
        if not isinstance(response, dict) or not response.get("plans"):
            continue
        plan = response["plans"][0]
        suffix = hashlib.sha256(row["key"].encode()).hexdigest()[:12]
        local_hand = "h-" + suffix
        if plan.get("admit") != "start" or not plan.get("lease_token"):
            denied.add(local_hand)
            continue
        aliases.update({local_hand: plan["hand_id"], "t-" + suffix: plan["task_id"],
                        "local-" + suffix: plan["lease_token"]})

    def translate(value):
        if isinstance(value, str):
            if value in denied:
                raise AdoptionRefused(value)
            return aliases.get(value, value)
        if isinstance(value, list):
            return [translate(item) for item in value]
        if isinstance(value, dict):
            return {field: translate(item) for field, item in value.items()}
        return value

    return "/".join(translate(part) for part in path.split("/")), translate(body), key


class DegradedMode:
    """Two failed controller calls within 30 s by default (hands.degraded; both numbers assumptions)."""

    def __init__(self, settings=None):
        self._failures: deque[float] = deque()
        self._lock = threading.Lock()
        self._degraded = False
        self._settings = settings or (lambda: {})

    def _limits(self) -> tuple[int, float]:
        try:
            values = self._settings() or {}
            return (max(1, int(values.get("controller_failures", 2))),
                    float(values.get("failure_window_seconds", 30)))
        except Exception:
            return 2, 30.0

    @property
    def is_degraded(self) -> bool:
        with self._lock:
            return self._degraded

    def failure(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        failures, window = self._limits()
        with self._lock:
            while self._failures and now - self._failures[0] > window:
                self._failures.popleft()
            self._failures.append(now)
            if len(self._failures) >= failures:
                self._degraded = True

    def success(self) -> None:
        with self._lock:
            self._failures.clear()
            self._degraded = False

    def admit(self, priority: str, running_count: int) -> tuple[bool, str]:
        if not self.is_degraded:
            return True, ""
        if priority != "owner":
            return False, "controller unavailable: only owner work can start"
        if running_count >= 1:
            return False, "controller unavailable: one running hand allowed"
        return True, ""


def register(ctx: Any) -> None:
    def pre_tool_call(tool_name="", args=None, session_id="", **kwargs):
        hand = None
        try:
            from .dispatch import get_runtime
            runtime = get_runtime(ctx)
            hand = runtime.registry.for_session(session_id, task_id=kwargs.get("task_id"))
            if hand is None or runtime.spool.writable:
                return None
            if hand.kind in {"coding", "ops", "integrate"}:
                hand.blocked = True
                runtime.system_stop(hand, "cannot record work safely")
                return {"action": "block", "message": "cannot record work safely"}
        except Exception:
            if hand is not None:
                return {"action": "block", "message": "cannot verify durable hand state"}
        return None

    ctx.register_hook("pre_tool_call", pre_tool_call)
