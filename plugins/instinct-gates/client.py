"""The gates' client for ``/v1/instinct``: the adapter's own settings and credential, and a hard deadline.

Every call returns what the sidecar answered or None, never raises, and never waits past its deadline: the request
runs on a daemon thread that is abandoned when the deadline passes (its own transport timeouts end it). None means
the gate keeps its existing path.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

DECIDE_ROUTE = "/v1/instinct/decide"
OUTCOME_ROUTE = "/v1/instinct/outcome"


def bounded(function, timeout: float) -> Any:
    """``function()``'s value, or None when it raises or has not returned within ``timeout`` seconds."""
    box: dict[str, Any] = {}
    done = threading.Event()

    def run() -> None:
        try:
            box["value"] = function()
        except Exception as error:
            logger.debug("gate request failed (%s)", type(error).__name__)
        finally:
            done.set()

    threading.Thread(target=run, name="colonymind-gates-request", daemon=True).start()
    return box.get("value") if done.wait(max(0.0, timeout)) else None


class GateClient:
    """``decide`` and ``outcome`` against the sidecar the adapter is configured for, under its credential."""

    def __init__(self, settings: Any = None, client: Any = None):
        from colonymind_hermes.client import ColonyMindClient, load_settings
        self.settings = settings if settings is not None else load_settings()
        self.client = client if client is not None else ColonyMindClient(self.settings)

    def _post(self, route: str, body: dict, timeout: float) -> Any:
        response = self.client.post(route, json=body, timeout=max(0.05, timeout))
        return response.json() if response.status_code == 200 else None

    def decide(self, body: dict, *, deadline_s: float) -> list[dict] | None:
        """The decisions for ``body`` (one request, or ``requests``), in order, or None. ``body`` carries the
        deadline the sidecar holds to (milliseconds); the caller waits at most ``deadline_s``."""
        body = {**body, "deadline_ms": max(1, min(5000, int(deadline_s * 1000)))}
        answer = bounded(lambda: self._post(DECIDE_ROUTE, body, deadline_s), deadline_s)
        wanted = len(body["requests"]) if "requests" in body else 1
        decisions = answer.get("decisions") if isinstance(answer, dict) and "requests" in body else [answer]
        if not isinstance(decisions, list) or len(decisions) != wanted or not all(
                isinstance(item, dict) and isinstance(item.get("apply"), bool) for item in decisions):
            return None
        return decisions

    def outcome(self, decisions: list[str], outcome: str, *, deadline_s: float) -> bool:
        answer = bounded(lambda: self._post(OUTCOME_ROUTE, {"decisions": decisions, "outcome": outcome},
                                            deadline_s), deadline_s)
        return isinstance(answer, dict)


__all__ = ["DECIDE_ROUTE", "OUTCOME_ROUTE", "GateClient", "bounded"]
