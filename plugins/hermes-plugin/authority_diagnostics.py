"""Process-local authority diagnostics, containing no session or call data."""
from __future__ import annotations

from collections import Counter
import threading

_COUNTERS: Counter[str] = Counter()
_LOCK = threading.Lock()


def ancestry_unreadable(logger, error: Exception) -> None:
    with _LOCK:
        _COUNTERS["ancestry_unreadable"] += 1
    logger.warning("ancestry_unreadable exception_type=%s", type(error).__name__)


def authority_counters() -> dict[str, int]:
    """A snapshot for the adapter status surface or an in-process collector."""
    with _LOCK:
        return {"ancestry_unreadable": _COUNTERS["ancestry_unreadable"]}
