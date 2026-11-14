"""Process-wide counters for failures that would otherwise pass without a trace.

Code that drops a write, stalls the event loop or ends a model call incomplete bumps a named counter here.
``snapshot()`` is what the mind status route reports under ``health_counters`` and what ``colonymind watch``
reads. Counters live for the life of the process; readers compare successive snapshots.
"""

from __future__ import annotations

import threading
from typing import Dict

_lock = threading.Lock()
_counts: Dict[str, int] = {}


def bump(name: str, n: int = 1) -> None:
    """Add ``n`` to the counter ``name`` (dotted, e.g. ``writes_dropped.comms_ledger``)."""
    with _lock:
        _counts[name] = _counts.get(name, 0) + n


def snapshot() -> Dict[str, int]:
    """Every counter bumped since the process started."""
    with _lock:
        return dict(_counts)
