"""Whether a live recall is running, so background indexing can leave the model endpoint to it.

An embedding batch running beside a turn's recall slows the query embedding and the rerank when the models share an
accelerator, and during an indexing backlog it can make a turn's recall miss its deadlines. Recall marks itself
while it collects and selects (``LIVE.running()``). The
source-vector lane starts no batch while it does: it waits for recall to end (up to ``YIELD_SECONDS``), and when
recall is still running after that it sends a small batch (``SMALL_BATCH``) instead of a full one, so the backlog
still drains under constant traffic. A batch already in flight is not interrupted.
"""
from __future__ import annotations

import asyncio
import threading
import time
from contextlib import contextmanager

YIELD_SECONDS = 8.0   # about one turn's whole recall budget
SMALL_BATCH = 4       # instead of 16
_POLL_SECONDS = 0.05


class LiveRecall:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = 0

    @contextmanager
    def running(self):
        """Mark a live recall for the block (nested and concurrent recalls count separately)."""
        with self._lock:
            self._running += 1
        try:
            yield
        finally:
            with self._lock:
                self._running -= 1

    def active(self) -> bool:
        return self._running > 0

    async def quiet(self, limit_s: float) -> bool:
        """Wait until no live recall runs, at most ``limit_s``; True when none does."""
        deadline = time.monotonic() + limit_s
        while self.active():
            left = deadline - time.monotonic()
            if left <= 0:
                return False
            await asyncio.sleep(min(_POLL_SECONDS, left))
        return True


LIVE = LiveRecall()


async def batch_size_beside_recall(batch_size: int) -> int:
    """The batch a background indexing step may send now: ``batch_size`` once no live recall runs (after waiting up
    to ``YIELD_SECONDS``), else at most ``SMALL_BATCH``."""
    if not LIVE.active() or await LIVE.quiet(YIELD_SECONDS):
        return batch_size
    return min(batch_size, SMALL_BATCH)
