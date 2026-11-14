"""A turn's time budget, derived from how long its host waits for the context.

A host waits a bounded time for ``/context/assemble`` and can name a shorter wait with ``wait_ms``. Recall used to stack
a 5 s query embedding limit (with up to three attempts under it), its rerank deadline and the judge's, which
together could run past that wait. Each turn now carries one budget, started when the request arrives:

- the query embedding gets one attempt of ``COLONYMIND_RECALL_EMBED_TIMEOUT_MS`` (default 1,500 ms);
- recall's rerank gets the lesser of its configured deadline (``COLONYMIND_RECALL_RERANK_TIMEOUT_MS``) and what is
  left of the wait less a reserve;
- the reserve is the per-turn judge's deadline (when selection is on) plus ``RESERVE_MARGIN_MS`` for the rest of
  the response, at most a third of the wait (recall comes first); the judge then gets the lesser of its own
  deadline and what is left less that margin.

With nothing left, recall starts no call: it keeps its keyword evidence and its bounded fallback. A host that
names no wait gets ``COLONYMIND_CONTEXT_HOST_WAIT_MS`` (default 7,500 ms). This is also the ceiling for named waits.
The assembly deadline covers every producer and response preparation, leaving a margin for returning the answer.

The budget also knows whether the turn's other lanes are still being assembled: a rerank answer that arrives after
recall's deadline but before they are done costs the turn nothing, and is used (``late_answer``).
"""
from __future__ import annotations

import asyncio
import math
import os
import inspect
import logging
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from functools import partial

from colonymind.health_counters import bump

WAIT_ENV = "COLONYMIND_CONTEXT_HOST_WAIT_MS"
DEFAULT_WAIT_MS = 7500
EMBED_ENV = "COLONYMIND_RECALL_EMBED_TIMEOUT_MS"
DEFAULT_EMBED_MS = 1500
RESERVE_MARGIN_MS = 500


def _ms(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default
    return min(max(value, low), high) if math.isfinite(value) else default


def embed_attempt_s() -> float:
    """The one attempt a live query embedding gets, in seconds."""
    return _ms(EMBED_ENV, DEFAULT_EMBED_MS, 1, 60_000) / 1000.0


@dataclass
class TurnBudget:
    wait_s: float
    judge_s: float = 0.0
    started: float = field(default_factory=time.monotonic)
    lanes_done: asyncio.Event = field(default_factory=asyncio.Event)
    sync_pool_skipped: int = 0
    sync_read_failed: int = 0
    read_errors: set = field(default_factory=set)
    failure_lock: threading.Lock = field(default_factory=threading.Lock)
    writes: list = field(default_factory=list)
    recall_exit: object = None

    def read_failed(self, error):
        with self.failure_lock:
            self.sync_read_failed += 1
            bump("context_reads_failed")
            key = (type(error), getattr(error, "sqlite_errorname", None))
            if key not in self.read_errors:
                self.read_errors.add(key)
                logging.getLogger(__name__).warning("context worker read failed (%s, sqlite_errorname=%s)",
                                                  type(error).__name__, key[1])

    @classmethod
    def for_host(cls, wait_ms: int | None = None) -> "TurnBudget":
        """The budget of a turn whose host waits ``wait_ms`` (or the configured default) for its context."""
        from colonymind.memory import compass
        settings = compass.selection_settings()
        configured = _ms(WAIT_ENV, DEFAULT_WAIT_MS, 1, 600_000)
        wait = configured if wait_ms is None else min(float(wait_ms), configured)
        return cls(wait_s=wait / 1000.0, judge_s=settings.timeout_s if settings.enabled else 0.0)

    @property
    def reserve_s(self) -> float:
        """What recall leaves for the judge and the rest of the response: at most a third of the wait."""
        return min(self.judge_s + RESERVE_MARGIN_MS / 1000.0, self.wait_s / 3.0)

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def assembly_left(self) -> float:
        """Time left for assembly, reserving at most a tenth of the wait for returning the response."""
        margin = min(RESERVE_MARGIN_MS / 1000.0, self.wait_s / 10.0)
        return max(0.0, self.wait_s - self.elapsed() - margin)

    def recall_left(self) -> float:
        """What recall may still spend, in seconds."""
        return max(0.0, self.wait_s - self.elapsed() - self.reserve_s)

    def embed_timeout(self) -> float:
        return min(embed_attempt_s(), self.recall_left())

    def rerank_timeout(self, configured_s: float) -> float:
        return min(configured_s, self.recall_left())

    def judge_timeout(self, configured_s: float) -> float:
        return max(0.0, min(configured_s, self.wait_s - self.elapsed() - RESERVE_MARGIN_MS / 1000.0))

    def assembling(self) -> bool:
        """Whether the turn's other lanes are still being assembled."""
        return not self.lanes_done.is_set()

    def lanes_assembled(self) -> None:
        self.lanes_done.set()

    async def late_answer(self, call: asyncio.Future) -> bool:
        """Wait for ``call`` (past its deadline) only while the other lanes are still being assembled and recall's
        budget lasts; True when it answered in that window. The call itself is never cancelled here."""
        left = self.recall_left()
        if not self.assembling() or left <= 0:
            return False
        lanes = asyncio.ensure_future(self.lanes_done.wait())
        try:
            done, _ = await asyncio.wait({call, lanes}, timeout=left, return_when=asyncio.FIRST_COMPLETED)
        finally:
            lanes.cancel()
        return call in done


_CURRENT: ContextVar[TurnBudget | None] = ContextVar("colonymind_turn_budget", default=None)
_WRITES: ContextVar[list | None] = ContextVar("colonymind_producer_writes", default=None)
_SECTION: ContextVar[str | None] = ContextVar("colonymind_write_section", default=None)
_INHERIT_SECTION = object()
_PENDING: set[asyncio.Task] = set()


def current() -> TurnBudget | None:
    """The budget of the context being assembled, if any (explicit searches have none)."""
    return _CURRENT.get()


@contextmanager
def turn(budget: TurnBudget):
    """Make ``budget`` the current turn's for the block (and the tasks it starts)."""
    token = _CURRENT.set(budget)
    try:
        yield budget
    finally:
        budget.writes.clear()
        _CURRENT.reset(token)


def embed_timeout() -> float:
    """The current turn's query-embedding attempt, or the configured attempt outside a turn."""
    budget = current()
    return budget.embed_timeout() if budget is not None else embed_attempt_s()


def defer_write(call, /, *args, **kwargs) -> None:
    """Record context side effects only once its response is ready and still in time."""
    from . import producers
    if producers.active():
        raise RuntimeError("context workers cannot schedule writes")
    budget = current()
    if budget is None:
        call(*args, **kwargs)
    else:
        writes = _WRITES.get()
        (budget.writes if writes is None else writes).append((_SECTION.get(), call, args, kwargs))


@contextmanager
def producer_writes(section=_INHERIT_SECTION):
    """Hold one producer's writes until it succeeds; a failed producer drops them."""
    writes = []
    parent = _WRITES.get()
    token = _WRITES.set(writes)
    section_token = _SECTION.set(_SECTION.get() if section is _INHERIT_SECTION else section)
    try:
        yield writes
    except BaseException:
        raise
    else:
        budget = current()
        if budget is not None:
            (budget.writes if parent is None else parent).extend(writes)
    finally:
        _WRITES.reset(token)
        _SECTION.reset(section_token)


def take_writes(sections):
    """Capture accepted producers' writes before the turn clears its proposals."""
    budget = current()
    visible = {section.id for section in sections}
    writes, budget.writes = budget.writes, []
    return [write for write in writes if write[0] is None or write[0] in visible]


async def finish_writes(writes, observation=None) -> None:
    """Run accepted writes outside assembly, offloading only the runtime-lock writers."""
    # The caller returns its response before this task starts any synchronous work.
    await asyncio.sleep(0)
    from . import recall_exit
    token = recall_exit.bind_post(observation)
    try:
        for _, call, args, kwargs in writes:
            try:
                function = call
                while isinstance(function, partial):
                    function = function.func
                name = getattr(function, "__qualname__", type(function).__qualname__)
                offload = (getattr(function, "__module__", ""), name) in {
                    ("colonymind.turns.source_vectors", "SourceVectors._owner_search.<locals>.record"),
                    ("colonymind.plasticity.learners.query_adapter", "Driver.record_trace"),
                    ("colonymind.plasticity.learners.query_adapter", "Driver.record_selection"),
                    # Selection is normally recorded inside this accepted callback.
                    ("colonymind.api.routers.host", "_learning_served"),
                }
                result = await asyncio.to_thread(call, *args, **kwargs) if offload else call(*args, **kwargs)
                if inspect.isawaitable(result):
                    await result
            except Exception as error:
                bump("context_writes_dropped")
                bump(f"writes_dropped.{name}")
                logging.getLogger(__name__).warning(
                    "accepted context write failed: %s (%s, sqlite_errorname=%s)",
                    name, type(error).__name__, getattr(error, "sqlite_errorname", None))
    finally:
        recall_exit.finish_post(observation, token)


def schedule_writes(writes, observation=None) -> None:
    """Keep the post-response write task alive without inheriting a turn budget."""
    if writes or observation is not None:
        context = copy_context()
        context.run(_CURRENT.set, None)
        context.run(_WRITES.set, None)
        context.run(_SECTION.set, None)
        task = asyncio.create_task(finish_writes(writes, observation), name="context-writes", context=context)
        _PENDING.add(task)
        task.add_done_callback(_PENDING.discard)


async def start_writes(writes, observation=None) -> None:
    """ASGI background hook: response bytes have been sent; start the loop task."""
    schedule_writes(writes, observation)
