"""One rerank call at a time per reranker, and never a queue.

A reranker server may not cancel a call its client gave up on: it computes the answer anyway, and the next call
waits behind that unfinished work. The per-turn judge, which runs after recall, then tends to miss its own deadline
whenever recall missed.

So recall and the judge start a call only when their reranker is idle. A caller that finds the gate held skips
reranking (``RerankBusy``) and uses its fallback at once instead of queueing behind work the server has not finished.

Recall's call is shielded: it keeps the reranker (this gate) until the server answers, whether or not recall still
waits, bounded by ``COLONYMIND_RERANK_HOLD_LIMIT_MS`` (default 30,000, the remote provider's own HTTP timeout), so
the judge that follows it in the same turn skips instead of waiting behind it. A caller that stopped waiting may
still use the answer if it arrives in a window the caller names (recall: while the turn's other lanes are still
being assembled, ``turn_budget.late_answer``).

The judge's call is not held (``hold=False``): the judge has no use for a late answer, and the next call is the next
turn's recall, which a held gate would make skip its rerank for as long as the server takes to finish the judge's
call. When the judge stops waiting (its deadline passed, or it was cancelled) its call is cancelled on this side and
the gate is free at once.

Single flight on the server lasts until the answer, the hold limit or a caller that does not hold giving up,
whichever comes first. The gate knows when an answer arrives, not when a server stops working on a call its client
gave up: once the gate is freed (the call is cancelled on this side), the next call may reach a server still
computing the first.

The gate is process-wide: one per reranker object, shared by every recall and judge call in the process.
"""
from __future__ import annotations

import asyncio
import math
import os
import threading
import weakref
from typing import Any, Awaitable, Callable

HOLD_ENV = "COLONYMIND_RERANK_HOLD_LIMIT_MS"
DEFAULT_HOLD_MS = 30_000


class RerankBusy(RuntimeError):
    """The reranker is still answering an earlier call: skipped, not queued."""


def hold_limit_s() -> float:
    try:
        value = float(os.environ.get(HOLD_ENV, "").strip() or DEFAULT_HOLD_MS)
    except ValueError:
        value = DEFAULT_HOLD_MS
    if not math.isfinite(value) or value <= 0:
        value = DEFAULT_HOLD_MS
    return value / 1000.0


class RerankGate:
    """The one call in flight on a reranker, if any."""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._lock = threading.Lock()

    def busy(self) -> bool:
        task = self._task
        if task is None or task.done():
            return False
        try:
            return not task.get_loop().is_closed()
        except Exception:
            return False

    def start(self, call: Callable[[], Awaitable[Any]]) -> asyncio.Task | None:
        """Start ``call()`` when the reranker is idle and return its task, which holds the gate until it ends or is
        released (``release``); None when an earlier call is still in flight."""
        with self._lock:
            if self.busy():
                return None
            task = asyncio.ensure_future(self._held(call))
            self._task = task
        task.add_done_callback(self._released)
        return task

    @staticmethod
    async def _held(call):
        return await asyncio.wait_for(call(), hold_limit_s())

    def release(self, task: asyncio.Task) -> None:
        """Free the gate from ``task`` at once and cancel it: its caller has no use for the answer."""
        with self._lock:
            if self._task is task:
                self._task = None
        task.cancel()

    def _released(self, task: asyncio.Task) -> None:
        with self._lock:
            if self._task is task:
                self._task = None
        if not task.cancelled():
            task.exception()  # retrieved here: its caller may have stopped waiting for it

    async def idle(self) -> None:
        """Wait until no call is in flight (tests; an orderly shutdown)."""
        task = self._task
        if task is not None and not task.done():
            await asyncio.gather(asyncio.shield(task), return_exceptions=True)


_GATES: "weakref.WeakKeyDictionary[Any, RerankGate]" = weakref.WeakKeyDictionary()
_UNWEAKABLE: dict[int, RerankGate] = {}
_REGISTRY = threading.Lock()


def gate_for(rerank_fn) -> RerankGate:
    """The gate of the reranker behind ``rerank_fn`` (a bound method's object, else the function itself)."""
    owner = getattr(rerank_fn, "__self__", None)
    if owner is None:
        owner = rerank_fn
    with _REGISTRY:
        try:
            gate = _GATES.get(owner)
            if gate is None:
                gate = _GATES[owner] = RerankGate()
        except TypeError:  # not weakly referenceable
            gate = _UNWEAKABLE.setdefault(id(owner), RerankGate())
        return gate


async def rerank_once(rerank_fn, query: str, documents: list, *, timeout_s: float,
                      late: Callable[[asyncio.Future], Awaitable[bool]] | None = None,
                      started: Callable[[], None] | None = None, hold: bool = True) -> tuple[Any, bool]:
    """``(results, late)``: the reranker's answer within ``timeout_s``, or after it within ``late``'s window.

    Raises ``RerankBusy`` at once when an earlier call is still in flight, and ``TimeoutError`` when the answer
    is not in by then. With ``hold`` (recall) the call is never cancelled by its caller: it holds the gate until the
    server answers or the hold limit passes. Without it (the judge) a call its caller stopped waiting for, past its
    window or cancelled, is cancelled and frees the gate at once. ``started`` is called once the call has started
    (the documents are sent), before its answer is awaited; never when it is skipped (busy, or no time left).
    """
    if timeout_s <= 0:
        raise TimeoutError("no time left for the rerank")
    gate = gate_for(rerank_fn)
    task = gate.start(lambda: rerank_fn(query, documents, top_k=len(documents)))
    if task is None:
        raise RerankBusy("the reranker is still answering an earlier call")
    try:
        if started is not None:
            started()
        done, _ = await asyncio.wait({task}, timeout=timeout_s)
        if task in done:
            return task.result(), False
        if late is not None and await late(task):
            return task.result(), True
        raise TimeoutError("the reranker did not answer within %.0f ms" % (timeout_s * 1000))
    finally:
        if not hold and not task.done():
            gate.release(task)
