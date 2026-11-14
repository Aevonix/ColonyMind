"""One pooled HTTP client per remote model provider, reused across calls.

A client opened and closed around every call pays a new connection, and its setup time, on every rerank and query
embedding, several times a turn. A provider keeps one ``httpx.AsyncClient`` per event loop instead. Its idle
connections expire after ``COLONYMIND_HTTP_KEEPALIVE_MS`` (default 4,000 ms): shorter than a typical server's own
keep-alive (5 s for the common Python servers) and a proxy's idle timeout, so the pool never offers a connection the
other end has already closed.

A pool never outlives its event loop. A connection belongs to the loop that opened it, and an open connection keeps
that loop referenced, so a pool left behind would hold its loop and sockets for good. Each pool is therefore closed
within its own loop's lifecycle: when it opens, it registers an async generator with the running loop, and a loop
that shuts down in order (``asyncio.run``, ``asyncio.Runner``, any shutdown that finalizes the loop's async
generators) closes it there, which closes the pool and forgets it. A loop closed without that shutdown can no longer
run anything; its pool is dropped the next time the provider is used, or closed, from another loop, and the garbage
collector then closes its sockets.

A provider dropped without ``close()`` while its loops run has its pools closed on those loops, in order. The pools
are kept apart from the provider (nothing in a pool refers back to it), so they are never garbage together with it:
the collector would close a pooled socket at once and leave the loop's own close for later, and that close would
then withdraw the loop's watch on a descriptor number another connection may hold by then, which that connection
would wait out to its time limit.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import weakref

logger = logging.getLogger(__name__)

KEEPALIVE_ENV = "COLONYMIND_HTTP_KEEPALIVE_MS"
DEFAULT_KEEPALIVE_MS = 4000


def keepalive_expiry_s() -> float:
    """How long an idle pooled connection is kept, in seconds (0 keeps none)."""
    try:
        value = float(os.environ.get(KEEPALIVE_ENV, "").strip() or DEFAULT_KEEPALIVE_MS)
    except ValueError:
        value = DEFAULT_KEEPALIVE_MS
    if value != value or value < 0:  # NaN or negative
        value = DEFAULT_KEEPALIVE_MS
    return min(value, 600_000) / 1000.0


class _Pool:
    """One loop's client, and the async generator that closes it when that loop shuts down."""

    __slots__ = ("client", "lifetime")

    def __init__(self, client) -> None:
        self.client = client
        self.lifetime = None


class KeepAliveClient:
    """A provider's pooled client: one per event loop, created on first use with ``kwargs``, closed with its loop, or
    on that loop when the provider is dropped without closing it."""

    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs
        self._pools: dict[asyncio.AbstractEventLoop, _Pool] = {}
        self._lock = threading.RLock()  # re-entered when an ended loop's pool is finished while it is held
        # The pools and the lock, never this object: dropping the provider then releases them in order (``_release``).
        weakref.finalize(self, _release, self._pools, self._lock).atexit = False

    def _open(self):
        import httpx
        # httpx's default pool sizes, with the idle expiry above.
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=keepalive_expiry_s())
        return httpx.AsyncClient(limits=limits, **self._kwargs)

    def get(self):
        """The running loop's client, opened on first use (or after it was closed)."""
        loop = asyncio.get_running_loop()
        with self._lock:
            _drop_ended_loops(self._pools)
            pool = self._pools.get(loop)
            if pool is None:
                pool = self._pools[loop] = _Pool(self._open())
                pool.lifetime = _lifetime(self._pools, self._lock, loop, pool)
                _start(pool.lifetime)
            elif pool.client.is_closed:
                pool.client = self._open()
            return pool.client

    async def aclose(self) -> None:
        """Close the running loop's pool and its connections (the next call opens a new one), and drop the pools of
        loops that were closed without shutting down theirs. Another running loop closes its own when it ends."""
        loop = asyncio.get_running_loop()
        with self._lock:
            _drop_ended_loops(self._pools)
            pool = self._pools.pop(loop, None)
        if pool is not None:
            await pool.lifetime.aclose()


async def _lifetime(pools: dict, lock, loop: asyncio.AbstractEventLoop, pool: _Pool):
    """Started on ``loop`` when its pool opens; the loop closes it when it shuts down, and the pool with it."""
    try:
        yield
    finally:
        with lock:
            if pools.get(loop) is pool:
                del pools[loop]
        if not loop.is_closed():
            await pool.client.aclose()


def _drop_ended_loops(pools: dict) -> None:
    """Forget the pools of loops closed without shutting down their async generators (called with the lock)."""
    for loop in [loop for loop in pools if loop.is_closed()]:
        _finish(pools.pop(loop))


def _finish(pool: _Pool) -> None:
    """Finish an ended loop's pool here: its loop is closed, so its lifetime ends without awaiting anything."""
    try:
        pool.lifetime.aclose().send(None)
    except StopIteration:
        pass
    except Exception:
        logger.debug("an ended loop's pool did not finish cleanly", exc_info=True)


def _release(pools: dict, lock) -> None:
    """A dropped provider's pools: each running loop closes its own pool, in order, and an ended loop's is finished
    here. Until its close runs, the scheduled close holds the pool, so the collector never finalizes its connections.
    Called from whichever thread dropped the provider (or ran the collector)."""
    with lock:
        held = list(pools.items())
        pools.clear()
    for loop, pool in held:
        if loop.is_closed():
            _finish(pool)
            continue
        try:
            loop.call_soon_threadsafe(loop.create_task, pool.lifetime.aclose())
        except RuntimeError:                  # the loop closed meanwhile
            _finish(pool)


def _start(lifetime) -> None:
    """Run ``lifetime`` to its first ``yield`` on the running loop, which registers it with that loop: the loop then
    closes it when it shuts down its async generators."""
    try:
        lifetime.asend(None).send(None)
    except StopIteration:
        return
    raise RuntimeError("a pool's lifetime must reach its first yield without waiting")
