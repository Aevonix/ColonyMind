"""Deadline-bound synchronous context reads, isolated from the shared executor.

A cancelled waiter does not free its slot: only the worker's completion does.
Every worker connection is read-only and closed before that slot is released.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
import threading
import time

from . import turn_budget

_NO_DEFAULT = object()


class Unavailable(RuntimeError):
    """A required context read could not supply a safe value."""


class Skipped(Unavailable):
    """An abandoned read exhausted the synchronous producer pool."""


class _DeadlineCursor(sqlite3.Cursor):
    def execute(self, *args, **kwargs):
        return self.connection.checked(super().execute, *args, **kwargs)

    def executemany(self, *args, **kwargs):
        return self.connection.checked(super().executemany, *args, **kwargs)

    def executescript(self, *args, **kwargs):
        return self.connection.checked(super().executescript, *args, **kwargs)


class _DeadlineConnection(sqlite3.Connection):
    def checked(self, operation, *args, **kwargs):
        try:
            self.check_deadline()
            try:
                return operation(*args, **kwargs)
            except sqlite3.OperationalError as exc:
                if getattr(exc, "sqlite_errorcode", None) != sqlite3.SQLITE_SCHEMA:
                    raise
                # A startup migration can invalidate a prepared worker read.
                # Reprepare once, within the same response deadline.
                self.check_deadline()
                return operation(*args, **kwargs)
        except Exception as exc:
            self.read.failed(exc)
            raise

    def check_deadline(self):
        if self.read.expired():
            raise sqlite3.OperationalError("context read deadline exceeded")

    def execute(self, *args, **kwargs):
        return self.checked(super().execute, *args, **kwargs)

    def executemany(self, *args, **kwargs):
        return self.checked(super().executemany, *args, **kwargs)

    def executescript(self, *args, **kwargs):
        return self.checked(super().executescript, *args, **kwargs)

    def cursor(self, factory=_DeadlineCursor):
        return self.checked(super().cursor, factory)


@dataclass
class _Read:
    deadline: float
    abandoned: threading.Event
    connections: list = field(default_factory=list)
    bound: dict = field(default_factory=dict)
    writes: list = field(default_factory=list)
    failures: set = field(default_factory=set)

    def failed(self, error):
        key = (type(error), getattr(error, "sqlite_errorname", None))
        if key not in self.failures:
            self.failures.add(key)
            budget = turn_budget.current()
            if budget is not None:
                budget.read_failed(error)

    def expired(self):
        return self.abandoned.is_set() or time.monotonic() >= self.deadline

    def connect(self, path, *, query_only=True):
        if self.expired():
            raise TimeoutError("context read deadline")
        if not path or str(path) == ":memory:":
            raise ValueError("context workers require a file-backed database")
        conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True,
                               timeout=min(.05, max(0.0, self.deadline - time.monotonic())),
                               factory=_DeadlineConnection)
        conn.read = self
        self.connections.append(conn)
        conn.row_factory = sqlite3.Row
        if query_only:
            conn.execute("PRAGMA query_only=ON")
        # Keyword TEMP tables may opt out; mode=ro still protects the main database.
        # PRAGMA optimize and ANALYZE must never run on the turn path.
        conn.set_progress_handler(lambda: int(self.expired()), 100)
        return conn


_READ: ContextVar[_Read | None] = ContextVar("colonymind_context_read", default=None)
_INLINE: ContextVar[bool] = ContextVar("colonymind_inline_reads", default=False)


@contextmanager
def inline_reads():
    """Run unbudgeted reads on the calling worker thread, which owns one connection for them.

    For a worker already off the event loop (voice recall): a thread hop would only
    open another connection. Budgeted reads still go through the pool.
    """
    token = _INLINE.set(True)
    try:
        yield
    finally:
        _INLINE.reset(token)


def active() -> bool:
    """Whether this code runs in a context read worker (never a writer)."""
    return _READ.get() is not None


def expired() -> bool:
    read = _READ.get()
    return read is not None and read.expired()


def limit_deadline(ends: float) -> None:
    """Narrow a producer's SQL deadline to its section's local allowance."""
    read = _READ.get()
    if read is not None:
        read.deadline = min(read.deadline, ends)


def read_failed(error) -> None:
    """Count a worker error handled by a producer that still returns partial state."""
    read = _READ.get()
    if read is not None:
        read.failed(error)


def connection(path, *, query_only=True):
    """A short-lived worker connection, or None for the store's normal writer path."""
    read = _READ.get()
    return read.connect(path, query_only=query_only) if read is not None else None


def after_read(call, /, *args, **kwargs):
    """Propose a write from a read snapshot; an abandoned producer discards it.

    The event-loop waiter transfers these proposals to the accepted response's
    write list, only after the entire read returns within its deadline.
    """
    read = _READ.get()
    if read is None:
        raise RuntimeError("write proposals require a context read snapshot")
    read.writes.append((call, args, kwargs))


class ReadableConnection:
    """Keep a store's writer thread-bound; resolve a separate connection on a read worker.

    The database path is captured when the writer is assigned, in its own thread.
    Bound connections are reused only inside one producer and closed with it.
    """
    def __set_name__(self, owner, name):
        self.key = "_context_connection_" + name

    def __set__(self, obj, conn):
        path = next((row[2] for row in conn.execute("PRAGMA database_list") if row[1] == "main"), "")
        obj.__dict__[self.key] = (conn, path)

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        writer, path = obj.__dict__[self.key]
        read = _READ.get()
        if read is None:
            return writer
        if path not in read.bound:
            read.bound[path] = read.connect(path)
        return read.bound[path]


class ProducerPool:
    """Bound worker concurrency; only abandoned reads exhaust admission.

    Healthy concurrent reads await a free slot within their turn's deadline,
    without putting work in the executor's queue.
    """
    def __init__(self, workers=16):
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="context-read")
        self.slots = threading.BoundedSemaphore(workers)
        self.workers = workers
        self.abandoned = set()
        self.lock = threading.Lock()

    async def read(self, call, /, *args, default=_NO_DEFAULT, **kwargs):
        budget = turn_budget.current()
        if budget is None:
            if _INLINE.get():
                return call(*args, **kwargs)
            return await asyncio.to_thread(call, *args, **kwargs)
        left = budget.assembly_left()
        if left <= 0:
            raise TimeoutError("context assembly deadline")
        while not self.slots.acquire(blocking=False):
            with self.lock:
                full = len(self.abandoned) >= self.workers
            if full:
                budget.sync_pool_skipped += 1
                if default is _NO_DEFAULT:
                    raise Skipped("abandoned context reads exhausted the pool")
                return default
            if budget.assembly_left() <= 0:
                raise TimeoutError("context assembly deadline")
            await asyncio.sleep(.001)
        left = budget.assembly_left()
        context = copy_context()
        abandoned = threading.Event()
        read = _Read(time.monotonic() + left, abandoned)

        def run():
            token = _READ.set(read)
            try:
                if read.expired():
                    raise TimeoutError("context read deadline")
                return call(*args, **kwargs)
            except Exception as exc:
                read.failed(exc)
                raise
            finally:
                for conn in read.connections:
                    conn.close()
                _READ.reset(token)

        try:
            future = self.executor.submit(context.run, run)
        except BaseException:
            self.slots.release()
            raise
        # This is the concurrent future, not its cancellable asyncio wrapper.
        def finished(done):
            with self.lock:
                self.abandoned.discard(done)
                self.slots.release()

        future.add_done_callback(finished)
        wrapped = asyncio.wrap_future(future)
        # Retrieve a late exception even when its waiter has left.
        wrapped.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        try:
            result = await asyncio.wait_for(asyncio.shield(wrapped), timeout=left)
            if budget.assembly_left() <= 0:
                raise TimeoutError("context assembly deadline")
            for call, args, kwargs in read.writes:
                turn_budget.defer_write(call, *args, **kwargs)
            return result
        finally:
            abandoned.set()
            with self.lock:
                if not future.done():
                    self.abandoned.add(future)


POOL = ProducerPool()


async def read(call, /, *args, default=_NO_DEFAULT, **kwargs):
    return await POOL.read(call, *args, default=default, **kwargs)
