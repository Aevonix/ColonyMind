"""One reentrant lock per state root, shared by requests, builds, rollback and erasure.

An RLock orders threads; flock orders independent sidecar and CLI processes. Separate
StateLock instances for the same root share the descriptor and nesting count.
On a running asyncio loop, contention raises StateLockBusy instead of waiting;
optional callers take their fallback and required maintenance belongs on workers.
Reentrancy is thread-owned: never hold this synchronous lock across an await.
"""
from __future__ import annotations

import asyncio
import fcntl
from functools import wraps
import threading
from pathlib import Path


class _Lease:
    def __init__(self):
        self.mutex = threading.RLock()
        self.depth = 0
        self.handle = None


_GUARD = threading.Lock()
_LEASES = {}


class StateLockBusy(BlockingIOError):
    """An event-loop caller must use its fallback or retry off the loop."""


def on_busy(default=None):
    """Skip optional learned work on contention, without swallowing other faults.

    Only use at a complete optional operation's boundary, never around required
    work such as erasure. Worker threads still wait in StateLock as before.
    """
    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except StateLockBusy:
                return default
        return call
    return decorate


class StateLock:
    def __init__(self, root):
        self.root = Path(root).resolve()
        with _GUARD:
            self._lease = _LEASES.setdefault(self.root, _Lease())

    def __enter__(self):
        lease = self._lease
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            blocking = True
        else:
            blocking = False
        # Both layers must be non-blocking on the loop: a CLI process can hold
        # the flock even when no other thread owns this process's mutex.
        if not lease.mutex.acquire(blocking=blocking):
            raise StateLockBusy("plasticity state is busy")
        try:
            if lease.depth == 0:
                self.root.mkdir(parents=True, exist_ok=True)
                lease.handle = (self.root / ".lock").open("a+b")
                try:
                    fcntl.flock(lease.handle.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
                except BlockingIOError as error:
                    raise StateLockBusy("plasticity state is busy") from error
            lease.depth += 1
        except BaseException:
            if lease.handle is not None:
                lease.handle.close()
                lease.handle = None
            lease.mutex.release()
            raise
        return self

    def __exit__(self, *args):
        lease = self._lease
        try:
            lease.depth -= 1
            if lease.depth == 0:
                fcntl.flock(lease.handle.fileno(), fcntl.LOCK_UN)
                lease.handle.close()
                lease.handle = None
        finally:
            lease.mutex.release()
