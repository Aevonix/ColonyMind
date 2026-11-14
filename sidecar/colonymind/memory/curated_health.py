"""Process-local curated incidents. Recall records flags; the existing outbox pull reports them."""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
import threading
from uuid import uuid4

from colonymind.util.temporal import now_utc


class _Health:
    def __init__(self):
        self.lock = threading.Lock()
        self.fail_open_recalls = 0
        # Monotonic across recovery and shared by every facade for this path.
        # A read that crossed any protection failure cannot publish its result.
        self.failure_version = 0
        self.incident = None
        self.last_error_class = None
        self.pending = deque()
        self.delivery_lock = threading.Lock()

    def degrade(self, error):
        with self.lock:
            self.failure_version += 1
            if self.incident is None:
                if error is not None:
                    self.last_error_class = type(error).__name__
                self.incident = {
                    'id': uuid4().hex, 'first_seen': now_utc().isoformat(),
                    'error_class': self.last_error_class,
                }
                self.pending.append(self.incident)

    def recover(self):
        with self.lock:
            self.incident = None

    def snapshot(self):
        with self.lock:
            incident = self.incident or {}
            return {'fail_open_recalls': self.fail_open_recalls, 'degraded': self.incident is not None,
                    'first_seen': incident.get('first_seen'), 'error_class': incident.get('error_class')}


@lru_cache(maxsize=None)
def _health(path):
    return _Health()


def for_path(path):
    return _health(str(path.resolve()))


def for_ledger(ledger):
    # Voice's borrowed facade and the native handle share the same incident.
    health = getattr(ledger, '_curated_health', None)
    if health is None:
        health = ledger._curated_health = for_path(ledger.db_path)
    return health


_recall = ContextVar('curated_recall', default=None)


@contextmanager
def recalling(scope=None):
    """One count across preparation, version checks, voice fill and plain-packet rebuilds."""
    active = _recall.get()
    scope = active if active is not None else (scope if scope is not None else set())
    token = _recall.set(scope)
    try:
        yield scope
    finally:
        _recall.reset(token)


def failed_open(ledger):
    health = for_ledger(ledger)
    # Reinitializing another handle can recover the path while an older,
    # disabled handle still fails open. That next recall opens a new incident.
    if health.incident is None:
        health.degrade(None)
    scope = _recall.get()
    with health.lock:
        health.failure_version += 1
        if scope is None or health not in scope:
            health.fail_open_recalls += 1
            if scope is not None:
                scope.add(health)


def owner_notices(ledger, outbox):
    """Drain on the existing mind outbox poll, never on a recall worker.

    Keep incidents pending through recovery and failed queue writes. The outbox's
    durable dedup key also covers a queue write that committed before raising.
    """
    if ledger is None or not outbox.owner_id:
        return
    health = for_ledger(ledger)
    with health.delivery_lock:
        while True:
            with health.lock:
                if not health.pending:
                    return
                incident = health.pending[0]
            key = 'curated-fail-open:' + incident['id']
            outbox.notice(
                type='health_notice', title='Curated memory protection is degraded', dedup_key=key,
                text=('Recall is withheld while curated memory protection is unavailable. '
                      'Nothing forgotten can resurface through recall. Restore curated memory protection.'),
                extra={'topic': key})
            with health.lock:
                health.pending.popleft()
