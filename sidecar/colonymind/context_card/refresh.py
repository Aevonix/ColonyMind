"""Coalesced change detection and publication, entirely outside GET."""
from __future__ import annotations

import asyncio
from contextlib import closing
from dataclasses import replace
from functools import lru_cache
import logging
import os
from pathlib import Path
import time
import sqlite3

from colonymind.reports.instance import signature
from colonymind.util import temporal
from .builder import build, fingerprint
from .store import CardStore

logger = logging.getLogger(__name__)


@lru_cache(maxsize=16)
def _resolved_owner(person: str, contact: str, legacy: str) -> str:
    from colonymind.api.auth import owner_person_id
    from colonymind.identity import get_owner_contact_id
    # No implicit 'owner' binding for a private derived card; it must match the
    # identity the existing API authority checks use.
    owner = person or get_owner_contact_id()
    return owner if owner and owner == owner_person_id() else ""


def configured_owner() -> str:
    # Cache the deprecated resolver per binding; environment changes still
    # invalidate immediately, without repeating the legacy warning on GET.
    return _resolved_owner(*(os.environ.get(key, "").strip() for key in
        ("COLONYMIND_OWNER_PERSON_ID", "COLONYMIND_OWNER_CONTACT_ID", "COLONYMIND_HOST_CONTACT_ID")))


def local_day() -> str:
    return temporal.now_in(temporal.agent_timezone()).date().isoformat()


def _wal_signature(path: Path):
    """WAL signature with an empty WAL treated as absent.

    Opening or closing a reader creates or deletes an empty WAL, which holds no
    commits. Any commit grows the WAL, and a checkpoint that truncates or
    removes a non-empty WAL rewrites the main file, so real changes still
    change the main or WAL signature.
    """
    wal = signature(Path(str(path) + "-wal"))
    return wal if wal and wal[1] else None


class RevisionProbe:
    """File/WAL change detection and a separate, read-only privacy watermark.

    WAL signatures catch commits before checkpoint. The periodic collector
    reconciles in-memory changes and time-dependent producers as well.
    """
    NAMES = ("turn-idempotency.db", "mind.db", "initiatives.db", "colonymind-work.db",
             "colonymind-preferences.db", "colonymind-commitments.db", "colonymind-comms.db")

    def __init__(self, state_dir: Path, *, owner_provider=configured_owner, day_provider=local_day):
        self.state_dir = state_dir
        self.owner_provider = owner_provider
        self.day_provider = day_provider
        self._privacy_key = None
        self._watermark = None

    def binding(self):
        """Only query the erasure head when the ledger metadata or owner changes.

        No source hydration, initialization or shared producer connections on
        GET. A missing ledger is distinct from a previously present ledger.
        Unreadable erasure metadata fails closed instead of reusing a watermark.
        """
        owner = self.owner_provider()
        path = self.state_dir / "turn-idempotency.db"
        key = (owner, signature(path), _wal_signature(path))
        if key != self._privacy_key:
            if key[1] is None:
                watermark = None
            else:
                with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True,
                                             timeout=0.025)) as conn:
                    conn.execute("PRAGMA query_only=ON")
                    from colonymind.memory import curated
                    watermark = curated.watermark(None, owner, conn=conn)
            self._privacy_key, self._watermark = key, watermark
        return owner, self._watermark

    def __call__(self):
        from colonymind.config import instance_home
        databases = [self.state_dir / name for name in self.NAMES]
        files = databases + [instance_home() / "identity.yaml", instance_home() / "colonymind.yaml"]
        files += [Path(os.environ.get("COLONYMIND_EVENT_JOURNAL_DIR") or self.state_dir / "events") / ".cursor"]
        return (self.owner_provider(), self.day_provider(), os.environ.get("COLONYMIND_OWNER_NAME", ""),
                os.environ.get("COLONYMIND_CONTEXT_CHARS_PER_TOKEN", ""),
                tuple(signature(path) for path in files) + tuple(_wal_signature(path) for path in databases))


class CardService:
    def __init__(self, store: CardStore, collector, probe, *, owner_provider=configured_owner,
                 clock=time.monotonic, reconcile_seconds=300, max_postponed_passes=2):
        self.store, self.collector, self.probe = store, collector, probe
        self.owner_provider, self.clock = owner_provider, clock
        self.reconcile_seconds = reconcile_seconds
        self.revision = None  # Persisted bytes are untrusted until reconciled.
        self.next_check = 0.0
        self.error = False
        if type(max_postponed_passes) is not int or max_postponed_passes < 1:
            raise ValueError("context_card_invalid_postponement_limit")
        self.max_postponed_passes = max_postponed_passes
        self._postponed = 0
        self._privacy_revision = None
        self._refresh_lock = asyncio.Lock()
        store.load(owner_provider())

    def fetch(self):
        """Only cached bytes and cheap revision metadata; never collect/build."""
        card = self.store.current
        if self.revision is None or card is None or card.owner_id != self.owner_provider():
            return None
        try:
            if self.probe.binding() != self._privacy_revision:
                return None
            changed = self.probe() != self.revision
        except (OSError, sqlite3.Error):
            return None
        return replace(card, stale=self.error or changed or bool(self._postponed))

    async def refresh(self) -> bool:
        async with self._refresh_lock:
            return await self._refresh()

    async def _refresh(self) -> bool:
        try:
            before = self.probe()
            privacy_before = self.probe.binding()
            if (self.store.current and (self.store.current.owner_id != privacy_before[0]
                    or (self._privacy_revision is not None and privacy_before != self._privacy_revision))):
                self.store.purge()
            if (before == self.revision and self.clock() < self.next_check
                    and not self.error and not self._postponed and self.store.current is not None):
                return False
            # Capture shared producer state on its loop, then hand only detached
            # values to the ledger worker. Finish it before lifecycle shutdown.
            if hasattr(self.collector, "prepare"):
                prepared = self.collector.prepare()
                task = asyncio.create_task(asyncio.to_thread(self.collector.collect_ledger, prepared))
            else:
                task = asyncio.create_task(asyncio.to_thread(self.collector.collect))
            try:
                inputs = await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise
            if inputs.owner_id != self.owner_provider():
                raise ValueError("context_card_owner_changed")
            current = self.store.current
            bound_owner, watermark = self.store.binding()
            erased = bool(bound_owner) and (bound_owner != inputs.owner_id
                                           or watermark != inputs.erasure_watermark)
            if erased:
                self.store.purge()
            # A metadata-only write can trigger collection without rendering.
            same = current and not erased and current.input_fingerprint == fingerprint(inputs)
            built = None if same else build(inputs)
            after = self.probe()
            privacy_after = self.probe.binding()
            if (privacy_before != privacy_after or inputs.owner_id != privacy_after[0]
                    or (privacy_after[1] is not None and inputs.erasure_watermark != privacy_after[1])):
                # Never force a publication across an erasure or owner rebind.
                if self.store.current and (self.store.current.owner_id != privacy_after[0]
                        or self.store.current.erasure_watermark != privacy_after[1]):
                    self.store.purge()
                self._postponed = 0
                return False
            if before != after:
                self._postponed += 1
                if self._postponed < self.max_postponed_passes:
                    return False
            if same:
                self.store.checked(replace(current, checked_at=temporal.now_utc().isoformat(),
                                           source_refs=inputs.source_refs, stale=False))
            else:
                self.store.publish(built, inputs)
            racing = before != after
            self.revision, self.error = before if racing else after, False
            self._privacy_revision = privacy_after
            self._postponed = 0
            self.next_check = self.clock() + (0 if racing else self.reconcile_seconds)
            return not same and (current is None or current.text != built.text)
        except Exception as error:
            self.error = True
            # No private text or exception strings in the log.
            from colonymind.health_counters import bump
            bump("context_card_refresh_failed")
            logger.warning("Context card background refresh failed (%s, sqlite_errorname=%s)",
                           type(error).__name__, getattr(error, "sqlite_errorname", None))
            return False

    async def run(self, *, interval_seconds=5):
        while True:
            await self.refresh()
            await asyncio.sleep(interval_seconds)
