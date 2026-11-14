"""One SQLite store on the primary: what changed since the last ship, read without writing it.

The store is opened read-only (``mode=ro``), so the replicator cannot alter a primary store.

WAL stores ship the committed frames of their log (``wal.py``). Between ticks the source holds
a read transaction, the *pin*: SQLite cannot restart the log under a reader that uses it, and
a reader on read-mark 0 blocks every backfill, so a restart seen while the pin was held
continuously loses no frame and the source simply follows the new log generation. The pin is
released for one tick when the log passes ``release_bytes`` so the application's checkpoints
can restart it; a restart seen after an unpinned moment may have lost frames, and the source
then *resynchronises*: a consistent snapshot through SQLite's backup API, diffed page by page
against the hashes of what was shipped. Rollback-journal stores are resynchronised whenever
their header's change counter moves. A periodic anti-entropy pass resynchronises every store:
it should find nothing, and the pages it does find are counted and repaired.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from colonymind.twin import wal

SQLITE_MAGIC = b'SQLite format 3\x00'
PAGE_HASH = 16
SMALL_DELTA_BYTES = 1 << 20     # a delta this small ships as pages even when it is most of the store
_EMPTY = b''


def page_hash(page: bytes, pgno: int = 0) -> bytes:
    """A page's hash; page 1 without the two header counters any copy of a database rewrites
    (the file change counter at 24 and version-valid-for at 92), which carry no content."""
    if pgno == 1 and len(page) >= 100:
        page = page[:24] + bytes(4) + page[28:92] + bytes(4) + page[96:]
    return hashlib.blake2b(page, digest_size=PAGE_HASH).digest()


def digest(hashes: list[bytes]) -> str:
    """One digest over the page hashes of a whole store; the twin recomputes it from its file."""
    return hashlib.blake2b(b''.join(hashes), digest_size=32).hexdigest()


def file_digest(path: str | os.PathLike[str], page_size: int) -> str:
    hashes = []
    with open(path, 'rb') as handle:
        for pgno, page in enumerate(iter(lambda: handle.read(page_size), b''), start=1):
            hashes.append(page_hash(page, pgno))
    return digest(hashes)


@dataclass(frozen=True)
class Header:
    page_size: int
    wal_mode: bool
    change_counter: int


def read_header(path: Path) -> Header | None:
    try:
        with open(path, 'rb') as handle:
            data = handle.read(100)
    except OSError:
        return None
    if len(data) < 100 or not data.startswith(SQLITE_MAGIC):
        return None
    (size,) = struct.unpack('>H', data[16:18])
    return Header(65536 if size == 1 else size, data[18] == 2 and data[19] == 2,
                  struct.unpack('>I', data[24:28])[0])


def is_sqlite(path: Path) -> bool:
    try:
        with open(path, 'rb') as handle:
            return handle.read(16) == SQLITE_MAGIC
    except OSError:
        return False


@dataclass
class Delta:
    """What the twin must write to reach the primary's state: pages, then the size in pages."""
    key: str
    page_size: int
    page_count: int
    pages: dict[int, bytes]
    reason: str
    seed: Path | None = None      # a full snapshot to start from, before ``pages``
    digest: str | None = None     # whole-store digest after this delta, for verification
    repaired: int = 0             # pages an anti-entropy pass found the fast path had missed


@dataclass
class Stats:
    wal_frames_pages: int = 0
    resyncs: int = 0
    seeds: int = 0
    lossless_restarts: int = 0
    releases: int = 0
    anti_entropy_runs: int = 0
    anti_entropy_pages: int = 0
    reasons: dict[str, int] = field(default_factory=dict)


class SourceError(RuntimeError):
    pass


class SpaceError(SourceError):
    """A snapshot would take the free space below the floor the live writers need."""


class SqliteSource:
    """Tracks one store; ``poll`` returns the delta since the previous poll, or None."""

    def __init__(self, key: str, path: Path, scratch: Path, *, release_bytes: int = 16 << 20,
                 seed_fraction: float = 0.25, seed_bytes: int = 64 << 20, min_free_bytes: int = 0):
        self.key, self.path, self.scratch = key, Path(path), Path(scratch)
        self.min_free_bytes = min_free_bytes
        self.release_bytes, self.seed_fraction, self.seed_bytes = release_bytes, seed_fraction, seed_bytes
        self.conn: sqlite3.Connection | None = None
        self.position: wal.Position | None = None
        self.pinned = False
        self.hashes: list[bytes] = []
        self.page_size = 0
        self.identity: tuple[int, int] | None = None
        self.counter: int | None = None
        self.seeded = False
        self.released_at_frames = 0
        self.stats = Stats()

    @property
    def wal_path(self) -> Path:
        return self.path.with_name(self.path.name + '-wal')

    # -- connection and pin ------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        if self.conn is None:
            uri = self.path.resolve().as_uri() + '?mode=ro'
            self.conn = sqlite3.connect(uri, uri=True, isolation_level=None, check_same_thread=False, timeout=0.5)
        return self.conn

    def _repin(self) -> None:
        conn = self._connect()
        if conn.in_transaction:
            conn.execute('COMMIT')
        conn.execute('BEGIN')
        conn.execute('SELECT count(*) FROM sqlite_master').fetchone()
        self.pinned = True

    def unpin(self) -> None:
        if self.conn is not None and self.conn.in_transaction:
            try:
                self.conn.execute('COMMIT')
            except sqlite3.Error:
                pass
        self.pinned = False

    def close(self) -> None:
        self.unpin()
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def forget(self) -> None:
        """Ship the store from nothing on the next poll (a new generation)."""
        self.close()
        self.position, self.hashes, self.seeded, self.identity, self.counter = None, [], False, None, None

    # -- polling -----------------------------------------------------------------------
    def poll(self, *, anti_entropy: bool = False) -> Delta | None:
        header = read_header(self.path)
        if header is None:
            raise SourceError('store is missing or not a SQLite database')
        stat = self.path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if not self.seeded or identity != self.identity or header.page_size != self.page_size:
            self.close()
            return self._guarded('seed' if not self.seeded else 'replaced', header, identity)
        if not header.wal_mode:
            self.unpin()
            self.position = None
            if header.change_counter != self.counter:
                return self._guarded('rollback-journal', header, identity)
            if anti_entropy:
                try:
                    return self._guarded('anti-entropy', header, identity)
                except SpaceError:
                    return None          # a check with no room waits for the next pass
            return None
        if self.position is None and not self.pinned:
            return self._guarded('entered-wal', header, identity)
        if anti_entropy and self.pinned and self._shortfall() is None:
            return self._guarded('anti-entropy', header, identity)
        return self._poll_wal(header, identity)     # a check with no room waits; the fast path does not

    def _shortfall(self) -> str | None:
        """A snapshot is a full copy of the store: it is taken only when the free space left
        after it stays above the floor, so replication never takes the space live writes need.

        It is measured inside the pin, on the snapshot the backup will copy: its page count
        times its page size is exactly what the backup writes, whatever a checkpoint does to the
        files' sizes meanwhile (two separate file sizes can miss a checkpoint between them). The
        log is added: its frames ship with the snapshot.
        """
        conn = self.conn
        if not (self.pinned and conn is not None and conn.in_transaction):
            raise SourceError(f'{self.key}: a snapshot is budgeted only inside the pin')
        self.scratch.mkdir(parents=True, exist_ok=True)
        page_size = conn.execute('PRAGMA page_size').fetchone()[0]
        needed = conn.execute('PRAGMA page_count').fetchone()[0] * page_size
        try:
            needed += self.wal_path.stat().st_size
        except FileNotFoundError:
            pass
        free = shutil.disk_usage(self.scratch).free
        if free - needed < self.min_free_bytes:
            return f'{self.key}: a snapshot needs {needed} bytes, {free} free, floor {self.min_free_bytes}'
        return None

    def _guarded(self, reason: str, header: Header, identity) -> Delta | None:
        """A failed snapshot leaves a gap in the pin: force a full resynchronisation next time."""
        try:
            return self._snapshot(reason, header, identity)
        except BaseException:
            self.unpin()
            self.position = None
            raise

    def _poll_wal(self, header: Header, identity) -> Delta | None:
        first = wal.scan(self.wal_path, self.position)
        if first.reset and not self.pinned:
            return self._guarded('restart-unpinned', header, identity)
        if first.reset:
            self.stats.lossless_restarts += 1
        pages, count = dict(first.pages), first.db_pages
        position = first.position if first.position is not None else (None if first.reset else self.position)
        frames = position.frames if position is not None else 0
        if frames * (wal.FRAME_HEADER + header.page_size) >= self.release_bytes and frames != self.released_at_frames:
            # Let the application's checkpoint restart the log; the next poll resynchronises
            # only if a restart really happened while unpinned.
            self.unpin()
            self.released_at_frames = frames
            self.stats.releases += 1
        else:
            self._repin()
            try:
                second = wal.scan(self.wal_path, position)
            except BaseException:
                # Whether the log restarted in the repin's gap is unknown: resynchronise.
                self.unpin()
                self.position = None
                raise
            if second.reset:
                return self._guarded('restart-in-gap', header, identity)
            pages.update(second.pages)
            count = second.db_pages or count
            position = second.position or position
        self.position = position
        if not pages and count is None:
            return None
        self.stats.wal_frames_pages += len(pages)
        return self._commit(pages, count or len(self.hashes), 'wal', header.page_size)

    def _commit(self, pages: dict[int, bytes], count: int, reason: str, page_size: int,
                *, seed: Path | None = None, with_digest: bool = False, repaired: int = 0) -> Delta:
        hashes = self.hashes
        if len(hashes) < count:
            hashes.extend([page_hash(bytes(page_size))] * (count - len(hashes)))
        del hashes[count:]
        for pgno, page in pages.items():
            if pgno <= count:
                hashes[pgno - 1] = page_hash(page, pgno)
        pages = {pgno: page for pgno, page in pages.items() if pgno <= count}
        self.stats.reasons[reason] = self.stats.reasons.get(reason, 0) + 1
        return Delta(self.key, page_size, count, pages, reason, seed,
                     digest(hashes) if with_digest else None, repaired)

    def _snapshot(self, reason: str, header: Header, identity) -> Delta | None:
        """A consistent snapshot diffed against the shipped hashes (a seed when mostly new)."""
        conn = self._connect()
        if not (header.wal_mode and self.pinned):
            self._repin()
        shortfall = self._shortfall()
        if shortfall is not None:
            raise SpaceError(shortfall)
        fd, name = tempfile.mkstemp(prefix='.snap-', dir=self.scratch)
        os.close(fd)
        snapshot = Path(name)
        try:
            target = sqlite3.connect(snapshot)
            try:
                conn.backup(target)      # inside the pin's read transaction: that snapshot exactly
            finally:
                target.close()
            if header.wal_mode:
                # Replaying the whole current log generation onto the snapshot yields the latest
                # commit: its frames up to the snapshot are already in it, and while the pin is
                # held the log restarts at most once, and only when every frame was backfilled.
                # A restart racing the read is caught by ``wal.scan`` (it reads the log again).
                overlay = wal.scan(self.wal_path)
            else:
                self.unpin()
                overlay = wal.Scan(None, {}, None, False)
            fresh = read_header(snapshot)
            page_size = fresh.page_size if fresh else header.page_size
            new_hashes, changed = [], {}
            limit = max(self.seed_bytes // page_size, 1)
            mostly_new = not self.seeded or page_size != self.page_size
            with open(snapshot, 'rb') as handle:
                for index, page in enumerate(iter(lambda: handle.read(page_size), b''), start=1):
                    value = page_hash(page, index)
                    new_hashes.append(value)
                    if mostly_new:
                        continue
                    if index > len(self.hashes) or self.hashes[index - 1] != value:
                        if len(changed) >= limit:
                            mostly_new, changed = True, {}
                        else:
                            changed[index] = page
            count = len(new_hashes)
            if (not mostly_new and len(changed) > self.seed_fraction * max(count, 1)
                    and len(changed) * page_size > SMALL_DELTA_BYTES):
                mostly_new, changed = True, {}
            if overlay.db_pages:
                count = overlay.db_pages
            self.page_size, self.identity, self.counter = page_size, identity, header.change_counter
            self.position = overlay.position if header.wal_mode else None
            self.pinned = header.wal_mode
            self.seeded = True
            self.stats.resyncs += 1
            if mostly_new:
                # The snapshot already holds the log's frames up to its point; ship only newer ones.
                self.stats.seeds += 1
                self.hashes = new_hashes
                newer = {p: page for p, page in overlay.pages.items()
                         if p > len(new_hashes) or new_hashes[p - 1] != page_hash(page, p)}
                delta = self._commit(newer, count, reason, page_size, seed=snapshot, with_digest=True)
                snapshot = None
                return delta
            # Ship only what differs from what was shipped: pages of the snapshot that changed
            # and log pages whose final image is new. A page the standby altered on its own is
            # therefore not overwritten silently; the digest check catches it.
            repaired = len(set(changed) - set(overlay.pages)) if reason == 'anti-entropy' else 0
            if reason == 'anti-entropy':
                self.stats.anti_entropy_runs += 1
                self.stats.anti_entropy_pages += repaired
            shipped = self.hashes
            ship = {p: page for p, page in changed.items() if p not in overlay.pages}
            for pgno, page in overlay.pages.items():
                if pgno > len(shipped) or shipped[pgno - 1] != page_hash(page, pgno):
                    ship[pgno] = page
            return self._commit(ship, count, reason, page_size, with_digest=True, repaired=repaired)
        finally:
            if snapshot is not None:
                snapshot.unlink(missing_ok=True)
