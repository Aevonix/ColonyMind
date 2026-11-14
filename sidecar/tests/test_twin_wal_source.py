"""The twin's primary-side readers: WAL frames and per-store deltas (docs/TWIN.md)."""

import os
import random
import shutil
import sqlite3
import struct
import threading
import time

import pytest

from colonymind.twin import wal
from colonymind.twin.segment import iter_pages, read_header, verified, write
from colonymind.twin.source import SqliteSource, file_digest


def _wal_db(path, autocheckpoint=1000):
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute(f'PRAGMA wal_autocheckpoint={autocheckpoint}')
    conn.execute('PRAGMA busy_timeout=5000')
    conn.execute('CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v BLOB)')
    return conn


def _apply(replica, delta):
    """What the standby does with one delta, minus the publication step."""
    if delta.seed is not None:
        shutil.copyfile(delta.seed, replica)
    fd = os.open(replica, os.O_RDWR | os.O_CREAT)
    try:
        for pgno, page in delta.pages.items():
            os.pwrite(fd, page, (pgno - 1) * delta.page_size)
        os.ftruncate(fd, delta.page_count * delta.page_size)
    finally:
        os.close(fd)
    if delta.digest:
        assert file_digest(replica, delta.page_size) == delta.digest


def _rows(path_or_conn, table='t'):
    if isinstance(path_or_conn, sqlite3.Connection):
        return list(path_or_conn.execute(f'SELECT * FROM {table} ORDER BY 1'))
    conn = sqlite3.connect(f'file:{path_or_conn}?mode=ro&immutable=1', uri=True)
    try:
        assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        return list(conn.execute(f'SELECT * FROM {table} ORDER BY 1'))
    finally:
        conn.close()


def _checksum_reference(data, big):
    s0 = s1 = 0
    for i in range(0, len(data), 8):
        x0, x1 = struct.unpack(('>' if big else '<') + 'II', data[i:i + 8])
        s0 = (s0 + x0 + s1) & 0xFFFFFFFF
        s1 = (s1 + x1 + s0) & 0xFFFFFFFF
    return s0, s1


def test_checksum_matches_the_documented_recurrence_both_byte_orders():
    data = os.urandom(4096)
    assert wal.checksum(data, False) == _checksum_reference(data, False)
    assert wal.checksum(data, True) == _checksum_reference(data, True)


def test_scan_returns_only_committed_frames_and_stops_at_a_torn_or_stale_frame(tmp_path):
    db = tmp_path / 's.db'
    conn = _wal_db(db, autocheckpoint=0)
    for _ in range(5):
        conn.execute('INSERT INTO t(v) VALUES (?)', (os.urandom(500),))
    full = wal.scan(str(db) + '-wal')
    assert full.position.frames > 0 and full.db_pages
    # An open transaction spills frames into the file; they are not committed and not read.
    size_before = (tmp_path / 's.db-wal').stat().st_size
    conn.execute('PRAGMA cache_size=1')
    conn.execute('BEGIN')
    for _ in range(50):
        conn.execute('INSERT INTO t(v) VALUES (?)', (os.urandom(3000),))
    assert (tmp_path / 's.db-wal').stat().st_size > size_before
    assert wal.scan(str(db) + '-wal').position == full.position
    conn.execute('ROLLBACK')
    # Corrupt one byte of the last frame: the scan ends at the commit before it.
    size = full.position.page_size + wal.FRAME_HEADER
    raw = bytearray((tmp_path / 's.db-wal').read_bytes())
    last = wal.HEADER + (full.position.frames - 1) * size + wal.FRAME_HEADER + 10
    raw[last] ^= 0xFF
    (tmp_path / 'copy-wal').write_bytes(bytes(raw))
    torn = wal.scan(tmp_path / 'copy-wal')
    assert torn.position.frames < full.position.frames
    # A header from another generation makes every frame stale.
    stale = bytearray(raw)
    stale[16:20] = struct.pack('>I', struct.unpack('>I', stale[16:20])[0] + 1)
    stale[24:32] = struct.pack('>II', *wal.checksum(bytes(stale[:24]), stale[3] == 0x83))
    (tmp_path / 'stale-wal').write_bytes(bytes(stale))
    assert wal.scan(tmp_path / 'stale-wal').position.frames == 0
    assert wal.scan(tmp_path / 'stale-wal', full.position).reset


def test_seed_then_wal_deltas_replicate_bit_for_bit(tmp_path):
    db, replica = tmp_path / 'p.db', tmp_path / 'r.db'
    conn = _wal_db(db)
    conn.execute('INSERT INTO t(v) VALUES (?)', (b'seed',))
    source = SqliteSource('p.db', db, tmp_path / 'scratch')
    first = source.poll()
    assert first.reason == 'seed' and first.seed is not None and first.digest
    _apply(replica, first)
    for round_ in range(20):
        for _ in range(random.randint(0, 8)):
            conn.execute('INSERT INTO t(v) VALUES (?)', (os.urandom(random.randint(1, 3000)),))
        if round_ % 5 == 0:
            conn.execute('DELETE FROM t WHERE id % 3 = 0')
        delta = source.poll()
        if delta is not None:
            assert delta.reason == 'wal' and delta.seed is None
            _apply(replica, delta)
    assert _rows(replica) == _rows(conn)
    check = source.poll(anti_entropy=True)
    assert check.reason == 'anti-entropy' and check.repaired == 0
    _apply(replica, check)          # verifies the whole-store digest
    source.close()


def test_concurrent_writer_with_constant_checkpoints_loses_nothing_and_never_errors(tmp_path):
    db, replica = tmp_path / 'p.db', tmp_path / 'r.db'
    conn = _wal_db(db, autocheckpoint=20)
    stop, errors = threading.Event(), []

    def writer():
        while not stop.is_set():
            try:
                conn.execute('INSERT INTO t(v) VALUES (?)', (os.urandom(random.randint(10, 3000)),))
                if random.random() < 0.2:
                    conn.execute('DELETE FROM t WHERE id IN (SELECT id FROM t ORDER BY random() LIMIT 1)')
            except sqlite3.Error as error:
                errors.append(error)
            time.sleep(random.random() * 0.002)

    source = SqliteSource('p.db', db, tmp_path / 'scratch', release_bytes=300_000)
    thread = threading.Thread(target=writer)
    thread.start()
    try:
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            delta = source.poll()
            if delta is not None:
                _apply(replica, delta)
            time.sleep(random.random() * 0.01)
    finally:
        stop.set()
        thread.join()
    while (delta := source.poll()) is not None:
        _apply(replica, delta)
    assert errors == []
    assert _rows(replica) == _rows(conn)
    assert source.stats.releases > 0, 'the pin must be released so the log can restart'
    # The log restarted under the load, so the primary's WAL stays bounded by the release size.
    assert os.path.getsize(str(db) + '-wal') < 4 * 300_000, os.path.getsize(str(db) + '-wal')
    final = source.poll(anti_entropy=True)
    assert final.repaired == 0
    source.close()


def test_restart_while_unpinned_is_resynchronised_not_trusted(tmp_path):
    db, replica = tmp_path / 'p.db', tmp_path / 'r.db'
    conn = _wal_db(db, autocheckpoint=0)
    conn.execute('INSERT INTO t(v) VALUES (?)', (b'a',))
    source = SqliteSource('p.db', db, tmp_path / 'scratch')
    _apply(replica, source.poll())
    source.unpin()                               # as after a release
    conn.execute('INSERT INTO t(v) VALUES (?)', (b'lost-if-trusted',))
    conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    conn.execute('INSERT INTO t(v) VALUES (?)', (b'b',))
    delta = source.poll()
    assert delta.reason == 'restart-unpinned' and delta.digest
    _apply(replica, delta)
    assert [r[1] for r in _rows(replica)] == [b'a', b'lost-if-trusted', b'b']


def test_rollback_journal_store_ships_changed_pages_on_counter_change(tmp_path):
    db, replica = tmp_path / 'rb.db', tmp_path / 'r.db'
    conn = sqlite3.connect(db, isolation_level=None)
    conn.execute('CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)')
    conn.executemany('INSERT INTO t(v) VALUES (?)', [(str(i) * 50,) for i in range(2000)])
    source = SqliteSource('rb.db', db, tmp_path / 'scratch')
    _apply(replica, source.poll())
    assert source.poll() is None                 # unchanged counter: nothing to do
    conn.execute("UPDATE t SET v = 'changed' WHERE id = 7")
    delta = source.poll()
    assert delta.reason == 'rollback-journal' and delta.seed is None and 0 < len(delta.pages) < 5
    _apply(replica, delta)
    assert _rows(replica) == _rows(conn)
    source.close()


def test_anti_entropy_repairs_what_the_fast_path_missed(tmp_path):
    db, replica = tmp_path / 'p.db', tmp_path / 'r.db'
    conn = _wal_db(db)
    conn.executemany('INSERT INTO t(v) VALUES (?)', [(os.urandom(100),) for _ in range(200)])
    conn.close()                                  # the last close checkpoints and removes the log
    conn = _wal_db(db)
    conn.execute('INSERT INTO t(v) VALUES (?)', (b'tail',))
    source = SqliteSource('p.db', db, tmp_path / 'scratch')
    _apply(replica, source.poll())
    in_log = set(wal.scan(str(db) + '-wal').pages)
    stale = next(p for p in range(2, source.page_size and len(source.hashes) + 1) if p not in in_log)
    # A change the fast path never shipped: the replica and the shipped hashes both hold a
    # page the primary does not.
    garbage = os.urandom(source.page_size)
    with open(replica, 'r+b') as handle:
        handle.seek((stale - 1) * source.page_size)
        handle.write(garbage)
    from colonymind.twin.source import page_hash
    source.hashes[stale - 1] = page_hash(garbage, stale)
    repair = source.poll(anti_entropy=True)
    assert repair.repaired == 1 and stale in repair.pages
    _apply(replica, repair)
    assert _rows(replica) == _rows(conn)
    assert source.stats.anti_entropy_pages == 1
    source.close()


def test_the_source_cannot_write_the_primary_store(tmp_path):
    db = tmp_path / 'p.db'
    _wal_db(db)
    source = SqliteSource('p.db', db, tmp_path / 'scratch')
    source.poll()
    source.unpin()
    with pytest.raises(sqlite3.OperationalError):
        source.conn.execute('INSERT INTO t(v) VALUES (1)')
    source.close()


def test_a_replaced_store_is_seeded_again(tmp_path):
    db = tmp_path / 'p.db'
    _wal_db(db).close()
    source = SqliteSource('p.db', db, tmp_path / 'scratch')
    source.poll()
    source.close()
    os.replace(db, tmp_path / 'old.db')
    fresh = _wal_db(db)
    fresh.execute('INSERT INTO t(v) VALUES (?)', (b'new',))
    delta = source.poll()
    assert delta.reason == 'replaced' and delta.digest
    source.close()


def test_segment_round_trip_and_tamper_detection(tmp_path):
    pages = {1: os.urandom(4096), 7: os.urandom(4096)}
    meta = {'key': 'a.db', 'page_size': 4096, 'page_count': 9, 'reason': 'wal', 'digest': None, 'seed': None}
    path = write(tmp_path, {'generation': 'g', 'seq': 3, 'created_at': 1.0}, [(meta, pages)])
    assert verified(path)
    header, base = read_header(path)
    assert header['seq'] == 3 and dict(iter_pages(path, base, header['entries'][0])) == pages
    raw = bytearray(path.read_bytes())
    raw[-3] ^= 1
    path.write_bytes(bytes(raw))
    assert not verified(path)


def test_a_log_restart_during_a_scan_never_rolls_back_a_shipped_page(tmp_path, monkeypatch):
    """A reader on read-mark 0 does not stop a restart: a scan must never return a torn prefix."""
    path = tmp_path / 'store.db'
    writer = _wal_db(path, autocheckpoint=0)
    writer.execute('CREATE TABLE u (id INTEGER PRIMARY KEY, v BLOB)')
    writer.execute("INSERT INTO t(id, v) VALUES (1, 'one')")
    writer.execute("UPDATE t SET v = 'two' WHERE id = 1")
    writer.execute('PRAGMA wal_checkpoint(PASSIVE)')       # every frame backfilled, the log not restarted
    real_open, state = open, {'seen': False, 'fired': False}

    class Pausing:
        """Reads the log; after the frame that commits 'one', the writer restarts it."""
        def __init__(self, handle):
            self.handle = handle

        def read(self, size=-1):
            if state['seen'] and not state['fired']:
                state['fired'] = True
                writer.execute('INSERT INTO u(v) VALUES (?)', (os.urandom(64 << 10),))
            data = self.handle.read(size)
            if size > wal.HEADER and b'one' in data and b'two' not in data:
                state['seen'] = True
            return data

        def seek(self, *args):
            return self.handle.seek(*args)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return self.handle.__exit__(*exc)

    def pausing_open(name, *args, **kwargs):
        handle = real_open(name, *args, **kwargs)
        return Pausing(handle) if str(name).endswith('-wal') and not state['fired'] else handle

    monkeypatch.setattr(wal, 'open', pausing_open, raising=False)
    source = SqliteSource('store.db', path, tmp_path / 'scratch')
    replica = tmp_path / 'replica.db'
    _apply(replica, source.poll())
    assert state['fired'], 'the restart must have happened in the middle of the scan'
    assert _rows(replica) == [(1, 'two')]
    assert _rows(replica, 'u') == _rows(writer, 'u')
    source.close()


class _RacingPath(type(__import__('pathlib').Path())):
    """The store's path: its second ``stat`` is followed at once by ``race()`` (a checkpoint or
    a commit landing between two separate size samples)."""
    race = None
    calls = 0

    def stat(self, *args, **kwargs):
        result = super().stat(*args, **kwargs)
        if self.name == 'store.db' and type(self).race is not None:
            type(self).calls += 1
            if type(self).calls == 2:
                type(self).race()
        return result


@pytest.mark.parametrize('journal', ['wal', 'rollback'])
def test_a_snapshot_is_budgeted_on_the_pinned_store_not_on_racing_file_sizes(tmp_path, monkeypatch, journal):
    import collections
    from colonymind.twin.source import SpaceError
    path = tmp_path / 'store.db'
    writer = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    writer.execute('PRAGMA page_size=4096')
    if journal == 'wal':
        writer.execute('PRAGMA journal_mode=WAL')
        writer.execute('PRAGMA wal_autocheckpoint=0')
    writer.execute('CREATE TABLE t (id INTEGER PRIMARY KEY, v BLOB)')
    writer.executemany('INSERT INTO t(v) VALUES (?)', [(os.urandom(3000),) for _ in range(100)])
    if journal == 'wal':
        writer.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        # The store doubles, but only in its log: a checkpoint moves it into the main file.
        writer.executemany('INSERT INTO t(v) VALUES (?)', [(os.urandom(3000),) for _ in range(100)])
        race = lambda: writer.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()   # noqa: E731
    else:
        race = lambda: writer.executemany(                                            # noqa: E731
            'INSERT INTO t(v) VALUES (?)', [(os.urandom(3000),) for _ in range(100)])
    main = path.stat().st_size
    usage = collections.namedtuple('usage', 'total used free')
    floor = 64 << 20
    free = floor + main + (main // 2)      # room for the main file as it was, not for the store doubled
    monkeypatch.setattr(shutil, 'disk_usage', lambda _: usage(1 << 40, 0, free))
    source = SqliteSource('store.db', path, tmp_path / 'scratch', min_free_bytes=floor)
    _RacingPath.race, _RacingPath.calls = race, 0
    source.path = _RacingPath(path)
    try:
        delta = source.poll()
    except SpaceError:
        delta = None
    finally:
        _RacingPath.race = None
    if delta is not None:
        taken = delta.seed.stat().st_size if delta.seed is not None else 0
        assert free - taken >= floor, f'the snapshot took {taken} bytes of {free}, leaving less than the floor'
    source.close()
