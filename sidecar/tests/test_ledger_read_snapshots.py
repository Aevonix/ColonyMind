"""Per-snapshot canonical readers retain read-only, thread and WAL boundaries."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import sqlite3
import threading
import time

import pytest

from colonymind.turns.idempotency import TurnIdempotencyLedger
from test_voice_affect import affect_app, add_appraisal, call, stamp  # noqa: F401


def assert_closed(connection):
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        connection.execute('SELECT 1')


def test_snapshot_owns_a_readonly_connection(tmp_path, monkeypatch):
    ledger = TurnIdempotencyLedger(tmp_path / 'sources.db')
    connect = sqlite3.connect
    opened = []

    def observe(*args, **kwargs):
        opened.append((args, kwargs))
        return connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, 'connect', observe)
    with ledger.read_snapshot() as first:
        assert first.execute('PRAGMA query_only').fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            first.execute('DELETE FROM turn_sources')
        # File-level read-only mode remains in force even if query_only is off.
        first.execute('PRAGMA query_only=OFF')
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            first.execute('DELETE FROM turn_sources')
        first.execute('PRAGMA query_only=ON')
    assert_closed(first)
    with ledger.read_snapshot() as second:
        assert second is not first
    assert_closed(second)
    assert len(opened) == 2
    assert opened[0][0][0].endswith('?mode=ro')
    assert opened[0][1]['uri'] is True


def test_snapshot_observes_commits_and_releases_wal(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'sources.db')
    with closing(ledger._connect()) as writer:
        with writer:
            writer.execute('CREATE TABLE snapshot_probe (value INTEGER)')
            writer.execute('INSERT INTO snapshot_probe VALUES (1)')
        with ledger.read_snapshot() as reader:
            assert reader.execute('SELECT value FROM snapshot_probe').fetchone()[0] == 1
            with writer:
                writer.execute('UPDATE snapshot_probe SET value=2')
            assert reader.execute('SELECT value FROM snapshot_probe').fetchone()[0] == 1
        assert_closed(reader)
        with ledger.read_snapshot() as reader:
            assert reader.execute('SELECT value FROM snapshot_probe').fetchone()[0] == 2
        assert tuple(writer.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()) == (0, 0, 0)


def test_snapshot_releases_after_failure_and_supports_nested_reads(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'sources.db')
    with pytest.raises(sqlite3.OperationalError, match='no such table'):
        with ledger.read_snapshot() as reader:
            reader.execute('SELECT * FROM missing_table')
    assert_closed(reader)
    with ledger.read_snapshot() as outer:
        with ledger.read_snapshot() as inner:
            assert inner is not outer
            assert inner.execute('SELECT count(*) FROM turn_sources').fetchone()[0] == 0
        assert outer.in_transaction
    assert_closed(inner)
    assert_closed(outer)


def test_snapshot_connections_belong_to_each_ledger_and_thread(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'sources.db')
    other = TurnIdempotencyLedger(ledger.db_path)
    with ledger.read_snapshot() as main:
        with other.read_snapshot() as separate:
            assert separate is not main

    def read():
        with ledger.read_snapshot() as worker:
            assert worker.execute('SELECT count(*) FROM turn_sources').fetchone()[0] == 0
            return worker

    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(read).result()
        second = pool.submit(read).result()
        assert second is not first
        assert all(conn is not main and conn is not separate for conn in (first, second))
        assert_closed(first)
        assert_closed(second)


def test_context_worker_keeps_its_deadline_and_connection_lifetime(tmp_path):
    from colonymind.memory import producers
    ledger = TurnIdempotencyLedger(tmp_path / 'sources.db')
    with ledger.read_snapshot() as ordinary:
        pass
    read = producers._Read(time.monotonic() + 10, threading.Event())
    token = producers._READ.set(read)
    try:
        with ledger.read_snapshot() as worker:
            assert worker is not ordinary
            assert worker.execute('SELECT count(*) FROM turn_sources').fetchone()[0] == 0
            read.abandoned.set()
            with pytest.raises(sqlite3.OperationalError, match='deadline'):
                worker.execute('SELECT 1')
        with pytest.raises(sqlite3.ProgrammingError, match='closed'):
            sqlite3.Connection.execute(worker, 'SELECT 1')
    finally:
        producers._READ.reset(token)
    with ledger.read_snapshot() as reader:
        assert reader is not ordinary
        assert reader.execute('SELECT 1').fetchone()[0] == 1


async def test_voice_readers_avoid_writer_setup_and_observe_erasure(affect_app, monkeypatch):
    state = affect_app
    source_id = add_appraisal(state, 'interest')
    lineage, _ = state.affect.source_input(source_id, 'owner')
    state.affect.create_event(contact_id='owner', valence=-0.8, arousal=0.1,
                              timestamp=stamp(state.now), source_lineage=lineage)
    connect = state.ledger._connect

    def unexpected_open():
        pytest.fail('voice read opened a writer connection or repeated schema setup')

    monkeypatch.setattr(state.ledger, '_connect', unexpected_open)
    for _ in range(3):
        assert await call(state) == 'Sam: frustrated, low energy · Nova: interested'
    with closing(connect()) as writer, writer:
        writer.execute('INSERT INTO source_projection_erasures VALUES (?,?)',
                       (source_id, 'external-erasure'))
    for _ in range(3):
        assert await call(state) == ''
    assert state.affect._conn.execute('SELECT count(*) FROM affect_events').fetchone()[0] == 1


def test_building_the_appraisal_view_opens_no_connection(tmp_path, monkeypatch):
    """The ledger set up and migrated the appraisal schema when it opened, also on an older file."""
    from colonymind.self_model.appraisals import AppraisalStore
    path = tmp_path / 'sources.db'
    with closing(sqlite3.connect(path)) as conn, conn:
        # An older store: ``appraisal_runs`` without ``enqueued_at``.
        conn.execute('''CREATE TABLE turn_sources (turn_id TEXT PRIMARY KEY, content_sha256 TEXT NOT NULL,
            contact_id TEXT NOT NULL, session_id TEXT NOT NULL, scope TEXT NOT NULL, messages_json TEXT NOT NULL,
            occurred_at TEXT, ingested_at TEXT NOT NULL)''')
        conn.execute("INSERT INTO turn_sources VALUES ('t-1','x','c','s','person','[]',NULL,'2026-09-20T10:00:00.000Z')")
        conn.execute('''CREATE TABLE appraisal_runs (turn_id TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0, next_attempt REAL NOT NULL DEFAULT 0, disposition TEXT, error TEXT)''')
        conn.execute("INSERT INTO appraisal_runs(turn_id) VALUES ('t-1')")
    ledger = TurnIdempotencyLedger(path)
    connect = sqlite3.connect
    opened = []

    def observe(*args, **kwargs):
        opened.append((args, kwargs))
        return connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, 'connect', observe)
    store = AppraisalStore(ledger, owner_id='owner')
    assert opened == []
    # A reader of the migrated column works on the view as built.
    assert set(store.pending_jobs()) == {'pending', 'running'}
    assert len(opened) == 1


def test_present_read_connections_are_as_before(tmp_path):
    """The snapshot's statement-level refusal and its use from another thread are its own."""
    ledger = TurnIdempotencyLedger(tmp_path / 'sources.db')
    with closing(ledger._read_connect()) as conn:
        assert conn.execute('PRAGMA query_only').fetchone()[0] == 0
        assert not conn.in_transaction
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            conn.execute('DELETE FROM turn_sources')
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(sqlite3.ProgrammingError, match='same thread'):
                pool.submit(conn.execute, 'SELECT 1').result()
