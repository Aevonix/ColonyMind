"""Bounded request work and revision stability at the review's history size.

Run the offline HTTP benchmark from sidecar/ with:
COLONYMIND_WORK_BENCHMARK=1 ../.venv/bin/python -m pytest -p no:cacheprovider \
    tests/test_work_scaling.py -k benchmark -s
It asserts p99 admission below 50 ms and projection visibility within 2 seconds.
"""
from contextlib import closing
import asyncio
import json
import math
import os
import sqlite3
import threading
import time

from httpx import ASGITransport, AsyncClient
import pytest

from colonymind.work import get_work_store
from colonymind.work import ledger as facts
from test_work_ingestion import app, payload, state  # noqa: F401
from test_work_projection import admit, envelope, operation, stores  # noqa: F401


def _derived(stores):
    admit(stores)
    stores.work.reconcile()
    before = stores.work.get_root('root', contact_id='owner')
    assert stores.work.write_derived('root', contact_id='owner', expected_revision=before['revision'],
                                    title='Checked file update', overlay='confirmed')
    return before['revision']


def _assert_derived(stores, revision):
    stores.work.reconcile()
    current = stores.work.get_root('root', contact_id='owner')
    assert current['revision'] == revision
    assert current['title'] == 'Checked file update'
    with closing(stores.work._connect()) as conn:
        assert conn.execute('SELECT verdict FROM work_overlays WHERE root_id=?', ('root',)).fetchone()[0] == 'confirmed'


def test_temporary_hands_database_failure_preserves_root_revision_and_derivatives(stores):
    revision = _derived(stores)
    (stores.path / 'hands.db').write_bytes(b'temporarily unreadable sqlite database')
    _assert_derived(stores, revision)
    (stores.path / 'hands.db').unlink()
    _assert_derived(stores, revision)


def test_projection_reads_never_begin_a_ledger_write(stores, monkeypatch):
    _derived(stores)
    original = stores.ledger._connect
    statements = []

    def connect():
        conn = original()

        def authorize(action, first, second, database, origin):
            if action in {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE}:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        conn.set_authorizer(authorize)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(stores.ledger, '_connect', connect)
    with closing(original()) as writer:
        writer.execute('BEGIN IMMEDIATE')
        assert stores.work.list_roots(contact_id='owner')
        assert stores.work.get_root('root', contact_id='owner')
        assert stores.work.summary(contact_id='owner')['total'] == 1
        writer.rollback()
    assert not any(query.upper().startswith(('BEGIN IMMEDIATE', 'BEGIN EXCLUSIVE')) for query in statements)


@pytest.mark.asyncio
async def test_admission_returns_while_only_its_root_rebuilds_off_loop(state, monkeypatch):
    store = get_work_store()
    assert store.wait_pending(timeout=2)
    for ident in ('existing-one', 'existing-two'):
        state.ledger.record_source(ident, contact_id='owner', session_id='session',
            messages=[{'role': 'user', 'content': f'Earlier request {ident}'}],
            work={'envelope': envelope(ident), 'principal_id': 'api-key'}, derive_claims=False)
    store.reconcile()
    rebuild = store.rebuild_root
    entered, release = threading.Event(), threading.Event()
    calls = []

    def paused(ident):
        calls.append((ident, threading.get_ident()))
        entered.set()
        release.wait(2)
        rebuild(ident)

    monkeypatch.setattr(store, 'rebuild_root', paused)
    try:
        async with AsyncClient(transport=ASGITransport(app=app()), base_url='http://example.test') as client:
            started = time.perf_counter()
            response = await client.post('/v1/host/turns/sync', json=payload('new-root'))
            assert time.perf_counter() - started < .5
            assert response.status_code == 200 and response.json()['work_recorded'] is True
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(.005)
            assert entered.is_set()
            heartbeat = asyncio.Event()
            asyncio.get_running_loop().call_soon(heartbeat.set)
            await asyncio.wait_for(heartbeat.wait(), timeout=.5)
            assert store.get_root('new-root', contact_id='owner') is None
            assert store.summary(contact_id='owner')['total'] == 2
    finally:
        release.set()
        assert store.wait_pending(timeout=2)
    assert [ident for ident, _ in calls] == ['new-root']
    assert calls[0][1] != threading.get_ident()
    assert store.get_root('new-root', contact_id='owner')['status'] == 'proven'


def _seed_review_history(ledger, *, history=20_000, roots=1_500):
    """Seed offline facts in one transaction; setup is outside the measured path."""
    stamp = '2026-09-01T00:00:00+00:00'
    with closing(ledger._connect()) as conn, conn:
        facts.initialize(conn)
        conn.executemany('''INSERT INTO turn_sources
            (turn_id,content_sha256,contact_id,session_id,scope,messages_json,occurred_at,ingested_at)
            VALUES (?,?,?,?,?,?,?,?)''',
            [(f'history-{index}', 'a' * 64, 'owner', 'history', 'person',
              json.dumps([{'role': 'user', 'content': f'Historical request {index} ' + 'x' * 1200},
                          {'role': 'assistant', 'content': 'Recorded'}]), None, stamp)
             for index in range(history)])
        for index in range(roots):
            ident = f'existing-root-{index}'
            messages = [{'role': 'user', 'content': f'Existing request {index}'}]
            conn.execute('''INSERT INTO turn_sources
                (turn_id,content_sha256,contact_id,session_id,scope,messages_json,occurred_at,ingested_at)
                VALUES (?,?,?,?,?,?,?,?)''',
                (ident, 'a' * 64, 'owner', 'session', 'person', json.dumps(messages), None, stamp))
            facts.admit_turn(conn, work=envelope(ident, operations=[
                operation(f'{ident}-operation-{number}') for number in range(3)]),
                source_id=ident, contact_id='owner', session_id='session', messages=messages,
                principal_id='api-key', occurred_at=stamp)


@pytest.mark.skipif(os.environ.get('COLONYMIND_WORK_BENCHMARK') != '1',
                    reason='offline 20k-source benchmark is opt-in')
@pytest.mark.asyncio
async def test_owner_admission_benchmark_at_review_scale(state):
    _seed_review_history(state.ledger)
    store = get_work_store()
    started = time.perf_counter()
    store.reconcile()
    maintenance_seconds = time.perf_counter() - started
    assert store.summary(contact_id='owner')['total'] == 1_500
    durations, projection_delays = [], []
    async with AsyncClient(transport=ASGITransport(app=app()), base_url='http://example.test') as client:
        for index in range(101):
            ident = f'measured-root-{index}'
            body = payload(ident)
            body['user_message']['content'] = f'Please update report {index}'
            body['work']['operations'] = [operation(f'{ident}-operation-{number}') for number in range(3)]
            started = time.perf_counter()
            response = await client.post('/v1/host/turns/sync', json=body)
            returned = time.perf_counter()
            assert response.status_code == 200, response.text
            assert response.json()['work_recorded'] is True
            assert store.wait_pending(timeout=2), f'Projection did not settle for {ident}'
            current = store.get_root(ident, contact_id='owner')
            visible = time.perf_counter()
            assert current and current['status'] == 'proven' and len(current['effects']) == 3
            if index:  # Warm imports, connection caches and the worker before measurement.
                durations.append((returned - started) * 1000)
                projection_delays.append(visible - returned)
    p99_ms = sorted(durations)[math.ceil(len(durations) * .99) - 1]
    metrics = {'history_source_rows': 20_000, 'existing_roots': 1_500,
               'operations_per_root': 3, 'measured_requests': len(durations),
               'request_p99_ms': round(p99_ms, 3), 'request_max_ms': round(max(durations), 3),
               'projection_max_seconds': round(max(projection_delays), 6),
               'maintenance_seconds': round(maintenance_seconds, 3)}
    print('\nWORK_BENCHMARK ' + json.dumps(metrics, sort_keys=True))
    assert p99_ms < 50, metrics
    assert max(projection_delays) < 2, metrics


from colonymind.memory import curated
def test_unrelated_owner_erasure_preserves_root_revision_and_derivatives(stores):
    revision = _derived(stores)
    stores.ledger.record_source('unrelated', contact_id='owner', session_id='other-session', messages=[{'role': 'user', 'content': 'An unrelated request'}], derive_claims=False)
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(stores.ledger)
    curated.forget_sources(stores.ledger, 'owner', source_ids=['unrelated'])
    assert historical_bytes(stores.ledger) == _native_before_withdrawal
    _assert_derived(stores, revision)

def test_unrelated_erasure_preserves_derivatives_during_pending_refresh(stores, monkeypatch):
    from colonymind.work.store import WorkStore
    revision = _derived(stores)
    with closing(stores.ledger._connect()) as conn, conn:
        facts.mark_dirty(conn, 'root')
    monkeypatch.setattr(WorkStore, 'schedule', lambda *args: None)
    deleted = []
    delete = WorkStore._delete_root

    def track_delete(conn, ident):
        deleted.append(ident)
        delete(conn, ident)
    monkeypatch.setattr(WorkStore, '_delete_root', staticmethod(track_delete))
    stores.ledger.record_source('unrelated', contact_id='owner', session_id='other-session', messages=[{'role': 'user', 'content': 'An independent request'}], derive_claims=False)
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(stores.ledger)
    curated.forget_sources(stores.ledger, 'owner', source_ids=['unrelated'])
    assert historical_bytes(stores.ledger) == _native_before_withdrawal
    assert deleted == []
    _assert_derived(stores, revision)

def test_historical_event_erasure_cannot_clear_new_derivatives(stores, monkeypatch):
    from colonymind.work.store import WorkStore
    monkeypatch.setattr(WorkStore, 'schedule', lambda *args: None)
    admit(stores)
    stores.ledger.record_source('late-source', contact_id='owner', session_id='session', messages=[{'role': 'tool', 'content': 'Later operation evidence'}], derive_claims=False)
    stores.work.admit_event({'event_id': 'late', 'root_turn_id': 'root', 'session_id': 'session', 'work': envelope(operations=[operation('late-operation', success=False)]), 'source_refs': stores.ledger.source_references(['late-source'], contact_id='owner', session_id='session')}, contact_id='owner', principal_id='host')
    stores.work.reconcile()
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(stores.ledger)
    curated.forget_sources(stores.ledger, 'owner', source_ids=['late-source'])
    assert historical_bytes(stores.ledger) == _native_before_withdrawal
    revision = _derived(stores)
    stores.ledger.record_source('unrelated', contact_id='owner', session_id='other-session', messages=[{'role': 'user', 'content': 'A later independent request'}], derive_claims=False)
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(stores.ledger)
    curated.forget_sources(stores.ledger, 'owner', source_ids=['unrelated'])
    assert historical_bytes(stores.ledger) == _native_before_withdrawal
    _assert_derived(stores, revision)

def test_stale_worker_cannot_recreate_an_erased_projection(stores, monkeypatch):
    from colonymind.work.store import WorkStore
    _derived(stores)
    newer = WorkStore(stores.ledger)
    monkeypatch.setattr(WorkStore, 'schedule', lambda *args: None)
    connect = stores.work._connect
    captured, release = (threading.Event(), threading.Event())
    errors = []

    def paused_connect():
        captured.set()
        assert release.wait(2)
        return connect()

    def stale_rebuild():
        try:
            stores.work.rebuild_root('root')
        except Exception as error:
            errors.append(error)
    monkeypatch.setattr(stores.work, '_connect', paused_connect)
    worker = threading.Thread(target=stale_rebuild)
    worker.start()
    try:
        assert captured.wait(1)
        from test_curated_memory import historical_bytes
        _native_before_withdrawal = historical_bytes(stores.ledger)
        curated.forget_sources(stores.ledger, 'owner', source_ids=['root'])
        assert historical_bytes(stores.ledger) == _native_before_withdrawal
        newer.rebuild_root('root')
        with closing(stores.ledger._connect()) as conn:
            assert conn.execute('SELECT pending FROM work_dirty WHERE root_turn_id=?', ('root',)).fetchone()[0] == 0
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive() and (not errors)
    with closing(newer._connect()) as conn:
        assert conn.execute('SELECT count(*) FROM work_roots').fetchone()[0] == 0
        assert conn.execute('SELECT count(*) FROM work_effects').fetchone()[0] == 0
