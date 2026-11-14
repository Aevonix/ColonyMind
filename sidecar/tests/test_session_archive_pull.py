"""The Hermes transport boundary: authority, timeout, replay and no local jobs."""
import asyncio
import builtins
from contextlib import closing
from dataclasses import replace
from datetime import timedelta
import sqlite3
import time
from types import SimpleNamespace

from httpx import ASGITransport, AsyncClient
import pytest

from colonymind.api.archive_queue import ArchiveQueueClient
from colonymind.api.auth import anonymous_authority, key_authority
from colonymind.api.legacy_scopes import required_scope
from colonymind.api.routers import host
from colonymind.session_archive.store import ArchiveStore, _digest
from colonymind.util import temporal
from test_session_archive import Engine, application, body, client

PLANS = '/v1/host/sessions/archive/plans'
ARCHIVE = '/v1/host/sessions/side/archive'


@pytest.fixture
def queue(tmp_path, monkeypatch):
    store = ArchiveStore(tmp_path / 'archive.db', clock=lambda: temporal.now_utc().timestamp())
    engine = ArchiveQueueClient(store, timeout=.05)
    monkeypatch.setattr(host, '_session_archive_store', store)
    monkeypatch.setattr(host, '_archive_engine', engine)
    return store


def receipt(plan):
    return {'destination_id': plan['destination_id'], 'archived': plan['request']['archived'],
            'tasks_reparented': True, 'receipt_id': 'receipt-1',
            'revision': _digest([plan['destination_id'], 'receipt-1']),
            'archived_at': plan['created_at'] if plan['request']['archived'] else None,
            'reparented': {'cron_jobs': 1}, 'operation_id': plan['operation_id']}


@pytest.mark.asyncio
async def test_production_queue_without_any_cron_import_timeout_then_replay(queue, monkeypatch):
    original = builtins.__import__
    def no_cron(name, *args, **kwargs):
        assert name != 'cron' and not name.startswith('cron.')
        assert name != 'colonymind.session_archive.reparent'
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_cron)
    async with client() as api:
        assert 'session_archive' not in host.supported_capabilities()
        assert (await api.post(ARCHIVE, json=body())).status_code == 409
        assert not queue.path.exists()
        assert (await api.get(PLANS + '?wait=0')).json() == {'plan': None}
        assert 'session_archive' in host.supported_capabilities()
        pending = await api.post(ARCHIVE, json=body())
        assert pending.status_code == 503
        assert pending.json() == {'error': 'source_unavailable', 'pending': True, 'operation_id': 'op'}
        assert queue.inventory()['items'] == []
        plan = (await api.get(PLANS + '?wait=0')).json()['plan']
        assert plan['authority']['speaks_for_owner'] is True
        assert plan['authority']['sessions:write'] is True
        result = receipt(plan)
        path = PLANS + '/op/receipt'
        payload = {'claim': plan['claim'], 'receipt': result}
        assert (await api.post(path, json=payload)).status_code == 200
        assert (await api.post(path, json=payload)).status_code == 200
        assert (await api.post(path, json={**payload, 'receipt': {**result, 'reparented': {'cron_jobs': 2}}})).status_code == 409
        assert (await api.post(ARCHIVE, json=body())).json() == result
        assert (await api.post(ARCHIVE, json=body(archived=False))).status_code == 409
        assert (await api.get(ARCHIVE)).json()['receipt'] == result
        assert len(queue.inventory()['items']) == 1


@pytest.mark.asyncio
async def test_heartbeat_expires_and_old_adapter_never_enables_it(queue, monkeypatch):
    async with client() as api:
        await api.get(PLANS + '?wait=0')
        stamp = temporal.now_utc()
        monkeypatch.setattr(temporal, 'now_utc', lambda: stamp + timedelta(seconds=61))
        assert 'session_archive' not in host.supported_capabilities()
        response = await api.post(ARCHIVE, json=body())
        assert response.status_code == 409 and response.json()['error'] == 'unsupported_capability'
        with queue._connect() as db:
            assert db.execute('SELECT count(*) FROM archive_plans').fetchone()[0] == 0


@pytest.mark.parametrize('authority', [anonymous_authority(), replace(key_authority(), owner_platforms=frozenset()),
    replace(key_authority(), scopes=frozenset({'sessions:write'})),
    replace(key_authority(), access=frozenset({'read'}))])
@pytest.mark.asyncio
async def test_worker_requires_its_own_scope_and_owner(queue, authority):
    async with client(authority) as api:
        response = await api.get(PLANS + '?wait=0')
        assert response.status_code == 403
    assert not queue.path.exists()
    assert required_scope('GET', PLANS) == 'sessions:archive-worker'
    assert required_scope('POST', PLANS + '/op/receipt') == 'sessions:archive-worker'


@pytest.mark.asyncio
async def test_claim_is_bound_to_credential_and_refusal_never_archives(queue):
    async with client() as api:
        await api.get(PLANS + '?wait=0')
        await api.post(ARCHIVE, json=body())
        plan = (await api.get(PLANS + '?wait=0')).json()['plan']
        payload = {'claim': plan['claim'], 'receipt': {'error': 'uncertain', 'refused': ['job-stale'],
                    'reason': 'cron_update_refused', 'half_applied': False, 'jobs_left_moved': []}}
        async with client(replace(key_authority(), credential_id='another-worker')) as other:
            assert (await other.post(PLANS + '/op/receipt', json=payload)).status_code == 403
        assert (await api.post(PLANS + '/op/receipt', json={**payload, 'claim': 'wrong'})).status_code == 403
        assert (await api.post(PLANS + '/op/receipt', json=payload)).status_code == 200
        result = await api.post(ARCHIVE, json=body())
        assert result.status_code == 409 and result.json() == payload['receipt']
        assert queue.inventory()['items'] == []


@pytest.mark.asyncio
async def test_async_wait_does_not_block_the_poll_or_receipt(queue, monkeypatch):
    monkeypatch.setattr(host._archive_engine, 'timeout', 2)
    async with client() as api:
        await api.get(PLANS + '?wait=0')
        pending = asyncio.create_task(api.post(ARCHIVE, json=body()))
        for _ in range(100):
            with closing(sqlite3.connect(queue.path.as_uri() + '?mode=ro', uri=True)) as db:
                if db.execute('SELECT 1 FROM archive_plans').fetchone():
                    break
            await asyncio.sleep(.01)
        polled = await api.get(PLANS + '?wait=2')
        plan = polled.json()['plan']
        assert plan is not None
        result = receipt(plan)
        await api.post(PLANS + '/op/receipt', json={'claim': plan['claim'], 'receipt': result})
        assert (await pending).json() == result


@pytest.mark.asyncio
async def test_pre_pull_receipt_replays_without_claiming_or_reparenting(queue):
    original = queue.archive('side', body(), Engine())
    async with client() as api:
        await api.get(PLANS + '?wait=0')
        assert (await api.post(ARCHIVE, json=body())).json() == original
        assert (await api.get(PLANS + '?wait=0')).json() == {'plan': None}


@pytest.mark.asyncio
async def test_reported_half_applied_jobs_remain_visible_until_recovery(queue):
    async with client() as api:
        await api.get(PLANS + '?wait=0')
        await api.post(ARCHIVE, json=body())
        plan = (await api.get(PLANS + '?wait=0')).json()['plan']
        await api.post(PLANS + '/op/receipt', json={'claim': plan['claim'], 'receipt': {
            'error': 'uncertain', 'half_applied': True, 'jobs_left_moved': ['job-stuck'],
            'tasks_reparented': False, 'restored': {'cron_jobs': 0}}})
        state = (await api.get(ARCHIVE)).json()
        assert state['half_applied'] and state['jobs_left_moved'] == ['job-stuck']
        await api.post(ARCHIVE, json=body(operation='recover'))
        state = (await api.get(ARCHIVE)).json()
        assert state['pending'] and state['half_applied'] and state['jobs_left_moved'] == ['job-stuck']


@pytest.mark.asyncio
async def test_worker_death_during_held_poll_stops_claims_and_expires_at_exactly_60_seconds(queue, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(queue, '_clock', lambda: now[0])
    loop = asyncio.get_running_loop()
    inspected, dead = asyncio.Event(), asyncio.Event()
    original = queue.claim_plan
    claims = []
    def claim(*args):
        claims.append(now[0])
        result = original(*args)
        loop.call_soon_threadsafe(inspected.set)
        return result
    monkeypatch.setattr(queue, 'claim_plan', claim)
    async def disconnected():
        return dead.is_set()
    request = SimpleNamespace(query_params={'wait': '25'}, is_disconnected=disconnected,
                              state=SimpleNamespace(colonymind_authority=key_authority()))
    poll = asyncio.create_task(host.archive_plans(request))
    await asyncio.wait_for(inspected.wait(), 2)
    dead.set()  # A worker vanishes after the incoming request, while the handler lives.
    queue.enqueue('side', body(), {'owner_id': 'owner'})
    response = await asyncio.wait_for(poll, 2)
    assert response.status_code == 200 and claims == [1000.0]
    with queue._connect() as db:
        assert db.execute('SELECT claim FROM archive_plans').fetchone()[0] is None
        assert float(db.execute("SELECT value FROM archive_meta WHERE key='worker_heartbeat'").fetchone()[0]) == 1000
    now[0] = 1059.999
    assert queue.worker_available()
    now[0] = 1060.0
    assert not queue.worker_available()


@pytest.mark.asyncio
async def test_held_empty_poll_writes_one_heartbeat_and_only_reads_the_queue(queue, monkeypatch):
    from contextlib import contextmanager
    writes = []
    original = queue._writer
    @contextmanager
    def writer():
        writes.append(True)
        with original() as db:
            yield db
    monkeypatch.setattr(queue, '_writer', writer)
    async with client() as api:
        assert (await api.get(PLANS + '?wait=1')).json() == {'plan': None}
    assert len(writes) == 1


@pytest.mark.asyncio
async def test_contended_availability_fails_closed_without_delaying_an_unrelated_asgi_request(queue, monkeypatch):
    queue.heartbeat()
    main_thread = __import__('threading').get_ident()
    original = queue.worker_available
    checked = []
    def available():
        checked.append(__import__('threading').get_ident())
        return original()
    monkeypatch.setattr(queue, 'worker_available', available)
    app = application()
    @app.get('/unrelated')
    async def unrelated():
        return {'ok': True}
    with closing(sqlite3.connect(queue.path, timeout=0)) as lock:
        lock.execute('BEGIN EXCLUSIVE')
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://localhost') as api:
            start = time.monotonic()
            archive, other = await asyncio.gather(api.get(ARCHIVE), api.get('/unrelated'))
            assert time.monotonic() - start < .5
            assert other.status_code == 200 and other.json() == {'ok': True}
            assert archive.status_code == 409 and archive.json()['error'] == 'unsupported_capability'
    assert checked and all(thread != main_thread for thread in checked)
