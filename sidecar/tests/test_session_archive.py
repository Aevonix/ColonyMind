"""Archive overlay and host boundary, using an engine seam with no Hermes access."""
from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import sys
from types import ModuleType

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from colonymind.api.auth import anonymous_authority, key_authority
from colonymind.api.credentials import CredentialTable, secret_digest
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import host
from colonymind.session_archive.store import ArchiveError, ArchiveStore


SOURCE = {'platform': 'whatsapp', 'chat_id': 'chat-side', 'thread_id': None}
MAIN = {'platform': 'whatsapp', 'chat_id': 'chat-main', 'thread_id': None}


def body(operation='op', archived=True, reparent=True, **overrides):
    return {'archived': archived, 'reparent_to_main': reparent, 'operation_id': operation,
            'main_session_id': 'session-main', 'source_origin': dict(SOURCE), 'main_origin': dict(MAIN),
            **overrides}


class Engine:
    def __init__(self):
        self.calls = []
        self.moved = False
        self.fail_apply = False
        self.fail_commit = False
        self.fail_restore = False
        self.crash = False
        self.no_work = False
        self.local = None

    def __call__(self, *, phase, session_id, request, intent, connection, record=None):
        self.calls.append(phase)
        reparent_to_main = request.get('reparent_to_main')
        if phase == 'validate':
            if session_id in {'unknown', 'main', 'guest'}:
                raise ArchiveError('not_found')
        elif phase == 'plan':
            return {'jobs': [] if self.no_work or not reparent_to_main else [
                {'id': 'job-1', 'original_origin': {'session': session_id},
                 'intended_origin': {'session': 'main'}}]}
        elif phase == 'apply':
            self.moved = bool(intent['jobs'])
            for job in intent['jobs']:
                job['attempted'] = True
            if record is not None:
                record(intent)
            if self.crash:
                raise SystemExit('process interrupted after cron update')
            if self.fail_apply:
                raise ArchiveError('uncertain')
            return {'tasks_reparented': bool(reparent_to_main or self.no_work),
                    'reparented': {'cron_jobs': len(intent['jobs'])}}
        elif phase == 'commit':
            if self.local:
                connection.execute('ATTACH DATABASE ? AS local', (str(self.local),))
                connection.execute("UPDATE local.parents SET parent='main'")
            if self.fail_commit:
                raise RuntimeError('local verification failed')
        elif phase == 'rollback':
            if self.fail_restore:
                return {'restored': {'cron_jobs': 0}, 'half_applied': True, 'jobs_left_moved': ['job-1']}
            count = int(self.moved)
            self.moved = False
            return {'restored': {'cron_jobs': count}, 'half_applied': False, 'jobs_left_moved': []}


@pytest.fixture
def service(tmp_path, monkeypatch):
    store, engine = ArchiveStore(tmp_path / 'private' / 'nested' / 'archive.db'), Engine()
    monkeypatch.setattr(host, '_session_archive_store', store)
    monkeypatch.setattr(host, '_session_archive_engine', lambda: engine)
    return store, engine


def application(authority=key_authority(), *, credentials=None):
    app = FastAPI()
    if credentials is not None:
        app.add_middleware(ApiKeyMiddleware, api_key='instance-key', credentials=credentials)
    elif authority is not None:
        @app.middleware('http')
        async def authorize(request, call_next):
            request.state.colonymind_authority = authority
            return await call_next(request)
    app.include_router(host.router)
    return app


def client(authority=key_authority(), *, credentials=None):
    return AsyncClient(transport=ASGITransport(app=application(authority, credentials=credentials)),
                       base_url='http://localhost')


@pytest.mark.asyncio
async def test_archive_receipt_replay_unarchive_and_retention(service):
    store, engine = service
    async with client() as api:
        cold = await api.get('/v1/host/sessions/side/archive')
        assert cold.status_code == 200
        assert cold.json() == {'destination_id': 'side', 'archived': False,
                               'revision': cold.json()['revision'], 'archived_at': None,
                               'receipt': None, 'half_applied': False, 'jobs_left_moved': []}
        first = await api.post('/v1/host/sessions/side/archive', json=body())
        assert first.status_code == 200, first.text
        receipt = first.json()
        assert set(receipt) == {'destination_id', 'archived', 'tasks_reparented', 'receipt_id', 'revision',
                                'archived_at', 'reparented', 'operation_id'}
        assert receipt['tasks_reparented'] is True and len(receipt['revision']) == 64
        before = list(engine.calls)
        assert (await api.post('/v1/host/sessions/side/archive', json=body())).json() == receipt
        assert engine.calls == before
        for session, changed in [('side', body(archived=False)), ('other', body())]:
            conflict = await api.post(f'/v1/host/sessions/{session}/archive', json=changed)
            assert conflict.status_code == 409 and conflict.json()['error'] == 'idempotency_conflict'
        state = (await api.get('/v1/host/sessions/side/archive')).json()
        assert state['receipt'] == receipt and state['revision'] == receipt['revision']
        inventory = (await api.get('/v1/host/sessions/archived')).json()
        assert inventory['items'] == [{key: receipt[key] for key in
            ('destination_id', 'revision', 'archived_at', 'receipt_id', 'tasks_reparented')}]
        restored = await api.post('/v1/host/sessions/side/archive', json=body('restore', False))
        assert restored.status_code == 200 and restored.json()['archived'] is False
        assert restored.json()['revision'] != receipt['revision']
        assert (await api.get('/v1/host/sessions/archived')).json()['items'] == []
        assert (await api.get('/v1/host/sessions/side/archive')).json()['receipt'] == restored.json()
        # An old completed operation stays a replay even after a newer state.
        assert (await api.post('/v1/host/sessions/side/archive', json=body())).json() == receipt
        assert (await api.get('/v1/host/sessions/side/archive')).json()['archived'] is False
        for response in (cold, first, restored):
            assert response.headers['cache-control'] == 'no-store'
    with closing(sqlite3.connect(store.path)) as db:
        assert db.execute('SELECT count(*) FROM archive_operations').fetchone()[0] == 2
        assert db.execute('SELECT count(*) FROM archive_sessions').fetchone()[0] == 1
    assert store.path.stat().st_mode & 0o777 == 0o600
    assert store.path.with_name('archive.db.lock').stat().st_mode & 0o777 == 0o600
    assert all(p.stat().st_mode & 0o777 == 0o700 for p in [store.path.parent, store.path.parent.parent])


@pytest.mark.asyncio
@pytest.mark.parametrize('bad', [None, [], {}, {'archived': True}, body(reparent=0),
    body(archived=1), body(archived='true'), body(operation=''), body(operation='x' * 129),
    body(operation=42), {**body(), 'extra': True},
    {key: value for key, value in body().items() if key != 'source_origin'},
    body(main_session_id='side'), body(main_session_id=''), body(main_session_id=42),
    body(source_origin=None), body(source_origin={}), body(source_origin='whatsapp:chat'),
    body(source_origin={'platform': 'whatsapp'}), body(main_origin={'chat_id': 'c'}),
    body(source_origin={'platform': '', 'chat_id': 'c'}),
    body(main_origin={'platform': 'whatsapp', 'chat_id': 'c', 'thread_id': 7}),
    body(main_origin={'platform': 'whatsapp', 'chat_id': 'c', 'extra': 'x'}),
    body(main_origin=dict(SOURCE)),
    body(main_origin={'platform': 'WhatsApp', 'chat_id': 'chat-side', 'thread_id': None}),
    # Structurally unusable addresses: accepting " " wrote that exact origin onto a
    # live job and answered 200 with tasks_reparented true.
    body(main_origin={'platform': ' ', 'chat_id': ' ', 'thread_id': None}),
    body(main_origin={'platform': 'whatsapp', 'chat_id': '   ', 'thread_id': None}),
    body(source_origin={'platform': '\t', 'chat_id': 'c', 'thread_id': None}),
    body(main_origin={'platform': 'whatsapp', 'chat_id': 'c', 'thread_id': ' '})])
async def test_strict_request_validation(service, bad):
    async with client() as api:
        result = await api.post('/v1/host/sessions/side/archive', content=json.dumps(bad))
        assert result.status_code == 400 and result.json() == {'error': 'invalid_request'}
        assert result.headers['cache-control'] == 'no-store'
    assert service[1].calls == [] and not service[0].path.exists()


@pytest.mark.asyncio
async def test_malformed_json(service):
    async with client() as api:
        result = await api.post('/v1/host/sessions/side/archive', content='{')
        assert result.status_code == 400 and result.json()['error'] == 'invalid_request'


@pytest.mark.asyncio
@pytest.mark.parametrize('session', ['unknown', 'main', 'guest'])
async def test_unknown_or_non_side_refused(service, session):
    async with client() as api:
        for response in [await api.get(f'/v1/host/sessions/{session}/archive'),
                         await api.post(f'/v1/host/sessions/{session}/archive', json=body())]:
            assert response.status_code == 404 and response.json()['error'] == 'not_found'
    assert 'plan' not in service[1].calls


@pytest.mark.asyncio
@pytest.mark.parametrize('authority', [None, anonymous_authority(),
    replace(key_authority(), owner_platforms=frozenset()),
    replace(key_authority(), access=frozenset({'read'})),
    replace(key_authority(), access=frozenset({'write'})),
    replace(key_authority(), access=frozenset({'hands'})),
    replace(key_authority(), access=frozenset({'tool-origin'})),
    replace(key_authority(), access=frozenset({'answer'})),
    replace(key_authority(), scopes=frozenset({'api:access'}))])
async def test_authority_is_checked_in_handlers(service, authority):
    async with client(authority) as api:
        for result in [await api.get('/v1/host/sessions/archived'),
                       await api.get('/v1/host/sessions/side/archive'),
                       await api.post('/v1/host/sessions/side/archive', json=body())]:
            assert result.status_code == 403 and result.json() == {'error': 'unauthorized'}
            assert result.headers['cache-control'] == 'no-store'
    assert service[1].calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize('scope', ['sessions:read', 'sessions:write', '*'])
async def test_real_bearer_middleware_scope_alternative(service, scope):
    table = CredentialTable.from_config({'carried': {'sha256': secret_digest('scoped-key'),
        'owner': ['companion'], 'contacts': 'all', 'scopes': [scope], 'unscoped_api': False}})
    headers = {'Authorization': 'Bearer scoped-key'}
    async with client(credentials=table) as api:
        for path in ['/v1/host/sessions/archived', '/v1/host/sessions/side/archive']:
            result = await api.get(path, headers=headers)
            assert result.status_code == (200 if scope in {'sessions:read', '*'} else 403), result.text
        result = await api.post('/v1/host/sessions/side/archive', json=body(), headers=headers)
        assert result.status_code == (200 if scope in {'sessions:write', '*'} else 403), result.text
        result = await api.get('/v1/host/sessions/archived')
        assert result.status_code == 401 and result.json() == {'error': 'unauthorized'}
        assert result.headers['cache-control'] == 'no-store'


@pytest.mark.asyncio
@pytest.mark.parametrize('query', ['limit=0', 'limit=201', 'limit=1.5', 'limit=no', 'cursor=',
                                  'limit=' + '9' * 5000, 'cursor=garbage', 'cursor=' + 'x' * 4097])
async def test_invalid_inventory(service, query):
    async with client() as api:
        result = await api.get('/v1/host/sessions/archived?' + query)
        assert result.status_code == 400 and result.json()['error'] == 'invalid_request'


def test_inventory_pages_ties_staleness_and_limit(service, monkeypatch):
    store, engine = service
    monkeypatch.setattr(store, '_stamp', lambda: '2026-10-18T18:40:00+00:00')
    for name in ['z', 'a', 'b']:
        store.archive(name, body(name), engine)
    first = store.inventory(1)
    assert first['counts'] == {'archived': 3} and first['items'][0]['destination_id'] == 'a'
    reopened = ArchiveStore(store.path)
    second = reopened.inventory(1, first['next_cursor'])
    assert second['items'][0]['destination_id'] == 'b'
    assert second['snapshot_at'] == first['snapshot_at'] and second['revision'] == first['revision']
    third = reopened.inventory(1, second['next_cursor'])
    assert third['items'][0]['destination_id'] == 'z' and third['next_cursor'] is None
    assert third['has_more'] is False
    with pytest.raises(ArchiveError, match='invalid_request'):
        store.inventory(2, first['next_cursor'])
    with pytest.raises(ArchiveError, match='invalid_request'):
        store.inventory(1, first['next_cursor'] + 'x')
    store.archive('a', body('restore-a', False), engine)
    with pytest.raises(ArchiveError, match='stale_revision'):
        store.inventory(1, first['next_cursor'])


@pytest.mark.parametrize('no_work', [True, False])
def test_reparent_false_is_accepted_without_claiming_movement(service, no_work):
    store, engine = service
    engine.no_work = no_work
    result = store.archive('side', body(reparent=False), engine)
    assert result['tasks_reparented'] is no_work
    assert result['reparented'] == {'cron_jobs': 0}


@pytest.mark.parametrize('failure', ['fail_apply', 'fail_commit'])
def test_refusal_restores_jobs_and_rolls_back_local_transaction(service, failure, tmp_path):
    store, engine = service
    engine.local = tmp_path / 'local.db'
    with closing(sqlite3.connect(engine.local)) as db, db:
        db.execute('CREATE TABLE parents (parent TEXT)')
        db.execute("INSERT INTO parents VALUES ('side')")
    setattr(engine, failure, True)
    with pytest.raises(ArchiveError, match='uncertain') as refused:
        store.archive('side', body(), engine)
    assert refused.value.details == {'destination_id': 'side', 'archived': False,
        'tasks_reparented': False, 'restored': {'cron_jobs': 1}, 'half_applied': False, 'jobs_left_moved': []}
    assert engine.moved is False and store.receipt('side', engine)['receipt'] is None
    with closing(sqlite3.connect(engine.local)) as db:
        assert db.execute('SELECT parent FROM parents').fetchone()[0] == 'side'
    with closing(sqlite3.connect(store.path)) as db:
        assert db.execute('SELECT state FROM archive_operations').fetchone()[0] == 'refused'
    # Failed operations retain their refusal rather than silently being rerun.
    calls = list(engine.calls)
    with pytest.raises(ArchiveError, match='uncertain'):
        store.archive('side', body(), engine)
    assert engine.calls == calls


@pytest.mark.asyncio
async def test_half_applied_visible_and_no_second_intent_until_recovery(service):
    store, engine = service
    engine.fail_apply = engine.fail_restore = True
    async with client() as api:
        result = await api.post('/v1/host/sessions/side/archive', json=body())
        assert result.status_code == 409 and result.json()['half_applied'] is True
        assert result.json()['jobs_left_moved'] == ['job-1']
        state = (await api.get('/v1/host/sessions/side/archive')).json()
        assert state['half_applied'] is True and state['jobs_left_moved'] == ['job-1']
        second = await api.post('/v1/host/sessions/side/archive', json=body('second'))
        assert second.json() == result.json()
        assert engine.calls.count('plan') == 1
        assert 'session_archive' in host.supported_capabilities()
        engine.fail_apply = engine.fail_restore = False
        assert (await api.post('/v1/host/sessions/side/archive', json=body('second'))).status_code == 200
        assert (await api.get('/v1/host/sessions/side/archive')).json()['half_applied'] is False


def test_restart_rolls_back_durable_intent_before_new_attempt(service):
    store, engine = service
    engine.crash = True
    with pytest.raises(SystemExit):
        store.archive('side', body(), engine)
    with closing(sqlite3.connect(store.path)) as db:
        saved = json.loads(db.execute('SELECT intent_json FROM archive_operations').fetchone()[0])
        assert saved['jobs'][0]['original_origin'] == {'session': 'side'}
        assert saved['jobs'][0]['intended_origin'] == {'session': 'main'}
    engine.crash = False
    engine.calls.clear()
    receipt = ArchiveStore(store.path).archive('side', body('retry'), engine)
    assert receipt['tasks_reparented'] is True
    assert engine.calls == ['rollback', 'validate', 'plan', 'apply', 'commit']


@pytest.mark.parametrize('journal', ['WAL', 'DELETE'])
def test_an_attached_store_cannot_produce_a_false_atomic_success(service, tmp_path, journal):
    # Reparenting moves recurring delivery only, so the receipt is a single-store
    # promise. An engine that attaches anything, in ANY journal mode, is refused
    # rather than handed a success receipt for a weaker transaction.
    store, engine = service
    engine.local = tmp_path / 'local-attached.db'
    with closing(sqlite3.connect(engine.local)) as db, db:
        db.execute(f'PRAGMA journal_mode={journal}')
        db.execute('CREATE TABLE parents (parent TEXT)')
        db.execute("INSERT INTO parents VALUES ('side')")
    with pytest.raises(ArchiveError, match='uncertain'):
        store.archive('side', body(), engine)
    assert engine.moved is False
    assert store.receipt('side', engine)['receipt'] is None
    with closing(sqlite3.connect(engine.local)) as db:
        assert db.execute('SELECT parent FROM parents').fetchone()[0] == 'side'


def test_overlapping_writer_is_busy(service):
    store, engine = service
    with store._writer():
        with pytest.raises(ArchiveError, match='busy'):
            ArchiveStore(store.path).archive('side', body(), engine)
    assert not engine.calls


@pytest.mark.asyncio
@pytest.mark.parametrize('grant', [
    {'owner': ['companion'], 'contacts': 'all', 'access': ['manage']},
    {'owner': [], 'contacts': 'all', 'access': ['manage']},
    {'owner': ['companion'], 'contacts': 'all', 'access': ['read']},
    {'owner': ['companion'], 'access': ['answer']},
    {'access': ['hands']}, {'access': ['tool-origin']},
    {'owner': [], 'scopes': ['*']},
])
async def test_real_credentials_need_both_owner_and_route_grant(service, grant):
    table = CredentialTable.from_config({'caller': {'sha256': secret_digest('caller-key'), **grant}})
    expected = 200 if grant.get('access') == ['manage'] and grant.get('owner') else 403
    async with client(credentials=table) as api:
        headers = {'Authorization': 'Bearer caller-key'}
        for result in [await api.get('/v1/host/sessions/archived', headers=headers),
                       await api.get('/v1/host/sessions/side/archive', headers=headers),
                       await api.post('/v1/host/sessions/side/archive', json=body(), headers=headers)]:
            assert result.status_code == expected, result.text
            assert result.headers['cache-control'] == 'no-store'
            if expected == 403:
                assert result.json() == {'error': 'unauthorized'}


@pytest.mark.asyncio
async def test_absent_wiring_and_engine_refuse_without_writes(service, monkeypatch):
    store, engine = service
    for missing in ['store', 'engine']:
        monkeypatch.setattr(host, '_session_archive_store', None if missing == 'store' else store)
        monkeypatch.setattr(host, '_session_archive_engine', lambda: None if missing == 'engine' else engine)
        assert 'session_archive' not in host.supported_capabilities()
        async with client() as api:
            for result in [await api.get('/v1/host/sessions/archived'),
                           await api.get('/v1/host/sessions/side/archive'),
                           await api.post('/v1/host/sessions/side/archive', json=body())]:
                assert result.status_code == 409 and result.json()['error'] == 'unsupported_capability'
    assert not store.path.exists()


def test_injected_seam_never_imports_cron(monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name != 'cron' and not name.startswith('cron.')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    monkeypatch.setattr(host, '_session_archive_store', object())
    monkeypatch.setattr(host, '_archive_engine', None)
    assert host._session_archive_engine() is None
    assert 'session_archive' not in host.supported_capabilities()
    engine = Engine()
    host.set_session_archive_engine(engine)
    assert host._session_archive_engine() is engine
    assert 'session_archive' in host.supported_capabilities()
    host.set_session_archive_engine(None)
    assert 'session_archive' not in host.supported_capabilities()


@pytest.mark.asyncio
async def test_source_failure_is_private_and_contract_status(service, monkeypatch):
    def broken(*args):
        raise OSError('private host path and diagnostics')
    monkeypatch.setattr(service[0], 'inventory', broken)
    async with client() as api:
        result = await api.get('/v1/host/sessions/archived')
        assert result.status_code == 503 and result.json() == {'error': 'source_unavailable'}
