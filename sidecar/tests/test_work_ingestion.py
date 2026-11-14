"""Both turn APIs freeze authenticated work beside their canonical owner source."""
from dataclasses import replace
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from colonymind.api.auth import key_authority
from colonymind.api.routers import host
from colonymind.work.store import WorkStore
from test_work_projection import envelope, operation


@pytest.fixture
def state(tmp_path, monkeypatch):
    from colonymind.turns import get_turn_idempotency_ledger
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(tmp_path))
    monkeypatch.setenv('COLONYMIND_URL', '')
    monkeypatch.setenv('COLONYMIND_OWNER_PERSON_ID', 'owner')
    monkeypatch.setattr(host, '_contacts_store', None)
    monkeypatch.setattr(host, '_comms_log', None)
    monkeypatch.setattr(host, '_telemetry', None)
    monkeypatch.setattr(host, '_attribute_turn', AsyncMock(return_value='owner'))
    monkeypatch.setattr(host, '_ensure_channel_id', AsyncMock(return_value='test:room'))
    ledger = get_turn_idempotency_ledger(tmp_path)
    yield SimpleNamespace(path=tmp_path, ledger=ledger)
    from colonymind.work import _store
    cached = _store(str(tmp_path))
    cached.wait_pending(timeout=2)


def app(authority=None):
    result = FastAPI()
    @result.middleware('http')
    async def auth(request, call_next):
        request.state.colonymind_authority = authority or key_authority()
        return await call_next(request)
    result.include_router(host.router)
    result.include_router(host.v2_router)
    return result


def payload(ident='root'):
    return {'identity': {'host_id': 'generic-host'},
        'context': {'session_id': 'session', 'contact_id': 'owner', 'channel_id': 'test:room', 'turn_id': ident},
        'sender': {'platform': 'test', 'user_id': 'owner-handle'},
        'user_message': {'role': 'user', 'content': 'Please update the report'},
        'assistant_message': {'role': 'assistant', 'content': 'A reply is not proof'},
        'summary': 'A bounded source summary', 'work': envelope(ident)}


@pytest.mark.asyncio
@pytest.mark.parametrize('version', [1, 2])
async def test_both_routes_work_retry_conflict_and_crash_repair(state, monkeypatch, version):
    import colonymind.work
    monkeypatch.setattr(colonymind.work, 'schedule_work', lambda *args: None)
    body = payload()
    async with AsyncClient(transport=ASGITransport(app=app()), base_url='http://local') as client:
        async def send(value):
            return await (client.post('/v1/host/turns/sync', json=value) if version == 1
                else client.put('/v2/host/turns/root', json=value))
        first = await send(body)
        assert first.status_code in {200, 201}
        assert first.json()['work_recorded'] is True
        assert not (state.path / 'colonymind-work.db').exists()
        retry = await send(body)
        assert retry.headers['Idempotency-Status'] == 'replayed'
        assert retry.json() == first.json()
        body['work']['operations'][0]['success'] = False
        conflict = await send(body)
        assert conflict.status_code == 409
    store = WorkStore(state.ledger)
    store.reconcile()
    assert store.get_root('root', contact_id='owner')['status'] == 'proven'


@pytest.mark.asyncio
@pytest.mark.parametrize(('lane', 'sender', 'expected'), [
    ('test:room', True, True), ('cron:job', True, False), ('kanban:job', True, False),
    ('background_review:job', True, False), ('test:room', False, False), ('cli:local', False, True),
])
async def test_owner_provenance_and_automated_exclusions(state, monkeypatch, lane, sender, expected):
    body = payload()
    body['context']['channel_id'] = lane
    monkeypatch.setattr(host, '_ensure_channel_id', AsyncMock(return_value=lane))
    if not sender:
        body.pop('sender')
        monkeypatch.setattr(host, '_attribute_turn', AsyncMock(return_value=None))
    else:
        body['sender']['platform'] = lane.split(':')[0]
    async with AsyncClient(transport=ASGITransport(app=app()), base_url='http://local') as client:
        result = await client.post('/v1/host/turns/sync', json=body)
    assert result.status_code == 200
    assert result.json()['work_recorded'] is expected
    store = WorkStore(state.ledger)
    store.reconcile()
    assert bool(store.get_root('root', contact_id='owner')) is expected


@pytest.mark.asyncio
async def test_missing_owner_platform_cannot_capture_root(state):
    authority = replace(key_authority(), owner_platforms=frozenset({'other'}))
    async with AsyncClient(transport=ASGITransport(app=app(authority)), base_url='http://local') as client:
        response = await client.post('/v1/host/turns/sync', json=payload())
    assert response.json()['accepted'] is False
    assert not (state.path / 'colonymind-work.db').exists()


@pytest.mark.asyncio
async def test_faculty_off_still_admits_source_but_never_work(state):
    (state.path / 'colonymind.yaml').write_text('mind:\n  enabled: false\n  faculties:\n    work_records: false\n')
    async with AsyncClient(transport=ASGITransport(app=app()), base_url='http://local') as client:
        result = await client.post('/v1/host/turns/sync', json=payload())
    assert result.status_code == 200
    assert result.json()['source_recorded'] is True
    assert result.json()['work_recorded'] is False
    assert result.json()['work_skipped_reason'] == 'work_records_disabled'
    assert not (state.path / 'colonymind-work.db').exists()
    with closing(state.ledger._connect()) as conn:
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='work_envelopes'").fetchone()


@pytest.mark.asyncio
async def test_mind_lessons_and_model_off_generic_host_still_records_work(state):
    (state.path / 'colonymind.yaml').write_text('mind:\n  enabled: false\n  faculties:\n    lessons: false\n')
    body = payload()
    body['work']['operations'] = [operation(tool='terminal', action='execute', evidence={'kind': 'none'})]
    async with AsyncClient(transport=ASGITransport(app=app()), base_url='http://local') as client:
        result = await client.post('/v1/host/turns/sync', json=body)
    assert result.json()['work_recorded'] is True
    store = WorkStore(state.ledger)
    store.reconcile()
    assert store.get_root('root', contact_id='owner')['status'] == 'unproven'


@pytest.mark.asyncio
async def test_persisted_legacy_retry_stays_identical_after_work_addition(state):
    body = payload()
    body.pop('work')
    async with AsyncClient(transport=ASGITransport(app=app()), base_url='http://local') as client:
        first = await client.post('/v1/host/turns/sync', json=body)
        from colonymind.turns import TurnIdempotencyLedger
        TurnIdempotencyLedger(state.ledger.db_path)
        retry = await client.post('/v1/host/turns/sync', json=body)
    assert retry.headers['Idempotency-Status'] == 'replayed'
    assert first.json() == retry.json()
    assert first.json()['work_recorded'] is None


@pytest.mark.asyncio
@pytest.mark.parametrize('session', ['', 'session'])
@pytest.mark.parametrize('version', [1, 2])
async def test_pre_work_reservation_replays_with_explicit_or_derived_session(state, session, version):
    from colonymind.api.schemas.host import TurnSyncRequest
    from colonymind.turns import TurnIdempotencyLedger, canonical_turn_digest
    body = payload()
    body.pop('work')
    body['context']['session_id'] = session
    legacy = TurnSyncRequest.model_validate(body).model_dump(mode='json')
    legacy.pop('work')
    if not session:
        basis = canonical_turn_digest({'principal': key_authority().principal_id, 'turn': legacy})
        legacy['context']['session_id'] = host.DERIVED_SESSION_PREFIX + basis[:32]
    digest = canonical_turn_digest(legacy)
    response = {'accepted': True, 'source_recorded': True, 'continuity_updated': False}
    state.ledger.reserve('root', digest)
    state.ledger.complete('root', digest, response)
    TurnIdempotencyLedger(state.ledger.db_path)
    async with AsyncClient(transport=ASGITransport(app=app()), base_url='http://local') as client:
        result = await (client.post('/v1/host/turns/sync', json=body) if version == 1
            else client.put('/v2/host/turns/root', json=body))
    assert result.status_code == 200
    assert result.headers['Idempotency-Status'] == 'replayed'
    assert all(result.json()[key] == value for key, value in response.items())
    assert result.json()['work_recorded'] is None
    with closing(state.ledger._connect()) as conn:
        assert conn.execute('SELECT count(*) FROM turn_sources').fetchone()[0] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('session', ['', 'session'])
@pytest.mark.parametrize('version', [1, 2])
@pytest.mark.parametrize('explicit_nulls', [False, True])
async def test_pre_capture_digest_replays_with_absent_or_null_metadata(state, session, version, explicit_nulls):
    import hashlib
    import json
    from colonymind.api.schemas.host import TurnSyncRequest
    from colonymind.turns import TurnIdempotencyLedger, canonical_turn_digest

    def old_hash(value):
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':'),
                                        sort_keys=True).encode()).hexdigest()

    body = payload()
    body['context']['session_id'] = session
    fields = ('request_id', 'model', 'error_type', 'failure_kind', 'duration_ms')
    if explicit_nulls:
        body['work']['operations'][0].update(dict.fromkeys(fields))
    legacy = TurnSyncRequest.model_validate(body).model_dump(mode='json')
    for op in legacy['work']['operations']:
        for field in fields:
            op.pop(field)
    if not session:
        basis = old_hash({'principal': key_authority().principal_id, 'turn': legacy})
        legacy['context']['session_id'] = host.DERIVED_SESSION_PREFIX + basis[:32]
    for field in ('checkpoint_messages', 'assistant_source_refs', 'assistant_input_refs', 'source_only'):
        legacy.pop(field)
    digest = old_hash(legacy)
    assert canonical_turn_digest(legacy) == digest
    response = {'accepted': True, 'source_recorded': True, 'continuity_updated': False, 'work_recorded': True}
    state.ledger.reserve('root', digest)
    state.ledger.complete('root', digest, response)
    TurnIdempotencyLedger(state.ledger.db_path)
    async with AsyncClient(transport=ASGITransport(app=app()), base_url='http://local') as client:
        result = await (client.post('/v1/host/turns/sync', json=body) if version == 1
            else client.put('/v2/host/turns/root', json=body))
    assert result.status_code == 200
    assert result.headers['Idempotency-Status'] == 'replayed'
    assert all(result.json()[key] == value for key, value in response.items())
    with closing(state.ledger._connect()) as conn:
        assert conn.execute('SELECT count(*) FROM turn_sources').fetchone()[0] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('message', [None, {'role': 'user', 'content': '   '}])
async def test_assistant_only_capture_does_not_claim_an_owner_request(state, message):
    body = payload()
    body['user_message'] = message
    async with AsyncClient(transport=ASGITransport(app=app()), base_url='http://local') as client:
        result = await client.post('/v1/host/turns/sync', json=body)
    assert result.status_code == 200
    assert result.json()['source_recorded'] is True
    assert result.json()['work_recorded'] is False
    assert result.json()['work_skipped_reason'] == 'ineligible_origin'


@pytest.mark.asyncio
@pytest.mark.parametrize(('module', 'block_type'), [
    ('test_source_media', 'image_url'), ('test_source_audio', 'input_audio'),
    ('test_source_documents', 'input_document'),
])
async def test_owner_media_without_caption_is_an_owner_request(state, module, block_type):
    from importlib import import_module
    body = payload()
    body['user_message'] = import_module(module).message()
    body['user_message']['content'] = [block for block in body['user_message']['content']
                                      if block['type'] == block_type]
    async with AsyncClient(transport=ASGITransport(app=app()), base_url='http://local') as client:
        result = await client.post('/v1/host/turns/sync', json=body)
    assert result.status_code == 200
    assert result.json()['work_recorded'] is True
    store = WorkStore(state.ledger)
    store.reconcile()
    assert store.get_root('root', contact_id='owner')['status'] == 'proven'


@pytest.mark.asyncio
@pytest.mark.parametrize('version', [1, 2])
async def test_work_root_mismatch_is_rejected_before_source_admission(state, version):
    body = payload()
    body['work']['root_turn_id'] = 'another-turn'
    async with AsyncClient(transport=ASGITransport(app=app()), base_url='http://local') as client:
        result = await (client.post('/v1/host/turns/sync', json=body) if version == 1
            else client.put('/v2/host/turns/root', json=body))
    assert result.status_code == 422
    with closing(state.ledger._connect()) as conn:
        assert conn.execute('SELECT count(*) FROM turn_sources').fetchone()[0] == 0


from colonymind.memory import curated
@pytest.mark.asyncio
@pytest.mark.parametrize('with_work', [False, True])
async def test_erased_owner_copy_keeps_source_and_replays_acknowledgment(state, with_work):
    async with AsyncClient(transport=ASGITransport(app=app()), base_url='http://local') as client:
        first = payload('first')
        first.pop('work')
        first['user_message']['content'] = 'yes'
        assert (await client.post('/v1/host/turns/sync', json=first)).status_code == 200
        from test_curated_memory import historical_bytes
        _native_before_withdrawal = historical_bytes(state.ledger)
        curated.forget_sources(state.ledger, 'owner', source_ids=['first'])
        assert historical_bytes(state.ledger) == _native_before_withdrawal
        second = payload('second')
        second['user_message']['content'] = 'yes'
        second['assistant_message']['content'] = 'Changed the file'
        if not with_work:
            second.pop('work')
        result = await client.post('/v1/host/turns/sync', json=second)
        retry = await client.post('/v1/host/turns/sync', json=second)
    assert result.status_code == retry.status_code == 200
    assert result.json() == retry.json()
    assert result.json()['source_recorded'] is True
    assert result.json()['skipped_reason'] == 'source_erased'
    if with_work:
        assert result.json()['work_recorded'] is False
    with closing(state.ledger._connect()) as conn:
        assert conn.execute("SELECT 1 FROM turn_sources WHERE turn_id='second'").fetchone()
