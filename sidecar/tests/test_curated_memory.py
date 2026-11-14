"""B07: curated overrides/tombstones preserve every underlying historical row.

All content and database fixtures are synthetic; no service or model network calls.
"""
import asyncio
from contextlib import closing
import sqlite3

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from onekey import KEY
from colonymind.api.credentials import CredentialTable, secret_digest
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import curated_memory, host
from colonymind.memory import curated
from colonymind.turns import TurnIdempotencyLedger
from test_memory_inventory import state, say, id_of, ledger_of  # noqa: F401

ROOT = '/v1/host/memory/curated'
PERSON = 'contact-a'
GUEST = 'curated-guest-synthetic-0000000'
READER = 'curated-reader-synthetic-00000'


def app_for(*, key=KEY):
    app = FastAPI()
    app.state.curated_ledger = ledger_of(__import__("colonymind").get_state_dir())
    curated_memory._active_ledger = app.state.curated_ledger
    app.include_router(curated_memory.router)
    app.include_router(host.router)
    app.add_middleware(ApiKeyMiddleware, api_key=key, credentials=CredentialTable.from_config({
        'api-key': {'owner': ['whatsapp'], 'contacts': 'all'},
        'guest': {'sha256': secret_digest(GUEST), 'owner': [], 'contacts': [PERSON], 'viewer': PERSON,
                  'scopes': ['memory:read', 'memory:search', 'memory:write']},
        'reader': {'sha256': secret_digest(READER), 'owner': [], 'contacts': [PERSON], 'viewer': PERSON,
                   'scopes': ['memory:read']}}))
    return app


@pytest.fixture
async def client(state, monkeypatch):
    from colonymind import vector
    monkeypatch.setattr(vector, 'get_store', lambda: None)
    monkeypatch.setattr(vector, 'get_pipeline', lambda: None)
    monkeypatch.setenv('COLONYMIND_RECALL_RERANK', 'off')
    async with AsyncClient(transport=ASGITransport(app=app_for()), base_url='http://test',
                           headers={'Authorization': 'Bearer ' + KEY}) as client:
        yield client


async def remember(client, ident='mem_tea', text='I prefer green tea.', request_id='remember-1'):
    return await client.post(ROOT, json={'id': ident, 'person_id': PERSON, 'content': text,
                                         'kind': 'preference', 'request_id': request_id})


async def change(client, ident, operation, revision, *, request_id='change-1', text=None):
    body = {'person_id': PERSON, 'request_id': request_id, 'expected_revision': revision}
    if text is not None:
        body['content'] = text
    return await client.post(f'{ROOT}/{ident}/{operation}', json=body)


async def search(client, **kwargs):
    return await client.get(ROOT + '/search', params={'person_id': PERSON, **kwargs})


async def recall(client, query):
    response = await client.post('/v1/host/memory/search', json={
        'identity': {'host_id': 'fixture'}, 'person_id': PERSON, 'session_id': 'later',
        'query': query, 'limit': 10})
    assert response.status_code == 200, response.text
    return response.json()


def historical_bytes(ledger):
    """Exact historical value bytes, including FTS, apart from append-only fence/feed events."""
    with closing(ledger._connect()) as conn:
        names = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                 if not row[0].startswith('curated_memory_') and row[0] not in {
                     'sqlite_sequence', 'source_erasures', 'source_erasure_revisions'}]
        return {name: sorted((tuple(bytes(value, 'utf-8') if isinstance(value, str) else value for value in row)
                              for row in conn.execute('SELECT * FROM "' + name.replace('"', '""') + '"')),
                             key=repr) for name in names}


@pytest.mark.asyncio
async def test_correct_forget_receipts_and_underlying_history_byte_identity(state, client):
    await say(state, 'office-source', 'My office is in River.')
    ident = 'claim:' + id_of(state, 'River')
    ledger = ledger_of(state)
    # Stand-ins for already existing Hermes/ColonyMind databases; B07 must never open them for writes.
    files = []
    for name in ('state.db', 'mind.db'):
        path = state / name
        with closing(sqlite3.connect(path)) as conn, conn:
            conn.execute('CREATE TABLE history (id INTEGER PRIMARY KEY, content BLOB)')
            conn.execute('INSERT INTO history VALUES (1,?)', (b'synthetic historical chat',))
        files.append((path, path.read_bytes()))
    before = historical_bytes(ledger)
    original = await client.get(f'{ROOT}/{ident}', params={'person_id': PERSON})
    assert original.status_code == 200, original.text
    assert original.json()['revision'] == 1 and 'River' in original.json()['content']
    assert original.json()['verification'] == 'unverified'
    relationships = await client.get(f'{ROOT}/{ident}/relationships', params={'person_id': PERSON})
    assert relationships.json()['revision'] == 1
    assert {ref['kind'] for ref in relationships.json()['relationship_refs']} == {'contact', 'entity'}
    fixed = await change(client, ident, 'correct', 1, text='My office is in Lake.')
    assert fixed.status_code == 200, fixed.text
    assert fixed.json() == {'id': ident, 'revision': 2, 'applied_at': fixed.json()['applied_at'],
                            'operation': 'correct', 'request_id': 'change-1'}
    assert historical_bytes(ledger) == before
    current = await client.get(f'{ROOT}/{ident}', params={'person_id': PERSON, 'revision': 2})
    assert current.json()['content'] == 'My office is in Lake.'
    assert (await search(client, query='River')).json()['items'] == []
    assert (await search(client, query='Lake')).json()['items'][0]['revision'] == 2
    assert 'Lake' in (await recall(client, 'Lake'))['content']
    assert 'River' not in (await recall(client, 'office'))['content']
    # A stale write is refused with a content-free error and changes no revision.
    stale = await change(client, ident, 'correct', 1, text='wrong stale value', request_id='stale')
    assert stale.status_code == 409 and stale.json()['detail']['code'] == 'revision_conflict'
    forgotten = await change(client, ident, 'forget', 2, request_id='forget-1')
    assert forgotten.status_code == 200 and forgotten.json()['revision'] == 3
    assert historical_bytes(ledger) == before
    for path, contents in files:
        assert path.read_bytes() == contents
    assert (await search(client, query='office')).json()['items'] == []
    assert (await recall(client, 'office'))['content'] == ''
    for suffix in ('', '/relationships'):
        hidden = await client.get(f'{ROOT}/{ident}{suffix}', params={'person_id': PERSON, 'revision': 1})
        assert hidden.status_code == 404 and 'River' not in hidden.text
    assert (await change(client, ident, 'correct', 3, text='resurrect', request_id='new')).status_code == 404
    assert (await change(client, ident, 'forget', 3, request_id='second-forget')).status_code == 404
    assert (await change(client, ident, 'forget', 2, request_id='forget-1')).json() == forgotten.json()
    assert (await change(client, ident, 'correct', 1, text='My office is in Lake.')).json() == fixed.json()
    # A new ledger instance proves events/receipts survive restart.
    restarted = TurnIdempotencyLedger(ledger.db_path)
    assert curated.search(restarted, PERSON)['items'] == []
    assert curated.mutate(restarted, PERSON, ident, 'forget', request_id='forget-1', expected_revision=2) == forgotten.json()
    with closing(ledger._connect()) as conn:
        tombstone = conn.execute("SELECT data_json FROM curated_memory_events WHERE operation='forget'").fetchone()
        assert tombstone[0] is None
        with pytest.raises(sqlite3.IntegrityError, match='append_only'):
            conn.execute('DELETE FROM curated_memory_events')
        with pytest.raises(sqlite3.IntegrityError, match='append_only'):
            conn.execute('UPDATE curated_memory_events SET revision=99')


@pytest.mark.asyncio
async def test_remember_dedupe_revision_preconditions_and_restart(state, client):
    first = await remember(client)
    assert first.status_code == 200, first.text
    assert first.json()['revision'] == 1
    assert (await remember(client)).json() == first.json()
    collision = await remember(client, text='different private text')
    assert collision.status_code == 409 and collision.json()['detail']['code'] == 'request_id_conflict'
    assert (await remember(client, request_id='different-request')).json()['detail']['code'] == 'memory_id_conflict'
    assert 'green tea' in (await recall(client, 'What tea do I prefer?'))['content']
    assert (await client.get(ROOT + '/mem_tea', params={'person_id': PERSON, 'revision': 2})).status_code == 409
    fixed = await change(client, 'mem_tea', 'correct', 1, text='I prefer black tea.')
    assert fixed.status_code == 200 and fixed.json()['revision'] == 2
    assert (await search(client, query='GREEN')).json()['items'] == []
    assert 'black tea' in (await recall(client, 'What tea do I prefer?'))['content']
    restarted = TurnIdempotencyLedger(ledger_of(state).db_path)
    assert curated.read(restarted, PERSON, 'mem_tea', 2)['content'] == 'I prefer black tea.'
    # Curated content does not require any chat source row: chat absence/deletion is not forget.
    with closing(restarted._connect()) as conn:
        assert conn.execute('SELECT count(*) FROM turn_sources').fetchone()[0] == 0
    assert (await remember(client)).json() == first.json()
    assert 'curated_memory_v1' in host.supported_capabilities()


@pytest.mark.asyncio
async def test_pagination_empty_unicode_edges_and_invalidation(state, client):
    empty = (await search(client)).json()
    assert empty['status'] == 'empty' and empty['items'] == [] and empty['next_cursor'] is None
    for ident in ('mem_c', 'mem_a', 'mem_b'):
        assert (await remember(client, ident, 'Straße tea ' + ident, ident)).status_code == 200
    first = (await search(client, query='STRASSE', limit=2)).json()
    assert [item['id'] for item in first['items']] == ['mem_a', 'mem_b']
    token = first['next_cursor']
    last = (await search(client, query='strasse', limit=2, cursor=token)).json()
    assert [item['id'] for item in last['items']] == ['mem_c'] and last['next_cursor'] is None
    # Tokens are stable across process/ledger instances; a page must match the result revision.
    assert curated.search(TurnIdempotencyLedger(ledger_of(state).db_path), PERSON,
                          query='strasse', limit=2, cursor=token) == last
    for kwargs in ({'cursor': 'bad'}, {'limit': 0}, {'limit': 101}, {'query': 'x' * 4097},
                   {'cursor': '999.' + first['read_revision'], 'limit': 2, 'query': 'strasse'}):
        assert (await search(client, **kwargs)).status_code == 422
    assert (await search(client, query='other', limit=2, cursor=token)).status_code == 409
    assert (await change(client, 'mem_b', 'forget', 1)).status_code == 200
    stale = await search(client, query='strasse', limit=2, cursor=token)
    assert stale.status_code == 409 and stale.json()['detail']['code'] == 'curated_changed_restart'
    assert [item['id'] for item in (await search(client, query='   ')).json()['items']] == ['mem_a', 'mem_c']


@pytest.mark.asyncio
async def test_auth_scopes_person_isolation_and_schema_refusals(state, client):
    await remember(client)
    for key in ('', 'incorrect', GUEST, READER):
        async with AsyncClient(transport=ASGITransport(app=app_for()), base_url='http://test',
                               headers={'Authorization': 'Bearer ' + key} if key else {}) as other:
            result = await other.get(ROOT + '/mem_tea', params={'person_id': PERSON})
            if key in ('', 'incorrect'):
                assert result.status_code == 401
                continue
            assert result.status_code == 200
            assert (await other.get(ROOT + '/mem_tea', params={'person_id': 'contact-b'})).status_code == 403
            if key == READER:
                assert (await search(other)).status_code == 403
                assert (await change(other, 'mem_tea', 'forget', 1)).status_code == 403
            else:
                assert (await search(other)).status_code == 200
                assert (await other.get(ROOT + '/mem_tea', params={'person_id': PERSON})).status_code == 200
    assert (await client.get(ROOT + '/mem_tea', params={'person_id': 'contact-b'})).status_code == 404
    assert (await client.get(ROOT + '/search')).status_code == 422
    for field, value in (('person_id', ' '), ('content', ' '), ('request_id', ''), ('id', 'bad'), ('extra', 'x')):
        payload = {'person_id': PERSON, 'id': 'mem_valid', 'content': 'content', 'request_id': 'valid', field: value}
        assert (await client.post(ROOT, json=payload)).status_code == 422
    assert (await change(client, 'mem_tea', 'forget', True)).status_code == 422
    async with AsyncClient(transport=ASGITransport(app=app_for(key='')), base_url='http://127.0.0.1') as dev:
        assert (await search(dev)).status_code == 403


@pytest.mark.asyncio
async def test_compare_and_append_concurrent_writers(client):
    await remember(client)
    results = await asyncio.gather(*(change(client, 'mem_tea', 'correct', 1,
                                            request_id=f'writer-{n}', text=f'tea value {n}') for n in range(2)))
    assert sorted(result.status_code for result in results) == [200, 409]
    assert (await client.get(ROOT + '/mem_tea', params={'person_id': PERSON})).json()['revision'] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['correct', 'forget'])
async def test_mutation_during_ranking_never_returns_stale_text(state, client, monkeypatch, operation):
    from colonymind.memory.search import collect_sources, select_memory
    await say(state, 'office-source', 'My office is in River.')
    ledger = ledger_of(state)
    ident = 'claim:' + id_of(state, 'River')
    collected = await collect_sources(ledger, query='office', contact_id=PERSON, session_id='later')
    real = host._memory_context_selector()

    class MutatingSelector:
        async def select_context(self, *args, **kwargs):
            selected, content = await real.select_context(*args, **kwargs)
            assert 'River' in content
            curated.mutate(ledger, PERSON, ident, operation, request_id='during-ranking', expected_revision=1,
                           content='My office is in Lake.' if operation == 'correct' else None)
            return selected, content

    packet = await select_memory(collected, query='office', selector=MutatingSelector(), timezone_name='UTC')
    assert 'River' not in packet.content and packet.selected == []
    assert 'River' not in (await recall(client, 'office'))['content']
    if operation == 'correct':
        assert 'Lake' in (await recall(client, 'office'))['content']


@pytest.mark.asyncio
async def test_hand_refusal_and_backend_failure_are_content_free(client, monkeypatch):
    from colonymind.hands import sessions
    app = app_for()
    app.state.hands_store = object()

    async def registered_hand(store, session_id):
        return {'hand_id': 'fixture-hand'}

    monkeypatch.setattr(sessions, 'lookup', registered_hand)
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test',
                           headers={'Authorization': 'Bearer ' + KEY,
                                    'X-ColonyMind-Session': 'fixture-child'}) as hand:
        refused = await remember(hand)
        assert refused.status_code == 403 and refused.json()['detail']['code'] == 'hand_session_refused'
    assert (await search(client)).json()['items'] == []

    def unavailable(*args, **kwargs):
        raise sqlite3.OperationalError('private backend exception must stay private')

    monkeypatch.setattr(curated, 'search', unavailable)
    failed = await search(client)
    assert failed.status_code == 503
    assert failed.json() == {'detail': {'code': 'curated_backend_unavailable'}}


@pytest.mark.asyncio
async def test_checkpoint_copy_cannot_bypass_curated_forget(state, client):
    await say(state, 'office-source', 'My office is in River.')
    ledger = ledger_of(state)
    ident = 'claim:' + id_of(state, 'River')
    ledger.record_source('checkpoint-copy', contact_id=PERSON, session_id='s1', scope='session',
                         messages=[{'role': 'user', 'content': 'My office is in River.'}])
    before = historical_bytes(ledger)
    assert (await change(client, ident, 'forget', 1)).status_code == 200
    response = await client.post('/v1/host/memory/search', json={
        'identity': {'host_id': 'fixture'}, 'person_id': PERSON, 'session_id': 's1', 'query': 'office'})
    assert response.status_code == 200, response.text
    assert response.json()['content'] == ''
    assert historical_bytes(ledger) == before


@pytest.mark.asyncio
async def test_explicit_override_changed_during_ranking_is_dropped(state, client):
    from colonymind.memory.search import collect_sources, select_memory
    await remember(client)
    ledger = ledger_of(state)
    collected = await collect_sources(ledger, query='tea', contact_id=PERSON, session_id='later')
    real = host._memory_context_selector()

    class MutatingSelector:
        async def select_context(self, *args, **kwargs):
            selected, content = await real.select_context(*args, **kwargs)
            assert 'green tea' in content
            curated.mutate(ledger, PERSON, 'mem_tea', 'correct', request_id='during-ranking',
                           expected_revision=1, content='I prefer black tea.')
            return selected, content

    result = await select_memory(collected, query='tea', timezone_name='UTC', selector=MutatingSelector())
    assert result.content == ''
    assert 'black tea' in (await recall(client, 'What tea do I prefer?'))['content']


def test_curated_access_logs_omit_private_queries_for_all_statuses():
    import copy
    import logging
    from colonymind.runtime_logging import RuntimeFormatter
    for status in (200, 403, 422, 503):
        for path in (ROOT + '/search', ROOT + '/mem_tea/relationships'):
            record = logging.LogRecord('uvicorn.access', logging.INFO, 'fixture', 1, '%s %s %s %s %d',
                                       ('fixture', 'GET', path + '?query=private-tea-words&person_id=private-person',
                                        '1.1', status), None)
            before = copy.deepcopy(record.__dict__)
            value = RuntimeFormatter().format(record)
            assert 'private-' not in value and '?<omitted>' in value
            assert record.__dict__ == before


@pytest.mark.asyncio
async def test_modern_read_credential_reads_only_its_bound_curated_memories(state, client):
    await remember(client)
    token = 'modern-reader-synthetic-00000000'
    app = FastAPI()
    app.state.curated_ledger = ledger_of(__import__("colonymind").get_state_dir())
    curated_memory._active_ledger = app.state.curated_ledger
    app.include_router(curated_memory.router)
    app.add_middleware(ApiKeyMiddleware, api_key=KEY, credentials=CredentialTable.from_config({
        'reader': {'sha256': secret_digest(token), 'owner': [], 'contacts': [PERSON], 'viewer': PERSON,
                   'access': ['read']}}))
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test',
                           headers={'Authorization': 'Bearer ' + token}) as reader:
        assert (await search(reader)).status_code == 200
        assert (await reader.get(ROOT + '/mem_tea', params={'person_id': PERSON})).status_code == 200
        assert (await search(reader, person_id='contact-b')).status_code == 403
        assert (await change(reader, 'mem_tea', 'forget', 1)).status_code == 403


@pytest.mark.asyncio
async def test_another_claim_quoting_the_same_source_does_not_reveal_forgotten_content(state, client):
    await say(state, 'shared-source', 'My office is in River. My tea is green.')
    office = 'claim:' + id_of(state, 'River')
    tea = 'claim:' + id_of(state, 'green')
    before = historical_bytes(ledger_of(state))
    assert (await search(client, query='River')).json()['items']
    assert (await change(client, office, 'forget', 1)).status_code == 200
    assert (await search(client, query='River')).json()['items'] == []
    assert (await client.get(f'{ROOT}/{tea}', params={'person_id': PERSON})).status_code == 404
    assert historical_bytes(ledger_of(state)) == before


@pytest.mark.asyncio
async def test_voice_recall_honors_curated_revisions_on_its_shared_connection(state, client):
    import threading
    from colonymind.memory.voice_recall import read_voice_packet
    await say(state, 'office-source', 'My office is in River.')
    ledger = ledger_of(state)
    ident = 'claim:' + id_of(state, 'River')
    before = historical_bytes(ledger)

    def voice():
        return read_voice_packet(ledger, query='office', contact_id=PERSON, session_id='later',
                                 k=5, max_tokens=600, timezone_name='UTC', cancelled=threading.Event())

    assert 'River' in (await asyncio.to_thread(voice)).content
    assert (await change(client, ident, 'correct', 1, text='My office is in Lake.')).status_code == 200
    corrected = await asyncio.to_thread(voice)
    assert 'Lake' in corrected.content and 'River' not in corrected.content
    assert (await change(client, ident, 'forget', 2, request_id='forget-voice')).status_code == 200
    assert (await asyncio.to_thread(voice)).content == ''
    assert historical_bytes(ledger) == before


def visible_source_messages(ledger, source_id):
    import json
    from colonymind.turns.idempotency import source_message_hash
    with ledger._connect() as db:
        row = db.execute('SELECT session_id,messages_json FROM turn_sources WHERE turn_id=?', (source_id,)).fetchone()
        if row is None:
            return []
        return [message for message in json.loads(row['messages_json'])
                if not curated.is_forgotten(ledger, {'source_id': source_id,
                    'message_hash': source_message_hash(row['session_id'], message)}, conn=db)]
