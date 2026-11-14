"""The one source-forget path retains native history and fences every recall pool."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import sqlite3
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from colonymind.api.retired_routes import SOURCE_FORGET
from colonymind.memory import curated
from colonymind.memory.search import collect_sources, select_memory
from colonymind.memory.selection import RecallSelector, keep_offered
from colonymind.api.routers import host
from colonymind.api.schemas.host import ContextSection
from colonymind.memory import compass
from colonymind.memory.trace import NULL_TRACE
from colonymind.turns import TurnIdempotencyLedger
from test_curated_memory import (PERSON, ROOT, GUEST, READER, app_for, client, state, say,
                                 ledger_of, historical_bytes, search, change, id_of, recall)
from test_curated_memory_review import packet

PATH = ROOT + '/sources/forget'


@pytest.mark.asyncio
async def test_source_tombstone_receipt_restart_and_no_native_writes(state, client, monkeypatch):
    await say(state, 'office-source', 'My office is in River.')
    await say(state, 'other-source', 'My tea is green.')
    ledger = ledger_of(state)
    ident = 'claim:' + id_of(state, 'River')
    # A prior corrected view must also disappear when its source is forgotten.
    assert (await change(client, ident, 'correct', 1, text='My office is in Lake.')).status_code == 200
    assert 'Lake' in (await recall(client, 'office'))['content']
    before = historical_bytes(ledger)
    external = []
    for name in ('state.db', 'mind.db'):
        path = state / name
        with closing(sqlite3.connect(path)) as conn, conn:
            conn.execute('CREATE TABLE history (content BLOB)')
            conn.execute('INSERT INTO history VALUES (?)', (b'retained native memory',))
        external.append((path, path.read_bytes()))
    connect = ledger._connect
    writes = []
    event_tables = {'curated_memory_events', 'curated_memory_fence_hashes',
                    'source_erasures', 'source_erasure_revisions'}

    def guarded():
        conn = connect()
        def authorize(action, table, column, database, trigger):
            if action in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE):
                writes.append((action, table))
                if table not in event_tables:
                    return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        conn.set_authorizer(authorize)
        return conn

    with monkeypatch.context() as patch:
        patch.setattr(ledger, '_connect', guarded)
        response = await client.post(PATH, json={'person_id': PERSON, 'source_ids': ['office-source']})
        assert response.status_code == 200, response.text
        first = response.json()
        assert first['source_erased'] is True and first['scope'] == 'curated_source_tombstone'
        assert 'historical data is retained' in first['note']
        assert set(first['receipt']) == {'id', 'revision', 'operation', 'applied_at', 'request_id'}
        assert first['receipt']['operation'] == 'forget' and first['receipt']['revision'] == 1
        assert (await client.post(PATH, json={'person_id': PERSON, 'source_ids': ['office-source', 'office-source']})).json() == first
    assert writes and all(action == sqlite3.SQLITE_INSERT and table in event_tables for action, table in writes)
    assert historical_bytes(ledger) == before
    assert all(path.read_bytes() == original for path, original in external)
    assert curated.forget_sources(TurnIdempotencyLedger(ledger.db_path), PERSON,
                                  source_ids=['office-source']) == first
    assert (await client.get(ROOT + '/' + ident, params={'person_id': PERSON})).status_code == 404
    assert not (await search(client, query='office')).json()['items']
    # A correction of forgotten evidence must not reach the ranking offer either.
    beliefs, quotations = curated.candidates(ledger, PERSON, 'office', [], [])
    assert beliefs == quotations == []
    assert not (await recall(client, 'office'))['content']
    assert 'green' in (await recall(client, 'tea'))['content']
    with closing(ledger._connect()) as conn:
        row = conn.execute('SELECT * FROM curated_memory_events WHERE memory_id=?', (first['receipt']['id'],)).fetchone()
        assert row['data_json'] is None and row['source_ids_json'] == '["office-source"]'


@pytest.mark.asyncio
async def test_exact_provider_text_resolution_is_read_only_and_session_bound(state, client):
    await say(state, 'office-source', 'My office is in River.')
    ledger = ledger_of(state)
    before = historical_bytes(ledger)
    for old_text, session, code in [('office is in River', 's1', 'source_not_found'),
                                     ('My office is in River.', 'another', 'source_not_found')]:
        response = await client.post(PATH, json={'person_id': PERSON, 'session_id': session, 'old_text': old_text})
        assert response.status_code == 422 and response.json()['detail']['code'] == code
    body = {'person_id': PERSON, 'session_id': 's1', 'old_text': 'My office is in River.'}
    first = await client.post(PATH, json=body)
    assert first.status_code == 200, first.text
    assert (await client.post(PATH, json=body)).json() == first.json()
    assert (await client.post(PATH, json={'person_id': PERSON, 'source_ids': ['office-source']})).json() == first.json()
    assert historical_bytes(ledger) == before
    ledger.record_source('checkpoint', contact_id=PERSON, session_id='s1', messages=[
        {'role': 'user', 'content': body['old_text']}])
    before = historical_bytes(ledger)
    response = await client.post(PATH, json=body)
    assert response.status_code == 409 and response.json()['detail']['code'] == 'ambiguous_source'
    assert historical_bytes(ledger) == before


@pytest.mark.asyncio
async def test_source_forget_authority_validation_and_removed_route_are_nonmutating(state, client):
    await say(state, 'office-source', 'My office is in River.')
    ledger = ledger_of(state)
    before = historical_bytes(ledger)
    for fields, status in [({'source_ids': ['missing']}, 422),
                           ({'source_ids': ['office-source', 'missing']}, 422),
                           ({'source_ids': ['office-source'], 'old_text': 'also'}, 422),
                           ({'source_ids': []}, 422), ({'source_ids': ['x' * 257]}, 422),
                           ({'source_ids': ['office-source'], 'extra': True}, 422)]:
        assert (await client.post(PATH, json={'person_id': PERSON, **fields})).status_code == status
    async with AsyncClient(transport=ASGITransport(app=app_for()), base_url='http://test') as other:
        body = {'person_id': PERSON, 'source_ids': ['office-source']}
        assert (await other.post(PATH, json=body)).status_code == 401
        assert (await other.post(PATH, json=body, headers={'Authorization': 'Bearer ' + READER})).status_code == 403
        assert (await other.post(PATH, json={**body, 'person_id': 'contact-b'},
                                 headers={'Authorization': 'Bearer ' + GUEST})).status_code == 403
    # Retirement happens before body validation and never initializes a backend.
    for contents in (b'not-json', b'{}', b'{"contact_id":"contact-a","source_ids":["office-source"]}'):
        gone = await client.post(SOURCE_FORGET, content=contents)
        assert gone.status_code == 410 and gone.json() == {'detail': {'code': 'memory_forget_moved'}}
    assert SOURCE_FORGET not in app_for().openapi()['paths']
    assert historical_bytes(ledger) == before
    with closing(ledger._connect()) as conn:
        assert conn.execute('SELECT count(*) FROM curated_memory_events').fetchone()[0] == 0
    assert 'River' in (await recall(client, 'office'))['content']


@pytest.mark.asyncio
async def test_source_receipts_are_atomic_across_concurrent_connections(state, client):
    for turn in ('a', 'b'):
        await say(state, turn, 'My office is in River.')
    ledger = ledger_of(state)
    before = historical_bytes(ledger)
    def forget(index):
        return curated.forget_sources(TurnIdempotencyLedger(ledger.db_path), PERSON,
                                      source_ids=['a', 'b'] if index % 2 else ['b', 'a', 'b'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(forget, range(8)))
    assert all(receipt == receipts[0] for receipt in receipts)
    assert historical_bytes(ledger) == before
    with closing(ledger._connect()) as conn:
        assert conn.execute('SELECT count(*) FROM curated_memory_events').fetchone()[0] == 1


def test_exact_source_resolution_finds_sealed_text_without_sealing_new_values(state):
    from colonymind.secrets.vault import Vault
    from onekey import KEY
    ledger = ledger_of(state)
    text = 'key ' + KEY
    ledger.record_source('sealed', contact_id=PERSON, session_id='s1', messages=[{'role': 'user', 'content': text}])
    vault_before = Vault(state).entries()
    before = historical_bytes(ledger)
    result = curated.forget_sources(ledger, PERSON, old_text=text, session_id='s1')
    assert result['source_ids'] == ['sealed']
    with pytest.raises(curated.Refused, match='source_not_found'):
        curated.forget_sources(ledger, PERSON, old_text='the password is Tr0ub4dor&3', session_id='s1')
    assert Vault(state).entries() == vault_before and historical_bytes(ledger) == before


def test_reserved_request_id_collision_cannot_claim_a_source_tombstone(state):
    ledger = ledger_of(state)
    ledger.record_source('source', contact_id=PERSON, session_id='s1', messages=[{'role': 'user', 'content': 'office'}])
    reserved = 'source-forget:' + curated._digest(['source-forget-v1', PERSON, ['source']])
    curated.mutate(ledger, PERSON, 'mem_collision', 'remember', request_id=reserved, content='independent')
    before = historical_bytes(ledger)
    with pytest.raises(curated.Refused, match='request_id_conflict'):
        curated.forget_sources(ledger, PERSON, source_ids=['source'])
    assert historical_bytes(ledger) == before
    with closing(ledger._connect()) as conn:
        assert conn.execute("SELECT count(*) FROM curated_memory_events WHERE operation='forget'").fetchone()[0] == 0


def test_source_tombstone_follows_reattribution_without_fencing_another_persons_copies(state):
    from colonymind.turns.idempotency import source_message_hash
    from colonymind.turns.source_attribution import correct as reattribute
    ledger = ledger_of(state)
    message = {'role': 'user', 'content': 'My office is in River.'}
    for turn, person in [('source', PERSON), ('old-copy', PERSON), ('independent', 'contact-b')]:
        ledger.record_source(turn, contact_id=person, session_id='s1', messages=[message])
    before = historical_bytes(ledger)
    first = curated.forget_sources(ledger, PERSON, source_ids=['source'])
    assert historical_bytes(ledger) == before
    independent = {'turn_id': 'independent', 'source_message_hash': source_message_hash('s1', message)}
    assert curated.current_candidates(ledger, [independent], contact_id='contact-b') == [independent]
    assert curated.forget_sources(ledger, PERSON, source_ids=['source']) == first
    old_copy = {'turn_id': 'old-copy', 'source_message_hash': source_message_hash('s1', message)}
    assert curated.current_candidates(ledger, [old_copy], contact_id=PERSON) == []
    reattribute(ledger, operation_id='move-forgotten', performed_by='operator', old_contact_id=PERSON,
                contact_id='contact-b', source_ids=['source'], evidence_refs=['operator:correction'])
    # Source ids are global: correcting attribution must not resurrect forgotten history.
    assert curated.forget_sources(ledger, PERSON, source_ids=['source']) == first
    with pytest.raises(curated.Refused, match='source_not_found'):
        curated.forget_sources(ledger, PERSON, source_ids=['source', 'never-owned'])
    assert curated.current_candidates(ledger, [{'turn_id': 'source'}], contact_id='contact-b') == []
    assert curated.current_candidates(ledger, [old_copy], contact_id=PERSON) == []
    assert curated.current_candidates(ledger, [independent], contact_id='contact-b') == [independent]


def test_conversation_forget_fences_historical_speaker_sources_and_their_copies(state):
    from colonymind.turns.idempotency import source_message_hash
    ledger = ledger_of(state)
    parent = 'observed-conversation:omi:sample'
    speaker = parent + '#speaker_1'
    message = {'role': 'user', 'content': 'My marina is in Harbor.'}
    ledger.record_source(parent, contact_id=PERSON, session_id=parent,
                         messages=[{'role': 'user', 'content': 'A conversation at the marina.'}])
    for source, person in [(speaker, 'contact-b'), ('speaker-copy', 'contact-b'), ('owner-copy', PERSON)]:
        ledger.record_source(source, contact_id=person, session_id=parent, messages=[message])
    before = historical_bytes(ledger)
    first = curated.forget_sources(ledger, PERSON, source_ids=[parent])
    assert first == curated.forget_sources(ledger, PERSON, source_ids=[parent])
    assert historical_bytes(ledger) == before
    copied = lambda source: {'turn_id': source, 'source_message_hash': source_message_hash(parent, message)}
    assert curated.current_candidates(ledger, [copied(speaker), copied('speaker-copy')], contact_id='contact-b') == []
    assert curated.current_candidates(ledger, [copied('owner-copy')], contact_id=PERSON) == [copied('owner-copy')]


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['text', 'voice'])
@pytest.mark.parametrize('inline_chars', [None, 1400, 150])
async def test_source_forget_after_collection_fences_the_offered_pool(state, client, path, inline_chars):
    await say(state, 'office-source', 'My office is in River.')
    ledger = ledger_of(state)
    collected = await collect_sources(ledger, query='office', contact_id=PERSON, session_id='later')
    with keep_offered() as offered:
        selected = await select_memory(collected, query='office', selector=RecallSelector(), timezone_name='UTC', limit=1)
    assert 'River' in selected.content and offered[-1].rows
    curated.forget_sources(ledger, PERSON, source_ids=['office-source'])
    # The exact same shared fence is applied to the complete pool by host repacking.
    assert curated.current_candidates(ledger, offered[-1].rows, contact_id=PERSON) == []
    recalled = SimpleNamespace(packet=selected, offered=offered[-1], ledger=ledger, contact_id=PERSON,
                               session_id='later', served=None)
    section = ContextSection(id=compass.RECALLED, title='Relevant Memories', body=selected.content, priority=90)
    sections, shown = await host._memory_within_allowance([section], recalled, inline_chars=inline_chars, trace=NULL_TRACE)
    assert all('River' not in item.body for item in sections) and not shown
    assert (await packet(ledger, path)).content == ''
