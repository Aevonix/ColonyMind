"""Review regressions use synthetic ledgers and never contact a running store."""
import asyncio
from contextlib import closing, contextmanager
import json
import threading
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from colonymind.api.routers import curated_memory, host
from colonymind.memory import curated
from colonymind.memory.search import collect_sources, select_memory
from colonymind.memory.selection import RecallSelector, keep_offered
from colonymind.memory.voice_recall import read_voice_packet, VoiceMemoryUnavailable
from colonymind.turns import TurnIdempotencyLedger
from test_curated_memory import (PERSON, ROOT, KEY, app_for, client, state, ledger_of,
                                 say, remember, change, search, historical_bytes)


async def packet(ledger, path='text'):
    if path == 'voice':
        return await asyncio.to_thread(read_voice_packet, ledger, query='office', contact_id=PERSON,
            session_id='later', k=5, max_tokens=600, timezone_name='UTC', cancelled=threading.Event())
    collected = await collect_sources(ledger, query='office', contact_id=PERSON, session_id='later')
    return await select_memory(collected, query='office', selector=RecallSelector(), timezone_name='UTC', limit=5)


async def plain_packet(ledger, monkeypatch, path):
    with monkeypatch.context() as patch:
        patch.setattr(curated, 'candidates', lambda ledger, person, query, beliefs, quotations: (beliefs, quotations))
        patch.setattr(curated, 'current_candidates', lambda ledger, rows, **kwargs: rows)
        return await packet(ledger, path)


async def withheld_packet(ledger, path='text'):
    if path == 'voice':
        with pytest.raises(VoiceMemoryUnavailable):
            await packet(ledger, path)
    else:
        result = await packet(ledger, path)
        assert not result.selected and result.content == '' and result.source_refs == []


def test_curated_initialization_error_preserves_native_ledger(tmp_path, monkeypatch):
    def failed(conn):
        conn.execute('CREATE TABLE curated_partial (id)')
        raise RuntimeError('synthetic initialization failure')
    monkeypatch.setattr(curated, 'initialize', failed)
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    assert ledger.record_source('native', contact_id=PERSON, session_id='s',
                                messages=[{'role': 'user', 'content': 'office in River'}])
    assert ledger.search_sources('office', contact_id=PERSON, session_id='s', limit=5)
    assert not curated.available(ledger)
    with closing(ledger._connect()) as conn:
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='curated_partial'").fetchone()
        assert 'office in River' in conn.execute('SELECT messages_json FROM turn_sources').fetchone()[0]


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['text', 'voice'])
async def test_missing_curated_table_returns_exact_plain_recall(state, monkeypatch, path):
    monkeypatch.setattr(curated, 'initialize', lambda conn: None)
    await say(state, 'native', 'My office is in River.')
    ledger = ledger_of(state)
    await withheld_packet(ledger, path)
    app_for()
    assert 'curated_memory_v1' not in host.supported_capabilities()


@pytest.mark.asyncio
@pytest.mark.parametrize('path,fail_at', [('text', 2), ('voice', 2), ('voice', 3)])
async def test_curated_read_failure_after_preparation_returns_exact_plain_recall(
        state, client, monkeypatch, path, fail_at):
    await say(state, 'native', 'My office is in River.')
    assert (await remember(client, text='My office has curated violet wallpaper.')).status_code == 200
    ledger = ledger_of(state)
    real = curated._snapshot
    calls = 0

    @contextmanager
    def failing(ledger):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise RuntimeError('synthetic curated read failure')
        with real(ledger) as conn:
            yield conn

    monkeypatch.setattr(curated, '_snapshot', failing)
    await withheld_packet(ledger, path)
    assert calls >= fail_at
    assert 'curated_memory_v1' not in host.supported_capabilities()


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['correct', 'forget'])
async def test_allowance_repacking_fences_entire_offered_pool(state, client, operation):
    from colonymind.memory import compass
    from colonymind.memory.trace import NULL_TRACE
    from colonymind.api.schemas.host import ContextSection
    await remember(client, 'mem_a', 'My office is in River.', 'short')
    await remember(client, 'mem_b', 'My office review notes: ' + 'Synthetic independent detail. ' * 90, 'long')
    ledger = ledger_of(state)

    class MutatingSelector(RecallSelector):
        async def select_context(self, *args, **kwargs):
            result = await super().select_context(*args, **kwargs)
            curated.mutate(ledger, PERSON, 'mem_a', operation, request_id='during-ranking',
                expected_revision=1, content='My office is in Lake.' if operation == 'correct' else None)
            return result

    collected = await collect_sources(ledger, query='office', contact_id=PERSON, session_id='later')
    with keep_offered() as offered:
        result = await select_memory(collected, query='office', selector=MutatingSelector(),
                                     timezone_name='UTC', limit=None)
    assert 'River' not in result.content
    recalled = SimpleNamespace(packet=result, offered=offered[-1], ledger=ledger, contact_id=PERSON,
                               session_id='later', served=None)
    section = ContextSection(id=compass.RECALLED, title='Relevant Memories', body=result.content, priority=90)
    sections, shown = await host._memory_within_allowance([section], recalled, inline_chars=1400, trace=NULL_TRACE)
    assert all('River' not in item.body for item in sections)
    assert all(row['id'] != 'mem_a' for row in shown)


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['correct', 'forget'])
@pytest.mark.parametrize('dependency', ['subject_message', 'claim_reference', 'nested_source'])
async def test_curated_reads_and_recall_share_recorded_dependency_fence(tmp_path, operation, dependency):
    from test_source_claim_earlier_subjects import Scripted, fact, say as source, extract, stored
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    root, derived = 'My sister lives in Denver.', 'She works in Denver.'
    model = Scripted({root: [fact(root, 'My sister', 'location', 'Denver')],
                      derived: [fact(derived, 'My sister', 'workplace', 'Denver', basis=(root, 'My sister'))]})
    source(ledger, 'root', root, minute=0)
    source(ledger, 'derived', derived, minute=2)
    await extract(ledger, model)
    rows = stored(ledger)
    root_id, dependent_id = ('claim:' + rows[turn]['id'] for turn in ('root', 'derived'))
    if dependency != 'subject_message':
        # Synthetic future/native dependency fields must use the same recursive fence.
        data = dict(rows['derived'])
        data.pop('subject_basis_message', None)
        data['recorded_dependencies'] = ([{'source_claim_id': rows['root']['id']}] if dependency == 'claim_reference'
                                          else [{'context': {'source_turn_id': 'root'}}])
        with closing(ledger._connect()) as conn, conn:
            conn.execute('UPDATE source_claims SET data_json=? WHERE id=?', (json.dumps(data), rows['derived']['id']))
    assert curated.read(ledger, 'owner', dependent_id)
    before = historical_bytes(ledger)
    curated.mutate(ledger, 'owner', root_id, operation, request_id='mutate', expected_revision=1,
                   content='My sister lives in Boulder.' if operation == 'correct' else None)
    assert dependent_id not in {item['id'] for item in curated.search(ledger, 'owner')['items']}
    with pytest.raises(curated.Refused, match='memory_not_found'):
        curated.read(ledger, 'owner', dependent_id)
    assert historical_bytes(ledger) == before
    collected = await collect_sources(ledger, query='Denver', contact_id='owner', session_id='chat')
    result = await select_memory(collected, query='Denver', selector=RecallSelector(), timezone_name='UTC')
    assert 'Denver' not in result.content
    voice = await asyncio.to_thread(read_voice_packet, ledger, query='Denver', contact_id='owner', session_id='chat',
        k=5, max_tokens=600, timezone_name='UTC', cancelled=threading.Event())
    assert 'Denver' not in voice.content


@pytest.mark.asyncio
@pytest.mark.parametrize('key', ['sequence', 'request', 'revision'])
async def test_insert_or_replace_cannot_replace_journal_events(state, client, key):
    await remember(client)
    ledger = ledger_of(state)
    with closing(ledger._connect()) as conn, conn:
        assert conn.execute('PRAGMA recursive_triggers').fetchone()[0] == 0
        row = dict(conn.execute('SELECT * FROM curated_memory_events').fetchone())
        original = dict(row)
        row.update(data_json='{}')
        if key != 'sequence':
            row.pop('sequence')
        if key != 'request':
            row['request_id'] = 'different-request'
        if key != 'revision':
            row['memory_id'] = 'mem_other'
        with pytest.raises(__import__('sqlite3').IntegrityError, match='append_only'):
            conn.execute('INSERT OR REPLACE INTO curated_memory_events (' + ','.join(row) + ') VALUES (' +
                         ','.join('?' for _ in row) + ')', list(row.values()))
        assert dict(conn.execute('SELECT * FROM curated_memory_events').fetchone()) == original


@pytest.mark.asyncio
async def test_curated_routes_use_startup_ledger_without_cold_migrations(state, client, monkeypatch):
    await remember(client)
    ledger = ledger_of(state)
    app = app_for()
    app.state.curated_ledger = ledger
    before = historical_bytes(ledger)
    def cold(*args, **kwargs):
        raise AssertionError('request attempted cold initialization/migration')
    monkeypatch.setattr('colonymind.turns.get_turn_idempotency_ledger', cold)
    monkeypatch.setattr(TurnIdempotencyLedger, '_initialize', cold)
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test',
                           headers={'Authorization': 'Bearer ' + KEY}) as bound:
        assert (await search(bound)).status_code == 200
        assert (await bound.get(ROOT + '/mem_tea', params={'person_id': PERSON})).status_code == 200
        assert (await bound.get(ROOT + '/mem_tea/relationships', params={'person_id': PERSON})).status_code == 200
        assert (await remember(bound, 'mem_new', request_id='new')).status_code == 200
        assert (await change(bound, 'mem_tea', 'correct', 1, text='black tea')).status_code == 200
        assert (await change(bound, 'mem_tea', 'forget', 2, request_id='forget')).status_code == 200
    assert historical_bytes(ledger) == before


@pytest.mark.asyncio
async def test_unbound_curated_routes_refuse_without_cold_initialization(state, monkeypatch):
    app = app_for()
    del app.state.curated_ledger
    calls = []
    ledger = ledger_of(state)
    def cold(*args, **kwargs):
        calls.append(True)
        return ledger
    monkeypatch.setattr('colonymind.turns.get_turn_idempotency_ledger', cold)
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test',
                           headers={'Authorization': 'Bearer ' + KEY}) as bound:
        response = await search(bound)
    assert not calls
    assert response.status_code == 503
    assert response.json() == {'detail': {'code': 'curated_backend_unavailable'}}


@pytest.mark.asyncio
async def test_inventory_reconstruction_does_not_hold_writer_lock(state, client, monkeypatch):
    from test_memory_inventory import id_of
    await say(state, 'native', 'My office is in River.')
    ledger = ledger_of(state)
    ident = 'claim:' + id_of(state, 'River')
    real = curated._items
    inspected = []
    def check(conn, person):
        with closing(ledger._connect()) as other:
            other.execute('PRAGMA busy_timeout=0')
            other.execute('BEGIN IMMEDIATE')
            other.rollback()
        inspected.append(True)
        return real(conn, person)
    monkeypatch.setattr(curated, '_items', check)
    curated.mutate(ledger, PERSON, ident, 'forget', request_id='forget', expected_revision=1)
    assert inspected


@pytest.mark.asyncio
@pytest.mark.parametrize('count', [0, 10000])
async def test_warm_recall_does_not_materialize_or_parse_nonmatching_events(state, monkeypatch, count):
    await say(state, 'native', 'My office is in River.')
    ledger = ledger_of(state)
    # Bulk fixture creation, not 10,000 inventory reconstructions.
    with closing(ledger._connect()) as conn, conn:
        for n in range(count):
            item = {'id': f'mem_{n}', 'person_id': PERSON, 'revision': 1, 'content': f'Synthetic zzzz {n}',
                    'kind': 'fact', 'state': 'active', 'verification': 'owner_asserted', 'applied_at': '2026-01-01',
                    'source_refs': [], 'relationship_refs': []}
            conn.execute('''INSERT INTO curated_memory_events
                (person_id,memory_id,revision,operation,request_id,request_digest,applied_at,data_json,source_ids_json)
                VALUES (?,?,1,'remember',?,'fixture','2026-01-01',?,'[]')''',
                (PERSON, item['id'], item['id'], json.dumps(item)))
    # Warm both paths, then count actual event reads/parses, not wall-clock scheduling noise.
    await packet(ledger)
    await packet(ledger, 'voice')
    reads, parsed = [], []
    real_events, real_loads = curated._events, json.loads
    def events(*args):
        reads.append(True)
        return real_events(*args)
    def loads(value, *args, **kwargs):
        if isinstance(value, str) and 'Synthetic zzzz' in value:
            parsed.append(True)
        return real_loads(value, *args, **kwargs)
    monkeypatch.setattr(curated, '_events', events)
    monkeypatch.setattr(curated.json, 'loads', loads)
    for path in ('text', 'voice'):
        assert 'River' in (await packet(ledger, path)).content
    assert not reads
    assert not parsed


@pytest.mark.asyncio
async def test_curated_api_read_failure_disables_overlay_and_capability(state, client, monkeypatch):
    await say(state, 'native', 'My office is in River.')
    await remember(client, text='My office has violet wallpaper.')
    ledger = ledger_of(state)
    def failing(*args, **kwargs):
        raise __import__('sqlite3').OperationalError('synthetic API backend read failure')
    monkeypatch.setattr(curated, 'search', failing)
    assert (await search(client)).status_code == 503
    assert 'curated_memory_v1' not in host.supported_capabilities()
    assert (await packet(ledger)).content
    assert curated.available(ledger)
    from copy import copy
    facade = copy(ledger)
    facade.__dict__.pop('_curated_available', None)
    assert (await packet(facade)).content


@pytest.mark.asyncio
async def test_cached_recall_observes_external_journal_and_native_witness_commits(state, client):
    from test_memory_inventory import id_of
    from colonymind.turns.idempotency import source_message_hash
    await say(state, 'native', 'My office is in River.')
    ledger = ledger_of(state)
    ident = 'claim:' + id_of(state, 'River')
    assert 'River' in (await packet(ledger)).content
    other = TurnIdempotencyLedger(ledger.db_path)
    curated.mutate(other, PERSON, ident, 'forget', request_id='external', expected_revision=1)
    assert not (await packet(ledger)).content
    # A folded restatement and its checkpoint copy arrive after the fence was cached.
    text = 'My office has a River location.'
    message = {'role': 'user', 'content': text}
    for turn in ('witness', 'checkpoint'):
        ledger.record_source(turn, contact_id=PERSON, session_id='s1', messages=[message])
    with closing(ledger._connect()) as conn, conn:
        original = conn.execute('SELECT data_json FROM source_claims WHERE id=?', (ident[6:],)).fetchone()[0]
        data = json.loads(original)
        data.update(evidence=text)
        conn.execute('''INSERT INTO source_claim_witnesses (claim_id,turn_id,message_hash,data_json)
                        VALUES (?,?,?,?)''', (ident[6:], 'witness', source_message_hash('s1', message), json.dumps(data)))
    assert 'River' not in (await packet(ledger)).content
    assert 'River' not in (await packet(ledger, 'voice')).content


@pytest.mark.asyncio
async def test_mutation_reconstructs_again_after_concurrent_inventory_commit(state, client, monkeypatch):
    from test_memory_inventory import id_of
    await say(state, 'native', 'My office is in River.')
    ledger = ledger_of(state)
    ident = 'claim:' + id_of(state, 'River')
    real = curated._items
    calls = []
    def changing(conn, person):
        result = real(conn, person)
        calls.append(True)
        if len(calls) == 1:
            other = TurnIdempotencyLedger(ledger.db_path)
            curated.mutate(other, PERSON, ident, 'correct', request_id='competitor', expected_revision=1,
                           content='My office is in Lake.')
        return result
    monkeypatch.setattr(curated, '_items', changing)
    with pytest.raises(curated.Refused, match='revision_conflict'):
        curated.mutate(ledger, PERSON, ident, 'forget', request_id='stale', expected_revision=1)
    assert len(calls) >= 2
    assert curated.read(ledger, PERSON, ident)['content'] == 'My office is in Lake.'


@pytest.mark.asyncio
async def test_allowance_curated_read_failure_rebuilds_exact_plain_packet(state, client, monkeypatch):
    from colonymind.memory import compass
    from colonymind.memory.trace import NULL_TRACE
    from colonymind.api.schemas.host import ContextSection
    from test_memory_inventory import id_of
    await say(state, 'native', 'My office is in River.')
    ledger = ledger_of(state)
    curated.mutate(ledger, PERSON, 'claim:' + id_of(state, 'River'), 'forget',
                   request_id='forget-native', expected_revision=1)
    await remember(client, 'mem_long', 'My office review notes: ' + 'Curated independent detail. ' * 90)
    collected = await collect_sources(ledger, query='office', contact_id=PERSON, session_id='later')

    async def select():
        with keep_offered() as offered:
            result = await select_memory(collected, query='office', selector=RecallSelector(),
                                         timezone_name='UTC', limit=None)
        return result, offered[-1]

    def recalled(result, offered):
        return SimpleNamespace(packet=result, offered=offered, ledger=ledger, contact_id=PERSON,
                               session_id='later', served=None, plain_recall=select)

    def section(result):
        return ContextSection(id=compass.RECALLED, title='Relevant Memories', body=result.content, priority=90)

    result, offered = await select()
    assert 'River' not in result.content and len(result.content) > 1400

    @contextmanager
    def failing(ledger):
        raise RuntimeError('synthetic curated read failure at allowance fitting')
        yield

    monkeypatch.setattr(curated, '_snapshot', failing)
    actual, actual_rows = await host._memory_within_allowance([section(result)], recalled(result, offered),
                                                            inline_chars=1400, trace=NULL_TRACE)
    assert actual == [] and actual_rows == []
