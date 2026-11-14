"""The review's second-check schedule applied to other public batch readers."""
from contextlib import closing

import pytest

from colonymind.memory import curated, inventory
from colonymind.turns import TurnIdempotencyLedger
from colonymind.turns.idempotency import source_message_hash
from test_curated_read_races import withdraw


def peer_between_checks(monkeypatch, ledger):
    predicate = curated.is_forgotten
    depth = calls = 0
    target = None
    fired = []

    def source(ref):
        if isinstance(ref, str):
            return ref
        if isinstance(ref, list):
            return source(ref[0])
        return (ref.get('turn_id') or ref.get('source_id') or ref.get('source_turn_id')
                or ref.get('source_uri', '').removeprefix('turn:'))

    def check(reader, ref, **kwargs):
        nonlocal depth, calls, target
        if not depth:
            calls += 1
            if calls == 1:
                target = source(ref)
            if calls == 2:
                assert target
                withdraw(ledger, target)
                fired.append(True)
        depth += 1
        try:
            return predicate(reader, ref, **kwargs)
        finally:
            depth -= 1

    monkeypatch.setattr(curated, 'is_forgotten', check)
    return fired


@pytest.fixture
def sources(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    for source in ('first', 'second', 'seed'):
        ledger.record_source(source, contact_id='fixture-owner', session_id='fixture-session',
            messages=[{'role': 'user', 'content': 'Reference ' + source}], derive_claims=False)
    curated.forget_sources(ledger, 'fixture-owner', source_ids=['seed'])
    return ledger


@pytest.mark.parametrize('reader', ['search_sources', 'search_stems', 'history', 'outcomes', 'query_log'])
def test_other_source_readers_recheck_complete_result(sources, tmp_path, monkeypatch, reader):
    from colonymind.instinct.outcomes import Sources
    from colonymind.plasticity.query_log import QueryLog
    from colonymind.turns.history_references import resolve
    ledger = sources
    refs = [{'session_id': 'fixture-session', 'message_hash': source_message_hash('fixture-session',
        {'role': 'user', 'content': 'Reference ' + source})} for source in ('first', 'second')]
    read = {'search_sources': lambda: ledger.search_sources('Reference',
                contact_id='fixture-owner', session_id='fixture-session'),
        'search_stems': lambda: ledger.search_stems('Reference',
                contact_id='fixture-owner', session_id='fixture-session'),
        'history': lambda: resolve(ledger, contact_id='fixture-owner', session_id='fixture-session', references=refs),
        'outcomes': lambda: Sources(tmp_path).erased(['first', 'second']),
        'query_log': lambda: QueryLog(tmp_path / 'queries.db').sources_current(['first', 'second'])}[reader]
    fired = peer_between_checks(monkeypatch, ledger)
    offered = read()
    assert fired
    assert offered == read()


@pytest.mark.asyncio
@pytest.mark.parametrize('reader', ['claim_rows', 'inventory', 'curated_search'])
async def test_other_claim_readers_recheck_complete_result(tmp_path, monkeypatch, reader):
    from colonymind.beliefs.source_projection import SourceClaimProjection
    from test_source_claim_projection import Model, claim
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    projection = SourceClaimProjection(ledger)
    for source in ('first', 'second'):
        text = 'My office is in ' + source + '.'
        ledger.record_source(source, contact_id='fixture-owner', session_id='fixture-session',
                             messages=[{'role': 'user', 'content': text}])
        assert await projection.process_one(Model({text: claim(text, source)}))
    ledger.record_source('seed', contact_id='fixture-owner', session_id='fixture-session',
                         messages=[{'role': 'user', 'content': 'Seed'}], derive_claims=False)
    curated.forget_sources(ledger, 'fixture-owner', source_ids=['seed'])

    def rows():
        with closing(ledger._connect()) as conn:
            return projection._rows(conn, 'fixture-owner', 'fixture-session')

    read = {'claim_rows': rows,
        'inventory': lambda: inventory.page(ledger, 'fixture-owner'),
        'curated_search': lambda: curated.search(ledger, 'fixture-owner')}[reader]
    fired = peer_between_checks(monkeypatch, ledger)
    offered = read()
    assert fired
    assert offered == read()


@pytest.mark.parametrize('reader', ['list_roots', 'summary'])
def test_other_work_readers_recheck_complete_result(tmp_path, monkeypatch, reader):
    from colonymind.work.store import WorkStore
    from test_work_api import retained_work
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    for source in ('first', 'second'):
        retained_work(ledger, source, contact='fixture-owner')
    work = WorkStore(ledger)
    work.reconcile()
    ledger.record_source('seed', contact_id='fixture-owner', session_id='fixture-session',
                         messages=[{'role': 'user', 'content': 'Seed'}], derive_claims=False)
    curated.forget_sources(ledger, 'fixture-owner', source_ids=['seed'])
    read = lambda: getattr(work, reader)(contact_id='fixture-owner')
    fired = peer_between_checks(monkeypatch, ledger)
    offered = read()
    assert fired
    assert offered == read()


@pytest.mark.asyncio
@pytest.mark.parametrize('reader', ['scan_all', 'search'])
async def test_vector_readers_recheck_complete_result(sources, tmp_path, monkeypatch, reader):
    from colonymind.vector import Collection
    from colonymind.vector.indexes import EmbeddingIdentity, IndexCatalog
    from colonymind.vector.store import VectorStore
    identity = EmbeddingIdentity('fixture', 'fixture', 'known', 2)
    store = VectorStore(str(tmp_path / 'vectors'), identity=identity, catalog=IndexCatalog(sources))
    await store.connect(2)
    await store.ensure_collections(2)
    try:
        for source in ('first', 'second'):
            await store.add(Collection.MEMORIES, id=source, text=source, vector=[1., 0.],
                            metadata={'source_uri': 'turn:' + source})
        read = (lambda: store.scan_all(Collection.MEMORIES)) if reader == 'scan_all' else (
            lambda: store.search(Collection.MEMORIES, [1., 0.]))
        assert len(await read()) == 2
        fired = peer_between_checks(monkeypatch, sources)
        offered = await read()
        assert fired
        fresh = await read()
        ids = (lambda rows: [row['id'] for row in rows]) if reader == 'scan_all' else (
            lambda rows: [row.id for row in rows])
        assert ids(offered) == ids(fresh)
        assert len(offered) == 1
        assert await store.count(Collection.MEMORIES) == 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_source_vector_search_fences_final_hydration(tmp_path, monkeypatch):
    from colonymind.turns import source_vectors
    from test_source_vectors import setup, drain
    ledger, store, _, projection = await setup(tmp_path)
    try:
        for source in ('first', 'second', 'seed'):
            ledger.record_source(source, contact_id='fixture-owner', session_id='fixture-session',
                messages=[{'role': 'user', 'content': 'Reference ' + source}], derive_claims=False)
        await drain(projection)
        curated.forget_sources(ledger, 'fixture-owner', source_ids=['seed'])
        search, hydrate = store.search, source_vectors.hydrate
        final = False
        calls = 0
        target = None

        async def searched(*args, **kwargs):
            nonlocal final
            result = await search(*args, **kwargs)
            final = True
            return result

        def checked(reader, metadata, **kwargs):
            nonlocal calls, target
            if final:
                calls += 1
                if calls == 1:
                    target = metadata['source_turn_id']
                if calls == 2:
                    withdraw(ledger, target)
            return hydrate(reader, metadata, **kwargs)

        monkeypatch.setattr(store, 'search', searched)
        monkeypatch.setattr(source_vectors, 'hydrate', checked)
        offered, media = await projection.search('Reference', contact_id='fixture-owner',
                                                 session_id='fixture-session')
        assert calls >= 2 and not media
        assert len(offered) == 1 and offered[0]['turn_id'] != target
    finally:
        await store.close()
