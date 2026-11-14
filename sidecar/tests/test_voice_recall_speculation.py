"""Speculative query vectors: lifecycle, canonical privacy and offline real gate."""
import asyncio
from dataclasses import replace

from httpx import ASGITransport, AsyncClient
import pytest

from onekey import KEY
from colonymind.memory.recall import estimate_voice_tokens
from colonymind.memory.voice_recall import VoiceEmbeddings, VoiceRequests, close_extension, _words
from test_canonical_memory_search import memory_app, annotate  # noqa: F401
from test_turn_source_evidence import source_app  # noqa: F401
from test_source_vectors import Pipeline, setup, drain
from test_voice_recall import ROUTE, body, call


@pytest.mark.parametrize('partial,final,expected', [
    ('Why does our car engine quit', 'Why does our car engine quit during cold weather?', True),
    ('WHERE is my workplace', 'Where is my workplace?', True),
    ('office', 'office', True),
    ('where is', 'where is my workplace', False),
    ('Where is my workplace', 'Where is my new workplace', False),
    ('Where is my workplace', 'Where is my workplace after I leave the train station', False),
    ('Where is my workplace', 'Where is my workplace actually', False),
    ('Where is my workplace', 'Where is my workplace not', False),
    ('Can I rent this flat', "Can I rent this flat instead", False),
    ('', 'where is my workplace', False),
])
def test_close_extension_rule(partial, final, expected):
    assert close_extension(_words(partial), final) is expected


async def attach(tmp_path, monkeypatch):
    from colonymind import vector
    ledger, store, pipeline, projection = await setup(tmp_path)
    await drain(projection)
    monkeypatch.setattr(vector, 'get_store', lambda: store)
    monkeypatch.setattr(vector, 'get_pipeline', lambda: pipeline)
    # Warm Lance schema/imports outside checks of the final path's budget.
    await projection.search('workplace', contact_id='person', session_id='later')
    return ledger, store, pipeline, projection


async def settled(cache):
    await asyncio.gather(*cache.jobs)
    await asyncio.sleep(0)  # done callbacks release job accounting


async def test_cached_vector_finds_paraphrase_without_final_embedding(memory_app, tmp_path, monkeypatch):
    app, ledger = memory_app
    ledger.record_source('office', contact_id='person', session_id='earlier',
        messages=[{'role': 'user', 'content': 'The office entrance code is birch-6.'}], derive_claims=False)
    _, _, pipeline, _ = await attach(tmp_path, monkeypatch)
    queries = []
    original = pipeline.embed_query
    async def tracked(query):
        queries.append(query)
        return await original(query)
    pipeline.embed_query = tracked
    from colonymind.memory.selection import RecallSelector
    async def forbidden(*args, **kwargs):
        raise AssertionError('voice must never rerank')
    monkeypatch.setattr(RecallSelector, '_reranked', forbidden)
    monkeypatch.setenv('COLONYMIND_RECALL_RERANK', 'on')
    monkeypatch.setenv('COLONYMIND_RECALL_RERANK_MIN_SCORE', '.9')
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test',
                           headers={'Authorization': 'Bearer ' + KEY}) as client:
        await call(client, query='Where is my workplace', speculative=True)
        await settled(app.state.voice_requests.embeddings)
        packet = await call(client, query='Where is my workplace entrance code?')
        assert packet['retrieval'] == {'semantic': 'ready', 'voice': 'local_hybrid'}
        assert 'birch-6' in packet['content']
        assert queries == ['Where is my workplace']
        assert estimate_voice_tokens(packet['content']) <= 600
        assert not app.state.voice_requests.embeddings.entries


async def test_late_embed_never_blocks_final_or_holds_read_slots(memory_app, tmp_path, monkeypatch):
    app, _ = memory_app
    _, _, pipeline, _ = await attach(tmp_path, monkeypatch)
    started, release = asyncio.Event(), asyncio.Event()
    async def blocked(query):
        started.set()
        await release.wait()
        return pipeline.vector(query)
    pipeline.embed_query = blocked
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test',
                           headers={'Authorization': 'Bearer ' + KEY}) as client:
        await call(client, query='Where is my hydrofoil', speculative=True)
        await started.wait()
        # Simulate an embed that exceeds the entire 50 ms voice budget.
        await asyncio.sleep(.075)
        packet = await asyncio.wait_for(call(client, query='Where is my hydrofoil departure?'), .05)
        assert packet['retrieval']['voice'] == 'local_keywords'
        assert 'Friday at nine' in packet['content']
        cache = app.state.voice_requests.embeddings
        assert not cache.entries and len(cache.jobs) == 1
        assert not app.state.voice_requests.jobs
        release.set()
        await settled(cache)
        assert not cache.entries and not cache.jobs


@pytest.mark.parametrize('component', range(5))
async def test_cache_key_isolates_every_authority_and_utterance_component(tmp_path, monkeypatch, component):
    _, store, pipeline, _ = await attach(tmp_path, monkeypatch)
    cache = VoiceEmbeddings()
    key = ('principal', 'credential', 'person', 'session', 'utterance')
    cache.start(key, 'where is my workplace', pipeline, store)
    await settled(cache)
    other = list(key); other[component] = 'other'
    assert cache.take(tuple(other), 'where is my workplace') is None
    assert cache.take(key, 'where is my workplace') is not None


@pytest.mark.parametrize('revision', ['text', 'pipeline', 'store', 'identity', 'query_format', 'query_transform', 'expired'])
async def test_revisions_and_expiry_discard_cached_vector(tmp_path, monkeypatch, revision):
    from colonymind import vector
    _, store, pipeline, _ = await attach(tmp_path, monkeypatch)
    cache = VoiceEmbeddings()
    key = ('p', 'c', 'person', '', 'u')
    cache.start(key, 'where is my workplace', pipeline, store)
    await settled(cache)
    query = 'where is my workplace'
    if revision == 'text': query = 'where is my bicycle'
    elif revision == 'pipeline': monkeypatch.setattr(vector, 'get_pipeline', lambda: Pipeline())
    elif revision == 'store': monkeypatch.setattr(vector, 'get_store', lambda: object())
    elif revision == 'identity': pipeline.index_identity = Pipeline('different').index_identity
    elif revision == 'query_format': pipeline.index_identity = replace(pipeline.index_identity, query_format='changed')
    elif revision == 'query_transform':
        pipeline.index_identity = replace(pipeline.index_identity, query_transform='query-adapter-1:' + 'a' * 64)
    else: cache.entries[key].created_at -= 121
    assert cache.take(key, query) is None
    assert not cache.entries


async def test_pending_embeddings_are_bounded_and_superseded_results_cannot_publish(tmp_path, monkeypatch):
    _, store, pipeline, _ = await attach(tmp_path, monkeypatch)
    release = asyncio.Event()
    started = asyncio.Queue()
    async def blocked(query):
        started.put_nowait(query)
        await release.wait()
        return pipeline.vector(query)
    pipeline.embed_query = blocked
    cache = VoiceEmbeddings(capacity=2, workers=1)
    first = ('p', 'c', 'person', '', 'u')
    second = (*first[:-1], 'v')
    cache.start(first, 'where is my workplace', pipeline, store)
    await started.get()
    cache.start(first, 'where is my circular room', pipeline, store)
    cache.start(second, 'where is my workplace', pipeline, store)
    assert len(cache.jobs) == 2 and second not in cache.entries
    assert cache.take(first, 'where is my circular room') is None
    release.set()
    await settled(cache)
    assert not cache.entries and not cache.jobs


async def test_completed_partial_embedding_deduplicates_and_failure_keeps_keywords(tmp_path, monkeypatch):
    _, store, pipeline, _ = await attach(tmp_path, monkeypatch)
    key = ('p', 'c', 'person', '', 'u')
    cache = VoiceEmbeddings()
    calls = []
    async def tracked(query):
        calls.append(query)
        if 'broken' in query: raise RuntimeError('offline embedder unavailable')
        return pipeline.vector(query)
    pipeline.embed_query = tracked
    cache.start(key, 'where is my workplace', pipeline, store)
    await settled(cache)
    cache.start(key, 'where is my workplace', pipeline, store)
    await settled(cache)
    assert len(calls) == 1 and cache.take(key, 'where is my workplace')
    cache.start(key, 'where is my broken workplace', pipeline, store)
    await settled(cache)
    assert cache.take(key, 'where is my broken workplace') is None


@pytest.mark.parametrize('action', ['scope', 'annotation', 'observed'])
async def test_vector_path_keeps_canonical_scope_and_current_evidence(memory_app, tmp_path, monkeypatch, action):
    app, ledger = memory_app
    if action == 'observed':
        from colonymind.turns import observed
        from test_observed_conversations import body as observed_body
        data = observed_body(('speaker_1', 'The office is beside the orchard.'),
                             summary=None, action_items=[])
        clocks = {name: data[name] for name in ('captured_at', 'ended_at')}
        observed.record(ledger, data, 'voice-observed', clocks, owner_id='person')
    _, store, pipeline, projection = await attach(tmp_path, monkeypatch)
    query = 'Where is my workplace' if action == 'observed' else 'Where is my vessel'
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test',
                           headers={'Authorization': 'Bearer ' + KEY}) as client:
        await call(client, query=query, speculative=True)
        await settled(app.state.voice_requests.embeddings)
        if action == 'annotation':
            annotate(ledger, 'vector-check', 'Friday is confirmed; nine is unsupported.')
        pass
        packet = await call(client, query=query)
        assert packet['retrieval']['semantic'] == 'ready'
        assert 'amber' not in packet['content'] and 'cobalt' not in packet['content']
        if action == 'annotation':
            assert 'nine is unsupported' in packet['content'] and packet['annotation_checks']
        elif action == 'erase': assert 'Friday at nine' not in packet['content']
        elif action == 'observed':
            assert 'observed_conversation' in packet['content']
            assert 'derived_unverified' in packet['content']
            assert 'not said to the agent' in packet['content']
            assert 'speaker_1' in packet['content']
        elif action == 'scope':
            await call(client, query=query, speculative=True, utterance_id='u-session', session_id='original')
            await settled(app.state.voice_requests.embeddings)
            packet = await call(client, query=query, utterance_id='u-session', session_id='original')
            assert 'cobalt' in packet['content'] and 'amber' not in packet['content']


async def test_real_embedding_voice_gate(memory_app, tmp_path, monkeypatch):
    """Optional qualification reads REAL vectors from an external file only."""
    from voice_recall_measurement import measure
    await measure(memory_app, tmp_path, monkeypatch)


async def test_final_call_only_hydrates_hits_searched_during_speech(memory_app, tmp_path, monkeypatch):
    app, ledger = memory_app
    ledger.record_source('office', contact_id='person', session_id='earlier',
        messages=[{'role': 'user', 'content': 'The office entrance code is birch-6.'}], derive_claims=False)
    _, store, _, _ = await attach(tmp_path, monkeypatch)
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test',
                           headers={'Authorization': 'Bearer ' + KEY}) as client:
        await call(client, query='Where is my workplace', speculative=True)
        await settled(app.state.voice_requests.embeddings)
        async def forbidden(*args, **kwargs):
            raise AssertionError('the final call must not search the vector index')
        monkeypatch.setattr(store, 'search', forbidden)
        packet = await call(client, query='Where is my workplace entrance code?')
        assert packet['retrieval'] == {'semantic': 'ready', 'voice': 'local_hybrid'}
        assert 'birch-6' in packet['content']


@pytest.mark.parametrize('k', [3, 5])
async def test_vector_fill_never_displaces_keyword_records(memory_app, tmp_path, monkeypatch, k):
    app, ledger = memory_app
    ledger.record_source('office', contact_id='person', session_id='earlier',
        messages=[{'role': 'user', 'content': 'The office keypad number is birch-6.'}], derive_claims=False)
    for index, detail in enumerate(('a revolving door', 'a ramp', 'a bike rack')):
        ledger.record_source(f'lobby-{index}', contact_id='person', session_id='earlier',
            messages=[{'role': 'user', 'content': f'My workplace entrance code panel sits by {detail}.'}],
            derive_claims=False)
    await attach(tmp_path, monkeypatch)
    query = 'Where is my workplace entrance code?'
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test',
                           headers={'Authorization': 'Bearer ' + KEY}) as client:
        keyword = await call(client, query=query, k=k, utterance_id='keyword-only')
        await call(client, query='Where is my workplace entrance', k=k, speculative=True, utterance_id='hybrid')
        await settled(app.state.voice_requests.embeddings)
        hybrid = await call(client, query=query, k=k, utterance_id='hybrid')
    assert keyword['retrieval']['voice'] == 'local_keywords' and hybrid['retrieval']['semantic'] == 'ready'
    assert keyword['count'] == 3 and 'birch-6' not in keyword['content']  # k=3 full; k=5 has room
    keyword_ids = [ref['source_id'] for ref in keyword['source_refs']]
    assert keyword_ids and set(keyword_ids) <= {ref['source_id'] for ref in hybrid['source_refs']}
    # Keyword records keep their exact rendering and lead the packet.
    body_lines = keyword['content'].splitlines()[2:]
    assert hybrid['content'].splitlines()[2:2 + len(body_lines)] == body_lines
    if keyword['count'] < k:
        assert 'birch-6' in hybrid['content'] and hybrid['count'] > keyword['count']
    else:
        assert hybrid['content'] == keyword['content']
