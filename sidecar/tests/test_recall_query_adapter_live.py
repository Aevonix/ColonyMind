"""Owner serving runs paired local searches while lessons retain the base vector."""
from contextlib import nullcontext
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from colonymind.memory.recall import calibration_fingerprint, provider_calibration_metadata
from colonymind.memory.selection import RecallSelector
from colonymind.plasticity.learners.query_adapter import RecallRequest
from colonymind.turns import source_vectors
from colonymind.vector.indexes import EmbeddingIdentity
from colonymind.vector.query import VectorResult
from test_plasticity_query_adapter_driver import Config, runtime  # noqa: F401


class Pipeline:
    index_identity = EmbeddingIdentity('fixture', 'fixture', '1', 4)

    async def embed_query(self, query):
        return [.6, .8, 0., 0.]


class Store:
    identity = Pipeline.index_identity
    def __init__(self):
        self.catalog = SimpleNamespace(read_generation=lambda identity: {'id': 'generation'})
        self.calls = []

    async def _table(self, *args, **kwargs):
        async def schema():
            return SimpleNamespace(names=['scope_key'])
        return SimpleNamespace(schema=schema)

    @staticmethod
    def _quoted(value):
        return repr(value)

    async def search(self, collection, vector, **kwargs):
        self.calls.append((list(vector), kwargs))
        return [VectorResult('chunk', .9, 'source text', {'source_turn_id': 'source', 'kind': 'source_quote'})]


@pytest.mark.asyncio
async def test_paired_search_is_once_per_owner_recall_and_base_memo_is_untouched(runtime, monkeypatch):
    driver = runtime.loaded['recall-test'].driver
    driver.document_identity = asdict(Pipeline.index_identity.document_identity)
    store, pipeline = Store(), Pipeline()
    monkeypatch.setattr(source_vectors, 'hydrate', lambda *a, **kw: {'turn_id': 'source', 'role': 'user',
                          'scope': 'person', 'content': 'source text'})
    request = RecallRequest.create(runtime, owner=True, turn_id='owner-turn')
    with source_vectors.turn_query_vectors():
        before = await source_vectors.query_vector(pipeline, 'question', 1)
        result = await source_vectors.SourceVectors(None, store, pipeline).search(
            'question', contact_id='owner', session_id='session', recall_request=request)
        again = await source_vectors.SourceVectors(None, store, pipeline).search(
            'question', contact_id='owner', session_id='session', recall_request=request)
        after = await source_vectors.query_vector(pipeline, 'question', 1)
    assert result == again
    assert before == after == [.6, .8, 0., 0.]
    assert len(store.calls) == 2
    records = runtime.loaded['recall-test'].comparator.records()
    assert len(records) == 1 and set(records[0]['output']) == {'hash'}
    assert records[0]['output'] == records[0]['reference_output']
    assert driver.queries.get('owner-turn') is not None
    for audience in ('guest', 'contact'):
        await source_vectors.SourceVectors(None, store, pipeline).search(
            'question', contact_id=audience, session_id='session')
    assert len(store.calls) == 4
    assert len(runtime.loaded['recall-test'].comparator.records()) == 1


@pytest.mark.asyncio
async def test_pinned_cutoff_overrides_environment_and_metadata_is_request_local(runtime, monkeypatch):
    provider = SimpleNamespace(calibration_metadata=lambda: {'model': 'fixture'})
    baseline = provider_calibration_metadata(provider)
    request = RecallRequest.create(runtime, owner=True, turn_id='owner-turn')
    request.transform = 'query-adapter-1:' + 'a' * 64
    request.cutoff = {'cutoff': .8, 'fingerprint': calibration_fingerprint(
        {**baseline, 'query_transform': request.transform})}
    monkeypatch.setenv('COLONYMIND_RECALL_RERANK', 'on')
    monkeypatch.setenv('COLONYMIND_RECALL_RERANK_MIN_SCORE', '.1')
    monkeypatch.setenv('COLONYMIND_RECALL_RERANK_CALIBRATION', calibration_fingerprint(baseline))
    async def rerank(query, documents, top_k):
        return [{'index': i, 'score': .7} for i in range(len(documents))]
    selector = RecallSelector(rerank)
    selector._rerank_calibration_metadata = lambda: provider_calibration_metadata(provider)
    for context, expected in ((request.use(), 0), (nullcontext(), 1)):
        with context:
            rows, _, cut = await selector._reranked('query', [{'id': 'a', 'content': 'text'}], 5)
        assert len(rows) == expected and cut
    assert provider_calibration_metadata(provider) == baseline


@pytest.mark.asyncio
async def test_vector_lane_carries_authenticated_request_without_inference_from_contact(runtime, monkeypatch):
    from colonymind.memory.search import LaneQuery, _vector_lane
    calls = []
    async def search(self, query, **kwargs):
        calls.append(kwargs)
        return [], []
    monkeypatch.setattr(source_vectors.SourceVectors, 'search', search)
    request = RecallRequest.create(runtime, owner=True, turn_id='authenticated-turn')
    await _vector_lane(LaneQuery(None, 'question', 'owner', 'session', recall_request=request))
    await _vector_lane(LaneQuery(None, 'question', 'owner', 'session'))
    assert calls[0]['recall_request'] is request
    assert calls[1].get('recall_request') is None


@pytest.mark.asyncio
async def test_learned_owner_transform_and_previous_comparator_leave_lessons_unchanged(runtime, monkeypatch):
    from test_plasticity_query_adapter_driver import promote
    driver = runtime.loaded['recall-test'].driver
    driver.document_identity = asdict(Pipeline.index_identity.document_identity)
    pinned = promote(runtime)
    store, pipeline = Store(), Pipeline()
    monkeypatch.setattr(source_vectors, 'hydrate', lambda *a, **kw: {'turn_id': 'source', 'role': 'user',
                          'scope': 'person', 'content': 'source text'})
    request = RecallRequest.create(runtime, owner=True, turn_id='transformed-owner')
    with source_vectors.turn_query_vectors():
        await source_vectors.SourceVectors(None, store, pipeline).search(
            'question', contact_id='owner', session_id='session', recall_request=request)
        lesson = await source_vectors.query_vector(pipeline, 'question', 1)
    assert store.calls[0][0] != lesson
    assert store.calls[1][0] == lesson
    assert request.pinned.closure.id == pinned.closure.id
    assert request.cutoff['id'] == pinned.closure.dependents['recall_cutoff']
    assert store.calls[0][1]['filter'] == store.calls[1][1]['filter']
    assert store.catalog.read_generation(store.identity)['id'] == 'generation'


@pytest.mark.asyncio
async def test_rollback_between_async_searches_retries_before_logging(runtime, monkeypatch):
    from test_plasticity_query_adapter_driver import promote
    driver = runtime.loaded['recall-test'].driver
    driver.document_identity = asdict(Pipeline.index_identity.document_identity)
    promote(runtime)
    store, pipeline = Store(), Pipeline()
    original = store.search
    async def search(*args, **kwargs):
        result = await original(*args, **kwargs)
        if len(store.calls) == 1:
            runtime._rollback('recall-test', reason='test concurrent rollback')
        return result
    store.search = search
    monkeypatch.setattr(source_vectors, 'hydrate', lambda *a, **kw: {'turn_id': 'source', 'role': 'user',
                          'scope': 'person', 'content': 'source text'})
    request = RecallRequest.create(runtime, owner=True, turn_id='racing-owner')
    await source_vectors.SourceVectors(None, store, pipeline).search(
        'question', contact_id='owner', session_id='session', recall_request=request)
    assert request.pinned.closure.mode == 'base' and request.cutoff is None
    records = runtime.loaded['recall-test'].comparator.records()
    assert len(records) == 1
    assert records[0]['closure'] == runtime.pin('recall-test', 'owner').closure.id
    assert len(store.calls) == 4


@pytest.mark.asyncio
async def test_provider_drift_never_serves_a_learned_transform_with_an_old_cutoff(runtime, monkeypatch):
    from test_plasticity_query_adapter_driver import promote
    driver = runtime.loaded['recall-test'].driver
    driver.document_identity = asdict(Pipeline.index_identity.document_identity)
    promote(runtime)
    store, pipeline = Store(), Pipeline()
    request = RecallRequest.create(runtime, owner=True, turn_id='configuration-drift')
    request.metadata = lambda: {'model': 'different-reranker'}
    with pytest.raises(ValueError, match='configuration'):
        await source_vectors.SourceVectors(None, store, pipeline).search(
            'question', contact_id='owner', session_id='session', recall_request=request)
    assert store.calls == []
    assert request.cutoff is None and request.transform == 'identity'
    assert not runtime.loaded['recall-test'].comparator.records()


@pytest.mark.asyncio
@pytest.mark.parametrize('enabled,declared', [(False, True), (True, False)])
async def test_faculty_off_and_absent_learner_keep_the_recall_packet_identical(tmp_path, monkeypatch, enabled, declared):
    from colonymind.memory.search import CollectedSources, select_memory
    from colonymind.plasticity.nightly import Runtime
    active = Runtime(Config(tmp_path, enabled=enabled, declared=declared))
    request = RecallRequest.create(active, owner=True, turn_id='turn')
    assert request is None
    store, pipeline = Store(), Pipeline()
    monkeypatch.setattr(source_vectors, 'hydrate', lambda *a, **kw: {
        'turn_id': 'source', 'role': 'user', 'scope': 'person', 'content': 'The cabinet contains green folders.'})
    vector_lane = source_vectors.SourceVectors(None, store, pipeline)
    baseline, _ = await vector_lane.search('folders', contact_id='owner', session_id='session')
    unchanged, _ = await vector_lane.search('folders', contact_id='owner', session_id='session', recall_request=request)
    assert baseline == unchanged
    selector = RecallSelector()
    before = await select_memory(CollectedSources(None, 'owner', 'session', None, hits=baseline),
                                 query='folders', selector=selector, timezone_name='UTC')
    after = await select_memory(CollectedSources(None, 'owner', 'session', None, hits=unchanged),
                                query='folders', selector=selector, timezone_name='UTC')
    assert before.public() == after.public()
    assert before.content.encode() == after.content.encode()
    assert not list(tmp_path.rglob('query-vectors.db'))
    active.close()


@pytest.mark.asyncio
async def test_interrupted_training_erasure_during_search_cannot_serve_dirty_pin(runtime, monkeypatch):
    from colonymind.turns.erasure_hooks import notify_source_erasure
    from test_plasticity_query_adapter_driver import promote
    driver = runtime.loaded['recall-test'].driver
    driver.document_identity = asdict(Pipeline.index_identity.document_identity)
    promote(runtime)
    store, pipeline = Store(), Pipeline()
    original = store.search
    async def search(*args, **kwargs):
        hits = await original(*args, **kwargs)
        if len(store.calls) == 1:
            original_erase = runtime.erase
            def interrupted(refs):
                raise RuntimeError('interrupted erasure')
            runtime.erase = interrupted
            try:
                with pytest.raises(RuntimeError):
                    notify_source_erasure(['source-a'])
            finally:
                runtime.erase = original_erase
        return hits
    store.search = search
    request = RecallRequest.create(runtime, owner=True, turn_id='during-erasure')
    with pytest.raises(ValueError, match='erasure'):
        await source_vectors.SourceVectors(None, store, pipeline).search(
            'question', contact_id='owner', session_id='session', recall_request=request)
    assert not runtime.loaded['recall-test'].comparator.records()
