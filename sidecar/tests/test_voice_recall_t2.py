"""T2 voice recall fixes and qualification."""
import asyncio
import importlib.util
import json
from pathlib import Path
import threading
from types import SimpleNamespace

from httpx import ASGITransport, AsyncClient
import pytest

from onekey import KEY
from colonymind.memory import voice_recall
from colonymind.memory.live_recall import LIVE
from colonymind.memory.recall import pack_memory_tokens, pack_memory_context, source_candidates
from test_canonical_memory_search import memory_app  # noqa: F401
from test_turn_source_evidence import source_app  # noqa: F401
from test_voice_recall import ROUTE, body, call
from test_voice_recall_speculation import attach, settled


@pytest.mark.parametrize('stage', ['embed', 'search'])
async def test_speculation_timeout_releases_live_and_worker_slot(memory_app, tmp_path, monkeypatch, stage):
    app, _ = memory_app
    _, store, pipeline, _ = await attach(tmp_path, monkeypatch)
    monkeypatch.setattr(voice_recall, 'SPECULATIVE_TIMEOUT_SECONDS', .02)
    entered, exited = asyncio.Event(), asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            exited.set()

    monkeypatch.setattr(pipeline if stage == 'embed' else voice_recall,
                        'embed_query' if stage == 'embed' else 'speculative_hits', blocked)
    cache = voice_recall.VoiceEmbeddings(workers=1)
    app.state.voice_requests = voice_recall.VoiceRequests()
    app.state.voice_requests.embeddings = cache
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test',
                           headers={'Authorization': 'Bearer ' + KEY}) as client:
        await call(client, query='Where is my hydrofoil', speculative=True)
        await entered.wait()
        await asyncio.wait_for(settled(cache), .5)
        assert exited.is_set() and not LIVE.active()
        assert not cache.entries and not cache.jobs and not cache.tasks
        assert not cache.slots.locked()
        packet = await call(client, query='Where is my hydrofoil departure?')
        assert packet['retrieval']['voice'] == 'local_keywords'
        assert 'Friday at nine' in packet['content']


@pytest.mark.parametrize('stage', ['embed', 'search'])
async def test_supersede_cancels_and_bounds_embeds_per_utterance(tmp_path, monkeypatch, stage):
    _, store, pipeline, _ = await attach(tmp_path, monkeypatch)
    cache = voice_recall.VoiceEmbeddings(workers=8)
    key = ('principal', 'credential', 'person', 'session', 'utterance')
    entered = asyncio.Queue()
    calls, cancelled, active, maximum = [], [], 0, 0
    original_embed = pipeline.embed_query

    async def tracked(query):
        calls.append(query)
        if stage == 'search':
            return await original_embed(query)
        return await blocked(query)

    async def blocked(*args, **kwargs):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        entered.put_nowait(args)
        try:
            await asyncio.Event().wait()
        finally:
            active -= 1
            cancelled.append(args)

    monkeypatch.setattr(pipeline, 'embed_query', tracked)
    if stage == 'search':
        monkeypatch.setattr(voice_recall, 'speculative_hits', blocked)
    for index in range(voice_recall.MAX_EMBEDS_PER_UTTERANCE):
        cache.start(key, f'where is my workplace {index}', pipeline, store)
        await asyncio.wait_for(entered.get(), .5)
    for index in range(100):
        cache.start(key, f'where is my workplace latest {index}', pipeline, store)
    await settled(cache)
    assert len(calls) == voice_recall.MAX_EMBEDS_PER_UTTERANCE
    assert len(cancelled) == len(calls) and maximum == 1
    assert not LIVE.active() and not cache.jobs and not cache.entries and not cache.tasks


async def test_rapid_replacements_use_latest_words_and_do_not_overlap(tmp_path, monkeypatch):
    _, store, pipeline, _ = await attach(tmp_path, monkeypatch)
    cache = voice_recall.VoiceEmbeddings()
    key = ('p', 'c', 'person', '', 'u')
    entered = asyncio.Event()
    calls = []
    async def embed(query):
        calls.append(query)
        if query.endswith('old'):
            entered.set()
            await asyncio.Event().wait()
        return pipeline.vector(query)
    monkeypatch.setattr(pipeline, 'embed_query', embed)
    cache.start(key, 'where is my workplace old', pipeline, store)
    await entered.wait()
    for text in ('middle', 'newer', 'latest'):
        cache.start(key, 'where is my workplace ' + text, pipeline, store)
    await asyncio.gather(*cache.jobs, return_exceptions=True)
    await asyncio.sleep(0)
    assert calls == ['where is my workplace old', 'where is my workplace latest']
    assert cache.take(key, calls[-1]) is not None


@pytest.mark.parametrize('cause', ['timeout', 'supersede'])
async def test_delayed_provider_cancellation_leaves_live_but_retains_slot(tmp_path, monkeypatch, cause):
    _, store, pipeline, _ = await attach(tmp_path, monkeypatch)
    deadline = asyncio.Event()

    async def until_deadline(awaitable, timeout):
        expired = asyncio.create_task(timeout.wait())
        try:
            done, _ = await asyncio.wait((awaitable, expired), return_when=asyncio.FIRST_COMPLETED)
            if awaitable in done:
                return await awaitable
            raise TimeoutError
        finally:
            awaitable.cancel()
            expired.cancel()
            await asyncio.gather(awaitable, expired, return_exceptions=True)

    # Exercise the real cancellation/cleanup path with an explicit deadline.
    # The replacement's real Lance search must not race an unrelated 20 ms timer.
    monkeypatch.setattr(voice_recall, 'SPECULATIVE_TIMEOUT_SECONDS', deadline)
    monkeypatch.setattr(voice_recall, 'asyncio', SimpleNamespace(**(vars(asyncio) | {'wait_for': until_deadline})))
    cache = voice_recall.VoiceEmbeddings(workers=1)
    key = ('p', 'c', 'person', '', 'u')
    entered, cancellation_seen, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []
    async def embed(query):
        calls.append(query)
        if query.endswith('old'):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_seen.set()
                await release.wait()
        return pipeline.vector(query)
    monkeypatch.setattr(pipeline, 'embed_query', embed)
    cache.start(key, 'where is my workplace old', pipeline, store)
    await entered.wait()
    if cause == 'supersede':
        cache.start(key, 'where is my workplace latest', pipeline, store)
    else:
        deadline.set()
    try:
        await asyncio.wait_for(cancellation_seen.wait(), .5)
        assert not LIVE.active() and cache.slots.locked()
        deadline.clear()
        if cause == 'timeout':
            cache.start(key, 'where is my workplace latest', pipeline, store)
        await asyncio.sleep(0)
        assert len(calls) == 1  # replacement must wait for provider cleanup
    finally:
        release.set()
        await asyncio.gather(*cache.jobs, return_exceptions=True)
        await asyncio.sleep(0)
    assert calls == ['where is my workplace old', 'where is my workplace latest']
    assert not cache.slots.locked() and not LIVE.active()
    assert cache.take(key, calls[-1]) is not None


@pytest.mark.parametrize('elapsed,ready', [(.024, True), (.025, False), (.026, False)])
def test_keyword_ready_deadline_preserves_packet_and_skips_extra_reads(memory_app, monkeypatch, elapsed, ready):
    _, ledger = memory_app
    clock = [100.]
    monkeypatch.setattr(voice_recall, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    keyword = voice_recall.read_voice_packet(ledger, query='hydrofoil', contact_id='person',
        session_id='later', k=5, max_tokens=600, timezone_name='UTC', cancelled=threading.Event())
    original_select = voice_recall.select_memory
    reads = []
    async def selection(*args, **kwargs):
        packet = await original_select(*args, **kwargs)
        clock[0] += elapsed
        return packet
    def hydration(*args, **kwargs):
        reads.append(True)
        return [], []
    monkeypatch.setattr(voice_recall, 'select_memory', selection)
    monkeypatch.setattr(voice_recall, 'hydrate_hits', hydration)
    result = voice_recall.read_voice_packet(ledger, query='hydrofoil', contact_id='person',
        session_id='later', k=5, max_tokens=600, timezone_name='UTC',
        cancelled=threading.Event(), embedding=object(), started_at=100.)
    assert bool(reads) is ready
    assert result.content == keyword.content and result.source_refs == keyword.source_refs
    assert result.retrieval['voice'] == ('local_hybrid' if ready else 'local_keywords')


async def test_hydration_uses_one_connection_and_fill_searches_lexical_media_once(memory_app, tmp_path, monkeypatch):
    from colonymind.turns.media import SourceMedia
    _, ledger = memory_app
    ledger.record_source('office', contact_id='person', session_id='earlier',
        messages=[{'role': 'user', 'content': 'The office keypad number is birch-6.'}], derive_claims=False)
    _, store, pipeline, _ = await attach(tmp_path, monkeypatch)
    cache = voice_recall.VoiceEmbeddings()
    key = ('p', 'c', 'person', 'later', 'u')
    cache.start(key, 'where is my workplace', pipeline, store)
    await settled(cache)
    entry = cache.take(key, 'where is my workplace keypad number')
    assert entry is not None
    connections = []
    original_connect = ledger._connect
    def connect():
        connections.append(True)
        return original_connect()
    monkeypatch.setattr(ledger, '_connect', connect)
    hits, _ = voice_recall.hydrate_hits(ledger, entry, contact_id='person', session_id='later', limit=5)
    assert hits and len(connections) == 1
    searches = []
    original_search = SourceMedia.search
    def search(*args, **kwargs):
        searches.append(True)
        return original_search(*args, **kwargs)
    monkeypatch.setattr(SourceMedia, 'search', search)
    monkeypatch.setattr(voice_recall, 'time', SimpleNamespace(monotonic=lambda: 100.))
    packet = await asyncio.to_thread(voice_recall.read_voice_packet, ledger,
        query='where is my workplace keypad number', contact_id='person', session_id='later',
        k=5, max_tokens=600, timezone_name='UTC', cancelled=threading.Event(), embedding=entry)
    assert 'birch-6' in packet.content and len(searches) == 1


@pytest.mark.parametrize('budget', [300, 600])
def test_packet_local_provenance_cache_preserves_exact_packing(memory_app, budget):
    _, ledger = memory_app
    ledger.record_source('packing-pair', contact_id='person', session_id='earlier', messages=[
        {'role': 'user', 'content': 'Which workspace repair?'},
        {'role': 'assistant', 'content': 'The workspace cart needs new wheels. ' * 80}])
    from colonymind.beliefs.source_projection import SourceClaimProjection
    from colonymind.beliefs.source_time import interpret_time_query
    from colonymind.util.temporal import now_utc
    hits = ledger.search_sources('workspace repair', contact_id='person', session_id='later', limit=10)
    _, rows = SourceClaimProjection(ledger).prepare_context([], hits, contact_id='person',
        session_id='later', time_query=interpret_time_query('workspace repair', now=now_utc()),
        classify_work_replies=False, include_conversation_inputs=True)
    rows += source_candidates(hits)
    assert rows
    rendering = voice_recall._VoiceRendering()
    before = json.dumps(rows, sort_keys=True, default=str)
    assert pack_memory_context(rows, limit=5, max_chars=4000) == pack_memory_context(
        rows, limit=5, max_chars=4000, renderer=rendering.rich)
    assert pack_memory_tokens(rows, limit=5, max_tokens=budget) == rendering.pack(
        rows, limit=5, max_tokens=budget)
    assert json.dumps(rows, sort_keys=True, default=str) == before


def test_provenance_marker_is_preserved_and_cache_expires_between_packets(monkeypatch):
    from colonymind.hands import provenance
    from test_voice_recall import rows
    memories = rows('The completed repair is ready.')[:1]
    memories[0].update(source_turn_id='hand-result', source_message_hash='message', role='assistant')
    marked = [False]
    def marker(row):
        return 'from the repair hand, E3, checked' if marked[0] and row.get('contact_id') else ''
    monkeypatch.setattr(provenance, 'recalled_hand_marker', marker)
    rendering = voice_recall._VoiceRendering()
    assert rendering.voice(memories) == voice_recall.render_voice_context(memories)
    marked[0] = True
    fresh = voice_recall._VoiceRendering()
    assert fresh.voice(memories) == voice_recall.render_voice_context(memories)
    assert 'from the repair hand' in fresh.voice(memories)


def measurement_module():
    script = Path(__file__).resolve().parents[2] / 'benchmarks/source_recall/voice_speculation_measurement.py'
    spec = importlib.util.spec_from_file_location('voice_speculation_measurement_t2', script)
    measurement = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(measurement)
    return measurement


@pytest.mark.parametrize('variant', ['q', 'q75'])
async def test_recorded_query_variant_cannot_be_overwritten_by_text_collision(variant):
    cases = [{'id': 'first', 'query': 'where is my office'},
             {'id': 'second', 'query': 'where is my office today now'}]
    corpus = {'model': 'recorded', 'dims': 3, 'single_query_ms': 0,
              'vectors': {'q:first': [1., 0., 0.], 'q75:first': [0., 0., 1.],
                          'q:second': [1., 1., 0.], 'q75:second': [0., 1., 0.]}}
    pipeline = measurement_module().RecordedPipeline(corpus, cases)
    pipeline.query_key = ('first', variant)
    vector = await pipeline.embed_query(pipeline.queries[pipeline.query_key])
    assert vector == corpus['vectors'][variant + ':first']
    assert pipeline.used[-1]['vector_key'] == variant + ':first'


async def test_real_embedding_t2_gate(memory_app, tmp_path, monkeypatch):
    measurement = measurement_module()
    await measurement.measure(memory_app, tmp_path, monkeypatch)
