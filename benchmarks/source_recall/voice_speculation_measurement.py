"""Optional offline qualification against an external, recorded embedding corpus."""
import asyncio
from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import threading

from httpx import ASGITransport, AsyncClient
import pytest

from onekey import KEY
from colonymind.memory.recall import estimate_voice_tokens
from colonymind.memory.voice_recall import VoiceRequests
from colonymind.turns.source_vectors import SourceVectors, chunks
from colonymind.vector.collections import Collection
from colonymind.vector.indexes import EmbeddingIdentity, IndexCatalog
from colonymind.vector.store import VectorStore
from test_voice_recall import ROUTE, body
from test_voice_recall_gate import CASES, NOW, ROOT, seed


class RecordedPipeline:
    """Replay real recorded embeddings; no model, network or synthetic vectors."""
    def __init__(self, corpus, cases):
        self.index_identity = EmbeddingIdentity(corpus['model'], corpus['model'], 'recorded-offline',
                                                corpus['dims'], normalization='l2')
        self.vectors = corpus['vectors']
        self.queries = {}
        self.used = []
        self.query_key = None
        self.calls = 0
        self.delay = corpus['single_query_ms'] / 1000
        self.started, self.release = asyncio.Event(), None
        for case in cases:
            words = case['query'].split()
            for variant, text in [('q', case['query']), ('q75', ' '.join(words[:int(.75 * len(words))]))]:
                self.queries[(case['id'], variant)] = text

    async def embed_query(self, query):
        key = self.query_key
        assert query == self.queries[key]
        vector_key = f'{key[1]}:{key[0]}'
        vector = self.vectors[vector_key]
        self.used.append({'text': query, 'vector_key': vector_key,
                         'sha256': hashlib.sha256(json.dumps(vector).encode()).hexdigest()})
        self.calls += 1
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        else:
            await asyncio.sleep(self.delay)
        return vector


async def recorded_index(ledger, store, corpus):
    """Bulk-load canonical chunk metadata into a fresh temporary Lance index.

    All benchmark/semantic vectors are real. The supplied single distractor
    prototype is repeated for the 10,000 background messages; it is not 10,000
    separately embedded conversations. This limitation is in the report.
    """
    projection = SourceVectors(ledger, store, None)
    table = await projection._table(store.catalog.write_generation(store.identity))
    rows, count = [], 0
    vectors = corpus['vectors']
    with closing(ledger._connect()) as conn:
        for source in conn.execute('SELECT * FROM turn_sources ORDER BY turn_id'):
            for text, meta in chunks(conn, source):
                turn = source['turn_id']
                vector = vectors.get('rec:' + turn, vectors.get('krec:' + turn, vectors['distractor']))
                store._validate_vector(vector)
                meta['embedding_fingerprint'] = store.identity.fingerprint
                rows.append({'id': meta['projection_id'], 'text': text, 'vector': vector,
                    'metadata': json.dumps(meta), 'scope_key': meta['scope_key'], 'modality': 'text',
                    'image_hash': '', 'image_ref': '', 'thumbnail_ref': '', 'caption': '',
                    'created_at': 0., 'updated_at': 0.})
                count += 1
                if len(rows) == 256:
                    await table.add(rows)
                    rows.clear()
        if rows: await table.add(rows)
    return count


def latency(values):
    ordered = sorted(values)
    return {'samples': len(values), 'p50_ms': statistics.median(values),
            'p95_ms': ordered[math.ceil(.95 * len(values)) - 1], 'max_ms': max(values)}


async def measure(memory_app, tmp_path, monkeypatch):
    path = os.environ.get('VOICE_RECALL_EMBEDDINGS')
    if not path or not Path(path).is_file():
        pytest.skip('set VOICE_RECALL_EMBEDDINGS to the external real embedding JSON')
    corpus = json.loads(Path(path).read_text())
    assert corpus['model'] == 'Qwen/Qwen3-Embedding-8B' and corpus['dims'] == 4096
    from colonymind import vector
    from colonymind.turns import get_turn_idempotency_ledger
    from colonymind import get_state_dir
    from colonymind.util import temporal
    app, ledger = memory_app
    monkeypatch.setattr(temporal, 'now_utc', lambda: NOW)
    monkeypatch.delenv('COLONYMIND_CONTEXT_CHARS_PER_TOKEN', raising=False)
    monkeypatch.setenv('COLONYMIND_RECALL_CONTEXT_MAX_CHARS', '4000')
    monkeypatch.setenv('COLONYMIND_RECALL_RERANK', 'off')
    sources, messages = seed(ledger)
    semantic = json.loads((ROOT / 'sidecar/tests/fixtures/voice_recall_semantic_cases.json').read_text())
    for record in semantic['records']:
        ledger.record_source(record['id'], contact_id='person', session_id='semantic-fixture',
            occurred_at='2026-09-05T12:00:00+00:00',
            messages=[{'role': 'user', 'content': record['content']}], derive_claims=False)
    keywords = json.loads(CASES.read_text())['queries']
    pipeline = RecordedPipeline(corpus, semantic['queries'] + keywords)
    store = VectorStore(str(tmp_path / 'real-vector-index'), identity=pipeline.index_identity,
                        catalog=IndexCatalog(ledger))
    await store.connect(corpus['dims']); await store.ensure_collections(corpus['dims'])
    indexed = await recorded_index(ledger, store, corpus)
    monkeypatch.setattr(vector, 'get_store', lambda: store)
    monkeypatch.setattr(vector, 'get_pipeline', lambda: pipeline)
    searched_vectors = []
    actual_search = store.search
    async def audited_search(collection, query_vector, *args, **kwargs):
        receipt = pipeline.used[-1]
        assert hashlib.sha256(json.dumps(query_vector).encode()).hexdigest() == receipt['sha256']
        searched_vectors.append(receipt)
        return await actual_search(collection, query_vector, *args, **kwargs)
    monkeypatch.setattr(store, 'search', audited_search)
    results, timings = [], {name: [] for name in (
        'keyword_only', 'q75_ready', 'full_ready', 'late', 'concurrent_embed', 'concurrent_search')}
    output = Path(os.environ.get('VOICE_RECALL_VECTOR_REPORT', str(tmp_path / 'voice-speculation.json')))
    counts = {'q75_ready': 0, 'full_ready': 0}
    sequence = 0
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test',
                           headers={'Authorization': 'Bearer ' + KEY}) as client:
        async def invoke(case, k, budget, variant, *, timed=True):
            nonlocal sequence
            sequence += 1
            utterance = f'measure-{sequence}'
            request = body(query=case['query'], k=k, max_tokens=budget, utterance_id=utterance)
            raw_hit_ids = []
            if variant != 'keyword_only':
                pipeline.query_key = (case['id'], 'q' if variant == 'full_ready' else 'q75')
                text = case['query'] if variant == 'full_ready' else ' '.join(
                    case['query'].split()[:int(.75 * len(case['query'].split()))])
                response = await client.post(ROUTE, json={**request, 'query': text,
                    'speculative': True, 'request_id': utterance + '-partial'})
                assert response.status_code == 200, response.text
                cache = app.state.voice_requests.embeddings
                if variant == 'late':
                    await pipeline.started.wait()
                    # It is still pending 75 ms later: beyond the 50 ms voice budget.
                    await asyncio.sleep(.075)
                else:
                    await asyncio.gather(*cache.jobs)
                    await asyncio.sleep(0)
                    ready_entry = next(iter(cache.entries.values()), None)
                    if ready_entry is not None and ready_entry.hits is not None:
                        raw_hit_ids = [str(hit_id) for hit_id, _ in ready_entry.hits]
            calls = pipeline.calls
            start = time.perf_counter_ns()
            response = await client.post(ROUTE, json={**request, 'request_id': utterance + '-final'})
            elapsed = (time.perf_counter_ns() - start) / 1e6
            assert response.status_code == 200, response.text
            packet = response.json()
            if variant == 'late':
                assert packet['retrieval'] == {'semantic': 'unavailable', 'voice': 'local_keywords'}
                assert any(not job.done() for job in cache.jobs), 'late simulation already finished'
            assert pipeline.calls == calls, 'final call started an embedding'
            assert packet['count'] <= k and estimate_voice_tokens(packet['content']) <= budget
            hits = {ref['source_id'] for ref in packet['source_refs']}
            score = len(hits & set(case['expected'])) / len(case['expected'])
            if timed and k == 5 and budget == 600:
                timings[variant].append(elapsed)
                if variant in counts: counts[variant] += packet['retrieval']['semantic'] == 'ready'
            return {'id': case['id'], 'recall': score, 'hits': sorted(hits),
                    'retrieval': packet['retrieval'], 'elapsed_ms': elapsed,
                    'speculative_hit_ids': raw_hit_ids,
                    'vector_used': pipeline.used[-1] if variant != 'keyword_only' else None}

        # Warm ASGI, canonical projection and Lance pages before timed samples.
        for case in semantic['queries'][:5]:
            await invoke(case, 5, 600, 'full_ready', timed=False)
        for k in (3, 4, 5):
            for budget in (300, 600):
                # Independent qualification sessions avoid filling retained final markers.
                app.state.voice_requests = VoiceRequests()
                row = {'k': k, 'max_tokens': budget}
                for name, cases in [('semantic', semantic['queries']), ('keyword', keywords)]:
                    variants = {}
                    for variant in ('keyword_only', 'q75_ready', 'full_ready'):
                        variants[variant] = [await invoke(case, k, budget, variant) for case in cases]
                    row[name] = {variant: {'recall': statistics.mean(q['recall'] for q in queries),
                        'hits': sum(q['recall'] > 0 for q in queries), 'queries': len(queries),
                        'per_query': queries} for variant, queries in variants.items()}
                results.append(row)
                output.write_text(json.dumps({'complete': False, 'quality': results}, indent=2) + '\n')
                print(f'VOICE_SPECULATION_PROGRESS k={k} budget={budget}', flush=True)

        app.state.voice_requests = VoiceRequests()
        late_rows = []
        for case in semantic['queries'] + keywords:
            pipeline.release = asyncio.Event()
            pipeline.started.clear()
            late_rows.append(await invoke(case, 5, 600, 'late'))
            assert not app.state.voice_requests.embeddings.entries
            pipeline.release.set()
            await asyncio.gather(*app.state.voice_requests.embeddings.jobs)
            await asyncio.sleep(0)
        pipeline.release = None

        # Pause an actual previous-partial SQLite statement via its progress
        # callback; run final reads on their own connection while it is in flight.
        # The embed lane uses the recorded 70.1 ms delay. The search lane runs
        # real Lance search concurrently, with no sleep or ready-hit substitution.
        route_ledger = get_turn_idempotency_ledger(get_state_dir())
        original_connect, original_keyword, original_search = route_ledger._connect, route_ledger.search_sources, store.search
        local = threading.local()
        keyword_started, keyword_release = threading.Event(), threading.Event()
        search_started = asyncio.Event()

        def concurrent_connect():
            conn = original_connect()
            if getattr(local, 'partial', False):
                def progress():
                    keyword_started.set()
                    if not keyword_release.wait(5):
                        return 1
                    return 0
                conn.set_progress_handler(progress, 1)
            return conn

        def concurrent_keyword(query, **kwargs):
            local.partial = query == partial_text
            try:
                return original_keyword(query, **kwargs)
            finally:
                local.partial = False

        async def concurrent_search(*args, **kwargs):
            search_started.set()
            return await original_search(*args, **kwargs)

        monkeypatch.setattr(route_ledger, '_connect', concurrent_connect)
        monkeypatch.setattr(route_ledger, 'search_sources', concurrent_keyword)
        monkeypatch.setattr(store, 'search', concurrent_search)
        concurrent_rows = {}
        for lane in ('concurrent_embed', 'concurrent_search'):
            app.state.voice_requests = VoiceRequests()
            lane_rows = []
            for case in semantic['queries'] + keywords:
                sequence += 1
                utterance = f'concurrent-{sequence}'
                partial_text = pipeline.queries[(case['id'], 'q75')]
                pipeline.query_key = (case['id'], 'q75')
                pipeline.started.clear()
                keyword_started.clear(); keyword_release.clear(); search_started.clear()
                request = body(query=case['query'], k=5, max_tokens=600, utterance_id=utterance)
                partial = asyncio.create_task(client.post(ROUTE, json={**request, 'query': partial_text,
                    'speculative': True, 'request_id': utterance + '-partial'}))
                try:
                    assert await asyncio.to_thread(keyword_started.wait, 3)
                    await asyncio.wait_for(pipeline.started.wait() if lane == 'concurrent_embed'
                                           else search_started.wait(), 3)
                    cache = app.state.voice_requests.embeddings
                    assert not partial.done() and any(not job.done() for job in cache.jobs)
                    assert app.state.voice_requests.jobs, 'previous canonical read is not in flight'
                    calls = pipeline.calls
                    start = time.perf_counter_ns()
                    response = await client.post(ROUTE, json={**request, 'request_id': utterance + '-final'})
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    assert response.status_code == 200, response.text
                    packet = response.json()
                    assert pipeline.calls == calls
                    assert packet['retrieval']['voice'] == 'local_keywords'
                    hits = {ref['source_id'] for ref in packet['source_refs']}
                    lane_rows.append({'id': case['id'], 'elapsed_ms': elapsed,
                        'recall': len(hits & set(case['expected'])) / len(case['expected']),
                        'retrieval': packet['retrieval'], 'both_in_flight_at_final_start': True})
                    timings[lane].append(elapsed)
                finally:
                    keyword_release.set()
                    await asyncio.gather(partial, *app.state.voice_requests.jobs,
                                         *app.state.voice_requests.embeddings.jobs, return_exceptions=True)
                    await asyncio.sleep(0)
                assert partial.result().status_code == 409
            concurrent_rows[lane] = lane_rows
            print('VOICE_CONCURRENT_PROGRESS ' + lane, flush=True)

    stats = {name: latency(values) for name, values in timings.items()}
    report = {'complete': True, 'sources': sources + 12, 'messages': messages + 12, 'indexed_chunks': indexed,
        'model': corpus['model'], 'dims': corpus['dims'], 'elapsed_speech_embed_ms': corpus['single_query_ms'],
        'latency': stats, 'ready_vectors_used': counts, 'quality': results,
        'concurrent_per_query': concurrent_rows,
        'vector_audit': {'searches': len(searched_vectors),
            'full_query_searches': sum(q['vector_key'].startswith('q:') for q in searched_vectors),
            'prefix_query_searches': sum(q['vector_key'].startswith('q75:') for q in searched_vectors),
            'all_search_vectors_match_recorded_variant': True,
            'semantic_full_prefix_vectors_differ': sum(
                corpus['vectors']['q:' + case['id']] != corpus['vectors']['q75:' + case['id']]
                for case in semantic['queries']),
            'semantic_hit_rankings_differ_at_5_600': sum(a['speculative_hit_ids'] != b['speculative_hit_ids']
                for a, b in zip(results[-1]['semantic']['q75_ready']['per_query'],
                               results[-1]['semantic']['full_ready']['per_query']))},
        'method': 'Authenticated in-process ASGI; warm serial calls; k=5/max_tokens=600 latency, '
            '90 queries per variant. Partial POST embeds recorded q75 word prefixes or full text with '
            '70.1 ms async delay, all elapsed before the timed final POST. Late embeddings remain blocked '
            'after 75 ms until its final returns. Frozen reference UTC 2026-09-05; '
            '120 keyword records + 12 semantic records + 5000 same-person two-message conversations '
            '(plus 3 existing scope fixtures). Fresh temporary Lance cosine search of real 4096-d vectors; '
            'the single supplied distractor vector is repeated for background messages and scope fixtures. '
            'Concurrent lanes start the final while a previous partial is inside a SQLite statement '
            '(progress callback barrier) and recorded embed or real Lance search is in flight. '
            'No unique-background embedding/ANN/live hardware qualification. '
            'Cache/arbitration reset between k/budget configurations. p95 nearest-rank.'}
    gates = {
        'semantic_above_half': all(row['semantic'][variant]['recall'] > .5 for row in results
                                   for variant in ('q75_ready', 'full_ready')),
        'keyword_no_drop': all(row['keyword'][variant]['recall'] >= row['keyword']['keyword_only']['recall']
                              for row in results for variant in ('q75_ready', 'full_ready')),
        'review_tables_unchanged_or_better': all(
            row['semantic'][variant]['recall'] >= expected / 12
            for row, expected in zip(results, (8, 7, 8, 9, 10, 11))
            for variant in ('q75_ready', 'full_ready')),
        'keyword_receipts_preserved': all(set(base['hits']) <= set(hybrid['hits'])
            for row in results for name in ('semantic', 'keyword')
            for variant in ('q75_ready', 'full_ready') for base, hybrid in zip(
                row[name]['keyword_only']['per_query'], row[name][variant]['per_query'])),
        'final_budget_50ms': all(stats[name]['p95_ms'] <= 50 for name in stats),
        'concurrent_keyword_unchanged': all(q['recall'] == baseline['recall']
            for rows in concurrent_rows.values() for q, baseline in zip(rows,
                results[-1]['semantic']['keyword_only']['per_query'] +
                results[-1]['keyword']['keyword_only']['per_query'])),
        'late_keyword_only': all(q['retrieval']['voice'] == 'local_keywords' for q in late_rows),
        'late_recall_unchanged': all(q['recall'] == baseline['recall'] for q, baseline in zip(late_rows,
            results[-1]['semantic']['keyword_only']['per_query'] + results[-1]['keyword']['keyword_only']['per_query'])),
    }
    report['gates'] = gates
    report['late_per_query'] = late_rows
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('\nVOICE_SPECULATION ' + json.dumps({key: value for key, value in report.items()
        if key not in ('quality', 'concurrent_per_query', 'late_per_query')}))
    print('VOICE_SPECULATION_QUALITY ' + json.dumps([
        {**{key: row[key] for key in ('k', 'max_tokens')},
         **{name: {variant: data['recall'] for variant, data in row[name].items()}
            for name in ('semantic', 'keyword')}} for row in results]))
    assert all(gates.values()), output
