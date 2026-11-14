"""Optional offline qualification against an external, recorded embedding corpus."""
import asyncio
from contextlib import closing
import json
import math
import os
from pathlib import Path
import statistics
import time

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
        self.calls = 0
        self.delay = corpus['single_query_ms'] / 1000
        self.started, self.release = asyncio.Event(), None
        for case in cases:
            words = case['query'].split()
            for variant, text in [('q', case['query']), ('q75', ' '.join(words[:int(.75 * len(words))]))]:
                self.queries[text] = self.vectors[f"{variant}:{case['id']}"]

    async def embed_query(self, query):
        self.calls += 1
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        else:
            await asyncio.sleep(self.delay)
        return self.queries[query]


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
    from colonymind.util import temporal
    app, ledger = memory_app
    monkeypatch.setattr(temporal, 'now_utc', lambda: NOW)
    monkeypatch.delenv('COLONYMIND_CONTEXT_CHARS_PER_TOKEN', raising=False)
    monkeypatch.setenv('COLONYMIND_RECALL_CONTEXT_MAX_CHARS', '4000')
    monkeypatch.setenv('COLONYMIND_RECALL_RERANK', 'off')
    sources, messages = seed(ledger)
    semantic = json.loads((Path(__file__).parent / 'fixtures/voice_recall_semantic_cases.json').read_text())
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
    results, timings = [], {'keyword_only': [], 'q75_ready': [], 'full_ready': [], 'late': []}
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
            if variant != 'keyword_only':
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
                    'retrieval': packet['retrieval'], 'elapsed_ms': elapsed}

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

    stats = {name: latency(values) for name, values in timings.items()}
    report = {'complete': True, 'sources': sources + 12, 'messages': messages + 12, 'indexed_chunks': indexed,
        'model': corpus['model'], 'dims': corpus['dims'], 'elapsed_speech_embed_ms': corpus['single_query_ms'],
        'latency': stats, 'ready_vectors_used': counts, 'quality': results,
        'method': 'Authenticated in-process ASGI; warm serial calls; k=5/max_tokens=600 latency, '
            '90 queries per variant. Partial POST embeds recorded q75 word prefixes or full text with '
            '70.1 ms async delay, all elapsed before the timed final POST. Late embeddings remain blocked '
            'after 75 ms until its final returns. Frozen reference UTC 2026-09-05; '
            '120 keyword records + 12 semantic records + 5000 same-person two-message conversations '
            '(plus 3 existing scope fixtures). Fresh temporary Lance cosine search of real 4096-d vectors; '
            'the single supplied distractor vector is repeated for background messages and scope fixtures. '
            'No unique-background embedding/ANN/concurrency/live hardware qualification. '
            'Cache/arbitration reset between k/budget configurations. p95 nearest-rank.'}
    gates = {
        'semantic_above_half': all(row['semantic'][variant]['recall'] > .5 for row in results
                                   for variant in ('q75_ready', 'full_ready')),
        'keyword_no_drop': all(row['keyword'][variant]['recall'] >= row['keyword']['keyword_only']['recall']
                              for row in results for variant in ('q75_ready', 'full_ready')),
        'final_budget_50ms': all(stats[name]['p95_ms'] < 50 for name in stats),
        'late_keyword_only': all(q['retrieval']['voice'] == 'local_keywords' for q in late_rows),
        'late_recall_unchanged': all(q['recall'] == baseline['recall'] for q, baseline in zip(late_rows,
            results[-1]['semantic']['keyword_only']['per_query'] + results[-1]['keyword']['keyword_only']['per_query'])),
    }
    report['gates'] = gates
    report['late_per_query'] = late_rows
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('\nVOICE_SPECULATION ' + json.dumps({key: value for key, value in report.items() if key != 'quality'}))
    print('VOICE_SPECULATION_QUALITY ' + json.dumps([
        {**{key: row[key] for key in ('k', 'max_tokens')},
         **{name: {variant: data['recall'] for variant, data in row[name].items()}
            for name in ('semantic', 'keyword')}} for row in results]))
    assert all(gates.values()), output
