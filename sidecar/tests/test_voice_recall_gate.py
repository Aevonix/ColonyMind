"""The voice gate: seeded canonical sources, route timings and fixed recall quality."""
from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import time

from httpx import ASGITransport, AsyncClient

from onekey import KEY
from colonymind.router.tokens import estimate_tokens
from colonymind.turns import TurnIdempotencyLedger
from test_canonical_memory_search import memory_app  # noqa: F401
from test_turn_source_evidence import source_app  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
CASES = Path(__file__).parent / 'fixtures/voice_recall_cases.json'
NOW = datetime(2026, 9, 5, 23, 0, tzinfo=timezone.utc)


def seed(ledger):
    fixture = json.loads((ROOT / 'benchmarks/source_recall/fixtures.json').read_text())
    for record in fixture['records']:
        ledger.record_source(record['id'], contact_id='person', session_id='fixture',
            occurred_at=record['at'] + 'T12:00:00+00:00',
            messages=[{'role': 'user', 'content': record['content']}], derive_claims=False)
    # Diverse ordinary work conversations, using the production ingestion/index writer.
    # Every distractor is in the searched person's scope; no tiny scoped-store shortcut.
    templates = [
        'The weekly planning meeting for workspace W-{i} lasts thirty minutes. We reviewed the repair budget '
        'and agreed to update the checklist after inspecting the equipment. The notes are in the workshop binder.',
        'At workspace W-{i}, the delivery parcel arrived before lunch. The reception desk logged the receipt '
        'and asked the maintenance coordinator to inspect the packaging. The dispatch team will follow up tomorrow.',
        'The calibration test for workspace W-{i} finished successfully. The technician measured a small '
        'alignment error and recorded the settings in the service log. A second measurement is scheduled next week.',
        'Workspace W-{i} uses a ceramic cup for afternoon tea. The cabinet contains spare labels and a blue '
        'storage crate. We moved the trolley beside the entrance so the facilities team can reach the supplies.',
        'The resource assigned to work order W-{i} is an inspection cart. The repair team ordered replacement '
        'wheels and expects the parts on Friday. The customer asked for a status report after the next inspection.',
    ]
    for i in range(5000):
        ledger.record_source(f'background-{i:05d}', contact_id='person', session_id=f'work-{i // 20}',
            occurred_at='2026-09-02T12:00:00+00:00', messages=[
                {'role': 'user', 'content': templates[i % len(templates)].format(i=i)},
                {'role': 'assistant', 'content': f'I recorded the workspace W-{i} update for the service review.'}],
            derive_claims=False)
    with closing(ledger._connect()) as conn:
        sources = conn.execute("SELECT count(*) FROM turn_sources WHERE contact_id='person'").fetchone()[0]
        messages = sum(len(json.loads(row[0])) for row in conn.execute(
            "SELECT messages_json FROM turn_sources WHERE contact_id='person'"))
    assert sources >= 5000 and messages >= 5000
    return sources, messages


async def test_voice_latency_and_quality_gate(memory_app, monkeypatch, tmp_path):
    from colonymind.util import temporal
    app, ledger = memory_app
    monkeypatch.setattr(temporal, 'now_utc', lambda: NOW)
    monkeypatch.delenv('COLONYMIND_CONTEXT_CHARS_PER_TOKEN', raising=False)
    monkeypatch.setenv('COLONYMIND_RECALL_CONTEXT_MAX_CHARS', '4000')
    sources, messages = seed(ledger)
    cases = json.loads(CASES.read_text())['queries']
    assert len(cases) >= 20 and all(case['expected'] for case in cases)
    original = json.loads((ROOT / 'benchmarks/source_recall/fixtures.json').read_text())['queries']
    assert all(case in original for case in cases)  # queries/expected hits never tuned for voice
    results, timings = [], {'voice': [], 'search': []}
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test',
                           headers={'Authorization': 'Bearer ' + KEY}) as client:
        async def invoke(route, case, k, index, max_tokens=600):
            request = {'identity': {'host_id': 'gate'}, 'person_id': 'person', 'session_id': 'later',
                       'query': case['query'], 'timezone': 'UTC'}
            if route == 'voice':
                request.update(k=k, max_tokens=max_tokens, utterance_id=f'gate-{index}', request_id=f'r-{index}')
                path = '/v1/host/memory/voice-recall'
            else:
                request.update(limit=k, as_of=NOW.isoformat())
                path = '/v1/host/memory/search'
            start = time.perf_counter_ns()
            response = await client.post(path, json=request)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            assert response.status_code == 200, response.text
            packet = response.json()
            assert packet['count'] <= k
            if route == 'voice':
                assert estimate_tokens(packet['content']) == packet['token_count'] <= max_tokens
            hits = {ref['source_id'] for ref in packet['source_refs']}
            return len(hits & set(case['expected'])) / len(case['expected']), elapsed

        # Warm imports, SQLite pages and thread executor outside timed samples.
        for i in range(10):
            for route in timings:
                await invoke(route, cases[i], 5, f'warm-{i}')
        for k in (3, 4, 5):
            for max_tokens in (300, 600):
                voice_scores, search_scores, per_query = [], [], []
                for i, case in enumerate(cases):
                    scores = {}
                    # Alternate order to reduce systematic page-cache/order bias.
                    for route in (('voice', 'search') if i % 2 else ('search', 'voice')):
                        score, elapsed = await invoke(route, case, k, f'{k}-{max_tokens}-{i}', max_tokens)
                        scores[route] = score
                        if k == 5 and max_tokens == 600:
                            timings[route].append(elapsed)
                    voice_scores.append(scores['voice'])
                    search_scores.append(scores['search'])
                    per_query.append({'id': case['id'], **scores})
                result = {'k': k, 'max_tokens': max_tokens, 'queries': len(cases),
                    'voice_recall': statistics.mean(voice_scores), 'search_recall': statistics.mean(search_scores),
                    'voice_hits': sum(score > 0 for score in voice_scores),
                    'search_hits': sum(score > 0 for score in search_scores), 'per_query': per_query}
                results.append(result)

    latency = {route: {'samples': len(values), 'p50_ms': statistics.median(values),
                       'p90_ms': sorted(values)[int(0.9 * (len(values) - 1))], 'max_ms': max(values)}
               for route, values in timings.items()}
    report = {'sources': sources, 'messages': messages, 'latency': latency, 'quality': results,
        'method': 'In-process authenticated ASGI POST, warm serial alternating routes, k=5/max_tokens=600 latency; '
                  'canonical text import of all frozen fixture records plus 5000 same-person work conversations; '
                  'macro expected-source recall from packet source_refs; no semantic index or inference; '
                  'all 78 original expected-hit queries, UTC reference 2026-09-05.'}
    output = Path(os.environ.get('VOICE_RECALL_REPORT', str(tmp_path / 'voice-recall-measurements.json')))
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('\nVOICE_RECALL_LATENCY ' + json.dumps(latency))
    print('VOICE_RECALL_QUALITY ' + json.dumps([{k: v for k, v in result.items() if k != 'per_query'}
                                              for result in results]))
    assert all(result['voice_recall'] >= result['search_recall'] for result in results), output
    assert latency['voice']['p50_ms'] < 50, output
