"""Offline review fixes; original tests and keyword fixtures stay read-only."""
import asyncio
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sqlite3
import statistics

from httpx import ASGITransport, AsyncClient
import pytest

from onekey import KEY
from colonymind.api.credentials import CredentialTable, secret_digest
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.memory.recall import estimate_voice_tokens, pack_memory_tokens
from colonymind.memory.voice_recall import VoiceBusy, VoiceRequests, VoiceSuperseded
from colonymind.router.tokens import estimate_tokens
from test_canonical_memory_search import memory_app  # noqa: F401
from test_turn_source_evidence import source_app  # noqa: F401
from test_voice_recall import ROUTE, body, rows
from test_voice_recall_gate import NOW, seed

SEMANTIC_CASES = Path(__file__).parent / 'fixtures/voice_recall_semantic_cases.json'
# Measured serving-model/old-estimator ratios from review. No serving tokenizer is
# available offline; these are an empirical envelope, not a universal theorem.
SYNTHETIC = [
    ('english', 'A cautious reader checks the complete recalled passage. ' * 120, 0.82),
    ('digits', '0123456789' * 700, 1.8),
    ('hex', 'deadbeef0123456789abcdef' * 350, 1.8),
    ('cjk', '我昨天去了东京出差' * 800, 2.45),
    ('emoji', '🙂🚀💡🧑🏽\u200d💻♥️' * 700, 3.9),
    ('symbols', '♜∑∞→★' * 1200, 3.9),
]


def old_default_count(text):
    """Review's default 4/3-character estimator, independent of env overrides."""
    if not text:
        return 0
    sample = text[:20000]
    dense = sum(not (c.isalpha() or c in " .,;:'\"!?-") for c in sample) / len(sample) > .25
    return math.ceil(len(text) / (3 if dense else 4))


@pytest.mark.parametrize('name,text,ratio', SYNTHETIC, ids=[x[0] for x in SYNTHETIC])
@pytest.mark.parametrize('override', ['1', '4', '8'])
def test_voice_counter_bounds_review_model_ratios(monkeypatch, name, text, ratio, override):
    monkeypatch.setenv('COLONYMIND_CONTEXT_CHARS_PER_TOKEN', override)
    assert estimate_voice_tokens(text) >= math.ceil(old_default_count(text) * ratio)
    assert estimate_voice_tokens(text) >= estimate_tokens(text)


@pytest.mark.parametrize('name,text,ratio', SYNTHETIC, ids=[x[0] for x in SYNTHETIC])
@pytest.mark.parametrize('budget', [300, 600])
@pytest.mark.parametrize('override', ['1', '4', '8'])
def test_complete_voice_packet_bounds_review_model_ratios(monkeypatch, name, text, ratio, budget, override):
    monkeypatch.setenv('COLONYMIND_CONTEXT_CHARS_PER_TOKEN', override)
    candidates = rows(text)
    original = deepcopy(candidates)
    selected, content = pack_memory_tokens(candidates, limit=5, max_tokens=budget)
    assert candidates == original
    assert estimate_voice_tokens(content) <= budget
    # Count each language segment at its measured ratio, and the surrounding
    # warning, legend, JSON/provenance and truncation notices as English ASCII.
    # The check uses the rendered packet, not its original oversized input.
    remainder = content
    quote_bound = 0
    for row in selected:
        encoded = json.dumps(row['content'], ensure_ascii=False)[1:-1]
        assert encoded in remainder
        remainder = remainder.replace(encoded, '', 1)
        quote_bound += math.ceil(old_default_count(row['content']) * ratio)
    assert quote_bound + math.ceil(old_default_count(remainder) * .82) <= budget
    if override != '1' and budget == 600:
        assert selected  # demonstrate nonempty capped retrieval, not just abstention


@pytest.mark.parametrize('text', ['123', 'a1b2', 'abcdef', 'deadbeef', '我', 'é', '\u0301', '\u200d', '🙂', '★', '—'])
def test_small_and_mixed_unicode_units_are_never_rounded_down(text):
    if not text.isascii():
        assert estimate_voice_tokens(text) >= (3 if text in ('🙂', '★', '—') else 1)
    assert estimate_voice_tokens('meeting ' + text * 250 + ' summary') >= estimate_voice_tokens(text * 250)


@pytest.mark.parametrize('name,text,ratio', SYNTHETIC, ids=[x[0] for x in SYNTHETIC])
@pytest.mark.parametrize('budget', [300, 600])
async def test_http_route_uses_strict_counter(memory_app, monkeypatch, name, text, ratio, budget):
    monkeypatch.setenv('COLONYMIND_CONTEXT_CHARS_PER_TOKEN', '8')
    app, ledger = memory_app
    ledger.record_source('review-multilingual', contact_id='person', session_id='fixture',
        messages=[{'role': 'user', 'content': 'Tokenproof ' + text}], derive_claims=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test',
                           headers={'Authorization': 'Bearer ' + KEY}) as client:
        response = await client.post(ROUTE, json=body(query='Tokenproof', max_tokens=budget))
    assert response.status_code == 200, response.text
    packet = response.json()
    assert estimate_voice_tokens(packet['content']) <= budget
    assert packet['token_count'] == estimate_tokens(packet['content']) <= budget


@pytest.mark.parametrize('other_scope', [('principal', 'other-credential'), ('other-principal', 'credential')])
async def test_one_credential_flood_cannot_make_another_busy(other_scope):
    coordinator = VoiceRequests()
    async def read(flag):
        return 'evidence'
    for index in range(coordinator.capacity):
        assert await coordinator.run(('principal', 'credential', 'person', '', str(index)),
                                     speculative=False, read=read) == 'evidence'
    with pytest.raises(VoiceBusy):
        await coordinator.run(('principal', 'credential', 'different-person', 'other-session', 'overflow'),
                              speculative=False, read=read)
    assert await coordinator.run((*other_scope, 'person', '', 'fresh'),
                                 speculative=False, read=read) == 'evidence'


async def test_http_busy_is_isolated_to_the_authenticated_credential(source_app):
    secrets = {'flood': 'review-flood-secret', 'independent': 'review-independent-secret'}
    table = CredentialTable.from_config({
        'api-key': {'owner': ['voice'], 'contacts': 'all'},
        **{name: {'sha256': secret_digest(secret), 'owner': [], 'contacts': 'own', 'access': ['read']}
           for name, secret in secrets.items()},
    })
    source_app.add_middleware(ApiKeyMiddleware, api_key=KEY, credentials=table)
    source_app.state.voice_requests = VoiceRequests(capacity=2)
    async with AsyncClient(transport=ASGITransport(app=source_app), base_url='http://test') as client:
        for index in range(3):
            response = await client.post(ROUTE,
                headers={'Authorization': 'Bearer ' + secrets['flood']},
                json=body(person_id='credential:flood', utterance_id=f'flood-{index}'))
            assert response.status_code == (200 if index < 2 else 503), response.text
        assert response.json()['detail']['code'] == 'voice_recall_busy'
        response = await client.post(ROUTE,
            headers={'Authorization': 'Bearer ' + secrets['independent']},
            json=body(person_id='credential:independent', utterance_id='fresh'))
        assert response.status_code == 200, response.text


async def test_completed_partials_release_entries_without_retention():
    coordinator = VoiceRequests(capacity=2)
    async def read(flag):
        return 'partial'
    for index in range(10):
        key = ('principal', 'credential', 'person', '', str(index))
        assert await coordinator.run(key, speculative=True, read=read) == 'partial'
        assert key not in coordinator.entries
        assert not coordinator.jobs and not coordinator.job_scopes


@pytest.mark.parametrize('error', [False, True])
async def test_abandoned_partial_keeps_job_limit_until_worker_finishes(error):
    coordinator = VoiceRequests(capacity=1, workers=2)
    started, release = asyncio.Event(), asyncio.Event()
    async def blocked(flag):
        started.set()
        await release.wait()
        if error:
            raise RuntimeError('offline test failure')
        return 'obsolete'
    async def read(flag):
        return 'other credential'
    key = ('principal', 'credential', 'person', '', 'partial')
    partial = asyncio.create_task(coordinator.run(key, speculative=True, read=blocked))
    await started.wait()
    partial.cancel()
    with pytest.raises(asyncio.CancelledError):
        await partial
    try:
        with pytest.raises(VoiceBusy):
            await coordinator.run((*key[:-1], 'another'), speculative=True, read=read)
        assert await coordinator.run(('principal', 'other-credential', *key[2:]),
                                     speculative=False, read=read) == 'other credential'
    finally:
        jobs = list(coordinator.jobs)
        release.set()
        await asyncio.gather(*jobs, return_exceptions=True)
    assert key not in coordinator.entries
    assert not coordinator.jobs and not coordinator.job_scopes


async def test_superseded_jobs_count_towards_their_credential_capacity():
    coordinator = VoiceRequests(capacity=2, workers=3)
    release = asyncio.Event()
    started = asyncio.Queue()
    async def blocked(flag):
        started.put_nowait(flag)
        await release.wait()
        return 'obsolete'
    async def read(flag):
        return 'independent'
    key = ('principal', 'credential', 'person', '', 'utterance')
    first = asyncio.create_task(coordinator.run(key, speculative=True, read=blocked))
    first_flag = await started.get()
    second = asyncio.create_task(coordinator.run(key, speculative=False, read=blocked))
    await started.get()
    try:
        with pytest.raises(VoiceSuperseded):
            await first
        assert first_flag.is_set()
        assert len(coordinator.entries) == 1 and len(coordinator.jobs) == 2
        with pytest.raises(VoiceBusy):
            await coordinator.run(key, speculative=False, read=read)
        assert await coordinator.run(('other-principal', 'credential', *key[2:]),
                                     speculative=False, read=read) == 'independent'
    finally:
        release.set()
        await asyncio.gather(second, *coordinator.jobs)
    assert not coordinator.jobs and not coordinator.job_scopes
    assert coordinator.entries[key].final


async def test_completed_partial_remains_fenced_until_response_leaves():
    coordinator = VoiceRequests()
    key = ('principal', 'credential', 'person', '', 'utterance')
    final_tasks = []
    async def final_read(flag):
        return 'final'
    async def partial_read(flag):
        # Schedule the final precisely when partial work finishes, before its
        # caller can finish response cleanup and return stale evidence.
        asyncio.get_running_loop().call_soon(lambda: final_tasks.append(asyncio.create_task(
            coordinator.run(key, speculative=False, read=final_read))))
        return 'stale partial'
    with pytest.raises(VoiceSuperseded):
        await coordinator.run(key, speculative=True, read=partial_read)
    assert await final_tasks[0] == 'final'
    assert coordinator.entries[key].final


def test_semantic_fixture_has_no_shared_content_words_or_porter_stems():
    fixture = json.loads(SEMANTIC_CASES.read_text())
    sources = {row['id']: row['content'] for row in fixture['records']}
    stop = 'a an and are as at be because below by can does due during each every for from how i in is it my of on our should that the their they to until was we when whenever where which while why with without'.split()
    with sqlite3.connect(':memory:') as conn:
        conn.execute("CREATE VIRTUAL TABLE words USING fts5(text, tokenize='porter unicode61')")
        conn.execute("CREATE VIRTUAL TABLE vocabulary USING fts5vocab(words, 'instance')")
        def terms(text):
            conn.execute('DELETE FROM words')
            conn.execute('INSERT INTO words(text) VALUES (?)', (text,))
            return {row[0] for row in conn.execute('SELECT term FROM vocabulary')}
        stop_stems = terms(' '.join(stop))
        assert len(fixture['queries']) >= 10
        for case in fixture['queries']:
            query = terms(case['query']) - stop_stems
            assert query
            for expected in case['expected']:
                shared = query & (terms(sources[expected]) - stop_stems)
                assert not shared, (case['id'], shared)


async def test_semantic_only_voice_recall_is_reported(memory_app, monkeypatch, tmp_path):
    from colonymind.util import temporal
    from colonymind import vector
    from colonymind.memory.selection import RecallSelector
    app, ledger = memory_app
    monkeypatch.setattr(temporal, 'now_utc', lambda: NOW)
    monkeypatch.delenv('COLONYMIND_CONTEXT_CHARS_PER_TOKEN', raising=False)
    def forbidden(*args, **kwargs):
        raise AssertionError('semantic qualification must stay inference-free')
    monkeypatch.setattr(vector, 'get_store', forbidden)
    monkeypatch.setattr(vector, 'get_pipeline', forbidden)
    monkeypatch.setattr(RecallSelector, '_reranked', forbidden)
    monkeypatch.setenv('COLONYMIND_RECALL_RERANK', 'on')
    monkeypatch.setenv('COLONYMIND_RECALL_RERANK_MIN_SCORE', '.9')
    sources, messages = seed(ledger)
    fixture = json.loads(SEMANTIC_CASES.read_text())
    for record in fixture['records']:
        ledger.record_source(record['id'], contact_id='person', session_id='semantic-fixture',
            occurred_at='2026-09-05T12:00:00+00:00',
            messages=[{'role': 'user', 'content': record['content']}], derive_claims=False)
    results = []
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test',
                           headers={'Authorization': 'Bearer ' + KEY}) as client:
        for k in (3, 4, 5):
            for budget in (300, 600):
                queries = []
                for case in fixture['queries']:
                    response = await client.post(ROUTE, json=body(query=case['query'], k=k,
                        max_tokens=budget, utterance_id=f"semantic-{k}-{budget}-{case['id']}"))
                    assert response.status_code == 200, response.text
                    packet = response.json()
                    assert packet['retrieval']['voice'] == 'local_keywords'
                    assert estimate_voice_tokens(packet['content']) <= budget
                    hits = sorted({ref['source_id'] for ref in packet['source_refs']})
                    score = len(set(hits) & set(case['expected'])) / len(case['expected'])
                    queries.append({'id': case['id'], 'expected': case['expected'],
                                    'source_refs': hits, 'count': packet['count'], 'recall': score})
                results.append({'k': k, 'max_tokens': budget, 'queries': len(queries),
                    'voice_recall': statistics.mean(row['recall'] for row in queries),
                    'voice_hits': sum(row['recall'] > 0 for row in queries),
                    'keyword_noise_packets': sum(row['count'] > 0 and row['recall'] == 0 for row in queries),
                    'per_query': queries})
    report = {'sources': sources + len(fixture['records']),
              'messages': messages + len(fixture['records']), 'quality': results,
              'method': '12 semantic-only queries without shared content words or Porter stems; '
                        'original 120 fixture records plus 5000 same-person conversations and '
                        '12 additional semantic sources; authenticated in-process voice ASGI; '
                        'keyword-only, vector lane and rerank cutoff skipped; no live-search parity claim.'}
    output = Path(os.environ.get('VOICE_RECALL_SEMANTIC_REPORT', str(tmp_path / 'semantic-measurements.json')))
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('\nVOICE_RECALL_SEMANTIC ' + json.dumps([{k: v for k, v in row.items() if k != 'per_query'}
                                                for row in results]))
