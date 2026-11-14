"""Seeded authenticated ASGI latency gate; no sockets or inference services."""
from contextlib import closing
import json
import math
import os
from pathlib import Path
import time
from uuid import uuid4

from httpx import ASGITransport, AsyncClient
import pytest

from colonymind.api.routers import host, social_state
from colonymind.memory.recall import estimate_voice_tokens
from test_voice_affect import affect_app, add_appraisal, call, stamp, ROUTE, KEY  # noqa: F401


def seed(state):
    # 64 distinct canonical receipts, plus a realistic much longer unlinked
    # imported affect history. All 10k events are in the measured owner's scope.
    identifiers = [add_appraisal(state, 'interest', age=i, name=f'seed-source-{i}') for i in range(64)]
    lineages = [state.affect.source_input(identifier, 'owner')[0] for identifier in identifiers]
    events = [(f'event-{i:05}', 'owner', -0.8, 0.1, 'explicit', None,
               stamp(state.now - (10000-i)/20), None,
               json.dumps(lineages[i]) if i < 64 else None) for i in range(10000)]
    with state.affect._conn:
        state.affect._conn.executemany('''INSERT INTO affect_events
            (id,contact_id,valence,arousal,source,trigger,timestamp,session_id,source_lineage_json)
            VALUES (?,?,?,?,?,?,?,?,?)''', events)
    state.affect._recompute_state('owner')
    # Keep 64 active heads; the other 9,936 records are appraisal history.
    # Controlled SQL fixtures reuse actual schemas and source receipts; no model
    # is invoked during seeding or during measured route requests.
    with closing(state.ledger._connect()) as conn, conn:
        row = dict(conn.execute('SELECT * FROM appraisal_records WHERE id=?', (identifiers[0],)).fetchone())
        history = [(f'history-{i:05}', row['owner_id'], row['subject_id'], f'history-{i:05}', row['kind'],
                    row['payload_json'], row['dependencies_json'], row['processor_json'], row['source_id'],
                    row['source_version'], state.now-3600-i, state.now-1800-i, 'settled', None)
                   for i in range(9936)]
        conn.executemany('INSERT INTO appraisal_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', history)
        counts = {'affect_events': state.affect._conn.execute('SELECT count(*) FROM affect_events').fetchone()[0],
                  'affect_owner_events': state.affect._conn.execute("SELECT count(*) FROM affect_events WHERE contact_id='owner'").fetchone()[0],
                  'source_linked_affect_events': state.affect._conn.execute('SELECT count(*) FROM affect_events WHERE source_lineage_json IS NOT NULL').fetchone()[0],
                  'canonical_sources': conn.execute('SELECT count(*) FROM turn_sources').fetchone()[0],
                  'appraisal_records': conn.execute('SELECT count(*) FROM appraisal_records').fetchone()[0],
                  'appraisal_heads': conn.execute('SELECT count(*) FROM appraisal_heads').fetchone()[0]}
    assert counts['affect_events'] == counts['affect_owner_events'] == counts['appraisal_records'] == 10000
    return counts


@pytest.mark.hardware_latency
async def test_voice_affect_seeded_latency_gate(affect_app, monkeypatch, hardware_latency):
    state = affect_app
    # Qualify store reads and exact output independently of scheduling. The
    # production deadline deliberately drops a late line; the boundary cases
    # below independently advance that decision clock through real store reads.
    # affect_app replaces host.time with a private namespace, so this controls
    # only the route's decision clock. No other perf_counter consumer is reached
    # by these requests; perf_counter_ns/thread_time_ns keep wall/CPU timing real.
    monkeypatch.setattr(host.time, 'perf_counter', lambda: state.elapsed)
    counts = seed(state)
    samples, cpu_samples, lines = [], [], []
    async with AsyncClient(transport=ASGITransport(app=state.app), base_url='http://fixture',
                           headers={'Authorization': 'Bearer '+KEY}) as client:
        async def invoke():
            start = time.perf_counter_ns()
            cpu_start = time.thread_time_ns()
            response = await client.post(ROUTE, json={'identity': {'host_id': 'seeded-gate'}, 'person_id': 'owner'})
            cpu_elapsed = (time.thread_time_ns()-cpu_start)/1e6
            elapsed = (time.perf_counter_ns()-start)/1e6
            assert response.status_code == 200, response.text
            assert set(response.json()) == {'line'}
            line = response.json()['line']
            assert line == 'Sam: frustrated, low energy · Nova: interested'
            assert len(line) <= 120 and estimate_voice_tokens(line) <= 25
            assert not any(char in line for char in '[]\u2013\u2014')
            return elapsed, cpu_elapsed, line
        for _ in range(20):
            await invoke()
        for _ in range(300):
            elapsed, cpu_elapsed, line = await invoke()
            samples.append(elapsed)
            cpu_samples.append(cpu_elapsed)
            lines.append(line)
    ordered = sorted(samples)
    percentile = lambda fraction: ordered[math.ceil(len(ordered)*fraction)-1]
    report = {'route': 'POST '+ROUTE, 'store_counts': counts,
        'method': 'httpx ASGITransport, authenticated middleware + body/schema validation + actual store readers + JSON response; warm serial, no sockets',
        'seed_method': 'controlled SQL history fixtures and real canonical source ingestion; recomputed AffectStore projection',
        'warmups': 20, 'samples': len(samples), 'nonempty_responses': sum(bool(line) for line in lines),
        'percentile': 'nearest rank', 'milliseconds': {'p50': percentile(.5), 'p95': percentile(.95), 'max': max(samples)},
        'sample_line': lines[0], 'cautious_tokens': estimate_voice_tokens(lines[0]),
        'characters': len(lines[0]), 'threshold_ms': 20,
        'scope': 'local warm serial qualification; excludes client transport and concurrent live load',
        'inference_calls': 0, 'gate_pass': percentile(.95) < 20}
    path = Path(os.environ.get('VOICE_AFFECT_REPORT', str(state.path/'voice-affect-measurements.json')))
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, sort_keys=True))
    hardware_latency(samples, cpu_samples, threshold_ms=20, statistic='p95',
                     enforced=os.environ.get('GITHUB_ACTIONS') != 'true')


@pytest.mark.parametrize('reader', ['speaker', 'appraisal'])
@pytest.mark.parametrize(('elapsed', 'expected'), [
    pytest.param(0.006, 'Sam: frustrated, low energy · Nova: interested', id='6ms'),
    pytest.param(math.nextafter(0.020, 0.0), 'Sam: frustrated, low energy · Nova: interested', id='before-20ms'),
    pytest.param(0.020, '', id='at-20ms'),
    pytest.param(math.nextafter(0.020, math.inf), '', id='after-20ms'),
])
async def test_voice_affect_read_deadline_boundary(affect_app, monkeypatch, reader, elapsed, expected):
    state = affect_app
    state.affect.create_event(contact_id='owner', valence=-0.8, arousal=0.1, timestamp=stamp(state.now))
    add_appraisal(state, name='marker-' + str(uuid4()))
    monkeypatch.setattr(host.time, 'perf_counter', lambda: state.elapsed)
    monkeypatch.setattr(social_state, 'appraisal_store', lambda: state.appraisals)
    store, method = ((state.affect, 'get_state') if reader == 'speaker'
                     else (state.appraisals, 'voice_appraisal'))
    original = getattr(store, method)
    reads = []

    def completing_read(*args, **kwargs):
        result = original(*args, **kwargs)
        state.elapsed = elapsed
        reads.append(result)
        return result

    monkeypatch.setattr(store, method, completing_read)
    # These are fixed contract boundaries, independent of the implementation's
    # constant: deriving them from DEADLINE_SECONDS would bless a 1 ms mutant.
    assert await call(state) == expected
    assert len(reads) == 1 and state.elapsed == elapsed
