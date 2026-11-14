"""Voice envelopes retry only withdrawals and share the worker's nested loop."""
import asyncio
import json
import os
import threading
import time
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from onekey import KEY
from colonymind.memory import curated
from colonymind.turns import TurnIdempotencyLedger
from test_canonical_memory_search import memory_app
from test_turn_source_evidence import source_app
from test_forget_fix8 import withdraw, traffic
from test_voice_affect import affect_app, add_appraisal, call


def emit(**record):
    if os.environ.get('FIX10_METRICS'):
        with open(os.environ['FIX10_METRICS'], 'a') as output:
            output.write(json.dumps(record) + '\n')


async def recall(app, identity):
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://fixture',
                           headers={'Authorization': 'Bearer ' + KEY}) as client:
        return await client.post('/v1/host/memory/voice-recall', json={
            'identity': {'host_id': 'fixture'}, 'person_id': 'person', 'session_id': 'later',
            'query': 'hydrofoil', 'k': 5, 'max_tokens': 600,
            'utterance_id': identity, 'request_id': identity})


def watch_cursor(monkeypatch):
    cursors = []
    original = curated._withdrawal_version
    def version(*args):
        result = original(*args)
        cursors.append(result)
        return result
    monkeypatch.setattr(curated, '_withdrawal_version', version)
    return cursors


@pytest.mark.parametrize('mode', ['one', 'sustained'])
async def test_voice_packet_ignores_ordinary_writes(memory_app, monkeypatch, mode):
    from colonymind.memory import search, voice_recall
    app, ledger = memory_app
    original, prepare = voice_recall._read_voice_packet, search._prepare_memory
    packets, preparations, writes, errors = [], [], [], []
    start, written, stop = threading.Event(), threading.Event(), threading.Event()
    cursors = watch_cursor(monkeypatch)
    def writer():
        start.wait()
        try:
            other = TurnIdempotencyLedger(ledger.db_path)
            while not stop.is_set():
                other.record_source(f'ordinary-{len(writes)}', contact_id='other', session_id='other',
                    messages=[{'role': 'user', 'content': 'Fixture'}], derive_claims=False)
                writes.append(time.perf_counter())
                written.set()
                if mode == 'one':
                    break
                stop.wait(.005)
        except Exception as error:
            errors.append(error)
            written.set()
    def packet(*args, **kwargs):
        packets.append(True)
        return original(*args, **kwargs)
    def delayed(*args, **kwargs):
        preparations.append(True)
        start.set()
        assert written.wait(5)
        time.sleep(.12)
        return prepare(*args, **kwargs)
    monkeypatch.setattr(voice_recall, '_read_voice_packet', packet)
    monkeypatch.setattr(search, '_prepare_memory', delayed)
    thread = threading.Thread(target=writer)
    thread.start()
    try:
        response = await recall(app, mode)
    finally:
        stop.set()
        start.set()
        await asyncio.to_thread(thread.join, 5)
    assert not thread.is_alive() and not errors
    rate = (len(writes)-1)/(writes[-1]-writes[0]) if len(writes)>1 else None
    emit(probe='ordinary', mode=mode, status=response.status_code, packets=len(packets),
         preparations=len(preparations), writes=len(writes), writes_per_second=rate,
         cursor_reads=len(cursors), memory='Friday at nine' in response.text)
    assert writes
    if mode == 'sustained':
        assert len(writes) >= 4 and rate >= 30
    assert response.status_code == 200
    assert 'Friday at nine' in response.json()['content']
    assert len(packets) == len(preparations) == 1
    assert len(cursors) == 2 and cursors[0] == cursors[1]


@pytest.mark.parametrize('mode', ['once', 'every'])
async def test_voice_withdrawal_after_packet_never_publishes_old_material(memory_app, monkeypatch, mode):
    from colonymind.memory import search, voice_recall
    app, ledger = memory_app
    original, prepare = voice_recall._read_voice_packet, search._prepare_memory
    packets, preparations = [], []
    cursors = watch_cursor(monkeypatch)
    def prepared(*args, **kwargs):
        preparations.append(True)
        return prepare(*args, **kwargs)
    def packet(*args, **kwargs):
        result = original(*args, **kwargs)
        packets.append(True)
        if len(packets) == 1:
            assert result is not None and 'Friday at nine' in result.content
            withdraw(ledger, 'report', 'person')
        elif mode == 'every':
            # A genuine new peer withdrawal moves the cursor in every attempt.
            source = f'withdrawn-{len(packets)}'
            ledger.record_source(source, contact_id='person', session_id='work',
                messages=[{'role': 'user', 'content': 'Fixture'}], derive_claims=False)
            withdraw(ledger, source, 'person')
        return result
    monkeypatch.setattr(voice_recall, '_read_voice_packet', packet)
    monkeypatch.setattr(search, '_prepare_memory', prepared)
    response = await recall(app, mode)
    emit(probe='withdrawal', mode=mode, status=response.status_code, packets=len(packets),
         preparations=len(preparations), cursor_reads=len(cursors),
         leaked='Friday at nine' in response.text)
    assert 'Friday at nine' not in response.text and 'report' not in response.text
    assert len(packets) == len(preparations) == (2 if mode == 'once' else 3)
    assert len(cursors) == len(packets)+1
    if mode == 'once':
        assert response.status_code == 200 and response.json()['content'] == ''
    else:
        assert response.status_code == 503
        assert response.json()['detail']['code'] == 'memory_backend_unavailable'


def test_sync_withdrawal_fence_lost_cursor_and_exception_reset(traffic, monkeypatch):
    ledger, _ = traffic
    version = curated._withdrawal_version(ledger)
    values = iter([version, None])
    monkeypatch.setattr(curated, '_withdrawal_version', lambda _: next(values))
    calls = []
    assert curated.withdrawal_fenced_read(ledger, lambda: calls.append(True) or 'old', empty=None) is None
    assert calls == [True] and not curated._withdrawal_reads.get()
    monkeypatch.setattr(curated, '_withdrawal_version', lambda _: version)
    def broken():
        raise ValueError('fixture')
    with pytest.raises(ValueError, match='fixture'):
        curated.withdrawal_fenced_read(ledger, broken, empty=None)
    assert not curated._withdrawal_reads.get()


async def test_voice_affect_ignores_ordinary_write(affect_app, monkeypatch):
    from colonymind.api.routers import host, social_state
    state = affect_app
    monkeypatch.setattr(social_state, 'appraisal_store', lambda: state.appraisals)
    monkeypatch.setattr(host, 'time', SimpleNamespace(time=lambda: state.now, perf_counter=lambda: 0.))
    add_appraisal(state)
    before = await call(state)
    assert before
    original = state.appraisals.voice_appraisal
    calls = []
    cursors = watch_cursor(monkeypatch)
    def writing(*args, **kwargs):
        calls.append(True)
        TurnIdempotencyLedger(state.ledger.db_path).record_source(
            f'ordinary-{len(calls)}', contact_id='other', session_id='other',
            messages=[{'role': 'user', 'content': 'Fixture'}], derive_claims=False)
        return original(*args, **kwargs)
    monkeypatch.setattr(state.appraisals, 'voice_appraisal', writing)
    offered = await call(state)
    assert offered == before
    assert len(calls) == 1 and len(cursors) == 2
