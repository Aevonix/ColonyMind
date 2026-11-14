"""Hot assembly ignores ordinary ledger churn and keeps its withdrawal envelope."""
import asyncio
from contextlib import closing
import json
import os
import threading
import time

import pytest
from httpx import ASGITransport, AsyncClient

from onekey import KEY
from colonymind.memory import curated
from colonymind.turns import TurnIdempotencyLedger
from test_source_annotations import annotated_app, REPORT
from test_turn_source_evidence import source_app
from test_forget_fix8 import traffic, withdraw, OWNER


def ordinary_write(path, number):
    TurnIdempotencyLedger(path).record_source(
        f'unrelated-{number}-{time.monotonic_ns()}', contact_id='unrelated', session_id='unrelated',
        messages=[{'role': 'user', 'content': 'Fixture'}], derive_claims=False)


@pytest.mark.parametrize('writer', ['none', 'one', '0.5', '0.05'])
async def test_context_assembly_ignores_ordinary_writer(annotated_app, monkeypatch, writer):
    from colonymind.api.routers import host
    from colonymind.memory.selection import RecallSelector
    app, ledger = annotated_app
    runs, writes, errors, cursors = [], [], [], []
    stop, start, written = threading.Event(), threading.Event(), threading.Event()
    version = getattr(curated, '_withdrawal_version', None)

    def cursor(*args):
        cursors.append(True)
        return version(*args)

    if version is not None:
        monkeypatch.setattr(curated, '_withdrawal_version', cursor)

    def sustained():
        start.wait()
        try:
            while not stop.is_set():
                ordinary_write(ledger.db_path, len(writes))
                writes.append(True)
                written.set()
                stop.wait(float(writer))
        except Exception as error:
            errors.append(error)
            written.set()

    class SlowSelector(RecallSelector):
        async def select_context(self, *args, **kwargs):
            runs.append(True)
            if len(runs) == 1:
                await asyncio.sleep(.075)
                if writer == 'one':
                    await asyncio.to_thread(ordinary_write, ledger.db_path, 0)
                    writes.append(True)
                elif writer != 'none':
                    start.set()
                    assert await asyncio.to_thread(written.wait, 5)
                await asyncio.sleep(.075)
            else:
                await asyncio.sleep(.15)
            return await super().select_context(*args, **kwargs)

    monkeypatch.setattr(host, '_context_recall_selector', (host._reranker, SlowSelector()))
    watermark = ledger.erasure_watermark('person')
    thread = threading.Thread(target=sustained) if writer not in {'none', 'one'} else None
    if thread:
        thread.start()
    before = time.perf_counter()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://fixture',
                               headers={'Authorization': 'Bearer ' + KEY}) as client:
            response = await client.post('/v1/host/context/assemble', json={
                'identity': {'host_id': 'fixture'},
                'context': {'contact_id': 'person', 'session_id': 'fresh'},
                'incoming_message': {'role': 'user', 'content': 'archive digest'}})
    finally:
        stop.set()
        start.set()
        if thread:
            await asyncio.to_thread(thread.join, 5)
            assert not thread.is_alive()
    data = response.json()
    memory = next((s for s in data.get('sections', []) if s['id'] == 'colonymind-memory'), None)
    if os.environ.get('FIX9_METRICS'):
        with open(os.environ['FIX9_METRICS'], 'a') as output:
            output.write(json.dumps(dict(writer=writer, runs=len(runs), writes=len(writes),
                sections=len(data.get('sections', [])), memory=bool(memory),
                wall_ms=round((time.perf_counter()-before)*1000, 3))) + '\n')
    assert not errors
    assert response.status_code == 200
    assert ledger.erasure_watermark('person') == watermark
    assert len(runs) == 1
    assert memory and REPORT in memory['body']
    assert not data['budget'].get('unavailable')
    if version is not None:
        assert len(cursors) == 2  # One outer envelope across sections, selection and allowance.
    assert bool(writes) is (writer != 'none')


@pytest.mark.parametrize('stage', ['selection', 'allowance', 'final_sections'])
async def test_mid_assembly_withdrawal_is_not_published(annotated_app, monkeypatch, stage):
    from colonymind.api.routers import host
    from colonymind.memory.selection import RecallSelector
    app, ledger = annotated_app
    fired = []
    original = getattr(host, '_memory_within_allowance' if stage == 'allowance' else '_turn_context_selection')

    async def forget():
        if not fired:
            await asyncio.to_thread(withdraw, ledger, 'report', 'person')
            fired.append(True)

    class WithdrawingSelector(RecallSelector):
        async def select_context(self, *args, **kwargs):
            selected = await super().select_context(*args, **kwargs)
            if stage == 'selection':
                await forget()
            return selected

    async def after(*args, **kwargs):
        result = await original(*args, **kwargs)
        await forget()
        return result

    monkeypatch.setattr(host, '_context_recall_selector', (host._reranker, WithdrawingSelector()))
    if stage != 'selection':
        monkeypatch.setattr(host, '_memory_within_allowance' if stage == 'allowance' else '_turn_context_selection', after)
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://fixture',
                           headers={'Authorization': 'Bearer ' + KEY}) as client:
        response = await client.post('/v1/host/context/assemble', json={
            'identity': {'host_id': 'fixture'},
            'context': {'contact_id': 'person', 'session_id': 'fresh'},
            'incoming_message': {'role': 'user', 'content': 'archive digest'}})
    assert response.status_code == 200
    assert fired
    assert ledger.erasure_watermark('person') > response.json()['source_erasure_watermark']
    assert response.json()['sections'] == []
    assert REPORT not in json.dumps(response.json())


@pytest.mark.parametrize('site', ['search', 'sections'])
async def test_standalone_hot_reader_rechecks_withdrawal(annotated_app, monkeypatch, site):
    from colonymind.api.routers import host
    from colonymind.memory import search, turn_budget
    app, ledger = annotated_app
    fired = []
    target, name = (search, '_select_memory') if site == 'search' else (host, '_assembled_lanes')
    original = getattr(target, name)

    async def after(*args, **kwargs):
        result = await original(*args, **kwargs)
        if not fired:
            material = result.content if site == 'search' else '\n'.join(s.body for s in result)
            assert REPORT in material  # The completed preparation actually carried the source.
            await asyncio.to_thread(withdraw, ledger, 'report', 'person')
            fired.append(True)
        return result

    monkeypatch.setattr(target, name, after)
    if site == 'search':
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://fixture',
                               headers={'Authorization': 'Bearer ' + KEY}) as client:
            response = await client.post('/v1/host/memory/search', json={
                'identity': {'host_id': 'fixture'}, 'person_id': 'person',
                'session_id': 'fresh', 'query': 'archive digest'})
        assert response.status_code == 200
        assert response.json()['content'] == ''
        assert response.json()['count'] == 0
    else:
        body = host.ContextAssembleRequest(identity={'host_id': 'fixture'},
            context={'contact_id': 'person', 'session_id': 'fresh'},
            incoming_message={'role': 'user', 'content': 'archive digest'})
        with turn_budget.turn(turn_budget.TurnBudget(wait_s=30.)):
            sections = await host._assemble_sections(body, viewer_person_id='person')
        assert REPORT not in '\n'.join(s.body for s in sections)
        assert not any(s.id == 'colonymind-memory' for s in sections)
    assert fired


@pytest.mark.parametrize('cursor', ['source', 'curated'])
async def test_withdrawal_fence_shares_retries_and_resets(traffic, cursor):
    ledger, _ = traffic
    outer, inner = [], []

    async def leaf():
        inner.append(True)
        if len(inner) == 1:
            if cursor == 'curated':
                await asyncio.to_thread(withdraw, ledger, 'first')
            else:
                # Exercise the source erasure cursor independently of curated events.
                def source_erasure():
                    with closing(ledger._connect()) as conn, conn:
                        ledger._append_erasure(conn, turn_id='first', contact_id=OWNER,
                            session_id='fixture-session', scope='person',
                            messages=[], removed=[], whole=True)
                await asyncio.to_thread(source_erasure)
        return len(inner)

    async def read():
        outer.append(True)
        return await curated.withdrawal_fenced_read(ledger, leaf, empty=0)

    result = await curated.withdrawal_fenced_read(ledger, read, empty=0)
    assert result == 2
    assert len(outer) == len(inner) == 2
    assert not curated._withdrawal_reads.get()
