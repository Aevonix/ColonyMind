"""Canonical forgetting must leave unrelated HTTP requests responsive."""
import asyncio
import threading
import time

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from colonymind.api.routers import host, curated_memory
from colonymind.memory import curated
from colonymind.turns import TurnIdempotencyLedger


@pytest.mark.asyncio
async def test_http_validation_progresses_during_real_canonical_erasure(tmp_path, monkeypatch, record_property):
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    ledger.record_source('diagnostic', contact_id='contact-fixture', session_id='fixture',
                         messages=[{'role': 'user', 'content': 'A disposable diagnostic fact.'}])
    monkeypatch.setattr('colonymind.turns.get_turn_idempotency_ledger', lambda _: ledger)
    monkeypatch.setattr('colonymind.vector.get_store', lambda: None)
    for name in ('_affect_store', '_comms_log'):
        monkeypatch.setattr(host, name, None)
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(tmp_path))
    entered, release = threading.Event(), threading.Event()
    read_rules = curated._resolve_sources
    worker_threads = set()

    def held_read(*args, **kwargs):
        worker_threads.add(threading.get_ident())
        if not entered.is_set():
            entered.set()
            assert release.wait(5), 'unrelated HTTP could not progress during canonical erasure'
        return read_rules(*args, **kwargs)

    monkeypatch.setattr(curated, '_resolve_sources', held_read)
    app = FastAPI()
    curated_memory.bind_ledger(app, ledger)
    app.include_router(curated_memory.router)
    from onekey import install_key, AUTH
    install_key(app)
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://fixture', headers=AUTH) as client:
        deletion = asyncio.create_task(client.post('/v1/host/memory/curated/sources/forget',
            json={'person_id': 'contact-fixture', 'source_ids': ['diagnostic']}))
        try:
            assert await asyncio.to_thread(entered.wait, 2), deletion.result().text if deletion.done() else 'erase did not start'
            assert not deletion.done()
            started = time.perf_counter()
            # Real route validation, while the real ledger's database read is
            # held. No deletion result or HTTP handler is substituted.
            response = await asyncio.wait_for(client.post('/v1/host/memory/curated/sources/forget', json={}), 1)
            record_property('unrelated_http_seconds', time.perf_counter() - started)
            assert response.status_code == 422
            assert not release.is_set() and not deletion.done()
        finally:
            release.set()
            result = await deletion
    assert result.status_code == 200 and result.json()['source_erased']
    assert threading.get_ident() not in worker_threads
    assert curated.is_forgotten(ledger, 'diagnostic')
    with ledger._connect() as conn:
        assert conn.execute("SELECT count(*) FROM turn_sources WHERE turn_id='diagnostic'").fetchone()[0] == 1
