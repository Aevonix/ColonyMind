"""Receipt polling only settles work that can still change."""

from types import SimpleNamespace
from unittest.mock import Mock
import sqlite3

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from colonymind.api.routers import host, transport_ingress_api as api
from colonymind.contacts.comms import CommsLog
from colonymind.contacts.transport_ingress import TransportIngress
from colonymind.self_model import reply_forecasts
from colonymind.turns import get_turn_idempotency_ledger


@pytest.fixture
async def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(tmp_path))
    comms = CommsLog(str(tmp_path / 'communications.db'))
    monkeypatch.setattr(host, '_comms_log', comms)
    monkeypatch.setattr(api, 'producer', lambda request: 'transport')
    forecasts = Mock()
    monkeypatch.setattr(reply_forecasts, 'reconcile_existing', forecasts)
    ledger = get_turn_idempotency_ledger(tmp_path)
    connect = Mock(wraps=ledger._connect)
    monkeypatch.setattr(ledger, '_connect', connect)
    app = FastAPI()
    app.include_router(api.router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://fixture') as client:
        value = SimpleNamespace(comms=comms, store=TransportIngress(comms._conn),
                                ledger=ledger, connect=connect, forecasts=forecasts, client=client)
        yield value
    value.comms._conn.close()


def bound(runtime, sequence=1, *, completed=False):
    store = runtime.store
    row = store.admit(producer='transport', account_id='account', epoch='epoch',
        sequence=sequence, event_id=f'event-{sequence}', contact_id='contact',
        occurred_at=100, journal_ref=f'journal:{sequence}', payload_digest='a' * 64,
        media_available=True, metadata={'channel': 'test'}, now=200)
    native = dict(session_id='session', task_id=f'task-{sequence}', turn_id=f'turn-{sequence}')
    batch = dict(producer='transport', receipt_ids=[row['receipt_id']], batch_id=f'batch-{sequence}')
    store.handoff(**batch)
    store.handoff(**batch, native_turn=native)
    if completed:
        store.complete(native_turn=native, source_versions={native['turn_id']: 'version'}, outcome='captured')
    return row['receipt_id'], native


async def poll(runtime, ids):
    response = await runtime.client.get('/ingress/receipts', params={'ids': ','.join(ids)})
    assert response.status_code == 200, response.text
    return response.json()['items']


@pytest.mark.asyncio
async def test_unchanged_settled_page_does_no_settlement_work(runtime, monkeypatch):
    ids = [bound(runtime, sequence, completed=True)[0] for sequence in range(1, 101)]
    first = await poll(runtime, ids)
    assert runtime.forecasts.call_count == 100
    runtime.forecasts.reset_mock()
    runtime.connect.reset_mock()
    settle = Mock(wraps=api.settle_receipt)
    monkeypatch.setattr(api, 'settle_receipt', settle)
    statements = []
    connect = sqlite3.connect

    def traced_connect(path, *args, **kwargs):
        conn = connect(path, *args, **kwargs)
        if path == runtime.comms._db_path:
            conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(sqlite3, 'connect', traced_connect)
    assert await poll(runtime, ids) == first
    assert settle.call_count == 0
    assert runtime.forecasts.call_count == 0
    assert runtime.connect.call_count == 0
    assert len([sql for sql in statements if sql.startswith('SELECT')]) == 1


def capture(runtime, native):
    runtime.ledger.record_source(native['turn_id'], contact_id='contact', session_id=native['session_id'],
        messages=[{'role': 'user', 'content': 'A neutral source.'}], derive_claims=False)
    runtime.ledger.reserve(native['turn_id'], 'b' * 64)
    runtime.ledger.complete(native['turn_id'], 'b' * 64, {'accepted': True, 'source_recorded': True})


@pytest.mark.asyncio
async def test_receipt_completing_between_polls_is_settled_once(runtime):
    receipt, native = bound(runtime)
    assert (await poll(runtime, [receipt]))[0]['state'] == 'handed_off'
    runtime.forecasts.assert_not_called()
    capture(runtime, native)
    completed = (await poll(runtime, [receipt]))[0]
    assert completed['state'] == 'completed'
    assert completed['source_versions_json']
    assert runtime.forecasts.call_count == 1
    assert await poll(runtime, [receipt]) == [completed]
    assert runtime.forecasts.call_count == 1


@pytest.mark.asyncio
async def test_failed_completion_is_retried(runtime, monkeypatch):
    receipt, native = bound(runtime)
    capture(runtime, native)
    original = TransportIngress.complete
    with monkeypatch.context() as patch:
        patch.setattr(TransportIngress, 'complete', Mock(side_effect=sqlite3.OperationalError('retry')))
        response = await runtime.client.get('/ingress/receipts', params={'ids': receipt})
        assert response.status_code == 503
    assert TransportIngress.complete is original
    assert runtime.store.get(receipt)['state'] == 'handed_off'
    assert (await poll(runtime, [receipt]))[0]['state'] == 'completed'
    assert runtime.forecasts.call_count == 1


@pytest.mark.asyncio
async def test_failed_forecast_settlement_is_retried(runtime):
    receipt, _ = bound(runtime, completed=True)
    runtime.forecasts.side_effect = [RuntimeError('retry'), None]
    assert (await poll(runtime, [receipt]))[0]['state'] == 'completed'
    assert not runtime.store.get(receipt)['metadata_settled']
    await poll(runtime, [receipt])
    assert runtime.store.get(receipt)['metadata_settled']
    await poll(runtime, [receipt])
    assert runtime.forecasts.call_count == 2


@pytest.mark.asyncio
async def test_failed_marker_write_is_retried(runtime, monkeypatch):
    receipt, _ = bound(runtime, completed=True)
    with monkeypatch.context() as patch:
        patch.setattr(TransportIngress, 'mark_settled', Mock(side_effect=sqlite3.OperationalError('retry')))
        response = await runtime.client.get('/ingress/receipts', params={'ids': receipt})
        assert response.status_code == 503
    assert not runtime.store.get(receipt)['metadata_settled']
    await poll(runtime, [receipt])
    await poll(runtime, [receipt])
    assert runtime.forecasts.call_count == 2


@pytest.mark.asyncio
async def test_settlement_survives_reopening_sidecar_store(runtime, monkeypatch):
    receipt, _ = bound(runtime, completed=True)
    first = await poll(runtime, [receipt])
    path = runtime.comms._db_path
    runtime.comms._conn.close()
    runtime.comms = CommsLog(path)
    runtime.store = TransportIngress(runtime.comms._conn)
    monkeypatch.setattr(host, '_comms_log', runtime.comms)
    runtime.forecasts.reset_mock()
    runtime.connect.reset_mock()
    assert await poll(runtime, [receipt]) == first
    runtime.forecasts.assert_not_called()
    runtime.connect.assert_not_called()


@pytest.mark.asyncio
async def test_changed_completed_native_binding_is_settled_again(runtime):
    receipt, native = bound(runtime, completed=True)
    await poll(runtime, [receipt])
    runtime.store.native_settled(producer='transport', native_turn=native, interrupted=True)
    following = dict(native, task_id='following-task', turn_id='following-turn')
    runtime.store.handoff(producer='transport', receipt_ids=[receipt], batch_id='batch-1', native_turn=following)
    await poll(runtime, [receipt])
    await poll(runtime, [receipt])
    assert runtime.forecasts.call_count == 2


@pytest.mark.asyncio
async def test_receipt_attachment_cannot_write_source_ledger(runtime, monkeypatch):
    receipt, _ = bound(runtime, completed=True)
    original = TransportIngress._attach_erasures
    checked = []

    def check_attachment(ingress, ledger):
        original(ingress, ledger)
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            ingress.conn.execute('DELETE FROM ingress_sources.source_erasures')
        checked.append(True)

    monkeypatch.setattr(TransportIngress, '_attach_erasures', check_attachment)
    await poll(runtime, [receipt])
    assert checked == [True]
