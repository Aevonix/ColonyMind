"""Offline regression coverage for the installed transport-receipt worker."""


import asyncio
import sqlite3
import threading
import time

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from colonymind.api.routers import host, transport_ingress_api as api
from colonymind.contacts.comms import CommsLog
from colonymind.contacts.transport_ingress import TransportIngress
from colonymind.turns.idempotency import get_turn_idempotency_ledger


@pytest.mark.parametrize('slow_part', ['receipt_rows', 'settle_receipt'])
async def test_receipt_work_yields_to_loop_and_closes_its_own_connection(receipts_env, monkeypatch, slow_part):
    comms, ids, native, ledger, app = receipts_env
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    owner = TransportIngress if slow_part == 'receipt_rows' else api
    original = getattr(owner, slow_part)
    workers, connections = [], []

    def slow(*args, **kwargs):
        workers.append(threading.get_ident())
        acknowledged = threading.Event()
        loop.call_soon_threadsafe(acknowledged.set)
        assert acknowledged.wait(5), 'receipt work prevented the loop from responding'
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, slow_part, slow)
    rows = TransportIngress.receipt_rows

    def observed_rows(self, **kwargs):
        connections.append(self.conn)
        return rows(self, **kwargs)

    monkeypatch.setattr(TransportIngress, 'receipt_rows', observed_rows)
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://synthetic') as client:
        response = await client.get('/ingress/receipts', params={'ids': ids[0]})
    assert response.status_code == 200, response.text
    assert workers and all(worker != loop_thread for worker in workers)
    assert connections and all(conn is not comms._conn for conn in connections)
    for conn in connections:
        with pytest.raises(sqlite3.ProgrammingError, match='closed'):
            conn.interrupt()


@pytest.fixture
def receipts_env(tmp_path, monkeypatch):
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(tmp_path))
    comms = CommsLog(str(tmp_path/'communications.db'))
    monkeypatch.setattr(host, '_comms_log', comms)
    monkeypatch.setattr(api, 'producer', lambda request: 'synthetic-producer')
    ingress = comms._transport_ingress
    ids = []
    for index in range(1, 4):
        row = ingress.admit(producer='synthetic-producer', account_id='account', epoch='epoch',
            sequence=index, event_id=f'event-{index}', contact_id='person',
            occurred_at=time.time()-1, journal_ref=f'journal-{index}', payload_digest='a'*64,
            media_available=True, metadata={'channel':'whatsapp', 'sender_ref':'synthetic'})
        ids.append(row['receipt_id'])
    native = {'session_id':'session', 'task_id':'task', 'turn_id':'source'}
    ingress.handoff(producer='synthetic-producer', receipt_ids=ids, batch_id='batch')
    ingress.handoff(producer='synthetic-producer', receipt_ids=ids, batch_id='batch', native_turn=native)
    ledger = get_turn_idempotency_ledger(tmp_path)
    app = FastAPI()
    app.include_router(api.router)
    yield comms, ids, native, ledger, app
    comms._conn.close()

async def test_order_settlement_once_and_one_hop(receipts_env, monkeypatch):
    comms, ids, native, ledger, app = receipts_env
    hops = []
    settlements = []
    real_thread = asyncio.to_thread
    async def tracked(*args, **kwargs):
        hops.append(1)
        return await real_thread(*args, **kwargs)
    monkeypatch.setattr(asyncio, 'to_thread', tracked)
    def settle(ingress, row):
        settlements.append(row['receipt_id'])
        ingress.complete(native_turn=native, source_versions={'source':'version'}, outcome='captured')
        for receipt in ids:
            ingress.mark_settled(ingress.get(receipt))
    monkeypatch.setattr(api, 'settle_receipt', settle)
    selected = [ids[2], ids[0], ids[2], ids[1]]
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://synthetic') as client:
        response = await client.get('/ingress/receipts', params={'ids':','.join(selected)})
    assert response.status_code == 200, response.text
    assert [row['receipt_id'] for row in response.json()['items']] == selected
    assert all(row['state'] == 'completed' for row in response.json()['items'])
    assert settlements == [ids[2]]
    assert hops == [1]
