"""Durable receipt settlement must be able to retry failed forecast work."""
import json
import sqlite3
from threading import RLock
from types import SimpleNamespace

import pytest

from colonymind.self_model import reply_forecasts as forecasts
from colonymind.api.routers import transport_ingress_api
from test_reply_forecasts import runtime, ingress, source_app, coverage, dispatch, reply, history
from test_transport_ingress_api import handoff, capture, status


@pytest.fixture
def matching_receipt(monkeypatch):
    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE predictions (prediction_id TEXT, domain TEXT, source TEXT, '
               'outcome TEXT, detail TEXT, created_at REAL)')
    conditions = {'wait_id': 'pending-wait', 'recipient_id': 'peer', 'producer': 'provider',
                  'account_id': 'neutral-account', 'provider_external_ref': 'whatsapp:provider-out'}
    db.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?)',
               ('prediction-one', 'expected_reply', forecasts.METHOD, 'pending',
                json.dumps({'conditions': conditions}), 1.0))
    store = SimpleNamespace(_conn=db, _lock=RLock())
    monkeypatch.setattr(forecasts, '_parts', lambda: (store,))
    yield {'metadata_json': json.dumps({'channel': 'whatsapp', 'reply_to_ref': 'provider-out'}),
           'contact_id': 'peer', 'producer': 'provider', 'account_id': 'neutral-account'}
    db.close()


def test_strict_receipt_reconciliation_propagates_wait_failure_and_retries(matching_receipt, monkeypatch):
    calls = []

    def reconcile(wait_id, *, strict=False):
        calls.append(wait_id)
        assert strict
        if len(calls) == 1:
            raise sqlite3.OperationalError('forecast store temporarily unavailable')

    monkeypatch.setattr(forecasts, 'reconcile', reconcile)
    with pytest.raises(sqlite3.OperationalError, match='temporarily unavailable'):
        forecasts.reconcile_existing(receipt=matching_receipt, strict=True)
    forecasts.reconcile_existing(receipt=matching_receipt, strict=True)
    assert calls == ['pending-wait', 'pending-wait']


def test_default_receipt_reconciliation_keeps_best_effort_behavior(matching_receipt, monkeypatch, caplog):
    calls = []

    def reconcile(wait_id, *, strict=False):
        calls.append(wait_id)
        assert not strict
        raise sqlite3.OperationalError('forecast store temporarily unavailable')

    monkeypatch.setattr(forecasts, 'reconcile', reconcile)
    forecasts.reconcile_existing(receipt=matching_receipt)
    assert calls == ['pending-wait']
    assert 'Declared reply forecast observation unavailable: OperationalError' in caplog.text


@pytest.mark.asyncio
async def test_receipt_settlement_retries_failed_forecast_evidence_read(runtime, monkeypatch):
    await coverage(runtime)
    await dispatch(runtime)
    runtime.now[0] += 20
    receipt = await reply(runtime, canonical=False)
    native = await handoff(runtime, receipt)

    def unavailable(*args):
        raise sqlite3.OperationalError('forecast evidence temporarily unavailable')

    with monkeypatch.context() as patch:
        patch.setattr(forecasts, '_parent_current', unavailable)
        response = await capture(runtime, native)
        assert response.status_code == 201, response.text
    store = transport_ingress_api.store()
    row = store.get(receipt['receipt_id'])
    assert row['state'] == 'completed'
    assert not row['metadata_settled']
    assert not history(runtime)['outcomes']

    assert (await status(runtime, receipt)).status_code == 200
    assert store.get(receipt['receipt_id'])['metadata_settled']
    assert history(runtime)['outcomes'][-1]['value'] is True
