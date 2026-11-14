"""Coverage work drains past recently checked forecasts without schema changes."""

import json
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from colonymind.api.routers import transport_ingress_api as api
from colonymind.self_model import reply_forecasts as forecasts


@pytest.fixture
def coverage_store(monkeypatch):
    db = sqlite3.connect(':memory:', check_same_thread=False)
    db.execute('CREATE TABLE predictions (prediction_id TEXT PRIMARY KEY, detail TEXT, domain TEXT, '
               'source TEXT, outcome TEXT, horizon REAL, created_at REAL)')
    class Store:
        pass
    instance = Store()
    instance._conn, instance._lock = db, threading.Lock()
    monkeypatch.setattr(forecasts, '_parts', lambda: (instance,))
    calls = []
    monkeypatch.setattr(forecasts, 'reconcile', lambda wait_id, **kwargs: calls.append(wait_id))
    clock = [1000.]
    monkeypatch.setattr(forecasts, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    yield instance, calls, clock
    db.close()


def add_rows(store, count):
    for number in range(count):
        detail = {'conditions': {'wait_id': str(number), 'producer': 'provider', 'account_id': 'account',
                                'recipient_id': 'recipient', 'provider_external_ref': 'chat:message'},
                  'origin_at': 10}
        store._conn.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?,?)',
                            (str(number), json.dumps(detail), 'expected_reply', forecasts.METHOD,
                             'unresolved', 20, number))
    store._conn.commit()


def check_coverage(**kwargs):
    return forecasts.reconcile_existing(producer='provider',
        coverage={'connected': True, 'connected_since': 0, 'account_id': 'account', 'observed_at': 30},
        **kwargs)


def test_second_coverage_call_skips_all_100_recently_checked_rows(coverage_store):
    store, calls, clock = coverage_store
    add_rows(store, 100)
    assert check_coverage() == 100
    assert check_coverage() == 0
    assert len(calls) == 100
    clock[0] += 600
    assert check_coverage() == 100


def test_coverage_drains_older_rows_and_receipts_bypass_throttle(coverage_store):
    store, calls, clock = coverage_store
    add_rows(store, 101)
    assert check_coverage() == 100
    assert check_coverage() == 1
    assert set(calls) == {str(number) for number in range(101)}
    receipt = {'metadata_json': json.dumps({'reply_to_ref': 'message', 'channel': 'chat'}),
               'contact_id': 'recipient', 'producer': 'provider', 'account_id': 'account'}
    assert forecasts.reconcile_existing(receipt=receipt, strict=True) == 100


def test_failed_checks_are_visible_and_remain_retryable(coverage_store, monkeypatch, caplog):
    from colonymind.health_counters import snapshot

    store, calls, clock = coverage_store
    add_rows(store, 1)
    before = snapshot().get('context_reads_failed', 0)

    def fail(wait_id, **kwargs):
        store._conn.execute('SELECT * FROM missing_table')

    monkeypatch.setattr(forecasts, 'reconcile', fail)
    assert check_coverage() == 0
    assert check_coverage() == 0
    assert snapshot().get('context_reads_failed', 0) == before + 2
    assert all('OperationalError sqlite_errorname=SQLITE_ERROR' in record.getMessage()
               for record in caplog.records)
    with pytest.raises(sqlite3.OperationalError):
        check_coverage(strict=True)


@pytest.mark.asyncio
async def test_coverage_route_reconciles_in_worker_thread(monkeypatch):
    loop_thread = threading.get_ident()
    observed = []
    reconciled = []
    monkeypatch.setattr(api, 'producer', lambda request: 'provider')
    monkeypatch.setattr(api, 'store', lambda: SimpleNamespace(
        observe_coverage=lambda **kwargs: observed.append(kwargs)))
    monkeypatch.setattr(forecasts, 'reconcile_existing',
                        lambda **kwargs: reconciled.append(threading.get_ident()))
    body = api.Coverage(account_id='account', epoch='epoch', connected_since=1.,
                        observed_at=2., watermark=0, sequence_floor=0, connected=True, unavailable=0)
    result = await api.record_coverage(body, None)
    assert observed
    assert result['recorded'] is True
    assert len(reconciled) == 1 and reconciled[0] != loop_thread
