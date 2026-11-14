"""Settled receipt pages still observe canonical erasure and validate scope."""

from contextlib import closing
from unittest.mock import Mock

import pytest

from colonymind.api.routers import transport_ingress_api as api
from test_transport_ingress_settlement import bound, poll, runtime


def erase_in_ledger(runtime, source_id, *, projection=False):
    # Model canonical erasure committing before its transport projection runs.
    with closing(runtime.ledger._connect()) as conn, conn:
        if projection:
            conn.execute('INSERT INTO source_projection_erasures VALUES (?,?)',
                         (source_id, 'erasure-cause'))
        else:
            conn.execute('INSERT INTO source_erasures '
                         '(contact_id,turn_id,session_id,message_hashes_json) VALUES (?,?,?,?)',
                         ('contact', source_id, 'session', '[]'))


@pytest.mark.asyncio
@pytest.mark.parametrize('projection', [False, True], ids=['source', 'projection-only'])
async def test_erasure_after_settlement_is_seen_without_forecast_retry(runtime, projection):
    receipt, native = bound(runtime, completed=True)
    assert (await poll(runtime, [receipt]))[0]['state'] == 'completed'
    runtime.forecasts.reset_mock()
    erase_in_ledger(runtime, native['turn_id'], projection=projection)
    if projection:
        with closing(runtime.ledger._connect()) as conn:
            assert conn.execute('SELECT count(*) FROM source_erasures').fetchone()[0] == 0

    erased = await poll(runtime, [receipt])

    assert erased[0]['state'] == 'erased'
    assert runtime.store.get(receipt)['contact_id'] is None
    assert runtime.store.get(receipt)['metadata_json'] == '{}'
    assert await poll(runtime, [receipt]) == erased
    runtime.forecasts.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('projection', [False, True], ids=['source', 'projection-only'])
async def test_superseded_source_erasure_after_completed_binding_replacement(runtime, projection):
    receipt, original = bound(runtime)
    batch = dict(producer='transport', receipt_ids=[receipt], batch_id='batch-1')
    captured = dict(original, task_id='captured-task', turn_id='captured-turn')
    following = dict(original, task_id='following-task', turn_id='following-turn')
    runtime.store.native_settled(producer='transport', native_turn=original, interrupted=True)
    runtime.store.handoff(**batch, native_turn=captured)
    runtime.store.complete(native_turn=captured,
                           source_versions={captured['turn_id']: 'version'}, outcome='captured')
    assert (await poll(runtime, [receipt]))[0]['state'] == 'completed'
    runtime.store.native_settled(producer='transport', native_turn=captured, interrupted=True)
    runtime.store.handoff(**batch, native_turn=following)
    assert (await poll(runtime, [receipt]))[0]['state'] == 'completed'
    runtime.forecasts.reset_mock()
    erase_in_ledger(runtime, original['turn_id'], projection=projection)

    assert (await poll(runtime, [receipt]))[0]['state'] == 'erased'
    runtime.forecasts.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('invalid', ['missing', 'other-producer'])
async def test_page_validates_every_id_before_any_settlement_mutation(runtime, monkeypatch, invalid):
    receipt, native = bound(runtime, completed=True)
    erase_in_ledger(runtime, native['turn_id'])
    if invalid == 'missing':
        invalid_receipt = 'missing-receipt'
    else:
        invalid_receipt = runtime.store.admit(producer='another-transport', account_id='account',
            epoch='epoch', sequence=2, event_id='event-2', contact_id='contact', occurred_at=100,
            journal_ref='journal:2', payload_digest='b' * 64, media_available=True,
            metadata={'channel': 'test'}, now=200)['receipt_id']
    before = runtime.store.get(receipt)
    settle = Mock(wraps=api.settle_receipt)
    erase = Mock(wraps=type(runtime.store).erase_sources)
    monkeypatch.setattr(api, 'settle_receipt', settle)
    monkeypatch.setattr(type(runtime.store), 'erase_sources', erase)

    response = await runtime.client.get('/ingress/receipts',
                                       params={'ids': ','.join([receipt, invalid_receipt])})

    assert response.status_code == 409, response.text
    assert response.json()['detail'] == 'ingress_receipt_scope_mismatch'
    assert runtime.store.get(receipt) == before
    settle.assert_not_called()
    erase.assert_not_called()
    runtime.forecasts.assert_not_called()
