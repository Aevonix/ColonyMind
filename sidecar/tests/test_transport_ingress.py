import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from colonymind.contacts.transport_ingress import (TransportIngress, adopt_retired_producers, ensure_schema,
                                                  retired_producer_rows)


def opened(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn, TransportIngress(conn)


def admit(store, sequence=1, **overrides):
    return store.admit(**(dict(producer='bridge', account_id='account', epoch='epoch', sequence=sequence,
        event_id='provider-' + str(sequence), contact_id='contact', occurred_at=100,
        journal_ref='epoch:' + str(sequence), payload_digest='a'*64, media_available=True,
        metadata={'channel': 'whatsapp', 'sender_ref': 'fixture'}, now=200) | overrides))


def test_two_connections_claim_once_and_native_source_settles_after_restart(tmp_path):
    path = tmp_path/'communications.db'
    conn, store = opened(path); ensure_schema(conn)
    receipt = admit(store)['receipt_id']; conn.close()
    barrier = Barrier(2)
    def claim():
        conn, store = opened(path)
        try:
            barrier.wait(timeout=5)
            return store.handoff(producer='bridge', receipt_ids=[receipt], batch_id='one')['may_dispatch']
        finally:
            conn.close()
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: claim(), range(2)))
    assert sorted(results) == [False, True]
    conn, store = opened(path)
    native = {'session_id': 'session', 'task_id': 'task', 'turn_id': 'source'}
    store.handoff(producer='bridge', receipt_ids=[receipt], batch_id='one', native_turn=native)
    assert store.for_canonical_turn(turn_id='source', contact_id='other') == []
    assert store.for_canonical_turn(turn_id='source', contact_id='contact')[0]['state'] == 'handed_off'
    store.complete(native_turn=native, source_versions={'source': 'v1'}, outcome='captured', now=210)
    assert store.get(receipt)['state'] == 'completed'
    with pytest.raises(ValueError, match='completion_conflict'):
        store.complete(native_turn=native, source_versions={'source': 'v2'}, outcome='captured')
    assert store.erase_sources(['source']) == [{'receipt_id': receipt, 'journal_ref': 'epoch:1'}]
    assert store.get(receipt)['state'] == 'erased'
    conn.close()


def test_lost_ack_repair_conflicts_and_erasure_before_canonical_completion(tmp_path):
    conn, store = opened(tmp_path/'comms.db'); ensure_schema(conn)
    receipt = admit(store, media_available=False)
    assert admit(store, media_available=False) == receipt
    with pytest.raises(ValueError, match='not_ready'):
        store.handoff(producer='bridge', receipt_ids=[receipt['receipt_id']], batch_id='one')
    repaired = admit(store, payload_digest='b'*64)
    assert repaired['media_available'] == 1
    with pytest.raises(ValueError, match='event_conflict'):
        admit(store, payload_digest='b'*64, contact_id='different')
    store.handoff(producer='bridge', receipt_ids=[receipt['receipt_id']], batch_id='one')
    native = {'session_id': 'session', 'task_id': 'task', 'turn_id': 'source'}
    store.handoff(producer='bridge', receipt_ids=[receipt['receipt_id']], batch_id='one', native_turn=native)
    assert store.erase_sources(['source'])
    store.complete(native_turn=native, source_versions={'source':'v1'}, outcome='captured')
    assert store.get(receipt['receipt_id'])['state'] == 'erased'
    conn.close()


def test_coverage_requires_connection_watermark_freshness_and_no_activity(tmp_path):
    conn, store = opened(tmp_path/'comms.db'); ensure_schema(conn)
    observation = dict(producer='bridge', account_id='account', epoch='epoch', connected_since=90,
        observed_at=120, watermark=0, connected=True, unavailable=0, now=120)
    store.observe_coverage(**observation)
    assert store.coverage(producer='bridge', account_id='account',contact_id='contact',since=100,now=121)['observed']
    assert not store.coverage(producer='bridge', account_id='account',contact_id='contact',since=80,now=121)['observed']
    store.observe_coverage(**(observation | {'watermark':1}))
    assert 'intake_gap' in store.coverage(producer='bridge',account_id='account',contact_id='contact',since=100,now=121)['reasons']
    admit(store)
    result = store.coverage(producer='bridge',account_id='account',contact_id='contact',since=100,now=121)
    assert result['reasons'] == ['recipient_activity_requires_review']
    assert store.coverage(producer='bridge',account_id='account',contact_id='other',since=100,now=121)['observed']
    assert not store.coverage(producer='bridge',account_id='account',contact_id='other',since=100,now=126)['observed']
    # A new, actually observed interval excludes older expired sequence gaps,
    # while an older since still cannot claim uninterrupted coverage.
    store.observe_coverage(**(observation | {'watermark':4,'sequence_floor':4,'connected_since':130,'observed_at':140,'now':140}))
    assert store.coverage(producer='bridge',account_id='account',contact_id='other',since=131,now=141)['observed']
    assert not store.coverage(producer='bridge',account_id='account',contact_id='other',since=120,now=141)['observed']
    with pytest.raises(ValueError,match='invalid_ingress_coverage'):
        store.observe_coverage(**(observation | {'sequence_floor':1}))
    conn.close()


@pytest.mark.parametrize('captured_before_interrupt', [False, True])
def test_interrupted_native_binding_allows_successor_without_replaying_input(tmp_path, captured_before_interrupt):
    conn, store = opened(tmp_path/'comms.db'); ensure_schema(conn)
    receipt = admit(store)['receipt_id']
    batch = dict(producer='bridge', receipt_ids=[receipt], batch_id='one')
    original = {'session_id': 'session', 'task_id': 'first-task', 'turn_id': 'first-turn'}
    following = dict(original, task_id='next-task', turn_id='next-turn')
    assert store.handoff(**batch)['may_dispatch'] is True
    store.handoff(**batch, native_turn=original)
    with pytest.raises(ValueError, match='native_turn_conflict'):
        store.handoff(**batch, native_turn=following)
    if captured_before_interrupt:
        store.complete(native_turn=original, source_versions={'first-turn': 'v1'}, outcome='captured')
    assert store.native_settled(producer='bridge', native_turn=original, interrupted=True)['released'] == 1
    conn.close()
    conn, store = opened(tmp_path/'comms.db')
    bound = store.handoff(**batch, native_turn=following)
    assert bound['native_bound'] and not bound['may_dispatch']
    assert bound['processing_complete'] is captured_before_interrupt
    assert store.handoff(**batch, native_turn=following) == bound
    assert store.native_settled(producer='bridge', native_turn=original, interrupted=True)['released'] == 0
    assert store.complete(native_turn=original, source_versions={'first-turn': 'v1'}, outcome='interrupted') == []
    if not captured_before_interrupt:
        assert store.get(receipt)['state'] == 'handed_off'
        store.complete(native_turn=following, source_versions={'next-turn': 'v1'}, outcome='captured')
    assert store.get(receipt)['state'] == 'completed'
    assert store.erase_sources(['first-turn']) == [{'receipt_id': receipt, 'journal_ref': 'epoch:1'}]
    assert store.get(receipt)['state'] == 'erased'
    conn.close()


def test_interrupted_binding_release_preserves_producer_session_and_replay_fences(tmp_path):
    conn, store = opened(tmp_path/'comms.db'); ensure_schema(conn)
    receipt = admit(store)['receipt_id']
    batch = dict(producer='bridge', receipt_ids=[receipt], batch_id='one')
    original = {'session_id': 'session', 'task_id': 'task', 'turn_id': 'first'}
    following = dict(original, turn_id='second')
    store.handoff(**batch); store.handoff(**batch, native_turn=original)
    assert store.native_settled(producer='other', native_turn=original, interrupted=True)['released'] == 0
    with pytest.raises(ValueError, match='native_turn_conflict'):
        store.handoff(**batch, native_turn=following)
    with pytest.raises(ValueError, match='interrupted_native_turn_required'):
        store.native_settled(producer='bridge', native_turn=original, interrupted=False)
    store.native_settled(producer='bridge', native_turn=original, interrupted=True)
    with pytest.raises(ValueError, match='native_turn_conflict'):
        store.handoff(**batch, native_turn=dict(following, session_id='other'))
    store.handoff(**batch, native_turn=following)
    store.native_settled(producer='bridge', native_turn=following, interrupted=True)
    with pytest.raises(ValueError, match='native_turn_conflict'):
        store.handoff(**batch, native_turn=original)
    assert json.loads(store.get(receipt)['native_turn_json']) == following
    conn.close()


@pytest.mark.parametrize('captured_before_interrupt', [False, True])
def test_column_based_interruptions_migrate_without_changing_receipts(tmp_path, captured_before_interrupt):
    path = tmp_path/'comms.db'
    conn, store = opened(path); ensure_schema(conn)
    original_columns = [row[1] for row in conn.execute('PRAGMA table_info(transport_ingress)')]
    receipt = admit(store)['receipt_id']
    untouched = admit(store, sequence=2)['receipt_id']
    batch = dict(producer='bridge', receipt_ids=[receipt], batch_id='one')
    native = {'session_id': 'session', 'task_id': 'task', 'turn_id': 'first'}
    store.handoff(**batch); store.handoff(**batch, native_turn=native)
    if captured_before_interrupt:
        store.complete(native_turn=native, source_versions={'first': 'v1'}, outcome='captured', now=210)
    before = [tuple(row) for row in conn.execute('SELECT * FROM transport_ingress ORDER BY receipt_id')]
    # Reproduce the schema and persisted flag from the first deadline-fix builds.
    with conn:
        conn.execute('DROP TABLE transport_ingress_interrupted')
        conn.execute('ALTER TABLE transport_ingress ADD COLUMN native_interrupted INTEGER NOT NULL DEFAULT 0')
        conn.execute('UPDATE transport_ingress SET native_interrupted=1 WHERE receipt_id=?', (receipt,))
    conn.close()

    conn, store = opened(path)
    ensure_schema(conn)
    marker = dict(conn.execute('SELECT * FROM transport_ingress_interrupted').fetchone())
    assert marker['receipt_id'] == receipt and json.loads(marker['native_turn_json']) == native
    assert marker['at'] > 0
    ensure_schema(conn)
    assert dict(conn.execute('SELECT * FROM transport_ingress_interrupted').fetchone()) == marker
    assert [row[1] for row in conn.execute('PRAGMA table_info(transport_ingress)')] == original_columns
    assert [tuple(row) for row in conn.execute('SELECT * FROM transport_ingress ORDER BY receipt_id')] == before
    assert store.get(untouched)['native_interrupted'] == 0
    assert store.get(receipt)['native_interrupted'] == 1
    following = dict(native, task_id='next-task', turn_id='next-turn')
    result = store.handoff(**batch, native_turn=following)
    assert not result['may_dispatch'] and result['processing_complete'] is captured_before_interrupt
    assert store.get(receipt)['native_interrupted'] == 0
    assert conn.execute('SELECT * FROM transport_ingress_interrupted').fetchall() == []
    assert store.erase_sources(['first']) == [{'receipt_id': receipt, 'journal_ref': 'epoch:1'}]
    conn.close()


def test_interruption_marker_is_exact_idempotent_and_consumed_atomically(tmp_path):
    conn, store = opened(tmp_path/'comms.db'); ensure_schema(conn)
    receipt = admit(store)['receipt_id']
    batch = dict(producer='bridge', receipt_ids=[receipt], batch_id='one')
    native = {'session_id': 'session', 'task_id': 'task', 'turn_id': 'first'}
    following = dict(native, task_id='next-task', turn_id='next-turn')
    store.handoff(**batch); store.handoff(**batch, native_turn=native)
    for _ in range(2):
        assert store.native_settled(producer='bridge', native_turn=native, interrupted=True)['released'] == 1
    assert conn.execute('SELECT COUNT(*) FROM transport_ingress_interrupted').fetchone()[0] == 1
    assert store.for_event(producer='bridge', account_id='account', epoch='epoch', sequence=1)['native_interrupted'] == 1
    assert store.for_canonical_turn(turn_id='first', contact_id='contact')[0]['native_interrupted'] == 1
    conn.execute("CREATE TRIGGER fail_handoff BEFORE UPDATE ON transport_ingress "
                 "BEGIN SELECT RAISE(ABORT, 'controlled handoff failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match='controlled handoff failure'):
        store.handoff(**batch, native_turn=following)
    assert store.get(receipt)['native_interrupted'] == 1
    assert conn.execute('SELECT * FROM transport_ingress_superseded').fetchall() == []
    conn.execute('DROP TRIGGER fail_handoff')
    store.handoff(**batch, native_turn=following)
    assert conn.execute('SELECT * FROM transport_ingress_interrupted').fetchall() == []
    # A kept side-table record cannot release a different tuple written with an earlier layout.
    with conn:
        conn.execute('INSERT INTO transport_ingress_interrupted VALUES (?,?,?)',
                     (receipt, json.dumps(native, sort_keys=True, separators=(',', ':')), 200))
    assert store.get(receipt)['native_interrupted'] == 0
    assert store.native_settled(producer='bridge', native_turn=native, interrupted=True)['released'] == 0
    with pytest.raises(ValueError, match='native_turn_conflict'):
        store.handoff(**batch, native_turn=dict(following, turn_id='third'))
    assert store.native_settled(producer='bridge', native_turn=following, interrupted=True)['released'] == 1
    assert store.get(receipt)['native_interrupted'] == 1
    conn.close()


def test_interruption_column_migration_rolls_back_on_failure_and_retries(tmp_path):
    conn, store = opened(tmp_path/'comms.db'); ensure_schema(conn)
    receipt = admit(store)['receipt_id']
    native = {'session_id': 'session', 'task_id': 'task', 'turn_id': 'first'}
    batch = dict(producer='bridge', receipt_ids=[receipt], batch_id='one')
    store.handoff(**batch); store.handoff(**batch, native_turn=native)
    with conn:
        conn.execute('ALTER TABLE transport_ingress ADD COLUMN native_interrupted INTEGER NOT NULL DEFAULT 0')
        conn.execute('UPDATE transport_ingress SET native_interrupted=1 WHERE receipt_id=?', (receipt,))
    conn.set_authorizer(lambda action, *_: sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_ALTER_TABLE
                        else sqlite3.SQLITE_OK)
    with pytest.raises(sqlite3.DatabaseError, match='not authorized'):
        ensure_schema(conn)
    conn.set_authorizer(None)
    assert conn.execute('SELECT native_interrupted FROM transport_ingress').fetchone()[0] == 1
    assert conn.execute('SELECT * FROM transport_ingress_interrupted').fetchall() == []
    ensure_schema(conn)
    assert 'native_interrupted' not in {row[1] for row in conn.execute('PRAGMA table_info(transport_ingress)')}
    assert store.get(receipt)['native_interrupted'] == 1
    conn.close()


def coverage(store, producer, account_id='account', observed_at=201, watermark=3):
    store.observe_coverage(producer=producer, account_id=account_id, epoch='epoch', connected_since=50,
                           observed_at=observed_at, watermark=watermark, connected=True, unavailable=0, now=observed_at)


def test_retired_producers_rows_are_adopted_by_the_instance_producer(tmp_path):
    """Rows stamped by principals that no longer exist move to the one producer the
    transport now presents: it reads, hands off and settles by the ids it journaled, a
    re-admit finds the adopted receipt, and the adopted coverage window is gap-free
    because every state moved. A second pass finds nothing."""
    conn, store = opened(tmp_path/'comms.db'); ensure_schema(conn)
    first, second, third = (admit(store, producer='retired-a', sequence=n) for n in (1, 2, 3))
    other = admit(store, producer='retired-b', account_id='other-account')
    store.handoff(producer='retired-a', receipt_ids=[second['receipt_id']], batch_id='old')
    native = {'session_id': 'session', 'task_id': 'task', 'turn_id': 'source-3'}
    store.handoff(producer='retired-a', receipt_ids=[third['receipt_id']], batch_id='old-3')
    store.handoff(producer='retired-a', receipt_ids=[third['receipt_id']], batch_id='old-3', native_turn=native)
    store.complete(native_turn=native, source_versions={'source-3': 'v1'}, outcome='captured', now=202)
    coverage(store, 'retired-a'); coverage(store, 'retired-b', account_id='other-account', watermark=1)
    ids = [first['receipt_id'], second['receipt_id'], third['receipt_id']]
    with pytest.raises(ValueError, match='scope_mismatch'):
        store.receipts(producer='api-key', receipt_ids=ids)
    with pytest.raises(ValueError, match='scope_mismatch'):
        store.handoff(producer='api-key', receipt_ids=[first['receipt_id']], batch_id='new')
    assert retired_producer_rows(conn, 'api-key') == {'receipts': 4, 'coverage': 2, 'producers': ['retired-a', 'retired-b']}

    assert adopt_retired_producers(conn, 'api-key') == {'receipts': 4, 'coverage': 2, 'kept': 0,
                                                         'producers': ['retired-a', 'retired-b']}
    assert [row['state'] for row in store.receipts(producer='api-key', receipt_ids=ids)] == ['admitted', 'handed_off', 'completed']
    assert store.handoff(producer='api-key', receipt_ids=[first['receipt_id']], batch_id='new')['may_dispatch'] is True
    assert store.receipts(producer='api-key', receipt_ids=[other['receipt_id']])[0]['state'] == 'admitted'
    checked = store.coverage(producer='api-key', account_id='account', contact_id='nobody', since=150, now=203)
    assert checked['observed'] and checked['watermark'] == 3
    assert admit(store, producer='api-key', sequence=1) == store.receipts(producer='api-key', receipt_ids=[first['receipt_id']])[0]
    assert conn.execute("SELECT COUNT(*) FROM transport_ingress").fetchone()[0] == 4
    assert retired_producer_rows(conn, 'api-key') == {'receipts': 0, 'coverage': 0, 'producers': []}
    assert adopt_retired_producers(conn, 'api-key') == {'receipts': 0, 'coverage': 0, 'kept': 0, 'producers': []}
    conn.close()


def test_adoption_keeps_what_the_instance_producer_already_holds(tmp_path):
    """An event the instance re-admitted before the upgrade keeps that receipt; the retired
    duplicate stays as it is and is not reported as pending. Coverage merges to the newest
    observation per account."""
    conn, store = opened(tmp_path/'comms.db'); ensure_schema(conn)
    old = admit(store, producer='retired')
    new = admit(store, producer='api-key')
    assert old['receipt_id'] != new['receipt_id']
    coverage(store, 'retired', observed_at=150, watermark=1)
    coverage(store, 'api-key', observed_at=160, watermark=1)
    coverage(store, 'retired', account_id='second', observed_at=170, watermark=1)
    coverage(store, 'api-key', account_id='second', observed_at=140, watermark=0)
    assert retired_producer_rows(conn, 'api-key') == {'receipts': 0, 'coverage': 2, 'producers': ['retired']}

    assert adopt_retired_producers(conn, 'api-key') == {'receipts': 0, 'coverage': 2, 'kept': 1, 'producers': ['retired']}
    assert store.get(old['receipt_id'])['producer'] == 'retired'
    assert admit(store, producer='api-key') == new
    rows = conn.execute('SELECT producer, account_id, observed_at FROM transport_ingress_coverage ORDER BY account_id').fetchall()
    assert [tuple(row) for row in rows] == [('api-key', 'account', 160.0), ('api-key', 'second', 170.0)]
    assert retired_producer_rows(conn, 'api-key') == {'receipts': 0, 'coverage': 0, 'producers': []}
    conn.close()
