"""Keep the receipt table usable after rolling back to an earlier layout."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from colonymind.contacts.transport_ingress import TransportIngress, ensure_schema


EARLIER_SCHEMA = (Path(__file__).parent / 'fixtures' / 'transport_ingress_earlier_layout.sql').read_text()


def _earlier_admit(conn, sequence):
    """Exercise the explicit column INSERT from an earlier layout, not an assumed positional writer.

    Fields and INSERT match TransportIngress.admit in the fixture
    commit; fixed valid inputs avoid reproducing its unrelated validation.
    """
    receipt_id = f'earlier-receipt-{sequence}'
    fields = dict(producer='bridge', account_id='account', epoch='epoch', sequence=sequence,
                  event_id=f'provider-{sequence}', contact_id='contact', occurred_at=100.0,
                  journal_ref=f'epoch:{sequence}', payload_digest='a' * 64,
                  media_available=1, metadata_json='{}')
    now = 200.0
    columns = ','.join(fields)
    with conn:
        conn.execute(f'INSERT INTO transport_ingress (receipt_id,{columns},created_at) '
                     f'VALUES ({",".join("?" for _ in range(len(fields)+2))})',
                     (receipt_id, *fields.values(), now))
    return receipt_id


def _table_info(conn):
    return [tuple(row) for row in conn.execute('PRAGMA table_info(transport_ingress)')]


def test_settlement_migration_keeps_existing_rows_and_defaults_to_pending(tmp_path):
    with closing(sqlite3.connect(tmp_path / 'comms.db')) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(EARLIER_SCHEMA)
        receipt = _earlier_admit(conn, 1)
        conn.execute('''UPDATE transport_ingress SET state='completed',
            native_turn_json='{"session_id":"session","task_id":"task","turn_id":"turn"}',
            source_versions_json='{"turn":"version"}',outcome='captured',completed_at=200''')
        conn.commit()
        original = [tuple(row) for row in conn.execute('SELECT * FROM transport_ingress')]
        ensure_schema(conn)
        ensure_schema(conn)
        assert [tuple(row) for row in conn.execute('SELECT * FROM transport_ingress')] == original
        assert conn.execute('SELECT COUNT(*) FROM transport_ingress_settled').fetchone()[0] == 0
        store = TransportIngress(conn)
        assert not store.get(receipt)['metadata_settled']
        store.mark_settled(store.get(receipt))
        ensure_schema(conn)
        assert store.get(receipt)['metadata_settled']
        following = _earlier_admit(conn, 2)
        assert not store.get(following)['metadata_settled']


@pytest.mark.parametrize('store_kind', ['fresh', 'earlier', 'earlier_deadline_fix'])
def test_transport_ingress_schema_matches_earlier(tmp_path, store_kind):
    with closing(sqlite3.connect(':memory:')) as earlier, closing(sqlite3.connect(tmp_path / 'comms.db')) as conn:
        earlier.executescript(EARLIER_SCHEMA)
        if store_kind != 'fresh':
            conn.executescript(EARLIER_SCHEMA)
            if store_kind == 'earlier_deadline_fix':
                conn.execute('ALTER TABLE transport_ingress ADD COLUMN native_interrupted INTEGER NOT NULL DEFAULT 0')
            receipt = _earlier_admit(conn, 1)
        ensure_schema(conn)
        ensure_schema(conn)

        expected = _table_info(earlier)
        actual = _table_info(conn)
        assert [row[1] for row in actual] == [row[1] for row in expected]
        assert actual == expected
        if store_kind != 'fresh':
            assert conn.execute('SELECT receipt_id FROM transport_ingress').fetchone() == (receipt,)


def test_earlier_admission_and_full_restore_accept_kept_interrupted_store(tmp_path):
    path = tmp_path / 'comms.db'
    with closing(sqlite3.connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(EARLIER_SCHEMA)
        receipt = _earlier_admit(conn, 1)
        ensure_schema(conn)
        store = TransportIngress(conn)
        batch = dict(producer='bridge', receipt_ids=[receipt], batch_id='batch')
        native = dict(session_id='session', task_id='task', turn_id='turn')
        store.handoff(**batch)
        store.handoff(**batch, native_turn=native)
        assert store.native_settled(producer='bridge', native_turn=native, interrupted=True)['released'] == 1

    # A process using an earlier layout reopening the kept store admits fresh input without knowing
    # about interruption bookkeeping. Full restore can copy its exact row shape.
    with closing(sqlite3.connect(path)) as kept, closing(sqlite3.connect(':memory:')) as restored:
        following = _earlier_admit(kept, 2)
        restored.executescript(EARLIER_SCHEMA)
        assert _table_info(kept) == _table_info(restored)
        restored.execute('ATTACH DATABASE ? AS kept', (str(path),))
        restored.execute('INSERT INTO transport_ingress SELECT * FROM kept.transport_ingress')
        assert restored.execute('SELECT receipt_id,state FROM transport_ingress ORDER BY sequence').fetchall() == [
            (receipt, 'handed_off'), (following, 'admitted')]
