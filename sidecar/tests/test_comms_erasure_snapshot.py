"""Erasure probes cover selected rows, and aggregates cannot admit unchecked writes."""
from contextlib import closing
import json
import sqlite3

import pytest

from colonymind.contacts.comms import CommsLog
from colonymind.turns import TurnIdempotencyLedger
from test_comms_recent_snapshot import historical_source_erasure


def erased_projection(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'sources.db')
    historical_source_erasure(ledger, 'person', turn_ids=['parent'])
    ledger.record_source('kept', contact_id='person', session_id='session',
                         messages=[{'role': 'user', 'content': 'Retained fixture'}], derive_claims=False)
    with closing(ledger._connect()) as conn, conn:
        conn.execute("INSERT INTO source_projection_erasures VALUES ('answer','parent')")
    log = CommsLog(str(tmp_path / 'comms.db'), source_ledger=ledger)
    lineage, _ = log.source_input('kept', 'person')
    with log._conn:
        for turn in ('answer', 'kept'):
            lineage = {**lineage, 'turn_id': turn}
            log._conn.execute('''INSERT INTO communications
                (id,contact_id,channel,direction,ts,source_lineage_json)
                VALUES (?,'person','sms','in','2099-01-01',?)''',
                (turn, json.dumps(lineage)))
    return log


@pytest.mark.parametrize('cached', [False, True])
def test_delete_restore_during_probe_covers_original_selection(tmp_path, monkeypatch, cached):
    """The exact races.py interleaving, isolating reconciliation's selected ids."""
    log = erased_projection(tmp_path)
    if cached:
        log._erased_pairs(['kept'])
    probe = log._erased_pairs

    def interleaved(*args):
        with sqlite3.connect(log._db_path) as writer:
            row = writer.execute("SELECT * FROM communications WHERE id='answer'").fetchone()
            writer.execute("DELETE FROM communications WHERE id='answer'")
        erased = probe(*args)
        with sqlite3.connect(log._db_path) as writer:
            writer.execute('INSERT INTO communications VALUES (' + ','.join('?' for _ in row) + ')', row)
        return erased

    monkeypatch.setattr(log, '_erased_pairs', interleaved)
    # Deliberately omit the outer aggregate transaction to reproduce both writes.
    with log._conn:
        log._reconcile_erasure_ids()
    assert log._conn.execute('SELECT id FROM communications').fetchall()[0][0] == 'kept'
    assert log._conn.execute('SELECT count(*) FROM communications').fetchone()[0] == 1


def test_cache_requires_every_selected_turn(tmp_path):
    log = erased_projection(tmp_path)
    assert ('person', 'answer') not in log._erased_pairs(['kept'])
    assert ('person', 'answer') in log._erased_pairs(['answer'])
    assert log._erased[2] == frozenset(['answer'])


def test_producer_probes_lineage_from_selected_rows(tmp_path, monkeypatch):
    from colonymind.memory import producers
    log = erased_projection(tmp_path)
    monkeypatch.setattr(producers, 'active', lambda: True)
    monkeypatch.setattr(log, '_invalid_sources', lambda rows: [])
    probe = log._erased_pairs
    selected = []

    def interleaved(turn_ids):
        selected.extend(turn_ids)
        with sqlite3.connect(log._db_path) as writer:
            writer.execute("DELETE FROM communications WHERE id='answer'")
        erased = probe(selected)
        with sqlite3.connect(log._db_path) as writer:
            writer.execute('''INSERT INTO communications
                (id,contact_id,direction,ts,source_lineage_json)
                VALUES ('answer','person','in','2099-01-01',?)''',
                (json.dumps({'turn_id': 'kept'}),))
        return erased

    monkeypatch.setattr(log, '_erased_pairs', interleaved)
    assert len(log.history('person')) == 1
    assert set(selected) == {'answer', 'kept'}


@pytest.mark.parametrize('method', ['rollup', 'counts', 'stats', 'inbound_since', 'recent'])
def test_aggregate_reserves_writer_until_rows_are_read(tmp_path, monkeypatch, method):
    log = erased_projection(tmp_path)
    reconcile = log._reconcile_erasure_ids
    attempts = []

    def checked(contact_id=None):
        reconcile(contact_id)
        # A stale writer would restore the erased row after cleanup and before
        # aggregation. The reservation must last beyond cleanup's DELETE.
        with sqlite3.connect(log._db_path, timeout=0) as writer:
            with pytest.raises(sqlite3.OperationalError, match='locked'):
                writer.execute('''INSERT INTO communications
                    (id,contact_id,direction,ts,source_lineage_json)
                    VALUES ('late','person','in','2099-01-01',?)''',
                    (json.dumps({'turn_id': 'answer'}),))
        attempts.append(True)

    monkeypatch.setattr(log, '_reconcile_erasure_ids', checked)
    args = () if method in ('rollup', 'recent') else ('person',)
    if method == 'inbound_since':
        args += ('2000-01-01',)
    result = getattr(log, method)(*args)
    assert attempts == [True]
    assert result
    assert not log._conn.in_transaction
    # Writers proceed once the result is materialized.
    with sqlite3.connect(log._db_path, timeout=0) as writer:
        writer.execute("INSERT INTO communications(id,contact_id,direction,ts) VALUES ('ok','person','in','2099-01-01')")
