"""Recent ledger responses retain the pre-optimization erasure and window semantics."""
from contextlib import closing
from datetime import datetime, timedelta
import json
import sqlite3

import pytest

from colonymind.contacts.comms import CommsLog
from colonymind.turns import TurnIdempotencyLedger


class LegacyCommsLog(CommsLog):
    def _erased_pairs(self, turn_ids=()):
        # The original full-ledger query, independent of the optimized probes.
        with closing(self._ledger()._connect()) as conn:
            return frozenset(tuple(row) for row in conn.execute('''
                SELECT e.contact_id,coalesce(r.source_turn_id,e.turn_id)
                FROM source_erasures e LEFT JOIN source_erasure_revisions r USING(sequence)
                UNION SELECT e.contact_id,p.turn_id FROM source_erasures e
                LEFT JOIN source_erasure_revisions r USING(sequence)
                JOIN source_projection_erasures p ON p.source_turn_id=coalesce(r.source_turn_id,e.turn_id)
            '''))


def historical_source_erasure(ledger, contact_id, *, turn_ids):
    """Seed the historical journal without the retired destructive eraser API."""
    with closing(ledger._connect()) as conn, conn:
        for turn_id in turn_ids:
            conn.execute("""INSERT INTO source_erasures
                (contact_id,turn_id,session_id,message_hashes_json) VALUES (?,?,?,?)""",
                (contact_id, turn_id, 'historical', '[]'))


@pytest.fixture
def clock(monkeypatch):
    now = [datetime(2026, 10, 19, 12)]
    connect = sqlite3.connect

    def frozen_connect(*args, **kwargs):
        conn = connect(*args, **kwargs)
        conn.create_function('datetime', 2, lambda origin, modifier:
            (now[0] + timedelta(days=int(modifier.split()[0]))).isoformat(' '))
        return conn

    monkeypatch.setattr(sqlite3, 'connect', frozen_connect)
    return now


@pytest.mark.parametrize('limit', [0, 1, 5, 500, 999])
@pytest.mark.parametrize('days', [0, 30])
def test_snapshot_equivalence_across_writes_and_erasure(tmp_path, clock, limit, days):
    ledger = TurnIdempotencyLedger(tmp_path / 'sources.db')
    logs = [kind(str(tmp_path / filename), source_ledger=ledger)
            for kind, filename in [(LegacyCommsLog, 'old.db'), (CommsLog, 'new.db')]]

    def compare():
        assert logs[0].recent_snapshot(limit=limit, since_days=days) == logs[1].recent_snapshot(
            limit=limit, since_days=days)

    compare()
    for log in logs:
        for channel, direction, stamp in [('sms', 'in', '2026-09-18 12:00:00'),
                ('sms', 'out', '2026-09-19 12:00:00'), ('mail', 'in', '2026-10-19 11:59:59'),
                ('mail', 'out', '2026-10-19 12:00:00')]:
            log.log('person', channel=channel, direction=direction, ts=stamp, summary='Fixture')
    compare()
    for log in logs:
        log.log_receipt(event_id='receipt', contact_id='person', channel='sms', direction='out',
            external_ref='provider', receipt_ref='receipt', occurred_at='2026-10-18T12:00:00Z',
            status='accepted')
        log.reattribute('person', 'merged')
    compare()
    ledger.record_source('source', contact_id='merged', session_id='session',
                         messages=[{'role': 'user', 'content': 'Fixture'}], derive_claims=False)
    for log in logs:
        lineage, _ = log.source_input('source', 'merged')
        log.log('merged', channel='sms', ts='2026-10-19 12:00:01', source_lineage=lineage)
    compare()
    historical_source_erasure(ledger, 'merged', turn_ids=['source'])
    compare()
    clock[0] += timedelta(seconds=1)
    compare()
    clock[0] += timedelta(days=1)
    compare()


def test_window_moves_without_writes_and_preserves_text_timestamp_boundary(tmp_path, clock):
    log = CommsLog(str(tmp_path / 'comms.db'))
    for stamp in ['2026-09-19 12:00:00', '2026-09-19T00:00:00+00:00']:
        log.log('person', channel='sms', ts=stamp)
    assert log.recent_snapshot()['by_channel'] == {'sms': {'in': 2, 'out': 0}}
    clock[0] += timedelta(seconds=1)
    assert log.recent_snapshot()['by_channel'] == {'sms': {'in': 1, 'out': 0}}
    clock[0] += timedelta(days=1)
    result = log.recent_snapshot()
    assert result['by_channel'] == {} and len(result['entries']) == 2


def test_new_linked_turn_invalidates_cache_without_new_tombstones(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'sources.db')
    log = CommsLog(str(tmp_path / 'comms.db'), source_ledger=ledger)
    ledger.record_source('parent', contact_id='person', session_id='session',
        messages=[{'role': 'user', 'content': 'Fixture'}], derive_claims=False)
    ledger.record_source('kept', contact_id='person', session_id='session',
        messages=[{'role': 'user', 'content': 'Independent'}], derive_claims=False)
    lineage, _ = log.source_input('kept', 'person')
    log.log('person', source_lineage=lineage)
    historical_source_erasure(ledger, 'person', turn_ids=['parent'])
    with closing(ledger._connect()) as conn, conn:
        conn.execute("INSERT INTO source_projection_erasures VALUES ('late-projection','parent')")
    assert log.rollup()['unknown']['in'] == 1  # Cache predates the new communication.
    lineage['turn_id'] = 'late-projection'
    # Simulate a stale external writer restoring a previously erased projection.
    with sqlite3.connect(log._db_path) as conn:
        conn.execute('''INSERT INTO communications(id,contact_id,channel,direction,ts,source_lineage_json)
            VALUES ('late','person','sms','in','2026-10-19',?)''', (json.dumps(lineage),))
    assert log.rollup() == {'unknown': {'in': 1, 'out': 0}}
    assert log._conn.execute("SELECT count(*) FROM communications WHERE id='late'").fetchone()[0] == 0


def test_projection_erasure_keeps_other_contacts_and_unlinked_rows(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'sources.db')
    log = CommsLog(str(tmp_path / 'comms.db'), source_ledger=ledger)
    ledger.record_source('parent', contact_id='person', session_id='session',
        messages=[{'role': 'user', 'content': 'Fixture'}], derive_claims=False)
    historical_source_erasure(ledger, 'person', turn_ids=['parent'])
    with closing(ledger._connect()) as conn, conn:
        conn.execute("INSERT INTO source_projection_erasures VALUES ('answer','parent')")
    for contact in ['person', 'other']:
        with sqlite3.connect(log._db_path) as conn:
            conn.execute('''INSERT INTO communications(id,contact_id,direction,ts,source_lineage_json)
                VALUES (?,?,'in','2026-10-19',?)''',
                (contact, contact, json.dumps({'turn_id': 'answer'})))
    log.log('person', channel='sms')
    assert log.rollup() == {'unknown': {'in': 1, 'out': 0}, 'sms': {'in': 1, 'out': 0}}
    assert log._conn.execute("SELECT count(*) FROM communications WHERE id='other'").fetchone()[0] == 1
