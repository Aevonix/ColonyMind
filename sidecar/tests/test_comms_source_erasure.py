"""Source forgetting reaches exact communication summaries without deleting receipts."""
from datetime import datetime, timedelta, timezone
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock

from httpx import ASGITransport, AsyncClient
import pytest

from colonymind.api.routers import host
from colonymind.contacts.comms import CommsLog
from colonymind.turns import TurnIdempotencyLedger
from colonymind.turns.idempotency import SourceErased
from test_turn_source_evidence import source_app


def turn(identifier, text, *, contact='person'):
    return {'identity': {'host_id': 'fixture'}, 'context': {
        'turn_id': identifier, 'session_id': 'shared-session', 'contact_id': contact,
        'channel_id': 'sms:fixture'}, 'user_message': {'role': 'user', 'content': text},
        'assistant_message': {'role': 'assistant', 'content': 'Recorded: ' + text},
        'summary': text}


def linked_log(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'canonical.db')
    ledger.record_source('original', contact_id='person', session_id='session',
        messages=[{'role': 'user', 'content': 'Copper inventory has seven crates.'}], derive_claims=False)
    log = CommsLog(str(tmp_path / 'communications.db'), source_ledger=ledger)
    lineage, _ = log.source_input('original', 'person')
    return ledger, log, lineage


def test_unbound_foreign_profile_read_preserves_linked_rows(tmp_path, monkeypatch):
    import colonymind
    ledger, log, lineage = linked_log(tmp_path)
    log.log('person', summary='A fact owned by the original profile.', source_lineage=lineage)
    monkeypatch.setattr(colonymind, 'get_state_dir', lambda: tmp_path / 'unrelated-profile')
    reopened = CommsLog(log._db_path)
    for read in (lambda: reopened.history('person'), reopened.recent,
                 lambda: reopened.counts('person'), reopened.purge_erased_sources):
        with pytest.raises(ValueError, match='communications_source_ledger_required'):
            read()
        assert reopened._conn.execute('SELECT count(*) FROM communications').fetchone()[0] == 1
    assert ledger.source_references(['original'], contact_id='person', session_id='session')
    assert not (tmp_path / 'unrelated-profile').exists()
    legacy = CommsLog(str(tmp_path / 'legacy-unlinked.db'))
    legacy.log('person', summary='An unlinked historical entry.')
    assert legacy.history('person')[0]['summary'] == 'An unlinked historical entry.'
    assert legacy.counts('person')['inbound'] == 1


def test_prose_validation_follows_selected_rows_not_whole_contact(tmp_path, monkeypatch):
    _, log, lineage = linked_log(tmp_path)
    for index in range(100):
        log.log('person', channel='sms' if index % 2 else 'voice', direction='out',
                summary='A neutral summary.', source_lineage=lineage)
    selected_counts = []
    validate = log._invalid_sources
    def selected(rows, turn_ids=None):
        selected_counts.append(len(rows))
        return validate(rows, turn_ids)
    monkeypatch.setattr(log, '_invalid_sources', selected)
    assert len(log.history('person', limit=3)) == 3
    assert log.last_outbound('person') is not None
    assert len(log.last_per_channel('person')) == 2
    assert len(log.recent(limit=2)) == 2
    assert selected_counts == [3, 1, 2, 2]
    assert log._conn.execute('SELECT count(*) FROM communications').fetchone()[0] == 100


def test_reattribute_moves_every_exchange_of_a_merged_contact(tmp_path):
    log = CommsLog(str(tmp_path / 'communications.db'),
                   source_ledger=TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db'))
    log.log('dup', summary='Asked about the lease.', direction='in')
    log.log('dup', summary='Sent the lease.', direction='out')
    log.log('keep', summary='Said hello.', direction='in')
    assert log.reattribute('dup', 'keep') == 2
    assert log.counts('dup') == {'inbound': 0, 'outbound': 0, 'channels': 0}
    assert log.counts('keep')['inbound'] == 2 and log.counts('keep')['outbound'] == 1
    assert log.reattribute('dup', 'keep') == 0 and log.reattribute('keep', 'keep') == 0


def _count_erasure_reads(ledger, monkeypatch):
    """How often a read walks every erasure (the join over the projection tombstones)."""
    reads = []
    connect = ledger._connect

    def traced():
        conn = connect()
        conn.set_trace_callback(lambda sql: reads.append(sql) if 'source_projection_erasures p' in sql else None)
        return conn
    monkeypatch.setattr(ledger, '_connect', traced)
    return reads


from colonymind.memory import curated
@pytest.mark.asyncio
async def test_ordinary_capture_forget_removes_only_linked_summaries(source_app, tmp_path, monkeypatch):
    from colonymind.api.middleware import ApiKeyMiddleware
    from onekey import KEY
    source_app.add_middleware(ApiKeyMiddleware, api_key=KEY)
    log = CommsLog(str(tmp_path / 'communications.db'), source_ledger=TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db'))
    monkeypatch.setattr(host, '_comms_log', log)
    log.log('person', summary='An older unlinked note.', session_id='shared-session')
    log.log_receipt(event_id='receipt-event', contact_id='person', channel='sms:fixture', direction='out', external_ref='provider-message', receipt_ref='transport-receipt', occurred_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), status='accepted')
    async with AsyncClient(transport=ASGITransport(app=source_app), base_url='http://fixture', headers={'Authorization': 'Bearer '+KEY}) as client:
        for identifier, text in [('first', 'Copper inventory contains seven crates.'), ('second', 'The blue bicycle needs a new bell.')]:
            response = await client.put('/v2/host/turns/' + identifier, json=turn(identifier, text))
            assert response.status_code == 201 and response.json()['source_recorded']
        assert any(('Copper inventory' in row['summary'] for row in log.history('person')))
        erased = await client.post('/v1/host/memory/curated/sources/forget', json={'person_id': 'person', 'source_ids': ['first']})
        assert erased.status_code == 200 and erased.json()['source_erased']
        assert not any(('Copper inventory' in row['summary'] for row in log.history('person')))
        assert erased.json()['scope'] == 'curated_source_tombstone'
        assert log.unlinked_summary_count('person') == 1
        assert any(('blue bicycle' in row['summary'] for row in log.recent()))
        assert any(('older unlinked' in row['summary'] for row in log.history('person')))
        assert log.outbound_receipts(contact_id='person', outbound_ref='provider-message')
        with sqlite3.connect(log._db_path) as db:
            assert db.execute("SELECT 1 FROM communications WHERE summary LIKE '%Copper inventory%'").fetchall()
        retry = await client.put('/v2/host/turns/first', json=turn('first', 'Copper inventory contains seven crates.'))
        assert retry.json()['source_recorded'] is True
        reopened = CommsLog(log._db_path, source_ledger=log._source_ledger)
        assert not any(('Copper inventory' in row['summary'] for row in reopened.history('person')))

def test_reopened_read_reconciles_erasure_and_keeps_unlinked_other_person(tmp_path):
    ledger, log, lineage = linked_log(tmp_path)
    log.log('person', summary='Copper inventory has seven crates.', source_lineage=lineage)
    with pytest.raises(SourceErased):
        log.log('other', summary='Cannot borrow another contact source.', source_lineage=lineage)
    log.log('other', summary='Copper inventory has seven crates.')
    curated.forget_sources(ledger, 'person', source_ids=['original'])
    reopened = CommsLog(log._db_path, source_ledger=ledger)
    assert reopened.recent() == [{'contact_id': 'other', **row} for row in log.history('other')]
    assert reopened.history('person') == [] and reopened.last_per_channel('person') == {}
    assert reopened.last_outbound('person') is None
    with sqlite3.connect(log._db_path) as db:
        assert db.execute('SELECT count(*) FROM communications').fetchone()[0] == 2
    with pytest.raises(SourceErased):
        log.log('person', summary='A stale retry.', source_lineage=lineage)

def test_erasure_during_projection_write_is_reconciled_before_return(tmp_path, monkeypatch):
    ledger, log, lineage = linked_log(tmp_path)
    visible, calls = (log._source_visible, [])

    def check(person, reference):
        calls.append(person)
        if len(calls) == 2:
            curated.forget_sources(ledger, 'person', source_ids=['original'])
        return visible(person, reference)
    monkeypatch.setattr(log, '_source_visible', check)
    with pytest.raises(SourceErased):
        log.log('person', summary='Copper inventory has seven crates.', source_lineage=lineage)
    assert len(calls) == 2
    assert log._conn.execute('SELECT count(*) FROM communications').fetchone()[0] == 1
    assert log.history('person') == []

def test_parent_erasure_removes_linked_answer_summary_and_keeps_user_source(tmp_path):
    ledger, log, _ = linked_log(tmp_path)
    references = ledger.source_references(['original'], contact_id='person', session_id='later')
    ledger.record_source('answer', contact_id='person', session_id='later', derive_claims=False, messages=[{'role': 'user', 'content': 'Please summarize the inventory.'}, {'role': 'assistant', 'content': 'Seven crates are available.', '_supplied_sources': references}])
    answer_lineage, _ = log.source_input('answer', 'person')
    log.log('person', direction='out', summary='Seven crates are available.', source_lineage=answer_lineage)
    result = curated.forget_sources(ledger, 'person', source_ids=['original'])
    assert curated.is_forgotten(ledger, 'answer')
    assert log.history('person') == []
    with sqlite3.connect(ledger.db_path) as db:
        remaining = json.loads(db.execute("SELECT messages_json FROM turn_sources WHERE turn_id='answer'").fetchone()[0])
    assert len(remaining) == 2
    assert visible_source_messages(ledger, 'answer') == [{'role': 'user', 'content': 'Please summarize the inventory.'}]

def test_legacy_rows_are_not_backlinked_by_identical_summary_or_session(tmp_path):
    path = tmp_path / 'legacy.db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE communications(id TEXT PRIMARY KEY, contact_id TEXT NOT NULL,\n            channel TEXT, direction TEXT NOT NULL, summary TEXT, session_id TEXT, ts TEXT NOT NULL)')
        db.execute("INSERT INTO communications VALUES ('old','person','sms','in',?,'session','2026-01-01')", ('Copper inventory has seven crates.',))
    ledger, _, _ = linked_log(tmp_path)
    log = CommsLog(str(path), source_ledger=ledger)
    curated.forget_sources(ledger, 'person', source_ids=['original'])
    assert log.purge_erased_sources(['original'], contact_id='person') == 0
    assert log.unlinked_summary_count('person') == 1
    assert log.history('person')[0]['summary'] == 'Copper inventory has seven crates.'

from test_curated_memory import visible_source_messages


def test_metadata_aggregates_use_erasure_ids_without_hashing_source_text(tmp_path, monkeypatch):
    import colonymind.turns.idempotency as sources
    ledger, log, lineage = linked_log(tmp_path)
    log.log('person', direction='out', summary='A linked summary.', source_lineage=lineage)
    log.log('person', direction='in', summary='Independent unlinked history.')
    with monkeypatch.context() as guard:
        guard.setattr(sources, 'source_message_hash', lambda *a: pytest.fail('aggregate rehashed canonical text'))
        assert log.counts('person') == {'inbound': 1, 'outbound': 1, 'channels': 1}
        assert log.stats('person')['total'] == 2
        assert log.rollup()['unknown'] == {'in': 1, 'out': 1}
        assert len(log.inbound_since('person', '2020-01-01')) == 1
    historical_source_erasure(ledger, 'person', source_ids=['original'])
    monkeypatch.setattr(sources, 'source_message_hash', lambda *a: pytest.fail('aggregate rehashed canonical text'))
    assert log.counts('person') == {'inbound': 1, 'outbound': 0, 'channels': 1}
    assert log.stats('person')['total'] == 1
    assert log.rollup()['unknown'] == {'in': 1, 'out': 0}
    assert log._conn.execute('SELECT count(*) FROM communications').fetchone()[0] == 1

def test_reads_walk_the_erasures_only_when_they_changed(tmp_path, monkeypatch):
    """A read reconciles against every erasure, but walks the erasure tables again only after they grew: with
    many erased sources the full walk takes seconds, and every read (the per-turn landscape, the ledger view)
    reconciles."""
    ledger, log, lineage = linked_log(tmp_path)
    ledger.record_source('kept', contact_id='person', session_id='session', messages=[{'role': 'user', 'content': 'The blue bicycle needs a new bell.'}], derive_claims=False)
    kept, _ = log.source_input('kept', 'person')
    log.log('person', summary='Copper inventory has seven crates.', source_lineage=lineage)
    log.log('person', summary='The blue bicycle needs a new bell.', source_lineage=kept)
    reads = _count_erasure_reads(ledger, monkeypatch)
    for _ in range(3):
        assert len(log.history('person')) == 2 and len(log.recent()) == 2
        log.rollup()
    assert len(reads) == 1, 'unchanged erasures are walked once, not on every read'
    historical_source_erasure(ledger, 'person', source_ids=['original'])
    assert [row['summary'] for row in log.history('person')] == ['The blue bicycle needs a new bell.']
    assert len(reads) == 2, 'a new erasure is seen on the next read'
    assert [row['summary'] for row in log.recent()] == ['The blue bicycle needs a new bell.']
    assert len(reads) == 2

def test_a_different_ledger_is_walked_even_with_the_same_counts(tmp_path, monkeypatch):
    """Two ledgers with one erasure each have equal counters: rebinding the log to the other one is still seen."""
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    ledger_a, log, lineage_a = linked_log(first)
    ledger_b = TurnIdempotencyLedger(second / 'canonical.db')
    ledger_b.record_source('other-source', contact_id='person', session_id='session', messages=[{'role': 'user', 'content': 'The blue bicycle needs a new bell.'}], derive_claims=False)

    def stored():
        with sqlite3.connect(log._db_path) as db:
            return [row[0] for row in db.execute('SELECT summary FROM communications')]
    log.log('person', summary='Copper inventory has seven crates.', source_lineage=lineage_a)
    historical_source_erasure(ledger_a, 'person', source_ids=['original'])
    log.rollup()
    assert stored() == []
    log._source_ledger = ledger_b
    kept, _ = log.source_input('other-source', 'person')
    log.log('person', summary='The blue bicycle needs a new bell.', source_lineage=kept)
    historical_source_erasure(ledger_b, 'person', source_ids=['other-source'])
    log.rollup()
    assert stored() == []


def historical_source_erasure(ledger, person, *, source_ids):
    # The independent historical journal reconciler still owns its legacy receipts.
    with ledger._connect() as db:
        for source in source_ids:
            db.execute('INSERT INTO source_erasures(contact_id,turn_id,session_id,message_hashes_json) VALUES (?,?,?,?)',
                       (person, source, 'historical', '[]'))
