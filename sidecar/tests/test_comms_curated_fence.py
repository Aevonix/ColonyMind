"""Communication metadata must not give withdrawn evidence new authority."""
from contextlib import closing
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from colonymind.contacts.comms import CommsLog
from colonymind.memory import curated, producers, turn_budget
from colonymind.memory.turn_budget import TurnBudget
from colonymind.turns import TurnIdempotencyLedger


@pytest.fixture
def traffic(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'sources.db')
    log = CommsLog(str(tmp_path / 'comms.db'), source_ledger=ledger)
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    for name, hour, person in [('withdraw', 1, 'person'), ('kept', 48, 'person'),
                               ('other', 2, 'other')]:
        ledger.record_source(name, contact_id=person, session_id=name,
                             messages=[{'role': 'user', 'content': name}], derive_claims=False)
        lineage, _ = log.source_input(name, person)
        stamp = (start + timedelta(hours=hour)).isoformat()
        log.log(person, ts=stamp, source_lineage=lineage, external_ref=name,
                reply_to_ref='delivery', receipt_ref='receipt:' + name,
                reaction='accepted', summary=name)
        log.log(person, direction='out', ts=stamp, source_lineage=lineage,
                external_ref='mind:reach_out:' + name, receipt_ref='sent:' + name,
                summary=name)
    refs = ledger.source_references(['withdraw'], contact_id='person', session_id='dependent')
    ledger.record_source('dependent', contact_id='person', session_id='dependent',
                         messages=[{'role': 'assistant', 'content': 'Derived response',
                                    '_supplied_sources': refs}], derive_claims=False)
    lineage, _ = log.source_input('dependent', 'person')
    log.log('person', ts=(start + timedelta(hours=2)).isoformat(),
            source_lineage=lineage, summary='Derived response')
    log.log('person', ts=(start + timedelta(hours=72)).isoformat(), summary='Unlinked history')
    return ledger, log, start


def native_rows(log):
    return [tuple(row) for row in log._conn.execute('SELECT * FROM communications ORDER BY id')]


@pytest.mark.parametrize('restart', [False, True])
@pytest.mark.asyncio
async def test_metadata_fences_withdrawals_on_writer_and_producer(traffic, monkeypatch, restart):
    ledger, log, start = traffic
    since = start.isoformat()
    assert len(log.inbound_since('person', since)) == 4
    assert len(log.mind_sends('person')) == 2
    before = native_rows(log)
    curated.forget_sources(ledger, 'person', source_ids=['withdraw'])
    if restart:
        ledger = TurnIdempotencyLedger(ledger.db_path)
        log = CommsLog(log._db_path, source_ledger=ledger)
    assert curated.is_forgotten(ledger, 'dependent')

    def read():
        # The actual worker connection must stay read-only; any cleanup fails.
        if producers.active():
            assert log._conn.execute('PRAGMA query_only').fetchone()[0] == 1
        return (log.inbound_since('person', since), log.mind_sends('person'),
                log.outbound_receipts(contact_id='person', outbound_ref='mind:reach_out:withdraw'),
                log.outbound_receipts(contact_id='person', outbound_ref='mind:reach_out:kept'),
                log.match_reply(contact_id='person', outbound_ref='delivery', since_iso=since,
                                until_iso=(start + timedelta(days=4)).isoformat()))

    expected = [(start + timedelta(hours=h)).isoformat() for h in (48, 72)]
    direct = read()
    with turn_budget.turn(TurnBudget(wait_s=7.5)):
        worker = await producers.read(read)
    assert worker == direct
    assert direct[0] == expected
    assert [row['intention_id'] for row in direct[1]] == ['kept']
    assert direct[2] == [] and len(direct[3]) == 1
    assert [row['external_ref'] for row in direct[4]['matches']] == ['kept']
    with closing(log.read_connection()) as connection:
        assert log.match_reply(contact_id='person', outbound_ref='delivery', since_iso=since,
                               until_iso=(start + timedelta(days=4)).isoformat(),
                               connection=connection) == direct[4]
    # Metadata paths bypass the prose projection validator, while still using
    # the shared curated fence for selected evidence and its dependencies.
    with monkeypatch.context() as guard:
        guard.setattr(log, '_invalid_sources', lambda *a: pytest.fail('metadata validated prose'))
        assert log.inbound_since('person', since) == expected
        assert [row['intention_id'] for row in log.mind_sends('person')] == ['kept']
    assert log.inbound_since('person', expected[-1]) == [expected[-1]]
    assert len(log.inbound_since('other', since)) == 1
    assert native_rows(log) == before
    assert ('person', 'withdraw') not in log._erased_pairs(['withdraw', 'dependent'])
    assert log.purge_erased_sources(contact_id='person') == 0
    assert native_rows(log) == before


@pytest.mark.asyncio
async def test_producer_does_not_delete_legacy_erasure_rows(traffic):
    ledger, log, start = traffic
    with ledger._connect() as conn:
        conn.execute('INSERT INTO source_erasures(contact_id,turn_id,session_id,message_hashes_json) '
                     'VALUES (?,?,?,?)', ('person', 'withdraw', 'withdraw', '[]'))
    before = native_rows(log)
    with turn_budget.turn(TurnBudget(wait_s=7.5)):
        stamps = await producers.read(log.inbound_since, 'person', start.isoformat())
    assert (start + timedelta(hours=1)).isoformat() not in stamps
    assert native_rows(log) == before


@pytest.mark.parametrize('restart', [False, True])
@pytest.mark.asyncio
async def test_affect_events_counts_and_observation_clock_withhold_without_deleting(traffic, tmp_path, restart):
    from colonymind.contacts.affect import AffectStore
    ledger, log, start = traffic
    affect = AffectStore(str(tmp_path / 'affect.db'), source_ledger=ledger)
    lineage, _ = log.source_input('withdraw', 'person')
    old = affect.create_event(contact_id='person', valence=0.2,
                             timestamp=start.isoformat())
    withdrawn = affect.create_event(contact_id='person', valence=-0.8,
        timestamp=(start + timedelta(hours=1)).isoformat(), source_lineage=lineage)
    before = [tuple(row) for row in affect._conn.execute('SELECT * FROM affect_events ORDER BY id')]
    curated.forget_sources(ledger, 'person', source_ids=['withdraw'])
    if restart:
        affect.close()
        affect = AffectStore(str(tmp_path / 'affect.db'),
                             source_ledger=TurnIdempotencyLedger(ledger.db_path))

    def read():
        if producers.active():
            assert affect._conn.execute('PRAGMA query_only').fetchone()[0] == 1
        return (affect.get_event(withdrawn['id']), affect.list_events(contact_id='person'),
                affect.count_events(contact_id='person'), affect.last_observed_at('person'),
                affect.project_state('person'))

    direct = read()
    with turn_budget.turn(TurnBudget(wait_s=7.5)):
        worker = await producers.read(read)
    assert worker == direct
    assert direct[0] is None
    assert [event['id'] for event in direct[1]] == [old['id']]
    assert direct[2:] == (1, start.isoformat(), None)
    assert not affect.detect_sustained_decline('person')
    assert [tuple(row) for row in affect._conn.execute('SELECT * FROM affect_events ORDER BY id')] == before


@pytest.mark.asyncio
async def test_affect_clock_passes_full_pages_of_withdrawn_events(traffic, tmp_path):
    from colonymind.contacts.affect import AffectStore
    ledger, log, start = traffic
    affect = AffectStore(str(tmp_path / 'paged-affect.db'), source_ledger=ledger)
    affect.create_event(contact_id='person', valence=0.2, timestamp=start.isoformat())
    lineage, _ = log.source_input('withdraw', 'person')
    event = affect.create_event(contact_id='person', valence=-0.8,
        timestamp=(start + timedelta(hours=1)).isoformat(), source_lineage=lineage)
    raw = affect._conn.execute('SELECT source_lineage_json FROM affect_events WHERE id=?',
                               (event['id'],)).fetchone()[0]
    with affect._conn:
        affect._conn.executemany('''INSERT INTO affect_events
            (id,contact_id,valence,arousal,source,timestamp,source_lineage_json)
            VALUES (?,?,?,?,?,?,?)''',
            [(f'withdrawn-{i:03}', 'person', -0.8, 0.5, 'explicit', event['timestamp'], raw)
             for i in range(225)])
    before = [tuple(row) for row in affect._conn.execute('SELECT * FROM affect_events ORDER BY id')]
    curated.forget_sources(ledger, 'person', source_ids=['withdraw'])
    assert affect.last_observed_at('person') == start.isoformat()
    with turn_budget.turn(TurnBudget(wait_s=7.5)):
        assert await producers.read(affect.last_observed_at, 'person') == start.isoformat()
    assert [tuple(row) for row in affect._conn.execute('SELECT * FROM affect_events ORDER BY id')] == before
