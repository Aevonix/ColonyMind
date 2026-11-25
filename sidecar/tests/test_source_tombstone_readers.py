"""Live readers withdraw retained evidence through the shared source fence."""
from contextlib import closing

import pytest

from colonymind.memory import curated
from colonymind.turns import TurnIdempotencyLedger
from test_transport_ingress_api import ingress
from test_turn_source_evidence import source_app


def record(ledger, source='private', *, person='owner', session='session', **message):
    ledger.record_source(source, contact_id=person, session_id=session,
        messages=[{'role': 'user', 'content': 'Send the private marina report.', **message}], derive_claims=False)


def test_current_state_survives_restart_and_never_scans_history_on_read(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path/'turn-idempotency.db')
    record(ledger)
    ref = ledger.source_references(['private'], contact_id='owner', session_id='later')[0]
    ledger.record_source('dependent', contact_id='owner', session_id='session', messages=[{
        'role': 'assistant', 'content': 'The marina report is ready.', '_supplied_sources': [ref]}], derive_claims=False)
    curated.forget_sources(ledger, 'owner', source_ids=['private'])
    reader = TurnIdempotencyLedger(ledger.db_path)
    # Invalidate the process cache as a fresh process would, including an unrelated native commit.
    curated._cache.cache_clear()
    record(reader, 'unrelated', session='another', content='The bicycle is blue.')
    traced = []
    connect = reader._connect
    def connection():
        db = connect()
        db.set_trace_callback(traced.append)
        return db
    reader._connect = connection
    curated._cache_of(reader).monitor.set_trace_callback(traced.append)
    assert curated.is_forgotten(reader, 'dependent')
    assert not reader.search_sources('marina', contact_id='owner', session_id='later')
    assert not reader.search_stems('marina', contact_id='owner', session_id='later')
    assert not curated.current_candidates(reader, [dict(turn_id='dependent')], contact_id='owner')
    assert not any('FROM turn_sources WHERE contact_id=' in sql for sql in traced)
    with closing(connect()) as db:
        assert db.execute('SELECT count(*) FROM turn_sources').fetchone()[0] == 3
        assert db.execute('SELECT count(*) FROM curated_memory_fence_hashes').fetchone()[0] >= 2


def test_startup_indexes_legacy_withdrawals_without_native_mutation(tmp_path):
    from test_curated_memory import historical_bytes
    ledger = TurnIdempotencyLedger(tmp_path/'turn-idempotency.db')
    record(ledger)
    ref = ledger.source_references(['private'], contact_id='owner', session_id='session')[0]
    for source, extra in [('derived', {'_supplied_sources': [ref]}), ('copy', {})]:
        ledger.record_source(source, contact_id='owner', session_id='session', messages=[{
            'role': 'assistant', 'content': 'The marina report is ready.', **extra}], derive_claims=False)
    with closing(ledger._connect()) as db, db:
        db.execute("INSERT INTO curated_memory_events(person_id,memory_id,revision,operation,request_id,"
                   "request_digest,applied_at,data_json,source_ids_json) VALUES (?,?,1,'forget',?,?,?,NULL,?)",
                   ('owner', 'sources:legacy', 'legacy', 'legacy', '2026-10-19T00:00:00+00:00', '["private"]'))
    before = historical_bytes(ledger)
    reader = TurnIdempotencyLedger(ledger.db_path)
    assert historical_bytes(reader) == before
    curated._cache.cache_clear()
    assert curated.is_forgotten(reader, 'copy')
    assert not reader.search_sources('marina', contact_id='owner', session_id='later')


def test_source_ids_remain_fenced_when_hash_index_cannot_be_read(tmp_path):
    import sqlite3
    from test_curated_memory import historical_bytes
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    record(ledger)
    curated.forget_sources(ledger, 'owner', source_ids=['private'])
    before = historical_bytes(ledger)
    curated._cache.cache_clear()
    monitor = curated._cache_of(ledger).monitor
    monitor.set_authorizer(lambda action, table, *_: sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_READ and table == 'curated_memory_fence_hashes' else sqlite3.SQLITE_OK)
    try:
        assert curated.is_forgotten(ledger, 'private')
        assert ledger.search_sources('marina', contact_id='owner', session_id='later') == []
        assert historical_bytes(ledger) == before
    finally:
        monitor.set_authorizer(None)


@pytest.mark.parametrize('source', ['private', 'turn:private'])
def test_native_turn_prefix_ids_and_their_copies_remain_withdrawn(tmp_path, source):
    from test_curated_memory import historical_bytes
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    record(ledger, source)
    record(ledger, 'turn:copy')
    before = historical_bytes(ledger)
    curated.forget_sources(ledger, 'owner', source_ids=[source])
    for ref in (source, 'turn:' + source, 'turn:copy', {'source_id': 'turn:copy'}):
        assert curated.is_forgotten(ledger, ref)
    assert ledger.search_sources('marina', contact_id='owner', session_id='later') == []
    assert historical_bytes(ledger) == before


def test_commitment_readers_hide_forgotten_source_and_keep_row(tmp_path):
    from colonymind.commitments.store import CommitmentStore
    ledger = TurnIdempotencyLedger(tmp_path/'turn-idempotency.db')
    record(ledger)
    store = CommitmentStore(tmp_path/'commitments.db')
    item = store.create(person_id='owner', description='Send the private marina report',
                        metadata={'source_turn': 'private'})
    assert store.get(item['id'])
    curated.forget_sources(ledger, 'owner', source_ids=['private'])
    assert store.get(item['id']) is None
    assert store.list(person_id='owner')['commitments'] == []
    assert store.get_pending_for_person('owner') == []
    with closing(store._connect()) as db:
        assert db.execute('SELECT description FROM commitments').fetchone()[0] == item['description']


@pytest.mark.parametrize('withdrawn', [False, True])
def test_commitment_listing_reads_its_sources_on_one_connection(tmp_path, monkeypatch, withdrawn):
    import sqlite3
    from colonymind.commitments.store import CommitmentStore
    ledger = TurnIdempotencyLedger(tmp_path/'turn-idempotency.db')
    for index in range(12):
        ledger.record_source(f'source-{index}', contact_id='owner', session_id='session',
                             messages=[{'role': 'user', 'content': f'Fixture item {index}'}], derive_claims=False)
    store = CommitmentStore(tmp_path/'commitments.db')
    for index in range(12):
        store.create(person_id='owner', description=f'Fixture item {index}',
                     metadata={'source_turn': f'source-{index}'} if index % 4 else None)
    if withdrawn:
        curated.forget_sources(ledger, 'owner', source_ids=['source-1', 'source-6'])
    with closing(store._connect()) as db:
        rows = db.execute('SELECT * FROM commitments ORDER BY made_at DESC').fetchall()
    expected = [row['id'] for row in rows if store._visible(row)]    # each row checked on its own connection
    assert len(expected) == (10 if withdrawn else 12)
    connect, opened = sqlite3.connect, []

    def counted(target, *args, **kwargs):
        if 'turn-idempotency.db' in str(target):
            opened.append(target)
        return connect(target, *args, **kwargs)

    monkeypatch.setattr(sqlite3, 'connect', counted)
    listed = store.list(person_id='owner', limit=5, offset=2)
    assert [item['id'] for item in listed['commitments']] == expected[2:7]
    assert listed['total'] == len(expected)
    assert len(opened) == 1


def test_later_dependent_answers_and_plain_copies_inherit_withdrawal(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path/'turn-idempotency.db')
    record(ledger)
    ref = ledger.source_references(['private'], contact_id='owner', session_id='session')[0]
    curated.forget_sources(ledger, 'owner', source_ids=['private'])
    ledger.record_source('late-answer', contact_id='owner', session_id='session', messages=[{
        'role': 'assistant', 'content': 'The marina report is ready.', '_supplied_sources': [ref]}],
        derive_claims=False)
    # A different process capturing a plain copy has no process-local lineage.
    curated._cache.cache_clear()
    ledger.record_source('plain-copy', contact_id='owner', session_id='session', messages=[{
        'role': 'assistant', 'content': 'The marina report is ready.'}], derive_claims=False)
    curated._cache.cache_clear()
    assert curated.is_forgotten(ledger, 'plain-copy')
    assert not ledger.search_sources('marina', contact_id='owner', session_id='later')
    with closing(ledger._connect()) as db:
        assert db.execute('SELECT count(*) FROM turn_sources').fetchone()[0] == 3
        assert db.execute('SELECT count(*) FROM curated_memory_fence_hashes').fetchone()[0] >= 2


def test_secret_aliases_remain_invisible_when_the_value_was_already_forgotten(tmp_path):
    import json
    from colonymind.secrets.refs import references_in
    ledger = TurnIdempotencyLedger(tmp_path/'turn-idempotency.db')
    message = {'role': 'user', 'content': 'The marina wifi password is violet-river-741.'}
    ledger.record_source('private', contact_id='owner', session_id='session', messages=[message], derive_claims=False)
    with closing(ledger._connect()) as db:
        stored = json.loads(db.execute("SELECT messages_json FROM turn_sources WHERE turn_id='private'").fetchone()[0])
    reference, = references_in(stored[0]['content'])
    assert ledger.vault.forget(reference.ref_id)
    curated.forget_sources(ledger, 'owner', source_ids=['private'])
    assert ledger.record_source('new-alias', contact_id='owner', session_id='session', messages=[message], derive_claims=False)
    assert curated.is_forgotten(ledger, 'new-alias')
    assert not ledger.search_sources('marina wifi', contact_id='owner', session_id='later')
    with closing(ledger._connect()) as db:
        assert db.execute('SELECT count(*) FROM turn_sources').fetchone()[0] == 2


def test_context_card_obligation_and_privacy_revision(tmp_path, monkeypatch):
    from colonymind.context_card.inputs import InputCollector, Obligation, PreparedInputs
    from colonymind.context_card.builder import CardInputs
    from colonymind.context_card.refresh import RevisionProbe
    from colonymind import turns
    ledger = TurnIdempotencyLedger(tmp_path/'turn-idempotency.db')
    record(ledger)
    # Use the real detached collector without starting any producer or worker.
    inputs = CardInputs(owner_id='owner', owner_name='Owner', local_day='2026-10-19')
    prepared = PreparedInputs(inputs=inputs, obligations=(Obligation('private', 'source', 'pending',
        'Send the private marina report', ''),), digest_created_at=None, comms=None)
    monkeypatch.setattr(turns, 'get_turn_idempotency_ledger', lambda *_: ledger)
    collector = InputCollector(tmp_path, host=None, mind_provider=lambda: None,
        owner_provider=lambda: 'owner', name_provider=lambda: 'Owner', day_provider=lambda: '2026-10-19')
    probe = RevisionProbe(tmp_path, owner_provider=lambda: 'owner')
    revision = probe.binding()
    before = collector.collect_ledger(prepared)
    assert any('private marina' in line for line in before.work)
    curated.forget_sources(ledger, 'owner', source_ids=['private'])
    assert probe.binding() != revision
    after = collector.collect_ledger(prepared)
    assert not any('private marina' in line for line in after.work)
    assert 'private' not in after.source_refs

def test_known_withdrawal_stays_fenced_during_curated_outage(tmp_path, monkeypatch):
    ledger = TurnIdempotencyLedger(tmp_path/'turn-idempotency.db')
    record(ledger)
    record(ledger, 'unrelated', session='other', content='The bicycle is blue.')
    # The successful append publishes its fallback before any later read.
    curated.forget_sources(ledger, 'owner', source_ids=['private'])
    curated.disable(ledger, RuntimeError('synthetic journal outage'))
    assert curated.is_forgotten(ledger, 'private')
    assert not curated.is_forgotten(ledger, 'unrelated')
    rows = [dict(turn_id='private'), dict(turn_id='unrelated')]
    assert curated.current_candidates(ledger, rows, contact_id='owner') == rows[1:]
    assert curated.candidates(ledger, 'owner', 'report', rows, [])[0] == rows[1:]


def test_degraded_reader_observes_a_later_withdrawal_from_another_process(tmp_path):
    import subprocess
    import sys
    ledger = TurnIdempotencyLedger(tmp_path/'turn-idempotency.db')
    record(ledger)
    record(ledger, 'unrelated', session='other', content='The bicycle is blue.')
    assert not curated.is_forgotten(ledger, 'private')
    curated.disable(ledger, RuntimeError('ordinary curated read outage'))
    script = ('from colonymind.turns import TurnIdempotencyLedger; '
              'from colonymind.memory import curated; import sys; '
              'curated.forget_sources(TurnIdempotencyLedger(sys.argv[1]), "owner", source_ids=["private"])')
    subprocess.run([sys.executable, '-c', script, str(ledger.db_path)], check=True)
    assert not curated.available(ledger)
    assert curated.is_forgotten(ledger, 'private')
    assert not ledger.search_sources('marina', contact_id='owner', session_id='later')
    assert ledger.search_sources('bicycle', contact_id='owner', session_id='later')


def test_current_fence_reopens_a_replaced_native_database(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path/'turn-idempotency.db')
    record(ledger)
    record(ledger, 'copy')
    assert not curated.is_forgotten(ledger, 'private')
    replacement = TurnIdempotencyLedger(tmp_path/'replacement'/'turn-idempotency.db')
    record(replacement)
    record(replacement, 'copy')
    curated.forget_sources(replacement, 'owner', source_ids=['private'])
    # Keep the prior files too, while simulating an atomic replacement during recovery.
    for suffix in ('', '-wal', '-shm'):
        old = type(ledger.db_path)(str(ledger.db_path)+suffix)
        new = type(ledger.db_path)(str(replacement.db_path)+suffix)
        if old.exists():
            old.rename(tmp_path/('prior.db'+suffix))
        if new.exists():
            new.rename(old)
    assert curated.is_forgotten(ledger, 'private')
    assert curated.is_forgotten(ledger, 'copy')
    with closing(ledger._connect()) as db:
        assert db.execute('SELECT count(*) FROM turn_sources').fetchone()[0] == 2

@pytest.mark.asyncio
async def test_transport_withdrawal_is_a_read_fence_and_retains_settlement(ingress):
    from test_transport_ingress_api import admit, handoff, capture, status
    receipt = await admit(ingress)
    turn = await handoff(ingress, receipt)
    assert (await capture(ingress, turn)).status_code == 201
    assert (await status(ingress, receipt)).json()['items'][0]['state'] == 'completed'
    from colonymind.api.routers.transport_ingress_api import store
    transport = store()
    before = dict(transport.get(receipt['receipt_id']))
    curated.forget_sources(ingress.ledger, ingress.person, source_ids=[turn['turn_id']])
    value = (await status(ingress, receipt)).json()['items'][0]
    assert value['state'] == 'erased' and value['source_versions_json'] is None
    assert dict(transport.get(receipt['receipt_id'])) == before
    replay = await admit(ingress, occurred_at=before['occurred_at'])
    assert replay['state'] == 'erased' and replay['source_versions_json'] is None
    from test_transport_ingress_api import PREFIX, headers
    rebound = await ingress.client.post(PREFIX+'/handoff', headers=headers(), json={
        'receipt_ids': [receipt['receipt_id']], 'batch_id': 'batch', 'native_turn': turn})
    assert rebound.status_code == 409
    assert dict(transport.get(receipt['receipt_id'])) == before

def test_withdrawal_is_current_inside_an_older_native_snapshot(tmp_path):
    from types import SimpleNamespace
    ledger = TurnIdempotencyLedger(tmp_path/'turn-idempotency.db')
    record(ledger)
    with closing(ledger._connect()) as native:
        native.execute('BEGIN')
        assert native.execute('SELECT count(*) FROM curated_memory_events').fetchone()[0] == 0
        curated.forget_sources(ledger, 'owner', source_ids=['private'])
        # A second process does not share the writer's process-local fallback.
        curated._cache.cache_clear()
        class Borrowed:
            def __getattr__(self, key):
                return getattr(native, key)
            def close(self):
                pass
        reader = SimpleNamespace(db_path=ledger.db_path, _connect=lambda: Borrowed())
        assert not curated.current_candidates(reader, [dict(source_id='private')], contact_id='owner')
        assert native.in_transaction
        assert native.execute('SELECT count(*) FROM curated_memory_events').fetchone()[0] == 0


def test_known_withdrawal_cannot_rewind_with_a_restored_journal(tmp_path):
    import sqlite3
    ledger = TurnIdempotencyLedger(tmp_path/'turn-idempotency.db')
    record(ledger)
    record(ledger, 'copy')
    archived = tmp_path/'before.db'
    with closing(ledger._connect()) as source, closing(sqlite3.connect(archived)) as target:
        source.backup(target)
    curated.forget_sources(ledger, 'owner', source_ids=['private'])
    candidate = dict(source_id='copy')
    assert not curated.current_candidates(ledger, [candidate], contact_id='owner')
    with closing(sqlite3.connect(archived)) as source, closing(ledger._connect()) as target:
        source.backup(target)
    assert not curated.current_candidates(ledger, [candidate], contact_id='owner')
    assert curated.is_forgotten(ledger, 'copy')

def test_freshness_cursor_handles_another_persons_historical_events(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path/'turn-idempotency.db')
    record(ledger)
    with closing(ledger._connect()) as db, db:
        db.execute('INSERT INTO source_erasures(contact_id,turn_id,session_id,message_hashes_json) '
                   'VALUES (?,?,?,?)', ('other', 'historical', 'history', '[]'))
    curated.forget_sources(ledger, 'owner', source_ids=['private'])
    first = curated.freshness_feed(ledger, 'owner')
    assert first['complete'] and first['head'] == curated.watermark(ledger, 'owner') == 2
    next_page = curated.freshness_feed(ledger, 'owner', after=first['through'])
    assert next_page['complete'] and not next_page['events']


def test_source_withdrawal_survives_corrupt_curated_content(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path/'turn-idempotency.db')
    record(ledger)
    ref = ledger.source_references(['private'], contact_id='owner', session_id='session')[0]
    ledger.record_source('derived', contact_id='owner', session_id='session', messages=[{
        'role': 'assistant', 'content': 'The marina report is ready.', '_supplied_sources': [ref]}], derive_claims=False)
    ledger.record_source('copy', contact_id='owner', session_id='session', messages=[{
        'role': 'assistant', 'content': 'The marina report is ready.'}], derive_claims=False)
    record(ledger, 'unrelated', session='other', content='The bicycle is blue.')
    curated.forget_sources(ledger, 'owner', source_ids=['private'])
    with closing(ledger._connect()) as db, db:
        db.execute("INSERT INTO curated_memory_events(person_id,memory_id,revision,operation,request_id,"
                   "request_digest,applied_at,data_json,source_ids_json) VALUES (?,?,1,'remember',?,?,?,?,'[]')",
                   ('owner', 'mem_broken', 'broken', 'broken', '2026-10-19T00:00:00+00:00', '{broken'))
    curated._cache.cache_clear()
    assert not ledger.search_sources('marina', contact_id='owner', session_id='later')
    assert ledger.search_sources('bicycle', contact_id='owner', session_id='later')
    assert not curated.current_candidates(ledger, [dict(source_id='copy')], contact_id='owner')


def test_work_report_withholds_root_without_changing_instance_files(tmp_path):
    from colonymind.work.store import WorkStore, read_summary
    from test_work_api import retained_work
    ledger = TurnIdempotencyLedger(tmp_path/'turn-idempotency.db')
    retained_work(ledger, 'private')
    work = WorkStore(ledger)
    work.reconcile()
    curated.forget_sources(ledger, 'owner-1', source_ids=['private'])
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    assert read_summary(work.path, contact_id='owner-1')['total'] == 0
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before

def test_work_maintenance_keeps_withdrawn_projection_rows(tmp_path):
    from colonymind.work.store import WorkStore
    from test_work_api import retained_work
    ledger = TurnIdempotencyLedger(tmp_path/'turn-idempotency.db')
    retained_work(ledger, 'private')
    work = WorkStore(ledger)
    work.reconcile()
    with closing(work._connect()) as db:
        before = [tuple(row) for row in db.execute('SELECT * FROM work_roots')]
    curated.forget_sources(ledger, 'owner-1', source_ids=['private'])
    work.reconcile()
    assert work.get_root('private', contact_id='owner-1') is None
    assert work.wait_pending()
    with closing(work._connect()) as db:
        assert [tuple(row) for row in db.execute('SELECT * FROM work_roots')] == before


def test_work_report_rechecks_withdrawal_after_copying_snapshot(tmp_path, monkeypatch):
    from colonymind.work import store
    from test_work_api import retained_work
    ledger = TurnIdempotencyLedger(tmp_path/'turn-idempotency.db')
    retained_work(ledger, 'private')
    work = store.WorkStore(ledger)
    work.reconcile()
    assert store.read_summary(work.path, contact_id='owner-1')['total'] == 1
    summarize = store._summary
    def race(rows, effects):
        if rows:
            curated.forget_sources(ledger, 'owner-1', source_ids=['private'])
        return summarize(rows, effects)
    monkeypatch.setattr(store, '_summary', race)
    assert store.read_summary(work.path, contact_id='owner-1')['total'] == 0
    with closing(work._connect()) as db:
        assert db.execute('SELECT count(*) FROM work_roots').fetchone()[0] == 1


@pytest.mark.asyncio
async def test_reextraction_limit_selects_current_sources_before_budget(tmp_path):
    from colonymind.beliefs.reextract import Reextraction
    from colonymind.beliefs.source_projection import SourceClaimProjection
    from test_source_claim_reextract import ledger_at, history, OFFICE, RIVER, TEA
    ledger = ledger_at(tmp_path)
    await history(ledger, [('first', RIVER, 9), ('second', TEA, 10)], OFFICE)
    curated.forget_sources(ledger, 'owner', source_ids=['first'])
    worker = Reextraction(SourceClaimProjection(ledger))
    assert worker.plan(limit=1)['sources'] == 1
    assert worker.start(limit=1)['queued'] == 1
    with closing(ledger._connect()) as db:
        assert [row[0] for row in db.execute('SELECT turn_id FROM source_reextract_jobs')] == ['second']
