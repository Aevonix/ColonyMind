"""Doctor exposes pending and failed vector turns from the source ledger."""
from colonymind import doctor
from colonymind.turns import TurnIdempotencyLedger
from colonymind.vector.indexes import IndexCatalog, EmbeddingIdentity


def test_doctor_reports_vector_backlog_without_retrying(tmp_path, monkeypatch):
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(tmp_path))
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    catalog = IndexCatalog(ledger)
    catalog.initialize(EmbeddingIdentity('fixture', 'fixture', 'known', 3), legacy_exists=False)
    generation = catalog.active()['id']
    for ident in ('complete', 'pending', 'failed', 'old-generation', 'missing-job'):
        ledger.record_source(ident, contact_id='c', session_id='s',
                             messages=[{'role': 'user', 'content': 'Fixture text.'}])
    with ledger._connect() as conn:
        conn.execute("UPDATE source_vector_jobs SET status='complete',generation_id=?", (generation,))
        conn.execute("UPDATE source_vector_jobs SET status='pending' WHERE turn_id IN ('pending','failed')")
        conn.execute("UPDATE source_vector_jobs SET error='synthetic failure' WHERE turn_id='failed'")
        conn.execute("UPDATE source_vector_jobs SET generation_id='previous' WHERE turn_id='old-generation'")
        conn.execute("DELETE FROM source_vector_jobs WHERE turn_id='missing-job'")
        before = list(map(tuple, conn.execute('SELECT * FROM source_vector_jobs ORDER BY turn_id')))
    checks = dict(doctor.LOCAL_CHECKS)
    assert 'vector-jobs' in checks
    result = checks['vector-jobs']()
    output = doctor.format_report([result], color=False)
    assert 'pending vector turns: 3; failed vector turns: 1' in output
    assert result.status == doctor.WARN
    with ledger._connect() as conn:
        assert list(map(tuple, conn.execute('SELECT * FROM source_vector_jobs ORDER BY turn_id'))) == before


def test_doctor_reports_zero_vector_backlog(tmp_path, monkeypatch):
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(tmp_path))
    TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    checks = dict(doctor.LOCAL_CHECKS)
    assert 'vector-jobs' in checks
    result = checks['vector-jobs']()
    assert result.status == doctor.PASS
    assert result.detail == 'pending vector turns: 0; failed vector turns: 0'


def test_pending_counts_only_due_unforgotten_jobs(tmp_path, monkeypatch):
    import time
    from colonymind.memory import curated
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(tmp_path))
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    catalog = IndexCatalog(ledger)
    catalog.initialize(EmbeddingIdentity('fixture', 'fixture', 'known', 3), legacy_exists=False)
    generation = catalog.active()['id']
    for name in ('due', 'forgotten', 'branch', 'leased', 'retrying', 'old-generation', 'complete'):
        ledger.record_source(name, contact_id='c', session_id='s',
                             messages=[{'role': 'user', 'content': name + ' fixture text'}])
    curated.forget_sources(ledger, 'c', source_ids=['forgotten'])
    with ledger._connect() as conn:
        conn.execute("DELETE FROM source_vector_jobs WHERE turn_id='branch'")
        conn.execute("UPDATE source_vector_jobs SET lease_until=? WHERE turn_id='leased'", (time.time()+3600,))
        conn.execute("UPDATE source_vector_jobs SET next_attempt=?,error='fixture retry' WHERE turn_id='retrying'", (time.time()+3600,))
        conn.execute("UPDATE source_vector_jobs SET status='complete',generation_id='old' WHERE turn_id='old-generation'")
        conn.execute("UPDATE source_vector_jobs SET status='complete',generation_id=? WHERE turn_id='complete'", (generation,))
        before = list(map(tuple, conn.execute('SELECT * FROM source_vector_jobs ORDER BY turn_id')))
    result = doctor.check_vector_jobs()
    assert result.detail == 'pending vector turns: 2; failed vector turns: 1'
    assert result.status == doctor.WARN
    with ledger._connect() as conn:
        assert list(map(tuple, conn.execute('SELECT * FROM source_vector_jobs ORDER BY turn_id'))) == before
        conn.execute("UPDATE source_vector_jobs SET status='complete',generation_id=?,error=NULL", (generation,))
    assert doctor.check_vector_jobs().status == doctor.PASS


def test_forgotten_and_branch_sources_never_leave_permanent_pending_work(tmp_path, monkeypatch):
    from colonymind.memory import curated
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(tmp_path))
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    IndexCatalog(ledger).initialize(EmbeddingIdentity('fixture', 'fixture', 'known', 3), legacy_exists=False)
    for name in ('forgotten', 'branch'):
        ledger.record_source(name, contact_id='c', session_id='s',
                             messages=[{'role': 'user', 'content': name + ' fixture text'}])
    curated.forget_sources(ledger, 'c', source_ids=['forgotten'])
    with ledger._connect() as conn:
        conn.execute("DELETE FROM source_vector_jobs WHERE turn_id='branch'")
        conn.execute("UPDATE source_vector_jobs SET error='obsolete failure' WHERE turn_id='forgotten'")
    result = doctor.check_vector_jobs()
    assert result.detail == 'pending vector turns: 0; failed vector turns: 0'
    assert result.status == doctor.PASS
