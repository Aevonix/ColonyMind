"""Forget metadata readers obey comms snapshot and erasure contracts together."""
from datetime import datetime, timezone
import sqlite3

import pytest

from colonymind.memory import curated, producers, turn_budget
from colonymind.memory.turn_budget import TurnBudget
from test_comms_curated_fence import traffic  # noqa: F401
from test_comms_erasure_snapshot import erased_projection


def test_batched_fence_refreshes_after_withdrawal_and_preserves_dependencies(traffic):
    ledger, log, _ = traffic
    refs = ['withdraw', 'dependent', 'kept', 'other']
    assert curated.forgotten_mask(ledger, refs) == [False] * 4
    before = [tuple(row) for row in log._conn.execute('SELECT * FROM communications ORDER BY id')]
    curated.forget_sources(ledger, 'person', source_ids=['withdraw'])
    assert curated.forgotten_mask(ledger, refs) == [True, True, False, False]
    assert curated.forgotten_mask(ledger, refs) == [curated.is_forgotten(ledger, ref) for ref in refs]
    assert [tuple(row) for row in log._conn.execute('SELECT * FROM communications ORDER BY id')] == before


def test_empty_fences_use_the_monitor_without_native_body_connections(traffic, monkeypatch):
    ledger, _, _ = traffic
    candidate = {'turn_id': 'kept'}
    with monkeypatch.context() as guard:
        guard.setattr(ledger, '_connect', lambda: pytest.fail('empty fence opened native body connection'))
        assert curated.forgotten_mask(ledger, ['kept']) == [False]
        assert curated.candidates(ledger, 'person', '', [], [candidate]) == ([], [candidate])
        assert curated.current_candidates(ledger, [candidate], contact_id='person') == [candidate]
    curated.forget_sources(ledger, 'person', source_ids=['kept'])
    assert curated.current_candidates(ledger, [candidate], contact_id='person') == []


def test_rollup_rechecks_withdrawals_committed_during_sql_aggregation(traffic, monkeypatch):
    ledger, log, start = traffic
    before = [tuple(row) for row in log._conn.execute('SELECT * FROM communications ORDER BY id')]
    original = log._reconciled_rows
    withdrew = False

    def select(*args, **kwargs):
        nonlocal withdrew
        rows = original(*args, **kwargs)
        if not withdrew:
            withdrew = True
            curated.forget_sources(ledger, 'person', source_ids=['withdraw'])
        return rows

    monkeypatch.setattr(log, '_reconciled_rows', select)
    since_days = max(1, (datetime.now(timezone.utc) - start).days + 1)
    assert log.rollup(since_days=since_days) == {'unknown': {'in': 3, 'out': 2}}
    assert withdrew
    assert not log._conn.in_transaction
    assert [tuple(row) for row in log._conn.execute('SELECT * FROM communications ORDER BY id')] == before


@pytest.mark.asyncio
async def test_metadata_producer_uses_read_only_snapshot_and_selected_lineage(tmp_path, monkeypatch):
    log = erased_projection(tmp_path)
    log._erased_pairs(['kept'])
    original = log._erased_pairs
    selections, statements = [], []

    def probe(turn_ids):
        selected = list(turn_ids)
        selections.append(set(selected))
        return original(selected)

    monkeypatch.setattr(log, '_erased_pairs', probe)

    def read():
        assert log._conn.execute('PRAGMA query_only').fetchone()[0] == 1
        log._conn.set_trace_callback(lambda sql: statements.append(sql)
                                     if sql in ('BEGIN', 'BEGIN IMMEDIATE', 'COMMIT') else None)
        assert log.counts('person') == {'inbound': 1, 'outbound': 0, 'channels': 1}
        assert log.stats('person')['total'] == 1
        assert log.inbound_since('person', '2000') == ['2099-01-01']
        assert not log._conn.in_transaction

    before = [tuple(row) for row in log._conn.execute('SELECT * FROM communications ORDER BY id')]
    with turn_budget.turn(TurnBudget(wait_s=7.5)):
        await producers.read(read)
    assert selections == [{'answer', 'kept'}] * 3
    assert statements == ['BEGIN', 'COMMIT'] * 3
    assert [tuple(row) for row in log._conn.execute('SELECT * FROM communications ORDER BY id')] == before


@pytest.mark.parametrize('method', ['counts', 'stats', 'rollup', 'inbound_since'])
def test_metadata_cleanup_releases_writer(tmp_path, method):
    log = erased_projection(tmp_path)
    args = () if method == 'rollup' else ('person',)
    if method == 'inbound_since':
        args += ('2000',)
    getattr(log, method)(*args)
    assert not log._conn.in_transaction
    assert log._conn.execute("SELECT 1 FROM communications WHERE id='answer'").fetchone() is None
    with sqlite3.connect(log._db_path, timeout=0) as writer:
        writer.execute("INSERT INTO communications(id,contact_id,direction,ts) VALUES ('late','person','in','2099-01-01')")


def test_metadata_failed_materialization_rolls_back_cleanup_and_releases_writer(tmp_path):
    log = erased_projection(tmp_path)
    with pytest.raises(sqlite3.OperationalError):
        log._metadata_rows('SELECT absent_column FROM communications', (), (), contact_id='person')
    assert not log._conn.in_transaction
    assert log._conn.execute("SELECT 1 FROM communications WHERE id='answer'").fetchone()
    with sqlite3.connect(log._db_path, timeout=0) as writer:
        writer.execute("DELETE FROM communications WHERE id='answer'")
