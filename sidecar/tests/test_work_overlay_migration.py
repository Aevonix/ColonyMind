"""Interrupted projection schema upgrades resume without losing verdicts."""
from contextlib import closing
import sqlite3
from types import SimpleNamespace

import pytest

from colonymind.turns import TurnIdempotencyLedger
from colonymind.work.store import WorkStore
from test_work_projection import admit


@pytest.mark.parametrize('completed_columns', [0, 1, 2, 3])
def test_interrupted_overlay_migration_resumes_verdict_writes(tmp_path, monkeypatch, completed_columns):
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(tmp_path))
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    with closing(sqlite3.connect(tmp_path / 'colonymind-work.db')) as conn:
        conn.execute("""CREATE TABLE work_overlays (
            root_id TEXT PRIMARY KEY, verdict TEXT, source_refs_json TEXT NOT NULL DEFAULT '[]')""")
        conn.execute("INSERT INTO work_overlays VALUES ('existing', 'confirmed', '[]')")
        conn.commit()
        for column in (
            "verdict_at TEXT NOT NULL DEFAULT ''",
            "verdict_id TEXT NOT NULL DEFAULT ''",
            "identity_json TEXT NOT NULL DEFAULT '{}'",
        )[:completed_columns]:
            conn.execute('ALTER TABLE work_overlays ADD COLUMN ' + column)
            assert not conn.in_transaction
        # Closing here simulates interruption after any committed ALTER.

    work = WorkStore(ledger)
    work = WorkStore(ledger)  # A completed upgrade is also safe to repeat.
    admit(SimpleNamespace(ledger=ledger, work=work))
    work.reconcile()
    root = work.get_root('root', contact_id='owner')
    assert work.write_derived('root', contact_id='owner', expected_revision=root['revision'],
                              overlay='corrected', verdict_order=('2026-01-01', 'verdict'))
    assert work.get_root('root', contact_id='owner')['overlay'] == 'corrected'
    with closing(work._connect()) as conn:
        assert tuple(conn.execute("SELECT * FROM work_overlays WHERE root_id='existing'").fetchone()) == (
            'existing', 'confirmed', '[]', '', '', '{}')
        assert tuple(conn.execute("SELECT verdict_at,verdict_id,identity_json FROM work_overlays "
                                  "WHERE root_id='root'").fetchone()) == ('2026-01-01', 'verdict', '{}')
