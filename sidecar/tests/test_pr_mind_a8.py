"""Claim projection preserves its SQLite fail-open path."""
from contextlib import closing
import sqlite3

from colonymind.beliefs.source_projection import SourceClaimProjection
from test_mind_consolidate import fx, CONTACT


async def test_sqlite_error_in_final_claim_filter_returns_collected_rows(fx, monkeypatch):
    claim_id = await fx.fact('claim-source', CONTACT, 'claim-session', 'My office is room 4.', 'room 4')
    def unavailable(*args, **kwargs):
        raise sqlite3.OperationalError('synthetic read failure')
    monkeypatch.setattr('colonymind.memory.curated.is_forgotten', unavailable)
    with closing(fx.ledger._connect()) as conn:
        rows = SourceClaimProjection(fx.ledger)._read_rows(conn, CONTACT, 'claim-session')
    assert [row['id'] for row in rows] == [claim_id]
    assert rows[0]['value'] == 'room 4'
