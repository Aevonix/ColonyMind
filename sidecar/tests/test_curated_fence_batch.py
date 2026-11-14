"""Fence lookup optimizations keep file identity and cache reset semantics."""
import os
import sqlite3
from contextlib import closing

import pytest

from colonymind.memory import curated
from colonymind.turns import TurnIdempotencyLedger
from test_plasticity_query_adapter_driver import runtime  # noqa: F401


def test_evidence_connection_rejects_writes(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'sources.db')
    ledger.record_source('source', contact_id='person', session_id='session',
                         messages=[{'role': 'user', 'content': 'Fixture'}], derive_claims=False)
    with closing(ledger._read_connect()) as conn:
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            conn.execute('DELETE FROM turn_sources')
        assert conn.execute('SELECT count(*) FROM turn_sources').fetchone()[0] == 1


def test_read_only_empty_legacy_journal_needs_no_fence_hash_schema(tmp_path):
    from types import SimpleNamespace
    path = tmp_path / 'sources.db'
    ledger = TurnIdempotencyLedger(path)
    with closing(ledger._connect()) as conn, conn:
        conn.execute('DROP TABLE curated_memory_fence_hashes')

    def connect():
        conn = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    reader = SimpleNamespace(db_path=path, _connect=connect)
    assert curated.forgotten_mask(reader, ['source']) == [False]
    assert curated.available(reader)
    with closing(connect()) as conn:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='curated_memory_fence_hashes'").fetchone() is None


def test_canonical_path_cache_follows_a_rebound_ledger_symlink(tmp_path):
    ledgers = [TurnIdempotencyLedger(tmp_path / (label + '.db')) for label in ('first', 'second')]
    for ledger in ledgers:
        ledger.record_source('source', contact_id='person', session_id='session',
                             messages=[{'role': 'user', 'content': 'Fixture'}], derive_claims=False)
    curated.forget_sources(ledgers[1], 'person', source_ids=['source'])
    alias = tmp_path / 'alias.db'
    alias.symlink_to(ledgers[0].db_path)
    reader = TurnIdempotencyLedger(alias)
    assert curated.forgotten_mask(reader, ['source']) == [False]
    replacement = tmp_path / 'replacement-link'
    replacement.symlink_to(ledgers[1].db_path)
    os.replace(replacement, alias)
    assert curated.forgotten_mask(reader, ['source']) == [True]


def test_canonical_path_cache_does_not_pin_a_cleared_fence_cache(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'sources.db')
    first = curated._cache_of(ledger)
    curated._cache.cache_clear()
    assert curated._cache_of(ledger) is not first


def test_missing_ledger_preserves_the_loud_fail_open_contract(tmp_path):
    from types import SimpleNamespace
    ledger = SimpleNamespace(db_path=tmp_path / 'absent.db')
    assert not curated.is_forgotten(ledger, 'source')
    assert not curated.available(ledger)


@pytest.mark.asyncio
async def test_qualification_hydrates_withdrawn_hits_before_selection(runtime, tmp_path):
    from dataclasses import asdict
    from colonymind.plasticity.recall_qualify import RecallScorer
    from test_plasticity_recall_qualify import FixtureIndex, FixturePipeline

    ledger = TurnIdempotencyLedger(tmp_path / 'sources.db')
    ledger.record_source('answer', contact_id='owner', session_id='past',
                         messages=[{'role': 'user', 'content': 'Fixture answer'}], derive_claims=False)
    index = FixtureIndex(ledger)
    curated.forget_sources(ledger, 'owner', source_ids=['answer'])
    driver = runtime.loaded['recall-test'].driver
    driver.document_identity = asdict(FixturePipeline.index_identity.document_identity)
    class Reranker:
        def calibration_metadata(self):
            return {'provider': 'fixture', 'model': 'fixture'}

        async def rerank(self, *args, **kwargs):
            pytest.fail('withdrawn candidate reached reranking')

    scorer = RecallScorer(driver, ledger=ledger, store=index, pipeline=FixturePipeline(),
                          reranker=Reranker(), owner='owner')
    result = await scorer.score({'query_id': 'probe', 'query': 'Fixture answer',
                                 'relevant': ['answer'], 'no_answer': False}, driver.learner.base())
    assert result['selected'] == []
    assert len(index.calls) == 1
    with ledger._connect() as conn:
        assert conn.execute('SELECT count(*) FROM turn_sources').fetchone()[0] == 1
        assert conn.execute('SELECT count(*) FROM source_erasures').fetchone()[0] == 1
