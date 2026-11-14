"""Maintenance must not send withdrawn vector prose back to an embedder."""
import json

import pytest

from colonymind.memory import curated
from colonymind.turns import TurnIdempotencyLedger
from colonymind.vector import Collection
from colonymind.vector.indexes import EmbeddingIdentity, IndexCatalog
from colonymind.vector.store import VectorStore


@pytest.mark.asyncio
async def test_scan_fences_curated_sources_after_restart_without_deleting_vectors(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'sources.db')
    identity = EmbeddingIdentity('fixture', 'fixture', 'known', 2)
    store = VectorStore(str(tmp_path / 'vectors'), identity=identity, catalog=IndexCatalog(ledger))
    await store.connect(2)
    await store.ensure_collections(2)
    for source in ('withdrawn', 'kept'):
        ledger.record_source(source, contact_id='person', session_id='session',
                             messages=[{'role': 'user', 'content': source}], derive_claims=False)
        await store.add(Collection.MEMORIES, id=source, text=source, vector=[1., 0.],
                        metadata={'source_uri': 'turn:' + source})
    table = await store._table(Collection.MEMORIES)
    before = await table.query().to_list()
    assert len(await store.scan_all(Collection.MEMORIES)) == 2
    curated.forget_sources(ledger, 'person', source_ids=['withdrawn'])
    assert [row['id'] for row in await store.scan_all(Collection.MEMORIES)] == ['kept']
    assert await store.count(Collection.MEMORIES) == 2  # physical health accounting
    await store.close()
    reopened = VectorStore(str(tmp_path / 'vectors'), identity=identity,
                           catalog=IndexCatalog(TurnIdempotencyLedger(ledger.db_path)))
    await reopened.connect(2)
    try:
        visible = await reopened.scan_all(Collection.MEMORIES)
        assert [row['id'] for row in visible] == ['kept']
        assert json.loads(visible[0]['metadata'])['source_uri'] == 'turn:kept'
        table = await reopened._table(Collection.MEMORIES)
        assert await table.query().to_list() == before
    finally:
        await reopened.close()
