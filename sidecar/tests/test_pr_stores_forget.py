"""Source forget calls the existing physical vector eraser after committing."""
import asyncio

import pytest

from colonymind.memory import curated
from colonymind.vector import Collection
from test_source_vectors import setup, drain, files_containing


@pytest.mark.asyncio
@pytest.mark.parametrize('worker', [True, False])
async def test_forget_purges_vector_files_and_copies(tmp_path, monkeypatch, worker):
    import colonymind.vector as vector
    ledger, store, _, projection = await setup(tmp_path)
    monkeypatch.setattr(vector, '_store', store)
    store.compaction.pause = 0
    text = 'The hydrofoil identifier is cedar-fixture-42.'
    for turn, messages in [('original', [{'role': 'user', 'content': text}]),
                           ('copy', [{'role': 'user', 'content': text},
                                     {'role': 'assistant', 'content': 'The office is open.'}])]:
        ledger.record_source(turn, contact_id='c', session_id='s', messages=messages)
    await drain(projection)
    assert files_containing(tmp_path / 'lancedb', 'cedar-fixture-42')
    try:
        if worker:
            receipt = await asyncio.to_thread(curated.forget_sources, ledger, 'c', source_ids=['original'])
        else:
            receipt = curated.forget_sources(ledger, 'c', source_ids=['original'])
        await asyncio.gather(*(asyncio.wrap_future(task) for task in store._erasure_tasks))
        assert receipt['source_erased'] is True
        assert files_containing(tmp_path / 'lancedb', 'cedar-fixture-42') == []
        table = await store._table(Collection.CONVERSATIONS)
        rows = await table.query().select(['text']).to_list()
        assert [row['text'] for row in rows] == ['The office is open.']
        # Historical canonical evidence remains as required by the existing contract.
        with ledger._connect() as conn:
            assert text in conn.execute("SELECT messages_json FROM turn_sources WHERE turn_id='original'").fetchone()[0]
        repeated = await asyncio.to_thread(curated.forget_sources, ledger, 'c', source_ids=['original'])
        assert repeated == receipt
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_committed_forget_returns_while_erase_is_blocked(tmp_path, monkeypatch):
    import colonymind.vector as vector
    ledger, store, _, projection = await setup(tmp_path)
    monkeypatch.setattr(vector, '_store', store)
    ledger.record_source('forget', contact_id='c', session_id='s',
                         messages=[{'role': 'user', 'content': 'Fixture memory'}])
    started, release = asyncio.Event(), asyncio.Event()
    eraser_loop, rewrites = [], []

    async def blocked(turn_ids, *, purge=True):
        eraser_loop.append(asyncio.get_running_loop())
        started.set()
        await release.wait()

    async def compact(reason):
        rewrites.append(reason)

    monkeypatch.setattr(store, 'erase_source_projections', blocked)
    monkeypatch.setattr(store.compaction, 'schedule', compact)
    request = asyncio.create_task(asyncio.to_thread(curated.forget_sources, ledger, 'c', source_ids=['forget']))
    try:
        await asyncio.wait_for(started.wait(), 2)
        done, _ = await asyncio.wait({request}, timeout=2)
        assert request in done, 'the committed fence must answer before erasure finishes'
        assert request.result()['source_erased']
        assert curated.is_forgotten(ledger, 'forget')
        assert eraser_loop == [asyncio.get_running_loop()]
        assert rewrites == []
    finally:
        release.set()
        await request
        await store.close()
    assert rewrites == ['erasure']


@pytest.mark.asyncio
@pytest.mark.parametrize('worker', [False, True])
async def test_failed_erase_cannot_fail_receipt_or_prevent_compaction_close(tmp_path, monkeypatch, caplog, worker):
    import colonymind.vector as vector
    ledger, store, _, _ = await setup(tmp_path)
    monkeypatch.setattr(vector, '_store', store)
    ledger.record_source('forget', contact_id='c', session_id='s',
                         messages=[{'role': 'user', 'content': 'Fixture memory'}])
    release, close_started = asyncio.Event(), asyncio.Event()
    compaction_closed = []
    real_close = store.compaction.close

    async def broken(turn_ids, *, purge=True):
        await release.wait()
        raise RuntimeError('fixture eraser failure')

    async def closed():
        compaction_closed.append(True)
        await real_close()

    monkeypatch.setattr(store, 'erase_source_projections', broken)
    monkeypatch.setattr(store.compaction, 'close', closed)
    try:
        if worker:
            receipt_task = asyncio.create_task(asyncio.to_thread(curated.forget_sources, ledger, 'c', source_ids=['forget']))
            done, _ = await asyncio.wait({receipt_task}, timeout=2)
            release.set()
            receipt = await receipt_task
            assert receipt_task in done
        else:
            receipt = curated.forget_sources(ledger, 'c', source_ids=['forget'])
        assert receipt['source_erased']
        closing = asyncio.create_task(store.close())
        await asyncio.sleep(0)
        release.set()
        await closing
        assert compaction_closed == [True]
        assert curated.available(ledger)
        assert 'Source vector erasure failed' in caplog.text
    finally:
        release.set()
        await store.close()


@pytest.mark.asyncio
async def test_scheduling_fault_does_not_fail_a_committed_forget(tmp_path, monkeypatch, caplog):
    import colonymind.vector as vector
    ledger, store, _, _ = await setup(tmp_path)
    monkeypatch.setattr(vector, '_store', store)
    ledger.record_source('forget', contact_id='c', session_id='s',
                         messages=[{'role': 'user', 'content': 'Fixture memory'}])

    def unavailable(turn_ids):
        raise RuntimeError('fixture closed event loop')

    monkeypatch.setattr(store, 'erase_forgotten_sources', unavailable)
    try:
        receipt = curated.forget_sources(ledger, 'c', source_ids=['forget'])
        assert receipt['source_erased'] and curated.is_forgotten(ledger, 'forget')
        assert curated.available(ledger)
        assert 'Could not schedule source vector erasure' in caplog.text
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_forget_schedules_only_fenced_turns_and_hash_holders(tmp_path, monkeypatch):
    import colonymind.vector as vector
    ledger, store, _, _ = await setup(tmp_path)
    monkeypatch.setattr(vector, '_store', store)
    selected = []
    monkeypatch.setattr(store, 'erase_forgotten_sources', lambda turns: selected.append(set(turns)))
    message = {'role': 'user', 'content': 'The fixture vessel is green.'}
    for turn, session, messages in [
        ('original', 's', [message]), ('copy', 's', [message, {'role': 'assistant', 'content': 'Still kept.'}]),
        ('unrelated', 's', [{'role': 'user', 'content': 'The office is open.'}]),
        ('other-session', 'other', [message]),
    ]:
        ledger.record_source(turn, contact_id='c', session_id=session, messages=messages)
    try:
        curated.forget_sources(ledger, 'c', source_ids=['original'])
        assert selected == [{'original', 'copy'}]
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['read', 'unavailable', 'predicate'])
async def test_erasure_never_deletes_on_an_uncertain_predicate(tmp_path, monkeypatch, caplog, fault):
    import sqlite3
    import colonymind.vector as vector
    ledger, store, _, projection = await setup(tmp_path)
    monkeypatch.setattr(vector, '_store', store)
    store.compaction.pause = 0
    for turn in ('forgotten', 'kept'):
        ledger.record_source(turn, contact_id='c', session_id='s',
            messages=[{'role': 'user', 'content': turn + ' fixture text'}])
    await drain(projection)
    # Commit the fence independently, then exercise the maintenance eraser with
    # an intentionally broad selection, so a conservative read cannot delete it.
    monkeypatch.setattr(store, 'erase_forgotten_sources', lambda turns: None)
    curated.forget_sources(ledger, 'c', source_ids=['forgotten'])
    original = curated._is_forgotten

    def unreadable(ledger_, ref, *, conn=None):
        if isinstance(ref, dict) and ref.get('source_projection'):
            raise sqlite3.OperationalError('fixture fence read failure')
        return original(ledger_, ref, conn=conn)

    def bad_predicate(meta):
        raise sqlite3.OperationalError('fixture predicate failure')

    if fault == 'read':
        monkeypatch.setattr(curated, '_is_forgotten', unreadable)
    elif fault == 'unavailable':
        curated.disable(ledger)
    else:
        monkeypatch.setattr(store.catalog, 'source_erased', bad_predicate)
    try:
        assert await store.erase_source_projections(['forgotten', 'kept'], purge=False) == 0
        table = await store._table(Collection.CONVERSATIONS)
        assert sorted(row['text'] for row in await table.query().select(['text']).to_list()) == [
            'forgotten fixture text', 'kept fixture text']
        with ledger._connect() as conn:
            assert conn.execute('SELECT count(*) FROM vector_projection_deletions').fetchone()[0] == 0
        assert 'Source vector erasure skipped: predicate unavailable' in caplog.text
        if fault != 'unavailable':
            assert curated.available(ledger)
    finally:
        curated.enable(ledger)
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('project_while_disabled', [False, True])
async def test_vectors_keep_working_with_fences_while_curated_is_disabled(tmp_path, monkeypatch, project_while_disabled):
    ledger, store, _, projection = await setup(tmp_path)
    monkeypatch.setattr(store, 'erase_forgotten_sources', lambda turns: None)
    ledger.record_source('gone', contact_id='c', session_id='s',
                         messages=[{'role': 'user', 'content': 'Forgotten fixture'}])
    await drain(projection)
    curated.forget_sources(ledger, 'c', source_ids=['gone'])
    ledger.record_source('kept', contact_id='c', session_id='s',
                         messages=[{'role': 'user', 'content': 'Current fixture'}])
    if not project_while_disabled:
        await drain(projection)
    curated.disable(ledger, RuntimeError('synthetic unrelated route fault'))
    try:
        if project_while_disabled:
            assert await projection.process_one()
            with ledger._connect() as conn:
                assert tuple(conn.execute("SELECT status,error FROM source_vector_jobs WHERE turn_id='kept'").fetchone()) == (
                    'complete', None)
        hits = await store.search(Collection.CONVERSATIONS, [1.0, 0.0, 0.0], limit=10)
        assert [hit.text for hit in hits] == ['Current fixture']
        assert not curated.available(ledger)
    finally:
        curated.enable(ledger)
        await store.close()


@pytest.mark.asyncio
async def test_vector_eligibility_fails_closed_on_a_fence_read_fault(tmp_path, monkeypatch):
    import sqlite3

    ledger, store, _, projection = await setup(tmp_path)
    monkeypatch.setattr(store, 'erase_forgotten_sources', lambda turns: None)
    for turn in ('gone', 'kept'):
        ledger.record_source(turn, contact_id='c', session_id='s',
                             messages=[{'role': 'user', 'content': turn + ' fixture'}])
    await drain(projection)
    curated.forget_sources(ledger, 'c', source_ids=['gone'])
    table = await store._table(Collection.CONVERSATIONS)
    import json
    metadata = [json.loads(row['metadata']) for row in await table.query().select(['metadata']).to_list()]

    def unreadable(*args, **kwargs):
        raise sqlite3.OperationalError('synthetic fence read fault')

    monkeypatch.setattr(curated, '_is_forgotten', unreadable)
    try:
        assert all(store.catalog.source_erased(meta) for meta in metadata)
    finally:
        curated.enable(ledger)
        await store.close()
