"""Titles cannot carry revoked text from another request or an old batch."""
from contextlib import closing
import json
import sqlite3

import pytest

from colonymind.turns.source_attribution import correct
from colonymind.work.night import WorkNight
from colonymind.work.store import WorkStore, read_summary
from test_work_night import NOW, record
from test_work_projection import stores, admit, envelope, operation

PRIVATE = 'Cobalt falcon merger dossier'
PUBLIC = 'Update public report'


def finish(coro):
    # The model stub never suspends and needs no socket or event loop.
    with pytest.raises(StopIteration):
        coro.send(None)


@pytest.mark.parametrize('action', ['correct', 'erase'])
@pytest.mark.parametrize('timing', ['during-call', 'after-call'])
def test_cross_batch_title_provenance(stores, monkeypatch, action, timing):
    monkeypatch.setattr(WorkStore, 'schedule', lambda *args: None)
    for ident, text in [('private-root', PRIVATE), ('owner-root', PUBLIC)]:
        admit(stores, envelope(ident, operations=[operation(evidence={'kind': 'none'})]),
              source_id=ident, messages=[{'role': 'user', 'content': text}])
    stores.work.reconcile()
    inputs = []

    def revoke():
        correct(stores.ledger, operation_id='guest-correction', performed_by='owner', old_contact_id='owner', contact_id='guest', source_ids=['private-root'], evidence_refs=['source:owner-confirmation'])

    async def call(night, **kw):
        batch = json.loads(kw['user'])
        inputs.append(batch)
        target = next((item for item in batch if item['request'] == PUBLIC), None)
        if target and timing == 'during-call':
            revoke()
        # Model misassociation is possible only when both requests are exposed.
        return {'titles': [{'id': item['id'], 'title': batch[0]['request']
                            if len(batch) == 1 else PRIVATE} for item in batch]}

    finish(WorkNight(stores.ledger, 'owner').run(record(), NOW, call=call))
    if timing == 'after-call':
        revoke()
    for repair in (False, True):
        if repair:
            stores.work.reconcile()
        assert stores.work.get_root('private-root', contact_id='owner') is None
        assert stores.work.get_root('owner-root', contact_id='owner')['title'] != PRIVATE
        assert PRIVATE not in [row['title'] for row in stores.work.list_roots(contact_id='owner')]
        assert PRIVATE not in stores.work.summary(contact_id='owner')['titles']['unproven']
        assert PRIVATE not in read_summary(stores.work.path, contact_id='owner')['titles']['unproven']
    assert all(len(batch) == 1 for batch in inputs)


def test_title_input_excludes_effect_labels(stores):
    admit(stores, envelope(operations=[operation(tool='private_tool_label')]))
    stores.work.reconcile()

    async def call(night, **kw):
        inputs = json.loads(kw['user'])
        assert len(inputs) == 1
        assert not inputs[0].get('tools')
        return {'titles': [{'id': inputs[0]['id'], 'title': 'Change the report'}]}

    finish(WorkNight(stores.ledger, 'owner').run(record(), NOW, call=call))


def legacy_title(stores):
    admit(stores)
    stores.work.reconcile()
    root = stores.work.get_root('root', contact_id='owner')
    refs = stores.ledger.source_references(['root'], contact_id='owner', session_id='session')
    with closing(stores.work._connect()) as conn, conn:
        conn.execute('DROP TABLE work_titles')
        conn.execute('''CREATE TABLE work_titles (root_id TEXT PRIMARY KEY, title TEXT NOT NULL,
            source_refs_json TEXT NOT NULL, revision TEXT NOT NULL)''')
        conn.execute('INSERT INTO work_titles VALUES (?,?,?,?)',
                     ('root', PRIVATE, json.dumps(refs), root['revision']))
        conn.execute('UPDATE work_roots SET title=?', (PRIVATE,))


def test_legacy_batch_title_is_hidden_and_retried(stores):
    legacy_title(stores)
    # Read-only callers must also fence titles before a writable store migrates.
    readonly = WorkStore(stores.ledger, read_only=True)
    assert readonly.get_root('root', contact_id='owner')['title'] is None
    assert readonly.list_roots(contact_id='owner')[0]['title'] is None
    assert read_summary(stores.work.path, contact_id='owner')['untitled'] == 1
    worker = WorkNight(stores.ledger, 'owner')
    assert len(worker._select(NOW)) == 1


def test_interrupted_title_migration_rolls_back_and_retries(stores, monkeypatch):
    legacy_title(stores)

    class InterruptedConnection(sqlite3.Connection):
        def execute(self, sql, *args):
            if sql.startswith('ALTER TABLE work_titles'):
                raise sqlite3.OperationalError('interrupted title migration')
            return super().execute(sql, *args)

    def connect(store):
        conn = sqlite3.connect(store.path, factory=InterruptedConnection)
        conn.row_factory = sqlite3.Row
        return conn

    with monkeypatch.context() as patch:
        patch.setattr(WorkStore, '_connect', connect)
        with pytest.raises(sqlite3.OperationalError, match='interrupted title migration'):
            WorkStore(stores.ledger)
    with closing(stores.work._connect()) as conn:
        assert conn.execute('SELECT title FROM work_titles').fetchone()[0] == PRIVATE
        assert conn.execute('SELECT title FROM work_roots').fetchone()[0] == PRIVATE
    assert WorkStore(stores.ledger, read_only=True).summary(contact_id='owner')['untitled'] == 1
    worker = WorkNight(stores.ledger, 'owner')
    assert worker.store.get_root('root', contact_id='owner')['title'] is None
    assert len(worker._select(NOW)) == 1


def test_correction_during_own_title_call_cannot_publish(stores, monkeypatch):
    monkeypatch.setattr(WorkStore, 'schedule', lambda *args: None)
    admit(stores)
    stores.work.reconcile()

    async def call(night, **kw):
        correct(stores.ledger, operation_id='root-correction', performed_by='owner',
                old_contact_id='owner', contact_id='guest', source_ids=['root'],
                evidence_refs=['source:owner-confirmation'])
        return {'titles': [{'id': 'w1', 'title': PRIVATE}]}

    finish(WorkNight(stores.ledger, 'owner').run(record(), NOW, call=call))
    with closing(stores.work._connect()) as conn:
        assert conn.execute('SELECT count(*) FROM work_titles').fetchone()[0] == 0
    assert stores.work.get_root('root', contact_id='owner') is None


@pytest.mark.parametrize('action', ['correct', 'erase'])
def test_title_dependencies_rechecked_without_projection_repair(stores, monkeypatch, action):
    admit(stores, envelope(operations=[operation(evidence={'kind': 'none'})]))
    stores.ledger.record_source('title-source', contact_id='owner', session_id='session',
        messages=[{'role': 'user', 'content': PRIVATE}], derive_claims=False)
    stores.work.reconcile()
    root = stores.work.get_root('root', contact_id='owner')
    refs = stores.ledger.source_references(['title-source'], contact_id='owner', session_id='session')
    assert stores.work.write_derived('root', contact_id='owner', expected_revision=root['revision'],
                                     title=PRIVATE, source_refs=refs)
    monkeypatch.setattr(WorkStore, 'schedule', lambda *args: None)
    monkeypatch.setattr('colonymind.work.store.invalidate_projection', lambda *args: None)
    correct(stores.ledger, operation_id='title-correction', performed_by='owner', old_contact_id='owner', contact_id='guest', source_ids=['title-source'], evidence_refs=['source:owner-confirmation'])
    assert stores.work.get_root('root', contact_id='owner')['title'] is None
    assert stores.work.list_roots(contact_id='owner')[0]['title'] is None
    assert stores.work.summary(contact_id='owner')['titles']['unproven'] == []
    assert read_summary(stores.work.path, contact_id='owner')['titles']['unproven'] == []
