"""Corrections fence work at commit, even when projection repair is unavailable."""
from contextlib import closing
from types import SimpleNamespace

import pytest

from colonymind.turns.source_attribution import correct
from colonymind.work.store import WorkStore, read_summary
from test_work_projection import stores, admit, envelope, operation


@pytest.mark.parametrize('target', ['root', 'evidence', 'descendant'])
@pytest.mark.parametrize('wake_fails', [False, True])
def test_correction_revokes_work_before_repair(stores, monkeypatch, target, wake_fails):
    ledger, store = stores.ledger, stores.work
    ledger.record_source('evidence', contact_id='owner', session_id='session',
                         messages=[{'role': 'user', 'content': 'Original verification'}], derive_claims=False)
    refs = ledger.source_references(['evidence'], contact_id='owner', session_id='session')
    if target == 'descendant':
        ledger.record_source('descendant', contact_id='owner', session_id='session',
            messages=[{'role': 'assistant', 'content': 'Verification copy', '_supplied_sources': refs}], derive_claims=False)
        refs = ledger.source_references(['descendant'], contact_id='owner', session_id='session')
    admit(stores, envelope(operations=[operation(source_refs=refs)]))
    admit(stores, envelope('unrelated'), source_id='unrelated')
    store.reconcile()
    before = store.get_root('root', contact_id='owner')
    wakes = []
    def schedule():
        wakes.append(True)
        if wake_fails:
            raise RuntimeError('worker unavailable')
    monkeypatch.setattr('colonymind.work._store', lambda path: SimpleNamespace(schedule=schedule))
    kwargs = dict(operation_id='correct-guest', performed_by='owner', old_contact_id='owner',
        contact_id='guest', source_ids=['root' if target == 'root' else 'evidence'],
        evidence_refs=['source:owner-confirmation'])
    result = correct(ledger, **kwargs)
    assert wakes == [True]
    assert store.get_root('root', contact_id='owner') is None
    assert [row['id'] for row in store.list_roots(contact_id='owner')] == ['unrelated']
    assert store.summary(contact_id='owner')['counts']['proven'] == 1
    assert read_summary(store.path, contact_id='owner')['counts']['proven'] == 1
    assert not store.write_derived('root', contact_id='owner', expected_revision=before['revision'], title='Stale work title')
    with closing(ledger._connect()) as conn:
        pending = tuple(conn.execute("SELECT generation,pending FROM work_dirty WHERE root_turn_id='root'").fetchone())
    assert pending == (2, 1)
    assert correct(ledger, **kwargs) == result
    with closing(ledger._connect()) as conn:
        assert tuple(conn.execute("SELECT generation,pending FROM work_dirty WHERE root_turn_id='root'").fetchone()) == pending
    reopened = WorkStore(ledger)
    reopened.schedule()
    assert reopened.wait_pending()
    after = reopened.get_root('root', contact_id='owner')
    assert after is None if target == 'root' else after['status'] != 'proven'
    assert reopened.get_root('root', contact_id='guest') is None
