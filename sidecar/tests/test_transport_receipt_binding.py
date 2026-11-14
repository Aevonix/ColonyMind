"""A restarted transport can distinguish an unbound claim from a native turn."""
import pytest

from colonymind.contacts.transport_ingress import ensure_schema
from test_transport_ingress import admit, opened


@pytest.mark.parametrize('bound', [False, True])
def test_receipt_view_reports_native_binding_after_reopen(tmp_path, bound):
    path = tmp_path / 'communications.db'
    conn, store = opened(path)
    try:
        ensure_schema(conn)
        receipt = admit(store)['receipt_id']
        batch = dict(producer='bridge', receipt_ids=[receipt], batch_id='a' * 32 + ':batch')
        store.handoff(**batch)
        if bound:
            store.handoff(**batch, native_turn={
                'session_id': 'session', 'task_id': 'task', 'turn_id': 'turn'})
    finally:
        conn.close()
    conn, store = opened(path)
    try:
        view = store.view(store.get(receipt))
        assert view['state'] == 'handed_off'
        assert view['native_bound'] is bound
        resumed = store.handoff(**(batch | {'batch_id': 'b' * 32 + ':batch'}))
        assert resumed['may_dispatch'] is False
        assert resumed['native_bound'] is bound
        assert resumed['uncertain_receipt_ids'] == ([] if bound else [receipt])
        assert store.get(receipt)['state'] == ('handed_off' if bound else 'uncertain')
    finally:
        conn.close()
