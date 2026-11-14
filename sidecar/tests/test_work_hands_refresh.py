"""Committed hands changes refresh exactly their frozen owner root."""
from contextlib import closing
import json
import sqlite3

import pytest

from colonymind.hands.store import _SCHEMA
from colonymind.work.hands import refresh
from colonymind.work.ledger import mark_dirty
from test_work_projection import stores, admit, envelope  # noqa: F401


def setup_hands(stores):
    reference = {'platform': 'test', 'chat': 'owner-chat', 'message_id': 'message-1'}
    admit(stores, envelope(operations=[], origin_ref=json.dumps(reference, sort_keys=True, separators=(',', ':'))))
    admit(stores, envelope('other'), source_id='other')
    path = stores.path / 'hands.db'
    with sqlite3.connect(path) as conn:
        conn.executescript(_SCHEMA)
        conn.execute('INSERT INTO tasks(id,origin,origin_ref,kind) VALUES (?,?,?,?)',
            ('task', 'owner_turn', json.dumps(reference), 'deep'))
        conn.execute('''INSERT INTO hands(id,task_id,attempt,intent_version,name,kind,effort,
            parent_session,executor,epoch,state,evidence) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
            ('hand', 'task', 1, 1, 'check', 'deep', 2, 'session', 'harness', '{}', 'done',
             json.dumps({'level': 'E0', 'scope': {'kind': 'build'}})))
    return path


def test_committed_hand_result_refreshes_only_exact_origin(stores, monkeypatch):
    path = setup_hands(stores)
    refreshed = []
    def enqueue(ledger, roots):
        refreshed.extend(roots)
        with closing(ledger._connect()) as conn, conn:
            for ident in roots:
                mark_dirty(conn, ident)
    monkeypatch.setattr('colonymind.work.refresh_work', enqueue)
    refresh(path)
    stores.work.reconcile()
    assert stores.work.get_root('root', contact_id='owner')['status'] == 'unproven'
    with sqlite3.connect(path) as conn:
        conn.execute('UPDATE hands SET evidence=? WHERE id=?',
            (json.dumps({'level': 'E3', 'scope': {'kind': 'build'}}), 'hand'))
    refresh(path)
    stores.work.reconcile()
    assert stores.work.get_root('root', contact_id='owner')['status'] == 'proven'
    assert refreshed == ['root', 'root']
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT count(*) FROM work_refresh_tasks').fetchone()[0] == 0
        conn.execute("UPDATE hands SET state='failed'")
        conn.rollback()
        assert conn.execute('SELECT count(*) FROM work_refresh_tasks').fetchone()[0] == 0


def test_failed_enqueue_preserves_committed_hand_refresh_for_retry(stores, monkeypatch):
    path = setup_hands(stores)
    def unavailable(*args):
        raise OSError('ledger unavailable')
    monkeypatch.setattr('colonymind.work.refresh_work', unavailable)
    with pytest.raises(OSError, match='ledger unavailable'):
        refresh(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT count(*) FROM work_refresh_tasks').fetchone()[0] == 1


def test_hands_origin_lookup_uses_its_index(stores):
    setup_hands(stores)
    with closing(stores.ledger._connect()) as conn:
        plan = conn.execute('''EXPLAIN QUERY PLAN SELECT root_turn_id FROM work_envelopes
            WHERE kind='root' AND invalidated=0 AND json_extract(envelope_json,'$.origin_ref')=?''',
            ('example',)).fetchall()
    assert any('work_envelope_origin' in row[3] for row in plan)
