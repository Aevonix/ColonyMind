"""Actual retry HTTP and SQLite history stay atomic for initiatives."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from secrets import token_hex

import sqlite3

from fastapi.testclient import TestClient
import pytest

from colonymind.api.routers import host
from colonymind.initiatives.store import InitiativeStore
from onekey import KEY, _principal, _write_keyring, keyed_host_app as _app


@pytest.fixture
def failed_work(tmp_path, monkeypatch):
    state = tmp_path/'state'
    state.mkdir()
    initiatives = InitiativeStore(state)
    accepted = initiatives.create(type='RESEARCH_DEEP_DIVE', description='Compare two neutral notes')
    initiatives.update(accepted.id, status='failed', attempt_count=1)
    monkeypatch.setattr(host, '_initiative_store', initiatives)
    keyring = tmp_path/'keys.json'
    _write_keyring(keyring, [
        _principal(principal='operator', secret=token_hex(24), scopes=['api:access']),
        _principal(principal='reader', secret=token_hex(24), scopes=['context:read'])])
    with TestClient(_app(api_key=KEY), raise_server_exceptions=False) as api:
        yield api, initiatives, accepted.id
    initiatives.close()


def snapshot(store, identifier):
    with closing(sqlite3.connect(store._db_path, timeout=.2)) as db:
        db.row_factory = sqlite3.Row
        row = dict(db.execute('SELECT * FROM initiatives WHERE id=?', (identifier,)).fetchone())
        history = [dict(item) for item in db.execute(
            'SELECT * FROM assignment_history WHERE initiative_id=? ORDER BY id', (identifier,))]
        return row, history


def retry(api, identifier):
    return api.post('/v1/host/initiatives/'+identifier+'/retry',
                    headers={'Authorization':'Bearer ' + KEY})




def test_history_write_failure_rolls_back_retry_and_releases_writer(failed_work):
    api, store, identifier = failed_work
    before = snapshot(store, identifier)
    with closing(sqlite3.connect(store._db_path)) as db, db:
        db.execute("""CREATE TRIGGER reject_retry_history BEFORE INSERT ON assignment_history
            WHEN NEW.action='retry' BEGIN SELECT RAISE(ABORT, 'forced history failure'); END""")
    response = retry(api, identifier)
    assert response.status_code == 500
    assert not store._db.in_transaction
    assert snapshot(store, identifier) == before
    with closing(sqlite3.connect(store._db_path, timeout=.2)) as db, db:
        db.execute('BEGIN IMMEDIATE')
        assert db.execute('SELECT status FROM initiatives WHERE id=?', (identifier,)).fetchone()[0] == 'failed'
        db.execute('DROP TRIGGER reject_retry_history')
    assert retry(api, identifier).status_code == 200
    assert store.get(identifier).status == 'pending'


def test_simultaneous_retry_requests_record_one_transition(failed_work):
    api, store, identifier = failed_work
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: retry(api, identifier), range(2)))
    assert sorted(response.status_code for response in responses) == [200, 400]
    row, history = snapshot(store, identifier)
    assert row['status'] == 'pending' and row['attempt_count'] == 1
    assert sum(event['action'] == 'retry' for event in history) == 1
    assert not store._db.in_transaction
