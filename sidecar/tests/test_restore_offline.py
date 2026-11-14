"""An equal-marker, same-inode restore requires readers to stop and reopen."""
from contextlib import closing
import json
import os
import sqlite3

import pytest

from colonymind.backup import create_full_backup, restore_full_backup
from colonymind.contacts.comms import CommsLog
from colonymind.turns import TurnIdempotencyLedger
from colonymind.util.state_access import state_access
from test_comms_recent_snapshot import historical_source_erasure


def test_same_inode_equal_markers_restore_requires_stop_and_fresh_reader(tmp_path):
    state = tmp_path / 'state'
    state.mkdir()
    (state / 'instance-id').write_text('restore-fixture')
    ledger = TurnIdempotencyLedger(state / 'turn-idempotency.db')
    historical_source_erasure(ledger, 'person', turn_ids=['parent'])
    with closing(ledger._connect()) as conn, conn:
        conn.executemany('INSERT INTO source_projection_erasures VALUES (?,?)',
                         [('answer', 'parent'), ('same-newest', 'parent')])
    log = CommsLog(str(state / 'colonymind-comms.db'), source_ledger=ledger)
    with log._conn:
        log._conn.execute('''INSERT INTO communications
            (id,contact_id,direction,ts,source_lineage_json)
            VALUES ('answer','person','in','2099-01-01',?)''', (json.dumps({'turn_id': 'answer'}),))
    archive = create_full_backup(state, tmp_path / 'backups', include_vectors=False)
    with closing(ledger._connect()) as conn, conn:
        conn.execute("UPDATE source_projection_erasures SET turn_id='unrelated' WHERE turn_id='answer'")
    assert ('person', 'answer') not in log._erased_pairs(['answer'])
    mark = log._erased[0][4:]
    inode = os.stat(ledger.db_path).st_ino
    before = (state / 'turn-idempotency.db').read_bytes()
    # The lease is what server.lifespan holds, including for managed services
    # without a CLI pid file. Refusal happens before any database is restored.
    with state_access(state):
        with pytest.raises(ValueError, match='Stop the sidecar'):
            restore_full_backup(archive, state)
    assert (state / 'turn-idempotency.db').read_bytes() == before
    log._conn.close()
    restore_full_backup(archive, state)
    assert os.stat(ledger.db_path).st_ino == inode
    reopened = CommsLog(str(state / 'colonymind-comms.db'),
                       source_ledger=TurnIdempotencyLedger(ledger.db_path))
    assert ('person', 'answer') in reopened._erased_pairs(['answer'])
    assert reopened._erased[0][4:] == mark
    assert reopened.rollup() == {}


def test_restore_refuses_live_legacy_pid(tmp_path):
    state = tmp_path / 'state'
    state.mkdir()
    archive = create_full_backup(state, tmp_path / 'backups', include_vectors=False)
    (state / 'sidecar.pid').write_text(str(os.getpid()))
    with pytest.raises(ValueError, match='Stop the sidecar'):
        restore_full_backup(archive, state)


@pytest.mark.asyncio
async def test_server_holds_state_lease_through_shutdown(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager
    from colonymind import server
    monkeypatch.setattr(server, '_state_dir', lambda: tmp_path)
    events = []

    @asynccontextmanager
    async def serving(app):
        with pytest.raises(ValueError, match='Stop the sidecar'):
            with state_access(tmp_path, restore=True):
                pass
        events.append('startup')
        yield
        with pytest.raises(ValueError, match='Stop the sidecar'):
            with state_access(tmp_path, restore=True):
                pass
        events.append('shutdown')

    monkeypatch.setattr(server, '_lifespan', serving)
    async with server.lifespan(None):
        with pytest.raises(ValueError, match='Stop the sidecar'):
            with state_access(tmp_path, restore=True):
                pass
    with state_access(tmp_path, restore=True):
        assert events == ['startup', 'shutdown']
        with pytest.raises(ValueError, match='Stop the sidecar'):
            with state_access(tmp_path):
                pass
