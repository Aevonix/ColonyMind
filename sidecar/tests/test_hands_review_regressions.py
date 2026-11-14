"""Crash recovery, request identity and deployment migration regressions."""
import asyncio
import sqlite3

from fastapi import HTTPException
import pytest

from colonymind.events import journal
from colonymind.hands import definitions
from colonymind.hands.store import HandsStore, SCHEMA_VERSION, _SCHEMA
from test_hands_controller import c  # noqa: F401
from test_hands_definitions import editable, events  # noqa: F401

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize('operation', ['create', 'replace', 'task'])
async def test_key_reuse_with_changed_body_is_conflict(c, operation):
    row = await c.save_definition(definitions.CreateDefinition(name='First'), 'create')
    if operation == 'create':
        call = lambda body: c.save_definition(body, 'reused')
        first = definitions.CreateDefinition(name='Second')
        changed = definitions.CreateDefinition(name='Third')
    elif operation == 'replace':
        call = lambda body: c.save_definition(body, 'reused', row['id'])
        first = editable(row, expected_version=1, instructions='first')
        changed = editable(row, expected_version=1, instructions='changed')
    else:
        call = lambda body: c.queue_definition_task(row['id'], body, 'reused')
        first = definitions.TaskBrief(brief='first')
        changed = definitions.TaskBrief(brief='changed')
    saved = await call(first)
    assert await call(first) == saved
    with pytest.raises(HTTPException) as error:
        await call(changed)
    assert error.value.status_code == 409
    assert error.value.detail == 'idempotency_key_reuse'
    assert await call(first) == saved


@pytest.mark.parametrize('missing_record', [False, True])
async def test_event_crash_retry_preserves_sequence_and_unblocks_following_events(c, missing_record, monkeypatch):
    row = await c.save_definition(definitions.CreateDefinition(name='First'), 'create')
    original = journal.replay_events(after_seq=0)['events']
    seq = original[-1]['seq']
    if missing_record:
        # Marker/cursor persisted but the event file was not installed yet.
        from pathlib import Path
        import os
        for path in Path(os.environ['COLONYMIND_EVENT_JOURNAL_DIR']).glob('*.json'):
            if json_record(path).get('seq') == seq:
                path.unlink()
    await c.execute('UPDATE hands_changes SET published=0')
    await c.store.db.commit()
    await c.publish_changes()
    assert (await c.one('SELECT count(*) n FROM hands_changes WHERE published=0'))['n'] == 0
    assert journal.current_sequence() == seq
    assert journal.replay_events(after_seq=0)['events'] == original
    await c.save_definition(editable(row, role='Updated'), 'edit', row['id'])
    replay = journal.replay_events(after_seq=0)['events']
    assert len(replay) == len(original) + 1
    assert replay[-1]['seq'] > seq
    assert all(event['data']['seq'] == event['seq'] for event in replay)


def json_record(path):
    import json
    return json.loads(path.read_text())


async def test_concurrent_first_open_migrates_once_and_keeps_existing_rows(tmp_path):
    path = tmp_path / 'upgrade.db'
    with sqlite3.connect(path) as db:
        db.executescript(_SCHEMA + '\nPRAGMA journal_mode=WAL; PRAGMA user_version=1;')
        db.execute("INSERT INTO tasks(id,origin,origin_ref,kind) VALUES('t-existing','own','ref','deep')")
    stores = [HandsStore(path) for _ in range(12)]
    try:
        results = await asyncio.gather(*(store.connect() for store in stores), return_exceptions=True)
        assert not [repr(result) for result in results if isinstance(result, BaseException)]
        assert (await stores[0].db.execute_fetchall('PRAGMA user_version'))[0][0] == SCHEMA_VERSION
        assert (await stores[0].db.execute_fetchall('SELECT id FROM tasks'))[0][0] == 't-existing'
        assert len(await stores[0].db.execute_fetchall('SELECT * FROM hands_change_identity')) == 1
        await stores[0].migrate()
        assert len(await stores[0].db.execute_fetchall('SELECT * FROM hands_change_identity')) == 1
    finally:
        await asyncio.gather(*(store.close() for store in stores))
