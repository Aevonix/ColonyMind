"""Offline gate for affect reads: no writes, identical decay and safe erasure."""
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from colonymind.contacts import affect as affect_module
from colonymind.memory.voice_affect import speaker_words
from test_voice_affect import affect_app, call, stamp  # noqa: F401
from test_voice_affect_gate import seed, test_voice_affect_seeded_latency_gate as latency_gate


def snapshot(store):
    """Include the WAL so a deferred checkpoint cannot hide a database write."""
    path = Path(store._db_path)
    files = {}
    for suffix in ('', '-wal', '-shm'):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            files[suffix] = (candidate.stat().st_mtime_ns, candidate.stat().st_size,
                             candidate.read_bytes() if suffix != '-shm' else None)
    return {
        'total_changes': store._conn.total_changes,
        'in_transaction': store._conn.in_transaction,
        'events': [tuple(row) for row in store._conn.execute('SELECT * FROM affect_events ORDER BY id')],
        'state': [tuple(row) for row in store._conn.execute('SELECT * FROM affect_state ORDER BY contact_id')],
        'files': files,
    }


@contextmanager
def assert_readonly(store):
    before = snapshot(store)
    attempts, statements = [], []

    def authorize(action, *args):
        if action not in {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}:
            attempts.append((action, args))
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    store._conn.set_authorizer(authorize)
    store._conn.set_trace_callback(statements.append)
    try:
        yield
    finally:
        store._conn.set_authorizer(None)
        store._conn.set_trace_callback(None)
        assert attempts == [], f'Attempted a non-read operation: {attempts}'
        assert statements and all(sql.lstrip().upper().startswith('SELECT ') for sql in statements)
        assert snapshot(store) == before


@pytest.mark.parametrize('observation_age', [0, 6 * 3600])
async def test_voice_call_never_writes_even_with_hours_old_state(affect_app, observation_age):
    state = affect_app
    state.affect.create_event(contact_id='owner', valence=-0.8, arousal=0.1,
                              timestamp=stamp(state.now-observation_age))
    state.affect._conn.execute('UPDATE affect_state SET last_updated=?', (stamp(state.now-6*3600),))
    state.affect._conn.commit()
    expected = 'Sam: frustrated, low energy' if observation_age == 0 else ''
    with assert_readonly(state.affect):
        for _ in range(3):
            assert await call(state) == expected
            state.now += 0.01


@pytest.mark.parametrize('hours', [-1, 0, 0.0000001, 0.5, 6, 72])
@pytest.mark.parametrize('decay_factor', [0.95, 0.8])
def test_projection_equals_get_state_at_same_instant(affect_app, hours, decay_factor):
    state = affect_app
    state.affect._decay_factor = decay_factor
    state.affect.create_event(contact_id='owner', valence=-0.8, arousal=0.1,
                              timestamp=stamp(state.now))
    state.affect.create_event(contact_id='owner', valence=0.1234, arousal=0.8765,
                              timestamp=stamp(state.now))
    state.affect._conn.execute('UPDATE affect_state SET last_updated=?', (stamp(state.now-hours*3600),))
    state.affect._conn.commit()
    now = datetime.fromtimestamp(state.now, timezone.utc)
    with assert_readonly(state.affect):
        projected = state.affect.project_state('owner', now=now)
    # The fixture freezes _now at this same instant. The ordinary reader still
    # persists positive decay and retains its full return shape.
    before = state.affect._conn.total_changes
    assert projected == state.affect.get_state('owner')
    assert state.affect._conn.total_changes == before + (1 if hours > 0 else 0)


def test_shared_decay_helper_is_pure_and_used_by_both_readers(affect_app, monkeypatch):
    state = affect_app
    state.affect.create_event(contact_id='owner', valence=-0.8, arousal=0.9,
                              timestamp=stamp(state.now))
    state.affect._conn.execute('UPDATE affect_state SET last_updated=?', (stamp(state.now-3600),))
    state.affect._conn.commit()
    original = affect_module._project_decay
    inputs = []

    def observe(value, **kwargs):
        before = dict(value)
        result = original(value, **kwargs)
        assert value == before and result is not value
        inputs.append((before, kwargs))
        return result

    monkeypatch.setattr(affect_module, '_project_decay', observe)
    with assert_readonly(state.affect):
        projection = state.affect.project_state('owner')
    assert projection == state.affect.get_state('owner')
    assert len(inputs) == 2 and inputs[0] == inputs[1]


async def test_other_contacts_invalid_sources_do_not_suppress_speaker(affect_app):
    state = affect_app
    state.affect.create_event(contact_id='other', valence=-0.8, timestamp=stamp(state.now))
    raw = json.dumps({'turn_id': 'missing', 'session_id': 's', 'message_hashes': []})
    state.affect._conn.execute('UPDATE affect_events SET source_lineage_json=?', (raw,))
    state.affect._conn.commit()
    state.affect.create_event(contact_id='owner', valence=0.8, arousal=0.9, timestamp=stamp(state.now))
    with assert_readonly(state.affect):
        assert await call(state) == 'Sam: pleased, high energy'


async def test_missing_contact_has_no_readonly_projection(affect_app):
    with assert_readonly(affect_app.affect):
        assert affect_app.affect.project_state('missing') is None
        assert await call(affect_app) == ''
    assert affect_app.affect.get_state('missing')['event_count'] == 0


def test_voice_never_falls_back_to_mutating_store_interface():
    calls = []
    store = SimpleNamespace(get_state=lambda person: calls.append(person))
    assert speaker_words(store, 'owner', now=0) == ''
    assert calls == []


def test_projection_context_resets_after_reader_failure(affect_app, monkeypatch):
    state = affect_app
    state.affect.create_event(contact_id='owner', valence=-0.8, timestamp=stamp(state.now))
    state.affect._conn.execute('UPDATE affect_state SET last_updated=?', (stamp(state.now-3600),))
    state.affect._conn.commit()
    original = state.affect.get_state

    def broken(person):
        raise RuntimeError('unavailable')

    before = snapshot(state.affect)
    with monkeypatch.context() as patch:
        patch.setattr(state.affect, 'get_state', broken)
        with pytest.raises(RuntimeError, match='unavailable'):
            state.affect.project_state('owner')
    assert snapshot(state.affect) == before
    assert original('owner')['last_updated'] == stamp(state.now)
    assert state.affect._conn.total_changes == before['total_changes'] + 1


async def test_voice_works_with_sqlite_readonly_connection(affect_app):
    state = affect_app
    state.affect.create_event(contact_id='owner', valence=-0.8, arousal=0.1, timestamp=stamp(state.now))
    state.affect._conn.execute('UPDATE affect_state SET last_updated=?', (stamp(state.now-3600),))
    state.affect._conn.commit()
    state.affect._conn.close()
    state.affect._conn = sqlite3.connect(Path(state.affect._db_path).as_uri() + '?mode=ro', uri=True)
    state.affect._conn.row_factory = sqlite3.Row
    # The first read on a fresh WAL connection creates -wal/-shm; that is SQLite's
    # own open, not a write by the voice path, so warm it before the snapshot.
    state.affect._conn.execute('SELECT 1 FROM affect_state LIMIT 1').fetchall()
    with assert_readonly(state.affect):
        assert await call(state) == 'Sam: frustrated, low energy'


@pytest.mark.parametrize(('valence', 'arousal'), [(float('inf'), 0.3), (-1.1, 0.3), (0, 1.1)])
def test_invalid_projected_state_is_silent(affect_app, valence, arousal):
    state = affect_app
    state.affect.create_event(contact_id='owner', valence=0.8, timestamp=stamp(state.now))
    state.affect._conn.execute('UPDATE affect_state SET current_valence=?, current_arousal=?',
                              (valence, arousal))
    state.affect._conn.commit()
    with assert_readonly(state.affect):
        assert speaker_words(state.affect, 'owner', now=state.now) == ''


@pytest.mark.hardware_latency
async def test_seeded_p95_and_readonly_gate(affect_app, monkeypatch, hardware_latency):
    # Reuse the 10k-event / 10k-appraisal p95 and exact line on every response.
    # Seed before installing the read-only guards; avoid seeding a second time.
    state = affect_app
    counts = seed(state)
    monkeypatch.setattr('test_voice_affect_gate.seed', lambda fixture: counts)
    # The authorizer fires per column read, so it would itself break the 20 ms
    # budget on 10k rows: time unhooked and prove no write by snapshot, then run
    # one call under the full statement/authorizer guard.
    before = snapshot(state.affect)
    await latency_gate(state, monkeypatch, hardware_latency)
    assert snapshot(state.affect) == before
    with assert_readonly(state.affect):
        assert speaker_words(state.affect, 'owner', now=state.now) == 'frustrated, low energy'


@pytest.mark.parametrize('invalidity', ['missing', 'messages', 'session', 'contact', 'scope', 'projection'])
async def test_any_invalid_source_suppresses_voice_without_purging(affect_app, invalidity):
    state = affect_app
    messages = [{'role': 'user', 'content': 'Synthetic affect evidence.'}]
    state.ledger.record_source('older-source', contact_id='owner', session_id='source-session',
        messages=messages, occurred_at=stamp(state.now-100), derive_claims=False)
    lineage, _ = state.affect.source_input('older-source', 'owner')
    state.affect.create_event(contact_id='owner', valence=-0.8, arousal=0.1,
        timestamp=stamp(state.now-100), source_lineage=lineage)
    # A newer valid, unlinked event must not hide the older invalid contribution.
    state.affect.create_event(contact_id='owner', valence=-0.8, arousal=0.1,
                              timestamp=stamp(state.now))
    assert await call(state) == 'Sam: frustrated, low energy'
    with closing(state.ledger._connect()) as conn, conn:
        if invalidity == 'missing':
            conn.execute("DELETE FROM turn_sources WHERE turn_id='older-source'")
        elif invalidity == 'projection':
            conn.execute('INSERT INTO source_projection_erasures VALUES (?,?)', ('older-source', 'erasure-cause'))
        else:
            column, value = {'messages': ('messages_json', json.dumps([{'role': 'user', 'content': 'Changed.'}])), 'session': ('session_id', 'changed-session'), 'contact': ('contact_id', 'other'), 'scope': ('scope', 'session')}[invalidity]
            conn.execute(f"UPDATE turn_sources SET {column}=? WHERE turn_id='older-source'", (value,))
    with assert_readonly(state.affect):
        assert state.affect.project_state('owner') is None
        for _ in range(2):
            assert await call(state) == ''
    # Other callers retain the original cleanup and recompute behavior.
    ordinary = state.affect.get_state('owner')
    assert ordinary['event_count'] == 1
    assert state.affect._conn.execute('SELECT count(*) FROM affect_events').fetchone()[0] == 1
