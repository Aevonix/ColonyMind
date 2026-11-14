"""Selective erasure across source, projection and disconnected-host boundaries."""
import json
import sqlite3
from unittest.mock import AsyncMock
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response, Request
import pytest
from colonymind.api.routers import host
from colonymind.turns import TurnIdempotencyLedger
from colonymind.memory import curated

@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    return TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")

def source(ledger, turn_id="turn-a", *, contact="contact-a", session="session-a", messages=None):
    messages = messages or [{"role": "user", "content": "The hydrofoil password is pâss-unique."}]
    ledger.record_source(turn_id, contact_id=contact, session_id=session, messages=messages)
    return messages

def queued(turn_id, messages, *, contact="contact-a"):
    return {"turn_id": turn_id, "contact_id": contact, "session_id": "session-a", "checkpoint_messages": messages}

def test_selective_tombstones_survive_restart(ledger):
    from colonymind.memory import curated
    messages = source(ledger)
    survivor = {"role": "user", "content": "The bicycle is blue."}
    source(ledger, "checkpoint-a", messages=messages + [survivor])
    source(ledger, "other-contact", contact="contact-b", messages=messages)
    source(ledger, "other-session", session="session-b", messages=messages)
    curated.forget_sources(ledger, 'contact-a', source_ids=['turn-a'])
    reopened = TurnIdempotencyLedger(ledger.db_path)
    def search(query, person='contact-a'):
        rows = reopened.search_sources(query, contact_id=person, session_id='session-a')
        return curated.current_candidates(reopened, rows, contact_id=person)
    assert curated.is_forgotten(reopened, 'turn-a')
    assert search('bicycle')
    assert all(row['turn_id'] == 'other-session' for row in search('password'))
    assert search('password', 'contact-b')
    with sqlite3.connect(ledger.db_path) as conn:
        checkpoint = json.loads(conn.execute("SELECT messages_json FROM turn_sources WHERE turn_id='checkpoint-a'").fetchone()[0])
    assert checkpoint[-1] == survivor and len(checkpoint) == 2
    assert 'hydrofoil' in checkpoint[0]['content']
    source(reopened, 'checkpoint-rewrapped', messages=messages + [survivor])
    assert all(row['turn_id'] == 'other-session' for row in search('password'))

def test_ambiguity_and_cross_contact_selection_are_non_mutating(ledger):
    message = source(ledger)
    source(ledger, "checkpoint-a", messages=message)
    with pytest.raises(ValueError, match="ambiguous_source"):
        curated.forget_sources(ledger, "contact-a", old_text=message[0]["content"], session_id="session-a")
    with pytest.raises(ValueError, match="source_not_found"):
        curated.forget_sources(ledger, "contact-b", source_ids=["turn-a"])
    assert curated.freshness_feed(ledger, "contact-a")["head"] == 0

def test_feed_pages_and_detects_restore_behind_host(ledger):
    source(ledger, "a")
    source(ledger, "b", session="session-b")
    curated.forget_sources(ledger, "contact-a", source_ids=["a"])
    curated.forget_sources(ledger, 'contact-a', source_ids=['b'])
    first = curated.freshness_feed(ledger, "contact-a", limit=1)
    assert first["complete"] is False
    second = curated.freshness_feed(ledger, "contact-a", after=first["through"])
    assert second["complete"] is True and second["head"] == 2
    with pytest.raises(ValueError, match="cursor"):
        curated.freshness_feed(ledger, "contact-a", after=3)


def test_repeat_erase_retains_derived_cleanup_targets(ledger):
    messages = source(ledger)
    source(ledger, "checkpoint-a", messages=messages + [{"role": "user", "content": "Unrelated."}])
    first = curated.forget_sources(ledger, "contact-a", source_ids=["turn-a"])
    again = curated.forget_sources(ledger, "contact-a", source_ids=["turn-a"])
    assert again == first


def test_historical_erasure_feed_still_pages_and_detects_restore(ledger):
    # Retained pre-cutover events remain readable without the retired writer.
    with ledger._connect() as conn:
        conn.executemany('INSERT INTO source_erasures(contact_id,turn_id,session_id,message_hashes_json) '
                         'VALUES (?,?,?,?)', [('contact-a', 'old-a', 'old-session', '[]'),
                                               ('contact-a', 'old-b', 'old-session', '[]')])
    first = ledger.erasure_feed('contact-a', limit=1)
    assert first['head'] == 2 and first['through'] == 1 and not first['complete']
    second = ledger.erasure_feed('contact-a', after=first['through'])
    assert second['complete'] and second['events'][0]['turn_id'] == 'old-b'
    with pytest.raises(ValueError, match='cursor'):
        ledger.erasure_feed('contact-a', after=3)
