"""PM-scoped outage behavior and recovery, preserving unrelated consumers."""
from contextlib import closing
import logging
import sqlite3
from types import SimpleNamespace

import pytest

from colonymind.memory import curated, curated_health
from test_forget_fix11 import (memory_app, source_app, request, break_curated,
                              assert_warning_is_content_free, forget_report, PRIVATE)
from test_mind_loop import pinned_clock
from test_hands_controller import c, plan
from test_mind_lessons import fx, admit, owner_turn, FIELDS


def outage(monkeypatch, ledger, unreadable=False):
    curated.disable(ledger, sqlite3.OperationalError('fixture'))
    if unreadable:
        def broken(*args, **kwargs):
            raise sqlite3.OperationalError('fixture')
        monkeypatch.setattr(curated, '_cache_of', broken)


@pytest.mark.parametrize('fence', ['fenced_read', 'withdrawal_fenced_read'])
@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('unreadable', [False, True])
async def test_shared_default_retains_unknown_start_fail_open(memory_app, monkeypatch, fence,
                                                              asynchronous, unreadable):
    _, ledger = memory_app
    outage(monkeypatch, ledger, unreadable)
    calls = []
    def read():
        calls.append(True)
        return 'native'
    async def aread():
        return read()
    result = getattr(curated, fence)(ledger, aread if asynchronous else read, empty='withheld')
    if asynchronous:
        result = await result
    assert result == 'native'
    assert calls == [True]


@pytest.mark.parametrize('reader', ['candidates', 'current_candidates'])
@pytest.mark.parametrize('unreadable', [False, True])
def test_other_candidate_callers_retain_native(memory_app, monkeypatch, reader, unreadable):
    _, ledger = memory_app
    candidate = {'turn_id': 'report', 'content': 'fixture'}
    outage(monkeypatch, ledger, unreadable)
    if reader == 'candidates':
        result = curated.candidates(ledger, 'person', 'fixture', [candidate], [candidate])
        assert result == ([candidate], [candidate])
    else:
        assert curated.current_candidates(ledger, [candidate], contact_id='person') == [candidate]


@pytest.mark.parametrize('unreadable', [False, True])
async def test_opinion_job_is_not_source_erased(memory_app, monkeypatch, unreadable):
    from colonymind.mind.opinions import run_one
    from colonymind.self_model.judgments import SelfJudgments
    _, ledger = memory_app
    ledger.record_source('opinion-fixture', contact_id='person', session_id='work',
                         messages=[{'role': 'user', 'content': 'fixture'}], derive_claims=False)
    store = SelfJudgments(ledger, owner_id='person')
    job = {'ref': 'opinion-fixture', 'kind': 'turn', 'enqueued_at': store.clock()}
    with closing(ledger._connect()) as conn, conn:
        conn.execute('INSERT OR IGNORE INTO opinion_jobs(ref,kind,contact_id,enqueued_at) VALUES (?,?,?,?)',
                     (job['ref'], job['kind'], 'person', job['enqueued_at']))
    monkeypatch.setattr(store, 'next_job', lambda: job)
    outage(monkeypatch, ledger, unreadable)
    router = SimpleNamespace(supports_function_routing=True, function_deadline_seconds=lambda **kw: 1)
    assert await run_one(store, router, enabled=True)
    with closing(ledger._connect()) as conn:
        row = conn.execute('SELECT disposition,done_at FROM opinion_jobs WHERE ref=?', (job['ref'],)).fetchone()
    assert row['disposition'] == 'no_premise' and row['done_at'] is not None


@pytest.mark.parametrize('unreadable', [False, True])
async def test_hands_guard_and_list_remain_available(c, memory_app, monkeypatch, unreadable):
    from colonymind.hands import wire
    _, ledger = memory_app
    hand = await plan(c)
    monkeypatch.setattr(c, 'ledger', ledger)
    outage(monkeypatch, ledger, unreadable)
    row = await c.guard(hand['hand_id'])
    assert row['id'] == hand['hand_id']
    listed = await c.list_hands(wire.ListHandsRequest())
    assert hand['hand_id'] in [item['id'] for item in listed['hands']]


@pytest.mark.parametrize('unreadable', [False, True])
def test_digest_native_claim_is_not_withheld(memory_app, monkeypatch, unreadable):
    from colonymind.contacts.digest import not_live
    _, ledger = memory_app
    with closing(ledger._connect()) as conn:
        row = conn.execute('SELECT id FROM source_claims WHERE turn_id=?', ('report',)).fetchone()
        # Use the actual digest membership check with a retained native claim.
        if row is None:
            conn.execute("INSERT INTO source_claims(id,turn_id,message_hash,subject_key,predicate,value_key,data_json) VALUES (?,?,?,?,?,?,?)",
                         ('claim:fixture-claim', 'report', 'fixture', 'fixture', 'fixture', 'fixture', '{}'))
            conn.commit()
            ident = 'claim:fixture-claim'
        else:
            ident = row[0]
        expected = not_live(conn, [ident], contact_id='person')
        assert expected == []
        outage(monkeypatch, ledger, unreadable)
        assert not_live(conn, [ident], contact_id='person') == expected


@pytest.mark.parametrize('unreadable', [False, True])
def test_lessons_entries_and_admit_keep_supersession(fx, monkeypatch, unreadable):
    owner_turn(fx, 'verdict-1', 'fixture')
    owner_turn(fx, 'verdict-2', 'fixture', session='day-02')
    first = admit(fx)
    assert first is not None
    before = fx.mind.lessons._entries()
    outage(monkeypatch, fx.ledger, unreadable)
    assert fx.mind.lessons._entries() == before
    second = admit(fx, fields={**FIELDS, 'content': 'replacement fixture'}, lineage=('verdict-2',))
    assert second is not None and second.supersedes == first.id
    curated.enable(fx.ledger)
    assert fx.mind.lessons.get(first.id).status == 'superseded'
    assert [item.id for item in fx.mind.lessons.all()] == [second.id]


@pytest.mark.parametrize('stage', ['start', 'lane', 'after', 'unknown_cursor', 'lost_cursor'])
async def test_context_preserves_all_nonmemory_sections(memory_app, monkeypatch, caplog, stage):
    from colonymind.api.routers import host
    app, ledger = memory_app
    from datetime import datetime, timezone
    from colonymind.util import temporal
    fixed = datetime(2026, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(host, '_temporal_now', lambda: fixed)
    monkeypatch.setattr(temporal, 'now_utc', lambda: fixed)
    healthy = await request(app, 'context')
    assert healthy.status_code == 200
    expected = [s for s in healthy.json()['sections'] if s['id'] != 'colonymind-memory']
    assert any(s['id'] == 'temporal-context' for s in expected)
    if stage in ('after', 'lost_cursor'):
        original = host._read_context_assemble
        async def after(*args, **kwargs):
            result = await original(*args, **kwargs)
            if stage == 'after':
                curated.disable(ledger, sqlite3.OperationalError('fixture'))
            else:
                monkeypatch.setattr(curated, '_withdrawal_version', lambda reader: None)
            return result
        monkeypatch.setattr(host, '_read_context_assemble', after)
    elif stage == 'unknown_cursor':
        monkeypatch.setattr(curated, '_withdrawal_version', lambda reader: None)
    else:
        break_curated(monkeypatch, ledger, stage)
    with caplog.at_level(logging.WARNING):
        response = await request(app, 'context')
    assert response.status_code == 200
    data = response.json()
    assert data['sections'] == expected
    assert data['budget']['unavailable'] is True
    assert_warning_is_content_free(caplog)


def test_owner_notice_describes_withheld_recall(memory_app):
    _, ledger = memory_app
    health = curated_health.for_ledger(ledger)
    health.pending.clear()
    curated.disable(ledger, sqlite3.OperationalError('fixture'))
    notices = []
    curated_health.owner_notices(ledger, SimpleNamespace(owner_id='person', notice=lambda **kw: notices.append(kw)))
    assert len(notices) == 1
    text = notices[0]['text']
    assert 'recall is withheld' in text.lower()
    assert 'nothing forgotten can resurface through recall' in text.lower()
    assert '—' not in text and not any(c.isdigit() for c in text)
    assert len(text) < 220


@pytest.mark.parametrize('consumer', ['voice', 'context', 'search'])
async def test_next_strict_read_recovers_without_restart(memory_app, consumer):
    app, ledger = memory_app
    assert (await request(app, consumer, 'warmup')).status_code == 200
    curated.disable(ledger, sqlite3.OperationalError('fixture'))
    version = curated_health.for_ledger(ledger).failure_version
    response = await request(app, consumer, 'recovered')
    assert response.status_code == 200
    assert curated.available(SimpleNamespace(db_path=ledger.db_path))
    assert curated_health.for_ledger(ledger).failure_version == version
    data = response.json()
    if consumer == 'context':
        assert any(s['id'] == 'colonymind-memory' for s in data['sections'])
        assert not data['budget'].get('unavailable', False)
    else:
        assert data['count'] > 0


@pytest.mark.parametrize('consumer', ['voice', 'context', 'search'])
async def test_failed_reprobe_is_once_and_bounded(memory_app, monkeypatch, consumer):
    app, ledger = memory_app
    await request(app, consumer, 'warmup')
    curated.disable(ledger, sqlite3.OperationalError('fixture'))
    original = sqlite3.connect
    original_reprobe = curated._reprobe
    probes = []
    probing = []
    def reprobe(reader):
        probing.append(True)
        try:
            return original_reprobe(reader)
        finally:
            probing.pop()
    def connect(*args, **kwargs):
        if probing:
            probes.append(kwargs.get('timeout'))
            raise sqlite3.OperationalError('fixture')
        return original(*args, **kwargs)
    monkeypatch.setattr(curated, '_reprobe', reprobe)
    monkeypatch.setattr(sqlite3, 'connect', connect)
    response = await request(app, consumer, 'unavailable')
    assert probes and len(probes) == 1
    assert 0 < probes[0] <= .1
    assert not curated.available(ledger)
    assert response.status_code == (503 if consumer == 'voice' else 200)


@pytest.mark.parametrize('fence', ['strict_fenced_read', 'strict_withdrawal_fenced_read'])
@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('failure', ['unknown', 'disabled', 'recovered', 'cursor'])
async def test_strict_fences_keep_failure_sticky(memory_app, monkeypatch, fence, asynchronous, failure):
    _, ledger = memory_app
    cursor = '_journal_version' if fence == 'strict_fenced_read' else '_withdrawal_version'
    if failure == 'unknown':
        monkeypatch.setattr(curated, cursor, lambda ledger: None)
    def read():
        if failure == 'cursor':
            monkeypatch.setattr(curated, cursor, lambda ledger: None)
        else:
            curated.disable(ledger, sqlite3.OperationalError('fixture'))
            if failure == 'recovered':
                curated.enable(ledger)
        return 'uncertified'
    async def aread():
        return read()
    # Step 1 used these strict semantics under the shared names.
    # This also runs the retained guarantee against that exact commit.
    function = getattr(curated, fence, None) or getattr(curated, fence.removeprefix('strict_'))
    result = function(ledger, aread if asynchronous else read, empty='withheld')
    if asynchronous:
        result = await result
    assert result == 'withheld'


@pytest.mark.parametrize('consumer', ['voice', 'context', 'search'])
async def test_a_partial_outage_the_probe_cannot_see_opens_one_incident(memory_app, monkeypatch, consumer):
    # The probe reads only the journal cursor; a state read that keeps failing must
    # not open a new incident (and a new HOME notice) on every strict read.
    app, ledger = memory_app
    forget_report(ledger)
    assert (await request(app, consumer, 'warm')).status_code == 200
    health = curated_health.for_ledger(ledger)
    health.pending.clear()

    def fail(*a, **k):
        raise sqlite3.OperationalError('fixture')
    monkeypatch.setattr(curated, '_state', fail)
    for i in range(5):
        response = await request(app, consumer, 'r%d' % i)
        assert PRIVATE not in response.text
    assert len({incident['id'] for incident in health.pending}) == 1
