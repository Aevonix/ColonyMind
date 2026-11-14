"""Unavailable curated protection withholds recall, without stopping text turns."""
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from onekey import KEY
from colonymind.memory import curated, curated_health
from colonymind.turns import TurnIdempotencyLedger
from test_canonical_memory_search import memory_app
from test_turn_source_evidence import source_app
from test_voice_affect import affect_app, add_appraisal, call


PRIVATE = 'The hydrofoil departure is Friday at nine.'


async def request(app, consumer, identity='fixture'):
    body = {'identity': {'host_id': 'fixture'}, 'person_id': 'person',
            'session_id': 'later', 'query': 'hydrofoil'}
    if consumer == 'voice':
        route = 'memory/voice-recall'
        body.update(k=5, max_tokens=600, utterance_id=identity, request_id=identity)
    elif consumer == 'search':
        route = 'memory/search'
    else:
        route = 'context/assemble'
        body = {'identity': {'host_id': 'fixture'},
                'context': {'contact_id': 'person', 'session_id': 'later'},
                'incoming_message': {'role': 'user', 'content': 'hydrofoil'}}
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://fixture',
                           headers={'Authorization': 'Bearer ' + KEY}) as client:
        return await client.post('/v1/host/' + route, json=body)


def forget_report(ledger):
    # A claim-level curated forget fences its source only in the curated layer.
    # Unlike forget_sources, it has no independent source-only tombstone fallback.
    with closing(ledger._connect()) as conn, conn:
        conn.execute('''INSERT INTO curated_memory_events
            (person_id,memory_id,revision,operation,request_id,request_digest,
             applied_at,data_json,source_ids_json) VALUES (?,?,?,?,?,?,?,?,?)''',
            ('person', 'claim:fixture', 2, 'forget', 'forget-fixture', 'fixture',
             '2026-01-01T00:00:00+00:00', None, '["report"]'))


def break_curated(monkeypatch, ledger, stage):
    fired = []
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError(PRIVATE)
    if stage == 'start':
        # Both withdrawal and leaf journal reads become unknown before recall.
        monkeypatch.setattr(curated, '_cache_of', fail)
    else:
        original = TurnIdempotencyLedger.search_sources
        def after_first_lane(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            if not fired:
                fired.append(True)
                curated.disable(self, sqlite3.OperationalError(PRIVATE))
                monkeypatch.setattr(curated, '_state', fail)
            return result
        monkeypatch.setattr(TurnIdempotencyLedger, 'search_sources', after_first_lane)
    return fired


def assert_warning_is_content_free(caplog):
    warnings = [record for record in caplog.records if record.levelno >= logging.WARNING
                and record.name.startswith('colonymind.memory')]
    assert warnings
    for record in warnings:
        assert PRIVATE not in record.getMessage()
        assert 'Friday' not in record.getMessage() and 'person' not in record.getMessage()
        assert record.exc_info is None


@pytest.mark.parametrize('stage', ['start', 'lane'])
async def test_voice_unavailable_never_publishes(memory_app, monkeypatch, caplog, stage):
    app, ledger = memory_app
    forget_report(ledger)
    assert PRIVATE not in (await request(app, 'voice', 'healthy')).text
    fired = break_curated(monkeypatch, ledger, stage)
    with caplog.at_level(logging.WARNING):
        response = await request(app, 'voice', 'broken')
    assert PRIVATE not in response.text
    assert response.status_code == 503
    assert response.json()['detail']['code'] == 'memory_backend_unavailable'
    assert stage == 'start' or fired


@pytest.mark.parametrize('stage', ['start', 'lane'])
@pytest.mark.parametrize('consumer', ['context', 'search'])
async def test_text_unavailable_withholds_memory_and_continues(memory_app, monkeypatch, caplog,
                                                               stage, consumer):
    app, ledger = memory_app
    forget_report(ledger)
    assert PRIVATE not in (await request(app, consumer)).text
    fired = break_curated(monkeypatch, ledger, stage)
    with caplog.at_level(logging.WARNING):
        response = await request(app, consumer)
    assert response.status_code == 200
    assert PRIVATE not in response.text
    data = response.json()
    if consumer == 'search':
        assert data['content'] == '' and data['count'] == 0 and data['source_refs'] == []
    else:
        assert not any(s['id'] == 'colonymind-memory' for s in data['sections'])
    assert stage == 'start' or fired
    assert_warning_is_content_free(caplog)


@pytest.mark.parametrize('stage', ['start', 'during'])
async def test_affect_unavailable_omits_line(affect_app, monkeypatch, stage):
    from colonymind.api.routers import host, social_state
    state = affect_app
    monkeypatch.setattr(social_state, 'appraisal_store', lambda: state.appraisals)
    monkeypatch.setattr(host, 'time', SimpleNamespace(time=lambda: state.now, perf_counter=lambda: 0.))
    add_appraisal(state)
    assert await call(state)
    if stage == 'start':
        break_curated(monkeypatch, state.ledger, 'start')
    else:
        original = state.appraisals.voice_appraisal
        def disabling(*args, **kwargs):
            result = original(*args, **kwargs)
            curated.disable(state.ledger, sqlite3.OperationalError(PRIVATE))
            return result
        monkeypatch.setattr(state.appraisals, 'voice_appraisal', disabling)
    assert await call(state) == ''


@pytest.fixture
def normal_app(monkeypatch, request, tmp_path):
    from colonymind.api.routers import host
    from colonymind.turns import idempotency
    from colonymind.util import temporal
    now = datetime(2026, 10, 20, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(idempotency, 'ledger_stamp', lambda: now.isoformat())
    monkeypatch.setattr(temporal, 'now_utc', lambda: now)
    monkeypatch.setattr(host, '_temporal_now', lambda: now)
    # The clock line names the configured assistant and zone. Pin what it is built from, as an instance
    # configures it: the default name, the zone in the environment, no stored zone, and a machine that
    # names none. The pinned bytes are then the same on every machine.
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(tmp_path))
    for name in ('COLONYMIND_PERSONA_NAME', 'COLONYMIND_DEFAULT_CONTACT_TIMEZONE', 'TZ'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('COLONYMIND_AGENT_TIMEZONE', 'America/New_York')
    monkeypatch.setattr(temporal, '_system_timezone', lambda: None)
    return request.getfixturevalue('memory_app')


@pytest.mark.parametrize('consumer', ['voice', 'context', 'search'])
async def test_available_output_matches_frozen_memory(normal_app, consumer):
    app, _ = normal_app
    response = await request(app, consumer)
    assert response.status_code == 200 and PRIVATE in response.text
    # Compare the complete memory output bytes. Context's budget/elapsed telemetry
    # is per-turn; its actual sections are the bytes delivered as model context.
    data = response.json()
    value = data['sections'] if consumer == 'context' else data
    raw = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    capture = os.environ.get('FIX11_NORMAL_OUTPUTS')
    if capture:
        path = Path(capture) / (consumer + '.json')
        if path.exists():
            assert raw == path.read_bytes()
        else:
            path.write_bytes(raw)
    # Voice and search use frozen outputs independent of capture.
    # Context includes the configured assistant and the zone's own abbreviation
    # in its clock line (normal_app pins both).
    expected = NORMAL_SHA256[consumer]
    assert hashlib.sha256(raw).hexdigest() == expected


NORMAL_SHA256 = {
    'voice': '773111548c2af01ee6d07a66652f262482ecedace997817ef50487697d12308d',
    'context': '23e1db7e230db7180177a05c6d8d01d171fbbb55db38307989b7faf4845265fd',
    'search': '5088a03be615967786a34dd7d0ba2ab49721dc4f0fa234dd1cebfc890781e463',
}


async def test_available_affect_keeps_exact_line(affect_app, monkeypatch):
    from colonymind.api.routers import host, social_state
    state = affect_app
    monkeypatch.setattr(social_state, 'appraisal_store', lambda: state.appraisals)
    monkeypatch.setattr(host, 'time', SimpleNamespace(time=lambda: state.now, perf_counter=lambda: 0.))
    add_appraisal(state)
    assert await call(state) == 'Nova: interested'


@pytest.mark.parametrize('stage', ['packet', 'recovered', 'failed_open', 'retry', 'cursor'])
async def test_voice_late_failure_is_sticky(memory_app, monkeypatch, stage):
    from colonymind.memory import voice_recall
    app, ledger = memory_app
    original = voice_recall._read_voice_packet
    calls = []
    def packet(reader, **kwargs):
        result = original(reader, **kwargs)
        calls.append(True)
        if stage == 'retry' and len(calls) == 1:
            forget_report(ledger)  # Move the real withdrawal cursor for a retry.
        elif stage == 'failed_open':
            curated_health.failed_open(reader)
        elif stage != 'cursor':
            curated.disable(reader, sqlite3.OperationalError(PRIVATE))
            if stage == 'recovered':
                curated.enable(reader)
        return result
    monkeypatch.setattr(voice_recall, '_read_voice_packet', packet)
    if stage == 'cursor':
        version = curated._withdrawal_version
        cursors = []
        def lost(reader):
            cursors.append(True)
            if len(cursors) == 2:
                curated.disable(reader, sqlite3.OperationalError(PRIVATE))
                return None
            return version(reader)
        monkeypatch.setattr(curated, '_withdrawal_version', lost)
    response = await request(app, 'voice')
    assert calls and len(calls) == (2 if stage == 'retry' else 1)
    assert response.status_code == 503
    assert response.json()['detail']['code'] == 'memory_backend_unavailable'
    assert PRIVATE not in response.text


@pytest.mark.parametrize('consumer', ['context', 'search'])
@pytest.mark.parametrize('failure', ['disabled', 'recovered', 'failed_open'])
async def test_text_failure_after_complete_preparation_is_withheld(memory_app, monkeypatch, caplog,
                                                                  consumer, failure):
    from colonymind.api.routers import host
    from colonymind.memory import search
    app, ledger = memory_app
    target, name = (host, '_read_context_assemble') if consumer == 'context' else (search, '_select_memory')
    original = getattr(target, name)
    calls = []
    async def after(*args, **kwargs):
        result = await original(*args, **kwargs)
        material = '\n'.join(s.body for s in result.sections) if consumer == 'context' else result.content
        assert PRIVATE in material
        calls.append(True)
        if failure == 'failed_open':
            curated_health.failed_open(ledger)
        else:
            curated.disable(ledger, sqlite3.OperationalError(PRIVATE))
            if failure == 'recovered':
                curated.enable(ledger)
        return result
    monkeypatch.setattr(target, name, after)
    with caplog.at_level(logging.WARNING):
        response = await request(app, consumer)
    assert calls == [True]
    assert response.status_code == 200
    assert PRIVATE not in response.text
    assert_warning_is_content_free(caplog)


@pytest.mark.parametrize('consumer', ['context', 'search'])
async def test_recovery_after_lane_does_not_certify_old_collection(memory_app, monkeypatch, caplog, consumer):
    app, _ = memory_app
    original = TurnIdempotencyLedger.search_sources
    calls = []
    def recovering(reader, *args, **kwargs):
        result = original(reader, *args, **kwargs)
        if not calls:
            calls.append(True)
            curated.disable(reader, sqlite3.OperationalError(PRIVATE))
            curated.enable(reader)
        return result
    monkeypatch.setattr(TurnIdempotencyLedger, 'search_sources', recovering)
    with caplog.at_level(logging.WARNING):
        response = await request(app, consumer)
    assert calls == [True] and response.status_code == 200
    assert PRIVATE not in response.text
    assert_warning_is_content_free(caplog)


@pytest.mark.parametrize('reader', ['candidates', 'current_candidates'])
@pytest.mark.parametrize('failure', ['disabled', 'snapshot'])
def test_shared_candidate_readers_never_fall_back(memory_app, monkeypatch, caplog, reader, failure):
    _, ledger = memory_app
    candidate = {'turn_id': 'report', 'content': PRIVATE}
    forget_report(ledger)
    if failure == 'disabled':
        curated.disable(ledger, sqlite3.OperationalError(PRIVATE))
        monkeypatch.setattr(curated, '_reprobe', lambda reader: None)
    else:
        def broken(*args, **kwargs):
            raise sqlite3.OperationalError(PRIVATE)
        monkeypatch.setattr(curated, '_snapshot', broken)
    with caplog.at_level(logging.WARNING):
        if reader == 'candidates':
            assert curated.strict_candidates(ledger, 'person', 'hydrofoil', [candidate], [candidate]) == ([], [])
        else:
            assert curated.strict_current_candidates(ledger, [candidate], contact_id='person') == []
    assert_warning_is_content_free(caplog)
