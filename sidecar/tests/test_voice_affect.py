"""Offline voice affect contract, privacy, freshness and fixed vocabulary."""
from contextlib import closing
from datetime import datetime, timezone
import itertools
import json
import time
from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
import pytest

from colonymind.api.credentials import CredentialTable, route_operation, secret_digest
from colonymind.api.legacy_scopes import required_scope
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import host, social_state
from colonymind.api.schemas.host import VoiceAffectResponse
from colonymind.contacts import affect as affect_module
from colonymind.contacts.affect import AffectStore
from colonymind.memory.recall import estimate_voice_tokens
from colonymind.memory.voice_affect import (
    APPRAISAL_WORDS, STALE_SECONDS, fresh_timestamp, render_line, speaker_words,
)
from colonymind.self_model.appraisals import AppraisalStore
from colonymind.turns import TurnIdempotencyLedger
from colonymind.turns.idempotency import source_message_hash

ROUTE = '/v1/host/memory/voice-affect'
KEY = 'voice-affect-synthetic-instance-key'
GUEST_KEY = 'voice-affect-synthetic-guest-key'


def stamp(now):
    return datetime.fromtimestamp(now, timezone.utc).isoformat()


def add_appraisal(state, dimension='interest', *, subject='owner', kind='appraisal',
                  status='current', age=0, observed_age=None, expires_in=21600, head=True,
                  name=None):
    """Load a controlled interpretation with a real canonical source receipt."""
    name = name or f'fixture-{dimension}-{subject}-{kind}-{age}'
    at = state.now - (age if observed_age is None else observed_age)
    messages = [{'role': 'user', 'content': 'Synthetic incident for the local store.'}]
    state.ledger.record_source(name, contact_id=subject, session_id=name,
        messages=messages, occurred_at=stamp(at), derive_claims=False)
    refs = state.ledger.source_references([name], contact_id=subject, session_id=name)
    assert len(refs) == 1
    dependency = {**refs[0], 'source_contact_id': subject,
                  'message_hash': source_message_hash(name, messages[0])}
    payload = {'dimension': dimension, 'topic': 'PRIVATE internal topic',
               'text': 'PRIVATE own appraisal reasoning', 'reason': 'PRIVATE evidence reason',
               'intensity': 'moderate', 'hint': 'warmth', 'support': [], 'contrary': [],
               'observed_at': at, 'raw_score': 0.987654}
    with closing(state.ledger._connect()) as conn, conn:
        conn.execute('INSERT INTO appraisal_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (name, 'owner', subject, name, kind, json.dumps(payload), json.dumps([dependency]),
             '{}', name, refs[0]['source_version'], state.now-age,
             None if expires_in is None else state.now+expires_in, status, None))
        if head:
            conn.execute('INSERT INTO appraisal_heads VALUES (?,?,?,?)', ('owner', subject, name, name))
    return name


@pytest.fixture
def affect_app(tmp_path, monkeypatch):
    from colonymind import turns
    # Keep exact seconds for SQLite's microsecond stamps, independent of runner
    # time. Consumers measuring real latency keep the real performance clock.
    state = SimpleNamespace(now=1_800_000_000.0, elapsed=0.0, path=tmp_path)
    state.ledger = TurnIdempotencyLedger(tmp_path/'sources.db')
    state.affect = AffectStore(str(tmp_path/'affect.db'), source_ledger=state.ledger)
    state.appraisals = AppraisalStore(state.ledger, owner_id='owner', clock=lambda: state.now)
    monkeypatch.setenv('COLONYMIND_OWNER_CONTACT_ID', 'owner')
    monkeypatch.setenv('COLONYMIND_OWNER_PERSON_ID', 'owner')
    monkeypatch.setenv('COLONYMIND_OWNER_NAME', 'Sam')
    monkeypatch.setenv('COLONYMIND_PERSONA_NAME', 'Nova')
    monkeypatch.delenv('COLONYMIND_CONTEXT_CHARS_PER_TOKEN', raising=False)
    monkeypatch.setattr(host, '_affect_store', state.affect)
    monkeypatch.setattr(host, '_contacts_store', None)
    monkeypatch.setattr(host, 'time', SimpleNamespace(time=lambda: state.now, perf_counter=time.perf_counter))
    monkeypatch.setattr(affect_module, 'time', SimpleNamespace(time=lambda: state.now))
    monkeypatch.setattr(turns, 'get_turn_idempotency_ledger', lambda *a, **k: state.ledger)
    state.table_config = {
        'api-key': {'owner': ['voice'], 'contacts': 'all'},
        'guest': {'sha256': secret_digest(GUEST_KEY), 'owner': [],
                  'contacts': ['guest-person', 'other'], 'viewer': 'guest-person',
                  'senders': 'listed', 'access': ['read']},
        'unbound': {'sha256': secret_digest('unbound-secret'), 'owner': [],
                    'contacts': 'all', 'access': ['read']},
        'own': {'sha256': secret_digest('own-secret'), 'owner': [],
                'contacts': 'own', 'access': ['read']},
        'legacy': {'sha256': secret_digest('legacy-secret'), 'owner': ['voice'],
                   'contacts': 'all', 'scopes': ['memory:search']},
        'wrong-platform': {'sha256': secret_digest('wrong-platform-secret'), 'owner': ['sms'],
                           'contacts': 'all', 'access': ['read']},
        'no-read': {'sha256': secret_digest('no-read-secret'), 'owner': ['voice'],
                    'contacts': 'all', 'access': ['write']},
    }
    state.app = FastAPI()
    state.app.include_router(host.router)
    state.app.add_middleware(ApiKeyMiddleware, api_key=KEY,
        credentials=CredentialTable.from_config(state.table_config))
    yield state
    state.affect.close()


@pytest.fixture(autouse=True)
def _controlled_read_deadline(request, monkeypatch):
    # This module checks functional decisions. The shared affect_app fixture is
    # also imported by real latency gates, which must retain their real clock.
    if 'affect_app' in request.fixturenames:
        state = request.getfixturevalue('affect_app')
        monkeypatch.setattr(host.time, 'perf_counter', lambda: state.elapsed)


async def call(state, *, person='owner', key=KEY, extra=None):
    async with AsyncClient(transport=ASGITransport(app=state.app), base_url='http://fixture',
            headers={'Authorization': 'Bearer '+key} if key else {}) as client:
        response = await client.post(ROUTE,
            json={'identity': {'host_id': 'synthetic'}, 'person_id': person, **(extra or {})})
    assert response.status_code == 200, response.text
    packet = response.json()
    assert set(packet) == {'line'}
    line = packet['line']
    assert len(line) <= 120 and estimate_voice_tokens(line) <= 25
    assert not any(c in line for c in '[]\u2013\u2014\r\n')
    assert 'PRIVATE' not in line and '0.987654' not in line
    return line


@pytest.mark.parametrize(('valence', 'mood'), [(-1, 'frustrated'), (-0.25, 'frustrated'),
    (-0.2499, 'neutral'), (0, 'neutral'), (0.2499, 'neutral'), (0.25, 'pleased'), (1, 'pleased')])
@pytest.mark.parametrize(('arousal', 'energy'), [(0, 'low energy'), (0.3499, 'low energy'),
    (0.35, 'steady energy'), (0.65, 'steady energy'), (0.6501, 'high energy'), (1, 'high energy')])
async def test_all_bins_and_boundaries(affect_app, valence, mood, arousal, energy):
    state = affect_app
    state.affect.create_event(contact_id='owner', valence=valence, arousal=arousal,
                              timestamp=stamp(state.now))
    assert await call(state) == f'Sam: {mood}, {energy}'


@pytest.mark.parametrize(('dimension', 'word'), list(APPRAISAL_WORDS.items()))
async def test_own_appraisal_vocabulary_and_exact_packet(affect_app, dimension, word):
    state = affect_app
    state.affect.create_event(contact_id='owner', valence=-0.8, arousal=0.1, timestamp=stamp(state.now))
    add_appraisal(state, dimension)
    assert await call(state) == f'Sam: frustrated, low energy · Nova: {word}'


async def test_empty_store_never_guesses_neutral(affect_app):
    assert await call(affect_app) == ''


@pytest.mark.parametrize('age', [1799, 1800, 1801, 21601, -1])
async def test_observation_freshness_not_decay_read_clock(affect_app, age):
    state = affect_app
    state.affect.create_event(contact_id='owner', valence=-0.8, arousal=0.1,
                              timestamp=stamp(state.now-age))
    expected = 'Sam: frustrated, low energy' if 0 <= age < STALE_SECONDS else ''
    for _ in range(2):
        assert await call(state) == expected
        state.now += 0.01


async def test_decay_event_and_repeated_reads_do_not_freshen_old_observation(affect_app):
    state = affect_app
    state.affect.create_event(contact_id='owner', valence=-0.8, timestamp=stamp(state.now-1800))
    state.affect.create_event(contact_id='owner', valence=-0.8, source=' DECAY ', timestamp=stamp(state.now))
    assert await call(state) == ''


async def test_decay_only_has_no_state_to_say(affect_app):
    state = affect_app
    state.affect.create_event(contact_id='owner', valence=0.9, source='decay', timestamp=stamp(state.now))
    assert await call(state) == ''


async def test_reuses_store_current_state_and_decay_instead_of_latest_event(affect_app):
    state = affect_app
    state.affect.create_event(contact_id='owner', valence=-1, arousal=0.1, timestamp=stamp(state.now-100))
    state.affect.create_event(contact_id='owner', valence=0.1, arousal=0.1, timestamp=stamp(state.now))
    assert await call(state) == 'Sam: frustrated, low energy'
    state.affect._conn.execute('UPDATE affect_state SET current_valence=-0.251, last_updated=?',
                              (stamp(state.now-1700),))
    state.affect._conn.commit()
    assert await call(state) == 'Sam: neutral, low energy'


@pytest.mark.parametrize('changes', [dict(age=1800), dict(observed_age=1800), dict(age=-1),
    dict(observed_age=-1), dict(expires_in=0), dict(expires_in=None),
    dict(status='withdrawn'), dict(status='reconsidering'), dict(status='settled'),
    dict(kind='behavior_hypothesis'), dict(kind='assessment'), dict(kind='preference'),
    dict(head=False), dict(subject='guest-person')])
async def test_absent_stale_expired_noncurrent_and_other_person_appraisals_omitted(affect_app, changes):
    add_appraisal(affect_app, **changes)
    assert await call(affect_app) == ''


async def test_latest_current_appraisal_only_no_fallback_to_older_mood(affect_app):
    state = affect_app
    add_appraisal(state, 'frustration', age=100)
    add_appraisal(state, 'satisfaction')
    assert await call(state) == 'Nova: pleased'
    with closing(state.ledger._connect()) as conn, conn:
        conn.execute("UPDATE appraisal_records SET expires_at=? WHERE id LIKE 'fixture-satisfaction%'", (state.now,))
    assert await call(state) == ''


async def test_each_part_independently_absent_or_stale(affect_app, monkeypatch):
    state = affect_app
    add_appraisal(state, 'annoyance')
    assert await call(state) == 'Nova: annoyed'
    monkeypatch.setattr(host, '_affect_store', None)
    assert await call(state) == 'Nova: annoyed'
    monkeypatch.setattr(host, '_affect_store', state.affect)
    state.affect.create_event(contact_id='owner', valence=0.8, arousal=0.9, timestamp=stamp(state.now))
    state.now += 1800
    state.affect.create_event(contact_id='owner', valence=0.8, arousal=0.9, timestamp=stamp(state.now))
    assert await call(state) == 'Sam: pleased, high energy'


@pytest.mark.parametrize(('key', 'person', 'allowed'), [
    (KEY, 'owner', True), (KEY, 'guest-person', False),
    (GUEST_KEY, 'guest-person', True), (GUEST_KEY, 'owner', False),
    (GUEST_KEY, 'other', False), ('unbound-secret', 'guest-person', False),
    ('unbound-secret', 'owner', False), ('wrong-platform-secret', 'owner', False),
    ('legacy-secret', 'owner', True), ('own-secret', 'credential:own', True),
    ('own-secret', 'owner', False), ('own-secret', 'guest-person', False),
])
async def test_privacy_owner_voice_or_verified_own_state_only(affect_app, key, person, allowed):
    state = affect_app
    for identity in ('owner', 'guest-person', 'other', 'credential:own'):
        state.affect.create_event(contact_id=identity, valence=-0.8, arousal=0.1,
                                  timestamp=stamp(state.now))
    add_appraisal(state, 'satisfaction')
    line = await call(state, person=person, key=key)
    if not allowed:
        assert line == ''
    elif person == 'owner':
        assert line == 'Sam: frustrated, low energy · Nova: pleased'
    else:
        assert line == 'Speaker: frustrated, low energy'


async def test_verified_contact_label_from_store_only(affect_app, monkeypatch):
    state = affect_app
    state.affect.create_event(contact_id='guest-person', valence=0.8, arousal=0.9,
                              timestamp=stamp(state.now))
    async def get(person):
        assert person == 'guest-person'
        return SimpleNamespace(given_name='Sam', display_name='Sam Example')
    monkeypatch.setattr(host, '_contacts_store', SimpleNamespace(get=get))
    assert await call(state, person='guest-person', key=GUEST_KEY) == 'Sam: pleased, high energy'


@pytest.mark.parametrize('variable', ['COLONYMIND_OWNER_CONTACT_ID', 'COLONYMIND_OWNER_PERSON_ID'])
async def test_missing_or_conflicting_instance_identity_fails_closed(affect_app, monkeypatch, variable):
    state = affect_app
    state.affect.create_event(contact_id='owner', valence=-0.8, timestamp=stamp(state.now))
    add_appraisal(state)
    monkeypatch.setenv(variable, 'different-instance-owner')
    assert await call(state) == ''


@pytest.mark.parametrize(('key', 'status'), [(None, 401), ('bad-key', 401), ('no-read-secret', 403)])
async def test_invalid_or_nonread_credentials_refused(affect_app, key, status):
    async with AsyncClient(transport=ASGITransport(app=affect_app.app), base_url='http://fixture',
            headers={'Authorization': 'Bearer '+key} if key else {}) as client:
        assert (await client.post(ROUTE,
            json={'identity': {'host_id': 'fixture'}, 'person_id': 'owner'})).status_code == status


async def test_keyless_loopback_never_gets_owner(affect_app):
    app = FastAPI()
    app.include_router(host.router)
    app.add_middleware(ApiKeyMiddleware, api_key=None,
                       credentials=CredentialTable.from_config(affect_app.table_config))
    async with AsyncClient(transport=ASGITransport(app=app, client=('127.0.0.1', 123)),
                           base_url='http://localhost') as client:
        response = await client.post(ROUTE, json={'identity': {'host_id': 'fixture'}, 'person_id': 'owner'})
        assert response.status_code == 200 and response.json() == {'line': ''}


@pytest.mark.parametrize('extra', [{'person_id': ' '}, {'person_id': ''}, {'person_id': 'x'*257},
    {'speaker_name': '[laughs]'}, {'subject_id': 'other'}, {'platform': 'sms'}, {'line': 'invented'}])
async def test_request_cannot_supply_affect_names_or_bypass_scope(affect_app, extra):
    async with AsyncClient(transport=ASGITransport(app=affect_app.app), base_url='http://fixture',
                           headers={'X-API-Key': KEY}) as client:
        response = await client.post(ROUTE,
            json={'identity': {'host_id': 'fixture'}, 'person_id': 'owner', **extra})
        assert response.status_code == 422


async def test_changed_source_cannot_leave_a_current_appraisal(affect_app):
    state = affect_app
    identifier = add_appraisal(state)
    assert await call(state) == 'Nova: interested'
    with closing(state.ledger._connect()) as conn, conn:
        conn.execute('UPDATE turn_sources SET messages_json=? WHERE turn_id=?',
                     (json.dumps([{'role': 'user', 'content': 'Changed source'}]), identifier))
    assert await call(state) == ''


@pytest.mark.parametrize('dimension', ['amusement', 'raw_score', 'calm', '[laughs]'])
async def test_no_unstored_emotion_or_raw_dimension_is_rendered(affect_app, dimension):
    add_appraisal(affect_app, dimension)
    assert await call(affect_app) == ''


async def test_legacy_appraisal_without_observed_clock_checks_source_clock(affect_app):
    state = affect_app
    identifier = add_appraisal(state)
    with closing(state.ledger._connect()) as conn, conn:
        payload = json.loads(conn.execute('SELECT payload_json FROM appraisal_records WHERE id=?',
                                        (identifier,)).fetchone()[0])
        del payload['observed_at']
        conn.execute('UPDATE appraisal_records SET payload_json=? WHERE id=?', (json.dumps(payload), identifier))
    assert await call(state) == 'Nova: interested'
    with closing(state.ledger._connect()) as conn, conn:
        conn.execute('UPDATE turn_sources SET occurred_at=? WHERE turn_id=?', (stamp(state.now-1800), identifier))
    assert await call(state) == ''


async def test_guest_reads_never_open_owner_appraisal_store(affect_app, monkeypatch):
    state = affect_app
    state.affect.create_event(contact_id='guest-person', valence=-0.8, arousal=0.1,
                              timestamp=stamp(state.now))
    calls = []
    def forbidden():
        calls.append('opened')
        raise AssertionError('guest opened owner appraisal store')
    monkeypatch.setattr(social_state, 'appraisal_store', forbidden)
    assert await call(state, person='guest-person', key=GUEST_KEY) == 'Speaker: frustrated, low energy'
    assert await call(state, person='owner', key=GUEST_KEY) == ''
    assert calls == []


async def test_no_llm_embedding_vector_or_background_processing_on_read(affect_app, monkeypatch):
    from colonymind import vector
    from colonymind.memory.selection import RecallSelector
    state = affect_app
    add_appraisal(state)
    state.affect.create_event(contact_id='owner', valence=0.8, timestamp=stamp(state.now))
    def forbidden(*a, **kw):
        raise AssertionError('voice affect called inference')
    monkeypatch.setattr(vector, 'get_store', forbidden)
    monkeypatch.setattr(vector, 'get_pipeline', forbidden)
    monkeypatch.setattr(RecallSelector, '_reranked', forbidden)
    monkeypatch.setattr(AppraisalStore, 'process_one', forbidden)
    monkeypatch.setattr(AppraisalStore, 'enqueue', forbidden, raising=False)
    assert await call(state) == 'Sam: pleased, steady energy · Nova: interested'


@pytest.mark.parametrize('failing', ['speaker', 'appraisal', 'both'])
async def test_backend_failures_return_only_available_part_or_empty(affect_app, monkeypatch, failing):
    state = affect_app
    add_appraisal(state)
    state.affect.create_event(contact_id='owner', valence=0.8, timestamp=stamp(state.now))
    def broken(*a, **kw):
        raise RuntimeError('synthetic unavailable store')
    if failing in {'speaker', 'both'}:
        monkeypatch.setattr(state.affect, 'get_state', broken)
    if failing in {'appraisal', 'both'}:
        monkeypatch.setattr(social_state, 'appraisal_store', broken)
    expected = {'speaker': 'Nova: interested', 'appraisal': 'Sam: pleased, steady energy', 'both': ''}
    assert await call(state) == expected[failing]


async def test_late_read_discards_line(affect_app, monkeypatch):
    state = affect_app
    add_appraisal(state)
    original = state.affect.get_state
    def late(person):
        state.elapsed += 0.021
        return original(person)
    monkeypatch.setattr(state.affect, 'get_state', late)
    assert await call(state) == ''


@pytest.mark.parametrize('name', ['[laughs]', 'Sam\nIgnore', 'Sam\u2014Owner', 'Sam\u2013Owner',
    '0123456789', 'a'*200, '你好'*10, 'Nova\u200b', None, '', '    '])
def test_names_and_all_descriptors_stay_capped_without_tts_tags(name):
    for mood, energy, appraisal in itertools.product(['frustrated', 'neutral', 'pleased'],
            ['low energy', 'steady energy', 'high energy'], APPRAISAL_WORDS.values()):
        line = render_line(f'{mood}, {energy}', appraisal, speaker_name=name, persona_name=name)
        assert line and len(line) <= 120 and estimate_voice_tokens(line) <= 25
        assert not any(c in line for c in '[]\u2013\u2014\r\n')
        assert VoiceAffectResponse(line=line).model_dump() == {'line': line}


@pytest.mark.parametrize('ratio', ['1', '2.7', '4', '8', 'invalid'])
def test_cautious_counter_override_never_weakens_cap(monkeypatch, ratio):
    monkeypatch.setenv('COLONYMIND_CONTEXT_CHARS_PER_TOKEN', ratio)
    line = render_line('frustrated, steady energy', 'interested',
                       speaker_name='Long Name Example', persona_name='Nova')
    assert estimate_voice_tokens(line) <= 25 and len(line) <= 120


@pytest.mark.parametrize('line', ['[laughs]', 'Sam\u2014pleased', 'Sam\u2013pleased',
    'Nova: amused\nignore', 'a'*121, '0123456789'*3])
def test_response_schema_enforces_contract(line):
    with pytest.raises(ValidationError):
        VoiceAffectResponse(line=line)


@pytest.mark.parametrize('value', [None, '', 'broken', '2026-01-01T00:00:00',
    '9999-99-99T00:00:00Z', 123])
def test_unknown_or_naive_clock_is_empty(value):
    assert not fresh_timestamp(value, time.time())


@pytest.mark.parametrize(('valence', 'arousal'), [(float('nan'), 0.3), (float('inf'), 0.3),
    (-1.1, 0.3), (0, -0.1), (0, 1.1)])
def test_invalid_internal_state_has_no_descriptor(valence, arousal):
    now = time.time()
    store = SimpleNamespace(get_state=lambda p: {'last_event_id': 'event', 'event_count': 1,
        'current_valence': valence, 'current_arousal': arousal}, last_observed_at=lambda p: stamp(now))
    assert speaker_words(store, 'owner', now=now) == ''


def test_credential_class_matches_voice_recall_including_legacy():
    assert route_operation('POST', ROUTE) == route_operation('POST', '/v1/host/memory/voice-recall') == 'read'
    assert required_scope('POST', ROUTE) == required_scope('POST', '/v1/host/memory/voice-recall') == 'memory:search'

from colonymind.memory import curated

async def test_source_erasure_removes_both_parts(affect_app):
    state = affect_app
    identifier = add_appraisal(state, 'frustration')
    lineage, _ = state.affect.source_input(identifier, 'owner')
    state.affect.create_event(contact_id='owner', valence=-0.8, arousal=0.1,
        timestamp=stamp(state.now), source_lineage=lineage)
    assert await call(state) == 'Sam: frustrated, low energy · Nova: frustrated'
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(state.ledger)
    _source_forget_receipt = curated.forget_sources(state.ledger, 'owner', source_ids=[identifier])
    assert historical_bytes(state.ledger) == _native_before_forget
    assert all(curated.is_forgotten(state.ledger, sid) for sid in _source_forget_receipt['source_ids'])
    assert await call(state) == ''
