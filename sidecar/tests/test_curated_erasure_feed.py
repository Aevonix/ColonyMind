"""Curated source withdrawals publish the voice gateway's existing feed contract."""
from contextlib import closing
import hashlib
import json
import re
import sqlite3

import pytest

from colonymind.memory import curated
from colonymind.turns import TurnIdempotencyLedger
from colonymind.turns.idempotency import canonical_turn_digest, source_message_hash
from test_curated_memory import PERSON, ROOT, client, historical_bytes, ledger_of, state

FEED = '/v1/host/memory/sources/erasures'
SESSION = ' session-é '
MESSAGES = [
    {'role': 'user', 'content': 'café 雪 e\u0301'},
    {'role': 'assistant', 'content': [{'type': 'text', 'text': '雪'}]},
    {'role': 'user', 'content': ''},
]


def voice_hash(session_id, message):
    # Copied contract from voice/voice_sources.py; no dependency on the consumer's own module.
    return hashlib.sha256(json.dumps({'session_id': session_id, 'role': message.get('role'),
        'content': message.get('content')}, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def voice_rules(events):
    # The consumer's required rule types and lowercase SHA-256 validation.
    for rule in events:
        assert isinstance(rule.get('session_id'), str)
        assert isinstance(rule.get('turn_id'), str)
        assert isinstance(rule.get('message_hashes'), list)
        assert all(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value)
                   for value in rule['message_hashes'])
        assert rule['whole_source'] is True
        assert isinstance(rule['source_turn_id'], str)
        assert re.fullmatch(r'[0-9a-f]{64}', rule['source_version'])
    return events


def stored(ledger, source):
    with closing(ledger._connect()) as conn:
        return json.loads(conn.execute('SELECT messages_json FROM turn_sources WHERE turn_id=?',
                                       (source,)).fetchone()[0])


@pytest.mark.parametrize('message', MESSAGES, ids=['unicode', 'list-content', 'empty-content'])
def test_sidecar_hash_matches_voice_without_preserved_input_identity(state, message):
    ledger = ledger_of(state)
    ledger.record_source('voice-source', contact_id=PERSON, session_id=SESSION,
                         messages=[message], derive_claims=False)
    retained = stored(ledger, 'voice-source')[0]
    assert retained == message
    assert source_message_hash(SESSION, retained) == voice_hash(SESSION, retained)


@pytest.mark.asyncio
@pytest.mark.parametrize('selection', ['ids', 'text'])
async def test_withdrawal_publishes_contact_scoped_voice_feed_once(state, client, selection):
    ledger = ledger_of(state)
    for source, person in [('voice-source', PERSON), ('other-source', 'contact-b')]:
        ledger.record_source(source, contact_id=person, session_id=SESSION,
                             messages=MESSAGES, derive_claims=False)
    before = historical_bytes(ledger)
    initial = (await client.get(FEED, params={'contact_id': PERSON})).json()
    assert initial['head'] == initial['through'] == 0 and initial['events'] == []
    other = await client.post(ROOT + '/sources/forget', json={
        'person_id': 'contact-b', 'source_ids': ['other-source']})
    assert other.status_code == 200
    assert (await client.get(FEED, params={'contact_id': PERSON})).json() == initial
    other_feed = (await client.get(FEED, params={'contact_id': 'contact-b'})).json()
    fields = ({'source_ids': ['voice-source']} if selection == 'ids' else
              {'old_text': MESSAGES[0]['content'], 'session_id': SESSION})
    body = {'person_id': PERSON, **fields}
    first = await client.post(ROOT + '/sources/forget', json=body)
    assert first.status_code == 200
    response = await client.get(FEED, params={'contact_id': PERSON, 'after': initial['head']})
    assert response.status_code == 200
    page = response.json()
    assert page['contact_id'] == PERSON and page['complete'] is True
    assert page['head'] == page['through'] > initial['head']
    rules = voice_rules(page['events'])
    assert len(rules) == 1
    rule = rules[0]
    assert rule['sequence'] == page['head']
    assert rule['session_id'] == SESSION and rule['source_turn_id'] == 'voice-source'
    assert rule['message_hashes'] == sorted({voice_hash(SESSION, message)
                                            for message in stored(ledger, 'voice-source')})
    assert rule['source_version'] == canonical_turn_digest(stored(ledger, 'voice-source'))
    # Whole-source consumers can suppress a turn without holding its messages.
    assert {r['source_turn_id'] for r in rules if r.get('whole_source', True)} == {'voice-source'}
    fresh = await client.post(FEED, json={'contact_id': PERSON, 'session_id': 'later',
        'source_refs': [{'source_id': 'voice-source', 'source_version': rule['source_version']}]})
    assert fresh.status_code == 200
    assert all(fresh.json()[key] == value for key, value in page.items())
    assert (await client.post(ROOT + '/sources/forget', json=body)).json() == first.json()
    assert (await client.post(ROOT + '/sources/forget', json={
        'person_id': PERSON, 'source_ids': ['voice-source', 'voice-source']})).json() == first.json()
    assert curated.forget_sources(TurnIdempotencyLedger(ledger.db_path), PERSON,
                                  source_ids=['voice-source']) == first.json()
    assert (await client.get(FEED, params={'contact_id': PERSON})).json() == page
    assert (await client.get(FEED, params={'contact_id': 'contact-b'})).json() == other_feed
    tail = (await client.get(FEED, params={'contact_id': PERSON, 'after': page['through']})).json()
    assert tail['head'] == page['head'] and tail['events'] == [] and tail['complete']
    assert historical_bytes(ledger) == before


@pytest.mark.parametrize('prior_rule', [False, True])
def test_feed_hashes_stored_content_and_preserved_input_identity(state, prior_rule):
    ledger = ledger_of(state)
    ledger.record_source('normalized', contact_id=PERSON, session_id=SESSION, messages=[{
        'role': 'user', 'content': [{'type': 'input_image', 'image_url': 'fixture:unretained'}]}],
        derive_claims=False)
    message = stored(ledger, 'normalized')[0]
    assert source_message_hash(SESSION, message) == message['_source_message_hash']
    assert source_message_hash(SESSION, message) != voice_hash(SESSION, message)
    if prior_rule:
        with closing(ledger._connect()) as conn, conn:
            ledger._append_erasure(conn, turn_id='normalized', contact_id=PERSON,
                session_id=SESSION, scope='person', messages=[message], removed=[message], whole=False)
    prior_head = ledger.erasure_feed(PERSON)['head']
    before = historical_bytes(ledger)
    curated.forget_sources(ledger, PERSON, source_ids=['normalized'])
    rule, = voice_rules(ledger.erasure_feed(PERSON, prior_head)['events'])
    assert rule['message_hashes'] == sorted({voice_hash(SESSION, message), message['_source_message_hash']})
    assert historical_bytes(ledger) == before


def test_feed_publication_retains_later_native_copies_and_dependencies(state):
    ledger = ledger_of(state)
    ledger.record_source('voice-source', contact_id=PERSON, session_id=SESSION,
                         messages=MESSAGES, derive_claims=False)
    ref, = ledger.source_references(['voice-source'], contact_id=PERSON, session_id=SESSION)
    curated.forget_sources(ledger, PERSON, source_ids=['voice-source'])
    page = ledger.erasure_feed(PERSON)
    for source, messages in [('copy', MESSAGES), ('dependent', [{
            'role': 'assistant', 'content': '', '_supplied_sources': [ref]}])]:
        assert ledger.record_source(source, contact_id=PERSON, session_id=SESSION,
                                    messages=messages, derive_claims=False)
        assert stored(ledger, source) == messages
        assert curated.is_forgotten(ledger, source)
    assert ledger.erasure_feed(PERSON) == page


@pytest.mark.parametrize('fail_after', [1, 2])
def test_tombstone_and_feed_roll_back_together(state, monkeypatch, fail_after):
    ledger = ledger_of(state)
    for source in ('a', 'b'):
        ledger.record_source(source, contact_id=PERSON, session_id=SESSION,
                             messages=MESSAGES, derive_claims=False)
    before = historical_bytes(ledger)
    append = ledger._append_erasure
    calls = []

    def interrupted(conn, **kwargs):
        result = append(conn, **kwargs)
        calls.append(result)
        # A separate reader sees neither half of the pending transaction.
        with closing(ledger._connect()) as observer:
            for table in ('curated_memory_events', 'source_erasures', 'source_erasure_revisions'):
                assert observer.execute('SELECT count(*) FROM ' + table).fetchone()[0] == 0
        if len(calls) == fail_after:
            raise sqlite3.OperationalError('fixture interruption')
        return result

    with monkeypatch.context() as patch:
        patch.setattr(ledger, '_append_erasure', interrupted)
        with pytest.raises(sqlite3.OperationalError, match='fixture interruption'):
            curated.forget_sources(ledger, PERSON, source_ids=['a', 'b'])
    assert len(calls) == fail_after
    with closing(ledger._connect()) as conn:
        for table in ('curated_memory_events', 'curated_memory_fence_hashes',
                      'source_erasures', 'source_erasure_revisions'):
            assert conn.execute('SELECT count(*) FROM ' + table).fetchone()[0] == 0
    assert historical_bytes(ledger) == before
    curated.forget_sources(ledger, PERSON, source_ids=['a'])
    first = ledger.erasure_feed(PERSON)
    curated.forget_sources(ledger, PERSON, source_ids=['a', 'b'])
    page = ledger.erasure_feed(PERSON)
    assert len(voice_rules(page['events'])) == 2 and page['head'] > first['head']
    assert len(ledger.erasure_feed(PERSON, first['head'])['events']) == 1
    assert historical_bytes(ledger) == before
