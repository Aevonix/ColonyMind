"""Safety assertions for five erasure probes, using synthetic stores."""
import asyncio
from contextlib import closing
from datetime import datetime, timezone
import json
import sqlite3
import threading
import time

import pytest

from colonymind.commitments.store import CommitmentStore
from colonymind.contacts.affect import AffectStore
from colonymind.contacts.comms import CommsLog
from colonymind.memory import curated
from colonymind.memory.voice_affect import speaker_words
from colonymind.memory.voice_recall import read_voice_packet
from colonymind.turns import TurnIdempotencyLedger
from test_curated_memory import PERSON, ROOT, client, historical_bytes, ledger_of, state
from test_curated_read_races import withdraw
from voice_consumer_snapshot import SourceOutbox, _filter, redact_source_payload, source_message_hash


@pytest.fixture
def linked(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    log = CommsLog(str(tmp_path / 'comms.db'), source_ledger=ledger)
    affect = AffectStore(str(tmp_path / 'affect.db'), source_ledger=ledger)
    now = datetime.now(timezone.utc).timestamp()
    for source in ('first', 'second', 'seed'):
        ledger.record_source(source, contact_id='fixture-owner', session_id='fixture-session',
            messages=[{'role': 'user', 'content': source}], derive_claims=False)
        if source != 'seed':
            lineage, _ = log.source_input(source, 'fixture-owner')
            affect.create_event(contact_id='fixture-owner', valence=.8, arousal=.9,
                source_lineage=lineage, timestamp=datetime.fromtimestamp(now, timezone.utc).isoformat())
    curated.forget_sources(ledger, 'fixture-owner', source_ids=['seed'])
    return ledger, affect, now


@pytest.mark.parametrize('method', ['get_state', 'project_state', 'speaker_words'])
def test_affect_complete_projection(linked, monkeypatch, method):
    ledger, affect, now = linked
    original = curated.is_forgotten
    checks = depth = 0
    target = None
    fired = []

    def check(reader, ref, **kwargs):
        nonlocal checks, depth, target
        if depth == 0:
            checks += 1
            if checks == 1:
                target = ref['turn_id']
            if checks == 2:
                withdraw(ledger, target)
                fired.append(True)
        depth += 1
        try:
            return original(reader, ref, **kwargs)
        finally:
            depth -= 1

    monkeypatch.setattr(curated, 'is_forgotten', check)
    read = lambda: (speaker_words(affect, 'fixture-owner', now=now) if method == 'speaker_words'
        else affect.project_state('fixture-owner', now=datetime.fromtimestamp(now, timezone.utc))
        if method == 'project_state' else affect.get_state('fixture-owner'))
    offered = read()
    assert fired
    assert offered == read()
    assert offered['event_count'] == 0 if method == 'get_state' else not offered


@pytest.mark.parametrize('stage', ['decay', 'description'])
def test_affect_fence_includes_final_preparation(linked, monkeypatch, stage):
    from colonymind.contacts import affect as module
    ledger, affect, now = linked
    target, attribute = (module, '_project_decay') if stage == 'decay' else (affect, 'last_observed_at')
    original = getattr(target, attribute)
    fired = []

    def prepare(*args, **kwargs):
        result = original(*args, **kwargs)
        if not fired:
            withdraw(ledger, 'first')
            fired.append(True)
        return result

    monkeypatch.setattr(target, attribute, prepare)
    offered = (affect.project_state('fixture-owner') if stage == 'decay'
               else speaker_words(affect, 'fixture-owner', now=now))
    assert fired and not offered


@pytest.mark.parametrize('method', ['list', 'get_pending_for_person', 'get_overdue',
    'get_open_for_counterpart', 'resolution_stats', 'recent_rejections'])
def test_commitment_complete_lists(linked, tmp_path, monkeypatch, method):
    ledger, _, _ = linked
    store = CommitmentStore(tmp_path / 'commitments.db')
    for source in ('first', 'second'):
        item = store.create(person_id='fixture-owner', description='Fixture task ' + source,
            metadata={'source_turn': source,
                'counterpart': 'Fixture', 'resolution': {'outcome': 'invalid'}})
        if method == 'recent_rejections':
            store.update(item['id'], status='cancelled')
    with closing(store._connect()) as conn, conn:
        conn.execute("UPDATE commitments SET due_at='2000-01-01T00:00:00+00:00'")
    visible = store._visible
    checks = 0
    target = None

    def check(row):
        nonlocal checks, target
        checks += 1
        if checks == 1:
            target = json.loads(row['metadata'])['source_turn']
        if checks == 2:
            withdraw(ledger, target)
        return visible(row)

    monkeypatch.setattr(store, '_visible', check)
    read = {'list': lambda: store.list(person_id='fixture-owner'),
        'get_pending_for_person': lambda: store.get_pending_for_person('fixture-owner'),
        'get_overdue': store.get_overdue,
        'get_open_for_counterpart': lambda: store.get_open_for_counterpart(['Fixture']),
        'resolution_stats': lambda: store.resolution_stats(days=30),
        'recent_rejections': store.recent_rejections}[method]
    offered = read()
    monkeypatch.setattr(store, '_visible', visible)
    assert checks >= 2
    assert offered == read()
    if method == 'list':
        assert offered['total'] == len(offered['commitments']) == 1
    elif method == 'resolution_stats':
        assert sum(row['created'] for row in offered['by_source'].values()) == 1
    else:
        assert len(offered) == 1


def make_packet(ledger, person):
    return read_voice_packet(ledger, query='Fixture preference', contact_id=person,
        session_id='fixture-session', k=5, max_tokens=600, timezone_name='UTC',
        cancelled=threading.Event())


@pytest.mark.parametrize('operation', ['forget', 'correct'])
def test_item_mutation_invalidates_prepared_packet_and_replays_once(state, operation):
    ledger = ledger_of(state)
    curated.mutate(ledger, PERSON, 'mem_fixture', 'remember', request_id='create',
                   content='Fixture preference', kind='preference')
    packet = make_packet(ledger, PERSON)
    assert packet and len(packet.selected) == 1 and packet.source_refs == []
    other = ledger.erasure_feed('contact-b')
    before = historical_bytes(ledger)
    kwargs = dict(request_id='withdraw', expected_revision=1,
                  content='Replacement' if operation == 'correct' else None)
    receipt = curated.mutate(ledger, PERSON, 'mem_fixture', operation, **kwargs)
    page = ledger.erasure_feed(PERSON, packet.watermark)
    assert page['head'] == page['through'] > packet.watermark
    rule, = page['events']
    assert rule['session_id'] == '' and rule['message_hashes'] == []
    assert rule['whole_source'] is False and isinstance(rule['turn_id'], str)
    assert curated.current_candidates(ledger, packet.selected, contact_id=PERSON) == []
    assert ledger.erasure_feed('contact-b') == other
    restarted = TurnIdempotencyLedger(ledger.db_path)
    assert curated.mutate(restarted, PERSON, 'mem_fixture', operation, **kwargs) == receipt
    assert restarted.erasure_feed(PERSON) == page
    assert restarted.erasure_feed(PERSON, page['through'])['events'] == []
    assert historical_bytes(ledger) == before


@pytest.mark.asyncio
async def test_item_forget_http_rejects_old_packet_watermark(state, client):
    response = await client.post(ROOT, json={'id': 'mem_fixture', 'person_id': PERSON,
        'content': 'Fixture preference', 'kind': 'preference', 'request_id': 'create'})
    assert response.status_code == 200
    ledger = ledger_of(state)
    ledger.record_source('present-input', contact_id=PERSON, session_id='fixture-session',
        messages=[{'role': 'user', 'content': 'Unrelated utterance'}], derive_claims=False)
    packet = await asyncio.to_thread(make_packet, ledger, PERSON)
    assert packet and len(packet.selected) == 1 and packet.source_refs == []
    deleted = await client.post(ROOT + '/mem_fixture/forget', json={
        'person_id': PERSON, 'request_id': 'forget', 'expected_revision': 1})
    assert deleted.status_code == 200
    feed = '/v1/host/memory/sources/erasures'
    get = await client.get(feed, params={'contact_id': PERSON, 'after': packet.watermark})
    post = await client.post(feed, json={'contact_id': PERSON, 'session_id': 'fixture-session',
        'after': packet.watermark, 'source_refs': ledger.source_references(['present-input'],
            contact_id=PERSON, session_id='fixture-session')})
    assert get.status_code == post.status_code == 200
    page, current = get.json(), post.json()
    assert all(current[key] == value for key, value in page.items())
    assert current['sources_current'] is True
    assert page['head'] > packet.watermark  # the phone comparator rejects this generation
    outbox = SourceOutbox(state / 'fixture-outbox.db')
    outbox.prepare()
    payload = {'turn_id': page['events'][0]['turn_id'], 'contact_id': PERSON,
        'session_id': '', 'checkpoint_messages': [{'role': 'assistant', 'content': 'Unrelated'}],
        'assistant_source_refs': [{'source_id': page['events'][0]['turn_id'], 'source_version': '0' * 64}],
        'assistant_input_refs': [{'source_id': page['events'][0]['turn_id'], 'input_message_hash': '0' * 64}]}
    outbox.enqueue(payload['turn_id'], payload)
    outbox.apply_erasure_page(PERSON, page)  # the real consumer validation accepts the marker
    assert outbox.lookup(payload['turn_id'])['payload'] == payload
    assert outbox.erasure_watermark(PERSON) == page['head']
    # The copied request filter removes the stale packet but leaves ordinary history.
    stamp = '[colonymind-recall-v1 ' + json.dumps({'contact_id': PERSON,
        'watermark': packet.watermark}) + ']\n'
    messages = [{'role': 'user', 'content': 'Unrelated history'},
        {'role': 'assistant', 'content': 'Unrelated reply'},
        {'role': 'user', 'content': 'Current\n' + stamp + packet.content + '[/colonymind-recall-v1]'}]
    filtered = _filter(messages, contact_id=PERSON, watermark=page['head'], rules=page['events'],
        fresh=True, aliases={}, current_content=messages[-1]['content'], current_input='Current')
    assert filtered[:2] == messages[:2]
    assert filtered[-1]['content'] == 'Current\n'


@pytest.mark.parametrize('operation', ['forget', 'correct'])
def test_item_event_and_feed_rollback_atomically(state, monkeypatch, operation):
    ledger = ledger_of(state)
    curated.mutate(ledger, PERSON, 'mem_fixture', 'remember', request_id='create', content='Fixture')
    append = ledger._append_erasure
    fired = []

    def interrupted(conn, **kwargs):
        append(conn, **kwargs)
        with closing(ledger._connect()) as observer:
            assert observer.execute('SELECT count(*) FROM curated_memory_events').fetchone()[0] == 1
            assert observer.execute('SELECT count(*) FROM source_erasures').fetchone()[0] == 0
        fired.append(True)
        raise sqlite3.OperationalError('fixture interruption')

    monkeypatch.setattr(ledger, '_append_erasure', interrupted)
    with pytest.raises(sqlite3.OperationalError, match='fixture interruption'):
        curated.mutate(ledger, PERSON, 'mem_fixture', operation, request_id='withdraw',
            expected_revision=1, content='Replacement' if operation == 'correct' else None)
    assert fired
    assert ledger.erasure_feed(PERSON)['head'] == 0
    assert curated.search(ledger, PERSON)['items'][0]['revision'] == 1


@pytest.mark.parametrize('delivery', ['pending', 'delivered'])
@pytest.mark.parametrize('kind', ['normalized', 'text', 'identity-control'])
def test_consumer_purges_whole_source_original(tmp_path, delivery, kind):
    ledger = TurnIdempotencyLedger(tmp_path / 'ledger.db')
    message = {'role': 'user', 'content': 'Fixture'} if kind == 'text' else {
        'role': 'user', 'content': [{'type': 'input_image', 'image_url': 'fixture:unretained'}]}
    payload = {'turn_id': 'original', 'contact_id': PERSON, 'session_id': 'fixture-session',
               'checkpoint_messages': [message]}
    ledger.record_source('original', contact_id=PERSON, session_id=payload['session_id'],
                         messages=[message], derive_claims=False)
    outbox = SourceOutbox(tmp_path / 'outbox.db')
    outbox.prepare()
    outbox.enqueue('original', payload)
    if delivery == 'delivered':
        assert outbox.drain(lambda *a, **kw: True, limit=1, timeout_seconds=1) == 1
    assert outbox.lookup('original')['state'] == delivery
    before = historical_bytes(ledger)
    curated.forget_sources(ledger, PERSON, source_ids=['original'])
    page = ledger.erasure_feed(PERSON)
    rule, = page['events']
    with closing(ledger._connect()) as conn:
        stored = json.loads(conn.execute('SELECT messages_json FROM turn_sources').fetchone()[0])[0]
    assert rule['message_hashes'] == sorted({source_message_hash(payload['session_id'], stored),
                                           source_message_hash(payload['session_id'], message)})
    if kind == 'identity-control':
        page['events'] = [dict(rule, turn_id=rule['source_turn_id'])]
    outbox.apply_erasure_page(PERSON, page)
    assert outbox.snapshot() == []
    assert outbox.erasure_watermark(PERSON) == page['head']
    assert redact_source_payload(payload, page['events']) is None
    assert rule['whole_source'] and rule['source_turn_id'] == 'original'
    if kind == 'normalized':
        assert rule['turn_id'] == 'original'
    assert historical_bytes(ledger) == before
    curated.forget_sources(TurnIdempotencyLedger(ledger.db_path), PERSON, source_ids=['original'])
    assert ledger.erasure_feed(PERSON)['head'] == page['head']
