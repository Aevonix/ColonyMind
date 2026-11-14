"""Original client media and complete affect reads survive the review schedules."""
import base64
from contextlib import closing
from datetime import datetime, timedelta, timezone
import io
import json

from PIL import Image
import pytest

from colonymind.contacts.affect import AffectStore
from colonymind.contacts.comms import CommsLog
from colonymind.memory import curated
from colonymind.turns import TurnIdempotencyLedger
from colonymind.turns.idempotency import canonical_turn_digest
from test_curated_read_races import withdraw
from voice_consumer_snapshot import SourceOutbox, check_request, source_message_hash


OWNER, SESSION = 'fixture-owner', 'fixture-session'


class FeedResponse:
    def __init__(self, page):
        self.page = page

    def raise_for_status(self):
        pass

    def json(self):
        return self.page


class FeedClient:
    def __init__(self, ledger):
        self.ledger = ledger

    def get(self, path, params, **kwargs):
        assert path == '/v1/host/memory/sources/erasures'
        return FeedResponse(self.ledger.erasure_feed(params['contact_id'], after=params['after']))


@pytest.mark.parametrize('kind', ['retained_image', 'retained_input_image',
                                  'unretained_image', 'plain_text_control'])
def test_original_client_media_is_withheld(tmp_path, kind):
    buffer = io.BytesIO()
    Image.new('RGB', (4, 4), (10, 20, 30)).save(buffer, 'PNG')
    png = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
    content = {
        'retained_image': [{'type': 'text', 'text': 'Fixture'},
                           {'type': 'image_url', 'image_url': {'url': png}}],
        'retained_input_image': [{'type': 'input_text', 'text': 'Fixture'},
                                 {'type': 'input_image', 'image_url': png}],
        'unretained_image': [{'type': 'input_image', 'image_url': 'fixture:unretained'}],
        'plain_text_control': 'Fixture',
    }[kind]
    original = {'role': 'user', 'content': content}
    ledger = TurnIdempotencyLedger(tmp_path / 'sources.db')
    ledger.record_source('original', contact_id=OWNER, session_id=SESSION,
                         messages=[original], derive_claims=False)
    with closing(ledger._connect()) as conn:
        messages = json.loads(conn.execute('SELECT messages_json FROM turn_sources').fetchone()[0])
    stored, = messages
    original_hash = source_message_hash(SESSION, original)
    stored_hash = source_message_hash(SESSION, stored)
    if kind != 'plain_text_control':
        assert stored['content'] != content
        assert stored['_source_message_hash'] == original_hash != stored_hash
    curated.forget_sources(ledger, OWNER, source_ids=['original'])
    page = ledger.erasure_feed(OWNER)
    rule, = page['events']
    assert rule['message_hashes'] == sorted({original_hash, stored_hash})
    assert rule['source_version'] == canonical_turn_digest(messages)
    assert rule['whole_source'] is True
    assert rule['turn_id'] == rule['source_turn_id'] == 'original'
    outbox = SourceOutbox(tmp_path / 'outbox.db')
    outbox.prepare()
    current = 'Fixture current input'
    history = [original, {'role': 'assistant', 'content': 'Fixture reply'},
               {'role': 'user', 'content': current}]
    result = check_request(FeedClient(ledger), outbox, contact_id=OWNER,
                           observed=history, messages=history, current_input=current)
    assert result['reason'] == 'source_erasure_checked'
    assert original not in result['messages']
    assert len(result['messages']) == 2
    assert result['messages'][-1] == history[-1]
    assert result['messages'][0]['role'] == 'user'
    assert result['messages'][0]['content'] != original['content']
    curated.forget_sources(TurnIdempotencyLedger(ledger.db_path), OWNER, source_ids=['original'])
    assert ledger.erasure_feed(OWNER) == page


@pytest.mark.parametrize('whole', [False, True])
@pytest.mark.parametrize('preserved', ['a' * 64, 'A' * 64, 'a' * 63, 'g' * 64, 7, None])
def test_publisher_unions_only_valid_preserved_identity(tmp_path, whole, preserved):
    ledger = TurnIdempotencyLedger(tmp_path / 'sources.db')
    message = {'role': 'user', 'content': 'Fixture', '_source_message_hash': preserved}
    with closing(ledger._connect()) as conn, conn:
        ledger._append_erasure(conn, turn_id='original', contact_id=OWNER,
            session_id=SESSION, scope='person', messages=[message],
            removed=[message, message], whole=whole)
    rule, = ledger.erasure_feed(OWNER)['events']
    expected = {source_message_hash(SESSION, message)}
    if preserved == 'a' * 64:
        expected.add(preserved)
    assert rule['message_hashes'] == sorted(expected)
    assert rule['whole_source'] is whole
    assert rule['source_version'] == canonical_turn_digest([message])


@pytest.fixture
def declining(tmp_path, monkeypatch):
    from colonymind.contacts import affect as module
    now = datetime(2026, 10, 20, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(module, '_now', lambda: now)
    ledger = TurnIdempotencyLedger(tmp_path / 'sources.db')
    log = CommsLog(str(tmp_path / 'comms.db'), source_ledger=ledger)
    affect = AffectStore(str(tmp_path / 'affect.db'), source_ledger=ledger)
    for index, valence in enumerate([-.4, -.5, -.8, -.9]):
        source = 'e' + str(index + 1)
        ledger.record_source(source, contact_id=OWNER, session_id=SESSION,
            messages=[{'role': 'user', 'content': source}], derive_claims=False)
        lineage, _ = log.source_input(source, OWNER)
        affect.create_event(contact_id=OWNER, valence=valence, arousal=.6,
            source_lineage=lineage, timestamp=(now - timedelta(minutes=10-index)).isoformat())
    ledger.record_source('seed', contact_id=OWNER, session_id=SESSION,
        messages=[{'role': 'user', 'content': 'Fixture seed'}], derive_claims=False)
    curated.forget_sources(ledger, OWNER, source_ids=['seed'])
    assert affect.detect_sustained_decline(OWNER) is True
    return ledger, affect


@pytest.mark.parametrize('method', ['detect_sustained_decline', 'trend'])
@pytest.mark.parametrize('withdrawn', [True, False])
def test_complete_decline_read(declining, monkeypatch, method, withdrawn):
    ledger, affect = declining
    original = affect._visible_events
    fired = []

    def visible(rows):
        if withdrawn and not fired:
            withdraw(ledger, 'e1')
            fired.append(True)
        return original(rows)

    monkeypatch.setattr(affect, '_visible_events', visible)
    offered = getattr(affect, method)(OWNER)
    monkeypatch.setattr(affect, '_visible_events', original)
    assert bool(fired) is withdrawn
    assert offered == getattr(affect, method)(OWNER)
    if method == 'trend':
        assert offered['declining'] is (not withdrawn)
        if withdrawn:
            assert offered == {'valence': 0.0, 'trend': 'stable', 'declining': False}
    else:
        assert offered is (not withdrawn)
