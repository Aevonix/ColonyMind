"""Exact audio originals, fallible timed transcripts and canonical erasure."""
import base64
import copy
import hashlib
import io
import json
import sqlite3
import wave

import httpx
import pytest

from colonymind.turns import TurnIdempotencyLedger
from colonymind.turns.audio import decode_audio, source_text
from colonymind.turns.idempotency import SourceErased, source_message_hash
from colonymind.turns.media import SourceMedia
from colonymind.beliefs.source_projection import SourceClaimProjection
from test_turn_source_evidence import source_app, recalled, envelope


def wav_bytes(rate=16000, frames=1600):
    data = io.BytesIO()
    with wave.open(data, 'wb') as clip:
        clip.setnchannels(1); clip.setsampwidth(2); clip.setframerate(rate); clip.writeframes(b'\x01\x00' * frames)
    return data.getvalue()


def message(data=None):
    data = data or wav_bytes()
    return {'role': 'user', 'content': [
        {'type': 'input_audio', 'input_audio': {'format': 'wav', 'data': base64.b64encode(data).decode()}},
        {'type': 'audio_transcript', 'audio_sha256': hashlib.sha256(data).hexdigest(),
         'segments': [{'start_ms': 0, 'end_ms': 100, 'text': 'The fixture lamp is violet.'}],
         'recognizer': {'model_id': 'fixture-asr', 'model_revision': 'unknown'},
         'received_at': '2026-09-09T12:00:00+00:00', 'captured_at': None}]}


def retained(ledger, turn='audio'):
    with sqlite3.connect(ledger.db_path) as db:
        return json.loads(db.execute('SELECT messages_json FROM turn_sources WHERE turn_id=?', (turn,)).fetchone()[0])[0]


@pytest.mark.parametrize('variant', ['mp3', 'truncated', 'too-long', 'remote', 'video', 'transcript-hash', 'segment-range'])
def test_unsupported_audio_and_invalid_transcripts_are_not_silently_claimed_retained(tmp_path, variant):
    original = message()
    if variant == 'mp3': original['content'][0]['input_audio']['format'] = 'mp3'
    if variant == 'truncated': original['content'][0]['input_audio']['data'] = base64.b64encode(wav_bytes()[:-2]).decode()
    if variant == 'too-long': original['content'][0]['input_audio']['data'] = base64.b64encode(wav_bytes(frames=16000 * 61)).decode()
    if variant == 'remote': original['content'][0] = {'type': 'audio_url', 'url': 'https://private.invalid/audio?token=secret'}
    if variant == 'video': original['content'][0] = {'type': 'input_video', 'data': 'untrusted-video-bytes'}
    if variant == 'transcript-hash': original['content'][1]['audio_sha256'] = '0'*64
    if variant == 'segment-range': original['content'][1]['segments'][0]['end_ms'] = 101
    ledger = TurnIdempotencyLedger(tmp_path/'sources.db')
    ledger.record_source('audio', contact_id='person', session_id='call', messages=[original])
    stored = retained(ledger)
    assert 'unretained' in json.dumps(stored)
    assert not source_text(stored['content'])
    assert 'secret' not in json.dumps(stored) and 'untrusted-video-bytes' not in json.dumps(stored)


@pytest.mark.asyncio
async def test_audio_protocol_rejects_predecessor_before_raw_byte_storage_and_keeps_literal_ids(source_app):
    from fastapi import HTTPException, Response
    from colonymind.api.routers.host import turns_sync_v2
    from colonymind.api.schemas.host import TurnSyncRequest
    body = TurnSyncRequest.model_validate({'identity': {'host_id': 'fixture'},
        'context': {'contact_id': 'person', 'session_id': 'call', 'turn_id': 'clip'}, 'user_message': message()})
    with pytest.raises(HTTPException) as error:
        await turns_sync_v2('source-media/audio/clip', body, Response())
    assert error.value.status_code == 409 and error.value.detail['code'] == 'turn_id_mismatch'
    identifier = 'source-media/audio/old-literal-id'
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=source_app), base_url='http://fixture') as client:
        assert (await client.put('/v2/host/turns/' + identifier, json=envelope(identifier, checkpoint=True))).status_code == 201

from colonymind.memory import curated

def test_audio_memory_salvage_recovers_only_current_owned_original(tmp_path):
    from colonymind import backup
    state = tmp_path/'state'; ledger = TurnIdempotencyLedger(state/'turn-idempotency.db')
    ledger.record_source('audio', contact_id='person', session_id='call', messages=[message()], derive_claims=False)
    (state/'colonymind-id').write_text('fixture-audio-colonymind')
    archive = backup.create_full_backup(state, tmp_path/'archives', include_vectors=False)
    asset = hashlib.sha256(wav_bytes()).hexdigest()
    SourceMedia(ledger).store._original_path(asset, 'audio/wav').unlink()
    destination = tmp_path/'salvage'
    result = backup.restore_source_memory(archive, destination, current_state=state)
    assert result['images_recovered_from_archive'] == 1 and result['runtime_authority_restored'] is False
    recovered = SourceMedia(TurnIdempotencyLedger(destination/'turn-idempotency.db'))
    assert recovered.read(asset, contact_id='person', session_id='later')[0] == wav_bytes()
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(recovered.ledger)
    _source_forget_receipt = curated.forget_sources(recovered.ledger, 'person', source_ids=['audio'])
    assert historical_bytes(recovered.ledger) == _native_before_forget
    assert all(curated.is_forgotten(recovered.ledger, sid) for sid in _source_forget_receipt['source_ids'])


def test_audio_original_backup_and_shared_owner_erasure(tmp_path):
    from colonymind import backup
    state = tmp_path / 'state'
    ledger = TurnIdempotencyLedger(state / 'turn-idempotency.db')
    for person in ('a', 'b'):
        ledger.record_source(person, contact_id=person, session_id='call', messages=[message()], derive_claims=False)
    archive = backup.create_full_backup(state, tmp_path / 'archives', include_vectors=False)
    destination = tmp_path / 'restore'
    assert backup.restore_full_backup(archive, destination)['source_images'] == 1
    media = SourceMedia(TurnIdempotencyLedger(destination / 'turn-idempotency.db'))
    asset = hashlib.sha256(wav_bytes()).hexdigest()
    assert media.read(asset, contact_id='a', session_id='later')[0] == wav_bytes()
    curated.forget_sources(media.ledger, 'a', source_ids=['a'])
    assert media.read(asset, contact_id='a', session_id='later')[0] == wav_bytes()
    assert media.read(asset, contact_id='b', session_id='later')[0] == wav_bytes()
    curated.forget_sources(media.ledger, 'b', source_ids=['b'])
    assert media.store._original_path(asset, 'audio/wav').exists()
    assert curated.is_forgotten(media.ledger, 'a') and curated.is_forgotten(media.ledger, 'b')
