"""Scoped complete-source opening, stable paging, and honest oversized conflicts."""
from datetime import datetime, timezone
import hashlib
import json

from httpx import ASGITransport, AsyncClient
import pytest

from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.beliefs.source_projection import SourceClaimProjection
from colonymind.beliefs.source_time import interpret_time_query
from colonymind.memory.recall import pack_memory_context
from colonymind.turns import TurnIdempotencyLedger
from colonymind.turns.source_read import read
from onekey import KEY, _principal, _write_keyring
from test_source_claim_projection import Model, claim
from test_turn_source_evidence import source_app


def ref(ledger, identifier='source', **kwargs):
    return ledger.source_references([identifier], contact_id='person', session_id='later', **kwargs)[0]


def opened(ledger, identifier='source', **kwargs):
    return read(ledger, contact_id='person', session_id='later', **ref(ledger, identifier), **kwargs)


def test_checkpoint_opening_stays_in_its_session_and_correction_changes_continuation(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path/'source.db')
    ledger.record_source('private-session', contact_id='person', session_id='original', scope='session',
                         messages=[{'role': 'user', 'content': 'Session-scoped appendix.'}])
    private_ref = ledger.source_references(['private-session'], contact_id='person', session_id='original')[0]
    with pytest.raises(ValueError, match='unavailable'):
        read(ledger, contact_id='person', session_id='later', **private_ref)
    text = 'Archive digest matched. ' + 'Recorded details. ' * 500
    ledger.record_source('source', contact_id='person', session_id='original',
                         messages=[{'role': 'assistant', 'content': text}])
    first = opened(ledger)
    correction = 'The digest was not compared; that claim is unsupported.'
    ledger.append_source_annotation(contact_id='person', session_id='later', annotation_id='note',
        **ref(ledger), excerpt='Archive digest matched.', correction=correction, author_principal='operator')
    with pytest.raises(ValueError, match='restart_at_zero'):
        opened(ledger, offset=first['next_offset'], read_revision=first['read_revision'])
    content, page = '', opened(ledger)
    while True:
        content += page['content']
        if page['complete']:
            break
        page = opened(ledger, offset=page['next_offset'], read_revision=page['read_revision'])
    assert correction in content and 'attributed_correction' in content
    assert len(page['source_refs']) == 2


def test_corrected_old_document_keeps_record_times_distinct_from_its_contents_and_correction(tmp_path, monkeypatch):
    from colonymind.turns import source_annotations
    ledger = TurnIdempotencyLedger(tmp_path/'source.db')
    observed_at = '2026-09-12T08:00:00+00:00'
    text = 'Service handbook, 2021 edition: cancellation requires a telephone call.'
    ledger.record_source('source', contact_id='person', session_id='original',
        messages=[{'role': 'tool', 'content': text}], occurred_at=observed_at, derive_claims=False)
    original = opened(ledger)
    assert json.loads(original['content'])['messages'][0]['content'] == text
    assert original['reported_at'] == observed_at
    assert original['recorded_at'] != observed_at
    assert 'supported historical or stable facts' in original['guidance']
    assert 'does not re-inspect its underlying subject' in original['guidance']

    monkeypatch.setattr(source_annotations, 'now_utc', lambda: datetime(2026, 9, 13, 9, tzinfo=timezone.utc))
    assert opened(ledger) == original
    correction = 'As of 2026-09-13, the service also accepts cancellation through its portal.'
    note = ledger.append_source_annotation(contact_id='person', session_id='later', annotation_id='policy',
        **ref(ledger), excerpt='cancellation requires a telephone call',
        correction=correction, author_principal='operator')
    corrected = opened(ledger)
    assert corrected['reported_at'] == original['reported_at']
    assert corrected['recorded_at'] == original['recorded_at']
    assert corrected['source_version'] == original['source_version']
    assert corrected['read_revision'] != original['read_revision']
    assert text in corrected['content'] and correction in corrected['content']
    assert '2026-09-13T09:00:00+00:00' in corrected['content']
    assert 'operator' in corrected['content']
    assert note['source_id'] in {r['source_id'] for r in corrected['source_refs']}


def retain_call(ledger, origin, number, content, *, sources=(), reason='Useful inspection evidence.'):
    sid = f'historical-observation-{number}'
    ledger.record_source(sid, contact_id='person', session_id='original', derive_claims=False,
        messages=[{'role': 'tool', 'content': content, '_native_tool_observation': 'native-tool-observation-v1',
                   '_observation_sources': [origin, *sources]}])
    return ref(ledger, sid)


def observation_origin(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path/'source.db')
    ledger.record_source('source', contact_id='person', session_id='original',
        messages=[{'role': 'user', 'content': 'Inspect the Corvus export and its checksum.'}], derive_claims=False)
    return ledger, ref(ledger)


from colonymind.memory import curated

@pytest.mark.parametrize('reported_at', [None, '2024-02-03T04:05:06+00:00'])
def test_long_procedure_opens_every_step_and_scope_or_revision_never_widens(tmp_path, reported_at):
    ledger = TurnIdempotencyLedger(tmp_path/'source.db')
    text = 'Pump procedure: isolate pressure. ' + 'Check the seal; ' * 700 + 'Only then reconnect power.'
    ledger.record_source('source', contact_id='person', session_id='original',
                         messages=[{'role': 'user', 'content': text}], occurred_at=reported_at)
    with ledger._connect() as conn:
        row = conn.execute("SELECT ingested_at,messages_json FROM turn_sources WHERE turn_id='source'").fetchone()
    recorded_at = row['ingested_at']
    original = json.dumps({'messages': json.loads(row['messages_json']),
        'reported_at': reported_at, 'recorded_at': recorded_at,
        'event_time': 'unknown unless explicitly stated in each source'}, ensure_ascii=False)
    first = opened(ledger)
    assert not first['complete'] and len(first['content']) == 4096
    assert 'reported_at' not in first['content']
    assert first['read_revision'] == hashlib.sha256(original.encode()).hexdigest()
    pages, page = [first['content']], first
    while True:
        assert page['reported_at'] == reported_at and page['recorded_at'] == recorded_at
        assert page['evidence_basis'] == 'retained_record'
        if page['complete']:
            break
        page = opened(ledger, offset=page['next_offset'], read_revision=page['read_revision'])
        pages.append(page['content'])
    assert ''.join(pages) == original
    assert opened(ledger) == first
    assert json.loads(''.join(pages))['messages'] == [{'role': 'user', 'content': text}]
    assert first['source_refs'] == [ref(ledger)]
    for scope in ({'contact_id': 'other', 'session_id': 'later'},):
        with pytest.raises(ValueError, match='unavailable'):
            read(ledger, **scope, **ref(ledger))
    with pytest.raises(ValueError, match='unavailable'):
        read(ledger, contact_id='person', session_id='later', source_id='source', source_version='0'*64)
    saved = ref(ledger)
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(ledger)
    _source_forget_receipt = curated.forget_sources(ledger, 'person', source_ids=['source'])
    assert historical_bytes(ledger) == _native_before_forget
    assert all(curated.is_forgotten(ledger, sid) for sid in _source_forget_receipt['source_ids'])
    read(ledger, contact_id='person', session_id='later', **saved)


@pytest.mark.asyncio
async def test_oversized_conflict_is_discoverable_and_history_pages_current_sources(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path/'source.db')
    projection = SourceClaimProjection(ledger)
    for index in range(10):
        text = f'My office is in District{index}.'
        ledger.record_source(f's{index}', contact_id='person', session_id='original',
            messages=[{'role': 'user', 'content': text}])
        await projection.process_one(Model({text: claim(text, f'District{index}')}))
    _, rows = projection.prepare_context([], ledger.search_sources('office', contact_id='person', session_id='later'),
        contact_id='person', session_id='later', time_query=interpret_time_query('office', now=datetime.now(timezone.utc)))
    selected, packet = pack_memory_context(rows)
    assert 'incomplete_assertion_history' in packet and 'history_anchor' in packet
    assert not any(f'District{i}' in packet for i in range(10))
    anchor = selected[0]['history_anchor']
    first = opened(ledger, anchor['source_id'], view='assertions', claim_id=anchor['claim_id'])
    second = opened(ledger, anchor['source_id'], view='assertions', claim_id=anchor['claim_id'],
                    offset=first['next_offset'], read_revision=first['read_revision'])
    assert first['total'] == 10 and not first['complete'] and second['complete']
    assertions = json.loads(first['content'])['assertions'] + json.loads(second['content'])['assertions']
    assert {c['value'] for c in assertions} == {f'District{i}' for i in range(10)}
    assert all(c['superseded_by'] is None and c['retracted_by'] is None for c in assertions)
    victim = next(c['turn_id'] for c in assertions if c['turn_id'] != anchor['source_id'])
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(ledger)
    _source_forget_receipt = curated.forget_sources(ledger, 'person', source_ids=[victim])
    assert historical_bytes(ledger) == _native_before_forget
    assert all(curated.is_forgotten(ledger, sid) for sid in _source_forget_receipt['source_ids'])
    with pytest.raises(ValueError, match='restart_at_zero'):
        opened(ledger, anchor['source_id'], view='assertions', claim_id=anchor['claim_id'],
               offset=first['next_offset'], read_revision=first['read_revision'])
    assert opened(ledger, anchor['source_id'], view='assertions', claim_id=anchor['claim_id'])['total'] == 9


def test_observation_open_carries_parent_and_result_corrections_and_rejects_attribution_or_erase(tmp_path):
    from colonymind.turns.source_attribution import correct
    ledger, origin = observation_origin(tmp_path)
    observation = retain_call(ledger, origin, 1, 'Checksum mismatch; files modified 0.')
    notes = []
    for number, target, excerpt, correction in (
        (1, origin, 'Corvus export', 'The request meant the staging export, not production.'),
        (2, observation, 'Checksum mismatch', 'The expected checksum was supplied manually.')):
        notes.append(ledger.append_source_annotation(contact_id='person', session_id='later',
            annotation_id=f'note-{number}', **target, excerpt=excerpt, correction=correction, author_principal='operator'))
    for result in (opened(ledger, observation['source_id']),):
        assert 'staging export' in result['content'] and 'supplied manually' in result['content']
        assert all({key:note[key] for key in ('source_id','source_version')} in result['source_refs'] for note in notes)
    moved = correct(ledger, operation_id='correct-person', performed_by='operator',
        old_contact_id='person', contact_id='actual-person', source_ids=['source'], evidence_refs=['owner-confirmation'])
    assert observation['source_id'] in moved['invalidated_source_ids']
    assert not ledger.source_references([observation['source_id']], contact_id='person', session_id='later')
    with pytest.raises(ValueError, match='unavailable'):
        read(ledger, contact_id='person', session_id='later', **observation)
    # A separate current origin exercises existing erasure traversal as well.
    ledger.record_source('erase-origin', contact_id='person', session_id='original',
        messages=[{'role':'user','content':'Inspect a separate archive.'}], derive_claims=False)
    erased = retain_call(ledger, ref(ledger, 'erase-origin'), 2, 'Separate archive failed.')
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(ledger)
    _source_forget_receipt = curated.forget_sources(ledger, 'person', source_ids=['erase-origin'])
    assert historical_bytes(ledger) == _native_before_forget
    assert all(curated.is_forgotten(ledger, sid) for sid in _source_forget_receipt['source_ids'])
    read(ledger, contact_id='person', session_id='later', **erased)
