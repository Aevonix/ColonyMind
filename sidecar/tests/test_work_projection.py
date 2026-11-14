"""Offline work facts survive projection loss without inventing completion."""
from contextlib import closing
from copy import deepcopy
import json
import hashlib
import sqlite3
from types import SimpleNamespace

import pytest

from colonymind.turns import TurnIdempotencyLedger, canonical_turn_digest
from colonymind.work.models import WorkEnvelope
from colonymind.work.store import WorkStore, read_summary
from colonymind.work.verification import verify


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(tmp_path))
    monkeypatch.setenv('COLONYMIND_URL', '')
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    return SimpleNamespace(ledger=ledger, work=WorkStore(ledger), path=tmp_path)


def operation(ident='write', **changes):
    return {'operation_id': ident, 'tool_call_id': ident, 'tool': 'write_file', 'action': 'write',
            'effect': 'write', 'resource_ref': 'resource:one', 'success': True,
            'evidence': {'kind': 'file_hash', 'method': 'native_post_write_sha256', 'verified': True},
            'source_refs': [], **changes}


def envelope(ident='root', operations=None, **changes):
    return {'version': 1, 'root_turn_id': ident, 'operations': operations if operations is not None else [operation()],
            'children': [], 'truncated': False, **changes}


def admit(stores, work=None, *, source_id='root', messages=None):
    work = work or envelope(source_id)
    return stores.ledger.record_source(source_id, contact_id='owner', session_id='session',
        messages=messages or [{'role': 'user', 'content': 'Please change the report.'}],
        work={'envelope': work, 'principal_id': 'host'}, derive_claims=False)


def root(stores, ident='root'):
    stores.work.wait_pending(timeout=2)
    stores.work.reconcile()  # Explicit maintenance barrier; API reads never repair.
    return stores.work.get_root(ident, contact_id='owner')


def event(work, ident='late'):
    return {'event_id': ident, 'root_turn_id': work['root_turn_id'], 'session_id': 'child-session',
            'work': work, 'source_refs': []}


def test_projection_is_repairable_after_admission_crash(stores, monkeypatch):
    admit(stores)
    # No projection write has happened, representing a crash at that boundary.
    with closing(stores.work._connect()) as conn:
        assert conn.execute('SELECT count(*) FROM work_roots').fetchone()[0] == 0
    assert root(stores)['status'] == 'proven'
    stores.work.path.unlink()
    assert root(stores)['status'] == 'proven'
    stores.work.path.unlink()
    stores.work = WorkStore(TurnIdempotencyLedger(stores.ledger.db_path))
    assert root(stores)['status'] == 'proven'


@pytest.mark.parametrize(('operation_value', 'status'), [
    (operation(evidence={'kind': 'none'}), 'ran'),
    (operation(success=False), 'failed'),
    (operation(success=None), 'unknown'),
    (operation(evidence={'kind': 'file_hash', 'method': 'sealed_sha256', 'intended_sha256': 'a' * 64, 'actual_sha256': 'b' * 64}), 'failed'),
    (operation(evidence={'kind': 'file_hash', 'method': 'sealed_sha256', 'intended_sha256': 'a' * 64, 'actual_sha256': 'a' * 64}), 'verified'),
    (operation(tool='patch', evidence={'kind': 'none'}), 'ran'),
    (operation(tool='send_message', action='send', evidence={'kind': 'message_receipt', 'provider_message_id': 'provider-id'}), 'verified'),
    (operation(tool='send_message', action='send', evidence={'kind': 'none'}), 'ran'),
    (operation(tool='cronjob', action='edit', evidence={'kind': 'saved_job', 'requested_fields': {'enabled': True}, 'saved_fields': {'enabled': True}}), 'verified'),
    (operation(tool='cronjob', action='edit', evidence={'kind': 'saved_job', 'requested_fields': {'enabled': True}, 'saved_fields': {'enabled': False}}), 'failed'),
    (operation(tool='mcp_read_document', effect='effect_unknown'), 'unknown'),
    (operation(tool='delegate_task', action='run'), 'dispatched'),
])
def test_tool_specific_postconditions(operation_value, status):
    assert verify(operation_value)[0] == status


def test_read_wrong_resource_and_failure_cannot_be_hidden(stores):
    operations = [operation('a', success=False), operation('b', resource_ref='resource:two'),
        operation('read', tool='read_file', action='read', resource_ref='resource:two', evidence={'kind': 'read'})]
    admit(stores, envelope(operations=operations))
    result = root(stores)
    assert result['status'] == 'failed'
    assert {item['operation_id']: item['status'] for item in result['effects']}['a'] == 'failed'


def test_exact_later_read_observed_is_not_verified(stores):
    admit(stores, envelope(operations=[operation(evidence={'kind': 'none'}),
        operation('read', tool='read_file', action='read', evidence={'kind': 'read'})]))
    result = root(stores)
    assert result['status'] == 'unproven'
    assert next(item for item in result['effects'] if item['operation_id'] == 'write')['status'] == 'observed'


def test_late_child_after_reply_and_event_conflicts(stores):
    admit(stores, envelope(operations=[operation(tool='delegate_task', action='run')],
                          children=[{'turn_id': 'child', 'parent_turn_id': 'root'}]))
    assert root(stores)['status'] == 'unproven'
    body = event(envelope(operations=[operation('child-write')], parent_turn_id='child'))
    assert stores.work.admit_event(body, contact_id='owner', principal_id='host')['duplicate'] is False
    assert stores.work.admit_event(body, contact_id='owner', principal_id='host')['duplicate'] is True
    assert root(stores)['status'] == 'partial'
    changed = deepcopy(body)
    changed['work']['operations'][0]['success'] = False
    with pytest.raises(ValueError, match='work_operation_conflict'):
        stores.work.admit_event(changed, contact_id='owner', principal_id='host')
    changed['event_id'] = 'different'
    with pytest.raises(ValueError, match='work_operation_conflict'):
        stores.work.admit_event(changed, contact_id='owner', principal_id='host')


def test_late_parent_event_and_unknown_parent(stores):
    admit(stores, envelope(operations=[]))
    late = event(envelope(operations=[operation('late')]))
    stores.work.admit_event(late, contact_id='owner', principal_id='host')
    assert root(stores)['status'] == 'proven'
    bad = event(envelope(operations=[], parent_turn_id='unrelated'), 'bad')
    with pytest.raises(ValueError, match='work_parent_missing'):
        stores.work.admit_event(bad, contact_id='owner', principal_id='host')


def test_empty_and_truncated_never_proven(stores):
    admit(stores, envelope(operations=[]))
    admit(stores, envelope('other', truncated=True), source_id='other')
    assert root(stores) is None
    assert root(stores, 'other')['status'] == 'partial'


def test_seals_work_before_digest_or_persistence(stores, monkeypatch):
    from colonymind.secrets import seal
    raw = 'sensitive-placeholder'
    original = seal.seal_text
    monkeypatch.setattr(seal, 'seal_text', lambda value, **kwargs: '[withheld]' if raw in value else original(value, **kwargs))
    admit(stores, envelope(operations=[operation(tool='send_message', action='send',
        evidence={'kind': 'message_receipt', 'receipt_id': raw})]))
    with closing(stores.ledger._connect()) as conn:
        row = conn.execute('SELECT envelope_json,content_sha256 FROM work_envelopes').fetchone()
    assert raw not in row[0]
    assert '[withheld]' in row[0]
    assert root(stores)['effects'][0]['resource_ref'] == 'resource:one'


def test_hands_exact_origin_and_verified_outcome(stores):
    origin = {'platform': 'test', 'chat': 'room', 'message_id': 'message'}
    admit(stores, envelope(operations=[], origin_ref=json.dumps(origin, sort_keys=True, separators=(',', ':'))))
    with sqlite3.connect(stores.path / 'hands.db') as conn:
        conn.executescript('''CREATE TABLE tasks(id,origin,origin_ref,intent_version);
            CREATE TABLE hands(id,task_id,state,evidence,intent_version);''')
        conn.execute('INSERT INTO tasks VALUES (?,?,?,?)', ('task', 'owner_turn', json.dumps(origin), 1))
        conn.execute('INSERT INTO hands VALUES (?,?,?,?,?)', ('hand', 'task', 'done', json.dumps({'level': 'E0', 'scope': {'kind': 'build'}}), 1))
    assert root(stores)['status'] == 'unproven'
    with sqlite3.connect(stores.path / 'hands.db') as conn:
        conn.execute('UPDATE hands SET evidence=?', (json.dumps({'level': 'E3', 'scope': {'kind': 'build', 'reference': 'artifact'}}),))
    assert root(stores)['status'] == 'proven'
    with sqlite3.connect(stores.path / 'hands.db') as conn:
        conn.execute('UPDATE hands SET state=?', ('failed',))
    assert root(stores)['status'] == 'failed'


def test_read_only_summary_and_pagination(stores):
    for index in range(3):
        admit(stores, envelope(str(index)), source_id=str(index))
    stores.work.reconcile()
    first = stores.work.list_roots(contact_id='owner', limit=2)
    second = stores.work.list_roots(contact_id='owner', limit=2, offset=2)
    assert len(first) == 2 and len(second) == 1
    assert read_summary(stores.work.path, contact_id='owner')['counts']['proven'] == 3


def test_absent_work_digest_preserves_legacy_payload():
    from colonymind.api.schemas.host import TurnSyncRequest
    body = TurnSyncRequest(identity={'host_id': 'test'}, context={'session_id': 'session', 'contact_id': 'owner'})
    current = body.model_dump(mode='json')
    legacy = {key: value for key, value in current.items() if key != 'work'}
    assert canonical_turn_digest(body) == canonical_turn_digest(legacy)


def test_model_bounds_and_version():
    with pytest.raises(ValueError):
        WorkEnvelope.model_validate(envelope(version=2))
    with pytest.raises(ValueError):
        WorkEnvelope.model_validate(envelope(operations=[operation()] * 257))


def test_work_only_sealed_value_is_committed_before_work_digest(stores):
    from colonymind.secrets.refs import references_in
    raw = 'Bearer ' + ('Ab12Cd34' * 4)
    admit(stores, envelope(operations=[operation(tool='send_message', action='send',
        evidence={'kind': 'message_receipt', 'receipt_id': 'Authorization: ' + raw})]))
    with closing(stores.ledger._connect()) as conn:
        row = conn.execute('SELECT envelope_json FROM work_envelopes').fetchone()[0]
        dump = '\n'.join(conn.iterdump())
    assert raw not in dump and hashlib.sha256(raw.encode()).hexdigest() not in dump
    references = references_in(json.loads(row)['operations'][0]['evidence']['receipt_id'])
    assert len(references) == 1
    assert stores.ledger.vault.resolve(references[0].ref_id) == raw.removeprefix('Bearer ')


@pytest.mark.parametrize('resource', ['/home/user/private.txt', 'https://example.test/private?token=value', 'file:private'])
def test_literal_resources_are_rejected(resource):
    with pytest.raises(ValueError):
        WorkEnvelope.model_validate(envelope(operations=[operation(resource_ref=resource)]))


def test_evidence_keys_cannot_carry_secrets():
    with pytest.raises(ValueError):
        WorkEnvelope.model_validate(envelope(operations=[operation(evidence={
            'kind': 'saved_job', 'requested_fields': {'private value as a key': True}})]))


def test_withheld_fields_never_verify():
    from colonymind.secrets.detect import WITHHELD
    value = operation(tool='cronjob', action='edit', evidence={'kind': 'saved_job',
        'requested_fields': {'prompt': WITHHELD}, 'saved_fields': {'prompt': WITHHELD}})
    assert verify(value)[0] == 'ran'


def test_projected_effect_ids_do_not_alias_between_roots(stores):
    admit(stores, envelope('a:b', operations=[operation('c')]), source_id='a:b')
    admit(stores, envelope('a', operations=[operation('b:c')]), source_id='a')
    stores.work.reconcile()
    assert len(root(stores, 'a:b')['effects']) == len(root(stores, 'a')['effects']) == 1
    with closing(stores.work._connect()) as conn:
        assert conn.execute('SELECT count(*) FROM work_effects').fetchone()[0] == 2


def test_late_job_check_binds_exact_operation_and_resource(stores):
    initial = operation('job-edit', tool='cronjob', action='edit', evidence={'kind': 'none'})
    admit(stores, envelope(operations=[initial]))
    check = operation('saved-read', tool='cronjob', action='get', evidence={'kind': 'saved_job',
        'method': 'native_saved_job_fields', 'checked_operation_id': 'job-edit',
        'requested_fields': {'paused': True}, 'saved_fields': {'paused': True}})
    stores.work.admit_event(event(envelope(operations=[check])), contact_id='owner', principal_id='host')
    assert root(stores)['status'] == 'proven'


def test_malformed_native_marker_cannot_certify_work(stores):
    admit(stores, envelope(operations=[operation(evidence={'kind': 'none'})]))
    refs = stores.ledger.source_references(['root'], contact_id='owner', session_id='session')
    source = {'role': 'tool', 'content': json.dumps({'success': True, 'verified': True}),
        '_native_tool_observation': 'native-tool-observation-v1', '_observation_sources': refs,
        'provenance': {'native': {'tool_call_id': 'write', 'tool_name': 'write_file',
                                   'turn_id': 'root', 'session_id': 'session'}}}
    stores.ledger.record_source('malformed', contact_id='owner', session_id='session', messages=[source])
    assert root(stores)['status'] == 'unproven'

from colonymind.memory import curated

def test_erased_verdict_invalidates_overlay_without_removing_root(stores):
    admit(stores)
    stores.ledger.record_source('verdict', contact_id='owner', session_id='session', messages=[{'role': 'user', 'content': 'Confirmed'}])
    refs = stores.ledger.source_references(['verdict'], contact_id='owner', session_id='session')
    before = root(stores)
    assert stores.work.write_derived('root', contact_id='owner', expected_revision=before['revision'], overlay='confirmed', source_refs=refs)
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(stores.ledger)
    _source_forget_receipt = curated.forget_sources(stores.ledger, 'owner', source_ids=['verdict'])
    assert historical_bytes(stores.ledger) == _native_before_forget
    assert all(curated.is_forgotten(stores.ledger, sid) for sid in _source_forget_receipt['source_ids'])
    assert root(stores)['status'] == 'proven'


def test_historical_tool_result_cannot_upgrade_work_and_keeps_erasure(stores):
    admit(stores, envelope(operations=[operation(evidence={'kind': 'none'})]))
    refs = stores.ledger.source_references(['root'], contact_id='owner', session_id='session')
    content = json.dumps({'success': True, 'verified': True})
    native = {'tool_call_id': 'write', 'tool_name': 'write_file', 'turn_id': 'root', 'session_id': 'session',
              'task_id': 'task', 'api_request_id': 'request', 'profile_id': 'a' * 64, 'message_id': 1,
              'timestamp': 1700000000.0, 'result_sha256': hashlib.sha256(content.encode()).hexdigest()}
    identifier = 'native-observation:' + hashlib.sha256(json.dumps(native, sort_keys=True,
        separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    stores.ledger.record_source(identifier, contact_id='owner', session_id='session',
        messages=[{'role': 'tool', 'content': content, '_native_tool_observation': 'native-tool-observation-v1',
                   '_observation_sources': refs, 'provenance': {'kind': 'native-tool-observation-v1',
                       'native': native, 'selection': {'author': 'model', 'reason': 'Exact original result'}}}],
        derive_claims=False)
    assert root(stores)['status'] == 'unproven'
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(stores.ledger)
    _source_forget_receipt = curated.forget_sources(stores.ledger, 'owner', source_ids=[identifier])
    assert historical_bytes(stores.ledger) == _native_before_forget
    assert all(curated.is_forgotten(stores.ledger, sid) for sid in _source_forget_receipt['source_ids'])
    assert root(stores)['status'] == 'unproven'


def test_erased_late_event_does_not_upgrade_surviving_root(stores):
    admit(stores)
    stores.ledger.record_source('late-source', contact_id='owner', session_id='session',
                                messages=[{'role': 'tool', 'content': 'A failed outcome'}])
    body = event(envelope(operations=[operation('late-failure', success=False)]))
    body['source_refs'] = stores.ledger.source_references(['late-source'], contact_id='owner', session_id='session')
    stores.work.admit_event(body, contact_id='owner', principal_id='host')
    assert root(stores)['status'] == 'failed'
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(stores.ledger)
    _source_forget_receipt = curated.forget_sources(stores.ledger, 'owner', source_ids=['late-source'])
    assert historical_bytes(stores.ledger) == _native_before_forget
    assert all(curated.is_forgotten(stores.ledger, sid) for sid in _source_forget_receipt['source_ids'])
    assert root(stores)['status'] == 'partial'


def test_whole_erasure_fences_replay_restart_and_repair(stores):
    admit(stores)
    assert root(stores)
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(stores.ledger)
    curated.forget_sources(stores.ledger, 'owner', source_ids=['root'])
    assert historical_bytes(stores.ledger) == _native_before_withdrawal
    assert root(stores) is None
    admit(stores)
    assert root(stores) is None
    stores.work = WorkStore(TurnIdempotencyLedger(stores.ledger.db_path))
    assert root(stores) is None

def test_partial_erasure_preserves_unrelated_operation_and_invalidates_derivatives(stores):
    ledger = stores.ledger
    ledger.record_source('evidence', contact_id='owner', session_id='session', messages=[{'role': 'tool', 'content': 'Old evidence'}, {'role': 'tool', 'content': 'Remaining evidence'}])
    refs = ledger.source_references(['evidence'], contact_id='owner', session_id='session')
    admit(stores, envelope(operations=[operation('bound', source_refs=refs), operation('independent')]))
    before = root(stores)
    assert stores.work.write_derived('root', contact_id='owner', expected_revision=before['revision'], title='Checked file work', overlay='confirmed', source_refs=refs)
    ledger.record_source('erase-one', contact_id='owner', session_id='session', messages=[{'role': 'tool', 'content': 'Old evidence'}])
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger)
    curated.forget_sources(ledger, 'owner', source_ids=['erase-one'])
    assert historical_bytes(ledger) == _native_before_withdrawal
    after = root(stores)
    assert after['title'] is None and after['status'] == 'partial'
    assert 'independent' in {item['operation_id'] for item in after['effects']}
    assert not stores.work.write_derived('root', contact_id='owner', expected_revision=before['revision'], title='Stale')
    with closing(stores.work._connect()) as conn:
        assert conn.execute('SELECT count(*) FROM work_overlays').fetchone()[0] == 0
    with closing(ledger._connect()) as conn:
        retained = json.loads(conn.execute("SELECT messages_json FROM turn_sources WHERE turn_id='evidence'").fetchone()[0])
    assert len(retained) == 2
    assert [m['content'] for m in visible_source_messages(ledger, 'evidence')] == ['Remaining evidence']

from test_curated_memory import visible_source_messages
