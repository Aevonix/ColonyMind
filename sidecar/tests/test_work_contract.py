"""Public work facts cannot certify another resource or an arbitrary tool."""
from copy import deepcopy
import json
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError
from referencing import Registry, Resource

from colonymind.work.models import Operation, WorkEnvelope, WorkEvent
from colonymind.work.verification import root_status, verify


SCHEMAS = Path(__file__).parents[1] / 'colonymind/work/schema'


def operation(**changes):
    return {
        'operation_id': 'call-1:0', 'tool_call_id': 'call-1', 'tool': 'write_file',
        'action': 'write', 'effect': 'write', 'resource_ref': 'resource:one',
        'success': True, 'evidence': {'kind': 'none'}, 'source_refs': [], **changes,
    }


def envelope(**changes):
    return {
        'version': 1, 'root_turn_id': 'owner-turn', 'parent_turn_id': None,
        'origin_ref': None, 'operations': [operation()], 'children': [],
        'truncated': False, **changes,
    }


def validator(name):
    schemas = [(path.name, json.loads(path.read_text())) for path in SCHEMAS.glob('*.json')]
    registry = Registry().with_resources((name, Resource.from_contents(value)) for name, value in schemas)
    return jsonschema.Draft202012Validator(dict(schemas)[name], registry=registry)


def test_frozen_examples_obey_both_runtime_and_published_schemas():
    work = envelope(operations=[operation(evidence={
        'kind': 'file_hash', 'method': 'native_post_write_sha256', 'verified': True})])
    validator('work-envelope.json').validate(work)
    parsed = WorkEnvelope.model_validate(work)
    validator('work-envelope.json').validate(parsed.model_dump(mode='json', exclude_none=True))
    event = {'event_id': 'late-1', 'root_turn_id': 'owner-turn', 'session_id': 'child-session',
             'work': envelope(parent_turn_id='owner-turn', children=[
                 {'turn_id': 'child-turn', 'parent_turn_id': 'owner-turn'}]), 'source_refs': []}
    validator('work-event.json').validate(event)
    WorkEvent.model_validate(event)


@pytest.mark.parametrize('field', ['arguments', 'result', 'command', 'content'])
def test_work_cannot_carry_raw_tool_payloads(field):
    work = envelope()
    work['operations'][0][field] = 'private words'
    with pytest.raises(ValidationError):
        WorkEnvelope.model_validate(work)
    with pytest.raises(jsonschema.ValidationError):
        validator('work-envelope.json').validate(work)


@pytest.mark.parametrize('tool', ['terminal', 'mcp_read_file', 'mcp_write_file', 'other_tool'])
def test_native_file_attestation_does_not_certify_other_tools(tool):
    value = operation(tool=tool, evidence={'kind': 'file_hash',
        'method': 'native_post_write_sha256', 'verified': True})
    assert verify(value)[0] != 'verified'


def test_verified_flag_alone_and_missing_resource_are_not_file_proof():
    value = operation(evidence={'kind': 'file_hash', 'verified': True})
    assert verify(value)[0] == 'ran'
    value['evidence']['method'] = 'native_post_write_sha256'
    value['resource_ref'] = None
    assert verify(value)[0] == 'ran'


def test_content_mismatch_and_call_failure_cannot_be_hidden_by_successful_target():
    bad = operation(evidence={'kind': 'file_hash', 'method': 'sealed_sha256',
                             'intended_sha256': 'a' * 64, 'actual_sha256': 'b' * 64})
    assert verify(bad)[0] == 'failed'
    good = operation(operation_id='call-2:0', resource_ref='resource:two', evidence={
        'kind': 'file_hash', 'method': 'native_post_write_sha256', 'verified': True})
    assert root_status([{'kind': 'write', 'status': verify(value)[0]} for value in [bad, good]]) == 'failed'
    good['success'] = False
    assert verify(good)[0] == 'failed'


def test_saved_job_missing_display_fields_are_uncheckable_but_false_comparisons_fail():
    value = operation(tool='cronjob', action='update', effect='edit', evidence={
        'kind': 'saved_job', 'requested_fields': {'paused': True, 'repeat': True},
        'saved_fields': {'paused': True}})
    assert verify(value) == ('ran', 'saved_fields_uncheckable')
    value['evidence']['saved_fields']['repeat'] = False
    assert verify(value) == ('failed', 'saved_fields_mismatch')
    value['evidence']['saved_fields']['repeat'] = True
    assert verify(value)[0] == 'verified'


def test_message_listing_and_delegation_success_cannot_be_sent_or_completed():
    listed = operation(tool='send_message', action='list', effect='send', evidence={
        'kind': 'message_receipt', 'provider_message_id': 'provider-1'})
    assert verify(listed)[0] != 'verified'
    delegated = operation(tool='delegate_task', action='dispatch', effect='dispatch', evidence={
        'kind': 'hand_outcome', 'verified': True, 'outcome': 'done'})
    assert verify(delegated)[0] == 'dispatched'


def test_duplicate_operations_and_cross_root_events_are_refused():
    work = envelope()
    work['operations'].append(deepcopy(work['operations'][0]))
    with pytest.raises(ValidationError):
        WorkEnvelope.model_validate(work)
    with pytest.raises(ValidationError):
        WorkEvent.model_validate({'event_id': 'event', 'root_turn_id': 'other-root',
            'session_id': 'session', 'work': envelope(), 'source_refs': []})


def test_hashes_require_explicit_sealed_representation():
    with pytest.raises(ValidationError):
        Operation.model_validate(operation(evidence={
            'kind': 'file_hash', 'intended_sha256': 'a' * 64, 'actual_sha256': 'a' * 64}))


def test_envelope_and_selected_evidence_have_hard_bounds():
    with pytest.raises(ValidationError):
        WorkEnvelope.model_validate(envelope(operations=[operation(operation_id=str(i)) for i in range(257)]))
    with pytest.raises(ValidationError):
        Operation.model_validate(operation(evidence={'kind': 'saved_job',
            'requested_fields': {'name': 'x' * 1025}}))


@pytest.mark.parametrize('changes', [
    {'action': 'Remind the owner about private results'},
    {'evidence': {'kind': 'saved_job', 'requested_fields': {'prompt': 'Private appointment'}}},
    {'evidence': {'kind': 'saved_job', 'saved_fields': {'repeat': 5}}},
    {'evidence': {'kind': 'hand_outcome', 'outcome': 'Private appointment completed'}},
])
def test_host_metadata_cannot_carry_argument_values(changes):
    work = envelope(operations=[operation(**changes)])
    with pytest.raises(ValidationError):
        WorkEnvelope.model_validate(work)
    with pytest.raises(jsonschema.ValidationError):
        validator('work-envelope.json').validate(work)


def test_unknown_tool_has_no_retained_action():
    parsed = Operation.model_validate(operation(tool='mcp_calendar', action='schedule'))
    assert parsed.action == ''


def test_mechanical_failure_kind_is_independent_of_semantic_verification():
    value = operation(request_id='request', model='producer', error_type=None, failure_kind=None,
        duration_ms=12, evidence={'kind': 'file_hash', 'method': 'sealed_sha256',
                                 'intended_sha256': 'a' * 64, 'actual_sha256': 'b' * 64})
    parsed = Operation.model_validate(value).model_dump(exclude_none=True)
    assert verify(parsed) == ('failed', 'content_hash_mismatch')
    assert parsed.get('failure_kind') is None and parsed.get('error_type') is None
    validator('work-envelope.json').validate(envelope(operations=[parsed]))


@pytest.mark.parametrize(('field', 'value'), [
    ('request_id', 'request'), ('model', 'producer'), ('error_type', 'tool_timeout'),
    ('failure_kind', 'timeout'), ('duration_ms', 0.0),
])
def test_canonical_hash_omits_only_null_capture_metadata(field, value):
    from colonymind.turns import canonical_turn_digest

    payload = {'work': envelope(operations=[operation(resource_ref=None)])}
    baseline = canonical_turn_digest(payload)
    with_nulls = deepcopy(payload)
    op = with_nulls['work']['operations'][0]
    op.update(dict.fromkeys(('request_id', 'model', 'error_type', 'failure_kind', 'duration_ms')))
    assert canonical_turn_digest(with_nulls) == baseline
    assert all(key in op for key in ('request_id', 'model', 'error_type', 'failure_kind', 'duration_ms'))
    op[field] = value
    assert canonical_turn_digest(with_nulls) != baseline
    del op[field]
    op.pop('resource_ref')
    assert canonical_turn_digest(with_nulls) != baseline


@pytest.mark.parametrize('changes', [
    {'failure_kind': 'semantically_wrong_arguments'}, {'duration_ms': -1},
    {'duration_ms': float('inf')}, {'model': 'x' * 257}, {'request_id': 'x' * 257},
])
def test_failure_metadata_bounds(changes):
    with pytest.raises(ValidationError):
        Operation.model_validate(operation(**changes))
