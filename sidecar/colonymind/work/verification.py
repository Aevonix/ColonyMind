"""Deterministic operation checks. A successful call alone is never proof."""
from __future__ import annotations


def effect_kind(operation):
    """Known tool semantics take priority; otherwise retain the host's effect."""
    tool, action = operation['tool'], operation['action']
    if tool in {'write_file', 'patch'}:
        return 'remove' if tool == 'patch' and operation.get('effect') == 'remove' else 'write'
    if tool in {'read_file', 'search_files', 'list_directory', 'web_search', 'web_extract',
                'session_search', 'skills_list', 'skill_view', 'kanban_get', 'kanban_list',
                'tool_search', 'tool_describe'}:
        return 'read'
    if tool == 'terminal':
        return 'read' if action in {'read', 'probe'} else 'background' if action == 'background' else 'execute'
    if tool == 'send_message':
        return 'read' if action == 'list' else 'send' if action == 'send' else 'effect_unknown'
    if tool in {'cronjob', 'cronjob_manage'}:
        return {'list': 'read', 'read': 'read', 'get': 'read', 'create': 'create', 'edit': 'edit',
                'update': 'edit', 'pause': 'edit', 'resume': 'edit', 'remove': 'remove',
                'run': 'dispatch', 'resnap': 'edit'}.get(action, 'effect_unknown')
    if tool in {'delegate_task', 'delegate', 'subagent', 'hands', 'kanban_create'}:
        return 'read' if action == 'list' else 'dispatch'
    return operation.get('effect', 'effect_unknown')


def verify(operation):
    """A status and evidence reason for this exact operation/resource pair."""
    kind = effect_kind(operation)
    evidence = operation.get('evidence') or {}
    if operation.get('success') is False:
        return 'failed', 'call_failed'
    if kind == 'effect_unknown':
        return 'unknown', 'unrecognized_effect'
    if kind == 'dispatch':
        return 'dispatched', 'outcome_pending'
    if operation.get('success') is not True:
        return 'unknown', 'outcome_missing'
    from colonymind.secrets.detect import WITHHELD
    import json
    if WITHHELD in json.dumps(evidence):
        return 'ran', 'evidence_withheld'
    if not operation.get('resource_ref'):
        return 'ran', 'resource_missing'
    if kind == 'write' and operation['tool'] in {'write_file', 'patch'} and evidence.get('kind') == 'file_hash':
        if evidence.get('method') == 'sealed_sha256':
            intended, actual = evidence.get('intended_sha256'), evidence.get('actual_sha256')
            if intended and actual:
                return ('verified', 'content_hash_matches') if intended == actual else ('failed', 'content_hash_mismatch')
        if evidence.get('method') == 'native_post_write_sha256':
            if evidence.get('verified') is True:
                return 'verified', 'native_content_hash_matches'
            if evidence.get('verified') is False:
                return 'failed', 'native_content_hash_mismatch'
    if kind == 'send' and operation['tool'] == 'send_message' and evidence.get('kind') == 'message_receipt':
        if evidence.get('receipt_id') or evidence.get('provider_message_id'):
            return 'verified', 'sent_receipt'
    if kind in {'create', 'edit'} and operation['tool'] in {'cronjob', 'cronjob_manage'} and evidence.get('kind') == 'saved_job':
        requested, saved = evidence.get('requested_fields'), evidence.get('saved_fields')
        if requested and saved is not None:
            if any(value is not True for value in requested.values()):
                return 'ran', 'saved_fields_uncheckable'
            if any(saved.get(key) is False for key in requested):
                return 'failed', 'saved_fields_mismatch'
            if any(key not in saved or saved[key] is not True for key in requested):
                return 'ran', 'saved_fields_uncheckable'
            return 'verified', 'saved_fields_match'
    return 'ran', 'successful_call_only'


def root_status(effects):
    statuses, verified = [], set()
    for item in reversed(effects):
        if item['kind'] == 'read':
            continue
        target = (item.get('tool'), item.get('resource_ref'))
        exact = all(target)
        if item['status'] == 'failed' and exact and target in verified:
            continue
        if item['status'] == 'verified' and exact:
            verified.add(target)
        statuses.append(item['status'])
    if 'failed' in statuses:
        return 'failed'
    if statuses and all(status == 'verified' for status in statuses):
        return 'proven'
    return 'partial' if 'verified' in statuses else 'unproven'
