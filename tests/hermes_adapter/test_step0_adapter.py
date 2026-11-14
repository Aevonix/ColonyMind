"""Native policy dispatch and delivery contracts for the two Step 0 revisions."""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

from colonymind_hermes import secrets as secret_module
from colonymind_hermes.answers import Answers
from colonymind_hermes.body import Body, BodyLedger, send_message
from colonymind_hermes.guard import effect_of


@pytest.fixture
def secrets_guard(tmp_path, monkeypatch):
    guard = secret_module.Secrets(NS(), NS(), NS(home=tmp_path, worker_profile='worker'))
    monkeypatch.setattr(guard, 'owner_session', lambda session: True)
    monkeypatch.setattr(secret_module, 'mind_run', lambda settings: False)
    return guard


def native_dispatch(monkeypatch, guard, tool, args):
    from hermes_cli import plugins
    manager = plugins.PluginManager()
    manager._hooks['pre_tool_call'] = [guard.pre_tool_call]
    monkeypatch.setattr(plugins, 'get_plugin_manager', lambda *a, **kw: manager)
    monkeypatch.setattr(plugins, '_resolve_hook_callback_timeout', lambda: 0)
    executed = []
    action, message = plugins.get_pre_tool_call_directive(tool, args, session_id='owner', tool_call_id='call')
    if action != 'block':
        executed.append((tool, args))
    return action, message, executed


@pytest.mark.parametrize('failure', ['owner', 'reference', 'named_reference'])
def test_secrets_guard_error_blocks_on_l_and_s(monkeypatch, secrets_guard, failure, caplog):
    def broken(*a, **kw):
        raise RuntimeError('synthetic private value must not appear')
    args = {'command': 'echo [secret S-AAAAAAAA: password]'}
    if failure == 'owner':
        monkeypatch.setattr(secrets_guard, 'owner_session', broken)
    elif failure == 'reference':
        monkeypatch.setattr(secret_module, 'reference_ids', broken)
    else:
        args = {'command': 'echo "$COLONYMIND_SECRET_S_AAAAAAAA"'}
        monkeypatch.setattr(secret_module, 'named_refs', broken)
    action, message, executed = native_dispatch(monkeypatch, secrets_guard, 'terminal', args)
    assert action == 'block' and not executed
    assert 'private value authorization is unavailable' in message
    assert 'synthetic private value' not in caplog.text + message


def test_secret_middleware_failure_leaves_reference_blocked(monkeypatch, secrets_guard):
    def broken(*a):
        raise RuntimeError('resolver unavailable')
    monkeypatch.setattr(secrets_guard, '_tool_request', broken)
    args = {'command': 'echo [secret S-AAAAAAAA: password]'}
    assert secrets_guard.tool_request('terminal', args, 'owner') is None
    action, _, executed = native_dispatch(monkeypatch, secrets_guard, 'terminal', args)
    assert action == 'block' and not executed


def test_authorized_secret_call_keeps_existing_behavior(monkeypatch, secrets_guard):
    program = 'echo "$COLONYMIND_SECRET_S_AAAAAAAA"'
    secrets_guard._bound[('owner', program)] = 1
    action, _, executed = native_dispatch(monkeypatch, secrets_guard, 'terminal', {'command': program})
    assert action is None and len(executed) == 1
    action, _, executed = native_dispatch(monkeypatch, secrets_guard, 'web_search', {'query': 'safe words'})
    assert action is None and len(executed) == 1


def test_partial_send_is_recorded_without_duplicate_retry(monkeypatch, tmp_path):
    from tools import send_message_tool as native
    detail = {'success': False, 'partial_success': True, 'message_id': 'sent-1',
              'error': 'Delivery incomplete: one attachment dropped', 'media_dropped': ['fixture.png']}
    calls = []
    monkeypatch.setattr(native, 'send_message_tool', lambda args: calls.append(args) or json.dumps(detail))
    outcome = send_message('telegram:fixture', 'words')
    assert outcome['result'] == 'sent' and outcome['detail'] == detail
    assert outcome['error'] == detail['error']
    assert Answers._succeeded(detail) is True
    assert len(calls) == 1
    calls.clear()
    reports = []
    message = {'id': 'notice', 'target': 'telegram:fixture', 'text': 'words'}
    client = NS(get=lambda *a, **kw: NS(status_code=200, is_success=True, json=lambda: {'messages': [message]}),
                post=lambda path, **kw: reports.append((path, kw['json'])) or NS(status_code=200, is_success=True))
    body = Body(client, None, None, None, ledger=BodyLedger(tmp_path / 'body.sqlite3'))
    assert body.send_outbox() == 1
    assert body.send_outbox() == 0
    assert len(calls) == 1
    assert reports[-1][1]['result'] == 'sent' and reports[-1][1]['error'] == detail['error']
    assert body.ledger.send('notice')['state'] == 'sent'


@pytest.mark.parametrize('detail', [{'success': False, 'error': 'unknown target'},
                                   {'success': False, 'partial_success': False, 'error': 'unknown target'}])
def test_failed_send_stays_failed(monkeypatch, detail):
    from tools import send_message_tool as native
    monkeypatch.setattr(native, 'send_message_tool', lambda args: detail)
    assert send_message('telegram:fixture', 'words')['result'] == 'failed'
    assert Answers._succeeded(detail) is False


def test_legacy_resnap_remains_effect_free():
    assert effect_of('cronjob_manage', {'action': 'resnap'}) is None
