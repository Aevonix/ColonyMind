"""Owner Ideas HTTP -> real durable queue/claim/plan/bind -> native dispatch gate."""
import importlib
import importlib.util
from pathlib import Path

import pytest

from hands_fake_hermes import Parent
from test_hands_definition_dispatch import setup, create  # noqa: F401
from test_hands_definition_pickup import pickup

pytestmark = pytest.mark.requires_sidecar


@pytest.fixture
def ideas_env(tmp_path, monkeypatch, setup):
    spec = importlib.util.spec_from_file_location('ideas_review_fixtures',
        Path(__file__).resolve().parents[2] / 'sidecar/tests/test_grounded_ideas.py')
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    state = tmp_path / 'ideas'
    state.mkdir()
    generator = fixtures.env.__wrapped__(state, monkeypatch)
    env = next(generator)
    try:
        yield fixtures, env
    finally:
        generator.close()


@pytest.mark.parametrize('source', ['mail', 'calendar'])
def test_actual_ideas_build_dispatches_scoped_google_reads_and_refuses_effects(setup, ideas_env, source):
    from colonymind.mind.ideas_build import HandsQueue
    runtime, native, client, ctl, http, package = setup
    fixtures, env = ideas_env
    reads = ['connectors__google__gmail_get', 'connectors__google__calendar_events']
    refused_tools = ['connectors__google__gmail_send', 'connectors__google__calendar_update',
              'connectors__google__gmail_create_draft', 'terminal', 'execute_code', 'send_message', 'read_file', 'colonymind_memory_search']
    definition = create(setup, tools=reads + refused_tools, connectors=['google'])
    env.ideas.config['definition_id'] = definition['id']
    env.ideas.queue = HandsQueue()
    http.app.include_router(fixtures.host.router)
    records = fixtures.mailbox()[:1] if source == 'mail' else fixtures.mailbox()[-1:]
    row, = http.portal.call(fixtures.generate, env, records)
    response = http.post('/v1/host/initiatives/' + row.id + '/execute', json={'mode': 'full'})
    assert response.status_code == 200, response.text
    task_id = env.store.get(row.id).context['idea']['execution']['taskId']
    snapshot = http.portal.call(ctl.task_definition, task_id)
    parent = Parent()
    runtime.remember_parent(parent)
    pickup(runtime)  # Real automatic pickup, with no kind override.
    assert len(native.calls) == 1
    hand, = runtime.registry.all()
    allow = importlib.import_module(package.__name__ + '.allow')
    tool = reads[0] if source == 'mail' else reads[1]
    args = ({'account': fixtures.ACCOUNT['account'], 'message_id': 'm-invoice'} if source == 'mail' else
            {'account': fixtures.ACCOUNT['account'], 'calendar_id': 'primary', 'time_min': '2026-10-19T15:00:00+00:00',
             'time_max': '2026-10-20T15:00:00+00:00', 'single_events': True, 'page_size': 16})
    assert allow.check_tool(hand, tool, args) is None
    assert hand.kind == 'ops'
    assert snapshot['source_reads'] == hand.plan['definition_policy']['source_reads']
    assert snapshot['source_reads'][0]['account'] == fixtures.ACCOUNT['account']
    for call in ({**args, 'account': 'another-account'}, {k: v for k, v in args.items() if k != 'account'},
                 {**args, 'message_id': 'unrelated-message'} if source == 'mail' else {**args, 'calendar_id': 'other'},
                 {**args, 'query': 'all private notes'}):
        assert allow.check_tool(hand, tool, call)
    for tool_name in refused_tools:
        assert allow.check_tool(hand, tool_name, args), tool_name
    assert allow.check_tool(hand, reads[1] if source == 'mail' else reads[0], args)
    # Recompiling the immutable snapshot preserves the source restriction.
    recovered = importlib.import_module(package.__name__ + '.definitions').compile_policy(snapshot)
    assert recovered == hand.plan['definition_policy']


def test_ordinary_deep_still_refuses_connector_even_when_definition_allows_it(setup):
    from types import SimpleNamespace
    package = setup[-1]
    allow = importlib.import_module(package.__name__ + '.allow')
    hand = SimpleNamespace(blocked=False, activity='', status='running', kind='deep',
        plan={'definition_policy': {'tools': ['connectors__google__gmail_get'], 'connectors': ['google']}})
    assert allow.check_tool(hand, 'connectors__google__gmail_get', {'account': 'owner', 'message_id': 'original'})


def test_frozen_reads_use_native_mcp_identity_and_refuse_ambiguous_servers(setup, monkeypatch):
    from types import SimpleNamespace
    from tools.registry import registry
    package = setup[-1]
    definitions = importlib.import_module(package.__name__ + '.definitions')
    allow = importlib.import_module(package.__name__ + '.allow')
    tool = 'mcp__owner_google__gmail_get'
    read = {'server': 'owner_google', 'account': 'selected-account', 'tool': 'gmail_get', 'provider_id': 'original'}
    hand = SimpleNamespace(blocked=False, activity='', status='running', kind='ops',
        plan={'definition_policy': {'tools': [tool], 'connectors': ['owner_google'], 'source_reads': [read]}})
    monkeypatch.setattr(registry, 'get_entry', lambda name: SimpleNamespace(toolset='mcp-owner_google') if name == tool else None)
    args = {'account': 'selected-account', 'message_id': 'original', 'format': 'full'}
    assert allow.check_tool(hand, tool, args) is None
    assert definitions.check_source_read([read], tool, {**args, 'message_id': 'private-notes'})
    monkeypatch.setattr(registry, 'get_entry', lambda name: None)
    assert allow.check_tool(hand, tool, args)  # No guessed MCP server identity.


@pytest.mark.parametrize('changes', [
    {'time_max': '2026-10-21T15:00:00Z'}, {'time_min': '2026-10-19T14:59:00Z'},
    {'single_events': False}, {'page_size': 100}, {'event_id': 'another-event'},
])
def test_calendar_scope_cannot_expand_window_or_arguments(setup, changes):
    package = setup[-1]
    definitions = importlib.import_module(package.__name__ + '.definitions')
    reads = [{'server': 'google', 'account': 'selected', 'tool': 'calendar_events', 'provider_id': 'event',
              'calendar_id': 'primary', 'time_min': '2026-10-19T15:00:00Z', 'time_max': '2026-10-20T15:00:00Z'}]
    args = {'account': 'selected', 'calendar_id': 'primary', 'time_min': '2026-10-19T15:00:00Z',
            'time_max': '2026-10-20T15:00:00Z', 'single_events': True, 'page_size': 16}
    assert definitions.check_source_read(reads, 'connectors__google__calendar_events', args) is None
    assert definitions.check_source_read(reads, 'connectors__google__calendar_events', {**args, **changes})


def test_unscoped_queue_preserves_preexisting_brief_hash(setup):
    import hashlib
    import json
    runtime, native, client, ctl, http, package = setup
    definition = create(setup)
    body = {'brief': 'Existing owner task'}
    task = client.post('/v1/hands/definitions/' + definition['id'] + '/tasks', body, 'preexisting-body')
    record = http.portal.call(ctl.one, 'SELECT request_hash FROM idempotency WHERE key=?', ('preexisting-body',))
    old_hash = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
    assert record['request_hash'] == old_hash
    assert client.post('/v1/hands/definitions/' + definition['id'] + '/tasks', body, 'preexisting-body') == task
    assert http.portal.call(ctl.get_task, task['id'])['kind'] == 'deep'
