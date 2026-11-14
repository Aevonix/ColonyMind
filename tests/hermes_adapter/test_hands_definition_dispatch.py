"""Named Hands use the real sidecar controller and existing detached executor."""
import importlib
import json
from types import SimpleNamespace

import pytest

from hands_fake_hermes import FakeHermes, Parent, load_hands

pytestmark = pytest.mark.requires_sidecar


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from colonymind.api.middleware import ApiKeyMiddleware
    from colonymind.api.routers import hands
    from colonymind.hands import controller
    from colonymind.events import broadcaster
    from colonymind.api.routers.host import broadcast_event
    monkeypatch.setenv('COLONYMIND_EVENT_JOURNAL_DIR', str(tmp_path / 'events'))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'hermes'))
    monkeypatch.setenv('COLONYMIND_HOME', str(tmp_path / 'instance'))
    monkeypatch.setattr(broadcaster, '_broadcast_fn', broadcast_event)
    config = {'max_running': 3, 'platforms': ['sms'], 'contracts': {'text-route': {
        'version': '1', 'family': 'text', 'modalities': ['text'], 'context_tokens': 100000,
        'max_output_tokens': 2000, 'tool_calls': {'qualified': True, 'success_share': 1.0}, 'decode_tok_s': 100,
        'json': True, 'effort': {'2': {'reasoning_effort': 'high'}},
        'admission': {'family': 'text', 'priority': 10}}},
        'routes': {role: ['text-route'] for role in ('deep', 'coding', 'verify', 'media')},
        'health': {'running_limits': {'text': 6}}}
    app = FastAPI()
    app.include_router(hands.runtime_router)
    app.include_router(hands.router)
    app.add_middleware(ApiKeyMiddleware, api_key='test-instance-key')
    with TestClient(app, headers={'Authorization': 'Bearer test-instance-key'}) as http:
        async def connect():
            c = controller.Controller(tmp_path / 'hands.db', config=config)
            await c.connect()
            c.health.record('text', running=0, waiting=0, kv=0.1)
            return c
        c = http.portal.call(connect)
        app.state.hands_store = c.store
        async def current():
            return c
        monkeypatch.setattr(controller, 'get_controller', current)
        class Client:
            def get(self, path, params=None):
                response = http.get(path, params=params)
                response.raise_for_status()
                return response.json()
            def post(self, path, body, key):
                response = http.post(path, json=body, headers={'Idempotency-Key': key})
                response.raise_for_status()
                return response.json()
        client = Client()
        package, native = load_hands(), FakeHermes()
        dispatch = importlib.import_module(package.__name__ + '.dispatch')
        runtime = dispatch.Runtime(config=config, client=client, spool_dir=tmp_path / 'spool',
                                   native=native, owner_check=lambda sid: sid.startswith('owner'))
        native.runtime = runtime
        runtime.install()
        package.register_route_resolver(lambda role, effort: native.native_policy())
        try:
            yield runtime, native, client, c, http, package
        finally:
            runtime.close()
            http.portal.call(c.close)


def create(setup, **fields):
    return setup[2].post('/v1/hands/definitions', {'name': 'Source reviewer', **fields}, 'create')


@pytest.mark.parametrize('by_name', [False, True])
def test_delegate_named_hand_through_wrapper_and_policy(setup, by_name):
    runtime, native, client, c, http, package = setup
    definition = create(setup, role='Research', instructions='SNAPSHOT-INSTRUCTIONS', tools=['read_file', 'terminal'])
    selector = definition['name'] if by_name else definition['id']
    answer = json.loads(native.delegate_module.delegate_task(tasks=[{'goal': 'Inspect', 'hand': selector}], parent_agent=Parent()))
    assert answer['hands'][0]['status'] == 'dispatched'
    assert len(native.calls) == 1 and native.calls[0]['background'] is True and not native.inline
    hand, = runtime.registry.all()
    assert hand.plan['definition_snapshot']['id'] == definition['id']
    assert 'SNAPSHOT-INSTRUCTIONS' in native.calls[0]['tasks'][0]['context']
    allow = importlib.import_module(package.__name__ + '.allow')
    assert allow.check_tool(hand, 'read_file', {}) is None
    assert allow.check_tool(hand, 'terminal', {})  # definition cannot widen kind=deep
    assert allow.check_tool(hand, 'search_files', {})  # definition narrows kind
    task = client.get('/v1/hands/tasks/' + hand.task_id)
    assert task['definition_id'] == definition['id'] and task['definition_version'] == 1
    assert not http.portal.call(c.all, 'SELECT * FROM permits')


@pytest.mark.parametrize('state', ['unknown', 'disabled', 'archived'])
def test_invalid_definition_never_dispatches(setup, state):
    runtime, native, client, c, http, _ = setup
    row = create(setup, enabled=state != 'disabled')
    if state == 'archived':
        assert http.delete('/v1/hands/definitions/' + row['id'], headers={'Idempotency-Key': 'archive'}).status_code == 200
    answer = runtime.dispatch_tasks([{'goal': 'Inspect', 'hand': 'missing' if state == 'unknown' else row['id']}], parent=Parent())
    refusal = answer['hands'][0]
    assert refusal['status'] == 'refused' and refusal['detail'] == state + '_hand_definition'
    assert refusal['hint'] and not native.calls and not native.inline
    assert not http.portal.call(c.all, 'SELECT * FROM hands')
    assert not http.portal.call(c.all, 'SELECT * FROM tasks')


def test_list_and_queued_task_use_live_parent_and_verified_return(setup):
    runtime, native, client, c, http, package = setup
    definition = create(setup, tools=['read_file'])
    tools = importlib.import_module(package.__name__ + '.tools')
    listed = tools.execute_hands({'action': 'list_hands'}, runtime=runtime, session_id='owner-session')
    assert listed == {'hands': [{key: definition[key] for key in ('id', 'name', 'role')}]}
    queued = client.post('/v1/hands/definitions/' + definition['id'] + '/tasks', {'brief': 'Inspect sources'}, 'queue')
    assert queued['dispatch_state'] == 'queued' and not native.calls
    request = {'action': 'dispatch', 'task_id': queued['id'], 'kind': 'research'}
    assert 'error' in tools.execute_hands(request, runtime=runtime, session_id='owner-session')
    parent = Parent()
    runtime.remember_parent(parent)
    assert tools.execute_hands({'action': 'pending'}, runtime=runtime, session_id=parent.session_id)['tasks']
    answer = tools.execute_hands(request, runtime=runtime, session_id=parent.session_id)
    assert answer['hands'][0]['status'] == 'dispatched' and len(native.calls) == 1
    hand, = runtime.registry.all()
    assert hand.task_id == queued['id'] and hand.parent_session_id == parent.session_id
    assert hand.kind == 'research'


def test_queued_task_without_detached_return_never_executes(setup):
    runtime, native, client, c, http, package = setup
    definition = create(setup)
    queued = client.post('/v1/hands/definitions/' + definition['id'] + '/tasks', {'brief': 'Inspect'}, 'queue')
    parent = Parent()
    runtime.remember_parent(parent)
    native.consumer = False
    definitions = importlib.import_module(package.__name__ + '.definitions')
    answer = definitions.dispatch_request(runtime, queued['id'], parent.session_id)
    assert answer['hands'][0]['reason'] == 'no_async' and not native.calls


def test_definition_policy_survives_restart(setup):
    runtime, native, client, c, http, package = setup
    row = create(setup, tools=['read_file'])
    runtime.dispatch_tasks([{'goal': 'Inspect', 'hand': row['id']}], parent=Parent())
    original, = runtime.registry.all()
    runtime.close()
    restored = type(runtime)(config=runtime.config, client=client, spool_dir=runtime.spool_dir,
                             native=native, owner_check=lambda sid: sid.startswith('owner'))
    try:
        hand, = restored.registry.all()
        assert hand.plan['definition_policy'] == original.plan['definition_policy']
    finally:
        restored.close()


@pytest.mark.parametrize('kind', ['deep', 'research', 'ops', 'coding', 'media'])
def test_empty_policy_removes_all_tools_and_cannot_grant_blocked_tools(kind):
    package = load_hands()
    allow = importlib.import_module(package.__name__ + '.allow')
    hand = SimpleNamespace(blocked=False, activity=None, status='running', kind=kind,
                           plan={'definition_policy': {'tools': [], 'connectors': []}})
    for tool in ('read_file', 'terminal', 'web_search', 'send_message'):
        assert allow.check_tool(hand, tool, {})
    hand.plan['definition_policy']['tools'] = ['send_message']
    assert allow.check_tool(hand, 'send_message', {})


def test_toolsets_unknown_names_connectors_and_bridge_restrictions():
    package = load_hands()
    definitions = importlib.import_module(package.__name__ + '.definitions')
    policy = definitions.compile_policy({'tools': ['web', 'future_tool'], 'connectors': ['calendar']})
    assert {'web_search', 'web_extract', 'future_tool'} <= set(policy['tools'])
    policy['tools'] += ['connectors__calendar__read', 'connectors__mail__read', 'tool_call', 'execute_code']
    hand = SimpleNamespace(plan={'definition_policy': policy})
    assert definitions.check_policy(hand, 'connectors__calendar__read', {}) is None
    assert definitions.check_policy(hand, 'connectors__mail__read', {})
    assert definitions.check_policy(hand, 'tool_call', {'calls': [{'name': 'send_message'}]})
    assert definitions.check_policy(hand, 'execute_code', {})


def test_named_hand_never_uses_degraded_dispatch(setup, monkeypatch):
    runtime, native, client, *_ = setup
    create(setup)
    runtime.degraded.failure()
    runtime.degraded.failure()
    def unavailable(*args, **kwargs):
        raise OSError('controller unavailable')
    monkeypatch.setattr(client, 'post', unavailable)
    answer = runtime.dispatch_tasks([{'goal': 'Inspect', 'hand': 'Source reviewer'}], parent=Parent())
    assert answer['hands'][0]['reason'] == 'definition_unavailable' and not native.calls
