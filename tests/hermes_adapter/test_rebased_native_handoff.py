"""Hands admission against real Hermes modules; the child-launch boundary is inert."""
from functools import wraps
import importlib
import json
import sys

from gateway.session_context import clear_session_vars, set_session_vars
from hands_fake_hermes import FakeController, Parent, load_hands


def test_handoff_uses_native_context_and_records_dispatch_before_child_binding(tmp_path, monkeypatch):
    package = load_hands()
    dispatch = importlib.import_module(package.__name__ + '.dispatch')
    native = dispatch.NativeHermes()
    client, calls = FakeController(), []
    policy = {'provider': 'text-route', 'model': 'test-model', 'fallback_providers': []}
    monkeypatch.setattr(native.policy, '_load_config', lambda: policy)
    monkeypatch.setattr(native.policy, '_get_max_async_children', lambda: 4)
    monkeypatch.delitem(sys.modules, 'colonymind_hands_front', raising=False)
    original = native.delegate_module.delegate_task

    @wraps(original)
    def launch(*args, **kwargs):
        assert kwargs['background'] is True
        assert len(kwargs['tasks']) == 1 and 'kind' not in kwargs['tasks'][0]
        assert kwargs['credentials_cfg']['provider'] == 'text-route'
        assert any(row['body'].get('events', [{}])[0].get('type') == 'dispatch_intent'
                   for row in runtime.spool.records() if row['body'].get('events'))
        calls.append(kwargs)
        runtime.subagent_start(parent_session_id='owner-session', child_session_id='test-child',
                               child_subagent_id='test-subagent')
        return json.dumps({'status': 'dispatched', 'delegation_id': 'test-delegation'})

    monkeypatch.setattr(native.delegate_module, 'delegate_task', launch)
    runtime = dispatch.Runtime(config={'platforms': ['sms'], 'max_running': 1, 'contracts': {
        'text-route': {'version': '1', 'family': 'text', 'modalities': ['text'],
                       'context_tokens': 100000, 'max_output_tokens': 2000,
                       'tool_calls': True, 'decode_tok_s': 100}}},
        client=client, spool_dir=tmp_path / 'spool', native=native,
        owner_check=lambda sid: sid == 'owner-session')
    runtime.install()
    package.register_route_resolver(lambda role, effort: native.native_policy())
    tokens = set_session_vars(platform='sms', session_id='owner-session', async_delivery=True)
    try:
        result = json.loads(native.delegate_module.delegate_task(
            tasks=[{'goal': 'Stage a local result', 'kind': 'research'}], parent_agent=Parent()))
        assert result['hands'][0]['status'] == 'dispatched', result
        assert len(calls) == 1
        hand = runtime.registry.all()[0]
        assert (hand.delegation_id, hand.child_session_id) == ('test-delegation', 'test-child')
        paths = [path.rsplit('/', 1)[-1] for path, _, _ in client.writes]
        assert paths.index('dispatched') < paths.index('bind')
    finally:
        clear_session_vars(tokens)
        runtime.close()
