"""Schema verdicts survive the native result builder and public stop dispatcher."""
from concurrent.futures import ThreadPoolExecutor
import importlib
import hashlib
import json
from types import SimpleNamespace

import pytest

from hands_fake_hermes import Parent
from hermes_source import HERMES_ROOT, assert_source_module, assert_supported_source
from test_hands_dispatch import setup


@pytest.fixture
def native_results(setup, monkeypatch):
    revision = assert_supported_source()
    monkeypatch.syspath_prepend(str(HERMES_ROOT))
    child_run = importlib.import_module('tools.delegate_tool_child_run')
    results = importlib.import_module('tools.delegate_tool_results')
    plugins = importlib.import_module('hermes_cli.plugins')
    assert_source_module(child_run)
    runtime, native, client = setup
    native.results_module = results
    runtime.install()
    hooks = []

    def public_stop(name, **kwargs):
        assert name == 'subagent_stop'
        assert 'schema_valid' not in kwargs
        hooks.append(kwargs)
        # S may dispatch callbacks on a worker with no originating ContextVar.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(runtime.subagent_stop, **kwargs).result(timeout=2)

    monkeypatch.setattr(plugins, 'invoke_hook', public_stop)

    def spawn(valid=False, parent=None, ident='call-1', errors=None):
        parent = parent or Parent()
        task = {'goal':'check the contract'}
        if valid is not None:
            task['output_schema'] = {'type':'object', 'required':['answer']}
        answer = runtime.dispatch_tasks([task], parent=parent, tool_call_id=ident)
        hand = runtime.registry.get(answer['hands'][0]['hand_id'])
        child = SimpleNamespace(session_id=hand.child_session_id)
        schema = child_run._SchemaOutcome(task.get('output_schema'), valid,
            errors if errors is not None else ['answer is required'] if valid is False else [], 1)
        entry = child_run._build_result_entry(child,
            {'completed':True, 'final_response':'{"other":"retained"}'}, 0, 1.0, schema)
        return parent, hand, child, entry

    return SimpleNamespace(runtime=runtime, native=native, client=client, hooks=hooks,
                           module=results, spawn=spawn, revision=revision)


def recorded(env):
    return [body for path, body, _ in env.client.writes if path.endswith('/result')]


def test_invalid_schema_native_result_never_reports_done(native_results):
    env = native_results
    parent, hand, child, entry = env.spawn(False)
    assert entry['status'] == ('failed' if env.revision not in {'S', 'S217'} else 'completed')
    summary = entry['summary']
    env.module._fire_subagent_stop_hooks([entry], {0:child}, parent)
    result, = recorded(env)
    assert result['status'] == hand.status == 'failed'
    assert result['summary_text'] == summary and 'retained' in summary
    evidence = [e['payload'] for _, body, _ in env.client.writes
                for e in body.get('events', []) if e['type'] == 'schema_validation']
    assert evidence == [{'schema_valid':False, 'schema_errors':['answer is required'], 'schema_retries':1}]
    assert not env.runtime._schema_verdicts


def test_schema_verdict_is_available_before_stop_callback(native_results, monkeypatch):
    env = native_results
    parent, hand, child, entry = env.spawn(False)
    original = env.runtime.subagent_stop

    def observe(**kwargs):
        key = (child.session_id, parent.session_id, parent._current_turn_id)
        assert env.runtime._schema_verdicts[key][0] is hand
        assert env.runtime._schema_verdicts[key][1]['schema_valid'] is False
        return original(**kwargs)

    monkeypatch.setattr(env.runtime, 'subagent_stop', observe)
    env.module._fire_subagent_stop_hooks([entry], {0:child}, parent)
    assert len(env.hooks) == len(recorded(env)) == 1


def test_schema_verdict_does_not_cross_children_or_attempts(native_results):
    env = native_results
    first = env.spawn(False, Parent('owner-one'), 'first')
    next_parent = Parent('owner-two')
    next_parent._current_turn_id = 'turn-2'
    second = env.spawn(True, next_parent, 'second')
    parent1, hand1, child1, entry1 = first
    parent2, hand2, child2, entry2 = second
    env.runtime.capture_schema_verdicts([entry1], {0:child1}, parent1)
    env.runtime.subagent_stop(child_session_id=child1.session_id, child_status='completed',
                             parent_session_id=parent1.session_id, parent_turn_id='wrong-turn')
    assert not recorded(env)
    env.module._fire_subagent_stop_hooks([entry2], {0:child2}, parent2)
    env.module._fire_subagent_stop_hooks([entry1], {0:child1}, parent1)
    env.module._fire_subagent_stop_hooks([entry1], {0:child1}, parent1)
    assert [result['status'] for result in recorded(env)] == ['done','failed']
    assert hand1.status == 'failed' and hand2.status == 'done'
    assert not env.runtime._schema_verdicts


@pytest.mark.parametrize('valid', [True, None])
def test_valid_and_unconstrained_children_keep_success(native_results, valid):
    env = native_results
    parent, hand, child, entry = env.spawn(valid)
    env.module._fire_subagent_stop_hooks([entry], {0:child}, parent)
    assert recorded(env)[0]['status'] == 'done'
    assert hand.status == 'done'


def test_uncorrelated_required_verdict_stays_unresolved(native_results):
    env = native_results
    parent, hand, child, entry = env.spawn(True)
    del entry['schema_valid']
    env.module._fire_subagent_stop_hooks([entry], {0:child}, parent)
    assert not recorded(env)
    assert hand.status == 'lost'
    assert len(env.native.calls) == 1


def test_native_failure_wins_over_schema_and_waiting_state(native_results):
    env = native_results
    parent, hand, child, entry = env.spawn(True)
    entry['status'] = 'failed'
    hand.activity = 'awaiting_input'
    env.module._fire_subagent_stop_hooks([entry], {0:child}, parent)
    assert recorded(env)[0]['status'] == 'failed'


def test_schema_wrapper_unload_restores_only_its_own_wrapper(native_results):
    env = native_results
    original = env.runtime._wrapped_results[1]
    own = env.module._fire_subagent_stop_hooks
    env.runtime.install()
    assert env.module._fire_subagent_stop_hooks is own
    env.module._fire_subagent_stop_hooks = original
    env.runtime.close()
    assert env.module._fire_subagent_stop_hooks is original


@pytest.mark.parametrize('valid', [True, False, None])
@pytest.mark.parametrize('rotate', [False, True])
def test_detached_schema_result_survives_parent_advancing(native_results, valid, rotate):
    env = native_results
    parent, hand, child, entry = env.spawn(valid)
    original_session = parent.session_id
    parent._current_turn_id = 'next-owner-turn'
    if rotate:
        parent.session_id = 'owner-rotated'
    env.module._fire_subagent_stop_hooks([entry], {0:child}, parent)
    result, = recorded(env)
    assert result['status'] == ('failed' if valid is False else 'done')
    assert hand.parent_session_id == original_session
    assert hand.plan['parent_turn_id'] == 'turn-1'
    assert not env.runtime._schema_verdicts


def test_captured_schema_cannot_finalize_a_rebound_attempt(native_results):
    env = native_results
    parent, first, child, entry = env.spawn(False)
    env.runtime.capture_schema_verdicts([entry], {0:child}, parent)
    env.runtime.registry.remove(first.hand_id)
    _, second, next_child, next_entry = env.spawn(None, parent, 'next-attempt')
    env.runtime.registry.bind(second, child.session_id)
    env.runtime.subagent_stop(child_session_id=child.session_id, child_status=entry['status'],
        parent_session_id=parent.session_id, parent_turn_id=parent._current_turn_id)
    assert not recorded(env)
    assert not second.plan.get('_result_submitted')
    env.module._fire_subagent_stop_hooks([next_entry], {0:next_child}, parent)
    assert len(recorded(env)) == 1
    assert recorded(env)[0]['status'] == 'done'


@pytest.mark.requires_sidecar
def test_large_schema_evidence_fits_controller_and_keeps_full_private_record(native_results):
    from colonymind.hands.controller import Controller
    env = native_results
    errors = [f'$.field{n}: ' + '\U0001f642' * 500 for n in range(10)]
    parent, hand, child, entry = env.spawn(False, errors=errors)
    env.runtime.effects.intent(hand.hand_id, 'external-write', 'terminal', {'command':'fixture effect'})
    env.module._fire_subagent_stop_hooks([entry], {0:child}, parent)
    result, = recorded(env)
    assert result['status'] == 'failed'
    assert len(env.native.calls) == 1
    assert env.runtime.continue_hand(hand).get('error') == 'reconciliation required'
    assert len(env.native.calls) == 1
    module = importlib.import_module(env.runtime.__module__.rsplit('.', 1)[0] + '.spool')
    reloaded = module.Spool(env.runtime.spool_dir)
    row, = [row for row in reloaded.records() if any(
        e['type'] == 'schema_validation' for e in row['body'].get('events', []))]
    full = row['metadata']['schema_verdict']
    assert full['schema_errors'] == errors
    payload = row['body']['events'][0]['payload']
    assert payload['schema_error_count'] == len(errors)
    assert payload['schema_errors'][0].startswith('$.field0:')
    assert payload['schema_evidence_hash'] == hashlib.sha256(
        json.dumps(full, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    inserted = []
    async def insert(*args, **kwargs):
        inserted.append(kwargs)
    async def update(*args, **kwargs):
        pass
    # This native controller method only awaits our in-memory storage calls,
    # so it can enforce its real payload cap without a socket loop.
    sink = SimpleNamespace(insert=insert, update=update, now=lambda:'2026-01-01T00:00:00Z')
    pending = Controller.log(sink, hand.hand_id, 'schema_validation', payload,
                             seq=1, at='2026-01-01T00:00:00Z')
    with pytest.raises(StopIteration):
        pending.send(None)
    assert len(inserted) == 1
    assert inserted[0]['payload']['schema_valid'] is False


def local_held_result(env, hand, child):
    """Let the selected native validator retry the owned terminal response."""
    child_run = importlib.import_module('tools.delegate_tool_child_run')
    middleware = importlib.import_module(env.runtime.__module__.rsplit('.', 1)[0] + '.middleware')
    hand.blocked, hand.activity = True, 'awaiting_approval'
    child._delegate_output_schema = hand.plan['output_schema']
    requests = []

    def local_response(*args, **kwargs):
        ids = dict(session_id=child.session_id, api_request_id=f'held-{len(requests)}',
                   provider='text-route', model='text-model', api_mode='chat_completions')
        requests.append(ids)
        request = middleware.llm_request(env.runtime, {'messages': [], 'tools': [{'name': 'terminal'}]}, **ids)
        response = middleware.llm_execution(env.runtime, request['request'],
            lambda payload: pytest.fail('a held child called its provider'), **ids)
        middleware.post_api_request(env.runtime, usage=None, **ids)
        assert response.usage is None and not env.runtime.registry.inflight
        return {'completed': True, 'final_response': response.choices[0].message.content}

    child.run_conversation = local_response
    result = local_response()
    schema = child_run._validate_child_output_schema(child, result, 0, hand.subagent_id, None)
    assert schema.valid is False and len(requests) == 2
    return child_run._build_result_entry(child, result, 0, 1.0, schema)


@pytest.mark.parametrize('restart', [False, True])
def test_declared_schema_hold_resumes_once_and_preserves_its_contract(native_results, restart, monkeypatch):
    env = native_results
    parent, hand, child, _ = env.spawn(False)
    original_schema = dict(hand.plan['output_schema'])
    entry = local_held_result(env, hand, child)
    # The public callback drops the schema fields. The owned wrapper must carry
    # both the schema verdict and the independently recorded local response.
    env.module._fire_subagent_stop_hooks([entry], {0: child}, parent)
    assert recorded(env)[0]['status'] == 'blocked'
    assert hand.plan['_result_submitted'] and not hand.tokens
    assert any(event['payload']['schema_valid'] is False for _, body, _ in env.client.writes
               for event in body.get('events', []) if event['type'] == 'schema_validation')
    env.runtime.emit(f'/v1/hands/{hand.hand_id}/hold', {'epoch': hand.epoch}, 'held-schema')
    env.native.live[hand.delegation_id]['status'] = 'completed'
    env.native.active_count = lambda: sum(row['status'] == 'running' for row in env.native.live.values())
    if restart:
        dispatch = importlib.import_module(env.runtime.__module__)
        env.runtime.close()
        env.runtime = dispatch.Runtime(config=env.runtime.config, client=env.client,
            spool_dir=env.runtime.spool_dir, native=env.native, owner_check=lambda sid: sid.startswith('owner'))
        env.native.runtime = env.runtime
        env.runtime.install()
        env.runtime.remember_parent(parent)
        hand = env.runtime.registry.get(hand.hand_id)
        assert hand.plan['output_schema'] == original_schema
        monkeypatch.setattr(importlib.import_module('hermes_cli.plugins'), 'invoke_hook',
                            lambda name, **kwargs: env.runtime.subagent_stop(**kwargs))
    grant = dict(permit_id='schema-permit', ask_id='schema-ask', hand_id=hand.hand_id,
                 parent_session=parent.session_id, state='granted', workdir='/fixture',
                 call={'tool': 'terminal', 'arguments': {'command': 'touch /fixture/report'}})
    original_get = env.client.get
    env.client.get = lambda path, params=None: ({'permits': [grant], 'refreshed': True}
        if path == '/v1/hands/permits/held' else original_get(path, params))
    try:
        env.runtime.resume_approvals(parent.session_id, [grant])
        env.runtime.maintain()
        resumed = env.runtime.registry.all()[-1]
        assert resumed is not hand and len(env.native.calls) == 2
        assert resumed.plan['output_schema'] == original_schema
        assert env.native.calls[-1]['tasks'][0]['output_schema'] == original_schema
        env.runtime.maintain()
        assert len(env.native.calls) == 2
        resumed_child = SimpleNamespace(session_id=resumed.child_session_id)
        invalid = dict(entry, summary='Invalid completed work')
        env.module._fire_subagent_stop_hooks([invalid], {0: resumed_child}, parent)
        assert recorded(env)[-1]['status'] == 'failed'
    finally:
        if restart:
            env.runtime.close()


@pytest.mark.parametrize('failure', ['error', 'interrupted', 'unverified'])
def test_local_hold_cannot_mask_native_failure_or_unverified_schema_result(native_results, failure):
    env = native_results
    parent, hand, child, entry = env.spawn(False)
    entry = local_held_result(env, hand, child)
    if failure == 'unverified':
        hand.local_requests.clear()
    else:
        entry.update(status='interrupted' if failure == 'interrupted' else 'failed', exit_reason=failure)
    env.module._fire_subagent_stop_hooks([entry], {0: child}, parent)
    assert recorded(env)[0]['status'] == ('stopped' if failure == 'interrupted' else 'failed')


def test_manual_continuation_retains_declared_schema(native_results):
    env = native_results
    parent, hand, child, entry = env.spawn(True)
    env.module._fire_subagent_stop_hooks([entry], {0: child}, parent)
    env.native.live[hand.delegation_id]['status'] = 'completed'
    env.native.active_count = lambda: 0
    env.runtime.continue_hand(hand, 'Please finish the result')
    assert env.native.calls[-1]['tasks'][0]['output_schema'] == hand.plan['output_schema']
