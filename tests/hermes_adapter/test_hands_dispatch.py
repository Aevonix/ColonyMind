"""OC2 admission and ownership acceptance tests, without sockets."""
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import threading

import pytest
from hands_fake_hermes import FakeController, FakeHermes, Parent, load_hands


@pytest.fixture
def setup(tmp_path):
    package = load_hands()
    dispatch = importlib.import_module(package.__name__ + '.dispatch')
    native, client = FakeHermes(), FakeController()
    config = {'max_running':3, 'platforms':['sms'], 'contracts': {'text-route': {
        'version':'1','family':'text','modalities':['text'],'context_tokens':100000,
        'max_output_tokens':2000,'tool_calls':True,'decode_tok_s':100}}}
    runtime = dispatch.Runtime(config=config, client=client, spool_dir=tmp_path/'spool',
                               native=native, owner_check=lambda sid:sid.startswith('owner'))
    native.runtime = runtime
    runtime.install()
    package.register_route_resolver(lambda role,effort:native.native_policy())
    yield runtime, native, client
    runtime.close()


def test_batch_is_three_calls_hands_and_names(setup):
    runtime, native, client = setup
    result = json.loads(native.delegate_module.delegate_task(tasks=[{'goal':str(i),'kind':'research'} for i in range(3)],parent_agent=Parent()))
    assert len(native.calls) == len(runtime.registry.all()) == 3
    assert len({row['name'] for row in result['hands']}) == 3
    assert all(len(c['tasks']) == 1 and c['background'] is True for c in native.calls)
    assert all('kind' not in c['tasks'][0] for c in native.calls)
    assert not native.inline
    paths = [path.rsplit('/',1)[-1] for path,_,_ in client.writes]
    assert paths.index('dispatched') < paths.index('bind')


@pytest.mark.parametrize('standing', ['owner', 'mind', 'system'])
def test_hand_retains_verified_parent_lane(setup, standing):
    from types import SimpleNamespace
    runtime, native, _ = setup
    inherited = []
    runtime.sessions = SimpleNamespace(standing=lambda sid: (standing, 'cli'),
        inherit=lambda child, parent, **kw: inherited.append(kw['standing']))
    native.delegate_module.delegate_task(goal='Inspect sources', parent_agent=Parent())
    hand, = runtime.registry.all()
    assert hand.plan['standing'] == standing
    assert inherited == [standing]
    plans = [row for row in runtime.spool.records() if row['path'] == '/v1/hands/plan']
    assert plans[0]['metadata']['standing'] == standing


@pytest.mark.parametrize('condition,reason', [('full','slots'),('consumer','no_async'),('platform','platform')])
def test_refusals_never_inline(setup,condition,reason):
    runtime,native,_ = setup
    if condition == 'full':
        for _ in range(3): native.cron()
    elif condition == 'consumer': native.consumer=False
    else: native.platform='api'
    result=json.loads(native.delegate_module.delegate_task(goal='check',parent_agent=Parent()))
    assert result['hands'][0]['reason'] == reason
    assert not native.calls and not native.inline


def test_cron_cannot_take_slot_between_check_and_admission(setup):
    runtime,native,_=setup
    native.capacity=2
    runtime.config['max_running']=1
    attempted, finished=threading.Event(),threading.Event()
    def cron():
        attempted.set()
        native.async_module.dispatch_async_delegation(goal='cron')
        finished.set()
    def race():
        thread=threading.Thread(target=cron)
        thread.start()
        assert attempted.wait(1)
        assert not finished.wait(.02)
    native.before_admission=race
    native.delegate_module.delegate_task(goal='check',parent_agent=Parent())
    assert finished.wait(1)
    assert len(native.calls)==1 and native.max_seen==2 and not native.inline


def test_four_dispatch_paths_and_cron_never_exceed_limit(setup):
    runtime,native,_=setup
    runtime.config['max_running']=2
    native.capacity=3
    def issue(i):
        parent=Parent('owner-' + str(i%2))
        if i%4 < 2:
            return runtime.dispatch_tasks([{'goal':str(i)}],parent=parent,tool_call_id=str(i))
        runtime.remember_parent(parent)
        return runtime.dispatch_tasks([{'goal':str(i)}],session_id=parent.session_id,tool_call_id=str(i))
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures=[pool.submit(issue,i) for i in range(50)]
        cron=pool.submit(native.async_module.dispatch_async_delegation,goal='cron')
        results=[f.result() for f in futures]
        cron.result()
    assert sum(h.status not in {'refused','lost','stopped','failed'} for h in runtime.registry.all())<=2
    assert native.max_seen<=3 and not native.inline
    assert sum(len(x['hands']) for x in results)==50


def test_repeated_plan_key_does_not_redispatch(setup):
    runtime,native,_=setup
    for _ in range(2):
        native.delegate_module.delegate_task(goal='check',parent_agent=Parent())
    assert len(native.calls)==1


def test_observer_failure_does_not_block_safe_tool(setup, monkeypatch):
    from test_step0_adapter import native_dispatch
    runtime, native, _ = setup
    class BrokenCallIds(dict):
        def __setitem__(self, key, value):
            raise RuntimeError('observer unavailable')
    runtime.call_ids = BrokenCallIds()
    monkeypatch.setattr(runtime, 'authorized_owner', lambda sid:True)
    action, _, executed = native_dispatch(monkeypatch, runtime, 'delegate_task', {'goal':'read safely'})
    assert action is None and len(executed) == 1


def test_registry_helpers_require_matching_owner(setup):
    runtime,native,_=setup
    native.delegate_module.delegate_task(goal='check',parent_agent=Parent())
    hand=runtime.registry.all()[0]
    with pytest.raises(PermissionError): runtime.stop(hand,'stop',calling_session='owner-other')
    with pytest.raises(PermissionError): runtime.steer(hand,'change',calling_session='owner-other')
    assert not native.stops and not native.steers
    runtime.stop(hand,'stop',calling_session='owner-session')
    assert len(native.stops)==1


def test_unexpected_inline_is_recorded_and_not_retried(setup):
    runtime,native,client=setup
    runtime.original_delegate=lambda **kwargs:json.dumps({'results':[{'summary':'kept'}]})
    result=runtime.dispatch_tasks([{'goal':'check'}],parent=Parent())
    assert result['hands'][0]['reason']=='ran_inline'
    assert any(e['type']=='ran_inline' for _,body,_ in client.writes for e in body.get('events',[]))


def test_provider_without_contract_never_dispatches(setup):
    runtime,native,_=setup
    runtime.config['contracts']={}
    result=runtime.dispatch_tasks([{'goal':'check'}],parent=Parent())
    assert result['hands'][0]['reason']=='contract'
    assert not native.calls


def test_pinned_stop_hook_preserves_failure_summary_and_actual_route(setup):
    import hashlib
    runtime,native,client=setup
    native.delegate_module.delegate_task(goal='check',parent_agent=Parent())
    hand=runtime.registry.all()[0]
    hand.families_used.add('text')
    hand.plan['contracts_used']=[{'provider':'text-route','version':'1','family':'text','model':'text-model'}]
    runtime.subagent_stop(child_session_id=hand.child_session_id,child_status='failed',child_summary='failed check')
    result=next(body for path,body,_ in client.writes if path.endswith('/result'))
    assert result['status']=='failed'
    assert result['summary_hash']==hashlib.sha256(b'failed check').hexdigest()
    assert result['contracts_used'][0]['family']=='text'
    assert hand.status=='failed'


@pytest.mark.requires_sidecar
def test_frozen_wire_accepts_every_dispatch_write(setup):
    pytest.importorskip('pydantic')
    from colonymind.hands import wire
    runtime,native,client=setup
    native.delegate_module.delegate_task(goal='check',parent_agent=Parent())
    hand=runtime.registry.all()[0]
    runtime.subagent_stop(child_session_id=hand.child_session_id,child_status='completed',child_summary='checked')
    schemas={'plan':wire.PlanRequest,'events':wire.EventsRequest,'dispatched':wire.DispatchedRequest,
             'bind':wire.BindRequest,'result':wire.ResultRequest}
    for path,body,key in client.writes:
        assert key
        schemas[path.rsplit('/',1)[-1]].model_validate(body)


def test_controller_queue_has_no_lease_to_release_and_never_executes(setup):
    runtime,native,client=setup
    original=client.post
    def queued(path,body,key):
        result=original(path,body,key)
        if path.endswith('/plan'):
            result['plans'][0].update(admit='queued',lease_token=None,reason='family')
        return result
    client.post=queued
    result=runtime.dispatch_tasks([{'goal':'check'}],parent=Parent())
    assert result['hands'][0]['status']=='queued'
    assert not native.calls
    assert not any(path.endswith('/release') for path,_,_ in client.writes)


def test_platform_capability_is_checked_before_every_plugin_dispatch(setup):
    runtime,native,_=setup
    parent=Parent()
    runtime.remember_parent(parent)
    native.consumer=False
    result=runtime.dispatch_tasks([{'goal':'continuation','continue':'h-'+'0'*12}],session_id=parent.session_id)
    assert result['hands'][0]['reason']=='no_async'
    assert not native.calls


def test_revised_attempt_uses_controller_hand_version(setup):
    runtime,native,client=setup
    original=client.get
    def revised(path,params=None):
        result=original(path,params)
        if path.startswith('/v1/hands/h-'): result['intent_version']=3
        return result
    client.get=revised
    runtime.dispatch_tasks([{'goal':'revised'}],parent=Parent())
    hand=runtime.registry.all()[0]
    runtime.subagent_stop(child_session_id=hand.child_session_id,child_status='completed',child_summary='updated')
    result=next(body for path,body,_ in client.writes if path.endswith('/result'))
    assert result['intent_version']==3


def test_restart_restores_attempt_and_repeated_plan_does_not_execute(setup):
    runtime,native,client=setup
    parent=Parent()
    runtime.dispatch_tasks([{'goal':'check'}],parent=parent)
    hand=runtime.registry.all()[0]
    runtime.close()
    restored=type(runtime)(config=runtime.config,client=client,spool_dir=runtime.spool_dir,
                           native=native,owner_check=lambda sid:sid.startswith('owner'))
    native.runtime=restored
    try:
        restored.install()
        saved=restored.registry.get(hand.hand_id)
        assert saved is not None and saved.plan['task']['goal']=='check'
        restored.dispatch_tasks([{'goal':'check'}],parent=parent)
        assert len(native.calls)==1
    finally:
        restored.close()


def test_authoritative_controller_refusal_never_enters_degraded_dispatch(setup):
    from types import SimpleNamespace
    runtime,native,client=setup
    class Refusal(Exception):
        response=SimpleNamespace(status_code=403)
    def refuse(*args,**kwargs): raise Refusal('refused')
    client.post=refuse
    parent=Parent()
    for ident in ('first','second'):
        result=runtime.dispatch_tasks([{'goal':'check'}],parent=parent,tool_call_id=ident)
        assert result['hands'][0]['reason']=='controller_refused'
    assert not native.calls and not runtime.degraded.is_degraded
    assert not runtime.spool.pending()


class QueueingController(FakeController):
    """The controller queues the next plan; its sweeper later reserves it with a lease."""
    def post(self, path, body, key):
        answer = super().post(path, body, key)
        if path.endswith('/plan') and self.queue_next:
            for plan in answer['plans']:
                plan.update(admit='queued', reason='capacity', lease_token=None)
                self.rows[plan['hand_id']]['state'] = 'planned'
        return answer

    def promote(self, hand_id, token='lease-promoted'):
        self.rows[hand_id].update(state='reserved', decisions={'admission': {'lease_token': token}})


def test_a_queued_hand_starts_when_the_sweeper_reserves_it(tmp_path):
    package = load_hands()
    dispatch = importlib.import_module(package.__name__ + '.dispatch')
    native, client = FakeHermes(), QueueingController()
    client.queue_next = True
    config = {'max_running':1, 'platforms':['sms'], 'contracts': {'text-route': {
        'version':'1','family':'text','modalities':['text'],'context_tokens':100000,
        'max_output_tokens':2000,'tool_calls':True,'decode_tok_s':100}}}
    runtime = dispatch.Runtime(config=config, client=client, spool_dir=tmp_path/'spool',
                               native=native, owner_check=lambda sid:sid.startswith('owner'))
    native.runtime = runtime
    runtime.install()
    package.register_route_resolver(lambda role,effort:native.native_policy())
    parent = Parent()  # the gateway keeps the owner's agent; the runtime holds it weakly
    try:
        result = json.loads(native.delegate_module.delegate_task(goal='check',parent_agent=parent))
        hand_id = result['hands'][0]['hand_id']
        assert result['hands'][0]['status'] == 'queued' and not native.calls
        runtime.maintain()
        assert not native.calls  # still queued in the controller: nothing starts
        client.promote(hand_id)
        runtime.maintain()
        assert len(native.calls) == 1 and native.calls[0]['background'] is True and not native.inline
        dispatched = [b for p, b, _ in client.writes if p.endswith(hand_id + '/dispatched')]
        assert dispatched and dispatched[0]['lease_token'] == 'lease-promoted'
        runtime.maintain()
        assert len(native.calls) == 1  # promotion starts it once
    finally:
        runtime.close()


def test_a_promoted_hand_without_its_parent_is_released_not_run(tmp_path):
    package = load_hands()
    dispatch = importlib.import_module(package.__name__ + '.dispatch')
    native, client = FakeHermes(), QueueingController()
    client.queue_next = True
    config = {'max_running':1, 'platforms':['sms'], 'contracts': {'text-route': {
        'version':'1','family':'text','modalities':['text'],'context_tokens':100000,
        'max_output_tokens':2000,'tool_calls':True,'decode_tok_s':100}}}
    runtime = dispatch.Runtime(config=config, client=client, spool_dir=tmp_path/'spool',
                               native=native, owner_check=lambda sid:sid.startswith('owner'))
    native.runtime = runtime
    runtime.install()
    package.register_route_resolver(lambda role,effort:native.native_policy())
    try:
        result = json.loads(native.delegate_module.delegate_task(goal='check',parent_agent=Parent()))
        hand_id = result['hands'][0]['hand_id']
        runtime.parents.clear()
        client.promote(hand_id)
        runtime.maintain()
        assert not native.calls
        released = [b for p, b, _ in client.writes if p.endswith(hand_id + '/release')]
        assert released and released[0]['reason'] == 'parent_unavailable'
    finally:
        runtime.close()


def test_a_queued_hand_holds_no_slot(setup):
    runtime, native, client = setup
    runtime.config['max_running'] = 1
    parent = Parent()
    queued = runtime.registry
    from importlib import import_module
    budgets = import_module(runtime.__class__.__module__.rsplit('.', 1)[0] + '.budgets')
    waiting = budgets.HandState('h-000000000009', task_id='t-000000000009', kind='research',
                                plan={'admit': 'queued', 'name': 'waiting'}, parent_session_id='owner-session',
                                epoch=runtime.epoch)
    waiting.status = 'planned'
    queued.add(waiting)
    result = json.loads(native.delegate_module.delegate_task(goal='check', parent_agent=parent))
    assert result['hands'][0]['status'] == 'dispatched' and len(native.calls) == 1


def refusing_controller(client, status, body):
    import httpx
    def refuse(path, payload, key):
        request = httpx.Request('POST', 'http://controller' + path)
        response = httpx.Response(status, json=body, request=request)
        raise httpx.HTTPStatusError('refused', request=request, response=response)
    client.post = refuse


def test_refused_plan_tells_the_front_why_and_what_to_change(setup):
    """Gate w18-20261007T182631Z: a bare controller_refused left the front no way to correct its plan."""
    runtime, native, client = setup
    refusing_controller(client, 422, {'detail': 'coding_requires_workdir_or_resources'})
    task = {'goal': 'Run the fixture command', 'kind': 'coding', 'context': 'see /home/owner/private/notes.txt'}
    answer = runtime.dispatch_tasks([task], parent=Parent())['hands'][0]
    assert answer['status'] == 'refused' and answer['reason'] == 'controller_refused'
    assert answer['detail'] == 'coding_requires_workdir_or_resources'
    assert 'use kind ops' in answer['hint'] and 'workdir' in answer['hint']
    assert '/home/owner' not in json.dumps(answer)
    assert not native.calls and not runtime.degraded.is_degraded


def test_validation_refusal_names_fields_never_values(setup):
    runtime, native, client = setup
    refusing_controller(client, 422, {'detail': [
        {'loc': ['body', 'tasks', 0, 'kind'], 'msg': 'Input should be coding or ops', 'type': 'enum',
         'input': 'secret-token-1234'},
        {'loc': ['body', 'tasks', 0, 'path with/slash'], 'msg': 'Extra inputs are not permitted', 'input': 'x'}]})
    answer = runtime.dispatch_tasks([{'goal': 'check', 'kind': 'coding'}], parent=Parent())['hands'][0]
    assert answer == {'status': 'refused', 'reason': 'controller_refused', 'detail': 'invalid_fields',
                      'fields': ['tasks.0.kind'], 'hint': answer['hint']} and 'schema' in answer['hint']
    assert 'secret-token' not in json.dumps(answer) and 'Input should' not in json.dumps(answer)


@pytest.mark.parametrize('body', [{'detail': 'Idempotency-Key must not be blank'}, {'detail': None}, 'not json'])
def test_refusal_without_a_stable_code_keeps_the_bare_answer(setup, body):
    runtime, native, client = setup
    refusing_controller(client, 422, body)
    answer = runtime.dispatch_tasks([{'goal': 'check'}], parent=Parent())['hands'][0]
    assert answer == {'status': 'refused', 'reason': 'controller_refused'}


def test_kind_the_controller_adjusted_is_reported_and_used(setup):
    runtime, native, client = setup
    original = client.post
    def adjusting(path, body, key):
        answer = original(path, body, key)
        if path.endswith('/plan'):
            answer['plans'][0]['kind'] = 'ops'
        return answer
    client.post = adjusting
    answer = runtime.dispatch_tasks([{'goal': 'Run the fixture command', 'kind': 'coding'}], parent=Parent())['hands'][0]
    assert answer['status'] == 'dispatched' and answer['kind'] == 'ops'
    assert answer['kind_adjusted'] == {'from': 'coding', 'to': 'ops'}
    assert runtime.registry.get(answer['hand_id']).kind == 'ops'
