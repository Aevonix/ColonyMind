"""App tasks use the maintenance loop and real controller admission, without a model turn."""
from concurrent.futures import ThreadPoolExecutor
import importlib
import threading
import time

import pytest

from hands_fake_hermes import FakeHermes, Parent, load_hands
from test_hands_definition_dispatch import setup, create  # noqa: F401

pytestmark = pytest.mark.requires_sidecar
BASE = '/v1/hands/definitions/queue/'


def enqueue(setup, definition=None, key='queue'):
    definition = definition or create(setup, tools=['read_file'])
    return setup[2].post('/v1/hands/definitions/' + definition['id'] + '/tasks', {'brief': 'Inspect sources'}, key)


def pickup(runtime):
    importlib.import_module(runtime.__class__.__module__.rsplit('.', 1)[0] + '.definitions').pump(runtime)


def test_enqueue_to_attempt_with_real_loop_under_ten_seconds(setup, record_property):
    runtime, native, client, c, http, _ = setup
    parent = Parent()
    runtime.remember_parent(parent)
    # Leave the configured legacy interval large: pickup must still meet its bound.
    runtime.config['spool'] = {'replay_seconds': 60}
    definition = create(setup, tools=['read_file'])
    # Do not wake on enqueue: exercise the worst phase of the bounded poll.
    runtime._resume_ready.clear()
    runtime.start()
    time.sleep(.1)
    started = time.monotonic()
    queued = enqueue(setup, definition)
    deadline = started + 10
    while time.monotonic() < deadline:
        rows = http.portal.call(c.all, 'SELECT * FROM hands WHERE task_id=? AND state=?', (queued['id'], 'running'))
        if rows:
            break
        time.sleep(.02)
    latency = time.monotonic() - started
    record_property('enqueue_to_attempt_seconds', round(latency, 4))
    print(f'enqueue_to_attempt_seconds={latency:.4f}')
    assert rows and latency <= 10
    assert len(native.calls) == 1 and native.calls[0]['background'] is True
    hand, = runtime.registry.all()
    assert rows[0]['parent_session'] == parent.session_id
    assert hand.task_id == queued['id'] and hand.plan['definition_policy']['tools'] == ['read_file']
    task = client.get('/v1/hands/tasks/' + queued['id'])
    assert task['dispatch_state'] is None and task['obligation_id']
    assert not http.portal.call(c.all, 'SELECT * FROM permits')


def test_two_runtimes_cannot_dispatch_the_same_request(setup, tmp_path):
    runtime, native, client, c, http, _ = setup
    package = load_hands()
    other_native = FakeHermes()
    other = importlib.import_module(package.__name__ + '.dispatch').Runtime(
        config=runtime.config, client=client, spool_dir=tmp_path / 'other-spool',
        native=other_native, owner_check=lambda sid: sid.startswith('owner'))
    other_native.runtime = other
    other.install()
    package.register_route_resolver(lambda role, effort: other_native.native_policy())
    parents = [Parent('owner-first'), Parent('owner-second')]
    runtime.remember_parent(parents[0])
    other.remember_parent(parents[1])
    queued = enqueue(setup)
    barrier = threading.Barrier(2)
    def race(instance):
        barrier.wait()
        pickup(instance)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(race, (runtime, other)))
        pickup(runtime)
        pickup(other)
        assert len(native.calls) + len(other_native.calls) == 1
        rows = http.portal.call(c.all, 'SELECT * FROM hands WHERE task_id=?', (queued['id'],))
        assert len(rows) == 1 and rows[0]['state'] == 'running'
    finally:
        other.close()


def test_no_owner_then_live_owner_dispatches_without_a_model_action(setup):
    runtime, native, client, *_ = setup
    queued = enqueue(setup)
    pickup(runtime)
    assert not native.calls
    assert client.get('/v1/hands/tasks/' + queued['id'])['dispatch_state'] == 'waiting_for_main_session'
    # An internal parent, child or dead weak parent grants no owner destination.
    parent = Parent()
    runtime.remember_parent(parent)
    pickup(runtime)
    pickup(runtime)
    assert len(native.calls) == 1


@pytest.mark.parametrize('refusal', ['no_async', 'capacity', 'contract'])
def test_admission_refusal_stays_queued_with_code_then_retries(setup, refusal):
    runtime, native, client, c, http, _ = setup
    queued = enqueue(setup)
    parent = Parent()
    runtime.remember_parent(parent)
    if refusal == 'no_async':
        native.consumer = False
    elif refusal == 'capacity':
        c.config['max_running'] = 0
    else:
        saved = c.config['routes'].copy()
        c.config['routes'] = {}
    pickup(runtime)
    assert not native.calls
    task = client.get('/v1/hands/tasks/' + queued['id'])
    assert task['dispatch_state'] == 'refused:' + refusal
    assert task['state'] == 'open'
    assert not http.portal.call(c.all, "SELECT * FROM hands WHERE state IN ('planned','reserved')")
    native.consumer = True
    c.config['max_running'] = 3
    if refusal == 'contract':
        c.config['routes'] = saved
    pickup(runtime)
    assert len(native.calls) == 1


@pytest.mark.parametrize('change', ['archived', 'disabled'])
@pytest.mark.parametrize('phase', ['before_claim', 'after_plan'])
def test_definition_changed_after_enqueue_is_never_dispatched(setup, change, phase, monkeypatch):
    runtime, native, client, c, http, _ = setup
    definition = create(setup)
    queued = enqueue(setup, definition)
    parent = Parent()
    runtime.remember_parent(parent)
    def change_definition():
        if change == 'archived':
            response = http.delete('/v1/hands/definitions/' + definition['id'], headers={'Idempotency-Key': 'archive'})
        else:
            from colonymind.hands.definitions import ReplaceDefinition
            body = {k: v for k, v in definition.items() if k in ReplaceDefinition.model_fields}
            response = http.patch('/v1/hands/definitions/' + definition['id'], json={**body, 'enabled': False},
                                  headers={'Idempotency-Key': 'disable'})
        assert response.status_code == 200
    if phase == 'before_claim':
        change_definition()
    else:
        route = runtime._route
        def race(*args):
            result = route(*args)
            change_definition()
            return result
        monkeypatch.setattr(runtime, '_route', race)
    pickup(runtime)
    assert not native.calls
    assert client.get('/v1/hands/tasks/' + queued['id'])['dispatch_state'] == 'refused:' + change + '_hand_definition'


def test_expired_claim_and_cached_plan_cannot_dispatch_after_reclaim(setup, monkeypatch):
    runtime, native, client, c, http, _ = setup
    queued = enqueue(setup)
    parent = Parent()
    runtime.remember_parent(parent)
    request = client.post(BASE + 'claim', {'parent_session': parent.session_id}, 'claim')['tasks'][0]
    plan = http.portal.call(c.plan, __import__('colonymind.hands.wire', fromlist=['PlanRequest']).PlanRequest(
        parent_session=parent.session_id, turn='real-turn', origin='own', origin_ref='actual-ingress',
        tasks=[{'goal': request['brief'], 'continue': queued['id'], 'queue_claim': request['claim_token']}]), 'stale-plan')
    async def expire():
        await c.execute("UPDATE definition_task_dispatch SET lease_until='2000-01-01' WHERE task_id=?", (queued['id'],))
        await c.store.db.commit()
    http.portal.call(expire)
    newer = client.post(BASE + 'claim', {'parent_session': parent.session_id}, 'claim2')['tasks'][0]
    assert newer['claim_token'] != request['claim_token']
    response = http.post(BASE + 'bind', json={'task_id': queued['id'], 'claim_token': request['claim_token'],
                         'hand_id': plan['plans'][0]['hand_id']}, headers={'Idempotency-Key': 'stale-bind'})
    assert response.status_code == 409
    assert not native.calls
    module = importlib.import_module(runtime.__class__.__module__.rsplit('.', 1)[0] + '.definitions')
    module.dispatch_claim(runtime, newer, parent.session_id)
    assert len(native.calls) == 1
    assert http.post(BASE + 'bind', json={'task_id': queued['id'], 'claim_token': newer['claim_token'],
        'hand_id': runtime.registry.all()[0].hand_id}, headers={'Idempotency-Key': 'bind-again'}).status_code == 409


def test_pickup_uses_most_recent_live_authorized_owner_context(setup):
    runtime, native, client, *_ = setup
    older, newest = Parent('owner-older'), Parent('owner-newer')
    runtime.remember_parent(older)
    runtime.remember_parent(newest)
    # Refreshing real ingress makes the older object the latest main session.
    runtime.remember_parent(older)
    queued = enqueue(setup)
    pickup(runtime)
    assert len(native.calls) == 1
    assert runtime.registry.all()[0].parent_session_id == older.session_id


def test_internal_or_revoked_parent_never_supplies_owner_authority(setup):
    runtime, native, client, *_ = setup
    from types import SimpleNamespace
    parent = Parent()
    runtime.remember_parent(parent)
    runtime.sessions = SimpleNamespace(standing=lambda sid: ('mind', 'internal'))
    queued = enqueue(setup)
    pickup(runtime)
    assert not native.calls
    assert client.get('/v1/hands/tasks/' + queued['id'])['dispatch_state'] == 'waiting_for_main_session'
    runtime.sessions = None
    runtime.owner_check = lambda sid: False
    pickup(runtime)
    assert not native.calls
    runtime.owner_check = lambda sid: True
    pickup(runtime)
    assert len(native.calls) == 1


def test_replayed_claim_can_bind_in_only_one_runtime(setup, tmp_path):
    runtime, native, client, *_ = setup
    package = load_hands()
    other_native = FakeHermes()
    other = importlib.import_module(package.__name__ + '.dispatch').Runtime(
        config=runtime.config, client=client, spool_dir=tmp_path / 'replay-spool',
        native=other_native, owner_check=lambda sid: sid.startswith('owner'))
    other_native.runtime = other
    other.install()
    package.register_route_resolver(lambda role, effort: other_native.native_policy())
    parent = Parent()
    runtime.remember_parent(parent)
    other.remember_parent(parent)
    queued = enqueue(setup)
    request = client.post(BASE + 'claim', {'parent_session': parent.session_id}, 'claim')['tasks'][0]
    barrier = threading.Barrier(2)
    def dispatch(instance):
        module = importlib.import_module(instance.__class__.__module__.rsplit('.', 1)[0] + '.definitions')
        barrier.wait()
        return module.dispatch_claim(instance, request, parent.session_id)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            answers = list(pool.map(dispatch, (runtime, other)))
        assert len(native.calls) + len(other_native.calls) == 1
        assert sum(answer['hands'][0]['status'] == 'dispatched' for answer in answers) == 1
        assert client.get('/v1/hands/tasks/' + queued['id'])['dispatch_state'] is None
    finally:
        other.close()


def test_repeated_unstarted_refusals_do_not_exhaust_attempt_or_cascade_budget(setup):
    runtime, native, client, c, http, _ = setup
    queued = enqueue(setup)
    parent = Parent()
    runtime.remember_parent(parent)
    native.consumer = False
    for _ in range(4):
        pickup(runtime)
    task = client.get('/v1/hands/tasks/' + queued['id'])
    assert task['dispatch_state'] == 'refused:no_async'
    assert task['usage'].get('hands', 0) == 0
    assert task['usage'].get('cascades', 0) == 0
    assert not native.calls
    native.consumer = True
    pickup(runtime)
    assert len(native.calls) == 1
    task = client.get('/v1/hands/tasks/' + queued['id'])
    assert task['usage']['hands'] == 1
    assert task['usage'].get('cascades', 0) == 0


def test_refused_older_request_does_not_starve_other_requests(setup, monkeypatch):
    runtime, native, client, c, *_ = setup
    c.config['health']['default_family_cap'] = 6
    definition = create(setup)
    blocked = client.post('/v1/hands/definitions/' + definition['id'] + '/tasks', {'brief': 'blocked'}, 'blocked')
    ready = client.post('/v1/hands/definitions/' + definition['id'] + '/tasks', {'brief': 'ready'}, 'ready')
    parent = Parent()
    runtime.remember_parent(parent)
    route = runtime._route
    def refuse_blocked(plan, task):
        if task['goal'] == 'blocked':
            raise ValueError('contract')
        return route(plan, task)
    monkeypatch.setattr(runtime, '_route', refuse_blocked)
    pickup(runtime)
    assert len(native.calls) == 1
    assert next(hand for hand in runtime.registry.all() if hand.status == 'running').task_id == ready['id']
    assert client.get('/v1/hands/tasks/' + blocked['id'])['dispatch_state'] == 'refused:contract'


def test_one_pass_picks_up_available_capacity_without_waiting_between_tasks(setup):
    runtime, native, client, c, *_ = setup
    c.config['health']['default_family_cap'] = 6
    definition = create(setup)
    for i in range(3):
        enqueue(setup, definition, 'queue-' + str(i))
    parent = Parent()
    runtime.remember_parent(parent)
    pickup(runtime)
    assert len(native.calls) == 3
