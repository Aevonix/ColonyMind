"""Socket-free request suppression and owner-first review retry regressions."""
from importlib import import_module

import pytest

from test_hands_front import event, owner_turn, result_rows, review_turn, rig  # noqa: F401


def request(rig, turn='turn-1', **ids):
    middleware = import_module(rig.front.__package__ + '.middleware')
    rig.runtime.registry.for_session = lambda *args, **kwargs: None
    payload = {'messages': [{'role': 'user', 'content': 'Native completion'}]}
    envelope = middleware.llm_request(rig.runtime, payload, session_id='owner-session', turn_id=turn, **ids)
    sent = []
    response = middleware.llm_execution(rig.runtime, envelope['request'],
        lambda value: sent.append(value) or 'provider response', session_id='owner-session', turn_id=turn, **ids)
    return sent, response


def settle(rig, turn, text='[SILENT]'):
    rig.guard.post_llm_call(session_id='owner-session', turn_id=turn, assistant_response=text)
    rig.guard.on_session_end(session_id='owner-session', turn_id=turn,
        completed=True, failed=False, interrupted=False)
    rig.gateway._release_turn_lease('route', rig.state.persistent.run_generation)


@pytest.mark.parametrize('queue', ['pending', 'fifo'])
def test_waiting_owner_drops_review_request_and_retries_after_owner(rig, queue):
    rig.client.rows = result_rows(1)
    review_turn(rig)
    owner = event('Owner asks now', 'message-2')
    if queue == 'pending':
        rig.pending['route'] = owner
    else:
        rig.fifo.append(owner)
    # No watchdog tick or accepted native redirect is needed before the request.
    rig.agent.redirect = lambda text: False
    sent, response = request(rig)
    assert sent == [], 'a deferred review must never reach the provider'
    assert response is not None, 'None is an invalid native response and triggers retries'
    state = rig.guard._state('owner-session', 'turn-1')
    assert state.owner_waiting and state.request_dropped
    assert rig.guard.finalize_response('anything', session_id='owner-session', turn_id='turn-1') == '[SILENT]'
    assert not rig.agent.interrupts
    settle(rig, 'turn-1')
    assert rig.ctx.injected == [], 'retry must wait until the owner has had a turn'
    assert not rig.guard.rendered
    assert not [path for path, _, _ in rig.client.writes if path.endswith(('/review', '/communicate'))]

    rig.pending.clear()
    rig.fifo.clear()
    rig.state.persistent.run_generation += 1
    owner_turn(rig, 'Owner asks now', 'turn-2', 'message-2')
    assert request(rig, 'turn-2')[0], 'the owner still calls the provider'
    settle(rig, 'turn-2', 'Owner answer')
    assert len(rig.ctx.injected) == 1
    retry = rig.ctx.injected[0]
    assert retry['session_key'] == 'route'
    # Gateway injection is structurally internal; the body retains the native receipt.
    assert retry['content'] == 'Native completion'
    assert rig.gateway._release_turn_lease('route', 2) is False
    assert len(rig.ctx.injected) == 1
    rig.state.persistent.run_generation += 1
    review_turn(rig, 'turn-3')
    assert request(rig, 'turn-3')[0], 'the deferred result is reviewed after the owner'
    settle(rig, 'turn-3', 'Reviewed result')
    assert len(rig.guard.rendered) == 1


@pytest.mark.parametrize('kind', ['internal', 'guest', 'other_chat', 'other_session', 'no_event', 'owner_turn', 'stale_generation', 'stale_agent', 'watchdog_off', 'child'])
def test_review_drop_is_scoped_to_a_live_review_and_waiting_owner(rig, kind):
    if kind == 'owner_turn':
        owner_turn(rig)
    else:
        review_turn(rig)
    waiting = event(internal=kind == 'internal', sender='guest' if kind == 'guest' else '5550100')
    if kind == 'other_chat':
        waiting.source.chat_id = 'other-chat'
    if kind != 'no_event':
        rig.pending['other-route' if kind == 'other_session' else 'route'] = waiting
    if kind == 'stale_generation':
        rig.state.persistent.run_generation += 1
    if kind == 'stale_agent':
        rig.gateway._running_agents['route'] = object()
    if kind == 'watchdog_off':
        rig.runtime.config = {'front': {'watchdog': False}}
    ids = {'platform': 'subagent'} if kind == 'child' else {}
    sent, _ = request(rig, **ids)
    assert len(sent) == 1
    assert not rig.ctx.injected


def test_waiting_owner_skips_review_digest_io(rig):
    rig.pending['route'] = event()
    def unexpected(*args, **kwargs):
        pytest.fail('a review that is yielding needs no controller digest')
    rig.client.get = unexpected
    review_turn(rig)
    assert request(rig)[0] == []


def test_review_predicate_is_exact_and_never_uses_owner_lookup(rig):
    review_turn(rig)
    def unexpected(*args):
        pytest.fail('the review predicate must be a local lookup')
    rig.runtime.authorized_owner = unexpected
    bridge = rig.guard.bridge
    assert bridge.is_review_turn('owner-session', 'turn-1') is True
    assert bridge.is_review_turn('owner-session', 'turn-missing') is False
    assert bridge.is_review_turn('other-session', 'turn-1') is False
    assert bridge.is_review_turn('owner-session', '') is False
    rig.guard._state('owner-session', 'turn-1').finished = True
    assert bridge.is_review_turn('owner-session', 'turn-1') is False


def test_retry_waits_for_all_queued_owner_turns_and_failed_injection_stays_pending(rig):
    review_turn(rig)
    rig.pending['route'] = event()
    assert request(rig)[0] == []
    settle(rig, 'turn-1')
    rig.pending.clear()
    rig.state.persistent.run_generation += 1
    owner_turn(rig, turn='turn-2', ident='message-2')
    rig.pending['route'] = event('Next owner message', 'message-3')
    settle(rig, 'turn-2', 'Answer')
    assert not rig.ctx.injected
    rig.pending.clear()
    rig.ctx.inject_message = lambda *args, **kwargs: False
    rig.state.persistent.run_generation += 1
    owner_turn(rig, turn='turn-3', ident='message-3')
    settle(rig, 'turn-3', 'Answer')
    assert rig.guard.metrics['injection_failed'] == 1
    rig.ctx.inject_message = lambda content, **kw: rig.ctx.injected.append({'content': content, **kw}) or True
    rig.state.persistent.run_generation += 1
    owner_turn(rig, turn='turn-4', ident='message-4')
    settle(rig, 'turn-4', 'Answer')
    assert len(rig.ctx.injected) == 1


def test_scheduled_retry_remains_pending_until_a_review_actually_starts(rig):
    review_turn(rig)
    rig.pending['route'] = event()
    assert request(rig)[0] == []
    settle(rig, 'turn-1')
    rig.pending.clear()
    rig.state.persistent.run_generation += 1
    owner_turn(rig, turn='turn-2', ident='message-2')
    settle(rig, 'turn-2', 'Answer')
    assert len(rig.ctx.injected) == 1
    # Scheduling may succeed while the native asynchronous route later refuses.
    rig.state.persistent.run_generation += 1
    owner_turn(rig, turn='turn-3', ident='message-3')
    settle(rig, 'turn-3', 'Answer')
    assert len(rig.ctx.injected) == 2


def test_queued_owner_follow_up_in_the_same_run_schedules_the_retry(rig):
    # Hermes runs the queued owner message as the review run's own follow-up: one generation, one settlement.
    review_turn(rig)
    rig.pending['route'] = event('Owner asks now', 'message-2')
    assert request(rig)[0] == []
    rig.pending.clear()
    owner_turn(rig, 'Owner asks now', 'turn-2', 'message-2')
    assert request(rig, 'turn-2')[0], 'the owner still calls the provider'
    rig.guard.post_llm_call(session_id='owner-session', turn_id='turn-1', assistant_response='[SILENT]')
    settle(rig, 'turn-2', 'Owner answer')
    assert [item['content'] for item in rig.ctx.injected] == ['Native completion']


def test_retry_stops_after_three_scheduled_attempts_without_a_review(rig):
    review_turn(rig)
    rig.pending['route'] = event()
    assert request(rig)[0] == []
    settle(rig, 'turn-1')
    rig.pending.clear()
    # No injected retry ever starts a classified review (its native receipt is gone, for instance).
    for number in range(2, 7):
        rig.state.persistent.run_generation += 1
        owner_turn(rig, turn=f'turn-{number}', ident=f'message-{number}')
        settle(rig, f'turn-{number}', 'Answer')
    assert len(rig.ctx.injected) == 3
    assert rig.guard.metrics['review_retry_abandoned'] == 1
    assert not rig.guard.deferred_reviews


def test_old_owner_settlement_cannot_schedule_a_newly_dropped_review(rig):
    owner_turn(rig, turn='turn-old', ident='old-message')
    settle(rig, 'turn-old', 'Old answer')
    rig.state.persistent.run_generation += 1
    review_turn(rig)
    rig.pending['route'] = event()
    assert request(rig)[0] == []
    rig.pending.clear()
    assert rig.gateway._release_turn_lease('route', 1) is False
    assert not rig.ctx.injected


def test_owner_arrival_between_request_and_execution_also_drops_without_a_send(rig):
    review_turn(rig)
    rig.runtime.registry.for_session = lambda *args, **kwargs: None
    middleware = import_module(rig.front.__package__ + '.middleware')
    ids = {'session_id': 'owner-session', 'turn_id': 'turn-1'}
    prepared = middleware.llm_request(rig.runtime, {'messages': []}, **ids)
    assert 'reason' not in prepared
    rig.pending['route'] = event()
    sent = []
    assert middleware.llm_execution(rig.runtime, prepared['request'], sent.append, **ids) is not None
    assert sent == []
    assert rig.guard._state('owner-session', 'turn-1').request_dropped


def scheduled_review(rig, *, accepted=True):
    """Drop one result, then let the owner's settlement schedule its replay."""
    rig.client.rows = result_rows(1)
    review_turn(rig)
    rig.pending['route'] = event('Owner asks now', 'message-2')
    assert request(rig)[0] == []
    settle(rig, 'turn-1')
    rig.pending.clear()
    rig.state.persistent.run_generation += 1
    owner_turn(rig, 'Owner asks now', 'turn-2', 'message-2')
    if not accepted:
        rig.ctx.inject_message = lambda *args, **kwargs: False
    settle(rig, 'turn-2', 'Owner answer')
    rig.state.persistent.run_generation += 1


def unmarked_review(rig, *, session='owner-session', route='route', text='Native completion', turn='replay-turn'):
    rig.agent._current_turn_id = turn
    return rig.guard.pre_llm_call(session_id=session, turn_id=turn, user_message=text,
        platform='sms', sender_id='5550100', chat_id='5550100', session_key=route,
        conversation_history=[{'role': 'user', 'content': text}])


def test_scheduled_review_queued_behind_another_owner_keeps_review_provenance(rig):
    scheduled_review(rig)
    owner_turn(rig, 'Another owner message', 'turn-3', 'message-3')
    # The asynchronous injection reached the busy adapter. Native queued follow-up loses display_kind.
    unmarked_review(rig)
    assert rig.guard.is_review_turn('owner-session', 'replay-turn')
    assert not rig.guard.owner_started_turn('owner-session', 'replay-turn')
    assert rig.guard.capture_owner_text('owner-session', 'replay-turn', 'Native completion') is None
    assert not rig.guard.deferred_reviews
    settle(rig, 'replay-turn', 'Reviewed result')
    assert len(rig.guard.rendered) == 1
    assert len(rig.ctx.injected) == 1


def test_identical_owner_words_never_claim_a_scheduled_review(rig):
    scheduled_review(rig)
    owner_turn(rig, 'Native completion', 'owner-repeat', 'message-3')
    assert not rig.guard.is_review_turn('owner-session', 'owner-repeat')
    assert rig.guard.owner_started_turn('owner-session', 'owner-repeat')
    assert rig.guard.capture_owner_text('owner-session', 'owner-repeat', 'Native completion') == 'Native completion'
    assert len(rig.guard.deferred_reviews) == 1
    assert request(rig, 'owner-repeat')[0]


@pytest.mark.parametrize('invalid', ['unscheduled', 'refused', 'receipt_missing', 'receipt_changed',
    'old_generation', 'stale_agent', 'other_session', 'other_route', 'other_message'])
def test_unmarked_replay_requires_accepted_scheduling_and_current_native_provenance(rig, invalid):
    scheduled_review(rig, accepted=invalid != 'refused')
    deferred = rig.guard.deferred_reviews[('owner-session', 'route')]
    kwargs = {}
    if invalid == 'unscheduled':
        deferred.scheduled = False
    elif invalid == 'receipt_missing':
        rig.guard._completion_texts = lambda state: {}
    elif invalid == 'receipt_changed':
        rig.guard._completion_texts = lambda state: {'h-0': 'Different native receipt'}
    elif invalid == 'old_generation':
        rig.state.persistent.run_generation = deferred.generation
    elif invalid == 'stale_agent':
        rig.gateway._running_agents['route'] = object()
    elif invalid == 'other_session':
        rig.runtime.authorized_owner = lambda sid: True
        kwargs['session'] = 'other-session'
    elif invalid == 'other_route':
        kwargs['route'] = 'other-route'
        rig.gateway._running_agents.clear()
    elif invalid == 'other_message':
        kwargs['text'] = 'Some other internal message'
    unmarked_review(rig, **kwargs)
    state = rig.guard._state(kwargs.get('session', 'owner-session'), 'replay-turn')
    assert state.mode == 'answer_now'
    assert not state.rendered
    assert not state.completions
    assert len(rig.guard.deferred_reviews) == 1
