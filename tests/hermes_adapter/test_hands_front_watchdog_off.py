"""The watchdog's rollback is a config change; the request bridge answers as before, and nothing restarts."""

from test_hands_front import begin_tool, end_tool, owner_turn, rig  # noqa: F401  (fixture and helper)


def test_watchdog_follows_the_live_config_and_the_request_bridge_is_unaffected(rig):  # noqa: F811
    live = {'front': {}}
    rig.runtime.config_loader = lambda: live
    rig.guard.start_threads = True
    owner_turn(rig, turn='turn-1', ident='message-1')
    first = rig.guard._state('owner-session', 'turn-1')
    assert first.thread is not None
    first.stopped.set()

    live['front'] = {'watchdog': False}
    owner_turn(rig, text='Owner asks again', turn='turn-2', ident='message-2')
    second = rig.guard._state('owner-session', 'turn-2')
    assert second.thread is None, 'watchdog off: no watchdog for the next turn'
    caps = rig.guard.request_caps('owner-session', 'turn-2')
    assert caps and caps == rig.guard.request_caps('owner-session', 'turn-1'), (
        'the request bridge answers the same with the watchdog off')

    live['front'] = {'watchdog': True}
    owner_turn(rig, text='Owner asks a third time', turn='turn-3', ident='message-3')
    third = rig.guard._state('owner-session', 'turn-3')
    assert third.thread is not None
    third.stopped.set()


def test_an_unreadable_live_config_keeps_the_watchdog(rig):  # noqa: F811
    def broken():
        raise OSError('unreadable')
    rig.runtime.config_loader = broken
    rig.runtime.config = {'front': {}}
    rig.guard.start_threads = True
    owner_turn(rig)
    state = rig.guard._state('owner-session', 'turn-1')
    assert state.thread is not None
    state.stopped.set()


def test_a_second_pre_llm_call_keeps_the_review_classification(rig):  # noqa: F811
    """Hermes runs the front's pre_llm_call and the adapter's observe_turn bridge for the same
    completion turn; the second call set the turn back to 'system', so the watchdog never redirected a review."""
    from test_hands_front import review_turn
    review_turn(rig)
    state = rig.guard._state('owner-session', 'turn-1')
    assert state.mode == 'review_result'
    rig.guard.observe_turn(session_id='owner-session', turn_id='turn-1', user_message='Native completion',
                           platform='sms', sender_id='5550100', chat_id='5550100', session_key='route',
                           conversation_history=[{'role': 'user', 'content': 'Native completion',
                                                  'display_kind': 'internal_notification'}])
    assert state.mode == 'review_result'
    begin_tool(rig)
    rig.pending['route'] = __import__('test_hands_front').event('Owner asks now', 'message-9')
    rig.guard.tick('owner-session', 'turn-1')
    assert not rig.agent.redirects and not rig.agent.steers
    end_tool(rig)
    assert rig.agent.steers == [rig.front.OWNER_WAITING_LINE]


def test_classification_runs_when_structure_arrives_on_the_second_call(rig):  # noqa: F811
    rig.agent._current_turn_id = 'turn-1'
    rig.guard.observe_turn(session_id='owner-session', turn_id='turn-1', user_message='Native completion',
                           platform='sms', sender_id='5550100', chat_id='5550100', session_key='route')
    state = rig.guard._state('owner-session', 'turn-1')
    from test_hands_front import review_turn
    review_turn(rig)
    assert state.mode == 'review_result'


def test_faculty_off_after_review_drop_restores_provider_and_blocks_retry_injection(rig):
    from importlib import import_module
    from test_hands_front import event, review_turn
    live = {'mind': {'faculties': {'hands': True}}}
    rig.runtime.config_loader = lambda: live
    rig.runtime.registry.for_session = lambda *args, **kwargs: None
    review_turn(rig)
    original_callback = rig.agent.stream_delta_callback
    rig.pending['route'] = event('Owner asks now', 'message-2')
    middleware = import_module(rig.front.__package__ + '.middleware')
    ids = {'session_id': 'owner-session', 'turn_id': 'turn-1'}
    prepared = middleware.llm_request(rig.runtime, {'messages': []}, **ids)
    state = rig.guard._state('owner-session', 'turn-1')
    assert prepared['reason'] == 'review_deferred_for_owner'
    assert state.request_dropped and state.owner_waiting
    assert rig.agent.stream_delta_callback is None
    live['mind']['faculties']['hands'] = False
    rig.guard.tick(**ids)
    sent = []
    def execute(request):
        return middleware.llm_execution(rig.runtime, request,
            lambda value: sent.append(value) or 'Provider answer', **ids)
    assert rig.guard.llm_execution(prepared['request'], execute, **ids) == 'Provider answer'
    assert len(sent) == 1
    assert not state.request_dropped and not state.owner_waiting
    assert rig.agent.stream_delta_callback is original_callback
    assert rig.guard.finalize_response('Provider answer', **ids) is None
    rig.guard.post_llm_call(assistant_response='Provider answer', **ids)
    rig.guard.on_native_turn_settled(outcome={'completed': True, 'failed': False, 'interrupted': False}, **ids)
    rig.pending.clear()
    rig.state.persistent.run_generation += 1
    owner_turn(rig, 'Owner asks now', 'turn-2', 'message-2')
    rig.guard.post_llm_call(session_id='owner-session', turn_id='turn-2', assistant_response='Owner answer')
    rig.guard.on_native_turn_settled(session_id='owner-session', turn_id='turn-2',
        outcome={'completed': True, 'failed': False, 'interrupted': False})
    rig.guard.on_gateway_turn_settled(session_id='owner-session', session_key='route', run_generation=2)
    assert rig.ctx.injected == [] and rig.agent.interrupts == 0
    assert rig.agent.redirects == [] and rig.agent.steers == []
