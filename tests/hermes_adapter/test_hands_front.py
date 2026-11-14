"""Front acceptance tests. Clock injection keeps the timing of tool work deterministic.

An owner turn has no call cap, no time limit and no forced thinking setting; test_owner_front_limits.py
pins that. Here the yield line is tested on the one turn that still gets it, a hand's result review.
"""
from collections import deque
from importlib import import_module
import threading
from types import SimpleNamespace as NS

import pytest

from hands_fake_hermes import FakeContext, attach_turn_leases, hold_turn_lease, load_hands


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class Agent:
    def __init__(self, sid='owner-session', turn='turn-1'):
        self.session_id, self._current_turn_id = sid, turn
        self.redirects, self.steers, self.interrupts = [], [], 0
        self._disable_streaming = False
        self.stream_delta_callback = lambda text: None
        self._stream_callback = lambda text: None
        self.interim_assistant_callback = lambda text: None
        self.reasoning_callback = lambda text: None

    def redirect(self, text):
        self.redirects.append(text)
        return True

    def steer(self, text):
        self.steers.append(text)
        return True

    def hard_interrupt(self):
        self.interrupts += 1


class Client:
    def __init__(self):
        self.rows, self.writes = [], []

    def get(self, path, params=None):
        if path.endswith('/digest'):
            return {'text': 'Ready results' if self.rows else '', 'items': self.rows}
        if path.rstrip('/') == '/v1/hands':
            return {'hands': self.rows}
        return next((r for r in self.rows if path.endswith('/' + r['hand_id'])), {})

    def post(self, path, body, key):
        self.writes.append((path, body, key))
        return {'ok': True}


def event(text='Owner asks', ident='message-1', internal=False, sender='5550100'):
    return NS(text=text, message_id=ident, internal=internal, metadata={},
              source=NS(platform='sms', user_id=sender, chat_id='5550100', chat_type='dm'),
              session_id='owner-session', session_key='route')


@pytest.fixture
def rig(tmp_path):
    package = load_hands()
    front = import_module(package.__name__ + '.front')
    agent, clock, client, ctx = Agent(), Clock(), Client(), FakeContext()
    client.settings = NS(hermes_home=tmp_path / 'hermes')
    ctx.injected = []
    ctx.inject_message = lambda content, **kw: ctx.injected.append({'content': content, **kw}) or True
    pending, fifo = {}, []
    adapter = NS(_pending_messages=pending)
    state = NS(persistent=NS(run_generation=1), conversation=NS(queued_events=fifo))
    gateway = NS(_running_agents={'route': agent}, adapters={'sms': adapter},
                 _peek_session_state=lambda key: state if key == 'route' else None,
                 _is_session_run_current=lambda key, generation: key == 'route' and generation == state.persistent.run_generation)
    attach_turn_leases(gateway, state)
    hold_turn_lease(gateway, state)
    runtime = NS(config={}, client=client, gateway=lambda: gateway,
                 authorized_owner=lambda sid: sid == 'owner-session', sessions=None,
                 registry=NS(for_session=lambda sid: None), emit=client.post)
    guard = front.FrontGuard(runtime, ctx, clock=clock, start_threads=False)
    guard.install_bridge()
    guard._completion_texts = lambda state: {f'h-{n}': 'Native completion' for n in range(8)}
    yield NS(front=front, guard=guard, agent=agent, clock=clock, client=client, ctx=ctx,
             gateway=gateway, state=state, pending=pending, fifo=fifo, runtime=runtime)
    guard.close()


def owner_turn(rig, text='Owner asks', turn='turn-1', ident='message-1'):
    hold_turn_lease(rig.gateway, rig.state)
    rig.agent._current_turn_id = turn
    rig.guard.pre_gateway_dispatch(event=event(text, ident), gateway=rig.gateway)
    return rig.guard.pre_llm_call(session_id='owner-session', turn_id=turn, user_message=text,
                                  platform='sms', sender_id='5550100', message_id=ident, session_key='route')


def review_turn(rig, turn='turn-1'):
    hold_turn_lease(rig.gateway, rig.state)
    rig.agent._current_turn_id = turn
    return rig.guard.pre_llm_call(session_id='owner-session', turn_id=turn,
        user_message='Native completion', platform='sms', sender_id='5550100', chat_id='5550100', session_key='route',
        conversation_history=[{'role': 'user', 'content': 'Native completion', 'display_kind': 'internal_notification'}])


def result_rows(count=8):
    return [dict(hand_id=f'h-{n}', id=f'h-{n}', task_id=f't-{n}', obligation_id=f'o-{n}',
                 parent_session='owner-session', state='done', result_hash=f'result-{n}',
                 intent_version=1, text=f'Result {n}', outcome='done') for n in range(count)]


def begin_tool(rig, call='call-1', turn='turn-1'):
    return rig.guard.pre_tool_call(session_id='owner-session', turn_id=turn,
                                  tool_name='read_file', tool_call_id=call, args={})


def end_tool(rig, call='call-1', turn='turn-1'):
    return rig.guard.post_tool_call(session_id='owner-session', turn_id=turn,
                                   tool_name='read_file', tool_call_id=call, result='done')


@pytest.mark.parametrize('tag', ['think', 'thinking', 'reasoning', 'thought', 'REASONING_SCRATCHPAD'])
def test_reasoning_is_not_visible_text(rig, tag):
    review_turn(rig)
    rig.guard.post_api_request(session_id='owner-session', turn_id='turn-1', assistant_message=NS(content=f'<{tag}>private</{tag}>'))
    begin_tool(rig)
    rig.clock.now = 30
    rig.guard.tick('owner-session', 'turn-1')
    assert not rig.agent.redirects and not rig.agent.steers
    end_tool(rig)
    assert rig.agent.steers == [rig.front.YIELD_LINE]


def test_visible_text_cancels_the_yield_line(rig):
    review_turn(rig)
    rig.guard.post_api_request(session_id='owner-session', turn_id='turn-1', assistant_message={'content': 'Answer'})
    begin_tool(rig)
    rig.clock.now = 30
    end_tool(rig)
    assert not rig.agent.steers and not rig.agent.redirects
    begin_tool(rig, 'call-2')
    rig.clock.now = 90
    rig.guard.tick('owner-session', 'turn-1')
    assert rig.agent.interrupts == 0
    end_tool(rig, 'call-2')
    assert rig.agent.interrupts == 0
    assert len(rig.ctx.injected) == 0
    assert not rig.agent.steers and not rig.agent.redirects


def test_generation_or_agent_replacement_prevents_stale_action(rig):
    review_turn(rig)
    begin_tool(rig)
    rig.clock.now = 100
    rig.state.persistent.run_generation = 2
    rig.guard.tick('owner-session', 'turn-1')
    end_tool(rig)
    assert not rig.agent.redirects and not rig.agent.steers and rig.agent.interrupts == 0


@pytest.mark.parametrize('slot', ['pending', 'fifo'])
def test_owner_arrival_read_only_queue_watch_steers_review_at_boundary(rig, slot):
    review_turn(rig)
    begin_tool(rig)
    message = event('Next question', 'message-2')
    if slot == 'pending':
        rig.pending['route'] = message
    else:
        rig.fifo.append(message)
    rig.clock.now = 0.2
    rig.guard.tick('owner-session', 'turn-1')
    assert not rig.agent.redirects and not rig.agent.steers
    end_tool(rig)
    assert rig.agent.steers == [rig.front.OWNER_WAITING_LINE]
    assert rig.pending.get('route') is message if slot == 'pending' else rig.fifo == [message]
    rig.guard.tick('owner-session', 'turn-1')
    assert len(rig.agent.steers) == 1 and not rig.agent.redirects


def test_a_refused_owner_steer_is_retried_at_the_next_tool_boundary(rig):
    review_turn(rig)
    answers = iter([False, False, True])
    taken = []
    rig.agent.steer = lambda text: taken.append(text) or next(answers)
    rig.pending['route'] = event('Next question', 'message-2')
    rig.guard.tick('owner-session', 'turn-1')
    assert taken == []
    for call in ('call-1', 'call-2', 'call-3'):
        assert begin_tool(rig, call).get('action') != 'block'
        end_tool(rig, call)
    assert taken == [rig.front.OWNER_WAITING_LINE] * 3
    assert rig.guard.metrics['redirect_refused'] == 1 and rig.guard.metrics['redirect_accepted'] == 1
    assert not rig.agent.redirects
    state = rig.guard._state('owner-session', 'turn-1')
    assert state.owner_waiting and state.owner_redirected and state.synthetic == [rig.front.OWNER_WAITING_LINE]


def test_internal_pending_event_does_not_preempt(rig):
    review_turn(rig)
    rig.pending['route'] = event(internal=True)
    rig.guard.tick('owner-session', 'turn-1')
    assert not rig.agent.redirects


def test_unmatched_owner_text_is_not_duplicate_review(rig):
    rig.guard.pre_gateway_dispatch(event=event(), gateway=rig.gateway)
    rig.guard.pre_llm_call(session_id='owner-session', turn_id='turn-1', user_message='[reply context]\n\nOwner asks',
                          platform='sms', sender_id='5550100', message_id='message-1', session_key='route')
    assert rig.guard.request_caps('owner-session', 'turn-1')['mode'] == 'answer_now'
    assert rig.guard.finalize_response('Owner answer', session_id='owner-session', turn_id='turn-1') is None
    # Bound by the platform message id, Hermes' own note ahead of what was sent keeps the turn his.
    assert rig.guard.capture_owner_text('owner-session', 'turn-1', '[reply context]\n\nOwner asks') == (
        '[reply context]\n\nOwner asks')
    assert rig.guard.owner_started_turn('owner-session', 'turn-1') is True


@pytest.mark.parametrize('sent,turn_text', [
    ('yes', '[Replying to: "Approve the change?"]\n\nyes'),
    ('', '"Please check the report"'),
    ('the caption', '[The user sent an image]\n\nthe caption'),
])
def test_hermes_enrichment_of_a_bound_owner_message_stays_his(rig, sent, turn_text):
    rig.guard.pre_gateway_dispatch(event=event(sent), gateway=rig.gateway)
    rig.guard.pre_llm_call(session_id='owner-session', turn_id='turn-1', user_message=turn_text,
                          platform='sms', sender_id='5550100', message_id='message-1', session_key='route')
    assert rig.guard.capture_owner_text('owner-session', 'turn-1', turn_text) == turn_text


def test_enriched_text_without_a_message_id_binding_or_with_an_unsplit_redirect_is_refused(rig):
    rig.guard.pre_gateway_dispatch(event=event('Owner asks', ident=''), gateway=rig.gateway)
    rig.guard.pre_llm_call(session_id='owner-session', turn_id='turn-1', user_message='[reply context]\n\nOwner asks',
                          platform='sms', sender_id='5550100', session_key='route')
    assert rig.guard.capture_owner_text('owner-session', 'turn-1', '[reply context]\n\nOwner asks') is None
    owner_turn(rig, turn='turn-2', ident='message-2')
    begin_tool(rig, turn='turn-2')
    rig.clock.now = 30
    end_tool(rig, turn='turn-2')
    text = 'Owner asks\n\n' + rig.front.YIELD_LINE + '\n\nlater words'
    assert rig.guard.capture_owner_text('owner-session', 'turn-2', '[note]\n\n' + text) is None


def test_ledger_refuses_an_unrecorded_line_and_a_replay(rig):
    owner_turn(rig)
    begin_tool(rig)
    rig.clock.now = 30
    end_tool(rig)
    synthetic = rig.front.YIELD_LINE
    # An owner turn is sent no runtime line, so the ledger recorded none and splits none off: no fuzzy subtraction.
    assert rig.guard.capture_owner_text('owner-session', 'turn-1', 'Owner asks\n\n' + synthetic) is None
    assert rig.guard.capture_owner_text('owner-session', 'turn-1', synthetic) is None
    rig.agent._current_turn_id = 'turn-2'
    rig.guard.pre_llm_call(session_id='owner-session', turn_id='turn-2', user_message='Owner asks', platform='sms')
    assert rig.guard.owner_started_turn('owner-session', 'turn-2') is False
    assert rig.guard.capture_owner_text('owner-session', 'turn-2', 'Owner asks') is None


def test_eight_results_render_once_and_duplicate_filters_outbound(rig):
    rig.client.rows = result_rows()
    first = review_turn(rig)
    assert all(f'Result {n}' in first['context'] for n in range(8))
    rig.guard.post_llm_call(session_id='owner-session', turn_id='turn-1', assistant_response='Consolidated answer')
    rig.guard.on_session_end(session_id='owner-session', turn_id='turn-1', completed=True, failed=False, interrupted=False)
    for n in range(2, 9):
        result = review_turn(rig, f'turn-{n}')
        assert 'already reviewed' in result['context']
        assert rig.guard.finalize_response('Unwanted prose', session_id='owner-session', turn_id=f'turn-{n}') == '[SILENT]'
        assert rig.agent._disable_streaming is True
        assert rig.agent.stream_delta_callback is None
        assert rig.agent.interim_assistant_callback is None
        rig.guard.post_llm_call(session_id='owner-session', turn_id=f'turn-{n}', assistant_response='[SILENT]')
    assert len([row for row in rig.client.writes if row[0].endswith('/review')]) == 8


def test_empty_digest_is_not_duplicate(rig):
    review_turn(rig)
    assert not rig.guard.request_caps('owner-session', 'turn-1')['duplicate']
    assert rig.guard.finalize_response('Answer', session_id='owner-session', turn_id='turn-1') is None


def test_silent_reply_never_communicates_obligation(rig):
    rig.client.rows = result_rows(1)
    review_turn(rig)
    rig.guard.post_llm_call(session_id='owner-session', turn_id='turn-1', assistant_response='[SILENT]')
    rig.guard.on_session_end(session_id='owner-session', turn_id='turn-1', completed=True, failed=False, interrupted=False)
    assert not any(p.endswith('/communicate') for p, _, _ in rig.client.writes)


@pytest.mark.parametrize('outcome', [None, {'completed': False, 'failed': True, 'interrupted': False}, {'completed': True, 'failed': False, 'interrupted': True}])
def test_unsuccessful_native_outcome_never_marks_reviewed(rig, outcome):
    rig.client.rows = result_rows(1)
    review_turn(rig)
    rig.guard.post_llm_call(session_id='owner-session', turn_id='turn-1', assistant_response='Partial text')
    assert not any(p.endswith('/review') for p, _, _ in rig.client.writes)
    rig.guard.on_session_end(session_id='owner-session', turn_id='turn-1', **(outcome or {}))
    assert not any(p.endswith('/review') for p, _, _ in rig.client.writes)
    assert not rig.guard.rendered


def test_review_communication_uses_origin_chat_and_leaves_delivery_unknown(rig):
    rig.client.rows = result_rows(1)
    rig.client.rows[0]['origin_ref'] = {'platform': 'sms', 'chat': '5550100', 'message_id': 'original'}
    review_turn(rig)
    rig.guard.post_llm_call(session_id='owner-session', turn_id='turn-1', assistant_response='Answer')
    rig.guard.on_session_end(session_id='owner-session', turn_id='turn-1', completed=True, failed=False, interrupted=False)
    writes = [body for path, body, _ in rig.client.writes if path.endswith('/communicate')]
    assert len(writes) == 1
    assert writes[0]['destination'] == {'platform': 'sms', 'chat': '5550100'}
    assert not any(path.endswith('/delivered') for path, _, _ in rig.client.writes)


def test_late_success_after_new_generation_cannot_commit_review(rig):
    rig.client.rows = result_rows(1)
    review_turn(rig)
    rig.guard.post_llm_call(session_id='owner-session', turn_id='turn-1', assistant_response='Answer')
    rig.state.persistent.run_generation += 1
    rig.guard.on_session_end(session_id='owner-session', turn_id='turn-1', completed=True, failed=False, interrupted=False)
    assert not rig.guard.rendered


def test_pending_review_receipt_is_cancelled_when_generation_changes(rig):
    rig.client.rows = result_rows(1)
    review_turn(rig)
    rig.guard.post_llm_call(session_id='owner-session', turn_id='turn-1', assistant_response='Answer')
    pending = []
    rig.guard.start_threads = True
    rig.guard._background = pending.append
    rig.guard.on_session_end(session_id='owner-session', turn_id='turn-1',
                            completed=True, failed=False, interrupted=False)
    assert len(pending) == 1
    rig.state.persistent.run_generation += 1
    pending[0]()
    assert not rig.guard.rendered
    assert not any(path.endswith('/review') for path, _, _ in rig.client.writes)


@pytest.mark.parametrize('blocked_path', ['review', 'communicate'])
def test_generation_change_during_review_response_cancels_remaining_receipts(rig, blocked_path):
    rig.client.rows = result_rows(2)
    for row in rig.client.rows:
        row['origin_ref'] = {'platform': 'sms', 'chat': '5550100', 'message_id': 'original'}
    review_turn(rig)
    rig.guard.post_llm_call(session_id='owner-session', turn_id='turn-1', assistant_response='Answer')
    response_waiting, respond = threading.Event(), threading.Event()
    original_emit = rig.runtime.emit
    def blocked_emit(path, body, key):
        result = original_emit(path, body, key)
        if path.endswith('/' + blocked_path):
            response_waiting.set()
            assert respond.wait(5)
        return result
    rig.runtime.emit = blocked_emit
    errors = []
    def settle():
        try:
            rig.guard.on_session_end(session_id='owner-session', turn_id='turn-1',
                                     completed=True, failed=False, interrupted=False)
        except BaseException as error:
            errors.append(error)
    worker = threading.Thread(target=settle)
    worker.start()
    try:
        assert response_waiting.wait(5)
        assert rig.guard.lock.acquire(timeout=1)
        try:
            rig.state.persistent.run_generation += 1
        finally:
            rig.guard.lock.release()
    finally:
        respond.set()
        worker.join(5)
    assert not worker.is_alive()
    assert not errors
    paths = [path for path, _, _ in rig.client.writes]
    expected = ['/v1/hands/obligations/o-0/review']
    if blocked_path == 'communicate':
        expected.append('/v1/hands/obligations/o-0/communicate')
    assert paths == expected
    assert not rig.guard.rendered


def test_arbitrary_internal_notification_is_not_a_duplicate_review(rig):
    rig.client.rows = result_rows(1)
    review_turn(rig)
    rig.guard.post_llm_call(session_id='owner-session', turn_id='turn-1', assistant_response='Answer')
    rig.guard.on_session_end(session_id='owner-session', turn_id='turn-1', completed=True, failed=False, interrupted=False)
    rig.agent._current_turn_id = 'turn-2'
    rig.guard.pre_llm_call(session_id='owner-session', turn_id='turn-2', user_message='Process exited with new evidence',
        platform='sms', sender_id='5550100', chat_id='5550100', session_key='route', conversation_history=[
            {'role': 'user', 'content': 'Process exited with new evidence', 'display_kind': 'internal_notification'}])
    assert rig.guard.request_caps('owner-session', 'turn-2')['mode'] == 'system'
    assert rig.guard.finalize_response('New answer', session_id='owner-session', turn_id='turn-2') is None


def test_controller_refusal_does_not_silence_future_review(rig):
    rig.client.rows = result_rows(1)
    rig.runtime.emit = lambda *args: {'error': 'controller_refused'}
    review_turn(rig)
    rig.guard.post_llm_call(session_id='owner-session', turn_id='turn-1', assistant_response='Answer')
    rig.guard.on_session_end(session_id='owner-session', turn_id='turn-1', completed=True, failed=False, interrupted=False)
    assert not rig.guard.rendered


def test_schema_timeout_inventory_and_unknown_tools(rig):
    owner_turn(rig)
    inventory = rig.guard.inventory_tools([
        {'name': 'mcp_wait', 'parameters': {'properties': {'timeout_seconds': {'type': 'integer'}}}},
        {'function': {'name': 'plugin_read', 'parameters': {'properties': {}}}},
    ])
    assert inventory['timeouts']['mcp_wait'] == 'timeout_seconds'
    assert 'plugin_read' in inventory['without_timeout']
    assert 'execute_code' in inventory['without_timeout']
    # The inventory is reported only: the timeout the caller asked for is left as it is.
    result = rig.guard.pre_tool_call(session_id='owner-session', turn_id='turn-1', tool_name='mcp_wait', args={'timeout_seconds': 90})
    assert result == {}
    background = rig.guard.pre_tool_call(session_id='owner-session', turn_id='turn-1', tool_name='terminal', args={'background': True, 'timeout': 120})
    assert background == {}


def test_stale_watchdog_disarms_after_agent_replacement(rig):
    owner_turn(rig)
    rig.gateway._running_agents['route'] = Agent(turn='replacement')
    rig.guard.tick('owner-session', 'turn-1')
    assert rig.guard._state('owner-session', 'turn-1').stopped.is_set()


def test_slow_controller_does_not_block_native_settlement(rig):
    import threading
    import time
    rig.client.rows = result_rows(1)
    review_turn(rig)
    rig.guard.start_threads = True
    entered, release = threading.Event(), threading.Event()
    def slow_emit(*args):
        entered.set()
        release.wait(2)
        return {'ok': True}
    rig.runtime.emit = slow_emit
    rig.guard.post_llm_call(session_id='owner-session', turn_id='turn-1', assistant_response='Answer')
    start = time.monotonic()
    try:
        rig.guard.on_session_end(session_id='owner-session', turn_id='turn-1', completed=True, failed=False, interrupted=False)
        assert time.monotonic() - start < 0.25
        assert entered.wait(0.5)
    finally:
        release.set()


def test_successful_render_receipts_survive_front_restart(rig, tmp_path):
    rig.runtime.spool_dir = tmp_path
    rig.client.rows = result_rows(1)
    review_turn(rig)
    rig.guard.post_llm_call(session_id='owner-session', turn_id='turn-1', assistant_response='Answer')
    rig.guard.on_session_end(session_id='owner-session', turn_id='turn-1', completed=True, failed=False, interrupted=False)
    restarted = rig.front.FrontGuard(rig.runtime, rig.ctx, clock=rig.clock, start_threads=False)
    try:
        assert restarted.rendered == rig.guard.rendered
        assert restarted.rendered
    finally:
        restarted.close()


def test_consolidated_digest_stays_bounded_and_names_every_rendered_result(rig):
    rig.client.rows = result_rows()
    for row in rig.client.rows:
        row['text'] = row['text'] + ' detail' * 300
    result = review_turn(rig)
    assert len(result['context']) <= 900
    assert all(f'Result {n}' in result['context'] for n in range(8))


def test_runtime_shutdown_retires_bridge_before_later_capture(rig):
    import sys
    import threading
    rig.runtime._closed = threading.Event()
    rig.runtime._closed.set()
    assert rig.guard.observe_turn(session_id='owner-session', turn_id='later', user_message='Hello') is None
    assert 'colonymind_hands_front' not in sys.modules
    assert rig.guard.capture_owner_text('owner-session', 'later', 'Hello') == 'Hello'


def provider_call(rig, turn='turn-1', during=None):
    """One provider call through the front's execution middleware; `during` runs while it is in flight."""
    def provider(request):
        if during:
            during()
        return 'provider-response'
    return rig.guard.llm_execution({}, provider, session_id='owner-session', turn_id=turn)


def test_owner_arrival_mid_review_request_drops_the_review_once_and_defers_it(rig):
    review_turn(rig)
    message = event('Next question', 'message-2')

    def arrive():
        rig.pending['route'] = message
        rig.guard.tick('owner-session', 'turn-1')
        rig.guard.tick('owner-session', 'turn-1')

    provider_call(rig, during=arrive)
    state = rig.guard._state('owner-session', 'turn-1')
    assert rig.agent.interrupts == 1 and state.request_dropped and state.owner_waiting
    assert not rig.agent.redirects and not rig.agent.steers
    assert rig.guard.metrics['review_request_dropped'] == 1
    assert len(rig.guard.deferred_reviews) == 1
    assert rig.pending.get('route') is message


def test_owner_turn_mid_request_is_never_cut_for_another_owner_message(rig):
    owner_turn(rig)

    def arrive():
        rig.pending['route'] = event('Second message', 'message-2')
        rig.clock.now = 600.0
        rig.guard.tick('owner-session', 'turn-1')

    provider_call(rig, during=arrive)
    assert rig.agent.interrupts == 0 and not rig.agent.redirects and not rig.agent.steers
    assert rig.guard.metrics['review_request_dropped'] == 0


def test_review_waits_for_its_tool_boundary_when_no_request_is_in_flight(rig):
    review_turn(rig)
    begin_tool(rig)
    rig.pending['route'] = event('Next question', 'message-2')
    rig.guard.tick('owner-session', 'turn-1')
    assert rig.agent.interrupts == 0 and rig.guard.metrics['review_request_dropped'] == 0


def test_faculty_off_keeps_a_review_request_untouched(rig):
    review_turn(rig)
    rig.runtime.config = {'mind': {'faculties': {'hands': False}}}

    def arrive():
        rig.pending['route'] = event('Next question', 'message-2')
        rig.guard.tick('owner-session', 'turn-1')

    provider_call(rig, during=arrive)
    assert rig.agent.interrupts == 0 and rig.guard.metrics['review_request_dropped'] == 0


def pre(rig, tool, call=None):
    return rig.guard.pre_tool_call(session_id='owner-session', turn_id='turn-1', tool_name=tool,
                                   tool_input={}, tool_call_id=call or '')


def test_hand_off_always_passes_and_is_not_counted(rig):
    # A hand-off passes on an ordinary turn and stays out of the turn's tool census; no other call is capped.
    owner_turn(rig)
    for n in range(5):
        assert pre(rig, 'delegate_task', f'hand-{n}').get('action') != 'block'
    for n in range(8):
        assert pre(rig, 'terminal', f'w-{n}').get('action') != 'block'
    assert pre(rig, 'terminal', 'w-9').get('action') != 'block'
    assert pre(rig, 'delegate_task', 'hand-after-nine-writes').get('action') != 'block'
    assert rig.guard._state('owner-session', 'turn-1').calls == {'read': 0, 'write': 9}


@pytest.mark.parametrize('flag', ('owner_waiting',))
def test_after_reply_now_only_a_few_hand_offs_pass(rig, flag):
    owner_turn(rig)
    state = next(iter(rig.guard.turns.values()))
    setattr(state, flag, True)
    assert pre(rig, 'terminal', 'w-1')['action'] == 'block'
    assert pre(rig, 'colonymind_hands', 'hands-1')['action'] == 'block'
    for n in range(3):
        assert pre(rig, 'delegate_task', f'hand-{n}').get('action') != 'block'
    assert pre(rig, 'delegate_task', 'hand-1').get('action') != 'block'  # the same call again is not a new one
    assert pre(rig, 'delegate_task', 'hand-3') == {'action': 'block', 'message': 'Reply now.'}


def test_a_duplicate_turn_never_hands_off(rig):
    owner_turn(rig)
    state = next(iter(rig.guard.turns.values()))
    state.duplicate = True
    assert pre(rig, 'delegate_task', 'hand-0')['action'] == 'block'
