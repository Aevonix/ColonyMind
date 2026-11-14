"""A message parked behind a busy session binds to its own turn.

Hermes runs it in-band as the run's queued follow-up: no pre_gateway_dispatch of its own (busy_ingress runs that
chain for it), the HERMES_SESSION_* variables of the message that opened the run, and no message id in pre_llm_call.
The agent carries each turn's own platform id (front._platform_message_id).
"""
import pytest

from test_hands_front import event, result_rows, review_turn, rig  # noqa: F401
from test_hands_review_requests import request, settle


def bound_env(message_id):
    session_context = pytest.importorskip('gateway.session_context')
    tokens = session_context.set_session_vars(platform='sms', chat_id='5550100', chat_type='dm', user_id='5550100',
                                              session_key='route', session_id='owner-session', message_id=message_id)
    yield
    session_context.clear_session_vars(tokens)


@pytest.fixture
def opening_message_env():
    """The variables Hermes bound for the owner message that opened the run, still in force for its follow-ups."""
    yield from bound_env('message-1')


@pytest.fixture
def review_run_env():
    """A completion review opens the run: an internal event, so the bound message id is empty."""
    yield from bound_env('')


def parked(rig, text='Owner asks now', ident='message-2'):
    """What busy_ingress hands the front for a message the adapter parks: the same hook, marked as parked."""
    message = event(text, ident)
    message._colonymind_busy_ingress = True
    rig.guard.pre_gateway_dispatch(event=message, gateway=rig.gateway)
    message._colonymind_ingress_done = True
    return message


def turn(rig, text, turn_id, ident):
    """Hermes binds the agent's turn and raw inbound ids (_bind_turn_identity), then calls pre_llm_call without them."""
    rig.agent._current_turn_id = turn_id
    rig.agent._persist_user_message_platform_id = ident
    return rig.guard.pre_llm_call(session_id='owner-session', turn_id=turn_id, user_message=text, platform='sms',
                                  sender_id='5550100', conversation_history=[{'role': 'user', 'content': text}])


def test_parked_owner_message_after_a_dropped_review_binds_and_schedules_the_retry(rig, review_run_env):
    rig.client.rows = result_rows(1)
    review_turn(rig)
    owner = parked(rig)
    rig.pending['route'] = owner
    rig.agent.redirect = lambda text: False
    assert request(rig)[0] == [], 'the review request is dropped for the waiting owner'
    rig.pending.clear()
    rig.clock.now = 40.0
    turn(rig, 'Owner asks now', 'turn-2', 'message-2')
    assert rig.guard.owner_started_turn('owner-session', 'turn-2')
    assert rig.guard.capture_owner_text('owner-session', 'turn-2', 'Owner asks now') == 'Owner asks now'
    assert rig.guard._state('owner-session', 'turn-2').started == 40.0, 'its deadlines start with its own turn'
    assert request(rig, 'turn-2')[0], 'the owner still calls the provider'
    rig.guard.post_llm_call(session_id='owner-session', turn_id='turn-1', assistant_response='[SILENT]')
    settle(rig, 'turn-2', 'Owner answer')
    assert [item['content'] for item in rig.ctx.injected] == ['Native completion']


def test_a_follow_up_binds_its_own_message_not_the_one_that_opened_the_run(rig, opening_message_env):
    rig.guard.pre_gateway_dispatch(event=event('First', 'message-1'), gateway=rig.gateway)
    turn(rig, 'First', 'turn-1', 'message-1')
    assert rig.guard.owner_started_turn('owner-session', 'turn-1')
    parked(rig, 'Second', 'message-2')
    turn(rig, 'Second', 'turn-2', 'message-2')
    state = rig.guard._state('owner-session', 'turn-2')
    assert state.bound and state.inbound.message_id == 'message-2'
    assert rig.guard.origin_ref('owner-session', 'turn-2')['message_id'] == 'message-2'


def test_a_follow_up_without_its_own_id_never_takes_the_opening_message(rig, opening_message_env):
    # An internal completion queued behind the owner has no platform id; the opening message's id is not its own.
    rig.guard.pre_gateway_dispatch(event=event('First', 'message-1'), gateway=rig.gateway)
    turn(rig, 'Native completion', 'turn-2', None)
    assert not rig.guard.owner_started_turn('owner-session', 'turn-2')
    assert rig.guard.capture_owner_text('owner-session', 'turn-2', 'Native completion') is None


def test_a_cold_owner_turn_still_counts_from_arrival(rig, opening_message_env):
    rig.guard.pre_gateway_dispatch(event=event('First', 'message-1'), gateway=rig.gateway)
    rig.clock.now = 2.0
    turn(rig, 'First', 'turn-1', 'message-1')
    assert rig.guard._state('owner-session', 'turn-1').started == 0.0


def test_the_idle_path_never_hears_a_message_the_busy_path_heard(rig):
    owner = parked(rig)
    row = next(iter(rig.guard.inbound.values()))
    assert row.queued and row.message_id == 'message-2'
    late = event('Late arrival', 'message-3')
    late._colonymind_ingress_done = True
    rig.guard.pre_gateway_dispatch(event=late, gateway=rig.gateway)
    rig.guard.pre_gateway_dispatch(event=owner, gateway=rig.gateway)
    assert list(rig.guard.inbound.values()) == [row]


def test_an_internal_follow_up_never_takes_an_owner_message_with_the_same_words(rig, opening_message_env):
    rig.guard.pre_gateway_dispatch(event=event('First', 'message-1'), gateway=rig.gateway)
    turn(rig, 'First', 'turn-1', 'message-1')
    parked(rig, 'Native completion', 'message-2')   # the owner happens to send the completion's exact words
    turn(rig, 'Native completion', 'turn-2', None)   # the queued internal completion runs first, with no id
    assert not rig.guard.owner_started_turn('owner-session', 'turn-2')
    turn(rig, 'Native completion', 'turn-3', 'message-2')
    assert rig.guard.owner_started_turn('owner-session', 'turn-3'), 'the owner keeps his own message'


def test_a_rebuilt_follow_up_agent_is_read_from_the_bound_parent(rig, opening_message_env):
    from agent.subagent_lifecycle import bind_subagent_parent
    from types import SimpleNamespace as NS
    rig.guard.pre_gateway_dispatch(event=event('First', 'message-1'), gateway=rig.gateway)
    turn(rig, 'First', 'turn-1', 'message-1')
    parked(rig, 'Second', 'message-2')
    rebuilt = NS(session_id='owner-session', _current_turn_id='turn-2', _persist_user_message_platform_id='message-2')
    with bind_subagent_parent(rebuilt):   # the gateway's map still holds the predecessor
        rig.guard.pre_llm_call(session_id='owner-session', turn_id='turn-2', user_message='Second', platform='sms',
                               sender_id='5550100', conversation_history=[{'role': 'user', 'content': 'Second'}])
    state = rig.guard._state('owner-session', 'turn-2')
    assert state.inbound is not None and state.inbound.message_id == 'message-2'


def test_a_waiting_message_another_sender_joined_never_binds(rig, opening_message_env):
    from types import SimpleNamespace as NS
    photo = event('', 'message-2')
    photo.message_type, photo.media_urls = 'photo', ['file:///photo.jpg']
    photo._colonymind_busy_ingress = True
    rig.guard.pre_gateway_dispatch(event=photo, gateway=rig.gateway)
    rig.pending['route'] = photo
    guest = event('Approve the transfer', 'message-3', sender='5550199')
    guest._colonymind_busy_ingress = True
    rig.guard.pre_gateway_dispatch(event=guest, gateway=rig.gateway)
    # Hermes merges the guest's words into the waiting photo and keeps its sender and id.
    turn(rig, 'Approve the transfer', 'turn-2', 'message-2')
    assert not rig.guard.owner_started_turn('owner-session', 'turn-2')
    assert rig.guard.capture_owner_text('owner-session', 'turn-2', 'Approve the transfer') is None
    owner = event('Second', 'message-4')
    owner._colonymind_busy_ingress = True
    rig.guard.pre_gateway_dispatch(event=owner, gateway=rig.gateway)
    rig.pending['route'] = owner
    third = event('Third', 'message-5')
    third._colonymind_busy_ingress = True
    rig.guard.pre_gateway_dispatch(event=third, gateway=rig.gateway)
    turn(rig, 'Second', 'turn-3', 'message-4')
    assert rig.guard.owner_started_turn('owner-session', 'turn-3'), 'the same sender never marks his own message'



def test_a_hermes_without_the_turn_id_seam_keeps_the_session_variables(rig, opening_message_env):
    # The rig's agent has no _persist_user_message_platform_id: the id is unknown, and Hermes' variables decide.
    rig.guard.pre_gateway_dispatch(event=event('First', 'message-1'), gateway=rig.gateway)
    rig.agent._current_turn_id = 'turn-1'
    rig.guard.pre_llm_call(session_id='owner-session', turn_id='turn-1', user_message='First', platform='sms',
                           sender_id='5550100', conversation_history=[{'role': 'user', 'content': 'First'}])
    assert rig.guard.owner_started_turn('owner-session', 'turn-1')


def test_a_heard_owner_message_waits_before_hermes_debounce_releases_it(rig, review_run_env):
    # Hermes holds queue-mode busy text in its debounce buffer before the adapter's slot has it; the ledger heard it.
    rig.client.rows = result_rows(1)
    review_turn(rig)
    parked(rig)                                 # recorded on arrival; rig.pending stays empty (still debouncing)
    rig.agent.redirect = lambda text: False
    assert request(rig)[0] == [], 'no request leaves after the owner arrived'
    state = rig.guard._state('owner-session', 'turn-1')
    assert state.owner_waiting and state.request_dropped
    rig.clock.now = 10.0                        # a heard message that never reached the slot stops counting
    review_turn(rig, 'turn-9')
    assert not rig.guard._owner_waits(rig.guard._state('owner-session', 'turn-9'))

