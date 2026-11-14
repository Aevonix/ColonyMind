"""The owner request is unchanged; review and system bounds remain selective."""
from copy import deepcopy
from importlib import import_module

import pytest

from test_hands_front import begin_tool, owner_turn, review_turn, rig  # noqa: F401


@pytest.mark.parametrize('field', ['max_tokens', 'max_completion_tokens', 'max_output_tokens'])
@pytest.mark.parametrize('nested', [False, True])
def test_owner_bound_above_32768_is_unchanged(rig, field, nested):
    owner_turn(rig)
    middleware = import_module(rig.front.__package__ + '.middleware')
    request = {'extra_body': {}}
    body = request['extra_body'] if nested else request
    body[field] = 65536
    expected = deepcopy(request)
    assert middleware._front_request(request, 'owner-session', turn_id='turn-1') == expected


@pytest.mark.parametrize('mode', ['review_result', 'system'])
@pytest.mark.parametrize('bound,expected', [(65536, 512), (128, 128), (None, None)])
def test_review_and_system_only_lower_supplied_bounds(rig, mode, bound, expected):
    owner_turn(rig)
    rig.guard._state('owner-session', 'turn-1').mode = mode
    middleware = import_module(rig.front.__package__ + '.middleware')
    request = {'extra_body': {}}
    if bound is not None:
        request['max_tokens'] = bound
    result = middleware._front_request(request, 'owner-session', turn_id='turn-1')
    if expected is None:
        assert 'max_tokens' not in result
    else:
        assert result['max_tokens'] == expected


def test_contact_request_is_outside_the_front_caps(rig):
    owner_turn(rig)
    rig.runtime.authorized_owner = lambda sid: False
    middleware = import_module(rig.front.__package__ + '.middleware')
    request = {'max_tokens': 65536}
    expected = deepcopy(request)
    assert rig.guard.request_caps('owner-session', 'turn-1') is None
    assert middleware._front_request(request, 'owner-session', turn_id='turn-1') == expected


@pytest.mark.parametrize('mode', ['answer_now', 'review_result', 'system'])
def test_request_without_extra_body_keeps_it_absent(rig, mode):
    owner_turn(rig)
    rig.guard._state('owner-session', 'turn-1').mode = mode
    middleware = import_module(rig.front.__package__ + '.middleware')
    request = {'messages': [{'role': 'user', 'content': 'Reply'}]}
    expected = deepcopy(request)
    assert middleware._front_request(request, 'owner-session', turn_id='turn-1') == expected
    assert 'extra_body' not in request


def test_confirmed_tool_reply_suppresses_review_yield(rig):
    review_turn(rig)
    state = rig.guard._state('owner-session', 'turn-1')
    begin_tool(rig, 'reply')
    rig.clock.now = 100
    rig.guard.post_tool_call(session_id='owner-session', turn_id='turn-1', tool_call_id='reply',
        tool_name='send_message', args={'target': state.platform + ':' + state.chat, 'message': 'Finished reply'},
        result={'success': True, 'message_id': 'sent-1'})
    assert state.visible
    assert not rig.agent.steers and not rig.agent.redirects and not rig.agent.interrupts
