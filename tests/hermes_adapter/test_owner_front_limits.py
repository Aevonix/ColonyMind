"""The installed owner front preserves caller limits and lets long work finish."""
from importlib import import_module
from copy import deepcopy

import pytest

from test_hands_front import begin_tool, end_tool, owner_turn, rig  # noqa: F401


@pytest.mark.parametrize('payload', [
    {}, {'max_tokens': 4096}, {'max_completion_tokens': 16384},
    {'extra_body': {'max_output_tokens': 32768}},
    {'chat_template_kwargs': {'enable_thinking': True}},
])
def test_owner_request_keeps_caller_bounds_and_thinking(rig, payload):
    owner_turn(rig)
    middleware = import_module(rig.front.__package__ + '.middleware')
    request = {'model': 'fixture', 'messages': [{'role': 'user', 'content': 'Work through this'}],
               'extra_body': {}, **payload}
    expected = deepcopy(request)
    assert middleware._front_request(request, 'owner-session', turn_id='turn-1') == expected


@pytest.mark.parametrize('tool,bucket', [('terminal', 'write'), ('tool_search', 'read')])
def test_owner_tool_calls_are_counted_without_a_cap(rig, tool, bucket):
    owner_turn(rig)
    for index in range(20):
        result = rig.guard.pre_tool_call(session_id='owner-session', turn_id='turn-1',
            tool_name=tool, tool_call_id=f'call-{index}', args={})
        assert result == {}
    state = rig.guard._state('owner-session', 'turn-1')
    assert state.calls[bucket] == 20


@pytest.mark.parametrize('seconds', [30, 120])
def test_owner_work_reaches_tool_boundary_without_yield_or_interrupt(rig, seconds):
    owner_turn(rig)
    begin_tool(rig)
    rig.clock.now = seconds
    rig.guard.tick('owner-session', 'turn-1')
    end_tool(rig)
    assert not rig.agent.steers and not rig.agent.redirects
    assert rig.agent.interrupts == 0
    assert not rig.ctx.injected


@pytest.mark.parametrize('arguments', [{}, {'timeout': 180}])
def test_owner_tool_timeout_is_left_to_the_caller(rig, arguments):
    owner_turn(rig)
    rig.guard.timeout_inventory = dict(rig.front.TIMEOUTS)
    assert rig.guard.pre_tool_call(session_id='owner-session', turn_id='turn-1',
        tool_name='terminal', tool_call_id='call', args=arguments) == {}
