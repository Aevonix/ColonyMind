"""Only a confirmed visible tool reply to this chat marks the turn visible."""

import json

import pytest

from test_hands_front import begin_tool, owner_turn, rig  # noqa: F401


def delivered_send(rig, *, result=None, args=None, tool_call_id='send-1', **kwargs):
    rig.guard.post_tool_call(session_id='owner-session', turn_id='turn-1', tool_call_id=tool_call_id,
        tool_name='send_message', args=args or {'target': 'sms:5550100', 'message': 'Here is your answer.'},
        result={'success': True, 'message_id': 'sent-1'} if result is None else result, **kwargs)


@pytest.mark.parametrize('as_json', [False, True])
def test_confirmed_tool_reply_marks_visible_and_its_tool_boundary_interrupts_nothing(rig, as_json):
    owner_turn(rig)
    begin_tool(rig, 'send-1')
    rig.clock.now = 100
    result = {'success': True, 'message_id': 'sent-1'}
    delivered_send(rig, result=json.dumps(result) if as_json else result)
    assert rig.agent.interrupts == 0
    assert rig.guard._state('owner-session', 'turn-1').visible


@pytest.mark.parametrize('result', [
    {}, 'unreadable', [], {'success': False}, {'success': 'true'}, {'success': 1},
    {'success': True, 'skipped': True}, {'success': True, 'error': 'not delivered'},
])
def test_ambiguous_failed_or_skipped_send_does_not_mark_visible(rig, result):
    owner_turn(rig)
    delivered_send(rig, result=result)
    assert not rig.guard._state('owner-session', 'turn-1').visible


@pytest.mark.parametrize('args', [
    {'target': 'sms:other', 'message': 'A reply elsewhere'},
    {'target': 'whatsapp:5550100', 'message': 'A different platform'},
    {'target': 'sms', 'message': 'Unresolved home channel'},
    {'target': 'sms:5550100', 'message': '   '},
    {'target': 'sms:5550100', 'message': '[SILENT]'},
    {'target': 'sms:5550100', 'message': '<think>Not visible</think>'},
    {'target': 'sms:5550100', 'message': 'reaction', 'action': 'react'},
])
def test_unrelated_or_invisible_send_does_not_mark_visible(rig, args):
    owner_turn(rig)
    delivered_send(rig, args=args)
    assert not rig.guard._state('owner-session', 'turn-1').visible


def test_late_duplicate_send_receipt_is_ignored(rig):
    owner_turn(rig)
    delivered_send(rig)
    state = rig.guard._state('owner-session', 'turn-1')
    assert state.visible
    state.visible = False
    delivered_send(rig)
    assert not state.visible


def test_confirmed_receipt_marks_visible_even_when_pre_tool_observer_was_unavailable(rig):
    owner_turn(rig)
    assert not rig.guard._state('owner-session', 'turn-1').visible
    delivered_send(rig)
    assert rig.guard._state('owner-session', 'turn-1').visible
