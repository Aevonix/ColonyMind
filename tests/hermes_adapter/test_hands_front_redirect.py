"""A follow-up that redirects a running turn stays the owner's when the front heard it.

Hermes' redirect cancels the model request a follow-up arrives during and joins the follow-up to the turn's user
message after a line of its own (CORRECTION); follow-ups that arrive before it applies are joined to each other after
a second line (ADDITIONAL). Capture keeps such a turn only when every follow-up is a message the front heard from the
turn's sender, in the turn's chat, on the busy path while the turn ran.
"""
import pytest

from colonymind_shared.midturn import finish_ingress, seal_rewrite
from test_hands_front import event, owner_turn, review_turn, rig  # noqa: F401

# The lines as Hermes writes them, spelled out here rather than read from the front.
CORRECTION = '\n\nUser correction during the turn: '
ADDITIONAL = '\n\n[Additional user correction]\n'


def heard(rig, text, ident, sender='5550100', chat='5550100'):
    """A follow-up the busy path hears (busy_ingress marks it) while the turn runs."""
    message = event(text, ident, sender=sender)
    message.source.chat_id = chat
    message._colonymind_busy_ingress = True
    rig.guard.pre_gateway_dispatch(event=message, gateway=rig.gateway)
    message._colonymind_ingress_done = True
    return message


def redirected(rig, *follow_ups, head='Owner asks', joiner=None):
    return head + CORRECTION + (joiner or ADDITIONAL).join(follow_ups)


def capture(rig, text, turn='turn-1'):
    return rig.guard.capture_owner_text('owner-session', turn, text)


def test_a_redirected_turn_keeps_his_message_and_the_follow_up_the_front_heard(rig):
    owner_turn(rig)
    rig.clock.now = 5.0
    heard(rig, 'Make it Sunday', 'message-2')
    rows = [(row.message_id, row.consumed) for row in rig.guard.inbound.values()]
    assert capture(rig, redirected(rig, 'Make it Sunday')) == 'Owner asks\n\nMake it Sunday'
    assert capture(rig, redirected(rig, 'Make it Sunday')) == 'Owner asks\n\nMake it Sunday', 'nothing is consumed'
    assert [(row.message_id, row.consumed) for row in rig.guard.inbound.values()] == rows
    assert rig.guard.metrics['capture_refused'] == 0
    assert rig.guard.owner_started_turn('owner-session', 'turn-1')


@pytest.mark.parametrize('joiner', [ADDITIONAL, CORRECTION], ids=['before', 'after'])
def test_follow_ups_joined_before_or_after_a_redirect_applied_are_each_his(rig, joiner):
    owner_turn(rig)
    rig.clock.now = 5.0
    heard(rig, 'Make it Sunday', 'message-2')
    heard(rig, 'and after nine', 'message-3')
    text = redirected(rig, 'Make it Sunday', 'and after nine', joiner=joiner)
    assert capture(rig, text) == 'Owner asks\n\nMake it Sunday\n\nand after nine'


def test_one_heard_message_stands_for_one_follow_up(rig):
    owner_turn(rig)
    rig.clock.now = 5.0
    heard(rig, 'ok', 'message-2')
    assert capture(rig, redirected(rig, 'ok', 'ok')) is None
    heard(rig, 'ok', 'message-3')
    assert capture(rig, redirected(rig, 'ok', 'ok')) == 'Owner asks\n\nok\n\nok'


def test_hermes_notes_ahead_of_his_message_stay_with_a_redirect(rig):
    rig.guard.pre_gateway_dispatch(event=event('yes'), gateway=rig.gateway)
    head = '[Replying to: "Approve the change?"]\n\nyes'
    rig.guard.pre_llm_call(session_id='owner-session', turn_id='turn-1', user_message=head, platform='sms',
                           sender_id='5550100', message_id='message-1', session_key='route')
    rig.clock.now = 5.0
    heard(rig, 'only the first part', 'message-2')
    assert capture(rig, redirected(rig, 'only the first part', head=head)) == head + '\n\nonly the first part'


def test_a_follow_up_sealed_after_the_front_heard_it_reads_as_sealed(rig):
    owner_turn(rig)
    rig.clock.now = 5.0
    message = event('my code is 4417', 'message-2')
    message._colonymind_busy_ingress = True
    rig.guard.pre_gateway_dispatch(event=message, gateway=rig.gateway)
    message.text = seal_rewrite(message, lambda text: 'my code is [sealed]')
    assert capture(rig, redirected(rig, 'my code is [sealed]')) == 'Owner asks\n\nmy code is [sealed]'


def test_a_merged_turn_takes_a_redirect_like_any_other(rig):
    for words, ident in (('First', 'message-1'), ('Second', 'message-2')):
        rig.guard.pre_gateway_dispatch(event=event(words, ident), gateway=rig.gateway)
    envelope = 'First\n\nSecond'
    members = [{'message_id': 'message-1', 'text': 'First'}, {'message_id': 'message-2', 'text': 'Second'}]
    history = [{'role': 'user', 'content': envelope, 'display_metadata': {
        'midturn_composition': {'version': 1, 'text': envelope, 'messages': members}}}]
    rig.guard.pre_llm_call(session_id='owner-session', turn_id='turn-1', user_message=envelope, platform='sms',
                           sender_id='5550100', message_id='message-2', session_key='route',
                           conversation_history=history)
    rig.clock.now = 5.0
    heard(rig, 'Third', 'message-3')
    assert capture(rig, redirected(rig, 'Third', head=envelope)) == 'First\n\nSecond\n\nThird'


def refuse_unheard(rig):
    pass


def refuse_idle_path(rig):
    rig.guard.pre_gateway_dispatch(event=event('Make it Sunday', 'message-2'), gateway=rig.gateway)


def refuse_other_sender(rig):
    heard(rig, 'Make it Sunday', 'message-2', sender='5550199')


def refuse_other_chat(rig):
    heard(rig, 'Make it Sunday', 'message-2', chat='5550111')


def refuse_heard_before_the_turn(rig):
    rig.clock.now = 0.0
    heard(rig, 'Make it Sunday', 'message-0')


def refuse_revoked_ingress(rig):
    message = heard(rig, 'Make it Sunday', 'message-2')
    message.text = 'Make it Sunday, unrecorded'
    finish_ingress(message._colonymind_text_receipt, message)


def refuse_mixed(rig):
    message = heard(rig, 'Make it Sunday', 'message-2')
    rig.pending['route'] = message
    heard(rig, 'Approve the transfer', 'message-3', sender='5550199')


def refuse_taken_by_another_turn(rig):
    heard(rig, 'Make it Sunday', 'message-2')
    owner_turn(rig, 'Make it Sunday', turn='turn-2', ident='message-2')


@pytest.mark.parametrize('arrange', [refuse_unheard, refuse_idle_path, refuse_other_sender, refuse_other_chat,
                                     refuse_heard_before_the_turn, refuse_revoked_ingress, refuse_mixed,
                                     refuse_taken_by_another_turn])
def test_a_follow_up_the_front_did_not_hear_from_him_during_the_turn_is_refused(rig, arrange):
    rig.clock.now = 2.0
    owner_turn(rig)
    rig.clock.now = 5.0
    arrange(rig)
    assert capture(rig, redirected(rig, 'Make it Sunday')) is None
    assert rig.guard.metrics['capture_refused'] == 1
    assert capture(rig, 'Owner asks') == 'Owner asks', 'his own message alone is still his'


def test_a_turn_he_did_not_start_never_takes_a_redirect(rig):
    review_turn(rig)
    heard(rig, 'Make it Sunday', 'message-2')
    assert capture(rig, redirected(rig, 'Make it Sunday', head='Native completion')) is None
    owner_turn(rig, turn='turn-2', ident='message-3')
    assert capture(rig, redirected(rig, 'Make it Sunday', head='Someone else asks'), 'turn-2') is None
