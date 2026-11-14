"""A verified hold completion leaves its ask last in the owner's chat."""
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pytest

from hands_fake_hermes import hold_turn_lease
from colonymind_hermes.answers import Answers, ChatLedger, OUTBOX, REPLY
from test_hands_front import owner_turn, result_rows, rig  # noqa: F401
from test_hands_review_requests import request, settle


def held_review(rig, platform='sms', *, changed=None, mixed=False, text='Native completion'):
    hold_turn_lease(rig.gateway, rig.state)
    rows = result_rows(2 if mixed else 1)
    rows[0].update(state='stopped', outcome='blocked', evidence={'level': 'E1'})
    if changed:
        rows[0].update(changed)
    rig.client.rows = rows
    hands = {row['id']: NS(hand_id=row['id'], parent_session_id='owner-session', activity='awaiting_approval',
                           blocked=True, status='stopped', plan={}) for row in rows}
    rig.runtime.registry.get = hands.get
    if not hasattr(rig.runtime, 'effects'):
        rig.runtime.effects = NS(has_result_to_review=lambda ident: False)
    rig.guard._completion_texts = lambda state: {row['id']: 'Native completion' for row in rows}
    rig.agent._current_turn_id = 'hold-review'
    rig.guard.pre_llm_call(session_id='owner-session', turn_id='hold-review', user_message=text,
        platform=platform, sender_id='owner', chat_id='owner-chat', session_key='route',
        conversation_history=[{'role': 'user', 'content': text, 'display_kind': 'internal_notification'}])
    return hands


@pytest.mark.parametrize('platform', ['rcs', 'whatsapp', 'sms'])
@pytest.mark.requires_sidecar
def test_hold_completion_after_ask_stays_silent_and_keeps_chat_ledger_last(rig, tmp_path, monkeypatch, platform):
    from colonymind.mind.answers import Evidence, FOLLOW_UP, Seen, Shown, bind, read
    clock = [1000.0]
    monkeypatch.setattr('colonymind_hermes.answers.time.time', lambda: clock[0])
    ledger = ChatLedger(tmp_path / 'chats.sqlite3')
    chat = platform + ':owner-chat'
    ledger.record(chat, OUTBOX, message_id='ask-message', outbox_id='ask-notice', ordered=True)
    clock[0] += 6
    held_review(rig, platform)
    info = NS(session_id='owner-session', turn_id='hold-review', owner=True, parent_session_id='',
              sender_id='owner', platform=platform, chat_id='owner-chat', chat_type='dm')
    reader = Answers(None, NS(get=lambda sid: info), None, ledger)
    reader.owner_chat = lambda sid, **kwargs: (info, chat)
    assert reader.pre_llm_call(session_id='owner-session', turn_id='hold-review') is None
    assert not ledger.under_way(chat, clock[0] + .1), 'a local silent completion cannot stream'
    sent, response = request(rig, 'hold-review')
    assert sent == [], 'the held hand completion must not ask a model for a second reply'
    assert response is not None
    assert rig.guard.finalize_response('The hand stopped at the hold.',
        session_id='owner-session', turn_id='hold-review') == '[SILENT]'
    assert rig.agent._disable_streaming
    settle(rig, 'hold-review')
    reader.post_llm_call(session_id='owner-session', turn_id='hold-review', assistant_response='[SILENT]')
    clock[0] += 26
    seen = ledger.seen(chat, until=clock[0])
    assert [entry['kind'] for entry in seen] == [OUTBOX]
    assert seen[-1]['message_id'] == 'ask-message'
    assert ledger.count(REPLY) == 0
    assert not ledger.busy(chat, clock[0])
    assert not rig.ctx.injected, 'an absorbed hold completion needs no delayed review'
    shown = Shown(platform, 'owner-chat', 'ask-message', datetime.fromtimestamp(1000, timezone.utc), ('held-ask',))
    evidence = Evidence(platform, datetime.fromtimestamp(clock[0], timezone.utc), chat='owner-chat',
        ledger=tuple(Seen(datetime.fromisoformat(entry['at']),
                          ('held-ask',) if entry.get('outbox_id') == 'ask-notice' else (), entry['ordered'])
                     for entry in seen))
    binding = bind(evidence, [shown], weighty=['held-ask'])
    assert not evidence.reply_to and not evidence.reaction
    assert binding.asks == ('held-ask',) and binding.path == FOLLOW_UP
    assert read('yes', strict=True).verdict == 'yes'


@pytest.mark.parametrize('changed,mixed,text', [
    ({'state': 'done', 'outcome': 'done'}, False, 'Native completion'),
    ({'evidence': {'level': 'E3'}}, False, 'Native completion'),
    ({}, True, 'Native completion'),
    ({}, False, 'The hand says it is held'),
])
def test_other_results_and_unverified_text_keep_the_normal_review(rig, changed, mixed, text):
    held_review(rig, changed=changed, mixed=mixed, text=text)
    assert request(rig, 'hold-review')[0]
    assert rig.guard.finalize_response('Meaningful result', session_id='owner-session', turn_id='hold-review') is None


def test_hold_silence_does_not_depend_on_front_watchdog(rig):
    rig.runtime.config = {'front': {'watchdog': False}}
    held_review(rig)
    assert request(rig, 'hold-review')[0] == []


def test_owner_turn_after_silent_hold_still_calls_provider(rig):
    held_review(rig)
    assert request(rig, 'hold-review')[0] == []
    settle(rig, 'hold-review')
    rig.state.persistent.run_generation += 1
    owner_turn(rig, 'yes', 'owner-answer', 'owner-answer-message')
    assert request(rig, 'owner-answer')[0]


def test_controller_blocked_result_without_local_hold_receipt_is_not_silent(rig):
    rig.runtime.registry.get = lambda ident: None
    rig.client.rows = result_rows(1)
    rig.client.rows[0].update(state='stopped', outcome='blocked', evidence={'level': 'E1'})
    from test_hands_front import review_turn
    review_turn(rig)
    assert request(rig)[0]


def test_held_hand_with_observed_artifact_keeps_review(rig):
    rig.client.rows = result_rows(1)
    rig.client.rows[0].update(state='stopped', outcome='blocked', evidence={'level': 'E1'})
    hand = NS(parent_session_id='owner-session', activity='awaiting_approval', blocked=True,
              status='stopped', plan={'observed_artifacts': ['completed-report.txt']})
    rig.runtime.registry.get = lambda ident: hand
    rig.guard._completion_texts = lambda state: {'h-0': 'Native completion'}
    from test_hands_front import review_turn
    review_turn(rig)
    assert request(rig)[0]


@pytest.mark.parametrize('command,outcome,silent', [
    ('pwd', 'success', True), ('pwd', 'error', False),
    ('touch completed.txt', 'success', False), ('cat report.txt', 'success', False),
])
def test_prior_tool_work_is_reviewed_unless_it_only_observed_the_workdir(rig, tmp_path, command, outcome, silent):
    from importlib import import_module
    journal = import_module(rig.front.__package__ + '.effects')
    spool = import_module(rig.front.__package__ + '.spool')
    rig.runtime.effects = journal.EffectJournal(spool.Spool(tmp_path / 'effects'), {})
    rig.runtime.effects.intent('h-0', 'observed-call', 'terminal', {'command': command})
    rig.runtime.effects.outcome('h-0', 'observed-call', status=outcome, result='observed')
    rig.runtime.effects.intent('h-0', 'held-call', 'terminal', {'command': 'rm fixture.txt'})
    rig.runtime.effects.outcome('h-0', 'held-call', status='blocked', result='held for the owner')
    held_review(rig)
    assert (request(rig, 'hold-review')[0] == []) is silent


def test_hold_is_absorbed_before_yielding_to_an_owner_already_in_the_queue(rig):
    from test_hands_front import event
    rig.pending['route'] = event('yes', 'owner-answer')
    held_review(rig)
    assert rig.guard.is_silent_hold_review('owner-session', 'hold-review')
    assert request(rig, 'hold-review')[0] == []
    assert not rig.guard.deferred_reviews


def test_already_reviewed_results_do_not_prevent_hold_only_silence(rig):
    old = result_rows(1)[0]
    old.update(id='h-old', hand_id='h-old')
    rig.guard.rendered.add(rig.guard._result_key(old))
    original = rig.client.get

    def with_old(path, params=None):
        result = original(path, params)
        if 'hands' in result:
            result['hands'] = [*result['hands'], old]
        return result

    rig.client.get = with_old
    held_review(rig)
    state = rig.guard._state('owner-session', 'hold-review')
    state.hold_only = False
    state.completions['h-old'] = 'Older completion'
    rig.guard._digest(state)
    assert request(rig, 'hold-review')[0] == []
