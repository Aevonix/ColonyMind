"""Hand approval delivery waits for the parent review, regardless of earlier tools."""
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pytest

from hands_fake_hermes import hold_turn_lease
from colonymind_hermes.answers import Answers, ChatLedger, REPLY, TURN
from test_hands_front import event, rig  # noqa: F401
from test_hands_hold_completion import held_review
from test_hands_review_requests import settle


def stamp(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def setup_reader(tmp_path, monkeypatch, platform='rcs', deadline=1180):
    clock = [1000.0]
    monkeypatch.setattr('colonymind_hermes.answers.time.time', lambda: clock[0])
    ledger = ChatLedger(tmp_path / 'chat.db')
    reader = Answers(None, None, None, ledger)
    message = {'type': 'ask_notice', 'hand_approval_waits': [
        {'hand_id': 'h-0', 'held_at': stamp(1000), 'deadline': stamp(deadline)}]}
    return clock, ledger, reader, message, platform + ':owner-chat'


@pytest.mark.parametrize('platform', ['rcs', 'whatsapp'])
@pytest.mark.parametrize('prior,response', [
    ('none', '[SILENT]'), ('pwd', '[SILENT]'),
    ('ls -la fixture fixture/work', 'Reviewed'), ('ls -la fixture fixture/work', '[SILENT]'),
    ('touch report.txt', 'Reviewed'), ('touch report.txt', '[SILENT]'),
])
def test_ask_waits_for_its_parent_review_end_and_gap(rig, tmp_path, monkeypatch, platform, prior, response):
    clock, ledger, reader, message, target = setup_reader(tmp_path, monkeypatch, platform)
    ledger.record(target, REPLY, at=980)  # The dispatch reply has already settled.
    assert reader.hold(message, target), 'an idle parent chat is not a completed hand review'
    rig.runtime.effects = NS(has_result_to_review=lambda ident: prior not in {'none', 'pwd'})
    clock[0] = 1010
    held_review(rig, platform)
    assert reader.hold(message, target)
    rig.guard.post_llm_call(session_id='owner-session', turn_id='hold-review', assistant_response=response)
    clock[0] = 1020
    assert reader.hold(message, target), 'post_llm_call alone has not ended the native review'
    settle(rig, 'hold-review', response)
    assert reader.hold(message, target)
    clock[0] += 4.999
    assert reader.hold(message, target)
    clock[0] += .002
    assert not reader.hold(message, target)


@pytest.mark.parametrize('platform', ['rcs', 'whatsapp'])
def test_deferred_review_waits_for_its_silent_replay_and_gap(rig, tmp_path, monkeypatch, platform):
    clock, _, reader, message, target = setup_reader(tmp_path, monkeypatch, platform)
    rig.runtime.effects = NS(has_result_to_review=lambda ident: True)
    queued = event('Owner asks now', 'owner-message', sender='owner')
    queued.source.platform = platform
    queued.source.chat_id = 'owner-chat'
    rig.pending['route'] = queued
    held_review(rig, platform)
    assert rig.guard._state('owner-session', 'hold-review').owner_waiting
    settle(rig, 'hold-review')
    clock[0] += 6
    assert rig.guard.approval_review_ended('h-0') is None
    assert reader.hold(message, target), 'yielding to the owner did not handle the hand result'

    rig.pending.clear()
    rig.state.persistent.run_generation += 1
    hold_turn_lease(rig.gateway, rig.state)
    rig.agent._current_turn_id = 'retry-review'
    rig.guard.pre_llm_call(session_id='owner-session', turn_id='retry-review', user_message='Native completion',
        platform=platform, sender_id='owner', chat_id='owner-chat', session_key='route',
        conversation_history=[{'role': 'user', 'content': 'Native completion',
                               'display_kind': 'internal_notification'}])
    assert not rig.guard.is_silent_hold_review('owner-session', 'retry-review')
    settle(rig, 'retry-review', '[SILENT]')
    assert rig.guard.approval_review_ended('h-0') == clock[0]
    assert reader.hold(message, target)
    clock[0] += 4.999
    assert reader.hold(message, target)
    clock[0] += .002
    assert not reader.hold(message, target)


@pytest.mark.parametrize('deadline', [1060, 1180])
@pytest.mark.parametrize('review', ['missing', 'hung'])
def test_missing_or_hung_completion_releases_at_cap(rig, tmp_path, monkeypatch, deadline, review):
    clock, ledger, reader, message, target = setup_reader(tmp_path, monkeypatch, deadline=deadline)
    if review == 'hung':
        rig.runtime.effects = NS(has_result_to_review=lambda ident: True)
        held_review(rig)
        ledger.record(target, TURN)
    clock[0] = deadline - .001
    assert reader.hold(message, target)
    clock[0] = deadline
    assert not reader.hold(message, target), 'a missing completion cannot extend the hand wall budget'


def test_other_hand_review_cannot_release_ask(rig, tmp_path, monkeypatch):
    clock, _, reader, message, target = setup_reader(tmp_path, monkeypatch)
    held_review(rig)
    settle(rig, 'hold-review')
    clock[0] += 6
    message['hand_approval_waits'][0]['hand_id'] = 'h-unrelated'
    assert reader.hold(message, target)


@pytest.mark.parametrize('platform', ['rcs', 'whatsapp', 'sms'])
@pytest.mark.parametrize('response', ['[SILENT]', 'The hand needs approval.'])
def test_review_settlement_hands_over_without_another_chat_wait(rig, tmp_path, monkeypatch, platform, response):
    clock, ledger, reader, message, target = setup_reader(tmp_path, monkeypatch, platform)
    ledger.record(target, TURN)
    held_review(rig, platform)
    clock[0] = 1010
    settle(rig, 'hold-review', response)
    assert reader.hold(message, target)
    clock[0] += 4.999
    assert reader.hold(message, target)
    clock[0] += .002
    assert not reader.hold(message, target), 'the settled native review owns this notice gap'


def test_missing_front_bridge_still_waits_but_only_to_cap(tmp_path, monkeypatch):
    import sys
    monkeypatch.delitem(sys.modules, 'colonymind_hands_front', raising=False)
    clock, _, reader, message, target = setup_reader(tmp_path, monkeypatch)
    assert reader.hold(message, target)
    clock[0] = 1180
    assert not reader.hold(message, target)


@pytest.mark.parametrize('kind', ['ask_show', 'ask_clarify'])
def test_recovery_after_hold_deadline_still_follows_owner_reply(tmp_path, monkeypatch, kind):
    clock, ledger, reader, message, target = setup_reader(tmp_path, monkeypatch)
    message['type'] = kind
    clock[0] = 1200
    ledger.record(target, TURN)
    assert reader.hold(message, target)
    clock[0] += 10
    ledger.record(target, REPLY)
    assert reader.hold(message, target)
    clock[0] += 5.001
    assert not reader.hold(message, target)


@pytest.mark.parametrize('response,prior_work', [('[SILENT]', False), ('Reviewed', True), ('[SILENT]', True)])
def test_ask_gap_starts_at_owned_lease_release_after_session_end(rig, tmp_path, monkeypatch, response, prior_work):
    clock, _, reader, message, target = setup_reader(tmp_path, monkeypatch)
    rig.runtime.effects = NS(has_result_to_review=lambda ident: prior_work)
    held_review(rig, 'rcs')
    rig.guard.post_llm_call(session_id='owner-session', turn_id='hold-review', assistant_response=response)
    rig.guard.on_session_end(session_id='owner-session', turn_id='hold-review',
                             completed=True, failed=False, interrupted=False)
    clock[0] += 30
    assert rig.guard.approval_review_ended('h-0') is None
    assert reader.hold(message, target), 'finalization still owns the native lease'
    assert rig.gateway._release_turn_lease('route', 1) is True
    assert rig.guard.approval_review_ended('h-0') == clock[0]
    clock[0] += 4.999
    assert reader.hold(message, target)
    clock[0] += .002
    assert not reader.hold(message, target)
    settled_at = rig.guard.approval_review_ended('h-0')
    assert rig.gateway._release_turn_lease('route', 1) is False
    assert rig.guard.approval_review_ended('h-0') == settled_at


@pytest.mark.parametrize('unfinished', ['running', 'interrupted', 'stale', 'replaced_agent', 'advanced_turn'])
def test_missing_or_stale_native_review_outcome_cannot_release_ask(rig, tmp_path, monkeypatch, unfinished):
    clock, _, reader, message, target = setup_reader(tmp_path, monkeypatch)
    held_review(rig, 'rcs')
    if unfinished == 'interrupted':
        rig.guard.agent_loop_stopped(session_id='owner-session', turn_id='hold-review')
    if unfinished == 'stale':
        rig.guard.on_session_end(session_id='owner-session', turn_id='hold-review',
                                 completed=True, failed=False, interrupted=False)
        rig.state.persistent.run_generation += 1
    if unfinished in {'replaced_agent', 'advanced_turn'}:
        if unfinished == 'replaced_agent':
            rig.gateway._running_agents['route'] = object()
        else:
            rig.agent._current_turn_id = 'next-turn'
        rig.guard.on_session_end(session_id='owner-session', turn_id='hold-review',
                                 completed=True, failed=False, interrupted=False)
    rig.gateway._release_turn_lease('route', 1)
    clock[0] += 30
    assert rig.guard.approval_review_ended('h-0') is None
    assert reader.hold(message, target)


@pytest.mark.parametrize('outcome', ['early_failure', 'failed_finalization', 'silent_early_exit'])
def test_failed_review_releases_ask_only_after_selected_native_cleanup(rig, tmp_path, monkeypatch, outcome):
    clock, _, reader, message, target = setup_reader(tmp_path, monkeypatch)
    rig.runtime.effects = NS(has_result_to_review=lambda ident: outcome != 'silent_early_exit')
    held_review(rig, 'rcs')
    state = rig.guard._state('owner-session', 'hold-review')
    if outcome == 'failed_finalization':
        rig.guard.on_session_end(session_id='owner-session', turn_id='hold-review',
                                 completed=False, failed=True, interrupted=False)
    # Both native sources have failure returns that skip on_session_end. The
    # actual selected cleanup method removes the agent before lease release.
    assert rig.gateway._release_running_agent_state('route', run_generation=1)
    assert 'route' not in rig.gateway._running_agents
    assert rig.gateway._release_running_agent_state('route', run_generation=1)
    clock[0] += 30
    assert reader.hold(message, target)
    assert rig.guard.approval_review_ended('h-0') is None
    assert rig.gateway._release_turn_lease('route', 1)
    assert rig.guard.approval_review_ended('h-0') == clock[0]
    assert state.outcome_recorded is (outcome == 'failed_finalization')
    assert not rig.guard.rendered
    assert not [path for path, _, _ in rig.client.writes if path.endswith(('/review', '/communicate'))]
    clock[0] += 5.001
    assert not reader.hold(message, target)


@pytest.mark.parametrize('invalid', ['unarmed', 'unbound', 'owner_waiting', 'stopped_before', 'stale_agent',
                                   'stopped_after', 'advanced_turn', 'rotated_session', 'stale_generation', 'release_failed'])
def test_early_review_cleanup_never_releases_unverified_or_stale_asks(rig, tmp_path, monkeypatch, invalid):
    clock, _, reader, message, target = setup_reader(tmp_path, monkeypatch)
    held_review(rig, 'rcs')
    state = rig.guard._state('owner-session', 'hold-review')
    if invalid == 'unarmed':
        state.armed = False
    elif invalid == 'unbound':
        state.agent = None
    elif invalid == 'owner_waiting':
        state.owner_waiting = True
    elif invalid == 'stopped_before':
        rig.guard.agent_loop_stopped(session_id='owner-session', turn_id='hold-review')
    elif invalid == 'stale_agent':
        rig.gateway._running_agents['route'] = object()
    rig.gateway._release_running_agent_state('route', run_generation=1)
    if invalid == 'stopped_after':
        rig.guard.agent_loop_stopped(session_id='owner-session', turn_id='hold-review')
    elif invalid == 'advanced_turn':
        rig.agent._current_turn_id = 'successor'
    elif invalid == 'rotated_session':
        rig.agent.session_id = 'successor-session'
    elif invalid == 'stale_generation':
        rig.state.persistent.run_generation += 1
    elif invalid == 'release_failed':
        rig.gateway._turn_leases.result = False
    rig.gateway._release_turn_lease('route', 1)
    clock[0] += 30
    assert rig.guard.approval_review_ended('h-0') is None
    assert reader.hold(message, target)
