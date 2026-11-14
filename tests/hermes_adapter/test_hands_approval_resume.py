"""Owner grants resume the executor without a model deciding to continue it."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
import importlib
import sys

import pytest

from hands_fake_hermes import Parent
from test_hands_dispatch import setup  # noqa: F401


def held(setup, *, platform='sms', workdir=None):
    runtime, native, client = setup
    native.platform = platform
    runtime.config['platforms'] = [platform]
    parent = Parent()
    task = {'goal': 'Remove the approved fixture', 'kind': 'coding', **({'workdir': workdir} if workdir else {})}
    runtime.dispatch_tasks([task], parent=parent)
    hand = runtime.registry.all()[0]
    hand.blocked, hand.activity = True, 'awaiting_approval'
    grant = {'permit_id': 'permit-one', 'ask_id': 'ask-one', 'hand_id': hand.hand_id,
             'parent_session': parent.session_id, 'state': 'granted',
             'call': {'tool': 'terminal', 'arguments': {'command': 'rm /fixture/report.txt'}},
             'workdir': '/fixture'}
    grants = [grant]
    original = client.get
    client.get = lambda path, params=None: ({'permits': list(grants), 'refreshed': True}
        if path == '/v1/hands/permits/held' else original(path, params))
    return runtime, native, client, parent, hand, grant, grants


def finish(runtime, native, hand):
    runtime.subagent_stop(child_session_id=hand.child_session_id, child_summary='Held for the owner')
    native.live[hand.delegation_id]['status'] = 'completed'
    # Native active_count counts only live slots in production.
    native.active_count = lambda: sum(row['status'] == 'running' for row in native.live.values())


@pytest.mark.parametrize('platform', ['sms', 'rcs', 'whatsapp', 'telegram'])
def test_grant_resumes_once_with_exact_call_and_normal_completion(setup, platform):
    runtime, native, client, parent, hand, grant, _ = held(setup, platform=platform)
    finish(runtime, native, hand)
    parent._current_turn_id = 'answer-turn'
    runtime.remember_parent(parent)
    line = runtime.resume_approvals(parent.session_id, [grant])
    assert 'automatically' in line and 'Do not continue' in line and 'perform the action yourself' in line
    runtime.maintain()
    resumed = runtime.registry.all()[-1]
    assert resumed is not hand and len(native.calls) == 2
    assert resumed.plan['task']['continue'] == hand.hand_id
    assert 'rm /fixture/report.txt' in resumed.plan['task']['context']
    assert resumed.parent_session_id == parent.session_id
    assert resumed.plan['origin_context'] == hand.plan['origin_context']
    assert runtime.continue_hand(hand)['status'] == 'approval_resume'
    runtime.resume_approvals(parent.session_id, [grant])
    runtime.maintain()
    assert len(native.calls) == 2
    runtime.subagent_stop(child_session_id=resumed.child_session_id, child_summary='Fixture removed')
    assert any(path == f'/v1/hands/{resumed.hand_id}/result' and body['status'] == 'done'
               for path, body, _ in client.writes)


def test_grant_waits_for_result_and_native_finalization(setup):
    runtime, native, _, parent, hand, grant, _ = held(setup)
    runtime.resume_approvals(parent.session_id, [grant])
    runtime.maintain()
    assert len(native.calls) == 1
    runtime.subagent_stop(child_session_id=hand.child_session_id, child_summary='Held')
    for status in ('running', 'stalling', 'finalizing'):
        native.live[hand.delegation_id]['status'] = status
        runtime.maintain()
        assert len(native.calls) == 1
    native.live[hand.delegation_id]['status'] = 'completed'
    runtime.maintain()
    assert len(native.calls) == 2


@pytest.mark.parametrize('state', ['waiting', 'refused', 'lapsed', 'stale', 'consumed'])
def test_no_refusal_or_lapse_never_resumes(setup, state):
    runtime, native, _, parent, hand, grant, grants = held(setup)
    finish(runtime, native, hand)
    grant['state'] = state
    assert not runtime.resume_approvals(parent.session_id, grants)
    runtime.maintain()
    assert len(native.calls) == 1


def test_grant_revoked_while_attempt_runs_never_resumes(setup):
    runtime, native, _, parent, hand, grant, grants = held(setup)
    runtime.resume_approvals(parent.session_id, [grant])
    grants.clear()
    finish(runtime, native, hand)
    runtime.maintain()
    assert len(native.calls) == 1


def test_persisted_grant_without_a_fresh_authority_read_does_not_resume(setup):
    runtime, native, client, parent, hand, grant, _ = held(setup)
    runtime.resume_approvals(parent.session_id, [grant])
    finish(runtime, native, hand)
    client.get = lambda *args, **kwargs: {'permits': [grant], 'refreshed': False}
    runtime.maintain()
    assert len(native.calls) == 1


def test_poll_recovers_grant_without_an_answer_hook_and_duplicate_controls_are_noops(setup):
    runtime, native, _, parent, hand, _, _ = held(setup)
    finish(runtime, native, hand)
    runtime.maintain()
    with ThreadPoolExecutor(max_workers=4) as pool:
        answers = list(pool.map(lambda _: runtime.continue_hand(hand), range(8)))
    assert all(answer['status'] == 'approval_resume' for answer in answers)
    runtime.dispatch_tasks([{'goal': 'Duplicate', 'continue': hand.hand_id}], parent=parent)
    assert len(native.calls) == 2


def test_other_parent_cannot_resume_a_grant(setup):
    runtime, native, _, _, hand, grant, _ = held(setup)
    finish(runtime, native, hand)
    assert not runtime.resume_approvals('owner-other', [grant])
    assert len(native.calls) == 1


@pytest.mark.parametrize('already_consumed', [False, True])
def test_answer_hook_queues_before_context_and_remembers_it(setup, tmp_path, monkeypatch, already_consumed):
    from colonymind_hermes.answers import Answers, ChatLedger
    runtime, native, _, parent, hand, grant, grants = held(setup)
    finish(runtime, native, hand)
    if already_consumed:
        runtime.maintain()
        grants.clear()
    monkeypatch.setitem(sys.modules, 'colonymind_hands_front', NS(resume_approvals=runtime.resume_approvals))
    reader = Answers(None, NS(get=lambda sid: None), None, ChatLedger(tmp_path / 'chats.db'))
    reader.owner_chat = lambda *args, **kwargs: (NS(), 'sms:owner-chat')
    reader._to_read = lambda *args, **kwargs: {'text': 'yes'}
    reader._post = lambda *args: {'context': 'The owner approved.', 'granted_hands': grants,
                                'approved': [grant['ask_id']]}
    result = reader.pre_llm_call(session_id=parent.session_id, turn_id='answer-turn')
    assert 'automatically' in result['context'] and 'Do not continue' in result['context']
    assert 'perform the action yourself' in reader.last_read(parent.session_id)
    runtime.maintain()
    assert len(native.calls) == 2


def test_restart_does_not_replay_an_already_dispatched_resume(setup):
    runtime, native, client, parent, hand, grant, _ = held(setup)
    finish(runtime, native, hand)
    runtime.resume_approvals(parent.session_id, [grant])
    runtime.maintain()
    dispatch = importlib.import_module(type(runtime).__module__)
    recovered = dispatch.Runtime(config=runtime.config, client=client, spool_dir=runtime.spool_dir,
                                 native=native, owner_check=lambda sid: sid.startswith('owner'))
    try:
        recovered.remember_parent(parent)
        recovered.resume_approvals(parent.session_id, [grant])
        recovered.maintain()
        assert len(native.calls) == 2
    finally:
        recovered.close()


def test_resume_waits_for_a_local_slot_without_losing_the_grant(setup):
    runtime, native, _, parent, hand, grant, _ = held(setup)
    finish(runtime, native, hand)
    runtime.config['max_running'] = 0
    runtime.resume_approvals(parent.session_id, [grant])
    runtime.maintain()
    assert len(native.calls) == 1 and runtime.approval_resumes.spool.pending()
    runtime.config['max_running'] = 1
    runtime.maintain()
    assert len(native.calls) == 2 and not runtime.approval_resumes.spool.pending()


def test_pending_dispatch_receipt_recovers_without_starting_a_second_child(setup, monkeypatch):
    runtime, native, _, parent, hand, grant, _ = held(setup)
    finish(runtime, native, hand)
    runtime.resume_approvals(parent.session_id, [grant])
    saved = runtime.approval_resumes.spool.acknowledge
    monkeypatch.setattr(runtime.approval_resumes.spool, 'acknowledge', lambda *args: None)
    runtime.maintain()
    assert runtime.approval_resumes.spool.pending() and len(native.calls) == 2
    monkeypatch.setattr(runtime.approval_resumes.spool, 'acknowledge', saved)
    runtime.maintain()
    assert not runtime.approval_resumes.spool.pending() and len(native.calls) == 2


def test_transient_controller_outage_keeps_the_same_resume_plan(setup):
    runtime, native, client, parent, hand, grant, _ = held(setup)
    finish(runtime, native, hand)
    runtime.resume_approvals(parent.session_id, [grant])
    original = client.post
    plan_keys = []
    def unavailable(path, body, key):
        if path.endswith('/plan'):
            plan_keys.append(key)
            raise ConnectionError('temporary controller outage')
        return original(path, body, key)
    client.post = unavailable
    runtime.maintain()
    assert runtime.approval_resumes.spool.pending() and len(native.calls) == 1
    client.post = original
    runtime.maintain()
    assert not runtime.approval_resumes.spool.pending() and len(native.calls) == 2
    assert next(key for path, _, key in client.writes if key.startswith('approval-resume:')) == plan_keys[0]


def test_owner_stop_cancels_queued_automatic_resume(setup):
    runtime, native, _, parent, hand, grant, _ = held(setup)
    runtime.resume_approvals(parent.session_id, [grant])
    finish(runtime, native, hand)
    runtime.stop(hand, 'Owner stopped this work', calling_session=parent.session_id)
    runtime.maintain()
    assert len(native.calls) == 1
    assert runtime.approval_resumes.cancelled(hand)


def resume(runtime, native, parent, hand, grant):
    finish(runtime, native, hand)
    runtime.resume_approvals(parent.session_id, [grant])
    runtime.maintain()
    resumed = runtime.registry.all()[-1]
    assert resumed is not hand and resumed.plan['task']['continue'] == hand.hand_id
    return resumed


def planned_task(client, hand):
    return next(body['tasks'][0] for path, body, key in client.writes
                if path.endswith('/plan') and body['tasks'][0].get('continue') == hand.hand_id)


def test_resume_keeps_the_original_workspace_never_the_held_call_directory(setup):
    """Gate w18-20261007T182631Z: the permit's cwd became the continuation's workspace."""
    runtime, native, client, parent, hand, grant, _ = held(setup, workdir='/fixture/repo')
    resumed = resume(runtime, native, parent, hand, grant)
    assert resumed.plan['workdir'] == resumed.plan['task']['workdir'] == '/fixture/repo'
    assert planned_task(client, hand)['workdir'] == '/fixture/repo'
    assert resumed.plan['approved_call'] == {'permit_id': 'permit-one', 'call': grant['call'], 'workdir': '/fixture'}
    context = resumed.plan['task']['context']
    assert 'Run it in /fixture through the normal permit check' in context
    assert 'does not widen your workspace' in context and 'needs its own approval' in context


def test_resume_without_an_original_workspace_has_none(setup):
    runtime, native, client, parent, hand, grant, _ = held(setup)
    resumed = resume(runtime, native, parent, hand, grant)
    assert 'workdir' not in resumed.plan['task'] and not resumed.plan.get('workdir')
    assert 'workdir' not in planned_task(client, hand)
    assert resumed.plan['approved_call']['workdir'] == '/fixture'


def test_a_second_resume_is_still_bounded_by_the_first_attempt(setup):
    runtime, native, client, parent, hand, grant, grants = held(setup, workdir='/fixture/repo')
    first = resume(runtime, native, parent, hand, grant)
    # A continuation recorded before this rule carried the held call's directory.
    first.plan['task']['workdir'] = first.plan['workdir'] = '/'
    first.blocked, first.activity = True, 'awaiting_approval'
    second_grant = {**grant, 'permit_id': 'permit-two', 'ask_id': 'ask-two', 'hand_id': first.hand_id, 'workdir': '/'}
    grants[:] = [second_grant]
    second = resume(runtime, native, parent, first, second_grant)
    assert second.plan['workdir'] == second.plan['task']['workdir'] == '/fixture/repo'
    assert second.plan['approved_call']['permit_id'] == 'permit-two'


def test_a_broken_continuation_chain_yields_no_workspace(setup):
    runtime, native, client, parent, hand, grant, _ = held(setup, workdir='/fixture/repo')
    resumes = importlib.import_module(type(runtime).__module__.rsplit('.', 1)[0] + '.resumes')
    assert resumes.original_workspace(runtime.registry, hand) == '/fixture/repo'
    hand.plan['task']['continue'] = 'h-000000000000'
    assert resumes.original_workspace(runtime.registry, hand) is None


def approved_in(setup, root):
    """A continuation whose original workspace contains the approved deletion."""
    runtime, native, client, parent, hand, grant, _ = held(setup, workdir=str(root))
    grant['call'] = {'tool': 'terminal', 'arguments': {'command': 'rm report.txt', 'workdir': str(root)}}
    grant['workdir'] = str(root)
    resumed = resume(runtime, native, parent, hand, grant)
    authority = importlib.import_module(type(runtime).__module__.rsplit('.', 1)[0] + '.authority')
    runtime.workdir_for = lambda tool, args, task: args.get('workdir')
    checks = []
    client._request = lambda method, path, *, key, params: checks.append(params) or dict(client.decision)
    client.decision = {'decision': 'allow', 'permit_id': 'permit-one'}
    call = lambda command, ident: authority.pre_tool_call(
        runtime, tool_name='terminal', args={'command': command, 'workdir': str(root)},
        session_id=resumed.child_session_id, tool_call_id=ident)
    return runtime, client, resumed, grant, authority, checks, call


def test_the_approved_call_consumes_its_permit_even_inside_the_workspace(setup, tmp_path):
    runtime, client, resumed, grant, authority, checks, call = approved_in(setup, tmp_path)
    assert call('rm report.txt', 'approved') is None
    assert [row['hand'] for row in checks] == [resumed.hand_id]
    exact = authority.exact_call('terminal', grant['call']['arguments'], str(tmp_path))
    assert checks[0]['call_digest'] == authority.call_digest(exact, str(tmp_path))
    # Another literal deletion keeps the original workspace's exemption and spends no permit.
    assert call('rm scratch.txt', 'other') is None and len(checks) == 1
    resumed.plan.pop('approved_call')
    assert call('rm report.txt', 'unbound') is None and len(checks) == 1


def test_a_spent_approval_holds_a_repeat_of_the_call_for_its_own_yes(setup, tmp_path):
    runtime, client, resumed, grant, authority, checks, call = approved_in(setup, tmp_path)
    client.decision = {'decision': 'hold', 'reason': 'no current unused owner permit'}
    verdict = call('rm report.txt', 'repeat')
    assert verdict['action'] == 'block' and 'held for the owner' in verdict['message']
    holds = [body for path, body, _ in client.writes if path == f'/v1/hands/{resumed.hand_id}/hold']
    assert holds and holds[-1]['call'] == authority.exact_call('terminal', grant['call']['arguments'], str(tmp_path))
    assert holds[-1]['workdir'] == str(tmp_path) and 'workdir' not in holds[-1]['call']['arguments']
    assert resumed.activity == 'awaiting_approval' and resumed.blocked


@pytest.mark.parametrize('command', ('rm report.txt; echo done', 'rm report.txt && ls', 'rm $HOME/report.txt'))
def test_a_compound_deletion_is_refused_with_how_to_hold_it(setup, tmp_path, command):
    # 2026-10-09 live: `rm <path>; echo ...` read as "authority check unavailable", so the hand stopped instead of
    # rerunning the deletion alone for the owner's yes.
    runtime, client, resumed, grant, authority, checks, call = approved_in(setup, tmp_path)
    client.decision = {'decision': 'hold', 'reason': 'no current unused owner permit'}
    verdict = call(command, 'compound-' + str(len(command)))
    assert verdict['action'] == 'block'
    assert 'its own terminal command' in verdict['message'] and 'unavailable' not in verdict['message']
    assert not [path for path, _, _ in client.writes if path.endswith('/hold')]
    assert not resumed.blocked and checks == []
    # The advised standalone form is then held for the owner's yes.
    held_verdict = call('rm report.txt', 'standalone')
    assert held_verdict['action'] == 'block' and 'held for the owner' in held_verdict['message']
    assert [path for path, _, _ in client.writes if path.endswith('/hold')]
    assert resumed.activity == 'awaiting_approval' and resumed.blocked


@pytest.mark.parametrize('tool, command, holdable', (
    ('terminal', 'rm report.txt; echo done', True),
    ('terminal', 'rm report.txt 2>/dev/null', True),
    ('terminal', 'sudo rm report.txt', False),
    ('terminal', 'git reset --hard', False),
    ('terminal', 'find . -name x -delete', False),
    ('execute_code', 'import os; os.remove("x")', False),
))
def test_refused_deletion_advice_matches_what_can_be_held(setup, tool, command, holdable):
    authority = importlib.import_module(held(setup)[0].__class__.__module__.rsplit('.', 1)[0] + '.authority')
    message = authority.unbound_deletion_message(tool, {'command': command, 'code': command})
    assert ('its own terminal command' in message) is holdable
    assert ('cannot be held' in message) is not holdable


@pytest.mark.parametrize('granted_with_workdir', (True, False))
def test_one_yes_covers_the_call_with_or_without_a_restated_workdir(setup, tmp_path, granted_with_workdir):
    # 2026-10-09 live: the grant bound `rm x` without workdir; the resumed hand ran it with workdir and was held again.
    runtime, client, resumed, grant, authority, checks, call = approved_in(setup, tmp_path)
    arguments = {'command': 'rm report.txt'}
    if granted_with_workdir:
        arguments['workdir'] = str(tmp_path)
    grant['call'] = {'tool': 'terminal', 'arguments': arguments}
    resumed.plan['approved_call'] = {'permit_id': 'permit-one', 'call': grant['call'], 'workdir': str(tmp_path)}
    granted_digest = authority.call_digest(authority.exact_call('terminal', arguments, str(tmp_path)), str(tmp_path))
    assert call('rm report.txt', 'approved') is None
    assert checks and checks[0]['call_digest'] == granted_digest


def test_a_different_workdir_is_still_a_different_call(setup, tmp_path):
    runtime, client, resumed, grant, authority, checks, call = approved_in(setup, tmp_path)
    other = tmp_path / 'other'
    other.mkdir()
    same = authority.exact_call('terminal', {'command': 'rm x', 'workdir': str(tmp_path)}, str(tmp_path))
    differ = authority.exact_call('terminal', {'command': 'rm x', 'workdir': str(other)}, str(tmp_path))
    assert same == {'tool': 'terminal', 'arguments': {'command': 'rm x'}}
    assert differ['arguments']['workdir'] == str(other)
