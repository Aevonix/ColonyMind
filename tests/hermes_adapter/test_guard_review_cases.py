"""Reviewer-only synthetic boundary checks. No external access or tool execution."""
import pytest

from test_unknown_standing import lane
from colonymind_hermes import capture, guard
REAL_MIND_RUN = guard.mind_run


def test_unreadable_ancestry_is_final_unknown(lane, monkeypatch, caplog):
    gate, sessions = lane
    monkeypatch.setattr(sessions, 'parent', lambda _: None)
    assert gate._cron_job('synthetic') is None
    # Round 2 distinguishes unreadable ancestry outside any identified cron
    # context from an identified cron run with unreadable creator evidence.
    assert gate._cron_origin('synthetic') == ''
    monkeypatch.setattr(guard, 'cron_context', lambda: True)
    assert gate._cron_origin('synthetic') == 'unknown'
    result = gate.pre_tool_call('web_search', {}, 'synthetic')
    assert result['action'] == 'block' and 'retry: false' in result['message']
    assert 'standing_unknown' in caplog.text


def test_sender_cache_cannot_reuse_owner_during_resolution_failure(lane, monkeypatch):
    gate, sessions = lane
    monkeypatch.setattr(capture, 'session_env', lambda: ('telegram', 'owner-1', 'group', 'group'))
    assert gate.pre_tool_call('terminal', {}, 'bound') is None
    monkeypatch.setattr(capture, 'session_env', lambda: ('telegram', 'guest-1', 'group', 'group'))
    def unavailable(*a, **kw):
        raise RuntimeError()
    monkeypatch.setattr(sessions.client, 'resolve_contact', unavailable)
    assert gate.pre_tool_call('terminal', {}, 'bound')['action'] == 'block'


def test_compressed_guest_child_keeps_its_snapshot(lane, monkeypatch):
    gate, sessions = lane
    sessions.observe(session_id='group', platform='telegram', sender_id='guest-1')
    gate.subagent_start(child_session_id='child', parent_session_id='group')
    sessions.observe(session_id='child', platform='subagent', parent_session_id='group')
    sessions.observe(session_id='group', platform='telegram', sender_id='owner-1')
    original_parent = sessions.parent
    monkeypatch.setattr(sessions, 'parent', lambda sid: 'child' if sid == 'compressed' else original_parent(sid))
    assert sessions.standing('child')[0] == 'guest'
    assert gate.pre_tool_call('terminal', {}, 'compressed')['action'] == 'block'


def test_compressed_unknown_child_keeps_its_snapshot(lane, monkeypatch):
    gate, sessions = lane
    sessions.inherit('child', 'group', standing='unknown', platform='subagent')
    sessions.observe(session_id='group', platform='telegram', sender_id='owner-1')
    sessions.observe(session_id='child', platform='subagent', parent_session_id='group')
    original_parent = sessions.parent
    monkeypatch.setattr(sessions, 'parent', lambda sid: 'child' if sid == 'compressed' else original_parent(sid))
    assert gate.pre_tool_call('terminal', {}, 'child')['action'] == 'block'
    assert gate.pre_tool_call('terminal', {}, 'compressed')['action'] == 'block'


def test_observation_after_child_compression_keeps_creator(lane):
    gate, sessions = lane
    sessions.observe(session_id='group', platform='telegram', sender_id='guest-1')
    gate.subagent_start(child_session_id='child', parent_session_id='group')
    sessions.observe(session_id='group', platform='telegram', sender_id='owner-1')
    # Hermes leaves agent._parent_session_id as the original creating parent on rotation.
    sessions.observe(session_id='compressed', platform='subagent', parent_session_id='group')
    assert gate.pre_tool_call('terminal', {}, 'compressed')['action'] == 'block'


def test_compressed_owner_child_survives_restart(lane, monkeypatch):
    gate, sessions = lane
    sessions.inherit('child', 'group', standing='owner', platform='subagent')
    restored = capture.SessionMap(sessions.settings, sessions.client)
    monkeypatch.setattr(restored, 'parent', lambda sid: {'compressed': 'child', 'child': 'group'}.get(sid, ''))
    restored_gate = guard.Guard(sessions.client, restored, sessions.settings)
    assert restored_gate.pre_tool_call('terminal', {}, 'child') is None
    assert restored_gate.pre_tool_call('terminal', {}, 'compressed') is None


def test_owner_kanban_on_worker_profile_keeps_creation_standing(lane, monkeypatch):
    gate, sessions = lane
    sessions.settings.worker_profile = 'synthetic-worker'
    monkeypatch.setenv('HERMES_KANBAN_TASK', 'task')
    monkeypatch.setenv('HERMES_PROFILE', 'synthetic-worker')
    monkeypatch.setattr(guard, 'mind_run', REAL_MIND_RUN)
    sessions.standings.put('kanban', 'task', 'owner', 'telegram')
    assert gate.automation_standing('worker')[0] == 'owner'


def test_unknown_kanban_on_worker_profile_has_no_tools(lane, monkeypatch):
    gate, sessions = lane
    sessions.settings.worker_profile = 'synthetic-worker'
    monkeypatch.setenv('HERMES_KANBAN_TASK', 'task')
    monkeypatch.setenv('HERMES_PROFILE', 'synthetic-worker')
    monkeypatch.setattr(guard, 'mind_run', REAL_MIND_RUN)
    assert gate.pre_tool_call('web_search', {}, 'worker')['action'] == 'block'


def test_recorded_guest_kanban_on_worker_profile_remains_guest(lane, monkeypatch):
    gate, sessions = lane
    sessions.settings.worker_profile = 'synthetic-worker'
    monkeypatch.setenv('HERMES_KANBAN_TASK', 'task')
    monkeypatch.setenv('HERMES_PROFILE', 'synthetic-worker')
    monkeypatch.setattr(guard, 'mind_run', REAL_MIND_RUN)
    sessions.standings.put('kanban', 'task', 'guest', 'telegram')
    assert gate.pre_tool_call('read_file', {'path': '/synthetic-host-file'}, 'worker')['action'] == 'block'


@pytest.mark.requires_sidecar
def test_retained_cron_requires_positive_creator_evidence(lane, monkeypatch):
    from types import SimpleNamespace
    from colonymind.mind.tick import Mind
    gate, sessions = lane
    # Use the real route's semantics with empty local authority state.
    service = SimpleNamespace(mind_state=SimpleNamespace(get=lambda _: None), _pending_around=lambda *a: False)
    response = Mind.cron_origin(service, 'job')
    monkeypatch.setattr(gate.client, 'has_mind_routes', lambda: True, raising=False)
    monkeypatch.setattr(gate.client, 'get', lambda *a, **kw: SimpleNamespace(is_success=True, json=lambda: response), raising=False)
    monkeypatch.setattr(gate, '_stored_job', lambda _: {'id': 'job', 'created_at': '2030-01-01'})
    assert sessions.standings.get('cron', 'job') is None
    assert gate.pre_tool_call('terminal', {}, 'cron_job_20300101_120000', 'cron:job:run')['action'] == 'block'


def test_normalized_cron_action_keeps_creator_during_interleaving(lane, monkeypatch):
    gate, sessions = lane
    monkeypatch.setattr(gate, '_guest_refusal', lambda *a: '')
    monkeypatch.setattr(gate, '_rules', lambda *a, **kw: None)
    sessions.observe(session_id='group', platform='telegram', sender_id='guest-1')
    assert gate.pre_tool_call('cronjob_manage', {'action': ' CREATE '}, 'group', tool_call_id='call') is None
    sessions.observe(session_id='group', platform='telegram', sender_id='owner-1')
    gate.post_tool_call('cronjob_manage', {'action': ' CREATE '}, {'success': True, 'job_id': 'job'},
                        session_id='group', tool_call_id='call')
    assert sessions.standings.get('cron', 'job')[0] == 'guest'


def test_evicted_creation_record_does_not_follow_later_group_sender(lane, monkeypatch):
    gate, sessions = lane
    monkeypatch.setattr(gate, '_guest_refusal', lambda *a: '')
    monkeypatch.setattr(gate, '_rules', lambda *a, **kw: None)
    sessions.observe(session_id='group', platform='telegram', sender_id='guest-1')
    assert gate.pre_tool_call('cronjob_manage', {'action': 'create'}, 'group', tool_call_id='call') is None
    for i in range(2048):
        gate._remember_creator('kanban_create', {}, 'other', '', str(i), '', ('owner', 'telegram'))
    sessions.observe(session_id='group', platform='telegram', sender_id='owner-1')
    gate.post_tool_call('cronjob_manage', {'action': 'create'}, {'success': True, 'job_id': 'job'},
                        session_id='group', tool_call_id='call')
    assert sessions.standings.get('cron', 'job')[0] in {'guest', 'unknown'}
