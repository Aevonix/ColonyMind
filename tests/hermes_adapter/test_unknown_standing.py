"""Unknown authority refuses all tools; inherited authority is a creation-time snapshot."""
import logging
from types import SimpleNamespace

import pytest

from colonymind_hermes import capture, guard


@pytest.fixture
def lane(tmp_path, monkeypatch):
    settings = SimpleNamespace(
        outbox_path=tmp_path / "outbox.sqlite3", key_file=tmp_path / "api.key",
        owner_handles=lambda: {"telegram": ["owner-1"]}, owner_contact_id=lambda: "owner-1",
        mind=lambda: {}, guests=lambda: {"tools": [], "executor_platforms": []})
    client = SimpleNamespace(resolve_contact=lambda *a, **kw: {"contact_id": "guest-1"})
    sessions = capture.SessionMap(settings, client)
    monkeypatch.setattr(sessions, "parent", lambda sid: (
        sessions.get(sid).parent_session_id if sessions.get(sid) else ""))
    monkeypatch.setattr(guard, "mind_run", lambda _: False)
    monkeypatch.setattr(guard, "cron_context", lambda: False)
    monkeypatch.setattr(guard, "_hand_session_check", lambda *a: False)
    monkeypatch.setattr(capture, "session_env", lambda: ("", "", "", ""))
    return guard.Guard(client, sessions, settings), sessions


@pytest.mark.parametrize("tool", [
    "terminal", "process_manage", "execute_code", "computer_use", "write_file", "patch",
    "browser_navigate", "browser_vault_list", "browser_vault_fill", "cronjob_manage",
    "delegate_task", "skill_manage", "kanban_create", "kanban_comment", "kanban_complete",
    "kanban_block", "kanban_unblock", "memory", "mcp__sample__read", "mcp_sample_read",
    "web_search", "read_file", "colonymind_memory_search", "unknown_plugin_tool",
])
def test_unknown_refuses_every_tool_family(lane, tool, caplog):
    gate, _ = lane
    with caplog.at_level(logging.WARNING):
        result = gate.pre_tool_call(tool, {"opaque": "private-argument-1"}, session_id="orphan-1")
    assert result["action"] == "block" and "retry: false" in result["message"]
    assert "standing_unknown" in caplog.text and "tool=" + tool in caplog.text
    assert "session_class=unregistered" in caplog.text
    assert "orphan-1" not in caplog.text
    assert "private-argument-1" not in caplog.text


def test_resolver_failure_has_the_same_final_refusal(lane, monkeypatch, caplog):
    gate, sessions = lane
    def unavailable(*a):
        raise RuntimeError("private-session-1")
    monkeypatch.setattr(sessions, "standing", unavailable)
    with caplog.at_level(logging.WARNING):
        result = gate.pre_tool_call("terminal", {}, "private-session-1")
    assert result["action"] == "block" and "retry: false" in result["message"]
    assert "standing_unknown" in caplog.text and "private-session-1" not in caplog.text


@pytest.mark.parametrize("sender,expected", [("owner-1", None), ("guest-1", "block")])
def test_child_inherits_parent_at_creation_and_survives_a_new_map(lane, sender, expected):
    gate, sessions = lane
    sessions.observe(session_id="group-1", platform="telegram", sender_id=sender)
    gate.subagent_start(child_session_id="child-1", parent_session_id="group-1")
    # A later speaker cannot upgrade or downgrade an existing child.
    sessions.observe(session_id="group-1", platform="telegram",
                     sender_id="guest-1" if sender == "owner-1" else "owner-1")
    sessions.observe(session_id="child-1", platform="subagent", parent_session_id="group-1")
    assert (gate.pre_tool_call("terminal", {}, "child-1") or {}).get("action") == expected
    restored = capture.SessionMap(sessions.settings, sessions.client)
    assert restored.standing("child-1")[0] == ("owner" if sender == "owner-1" else "guest")


def test_parentless_child_cannot_borrow_bound_owner(lane, monkeypatch):
    gate, sessions = lane
    gate.subagent_start(child_session_id="child-1")
    monkeypatch.setattr(capture, "session_env", lambda: ("telegram", "owner-1", "group-1", "group"))
    assert sessions.standing("child-1")[0] == "unknown"
    assert gate.pre_tool_call("terminal", {}, "child-1")["action"] == "block"


def test_observed_orphan_without_a_creation_hook_is_unknown(lane, monkeypatch):
    gate, sessions = lane
    monkeypatch.setattr(capture, "session_env", lambda: ("telegram", "owner-1", "group-1", "group"))
    sessions.observe(session_id="orphan-1", platform="subagent")
    assert sessions.standing("orphan-1")[0] == "unknown"
    assert gate.pre_tool_call("terminal", {}, "orphan-1")["action"] == "block"


@pytest.mark.parametrize("senders", [
    ["owner-1", "guest-1"], ["guest-1", "owner-1"],
    ["owner-1", "guest-1", "owner-1", "guest-1"],
])
@pytest.mark.parametrize("observe", [True, False])
def test_group_standing_is_per_current_sender(lane, monkeypatch, senders, observe):
    gate, sessions = lane
    for sender in senders:
        monkeypatch.setattr(capture, "session_env", lambda sender=sender: (
            "telegram", sender, "group-1", "group"))
        if observe:
            sessions.observe(session_id="group-session-1", platform="telegram", sender_id=sender)
        result = gate.pre_tool_call("terminal", {}, "group-session-1")
        assert (result or {}).get("action") == (None if sender == "owner-1" else "block")


def test_unknown_cron_cannot_use_the_mind_lane(lane, monkeypatch):
    gate, _ = lane
    monkeypatch.setattr(gate, "_cron_origin", lambda *a: "unknown")
    assert gate.pre_tool_call("terminal", {}, "cron-job-1")["action"] == "block"


def test_registered_hand_needs_explicit_standing(lane, monkeypatch):
    gate, _ = lane
    monkeypatch.setattr(guard, "_hand_session_check", lambda *a: True)
    monkeypatch.setattr(guard, "_hand_session_standing", lambda *a: "unknown")
    assert gate.pre_tool_call("terminal", {}, "child-1")["action"] == "block"
    monkeypatch.setattr(guard, "_hand_session_standing", lambda *a: "owner")
    assert gate.pre_tool_call("terminal", {}, "child-1") is None


@pytest.mark.parametrize("standing", ["owner", "mind", "system", "guest"])
def test_explicit_child_does_not_need_parent_store_again(lane, monkeypatch, standing):
    gate, sessions = lane
    sessions.inherit("child-1", "parent-1", standing=standing, platform="subagent")
    def unavailable(*a):
        raise RuntimeError("parent store unavailable")
    monkeypatch.setattr(sessions, "parent", unavailable)
    assert gate.standing("child-1")[0] == standing


@pytest.mark.parametrize("sender,standing", [("owner-1", "owner"), ("guest-1", "guest")])
def test_successful_cron_creation_records_creator(lane, sender, standing):
    gate, sessions = lane
    sessions.observe(session_id="parent-1", platform="telegram", sender_id=sender)
    gate._remember_creator("cronjob_manage", {"action": "create"}, "parent-1", "", "call-1", "",
                           sessions.standing("parent-1"))
    gate.post_tool_call("cronjob_manage", {"action": "create"},
                       {"success": True, "job": {"id": "job-1"}}, session_id="parent-1", tool_call_id="call-1")
    assert sessions.standings.get("cron", "job-1")[0] == standing


def test_cron_run_records_registered_job_standing_when_created(lane, monkeypatch):
    gate, sessions = lane
    monkeypatch.setattr(gate, "_cron_origin", lambda *a: "owner")
    info = sessions.observe(session_id="cron_job-1_20300101_120000", platform="cron", task_id="cron:job-1:run")
    assert info.inherited_standing == "owner"
    assert gate.pre_tool_call("terminal", {}, info.session_id) is None


def test_dispatcher_records_mind_standing_at_creation(lane, monkeypatch):
    gate, sessions = lane
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    sessions.standings.put("kanban", "task-1", "mind", "worker")
    info = sessions.observe(session_id="worker-1", platform="cli")
    assert info.inherited_standing == "mind"
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    assert gate.standing("worker-1")[0] == "mind"


@pytest.mark.parametrize("standing,expected", [("owner", None), ("guest", "block"), ("unknown", "block")])
def test_ordinary_worker_needs_its_tasks_recorded_standing(lane, monkeypatch, standing, expected):
    gate, sessions = lane
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    sessions.standings.put("kanban", "task-1", standing, "telegram", "parent-1")
    assert (gate.pre_tool_call("terminal", {}, "worker-1") or {}).get("action") == expected


def test_cron_result_keeps_creating_group_turn(lane):
    gate, sessions = lane
    sessions.observe(session_id="group-1", platform="telegram", sender_id="owner-1", turn_id="turn-1")
    assert gate.pre_tool_call("cronjob_manage", {"action": "create"}, "group-1", tool_call_id="call-1") is None
    sessions.observe(session_id="group-1", platform="telegram", sender_id="guest-1", turn_id="turn-2")
    gate.post_tool_call("cronjob_manage", {"action": "create"}, {"success": True, "job_id": "job-1"},
                       session_id="group-1", tool_call_id="call-1")
    assert sessions.standings.get("cron", "job-1")[0] == "owner"


@pytest.mark.parametrize("sender,standing", [("owner-1", "owner"), ("guest-1", "guest")])
def test_delegation_keeps_initiating_group_turn(lane, monkeypatch, sender, standing):
    gate, sessions = lane
    monkeypatch.setattr(gate, "_guest_refusal", lambda *a: "")
    monkeypatch.setattr(gate, "_rules", lambda *a, **kw: None)
    sessions.observe(session_id="group-1", platform="telegram", sender_id=sender, turn_id="turn-1")
    assert gate.pre_tool_call("delegate_task", {}, "group-1", turn_id="turn-1") is None
    sessions.observe(session_id="group-1", platform="telegram",
                     sender_id="guest-1" if sender == "owner-1" else "owner-1", turn_id="turn-2")
    gate.subagent_start(child_session_id="child-1", parent_session_id="group-1", parent_turn_id="turn-1")
    assert sessions.standing("child-1")[0] == standing


def test_snapshots_follow_the_instance_across_profiles_and_never_cross_instances(lane, tmp_path):
    _, original = lane
    values = dict(vars(original.settings), home=tmp_path / "instance-1")
    first = capture.SessionMap(SimpleNamespace(**values), original.client)
    first.inherit("child-1", "parent-1", standing="owner")
    values["outbox_path"] = tmp_path / "other-profile/outbox.sqlite3"
    same_instance = capture.SessionMap(SimpleNamespace(**values), original.client)
    assert same_instance.standing("child-1")[0] == "owner"
    values["home"] = tmp_path / "instance-2"
    other_instance = capture.SessionMap(SimpleNamespace(**values), original.client)
    assert other_instance.standings.get("session", "child-1") is None


@pytest.mark.parametrize("standing", ["guest", "mind"])
def test_authority_store_cannot_be_read_through_an_alias(lane, tmp_path, standing):
    gate, sessions = lane
    sessions.inherit("child-1", "parent-1", standing=standing, platform="telegram")
    alias = tmp_path / "attachment-1"
    alias.symlink_to(sessions.standings.path)
    result = gate.pre_tool_call("read_file", {"path": str(alias)}, "child-1")
    assert result["action"] == "block"
