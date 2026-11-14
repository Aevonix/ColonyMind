"""A durable creator record keeps owner cron work live after retention and restart."""
from types import SimpleNamespace

import pytest
from uuid import uuid4

from colonymind_hermes import capture, guard
from test_unknown_standing import lane

REAL_MIND_RUN = guard.mind_run


@pytest.mark.parametrize("task_standing", [None, "unknown", "guest", "owner", "mind"])
@pytest.mark.parametrize("matching_task", [False, True])
@pytest.mark.parametrize("observed", [False, True])
def test_mind_result_reporting_does_not_publish_missing_creation_authority(
        lane, monkeypatch, task_standing, matching_task, observed):
    gate, sessions = lane
    task, session, call, job = ["marker-" + str(uuid4()) for _ in range(4)]
    reported_task = task if matching_task else "marker-" + str(uuid4())
    monkeypatch.setenv("HERMES_KANBAN_TASK", task)
    monkeypatch.setenv("HERMES_PROFILE", "colonymind-act")
    monkeypatch.setattr(guard, "mind_run", REAL_MIND_RUN)
    if task_standing:
        sessions.standings.put("kanban", task, task_standing, "worker")
    if observed:
        sessions.observe(session_id=session, platform="cli", task_id=task)
        assert sessions.creation_standing(session)[0] == (task_standing or "unknown")
    else:
        assert sessions.creation_standing(session) is None
    sent = []
    monkeypatch.setattr(gate.client, "post", lambda route, **kw: sent.append((route, kw["json"])), raising=False)
    # No pre-call creator record: the result can never publish owner/mind authority.
    gate.post_tool_call("cronjob_manage", {"action": "create"}, {"success": True, "job_id": job},
                        session_id=session, task_id=reported_task, tool_call_id=call)
    assert sessions.standings.get("cron", job)[0] == "unknown"
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    assert gate.pre_tool_call("terminal", {}, "cron_" + job + "_20300101_120000")["action"] == "block"
    expected = [(guard.CRON_ORIGIN_ROUTE, {"job_id": job, "task_id": task, "tool_call_id": call,
                                         "action": "create", "ok": True})]
    assert sent == (expected if observed and task_standing == "mind" and matching_task else [])


@pytest.fixture
def terminal_report(lane, monkeypatch):
    from agent.tool_executor import _ToolCallRef, _blocked_tool_result
    import model_tools

    gate, _ = lane
    sent, delivered = [], []
    monkeypatch.setattr(gate.client, "post", lambda route, **kw: sent.append((route, kw["json"])), raising=False)

    def deliver(**kw):
        delivered.append(kw["status"])
        gate.post_tool_call(kw["function_name"], kw["function_args"], kw["result"],
                            session_id=kw["session_id"], task_id=kw["task_id"], tool_call_id=kw["tool_call_id"],
                            status=kw["status"])

    monkeypatch.setattr(model_tools, "_emit_post_tool_call_hook", deliver)

    def report(session, task, call, tool, args, value, *, blocked=False):
        ref = _ToolCallRef(tool, args, task, call, [])
        agent = SimpleNamespace(session_id=session)
        if blocked:
            _blocked_tool_result(agent, ref, block_body={"error": value},
                                 block_error_type="plugin_block", guardrail_decision=None)
        else:
            ref.emit_post(agent, value, status="ok", duration_ms=1, error_type=None, error_message=None)
        assert delivered == ["blocked" if blocked else "ok"]
        return sent

    return report


@pytest.mark.parametrize("session_standing", [None, "observed-guest", "guest", "unknown", "owner", "mind"])
@pytest.mark.parametrize("kind", ["cron-success", "cron-failure", "child"])
def test_missing_creator_reporting_requires_positive_session(lane, monkeypatch, terminal_report,
                                                            session_standing, kind):
    gate, sessions = lane
    task, session, call, target = ["marker-" + str(uuid4()) for _ in range(4)]
    monkeypatch.setenv("HERMES_KANBAN_TASK", task)
    monkeypatch.setattr(guard, "mind_run", REAL_MIND_RUN)
    if session_standing == "observed-guest":
        sessions.observe(session_id=session, platform="telegram", sender_id="marker-" + str(uuid4()))
        assert sessions.creation_standing(session) is None
        assert gate.standing(session, task)[0] == "guest"
    elif session_standing:
        sessions.standings.put("kanban", task, session_standing, "worker")
        sessions.observe(session_id=session, platform="cli", task_id=task)
        assert sessions.creation_standing(session) == (session_standing, "worker")
    else:
        assert sessions.creation_standing(session) is None
        assert gate.standing(session, task)[0] == "unknown"
    sessions.standings.put("kanban", task, "mind", "worker")
    child = kind == "child"
    tool, args = ("kanban_create", {}) if child else ("cronjob_manage", {"action": "create"})
    value = {"ok": True, "task_id": target} if child else {"success": kind == "cron-success", "job_id": target}
    assert sessions.standings.get("creation", gate._creator_key(session, task, tool, call)) is None
    sent = terminal_report(session, task, call, tool, args, value)
    expected = (guard.LINEAGE_ROUTE, {"child": target, "parent": task}) if child else (
        guard.CRON_ORIGIN_ROUTE, {"job_id": target if kind == "cron-success" else "", "task_id": task,
                                  "tool_call_id": call, "action": "create", "ok": kind == "cron-success"})
    assert sent == ([expected] if session_standing in {"owner", "mind"} else [])
    if session_standing == "observed-guest":
        assert sessions.creation_standing(session) is None
        assert gate.standing(session, task)[0] == "guest"
    if kind != "cron-failure":
        assert sessions.standings.get("kanban" if child else "cron", target)[0] == "unknown"
        monkeypatch.delenv("HERMES_KANBAN_TASK")
        restored = capture.SessionMap(sessions.settings, sessions.client)
        again = guard.Guard(gate.client, restored, sessions.settings)
        if child:
            monkeypatch.setenv("HERMES_KANBAN_TASK", target)
            verdict = again.pre_tool_call("terminal", {}, "marker-" + str(uuid4()), target)
        else:
            verdict = again.pre_tool_call("terminal", {}, "cron_" + target + "_20300101_120000")
        assert verdict["action"] == "block" and "retry: false" in verdict["message"]


@pytest.mark.parametrize("matching", [False, True])
def test_unrecorded_final_veto_cannot_later_report_as_mind(lane, monkeypatch, terminal_report, matching):
    gate, sessions = lane
    task, other, session, call = ["marker-" + str(uuid4()) for _ in range(4)]
    monkeypatch.setenv("HERMES_KANBAN_TASK", task)
    monkeypatch.setattr(guard, "mind_run", REAL_MIND_RUN)
    hook_task = task if matching else other
    assert sessions.creation_standing(session) is None
    assert gate.standing(session, hook_task)[0] == "unknown"
    args = {"action": "create"}
    verdict = gate.pre_tool_call("cronjob_manage", args, session, hook_task, tool_call_id=call)
    assert verdict["action"] == "block" and "retry: false" in verdict["message"]
    assert sessions.standings.get("creation", gate._creator_key(session, hook_task, "cronjob_manage", call)) is None
    sessions.standings.put("kanban", task, "mind", "worker")
    assert terminal_report(session, hook_task, call, "cronjob_manage", args, verdict["message"], blocked=True) == []
    assert sessions.creation_standing(session) is None


@pytest.mark.parametrize("standing", ["unknown", "guest"])
def test_mind_result_reporting_respects_retained_session_snapshot(lane, monkeypatch, standing):
    from agent.tool_executor import _ToolCallRef, _blocked_tool_result
    import model_tools

    gate, sessions = lane
    task, session, call = ["marker-" + str(uuid4()) for _ in range(3)]
    monkeypatch.setenv("HERMES_KANBAN_TASK", task)
    monkeypatch.setattr(guard, "mind_run", REAL_MIND_RUN)
    if standing == "guest":
        sessions.standings.put("kanban", task, "guest", "worker")
    observed = sessions.observe(session_id=session, platform="cli", task_id=task)
    assert observed.inherited_standing == standing
    sessions.standings.put("kanban", task, "mind", "worker")
    assert gate.standing(session, task)[0] == standing
    sent = []
    monkeypatch.setattr(gate.client, "post", lambda route, **kw: sent.append((route, kw["json"])), raising=False)
    args = {"action": "create"}
    error = call
    if standing == "unknown":
        verdict = gate.pre_tool_call("cronjob_manage", args, session, task, tool_call_id=call)
        assert verdict["action"] == "block" and "retry: false" in verdict["message"]
        error = verdict["message"]
    # Guest covers missing pre-hook evidence; unknown covers a genuine final veto.
    key = gate._creator_key(session, task, "cronjob_manage", call)
    assert sessions.standings.get("creation", key) is None
    delivered = []

    def deliver(**kw):
        delivered.append(kw["status"])
        gate.post_tool_call(kw["function_name"], kw["function_args"], kw["result"],
                            session_id=kw["session_id"], task_id=kw["task_id"], tool_call_id=kw["tool_call_id"],
                            status=kw.get("status"))

    monkeypatch.setattr(model_tools, "_emit_post_tool_call_hook", deliver)
    _blocked_tool_result(SimpleNamespace(session_id=session),
                         _ToolCallRef("cronjob_manage", args, task, call, []),
                         block_body={"error": error}, block_error_type="plugin_block", guardrail_decision=None)
    assert delivered == ["blocked"]
    assert sent == []


@pytest.mark.parametrize("standing", ["owner", "guest", "unknown"])
def test_job_creator_record_survives_missing_job_and_session_records(lane, monkeypatch, standing):
    _, sessions = lane
    sessions.standings.put("cron", "job", standing, "telegram", "unretained-parent")
    restored = capture.SessionMap(sessions.settings, sessions.client)
    monkeypatch.setattr(restored, "parent", lambda _: "")
    gate = guard.Guard(restored.client, restored, restored.settings)

    def no_retained_job(*a, **kw):
        pytest.fail("positive durable creator evidence must not need a retained job or a sidecar")
    monkeypatch.setattr(gate, "_stored_job", no_retained_job)
    session, task = "cron_job_20300101_120000", "cron:job:run"
    assert gate.standing(session, task)[0] == standing
    assert (gate.pre_tool_call("terminal", {}, session, task) or {}).get("action") == (
        None if standing == "owner" else "block")
    info = restored.observe(session_id=session, platform="cron", task_id=task)
    assert info.inherited_standing == standing

    again = capture.SessionMap(sessions.settings, sessions.client)
    monkeypatch.setattr(again, "parent", lambda sid: session if sid == "compressed" else "")
    continued = guard.Guard(again.client, again, again.settings)
    assert continued.standing("compressed")[0] == standing
    assert (continued.pre_tool_call("terminal", {}, "compressed") or {}).get("action") == (
        None if standing == "owner" else "block")


def test_authenticated_owner_can_readmit_job_without_upgrading_an_old_unknown_run(lane, monkeypatch):
    gate, sessions = lane
    sessions.standings.put("cron", "job", "unknown")
    sessions.inherit("old-run", "", standing="unknown", platform="cron")
    sessions.observe(session_id="owner-turn", platform="telegram", sender_id="owner-1")
    args = {"action": " UPDATE ", "job_id": "job"}
    assert gate.pre_tool_call("cronjob_manage", args, "owner-turn", tool_call_id="readmit") is None
    gate.post_tool_call("cronjob_manage", args, {"success": True, "job_id": "job"},
                        session_id="owner-turn", tool_call_id="readmit")
    restored = capture.SessionMap(sessions.settings, sessions.client)
    monkeypatch.setattr(restored, "parent", lambda _: "")
    resumed = guard.Guard(restored.client, restored, restored.settings)
    assert resumed.pre_tool_call("terminal", {}, "cron_job_20300102_120000", "cron:job:run") is None
    assert resumed.pre_tool_call("terminal", {}, "old-run")["action"] == "block"
