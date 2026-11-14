"""Creation evidence survives completion, compression and process restart."""
from types import SimpleNamespace

import pytest

from conftest import probe
from test_unknown_standing import lane
from test_guard import GUARD_CODE
from colonymind_hermes import capture, guard


@pytest.mark.parametrize("standing", ["owner", "guest", "unknown"])
def test_stock_compression_observation_and_restart_keep_nearest_snapshot(home, sidecar, standing):
    first = probe(GUARD_CODE + '''
sessions = colonymind_hermes._BODY.sessions
sessions.inherit('child', 'group', standing=%r, platform='subagent')
from hermes_state_registry import acquire, release_or_close
db = acquire()
db.create_session('group', 'telegram')
db.create_session('child', 'subagent', parent_session_id='group')
db.create_session('compressed', 'subagent', parent_session_id='child')
db.create_session('compressed-again', 'subagent', parent_session_id='compressed')
release_or_close(db)
emit(saved=True)
''' % standing, home)
    assert first == {"saved": True}
    # New process; the creating group is now speaking as the owner. Hermes'
    # pre-call hook still supplies that group as _parent_session_id.
    second = probe(GUARD_CODE + '''
owner('group')
invoke_hook('pre_llm_call', session_id='compressed-again', platform='subagent', sender_id='',
            parent_session_id='group', turn_id='continued-turn', task_id='t', user_message='synthetic',
            conversation_history=[], is_first_turn=False, model='m')
sessions = colonymind_hermes._BODY.sessions
emit(standing=sessions.standing('compressed-again')[0],
     blocked=check('terminal', {}, 'compressed-again')['action'] == 'block')
''', home)
    assert second == {"standing": standing, "blocked": standing != "owner"}
    third = probe(GUARD_CODE + '''
emit(standing=colonymind_hermes._BODY.sessions.standing('compressed-again')[0],
     blocked=check('terminal', {}, 'compressed-again')['action'] == 'block')
''', home)
    assert third == second


@pytest.mark.parametrize("standing", ["owner", "mind", "system", "guest", "unknown"])
def test_nearest_explicit_ancestor_wins_and_cannot_be_overwritten(lane, monkeypatch, standing):
    gate, sessions = lane
    sessions.inherit("root", "", standing="owner")
    sessions.inherit("child", "root", standing=standing)
    monkeypatch.setattr(sessions, "parent", lambda sid: {"compressed": "child", "child": "root"}.get(sid, ""))
    assert gate.standing("compressed")[0] == standing
    sessions.inherit("compressed", "root", standing="owner")
    assert gate.standing("compressed")[0] == standing


@pytest.mark.parametrize("tool,action,kind,result", [
    ("cronjob_manage", " CREATE ", "cron", {"success": True, "job_id": "created"}),
    ("cronjob_manage", " UPDATE ", "cron", {"success": True, "job_id": "created"}),
    ("kanban_create", "", "kanban", {"ok": True, "task_id": "created"}),
])
@pytest.mark.parametrize("standing", ["owner", "guest"])
def test_outstanding_creator_survives_restart(lane, monkeypatch, tool, action, kind, result, standing):
    gate, sessions = lane
    gate._remember_creator(tool, {"action": action}, "group", "", "call", "", (standing, "telegram"))
    restored = capture.SessionMap(sessions.settings, sessions.client)
    monkeypatch.setattr(restored, "parent", lambda _: "")
    restored.observe(session_id="group", platform="telegram", sender_id="owner-1")
    resumed = guard.Guard(sessions.client, restored, sessions.settings)
    resumed.post_tool_call(tool, {"action": action}, result, session_id="group", tool_call_id="call")
    assert restored.standings.get(kind, "created")[0] == standing


@pytest.mark.parametrize("tool,kind,result", [
    ("cronjob_manage", "cron", {"success": True, "job_id": "created"}),
    ("kanban_create", "kanban", {"ok": True, "task_id": "created"}),
])
def test_missing_creator_never_borrows_current_owner(lane, tool, kind, result):
    gate, sessions = lane
    sessions.observe(session_id="group", platform="telegram", sender_id="owner-1")
    gate.post_tool_call(tool, {"action": "create"}, result, session_id="group", tool_call_id="missing")
    assert sessions.standings.get(kind, "created")[0] == "unknown"


def test_failed_creation_publication_keeps_outstanding_owner_evidence(lane):
    gate, sessions = lane
    gate._remember_creator("cronjob_manage", {"action": "create"}, "group", "", "call", "", ("owner", "cli"))
    with sessions.standings.connect() as db:
        db.execute("CREATE TRIGGER reject_publication BEFORE INSERT ON standing WHEN NEW.kind='cron' "
                   "BEGIN SELECT RAISE(ABORT, 'synthetic unavailable destination'); END")
    gate.post_tool_call("cronjob_manage", {"action": "create"}, {"success": True, "job_id": "created"},
                       session_id="group", tool_call_id="call")
    key = gate._creator_key("group", "", "cronjob_manage", "call")
    assert sessions.standings.get("creation", key) == ("owner", "cli")
    assert sessions.standings.get("cron", "created") is None
    with sessions.standings.connect() as db:
        db.execute("DROP TRIGGER reject_publication")
    gate.post_tool_call("cronjob_manage", {"action": "create"}, {"success": True, "job_id": "created"},
                       session_id="group", tool_call_id="call")
    assert sessions.standings.get("cron", "created") == ("owner", "cli")
    assert sessions.standings.get("creation", key) is None


@pytest.mark.parametrize("standing", ["owner", "mind", "guest", "unknown"])
def test_error_read_uses_task_record_even_on_worker_profile(lane, monkeypatch, standing):
    gate, sessions = lane
    sessions.settings.worker_profile = "worker"
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task")
    sessions.standings.put("kanban", "task", standing)
    assert gate._error_read("web_search", {}, "worker") is (standing in {"owner", "mind"})


@pytest.mark.parametrize("standing", ["owner", "guest", "unknown"])
def test_initiating_delegate_turn_survives_restart(lane, monkeypatch, standing):
    gate, sessions = lane
    gate._remember_creator("delegate_task", {}, "group", "", "call", "turn", (standing, "telegram"))
    restored = capture.SessionMap(sessions.settings, sessions.client)
    monkeypatch.setattr(restored, "parent", lambda _: "")
    restored.observe(session_id="group", platform="telegram", sender_id="owner-1", turn_id="later")
    resumed = guard.Guard(sessions.client, restored, sessions.settings)
    resumed.subagent_start(child_session_id="child", parent_session_id="group", parent_turn_id="turn")
    assert restored.standing("child")[0] == standing


def test_dispatch_records_authority_before_board_publication(lane, tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from colonymind_hermes.body import Body

    _, sessions = lane
    settings = SimpleNamespace(worker_profile="worker", budget=lambda key, default: default)
    dispatcher = SimpleNamespace(settings=settings, sessions=sessions, _key=Body._key)
    conn = connect(tmp_path / "board.db")
    try:
        task = Body._create_task(dispatcher, kb, conn, "intention", {"title": "Synthetic task"})
        assert sessions.standings.get("kanban", task) == ("mind", "worker")
        assert kb.get_task(conn, task).status == "ready"
        sessions.standings.put("kanban", task, "guest")
        assert Body._create_task(dispatcher, kb, conn, "intention", {}) == task
        assert sessions.standings.get("kanban", task)[0] == "guest"

        def unavailable(*a, **kw):
            raise OSError("synthetic unavailable authority store")
        monkeypatch.setattr(sessions.standings, "put", unavailable)
        with pytest.raises(OSError):
            Body._create_task(dispatcher, kb, conn, "another", {})
        assert len(kb.list_tasks(conn)) == 1
    finally:
        conn.close()
