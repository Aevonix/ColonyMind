"""Activity is a bounded, lossy observer, with no execution-path I/O."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
import os
from pathlib import Path
import select
import sqlite3
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
import yaml

from conftest import MIND_PRELUDE, PRELUDE, build_home, probe
import colonymind_hermes
from colonymind_hermes import body as activity
from colonymind_hermes.body import Activity
from colonymind_hermes.capture import TurnOutbox, deliver

NATIVE = dict(session_id="thread-1", turn_id="turn-1")
PRIVATE = "PRIVATE-NESTED-CANARY-approval-command-7a6d"
HOOKS = ("pre_llm_call", "pre_tool_call", "pre_approval_request", "post_approval_response",
         "post_tool_call", "post_llm_call", "native_turn_settled")


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "activity daemon did not reach expected state"
        time.sleep(.005)


@pytest.fixture
def observer(tmp_path):
    client = SimpleNamespace(post=lambda *a, **kw: httpx.Response(200, json={"accepted": True}))
    value = Activity(tmp_path / "capture.sqlite3", client)
    yield value
    value.close()
    if value._thread:
        value._thread.join(3)
        assert not value._thread.is_alive()


def latest(observer):
    with observer._lock:
        return [{"scope": session, "activity_code": scope.state()[0], "count": scope.state()[1]}
                for session, scope in sorted(observer._scopes.items()) if scope.state()[0] != "idle"]


def test_run_tools_approval_and_late_callbacks_do_not_store_private_data(observer):
    assert latest(observer) == []
    observer.pre_llm_call(**NATIVE, user_message=PRIVATE)
    assert latest(observer) == [{"scope": "thread-1", "activity_code": "thinking", "count": 1}]
    tool = dict(NATIVE, tool_call_id="call-1", tool_name=PRIVATE, args={"deep": {"secret": PRIVATE}})
    assert observer.pre_tool_call(**tool) is None
    assert latest(observer)[0]["activity_code"] == "tool_running"
    before = dict(observer._pending)
    observer.pre_tool_call(**tool)
    assert observer._pending == before
    observer.pre_approval_request(**tool, surface="gateway", command=PRIVATE, description=PRIVATE)
    assert latest(observer)[0]["activity_code"] == "waiting_approval"
    observer.post_approval_response(**tool, choice="once")
    assert latest(observer)[0]["activity_code"] == "tool_running"
    observer.post_tool_call(**tool, result={"nested": [PRIVATE]})
    assert latest(observer)[0]["activity_code"] == "thinking"
    observer.post_llm_call(**NATIVE, assistant_response=PRIVATE)
    assert latest(observer) == []
    for callback in (observer.pre_tool_call, observer.pre_approval_request, observer.post_tool_call, observer.pre_llm_call):
        callback(**tool)
    assert latest(observer) == []
    observer.start()
    wait_for(lambda: observer.available)
    with closing(observer._connect()) as conn:
        # The daemon can still be committing or clearing its delivery snapshot.
        conn.execute("PRAGMA busy_timeout=5000")
        for table in ("scopes", "pending"):
            assert PRIVATE not in json.dumps([tuple(r) for r in conn.execute("SELECT * FROM " + table)])


def test_multiple_tools_and_approval_response_keep_other_waits(observer):
    observer.pre_llm_call(**NATIVE)
    for call in ("a", "b"):
        observer.pre_tool_call(**NATIVE, tool_call_id=call)
        observer.pre_approval_request(**NATIVE, tool_call_id=call, request_id="request-" + call, surface="gateway")
    observer.post_approval_response(**NATIVE, tool_call_id="a", request_id="request-a")
    assert latest(observer)[0]["activity_code"] == "waiting_approval"
    observer.post_tool_call(**NATIVE, tool_call_id="b")
    assert latest(observer)[0]["activity_code"] == "tool_running"
    observer.post_tool_call(**NATIVE, tool_call_id="a")
    assert latest(observer)[0]["activity_code"] == "thinking"


def test_smart_guardian_is_not_a_human_approval_wait(observer):
    observer.pre_llm_call(**NATIVE)
    observer.pre_tool_call(**NATIVE, tool_call_id="a")
    before = latest(observer)
    observer.pre_approval_request(**NATIVE, tool_call_id="a", surface="smart", command=PRIVATE)
    assert latest(observer) == before


@pytest.mark.parametrize("outcome,code", [({"failed": True}, "error"),
    ({"failed": True, "interrupted": True}, None), ({"completed": True}, None)])
def test_native_settlement_ends_failed_and_interrupted_runs(observer, outcome, code):
    observer.pre_llm_call(**NATIVE)
    observer.pre_tool_call(**NATIVE, tool_call_id="a")
    observer.native_turn_settled(**NATIVE, outcome={**outcome, "failure_reason": PRIVATE})
    assert latest(observer) == ([] if code is None else [{"scope": "thread-1", "activity_code": code, "count": 0}])
    observer.pre_llm_call(session_id="thread-1", turn_id="next-turn")
    assert latest(observer) == [{"scope": "thread-1", "activity_code": "thinking", "count": 1}]


def test_same_process_reload_reconciles_and_revision_survives(observer):
    calls = []
    observer.client.post = lambda *a, **kw: calls.append(kw["json"]) or httpx.Response(200, json={"accepted": True})
    observer.pre_llm_call(**NATIVE)
    observer.start()
    wait_for(lambda: calls)
    restarted = Activity(observer.path.with_name("capture.sqlite3"), observer.client)
    restarted.start()
    try:
        wait_for(lambda: any(row["activities"] == [] for row in calls))
        assert calls[-1]["producer_id"] == calls[0]["producer_id"]
        assert calls[-1]["revision"] > calls[0]["revision"]
        observer.pre_tool_call(**NATIVE, tool_call_id="late-old-load")
        wait_for(lambda: observer.failure_category == "OSError")
        assert calls[-1]["activities"] == []
    finally:
        restarted.close()
        restarted._thread.join(3)


def test_live_peer_preserved_dead_peer_reconciled(observer, monkeypatch):
    observer.pre_llm_call(**NATIVE)
    observer.start()
    wait_for(lambda: observer.available)
    with closing(observer._connect()) as conn, conn:
        # Allow the daemon's short transaction to finish before the test writes.
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("UPDATE owners SET pid=? WHERE owner=?", (999999, observer.owner))
    monkeypatch.setattr(activity, "_alive", lambda pid, birth: pid == 999999 or pid == observer.pid)
    calls = []
    peer = Activity(observer.path.with_name("capture.sqlite3"), SimpleNamespace(
        post=lambda *a, **kw: calls.append(kw["json"]) or httpx.Response(200, json={"accepted": True})))
    peer.pre_llm_call(session_id="thread-2", turn_id="turn-2")
    peer.start()
    try:
        wait_for(lambda: calls and len(calls[-1]["activities"]) == 2)
        monkeypatch.setattr(activity, "_alive", lambda pid, birth: pid != 999999)
        wait_for(lambda: calls[-1]["activities"] == [{"scope": "thread-2", "activity_code": "thinking", "count": 1}])
    finally:
        peer.close()
        peer._thread.join(3)
    assert calls[-1]["activities"] == []


def test_concurrent_native_turns_are_counted_once(observer):
    def start(ident):
        observer.pre_llm_call(session_id="thread-1", turn_id=f"turn-{ident}")
        observer.pre_llm_call(session_id="thread-1", turn_id=f"turn-{ident}")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(start, range(20)))
    assert latest(observer) == [{"scope": "thread-1", "activity_code": "thinking", "count": 20}]


def test_missing_identity_and_observer_failure_do_not_leak_or_raise(observer, monkeypatch, caplog):
    observer.pre_llm_call(session_id="", turn_id="missing-session")
    observer.pre_tool_call(**NATIVE, tool_call_id="unbound")
    assert latest(observer) == []
    def failed(*args, **kwargs):
        raise RuntimeError(PRIVATE)
    monkeypatch.setattr(observer, "_change", failed)
    assert observer.pre_tool_call(**NATIVE, args={"nested": PRIVATE}) is None
    assert observer.available is False and PRIVATE not in caplog.text


def test_worker_publishes_coalesced_state_without_body_or_dispatch(home, sidecar):
    result = probe('''
activity = colonymind_hermes._ACTIVITY
body = colonymind_hermes._BODY
assert body._thread is None and body._dispatcher is False
native = dict(session_id="worker-session", turn_id="worker-turn")
activity.pre_llm_call(**native)
activity.pre_tool_call(**native, tool_call_id="worker-call")
activity.pre_approval_request(**native, tool_call_id="worker-call", surface="gateway")
deadline = time.monotonic() + 5
while time.monotonic() < deadline:
    if activity._initialized and activity._last_activities == [{"scope":"worker-session","activity_code":"waiting_approval","count":1}]:
        break
    time.sleep(.01)
else:
    raise AssertionError((str(activity.path), activity.failure_category, activity._initialized, activity._pending, activity._local, activity._thread.is_alive()))
activity.close()
activity._thread.join(3)
emit(thread=body._thread is not None, daemon=activity._thread.daemon,
     dispatcher=body._dispatcher, mind_ticks=body._mind_ticks,
     module=str(__import__("pathlib").Path(colonymind_hermes.__file__).resolve()))
''', home, prelude=MIND_PRELUDE, env={"HERMES_KANBAN_TASK": "worker-task"})
    assert result["thread"] is False and result["daemon"] is True
    assert result["dispatcher"] is False and result["mind_ticks"] == 0
    # The worker loaded the adapter this interpreter resolves: the checkout in a source tree, the installed wheel in CI.
    assert result["module"] == str(Path(colonymind_hermes.__file__).resolve())
    published = sidecar.calls("/v1/host/activity/snapshot", "POST")
    assert any(row["json"]["activities"] == [{"scope": "worker-session", "activity_code": "waiting_approval", "count": 1}] for row in published)
    assert published[-1]["json"]["activities"] == []
    assert sidecar.calls("/v1/mind/dispatch") == []


def test_late_native_failure_cannot_restore_an_error_after_a_successor_started(observer):
    observer.pre_llm_call(**NATIVE)
    observer.post_llm_call(**NATIVE)
    successor = dict(session_id=NATIVE["session_id"], turn_id="successor-turn")
    observer.pre_llm_call(**successor)
    observer.native_turn_settled(**NATIVE, outcome={"failed": True})
    observer.post_llm_call(**successor)
    assert latest(observer) == []
    observer.native_turn_settled(**NATIVE, outcome={"failed": True})
    assert latest(observer) == []


@pytest.mark.parametrize("dependency", ["sqlite", "sidecar"])
def test_every_hook_p99_under_5ms_and_hermes_budget_never_blocks(observer, monkeypatch, dependency):
    entered, release = threading.Event(), threading.Event()
    if dependency == "sidecar":
        def unresponsive(*a, **kw):
            assert kw["timeout"] == .1
            entered.set()
            release.wait(5)
            raise httpx.ReadTimeout("unresponsive")
        observer.client.post = unresponsive
    observer.start()
    lock = None
    try:
        if dependency == "sqlite":
            wait_for(lambda: observer.available)
            lock = sqlite3.connect(observer.path, isolation_level=None)
            lock.execute("BEGIN EXCLUSIVE")
            wait_for(lambda: observer.failure_category == "OperationalError")
        else:
            assert entered.wait(3)
        # Each hook is measured 1,000 times, with realistic native identities.
        timings = {name: [] for name in HOOKS}
        for i in range(1000):
            native = dict(session_id="latency", turn_id=f"turn-{i}", tool_call_id="call", surface="gateway")
            for name in HOOKS:
                start = time.perf_counter_ns()
                assert getattr(observer, name)(**native) is None
                timings[name].append((time.perf_counter_ns() - start) / 1e6)
        p99 = {name: sorted(values)[989] for name, values in timings.items()}
        print("HOOK_LATENCY_MS", dependency, json.dumps(p99, sort_keys=True))
        assert max(p99.values()) < 5
        from hermes_cli.plugins import PluginManager
        import hermes_cli.plugins as hp
        monkeypatch.setattr(hp, "_resolve_hook_callback_timeout", lambda: .1)
        manager = PluginManager()
        for name in HOOKS:
            manager._hooks[name] = [getattr(observer, name)]
            assert manager.invoke_hook(name, **NATIVE, tool_call_id="budget", surface="gateway") == []
    finally:
        if lock:
            lock.close()
        release.set()


@pytest.mark.parametrize("failure", ["constructor", "sqlite"])
def test_activity_initialization_failure_preserves_all_existing_registrations(home, failure):
    injected = '''
import sqlite3
from colonymind_hermes.body import Activity
failure = %r
def fail(*a, **kw):
    raise sqlite3.OperationalError("activity only")
if failure == "constructor":
    Activity.__init__ = fail
else:
    Activity._connect = fail
''' % failure
    prelude = PRELUDE.replace("manager.discover_and_load(force=True)", injected + "\nmanager.discover_and_load(force=True)")
    result = probe('''
expected = {"pre_llm_call":3,"post_llm_call":2,"pre_tool_call":2,"post_tool_call":4,
            "subagent_start":2,"pre_gateway_dispatch":3,"pre_api_request":1,
            "post_api_request":1,"api_request_error":1,"on_skill_lifecycle":1,
            "on_interim_message":1,"transform_llm_output":1,"gateway_platform_event":1,
            "on_kanban_dispatch_tick":1}
counts = {name:len(manager._hooks[name]) for name in expected}
emit(ok=loaded.enabled, counts=counts, expected=expected,
     available=colonymind_hermes.activity_capabilities()["available"],
     commands=list(manager._plugin_commands), sections=list(manager._system_prompt_sections))
''', home, prelude=prelude, env={"COLONYMIND_BODY_THREAD": "0"})
    assert result["ok"] and result["available"] is False
    assert all(result["counts"][key] >= count for key, count in result["expected"].items())
    assert "mind" in result["commands"] and "colonymind" in result["sections"]


def test_activity_503_or_stalled_delivery_never_delays_conversation(observer, tmp_path):
    outbox = TurnOutbox(tmp_path / "capture.sqlite3")
    entered, release = threading.Event(), threading.Event()
    def activity_post(*args, **kwargs):
        entered.set()
        release.wait(5)
        return httpx.Response(503)
    observer.client.post = activity_post
    observer.start()
    try:
        assert entered.wait(3)
        # Includes a legacy row, reproducing the reviewer's queue ordering.
        outbox.enqueue("activity:legacy:1", {"kind": "activity_snapshot"})
        outbox.enqueue("conversation-turn", {"turn_id": "conversation-turn", "session_id": "conversation", "contact_id": "owner"})
        calls = []
        def healthy_post(path, **kwargs):
            calls.append(path)
            return httpx.Response(503 if path.endswith("/activity/snapshot") else 200, json={"accepted": True})
        start = time.perf_counter()
        assert outbox.drain(lambda payload: deliver(SimpleNamespace(post=healthy_post), None, None, payload)) == 2
        assert time.perf_counter() - start < .1
        assert calls == ["/v1/host/turns/sync"]
    finally:
        release.set()


def test_completed_history_and_failure_backlog_are_bounded(observer):
    observer.client.post = lambda *a, **kw: httpx.Response(503)
    observer.start()
    for i in range(2500):
        native = dict(session_id="history", turn_id=f"ended-{i}")
        observer.pre_llm_call(**native)
        observer.post_llm_call(**native)
        observer.native_turn_settled(**native)
    wait_for(lambda: observer._initialized)
    assert len(observer._finished) <= activity.MAX_FINISHED
    assert observer._scopes == {} and len(observer._pending) <= 1
    with closing(observer._connect()) as conn:
        conn.execute("PRAGMA busy_timeout=5000")
        assert conn.execute("SELECT COUNT(*) FROM scopes").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM pending").fetchone()[0] == 1


def test_upgrade_preserves_the_existing_installation_revision(observer):
    outbox = TurnOutbox(observer.source)
    with closing(outbox._connect()) as conn, conn:
        conn.execute("CREATE TABLE activity_revision (id INTEGER PRIMARY KEY, revision INTEGER)")
        conn.execute("INSERT INTO activity_revision VALUES (1,500)")
    published = []
    observer.client.post = lambda *a, **kw: published.append(kw["json"]) or httpx.Response(200, json={"accepted": True})
    observer.start()
    wait_for(lambda: published)
    assert published[0]["revision"] == 501
    assert outbox.rows() == []


def test_actual_unresponsive_http_does_not_enter_any_hook(observer):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from colonymind_hermes.client import ColonyMindClient
    entered, release = threading.Event(), threading.Event()
    class Blackhole(BaseHTTPRequestHandler):
        def do_POST(self):
            entered.set()
            release.wait(3)
            self.close_connection = True
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Blackhole)
    server.daemon_threads = True
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    observer.client = ColonyMindClient(url=f"http://127.0.0.1:{server.server_port}", api_key="test")
    observer.start()
    database_lock = None
    try:
        assert entered.wait(3)
        database_lock = sqlite3.connect(observer.path, isolation_level=None)
        database_lock.execute("BEGIN EXCLUSIVE")
        timings = {name: [] for name in HOOKS}
        for i in range(1000):
            for name in HOOKS:
                start = time.perf_counter_ns()
                getattr(observer, name)(session_id="http-latency", turn_id=f"t-{i}", tool_call_id="call", surface="gateway")
                timings[name].append((time.perf_counter_ns() - start) / 1e6)
        p99 = {name: sorted(values)[989] for name, values in timings.items()}
        print("HOOK_LATENCY_MS actual_http_and_locked_sqlite", json.dumps(p99, sort_keys=True))
        assert max(p99.values()) < 5
    finally:
        if database_lock:
            database_lock.close()
        release.set()
        server.shutdown()
        server.server_close()


def test_settlement_observation_reuses_the_existing_capture_callback(home):
    result = probe('''
from colonymind_hermes.capture import Capture
capture = next(cb.__self__ for cb in manager._hooks["pre_llm_call"]
               if isinstance(getattr(cb, "__self__", None), Capture))
activity = colonymind_hermes._ACTIVITY
assert capture.activity_observer.__self__ is activity
native = dict(session_id="settlement-session", turn_id="settlement-turn")
activity.pre_llm_call(**native)
if "on_native_turn_settled" in VALID_HOOKS:
    callbacks = manager._hooks["on_native_turn_settled"]
    assert len(callbacks) == 2
    for callback in callbacks:
        callback(**native, outcome={"failed": True})
else:
    capture.native_turn_settled(**native, outcome={"failed": True})
scope = activity._scopes["settlement-session"]
emit(code=scope.state()[0], count=scope.state()[1])
''', home, prelude=MIND_PRELUDE, env={"COLONYMIND_BODY_THREAD": "0"})
    assert result == {"code": "error", "count": 0}


def test_broken_settlement_observer_cannot_prevent_capture_cleanup():
    from colonymind_hermes.capture import Capture
    calls = []
    capture = Capture.__new__(Capture)
    capture.work = SimpleNamespace(info=lambda *a, **kw: calls.append("capture") or None)
    def failed(**kwargs):
        raise RuntimeError(PRIVATE)
    capture.activity_observer = failed
    capture.native_turn_settled(**NATIVE)
    assert calls == ["capture"]


def test_older_parallel_completion_cannot_discard_the_latest_turns_failure(observer):
    newer = dict(session_id=NATIVE["session_id"], turn_id="newer-turn")
    observer.pre_llm_call(**NATIVE)
    observer.pre_llm_call(**newer)
    observer.post_llm_call(**newer)
    observer.post_llm_call(**NATIVE)
    observer.native_turn_settled(**NATIVE, outcome={"completed": True})
    observer.native_turn_settled(**newer, outcome={"failed": True})
    assert latest(observer) == [{"scope": "thread-1", "activity_code": "error", "count": 0}]


def test_contended_activity_lock_never_vetoes_tools(observer, monkeypatch):
    from hermes_cli import plugins
    monkeypatch.setattr(plugins, "_resolve_hook_callback_timeout", lambda: .1)
    manager = plugins.PluginManager()
    # A lock held by any thread must cause a dropped observation, never a wait.
    with observer._lock:
        for name in ("pre_tool_call", *HOOKS):
            manager._hooks[name] = [getattr(observer, name)]
            started = time.monotonic()
            assert manager.invoke_hook(name, **NATIVE, tool_call_id="call", surface="gateway") == []
            assert time.monotonic() - started < .05
    assert latest(observer) == []


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
@pytest.mark.parametrize("locked", [True, False], ids=["held-lock", "free-lock"])
def test_forked_activity_is_disabled_without_blocking_hermes(observer, monkeypatch, locked):
    from hermes_cli import plugins
    monkeypatch.setattr(plugins, "_resolve_hook_callback_timeout", lambda: .1)
    observer.pre_llm_call(**NATIVE)
    observer.start()
    wait_for(lambda: observer.available)
    held, release = threading.Event(), threading.Event()
    def holder():
        with observer._lock:
            held.set()
            release.wait(5)
    worker = threading.Thread(target=holder)
    if locked:
        worker.start()
        assert held.wait(2)
    reader, writer = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(reader)
            manager = plugins.PluginManager()
            manager._hooks["pre_tool_call"] = [observer.pre_tool_call]
            started = time.monotonic()
            result = manager.invoke_hook("pre_tool_call", **NATIVE, tool_call_id="child-call")
            os.write(writer, json.dumps(dict(result=result, elapsed=time.monotonic() - started,
                available=observer.available, pending=observer._pending, local=observer._local,
                scopes=len(observer._scopes), pid=observer.pid, actual_pid=os.getpid(),
                daemon_alive=bool(observer._thread and observer._thread.is_alive()))).encode())
        finally:
            os._exit(0)
    os.close(writer)
    try:
        assert select.select([reader], [], [], 3)[0], "child activity callback hung"
        state = json.loads(os.read(reader, 65536))
    finally:
        os.close(reader)
        release.set()
        if locked:
            worker.join(2)
        os.waitpid(pid, 0)
    assert state["result"] == []
    assert state["elapsed"] < .05
    assert state["available"] is False and state["daemon_alive"] is False
    assert state["pending"] == state["local"] == {} and state["scopes"] == 0
    assert state["pid"] == state["actual_pid"]
    # The parent still owns and publishes its original scope.
    assert observer.available and observer._thread.is_alive()
    assert latest(observer)[0]["activity_code"] == "thinking"


def test_unique_completed_sessions_are_bounded_while_sqlite_is_locked(observer):
    observer.pre_llm_call(**NATIVE)
    observer._persist()
    high_water = {"pending": 0, "local": 0}
    with closing(sqlite3.connect(observer.path, isolation_level=None)) as locked:
        locked.execute("BEGIN EXCLUSIVE")
        observer.post_llm_call(**NATIVE)
        observer.native_turn_settled(**NATIVE)
        for i in range(50000):
            native = dict(session_id=f"completed-session-{i}", turn_id=f"turn-{i}")
            observer.pre_llm_call(**native)
            observer.post_llm_call(**native)
            observer.native_turn_settled(**native)
            high_water["pending"] = max(high_water["pending"], len(observer._pending))
            if i % 1000 == 999:
                with pytest.raises(sqlite3.OperationalError):
                    observer._persist()
                high_water["local"] = max(high_water["local"], len(observer._local))
        assert observer._scopes == {}
        assert len(observer._finished) <= activity.MAX_FINISHED
        assert max(high_water.values()) <= activity.MAX_SCOPES, high_water
        assert observer._pending == observer._local == {}
    payload, _ = observer._persist()
    assert payload["activities"] == []
    with closing(observer._connect()) as conn:
        conn.execute("PRAGMA busy_timeout=5000")
        assert conn.execute("SELECT COUNT(*) FROM scopes").fetchone()[0] == 0


def test_native_gateway_stop_closes_activity_without_plugin_unload(home, sidecar):
    result = probe('''
import asyncio
import sqlite3
from pathlib import Path
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
activity = colonymind_hermes._ACTIVITY
activity.pre_llm_call(session_id="shutdown-session", turn_id="shutdown-turn")
deadline = time.monotonic() + 5
while not activity.available and time.monotonic() < deadline:
    time.sleep(.005)
assert activity.available
async def stop_gateway():
    runner = GatewayRunner(config=GatewayConfig(multiplex_profiles=False, loop_watchdog=False,
        sessions_dir=Path(os.environ["HERMES_HOME"]) / "sessions"))
    await runner.stop()
    return runner._shutdown_event.is_set()
shutdown_complete = asyncio.run(stop_gateway())
activity._thread.join(.3)
with sqlite3.connect(activity.path) as conn:
    owners = conn.execute("SELECT COUNT(*) FROM owners").fetchone()[0]
state = dict(shutdown_complete=shutdown_complete, alive=activity._thread.is_alive(),
    closing=activity._closing, available=activity.available, owners=owners)
manager.unload()
emit(**state)
''', home, env={"COLONYMIND_BODY_THREAD": "0"})
    assert result == {"shutdown_complete": True, "alive": False, "closing": True,
                      "available": False, "owners": 0}
    assert sidecar.calls("/v1/host/activity/snapshot", "POST")[-1]["json"]["activities"] == []


def test_profile_discovery_keeps_first_activity_owner(tmp_path):
    home = build_home(tmp_path, 1, "http://127.0.0.1:1")
    second_root = tmp_path / "second-profile"
    second_root.mkdir()
    second = build_home(second_root, 1, "http://127.0.0.1:1")
    shared = tmp_path / "shared-outbox.sqlite3"
    for profile in (home, second):
        config = yaml.safe_load((profile.hermes / "config.yaml").read_text())
        config["plugins"]["colonymind"]["turn_outbox_path"] = str(shared)
        (profile.hermes / "config.yaml").write_text(yaml.safe_dump(config))
    prelude = '''
import httpx
real_client = httpx.Client
def offline_client(*args, **kwargs):
    kwargs["transport"] = httpx.MockTransport(lambda request: httpx.Response(200, json={"accepted": True}))
    return real_client(*args, **kwargs)
httpx.Client = offline_client
''' + PRELUDE
    result = probe('''
from hermes_cli.plugins_loader import _plugin_home_scope
from pathlib import Path
def activity_of(manager):
    return next(cb.__self__ for cb in manager._hooks["pre_llm_call"]
        if isinstance(getattr(cb, "__self__", None), colonymind_hermes.body.Activity))
def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "profile activity did not publish"
        time.sleep(.005)
first = activity_of(manager)
first.pre_llm_call(session_id="first-profile", turn_id="first-turn")
wait_for(lambda: first.available and len(first._last_activities) == 1)
with _plugin_home_scope(Path(%r)):
    second_manager = get_plugin_manager()
    second_manager.discover_and_load(force=True)
assert not second_manager._plugins["colonymind"].hooks_registered
assert colonymind_hermes._ACTIVITY is first
snapshot = list(first._last_activities)
second_manager.unload()
# Verify the first worker can still publish new observations after the second load.
first.pre_tool_call(session_id="first-profile", turn_id="first-turn", tool_call_id="still-live")
deadline = time.monotonic() + 1
while first.failure_category is None and time.monotonic() < deadline:
    if any(row["scope"] == "first-profile" and row["activity_code"] == "tool_running"
           for row in first._last_activities):
        break
    time.sleep(.005)
state = dict(snapshot=snapshot, first_available=first.available,
    first_failure=first.failure_category, first_latest=first._last_activities)
manager.unload()
first._thread.join(3)
emit(**state)
''' % str(second.hermes), home, prelude=prelude, env={"COLONYMIND_BODY_THREAD": "0"})
    expected = [{"scope": "first-profile", "activity_code": "thinking", "count": 1}]
    assert result["snapshot"] == expected
    expected[0]["activity_code"] = "tool_running"
    assert result["first_latest"] == expected
    assert result["first_available"] and result["first_failure"] is None
