"""Profile discovery must not detach the gateway's owner session map."""
from collections import defaultdict
import json
import logging
import sys

import httpx
import pytest

import colonymind_hermes as plugin
from conftest import PRELUDE, build_home, probe
from guard_offline import MOCK_HTTP, offline_home


class Context:
    def __init__(self):
        self.hooks = defaultdict(list)
        self.tools = []
        self.commands = []
        self.sections = []
        self.unloads = []
        self.middleware = []
        self.redactions = []

    def register_hook(self, name, callback):
        self.hooks[name].append(callback)
        return object()

    def register_tool(self, **registration):
        self.tools.append(registration)
        return object()

    def register_command(self, *args, **kwargs):
        self.commands.append((args, kwargs))

    def register_system_prompt_section(self, *args):
        self.sections.append(args)

    def on_unload(self, callback):
        self.unloads.append(callback)

    def register_middleware(self, *args):
        self.middleware.append(args)

    def register_redaction_patterns(self, patterns):
        self.redactions.extend(patterns)


@pytest.fixture
def registration(tmp_path, monkeypatch):
    home = build_home(tmp_path, 1, "http://127.0.0.1:1")
    monkeypatch.setenv("HERMES_HOME", str(home.hermes))
    monkeypatch.setenv("COLONYMIND_HOME", str(home.instance))
    monkeypatch.setenv("COLONYMIND_SKIP_DOTENV", "1")
    monkeypatch.setenv("COLONYMIND_BODY_THREAD", "1")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(plugin, "_BODY", None)
    monkeypatch.setattr(plugin, "_ACTIVITY", None)
    monkeypatch.setattr(plugin, "_REGISTERED_TOOLS", None, raising=False)
    # Restore the bridge even though register() itself writes sys.modules.
    monkeypatch.setitem(sys.modules, "colonymind_capture", object())
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: real_client(
        *a, **{**kw, "transport": httpx.MockTransport(lambda r: httpx.Response(404, json={}))}))
    starts = defaultdict(list)
    for name in ("Body", "ArchiveWorker", "Activity"):
        monkeypatch.setattr(getattr(plugin, name), "start",
                            lambda self, *a, _name=name: starts[_name].append(self))
    return home, starts


@pytest.mark.parametrize("process", ["gateway", "kanban", "cli"])
def test_register_twice_preserves_process_owner(registration, monkeypatch, caplog, process):
    home, starts = registration
    if process == "kanban":
        monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    elif process == "cli":
        monkeypatch.setenv("COLONYMIND_BODY_THREAD", "0")
    first, second = Context(), Context()
    plugin.register(first)
    body, activity = plugin._BODY, plugin._ACTIVITY
    bridge = sys.modules["colonymind_capture"]
    capture = first.hooks["pre_llm_call"][0].__self__
    assert capture.sessions is body.sessions
    assert bridge.record_hand_refs.__self__ is capture
    capture.pre_llm_call(session_id="owner-before", platform="telegram", sender_id="1001")
    owner = body.sessions.get("owner-before")
    assert owner is not None and owner.sender_id == "1001"
    assert len(starts["Body"]) == (1 if process == "gateway" else 0)
    assert len(starts["ArchiveWorker"]) == len(starts["Activity"]) == 1
    first_starts = {name: list(workers) for name, workers in starts.items()}
    evidence = (home.instance / "hermes-plugin-registration.json").read_bytes()

    other_home = home.hermes.parent / "other-profile"
    monkeypatch.setenv("HERMES_HOME", str(other_home))
    with caplog.at_level(logging.WARNING, logger=plugin.__name__):
        plugin.register(second)

    assert plugin._BODY is body
    assert plugin._BODY.sessions is capture.sessions
    assert plugin._BODY.sessions.get("owner-before") is owner
    assert sys.modules["colonymind_capture"] is bridge
    assert plugin._ACTIVITY is activity
    assert dict(starts) == first_starts
    capture.pre_llm_call(session_id="owner-after", platform="telegram", sender_id="1001")
    assert plugin._BODY.sessions.get("owner-after").sender_id == "1001"
    assert (home.instance / "hermes-plugin-registration.json").read_bytes() == evidence
    # Discovery keeps the toolset pin, but cannot call into the owner's state.
    assert len(second.tools) == len(first.tools)
    assert {row["toolset"] for row in second.tools} == {"colonymind"}
    for row, original in zip(second.tools, first.tools):
        assert {k: v for k, v in row.items() if k not in ("handler", "check_fn")} == {
            k: v for k, v in original.items() if k not in ("handler", "check_fn")}
        assert row["handler"] is not original["handler"]
        assert row.get("check_fn") is None
        assert json.loads(row["handler"]({"text": "PRIVATE-TURN"}, session_id="owner-before")) == {
            "error": "ColonyMind is active for another profile in this process"}
    warnings = [r for r in caplog.records if r.name == plugin.__name__ and r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert str(other_home.resolve()) in warnings[0].message
    assert "gets no ColonyMind" in warnings[0].message
    assert "PRIVATE-TURN" not in caplog.text
    assert not any((second.hooks, second.commands, second.sections, second.unloads,
                    second.middleware, second.redactions))


@pytest.mark.parametrize("stage", ["hook", "archive"])
def test_failed_registration_rolls_back_and_retries(registration, monkeypatch, stage):
    home, starts = registration
    failed = Context()
    if stage == "hook":
        def fail(*args):
            assert plugin._BODY is not None
            raise RuntimeError("injected hook failure")
        monkeypatch.setattr(failed, "register_hook", fail)
    else:
        def fail(self, ctx):
            assert plugin._BODY is not None and plugin._ACTIVITY is not None
            raise RuntimeError("injected archive failure")
        monkeypatch.setattr(plugin.ArchiveWorker, "start", fail)
    with pytest.raises(RuntimeError, match="injected"):
        plugin.register(failed)
    assert plugin._BODY is None
    assert plugin._ACTIVITY is None
    assert plugin._REGISTERED_TOOLS is None
    assert "colonymind_capture" not in sys.modules
    if stage == "archive":
        assert starts["Body"][0]._stop.is_set()
        assert starts["Activity"][0]._done.is_set()
    monkeypatch.setattr(plugin.ArchiveWorker, "start", lambda self, ctx: starts["ArchiveWorker"].append(self))
    fresh = Context()
    plugin.register(fresh)
    body, activity, tools = plugin._BODY, plugin._ACTIVITY, plugin._REGISTERED_TOOLS
    # Hermes may dispose the failed attempt after a successful retry.
    for cleanup in reversed(failed.unloads):
        cleanup()
    assert plugin._BODY is body and plugin._ACTIVITY is activity
    assert plugin._REGISTERED_TOOLS is tools
    assert fresh.hooks and fresh.commands and fresh.sections
    for cleanup in reversed(fresh.unloads):
        cleanup()
    assert plugin._BODY is plugin._ACTIVITY is plugin._REGISTERED_TOOLS is None


@pytest.mark.parametrize("park_at", ["settings", "owner_tracked", "activity_start", "before_publish"])
def test_timed_out_load_allows_full_rediscovery(offline_home, park_at):
    # Exercise Hermes's real deadline worker, abandoned context and failure
    # disposal, including release before and after acquiring process resources.
    prelude = MOCK_HTTP + PRELUDE.replace("manager.discover_and_load(force=True)", '''
import colonymind_hermes
from hermes_cli import plugins_loader
# On a Hermes without the hook on_native_turn_settled the adapter's first registration in a process imports the
# agent loop, and that import alone outlasts the 0.1 s deadline below: Hermes then refuses the later hook
# registrations and the loader never starts the activity. Imported here, every park point is reached in time.
import agent.turn_facade, agent.conversation_loop
resume = threading.Event()
entered = threading.Event()
workers = []
load_settings = colonymind_hermes.load_settings
on_unload = colonymind_hermes._Registration.on_unload
def park():
    workers.append(threading.current_thread())
    entered.set()
    assert resume.wait(10), "test did not resume the abandoned loader"
def delayed_settings():
    if PARK_AT == "settings":
        park()
    return load_settings()
def tracked_unload(self, callback):
    result = on_unload(self, callback)
    if PARK_AT == "owner_tracked" and callback.__name__ == "release_owner":
        park()
    return result
colonymind_hermes.load_settings = delayed_settings
colonymind_hermes._Registration.on_unload = tracked_unload
starts = {"Body": [], "ArchiveWorker": [], "Activity": []}
for name in starts:
    getattr(colonymind_hermes, name).start = lambda self, *a, _name=name: starts[_name].append(self)
archive_start = colonymind_hermes.ArchiveWorker.start
activity_start = colonymind_hermes.Activity.start
def delayed_activity(self):
    activity_start(self)
    if PARK_AT == "activity_start":
        park()
colonymind_hermes.Activity.start = delayed_activity
def delayed_archive(self, ctx):
    archive_start(self, ctx)
    if PARK_AT == "before_publish":
        park()
colonymind_hermes.ArchiveWorker.start = delayed_archive
resolve_timeout = plugins_loader._resolve_plugin_load_timeout
plugins_loader._resolve_plugin_load_timeout = lambda: 0.1
disposed = threading.Event()
errors = []
def discover():
    try:
        manager.discover_and_load(force=True)
    except BaseException as error:
        errors.append(error)
    finally:
        disposed.set()
loader = threading.Thread(target=discover)
loader.start()
try:
    assert entered.wait(5), "loader never reached the park point"
    assert disposed.wait(2), "Hermes disposal waited on the parked registration"
    assert not errors, errors
    assert all(worker.is_alive() for worker in workers) and not resume.is_set()
    failed = manager._plugins["colonymind"]
    assert not failed.enabled and "load timed out" in failed.error, failed.error
    assert colonymind_hermes._BODY is colonymind_hermes._ACTIVITY is None
    assert colonymind_hermes._REGISTERED_TOOLS is None
finally:
    resume.set()
    loader.join(10)
    for worker in workers:
        worker.join(10)
        assert not worker.is_alive()
    assert not loader.is_alive()
    plugins_loader._resolve_plugin_load_timeout = resolve_timeout
    colonymind_hermes.load_settings = load_settings
    colonymind_hermes._Registration.on_unload = on_unload
    colonymind_hermes.ArchiveWorker.start = archive_start
    colonymind_hermes.Activity.start = activity_start
assert colonymind_hermes._BODY is colonymind_hermes._ACTIVITY is None
assert colonymind_hermes._REGISTERED_TOOLS is None
assert "colonymind_capture" not in sys.modules
before_retry = {name: len(items) for name, items in starts.items()}
manager.discover_and_load(force=True)
'''.replace("PARK_AT", repr(park_at)))
    result = probe('''
assert loaded.hooks_registered, "rediscovery replayed refusing tools after timeout"
assert loaded.tools_registered
capture = sys.modules["colonymind_capture"].record_hand_refs.__self__
assert capture.sessions is colonymind_hermes._BODY.sessions
capture.pre_llm_call(session_id="after-timeout", platform="telegram", sender_id="1001")
assert colonymind_hermes._BODY.sessions.get("after-timeout") is not None
assert all(len(items) == before_retry[name] + 1 for name, items in starts.items()), starts
manager.unload()
emit(cleared=colonymind_hermes._REGISTERED_TOOLS is None and
     colonymind_hermes._BODY is None and colonymind_hermes._ACTIVITY is None)
''', offline_home, prelude=prelude, env={"COLONYMIND_BODY_THREAD": "1"})
    assert result["cleared"]


def test_released_load_allows_full_registration(registration, monkeypatch):
    # Deterministically dispose after hooks were accepted but before ownership
    # is published, covering the other side of the abandoned-load guard.
    failed = Context()
    start = plugin.ArchiveWorker.start
    def dispose(self, ctx):
        for cleanup in reversed(failed.unloads):
            cleanup()
    monkeypatch.setattr(plugin.ArchiveWorker, "start", dispose)
    plugin.register(failed)
    monkeypatch.setattr(plugin.ArchiveWorker, "start", start)
    fresh = Context()
    plugin.register(fresh)
    assert fresh.hooks, "registration replayed refusing tools after release"
    assert fresh.tools and fresh.commands and fresh.sections
    capture = fresh.hooks["pre_llm_call"][0].__self__
    assert capture.sessions is plugin._BODY.sessions
    for cleanup in reversed(fresh.unloads):
        cleanup()


def test_replay_only_includes_accepted_tools(registration, monkeypatch):
    first, second = Context(), Context()
    register_tool = first.register_tool
    def accept_tool(**row):
        if row["name"] != "colonymind_consult":
            return register_tool(**row)
    monkeypatch.setattr(first, "register_tool", accept_tool)
    plugin.register(first)
    plugin.register(second)
    assert first.tools
    assert {row["name"] for row in second.tools} == {row["name"] for row in first.tools}
    for cleanup in reversed(first.unloads):
        cleanup()


@pytest.mark.parametrize("targeted_unload", [False, True])
def test_hermes_owner_unload_and_force_reload(offline_home, targeted_unload):
    prelude = MOCK_HTTP + PRELUDE.replace("manager.discover_and_load(force=True)", '''
import colonymind_hermes
starts = {"Body": [], "ArchiveWorker": [], "Activity": []}
for name in starts:
    getattr(colonymind_hermes, name).start = lambda self, *a, _name=name: starts[_name].append(self)
manager.discover_and_load(force=True)
''')
    result = probe('''
first_body, first_activity = colonymind_hermes._BODY, colonymind_hermes._ACTIVITY
first_bridge = sys.modules["colonymind_capture"]
old_tools = set(loaded.tools_registered)
if %r:
    assert manager.unload("colonymind")
    assert colonymind_hermes._REGISTERED_TOOLS is None
    assert colonymind_hermes._BODY is colonymind_hermes._ACTIVITY is None
    assert "colonymind_capture" not in sys.modules
manager.discover_and_load(force=True)
reloaded = manager._plugins["colonymind"]
assert reloaded.enabled and not reloaded.error, reloaded.error
assert colonymind_hermes._BODY is not first_body
assert colonymind_hermes._ACTIVITY is not first_activity
assert first_body._stop.is_set() and first_activity._done.is_set()
assert sys.modules["colonymind_capture"] is not first_bridge
assert set(reloaded.tools_registered) == old_tools
capture = sys.modules["colonymind_capture"].record_hand_refs.__self__
assert capture.sessions is colonymind_hermes._BODY.sessions
capture.pre_llm_call(session_id="after-reload", platform="telegram", sender_id="1001")
assert colonymind_hermes._BODY.sessions.get("after-reload") is not None
assert reloaded.hooks_registered
assert all(len(workers) == 2 for workers in starts.values()), starts
manager.unload()
emit(cleared=colonymind_hermes._REGISTERED_TOOLS is None and
     colonymind_hermes._BODY is None and colonymind_hermes._ACTIVITY is None)
''' % targeted_unload, offline_home, prelude=prelude, env={"COLONYMIND_BODY_THREAD": "1"})
    assert result["cleared"]


@pytest.mark.parametrize("config, enabled", [
    ({}, True),
    ({"platform_toolsets": {"cli": ["terminal", "colonymind"]}}, True),
    ({"platform_toolsets": {"cli": ["terminal"]},
      "known_plugin_toolsets": {"cli": ["colonymind"]}}, False),
    ({"agent": {"disabled_toolsets": ["colonymind"]}}, False),
])
def test_worker_toolset_pin_survives_second_manager(offline_home, config, enabled):
    offline_home.write_config(**config)
    prelude = MOCK_HTTP + PRELUDE.replace("manager.discover_and_load(force=True)", """
import colonymind_hermes
colonymind_hermes.Body.start = lambda self: None
colonymind_hermes.ArchiveWorker.start = lambda self, ctx: None
colonymind_hermes.Activity.start = lambda self: None
manager.discover_and_load(force=True)
""")
    result = probe('''
from pathlib import Path
from hermes_cli.kanban_db_dispatch import _resolve_worker_cli_toolsets, _worker_profile_scope
from hermes_cli.plugins import get_plugin_toolset_keys_nowait
from tools.registry import registry
home = Path(os.environ["HERMES_HOME"])
worker = home.parent / "worker-profile"
worker.mkdir()
(worker / "config.yaml").write_bytes((home / "config.yaml").read_bytes())
body = colonymind_hermes._BODY
bridge = sys.modules["colonymind_capture"]
first = _resolve_worker_cli_toolsets(str(home))
first_keys = get_plugin_toolset_keys_nowait()
second = _resolve_worker_cli_toolsets(str(worker))
with _worker_profile_scope(str(worker)):
    other = get_plugin_manager()
    second_keys = get_plugin_toolset_keys_nowait()
    other_loaded = other._plugins["colonymind"]
    assert other is not manager and other_loaded.enabled, other_loaded.error
    assert set(other_loaded.tools_registered) == set(loaded.tools_registered)
    assert not other_loaded.hooks_registered
    for name in other_loaded.tools_registered:
        secondary = registry.get_entry(name, scope=other.scope_key)
        primary = registry.get_entry(name, scope=manager.scope_key)
        assert secondary.schema == primary.schema
        assert secondary.handler is not primary.handler
        assert secondary.check_fn is None
        assert json.loads(secondary.handler({}, session_id="owner"))["error"] == (
            "ColonyMind is active for another profile in this process")
    other.unload()
emit(first=first, second=second, first_keys=sorted(first_keys), second_keys=sorted(second_keys),
     body_preserved=body is colonymind_hermes._BODY,
     bridge_preserved=bridge is sys.modules.get("colonymind_capture"))
''', offline_home, prelude=prelude)
    assert result["first"] is not None
    assert result["first"] == result["second"]
    assert ("colonymind" in result["second"]) is enabled
    assert result["first_keys"] == result["second_keys"]
    assert "colonymind" in result["second_keys"]
    assert result["body_preserved"] and result["bridge_preserved"]
