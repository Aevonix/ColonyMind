"""Runtime recovery exercises the real wrapper against socket-free Hermes seams."""

from copy import deepcopy
import importlib
import json
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from hands_fake_hermes import FakeContext, FakeController, FakeHermes, Parent, load_hands


@pytest.fixture
def factory(tmp_path):
    runtimes = []

    def create(*, directory="spool", client=None, faculty=True):
        package = load_hands()
        dispatch = importlib.import_module(package.__name__ + ".dispatch")
        native, controller = FakeHermes(), client or FakeController()
        config = {"faculty": faculty, "max_running": 3, "platforms": ["sms"],
                  "contracts": {"text-route": {
                      "version": "1", "family": "text", "modalities": ["text"],
                      "context_tokens": 100000, "max_output_tokens": 2000,
                      "tool_calls": True, "decode_tok_s": 100}}}
        runtime = dispatch.Runtime(config=config, client=controller, spool_dir=tmp_path / directory,
                                   native=native, owner_check=lambda sid: sid.startswith("owner"))
        native.runtime = runtime
        runtime.install()
        package.register_route_resolver(lambda role, effort: native.native_policy())
        runtimes.append(runtime)
        return runtime, native, controller

    yield create
    for runtime in runtimes:
        runtime.close()


def outage_hand(runtime, client):
    client.down = True
    parent = Parent()
    first = runtime.dispatch_tasks([{"goal": "first", "kind": "coding"}], parent=parent,
                                   tool_call_id="first")
    assert first["hands"][0]["reason"] == "controller_unavailable"
    second = runtime.dispatch_tasks([{"goal": "second", "kind": "coding"}], parent=parent,
                                    tool_call_id="second")
    assert second["hands"][0]["status"] == "dispatched"
    return runtime.registry.get(second["hands"][0]["hand_id"])


def test_recovered_local_dispatch_replays_using_server_hand_and_lease(factory):
    runtime, native, client = factory()
    hand = outage_hand(runtime, client)
    row = runtime.effects.intent(hand.hand_id, "effect-a", "terminal", {"command": "touch file"})
    runtime.emit(row["path"], row["body"], row["key"])
    client.down = False
    runtime.maintain()
    assert runtime.spool.pending() == []
    remote = next(response["plans"][0] for key, response in client.responses.items()
                  if key.endswith(":second:0"))
    dispatched = [(path, body) for path, body, _ in client.writes if path.endswith("/dispatched")]
    assert dispatched == [(f"/v1/hands/{remote['hand_id']}/dispatched", {
        "epoch": runtime.epoch, "delegation_id": hand.delegation_id,
        "lease_token": remote["lease_token"]})]
    assert any(path == f"/v1/hands/{remote['hand_id']}/events" and body.get("effects")
               for path, body, _ in client.writes)
    execution_writes = lambda: [row for row in client.writes
                               if row[0] != '/v1/hands/definitions/queue/claim']
    writes = deepcopy(execution_writes())
    runtime.maintain()
    assert execution_writes() == writes
    assert len(native.calls) == 1


def test_recovered_server_identity_keeps_the_local_executor_for_owner_controls(factory):
    runtime, native, client = factory()
    hand = outage_hand(runtime, client)
    client.down = False
    runtime.maintain()
    remote = next(response["plans"][0] for key, response in client.responses.items()
                  if key.endswith(":second:0"))
    client.get = lambda path, params=None: {
        "id": remote["hand_id"], "task_id": remote["task_id"], "parent_session": hand.parent_session_id,
        "name": remote["name"], "state": "running", "intent_version": 1}
    controls = importlib.import_module(type(runtime).__module__.rsplit(".", 1)[0] + ".tools")
    result = controls.execute_hands({"action": "stop", "hand": remote["hand_id"]},
                                    runtime=runtime, session_id=hand.parent_session_id)
    assert not result.get("error"), result
    assert hand.blocked
    assert native.stops


def test_refused_recovery_adoption_stops_the_local_attempt(factory):
    class RefusingController(FakeController):
        def post(self, path, body, key):
            result = super().post(path, body, key)
            if path.endswith("/plan"):
                result["plans"][0].update(admit="refuse", reason="resource_conflict", lease_token=None)
            return result

    runtime, native, client = factory(client=RefusingController())
    hand = outage_hand(runtime, client)
    client.down = False
    runtime.maintain()
    assert hand.blocked
    assert native.stops
    assert runtime.spool.pending(), "Unadopted execution evidence must remain durable"


def test_disabled_faculty_uses_local_controls_without_creating_controller_plan(factory):
    runtime, native, client = factory(faculty=False)
    result = runtime.dispatch_tasks([{"goal": "check", "kind": "research"}], parent=Parent())
    assert result["hands"][0]["status"] == "dispatched"
    assert len(native.calls) == 1
    assert not any(path == "/v1/hands/plan" for path, _, _ in client.writes)
    runtime.maintain()
    assert not any(path == "/v1/hands/plan" for path, _, _ in client.writes)


def test_start_replays_old_open_effect_and_reconciles_without_waiting_ten_seconds(factory):
    old, old_native, client = factory()
    result = old.dispatch_tasks([{"goal": "check", "kind": "coding"}], parent=Parent())
    hand = old.registry.get(result["hands"][0]["hand_id"])
    old.effects.intent(hand.hand_id, "effect-a", "terminal", {"command": "touch file"})
    old.close()
    observed = threading.Event()
    post = client.post

    def observe(path, body, key):
        response = post(path, body, key)
        if path == "/v1/hands/reconcile":
            observed.set()
        return response

    client.post = observe
    new, native, _ = factory(client=client)
    new.epoch = {"owner_pid": old.epoch["owner_pid"] + 1,
                 "owner_started_at": old.epoch["owner_started_at"] + 1}
    new.effects.epoch = new.epoch
    new.start()
    assert observed.wait(0.3), "Load must replay and reconcile before the periodic ten-second wait"
    assert not native.calls
    effects_index = max(i for i, (_, body, _) in enumerate(client.writes) if body.get("effects"))
    reconcile_index = next(i for i, (path, _, _) in enumerate(client.writes) if path == "/v1/hands/reconcile")
    assert effects_index < reconcile_index


def test_clean_start_and_maintenance_only_poll_the_definition_queue(factory):
    runtime, native, client = factory()
    runtime.start()
    runtime.maintain()
    assert client.writes
    assert all(path == '/v1/hands/definitions/queue/claim' and body == {'parent_session': None}
               for path, body, _ in client.writes)
    assert not native.calls


def test_unwritable_spool_refuses_new_dispatch_without_uncontrolled_native_call(factory):
    runtime, native, client = factory()
    runtime.spool.writable = False
    response = json.loads(native.delegate_module.delegate_task(goal="check", parent_agent=Parent()))
    assert response.get("error") or response["hands"][0]["status"] == "refused"
    assert not native.calls
    assert client.writes == []


def test_stopped_attempt_with_unconfirmed_effect_requires_reconciliation(factory):
    runtime, native, client = factory()
    parent = Parent()
    result = runtime.dispatch_tasks([{"goal": "check", "kind": "coding"}], parent=parent)
    hand = runtime.registry.get(result["hands"][0]["hand_id"])
    runtime.effects.intent(hand.hand_id, "effect-a", "terminal", {"command": "touch file"})
    runtime.stop(hand, "cancelled")
    response = runtime.continue_hand(hand)
    assert response.get("error") == "reconciliation required"
    assert len(native.calls) == 1


def test_fsync_failure_during_plugin_load_installs_a_dispatch_refusal(tmp_path, monkeypatch):
    package = load_hands()
    dispatch = importlib.import_module(package.__name__ + ".dispatch")
    spool_module = importlib.import_module(package.__name__ + ".spool")
    native = FakeHermes()
    monkeypatch.setattr(dispatch, "NativeHermes", lambda: native)
    adapter = ModuleType("colonymind_hermes")
    adapter.__path__ = []
    client = ModuleType("colonymind_hermes.client")
    client.load_settings = lambda: SimpleNamespace(home=tmp_path)
    client.read_yaml = lambda path: {}
    monkeypatch.setitem(sys.modules, "colonymind_hermes", adapter)
    monkeypatch.setitem(sys.modules, "colonymind_hermes.client", client)
    monkeypatch.setitem(sys.modules, "tools.delegate_tool", native.delegate_module)

    def fail_sync(fd):
        raise OSError("disk unavailable")

    monkeypatch.setattr(spool_module.os, "fsync", fail_sync)
    dispatch.register(FakeContext())
    result = json.loads(native.delegate_module.delegate_task(goal="check", parent_agent=Parent()))
    assert result["error"] == "hands runtime unavailable"
    assert not native.calls
