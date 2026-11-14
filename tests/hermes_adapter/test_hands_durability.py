"""Durable hand transitions survive lost acknowledgements and executor restarts."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def modules():
    name = "hands_durability_test"
    package = ModuleType(name)
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "plugins/colonymind-hands")]
    sys.modules[name] = package
    loaded = {part: importlib.import_module(f"{name}.{part}")
              for part in ("spool", "effects", "reconcile", "degraded")}
    yield SimpleNamespace(**loaded)
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            sys.modules.pop(key, None)


EPOCH = {"owner_pid": 100, "owner_started_at": 1.0}
NEXT_EPOCH = {"owner_pid": 101, "owner_started_at": 2.0}
HAND = "h-0123456789ab"


def test_spool_fsyncs_before_delivery_and_preserves_private_permissions(modules, tmp_path, monkeypatch):
    synced = []
    fsync = os.fsync

    def sync(fd):
        synced.append(fd)
        return fsync(fd)

    monkeypatch.setattr(modules.spool.os, "fsync", sync)
    spool = modules.spool.Spool(tmp_path / "spool")
    spool.append("/v1/hands/plan", {"tasks": [{"goal": "check"}]}, "plan:one")
    assert synced
    sent = []

    def send(path, body, key):
        assert synced
        sent.append(key)
        return {"plans": [{"hand_id": HAND}]}

    assert spool.replay(send) == 1
    assert sent == ["plan:one"]
    for path in (tmp_path / "spool").glob("*.jsonl"):
        assert path.stat().st_mode & 0o777 == 0o600
    restored = modules.spool.Spool(tmp_path / "spool")
    assert restored.pending() == []
    assert restored.response("plan:one") == {"plans": [{"hand_id": HAND}]}


def test_lost_ack_replays_original_key_without_duplicate_effects(modules, tmp_path):
    spool = modules.spool.Spool(tmp_path)
    spool.append("/v1/hands/plan", {"tasks": [1]}, "plan:once")
    spool.append("/v1/hands/plan", {"tasks": [1]}, "plan:once")
    responses = {}
    attempts = []

    def send(path, body, key):
        attempts.append(key)
        responses.setdefault(key, {"number": len(responses) + 1})
        if len(attempts) == 1:
            raise ConnectionError("ack lost")
        return responses[key]

    assert spool.replay(send) == 0
    restored = modules.spool.Spool(tmp_path)
    assert restored.replay(send) == 1
    assert restored.replay(send) == 0
    assert attempts == ["plan:once", "plan:once"]
    assert len(responses) == 1


def test_spool_keeps_order_and_rejects_changed_idempotency_payload(modules, tmp_path):
    spool = modules.spool.Spool(tmp_path)
    spool.append("/first", {"n": 1}, "one")
    spool.append("/second", {"n": 2}, "two")
    with pytest.raises(ValueError):
        spool.append("/first", {"n": 3}, "one")
    attempted = []
    assert spool.replay(lambda path, body, key: attempted.append(key)) == 0
    assert attempted == ["one"]
    assert [row["key"] for row in spool.pending()] == ["one", "two"]


def test_spool_recovers_only_torn_tail(modules, tmp_path):
    spool = modules.spool.Spool(tmp_path)
    spool.append("/first", {}, "one")
    path, = tmp_path.glob("*.jsonl")
    with path.open("ab") as stream:
        stream.write(b'{"type":"request",')
    restored = modules.spool.Spool(tmp_path)
    restored.append("/second", {}, "two")
    assert [row["key"] for row in restored.pending()] == ["one", "two"]
    assert all(json.loads(line) for line in path.read_text().splitlines())


def test_failed_fsync_prevents_effect_execution(modules, tmp_path, monkeypatch):
    spool = modules.spool.Spool(tmp_path)
    journal = modules.effects.EffectJournal(spool, EPOCH)

    def broken(fd):
        raise OSError("disk unavailable")

    monkeypatch.setattr(modules.spool.os, "fsync", broken)
    with pytest.raises(OSError):
        journal.intent(HAND, "call-a", "terminal", {"command": "touch file"})
    assert not spool.writable


def test_effect_intent_precedes_tool_and_outcome_closes_it(modules, tmp_path):
    spool = modules.spool.Spool(tmp_path)
    journal = modules.effects.EffectJournal(spool, EPOCH)
    journal.intent(HAND, "call-a", "terminal", {"command": "touch file"}, "/work")
    open_effect, = journal.open_effects(HAND)
    assert open_effect["key"] == f"eff:{HAND}:call-a"
    assert open_effect["done_at"] is None
    assert spool.pending()[0]["body"]["effects"][0]["key"] == open_effect["key"]
    journal.outcome(HAND, "call-a", status="success", result={"exit_code": 0})
    assert journal.open_effects(HAND) == []
    assert len(spool.pending()) == 2
    restored = modules.effects.EffectJournal(modules.spool.Spool(tmp_path), NEXT_EPOCH)
    assert restored.open_effects(HAND) == []


def test_degraded_window_and_owner_only_single_admission(modules):
    mode = modules.degraded.DegradedMode()
    mode.failure(now=0)
    assert not mode.is_degraded
    mode.failure(now=31)
    assert not mode.is_degraded
    mode.failure(now=45)
    assert mode.is_degraded
    assert mode.admit("owner", 0)[0]
    assert not mode.admit("owner", 1)[0]
    assert not mode.admit("own", 0)[0]
    mode.success()
    assert not mode.is_degraded


def test_restart_reads_hermes_ledger_and_open_effect_never_restarts(modules, tmp_path):
    spool = modules.spool.Spool(tmp_path)
    spool.append(f"/v1/hands/{HAND}/dispatched", {
        "epoch": EPOCH, "delegation_id": "delegation-1", "lease_token": "lease-1",
    }, f"disp:{HAND}:old")
    spool.append(f"/v1/hands/{HAND}/bind", {
        "epoch": EPOCH, "executor_ref": "child-1", "child_session": "session-1",
    }, f"bind:{HAND}:old")
    modules.effects.EffectJournal(spool, EPOCH).intent(
        HAND, "call-a", "terminal", {"command": "touch file"})
    restored = modules.spool.Spool(tmp_path)
    emitted = []
    runtime = SimpleNamespace(spool=restored, epoch=NEXT_EPOCH,
                              effects=modules.effects.EffectJournal(restored, NEXT_EPOCH),
                              emit=lambda path, body, key: emitted.append((path, body, key)),
                              registry=SimpleNamespace(all=lambda: []))
    looked_up = []

    def lookup(delegation_id):
        looked_up.append(delegation_id)
        return {"delegation_id": delegation_id, "state": "unknown"}

    outcomes = modules.reconcile.reconcile(runtime, ledger_lookup=lookup)
    assert looked_up == ["delegation-1"]
    assert outcomes == {HAND: "unknown_effect"}
    assert emitted[-1][0] == "/v1/hands/reconcile"
    assert emitted[-1][1]["ledger"][0]["status"] == "unknown"
    assert emitted[-1][1]["ledger"][0]["epoch"] == EPOCH


def test_restart_without_open_effect_is_lost(modules, tmp_path):
    spool = modules.spool.Spool(tmp_path)
    spool.append(f"/v1/hands/{HAND}/dispatched", {
        "epoch": EPOCH, "delegation_id": "delegation-1", "lease_token": "lease-1",
    }, "disp:old")
    runtime = SimpleNamespace(spool=spool, epoch=NEXT_EPOCH,
                              effects=modules.effects.EffectJournal(spool, NEXT_EPOCH),
                              emit=lambda *args: {}, registry=SimpleNamespace(all=lambda: []))
    assert modules.reconcile.reconcile(runtime, ledger_lookup=lambda ref: None) == {HAND: "lost"}


def hook_runtime(modules, tmp_path):
    from hands_fake_hermes import FakeContext

    context = FakeContext()
    spool = modules.spool.Spool(tmp_path)
    hand = SimpleNamespace(hand_id=HAND, kind="coding", blocked=False, plan={})
    runtime = SimpleNamespace(spool=spool, epoch=EPOCH,
                              effects=modules.effects.EffectJournal(spool, EPOCH),
                              registry=SimpleNamespace(for_session=lambda sid, **kwargs:
                                                       hand if sid == "child" else None),
                              emit=lambda path, body, key: spool.append(path, body, key),
                              stop=lambda hand, reason: None)
    dispatch = ModuleType("hands_durability_test.dispatch")
    dispatch.get_runtime = lambda ctx: runtime
    sys.modules[dispatch.__name__] = dispatch
    modules.effects.register(context)
    modules.degraded.register(context)
    return context, runtime, hand


def test_real_hook_shape_journals_before_tool_and_owner_is_unchanged(modules, tmp_path):
    context, runtime, hand = hook_runtime(modules, tmp_path)
    assert context.fire("pre_tool_call", tool_name="terminal", args={"command": "touch file"},
                        session_id="owner", tool_call_id="owner-call") == [None, None]
    assert runtime.spool.records() == []
    assert context.fire("pre_tool_call", tool_name="terminal", args={"command": "touch file"},
                        session_id="child", tool_call_id="child-call") == [None, None]
    assert runtime.effects.open_effects(HAND)
    context.fire("post_tool_call", tool_name="terminal", args={"command": "touch file"},
                 session_id="child", tool_call_id="child-call", status="success", result="done")
    assert runtime.effects.open_effects(HAND) == []


def test_timed_out_tool_stays_open_and_duplicate_block_cannot_confirm_it(modules, tmp_path):
    context, runtime, hand = hook_runtime(modules, tmp_path)
    params = dict(tool_name="terminal", args={"command": "touch file"},
                  session_id="child", tool_call_id="child-call")
    context.fire("pre_tool_call", **params)
    context.fire("post_tool_call", **params, status="cancelled", result="timeout")
    assert runtime.effects.open_effects(HAND)
    directives = context.fire("pre_tool_call", **params)
    assert directives[0]["action"] == "block"
    context.fire("post_tool_call", **params, status="blocked", result="duplicate")
    assert runtime.effects.open_effects(HAND)


def test_duplicate_veto_cannot_close_an_unconfirmed_original_effect(modules, tmp_path):
    context, runtime, hand = hook_runtime(modules, tmp_path)
    params = dict(tool_name="terminal", args={"command": "touch file"},
                  session_id="child", tool_call_id="child-call")
    context.fire("pre_tool_call", **params)
    assert context.fire("pre_tool_call", **params)[0]["action"] == "block"
    context.fire("post_tool_call", **params, status="blocked", result="duplicate")
    assert runtime.effects.open_effects(HAND)


def test_budget_or_allow_veto_is_not_journaled_as_an_effect(modules, tmp_path):
    context, runtime, hand = hook_runtime(modules, tmp_path)
    hand.refused_calls = {"vetoed-call"}
    directives = context.fire("pre_tool_call", tool_name="terminal", args={"command": "touch file"},
                              session_id="child", tool_call_id="vetoed-call")
    assert directives[0]["action"] == "block"
    assert runtime.spool.records() == []


def test_broken_spool_refuses_coding_tool_without_raising_into_hermes(modules, tmp_path):
    context, runtime, hand = hook_runtime(modules, tmp_path)
    runtime.spool.writable = False
    directives = context.fire("pre_tool_call", tool_name="terminal", args={"command": "touch file"},
                              session_id="child", tool_call_id="child-call")
    assert all(result["action"] == "block" for result in directives)
    assert hand.blocked


def test_journal_keeps_exact_call_for_reconciliation_without_expanding_wire_contract(modules, tmp_path):
    journal = modules.effects.EffectJournal(modules.spool.Spool(tmp_path), EPOCH)
    journal.intent(HAND, "call-a", "terminal", {"command": "touch file"}, "/work")
    assert journal.call(HAND, "call-a") == {
        "tool": "terminal", "arguments": {"command": "touch file"}, "workdir": "/work"}
    row, = journal.spool.records()
    assert set(row["body"]) == {"epoch", "events", "effects"}


def test_degraded_replay_adopts_controller_ids_and_lease_from_durable_plan_ack(modules, tmp_path):
    import hashlib

    spool = modules.spool.Spool(tmp_path)
    key = "plan:owner:turn:call:0"
    suffix = hashlib.sha256(key.encode()).hexdigest()[:12]
    local = "h-" + suffix
    spool.append("/v1/hands/plan", {"tasks": [{"goal": "check"}]}, key)
    spool.acknowledge(key, {"plans": [{"hand_id": HAND, "task_id": "t-0123456789ab",
                                     "lease_token": "remote-lease", "admit": "start"}]})
    path, body, replay_key = modules.degraded.remap_request(
        spool, f"/v1/hands/{local}/dispatched", {"epoch": EPOCH,
                "delegation_id": "delegation-1", "lease_token": "local-" + suffix}, "disp:original")
    assert path == f"/v1/hands/{HAND}/dispatched"
    assert body["lease_token"] == "remote-lease"
    assert replay_key == "disp:original"


def test_recorded_handle_kill_rechecks_owner_and_process_identity(modules):
    process = SimpleNamespace(id="process-1", owner_task_id="child-task", pid=123,
                              host_start_time=40, pid_scope="host", exited=False)
    killed = []
    registry = SimpleNamespace(get=lambda sid: process,
                               kill_process=lambda sid, **kwargs: killed.append(sid))
    hand = SimpleNamespace(plan={})
    assert modules.reconcile.record_process(hand, {"session_id": "process-1", "pid": 123},
        task_id="child-task", registry=registry, start_time=lambda pid: 40)
    assert modules.reconcile.stop_recorded_processes(hand, registry=registry,
                                                    start_time=lambda pid: 41) == []
    process.owner_task_id = "other-task"
    assert modules.reconcile.stop_recorded_processes(hand, registry=registry,
                                                    start_time=lambda pid: 40) == []
    process.owner_task_id = "child-task"
    assert modules.reconcile.stop_recorded_processes(hand, registry=registry,
                                                    start_time=lambda pid: 40) == ["process-1"]
    assert killed == ["process-1"]


def test_foreground_result_and_remote_handle_are_never_guessed_for_kill(modules):
    process = SimpleNamespace(id="process-1", owner_task_id="child-task", pid=123,
                              host_start_time=40, pid_scope="sandbox", exited=False)
    registry = SimpleNamespace(get=lambda sid: process)
    hand = SimpleNamespace(plan={})
    assert not modules.reconcile.record_process(hand, {"exit_code": 0}, task_id="child-task",
                                               registry=registry, start_time=lambda pid: 40)
    assert not modules.reconcile.record_process(hand, {"session_id": "process-1", "pid": 123},
        task_id="child-task", registry=registry, start_time=lambda pid: 40)


@pytest.mark.parametrize("stopped", [False, True])
def test_orphan_or_stopped_attempt_with_open_effect_remains_unknown(modules, tmp_path, stopped):
    spool = modules.spool.Spool(tmp_path)
    journal = modules.effects.EffectJournal(spool, EPOCH)
    journal.intent(HAND, "call-a", "terminal", {"command": "touch file"})
    if stopped:
        spool.append(f"/v1/hands/{HAND}/stop", {"epoch": EPOCH, "reason": "cancelled"}, "stop:one")
    runtime = SimpleNamespace(spool=spool, epoch=NEXT_EPOCH,
                              effects=modules.effects.EffectJournal(spool, NEXT_EPOCH),
                              emit=lambda *args: {}, registry=SimpleNamespace(all=lambda: []))
    assert modules.reconcile.reconcile(runtime, ledger_lookup=lambda ref: None) == {HAND: "unknown_effect"}
