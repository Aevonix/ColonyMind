"""Durable scenarios use real SQLite, controller transitions and approval checks.

These require asyncio's local socket pair and are marked for the integrator when
the execution sandbox denies it. No live memory or inference service is opened.
"""

import asyncio
import hashlib
import json
from pathlib import Path
import sys

from fastapi import HTTPException
import pytest

from colonymind.hands import approvals, controller, locks, wire
from owner_answer import owner_answers
from test_mind_loop import Fixture, pinned_clock  # noqa: F401

from .support import assert_scenario, disable_branch, paired
from .controller_support import EPOCH, NEXT_EPOCH, bind_plan, database, finish, plan, running  # noqa: F401


pytestmark = [pytest.mark.hands_sockets, pytest.mark.asyncio]


@paired
async def test_s03_restart_after_effect_intent(database, monkeypatch, mutated, runtime_factory):
    c = database
    attempt = await running(c, kind="ops", workdir=str(Path(c.store.path).parent),
                            accept={"probe": "printf probe-observed"})
    await c.events(attempt["hand_id"], wire.EventsRequest(epoch=EPOCH, effects=[{
        "key": f"eff:{attempt['hand_id']}:write", "tool": "terminal", "call_hash": "b" * 64,
        "workdir": str(Path(c.store.path).parent), "intent_at": c.now()}]), "intent")
    await c.close()
    await c.connect()
    result = await c.reconcile(wire.ReconcileRequest(epoch=NEXT_EPOCH, live=[], ledger=[]), "restart")
    hand = result["hands"][0]
    # The real plugin runner submits its observation through the controller route.
    from colonymind.api.routers import hands as routes
    monkeypatch.setattr(controller, '_runtime', c)
    rig = runtime_factory()
    loop = asyncio.get_running_loop()
    def get(path, params=None):
        operation = c.get_task(path.rsplit('/', 1)[-1]) if '/tasks/' in path else c.get_hand(path.rsplit('/', 1)[-1])
        return asyncio.run_coroutine_threadsafe(operation, loop).result()
    def emit(path, body, key):
        return asyncio.run_coroutine_threadsafe(routes.probe(hand['id'], wire.ProbeRequest(**body), key), loop).result()
    rig.runtime.client.get = get
    rig.runtime.emit = emit
    await asyncio.to_thread(rig.runtime.probe_unknown, hand['id'])
    if mutated:
        disable_branch(monkeypatch, controller.Controller, "plan", "unreconciled and not item.reconciled")
    continuation = await plan(c, "retry", continues=attempt["hand_id"])
    digest = await c.digest(wire.DigestRequest(session="owner-session"))
    probe_shown = "probe-observed" in digest["text"]
    assert_scenario(3, {"state": hand["state"], "retries": int(continuation["admit"] == "start"),
        "continue_refused": continuation.get("reason") == "unreconciled_effect",
        "probe_shown": probe_shown}, mutated)


@paired
async def test_s04_completion_delivered_twice(database, monkeypatch, mutated):
    c = database
    artifact = Path(c.store.path).parent / "artifact.txt"
    from colonymind.turns.idempotency import TurnIdempotencyLedger
    from importlib import import_module
    from hands_fake_hermes import load_hands
    c.ledger = TurnIdempotencyLedger(Path(c.store.path).parent / 'turns.db')
    c.owner_id = 'owner'
    command = [sys.executable, "-c", "from pathlib import Path; import sys; "
               "assert Path(sys.argv[1]).read_text() == 'checked artifact'; print('artifact=checked')", str(artifact)]
    attempt = await running(c, kind='ops', accept={"probe": command, "requirements": ['artifact=checked']})
    artifact.write_text("checked artifact")
    task = await c.get_task(attempt['task_id'])
    check = import_module(load_hands().__name__ + '.checks').CheckRunner()
    receipts = check.finalize(check.run(kind='ops', acceptance=task['intents']['1']['acceptance'],
                                        workdir=artifact.parent), artifact.parent)
    first = await finish(c, attempt, receipts=receipts)
    assert first["evidence"]["level"] == "E3", "Real saved-plan check must reach E3"
    if mutated:
        disable_branch(monkeypatch, controller.Controller, "write", "prior")
    await c.close()
    await c.connect()
    try:
        second = await finish(c, attempt, receipts=receipts)
    except HTTPException:
        second = None
    entries = await c.all("SELECT * FROM derivations WHERE hand_id=? AND derived_kind='autobiography'",
                          (attempt["hand_id"],))
    notices = await c.all("SELECT id FROM notices WHERE id=?", (f"ntc:{attempt['hand_id']}:end",))
    results = await c.all("SELECT key FROM idempotency WHERE path=?", (f"/{attempt['hand_id']}/result",))
    assert_scenario(4, {"results": len(results), "entries": len(entries), "notices": len(notices),
        "evidence": first["evidence"]["level"], "same_result": first == second,
        "entry_hand": entries[0]["hand_id"] if entries else None}, mutated)


@paired
async def test_s06_goal_revised_mid_run(database, monkeypatch, mutated):
    c = database
    attempt = await running(c)
    revised = await c.revise(attempt["task_id"], wire.ReviseRequest(goal="Inspect the new artifact"), "revise")
    if mutated:
        disable_branch(monkeypatch, controller.Controller, "result",
                       "task['intent_version'] != hand['intent_version']")
    outcome = await finish(c, attempt)
    task = await c.get_task(attempt["task_id"])
    digest = await c.digest(wire.DigestRequest(session="owner-session"))
    item = next(item for item in digest["items"] if item.get("hand_id") == attempt["hand_id"])
    assert_scenario(6, {"version": revised["intent_version"], "outcome": outcome["outcome"],
        "task_state": task["state"], "relay_done": ": done" in item["text"]}, mutated)


@paired
async def test_s07_repository_writers_serialize(database, tmp_path, monkeypatch, mutated):
    c = database
    # Fixture headroom isolates the resource guard from the independent slot cap.
    c.config["max_running"] = 2
    c.config["health"]["default_family_cap"] = 2
    resource = "repo:" + str(tmp_path / "workspace")
    first = await running(c, "writer-one", kind="coding", touches=[resource])
    if mutated:
        async def no_lock(*args):
            return None
        monkeypatch.setattr(locks, "check", no_lock)
        monkeypatch.setattr(locks, "acquire", no_lock)
    second = await plan(c, "writer-two", kind="coding", touches=[resource])
    artifact = tmp_path / "write-order.txt"

    def write(line):
        with artifact.open("a") as stream:
            stream.write(line + "\n")

    write("first:start")
    if second["admit"] == "start":
        write("second:start")
    write("first:end")
    await c.stop(first["hand_id"], wire.StopRequest(reason="First writer finished"), "finish-one")
    swept = await c.sweep()
    promoted = second["hand_id"] in swept["reserved"]
    if promoted:
        write("second:start")
    if promoted or second["admit"] == "start":
        write("second:end")
    assert_scenario(7, {"second_admit": second["admit"], "reason": second.get("reason"),
        "writes": artifact.read_text().splitlines(), "promoted": promoted}, mutated)


@paired
async def test_s12_approved_file_changed_before_call(database, tmp_path, monkeypatch, mutated):
    c = database
    mind = Fixture(tmp_path / "mind")
    try:
        attempt = await running(c, kind="coding")
        file = tmp_path / "approved.txt"
        file.write_text("approved artifact")
        before = hashlib.sha256(file.read_bytes()).hexdigest()
        body = wire.HoldRequest(epoch=EPOCH, call={"tool": "terminal", "arguments": {
            "command": f"rm {file}"}}, workdir=str(tmp_path), targets=[{
                "kind": "file", "target": str(file), "value": before}])
        service = approvals.Approvals(c.store, mind.mind)
        held = await service.hold(attempt["hand_id"], body, "hold")
        await owner_answers(mind.mind, held.ask_id)
        file.write_text("changed artifact")
        after = hashlib.sha256(file.read_bytes()).hexdigest()
        if mutated:
            disable_branch(monkeypatch, approvals.Approvals, "_check", "bound != targets")
        result = await service.check(wire.PermitCheckRequest(hand=attempt["hand_id"],
            call_digest=approvals.call_digest(body.call, body.workdir), targets_now=json.dumps([
                {"kind": "file", "target": str(file), "value": after}])), "check-after-change")
        if result.decision == "allow":
            file.unlink()  # Only this disposable fixture artifact is executed against.
        permits = await c.all("SELECT * FROM permits")
        old = next(row for row in permits if row["id"] == held.permit_id)
        fresh = [row for row in permits if row["id"] != held.permit_id]
        assert_scenario(12, {"old_state": old["state"], "ask_ids": [row["ask_id"] for row in permits],
            "new_state": fresh[0]["state"] if fresh else None, "executed": int(not file.exists()),
            "old_hash": before, "new_hash": after}, mutated)
    finally:
        mind.store.close()


@paired
async def test_s13_two_corrections_conflicts_outage_recovery(database, monkeypatch, mutated):
    c = database
    first = await running(c, goal="Inspect the first artifact")
    tid = first["task_id"]
    second_goal, final_goal = "Inspect the second artifact", "Inspect the latest artifact"
    versions, conflicts = [1], []
    revised = await c.revise(tid, wire.ReviseRequest(goal=second_goal), "correction-one")
    versions.append(revised["intent_version"])
    one = await finish(c, first, summary="The first artifact wins")
    conflicts.append(one["outcome"])
    c.health.record("text-family", running=0, waiting=1, kv=0.1)
    # Ask for a retry while the engine is unavailable. It must wait until a new
    # healthy reading and a released stage-1 reservation allow the next attempt.
    waiting = await plan(c, "outage-retry", continues=tid)
    outage = waiting["admit"] != "start"
    c.health.record("text-family", running=0, waiting=0, kv=0.1)
    swept = await c.sweep()
    assert waiting["hand_id"] in swept["reserved"]
    promoted = await c.get_hand(waiting["hand_id"])
    second = await bind_plan(c, {**waiting, "lease_token": promoted["decisions"]["admission"]["lease_token"]},
                             "attempt-two")
    revised = await c.revise(tid, wire.ReviseRequest(goal=final_goal), "correction-two")
    versions.append(revised["intent_version"])
    if mutated:
        disable_branch(monkeypatch, controller.Controller, "result",
                       "task['intent_version'] != hand['intent_version']")
    two = await finish(c, second, version=2, summary="The second artifact wins")
    conflicts.append(two["outcome"])
    c.health.record("text-family", running=0, waiting=0, kv=0.1)
    task = await c.get_task(tid)
    recovered = False
    final_version, final_hash = task["intent_version"], ""
    if task["state"] != "done":
        last = await running(c, "latest-attempt", continues=tid)
        done = await finish(c, last, version=3, summary=final_goal)
        recovered = done["outcome"] == "done"
        current = await c.get_task(tid)
        final_hash = current["intents"]["3"]["goal_hash"]
        await c.told(current["obligation_id"], wire.ToldRequest(turn_ref="owner-final-reply"), "communicated")
    obligations = await c.all("SELECT state FROM obligations WHERE task_id=? ORDER BY rowid", (tid,))
    assert_scenario(13, {"versions": versions, "conflicts": conflicts, "outage": outage,
        "recovered": recovered, "final_version": final_version, "final_goal": final_hash,
        "expected_goal": hashlib.sha256(final_goal.encode()).hexdigest(),
        "obligations": [row["state"] for row in obligations]}, mutated)
