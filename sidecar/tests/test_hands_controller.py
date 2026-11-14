"""OC1 acceptance tests. All execution is local and uses a temporary database."""

import asyncio
import os
import socket
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from fastapi import HTTPException

from colonymind.config import DEFAULTS
from colonymind.hands import controller, wire


EPOCH = {"owner_pid": 123, "owner_started_at": 1000.0}
NEW_EPOCH = {"owner_pid": 124, "owner_started_at": 2000.0}
REF = {"platform": "test", "chat": "owner-chat", "message_id": "message-1"}


def contract(family="family-a", modalities=None):
    return {"version": "1", "family": family, "modalities": modalities or ["text"],
            "context_tokens": 100000, "max_output_tokens": 10000,
            "tool_calls": {"qualified": True, "success_share": 1.0}, "json": True,
            "effort": {"2": {"reasoning_effort": "high"}}, "decode_tok_s": 40,
            "admission": {"family": family, "priority": 10}}


@pytest_asyncio.fixture
async def c(tmp_path):
    if not os.environ.get("HANDS_TEST_POLL_LOOP"):
        left, right = socket.socketpair()
        try:
            right.send(b"x")
        except PermissionError:
            pytest.skip("Integrator: aiosqlite requires local socket-pair wakeups; sandbox denies them")
        finally:
            left.close()
            right.close()
    config = deepcopy(DEFAULTS["hands"])
    config.update(max_running=2, contracts={"provider-a": contract()},
                  routes={role: ["provider-a"] for role in ("deep", "coding", "media", "verify")})
    config["health"].update(default_family_cap=20, running_limits={"family-a": 6})
    config["locks"]["max_waiters"] = 1
    now = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
    instance = controller.Controller(tmp_path / "hands.db", config=config, clock=lambda: now[0])
    await instance.connect()
    instance.health.record("family-a", running=0, waiting=0, kv=0.1)
    instance.test_time = now
    yield instance
    await instance.close()


async def plan(c, key="plan-1", origin="owner_turn", **task):
    body = wire.PlanRequest(parent_session="parent", turn="turn-1", origin=origin,
                            origin_ref=REF, tasks=[{"goal": "Check the artifact", "kind": "deep", **task}])
    return (await c.plan(body, key))["plans"][0]


async def running(c, key="plan-1", **task):
    p = await plan(c, key, **task)
    assert p["admit"] == "start", p
    await c.dispatched(p["hand_id"], wire.DispatchedRequest(
        epoch=EPOCH, delegation_id=p["hand_id"], lease_token=p["lease_token"]), key + ":dispatch")
    await c.bind(p["hand_id"], wire.BindRequest(
        epoch=EPOCH, executor_ref=p["hand_id"], child_session=key + ":child"), key + ":bind")
    return p


async def finish(c, p, status="done", epoch=None, version=1):
    return await c.result(p["hand_id"], wire.ResultRequest(
        epoch=epoch or EPOCH, intent_version=version, status=status, summary_ref="branch-1",
        summary_hash="a" * 64, tool_digest=[], receipts=[], families_used=["family-a"]),
        p["hand_id"] + ":result")


async def test_fifty_concurrent_plans_never_exceed_two_and_replay(c):
    plans = await asyncio.gather(*(plan(c, f"plan-{i}") for i in range(50)))
    assert sum(p["admit"] == "start" for p in plans) == 2
    assert sum(p["admit"] == "queued" for p in plans) == 4
    assert sum(p["admit"] == "refuse" for p in plans) == 44
    assert await plan(c, "plan-0") == plans[0]
    assert (await c.one("SELECT count(*) AS n FROM hands"))["n"] == 50


async def test_fifty_plans_from_four_database_connections(c):
    peers = [controller.Controller(c.store.path, config=c.config, clock=c.clock) for _ in range(4)]
    try:
        for peer in peers:
            await peer.connect()
            peer.health.record("family-a", running=0, waiting=0, kv=0.1)
        plans = await asyncio.gather(*(plan(peers[n % 4], f"parallel-{n}") for n in range(50)))
        assert sum(p["admit"] == "start" for p in plans) == 2
        assert (await c.one("SELECT count(*) AS n FROM reservations WHERE state='held'"))["n"] == 2
    finally:
        for peer in peers:
            await peer.close()


async def test_reservation_expiry_and_restart_idempotence(c):
    p = await plan(c)
    await plan(c, "second")
    c.test_time[0] += timedelta(seconds=91)
    await c.sweep()
    assert (await c.get_hand(p["hand_id"]))["state"] == "expired"
    assert (await plan(c, "third"))["admit"] == "start"
    await c.close()
    await c.connect()
    assert await plan(c) == p


async def test_epoch_and_prior_state_are_guarded(c):
    p = await running(c)
    with pytest.raises(HTTPException, match="stale_epoch"):
        await finish(c, p, epoch=NEW_EPOCH)
    assert (await c.get_hand(p["hand_id"]))["state"] == "running"
    await finish(c, p)
    with pytest.raises(HTTPException):
        await c.bind(p["hand_id"], wire.BindRequest(
            epoch=EPOCH, executor_ref=p["hand_id"], child_session="other"), "late-bind")


async def test_revision_supersedes_old_result_and_explicit_accept_keeps_evidence(c):
    p = await running(c)
    task = await c.revise(p["task_id"], wire.ReviseRequest(goal="Check revised artifact"), "revise")
    assert task["intent_version"] == 2
    assert (await finish(c, p))["outcome"] == "superseded"
    accepted = await c.accept(p["task_id"], wire.AcceptRequest(
        hand=p["hand_id"], reason="Useful for revised goal", for_version=2), "accept")
    assert accepted["state"] == "done"
    assert (await c.get_hand(p["hand_id"]))["evidence"]["level"] == "E0"


async def test_second_writer_waits_third_refused(c):
    a = await plan(c, "first", touches=["repo:example"])
    b = await plan(c, "second", touches=["repo:example"])
    d = await plan(c, "third", touches=["repo:example"])
    assert [p["admit"] for p in (a, b, d)] == ["start", "queued", "refuse"]
    assert b["reason"] == "lock" and d["reason"] == "lock_queue_full"
    await c.stop(a["hand_id"], wire.StopRequest(reason="Finished"), "stop")
    promoted = await c.sweep()
    assert b["hand_id"] in promoted["reserved"]


async def test_seventh_hand_spends_whole_task_budget(c):
    p = await running(c)
    for attempt in range(6):
        await finish(c, p, status="partial")
        if attempt < 5:
            p = await running(c, f"attempt-{attempt}", continues=p["task_id"])
    seventh = await plan(c, "seventh", continues=p["task_id"])
    assert seventh["admit"] == "refuse" and seventh["reason"] == "task_budget"


async def test_two_failures_start_cooldown(c):
    p = await running(c)
    await finish(c, p, status="failed")
    p = await running(c, "again", continues=p["task_id"])
    await finish(c, p, status="failed")
    task = await c.get_task(p["task_id"])
    assert task["failures_in_row"] == 2 and task["cooldown_until"] > c.now()
    assert (await plan(c, "too-soon", continues=p["task_id"]))["reason"] == "cooldown"


async def test_daily_limit_holds_own_and_mind_never_owner(c):
    c.config["budgets"]["day"]["max_hands"] = 1
    p = await running(c, origin="own")
    await finish(c, p)
    for origin in ("own", "mind"):
        held = await plan(c, origin, origin=origin)
        assert held["admit"] != "start" and held["reason"] == "daily_budget"
    assert (await plan(c, "owner"))["admit"] == "start"


async def test_old_epoch_open_effect_unknown_never_restarts_without_reconciliation(c):
    p = await running(c)
    await c.events(p["hand_id"], wire.EventsRequest(epoch=EPOCH, effects=[wire.EffectInput(
        key=f"eff:{p['hand_id']}:call", tool="terminal", call_hash="b" * 64,
        workdir="/workspace", intent_at=c.now())]), "effect")
    result = await c.reconcile(wire.ReconcileRequest(epoch=NEW_EPOCH, live=[], ledger=[]), "restart")
    assert result["hands"][0]["state"] == "unknown_effect"
    assert (await plan(c, "retry", continues=p["hand_id"]))["reason"] == "unreconciled_effect"
    assert (await plan(c, "retry-task", continues=p["task_id"]))["reason"] == "unreconciled_effect"
    assert (await plan(c, "reviewed", continues=p["hand_id"], reconciled=True))["admit"] == "start"


async def test_usage_is_monotonic_daily_attributed_and_task_budget_persists(c):
    p = await running(c, origin="own")
    c.config["budgets"]["day"]["tokens_per_family"]["family-a"] = 20
    async def usage(tokens, key):
        return await c.events(p["hand_id"], wire.EventsRequest(epoch=EPOCH, events=[wire.EventInput(
            seq=tokens, at=c.now(), type="usage", payload={"tokens": tokens, "family": "family-a"},
            source="plugin", key=key)]), key)
    await usage(20, "usage-1")
    with pytest.raises(HTTPException):
        await usage(10, "usage-2")
    await finish(c, p, "partial")
    assert (await c.get_task(p["task_id"]))["usage"]["tokens"] == 20
    assert (await plan(c, "own-held", origin="own"))["reason"] == "daily_budget"
    assert (await plan(c, "mind-held", origin="mind"))["reason"] == "daily_budget"
    owner = await running(c, "owner-work")
    await c.update("tasks", "id", owner["task_id"], usage={"hands": 1, "tokens": 900000})
    await c.store.db.commit()
    await finish(c, owner, "partial")
    assert (await plan(c, "task-exhausted", continues=owner["task_id"]))["reason"] == "task_budget"


async def test_completed_effect_is_lost_on_restart_and_outcome_needs_intent(c):
    p = await running(c)
    intent = {"key": f"eff:{p['hand_id']}:call", "tool": "terminal", "call_hash": "b" * 64,
              "workdir": "/workspace", "intent_at": c.now()}
    outcome = {**intent, "outcome": "ok", "exit_code": 0, "done_at": c.now()}
    with pytest.raises(HTTPException):
        await c.events(p["hand_id"], wire.EventsRequest(epoch=EPOCH, effects=[outcome]), "too-early")
    await c.events(p["hand_id"], wire.EventsRequest(epoch=EPOCH, effects=[intent]), "intent")
    await c.events(p["hand_id"], wire.EventsRequest(epoch=EPOCH, effects=[outcome]), "outcome")
    await c.events(p["hand_id"], wire.EventsRequest(epoch=EPOCH, effects=[outcome]), "duplicate-outcome")
    assert (await c.reconcile(wire.ReconcileRequest(epoch=NEW_EPOCH, live=[], ledger=[]), "restart"))["hands"][0]["state"] == "lost"


@pytest.mark.parametrize("reason", ["token_budget", "wall_budget"])
async def test_plugin_budget_stop_still_records_the_partial_report(c, reason):
    from hashlib import sha256
    p = await running(c, "budget-" + reason)
    await c.stop(p["hand_id"], wire.StopRequest(reason=reason, epoch=EPOCH), "stop-" + reason)
    assert (await c.get_hand(p["hand_id"]))["state"] == "stopped"
    report = "Outcome: partial. Two of three sources checked."
    answer = await c.result(p["hand_id"], wire.ResultRequest(
        epoch=EPOCH, intent_version=1, status="partial", summary_ref="branch-1", summary_text=report,
        summary_hash=sha256(report.encode()).hexdigest(), tool_digest=[], receipts=[],
        families_used=["family-a"]), p["hand_id"] + ":result")
    assert answer["outcome"] == "stopped" and answer["next_step"] == "wait"
    hand = await c.get_hand(p["hand_id"])
    assert hand["outcome"] == "partial" and hand["decisions"]["result_text"] == report
    assert (await c.get_task(p["task_id"]))["state"] == "waiting"


async def test_other_stops_still_refuse_a_late_result(c):
    p = await running(c, "owner-stop")
    await c.stop(p["hand_id"], wire.StopRequest(reason="Stopped by owner", epoch=EPOCH), "stop-owner")
    with pytest.raises(HTTPException):
        await finish(c, p, "partial")


async def test_checking_holds_capacity_and_blocking_question_ends_attempt(c):
    p = await running(c)
    await running(c, "second")
    await c.events(p["hand_id"], wire.EventsRequest(epoch=EPOCH, events=[wire.EventInput(
        seq=1, at=c.now(), type="checking", payload={}, source="plugin", key="checking")]), "checking")
    assert (await plan(c, "third"))["admit"] == "queued"
    with pytest.raises(HTTPException):
        await c.question(p["hand_id"], wire.QuestionRequest(epoch=EPOCH, question="Too late"), "late-question")
    await finish(c, p)
    another = await running(c, "question-hand")
    await c.question(another["hand_id"], wire.QuestionRequest(epoch=EPOCH, question="Need input", blocking=True), "ask")
    assert (await c.get_hand(another["hand_id"]))["activity"] == "awaiting_input"
    with pytest.raises(HTTPException):
        await c.events(another["hand_id"], wire.EventsRequest(epoch=EPOCH), "tool-after-block")


async def test_erasure_scrubs_branches_questions_relays_and_replay(c):
    p = await running(c)
    q = await c.question(p["hand_id"], wire.QuestionRequest(
        epoch=EPOCH, question="Private question", assumption="Private assumption"), "question")
    await finish(c, p)
    await c.execute("INSERT INTO derivations VALUES ('relay', 'private-relay', ?, 'E0', 'branch-1', 'live')",
                    (p["hand_id"],))
    await c.store.db.commit()
    await c.erase_parent("parent", "erase")
    assert (await c.get_hand(p["hand_id"]))["branch"] is None
    row = await c.one("SELECT * FROM questions WHERE id=?", (q["id"],))
    assert row["text"] is None and row["assumption"] is None and row["answer"] is None
    assert not await c.all("SELECT * FROM derivations WHERE derived_kind='relay'")
    rows = await c.all("SELECT response FROM idempotency")
    assert "Private question" not in str(rows) and "branch-1" not in str(rows)


async def test_http_roundtrip_serializes_shared_records(c, monkeypatch):
    import httpx
    from fastapi import FastAPI, Request
    from colonymind.api.auth import key_authority
    from colonymind.api.routers.hands import router
    async def runtime():
        return c
    monkeypatch.setattr(controller, "get_controller", runtime)
    app = FastAPI()
    @app.middleware("http")
    async def owner(request: Request, call_next):
        request.state.colonymind_authority = key_authority()
        return await call_next(request)
    app.include_router(router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        body = {"parent_session": "parent", "turn": "turn-1", "origin": "owner_turn", "origin_ref": REF,
                "tasks": [{"goal": "Check the artifact", "kind": "deep"}]}
        response = await client.post("/v1/hands/plan", json=body, headers={"Idempotency-Key": "http-plan"})
        assert response.status_code == 200, response.text
        p = response.json()["plans"][0]
        response = await client.get("/v1/hands/" + p["hand_id"])
        assert response.status_code == 200, response.text
        assert response.json()["state"] == "reserved"
        response = await client.post("/v1/hands/" + p["hand_id"] + "/dispatched", json={
            "epoch": EPOCH, "delegation_id": "delegation", "lease_token": p["lease_token"]},
            headers={"Idempotency-Key": "http-dispatch"})
        assert response.status_code == 200, response.text
        assert response.json()["epoch"] == EPOCH


async def test_plugin_order_intent_then_dispatch_then_bind_by_subagent(c):
    # The plugin journals its dispatch intent before Hermes has a delegation id, and
    # binds with the child's subagent id, which differs from the delegation id.
    p = await plan(c)
    intent = wire.EventsRequest(epoch=EPOCH, events=[{"seq": 1, "at": "2026-01-01T00:00:00Z", "type": "dispatch_intent",
        "payload": {"task_id": p["task_id"], "intent_version": 1}, "source": "plugin", "key": "ev:x:1"}], effects=[])
    await c.events(p["hand_id"], intent, "intent")
    assert (await c.get_hand(p["hand_id"]))["state"] == "reserved"
    for refused in (wire.EventsRequest(epoch=EPOCH, events=[{"seq": 2, "at": "2026-01-01T00:00:00Z", "type": "progress",
                        "payload": {}, "source": "plugin", "key": "ev:x:2"}], effects=[]),):
        with pytest.raises(HTTPException) as caught:
            await c.events(p["hand_id"], refused, "early-progress")
        assert caught.value.status_code == 409
    await c.dispatched(p["hand_id"], wire.DispatchedRequest(epoch=EPOCH, delegation_id="delegation-1",
                                                            lease_token=p["lease_token"]), "dispatch")
    bound = await c.bind(p["hand_id"], wire.BindRequest(epoch=EPOCH, executor_ref="subagent-1",
                                                        child_session="child-1"), "bind")
    assert bound["state"] == "running" and bound["executor_ref"] == "subagent-1"
    assert bound["decisions"]["delegation_id"] == "delegation-1"
    rotated = await c.bind(p["hand_id"], wire.BindRequest(epoch=EPOCH, executor_ref="subagent-1",
                                                          child_session="child-1-rotated"), "bind-rotated")
    assert rotated["child_session"] == "child-1-rotated"
    with pytest.raises(HTTPException) as caught:
        await c.bind(p["hand_id"], wire.BindRequest(epoch=EPOCH, executor_ref="subagent-2",
                                                    child_session="child-2"), "bind-other")
    assert caught.value.detail == "executor_ref"


async def test_sweeper_passes_leave_no_idempotency_rows(c):
    await plan(c)
    before = (await c.one("SELECT count(*) AS n FROM idempotency"))["n"]
    for _ in range(5):
        await c.sweep()
    assert (await c.one("SELECT count(*) AS n FROM idempotency"))["n"] == before


async def test_non_owner_key_refused_before_controller(tmp_path):
    import httpx
    from fastapi import FastAPI
    from colonymind.api.routers.hands import router
    app = FastAPI()
    app.include_router(router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/v1/hands/", headers={"Authorization": "Bearer guest"})
    assert response.status_code == 403


async def test_report_and_cli_read_without_mutating(c):
    from argparse import ArgumentParser
    from colonymind.hands.cli import add_parser
    from colonymind.reports.hands import collect
    await running(c)
    report = collect(path=c.store.path)
    assert report["hands"]["running"] == 1 and report["obligations"]["open"] == 1
    parser = ArgumentParser()
    add_parser(parser.add_subparsers(dest="command"))
    for command in ("list", "tasks", "obligations", "report"):
        assert parser.parse_args(["hands", command]).hands_command == command
    assert parser.parse_args(["hands", "revise", "t-0123456789ab", "--goal", "Changed goal"]).goal == "Changed goal"


@pytest.mark.skipif(not os.environ.get("HANDS_INTEGRATOR_RUNNER"),
                    reason="Integrator: requires live Hermes, shared dispatch lock and manual cron runner from OC2")
def test_live_hermes_fifty_dispatches_with_cron():
    """Integrator supplies a harness and runs the full 3.5 cross-unit acceptance."""
    import os
    import subprocess
    subprocess.run([os.environ["HANDS_INTEGRATOR_RUNNER"], "concurrent-dispatch", "--plans", "50",
                    "--max-running", "2", "--manual-cron", "--assert-no-inline"], check=True)


async def test_served_controller_keeps_its_database_in_the_instance_directory(tmp_path, monkeypatch):
    # The instance directory holds colonymind.yaml, instinct.db and usage.db; hands.db sits beside them.
    from types import SimpleNamespace
    config = SimpleNamespace(home=tmp_path, data={"hands": deepcopy(DEFAULTS["hands"]), "mind": {"faculties": {}}})
    monkeypatch.setattr(controller, "load_config", lambda: config)
    monkeypatch.setattr(controller, "_runtime", None)
    monkeypatch.setattr(controller, "_runtime_lock", None)
    monkeypatch.setattr(controller.Controller, "start_maintenance", lambda self: None)
    served = await controller.get_controller()
    try:
        assert (tmp_path / "hands.db").is_file()
        assert not (tmp_path / "data").exists()
    finally:
        await served.close()


@pytest.mark.parametrize("needs", [[], ["shell"]])
async def test_shell_only_coding_without_a_workdir_is_planned_as_ops_and_recorded(c, needs):
    """Gate w18-20261007T182631Z: the front chose coding for a plain shell task with no workdir."""
    p = await plan(c, kind="coding", needs=needs, accept={"probe": "true"})
    assert p["kind"] == "ops" and p["admit"] == "start"
    task = await c.get_task(p["task_id"])
    assert task["kind"] == "ops" and task["resources"] == ["ops:local"]
    adjusted = {"from": "coding", "to": "ops", "reason": "coding_without_workdir_or_resources"}
    hand = await c.get_hand(p["hand_id"])
    assert hand["kind"] == "ops" and hand["decisions"]["kind_adjusted"] == adjusted
    events = await c.all("SELECT type, payload FROM events WHERE hand_id=?", (p["hand_id"],))
    assert {"type": "kind_adjusted", "payload": adjusted} in [dict(row) for row in events]
    assert p["budgets"] == c.config["budgets"]["attempt"]["ops"]


@pytest.mark.parametrize("task", [
    {"kind": "coding", "needs": ["code"]},
    {"kind": "coding", "needs": ["shell", "web"]},
    {"kind": "coding", "accept": {"test_command": "pytest -q"}},
    {"kind": "integrate"},
])
async def test_coding_that_names_code_work_still_needs_a_workdir_or_resources(c, task):
    with pytest.raises(HTTPException) as refused:
        await plan(c, **task)
    assert refused.value.status_code == 422 and refused.value.detail == "coding_requires_workdir_or_resources"
    assert (await c.one("SELECT count(*) AS n FROM hands"))["n"] == 0


@pytest.mark.parametrize("task", [{"workdir": "/workspace/repo"}, {"touches": ["service:mail"]}])
async def test_coding_with_a_workdir_or_resources_keeps_its_kind(c, task):
    p = await plan(c, kind="coding", **task)
    hand = await c.get_hand(p["hand_id"])
    assert p["kind"] == hand["kind"] == "coding" and "kind_adjusted" not in hand["decisions"]


async def test_a_continuation_of_an_adjusted_task_keeps_ops(c):
    first = await running(c, kind="coding", needs=["shell"])
    await finish(c, first, status="needs_input")
    # The plugin's continuation copies the task fields the front sent, including its kind.
    p = await plan(c, "plan-2", kind="coding", needs=["shell"], **{"continue": first["hand_id"]})
    hand = await c.get_hand(p["hand_id"])
    assert p["kind"] == hand["kind"] == "ops" and hand["continues"] == first["hand_id"]
    assert hand["decisions"]["kind_adjusted"]["from"] == "coding"
