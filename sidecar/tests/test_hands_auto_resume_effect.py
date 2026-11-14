"""A real owner grant drives one continuation through its permit and effect."""

import importlib
import itertools
import subprocess
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from starlette.requests import Request

from colonymind.api.auth import key_authority
from colonymind.api.routers import mind as mind_routes
from colonymind.hands import approvals, controller, wire
from test_hands_controller import contract
from test_hands_permits import event_loop, pinned_clock  # noqa: F401
from test_mind_loop import Fixture


@pytest.fixture
def world(tmp_path, monkeypatch, event_loop, request):
    from history_fixtures import require_history
    # The ops-only case does not clone a coding workspace and needs no history.
    if request.node.name in {
        'test_answer_automatically_runs_the_held_effect_once_and_completes',
        'test_a_held_call_in_a_broad_cwd_never_widens_the_continuation_workspace',
    }:
        require_history(Path(__file__).resolve().parents[2], 'HEAD')
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tests/hermes_adapter"))
    from hands_fake_hermes import FakeHermes, Parent, load_hands

    mind = Fixture(tmp_path / "mind")
    config = {"max_running": 3, "platforms": ["rcs"], "contracts": {"text-route": contract("text")},
              "routes": {role: ["text-route"] for role in ("deep", "coding", "media", "verify")},
              "health": {"default_family_cap": 20, "running_limits": {"text": 6}}}
    current = controller.Controller(tmp_path / "hands.db", config=config, clock=lambda: mind.now)
    run = event_loop.run_until_complete
    run(current.connect())
    current.health.record("text", running=0, waiting=0, kv=0.1)
    service = approvals.Approvals(current.store, mind.mind)
    package = load_hands()
    dispatch = importlib.import_module(package.__name__ + ".dispatch")
    authority = importlib.import_module(package.__name__ + ".authority")
    effects = importlib.import_module(package.__name__ + ".effects")
    native = FakeHermes(platform="rcs")
    writes = []

    class LocalController:
        def post(self, path, body, key):
            try:
                return self._post(path, body, key)
            except HTTPException as error:
                # As over HTTP: ControllerClient raises the refusal's status with its JSON detail.
                request = httpx.Request("POST", "http://controller" + path)
                response = httpx.Response(error.status_code, json={"detail": error.detail}, request=request)
                raise httpx.HTTPStatusError(str(error.status_code), request=request, response=response) from None

        def _post(self, path, body, key):
            writes.append((path, body, key))
            action = path.rsplit("/", 1)[-1]
            schema = getattr(wire, action.title() + "Request")
            request = schema.model_validate(body)
            if action in {"plan", "reconcile"}:
                return run(getattr(current, action)(request, key))
            ident = path.split("/")[-2]
            target = service if action == "hold" else current
            value = run(getattr(target, action)(ident, request, key))
            return value.model_dump() if hasattr(value, "model_dump") else value

        def get(self, path, params=None):
            if path == "/v1/hands/permits/held":
                run(service.refresh_pending())
                return {**run(service.held_permits(params["session"])).model_dump(), "refreshed": True}
            ident = path.rsplit("/", 1)[-1]
            return run(current.get_task(ident) if "/tasks/" in path else current.get_hand(ident))

        def _request(self, method, path, *, params, key):
            assert method == "GET" and path == "/v1/hands/permits/check"
            return run(service.check(wire.PermitCheckRequest(**params), key)).model_dump()

    runtime = dispatch.Runtime(config=current.config, client=LocalController(), spool_dir=tmp_path / "spool",
                               native=native, owner_check=lambda sid: sid == "owner-session")
    native.runtime = runtime
    # This fake executor supplies its effective cwd without importing Hermes
    # into the sidecar's interpreter. Authority still checks the exact effect.
    runtime.workdir_for = lambda tool, args, task_id: args.get("workdir") if tool == "terminal" else None
    runtime.install()
    package.register_route_resolver(lambda role, effort: native.native_policy())
    hooks = {}
    effects.register(SimpleNamespace(hands_runtime=runtime,
        register_hook=lambda name, fn: hooks.update({name: fn})))
    monkeypatch.setattr(mind_routes, "_mind", mind.mind)
    world = SimpleNamespace(run=run, current=current, runtime=runtime, native=native, writes=writes,
                            mind=mind, parent=Parent(), executions=[], answers=itertools.count(1))

    def execute(hand, call_id, arguments, effect=None):
        call = dict(tool_name="terminal", args=arguments, session_id=hand.child_session_id, tool_call_id=call_id)
        verdict = hooks["pre_tool_call"](**call) or authority.pre_tool_call(runtime, **call)
        if verdict:
            hooks["post_tool_call"](**call, status="blocked", result=verdict)
            return verdict
        # The local executor only reaches the filesystem after the real exact-call
        # permit check; no shell, model, socket or device is involved.
        world.executions.append(hand.hand_id)
        if effect is not None:
            effect()
        hooks["post_tool_call"](**call, status="success", result={"exit_code": 0})

    def permits():
        return [dict(row) for row in run(current.store.db.execute_fetchall("SELECT * FROM permits ORDER BY rowid"))]

    def approve(permit):
        number = next(world.answers)
        app = FastAPI()
        app.state.hands_store = current.store
        request = Request({"type": "http", "method": "POST", "path": "/v1/mind/answers", "headers": [],
                           "app": app, "state": {"colonymind_authority": key_authority()}})
        mind.mind.record_shown([permit["ask_id"]], platform="rcs", chat="5550100",
                               message_id=f"ask-message-{number}", at=mind.now - timedelta(seconds=30))
        return run(mind_routes.answers(mind_routes.AnswersBody(platform="rcs", contact_id=mind.mind.owner_id,
            chat_id="5550100", chat_type="dm", message_id=f"owner-answer-{number}", turn_id=f"owner-answer-{number}",
            arrived_at=mind.now.isoformat(), text="yes", reply_to_message_id=f"ask-message-{number}"), request))

    def resume(held, answer):
        """The holding attempt ends; the owner's grant starts exactly one continuation."""
        calls = len(native.calls)
        assert "automatically" in runtime.resume_approvals(world.parent.session_id, answer["granted_hands"])
        runtime.subagent_stop(child_session_id=held.child_session_id, child_summary="Held for approval")
        native.live[held.delegation_id]["status"] = "completed"
        native.active_count = lambda: sum(row["status"] == "running" for row in native.live.values())
        runtime.maintain()
        assert len(native.calls) == calls + 1
        resumed = runtime.registry.all()[-1]
        assert resumed.plan["task"]["continue"] == held.hand_id
        return resumed

    world.execute, world.permits, world.approve, world.resume = execute, permits, approve, resume
    try:
        yield world
    finally:
        runtime.close()
        run(current.close())
        mind.store.close()


def clone(path):
    # Coding acceptance binds a git tree. Keep that tree and the terminal cwd
    # independent of earlier tests that loaded the native terminal module.
    source = Path(__file__).resolve().parents[2]
    from history_fixtures import require_history
    require_history(source, 'HEAD')
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(source), str(path)],
                   check=True, capture_output=True)
    return path


def test_answer_automatically_runs_the_held_effect_once_and_completes(world, tmp_path):
    run, current, runtime, native = world.run, world.current, world.runtime, world.native
    workdir = clone(tmp_path / "work")
    target = tmp_path / "report.txt"
    target.write_text("approved fixture")
    arguments = {"command": f"rm {target}", "workdir": str(workdir)}

    runtime.dispatch_tasks([{"goal": "Remove the approved fixture", "kind": "coding",
                             "workdir": str(workdir), "accept": {"test_command": f"test ! -e {target}"}}],
                           parent=world.parent)
    original = runtime.registry.all()[0]
    assert original.status == "running"
    assert world.execute(original, "held-call", arguments)["action"] == "block" and target.exists()
    permit = world.permits()[0]

    answer = world.approve(permit)
    assert "automatically" in runtime.resume_approvals(world.parent.session_id, answer["granted_hands"])
    runtime.maintain()
    assert len(native.calls) == 1 and target.exists()

    runtime.subagent_stop(child_session_id=original.child_session_id, child_summary="Held for approval")
    assert original.status == "stopped"
    native.live[original.delegation_id]["status"] = "completed"
    native.active_count = lambda: sum(row["status"] == "running" for row in native.live.values())
    runtime.maintain()
    assert len(native.calls) == 2
    resumed = runtime.registry.all()[-1]
    assert resumed.plan["task"]["continue"] == original.hand_id
    assert resumed.plan["origin_context"] == original.plan["origin_context"]
    assert resumed.parent_session_id == world.parent.session_id
    assert world.execute(resumed, "approved-call", arguments, target.unlink) is None
    state = run(current.store.db.execute_fetchall("SELECT state FROM permits WHERE id=?", (permit["id"],)))[0][0]
    assert state == "consumed" and not target.exists() and world.executions == [resumed.hand_id]
    receipt = runtime.effects.get(resumed.hand_id, "approved-call")
    assert receipt["outcome"] == "ok" and receipt["exit_code"] == 0 and receipt["done_at"]
    assert receipt["workdir"] == permit["workdir"]
    runtime.subagent_stop(child_session_id=resumed.child_session_id, child_summary="Fixture removed")
    assert resumed.status == "done"
    assert any(path == f"/v1/hands/{resumed.hand_id}/result" and body["status"] == "done"
               for path, body, _ in world.writes)
    runtime.resume_approvals(world.parent.session_id, answer["granted_hands"])
    assert runtime.continue_hand(original)["status"] == "approval_resume"
    runtime.maintain()
    assert len(native.calls) == 2 and world.executions == [resumed.hand_id]


def held_in_broad_cwd(world, home, task):
    """Gate w18-20261007T182631Z: a held relative rm whose cwd is a broad directory."""
    target, other = home / "report.txt", home / "other.txt"
    target.write_text("approved fixture")
    other.write_text("never approved")
    arguments = {"command": "rm report.txt", "workdir": str(home)}
    world.runtime.dispatch_tasks([task], parent=world.parent)
    original = world.runtime.registry.all()[0]
    assert world.execute(original, "held-call", arguments)["action"] == "block" and target.exists()
    permit = world.permits()[0]
    assert permit["workdir"] == str(home) and permit["state"] == "waiting"
    resumed = world.resume(original, world.approve(permit))
    return resumed, permit, arguments, target, other


def test_a_held_call_in_a_broad_cwd_never_widens_the_continuation_workspace(world, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    workdir = clone(home / "work")
    resumed, permit, arguments, target, other = held_in_broad_cwd(world, home, {
        "goal": "Remove the approved fixture", "kind": "coding", "workdir": str(workdir)})
    assert resumed.plan["workdir"] == resumed.plan["task"]["workdir"] == str(workdir)
    assert world.run(world.current.get_hand(resumed.hand_id))["decisions"]["workdir"] == str(workdir)
    assert world.execute(resumed, "approved-call", arguments, target.unlink) is None
    assert {row["id"]: row for row in world.permits()}[permit["id"]]["state"] == "consumed"
    # A literal deletion inside the original task's own workspace keeps that task's exemption.
    scratch = workdir / "scratch.txt"
    scratch.write_text("hand scratch")
    assert world.execute(resumed, "workspace-delete", {"command": "rm scratch.txt", "workdir": str(workdir)},
                         scratch.unlink) is None
    assert not scratch.exists() and len(world.permits()) == 1
    verdict = world.execute(resumed, "second-delete", {"command": "rm other.txt", "workdir": str(home)}, other.unlink)
    assert verdict["action"] == "block" and other.exists()
    second = world.permits()[-1]
    assert second["hand_id"] == resumed.hand_id and second["state"] == "waiting" and second["workdir"] == str(home)


def test_a_continuation_of_a_task_without_a_workspace_holds_every_other_deletion(world, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    resumed, permit, arguments, target, other = held_in_broad_cwd(world, home, {
        "goal": "Remove the approved fixture", "kind": "ops"})
    assert "workdir" not in resumed.plan["task"] and not resumed.plan.get("workdir")
    assert world.run(world.current.get_hand(resumed.hand_id))["decisions"]["workdir"] is None
    assert world.execute(resumed, "approved-call", arguments, target.unlink) is None
    rows = {row["id"]: row for row in world.permits()}
    assert rows[permit["id"]]["state"] == "consumed" and not target.exists()
    verdict = world.execute(resumed, "second-delete", {"command": "rm other.txt", "workdir": str(home)}, other.unlink)
    assert verdict["action"] == "block" and "held for the owner" in verdict["message"]
    assert other.exists() and world.executions == [resumed.hand_id]
    second = [row for row in world.permits() if row["id"] != permit["id"]]
    assert len(second) == 1 and second[0]["hand_id"] == resumed.hand_id and second[0]["state"] == "waiting"
    assert second[0]["ask_id"] != permit["ask_id"] and second[0]["workdir"] == str(home)
