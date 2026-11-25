"""OC5 plugin acceptance, without a live Hermes or network transport."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType

import pytest

needs_hermes = pytest.mark.skipif(importlib.util.find_spec("hermes_time") is None,
                                  reason="needs stock Hermes in the test interpreter")


@pytest.fixture
def plugin(monkeypatch, tmp_path):
    name = "_oc5_hands"
    root = Path(__file__).resolve().parents[2] / "plugins/colonymind-hands"
    package = ModuleType(name)
    package.__path__ = [str(root)]
    package.__package__ = name
    monkeypatch.setitem(sys.modules, name, package)
    authority = importlib.import_module(name + ".authority")
    budgets = importlib.import_module(name + ".budgets")
    work = tmp_path / "work"
    work.mkdir()
    writes, checks, reads, permits = [], [], [], []
    def request(method, path, **kw):
        checks.append((method, path, kw))
        return {"decision": "hold"}
    def get(path, params=None):
        reads.append((path, params))
        assert path == "/v1/hands/permits/held"
        return {"permits": list(permits)}
    client = SimpleNamespace(_request=request, get=get)
    runtime = SimpleNamespace(registry=budgets.HandRegistry(), config={"floor": {
        "publish_targets": ["https://publish.example.test"], "owner_account_hosts": ["account.example.test"]}},
        client=client, epoch={"owner_pid": 123, "owner_started_at": 100},
        ingress=SimpleNamespace(owner_started_turn=lambda sid: sid == "owner-session"))
    runtime.workdir_for = lambda tool, args, task: str(work)
    runtime.emit = lambda path, body, key: writes.append((path, body, key)) or {"permit_id": "p-1", "ask_id": "a-1"}
    hand = budgets.HandState("h-000000000001", task_id="t-000000000001", kind="coding",
        parent_session_id="owner-session", epoch=runtime.epoch, plan={"workdir": str(work), "origin": "mind"})
    runtime.registry.add(hand)
    runtime.registry.bind(hand, "child-session")
    return SimpleNamespace(authority=authority, runtime=runtime, hand=hand, writes=writes, checks=checks,
                           reads=reads, permits=permits, work=work)


def before(plugin, command, *, session="child-session", **ids):
    return plugin.authority.pre_tool_call(plugin.runtime, tool_name="terminal", args={"command": command},
                                          session_id=session, tool_call_id="call-1", **ids)


def test_recognized_deletion_outside_is_held_inside_runs_and_unknown_commands_run(plugin, tmp_path):
    inside = plugin.work / "temporary.txt"
    inside.write_text("temporary")
    outside = tmp_path / "report.txt"
    outside.write_text("report")
    assert before(plugin, "rm temporary.txt") is None
    assert before(plugin, "python -m pytest") is None
    verdict = before(plugin, f"rm {outside}")
    assert verdict["action"] == "block"
    assert outside.exists() and len(plugin.writes) == 1
    body = plugin.writes[0][1]
    assert body["call"] == {"tool": "terminal", "arguments": {"command": f"rm {outside}"}}
    assert body["targets"] == [{"kind": "file", "target": str(outside),
        "value": hashlib.sha256(outside.read_bytes()).hexdigest()}]
    assert plugin.hand.activity == "awaiting_approval"


def test_workspace_symlink_and_compound_command_do_not_exempt_deletion(plugin, tmp_path):
    outside = tmp_path / "outside"
    outside.write_text("keep")
    (plugin.work / "escape").symlink_to(outside)
    assert before(plugin, "rm escape")["action"] == "block"
    plugin.hand.activity, plugin.hand.blocked = "queued", False
    assert before(plugin, "rm local; rm ../outside")["action"] == "block"


@pytest.mark.parametrize("command", ["curl https://api.stripe.com/v1/transfers", "passwd",
    "bulk_send messages", "curl -X POST https://publish.example.test/upload",
    "curl https://account.example.test/settings"])
def test_floor_classes_and_configured_destinations_wait(plugin, command):
    assert before(plugin, command)["action"] == "block"
    assert len(plugin.writes) == 1


@pytest.mark.parametrize("session", ["child-session", "owner-session"])
def test_url_with_an_unreadable_port_neither_blocks_nor_aborts_recognition(plugin, caplog, session):
    """A port that is no number in range is read as no port: the call is judged like any other, with no warning."""
    for command in ("curl https://docs.example.test:notaport/page", "curl https://docs.example.test:99999/page"):
        assert before(plugin, command, session=session) is None
    assert not plugin.writes and not plugin.checks
    assert not [record for record in caplog.records if record.levelname in {"WARNING", "ERROR"}]


@pytest.mark.parametrize("command, destination", [
    ("curl -X POST https://publish.example.test:notaport/upload", "https://publish.example.test/upload"),
    ("curl https://account.example.test:65536/settings", "https://account.example.test/settings")])
def test_configured_destination_with_an_unreadable_port_is_still_recognized(plugin, caplog, command, destination):
    assert before(plugin, command)["action"] == "block"
    assert len(plugin.writes) == 1
    assert plugin.writes[0][1]["targets"] == [{"kind": "destination", "target": destination, "value": destination}]
    assert not [record for record in caplog.records if record.levelname in {"WARNING", "ERROR"}]


def test_owner_call_with_an_unreadable_port_meets_the_held_target_it_names(plugin, caplog):
    held_permit(plugin, "curl -X POST https://publish.example.test/upload")
    verdict = before(plugin, "curl -X POST https://publish.example.test:notaport/upload", session="owner-session")
    assert_continue(plugin, verdict)
    assert "this target is held" in verdict["message"]
    assert not [record for record in caplog.records if record.levelname in {"WARNING", "ERROR"}]


def test_owner_front_runs_floor_effect_and_review_cannot_borrow_a_hand(plugin):
    command = "curl https://api.stripe.com/v1/transfers"
    assert before(plugin, command, session="owner-session") is None
    plugin.runtime.ingress.owner_started_turn = lambda sid: False
    assert before(plugin, command, session="owner-session", owner_started=True)["action"] == "block"
    assert not plugin.writes and not plugin.checks
    assert before(plugin, "python -m pytest", session="owner-session") is None


def test_targets_recomputed_on_each_call_and_check_sends_idempotency_header(plugin, tmp_path):
    file = tmp_path / "outside"
    file.write_text("original")
    plugin.runtime.client._request = lambda method, path, **kw: plugin.checks.append(kw) or {"decision": "allow"}
    assert before(plugin, f"rm {file}") is None
    first = json.loads(plugin.checks[0]["params"]["targets_now"])
    file.write_text("modified")
    assert before(plugin, f"rm {file}") is None
    second = json.loads(plugin.checks[1]["params"]["targets_now"])
    assert first != second and plugin.checks[0]["key"]


def test_authority_failure_closes_hand_but_not_owner_without_waiting_hand(plugin):
    def broken(*args, **kwargs):
        raise ConnectionError("offline")
    plugin.runtime.client._request = broken
    plugin.runtime.client.get = broken
    assert before(plugin, "passwd")["action"] == "block"
    assert before(plugin, "passwd", session="owner-session") is None
    plugin.runtime.registry.for_session = broken
    assert before(plugin, "passwd", session="owner-session")["action"] == "block"


def test_the_call_digest_matches_the_sidecar(plugin):
    from colonymind.hands import approvals, wire
    call = {"tool": "terminal", "arguments": {"command": "rm report.txt", "timeout": 30}}
    assert plugin.authority.call_digest(call, str(plugin.work)) == approvals.call_digest(
        wire.ExactCall(**call), str(plugin.work))


def test_git_push_matches_configured_remote_and_binds_commit(plugin, monkeypatch):
    plugin.runtime.config["floor"]["publish_targets"] = ["https://publish.example.test/repo"]
    git = {("remote", "get-url", "--push", "origin"): "https://publish.example.test/repo",
           ("rev-parse", "--verify", "--end-of-options", "main^{commit}"): "a" * 40}
    monkeypatch.setattr(plugin.authority, "_git", lambda workdir, *args: git[args])
    assert before(plugin, "git push origin main")["action"] == "block"
    targets = plugin.writes[0][1]["targets"]
    assert any(t["kind"] == "git_ref" and t["value"] == "a" * 40 for t in targets)
    assert any(t["kind"] == "destination" and t["value"] == "https://publish.example.test/repo" for t in targets)


def test_upload_file_and_service_probe_bind_values(plugin, tmp_path):
    file = tmp_path / "report.txt"
    file.write_text("report")
    plugin.hand.plan["floor_targets"] = [{"kind": "service", "target": "service-status"}]
    plugin.runtime.probe_target = lambda kind, target: b"version 1"
    command = f"curl --data-binary @{file} https://publish.example.test/upload"
    assert before(plugin, command)["action"] == "block"
    targets = plugin.writes[0][1]["targets"]
    assert any(t["kind"] == "file" and t["value"] == hashlib.sha256(b"report").hexdigest() for t in targets)
    assert any(t["kind"] == "service" and t["value"] == hashlib.sha256(b"version 1").hexdigest() for t in targets)


@needs_hermes
def test_tripwire_refuses_hand_key_read_and_answer_call_but_owner_still_runs(monkeypatch, tmp_path):
    from colonymind_hermes import guard
    reports = []
    client = SimpleNamespace(has_mind_routes=lambda: True,
                             post=lambda path, **kwargs: reports.append((path, kwargs)))
    sessions = SimpleNamespace(standing=lambda sid: ("owner", "sms"), is_owner=lambda sid: True,
                               standings=SimpleNamespace(get=lambda *args: None),
                               get=lambda sid: SimpleNamespace(sender_id="owner-1"))
    instance = guard.Guard(client, sessions, SimpleNamespace(key_file=tmp_path / "api.key",
        guests=lambda: {"executor_platforms": [], "tools": []}, mind=lambda: {}))
    monkeypatch.setattr(guard, "mind_run", lambda settings: False)
    monkeypatch.setattr(instance, "_cron_origin", lambda *args: "")
    monkeypatch.setattr(guard, "_hand_session_check", lambda sid, task: sid == "child-session")
    monkeypatch.setattr(guard, "_hand_session_standing", lambda sid, task: "owner")
    for tool, args in (("read_file", {"path": str(tmp_path / "api.key")}),
                       ("terminal", {"command": "curl http://localhost/v1/mind/answers"})):
        assert instance.pre_tool_call(tool_name=tool, args=args, session_id="child-session")["action"] == "block"
        assert instance.pre_tool_call(tool_name=tool, args=args, session_id="owner-session") is None
    assert any(row[1]["json"]["run"] == "hand" and row[1]["json"]["refused"] for row in reports)


def test_terminal_cwd_controls_workspace_exemption_not_the_plan(plugin, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "temporary.txt").write_text("keep")
    plugin.runtime.workdir_for = lambda tool, args, task: str(outside)
    assert before(plugin, "rm temporary.txt")["action"] == "block"
    assert plugin.writes[0][1]["workdir"] == str(outside)
    assert plugin.writes[0][1]["targets"][0]["target"] == str(outside / "temporary.txt")


def test_unavailable_workdir_refuses_recognized_effect_without_exempting_it(plugin):
    plugin.runtime.workdir_for = lambda tool, args, task: None
    assert before(plugin, "rm temporary.txt")["action"] == "block"
    assert not plugin.writes
    assert before(plugin, "python -m pytest") is None


def test_missing_ingress_ledger_preserves_owner_exemption_but_known_review_does_not(plugin, monkeypatch):
    command = "curl https://api.stripe.com/v1/transfers"
    runtime = plugin.runtime
    del runtime.ingress
    monkeypatch.delitem(sys.modules, "colonymind_hands_front", raising=False)
    assert before(plugin, command, session="owner-session") is None
    bridge = ModuleType("colonymind_hands_front")
    monkeypatch.setitem(sys.modules, "colonymind_hands_front", bridge)
    for provenance in (True, False, None):
        bridge.owner_started_turn = lambda sid, turn: provenance
        verdict = before(plugin, command, session="owner-session", turn_id="turn-7")
        if provenance is True:
            assert verdict is None
        else:
            assert verdict["action"] == "block"
    def broken(*args):
        raise RuntimeError("ledger unavailable")
    bridge.owner_started_turn = broken
    assert before(plugin, command, session="owner-session")["action"] == "block"
    assert not plugin.writes and not plugin.checks
    assert before(plugin, command)["action"] == "block"


def held_permit(plugin, command, *, state="waiting", parent_session="owner-session", targets=None):
    call = {"tool": "terminal", "arguments": {"command": command}}
    permit = {"hand_id": plugin.hand.hand_id, "parent_session": parent_session, "state": state,
              "call_digest": plugin.authority.call_digest(call, str(plugin.work)),
              "targets": targets if targets is not None else plugin.authority.bound_targets(
                  "terminal", call["arguments"], plugin.work, plugin.hand, plugin.runtime)}
    plugin.permits.append(permit)
    return permit


def assert_continue(plugin, verdict):
    assert verdict["action"] == "block"
    assert "runtime resumes" in verdict["message"]
    assert "automatically" in verdict["message"] and "Do not continue" in verdict["message"]
    assert plugin.hand.hand_id in verdict["message"]
    assert not plugin.writes and not plugin.checks


def test_owner_started_front_deletion_outside_workspace_runs_without_hold(plugin, tmp_path):
    outside = tmp_path / "owner-report.txt"
    outside.write_text("owner requested deletion")
    assert before(plugin, f"rm {outside}", session="owner-session") is None
    assert outside.exists()
    assert not plugin.writes and not plugin.checks


def test_owner_without_held_work_runs_even_when_workdir_reader_fails(plugin):
    def unavailable(*args, **kwargs):
        raise OSError("working directory unavailable")
    plugin.runtime.workdir_for = unavailable
    assert before(plugin, "passwd", session="owner-session") is None
    assert not plugin.writes and not plugin.checks


def test_owner_without_held_work_runs_push_with_unhandled_binding_option(plugin, monkeypatch):
    plugin.runtime.config["floor"]["publish_targets"] = ["https://publish.example.test/repo"]
    monkeypatch.setattr(plugin.authority, "_git", lambda *args: pytest.fail("owner exemption needs no git probe"))
    assert before(plugin, "git push --force-with-lease=main origin HEAD", session="owner-session") is None
    assert not plugin.writes and not plugin.checks


@pytest.mark.parametrize("state", ["waiting", "granted"])
def test_owner_answer_replay_names_the_holding_hand_without_consuming_permit(plugin, tmp_path, state):
    outside = tmp_path / "held-report.txt"
    outside.write_text("held")
    command = f"rm {outside}"
    held_permit(plugin, command, state=state)
    assert_continue(plugin, before(plugin, command, session="owner-session"))


@pytest.mark.parametrize("command", ["rm -f {target}", "unlink {target}"])
@pytest.mark.parametrize("state", ["waiting", "granted"])
def test_owner_cannot_change_command_to_delete_a_held_target(plugin, tmp_path, command, state):
    outside = tmp_path / "held-report.txt"
    outside.write_text("held")
    held_permit(plugin, f"rm {outside}", state=state)
    assert_continue(plugin, before(plugin, command.format(target=outside), session="owner-session"))


@pytest.mark.parametrize("held_target", [True, False])
def test_recognized_move_to_trash_only_blocks_the_held_target(plugin, tmp_path, monkeypatch, held_target):
    outside = tmp_path / "held-report.txt"
    outside.write_text("held")
    held_permit(plugin, f"rm {outside}")
    moved = outside if held_target else tmp_path / "owner-report.txt"
    if not held_target:
        moved.write_text("owner")
    guard = plugin.authority._guard()
    effect_of = guard.effect_of
    def recognize_move(tool, args, workdir):
        if tool == "terminal" and args.get("command", "").startswith("mv "):
            return SimpleNamespace(cls="irreversible_deletion", floor=True)
        return effect_of(tool, args, workdir)
    monkeypatch.setattr(guard, "effect_of", recognize_move)
    verdict = before(plugin, f"mv {moved} {tmp_path / 'trash'}", session="owner-session")
    if held_target:
        assert_continue(plugin, verdict)
    else:
        assert verdict is None
        assert not plugin.writes and not plugin.checks


def test_changed_content_does_not_let_owner_replay_the_same_held_target(plugin, tmp_path):
    outside = tmp_path / "held-report.txt"
    outside.write_text("held")
    held_permit(plugin, f"rm {outside}")
    outside.write_text("new contents")
    assert_continue(plugin, before(plugin, f"unlink {outside}", session="owner-session"))


def test_call_digest_blocks_even_without_shared_target_bindings(plugin):
    command = "passwd"
    held_permit(plugin, command, targets=[])
    assert_continue(plugin, before(plugin, command, session="owner-session"))


def test_missing_ingress_ledger_does_not_let_owner_take_a_held_call(plugin, monkeypatch):
    held_permit(plugin, "passwd", targets=[])
    del plugin.runtime.ingress
    monkeypatch.delitem(sys.modules, "colonymind_hands_front", raising=False)
    assert_continue(plugin, before(plugin, "passwd", session="owner-session"))


def test_owner_can_delete_a_different_target_while_same_session_hand_waits(plugin, tmp_path):
    outside, other = tmp_path / "held-report.txt", tmp_path / "owner-report.txt"
    outside.write_text("held")
    other.write_text("owner")
    held_permit(plugin, f"rm {outside}")
    plugin.hand.activity = "awaiting_approval"
    assert before(plugin, f"rm {other}", session="owner-session") is None
    assert not plugin.writes and not plugin.checks


def test_other_parent_session_held_target_does_not_block_owner(plugin, tmp_path):
    outside = tmp_path / "held-report.txt"
    outside.write_text("held elsewhere")
    held_permit(plugin, f"rm {outside}", parent_session="other-owner-session")
    assert before(plugin, f"rm {outside}", session="owner-session") is None
    assert not plugin.writes and not plugin.checks


def test_review_turn_cannot_borrow_granted_permit_even_with_owner_argument(plugin, tmp_path):
    outside = tmp_path / "held-report.txt"
    outside.write_text("held")
    held_permit(plugin, f"rm {outside}", state="granted")
    plugin.runtime.ingress.owner_started_turn = lambda sid: False
    assert before(plugin, f"rm {outside}", session="owner-session", owner_started=True)["action"] == "block"
    assert not plugin.writes and not plugin.checks


@pytest.mark.parametrize("state", ["waiting", "granted"])
def test_restart_reads_held_targets_from_controller_with_empty_registry(plugin, tmp_path, state):
    outside = tmp_path / "held-report.txt"
    outside.write_text("held before restart")
    held_permit(plugin, f"rm {outside}", state=state)
    plugin.runtime.registry = type(plugin.runtime.registry)()
    assert_continue(plugin, before(plugin, f"unlink {outside}", session="owner-session"))
    assert plugin.reads == [("/v1/hands/permits/held", {"session": "owner-session"})]


@pytest.mark.parametrize("controller_available", [True, False])
def test_restart_without_waiting_hand_does_not_block_owner(plugin, tmp_path, controller_available):
    outside = tmp_path / "owner-report.txt"
    outside.write_text("owner")
    plugin.runtime.registry = type(plugin.runtime.registry)()
    if not controller_available:
        def unavailable(*args, **kwargs):
            raise ConnectionError("controller unavailable")
        plugin.runtime.client.get = unavailable
    assert before(plugin, f"rm {outside}", session="owner-session") is None
    assert not plugin.writes and not plugin.checks


@pytest.mark.parametrize("state", ["waiting", "granted"])
def test_restart_binding_comparison_failure_only_blocks_when_controller_reports_waiting(plugin, state, monkeypatch):
    held_permit(plugin, "passwd", state=state, targets=[])
    plugin.runtime.registry = type(plugin.runtime.registry)()
    comparisons = []
    def unavailable(*args, **kwargs):
        comparisons.append((args, kwargs))
        raise OSError("target bindings unavailable")
    monkeypatch.setattr(plugin.authority, "bound_targets", unavailable)
    verdict = before(plugin, "curl https://api.stripe.com/v1/transfers", session="owner-session")
    assert len(comparisons) == 1
    if state == "waiting":
        assert_continue(plugin, verdict)
        assert "held target comparison unavailable" in verdict["message"]
    else:
        assert verdict is None
    assert not plugin.writes and not plugin.checks


@pytest.mark.parametrize("waiting", [True, False])
def test_unreadable_held_targets_only_close_owner_turn_if_same_session_hand_waits(plugin, waiting):
    def unavailable(*args, **kwargs):
        raise ConnectionError("controller unavailable")
    plugin.runtime.client.get = unavailable
    plugin.hand.activity = "awaiting_approval" if waiting else "queued"
    verdict = before(plugin, "passwd", session="owner-session")
    if waiting:
        assert_continue(plugin, verdict)
        assert "unavailable" in verdict["message"] or "cannot read" in verdict["message"]
    else:
        assert verdict is None
    assert not plugin.writes and not plugin.checks


def test_unreadable_held_targets_from_other_session_do_not_block_owner(plugin):
    def unavailable(*args, **kwargs):
        raise ConnectionError("controller unavailable")
    plugin.runtime.client.get = unavailable
    plugin.hand.activity = "awaiting_approval"
    plugin.hand.parent_session_id = "other-owner-session"
    assert before(plugin, "passwd", session="owner-session") is None
    assert not plugin.writes and not plugin.checks


def test_nonfloor_owner_call_runs_while_held_targets_are_unreadable_and_hand_waits(plugin):
    def unavailable(*args, **kwargs):
        raise ConnectionError("controller unavailable")
    plugin.runtime.client.get = unavailable
    plugin.hand.activity = "awaiting_approval"
    assert before(plugin, "python -m pytest", session="owner-session") is None
    assert not plugin.writes and not plugin.checks


@pytest.mark.parametrize("waiting", [True, False])
def test_runtime_restores_waiting_state_from_disk_before_offline_owner_check(plugin, tmp_path, waiting):
    dispatch = importlib.import_module(plugin.authority.__package__ + ".dispatch")
    spool_module = importlib.import_module(plugin.authority.__package__ + ".spool")
    spool_path = tmp_path / "saved-spool"
    key = "plan:owner-session:turn-1:call-1:0"
    hand_id = "h-" + hashlib.sha256(key.encode()).hexdigest()[:12]
    outside = tmp_path / "held-report.txt"
    outside.write_text("persisted target")
    spool = spool_module.Spool(spool_path)
    spool.append("/v1/hands/plan", {"parent_session": "owner-session", "tasks": [
        {"kind": "coding", "goal": "Remove temporary report"}]}, key)
    spool.append(f"/v1/hands/{hand_id}/dispatched", {
        "epoch": plugin.runtime.epoch, "delegation_id": "previous-delegation"}, "dispatched:1")
    if waiting:
        spool.append(f"/v1/hands/{hand_id}/hold", {
            "epoch": plugin.runtime.epoch, "call": {"tool": "terminal", "arguments": {"command": f"rm {outside}"}},
            "workdir": str(plugin.work), "targets": [{"kind": "file", "target": str(outside),
                "value": hashlib.sha256(outside.read_bytes()).hexdigest()}]}, "hold:1")
    spool.close()

    def unavailable(path, params=None):
        plugin.reads.append((path, params))
        raise ConnectionError("controller unavailable after restart")
    native = SimpleNamespace(delegate_module=SimpleNamespace(delegate_task=lambda **kwargs: None))
    runtime = dispatch.Runtime(config=plugin.runtime.config, client=SimpleNamespace(get=unavailable),
                               spool_dir=spool_path, native=native)
    try:
        runtime.ingress = plugin.runtime.ingress
        runtime.workdir_for = plugin.runtime.workdir_for
        restored = runtime.registry.get(hand_id)
        assert restored is not None and restored.parent_session_id == "owner-session"
        assert restored.activity == ("awaiting_approval" if waiting else "queued")
        persisted = runtime.spool.records()
        plugin.runtime, plugin.hand = runtime, restored
        verdict = before(plugin, f"rm {outside}", session="owner-session")
        if waiting:
            assert_continue(plugin, verdict)
            assert "held targets unavailable" in verdict["message"]
        else:
            assert verdict is None
        assert plugin.reads == [("/v1/hands/permits/held", {"session": "owner-session"})]
        assert runtime.spool.records() == persisted
    finally:
        runtime.close()


@pytest.mark.parametrize("state", ["stopped", "waiting", "refused"])
def test_stopped_waiting_and_refused_hand_calls_never_consume_permit(plugin, state):
    if state == "stopped":
        plugin.hand.status = "stopped"
    elif state == "waiting":
        plugin.hand.activity = "awaiting_approval"
    else:
        plugin.hand.refused_calls.add("call-1")
    plugin.runtime.client._request = lambda *args, **kwargs: pytest.fail("a refused hand must not consume a permit")
    assert before(plugin, "passwd")["action"] == "block"
    assert not plugin.writes


def test_owner_turn_ordinary_calls_never_wait_on_the_controller(plugin, tmp_path):
    # Owner turns stay I/O free for calls that are no floor effect: the held-work read happens only for a
    # recognized one, so a slow controller cannot slow the owner's ordinary tools.
    plugin.permits.append({"hand_id": plugin.hand.hand_id, "parent_session": "owner-session", "state": "waiting",
                           "call_digest": "0" * 64, "targets": []})
    assert before(plugin, "ls -la", session="owner-session") is None
    assert before(plugin, "echo hello > " + str(plugin.work / "note.txt"), session="owner-session") is None
    assert plugin.reads == []
    outside = tmp_path / "old.log"
    outside.write_text("old")
    before(plugin, "rm " + str(outside), session="owner-session")
    assert plugin.reads == [("/v1/hands/permits/held", {"session": "owner-session"})]
