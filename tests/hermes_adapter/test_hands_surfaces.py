"""Public hands surfaces stay bound to trusted sessions and delivery receipts."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "hands_surfaces_under_test"


@pytest.fixture
def modules():
    package = ModuleType(PACKAGE)
    package.__path__ = [str(ROOT / "plugins" / "colonymind-hands")]
    sys.modules[PACKAGE] = package
    result = SimpleNamespace(**{name: importlib.import_module(f"{PACKAGE}.{name}")
                               for name in ("tools", "commands", "notices")})
    yield result
    for name in list(sys.modules):
        if name == PACKAGE or name.startswith(PACKAGE + "."):
            del sys.modules[name]


HAND = "h-0123456789ab"
TASK = "t-0123456789ab"


class Client:
    def __init__(self):
        self.calls = []
        self.rows = [{"id": HAND, "task_id": TASK, "parent_session": "owner",
                      "name": "Scout", "state": "running", "intent_version": 1}]
        self.pending = []

    def get(self, path, params=None):
        self.calls.append((path, params))
        if path == "/v1/hands/":
            return {"hands": self.rows, "cursor": ""}
        if path == "/v1/hands/notices":
            return {"notices": self.pending, "cursor": ""}
        if path == "/v1/hands/digest":
            return {"items": [{"hand_id": HAND, "question_id": "q-1",
                               "obligation_id": "o-1"}], "cursor": "", "text": ""}
        if path == f"/v1/hands/tasks/{TASK}":
            return {"id": TASK, "obligation_id": "o-1", "intent_version": 1}
        if path == f"/v1/hands/{HAND}":
            return dict(self.rows[0])
        raise AssertionError(path)


@pytest.fixture
def runtime():
    hand = SimpleNamespace(hand_id=HAND, task_id=TASK, parent_session_id="owner",
                           child_session_id="child", plan={"name": "Scout"},
                           status="running", activity="generating", blocked=False,
                           seq=0, epoch={"owner_pid": 1, "owner_started_at": 1.0})
    registry = SimpleNamespace(for_session=lambda session: hand if session == "child" else None,
                               get=lambda key: hand if key == HAND else None,
                               all=lambda: [hand])
    calls = []
    def emit(path, body, key):
        calls.append((path, body, key))
        if path.endswith("/question"):
            return {"id": "q-1", "hand_id": HAND, "blocking": body["blocking"]}
        if "/questions/" in path:
            return {"id": "q-1", "hand_id": HAND, "blocking": hand.blocked}
        return {"ok": True}
    return SimpleNamespace(registry=registry, hand=hand, client=Client(), writes=calls,
                           actions=[], emit=emit, config={}, epoch=hand.epoch,
                           authorized_owner=lambda sid: sid == "owner",
                           stop=lambda hand, reason, **kw: calls.append(("stop", hand.hand_id, reason)),
                           steer=lambda hand, text, **kw: calls.append(("steer", hand.hand_id, text)),
                           continue_hand=lambda hand, **kw: calls.append(("continue", hand.hand_id, kw)))


def test_owner_and_child_authority_is_not_taken_from_tool_arguments(modules, runtime):
    for session in ("guest", "child", ""):
        answer = modules.tools.execute_hands({"action": "stop", "hand": HAND,
                                              "session_id": "owner"}, runtime=runtime,
                                             session_id=session)
        assert answer["error"]
    assert not runtime.writes
    assert not runtime.client.calls


def test_cross_session_hand_refused_before_registry_control(modules, runtime):
    runtime.client.rows[0]["parent_session"] = "different-owner-session"
    answer = modules.tools.execute_hands({"action": "stop", "hand": "Scout"},
                                         runtime=runtime, session_id="owner")
    assert answer["error"]
    assert not runtime.writes


@pytest.mark.parametrize("action", ["list", "show", "stop", "steer", "name",
                                    "revise", "told", "accept", "continue", "answer"])
def test_owner_actions_use_frozen_routes(modules, runtime, action):
    arguments = {"action": action, "hand": "Scout", "text": "Use the current scope",
                 "name": "Scout-next", "goal": "Revised goal", "reason": "Reviewed evidence"}
    result = modules.tools.execute_hands(arguments, runtime=runtime, session_id="owner", turn_id="turn-1")
    assert "error" not in result, result
    if action == "told":
        assert runtime.writes[-1][0] == "/v1/hands/obligations/o-1/told"
        assert runtime.writes[-1][1] == {"turn_ref": "turn-1"}
    if action == "accept":
        assert runtime.writes[-1][1] == {"hand": HAND, "reason": "Reviewed evidence", "for_version": 1}
    if action == "answer":
        assert runtime.writes[0][0] == "/v1/hands/questions/q-1/answer"
        assert runtime.writes[0][1]["answered_by"] == "owner"
        assert runtime.writes[1][:2] == ("steer", HAND)


def test_continuation_of_unknown_effect_requires_reconciliation(modules, runtime):
    runtime.client.rows[0]["state"] = "unknown_effect"
    runtime.hand.status = "unknown_effect"
    answer = modules.tools.execute_hands({"action": "continue", "hand": HAND},
                                         runtime=runtime, session_id="owner")
    assert answer["error"]
    assert not runtime.writes
    answer = modules.tools.execute_hands({"action": "continue", "hand": HAND, "reconciled": True},
                                         runtime=runtime, session_id="owner")
    assert "error" not in answer


def test_ask_is_hand_only_and_blocking_is_immediate(modules, runtime):
    assert modules.tools.execute_ask({"question": "Which format?"}, runtime=runtime,
                                     session_id="owner")["error"]
    response = modules.tools.execute_ask({"question": "Which format?", "assumption": "JSON"},
                                         runtime=runtime, session_id="child")
    assert response["blocking"] is False
    assert runtime.hand.blocked is False
    response = modules.tools.execute_ask({"question": "Which format?", "blocking": True},
                                         runtime=runtime, session_id="child")
    assert response["blocking"] is True
    assert runtime.hand.blocked is True
    assert runtime.hand.activity == "awaiting_input"
    assert runtime.writes[-1][1]["epoch"] == runtime.epoch


def test_blocking_question_stays_blocked_on_controller_failure(modules, runtime):
    def failing(*args):
        raise OSError("unavailable")
    runtime.emit = failing
    result = modules.tools.execute_ask({"question": "Which format?", "blocking": True},
                                       runtime=runtime, session_id="child")
    assert result["error"]
    assert runtime.hand.blocked is True


def test_blocking_answer_queues_continuation(modules, runtime):
    runtime.hand.blocked = True
    runtime.hand.status = "awaiting_input"
    result = modules.tools.execute_hands({"action": "answer", "hand": HAND, "text": "CSV"},
                                         runtime=runtime, session_id="owner")
    assert "error" not in result
    assert runtime.writes[-1] == ("continue", HAND, {"answer": "CSV", "reconciled": False})


def test_invalid_name_never_reaches_controller(modules, runtime):
    for name in ("bad name", "a" * 25, "hands", "colonymind_hands"):
        result = modules.tools.execute_hands({"action": "name", "hand": HAND, "name": name},
                                             runtime=runtime, session_id="owner")
        assert result["error"]
    assert not runtime.writes


@pytest.mark.parametrize("action", ["list", "show", "steer", "stop"])
def test_controller_outage_keeps_local_owner_controls(modules, runtime, action):
    def unavailable(*args, **kwargs):
        raise ConnectionError("controller down")
    runtime.client.get = unavailable
    result = modules.tools.execute_hands({"action": action, "hand": "Scout", "text": "Pause that work"},
                                         runtime=runtime, session_id="owner")
    assert "error" not in result, result
    if action == "list":
        assert result["hands"][0]["id"] == HAND
    if action in {"steer", "stop"}:
        assert runtime.writes[0][:2] == (action, HAND)


@pytest.mark.parametrize("action", ["name", "revise", "told", "accept", "continue", "answer"])
def test_controller_outage_refuses_mutations_requiring_controller(modules, runtime, action):
    def unavailable(*args, **kwargs):
        raise ConnectionError("controller down")
    runtime.client.get = unavailable
    result = modules.tools.execute_hands({"action": action, "hand": "Scout", "text": "scope", "name": "Scout2",
                                          "goal": "goal", "reason": "Reviewed evidence"},
                                         runtime=runtime, session_id="owner", turn_id="turn-1")
    assert result["error"]
    assert not runtime.writes


def test_outage_local_rows_cannot_control_other_owner_session(modules, runtime):
    runtime.client.get = lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionError())
    runtime.hand.parent_session_id = "owner-other"
    result = modules.tools.execute_hands({"action": "stop", "hand": HAND}, runtime=runtime, session_id="owner")
    assert result["error"]
    assert not runtime.writes


def test_accept_supports_explicit_current_intent_version(modules, runtime):
    result = modules.tools.execute_hands({"action": "accept", "hand": HAND, "reason": "Checked the new criteria",
                                          "for_version": 2}, runtime=runtime, session_id="owner")
    assert "error" not in result
    assert runtime.writes[0][1]["for_version"] == 2


@pytest.mark.parametrize("version", [0, -1, True, "2"])
def test_accept_requires_positive_integer_version(modules, runtime, version):
    result = modules.tools.execute_hands({"action": "accept", "hand": HAND, "reason": "Checked", "for_version": version},
                                         runtime=runtime, session_id="owner")
    assert result["error"]
    assert not runtime.writes


def test_command_stops_only_callers_hands_and_explains_between_turns(modules, runtime):
    runtime.client.rows.append({"id": "h-abcdef012345", "name": "Other", "state": "running",
                                "task_id": TASK, "parent_session": "someone-else"})
    text = modules.commands.run("stop all", runtime=runtime, session_id="owner")
    assert "between turns" in modules.commands.USAGE.lower()
    assert "error" not in text.lower()
    assert [call[1] for call in runtime.writes if call[0] == "stop"] == [HAND]


def notice(number=1, budget="routine", at="1970-01-01T00:00:00+00:00"):
    return {"id": f"ntc:{HAND}:{number}", "budget": budget, "line": f"Scout update {number}.",
            "queued_at": at, "hand_id": HAND, "attempt": 1}


def emitter(modules, runtime, tmp_path, sink, clock):
    return modules.notices.NoticeEmitter(client=runtime.client, emit=runtime.emit, sink=sink,
                                        ledger_path=tmp_path / "notices.sqlite3", now=lambda: clock[0])


def test_notice_merge_receipt_and_restart_retry_are_idempotent(modules, runtime, tmp_path):
    delivered = []
    clock = [10.0]
    def sink(**kwargs):
        delivered.append(kwargs)
        return {"platform_message_id": "platform-1"}
    runtime.client.pending = [notice(), notice(2)]
    first = emitter(modules, runtime, tmp_path, sink, clock)
    first.pump()
    assert not delivered
    clock[0] = 71
    first.pump()
    assert len(delivered) == 1
    assert len(delivered[0]["notices"]) == 2
    assert [entry[1] for entry in runtime.writes] == [{"platform_message_id": "platform-1"}] * 2
    second = emitter(modules, runtime, tmp_path, sink, clock)
    second.pump()
    assert len(delivered) == 1


@pytest.mark.parametrize("receipt", [None, {}, {"success": True}, {"platform_message_id": ""}])
def test_notice_requires_actual_platform_receipt(modules, runtime, tmp_path, receipt):
    runtime.client.pending = [notice()]
    out = emitter(modules, runtime, tmp_path, lambda **kwargs: receipt, [71.0])
    out.pump()
    assert not runtime.writes


def test_notice_budget_classes_reset_at_local_midnight(modules, runtime, tmp_path):
    delivered = []
    clock = [71.0]
    def sink(**kwargs):
        delivered.append(kwargs)
        return {"platform_message_id": str(len(delivered))}
    out = emitter(modules, runtime, tmp_path, sink, clock)
    for index in range(31):
        runtime.client.pending = [notice(index)]
        out.pump()
        clock[0] += 61
    assert len(delivered) == 30
    runtime.client.pending = [notice(100, "actionable")]
    out.pump()
    assert len(delivered) == 31
    clock[0] += 86400
    runtime.client.pending = [notice(101)]
    out.pump()
    assert len(delivered) == 32


def test_notice_composer_refuses_dashes_and_oversize(modules, runtime, tmp_path):
    delivered = []
    out = modules.notices.NoticeEmitter(
        client=runtime.client, emit=runtime.emit, sink=lambda **kw: delivered.append(kw),
        ledger_path=tmp_path / "notices.sqlite3", now=lambda: 71.0,
        settings=lambda: {"forbidden_patterns": [r"[\u2013\u2014]"]},
    )
    for line in ("bad\u2014line", "bad\u2013line", "x" * 201):
        runtime.client.pending = [{**notice(), "line": line}]
        out.pump()
    assert not delivered


def test_notice_uncertain_sink_reuses_delivery_id_after_restart(modules, runtime, tmp_path):
    sent = []
    runtime.client.pending = [notice()]
    def uncertain(**kwargs):
        sent.append(kwargs)
        raise OSError("response lost after platform accepted")
    first = emitter(modules, runtime, tmp_path, uncertain, [71.0])
    first.pump()
    assert len(sent) == 1
    assert not runtime.writes
    def receipt(**kwargs):
        sent.append(kwargs)
        return {"platform_message_id": "original-platform-id"}
    second = emitter(modules, runtime, tmp_path, receipt, [80.0])
    second.pump()
    assert sent[0]["delivery_id"] == sent[1]["delivery_id"]
    assert runtime.writes[0][1] == {"platform_message_id": "original-platform-id"}


def test_actionable_limit_and_hourly_overflow_are_separate(modules, runtime, tmp_path):
    sent, clock = [], [71.0]
    def sink(**kwargs):
        sent.append(kwargs)
        return {"platform_message_id": str(len(sent))}
    out = emitter(modules, runtime, tmp_path, sink, clock)
    for number in range(22):
        runtime.client.pending = [notice(number, "actionable")]
        out.pump()
        clock[0] += 61
    assert len(sent) == 21
    assert sent[-1]["notices"][0]["line"] == "1 more events; /hands"
    clock[0] += 3600
    out.pump()
    assert len(sent) == 22
    assert sent[-1]["notices"][0]["line"] == "2 more events; /hands"


def test_retry_after_midnight_counts_against_new_days_budget(modules, runtime, tmp_path):
    sent, clock, ready = [], [71.0], [False]
    def sink(**kwargs):
        sent.append(kwargs)
        return {"platform_message_id": str(len(sent))} if ready[0] else None
    out = emitter(modules, runtime, tmp_path, sink, clock)
    runtime.client.pending = [notice()]
    out.pump()
    assert len(sent) == 1
    ready[0] = True
    clock[0] += 86400
    out.pump()
    for number in range(2, 32):
        clock[0] += 61
        runtime.client.pending = [notice(number)]
        out.pump()
    assert len(sent) == 31  # One uncertain send yesterday and thirty reservations today.


@pytest.fixture(params=["sms", "rcs"])
def integrated(tmp_path, request):
    from hands_fake_hermes import FakeContext, FakeController, FakeHermes, Parent, load_hands
    package = load_hands()
    dispatch = importlib.import_module(package.__name__ + ".dispatch")
    native, client, ctx = FakeHermes(platform=request.param), FakeController(), FakeContext()
    config = {"max_running": 1, "platforms": [request.param], "contracts": {"text-route": {
        "version": "1", "family": "text", "modalities": ["text"], "context_tokens": 100000,
        "max_output_tokens": 2000, "tool_calls": True, "decode_tok_s": 100}}}
    runtime = dispatch.Runtime(config=config, client=client, spool_dir=tmp_path / "spool",
                               native=native, owner_check=lambda sid: sid == "owner-session")
    native.runtime = runtime
    ctx.hands_runtime = runtime
    runtime.sessions = SimpleNamespace(sender_is_owner=lambda platform, user: user == "owner-id",
                                       inherit=lambda *args, **kwargs: None)
    package.register_route_resolver(lambda role, effort: native.native_policy())
    package.register(ctx)
    parent = Parent()
    native.delegate_module.delegate_task(goal="Inspect sources", parent_agent=parent)
    yield package, runtime, native, client, ctx, parent
    runtime.close()


def test_registered_ask_blocks_real_guard_after_question(integrated):
    _, runtime, _, client, ctx, _ = integrated
    hand, = runtime.registry.all()
    ask = ctx.tools["colonymind_ask"]["handler"]
    response = json.loads(ask({"question": "Which scope?", "blocking": True},
                              session_id=hand.child_session_id))
    assert response["blocking"] is True
    replies = ctx.fire("pre_tool_call", tool_name="web_search", args={"query": "next"},
                       session_id=hand.child_session_id)
    assert any(isinstance(reply, dict) and reply.get("action") == "block" for reply in replies), replies
    assert client.writes[-1][0].endswith("/question")


def test_registered_command_only_stops_on_idle_dispatch(integrated):
    _, runtime, native, client, ctx, _ = integrated
    hand, = runtime.registry.all()
    row = {"id": hand.hand_id, "task_id": hand.task_id, "name": hand.plan["name"],
           "state": "running", "parent_session": "owner-session", "intent_version": 1}
    client.get = lambda path, params=None: {"hands": [row]} if path == "/v1/hands/" else row
    gateway = SimpleNamespace(_session_key_for_source=lambda source: "source-key",
                              session_store=SimpleNamespace(peek_session_id=lambda key: "owner-session"))
    event = SimpleNamespace(text="/hands stop all", message_id="message-2",
                            source=SimpleNamespace(platform=native.platform, user_id="owner-id"))
    assert not any(reply and reply.get("action") == "skip"
                   for reply in ctx.fire("pre_gateway_dispatch", event=event, gateway=gateway))
    assert not native.stops
    response = ctx.commands["hands"]("stop all")
    assert "error" not in response.lower()
    assert len(native.stops) == 1
    assert "require" in ctx.commands["hands"]("stop all").lower()


def test_registered_gateway_command_rejects_nonowner_source(integrated):
    _, runtime, native, _, ctx, _ = integrated
    event = SimpleNamespace(text="/hands stop all",
                            source=SimpleNamespace(platform=native.platform, user_id="guest-id"))
    gateway = SimpleNamespace(_session_key_for_source=lambda source: "source-key",
                              session_store=SimpleNamespace(peek_session_id=lambda key: "owner-session"))
    replies = ctx.fire("pre_gateway_dispatch", event=event, gateway=gateway)
    assert any(reply and reply.get("action") == "skip" for reply in replies)
    assert not native.stops


def test_generic_handler_uses_pinned_hermes_observability_ids(modules, runtime, monkeypatch):
    from contextvars import ContextVar
    from hands_fake_hermes import FakeContext
    context = ModuleType("tools.approval_context")
    for name, value in (("_approval_session_id", "owner"), ("_approval_turn_id", "actual-current-turn"),
                        ("_approval_tool_call_id", "actual-call")):
        variable = ContextVar(name, default=value)
        setattr(context, name, variable)
    monkeypatch.setitem(sys.modules, "tools.approval_context", context)
    monkeypatch.setitem(sys.modules, PACKAGE + ".dispatch", SimpleNamespace(get_runtime=lambda ctx: runtime))
    ctx = FakeContext()
    modules.tools.register(ctx)
    handler = ctx.tools["colonymind_hands"]["handler"]
    arguments = {"action": "told", "hand": HAND}
    for _ in range(2):
        result = json.loads(handler(arguments, session_id="owner", task_id="owner"))
        assert "error" not in result
    assert runtime.writes[0][1] == {"turn_ref": "actual-current-turn"}
    assert runtime.writes[0][2] == runtime.writes[1][2]


def test_observability_ids_from_other_session_are_not_used(modules, runtime, monkeypatch):
    from contextvars import ContextVar
    from hands_fake_hermes import FakeContext
    context = ModuleType("tools.approval_context")
    for name, value in (("_approval_session_id", "other"), ("_approval_turn_id", "other-turn"),
                        ("_approval_tool_call_id", "other-call")):
        setattr(context, name, ContextVar(name, default=value))
    monkeypatch.setitem(sys.modules, "tools.approval_context", context)
    monkeypatch.setitem(sys.modules, PACKAGE + ".dispatch", SimpleNamespace(get_runtime=lambda ctx: runtime))
    ctx = FakeContext()
    modules.tools.register(ctx)
    result = json.loads(ctx.tools["colonymind_hands"]["handler"]({"action": "told", "hand": HAND}, session_id="owner"))
    assert result["error"]
    assert not runtime.writes


def test_journal_failure_notice_needs_no_runtime_and_sends_once(modules):
    calls = []
    def sink(**kwargs):
        calls.append(kwargs)
        return {"platform_message_id": "journal-failure-message"}
    modules.notices.register_notice_sink(sink)
    modules.notices.journal_failure()
    modules.notices.journal_failure()
    assert len(calls) == 1
    row, = calls[0]["notices"]
    assert row["budget"] == "actionable"
    assert "journal" in row["line"].lower()
    assert calls[0]["delivery_id"]


def test_journal_failure_retries_same_id_until_receipt_then_stops(modules, runtime):
    calls, receipts = [], [None, {"platform_message_id": "message"}]
    def sink(**kwargs):
        calls.append(kwargs)
        return receipts.pop(0)
    runtime.notice_sink = sink
    modules.notices.journal_failure(runtime)
    modules.notices.journal_failure(runtime)
    modules.notices.journal_failure(runtime)
    assert len(calls) == 2
    assert calls[0]["delivery_id"] == calls[1]["delivery_id"]
    assert not runtime.writes


def test_journal_failure_without_sink_does_not_choose_transport(modules):
    assert modules.notices.journal_failure() == {"disabled": True}


def test_journal_failure_uncertain_retries_are_bounded(modules):
    calls = []
    modules.notices.register_notice_sink(lambda **kwargs: calls.append(kwargs))
    for _ in range(21):
        modules.notices.journal_failure()
    assert len(calls) == 20
