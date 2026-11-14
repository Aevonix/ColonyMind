"""In-process hand policy using the pinned Hermes hook and middleware payloads."""
from copy import deepcopy
import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def guards(monkeypatch):
    name = "_hands_guards_tests"
    path = Path(__file__).resolve().parents[2] / "plugins/colonymind-hands"
    package = SimpleNamespace(__path__=[str(path)], __package__=name)
    monkeypatch.setitem(sys.modules, name, package)
    modules = []
    for suffix in ("budgets", "allow", "middleware"):
        fullname = f"{name}.{suffix}"
        spec = importlib.util.spec_from_file_location(fullname, path / f"{suffix}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, fullname, module)
        spec.loader.exec_module(module)
        modules.append(module)
    budgets, allow, middleware = modules
    runtime = SimpleNamespace(registry=budgets.HandRegistry(), config={}, events=[], stops=[], steers=[])
    runtime.emit = lambda path, body, key: runtime.events.append((path, body, key))
    # Budgets, clocks and accounting act through the runtime's own system path. The owner-session
    # stop and steer are for sessions that ask; a hand guard must never need one.
    runtime.stop = lambda *args, **kwargs: pytest.fail("a hand guard used the owner-session stop")
    runtime.steer = lambda *args, **kwargs: pytest.fail("a hand guard used the owner-session steer")
    runtime.interrupts = []

    def system_stop(hand, reason, interrupt=True):
        runtime.stops.append((hand.hand_id, reason))
        if interrupt:
            runtime.interrupts.append((hand.hand_id, reason))

    runtime.system_stop = system_stop
    runtime.system_steer = lambda hand, text: runtime.steers.append((hand.hand_id, text))
    monkeypatch.setitem(sys.modules, f"{name}.dispatch", SimpleNamespace(get_runtime=lambda ctx=None: runtime))
    return SimpleNamespace(budgets=budgets, allow=allow, middleware=middleware, runtime=runtime)


def hand(guards, kind="research", **plan):
    state = guards.budgets.HandState("h-one", task_id="t-one", kind=kind, plan=plan)
    guards.runtime.registry.add(state)
    guards.runtime.registry.bind(state, "child-one", subagent_id="sa-one")
    return state


def before(guards, tool, args=None, **ids):
    return guards.budgets.pre_tool_call(guards.runtime, tool_name=tool, args=args or {},
                                        session_id=ids.pop("session_id", "child-one"), **ids)


def after(guards, result, tool="web_search", **ids):
    return guards.budgets.post_tool_call(guards.runtime, tool_name=tool, result=result,
                                         session_id=ids.pop("session_id", "child-one"), **ids)


def allowed(guards, tool, args=None, **ids):
    return guards.allow.pre_tool_call(guards.runtime, tool_name=tool, args=args or {},
                                      session_id=ids.pop("session_id", "child-one"), **ids)


def test_thirteenth_search_refused_and_rotation_keeps_counters(guards):
    state = hand(guards)
    for index in range(12):
        assert before(guards, "web_search", {"query": str(index)}) is None
    assert before(guards, "web_search", {"query": "13"}, session_id="rotated", task_id="sa-one")["action"] == "block"
    assert guards.runtime.registry.for_session("rotated") is state
    assert state.searches == 12
    assert "search budget spent" in before(guards, "web_search", {"query": "14"})["message"]


def test_repeated_arguments_refuse_third_call_even_key_order_changes(guards):
    hand(guards)
    assert before(guards, "read_file", {"path": "x", "limit": 10}) is None
    assert before(guards, "read_file", {"limit": 10, "path": "x"}) is None
    assert before(guards, "read_file", {"path": "x", "limit": 10})["action"] == "block"


def test_no_new_information_warns_at_three_and_refuses_after_five(guards):
    state = hand(guards)
    result = {"results": [{"url": "https://example.test/a", "content": "same evidence"}]}
    after(guards, result)
    for index in range(5):
        after(guards, {**result, "query": str(index), "duration": index})
    assert state.no_new_information == 5
    assert guards.runtime.steers == [(state.hand_id, "No new information in three searches: write up what you have.")]
    assert before(guards, "web_extract", {"url": "https://example.test/a"})["action"] == "block"
    after(guards, {"url": "https://example.test/a", "content": "changed evidence"})
    assert state.no_new_information == 0


def test_awaiting_input_blocks_all_tools_and_freezes_clock(guards):
    state = hand(guards)
    state.activity = "awaiting_input"
    for tool in ("read_file", "web_search", "colonymind_ask", "terminal"):
        assert before(guards, tool)["action"] == "block"
    guards.budgets.tick(guards.runtime, now=state.started_at + 100000)
    assert guards.runtime.stops == []


def test_queued_never_stale_and_generating_is_steered_then_stopped(guards):
    state = hand(guards)
    start = state.started_at
    guards.budgets.tick(guards.runtime, now=start + 100)
    assert not guards.runtime.stops and not guards.runtime.steers
    state.activity = "generating"
    state.activity_since = start
    state.expected_seconds = 10
    guards.budgets.tick(guards.runtime, now=start + 21)
    assert guards.runtime.steers == [(state.hand_id, "status?")]
    guards.budgets.tick(guards.runtime, now=start + 22)
    assert not guards.runtime.stops
    guards.budgets.tick(guards.runtime, now=start + 42)
    assert guards.runtime.stops == [(state.hand_id, "stale")]


def test_wall_warning_hard_stop_and_effort_three(guards):
    state = hand(guards, effort=3)
    # An integral start makes the exact boundary subtraction representable.
    start = state.started_at = float(int(state.started_at))
    guards.budgets.tick(guards.runtime, now=start + 1919)
    assert not guards.runtime.steers and not guards.runtime.stops
    guards.budgets.tick(guards.runtime, now=start + 1920)
    assert guards.runtime.steers == [(state.hand_id, "wrap up and report")]
    guards.budgets.tick(guards.runtime, now=start + 2399)
    assert not guards.runtime.stops
    assert guards.runtime.steers == [(state.hand_id, "wrap up and report")]
    guards.budgets.tick(guards.runtime, now=start + 2400)
    guards.budgets.tick(guards.runtime, now=start + 2401)
    assert guards.runtime.stops == [(state.hand_id, "wall_budget")]


@pytest.mark.parametrize("tool", ["terminal", "execute_code", "browser_cdp", "write_file", "edit_file", "mcp_arbitrary", "delegate_task"])
def test_research_only_allows_listed_tools(guards, tool):
    hand(guards)
    assert allowed(guards, tool)["action"] == "block"
    for read in ("web_search", "web_extract", "read_file", "search_files", "colonymind_memory_search", "colonymind_ask"):
        assert allowed(guards, read) is None


@pytest.mark.parametrize("tool,args", [("colonymind_memory_forget", {}), ("send_message", {}),
    ("colonymind_secret", {}), ("colonymind_people", {}), ("colonymind_self", {"action": "answer"}),
    ("colonymind_hands", {"action": "accept"}), ("clarify", {}), ("cronjob", {}), ("memory", {"action": "add"}),
    ("computer_use", {"action": "capture"})])
def test_named_authority_tools_refused_for_shell_hands_but_owner_unchanged(guards, tool, args):
    hand(guards, "coding", origin="mind")
    assert allowed(guards, tool, args)["action"] == "block"
    assert allowed(guards, tool, args, session_id="owner") is None
    assert allowed(guards, "terminal", {"command": "python -m pytest"}) is None


def test_allow_failure_blocks_hand_without_leaking_exception(guards, monkeypatch):
    hand(guards)
    monkeypatch.setattr(guards.allow, "check_tool", lambda *a, **kw: (_ for _ in ()).throw(ValueError("bad policy")))
    assert allowed(guards, "read_file")["action"] == "block"
    assert allowed(guards, "read_file", session_id="owner") is None


def test_media_writes_stay_in_workspace_including_symlinks(guards, tmp_path):
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "escape").symlink_to(tmp_path, target_is_directory=True)
    hand(guards, "media", cwd=str(workspace))
    assert allowed(guards, "write_file", {"path": "image.txt"}) is None
    for path in ("../outside", str(tmp_path / "outside"), "escape/outside"):
        assert allowed(guards, "write_file", {"path": path})["action"] == "block"


def test_audio_contract_uses_actual_provider_after_fallback(guards):
    state = hand(guards)
    guards.runtime.config = {"contracts": {
        "voice": {"modalities": ["text", "audio"], "family": "voice", "version": "v1"},
        "text": {"modalities": ["text"], "family": "text", "version": "v2"}}}
    request = {"messages": [{"role": "user", "content": [{"type": "text", "text": "transcript"},
        {"type": "input_audio", "input_audio": {"data": "bytes", "format": "wav"}}]}]}
    original = deepcopy(request)
    for provider, keep in (("voice", True), ("text", False), ("unknown", False)):
        result = guards.middleware.llm_request(guards.runtime, request=request, session_id="child-one", provider=provider)
        parts = result["request"]["messages"][0]["content"]
        assert any(part.get("type") == "input_audio" for part in parts) is keep
        if not keep:
            assert any("audio omitted" in part.get("text", "") for part in parts)
    assert request == original
    assert state.activity == "generating"


def test_identity_replaces_embedded_anchor_once_and_absent_stays_unchanged(guards, caplog):
    state = hand(guards, identity="You are the same assistant, working on a separate piece.")
    anchor = guards.middleware.IDENTITY_ANCHOR
    request = {"messages": [{"role": "system", "content": "Other instructions\n" + anchor + "\n" + anchor}]}
    result = guards.middleware.llm_request(guards.runtime, request=request, session_id="child-one")
    content = result["request"]["messages"][0]["content"]
    assert content.count(state.plan["identity"]) == 1 and content.count(anchor) == 1
    missing = {"messages": [{"role": "system", "content": "No identity anchor here."}]}
    for _ in range(2):
        assert guards.middleware.llm_request(guards.runtime, request=missing, session_id="child-one")["request"] == missing
    assert sum("identity anchor" in record.message for record in caplog.records) == 1


@pytest.mark.parametrize("settings,expected", [
    ({}, "Return everything to your main conversation. Do not contact anyone from here. "
         "Be truthful and precise. Give receipts and uncertainty."),
    ({"persona_addendum": "Host identity: {name} <name>.\n  Keep this spacing.  ",
      "style": "  Prefer concise answers — with café examples.  "},
     "Host identity: Ledger Ledger.\n  Keep this spacing.   "
     "Return everything to your main conversation. Do not contact anyone from here. "
     "Be truthful and precise. Give receipts and uncertainty.   "
     "Prefer concise answers — with café examples.  "),
])
def test_outbound_request_renders_host_settings_or_neutral_default(guards, settings, expected):
    hand(guards, name="Ledger")
    guards.runtime.config = {"hands": settings}
    anchor = guards.middleware.IDENTITY_ANCHOR
    request = {"messages": [{"role": "system", "content": "Before\n" + anchor + "\nAfter"}]}
    result = guards.middleware.llm_request(guards.runtime, request=request, session_id="child-one")
    assert result["request"]["messages"][0]["content"] == "Before\n" + expected + "\nAfter"
    assert request["messages"][0]["content"] == "Before\n" + anchor + "\nAfter"


def test_usage_and_actual_family_accounted_once_per_request(guards):
    state = hand(guards)
    guards.runtime.config = {"contracts": {"primary": {"family": "a", "version": "1"},
        "fallback": {"family": "b", "version": "2"}}, "budgets": {"attempt": {"research": {"tokens": 30}}}}
    for provider, request_id, usage in (("primary", "r1", {"prompt_tokens": 10, "completion_tokens": 5}),
        ("primary", "r1", {"prompt_tokens": 10, "completion_tokens": 5}),
        ("fallback", "r2", {"input_tokens": 10, "output_tokens": 5})):
        guards.middleware.post_api_request(guards.runtime, session_id="child-one", provider=provider,
                                           api_request_id=request_id, usage=usage)
    assert state.tokens == 30
    assert state.families_used == {"a", "b"}
    assert state.contract_versions == {("primary", "1"), ("fallback", "2")}
    assert guards.runtime.stops == [(state.hand_id, "token_budget")]


def test_guard_registration_uses_actual_hermes_payloads(guards):
    callbacks = {}
    ctx = SimpleNamespace(register_hook=lambda name, fn: callbacks.setdefault(name, []).append(fn),
        register_middleware=lambda name, fn: callbacks.setdefault(name, []).append(fn))
    for module in (guards.budgets, guards.allow, guards.middleware):
        module.register(ctx)
    hand(guards)
    result = callbacks["llm_request"][0](request={"messages": []}, session_id="child-one", task_id="sa-one",
        platform="subagent", model="small", provider="text", base_url="http://local", api_request_id="r1")
    assert "request" in result
    assert any(fn(tool_name="send_message", args={}, session_id="child-one", task_id="sa-one")
               for fn in callbacks["pre_tool_call"])


def test_controller_budget_is_final_and_not_multiplied_twice(guards):
    state = hand(guards, effort=3, budgets={"wall_seconds": 50, "tokens": 20, "tool_calls": 1, "web_searches": 1})
    assert guards.budgets.limits(guards.runtime, state) == state.plan["budgets"]
    assert before(guards, "read_file", {"path": "a"}) is None
    assert before(guards, "read_file", {"path": "b"})["action"] == "block"


def test_unknown_owner_route_strips_audio_without_touching_identity(guards):
    request = {"messages": [{"role": "system", "content": guards.middleware.IDENTITY_ANCHOR},
        {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "private"}}]}]}
    result = guards.middleware.llm_request(guards.runtime, request=request, session_id="owner", provider="unknown")
    assert result["request"]["messages"][0]["content"] == guards.middleware.IDENTITY_ANCHOR
    assert result["request"]["messages"][1]["content"][0]["type"] == "text"


def test_family_cap_queues_and_cancelled_request_never_reaches_provider(guards):
    import threading
    import time
    first = hand(guards)
    second = guards.budgets.HandState("h-two", kind="research", child_session_id="child-two")
    guards.runtime.registry.add(second)
    guards.runtime.config = {"contracts": {"local": {"family": "one", "modalities": ["text"], "version": "1"}},
                             "health": {"default_family_cap": 1}}
    guards.middleware.llm_request(guards.runtime, request={}, session_id="child-one", provider="local")
    completed = threading.Event()
    def request():
        guards.middleware.llm_request(guards.runtime, request={}, session_id="child-two", provider="local")
        completed.set()
    worker = threading.Thread(target=request, daemon=True)
    worker.start()
    time.sleep(0.03)
    assert second.activity == "queued" and not completed.is_set()
    second.blocked = True
    assert completed.wait(1)
    downstream = []
    assert guards.middleware.llm_execution(guards.runtime, request={}, next_call=lambda payload: downstream.append(payload),
                                           session_id="child-two") is None
    assert not downstream
    guards.middleware.post_api_request(guards.runtime, session_id="child-one", provider="local", api_request_id="first", usage={})
    third = guards.budgets.HandState("h-three", kind="research", child_session_id="child-three")
    guards.runtime.registry.add(third)
    guards.middleware.llm_request(guards.runtime, request={}, session_id="child-three", provider="local")
    assert third.activity == "generating"


def test_actual_route_coordinates_resolve_generic_provider_after_fallback(guards):
    hand(guards, routes=[{"provider": "openai", "model": "v", "base_url": "http://voice", "contract": {"provider": "voice"}},
        {"provider": "openai", "model": "t", "base_url": "http://text", "contract": {"provider": "text"}}])
    guards.runtime.config = {"contracts": {"voice": {"family": "a", "version": "1", "modalities": ["audio", "text"]},
        "text": {"family": "b", "version": "1", "modalities": ["text"]}}}
    request = {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "x"}}]}]}
    for model, host, expected in (("v", "http://voice", "input_audio"), ("t", "http://text", "text")):
        result = guards.middleware.llm_request(guards.runtime, request=request, session_id="child-one", provider="openai", model=model, base_url=host)
        assert result["request"]["messages"][0]["content"][0]["type"] == expected


def test_tool_receipts_contain_hashes_and_sources_without_tool_prose(guards):
    import hashlib
    state = hand(guards)
    guards.budgets.post_tool_call(guards.runtime, tool_name="web_extract", session_id="child-one", tool_call_id="call-1",
        args={"url": "https://example.test/source"}, result={"url": "https://example.test/source", "content": "Evidence text"})
    receipt = state.plan["tool_digest"][0]
    assert receipt["tool"] == "web_extract" and receipt["url"] == "https://example.test/source"
    assert receipt["content_hash"] == hashlib.sha256(b"Evidence text").hexdigest()
    assert "Evidence text" not in str(state.plan["tool_digest"])


def test_actual_contract_history_includes_served_model(guards):
    state = hand(guards)
    guards.runtime.config = {"contracts": {"text": {"family": "one", "version": "3"}}}
    guards.middleware.post_api_request(guards.runtime, session_id="child-one", provider="text", model="requested", response_model="served", usage={}, api_request_id="r1")
    assert state.plan["contracts_used"] == [{"provider": "text", "family": "one", "version": "3", "model": "served"}]


def test_refused_calls_are_marked_before_later_effect_hook_runs(guards):
    state = hand(guards, "coding", budgets={"wall_seconds": 600, "tokens": 100, "tool_calls": 0, "web_searches": 0})
    assert before(guards, "terminal", {"command": "date"}, tool_call_id="budget-call")["action"] == "block"
    assert "budget-call" in state.refused_calls
    assert allowed(guards, "send_message", {}, tool_call_id="policy-call")["action"] == "block"
    assert "policy-call" in state.refused_calls


def test_blocking_question_still_allows_tool_free_final_report(guards):
    state = hand(guards)
    state.blocked = True
    state.activity = "awaiting_input"
    request = {"messages": [], "tools": [{"name": "read_file"}], "tool_choice": "required"}
    result = guards.middleware.llm_request(guards.runtime, request=request, session_id="child-one")
    assert "tools" not in result["request"] and "tool_choice" not in result["request"]
    assert state.activity == "awaiting_input"
    assert guards.middleware.llm_execution(guards.runtime, request=result["request"], next_call=lambda payload: "final report",
                                           session_id="child-one") == "final report"
    assert before(guards, "read_file")["action"] == "block"


def test_missing_runtime_keeps_owner_tools_usable_and_blocks_children(guards):
    for module in (guards.allow, guards.budgets):
        assert module.pre_tool_call(None, tool_name="terminal", args={}, session_id="owner") is None
        assert module.pre_tool_call(None, tool_name="terminal", args={}, session_id="child", task_id="sa-one")["action"] == "block"


def test_refused_tool_results_do_not_claim_execution_receipts(guards):
    state = hand(guards)
    assert allowed(guards, "terminal", {}, tool_call_id="denied")["action"] == "block"
    after(guards, {"error": "refused"}, tool="terminal", tool_call_id="denied")
    assert not state.plan.get("tool_digest")


def test_recovery_alias_preserves_one_hand_and_child_bindings(guards):
    state = hand(guards)
    guards.runtime.registry.alias("h-remote", state.hand_id)
    assert guards.runtime.registry.get("h-remote") is state
    assert guards.runtime.registry.all() == [state]
    assert guards.runtime.registry.for_session("child-one") is state


def test_owner_model_call_continues_without_runtime(guards):
    assert guards.middleware.llm_execution(None, request={}, next_call=lambda payload: "owner reply", session_id="owner") == "owner reply"
    assert guards.middleware.llm_execution(None, request={}, next_call=lambda payload: pytest.fail("unbound child request"),
        session_id="child", task_id="sa-one") is None


def test_late_response_hook_cannot_release_newer_request_slot(guards):
    state = hand(guards)
    guards.runtime.config = {"contracts": {"local": {"family": "one", "version": "1", "modalities": ["text"]}}}
    guards.middleware.llm_request(guards.runtime, request={}, session_id="child-one", provider="local", api_request_id="old")
    guards.middleware.llm_request(guards.runtime, request={}, session_id="child-one", provider="local", api_request_id="new")
    guards.middleware.post_api_request(guards.runtime, session_id="child-one", provider="local", api_request_id="old", usage={})
    assert guards.runtime.registry.inflight[state.hand_id] == "one"
    assert state.activity == "generating"


@pytest.mark.parametrize("tool, args, seconds", (
    ("execute_code", {"code": "x", "timeout": None}, 300.0),
    ("execute_code", {"code": "x", "timeout": 30}, 300.0),
    ("execute_code", {"code": "x"}, 300.0),
    ("terminal", {"command": "x", "timeout": None}, 180.0),
    ("terminal", {"command": "x", "timeout": "45"}, 45.0),
    ("terminal", {"command": "x", "timeout": "soon"}, 180.0),
    ("terminal", {"command": "x", "timeout": True}, 180.0),
    ("terminal", {"command": "x", "timeout": -5}, 1.0),
    ("terminal", {"command": "x", "timeout": "nan"}, 180.0),
    ("terminal", {"command": "x", "timeout": "inf"}, 180.0),
    ("terminal", {"command": "x", "timeout": 10 ** 400}, 180.0),
    ("terminal", {"command": "x", "timeout": [5]}, 180.0),
    ("process_manage", {"action": "wait", "timeout": 40}, 40.0),
    ("browser_cdp", {"method": "x", "timeout": None}, 120.0),
    ("manage_connections", {"action": "wait", "timeout_seconds": None}, 120.0),
    ("manage_connections", {"action": "wait", "timeout_seconds": 20, "timeout": 99}, 20.0),
    ("read_file", {"path": "x", "timeout": 5}, 120.0),
))
def test_unusable_timeouts_never_fail_the_budget_check(guards, tool, args, seconds):
    # 2026-10-08 live: execute_code with "timeout": null raised in float() and the refusal stopped the hand.
    state = hand(guards, kind="ops")
    assert before(guards, tool, args, tool_call_id="call-1") is None
    assert state.expected_seconds == seconds
    assert state.tools == 1 and state.activity == "executing" and guards.runtime.stops == []


def test_non_finite_arguments_are_refused_without_counting_or_stopping(guards):
    # The effect spool refuses NaN/Infinity; the budget check refuses first, as before, so the hand is not stopped.
    state = hand(guards, kind="ops")
    result = before(guards, "terminal", {"command": "x", "timeout": float("nan")}, tool_call_id="call-1")
    assert result["action"] == "block"
    assert state.tools == 0 and guards.runtime.stops == []
