"""Front limits compose with hands guards in the one native request callback.

The bridge here is a stub: the cap it names only lowers a bound the caller sent. No bound is added to a
request that has none, and the caller's thinking setting is left as it is.
"""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def guarded(monkeypatch):
    name = "_hands_front_middleware_tests"
    path = Path(__file__).resolve().parents[2] / "plugins/colonymind-hands"
    package = ModuleType(name)
    package.__path__ = [str(path)]
    monkeypatch.setitem(sys.modules, name, package)
    modules = {}
    for suffix in ("budgets", "middleware"):
        fullname = name + "." + suffix
        spec = importlib.util.spec_from_file_location(fullname, path / (suffix + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, fullname, module)
        spec.loader.exec_module(module)
        modules[suffix] = module
    runtime = SimpleNamespace(registry=modules["budgets"].HandRegistry(), config={})
    bridge = SimpleNamespace(request_caps=lambda sid, tid="": {
        "mode": "review_result", "max_tokens": 512, "duplicate": False})
    monkeypatch.setitem(sys.modules, "colonymind_hands_front", bridge)
    dispatch = SimpleNamespace(get_runtime=lambda ctx=None: runtime)
    monkeypatch.setitem(sys.modules, name + ".dispatch", dispatch)
    return SimpleNamespace(**modules, runtime=runtime, bridge=bridge, dispatch=dispatch,
                           package=name, path=path)


@pytest.mark.parametrize("mode,cap", [("owner", 0), ("review_result", 512), ("system", 512)])
@pytest.mark.parametrize("token_field", ["max_tokens", "max_completion_tokens", "max_output_tokens"])
def test_front_cap_reaches_each_route_and_retry_without_losing_audio_guard(guarded, mode, cap, token_field):
    calls = []
    guarded.bridge.request_caps = lambda sid, tid="": calls.append((sid, tid)) or {
        "mode": mode, "max_tokens": cap, "duplicate": False}
    request = {token_field: 8192, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Please check this."},
        {"type": "input_audio", "input_audio": {"data": "sample"}}]}],
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True, "other": "kept"},
                       token_field: 9000, "top_k": 20}}
    original = deepcopy(request)
    for provider in ("primary", "fallback"):
        payload = guarded.middleware.llm_request(guarded.runtime, request, session_id="owner-session",
            turn_id="turn-1", provider=provider)["request"]
        assert payload[token_field] == (cap or 8192)
        assert payload["extra_body"][token_field] == (cap or 9000)
        assert payload["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True, "other": "kept"}
        assert payload["extra_body"]["top_k"] == 20
        assert payload["messages"][0]["content"][1]["type"] == "text"
    assert calls == [("owner-session", "turn-1")] * 2
    assert request == original


def test_lower_existing_caps_are_preserved_and_no_missing_cap_is_added(guarded):
    assert guarded.middleware.llm_request(guarded.runtime, {"max_tokens": 120})["request"]["max_tokens"] == 120
    assert "max_tokens" not in guarded.middleware.llm_request(guarded.runtime, {"messages": []})["request"]
    assert "max_output_tokens" not in guarded.middleware.llm_request(guarded.runtime, {"input": []})["request"]
    request = {"max_tokens": 120, "extra_body": {"max_tokens": 8192}}
    payload = guarded.middleware.llm_request(guarded.runtime, request)["request"]
    assert payload["max_tokens"] == payload["extra_body"]["max_tokens"] == 120


def test_duplicate_review_keeps_native_input_but_removes_tools(guarded):
    guarded.bridge.request_caps = lambda sid, tid="": {"mode": "review_result", "max_tokens": 512, "duplicate": True}
    request = {"messages": [{"role": "user", "content": "Native completion event"}],
               "tools": [{"name": "terminal"}], "tool_choice": "required", "parallel_tool_calls": True}
    payload = guarded.middleware.llm_request(guarded.runtime, request)["request"]
    assert payload["messages"] == request["messages"]
    assert "max_tokens" not in payload
    assert not {"tools", "tool_choice", "parallel_tool_calls"} & payload.keys()


def test_child_identity_and_thinking_remain_independent_of_front_caps(guarded):
    hand = guarded.budgets.HandState("hand-1", kind="research", plan={"identity": "Shared assistant identity."})
    guarded.runtime.registry.add(hand)
    guarded.runtime.registry.bind(hand, "child-session", subagent_id="sa-1")
    request = {"max_tokens": 4096, "system": guarded.middleware.IDENTITY_ANCHOR,
               "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
    payload = guarded.middleware.llm_request(guarded.runtime, request, session_id="child-session")["request"]
    assert payload["max_tokens"] == 4096
    assert payload["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert payload["system"].startswith("Shared assistant identity.")


@pytest.mark.parametrize("failure", ["resolve", "contract", "runtime"])
def test_front_caps_survive_guard_errors_and_registration_runtime_failure(guarded, monkeypatch, failure):
    def unavailable(*args, **kwargs):
        raise RuntimeError("unavailable")
    callbacks = {}
    ctx = SimpleNamespace(register_middleware=lambda name, fn: callbacks.setdefault(name, []).append(fn),
                          register_hook=lambda *args: None)
    if failure == "runtime":
        guarded.dispatch.get_runtime = unavailable
    else:
        monkeypatch.setattr(guarded.middleware, "resolve_hand" if failure == "resolve" else "contract_for", unavailable)
    guarded.middleware.register(ctx)
    request = {"max_tokens": 8192, "messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": "sample"}}]}]}
    payload = callbacks["llm_request"][0](request, session_id="owner-session", turn_id="turn-1")["request"]
    assert payload["max_tokens"] == 512
    assert "chat_template_kwargs" not in payload and "chat_template_kwargs" not in payload.get("extra_body", {})
    assert payload["messages"][0]["content"][0]["type"] == "text"
    assert len(callbacks["llm_request"]) == 1


def test_absent_bridge_and_unclassified_voice_do_not_gain_front_caps(guarded, monkeypatch):
    guarded.runtime.config = {"contracts": {"audio-route": {"modalities": ["audio"]}}}
    request = {"max_tokens": 4096, "messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": "sample"}}]}]}
    guarded.bridge.request_caps = lambda sid, tid="": None
    assert guarded.middleware.llm_request(guarded.runtime, request, session_id="voice", provider="audio-route")["request"] == request
    monkeypatch.delitem(sys.modules, "colonymind_hands_front")
    assert guarded.middleware.llm_request(guarded.runtime, request, provider="audio-route")["request"] == request


def test_front_bridge_failure_does_not_undo_audio_or_identity_guards(guarded):
    def unavailable(*args):
        raise RuntimeError("ledger unavailable")
    guarded.bridge.request_caps = unavailable
    request = {"max_tokens": 4096, "messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": "sample"}}]}]}
    payload = guarded.middleware.llm_request(guarded.runtime, request)["request"]
    assert payload["messages"][0]["content"][0]["type"] == "text"


def test_missing_child_runtime_does_not_apply_owner_caps(guarded):
    request = {"max_tokens": 4096, "messages": []}
    payload = guarded.middleware.llm_request(None, request, session_id="child-session", platform="subagent")["request"]
    assert payload == request


def test_pinned_native_request_chain_preserves_front_caps_and_audio(guarded, monkeypatch):
    configured = os.environ.get("HERMES_SOURCE_ROOT")
    if not configured:
        pytest.skip("integrator: set HERMES_SOURCE_ROOT to the pinned Hermes checkout")
    source = Path(configured)
    from hermes_source import assert_supported_source
    assert_supported_source(source)
    name = "_hands_front_native_middleware"
    spec = importlib.util.spec_from_file_location(name, source / "hermes_cli/middleware.py")
    native = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, native)
    spec.loader.exec_module(native)
    callbacks = {}
    ctx = SimpleNamespace(register_middleware=lambda key, fn: callbacks.setdefault(key, []).append(fn),
                          register_hook=lambda *args: None)
    guarded.middleware.register(ctx)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", SimpleNamespace(
        has_middleware=lambda key: bool(callbacks.get(key)),
        invoke_middleware=lambda key, **kwargs: [fn(**kwargs) for fn in callbacks[key]]))
    request = {"max_tokens": 8192, "messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": "sample"}}]}]}
    result = native.apply_llm_request_middleware(request, session_id="owner-session", turn_id="turn-1")
    assert result.payload["max_tokens"] == 512
    assert ("chat_template_kwargs" not in result.payload
            and "chat_template_kwargs" not in result.payload.get("extra_body", {}))
    assert result.payload["messages"][0]["content"][0]["type"] == "text"
    assert result.original_payload == request
    assert len(callbacks["llm_request"]) == 1


@pytest.mark.parametrize("bridge_result", [{"platform": "sms", "chat": "chat-1", "message_id": "message-1"}, None])
@pytest.mark.parametrize("chat_type", ["dm", "group", "shared", "guest"])
def test_dispatch_origin_comes_only_from_ingress_bridge(guarded, monkeypatch, bridge_result, chat_type):
    fullname = guarded.package + ".dispatch"
    monkeypatch.delitem(sys.modules, fullname)
    spec = importlib.util.spec_from_file_location(fullname, guarded.path / "dispatch.py")
    dispatch = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, fullname, dispatch)
    spec.loader.exec_module(dispatch)
    seen = []
    guarded.bridge.origin_ref = lambda sid, tid: seen.append((sid, tid)) or bridge_result
    env = {'HERMES_SESSION_PLATFORM': 'sms', 'HERMES_SESSION_CHAT_ID': 'chat-1',
           'HERMES_SESSION_CHAT_TYPE': chat_type}
    monkeypatch.setitem(sys.modules, "gateway.session_context", SimpleNamespace(get_session_env=lambda key: env.get(key, '')))
    monkeypatch.setitem(sys.modules, "tools.delegate_tool_dispatch", SimpleNamespace(
        _capture_origin=lambda: ("wake-session", None, None, None, True)))
    parent = SimpleNamespace(session_id="owner-session", _current_turn_id="turn-1")
    native = dispatch.NativeHermes.__new__(dispatch.NativeHermes)
    context = native.context(parent)
    assert context["origin_ref"] == (bridge_result if chat_type == 'dm' else None)
    assert seen == [("owner-session", "turn-1")]
    def unavailable(*args):
        raise RuntimeError("ledger unavailable")
    guarded.bridge.origin_ref = unavailable
    assert native.context(parent)["origin_ref"] is None
    monkeypatch.delitem(sys.modules, "colonymind_hands_front")
    assert native.context(parent)["origin_ref"] == ('turn-1' if chat_type == 'dm' else None)
