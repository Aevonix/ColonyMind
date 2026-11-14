"""Opt-in review suppression against the pinned native middleware and transports."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("COLONYMIND_HERMES_INTEGRATION") != "1",
    reason="integrator: opt in with COLONYMIND_HERMES_INTEGRATION=1 and HERMES_SOURCE_ROOT",
)


@pytest.mark.parametrize("api_mode", [
    "chat_completions", "codex_responses", "anthropic_messages", "bedrock_converse",
])
@pytest.mark.parametrize("streaming", [False, True])
def test_deferred_review_never_calls_native_transport_and_settles_silently(tmp_path, api_mode, streaming):
    _run_probe(tmp_path, api_mode, streaming)


@pytest.mark.parametrize("api_mode", [
    "chat_completions", "codex_responses", "anthropic_messages", "bedrock_converse",
])
@pytest.mark.parametrize("streaming", [False, True])
def test_held_child_uses_native_completion_without_another_provider_call(tmp_path, api_mode, streaming):
    _run_probe(tmp_path, "held:" + api_mode, streaming)


def test_deferred_review_keeps_provenance_in_native_queued_followup(tmp_path):
    _run_probe(tmp_path, "queued_followup", False)


def _run_probe(tmp_path, api_mode, streaming):
    source = Path(os.environ["HERMES_SOURCE_ROOT"])
    from hermes_source import assert_supported_source
    assert_supported_source(source)
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "config.yaml").write_text("terminal:\n  env: local\n", encoding="utf-8")
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("HERMES_", "COLONYMIND_", "TERMINAL_"))}
    environment.update(
        HERMES_HOME=str(profile), HERMES_SOURCE_ROOT=str(source),
        PYTHON_DOTENV_DISABLED="1", PYTHONDONTWRITEBYTECODE="1", TERMINAL_ENV="local",
    )
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), api_mode, str(int(streaming))],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _queued_followup_probe():
    """Exercise native queued replay with the selected display-kind contract."""
    import asyncio
    from types import SimpleNamespace as NS

    sys.path.insert(0, os.environ["HERMES_SOURCE_ROOT"])
    from gateway.platforms.base import MessageEvent
    from gateway.run_turn import GatewayTurnMixin
    from hermes_source import assert_supported_source
    revision = assert_supported_source()
    from test_hands_front import event, owner_turn, rig
    from test_hands_review_requests import scheduled_review, settle

    fixture = rig.__wrapped__(Path(os.environ['HERMES_HOME']))
    state = next(fixture)
    try:
        scheduled_review(state)
        owner_turn(state, "Another owner message", "owner-race", "message-3")
        injected = MessageEvent(text=state.ctx.injected[0]["content"], message_id="", internal=True,
            source=event().source)
        observations = []

        async def noop(*args, **kwargs):
            return None

        async def prepare(**kwargs):
            return kwargs["event"].text

        async def run_agent(**kwargs):
            # Public sources preserve internal provenance on the queued turn.
            display_kind = kwargs.get("persist_user_display_kind")
            assert display_kind == (None if revision not in {"S", "S217"} else "internal_notification")
            state.agent._current_turn_id = "replay-turn"
            state.guard.pre_llm_call(session_id=kwargs["session_id"], turn_id="replay-turn",
                user_message=kwargs["message"], platform="sms", sender_id="5550100", chat_id="5550100",
                session_key=kwargs["session_key"],
                conversation_history=[{"role": "user", "content": kwargs["message"], "display_kind": display_kind}])
            assert state.guard.is_review_turn("owner-session", "replay-turn")
            assert not state.guard.owner_started_turn("owner-session", "replay-turn")
            assert state.guard.capture_owner_text("owner-session", "replay-turn", kwargs["message"]) is None
            settle(state, "replay-turn", "Reviewed result")
            observations.append(True)
            return {"completed": True, "messages": [], "final_response": "Reviewed result"}

        slots = {}

        def session_state(key):
            # A source that runs the follow-up as its own sender's turn asks the runner for the session's slot:
            # the source's own state class, so the turn it binds is the native one.
            from gateway.session_state import SessionState
            return slots.setdefault(key, SessionState())

        runner = NS(_MAX_INTERRUPT_DEPTH=10, _is_goal_continuation_event=lambda event: False,
            _session_state=session_state,
            _session_key_for_source=lambda source: "route", _prepare_profile_scoped_inbound_message_text=prepare,
            _pinned_channel_inputs=lambda key, prompt, source, **kwargs: (prompt, source),
            _reply_anchor_for_event=lambda event: None, _adapter_for_source=lambda source: None,
            _delivery_adapter_for=lambda source: None, _intake_adapter_for=lambda source: None,
            _refresh_agent_cache_message_count=noop, _run_agent=run_agent,
            _run_agent_deliver_first_response=noop)
        turn = NS(source=injected.source, session_id="owner-session", session_key="route", run_generation=3,
            _interrupt_depth=0, history=[], _status_thread_metadata={}, context_prompt="Assistant", result_holder=[{}])
        asyncio.run(GatewayTurnMixin._run_agent_queued_followup(runner, turn, None, injected.text, injected,
            "Owner answer", {"completed": True, "messages": []}, None))
        assert observations == [True]
        assert len(state.guard.rendered) == len(state.ctx.injected) == 1
        assert not state.guard.deferred_reviews
    finally:
        fixture.close()


def _native_probe(api_mode, streaming, *, held=False):
    from copy import deepcopy
    from importlib import import_module
    import threading
    from types import SimpleNamespace as NS
    from unittest.mock import patch

    from hands_fake_hermes import FakeContext, load_hands

    sys.path.insert(0, os.environ["HERMES_SOURCE_ROOT"])
    from agent.transports import get_transport
    from agent.turn_api_call import perform_api_call
    from agent.turn_recovery import validate_response_shape
    from agent.turn_retry_state import TurnRetryState
    from agent.turn_usage import record_response_usage
    from agent.usage_pricing import normalize_usage
    from gateway.response_filters import is_intentional_silence_response
    from hermes_cli.middleware import apply_llm_request_middleware
    import hermes_cli.plugins as native_plugins

    package = load_hands()
    middleware = import_module(package.__name__ + ".middleware")
    dispatch = import_module(package.__name__ + ".dispatch")
    budgets = import_module(package.__name__ + ".budgets")
    runtime = NS(registry=budgets.HandRegistry(), config={})
    if held:
        runtime.registry.add(budgets.HandState("held-hand", child_session_id="owner-session",
            parent_session_id="parent-session", blocked=True, activity="awaiting_approval"))
    ctx = FakeContext()
    calls = []
    bridge = NS(
        request_caps=lambda sid, tid="": {"mode": "review_result", "max_tokens": 512, "duplicate": False},
        defer_review_request=lambda sid, tid="": calls.append((sid, tid)) or True,
    )
    plugin_manager = native_plugins.get_plugin_manager()
    plugin_manager._middleware = ctx.middleware

    def forbidden(*args, **kwargs):
        raise AssertionError("deferred review reached the provider")

    transport = get_transport(api_mode)
    assert transport is not None
    usage_notes = []
    agent = NS(
        api_mode=api_mode, provider="test-provider", model="test-model", base_url="https://model.invalid/v1",
        session_id="owner-session", platform="sms", _disable_streaming=not streaming,
        _has_stream_consumers=lambda: streaming, _get_transport=lambda: transport,
        _interruptible_streaming_api_call=forbidden, _interruptible_api_call=forbidden,
        _model_request_active=threading.Event(), _pending_redirect_lock=threading.RLock(),
        _pending_redirect=None, _has_pending_redirect=lambda: False,
        thinking_callback=None, session_api_calls=0,
        session_input_tokens=321, session_output_tokens=123, session_estimated_cost_usd=0.25,
        context_compressor=NS(
            awaiting_real_usage_after_compression=False,
            note_usage_less_response=lambda: usage_notes.append(True), update_from_response=forbidden,
        ),
    )
    request = {"model": agent.model, "messages": [{"role": "user", "content": "Native completion"}]}
    original = deepcopy(request)
    ids = dict(session_id=agent.session_id, turn_id="review-turn", api_mode=api_mode, model=agent.model)
    with patch.object(dispatch, "get_runtime", lambda context=None: runtime), \
            patch.dict(sys.modules, {"colonymind_hands_front": bridge}), \
            patch.object(native_plugins, "has_middleware", lambda kind: bool(ctx.middleware.get(kind))), \
            patch.object(native_plugins, "invoke_middleware",
                         lambda kind, **kwargs: [callback(**kwargs) for callback in ctx.middleware[kind]]), \
            patch.object(native_plugins, "get_plugin_manager", lambda: plugin_manager):
        middleware.register(ctx)
        assert len(ctx.middleware["llm_request"]) == len(ctx.middleware["llm_execution"]) == 1
        prepared = apply_llm_request_middleware(request, **ids)
        assert request == original
        assert prepared.payload["messages"] == original["messages"]
        verdict = perform_api_call(
            agent, api_kwargs=prepared.payload, _original_api_kwargs=prepared.original_payload,
            _llm_middleware_trace=prepared.trace, _moa_prepared_request=None,
            _retry=TurnRetryState(), thinking_spinner=None, retry_count=0, api_call_count=1,
            api_request_id="review-turn:api:1", effective_task_id="task", turn_id="review-turn",
            interrupted=False,
        )
        review_calls = list(calls)
        if api_mode == "chat_completions" and not streaming:
            _check_complete_native_turn(forbidden, held=held, runtime=runtime, dispatch=dispatch)
    assert (not review_calls) if held else (review_calls and set(review_calls) == {("owner-session", "review-turn")})
    assert verdict.action == "fallthrough" and not verdict.interrupted
    assert not agent._model_request_active.is_set()
    response = verdict.response
    assert validate_response_shape(agent, response) == (False, [])
    normalized = transport.normalize_response(response)
    expected = "Waiting for owner approval. The held action has not run." if held else "[SILENT]"
    assert normalized.content == expected and normalized.finish_reason == "stop"
    assert not normalized.tool_calls
    assert is_intentional_silence_response(normalized.content) is not held
    usage = normalize_usage(getattr(response, "usage", None), api_mode=api_mode, provider=agent.provider)
    assert usage.total_tokens == 0
    record_response_usage(agent, response, messages=original["messages"], api_call_count=1,
                          api_duration=0, compression_attempts=0, max_compression_attempts=3)
    assert (agent.session_input_tokens, agent.session_output_tokens, agent.session_estimated_cost_usd) == (321, 123, 0.25)
    assert usage_notes == [True], "local settlement must not reset the provider token anchor"


def _check_complete_native_turn(forbidden, *, held=False, runtime=None, dispatch=None):
    """The normal turn loop must reach both completion hooks without retrying."""
    from unittest.mock import patch

    import hermes_cli.lifecycle as lifecycle
    from run_agent import AIAgent
    from types import SimpleNamespace as NS

    observed = []

    def observe(name, **values):
        observed.append((name, values))
        return []

    with patch("model_tools.get_tool_definitions", return_value=[]), \
            patch("model_tools.check_toolset_requirements", return_value={}), \
            patch("agent.process_bootstrap.OpenAI"), \
            patch.object(lifecycle, "invoke_hook", observe):
        agent = AIAgent(
            model="test-model", api_mode="chat_completions", provider="custom",
            base_url="https://model.invalid/v1", api_key="test-only-key", quiet_mode=True,
            max_iterations=3, skip_context_files=True, skip_memory=True, skip_background_review=True,
            session_id="owner-session", platform="sms",
        )
        agent._interruptible_streaming_api_call = forbidden
        agent._interruptible_api_call = forbidden
        if held:
            from tools.delegate_tool_results import _run_child_lifecycle
            import hermes_cli.plugins as native_plugins

            records = []
            runtime.event = lambda *args: None
            runtime._checks = lambda hand: []
            runtime.epoch, runtime.epoch_key = {}, "test"
            runtime._schema_verdicts, runtime.native = {}, NS(inactivity_timeout=False)
            runtime.effects = NS(open_effects=lambda hand_id: [])
            runtime.emit = lambda path, body, key: records.append((path, body)) or {}

            def child_hook(name, **values):
                observed.append((name, values))
                if name == "subagent_stop":
                    dispatch.Runtime.subagent_stop(runtime, **values)
                return []

            with patch.object(native_plugins, "invoke_hook", child_hook):
                result = _run_child_lifecycle(0, "Native completion", agent, NS(session_id="parent-session"))
            assert result["status"] == result["exit_reason"] == "completed"
            assert result["summary"] == "Waiting for owner approval. The held action has not run."
            assert result["api_calls"] == 1
            assert "subagent_stop" in {name for name, _ in observed}
            assert len(records) == 1 and records[0][0].endswith("/result")
            assert records[0][1]["status"] == "blocked"
            return
        result = agent.run_conversation("Native completion", system_message="Assistant.")
    assert result["completed"] is True and not result["interrupted"]
    assert result["final_response"] == "[SILENT]"
    assert result["api_calls"] == 1
    assert result["messages"][-1]["content"] == "[SILENT]"
    assert {"post_llm_call", "on_session_end"} <= {name for name, _ in observed}


if __name__ == "__main__":
    from unittest.mock import patch

    with patch("socket.socket.connect", side_effect=AssertionError("native review probe must remain offline")):
        if sys.argv[1] == "queued_followup":
            _queued_followup_probe()
        elif sys.argv[1].startswith("held:"):
            _native_probe(sys.argv[1][5:], bool(int(sys.argv[2])), held=True)
        else:
            _native_probe(sys.argv[1], bool(int(sys.argv[2])))
