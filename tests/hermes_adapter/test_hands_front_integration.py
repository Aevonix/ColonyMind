"""Integrator-only probes against a pinned Hermes source tree.

Run with COLONYMIND_HERMES_INTEGRATION=1 and HERMES_SOURCE_ROOT pointing at the
pinned checkout, using an interpreter with Hermes dependencies installed. Each
probe has a fresh process and temporary profile. Transport methods record actual
Hermes consumer output; the probes do not contact an external model or service.
"""
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


def _probe(name, tmp_path):
    source = os.environ.get("HERMES_SOURCE_ROOT")
    assert source and (Path(source) / "gateway/run.py").is_file(), "HERMES_SOURCE_ROOT must select pinned Hermes"
    from hermes_source import assert_supported_source
    assert_supported_source(source)
    environment = dict(os.environ)
    environment.update(
        HERMES_HOME=str(tmp_path / "profile"),
        TERMINAL_CWD=str(tmp_path),
        TERMINAL_ENV="local",
        HERMES_GATEWAY_BUSY_ACK_ENABLED="false",
        HERMES_GATEWAY_BUSY_TEXT_DEBOUNCE_SECONDS="0",
        HERMES_HUMAN_DELAY_MODE="off",
        PYTHONDONTWRITEBYTECODE="1",
    )
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "config.yaml").write_text("terminal:\n  env: local\n  timeout: 30\n", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), name],
        env=environment, cwd=tmp_path, capture_output=True, text=True, timeout=35,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_busy_queue_owner_arrival_steers_at_boundary_without_dispatch_hook(tmp_path):
    _probe("busy_queue", tmp_path)


def test_parked_owner_message_binds_its_follow_up_and_schedules_the_retry(tmp_path):
    _probe("busy_follow_up", tmp_path)


def test_native_streaming_duplicate_review_has_no_outbound_message(tmp_path):
    _probe("native_duplicate", tmp_path)


def test_slow_request_wait_is_never_aborted_by_a_front_deadline(tmp_path):
    _probe("request_wait", tmp_path)


def test_terminal_boundary_yield_and_native_background_handoff(tmp_path):
    _probe("terminal_yield", tmp_path)


def test_pending_agent_can_bind_after_tracker_promotion(tmp_path):
    _probe("late_binding", tmp_path)


def test_native_tool_dispatcher_gets_no_timeout_or_call_cap_and_honors_a_front_block(tmp_path):
    _probe("tool_directives", tmp_path)


def test_native_coalesced_completion_header_is_recognized(tmp_path):
    _probe("coalesced_batch", tmp_path)


def _fixture(*, start_threads=False):
    import asyncio
    import importlib
    import threading
    import time
    from types import SimpleNamespace

    from agent.interrupt_control import InterruptControlMixin
    from agent.subagent_lifecycle import bind_subagent_parent
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.base import BasePlatformAdapter, SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource, build_session_key
    from gateway.session_context import set_session_vars
    from tools import async_delegation
    from tools.process_registry_notifications import format_process_notification
    from hands_fake_hermes import FakeContext, FakeController, FakeHermes, load_hands

    class Agent(InterruptControlMixin):
        def __init__(self):
            self.session_id = "owner-session"
            self._current_turn_id = "turn-1"
            self._delegate_depth = 0
            self._pending_redirect_lock = threading.Lock()
            self._pending_steer_lock = threading.Lock()
            self._pending_redirect = self._pending_steer = None
            self._interrupt_requested = False
            self._hard_interrupt_requested = threading.Event()
            self._model_request_active = threading.Event()
            self._model_request_active.set()
            self._execution_thread_id = None
            self._tool_worker_threads = set()
            self._tool_worker_threads_lock = threading.RLock()
            self._active_children = []
            self._active_children_lock = threading.RLock()
            self.quiet_mode = True
            self._executing_tools = False
            self._disable_streaming = False
            self.base_url = "http://model.invalid/v1"
            self.provider = "test-provider"
            self._has_stream_consumers = lambda: bool(getattr(self, "stream_delta_callback", None))

    class Adapter(BasePlatformAdapter):
        MAX_MESSAGE_LENGTH = 39000
        draft_stream_is_message = True

        def __init__(self):
            super().__init__(PlatformConfig(), Platform.SMS)
            self.outbound = []
            self._typing_paused = set()
            self._busy_text_debounce_seconds = 0

        async def connect(self):
            return True

        async def disconnect(self):
            pass

        async def get_chat_info(self, chat_id):
            return {"chat_id": chat_id, "chat_type": "dm"}

        async def send_typing(self, chat_id, metadata=None):
            pass

        async def stop_typing(self, chat_id, metadata=None):
            pass

        def supports_draft_streaming(self, chat_type=None, metadata=None):
            return True

        async def send(self, chat_id, content, reply_to=None, metadata=None, **kwargs):
            self.outbound.append(("send", content))
            return SendResult(success=True, message_id="reply-1")

        async def send_draft(self, *, chat_id, draft_id, content, metadata=None):
            self.outbound.append(("draft", content))
            return SendResult(success=True, message_id=None)

        async def edit_message(self, chat_id, message_id, content, **kwargs):
            self.outbound.append(("edit", content))
            return SendResult(success=True, message_id=message_id)

    class Controller(FakeController):
        def __init__(self):
            super().__init__()
            self.items = [dict(hand_id="h-" + str(n) * 12, task_id="t-" + str(n) * 12,
                               obligation_id="o-" + str(n) * 12, text=f"result {n}") for n in range(1, 9)]

        def post(self, path, body, key):
            if path.endswith("/digest"):
                return {"text": "Eight results ready", "items": self.items, "cursor": "cursor-1"}
            return super().post(path, body, key)

        def get(self, path, params=None):
            if "digest" in path:
                return {"text": "Eight results ready", "items": self.items, "cursor": "cursor-1"}
            rows = [dict(item, id=item["hand_id"], parent_session="owner-session", state="done",
                         status="done", outcome="done", result_hash="a" * 64, intent_version=1)
                    for item in self.items]
            return next((row for row in rows if path.endswith(row["id"])), {"hands": rows})

    package = load_hands()
    dispatch = importlib.import_module(package.__name__ + ".dispatch")
    budgets = importlib.import_module(package.__name__ + ".budgets")
    front = importlib.import_module(package.__name__ + ".front")
    native, client, ctx = FakeHermes(), Controller(), FakeContext()
    ctx.injected = []
    def inject_message(content, role="user", *, session_key=None):
        ctx.injected.append((content, dict(session_key=session_key, role=role)))
        return True
    ctx.inject_message = inject_message
    runtime = dispatch.Runtime(config={}, client=client, spool_dir=Path(os.environ["HERMES_HOME"]) / "spool",
                               native=native, owner_check=lambda sid: sid == "owner-session")
    now = [100.0]
    guard = front.FrontGuard(runtime, ctx, clock=lambda: now[0], start_threads=start_threads)
    agent, adapter = Agent(), Adapter()
    source = SessionSource(platform=Platform.SMS, chat_id="owner-chat", chat_type="dm", user_id="owner")
    key = build_session_key(source)
    native.async_module._db_path = async_delegation._db_path
    completion_events = []
    for n, item in enumerate(client.items, 1):
        delegation_id = f"delegation-{n}"
        timestamp = time.time()
        record = dict(delegation_id=delegation_id, parent_session_id=agent.session_id,
                      session_key=key, dispatched_at=timestamp, goal=f"Check item {n}")
        event = dict(record, type="async_delegation", status="completed", completed_at=timestamp,
                     summary=f"Result {n}", duration_seconds=1)
        async_delegation._persist_dispatch(record)
        async_delegation._persist_completion(event, {"summary": event["summary"]})
        runtime.registry.add(budgets.HandState(hand_id=item["hand_id"], task_id=item["task_id"],
            parent_session_id=agent.session_id, delegation_id=delegation_id, status="done"))
        completion_events.append(event)
    gateway = object.__new__(GatewayRunner)
    gateway._session_state(key).turn.agent = agent
    generation = gateway._begin_session_run_generation(key)
    gateway._turn_leases = SimpleNamespace(release=lambda token: True)
    gateway._session_state(key).turn.lease_tokens[generation] = SimpleNamespace(session_id=agent.session_id)
    gateway.adapters = {Platform.SMS: adapter}
    gateway._adapter_for_source = lambda source: adapter
    gateway._busy_input_mode = "queue"
    gateway._effective_busy_input_mode = lambda source: "queue"
    gateway._effective_busy_text_mode = lambda source: "queue"
    gateway._is_user_authorized_for_source = lambda source: True
    gateway._admit_bot_message_for_source = lambda source: True
    gateway._draining = False
    async def no_approval(*args):
        return False
    gateway._route_plaintext_approval_while_busy = no_approval
    gateway.session_store = SimpleNamespace(
        config=SimpleNamespace(multiplex_profiles=False),
        _entries={key: SimpleNamespace(session_id=agent.session_id, session_key=key)},
        _lock=threading.RLock(), _ensure_loaded_locked=lambda: None,
    )
    adapter._busy_session_handler = gateway._handle_active_session_busy_message
    adapter._active_sessions[key] = asyncio.Event()
    gateway._bind_adapter_run_generation(adapter, key, generation)
    runtime.gateway = gateway
    guard.pre_gateway_dispatch(event=MessageEvent(text="Initial owner message", source=source,
                               message_id="owner-message-0"), gateway=gateway,
                               session_key=key, session_store=gateway.session_store)

    def start(internal=False, turn="turn-1", internal_text=None):
        agent._current_turn_id = turn
        current = gateway._session_state(key)
        current.turn.lease_tokens.setdefault(current.persistent.run_generation, SimpleNamespace(session_id=agent.session_id))
        completion = completion_events[min(int(turn.rsplit("-", 1)[-1]) - 1, 7)]
        message = (internal_text or format_process_notification(completion)) if internal else "Please check this"
        event = MessageEvent(text=message, source=source,
                             internal=internal, message_id=None if internal else "owner-message-1")
        if not internal:
            guard.pre_gateway_dispatch(event=event, gateway=gateway, session_key=key,
                                       session_store=gateway.session_store)
        row = {"role": "user", "content": event.text}
        if internal:
            row["display_kind"] = "internal_notification"
        # _bind_turn_identity: the turn's own raw inbound id, None for an internal event.
        agent._persist_user_message_platform_id = event.message_id
        set_session_vars(platform="sms", chat_id=source.chat_id, chat_type="dm", user_id=source.user_id,
                         session_key=key, session_id=agent.session_id, message_id=event.message_id or "")
        # run_conversation binds this before the gateway's 50ms tracker promotion.
        with bind_subagent_parent(agent):
            guard.pre_llm_call(session_id=agent.session_id, turn_id=turn, user_message=event.text,
                               platform="sms", sender_id=source.user_id, conversation_history=[row])
        return event

    return SimpleNamespace(**locals())


def _begin_tool(env, call='call-1', turn='turn-1'):
    env.agent._model_request_active.clear()
    env.agent._executing_tools = True
    return env.guard.pre_tool_call(session_id=env.agent.session_id, turn_id=turn,
        tool_name='read_file', tool_call_id=call, args={})


def _end_tool(env, call='call-1', turn='turn-1'):
    env.guard.post_tool_call(session_id=env.agent.session_id, turn_id=turn,
        tool_name='read_file', tool_call_id=call, result='done')
    env.agent._executing_tools = False


def _busy_queue():
    import asyncio
    import time
    from gateway.platforms.event import MessageEvent

    env = _fixture(start_threads=True)
    try:
        review = env.start(internal=True)
        owner = MessageEvent(text="New owner message", source=env.source, message_id="owner-message-2")
        started = time.monotonic()
        asyncio.run(env.adapter._handle_message_while_active(owner, env.key))
        assert env.adapter._pending_messages[env.key] is owner
        time.sleep(0.15)
        assert env.agent._pending_redirect is None and env.agent._pending_steer is None
        _begin_tool(env)
        _end_tool(env)
        assert env.agent._pending_steer, "queued owner guidance belongs at the tool boundary"
        assert env.agent._pending_redirect is None
        assert time.monotonic() - started < 2
        assert env.adapter._pending_messages[env.key] is owner, "watchdog must not consume queued owner text"

        async def drain_after_boundary():
            owner_ran = asyncio.Event()
            async def handler(event):
                if event is review:
                    assert env.agent._pending_steer
                elif event is owner:
                    owner_ran.set()
                return None
            env.adapter._message_handler = handler
            env.adapter._start_session_processing(review, env.key)
            await asyncio.wait_for(owner_ran.wait(), 2)
            assert time.monotonic() - started < 2
            await env.adapter.cancel_background_tasks()
        asyncio.run(drain_after_boundary())

        # The preceding turn consumed its guidance. Drain it explicitly in case
        # clear_interrupt retains an unconsumed steer.
        env.agent._drain_pending_steer()
        env.agent.clear_interrupt()
        env.start(internal=True, turn="turn-2")
        internal = MessageEvent(text="another completion", source=env.source, internal=True)
        env.adapter._pending_messages[env.key] = internal
        env.gateway._enqueue_fifo(env.key, owner, env.adapter)
        env.guard.tick(env.agent.session_id, 'turn-2')
        assert env.agent._pending_redirect is None and env.agent._pending_steer is None
        _begin_tool(env, turn='turn-2')
        _end_tool(env, turn='turn-2')
        assert env.agent._pending_steer, "owner in runner FIFO must also get boundary guidance"
        assert env.gateway._peek_session_state(env.key).conversation.queued_events == [owner]
    finally:
        env.guard.close()
        env.runtime.close()


def _native_duplicate():
    import asyncio
    from agent.turn_api_call import _should_stream
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
    from gateway.run import _AGENT_PENDING_SENTINEL

    env = _fixture()
    async def deliver(turn, text, internal_text=None):
        consumer = GatewayStreamConsumer(env.adapter, env.source.chat_id,
            StreamConsumerConfig(transport="auto", chat_type="dm", edit_interval=0.01,
                                 buffer_threshold=1, cursor=""))
        env.agent.stream_delta_callback = consumer.on_delta
        env.agent.interim_assistant_callback = consumer.on_delta
        if turn == "turn-2":
            env.gateway._session_state(env.key).turn.agent = _AGENT_PENDING_SENTINEL
        env.start(internal=True, turn=turn, internal_text=internal_text)
        task = asyncio.create_task(consumer.run())
        callback = env.agent.stream_delta_callback
        if _should_stream(env.agent) and callback:
            callback(text)
        await asyncio.sleep(0.04)
        env.gateway._session_state(env.key).turn.agent = env.agent
        # The native transform_llm_output hook omits turn_id.
        final = env.guard.finalize_response(response_text=text, session_id=env.agent.session_id) or text
        consumer.finish(final)
        await asyncio.wait_for(task, 2)
        env.guard.post_llm_call(session_id=env.agent.session_id, turn_id=turn, assistant_response=final)
        env.guard.on_session_end(session_id=env.agent.session_id, turn_id=turn,
            completed=True, failed=False, interrupted=False)
        return final

    async def run():
        assert await deliver("turn-1", "All eight results are ready.") != "[SILENT]"
        assert env.adapter.outbound
        env.adapter.outbound.clear()
        for n in range(2, 9):
            final = await deliver(f"turn-{n}", "Duplicate model output that must stay private.")
            assert final == "[SILENT]"
        assert env.adapter.outbound == [], "native draft, edit and send paths must all stay silent"
        final = await deliver("turn-9", "The background command finished.",
                              internal_text="A separate background command has completed.")
        assert final != "[SILENT]", "internal provenance alone does not identify a completion review"
        assert env.adapter.outbound
    try:
        asyncio.run(run())
    finally:
        env.guard.close()
        env.runtime.close()


def _request_wait():
    import asyncio
    import threading
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

    env = _fixture(start_threads=True)
    reader, writer = os.pipe()
    returned = threading.Event()
    received, aborts = [], []
    def receive():
        received.append(os.read(reader, 1))
        returned.set()
    worker = threading.Thread(target=receive, daemon=True)
    try:
        env.start()
        def abort(reason):
            aborts.append(reason)
            os.close(writer)
        env.agent._active_request_abort = abort
        worker.start()
        for elapsed in (30, 90, 150):
            env.now[0] = 100 + elapsed
            env.guard.tick(env.agent.session_id, "turn-1")
            assert not returned.wait(0.05), "elapsed time must leave the in-flight request waiting"
        assert not aborts and env.agent._pending_redirect is None
        assert not env.agent._hard_interrupt_requested.is_set()
        assert env.ctx.injected == []
        os.write(writer, b'a')
        assert returned.wait(2) and received == [b'a']
        answer = 'The slow first model response is complete.'
        async def deliver():
            consumer = GatewayStreamConsumer(env.adapter, env.source.chat_id,
                StreamConsumerConfig(transport='auto', chat_type='dm', cursor=''))
            consumer.on_delta(answer)
            consumer.finish(answer)
            await asyncio.wait_for(consumer.run(), 2)
        asyncio.run(deliver())
        env.guard.post_llm_call(session_id=env.agent.session_id, turn_id='turn-1', assistant_response=answer)
        assert env.adapter.outbound == [('send', answer)]
    finally:
        for descriptor in (writer, reader):
            try:
                os.close(descriptor)
            except OSError:
                pass
        worker.join(2)
        env.guard.close()
        env.runtime.close()


def _terminal_yield():
    import json
    from tools.terminal_tool import terminal_tool
    from tools.process_registry import process_registry

    env = _fixture()
    background = None
    try:
        # A hand's result review: the one turn that still gets the yield line at a tool boundary.
        env.start(internal=True)
        _begin_tool(env)
        result = json.loads(terminal_tool(command="sleep 0.05", timeout=30,
            task_id="front-yield-integration", session_id=env.agent.session_id,
            workdir=os.environ["TERMINAL_CWD"], _host_local=True))
        assert result.get('exit_code') == 0, result
        env.now[0] += 30
        env.guard.tick(env.agent.session_id, "turn-1")
        assert env.agent._pending_steer is None and env.agent._pending_redirect is None
        _end_tool(env)
        assert env.agent._pending_steer == env.front.YIELD_LINE
        assert not env.agent._hard_interrupt_requested.is_set()
        # A subsequent explicit handoff still uses Hermes' real background registry.
        background = json.loads(terminal_tool(command="sleep 120", background=True,
            task_id="front-background-integration", session_id=env.agent.session_id,
            workdir=os.environ["TERMINAL_CWD"], _host_local=True))
        assert background.get('session_id'), background
    finally:
        if background and background.get('session_id'):
            process_registry.kill_process(background['session_id'])
        env.guard.close()
        env.runtime.close()


def _late_binding():
    from gateway.run import _AGENT_PENDING_SENTINEL

    env = _fixture()
    try:
        env.gateway._session_state(env.key).turn.agent = _AGENT_PENDING_SENTINEL
        # A hand's result review: its boundary line is the evidence that the late binding worked.
        env.start(internal=True)
        env.gateway._session_state(env.key).turn.agent = env.agent
        _begin_tool(env)
        env.now[0] += 30
        env.guard.tick(env.agent.session_id, "turn-1")
        assert env.agent._pending_redirect is None
        _end_tool(env)
        assert env.agent._pending_steer, "first hook may precede gateway's 50ms agent tracker"
        env.agent.clear_interrupt()
        env.gateway._begin_session_run_generation(env.key)
        env.now[0] += 90
        env.guard.tick(env.agent.session_id, "turn-1")
        assert not env.agent._hard_interrupt_requested.is_set(), "old watchdog must reject a newer generation"
    finally:
        env.guard.close()
        env.runtime.close()


def _tool_directives():
    from unittest.mock import patch
    from hermes_cli.plugins import _dispatch_pre_tool_call_hooks

    env = _fixture()
    try:
        env.start()
        def invoke(name, **kwargs):
            assert name == "pre_tool_call"
            return [env.guard.pre_tool_call(**kwargs)]
        with patch("hermes_cli.lifecycle.invoke_hook", side_effect=invoke):
            block, changed = _dispatch_pre_tool_call_hooks("terminal",
                {"command": "sleep 120", "timeout": 120}, session_id=env.agent.session_id,
                turn_id="turn-1", tool_call_id="call-1")
            assert block is None
            assert changed is None, "the caller's timeout stands: the front sends no modify directive"
            for n in range(2, 9):
                block, changed = _dispatch_pre_tool_call_hooks("edit_file", {"path": "example.txt"},
                    session_id=env.agent.session_id, turn_id="turn-1", tool_call_id=f"call-{n}")
                assert block is None
            block, changed = _dispatch_pre_tool_call_hooks("edit_file", {"path": "example.txt"},
                session_id=env.agent.session_id, turn_id="turn-1", tool_call_id="call-9")
            assert block is None, "a ninth write passes: calls are counted, never capped"
            # The front still blocks once the turn must reply; that directive has to reach the dispatcher.
            env.guard._state(env.agent.session_id, "turn-1").owner_waiting = True
            block, changed = _dispatch_pre_tool_call_hooks("edit_file", {"path": "example.txt"},
                session_id=env.agent.session_id, turn_id="turn-1", tool_call_id="call-10")
            assert block, "native Hermes must recognize the front's block directive"
    finally:
        env.guard.close()
        env.runtime.close()


def _coalesced_batch():
    import asyncio

    env = _fixture()
    delivered = []
    async def ready(event):
        return True
    async def deliver(text, event, **kwargs):
        delivered.append(text)
        return True
    try:
        env.gateway._completion_delivery_identity = lambda event: event["delegation_id"]
        env.gateway._completion_identity_seen = lambda identity: False
        env.gateway._completion_delivery_ready = ready
        env.gateway._deliver_completion_notification = deliver
        accepted = asyncio.run(env.gateway._deliver_async_delegation_group_scoped(env.completion_events))
        assert accepted is True and len(delivered) == 1
        # Use the native coalescer's full output, including its exact batch header.
        env.start(internal=True, internal_text=delivered[0])
        assert env.guard.request_caps(env.agent.session_id, "turn-1")["mode"] == "review_result"
        assert env.guard.owner_started_turn(env.agent.session_id, "turn-1") is False
        env.guard.post_llm_call(session_id=env.agent.session_id, turn_id="turn-1",
                                assistant_response="All eight results are ready.")
        env.guard.on_session_end(session_id=env.agent.session_id, turn_id="turn-1",
            completed=True, failed=False, interrupted=False)
        env.start(internal=True, turn="turn-2", internal_text=delivered[0])
        assert env.guard.finalize_response(response_text="Duplicate batch.",
                                           session_id=env.agent.session_id) == "[SILENT]"
    finally:
        env.guard.close()
        env.runtime.close()


def _busy_follow_up():
    """The review is dropped for a parked owner message, whose in-band follow-up turn binds to
    it under the review's session variables, and the run's one settlement schedules the deferred review."""
    import asyncio
    from types import SimpleNamespace

    from agent.subagent_lifecycle import bind_subagent_parent
    from gateway.platforms.event import MessageEvent
    from hermes_cli import lifecycle
    from colonymind_hermes import busy_ingress

    env = _fixture()
    try:
        hooks = {"pre_gateway_dispatch": [env.guard.pre_gateway_dispatch]}
        def invoke(name, **kwargs):
            return [result for hook in hooks.get(name, []) if (result := hook(**kwargs)) is not None]
        async def ainvoke(name, **kwargs):
            return invoke(name, **kwargs)
        lifecycle.invoke_hook, lifecycle.ainvoke_hook = invoke, ainvoke
        env.gateway.config = SimpleNamespace(multiplex_profiles=False)
        assert busy_ingress.install(env.gateway) == 1
        review = env.start(internal=True)
        owner = MessageEvent(text="New owner message", source=env.source, message_id="owner-message-2")
        asyncio.run(env.adapter._handle_message_while_active(owner, env.key))
        assert env.adapter._pending_messages[env.key] is owner, "Hermes parks it behind the review"
        row = next(row for row in env.guard.inbound.values() if row.message_id == "owner-message-2")
        assert row.queued and busy_ingress.seen(owner)
        assert env.guard.defer_review_request(env.agent.session_id, "turn-1") is True
        settled = {"completed": True, "failed": False, "interrupted": False}
        env.guard.post_llm_call(session_id=env.agent.session_id, turn_id="turn-1", assistant_response="[SILENT]")
        env.guard.on_session_end(session_id=env.agent.session_id, turn_id="turn-1", **settled)
        assert env.adapter._pending_messages.pop(env.key) is owner, "the in-band drain takes it"
        # Hermes binds only the agent's ids for the follow-up; the review's session variables stay in force.
        env.agent._current_turn_id = "turn-2"
        env.agent._persist_user_message_platform_id = owner.message_id
        with bind_subagent_parent(env.agent):
            env.guard.pre_llm_call(session_id=env.agent.session_id, turn_id="turn-2", user_message=owner.text,
                                   platform="sms", sender_id=env.source.user_id,
                                   conversation_history=[{"role": "user", "content": owner.text}])
        assert env.guard.owner_started_turn(env.agent.session_id, "turn-2") is True
        assert env.guard.capture_owner_text(env.agent.session_id, "turn-2", owner.text) == owner.text
        env.guard.post_llm_call(session_id=env.agent.session_id, turn_id="turn-2", assistant_response="Owner answer")
        env.guard.on_session_end(session_id=env.agent.session_id, turn_id="turn-2", **settled)
        assert env.ctx.injected == []
        assert env.gateway._release_turn_lease(env.key, env.generation) is True
        assert env.gateway._release_turn_lease(env.key, env.generation) is False
        assert [content for content, _ in env.ctx.injected] == [review.text], "the deferred review is scheduled"
        assert env.ctx.injected[0][1]["session_key"] == env.key
    finally:
        env.guard.close()


if __name__ == "__main__":
    sys.path.insert(0, os.environ["HERMES_SOURCE_ROOT"])
    globals()["_" + sys.argv[1]]()
