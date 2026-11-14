"""Scenario stimuli exercise the real plugin and the pinned fake Hermes seams.

The fake engine in S02 models Hermes' stop contract. Its running counter is not
runtime proof about an inference server; test_clone.py requires that separately.
"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import io
import json
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import uuid

import pytest

from hands_fake_hermes import FakeContext, Parent
from .support import assert_scenario, needs_unit, paired


@paired
def test_s01_one_slot_contention(runtime_factory, monkeypatch, mutated):
    rig = runtime_factory()
    previous = rig.start(key="previous-attempt")
    rig.runtime.subagent_stop(child_session_id=previous.child_session_id,
                              child_status="completed", child_summary="First attempt complete")
    rig.native.live.clear()  # Hermes releases its slot after the stop hook returns.
    rig.native.calls.clear()
    if mutated:
        monkeypatch.setattr(rig.runtime, "_admit", lambda hand, plan, task, raw, parent, context, options:
            rig.runtime._spawn(hand, raw, parent, rig.native.native_policy(), options or {}, context))
    parents = [rig.parent, Parent("owner-b")]
    for parent in parents:
        rig.runtime.remember_parent(parent)
    paths = ["owner-a", "owner-b", "escalation", "continuation"]
    barrier = threading.Barrier(len(paths) + 1)

    def issue(index):
        barrier.wait(timeout=5)  # Fixture deadlock bound, not a product deadline.
        task = {"goal": paths[index], "kind": "deep"}
        if index < 2:
            return rig.runtime.dispatch_tasks([task], parent=parents[index], tool_call_id=paths[index])
        if index == 2:
            task.update({"continue": previous.hand_id, "effort": 3})
            return rig.runtime.dispatch_tasks([task], session_id=parents[0].session_id,
                                              tool_call_id=paths[index])
        return rig.runtime.continue_hand(previous)

    def cron():
        barrier.wait(timeout=5)
        return rig.native.async_module.dispatch_async_delegation(goal="manual cron")

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(issue, index) for index in range(4)]
        cron_future = pool.submit(cron)
        answers = [future.result() for future in futures]
        cron_future.result()
    active = [h for h in rig.runtime.registry.all() if h.status in {"running", "dispatched", "checking"}]
    assert_scenario(1, {"peak_hands": len(active), "inline": len(rig.native.inline),
                       "answers": sum(len(a["hands"]) for a in answers), "paths": paths}, mutated)


@paired
def test_s02_cancel_mid_tool(runtime_factory, monkeypatch, mutated):
    rig = runtime_factory()
    hand = rig.start()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    stream = io.BytesIO(b"model stream")
    engine = {"running": 1}
    session = SimpleNamespace(id="process-one", pid=process.pid, pid_scope="host", exited=False,
                              owner_task_id=hand.subagent_id, host_start_time=1.0)
    registry = SimpleNamespace(get=lambda ident: session if ident == session.id else None,
        kill_process=lambda *a, **kw: (process.terminate(), process.wait(timeout=5)))
    reconcile = rig.module("reconcile")
    monkeypatch.setattr(reconcile, "_process_dependencies", lambda *a: (registry, lambda pid: 1.0))
    try:
        assert reconcile.record_process(hand, {"session_id": session.id, "pid": process.pid},
                                        task_id=hand.subagent_id)

        def stop(ident):
            rig.native.stops.append(ident)
            stream.close()
            engine["running"] -= 1
            return True

        monkeypatch.setattr(rig.native, "stop", stop)
        if mutated:
            # The plugin's own cancellation of recorded process handles.
            monkeypatch.setattr(reconcile, "stop_recorded_processes", lambda hand, **kw: [])
        rig.runtime.stop(hand, "owner cancelled", calling_session=hand.parent_session_id)
        assert_scenario(2, {"process_gone": process.poll() is not None, "stream_closed": stream.closed,
                           "engine_before": 1, "engine_after": engine["running"], "state": hand.status}, mutated)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        stream.close()


@paired
def test_s05_notice_timeout_and_restart(runtime_factory, tmp_path, monkeypatch, mutated):
    rig = runtime_factory()
    notices = rig.module("notices")
    row = {"id": "ntc:one:end", "budget": "routine", "line": "Artifact checked.",
           "queued_at": "2026-01-01T00:00:00+00:00", "hand_id": "h-000000000001", "attempt": 1}
    client = SimpleNamespace(get=lambda *a, **kw: {"notices": [row]})
    ids, lines, receipts = [], {}, []

    def sink(*, delivery_id, notices):
        ids.append(delivery_id)
        lines.setdefault(delivery_id, list(notices))
        if len(ids) == 1:
            raise TimeoutError("platform sent; acknowledgement lost")
        return {"platform_message_id": "receipt-one"}

    if mutated:
        deliver = notices.NoticeEmitter._deliver
        monkeypatch.setattr(notices.NoticeEmitter, "_deliver", lambda self, db, batch:
            deliver(self, db, {**dict(batch), "id": "new-" + uuid.uuid4().hex}))
    options = dict(client=client, emit=lambda path, body, key: receipts.append(body), sink=sink,
                   ledger_path=tmp_path / "notice.sqlite3", now=lambda: 1767225661)
    notices.NoticeEmitter(**options).pump()
    notices.NoticeEmitter(**options).pump()
    assert_scenario(5, {"lines": len(lines), "delivery_ids": ids, "attempts": len(ids),
                       "receipt": receipts[0]["platform_message_id"] if receipts else None}, mutated)


@paired
def test_s08_engine_failure_contract_and_audio(runtime_factory, monkeypatch, mutated):
    rig = runtime_factory()
    audio = deepcopy(rig.runtime.config["contracts"]["text-route"])
    audio.update(family="audio-family", modalities=["text", "audio"])
    rig.runtime.config["contracts"]["audio-route"] = audio
    rig.native.native_policy = lambda: {"provider": "audio-route", "model": "audio-model",
        "fallback_providers": [{"provider": "unknown-route", "model": "invalid-model"},
                               {"provider": "text-route", "model": "text-model"}]}
    hand = rig.start()
    mw = rig.module("middleware")
    payload = {"messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": "fixture", "format": "wav"}},
        {"type": "text", "text": "transcript"}]}]}
    mw.llm_request(rig.runtime, payload, session_id=hand.child_session_id, provider="audio-route", api_request_id="a")
    mw.post_api_request(rig.runtime, session_id=hand.child_session_id, provider="audio-route",
                        model="audio-model", api_request_id="a", usage={"total_tokens": 1})
    mw.api_request_error(rig.runtime, session_id=hand.child_session_id, api_request_id="outage")
    if mutated:
        monkeypatch.setattr(mw, "_strip_audio", lambda value: (value, 0))
    response = mw.llm_request(rig.runtime, payload, session_id=hand.child_session_id,
                              provider="text-route", api_request_id="b")
    served = []
    mw.llm_execution(rig.runtime, response["request"], served.append,
                     session_id=hand.child_session_id, api_request_id="b")
    mw.post_api_request(rig.runtime, session_id=hand.child_session_id, provider="text-route",
                        model="text-model", api_request_id="b", usage={"total_tokens": 1})
    chain = rig.native.calls[0]["credentials_cfg"]["fallback_providers"]
    assert_scenario(8, {"fallback_valid": all(item["provider"] == "text-route" for item in chain) and bool(chain),
        "families": sorted(hand.families_used), "audio_parts": json.dumps(served).count('"input_audio"'),
        "fallback_called": bool(served)}, mutated)


@paired
def test_s09_sidecar_down_spool_replays_once(runtime_factory, monkeypatch, mutated):
    rig = runtime_factory(max_running=4)
    rig.client.down = True
    first = rig.runtime.dispatch_tasks([{"goal": "first"}], parent=rig.parent, tool_call_id="first")
    assert first["hands"][0]["reason"] == "controller_unavailable"
    hand = rig.start(key="second")
    own = rig.runtime.dispatch_tasks([{"goal": "initiative", "priority": "own"}], parent=rig.parent,
                                    tool_call_id="own")
    owner = rig.runtime.dispatch_tasks([{"goal": "other owner work"}], parent=rig.parent, tool_call_id="third")
    assert owner["hands"][0]["reason"] == "slots"
    degraded = rig.runtime.degraded.is_degraded
    rig.runtime.subagent_stop(child_session_id=hand.child_session_id, child_status="completed", child_summary="result")
    assert rig.runtime.spool.pending()
    if mutated:
        # Disable durable acknowledgement in the production spool. A second
        # maintenance pass must still be distinguishable from a successful drain.
        monkeypatch.setattr(rig.runtime.spool, "acknowledge", lambda *a, **kw: None)
    rig.client.down = False
    rig.runtime.maintain()
    rig.runtime.maintain()
    keys = [key for _, _, key in rig.client.writes]
    assert_scenario(9, {"degraded": degraded, "peak_hands": len(rig.native.calls),
        "own_refused": own["hands"][0]["reason"] == "degraded_owner_only",
        "pending": len(rig.runtime.spool.pending()), "duplicates": len(keys) - len(set(keys)),
        "executions": len(rig.native.calls)}, mutated)


@paired
@pytest.mark.parametrize('arrival_phase', ['in_flight', 'before_request'])
def test_s10_owner_arrives_during_review(runtime_factory, monkeypatch, mutated, arrival_phase):
    """Busy mode queue: Hermes parks an arrival for a running session in the
    adapter's pending slot and runs pre_gateway_dispatch only when that queued
    event is processed after the turn (gateway/run_busy.py, pinned source). The
    arrival here takes that path; firing the hook at arrival would test a seam
    Hermes does not have. The local baseline and elapsed time end at the owner's
    pre_llm_call turn-start hook. Actual gateway scheduling needs clone replay.
    """
    rig = runtime_factory()
    front, dispatch = rig.module("front"), rig.module("dispatch")
    monkeypatch.setattr(dispatch, "_RUNTIME", rig.runtime)
    ctx = FakeContext()
    front.register(ctx)
    redirected = []
    interrupted = threading.Event()
    starts = []

    def observe_start(turn_id, **_):
        starts.append((turn_id, time.monotonic()))

    ctx.hooks['pre_llm_call'].insert(0, observe_start)
    session, turn, session_key = rig.parent.session_id, 'review-turn', 'owner-route'
    def native_redirect(text):
        if arrival_phase == 'before_request':
            return False
        redirected.append(text)
        return True

    agent = SimpleNamespace(session_id=session, _current_turn_id=turn, redirect=native_redirect,
                            hard_interrupt=interrupted.set)
    adapter = SimpleNamespace(_pending_messages={})
    # Generation is a fixture assumption; the watchdog checks its exact binding.
    state = SimpleNamespace(persistent=SimpleNamespace(run_generation=1),
                            conversation=SimpleNamespace(queued_events=[]))
    gateway = SimpleNamespace(_running_agents={session_key: agent}, adapters={"sms": adapter},
        _peek_session_state=lambda key: state if key == session_key else None,
        _is_session_run_current=lambda key, generation: key == session_key and generation == state.persistent.run_generation)
    rig.runtime.gateway = gateway
    guard = rig.runtime.ingress
    guard._completion_texts = lambda state: {'h-completed': 'Native completion'}
    owner_text = 'Owner asks for a status'
    source = SimpleNamespace(platform='sms', user_id='5550100', chat_id='5550100')

    def owner_event(message_id):
        return SimpleNamespace(internal=False, source=source, text=owner_text,
            session_id=session, session_key=session_key, message_id=message_id)

    def start_owner(event, owner_turn):
        agent._current_turn_id = owner_turn
        ctx.fire('pre_gateway_dispatch', event=event, gateway=gateway)
        ctx.fire('pre_llm_call', session_id=session, turn_id=owner_turn, user_message=owner_text,
            platform='sms', sender_id='5550100', chat_id='5550100', session_key=session_key,
            message_id=event.message_id)

    # Measure this fake gateway's no-review turn starts in the same session.
    # The clone driver independently measures at least three samples on Hermes.
    baseline = []
    for index in range(3):
        sample_turn = f'owner-baseline-{index}'
        event = owner_event(sample_turn)
        arrived = time.monotonic()
        start_owner(event, sample_turn)
        baseline.append(starts[-1][1] - arrived)
        guard.agent_loop_stopped(session_id=session, turn_id=sample_turn)
    agent._current_turn_id = turn
    if mutated:
        monkeypatch.setattr(guard, '_owner_waits', lambda state: False)
    ctx.fire('pre_llm_call', session_id=session, turn_id=turn, user_message='Native completion',
        platform='sms', sender_id='5550100', chat_id='5550100', session_key=session_key,
        conversation_history=[{'role': 'user', 'content': 'Native completion', 'display_kind': 'internal_notification'}])
    assert guard._state(session, turn).mode == 'review_result'
    review_start = len(starts) - 1
    event = owner_event('owner-after-review')
    arrived = time.monotonic()
    adapter._pending_messages[session_key] = event  # Hermes' queue-mode enqueue, no hook
    event._gateway_accepted = True
    if arrival_phase == 'before_request':
        middleware = rig.module('middleware')
        payload = {'messages': [{'role': 'user', 'content': 'Native completion'}]}
        envelope = middleware.llm_request(rig.runtime, payload, session_id=session, turn_id=turn)
        sent = []
        response = middleware.llm_execution(rig.runtime, envelope['request'],
            lambda request: sent.append(request) or 'Provider response', session_id=session, turn_id=turn)
        assert len(sent) == int(mutated), 'a dropped review must not reach the transport'
        assert response is not None
        assert guard._state(session, turn).request_dropped is (not mutated)
        assert not redirected, 'Hermes refuses a redirect before a request is in flight'
    else:
        # The current watchdog cancels only a review's own provider request.
        # Exercise its execution middleware so model_waiting has the same
        # lifetime as the native call instead of pretending a request is active.
        def provider(request):
            assert guard._state(session, turn).model_waiting
            interrupted.wait(2.5)  # Fixture deadlock bound.
            return 'Provider response'

        guard.llm_execution({}, provider, session_id=session, turn_id=turn)
        assert not guard._state(session, turn).model_waiting
        assert interrupted.is_set() is (not mutated)
        assert not redirected
    request_dropped = guard._state(session, turn).request_dropped
    if interrupted.is_set() or arrival_phase == 'before_request':
        guard.agent_loop_stopped(session_id=session, turn_id=turn)
        start_owner(adapter._pending_messages.pop(session_key), 'owner-after-review')
    following = starts[review_start + 1:]
    elapsed = (following[0][1] if following else time.monotonic()) - arrived
    guard.close()
    assert_scenario(10, {"next_turn": "owner" if following and following[0][0] == 'owner-after-review' else "review",
                        "elapsed": elapsed, "redirected": bool(redirected), "request_dropped": request_dropped,
                        "baseline": baseline}, mutated)


@paired
def test_s11_blocking_question_stops_tools(runtime_factory, monkeypatch, mutated):
    rig = runtime_factory()
    hand = rig.start()
    tools, budgets = rig.module("tools"), rig.module("budgets")
    answer = tools.execute_ask({"question": "Which artifact?", "blocking": True}, runtime=rig.runtime,
                               session_id=hand.child_session_id, request_id="blocking-question")
    assert answer["blocking"] is True
    if mutated:
        monkeypatch.setattr(budgets, "pre_tool_call", lambda *a, **kw: None)
    verdict = budgets.pre_tool_call(rig.runtime, "web_search", {"query": "artifact"}, hand.child_session_id)
    blocked = isinstance(verdict, dict) and verdict.get("action") == "block"
    assert_scenario(11, {"activity": hand.activity, "executed": 0 if blocked else 1, "blocked": blocked}, mutated)


@paired
def test_s14_unsupported_platform_never_inline(runtime_factory, monkeypatch, mutated):
    rig = runtime_factory()
    if mutated:
        rig.runtime.config["platforms"] += ["rcs", "api", "voice", "unknown"]
    reasons, refused = [], []
    for platform in ("rcs", "api", "voice", "unknown"):
        rig.native.platform = platform
        result = rig.runtime.dispatch_tasks([{"goal": "check artifact"}], parent=rig.parent, tool_call_id=platform)
        answer = result["hands"][0]
        reasons.append(answer.get("reason"))
        if answer["status"] == "refused":
            refused.append(platform)
    assert_scenario(14, {"refused": refused, "reasons": reasons, "dispatches": len(rig.native.calls),
        "inline": len(rig.native.inline), "front_continued": rig.runtime.authorized_owner(rig.parent.session_id)}, mutated)
