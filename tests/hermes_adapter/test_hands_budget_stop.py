"""Budget stops end with the hand's report: system stops, one final call, cache-aware accounting.

Cached prompt tokens do not exhaust the research budget as if they were new input.
A budget stop on a delegate thread bypasses owner-session lookup and allows one
final model call, so the hand can return its report.
"""
from copy import deepcopy
import importlib
import json
import threading
from types import SimpleNamespace

import pytest

from hands_fake_hermes import FakeController, FakeHermes, Parent, load_hands
from test_hands_guards import guards, hand  # noqa: F401

DELEGATE = threading.local()
IDS = dict(provider="text-route", model="text-model", api_mode="chat_completions", platform="subagent")
TOOLS = [{"type": "function", "function": {"name": "web_search", "parameters": {}}}]


def owner_check(sid):
    # A delegate thread cannot resolve the owner's session.
    return sid.startswith("owner") and not getattr(DELEGATE, "active", False)


def in_delegate(fn):
    """Run fn on a thread where, as in a delegated child, the owner session does not resolve."""
    outcome = {}

    def run():
        DELEGATE.active = True
        try:
            outcome["value"] = fn()
        except BaseException as error:  # noqa: BLE001 - the test re-raises it on its own thread
            outcome["error"] = error

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(30)
    assert not thread.is_alive()
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


@pytest.fixture
def rig(tmp_path):
    package = load_hands()
    module = lambda name: importlib.import_module(package.__name__ + "." + name)
    native, client = FakeHermes(), FakeController()
    config = {"max_running": 3, "platforms": ["sms"], "contracts": {"text-route": {
        "version": "1", "family": "text", "modalities": ["text"], "context_tokens": 100000,
        "max_output_tokens": 2000, "tool_calls": True, "decode_tok_s": 100}}}
    runtime = module("dispatch").Runtime(config=config, client=client, spool_dir=tmp_path / "spool",
                                          native=native, owner_check=owner_check)
    native.runtime = runtime
    runtime.install()
    package.register_route_resolver(lambda role, effort: native.native_policy())
    native.delegate_module.delegate_task(tasks=[{"goal": "Research the question", "kind": "research"}],
                                         parent_agent=Parent())
    state = runtime.registry.all()[0]
    yield SimpleNamespace(runtime=runtime, native=native, client=client, hand=state,
                          middleware=module("middleware"), budgets=module("budgets"))
    runtime.close()


def summary(prompt, cached, output):
    """Hermes' post_api_request usage: agent.api_request_hooks._usage_summary_for_api_request_hook."""
    return {"input_tokens": prompt - cached, "output_tokens": output, "cache_read_tokens": cached,
            "cache_write_tokens": 0, "reasoning_tokens": 0, "request_count": 1,
            "prompt_tokens": prompt, "total_tokens": prompt + output}


# The incident's shape: 11 calls, context 15.5K to 56K, 68 to 98 percent prefix cache hits.
PROMPTS = (15500, 19500, 23500, 27500, 31500, 35500, 39500, 43500, 47500, 51500, 56000)
HITS = (0.68, 0.90, 0.93, 0.95, 0.96, 0.97, 0.97, 0.98, 0.98, 0.98, 0.98)
INCIDENT = [summary(prompt, round(prompt * hit), 4100) for prompt, hit in zip(PROMPTS, HITS)]


def model_call(rig, request_id, usage=None, content="", raises=None):
    """One native model call through the hand middleware, as Hermes' turn loop makes it."""
    state = rig.hand
    ids = dict(IDS, session_id=state.child_session_id, task_id=state.subagent_id, api_request_id=request_id)
    request = {"messages": [{"role": "user", "content": "work"}], "tools": deepcopy(TOOLS),
               "tool_choice": "auto", "max_tokens": 32768}
    envelope = rig.middleware.llm_request(rig.runtime, request=request, **ids)
    sent = []

    def provider(payload):
        sent.append(payload)
        if raises is not None:
            raise raises
        return SimpleNamespace(id=request_id, model="text-model", usage=None, choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=None), finish_reason="stop")])

    response = rig.middleware.llm_execution(rig.runtime, envelope["request"], provider, **ids)
    if response is not None:
        text = response.choices[0].message.content
        rig.middleware.post_api_request(rig.runtime, usage=usage if sent else None,
                                        assistant_message=SimpleNamespace(content=text), **ids)
    return envelope["request"], sent, response


def tool_call(rig, call_id):
    state = rig.hand
    return rig.budgets.pre_tool_call(rig.runtime, tool_name="web_search", args={"query": call_id},
                                     session_id=state.child_session_id, tool_call_id=call_id,
                                     task_id=state.subagent_id)


def writes(rig, action):
    return [body for path, body, _ in rig.client.writes if path.endswith("/" + action)]


def test_incident_context_is_charged_for_uncached_prompt_and_output(rig):
    def child():
        assert not rig.runtime.authorized_owner(rig.hand.parent_session_id)
        for index, usage in enumerate(INCIDENT):
            _, sent, _ = model_call(rig, f"r{index}", usage, content="searching")
            assert len(sent) == 1
            assert tool_call(rig, f"call-{index}") is None

    in_delegate(child)
    # The old count (raw) crossed the 400K research budget; what the hand cost did not.
    assert rig.hand.raw_tokens == sum(u["total_tokens"] for u in INCIDENT) == 436100
    assert rig.hand.tokens == sum(u["input_tokens"] + u["output_tokens"] for u in INCIDENT) == 62510
    assert not rig.hand.stop_reason and not rig.native.stops and not writes(rig, "stop")


@pytest.mark.parametrize("cache", [True, False], ids=["cached-context", "no-cache-fields"])
def test_incident_budget_stop_from_delegate_ends_with_one_final_report(rig, caplog, cache):
    usages = INCIDENT if cache else [{"prompt_tokens": u["prompt_tokens"], "completion_tokens": 4100}
                                     for u in INCIDENT]
    if cache:
        rig.hand.plan["budgets"]["tokens"] = 60000  # crossed by the charged count at call 11
    report = "Outcome: partial. Found three sources; the fourth was not checked."
    seen = {}

    def child():
        assert not rig.runtime.authorized_owner(rig.hand.parent_session_id)
        for index, usage in enumerate(usages):
            model_call(rig, f"r{index}", usage, content="searching")
            if rig.hand.stop_reason:
                break
            assert tool_call(rig, f"call-{index}") is None
        seen["stopped_at"] = index
        seen["refusal"] = tool_call(rig, "after-stop")
        seen["final"] = model_call(rig, "final", {"prompt_tokens": 60000, "completion_tokens": 900}, report)
        seen["later"] = model_call(rig, "later", {"prompt_tokens": 1, "completion_tokens": 1})

    in_delegate(child)
    state = rig.hand
    assert seen["stopped_at"] == 10 and state.stop_reason == "token_budget"
    assert "hand API accounting failed" not in caplog.text and "accounting_unavailable" not in caplog.text
    assert writes(rig, "stop") == [{"reason": "token_budget", "epoch": rig.runtime.epoch}]
    # The executor keeps running for its report: no interrupt, tools refused with the report prompt.
    assert rig.native.stops == []
    assert seen["refusal"]["action"] == "block" and "write the final report now" in seen["refusal"]["message"]
    request, sent, response = seen["final"]
    assert len(sent) == 1 and "tools" not in sent[0] and "tool_choice" not in sent[0]
    assert sent[0]["max_tokens"] == 8192
    assert response.choices[0].message.content == report
    # Later calls never reach the provider; the native loop ends on the report.
    _, later_sent, later = seen["later"]
    assert later_sent == [] and later is not None
    assert later.choices[0].message.content == report and later.usage is None
    rig.runtime.subagent_stop(child_session_id=state.child_session_id, child_status="completed",
                              child_summary=report, parent_session_id=state.parent_session_id)
    result, = writes(rig, "result")
    assert result["status"] == "partial" and result["summary_text"] == report
    usage = next(event["payload"] for body in writes(rig, "events") for event in body["events"]
                 if event["type"] == "usage")
    assert usage["tokens"] == state.tokens and usage["raw_tokens"] == state.raw_tokens
    assert state.raw_tokens >= state.tokens
    # This fake controller answers without an outcome; the real one keeps the hand stopped (sidecar test).
    assert state.status == "done"


def test_accounting_failure_stops_cleanly_without_a_none_loop(rig):
    def child():
        model_call(rig, "bad", {"total_tokens": "unreadable"}, content="searching")
        return model_call(rig, "after", {"prompt_tokens": 1, "completion_tokens": 1})

    _, sent, response = in_delegate(child)
    state = rig.hand
    assert state.stop_reason == "accounting_unavailable"
    assert writes(rig, "stop") == [{"reason": "accounting_unavailable", "epoch": rig.runtime.epoch}]
    assert rig.native.stops == [state.subagent_id] and state.status == "stopped"
    # Even if the interrupt is not honoured, the next call ends locally with the reason.
    assert sent == [] and response is not None
    assert "token accounting was unavailable" in response.choices[0].message.content
    rig.runtime.subagent_stop(child_session_id=state.child_session_id, child_status="completed",
                              child_summary=response.choices[0].message.content,
                              parent_session_id=state.parent_session_id)
    assert writes(rig, "result")[0]["status"] == "stopped"


def test_owner_requested_stops_and_steers_still_require_the_owner_session(rig):
    state = rig.hand

    def child():
        with pytest.raises(PermissionError):
            rig.runtime.stop(state, "stop")
        with pytest.raises(PermissionError):
            rig.runtime.steer(state, "change")

    in_delegate(child)
    with pytest.raises(PermissionError):
        rig.runtime.stop(state, "stop", calling_session="owner-other")
    with pytest.raises(PermissionError):
        rig.runtime.steer(state, "change", calling_session="owner-other")
    assert not rig.native.stops and not rig.native.steers and not writes(rig, "stop")
    # The runtime's own path needs no owner session and keeps the stop's effects.
    in_delegate(lambda: rig.runtime.system_stop(state, "wall_budget"))
    assert rig.native.stops == [state.subagent_id] and state.status == "stopped" and state.blocked
    assert writes(rig, "stop") == [{"reason": "wall_budget", "epoch": rig.runtime.epoch}]


def test_clock_steer_and_wall_stop_work_from_a_delegate_thread(rig):
    state = rig.hand
    # Keep the runtime's real clock nearby, but make exact boundary arithmetic
    # independent of the fractional monotonic timestamp assigned at dispatch.
    start = state.started_at = float(int(state.started_at))
    in_delegate(lambda: rig.budgets.tick(rig.runtime, now=start + 959))
    assert not rig.native.steers and not rig.native.stops
    in_delegate(lambda: rig.budgets.tick(rig.runtime, now=start + 960))
    assert rig.native.steers == [(state.subagent_id, "wrap up and report")]
    in_delegate(lambda: rig.budgets.tick(rig.runtime, now=start + 1199))
    assert not rig.native.stops
    assert rig.native.steers == [(state.subagent_id, "wrap up and report")]
    in_delegate(lambda: rig.budgets.tick(rig.runtime, now=start + 1200))
    assert state.stop_reason == "wall_budget" and rig.native.stops == [state.subagent_id]
    assert state.final_report == ""


@pytest.mark.parametrize("usage,charged,raw", [
    (summary(20000, 18000, 500), 2500, 20500),
    ({"prompt_tokens": 20000, "completion_tokens": 500, "total_tokens": 20500,
      "prompt_tokens_details": {"cached_tokens": 18000}}, 2500, 20500),
    ({"input_tokens": 20000, "output_tokens": 500, "total_tokens": 20500,
      "input_tokens_details": {"cached_tokens": 18000}}, 2500, 20500),
    ({"input_tokens": 2000, "cache_read_input_tokens": 18000, "cache_creation_input_tokens": 100,
      "output_tokens": 500}, 2600, 20600),
    ({"prompt_tokens": 20000, "prompt_cache_hit_tokens": 18000, "completion_tokens": 500}, 2500, 20500),
    ({"prompt_tokens": 20000, "cached_tokens": 18000, "completion_tokens": 500}, 2500, 20500),
    ({"input_tokens": 2000, "cache_read_tokens": 18000, "output_tokens": 500}, 2500, 20500),
    ({"prompt_tokens": 20000, "completion_tokens": 500}, 20500, 20500),
    ({"prompt_tokens": 20000, "completion_tokens": 500, "prompt_tokens_details": None}, 20500, 20500),
    ({"prompt_tokens": 100, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 400}}, 5, 105),
    ({"total_tokens": 7}, 7, 7),
    (None, 0, 0),
], ids=["hermes", "chat", "responses", "anthropic", "deepseek", "kimi", "hermes-no-prompt", "no-cache",
        "null-details", "cache-clamped", "total-only", "none"])
def test_usage_shapes_charge_uncached_prompt_and_keep_raw(guards, usage, charged, raw):
    assert guards.middleware._usage_tokens(usage) == (charged, raw)
    state = hand(guards)
    guards.middleware.post_api_request(guards.runtime, session_id="child-one", api_request_id="r1", usage=usage)
    assert (state.tokens, state.raw_tokens) == (charged, raw)


def test_malformed_usage_still_stops_for_accounting(guards):
    state = hand(guards)
    guards.middleware.post_api_request(guards.runtime, session_id="child-one", api_request_id="r1",
                                       usage={"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": "x"}})
    assert state.stop_reason == "accounting_unavailable"
    assert guards.runtime.interrupts == [(state.hand_id, "accounting_unavailable")]


def call(guards, request_id, provider, request=None):
    ids = dict(IDS, session_id="child-one", api_request_id=request_id)
    envelope = guards.middleware.llm_request(guards.runtime, request=request or {
        "messages": [], "tools": deepcopy(TOOLS), "tool_choice": "auto", "max_tokens": 32768}, **ids)
    return envelope["request"], guards.middleware.llm_execution(guards.runtime, envelope["request"], provider, **ids)


def reply(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None),
                                                    finish_reason="stop")], usage=None)


def test_exactly_one_final_call_and_later_calls_are_refused(guards):
    state = hand(guards)
    guards.budgets.stop(guards.runtime, state, "token_budget")
    assert guards.runtime.stops == [(state.hand_id, "token_budget")] and guards.runtime.interrupts == []
    provider = []
    failing = lambda payload: (provider.append(payload), (_ for _ in ()).throw(ConnectionError("reset")))
    with pytest.raises(ConnectionError):
        call(guards, "final-1", failing)
    # A provider error before any reply leaves the call to Hermes' own bounded retry.
    request, response = call(guards, "final-2", lambda payload: provider.append(payload) or reply("REPORT"))
    assert len(provider) == 2 and "tools" not in request and request["max_tokens"] == 8192
    assert state.final_report == "used" and response.choices[0].message.content == "REPORT"
    guards.middleware.post_api_request(guards.runtime, session_id="child-one", api_request_id="final-2",
        usage={"prompt_tokens": 10, "completion_tokens": 5}, assistant_message=SimpleNamespace(content="REPORT"))
    for later in ("later-1", "later-2"):
        request, response = call(guards, later, lambda payload: pytest.fail("a later call reached the provider"))
        assert "tools" in request and response.choices[0].message.content == "REPORT"
        guards.middleware.post_api_request(guards.runtime, session_id="child-one", api_request_id=later,
                                           usage=None, assistant_message=SimpleNamespace(content="REPORT"))
    assert (state.tokens, state.raw_tokens) == (15, 15)
    assert state.local_requests == {"later-1", "later-2"}


def test_final_report_deadline_interrupts_the_executor(guards, monkeypatch):
    monkeypatch.setattr(guards.budgets, "time", SimpleNamespace(monotonic=lambda: 1000.0))
    state = hand(guards)
    guards.budgets.stop(guards.runtime, state, "token_budget")
    state.status = "stopped"  # the runtime's system stop sets it; this fake runtime records only
    since = state.final_report_since
    guards.budgets.tick(guards.runtime, now=since + 299)
    assert guards.runtime.interrupts == []
    guards.budgets.tick(guards.runtime, now=since + 300)
    assert guards.runtime.interrupts == [(state.hand_id, "token_budget")] and state.final_report == "expired"
    _, response = call(guards, "late", lambda payload: pytest.fail("an expired report reached the provider"))
    assert "token budget was spent" in response.choices[0].message.content


def test_recorded_result_ends_the_final_report_clock(guards):
    state = hand(guards)
    guards.budgets.stop(guards.runtime, state, "token_budget")
    state.status, state.plan["_result_submitted"] = "done", True
    guards.budgets.tick(guards.runtime, now=state.final_report_since + 10000)
    assert guards.runtime.interrupts == []


def test_wall_budget_interrupts_at_once_and_refuses_locally(guards):
    state = hand(guards)
    guards.budgets.tick(guards.runtime, now=state.started_at + 1201)  # past the 1200 s cap whatever the rounding
    assert guards.runtime.interrupts == [(state.hand_id, "wall_budget")] and state.final_report == ""
    request, response = call(guards, "after", lambda payload: pytest.fail("a stopped hand reached the provider"))
    assert response.choices[0].message.content.startswith("This hand ended because its time budget was spent")


@pytest.mark.parametrize("payload,expected", [
    ({"messages": [], "max_tokens": 32768}, {"max_tokens": 8192}),
    ({"messages": [], "max_tokens": 1000}, {"max_tokens": 1000}),
    ({"messages": []}, {"max_tokens": 8192}),
    ({"messages": [], "max_completion_tokens": 50000, "extra_body": {"max_tokens": 9000}},
     {"max_completion_tokens": 8192, "extra_body": {"max_tokens": 8192}}),
    ({"input": [], "max_output_tokens": 64000}, {"max_output_tokens": 8192}),
    ({"input": []}, {"max_output_tokens": 8192}),
    ({"modelId": "m", "messages": [], "inferenceConfig": {"maxTokens": 16000}}, {"inferenceConfig": {"maxTokens": 8192}}),
    ({"modelId": "m", "messages": []}, {}),
])
def test_final_report_output_is_bounded_in_each_wire_shape(guards, payload, expected):
    state = hand(guards)
    guards.budgets.stop(guards.runtime, state, "token_budget")
    envelope = guards.middleware.llm_request(guards.runtime, request=deepcopy(payload), session_id="child-one",
                                             api_request_id="final", provider="local")
    sent = envelope["request"]
    for key, value in expected.items():
        assert sent[key] == value
    if not expected:
        assert set(sent) == set(payload)
    assert json.dumps(sent)  # still a plain request envelope
