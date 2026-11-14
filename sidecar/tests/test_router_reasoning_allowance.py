"""A caller's ``max_output_tokens`` is its answer's budget; a binding whose model reasons first is sent more.

A local OpenAI-compatible endpoint over real HTTP stands in for a model that spends output tokens thinking
before its answer, as vLLM does with ``chat_template_kwargs.enable_thinking``: when ``max_tokens`` runs out
before the answer is written, the completion ends with ``finish_reason: length`` and no final content.
"""
from __future__ import annotations

import json
import math
import os
import time

import pytest

from model_endpoint import ModelEndpoint
from colonymind.router.functions import DEFAULT_REASONING_TOKENS, PROMPT_MARGIN, request_output_tokens
from colonymind.router.router import LLMRouter
from colonymind.router.tokens import estimate_tokens
from colonymind.usage import charged_tokens, function_usage

THINKING = {"chat_template_kwargs": {"enable_thinking": True}}


class ThinkingModel:
    """Answers ``/v1/chat/completions`` after ``think`` tokens of reasoning, inside the request's ``max_tokens``.

    ``answers`` maps a response-schema name (or None, for a request without one) to a JSON object or to a
    callable of the request body. Each reply reports its usage: ``prompt`` input tokens and what it generated.
    """

    def __init__(self, *, think=600, prompt=200, answers=None, thinks=None):
        self.think, self.prompt, self.answers = think, prompt, dict(answers or {})
        self.reported = []      # each reply's total_tokens
        # Whether a request makes the model think; by default only one that turns thinking on.
        self.thinks = thinks or (lambda body: (body.get("chat_template_kwargs") or {}).get("enable_thinking") is True)

    def __call__(self, path, body):
        name = ((body.get("response_format") or {}).get("json_schema") or {}).get("name")
        answer = self.answers.get(name, {"answer": "neutral"})
        text = json.dumps(answer(body) if callable(answer) else answer)
        think = self.think if self.thinks(body) else 0
        answer_tokens = max(1, len(text) // 4)
        cap = body.get("max_tokens")
        if cap is not None and think + answer_tokens > cap:
            # The cap ran out while the model was still thinking: no final answer.
            message = {"role": "assistant", "content": None, "reasoning_content": "neutral scratch " * 8}
            choice, completion = {"index": 0, "message": message, "finish_reason": "length"}, cap
        else:
            message = {"role": "assistant", "content": text}
            if think:
                message["reasoning_content"] = "neutral scratch " * 8
            choice, completion = {"index": 0, "message": message, "finish_reason": "stop"}, think + answer_tokens
        self.reported.append(self.prompt + completion)
        return 200, {"id": "neutral", "object": "chat.completion", "created": 1, "model": body["model"],
                     "choices": [choice], "usage": {"prompt_tokens": self.prompt, "completion_tokens": completion,
                                                    "total_tokens": self.prompt + completion}}


def pool_config(url, **deliberate):
    """``deliberate`` thinks (its request turns thinking on); ``interactive`` does not."""
    return {"provider": "vllm", "apiKey": "neutral-key", "models": {},
            "modelPool": {
                "interactive": {"model": "fast-neutral", "baseUrl": url + "/v1", "contextTokens": 32000,
                                "maxTokens": 4096, "supportsJsonSchema": True},
                "deliberate": {"model": "strong-neutral", "baseUrl": url + "/v1", "contextTokens": 128000,
                               "maxTokens": 8192, "supportsJsonSchema": True, "extraBody": THINKING, **deliberate},
            },
            "functionRoles": {"reasoning": {"candidates": ["deliberate"], "timeoutSeconds": 10, "deadlineSeconds": 20},
                              "extraction": {"candidates": ["interactive"], "timeoutSeconds": 10, "deadlineSeconds": 20},
                              "chat": ["interactive"], "planning": ["deliberate"], "judging": ["deliberate"],
                              "coding": ["deliberate"], "vision": []}}


def make_router(cfg):
    router = LLMRouter(tiers={})
    router.configure(cfg)
    return router


SCHEMA = {"name": "neutral_relation", "schema": {"type": "object", "properties": {"relation": {"type": "string"}},
                                                 "required": ["relation"], "additionalProperties": False}}


async def ask(router, role, cap=None, **context):
    return await router.complete([{"role": "user", "content": "Neutral routing fixture."}],
                                 context={"function_role": role, "allow_fallback": False, "response_schema": SCHEMA,
                                          **({"max_output_tokens": cap} if cap is not None else {}), **context})


@pytest.mark.asyncio
async def test_a_thinking_binding_gets_the_answer_budget_plus_its_allowance_and_answers():
    model = ThinkingModel(think=600, answers={"neutral_relation": {"relation": "adds"}})
    with ModelEndpoint(model) as endpoint:
        router = make_router(pool_config(endpoint.url))
        response = await ask(router, "reasoning", 40)
        assert json.loads(response.content) == {"relation": "adds"} and response.binding == "deliberate"
        sent = endpoint.requests[-1][1]
        assert sent["max_tokens"] == 40 + DEFAULT_REASONING_TOKENS == response.client_max_tokens
        assert sent["chat_template_kwargs"] == {"enable_thinking": True}      # thinking stays on
        # The same cap the router reports to a caller that sizes its own budget.
        assert router.function_output_tokens(context={"function_role": "reasoning", "max_output_tokens": 40}) \
            == 40 + DEFAULT_REASONING_TOKENS
        assert router.routing_status()["models"]["deliberate"]["reasoning_tokens"] == DEFAULT_REASONING_TOKENS


@pytest.mark.asyncio
async def test_a_binding_that_does_not_think_gets_exactly_the_answer_budget():
    model = ThinkingModel(answers={"neutral_relation": {"relation": "adds"}})
    with ModelEndpoint(model) as endpoint:
        router = make_router(pool_config(endpoint.url))
        response = await ask(router, "extraction", 40)
        assert response.binding == "interactive" and endpoint.requests[-1][1]["max_tokens"] == 40
        assert "chat_template_kwargs" not in endpoint.requests[-1][1]
        assert router.function_output_tokens(context={"function_role": "extraction", "max_output_tokens": 40}) == 40
        assert router.routing_status()["models"]["interactive"]["reasoning_tokens"] == 0
        # Thinking explicitly off is not thinking.
        cfg = pool_config(endpoint.url, extraBody={"chat_template_kwargs": {"enable_thinking": False}})
        off = make_router(cfg)
        await ask(off, "reasoning", 40)
        assert endpoint.requests[-1][1]["max_tokens"] == 40


@pytest.mark.asyncio
async def test_without_the_allowance_the_thinking_model_runs_out_and_the_answer_is_still_incomplete():
    """``reasoningTokens: 0`` sends exactly the caller's cap: the night's failure, kept as an incomplete
    attempt (never a reasoning-only answer), so a still-truncated answer keeps its existing semantics."""
    model = ThinkingModel(think=600, answers={"neutral_relation": {"relation": "adds"}})
    with ModelEndpoint(model) as endpoint:
        router = make_router(pool_config(endpoint.url, reasoningTokens=0))
        with pytest.raises(RuntimeError, match="No eligible local model completed function reasoning") as raised:
            await ask(router, "reasoning", 40)
        assert "incomplete_final_answer" in str(raised.value) and "scratch" not in str(raised.value)
        assert endpoint.requests[-1][1]["max_tokens"] == 40


@pytest.mark.asyncio
async def test_an_answer_still_truncated_after_the_allowance_is_incomplete_and_may_fall_back():
    model = ThinkingModel(think=9000, answers={"neutral_relation": {"relation": "adds"}},
                          thinks=lambda body: body["model"] == "strong-neutral")
    with ModelEndpoint(model) as endpoint:
        cfg = pool_config(endpoint.url)
        cfg["functionRoles"]["reasoning"]["candidates"] = ["deliberate", "interactive"]
        router = make_router(cfg)
        response = await ask(router, "reasoning", 40, allow_fallback=True)
        assert response.binding == "interactive"
        assert response.prior_attempts == [{"binding": "deliberate", "model": "openai/strong-neutral",
                                             "status": "failed", "reason": "incomplete_final_answer"}]
        assert [body["max_tokens"] for _, body in endpoint.requests] == [40 + DEFAULT_REASONING_TOKENS, 40]


@pytest.mark.asyncio
async def test_the_allowance_is_configurable_and_never_exceeds_max_tokens_or_the_context():
    model = ThinkingModel(think=100, answers={"neutral_relation": {"relation": "adds"}})
    with ModelEndpoint(model) as endpoint:
        await ask(make_router(pool_config(endpoint.url, reasoningTokens=1000)), "reasoning", 250)
        assert endpoint.requests[-1][1]["max_tokens"] == 1250
        # maxTokens bounds the whole completion: the allowance shrinks, the answer keeps its budget.
        await ask(make_router(pool_config(endpoint.url, maxTokens=2000)), "reasoning", 1200)
        assert endpoint.requests[-1][1]["max_tokens"] == 2000
        await ask(make_router(pool_config(endpoint.url, maxTokens=2000)), "reasoning", 3000)
        assert endpoint.requests[-1][1]["max_tokens"] == 2000                  # as before: the binding's cap
        # The declared context bounds it too, after the prompt's room: the router's own estimate of the
        # prompt and the schema it is sent, with a margin for what that rough estimate misses.
        prompt = (estimate_tokens("Neutral routing fixture.") + 8
                  + estimate_tokens(json.dumps({"type": "json_schema", "json_schema": {**SCHEMA, "strict": True}})))
        room = math.ceil(PROMPT_MARGIN * prompt)
        assert PROMPT_MARGIN == 1.25 and room > prompt
        await ask(make_router(pool_config(endpoint.url, contextTokens=1500)), "reasoning", 250)
        assert endpoint.requests[-1][1]["max_tokens"] == 1500 - room
        # A context that fits the answer but not answer+allowance keeps the binding eligible (answer intact).
        await ask(make_router(pool_config(endpoint.url, contextTokens=300)), "reasoning", 40)
        assert endpoint.requests[-1][1]["max_tokens"] == 300 - room and 300 - room > 40
        # Where the margin leaves less than the answer, the answer still keeps its whole budget (this model
        # thinks 100 tokens first, so 40 cannot hold its answer: the attempt is incomplete, as without one).
        with pytest.raises(RuntimeError, match="incomplete_final_answer"):
            await ask(make_router(pool_config(endpoint.url, contextTokens=prompt + 45)), "reasoning", 40)
        assert endpoint.requests[-1][1]["max_tokens"] == 40
        # A context too small for the answer itself leaves the binding ineligible, as before.
        with pytest.raises(ValueError, match="No eligible local model"):
            make_router(pool_config(endpoint.url, contextTokens=prompt + 39)).function_config(
                context={"function_role": "reasoning", "max_output_tokens": 40,
                         "estimated_input_tokens": prompt})


@pytest.mark.asyncio
async def test_a_caller_without_a_cap_still_gets_the_bindings_own_max_tokens():
    model = ThinkingModel(think=100)
    with ModelEndpoint(model) as endpoint:
        router = make_router(pool_config(endpoint.url))
        await ask(router, "reasoning")
        assert endpoint.requests[-1][1]["max_tokens"] == 8192
        await ask(router, "extraction")
        assert endpoint.requests[-1][1]["max_tokens"] == 4096


@pytest.mark.asyncio
async def test_a_legacy_tier_binding_that_thinks_gets_the_allowance_too():
    model = ThinkingModel(think=600, answers={None: {"relation": "adds"}})     # no schema declared: prompt-only
    with ModelEndpoint(model) as endpoint:
        router = make_router({"provider": "vllm", "apiKey": "neutral-key", "baseUrl": endpoint.url + "/v1",
                              "models": {"large": {"model": "strong-neutral", "extraBody": THINKING,
                                                   "maxTokens": 8192, "reasoningTokens": 2048}}})
        response = await ask(router, "reasoning", 40)
        assert json.loads(response.content) == {"relation": "adds"}
        assert endpoint.requests[-1][1]["max_tokens"] == 40 + 2048


class ContextBoundModel:
    """A server whose real window, after the prompt it really sees (larger than the router's rough estimate),
    leaves ``fits`` completion tokens: a larger ``max_tokens`` is refused with the 400 vLLM sends."""

    def __init__(self, inner, *, fits):
        self.inner, self.fits = inner, fits

    def __call__(self, path, body):
        cap = body.get("max_tokens") or 0
        if cap > self.fits:
            return 400, {"object": "error", "type": "BadRequestError", "param": None, "code": 400,
                         "message": f"This model's maximum context length is {7000 + self.fits} tokens. However, "
                                    f"you requested {7000 + cap} tokens (7000 in the messages, {cap} in the "
                                    "completion). Please reduce the length of the messages or completion."}
        return self.inner(path, body)


def allowance_cap(context_tokens=128000, answer=40):
    """What the default allowance sends for ``ask``'s prompt on ``deliberate``."""
    prompt = (estimate_tokens("Neutral routing fixture.") + 8
              + estimate_tokens(json.dumps({"type": "json_schema", "json_schema": {**SCHEMA, "strict": True}})))
    return min(8192, answer + DEFAULT_REASONING_TOKENS, context_tokens - math.ceil(PROMPT_MARGIN * prompt))


@pytest.mark.asyncio
async def test_a_context_window_refusal_of_the_allowance_retries_the_same_binding_with_the_answer_budget():
    model = ContextBoundModel(ThinkingModel(think=10, answers={"neutral_relation": {"relation": "adds"}}), fits=500)
    with ModelEndpoint(model) as endpoint:
        router = make_router(pool_config(endpoint.url, contextTokens=8000))
        response = await ask(router, "reasoning", 40)
        assert json.loads(response.content) == {"relation": "adds"} and response.binding == "deliberate"
        assert [body["max_tokens"] for _, body in endpoint.requests] == [allowance_cap(8000), 40]
        # One attempt of the same binding: no failed attempt is reported, and the cap it answered under is.
        assert response.prior_attempts == [] and response.client_max_tokens == 40
        # A refused size is no reason to cool the endpoint down: the next call reaches it again.
        await ask(router, "reasoning", 40)
        assert [body["max_tokens"] for _, body in endpoint.requests][2:] == [allowance_cap(8000), 40]


@pytest.mark.asyncio
async def test_a_context_window_refusal_is_retried_once_and_only_where_an_allowance_was_added():
    model = ContextBoundModel(ThinkingModel(think=10), fits=20)
    with ModelEndpoint(model) as endpoint:
        router = make_router(pool_config(endpoint.url))
        # No allowance was added (the binding does not think): the refusal fails the attempt, as before.
        with pytest.raises(RuntimeError, match="attempts=ContextWindowExceededError$"):
            await ask(router, "extraction", 40)
        assert [body["max_tokens"] for _, body in endpoint.requests] == [40]
        # A thinking binding retries once with the answer's budget; refused again, it fails.
        with pytest.raises(RuntimeError, match="attempts=ContextWindowExceededError$"):
            await ask(router, "reasoning", 40)
        assert [body["max_tokens"] for _, body in endpoint.requests][1:] == [allowance_cap(), 40]


def test_the_prompts_room_has_a_margin_and_counts_tool_schemas():
    router = make_router(pool_config("http://127.0.0.1:9", contextTokens=10000))
    binding = router._snapshot.bindings["deliberate"]
    assert request_output_tokens(binding, 40, {"estimated_input_tokens": 4000}) == 40 + DEFAULT_REASONING_TOKENS
    # 10,000 - 1.25 x (4,000 + 2,000) leaves 2,500 for the answer and its allowance together.
    assert request_output_tokens(binding, 40, {"estimated_input_tokens": 4000, "tool_schema_tokens": 2000}) == 2500
    assert request_output_tokens(binding, 40, {"estimated_input_tokens": 7000, "tool_schema_tokens": 2000}) == 40
    assert request_output_tokens(binding, 40, {"estimated_input_tokens": 4000}, reasoning=False) == 40


@pytest.mark.asyncio
async def test_a_tool_calls_schemas_are_part_of_the_prompts_room():
    tools = [{"type": "function", "function": {"name": "neutral_lookup", "description": "neutral " * 400,
                                               "parameters": {"type": "object", "properties": {}}}}]
    model = ThinkingModel(think=10, answers={None: {"relation": "adds"}})
    with ModelEndpoint(model) as endpoint:
        router = make_router(pool_config(endpoint.url, contextTokens=3000, supportsTools=True))
        await router.complete([{"role": "user", "content": "Neutral routing fixture."}], tools=tools,
                              context={"function_role": "reasoning", "allow_fallback": False, "max_output_tokens": 40})
        prompt = estimate_tokens("Neutral routing fixture.") + 8
        room = math.ceil(PROMPT_MARGIN * (prompt + estimate_tokens(json.dumps(tools))))
        assert endpoint.requests[-1][1]["max_tokens"] == 3000 - room and room > 400


@pytest.mark.asyncio
async def test_a_call_no_attempt_completed_carries_what_its_attempts_reported():
    model = ThinkingModel(think=600, answers={"neutral_relation": {"relation": "adds"}}, thinks=lambda body: True)
    with ModelEndpoint(model) as endpoint:
        router = make_router(pool_config(endpoint.url, reasoningTokens=0))
        with pytest.raises(RuntimeError) as raised:
            await ask(router, "reasoning", 40)
        # The completion ran out at its cap while thinking: the endpoint still reported using it.
        assert raised.value.usage == {"prompt_tokens": 200, "completion_tokens": 40, "total_tokens": 240}
        assert charged_tokens(raised.value) == 240 == sum(model.reported)
        # Every failed attempt of a fallback adds what it reported.
        cfg = pool_config(endpoint.url, reasoningTokens=0)
        cfg["functionRoles"]["reasoning"]["candidates"] = ["deliberate", "interactive"]
        with pytest.raises(RuntimeError) as raised:
            await ask(make_router(cfg), "reasoning", 40, allow_fallback=True)
        assert raised.value.usage == {"prompt_tokens": 400, "completion_tokens": 80, "total_tokens": 480}
        # An error that never reached an endpoint carries nothing to charge.
        assert function_usage(ValueError("neutral")) == {} and charged_tokens(ValueError("neutral")) == 0


@pytest.mark.asyncio
async def test_an_answer_after_a_failed_attempt_carries_that_attempts_usage_too():
    model = ThinkingModel(think=9000, answers={"neutral_relation": {"relation": "adds"}},
                          thinks=lambda body: body["model"] == "strong-neutral")
    with ModelEndpoint(model) as endpoint:
        cfg = pool_config(endpoint.url)
        cfg["functionRoles"]["reasoning"]["candidates"] = ["deliberate", "interactive"]
        response = await ask(make_router(cfg), "reasoning", 40, allow_fallback=True)
        first = 200 + 40 + DEFAULT_REASONING_TOKENS
        assert response.prior_usage == {"prompt_tokens": 200, "completion_tokens": first - 200, "total_tokens": first}
        assert charged_tokens(response) == first + response.usage["total_tokens"] == sum(model.reported)
        assert function_usage(response)["prompt_tokens"] == 400


@pytest.mark.asyncio
async def test_each_qualification_observation_records_the_bindings_reasoning_allowance():
    from colonymind.qualification.runner import ObservedRouter
    model = ThinkingModel(think=100, answers={"neutral_relation": {"relation": "adds"}})
    with ModelEndpoint(model) as endpoint:
        observations = []
        observed = ObservedRouter(make_router(pool_config(endpoint.url, reasoningTokens=1024)), observations,
                                  "deliberate", qualification_role="reasoning")
        await ask(observed, "reasoning", 40)
        await ask(observed, "extraction", 40)
        assert [(row["binding_reasoning_tokens"], row["requested_max_output_tokens"], row["client_max_tokens"])
                for row in observations] == [(1024, 40, 40 + 1024), (0, 40, 40)]
        # A completion the router rejected keeps the allowance it was sent under too.
        observations.clear()
        failing = ObservedRouter(make_router(pool_config(endpoint.url, reasoningTokens=10)), observations,
                                 "deliberate", qualification_role="reasoning")
        with pytest.raises(RuntimeError):
            await ask(failing, "reasoning", 40)
        assert observations[0]["binding_reasoning_tokens"] is None
        assert observations[0]["rejected_completion"]["binding_reasoning_tokens"] == 10


# ---------------------------------------------------------------------------
# The persisted LLM config: the field is optional, validated, and an old config loads unchanged
# ---------------------------------------------------------------------------

def test_a_config_without_the_field_loads_unchanged():
    cfg = pool_config("http://127.0.0.1:9")
    del cfg["modelPool"]["deliberate"]["extraBody"]
    status = make_router(cfg).routing_status()
    assert status["reload_error"] is None
    assert {name: model["reasoning_tokens"] for name, model in status["models"].items()} == \
        {"interactive": 0, "deliberate": 0}
    # The thinking binding of an existing config needs no new field: its default allowance applies.
    assert make_router(pool_config("http://127.0.0.1:9")).routing_status()["models"]["deliberate"][
        "reasoning_tokens"] == DEFAULT_REASONING_TOKENS


@pytest.mark.parametrize("value, held", [(0, 0), (1, 1), (2048, 2048), (16384, 16384), (4096.0, 4096),
                                         (2 ** 20, 2 ** 20), (1.5, 1)])
def test_a_valid_reasoning_allowance_is_accepted(value, held):
    """As lenient as the binding's other numbers (``number``): a whole-number float is that number, and a
    fraction is cut as ``maxTokens``' would be, since an invalid persisted value drops all routing."""
    status = make_router(pool_config("http://127.0.0.1:9", reasoningTokens=value)).routing_status()
    assert status["models"]["deliberate"]["reasoning_tokens"] == held
    # An explicit allowance also serves a model that always reasons, with no thinking switch in its body.
    cfg = pool_config("http://127.0.0.1:9")
    cfg["modelPool"]["interactive"]["reasoningTokens"] = value
    assert make_router(cfg).routing_status()["models"]["interactive"]["reasoning_tokens"] == held


@pytest.mark.parametrize("value", [-1, -0.5, True, "4096", None, 2 ** 20 + 1, float("inf"), float("nan"), [4096]])
def test_an_invalid_reasoning_allowance_rejects_the_config_with_a_clear_error(value, tmp_path):
    with pytest.raises(ValueError, match="reasoningTokens must be a number of tokens from 0 to 1048576"):
        make_router(pool_config("http://127.0.0.1:9", reasoningTokens=value))
    # A reload keeps the last valid configuration, as for any other invalid field.
    path = tmp_path / "config.json"
    path.write_text(json.dumps(pool_config("http://127.0.0.1:9")))
    router = LLMRouter(tiers={})
    router.configure(json.loads(path.read_text()), config_path=path)
    revision = router.routing_status()["config_revision"]
    path.write_text(json.dumps(pool_config("http://127.0.0.1:9", reasoningTokens=value)))
    os.utime(path, ns=(1, 1))
    status = router.routing_status()
    assert status["reload_error"] == "ValueError" and status["config_revision"] == revision
