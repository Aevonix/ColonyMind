"""The night's reasoning calls on a binding that thinks before it answers (the live ``deliberate`` binding).

The real router and the real night, against a local endpoint whose model spends output tokens reasoning
before its JSON answer. Before the router sent a reasoning allowance, every capped call (a 40-token relation,
a 250-token episode, a 600-token narrative) ran out while thinking and the night recorded
``RuntimeError: No eligible local model completed function reasoning`` for each one.
"""
from __future__ import annotations

import pytest

from model_endpoint import ModelEndpoint
from colonymind.mind.consolidate import RELATION_OUTPUT_TOKENS, RELATION_TASK, TASK_DIGEST, TASK_EPISODE, TASK_NARRATIVE
from colonymind.router.functions import DEFAULT_REASONING_TOKENS
from colonymind.router.router import LLMRouter
from test_mind_consolidate import CONTACT, OWNER, Fixture, cite_claims, cite_uuids, settled_task
from test_router_reasoning_allowance import THINKING, ThinkingModel

THINK = 600                     # more than any of these answers' caps, far less than the allowance
CAPS = {TASK_NARRATIVE: 600, RELATION_TASK: RELATION_OUTPUT_TOKENS, TASK_EPISODE: 250}


def thinking_router(url, **deliberate):
    router = LLMRouter(tiers={})
    router.configure({"provider": "vllm", "apiKey": "neutral-key", "models": {},
                      "modelPool": {"deliberate": {"model": "strong-neutral", "baseUrl": url + "/v1",
                                                   "contextTokens": 128000, "maxTokens": 8192,
                                                   "supportsJsonSchema": True, "extraBody": THINKING,
                                                   **deliberate}},
                      "functionRoles": {"reasoning": {"candidates": ["deliberate"], "timeoutSeconds": 10,
                                                      "deadlineSeconds": 20}}})
    return router


def night_model():
    return ThinkingModel(think=THINK, answers={
        TASK_NARRATIVE: lambda body: cite_uuids(body["messages"], {}),
        TASK_DIGEST: lambda body: cite_claims(body["messages"], {}),
        TASK_EPISODE: {"summary": "They asked for the slides; the assistant promised them by Friday."},
        RELATION_TASK: {"relation": "replaces"},
    })


async def a_night_of_work(fx):
    """A settled task (the narrative), two statements that differ (a relation decision) and a session
    (an episode summary): three reasoning-role calls with small answer caps."""
    settled_task(fx)
    await fx.fact("turn-a", OWNER, "telegram-1", "My office is room 4.", "room 4")
    fx.shift(days=1)
    await fx.fact("turn-b", OWNER, "discord-2", "My office is room 7.", "room 7")
    for index in range(4):
        fx.turn(f"e-{index}", CONTACT, "sms-1", f"Can you send me slide {index}?", "On its way.")


def sent(endpoint):
    """Each consolidation request's schema name and completion cap, in order."""
    return [(body["response_format"]["json_schema"]["name"], body["max_tokens"]) for _, body in endpoint.requests]


async def test_the_nights_reasoning_calls_answer_when_the_model_thinks_first(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    monkeypatch.setenv("COLONYMIND_SOURCE_CLAIMS", "off")            # the claims are extracted by the fixture
    model = night_model()
    with ModelEndpoint(model) as endpoint:
        fx = Fixture(tmp_path, router=thinking_router(endpoint.url))
        await a_night_of_work(fx)
        night = await fx.mind.consolidate()
        assert night["errors"] == []
        assert night["counts"]["narrative"] >= 1 and night["counts"]["contradictions"] == 1
        assert night["counts"]["episodes"] == 1
        tasks = [name for name, _ in sent(endpoint)]
        assert {TASK_NARRATIVE, RELATION_TASK, TASK_EPISODE} <= set(tasks)
        # Each call was sent its own answer cap plus the binding's reasoning allowance; thinking stayed on.
        for name, cap in sent(endpoint):
            if name in CAPS:
                assert cap == CAPS[name] + DEFAULT_REASONING_TOKENS
        assert all(body["chat_template_kwargs"] == {"enable_thinking": True} for _, body in endpoint.requests)
        # The real usage, thinking included, is what the night charged, and it did not stop after one call.
        assert night["calls"] == len(endpoint.requests) >= 3 and not night["counts"].get("budget_stops")
        assert night["tokens"] == sum(model.reported) and night["peak"] == max(model.reported) > THINK
        assert fx.store.get(night["id"]).cost_tokens == night["tokens"]
        fx.store.close()


async def test_without_an_allowance_the_night_retries_at_the_binding_cap(tmp_path, monkeypatch):
    """``reasoningTokens: 0`` spends the first cap thinking, then the larger cap permits an answer."""
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    monkeypatch.setenv("COLONYMIND_SOURCE_CLAIMS", "off")
    model = night_model()
    with ModelEndpoint(model) as endpoint:
        fx = Fixture(tmp_path, router=thinking_router(endpoint.url, reasoningTokens=0))
        await a_night_of_work(fx)
        night = await fx.mind.consolidate()
        assert night["calls"] > 0 and night["errors"] == []
        requests = sent(endpoint)
        assert len(requests) == 2 * night["calls"]
        assert all(requests[i + 1] == (name, 8192) for i, (name, _) in enumerate(requests) if i % 2 == 0)
        assert all(cap == CAPS[name] for i, (name, cap) in enumerate(requests) if i % 2 == 0 and name in CAPS)
        # Every logical call pays for both attempts.
        assert night["tokens"] == sum(model.reported) > 0
        assert night["peak"] == max(sum(model.reported[i:i + 2]) for i in range(0, len(model.reported), 2))
        assert fx.store.get(night["id"]).cost_tokens == night["tokens"]
        fx.store.close()


async def test_a_night_whose_calls_all_run_out_still_pays_for_them_and_its_budget_stops(tmp_path, monkeypatch):
    """A model that never finishes thinking inside even the allowance: every call fails, every call is charged
    what the endpoint reported, and the night's share stops the calls instead of letting them run on."""
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    monkeypatch.setenv("COLONYMIND_SOURCE_CLAIMS", "off")
    model = ThinkingModel(think=10 ** 6)
    with ModelEndpoint(model) as endpoint:
        # 0.05 x 200,000 = 10,000 tokens: the retry gets the 4,904-token cap that still fits.
        fx = Fixture(tmp_path, config={"budgets": {"learn_share": 0.05}}, router=thinking_router(endpoint.url))
        await a_night_of_work(fx)
        night = await fx.mind.consolidate()
        assert night["budget"] == 10000 and night["calls"] == 0
        assert len(endpoint.requests) == len(model.reported) == 2
        assert sent(endpoint) == [(TASK_NARRATIVE, 4696), (TASK_NARRATIVE, 4904)]
        assert night["tokens"] == sum(model.reported) == sum(200 + cap for _, cap in sent(endpoint))
        assert night["exhausted"] is True and night["counts"]["budget_stops"] == 1
        assert all(error.endswith(": RuntimeError") for error in night["errors"]) and len(night["errors"]) == 1
        assert fx.store.get(night["id"]).cost_tokens == night["tokens"]
        fx.store.close()


async def test_the_nights_budget_check_counts_the_reasoning_allowance(tmp_path, monkeypatch):
    """A call is made only while what is left of the night's share covers what it may generate: its
    answer cap plus the allowance, not the answer cap alone."""
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    monkeypatch.setenv("COLONYMIND_SOURCE_CLAIMS", "off")
    with ModelEndpoint(night_model()) as endpoint:
        # 0.02 x 200,000 = 4,000: room for a 600-token answer, not for 600 + 4,096.
        fx = Fixture(tmp_path, config={"budgets": {"learn_share": 0.02}}, router=thinking_router(endpoint.url))
        settled_task(fx)
        night = await fx.mind.consolidate()
        assert night["budget"] == 4000 and night["calls"] == 0 and endpoint.requests == []
        assert night["exhausted"] is True and night["counts"]["budget_stops"] == 1
        fx.store.close()
    with ModelEndpoint(night_model()) as endpoint:
        # A smaller declared allowance fits the same share.
        fx = Fixture(tmp_path / "small", config={"budgets": {"learn_share": 0.02}},
                     router=thinking_router(endpoint.url, reasoningTokens=1024))
        settled_task(fx)
        night = await fx.mind.consolidate()
        assert night["calls"] == 1 and night["errors"] == [] and sent(endpoint) == [(TASK_NARRATIVE, 600 + 1024)]
        fx.store.close()
