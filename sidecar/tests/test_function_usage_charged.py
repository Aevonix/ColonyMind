"""A function call no attempt completed still used tokens, and every caller that charges a budget pays for it.

A completion that runs out while the model is still thinking reports using its whole cap. The router puts the summed
reports of a call's failed attempts on the ``RuntimeError`` it raises (``usage``), and on a later answer as
``prior_usage``; ``colonymind.usage.function_usage``/``charged_tokens`` read either. Each caller below charges a
failed call's reported tokens where it charges an answered one's.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from colonymind.beliefs.source_projection import _Metered
from colonymind.instinct.deliberate import decide, usage_tokens
from colonymind.mind import plain
from colonymind.mind.compose import Composer
from colonymind.mind.deliberate import Deliberation
from colonymind.mind.drift import TASK as BATTERY_TASK
from colonymind.usage import charged_tokens, function_usage
from test_mind_affect_loop import candidate, concern
from test_mind_drift import OWNER, DriftRouter, World

SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string"}}, "required": ["verdict"]}


def ran_out(total=240):
    """The router's error when every attempt ran out while thinking, as it raises it."""
    error = RuntimeError("No eligible local model completed function reasoning; attempts=incomplete_final_answer")
    error.usage = {"prompt_tokens": 200, "completion_tokens": total - 200, "total_tokens": total}
    return error


class FailingRouter:
    supports_function_routing = True

    def __init__(self, error):
        self.error, self.calls = error, 0

    def function_deadline_seconds(self, *, context=None):
        return 5

    async def complete(self, messages, *, context=None, **kwargs):
        self.calls += 1
        raise self.error


class AnsweringRouter(FailingRouter):
    """Answers after a failed attempt on another binding: the answer carries that attempt's usage too."""

    def __init__(self, content):
        super().__init__(None)
        self.content = content

    async def complete(self, messages, *, context=None, **kwargs):
        self.calls += 1
        return SimpleNamespace(content=self.content, usage={"prompt_tokens": 100, "completion_tokens": 20,
                                                            "total_tokens": 120},
                               prior_usage={"prompt_tokens": 200, "completion_tokens": 40, "total_tokens": 240})


def test_what_a_call_used_is_read_off_an_answer_or_an_error():
    assert function_usage(ran_out()) == {"prompt_tokens": 200, "completion_tokens": 40, "total_tokens": 240}
    answered = SimpleNamespace(usage={"total_tokens": 120}, prior_usage={"total_tokens": 240})
    assert charged_tokens(answered) == 360 == usage_tokens(answered)
    # Prompt plus completion where no total was reported; nothing at all for an error no endpoint answered.
    assert charged_tokens(SimpleNamespace(usage={"prompt_tokens": 7, "completion_tokens": 3})) == 10
    assert charged_tokens(TimeoutError()) == 0 and charged_tokens(SimpleNamespace(usage=None)) == 0
    assert charged_tokens(SimpleNamespace(usage={"total_tokens": True, "prompt_tokens": -4})) == 0


async def test_a_typed_decision_charges_a_call_that_failed():
    charged = []
    assert await decide(FailingRouter(ran_out()), task="neutral_task", system="neutral", data={}, schema=SCHEMA,
                        field="verdict", allowed=("yes",), on_usage=charged.append) is None
    assert charged == [240]
    # A failure no endpoint reported anything for charges nothing; a charge that fails never escapes.
    charged.clear()
    await decide(FailingRouter(TimeoutError()), task="neutral_task", system="neutral", data={}, schema=SCHEMA,
                 field="verdict", allowed=("yes",), on_usage=charged.append)
    assert charged == []

    def broken(tokens):
        raise OSError("ledger unavailable")
    assert await decide(FailingRouter(ran_out()), task="neutral_task", system="neutral", data={}, schema=SCHEMA,
                        field="verdict", allowed=("yes",), on_usage=broken) is None
    # An answer after a failed attempt charges both.
    assert await decide(AnsweringRouter(json.dumps({"verdict": "yes"})), task="neutral_task", system="neutral",
                        data={}, schema=SCHEMA, field="verdict", allowed=("yes",), on_usage=charged.append) == "yes"
    assert charged == [360]


async def test_a_plain_summary_charges_a_call_that_failed():
    charged = []
    text = "The first sentence is long enough to need a summary. " * 4
    assert await plain.summarize(FailingRouter(ran_out()), text, 40, on_usage=charged.append) is None
    assert charged == [240]


async def test_a_composition_charges_a_call_that_failed():
    composer = Composer(FailingRouter(ran_out()))
    text, tokens = await composer.compose(purpose="check_in", recipient_name="Sam", packet="neutral", topic="slides")
    assert tokens == 240 == composer.tokens_total and text.startswith("Hi Sam")


async def test_a_deliberation_charges_a_call_that_failed():
    deliberation = Deliberation(FailingRouter(ran_out()))
    research = candidate(type="research", drive="curiosity", dedup_base="research:x", priority=0.5, open_ended=True)
    shaped = await deliberation.form(concern(), research)
    assert deliberation.last_error == "RuntimeError" and deliberation.tokens_total == 240
    assert shaped.cost_tokens == 240                    # the tick charges the candidate's cost to the day


async def test_the_identity_check_charges_a_battery_call_that_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)

    class RanOut(DriftRouter):
        async def complete(self, messages, *, context=None, **kwargs):
            if (context or {}).get("task") == BATTERY_TASK:
                self.calls.append((messages, context))
                raise ran_out(4336)
            return await super().complete(messages, context=context, **kwargs)

    world = World(tmp_path, router=RanOut())
    try:
        summary = await world.mind.tick(force=True)
        assert summary["drift"]["trigger"] == "daily"
        [row] = world.checks()
        assert row.cost_tokens == 4336 and row.context["battery_error"] == "RuntimeError"
    finally:
        world.store.close()


async def test_a_dry_runs_meter_counts_a_call_that_failed():
    metered = _Metered(FailingRouter(ran_out()))
    with pytest.raises(RuntimeError):
        await metered.complete([{"role": "user", "content": "neutral"}], context={"task": "neutral_task"})
    assert metered.usage() == {"calls": 0, "prompt_tokens": 200, "completion_tokens": 40, "total_tokens": 240}
    answered = _Metered(AnsweringRouter("{}"))
    await answered.complete([{"role": "user", "content": "neutral"}], context={"task": "neutral_task"})
    assert answered.usage() == {"calls": 1, "prompt_tokens": 300, "completion_tokens": 60, "total_tokens": 360}
