"""A thinking binding's reasoning allowance lets it generate longer; the callers' own deadlines still bound it.

The owner's turn waits on the owner_instruction call (``instinct.deliberate.TURN_DEADLINE``, tighter than the role's
deadline). With the allowance a thinking model is sent a larger cap and may take longer, but never past that bound:
an endpoint slower than the turn's deadline leaves the turn to the reader's own fallback in time.
"""
from __future__ import annotations

import time

from model_endpoint import ModelEndpoint
from colonymind.instinct import deliberate as typed
from colonymind.router.functions import DEFAULT_REASONING_TOKENS
from test_mind_outreach_loop import make  # noqa: F401  (make is a pytest fixture)
from test_mind_outreach_reactions import say
from test_router_reasoning_allowance import ThinkingModel, make_router, pool_config

TURN = 0.5          # the turn's deadline in these tests (10 s live); decide adds one second of grace
SLOW = 6.0          # the endpoint answers after this long: far past the turn's bound


def instruction_model():
    return ThinkingModel(think=10, answers={"mind_owner_instruction": {"instruction": "stop"}})


async def test_the_owner_turns_deadline_bounds_a_thinking_binding(monkeypatch):
    monkeypatch.setattr(typed, "TURN_DEADLINE", TURN)
    with ModelEndpoint(instruction_model(), delay=SLOW) as endpoint:
        router = make_router(pool_config(endpoint.url))
        started = time.monotonic()
        assert await typed.owner_instruction(router, "Please stop for today.") is None
        assert time.monotonic() - started < TURN + 1 + 0.5
        # The call that was cut off was a thinking request with the allowance on top of the answer's cap.
        [(_, body)] = endpoint.requests
        assert body["max_tokens"] == typed.MAX_OUTPUT_TOKENS + DEFAULT_REASONING_TOKENS
        assert body["chat_template_kwargs"] == {"enable_thinking": True}
    # In time, the same thinking binding answers within the bound.
    with ModelEndpoint(instruction_model()) as endpoint:
        assert await typed.owner_instruction(make_router(pool_config(endpoint.url)), "Please stop.") == "stop"


async def test_an_owner_turn_is_never_held_past_its_deadline_by_a_thinking_binding(make, monkeypatch):  # noqa: F811
    """Through the mind's own turn path: a stop beside someone else's words leaves the reader unsure, so the
    owner_instruction point asks the router; the endpoint is far slower than the turn's deadline."""
    monkeypatch.setattr(typed, "TURN_DEADLINE", TURN)
    fx = make(config={"faculties": {"owner_gate": True}})
    with ModelEndpoint(instruction_model(), delay=SLOW) as endpoint:
        fx.mind.router = make_router(pool_config(endpoint.url))
        started = time.monotonic()
        await say(fx, 'My boss wrote "stop checking in on the team". How do I answer that?', "t-reply")
        held = time.monotonic() - started
        assert [body["max_tokens"] for path, body in endpoint.requests if path.endswith("/chat/completions")] \
            == [typed.MAX_OUTPUT_TOKENS + DEFAULT_REASONING_TOKENS]
        assert held < TURN + 1 + 1.5 < SLOW
