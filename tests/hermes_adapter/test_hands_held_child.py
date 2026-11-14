"""An approval hold ends the child through native completion without another provider call."""

from copy import deepcopy

import pytest

from test_hands_guards import guards, hand  # noqa: F401


@pytest.mark.parametrize("api_mode", ["chat_completions", "anthropic_messages", "codex_responses", "bedrock_converse"])
@pytest.mark.parametrize("request_id", ["held-final", ""])
def test_held_child_finishes_locally_without_provider_or_producer_accounting(guards, monkeypatch, api_mode, request_id):
    state = hand(guards)
    state.blocked, state.activity = True, "awaiting_approval"
    guards.runtime.config = {"contracts": {"local": {"family": "local", "version": "1"}}}
    state.tokens = 57
    state.families_used.add("earlier")
    ids = dict(session_id="child-one", provider="local", model="local-model", api_mode=api_mode,
               api_request_id=request_id)
    request = {"messages": [], "tools": [{"name": "terminal"}], "tool_choice": "required"}
    original = deepcopy(request)
    monkeypatch.setattr(guards.middleware, "_admit_request",
                        lambda *args: pytest.fail("a held child waited for a provider slot"))
    envelope = guards.middleware.llm_request(guards.runtime, request=request, **ids)
    response = guards.middleware.llm_execution(guards.runtime, envelope["request"],
        lambda payload: pytest.fail("a held child called the provider again"), **ids)
    assert request == original and "tools" not in envelope["request"]
    assert response is not None
    if api_mode == "anthropic_messages":
        text = response.content[0].text
        assert response.stop_reason == "end_turn"
    elif api_mode == "codex_responses":
        text = response.output[0].content[0].text
        assert response.status == "completed"
    else:
        text = response.choices[0].message.content
        assert response.choices[0].finish_reason == "stop"
    assert text == "Waiting for owner approval. The held action has not run."
    assert response.usage is None
    guards.middleware.post_api_request(guards.runtime, usage=None, **ids)
    assert state.tokens == 57 and state.families_used == {"earlier"}
    assert not state.plan.get("contracts_used")
    assert not guards.runtime.registry.inflight
    assert state.blocked and state.activity == "awaiting_approval"
