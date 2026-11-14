"""Role time settings reach validation, retry budgets and the HTTP transport."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest

from colonymind.router.router import LLMRouter
import colonymind.router.router as routing
from test_grounded_ideas import env as ideas_env, generate  # noqa: F401
from test_source_appraisals import state as appraisal_state, source, Processor  # noqa: F401


def config(**limits):
    return {
        "provider": "vllm",
        "models": {},
        "modelPool": {
            name: {"model": name, "baseUrl": "http://127.0.0.1:9/v1"}
            for name in ("first", "second", "third")
        },
        "functionRoles": {"reasoning": {"candidates": ["first", "second", "third"], **limits}},
    }


def test_reasoning_accepts_900_1800():
    router = LLMRouter(tiers={})
    router.configure(config(timeoutSeconds=900, deadlineSeconds=1800))
    role = router._snapshot.roles["reasoning"]
    assert role.timeout_seconds == 900
    assert role.deadline_seconds == 1800
    assert router.function_deadline_seconds(context={"task": "mind_lessons"}) == 1800


@pytest.mark.parametrize("seconds", [.05, 300, 600, 1e10])
def test_finite_time_settings_have_no_policy_ceiling(seconds):
    router = LLMRouter(tiers={})
    router.configure(config(timeoutSeconds=seconds, deadlineSeconds=seconds))
    role = router._snapshot.roles["reasoning"]
    assert role.timeout_seconds == role.deadline_seconds == seconds


@pytest.mark.parametrize("key", ["timeoutSeconds", "deadlineSeconds"])
@pytest.mark.parametrize("invalid", ["900", float("nan"), float("inf"), float("-inf"), 0, -1, True, None, .049])
def test_invalid_time_settings_still_rejected(key, invalid):
    with pytest.raises(ValueError, match="Invalid routing capability or limit"):
        LLMRouter(tiers={}).configure(config(**{key: invalid}))


def test_absent_time_keys_keep_every_role_default():
    router = LLMRouter(tiers={})
    router.configure(config())
    for name, role in router._snapshot.roles.items():
        expected = (120, 180) if name in {"reasoning", "planning", "judging", "coding"} else (20, 40)
        assert (role.timeout_seconds, role.deadline_seconds) == expected


@pytest.mark.asyncio
async def test_retry_timers_use_900_and_stop_at_total_1800(monkeypatch):
    router = LLMRouter(tiers={})
    router.configure(config(timeoutSeconds=900, deadlineSeconds=1800))
    now = [100.0]
    timers, calls = [], []

    class Timer:
        def __init__(self, seconds):
            timers.append(seconds)
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        def expired(self):
            return True

    async def fail(**kwargs):
        calls.append(kwargs["config"].model_id)
        # First attempt consumes 1000 wall seconds including dispatch overhead.
        # The second must receive only 800 seconds and exhaust the total budget.
        now[0] += 1000 if len(calls) == 1 else 800
        raise TimeoutError("simulated attempt exhaustion")

    monkeypatch.setattr(routing, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(routing, "asyncio", SimpleNamespace(timeout=Timer))
    monkeypatch.setattr(router, "_litellm_call", fail)
    with pytest.raises(RuntimeError, match="No eligible local model"):
        await router.complete([{"role": "user", "content": "Test"}], context={"task": "mind_lessons"})
    assert timers == [900, 800]
    assert calls == ["openai/first", "openai/second"]
    assert now[0] == 1900


@pytest.mark.asyncio
async def test_900_reaches_actual_openai_http_request(monkeypatch):
    """Keep LiteLLM and OpenAI real; intercept only HTTP, including timeout extensions."""
    router = LLMRouter(tiers={})
    router.configure(config(timeoutSeconds=900, deadlineSeconds=1800))
    captured = []
    real_client = routing._local_http_client

    async def respond(request):
        captured.append(deepcopy(request.extensions["timeout"]))
        return httpx.Response(200, json={
            "id": "timeout-test", "object": "chat.completion", "created": 1, "model": "first",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Done"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    def client(*args, **kwargs):
        http_client = real_client(*args, **kwargs)
        http_client._transport = httpx.MockTransport(respond)
        return http_client

    monkeypatch.setattr(routing, "_local_http_client", client)
    result = await router.complete([{"role": "user", "content": "Test"}], context={"task": "mind_lessons"})
    assert result.content == "Done"
    assert len(captured) == 1
    assert captured[0] == {"connect": 900, "read": 900, "write": 900, "pool": 900}


class DeadlineRouter:
    supports_function_routing = True

    def __init__(self, deadline):
        self.deadline = deadline
        self.calls = []

    def function_deadline_seconds(self, *, context=None):
        return self.deadline

    async def complete(self, *args, context=None, **kwargs):
        self.calls.append(context)
        raise RuntimeError("offline endpoint")


@pytest.mark.parametrize("deadline, expected", [(1800, 1800), (600, 600), (599, 600), (180, 185)])
def test_opinion_deadline_follows_large_role_and_preserves_existing_grace(deadline, expected):
    from colonymind.mind.opinions import _deadline
    assert _deadline(DeadlineRouter(deadline)) == expected


@pytest.mark.parametrize("role", ["extraction", "judging"])
def test_source_claim_deadlines_follow_role(role):
    from colonymind.beliefs.source_claims import _role_timeout_seconds
    assert _role_timeout_seconds(DeadlineRouter(1800), role) == 1805
    assert _role_timeout_seconds(DeadlineRouter(float("inf")), role) == 40


@pytest.mark.asyncio
@pytest.mark.parametrize("caller, grace", [
    ("lessons", 5), ("deliberate", 5), ("compose", 5), ("plain", 1), ("instinct", 1), ("drift", 5),
])
async def test_callers_use_large_role_deadline(monkeypatch, caller, grace):
    from colonymind.instinct.deliberate import decide
    from colonymind.mind import plain
    from colonymind.mind.compose import Composer
    from colonymind.mind.consolidate import Consolidation, Night
    from colonymind.mind.deliberate import Deliberation
    from colonymind.mind.drift import DriftCheck, Record
    from test_mind_affect_loop import candidate, concern

    router = DeadlineRouter(1800)
    waits = []

    async def wait_for(operation, timeout):
        waits.append(timeout)
        return await operation

    monkeypatch.setattr(asyncio, "wait_for", wait_for)
    if caller == "lessons":
        night = Night(local_date="2026-11-09", started_at="", since="", note_id="test", budget=10000)
        consolidation = Consolidation(store=None, ledger=None, concerns=None, mind_state=None, contacts=None,
                                      router=router, autobiography=None, owner_id=None, budgets=None,
                                      tokens_allowed=lambda: True, faculties={})
        await consolidation._call(night, task="mind_lessons", system="test", user="test", schema={}, max_output_tokens=10)
    elif caller == "deliberate":
        await Deliberation(router).form(concern(), candidate(type="research", drive="curiosity",
                                       dedup_base="research:x", priority=.5, open_ended=True))
    elif caller == "compose":
        await Composer(router).compose(purpose="check_in", recipient_name="Sam", packet="test", topic="slides")
    elif caller == "plain":
        await plain.summarize(router, "A long sentence about the slides.", 20)
    elif caller == "instinct":
        await decide(router, task="test", system="test", data={}, schema={}, field="verdict", allowed=("yes",))
    else:
        drift = DriftCheck(store=None, ledger=None, mind_state=None, router=router, owner_id=None, clock=lambda: None)
        await drift._ask(Record())
    assert len(router.calls) == 1
    assert waits == [1800 + grace]


@pytest.mark.asyncio
async def test_idea_synthesis_uses_large_deadline(ideas_env, monkeypatch):
    waits = []

    async def wait_for(operation, timeout):
        waits.append(timeout)
        return await operation

    monkeypatch.setattr(asyncio, "wait_for", wait_for)
    monkeypatch.setattr(ideas_env.model, "function_deadline_seconds", lambda **_: 1800, raising=False)
    await generate(ideas_env)
    assert waits and all(value == 1805 for value in waits)


@pytest.mark.asyncio
async def test_appraisal_job_accepts_large_deadline(appraisal_state, monkeypatch):
    source(appraisal_state, "timeout-test", "The export failed after its validation step.")
    router = Processor()
    waits = []

    async def wait_for(operation, timeout):
        waits.append(timeout)
        return await operation

    monkeypatch.setattr(asyncio, "wait_for", wait_for)
    monkeypatch.setattr(router, "function_deadline_seconds", lambda **_: 1800)
    assert await appraisal_state.process_one(router)
    assert len(router.requests) == 1
    assert waits == [1805]


@pytest.mark.asyncio
async def test_commitment_job_accepts_large_deadline_and_extends_lease(tmp_path, monkeypatch):
    from colonymind.commitments.extract import CommitmentExtractor
    from colonymind.commitments.store import CommitmentStore
    from colonymind.turns import TurnIdempotencyLedger

    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", "owner")
    store = CommitmentStore(db_path=tmp_path / "commitments.db")
    ledger = TurnIdempotencyLedger(tmp_path / "sources.db")
    extractor = CommitmentExtractor(ledger, lambda: store)
    ledger.record_source("timeout-test", contact_id="owner", session_id="session", messages=[
        {"role": "user", "content": "Remind me to call the dentist tomorrow."},
        {"role": "assistant", "content": "Noted."}])
    waits, leases = [], []

    class Router(DeadlineRouter):
        async def complete(self, *args, **kwargs):
            with ledger._connect() as connection:
                leases.append(connection.execute(
                    "SELECT lease_until FROM commitment_runs WHERE turn_id='timeout-test'").fetchone()[0])
            return await super().complete(*args, **kwargs)

    async def wait_for(operation, timeout):
        waits.append(timeout)
        return await operation

    router = Router(1800)
    monkeypatch.setattr(asyncio, "wait_for", wait_for)
    assert await extractor.process_one(router)
    assert len(router.calls) == 1
    assert waits == [1805]
    assert leases[0] > extractor.clock() + 1800
