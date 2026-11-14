"""Background thinking is charged and failed refreshes back off."""
from datetime import timedelta
from types import SimpleNamespace

import pytest

from colonymind.mind import gate
from colonymind.mind.ideas import RETRY_AFTER
from test_grounded_ideas import env, mailbox
from test_mind_consolidate import fx


@pytest.mark.parametrize('fail', [False, True])
async def test_refresh_charges_every_attempt_in_one_row(env, monkeypatch, fail):
    env.ideas.ingest(mailbox())
    sources = env.ideas.evidence(env.clock[0])
    monkeypatch.setattr(env.ideas, 'chunks', lambda _: [sources[:1], sources[1:]])
    calls = []
    async def complete(**kwargs):
        calls.append(kwargs)
        if fail:
            error = RuntimeError('synthetic unavailable')
            error.usage = {'total_tokens': 120}
            raise error
        return SimpleNamespace(content='{"offers": []}', usage={'total_tokens': 120}, cost_usd=0, model_id='test')
    monkeypatch.setattr(env.model, 'complete', complete)
    await env.ideas.refresh(env.clock[0], env.ideas.meta().get('generation', 0))
    rows = env.store.intentions(kind=['note'], limit=100)
    assert len(rows) == 1 and rows[0].cost_tokens == 240
    assert env.store.tokens_since(env.clock[0] - timedelta(days=1)) == 240
    assert env.ideas.meta()['last_refresh']['cost_tokens'] == 240
    assert len(calls) == 2


async def test_budget_is_rechecked_between_synthesis_calls(env, monkeypatch):
    env.ideas.ingest(mailbox())
    sources = env.ideas.evidence(env.clock[0])
    monkeypatch.setattr(env.ideas, 'chunks', lambda _: [sources[:1], sources[1:]])
    env.mind.authority = SimpleNamespace(tokens_allowed=lambda: env.store.tokens_since(
        env.clock[0] - timedelta(days=1)) < 100)
    calls = []
    async def complete(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content='{"offers": []}', usage={'total_tokens': 120}, cost_usd=0, model_id='test')
    monkeypatch.setattr(env.model, 'complete', complete)
    await env.ideas.refresh(env.clock[0], env.ideas.meta().get('generation', 0))
    assert len(calls) == 1
    assert env.ideas.meta()['last_refresh']['failure'] == 'RuntimeError'


async def test_repeated_failures_double_retry_gap_and_success_resets(env, monkeypatch):
    env.ideas.ingest(mailbox())
    async def fail(**kwargs):
        raise RuntimeError('synthetic unavailable')
    monkeypatch.setattr(env.model, 'complete', fail)
    for _ in range(2):
        await env.ideas.refresh(env.clock[0], env.ideas.meta().get('generation', 0))
    env.clock[0] += RETRY_AFTER + timedelta(seconds=1)
    await env.ideas.tick(env.clock[0])
    assert env.ideas.meta().get('pending_at') is None
    env.clock[0] += RETRY_AFTER
    await env.ideas.tick(env.clock[0])
    assert env.ideas.meta().get('pending_at') is not None
    async def success(**kwargs):
        return SimpleNamespace(content='{"offers": []}', usage={}, cost_usd=0, model_id='test')
    monkeypatch.setattr(env.model, 'complete', success)
    await env.ideas.refresh(env.clock[0], env.ideas.meta().get('generation', 0))
    assert env.ideas.meta()['consecutive_failures'] == 0


@pytest.mark.parametrize('fail', [False, True])
async def test_judge_cost_reaches_mind_day_budget(fx, fail):
    class Router:
        supports_function_routing = True
        async def complete(self, **kwargs):
            if fail:
                error = RuntimeError('synthetic unavailable')
                error.usage = {'total_tokens': 90}
                raise error
            return SimpleNamespace(content='{"choice": "digest"}', usage={'total_tokens': 90})
    fx.mind.router = Router()
    before = fx.store.tokens_since(fx.now - timedelta(days=1))
    await fx.mind.owner_gate.judge('system', 'prompt')
    assert fx.store.tokens_since(fx.now - timedelta(days=1)) == before + 90


def test_state_shows_last_ideas_refresh(fx):
    refresh = {'at': fx.now.isoformat(), 'failure': 'partial', 'published': 1}
    fx.mind.ideas.save_meta({'last_refresh': refresh})
    assert fx.mind.state()['ideas']['last_refresh'] == refresh


async def test_evidence_failure_also_records_retry_state(env, monkeypatch):
    def unavailable(now):
        raise RuntimeError('synthetic evidence failure')
    monkeypatch.setattr(env.ideas, 'evidence', unavailable)
    assert await env.ideas.refresh(env.clock[0], 0) == []
    meta = env.ideas.meta()
    assert meta['last_refresh']['failure'] == 'RuntimeError'
    assert meta['last_refresh']['model_calls'] == 0
    assert meta['consecutive_failures'] == 1
