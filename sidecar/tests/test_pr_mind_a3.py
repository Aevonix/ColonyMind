"""Night and idea calls can reach the next configured binding."""
import json

from types import SimpleNamespace

from colonymind.router.router import LLMResponse
from colonymind.router.tiers import ModelTier
from colonymind.mind.consolidate import TASK_NARRATIVE
from test_grounded_ideas import env, mailbox
from test_mind_consolidate import Fixture, cite_uuids, settled_task
from test_router_reasoning_allowance import ThinkingModel, make_router, pool_config


def in_process_router(monkeypatch, model, *, fallback=True, **binding):
    config = pool_config('http://127.0.0.1:9999', **binding)
    if fallback:
        config['functionRoles']['reasoning']['candidates'] = ['deliberate', 'interactive']
    router = make_router(config)
    requests = []

    async def at_endpoint(snapshot, binding, operation):
        return await operation('127.0.0.1')

    async def complete(**kwargs):
        cfg = kwargs['config']
        body = {'model': cfg.model_id.removeprefix('openai/'), 'messages': kwargs['messages'],
                'max_tokens': kwargs['max_output_tokens'],
                'response_format': kwargs.get('response_format')}
        requests.append(body)
        _, result = model('/v1/chat/completions', body)
        choice = result['choices'][0]
        return LLMResponse(request_id=kwargs['request_id'], tier_used=ModelTier.LARGE,
                           model_id=cfg.model_id, content=choice['message']['content'],
                           usage=result['usage'], latency_ms=1, cost_usd=0,
                           raw=SimpleNamespace(choices=[SimpleNamespace(
                               finish_reason=choice['finish_reason'],
                               message=SimpleNamespace(**choice['message']))]))

    monkeypatch.setattr(router, '_at_endpoint', at_endpoint)
    monkeypatch.setattr(router, '_litellm_call', complete)
    return router, requests


def first_unavailable(model):
    def answer(path, body):
        status, response = model(path, body)
        if body['model'] == 'strong-neutral':
            response['choices'][0]['message']['content'] = ''
        return status, response
    return answer


async def test_night_uses_second_binding_and_charges_both(tmp_path, monkeypatch):
    monkeypatch.setenv('COLONYMIND_SOURCE_CLAIMS', 'off')
    model = ThinkingModel(answers={TASK_NARRATIVE: lambda body: cite_uuids(body['messages'], {})})
    router, requests = in_process_router(monkeypatch, first_unavailable(model))
    fx = Fixture(tmp_path, router=router)
    try:
        settled_task(fx)
        night = await fx.mind.consolidate()
        assert night['errors'] == []
        assert night['counts']['narrative'] > 0
        assert [body['model'] for body in requests] == ['strong-neutral', 'fast-neutral']
        assert fx.store.get(night['id']).cost_tokens == sum(model.reported)
    finally:
        fx.store.close()


async def test_ideas_use_second_binding(env, monkeypatch):
    model = ThinkingModel(answers={None: {'offers': []}})
    env.mind.router, requests = in_process_router(monkeypatch, first_unavailable(model))
    env.ideas.ingest(mailbox())
    await env.ideas.refresh(env.clock[0], 0)
    assert env.ideas.meta()['last_refresh']['failure'] is None
    assert [body['model'] for body in requests] == ['strong-neutral', 'fast-neutral']


async def test_idea_copy_retry_allows_fallback(env, monkeypatch):
    contexts = []
    async def complete(messages, *, context):
        from types import SimpleNamespace
        contexts.append(context)
        return SimpleNamespace(content=json.dumps({'offers': [], 'copies': []}),
                               usage={}, cost_usd=0, model_id='test')
    def publish(offers, sources, now, *, copy_failures=None):
        if len(contexts) == 1:
            copy_failures.append(({'family': 'packet', 'anchor': 'Invoice', 'facts': []}, 'copy'))
        return []
    monkeypatch.setattr(env.model, 'complete', complete)
    monkeypatch.setattr(env.ideas, 'publish', publish)
    env.ideas.ingest(mailbox())
    await env.ideas.refresh(env.clock[0], 0)
    assert len(contexts) >= 2
    assert all(context.get('allow_fallback', True) for context in contexts)
