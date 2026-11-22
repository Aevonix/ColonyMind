"""Night admission covers the largest eligible fallback request."""
import pytest

from colonymind.mind.consolidate import TASK_NARRATIVE
from test_mind_consolidate import Fixture, cite_uuids, settled_task
from test_pr_mind_a3 import in_process_router
from test_router_reasoning_allowance import ThinkingModel, make_router, pool_config


def config():
    value = pool_config('http://127.0.0.1:9999')
    value['functionRoles']['reasoning']['candidates'] = ['interactive', 'deliberate']
    return value


@pytest.mark.parametrize('answer,expected', [(600, 4696), (None, 8192)])
def test_planned_cap_includes_only_eligible_and_allowed_fallbacks(answer, expected):
    router = make_router(config())
    context = {'task': TASK_NARRATIVE}
    if answer is not None:
        context['max_output_tokens'] = answer
    assert router.function_output_tokens(context=context) == expected
    assert router.function_output_tokens(context={**context, 'allow_fallback': True}) == expected
    assert router.function_output_tokens(context={**context, 'allow_fallback': False}) == (answer or 4096)
    # Filtering out the first binding changes which one is first, even without fallback.
    assert router.function_output_tokens(context={**context, 'allow_fallback': False,
                                                  'required_context_tokens': 64000}) == expected
    assert router.function_output_tokens(context={**context, 'required_context_tokens': 256000}) is None


async def test_night_refuses_larger_fallback_and_unreached_stages_go_first(tmp_path, monkeypatch):
    monkeypatch.setenv('COLONYMIND_SOURCE_CLAIMS', 'off')
    model = ThinkingModel(answers={TASK_NARRATIVE: lambda body: cite_uuids(body['messages'], {})})
    def answer(path, body):
        if body['model'] == 'fast-neutral':
            raise ConnectionError('first binding unavailable')
        return model(path, body)
    router, requests = in_process_router(monkeypatch, answer)
    router.configure(config())
    fx = Fixture(tmp_path, router=router, config={'budgets': {'llm_tokens_per_day': 4000}})
    try:
        settled_task(fx)
        first = await fx.mind.consolidate()
        assert requests == []
        assert first['exhausted'] and first['counts']['budget_stops'] == 1
        assert first['tokens'] == first['calls'] == 0 and first['errors'] == []
        assert first['done'] == []
        fx.shift(days=1)
        fx.restart()
        second = await fx.mind.consolidate()
        assert second['done'] == ['contradictions', 'digests', 'pages', 'episodes']
        assert second['exhausted'] and requests == []
    finally:
        fx.store.close()
