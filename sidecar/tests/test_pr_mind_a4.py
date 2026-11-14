"""A night can buy a smaller length retry without losing reported usage."""
import pytest

from colonymind.mind.consolidate import TASK_NARRATIVE
from test_mind_consolidate import Fixture, cite_uuids, settled_task
from test_pr_mind_a3 import in_process_router
from test_router_reasoning_allowance import ThinkingModel, ask


@pytest.mark.parametrize('budget,caps', [(15000, [4696, 6304]), (16888, [4696, 8192]),
                                         (11000, [4696])])
async def test_night_retries_with_affordable_cap(tmp_path, monkeypatch, budget, caps):
    monkeypatch.setenv('COLONYMIND_SOURCE_CLAIMS', 'off')
    model = ThinkingModel(think=5000, prompt=2000, thinks=lambda body: True,
                          answers={TASK_NARRATIVE: lambda body: cite_uuids(body['messages'], {})})
    router, requests = in_process_router(monkeypatch, model, fallback=False)
    fx = Fixture(tmp_path, router=router, config={'budgets': {'learn_share': budget / 200000}})
    try:
        settled_task(fx)
        night = await fx.mind.consolidate()
        assert [body['max_tokens'] for body in requests] == caps
        assert night['tokens'] == sum(model.reported) == fx.store.get(night['id']).cost_tokens
        assert night['tokens'] <= budget
        assert bool(night['errors']) == (len(caps) == 1)
        assert night['calls'] == int(len(caps) == 2)
    finally:
        fx.store.close()


@pytest.mark.parametrize('approved,expected', [(5500, 5500), (20000, 8192), (4000, None), (0, None)])
async def test_router_uses_smaller_of_binding_and_approved_cap(monkeypatch, approved, expected):
    model = ThinkingModel(think=5000, thinks=lambda body: True)
    router, requests = in_process_router(monkeypatch, model, fallback=False)
    async def budget(cap, spent):
        assert cap == 8192 and spent['total_tokens'] == 4896
        return approved
    if expected is None:
        with pytest.raises(RuntimeError):
            await ask(router, 'reasoning', 600, length_retry_budget=budget)
    else:
        await ask(router, 'reasoning', 600, length_retry_budget=budget)
    assert [body['max_tokens'] for body in requests] == [4696] + ([expected] if expected else [])


async def test_smaller_retry_spends_share_before_later_stages(tmp_path, monkeypatch):
    from test_mind_consolidate_thinking import a_night_of_work

    monkeypatch.setenv('COLONYMIND_SOURCE_CLAIMS', 'off')
    model = ThinkingModel(think=10**6, thinks=lambda body: True)
    router, requests = in_process_router(monkeypatch, model, fallback=False)
    fx = Fixture(tmp_path, router=router, config={'budgets': {'learn_share': 0.05}})
    try:
        await a_night_of_work(fx)
        night = await fx.mind.consolidate()
        assert [body['max_tokens'] for body in requests] == [4696, 4904]
        assert night['tokens'] == sum(model.reported) == 10000
        assert night['calls'] == 0 and len(night['errors']) == 1
        assert night['exhausted'] and night['counts']['budget_stops'] == 1
        assert fx.store.get(night['id']).cost_tokens == 10000
    finally:
        fx.store.close()
