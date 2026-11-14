"""Nightly length retries keep the binding, deadline, accounting and budget."""
from contextlib import closing

import pytest

from model_endpoint import ModelEndpoint
from colonymind import health_counters, usage
from colonymind.mind.consolidate import TASK_NARRATIVE
from colonymind.work.night import TASK as TITLE_TASK, WorkNight
from test_mind_consolidate import Fixture, OWNER, settled_task
from test_mind_consolidate_thinking import sent, thinking_router
from test_router_reasoning_allowance import ThinkingModel, ask, make_router, pool_config
from test_work_projection import envelope


@pytest.mark.parametrize("cap,think,caps", [(1200, 11000, [9392, 12288]),
                                         (12288, 20000, [12288]),
                                         (1200, 20000, [9392, 12288])])
async def test_length_retry_is_once_on_the_same_binding(tmp_path, monkeypatch, cap, think, caps):
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    model = ThinkingModel(think=think)
    checks = []

    async def budget(max_tokens, spent):
        checks.append((max_tokens, spent))
        return True

    with ModelEndpoint(model) as endpoint:
        router = thinking_router(endpoint.url, maxTokens=12288, reasoningTokens=8192)
        if think > 12288:
            with pytest.raises(RuntimeError) as raised:
                await ask(router, "reasoning", cap, length_retry_budget=budget)
            outcome = raised.value
            assert [item["max_tokens"] for item in outcome.attempts] == caps
            assert all(item["reason"] == "incomplete_final_answer" for item in outcome.attempts)
        else:
            outcome = await ask(router, "reasoning", cap, length_retry_budget=budget)
            assert outcome.binding == "deliberate"
        assert [body["max_tokens"] for _, body in endpoint.requests] == caps
        assert {body["model"] for _, body in endpoint.requests} == {"strong-neutral"}
        assert len(checks) == len(caps) - 1
        assert usage.charged_tokens(outcome) == sum(model.reported)
        if checks:
            assert checks == [(12288, {"prompt_tokens": 200, "completion_tokens": 9392,
                                      "total_tokens": 9592})]
    assert usage.flush()
    rows = usage.rows(tmp_path)
    assert [row["outcome"] for row in rows] == ["incomplete"] * (len(caps) - 1) + [
        "incomplete" if think > 12288 else "ok"]


@pytest.mark.parametrize("finish", ["content_filter", "stop"])
async def test_non_length_incomplete_does_not_retry(finish):
    from test_model_usage import chat_answer

    async def budget(*args):
        pytest.fail("Only length cut-offs may retry")

    with ModelEndpoint(chat_answer(content="", finish=finish)) as endpoint:
        with pytest.raises(RuntimeError):
            await ask(thinking_router(endpoint.url), "reasoning", 600, length_retry_budget=budget)
        assert len(endpoint.requests) == 1


@pytest.mark.parametrize("share,allowed,caps", [(0.25, True, [4696, 8192]),
                                               (0.05, True, [4696, 4904]),
                                               (0.25, False, [4696])])
async def test_incomplete_night_reports_status_counter_and_safe_attempts(tmp_path, monkeypatch, caplog,
                                                                      share, allowed, caps):
    monkeypatch.setenv("COLONYMIND_SOURCE_CLAIMS", "off")
    model = ThinkingModel(think=20000)
    key = "mind_incomplete." + TASK_NARRATIVE
    before = health_counters.snapshot().get(key, 0)
    with ModelEndpoint(model) as endpoint:
        fx = Fixture(tmp_path, config={"budgets": {"learn_share": share}}, router=thinking_router(endpoint.url))
        settled_task(fx)
        fx.shift(days=1)
        fx.mind.consolidation.tokens_allowed = lambda: allowed or not endpoint.requests
        result = await fx.mind._consolidation_step(fx.now, force=True)
        assert result == "incomplete"
        assert health_counters.snapshot()[key] == before + 1
        state = fx.mind._consolidation_state()
        assert state["errors"] == [TASK_NARRATIVE + ": RuntimeError"]
        assert state["last_tokens"] == sum(model.reported)
        assert [cap for _, cap in sent(endpoint)] == caps
        assert "incomplete_final_answer" in caplog.text and "4696" in caplog.text
        assert "scratch" not in caplog.text
        assert not state["counts"].get("budget_stops")
        fx.store.close()


async def test_work_titles_use_the_nightly_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_SOURCE_CLAIMS", "off")
    model = ThinkingModel(think=6000, answers={TITLE_TASK: {"titles": [{"id": "w1", "title": "Update the report"}]}})
    with ModelEndpoint(model) as endpoint:
        config = pool_config(endpoint.url)
        config["functionRoles"]["extraction"]["candidates"] = ["deliberate"]
        fx = Fixture(tmp_path, config={"faculties": {"consolidation": False, "work_records": True}},
                     router=make_router(config))
        fx.ledger.record_source("root", contact_id=OWNER, session_id="session",
            messages=[{"role": "user", "content": "Update the report."}],
            work={"envelope": envelope(), "principal_id": "host"}, derive_claims=False)
        work = WorkNight(fx.ledger, OWNER)
        work.store.reconcile()
        night = await fx.mind.consolidate()
        assert night["errors"] == [] and night["counts"]["work_titled"] == 1
        assert sent(endpoint) == [(TITLE_TASK, 768 + 4096), (TITLE_TASK, 8192)]
        assert night["tokens"] == sum(model.reported)
        assert work.store.get_root("root", contact_id=OWNER)["title"] == "Update the report"
        fx.store.close()


@pytest.mark.parametrize('age', ['backlog', 'watermark', 'before_night', 'during_night'])
async def test_claim_settle_timeout_only_counts_this_nights_claims(tmp_path, monkeypatch, age):
    from colonymind.turns import projection_backlog

    monkeypatch.setenv("COLONYMIND_SOURCE_CLAIMS", "on")
    monkeypatch.setattr("colonymind.mind.consolidate.CLAIM_SETTLE_S", 0)
    fx = Fixture(tmp_path)
    fx.ledger.record_source("waiting", contact_id=OWNER, session_id="session",
                            messages=[{"role": "user", "content": "The office is room four."}])
    fx.shift(days=1)
    with closing(fx.ledger._connect()) as conn, conn:
        watermark = projection_backlog.watermark(conn)
        enqueued = {'backlog': watermark - 1, 'watermark': watermark,
                    'before_night': fx.now.timestamp() - 1, 'during_night': fx.now.timestamp()}[age]
        conn.execute('UPDATE source_claim_jobs SET enqueued_at=?', (enqueued,))
    # The worker's split: enqueued at or after the watermark is new work, which the night settles.
    current = age in {'watermark', 'before_night', 'during_night'}
    key = 'mind_incomplete.claims_settle'
    before = health_counters.snapshot().get(key, 0)
    assert await fx.mind._consolidation_step(fx.now, force=True) == ('incomplete' if current else 'done')
    state = fx.mind._consolidation_state()
    assert ('claims_settle_timeout' in state['errors']) == current
    assert state['counts'].get('claims_unfinished', 0) == int(current)
    assert state['counts']['claims_backlog'] == int(not current)
    assert health_counters.snapshot().get(key, 0) == before + int(current)
    fx.store.close()
