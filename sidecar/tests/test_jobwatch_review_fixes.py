"""Regression cases from the independent job watch review."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from colonymind import config, health_counters
from colonymind.backup import _copy_private_tree, create_full_backup, restore_full_backup
from colonymind.mind.consolidate import Consolidation, Night
from colonymind.router.functions import PROMPT_MARGIN
from colonymind.router.tokens import estimate_tokens
from test_router_reasoning_allowance import make_router, pool_config


async def test_declined_retry_leaves_room_for_next_task():
    calls = []
    spent = {'prompt_tokens': 1500, 'completion_tokens': 9392, 'total_tokens': 10892}

    async def complete(messages, context):
        calls.append(context['task'])
        if context['task'] == 'first':
            assert await context['length_retry_budget'](12288, spent) == 10608
            error = RuntimeError('incomplete_final_answer')
            error.usage = spent
            error.attempts = []
            raise error
        return SimpleNamespace(content='{"ok": true}', usage={'total_tokens': 1900}, prior_usage={})

    runner = SimpleNamespace(available=True, tokens_allowed=lambda: True,
        store=SimpleNamespace(update=lambda *a, **k: None),
        router=SimpleNamespace(complete=complete,
            function_output_tokens=lambda context: context['max_output_tokens'] + 8192))
    runner._output_cap = lambda task, cap: Consolidation._output_cap(runner, task, cap)
    runner._charge = lambda night, tokens: Consolidation._charge(runner, night, tokens)
    night = Night(local_date='d', started_at='s', since='s', note_id='n', budget=50000)
    night.tokens, night.peak = 27000, 10900
    before = health_counters.snapshot().get('mind_incomplete.first', 0)
    assert await Consolidation._call(runner, night, task='first', system='s', user='u', schema={},
                                     max_output_tokens=1200) is None
    assert night.tokens == 37892 and not night.exhausted and not night.counts.get('budget_stops')
    assert night.errors == ['first: RuntimeError']
    assert health_counters.snapshot()['mind_incomplete.first'] == before + 1
    assert await Consolidation._call(runner, night, task='second', system='s', user='u', schema={},
                                     max_output_tokens=1200) == {'ok': True}
    assert calls == ['first', 'second'] and night.tokens == 39792


async def test_length_retry_respects_remaining_context(monkeypatch):
    import math
    prompt = 'Neutral context. ' * 1000
    room = math.ceil(PROMPT_MARGIN * (estimate_tokens(prompt) + 8))
    router = make_router(pool_config('http://127.0.0.1:9', contextTokens=room + 600,
                                    maxTokens=8192, reasoningTokens=100))
    caps, approvals = [], []

    async def endpoint(snapshot, binding, call):
        return await call('127.0.0.1')

    async def complete(**kwargs):
        cap = kwargs['max_output_tokens']
        caps.append(cap)
        assert cap <= 600
        finish = 'length' if len(caps) == 1 else 'stop'
        content = '' if finish == 'length' else '{"ok": true}'
        return SimpleNamespace(content=content, usage={'prompt_tokens': room, 'completion_tokens': cap,
            'total_tokens': room + cap}, latency_ms=1,
            raw=SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish,
                message=SimpleNamespace(content=content))]))

    async def budget(cap, spent):
        approvals.append(cap)
        return True

    monkeypatch.setattr(router, '_at_endpoint', endpoint)
    monkeypatch.setattr(router, '_litellm_call', complete)
    result = await router.complete([{'role': 'user', 'content': prompt}], context={
        'function_role': 'reasoning', 'max_output_tokens': 40, 'allow_fallback': False,
        'length_retry_budget': budget})
    assert result.content == '{"ok": true}'
    assert caps == [140, 600] and approvals == [600]


@pytest.mark.parametrize('watch,key', [({'notice_command': 'PRIVATE_VALUE'}, 'watch.notice_command'),
                                      ({'enabled': 'PRIVATE_VALUE'}, 'watch.enabled'),
                                      ('PRIVATE_VALUE', 'watch')])
def test_bad_watch_is_isolated_and_warning_names_only_key(watch, key, monkeypatch, caplog):
    monkeypatch.setattr(config, '_watch_warnings', set())
    data = config._merge(config.DEFAULTS, {'watch': watch})
    for _ in range(2):
        assert config.validate(deepcopy(data))['watch'] == config.DEFAULTS['watch']
    assert sum(key in record.message for record in caplog.records) == 1
    assert 'PRIVATE_VALUE' not in caplog.text
    with pytest.raises(config.ConfigError, match=key) as raised:
        config.validate(deepcopy(data), strict_watch=True)
    assert 'PRIVATE_VALUE' not in str(raised.value)


def test_backup_preserves_existing_directories_and_private_exec(tmp_path):
    state = tmp_path / 'state'
    state.mkdir()
    output = tmp_path / 'output'
    output.mkdir(mode=0o755)
    output.chmod(0o755)
    archive = create_full_backup(state, output, include_vectors=False)
    assert output.stat().st_mode & 0o777 == 0o755
    new_output = tmp_path / 'new-output'
    create_full_backup(state, new_output, include_vectors=False)
    assert new_output.stat().st_mode & 0o777 == 0o700
    restored = tmp_path / 'restore'
    restored.mkdir(mode=0o750)
    restored.chmod(0o750)
    restore_full_backup(archive, restored)
    assert restored.stat().st_mode & 0o777 == 0o750
    source = tmp_path / 'scripts'
    source.mkdir()
    (source / 'executable').write_text('#!/bin/sh\n')
    (source / 'executable').chmod(0o755)
    (source / 'data').write_text('data')
    destination = tmp_path / 'copy'
    _copy_private_tree(source, destination)
    assert destination.stat().st_mode & 0o777 == 0o700
    assert (destination / 'executable').stat().st_mode & 0o777 == 0o700
    assert (destination / 'data').stat().st_mode & 0o777 == 0o600
    destination.chmod(0o750)
    _copy_private_tree(source, destination)
    assert destination.stat().st_mode & 0o777 == 0o750
