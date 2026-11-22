"""A failed, exhausted or cancelled stage cannot monopolise later nights."""
import asyncio
from datetime import timedelta

import pytest

from colonymind.mind.consolidate import LAST_KEY, NIGHT_TASKS, STAGES, STAGE_PREFIX
from test_pr_mind_a1 import fx


def stages(fx, monkeypatch, entered, stop=None):
    c = fx.mind.consolidation
    c.faculties = dict.fromkeys(('consolidation', 'lessons', 'work_records', 'wishes'), True)
    for name in NIGHT_TASKS:
        async def run(night, now, name=name):
            entered.append(name)
            if name == 'work':
                if stop == 'budget':
                    night.exhausted = True
                elif stop == 'failure':
                    raise RuntimeError('work failed')
                elif stop == 'deadline':
                    await asyncio.sleep(30)
        monkeypatch.setattr(c, STAGES[name], run)
    return c


@pytest.mark.parametrize('stop', ['budget', 'failure', 'deadline'])
async def test_repeated_cuts_rotate_by_entry_night_across_restarts(fx, monkeypatch, stop):
    monkeypatch.setattr('colonymind.mind.consolidate.RUN_DEADLINE_S', .1)
    first = list(NIGHT_TASKS[:3])
    rotated = list(NIGHT_TASKS[3:]) + first
    expected = [list(NIGHT_TASKS)] * 3 if stop == 'failure' else [first, rotated, first]
    for wanted in expected:
        entered = []
        c = stages(fx, monkeypatch, entered, stop)
        night = await c.run(force=True)
        assert entered == wanted
        assert night['done'] == [name for name in wanted if name != 'work']
        if stop == 'deadline':
            assert night['errors'] == ['deadline']
        elif stop == 'failure':
            assert night['errors'] == ['work: RuntimeError']
        else:
            assert night['exhausted'] and night['errors'] == []
        stamp = fx.now.isoformat()
        fx.shift(days=1)
        fx.restart()
        assert all(fx.mind.mind_state.get(STAGE_PREFIX + name)['updated_at'] == stamp
                   for name in entered)


async def test_legacy_marker_does_not_enter_disabled_stages_or_create_a_cycle(fx, monkeypatch):
    old = fx.now - timedelta(days=2)
    fx.mind.mind_state.set(LAST_KEY, text=old.date().isoformat(), now=old)
    entered = []
    c = stages(fx, monkeypatch, entered)
    c.faculties['lessons'] = c.faculties['work_records'] = False
    await c.run(force=True)
    assert entered == ['narrative', 'wishes', 'contradictions', 'digests', 'pages', 'episodes']
    assert c.mind_state.get(STAGE_PREFIX + 'lessons') is None
    assert c.mind_state.get(STAGE_PREFIX + 'work') is None
    assert c.mind_state.get('consolidation.cycle') is None
    fx.shift(days=1)
    fx.restart()
    entered.clear()
    await stages(fx, monkeypatch, entered).run(force=True)
    assert entered == ['lessons', 'work', 'narrative', 'wishes', 'contradictions', 'digests', 'pages', 'episodes']
    fx.shift(days=1)
    fx.restart()
    entered.clear()
    await stages(fx, monkeypatch, entered).run(force=True)
    assert entered == list(NIGHT_TASKS)
