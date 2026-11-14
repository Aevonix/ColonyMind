"""Interrupted nights resume durable stage and episode progress."""
import asyncio
from contextlib import closing
from datetime import timedelta

import pytest

from colonymind.mind.consolidate import LAST_KEY, NIGHT_TASKS, STAGES
from colonymind.work.night import WorkNight
from test_mind_consolidate import CONTACT, Fixture, OWNER, episode_rows


@pytest.fixture
def fx(tmp_path, monkeypatch):
    monkeypatch.setenv('COLONYMIND_OWNER_CONTACT_ID', OWNER)
    monkeypatch.setenv('COLONYMIND_SOURCE_CLAIMS', 'off')
    fixture = Fixture(tmp_path)
    yield fixture
    fixture.store.close()


@pytest.mark.parametrize('cut', ['budget', 'deadline'])
async def test_completed_stages_survive_a_cut_and_restart(fx, monkeypatch, cut):
    c = fx.mind.consolidation
    c.faculties = {name: True for name in ('consolidation', 'lessons', 'wishes', 'work_records')}
    called = []
    async def stage(name, night, now):
        called.append(name)
        if name == 'lessons':
            if cut == 'budget':
                night.exhausted = True
            else:
                raise asyncio.TimeoutError
    for name in NIGHT_TASKS:
        async def run(night, now, name=name):
            await stage(name, night, now)
        monkeypatch.setattr(c, STAGES[name], run)
    first = await c.run(force=True)
    assert first['done'] == (['narrative'] if cut == 'budget' else [n for n in NIGHT_TASKS if n != 'lessons'])
    assert called == (['narrative', 'lessons'] if cut == 'budget' else list(NIGHT_TASKS))
    # A TimeoutError raised by a stage is an ordinary stage failure; the other stages can finish.
    fx.shift(days=1)
    fx.restart()
    c = fx.mind.consolidation
    c.faculties = {name: True for name in ('consolidation', 'lessons', 'wishes', 'work_records')}
    called.clear()
    for name in NIGHT_TASKS:
        async def run(night, now, name=name):
            called.append(name)
        monkeypatch.setattr(c, STAGES[name], run)
    second = await c.run(force=True)
    expected = list(NIGHT_TASKS[2:] + NIGHT_TASKS[:2]) if cut == 'budget' else list(NIGHT_TASKS)
    assert called == expected
    assert second['done'] == expected


async def test_episode_backlog_uses_legacy_anchor_and_expires_after_seven_days(fx):
    old = fx.now - timedelta(days=3)
    fx.mind.mind_state.set(LAST_KEY, text=old.date().isoformat(), now=old)
    for session in range(10):
        for turn in range(3):
            fx.turn(f't-{session}-{turn}', CONTACT, f's-{session}', 'Slides?', at=old + timedelta(hours=1))
    first = await fx.mind.consolidate()
    assert len(episode_rows(fx)) == 8
    assert 'episodes' not in first['done']
    fx.shift(days=8)
    fx.restart()
    second = await fx.mind.consolidate()
    assert len(episode_rows(fx)) == 8
    assert second['counts']['episodes'] == 0
    assert 'episodes' in second['done']


async def test_whole_night_deadline_keeps_completed_stage(fx, monkeypatch):
    c = fx.mind.consolidation
    called = []
    enabled = ['narrative', 'contradictions', 'digests', 'episodes']
    for name in enabled:
        async def run(night, now, name=name):
            called.append(name)
            if name == 'contradictions':
                await asyncio.sleep(10)
        monkeypatch.setattr(c, STAGES[name], run)
    monkeypatch.setattr('colonymind.mind.consolidate.RUN_DEADLINE_S', .05)
    first = await c.run(force=True)
    assert 'deadline' in first['errors']
    assert called == ['narrative', 'contradictions']
    fx.shift(days=1)
    called.clear()
    for name in enabled:
        async def run(night, now, name=name):
            called.append(name)
        monkeypatch.setattr(c, STAGES[name], run)
    await c.run(force=True)
    assert called == ['digests', 'episodes', 'narrative', 'contradictions']


async def test_exhausted_work_does_not_select_or_backoff(fx, monkeypatch):
    from colonymind.mind.consolidate import Night
    work = WorkNight(fx.ledger, OWNER)
    selected = []
    monkeypatch.setattr(work, '_select', lambda now: selected.append(now) or [])
    night = Night(local_date=fx.now.date().isoformat(), started_at=fx.now.isoformat(), exhausted=True)
    await work.run(night, fx.now, call=None)
    assert selected == []


async def test_first_sighting_keeps_the_initial_episode_window(fx):
    old = fx.now - timedelta(days=30)
    fx.mind.mind_state.set(LAST_KEY, text="", now=old)
    for turn in range(3):
        fx.turn(f'old-{turn}', CONTACT, 'old-session', 'Slides?', at=old + timedelta(hours=1))
    night = await fx.mind.consolidate()
    assert night['counts']['episodes'] == 0
    assert episode_rows(fx) == []
