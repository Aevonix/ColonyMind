"""Episode selection keeps bounded progress without losing continued sessions."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from colonymind.mind.consolidate import LAST_KEY, TASK_EPISODE
from test_mind_consolidate import CONTACT, Fixture, OWNER, episode_rows

ANCHOR = 'consolidation.episodes.last'


@pytest.fixture
def fx(tmp_path, monkeypatch):
    monkeypatch.setenv('COLONYMIND_OWNER_CONTACT_ID', OWNER)
    monkeypatch.setenv('COLONYMIND_SOURCE_CLAIMS', 'off')
    fixture = Fixture(tmp_path, at=datetime(2026, 10, 20, 8, tzinfo=timezone.utc))
    yield fixture
    fixture.store.close()


def session(fx, name, at, *, prefix='', empty=False):
    for turn in range(3):
        fx.turn(f'{prefix}{name}-{turn}', CONTACT, name,
                '' if empty else f'{prefix}Question {turn}?', '' if empty else 'Noted.', at=at)


def seed(fx, days):
    old = fx.now - timedelta(days=days)
    fx.mind.mind_state.set(LAST_KEY, text=old.date().isoformat(), now=old)
    return old


def summaries(fx):
    return [(json.loads(row['messages_json'])[0]['metadata']['session'], row['turn_id'])
            for row in episode_rows(fx)]


async def test_episode_window_never_extends_past_seven_days(fx):
    seed(fx, 30)
    session(fx, 'expired', fx.now - timedelta(days=7, seconds=1))
    session(fx, 'boundary', fx.now - timedelta(days=7))
    session(fx, 'recent', fx.now - timedelta(hours=1))
    night = await fx.mind.consolidate()
    assert {name for name, _ in summaries(fx)} == {'boundary', 'recent'}
    assert night['counts']['episodes'] == 2
    assert 'episodes' in night['done']
    assert fx.mind.mind_state.get(ANCHOR)['updated_at'] == fx.now.isoformat()


async def test_empty_sessions_use_no_places_and_leave_no_backlog(fx):
    seed(fx, 3)
    for index in range(12):
        session(fx, f'empty-{index}', fx.now - timedelta(days=2), empty=True)
    for index in range(8):
        session(fx, f'readable-{index}', fx.now - timedelta(hours=2, minutes=8-index))
    night = await fx.mind.consolidate()
    assert [name for name, _ in summaries(fx)] == [f'readable-{index}' for index in range(8)]
    assert fx.router.tasks().count(TASK_EPISODE) == 8
    assert night['counts']['episodes'] == 8
    assert 'episodes_pending' not in night['counts'] and night['errors'] == []
    assert 'episodes' in night['done']


async def test_backlog_and_continued_session_are_written_under_each_local_night(fx):
    old = seed(fx, 3)
    fx.mind.consolidation.tz = timezone(timedelta(hours=-10))
    for index in range(10):
        session(fx, f's-{index}', old + timedelta(hours=1, minutes=index))
    first = await fx.mind.consolidate()
    assert dict(summaries(fx)) == {
        f's-{index}': f'mind:episode:s-{index}:{first["local_date"]}:episode_summary' for index in range(8)}
    assert first['local_date'] == '2026-10-19'
    assert first['counts']['episodes_pending'] == 1 and 'episodes' not in first['done']
    assert fx.mind.mind_state.get(ANCHOR)['updated_at'] == old.isoformat()
    fx.shift(days=1)
    fx.restart()
    fx.mind.consolidation.tz = timezone(timedelta(hours=-10))
    session(fx, 's-0', fx.now - timedelta(hours=1), prefix='continued-')
    second = await fx.mind.consolidate()
    sent = [messages[-1]['content'] for messages, context in fx.router.calls if context['task'] == TASK_EPISODE]
    assert [text.split(' in session ')[1].split(' (')[0] for text in sent] == [
        *(f's-{index}' for index in range(8)), 's-8', 's-9', 's-0']
    assert 'continued-Question' in sent[-1]
    assert second['counts']['episodes'] == 3 and 'episodes' in second['done']
    assert all(f':{second["local_date"]}:' in key for name, key in summaries(fx)
               if name in {'s-8', 's-9'})
    assert fx.mind.mind_state.get(ANCHOR)['updated_at'] == fx.now.isoformat()


@pytest.mark.parametrize('name', ['prefix', "arbitrary%_:'session"])
async def test_unchanged_sessions_do_not_repeat_and_lookup_is_exact(fx, name):
    session(fx, name, fx.now)
    session(fx, name + ':other', fx.now)
    await fx.mind.consolidate()
    fx.shift(days=1)
    session(fx, name + ':other', fx.now - timedelta(hours=1), prefix='new-')
    night = await fx.mind.consolidate()
    assert night['counts']['episodes'] == 1
    assert [n for n, _ in summaries(fx)].count(name) == 1
    assert [n for n, _ in summaries(fx)].count(name + ':other') == 2


async def test_same_date_continuation_waits_without_advancing_anchor(fx):
    session(fx, 'continued', fx.now)
    first = await fx.mind.consolidate()
    anchor = fx.now.isoformat()
    fx.shift(hours=1)
    session(fx, 'continued', fx.now, prefix='new-')
    second = await fx.mind.consolidate()
    assert second['local_date'] == first['local_date']
    assert second['counts']['episodes'] == 0
    assert second['counts']['episodes_pending'] == 1 and 'episodes' not in second['done']
    assert fx.mind.mind_state.get(ANCHOR)['updated_at'] == anchor
    fx.shift(days=1)
    fx.restart()
    third = await fx.mind.consolidate()
    assert third['counts']['episodes'] == 1 and 'episodes' in third['done']
    assert len(summaries(fx)) == 2


async def test_failed_call_keeps_legacy_anchor_until_retry_finishes(fx):
    old = seed(fx, 4)
    session(fx, 'retry', old + timedelta(hours=1))
    def fail(messages, context):
        raise RuntimeError('episode call failed')
    fx.router.answers[TASK_EPISODE] = fail
    first = await fx.mind.consolidate()
    assert first['errors'] == [f'{TASK_EPISODE}: RuntimeError']
    assert 'episodes' not in first['done']
    assert fx.mind.mind_state.get(ANCHOR)['updated_at'] == old.isoformat()
    fx.shift(days=1)
    fx.restart()
    fx.router.answers[TASK_EPISODE] = {'summary': 'The question was answered.'}
    second = await fx.mind.consolidate()
    assert second['counts']['episodes'] == 1 and 'episodes' in second['done']
    assert fx.router.tasks().count(TASK_EPISODE) == 2
    assert fx.mind.mind_state.get(ANCHOR)['updated_at'] == fx.now.isoformat()


async def test_disabled_episodes_preserve_the_original_legacy_anchor(fx):
    old = seed(fx, 4)
    session(fx, 'waiting', old + timedelta(hours=1))
    c = fx.mind.consolidation
    c.faculties = {'consolidation': False, 'lessons': True, 'work_records': False, 'wishes': False}
    await c.run(force=True)
    assert c.mind_state.get('consolidation.stage.episodes') is None
    assert c.mind_state.get(ANCHOR)['updated_at'] == old.isoformat()
    fx.shift(days=1)
    fx.restart()
    night = await fx.mind.consolidate()
    assert night['counts']['episodes'] == 1
    assert fx.mind.mind_state.get(ANCHOR)['updated_at'] == fx.now.isoformat()
