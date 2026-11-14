"""Waiting starts with the first unsummarised turn in the bounded window."""
from datetime import timedelta

from colonymind.mind.consolidate import TASK_EPISODE
from test_mind_consolidate import CONTACT
from test_pr_mind_episodes import fx, seed, session, summaries


async def test_continued_session_keeps_its_place_until_summarised(fx):
    seed(fx, 1)
    expected = [
        [f'other-0-{index}' for index in range(8)],
        ['active', *(f'other-1-{index}' for index in range(7))],
        ['other-1-7', *(f'other-2-{index}' for index in range(7))],
        ['other-2-7', 'active', *(f'other-3-{index}' for index in range(6))],
    ]
    for day, wanted in enumerate(expected):
        for index in range(8):
            session(fx, f'other-{day}-{index}', fx.now - timedelta(hours=3, minutes=-index))
        session(fx, 'active', fx.now - timedelta(minutes=30), prefix=f'day-{day}-')
        first_call = len(fx.router.calls)
        night = await fx.mind.consolidate()
        sent = [messages[-1]['content'].split(' in session ')[1].split(' (')[0]
                for messages, context in fx.router.calls[first_call:] if context['task'] == TASK_EPISODE]
        assert sent == wanted
        assert sorted(name for name, key in summaries(fx) if f':{night["local_date"]}:' in key) == sorted(wanted)
        assert night['counts']['episodes'] == 8 and night['counts']['episodes_pending'] == 1
        fx.shift(days=1)
        fx.restart()


def test_waiting_order_ignores_turns_before_the_window_and_breaks_ties(fx):
    session(fx, 'outside', fx.now - timedelta(days=8), prefix='old-')
    session(fx, 'outside', fx.now - timedelta(hours=1), prefix='new-')
    session(fx, 'z-tied', fx.now - timedelta(hours=2))
    session(fx, 'a-tied', fx.now - timedelta(hours=2))
    fx.turn('active-first', CONTACT, 'active', 'First question?', at=fx.now - timedelta(hours=3))
    session(fx, 'active', fx.now - timedelta(minutes=10), prefix='later-')
    c = fx.mind.consolidation
    rows = c._sessions(fx.now, c.local_date(fx.now), fx.now - timedelta(days=30))
    assert [row['session_id'] for row in rows] == ['active', 'a-tied', 'z-tied', 'outside']
    assert next(row['turns'] for row in rows if row['session_id'] == 'outside') == 3
