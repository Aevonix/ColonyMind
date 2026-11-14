"""Which summary of a session the lookup takes: the latest date in the row id that is not forgotten.

The lookup reads a session's summaries through their ids, so the date in the id orders them, not the time
they were stored and not the order they were inserted in; a summary under any other id is not seen. A row
in the session's id range that names another session, or that belongs to another contact, is not taken.
"""
from datetime import timedelta

import pytest

from colonymind.memory import curated
from colonymind.mind.consolidate import TASK_EPISODE
from test_mind_consolidate import CONTACT
from test_pr_mind_episodes import ANCHOR, fx, seed, session, summaries


def summary(fx, name, date, at, digest=''):
    """One summary of the session, written by the product's writer at ``at``; returns its row id."""
    now, fx.now = fx.now, at
    event = f'episode:{name}:{date}' + (f':{digest}' if digest else '')
    assert fx.mind.consolidation.autobiography.record(event, 'episode_summary', f'The answer of {date}.',
                                                      contact_id=CONTACT, session=name)
    fx.now = now
    return f'mind:{event}:episode_summary'


def lookup(fx, monkeypatch, days=1):
    """One ``_sessions`` call: its candidates, and the rows it asked about being forgotten, in order."""
    asked = []
    original = curated.is_forgotten

    def counted(ledger, ref, **kwargs):
        asked.append(ref)
        return original(ledger, ref, **kwargs)

    c = fx.mind.consolidation
    with monkeypatch.context() as patch:
        patch.setattr(curated, 'is_forgotten', counted)
        candidates = c._sessions(fx.now, c.local_date(fx.now), fx.now - timedelta(days=days))
    return [(row['session_id'], row['summarised_tonight'], row['waiting_at']) for row in candidates], asked


def disagreeing(fx):
    """Three summaries of ``long`` whose insertion order, stored time and id date name three different rows."""
    now, c = fx.now, fx.mind.consolidation
    today = summary(fx, 'long', c.local_date(now), now - timedelta(hours=6))                    # stored first
    summary(fx, 'long', c.local_date(now - timedelta(days=2)), now - timedelta(hours=2))        # stored last
    yesterday = summary(fx, 'long', c.local_date(now - timedelta(days=1)), now - timedelta(hours=4))  # inserted last
    session(fx, 'long', now - timedelta(hours=5), prefix='early-')
    session(fx, 'long', now - timedelta(hours=3), prefix='late-')
    return today, yesterday


def test_lookup_takes_the_latest_date_in_the_id_not_the_stored_time_or_the_insertion_order(fx, monkeypatch):
    today, _ = disagreeing(fx)
    # By stored time the session would be done; by insertion order it would wait since the late turns only.
    assert lookup(fx, monkeypatch) == (
        [('long', True, (fx.now - timedelta(hours=5)).isoformat())], [today])


def test_a_forgotten_newest_summary_gives_way_to_the_next_date_in_the_id(fx, monkeypatch):
    today, yesterday = disagreeing(fx)
    curated.forget_sources(fx.ledger, CONTACT, source_ids=[today])
    assert lookup(fx, monkeypatch) == (
        [('long', False, (fx.now - timedelta(hours=3)).isoformat())], [today, yesterday])


@pytest.mark.parametrize('digest, plain_taken', [
    ('0123456789abcdef', True), ('e123456789abcdef', True), ('f123456789abcdef', False)])
async def test_two_standing_summaries_of_one_date_the_plain_id_is_taken_unless_the_digest_starts_with_f(
        fx, monkeypatch, digest, plain_taken):
    now, date, old = fx.now, fx.mind.consolidation.local_date(fx.now), seed(fx, 1)
    session(fx, 'twice', now - timedelta(hours=5), prefix='early-')
    plain = summary(fx, 'twice', date, now - timedelta(hours=4))
    session(fx, 'twice', now - timedelta(hours=3), prefix='late-')
    second = summary(fx, 'twice', date, now - timedelta(hours=2), digest)
    late = (now - timedelta(hours=3)).isoformat()
    assert lookup(fx, monkeypatch) == (([('twice', True, late)], [plain]) if plain_taken else ([], [second]))
    first = await fx.mind.consolidate()
    assert fx.router.tasks().count(TASK_EPISODE) == 0 and first['counts']['episodes'] == 0
    if not plain_taken:
        # The later summary is taken: the session is done.
        assert 'episodes_pending' not in first['counts'] and 'episodes' in first['done']
        assert fx.mind.mind_state.get(ANCHOR)['updated_at'] == now.isoformat()
        return
    # The earlier summary is taken: the turns after it wait as backlog under the same date, without a call.
    assert first['counts']['episodes_pending'] == 1 and 'episodes: backlog' in first['errors']
    assert 'episodes' not in first['done'] and fx.mind.mind_state.get(ANCHOR)['updated_at'] == old.isoformat()
    fx.shift(days=1)
    fx.restart()
    following = await fx.mind.consolidate()
    # The next night summarises the session once more, under its own date; that summary is then the latest.
    assert following['counts']['episodes'] == 1 and 'episodes' in following['done']
    assert fx.router.tasks().count(TASK_EPISODE) == 1
    again = f'mind:episode:twice:{following["local_date"]}:episode_summary'
    assert sorted(summaries(fx)) == sorted([('twice', plain), ('twice', second), ('twice', again)])
    assert lookup(fx, monkeypatch, days=3) == ([], [again])


@pytest.mark.parametrize('event', ['planted-summary', 'episode:other:{date}'])
async def test_a_summary_row_under_another_id_is_not_found(fx, monkeypatch, event):
    now, c = fx.now, fx.mind.consolidation
    session(fx, 'planted', now - timedelta(hours=3))
    fx.now = now - timedelta(hours=1)
    assert c.autobiography.record(event.format(date=c.local_date(now)), 'episode_summary', 'A planted answer.',
                                  contact_id=CONTACT, session='planted')
    fx.now = now
    # The row says it is a summary of the session, later than its turns, but its id is not the session's.
    assert lookup(fx, monkeypatch) == ([('planted', False, (now - timedelta(hours=3)).isoformat())], [])
    night = await fx.mind.consolidate()
    assert night['counts']['episodes'] == 1 and fx.router.tasks().count(TASK_EPISODE) == 1
    assert ('planted', f'mind:episode:planted:{night["local_date"]}:episode_summary') in summaries(fx)


def test_a_summary_of_a_session_whose_id_continues_this_one_is_not_taken(fx, monkeypatch):
    now, date = fx.now, fx.mind.consolidation.local_date(fx.now)
    session(fx, 'parent', now - timedelta(hours=3))
    session(fx, 'parent:child', now - timedelta(hours=3), prefix='child-')
    child = summary(fx, 'parent:child', date, now - timedelta(hours=1))
    # The child's summary lies in the id range of ``parent`` and is later than its turns: only the session
    # that the row names keeps ``parent`` waiting. The one row asked about is asked for the child.
    assert child.startswith('mind:episode:parent:')
    assert lookup(fx, monkeypatch) == ([('parent', False, (now - timedelta(hours=3)).isoformat())], [child])


def test_a_summary_of_the_same_session_id_under_another_contact_is_not_taken(fx, monkeypatch):
    now, c, other = fx.now, fx.mind.consolidation, 'p-03'
    session(fx, 'shared', now - timedelta(hours=3))
    for turn in range(3):
        fx.turn(f'theirs-{turn}', other, 'shared', f'Their question {turn}?', at=now - timedelta(hours=5))
    fx.now = now - timedelta(hours=1)
    assert c.autobiography.record(f'episode:shared:{c.local_date(now)}', 'episode_summary', 'Their answer.',
                                  contact_id=other, session='shared')
    fx.now = now
    theirs = f'mind:episode:shared:{c.local_date(now)}:episode_summary'
    # The row carries this session's id and is later than its turns: only the contact that the row belongs
    # to keeps this contact's session waiting. The one row asked about is asked for the other contact.
    assert lookup(fx, monkeypatch) == ([('shared', False, (now - timedelta(hours=3)).isoformat())], [theirs])
