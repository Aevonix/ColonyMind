"""Episode history and backlog reads stop as soon as their answer is known."""
from datetime import timedelta

import pytest

from colonymind.memory import curated
from colonymind.mind.consolidate import EPISODES_PER_NIGHT, TASK_EPISODE
from test_mind_consolidate import CONTACT
from test_pr_mind_episodes import ANCHOR, fx, seed, session


@pytest.mark.parametrize('forgotten', [0, 2])
def test_summary_lookup_checks_only_the_newest_surviving_prefix(fx, monkeypatch, forgotten):
    c = fx.mind.consolidation
    ids = []
    for day in range(6):
        session(fx, 'long', fx.now - timedelta(hours=1), prefix=f'day-{day}-')
        event = f'episode:long:{c.local_date(fx.now)}'
        assert c.autobiography.record(event, 'episode_summary', f'The question for day {day} was answered.',
                                     contact_id=CONTACT, session='long',
                                     lineage=[f'day-{day}-long-{turn}' for turn in range(3)])
        ids.append(f'mind:{event}:episode_summary')
        if day < 5:
            fx.shift(days=1)
    if forgotten:
        curated.forget_sources(fx.ledger, CONTACT, source_ids=ids[-forgotten:])
    fx.shift(hours=1)
    session(fx, 'long', fx.now, prefix='continued-')
    checked = []
    original = curated.is_forgotten

    def counted(ledger, ref, **kwargs):
        if isinstance(ref, str) and ref.startswith('mind:episode:'):
            checked.append(ref)
        return original(ledger, ref, **kwargs)

    monkeypatch.setattr(curated, 'is_forgotten', counted)
    candidates = c._sessions(fx.now, c.local_date(fx.now), fx.now - timedelta(days=1))
    assert [(row['session_id'], row['summarised_tonight']) for row in candidates] == [
        ('long', forgotten == 0)]
    assert checked == list(reversed(ids))[:forgotten + 1]


@pytest.mark.parametrize('stop', ['limit', 'exhausted'])
async def test_transcripts_stop_at_the_first_readable_backlog(fx, monkeypatch, stop):
    old = seed(fx, 3)
    c = fx.mind.consolidation
    now = fx.now
    fx.now = now - timedelta(hours=2)
    assert c.autobiography.record(f'episode:deferred:{c.local_date(now)}', 'episode_summary',
                                 'An earlier answer.', contact_id=CONTACT, session='deferred')
    fx.now = now
    session(fx, 'deferred', now - timedelta(hours=1))
    limit = EPISODES_PER_NIGHT if stop == 'limit' else 2
    names = [f'readable-{index:02}' for index in range(16)]
    names[limit:limit] = ['empty-a', 'empty-b']
    for index, name in enumerate(names):
        session(fx, name, now - timedelta(minutes=50-index), empty=name.startswith('empty-'))
    reads = []
    transcript, call = c._transcript, c._call
    calls = 0

    def counted(contact_id, session_id):
        reads.append(session_id)
        return transcript(contact_id, session_id)

    async def exhausting(night, **kwargs):
        nonlocal calls
        answer = await call(night, **kwargs)
        if kwargs['task'] == TASK_EPISODE:
            calls += 1
            if stop == 'exhausted' and calls == limit:
                night.exhausted = True
        return answer

    monkeypatch.setattr(c, '_transcript', counted)
    monkeypatch.setattr(c, '_call', exhausting)
    night = await fx.mind.consolidate()
    assert fx.router.tasks().count(TASK_EPISODE) == limit
    assert night['counts']['episodes'] == limit
    assert reads == ['deferred', *names[:limit + 3]]
    assert night['counts']['episodes_pending'] == 1 and 'episodes' not in night['done']
    assert c.mind_state.get(ANCHOR)['updated_at'] == old.isoformat()
