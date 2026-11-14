"""A failed entry marker belongs to its stage and leaves the night able to finish."""
import sqlite3

from colonymind.mind.consolidate import LAST_KEY, NIGHT_TASKS, STAGE_PREFIX
from test_pr_mind_stage_order import fx, stages


async def test_failed_entry_write_skips_stage_finishes_night_and_retries_first(fx, monkeypatch):
    entered = []
    c = stages(fx, monkeypatch, entered)
    original = c.mind_state.set

    def locked(key, **kwargs):
        if key == STAGE_PREFIX + 'contradictions':
            raise sqlite3.OperationalError('database is locked')
        return original(key, **kwargs)

    monkeypatch.setattr(c.mind_state, 'set', locked)
    night = await c.run(force=True)
    expected = [name for name in NIGHT_TASKS if name != 'contradictions']
    assert entered == night['done'] == expected
    assert night['errors'] == ['contradictions: OperationalError']
    assert c.mind_state.get(STAGE_PREFIX + 'contradictions') is None
    assert all(c.mind_state.get(STAGE_PREFIX + name)['updated_at'] == fx.now.isoformat()
               for name in expected)
    assert c.mind_state.get(LAST_KEY)['text'] == night['local_date']
    row = fx.store.get(night['id'])
    assert row.completed_at is not None and 'contradictions: OperationalError' in row.result
    assert not c.running

    fx.shift(days=1)
    fx.restart()
    entered.clear()
    next_night = await stages(fx, monkeypatch, entered).run(force=True)
    assert entered == next_night['done'] == ['contradictions', *expected]
    assert next_night['errors'] == []
