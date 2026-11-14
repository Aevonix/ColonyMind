"""The controller binds approval timing to the actual hand budget."""
from datetime import timedelta
import json

import pytest

from test_hands_permits import fx  # noqa: F401
from test_mind_loop import pinned_clock  # noqa: F401
from colonymind.mind.outbox import DueAsk, REMINDER


@pytest.mark.parametrize('wall,elapsed,remaining', [(600, 20, 180), (60, 20, 40), (60, 70, 0)])
async def test_outbox_carries_fixed_hand_review_deadline(fx, wall, elapsed, remaining):
    fixture, store, service, body, _ = fx
    mind = fixture.mind
    now = mind.clock()
    started = now - timedelta(seconds=elapsed)
    await store.db.execute('UPDATE hands SET started_at=?,budgets=?',
                           (started.isoformat(), json.dumps({'wall_seconds': wall})))
    await store.db.commit()
    held = await service.hold('h-000000000001', body, 'hold')
    ask = mind.store.get(held.ask_id)
    await mind._deliver_asks([ask], now)
    messages = mind.outbox.ready()
    assert len(messages) == 1
    timing = messages[0].get('hand_approval_waits')
    assert timing == [{'hand_id': 'h-000000000001', 'held_at': now.isoformat(),
                       'deadline': (now + timedelta(seconds=remaining)).isoformat()}]
    assert await service.hold('h-000000000001', body, 'repeat') == held
    mind._show_ask(ask, 'The held call', key='show', now=now, why='owner asked')
    assert all(message['hand_approval_waits'] == timing for message in mind.outbox.ready())


async def test_later_reminder_keeps_ordinary_chat_pause(fx):
    fixture, _, service, body, _ = fx
    mind = fixture.mind
    held = await service.hold('h-000000000001', body, 'hold')
    ask = mind.store.get(held.ask_id)
    reminder = mind.outbox._queue_ask(DueAsk(ask, REMINDER, own=True), name=None)
    message = next(row for row in mind.outbox.ready() if row['id'] == reminder.id)
    assert 'hand_approval_waits' not in message
