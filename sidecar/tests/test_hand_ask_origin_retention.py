"""A verified owner request outlives controller cleanup, without trusting ask text."""
from contextlib import closing
import sqlite3

import pytest

from test_ask_pause import (conversation, fx, last_verdict, ready_ids, request_hand_ask)
from test_mind_loop import pinned_clock  # noqa: F401


def retire_controller(fx):
    with closing(sqlite3.connect(fx.state / 'hands.db')) as db:
        db.execute('DELETE FROM hands')
        db.execute('DELETE FROM tasks')
        db.commit()


@pytest.mark.parametrize('platform', ['rcs', 'whatsapp', 'sms', 'api'])
async def test_verified_owner_ask_survives_review_cleanup_and_restart(fx, platform):
    message, _ = await request_hand_ask(fx, platform=platform)
    assert message.id in await ready_ids(fx)
    first = last_verdict(fx, message)
    retire_controller(fx)
    fx.shift(seconds=30)
    conversation(fx, 'parent-review', said='Internal hand completion', reply='The hand needs approval.')
    fx.shift(seconds=6)
    fx.restart()
    for _ in range(5):
        assert message.id in await ready_ids(fx)
        assert last_verdict(fx, message)['id'] == first['id']
        fx.shift(seconds=1)


async def test_unverified_ask_cannot_gain_exemption_after_controller_cleanup(fx):
    message, _ = await request_hand_ask(fx)
    retire_controller(fx)
    assert message.id not in await ready_ids(fx)
    assert last_verdict(fx, message)['stage'] == 'pause'


async def test_late_owner_source_releases_an_existing_conversational_pause(fx):
    message, _ = await request_hand_ask(fx)
    with closing(fx.ledger._connect()) as db:
        db.execute("DELETE FROM turn_sources WHERE turn_id='owner-request'")
        db.commit()
    conversation(fx, 'other-reply')
    fx.shift(seconds=6)
    assert message.id not in await ready_ids(fx)
    conversation(fx, 'owner-request')
    fx.shift(seconds=6)
    assert message.id in await ready_ids(fx)
    assert last_verdict(fx, message)['inputs']['message']['owner_started'] is True


@pytest.mark.parametrize('change', ['missing', 'guest', 'session', 'no_user'])
async def test_retained_origin_still_requires_valid_owner_source(fx, change):
    message, _ = await request_hand_ask(fx)
    assert message.id in await ready_ids(fx)
    retire_controller(fx)
    with closing(fx.ledger._connect()) as db:
        if change == 'missing':
            db.execute("DELETE FROM turn_sources WHERE turn_id='owner-request'")
        elif change == 'guest':
            db.execute("UPDATE turn_sources SET contact_id='guest' WHERE turn_id='owner-request'")
        elif change == 'session':
            db.execute("UPDATE turn_sources SET session_id='unrelated' WHERE turn_id='owner-request'")
        else:
            db.execute("UPDATE turn_sources SET messages_json='[]' WHERE turn_id='owner-request'")
        db.commit()
    conversation(fx, 'another-turn')
    fx.shift(seconds=6)
    assert message.id not in await ready_ids(fx)
    assert last_verdict(fx, message)['stage'] == 'pause'
