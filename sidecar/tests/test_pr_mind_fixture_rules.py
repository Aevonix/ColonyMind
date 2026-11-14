"""Test-fixture checks must observe the feedback the routes actually write."""
import pytest

from colonymind.api.routers import host
from test_grounded_ideas import env, generate, mailbox


async def test_unexpected_taste_write_fails_outside_the_route_handler(env):
    row = (await generate(env, mailbox()[:1]))[0]
    with pytest.raises(pytest.fail.Exception, match='unexpected taste write'):
        await host.initiative_feedback(row.id, request=None, body={'direction': 'up', 'surface': 'ideas'})
    # The single allowed write still passes through this same fixture.
    await host.respond_to_initiative(row.id, action='dismissed', details=None, request=None,
                                     mode=None, item_ids=None, itemIds=None, origin=None)
    assert env.store.get(row.id).status == 'cancelled'
    with pytest.raises(pytest.fail.Exception, match='unexpected taste write'):
        host._feedback_store.record('unrelated', 'dismissed', source=row.id)
