"""Outside source routes require the owner's credential and scope all reads."""
import pytest
from httpx import ASGITransport, AsyncClient

from colonymind.api.routers import mind as mind_router
from test_mind_loop import AUTH, OWNER, CONTACT, pinned_clock  # noqa: F401
from test_mind_observations_sources import fx, observation  # noqa: F401


async def test_owner_posts_separate_source_and_reads_it_but_contact_view_is_empty(fx):
    async with AsyncClient(transport=ASGITransport(app=fx.app()), base_url='http://test', headers=AUTH) as client:
        response = await client.post('/v1/mind/observations/sources/sensor',
            json={'source': 'sensor', 'observations': [observation(fx)]})
        assert response.status_code == 200 and response.json()['accepted'] == 1
        await client.post('/v1/mind/observations', json={'stale_tasks': []})
        own = await client.get('/v1/mind/observations')
        assert own.status_code == 200 and len(own.json()['observations']) == 1
        other = await client.get('/v1/mind/observations', params={'viewer': CONTACT})
        assert other.json() == {'observations': [], 'sources': []}
        mismatch = await client.post('/v1/mind/observations/sources/sensor',
            json={'source': 'other', 'observations': []})
        assert mismatch.status_code == 422
    mind_router.set_mind(None)


async def test_guest_cannot_read_or_post_even_if_it_claims_the_owner(fx):
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(mind_router.router)
    mind_router.set_mind(fx.mind)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
            response = await client.post('/v1/mind/observations/sources/sensor',
                json={'source': 'sensor', 'observations': [observation(fx)]})
            assert response.status_code == 403
            assert (await client.get('/v1/mind/observations', params={'viewer': OWNER})).json() == {
                'observations': [], 'sources': []}
    finally:
        mind_router.set_mind(None)


async def test_off_posts_are_ignored_and_reported_without_observation_content(fx):
    fx.mind.faculties['senses'] = False
    async with AsyncClient(transport=ASGITransport(app=fx.app()), base_url='http://test', headers=AUTH) as client:
        posted = await client.post('/v1/mind/observations/sources/sensor',
            json={'source': 'sensor', 'observations': [observation(fx)]})
        assert posted.json()['ignored'] == 1
        fx.shift(seconds=1)
        report = (await client.get('/v1/mind/observations')).json()
        assert report['enabled'] is False and report['observations'] == []
        assert report['sources'] == [{'source': 'sensor', 'surprises': 0, 'concerns': 0,
                                     'messages': 0, 'ignored': 1}]
    mind_router.set_mind(None)
