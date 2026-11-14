"""Claim-dependent plans have no routes after commitment-work retirement."""
import pytest
from httpx import ASGITransport, AsyncClient

from colonymind.server import create_app
from colonymind.api.schemas.host import TurnSyncRequest, TurnSyncResponse
from onekey import AUTH, KEY


def test_plan_routes_and_transport_media_are_removed():
    routes = {(method.upper(), path) for path, methods in create_app().openapi()['paths'].items()
              for method in methods}
    for suffix in ('bind-plan', 'verify-plan', 'check-plan'):
        assert ('POST', '/v1/host/temporal-followups/{wait_id}/' + suffix) not in routes
    assert ('POST', '/v1/host/temporal-followups') not in routes
    assert ('GET', '/v1/host/temporal-followups/{wait_id}') in routes
    assert ('POST', '/v1/host/temporal-followups/{wait_id}/change') in routes
    assert ('PUT', '/v2/host/turns/source-media/document/{turn_id}') in routes
    assert 'transport_media' not in TurnSyncRequest.model_fields
    assert 'transport_media' not in TurnSyncResponse.model_fields


@pytest.mark.asyncio
@pytest.mark.parametrize('suffix', ['', '/wait/bind-plan', '/wait/verify-plan', '/wait/check-plan'])
async def test_removed_temporal_routes_return_not_found_or_method_not_allowed(monkeypatch, suffix):
    monkeypatch.setenv('COLONYMIND_API_KEY', KEY)
    async with AsyncClient(transport=ASGITransport(app=create_app()), base_url='http://test') as client:
        response = await client.post('/v1/host/temporal-followups' + suffix, headers=AUTH, json={})
    assert response.status_code in {404, 405}


def test_generic_sync_treats_transport_media_as_an_unknown_field():
    body = {'identity': {'host_id': 'test-host'},
            'context': {'session_id': 'session', 'contact_id': 'owner'},
            'user_message': {'role': 'user', 'content': 'Native input'}}
    ordinary = TurnSyncRequest.model_validate(body)
    retired = TurnSyncRequest.model_validate({**body, 'transport_media': {'caption': 'Old caption'}})
    assert retired.model_dump() == ordinary.model_dump()
    assert retired.user_message.content == 'Native input'
