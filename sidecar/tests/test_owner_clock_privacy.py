"""The review's ten privacy reproductions, with home-clock and cache boundary checks."""
from datetime import datetime, timezone
from uuid import uuid4
import pytest
from httpx import ASGITransport, AsyncClient
from test_context_packet import wired, app_with_key, assemble_body
from test_owner_continuity import continuity
from test_colonymind_memory_provider import (provider_mod, _privacy_posture, _instance, _make_provider,
    _FakeHttpx, _RESOLVE, _TEMPORAL, _ASSEMBLE, _install_session_context)
from onekey import KEY

@pytest.mark.asyncio
@pytest.mark.parametrize('private',[False,None])
async def test_temporal_hides_owner_location_without_private_signal(continuity,wired,private):
    owner,_,_,_,now=continuity
    secret='marker-'+str(uuid4())
    row=wired.rows[owner]
    row.timezone='Asia/Tokyo'
    row.location_city=secret
    row.location_reported_at=now.isoformat()
    params={'contact_id':owner,'channel':'telegram'}
    if private is not None:
        params['owner_private_turn']=private
    async with AsyncClient(transport=ASGITransport(app=app_with_key()),base_url='http://test') as client:
        response=await client.get('/v1/host/context/temporal',params=params,headers={'Authorization':'Bearer '+KEY})
    assert response.status_code==200,response.text
    assert secret not in response.text,response.text
    assert 'Asia/Tokyo' not in response.text,response.text
    assert response.json() == {'id': 'temporal-context', 'title': 'Current Time',
        'body': "Nova's time: Saturday 10:14 AM EDT (America/New_York)."}

@pytest.mark.asyncio
@pytest.mark.parametrize('audience,metadata',[
    ('viewer',{'owner_private_turn':False,'chat_type':'group'}),
    ('viewer',{}),
    ('viewer',{'owner_private_turn':True,'chat_type':'shared'}),
    ('viewer',{'owner_private_turn':True,'chat_type':'group'}),
    ('viewer',{'owner_private_turn':True,'run':'cron'}),
    ('viewer',{'owner_private_turn':True,'kanban_task':'synthetic-task'}),
])
async def test_assembly_hides_owner_location_from_shared_turn(continuity,wired,audience,metadata):
    owner,_,_,_,now=continuity
    secret='marker-'+str(uuid4())
    row=wired.rows[owner]
    row.timezone='Asia/Tokyo'
    row.location_city=secret
    row.location_reported_at=now.isoformat()
    body=assemble_body(owner,query='a substantive synthetic request',audience=audience)
    body['context'].update(channel_id='telegram',metadata=metadata)
    async with AsyncClient(transport=ASGITransport(app=app_with_key()),base_url='http://test') as client:
        response=await client.post('/v1/host/context/assemble',json=body,headers={'Authorization':'Bearer '+KEY})
    assert response.status_code==200,response.text
    assert secret not in str(response.json()['sections']),str(response.json()['sections'])
    assert 'Asia/Tokyo' not in str(response.json()['sections'])
    clock = next(section for section in response.json()['sections'] if section['id'] == 'temporal-context')
    assert clock['body'] == "Nova's time: Saturday 10:14 AM EDT (America/New_York)."

@pytest.mark.asyncio
async def test_explicit_shared_person_boundary_is_safe(continuity,wired,monkeypatch):
    owner,_,_,_,now=continuity
    secret='marker-'+str(uuid4())
    row=wired.rows[owner]
    row.timezone='Asia/Tokyo'
    row.location_city=secret
    row.location_reported_at=now.isoformat()
    monkeypatch.setenv('COLONYMIND_SHARED_PERSON_ID','synthetic-shared')
    async with AsyncClient(transport=ASGITransport(app=app_with_key()),base_url='http://test') as client:
        denied=await client.post('/v1/host/context/assemble',
            json=assemble_body(owner,audience='shared'),headers={'Authorization':'Bearer '+KEY})
        response=await client.post('/v1/host/context/assemble',
            json=assemble_body('synthetic-shared',audience='shared'),headers={'Authorization':'Bearer '+KEY})
    assert denied.status_code==403,denied.text
    assert response.status_code==200,response.text
    assert secret not in response.text and 'Asia/Tokyo' not in response.text,response.text

@pytest.mark.parametrize('cached',[False,True])
def test_provider_group_prefetch_hides_location(provider_mod,monkeypatch,cached):
    secret='marker-'+str(uuid4())
    fake=_FakeHttpx(routes={_RESOLVE:{'contact_id':'cid-base'},_ASSEMBLE:{'sections':[]},
        _TEMPORAL:{'body':secret,'owner_timezone':'Asia/Tokyo'}})
    monkeypatch.setenv('COLONYMIND_AGENT_TIMEZONE','Pacific/Honolulu')
    provider=_make_provider(provider_mod,fake,monkeypatch)
    set_turn=_install_session_context(monkeypatch)
    if cached:
        set_turn(platform='cli')
        assert secret in provider.prefetch('synthetic owner request',session_id='private')
        assert provider._temporal_cache_key == ('cid-base', True)
        assert [r for r in fake.requests if r['url'].endswith(_TEMPORAL[1])][-1]['params']['owner_private_turn'] is True
    temporal_calls = len([r for r in fake.requests if r['url'].endswith(_TEMPORAL[1])])
    set_turn(platform='telegram',sender='synthetic-owner',chat='synthetic-group')
    assert not provider._owner_private_turn('group','cid-base')
    result=provider.prefetch('synthetic group request',session_id='group')
    assert secret not in result,result
    assert 'HST (Pacific/Honolulu).' in result and 'Asia/Tokyo' not in result
    assert len([r for r in fake.requests if r['url'].endswith(_TEMPORAL[1])]) == temporal_calls

@pytest.mark.parametrize('kind',['guest_greeting','group_greeting','guest_prefetch'])
def test_learned_owner_zone_does_not_reach_other_audience(provider_mod,monkeypatch,kind):
    from zoneinfo import ZoneInfo
    class HostDatetime(datetime):
        def astimezone(self,tz=None):
            return super().astimezone(tz or ZoneInfo('America/New_York'))
    fake=_FakeHttpx(routes={_TEMPORAL:{'body':'private clock','owner_timezone':'Asia/Tokyo'},
        _ASSEMBLE:{'sections':[]},_RESOLVE:{'contact_id':'guest'}})
    monkeypatch.setenv('COLONYMIND_AGENT_TIMEZONE','Pacific/Honolulu')
    provider=_make_provider(provider_mod,fake,monkeypatch)
    provider._timezone=''
    monkeypatch.setattr(provider_mod,'datetime',HostDatetime)
    monkeypatch.setattr(provider_mod._ttime,'time',lambda:datetime(2026,10,17,14,14,tzinfo=timezone.utc).timestamp())
    set_turn=_install_session_context(monkeypatch)
    set_turn(platform='cli')
    provider._fresh_temporal_block_sync(contact_id='cid-base')
    assert provider._owner_timezone=='Asia/Tokyo'
    if kind=='group_greeting':
        fake.routes[_RESOLVE]={'contact_id':'cid-base'}
    set_turn(platform='telegram',sender='synthetic-sender',chat='synthetic-group' if kind=='group_greeting' else 'synthetic-sender')
    result=(provider.prefetch('synthetic guest question',session_id='guest') if kind=='guest_prefetch'
        else provider.turn_clock(message='hi',session_id='other'))
    assert '11:14 PM JST' not in result,result
    assert "Nova's time: Saturday 4:14 AM HST (Pacific/Honolulu)." in result
    assert 'Asia/Tokyo' not in result and 'location' not in result


@pytest.mark.asyncio
@pytest.mark.parametrize('private', [False, True])
async def test_guest_temporal_ignores_private_signal_and_override(continuity, private):
    _, guest, _, _, _ = continuity
    async with AsyncClient(transport=ASGITransport(app=app_with_key()), base_url='http://test') as client:
        response = await client.get('/v1/host/context/temporal', params={
            'contact_id': guest, 'owner_private_turn': private, 'tz': 'Asia/Tokyo'},
            headers={'Authorization': 'Bearer ' + KEY})
    assert response.status_code == 200, response.text
    assert response.json() == {'id': 'temporal-context', 'title': 'Current Time',
        'body': "Nova's time: Saturday 10:14 AM EDT (America/New_York)."}


@pytest.mark.parametrize('spill', [False, True])
def test_prefetch_and_spill_use_the_requested_sessions_privacy(provider_mod, monkeypatch, spill):
    secret = 'marker-' + str(uuid4())
    sections = [{'id': 'colonymind-memory', 'title': 'Relevant Memories', 'body': 'x' * 1000},
                {'id': 'state', 'title': 'State', 'body': 'y' * 1000}] if spill else []
    fake = _FakeHttpx(routes={_ASSEMBLE: {'sections': sections},
        _TEMPORAL: {'body': secret, 'owner_timezone': 'Asia/Tokyo'}})
    monkeypatch.setenv('COLONYMIND_AGENT_TIMEZONE', 'Pacific/Honolulu')
    provider = _make_provider(provider_mod, fake, monkeypatch)
    _install_session_context(monkeypatch)(platform='cli')
    monkeypatch.setattr(provider, '_owner_private_turn', lambda session, contact: session == 'private')
    assert secret in provider._fresh_temporal_block_sync(contact_id='cid-base', session_id='private')
    monkeypatch.setattr(provider, '_spill_limit', lambda: 500)
    result = provider.prefetch('synthetic shared request', session_id='public')
    assert secret not in result and 'Asia/Tokyo' not in result and 'location' not in result
    assert 'HST (Pacific/Honolulu).' in result
    assert len([r for r in fake.requests if r['url'].endswith(_TEMPORAL[1])]) == 1
    if spill:
        assert 'held back' in result.lower()


def test_temporal_cache_key_includes_privacy_class(provider_mod, monkeypatch):
    fake = _FakeHttpx(routes={_TEMPORAL: {'body': 'fresh private clock'}})
    provider = _make_provider(provider_mod, fake, monkeypatch)
    _install_session_context(monkeypatch)(platform='cli')
    provider._temporal_cache = (provider_mod._ttime.monotonic(), 'wrong privacy class')
    provider._temporal_cache_key = ('cid-base', False)
    result = provider._fresh_temporal_block_sync(contact_id='cid-base')
    assert 'fresh private clock' in result and 'wrong privacy class' not in result
    assert provider._temporal_cache_key == ('cid-base', True)


def test_public_provider_rejects_private_clocks_from_older_assembly(provider_mod, monkeypatch):
    secret = 'marker-' + str(uuid4())
    fake = _FakeHttpx(routes={_RESOLVE: {'contact_id': 'cid-base'}, _ASSEMBLE: {'sections': [
        {'id': 'temporal-context', 'title': 'Legacy clock', 'body': secret + ' Asia/Tokyo'}]}})
    monkeypatch.setenv('COLONYMIND_AGENT_TIMEZONE', 'Pacific/Honolulu')
    provider = _make_provider(provider_mod, fake, monkeypatch)
    _install_session_context(monkeypatch)(platform='telegram', sender='owner', chat='group')
    assert provider._prefetch_sections('synthetic group request', session_id='group') == []
    result = provider.prefetch('synthetic group request', session_id='group')
    assert secret not in result and 'Asia/Tokyo' not in result
    assert 'HST (Pacific/Honolulu).' in result
