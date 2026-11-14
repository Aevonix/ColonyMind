"""The five reproductions from the owner-continuity review, with audience and paging coverage."""
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4
import time

from httpx import ASGITransport, AsyncClient
import pytest

from test_context_packet import wired, app_with_key, assemble_body
from test_owner_continuity import continuity, brief, needs_hermes
from test_colonymind_memory_provider import (
    provider_mod, _privacy_posture, _instance, _make_provider, _FakeHttpx, _RESOLVE,
    _TEMPORAL, _ASSEMBLE, _install_session_context,
)
from onekey import KEY


def marker():
    return 'marker-' + str(uuid4())


def test_audit_group_greeting(provider_mod, monkeypatch):
    secret, sender, chat, session = [marker() for _ in range(4)]
    fake = _FakeHttpx(routes={_RESOLVE: {'contact_id': 'cid-base'},
                             _TEMPORAL: {'continuity': {'body': secret}}})
    provider = _make_provider(provider_mod, fake, monkeypatch)
    _install_session_context(monkeypatch)(platform='telegram', sender=sender, chat=chat)
    assert not provider._owner_private_turn(session, provider._contact_id)
    result = provider.turn_clock(session_id=session, message='hi')
    assert secret not in result
    assert result == provider._clock_line()
    assert not any(call['url'].endswith(_TEMPORAL[1]) for call in fake.requests)


@pytest.mark.asyncio
@pytest.mark.parametrize('metadata', [
    {'chat_type': 'group'}, {'chat_type': 'shared'}, {},
    {'chat_type': 'group', 'owner_private_turn': True},
    {'owner_private_turn': True, 'run': 'cron'},
])
async def test_audit_group_assembly(continuity, metadata):
    owner, _, log, _, now = continuity
    secret = marker()
    log.log(owner, channel='whatsapp:' + marker(), summary=secret, ts=now.isoformat())
    body = assemble_body(owner, query=marker())
    body['context'].update(channel_id='telegram:' + marker(),
                           metadata={**metadata, 'chat_id': marker()})
    async with AsyncClient(transport=ASGITransport(app=app_with_key()), base_url='http://test') as client:
        response = await client.post('/v1/host/context/assemble', json=body,
                                     headers={'Authorization': 'Bearer ' + KEY})
    assert response.status_code == 200, response.text
    assert secret not in str(response.json()['sections'])


@needs_hermes
def test_audit_hook_total_deadline(provider_mod, monkeypatch):
    import colonymind_memory
    from agent.subagent_lifecycle import bind_subagent_parent
    fake = _FakeHttpx(routes={_RESOLVE: {'contact_id': 'cid-base'}}, delay=2)
    provider = _make_provider(provider_mod, fake, monkeypatch)
    monkeypatch.setattr(colonymind_memory, 'ColonyMindMemoryProvider', type(provider))
    hooks = []
    colonymind_memory.register(SimpleNamespace(
        register_memory_provider=lambda provider: None,
        register_hook=lambda name, callback: hooks.append(callback)))
    start = time.monotonic()
    session = marker()
    agent = SimpleNamespace(session_id=session, _memory_manager=SimpleNamespace(providers=[provider]))
    with bind_subagent_parent(agent):
        result = hooks[0](user_message='hi', platform='telegram', sender_id=marker(), session_id=session)
    assert time.monotonic() - start < 1.8
    assert result == {'context': provider._clock_line()}
    assert fake.timeouts and all(0 < timeout <= 1.5 for timeout in fake.timeouts)


@pytest.mark.asyncio
async def test_audit_handle_from_outside_window(continuity):
    owner, _, log, _, now = continuity
    handle = marker()
    log.log(owner, channel='whatsapp:' + handle, summary=marker(),
            ts=(now - timedelta(hours=73)).isoformat())
    log.log(owner, channel='whatsapp', summary='Contact @' + handle, ts=now.isoformat())
    text = (await brief(owner))['continuity']['body']
    assert handle not in text


def test_audit_index_and_bounded_read(continuity):
    owner, _, log, _, now = continuity
    channel = 'whatsapp:' + marker()
    data = [(marker(), owner, channel, 'in', marker(), (now - timedelta(minutes=i)).isoformat())
            for i in range(13000)]
    log._conn.executemany(
        'INSERT INTO communications(id,contact_id,channel,direction,summary,ts) VALUES(?,?,?,?,?,?)', data)
    log._conn.commit()
    sql = []
    log._conn.set_trace_callback(sql.append)
    rows = log.history(owner, limit=None, since=(now - timedelta(hours=72)).isoformat(),
                       conversation_only=True)
    query = next(q for q in sql if q.startswith('SELECT id,contact_id,source_lineage_json'))
    plan = [tuple(row) for row in log._conn.execute('EXPLAIN QUERY PLAN ' + query)]
    assert len(rows) <= 100
    assert 'LIMIT 100' in query and 'julianday' not in query
    assert any('idx_comms_contact' in str(row) and 'ts>' in str(row) for row in plan)
    assert not any('TEMP B-TREE' in str(row) for row in plan)
    sql.clear()
    elsewhere = log.elsewhere_history(owner, 'companion',
        since=(now - timedelta(hours=72)).isoformat(), until=now.isoformat())
    assert len(elsewhere) == 6
    selection = next(q for q in sql if q.startswith('SELECT id,ts,rowid AS position'))
    selection_plan = list(log._conn.execute('EXPLAIN QUERY PLAN ' + selection))
    assert 'LIMIT 100' in selection and 'julianday' not in selection
    assert any('idx_comms_contact' in str(tuple(row)) and 'ts>' in str(tuple(row))
               for row in selection_plan)
    assert not any('TEMP B-TREE' in str(tuple(row)) for row in selection_plan)


@pytest.mark.asyncio
async def test_temporal_requires_explicit_private_audience(continuity):
    owner, _, log, _, now = continuity
    secret = marker()
    log.log(owner, channel='whatsapp', summary=secret, ts=now.isoformat())
    async with AsyncClient(transport=ASGITransport(app=app_with_key()), base_url='http://test') as client:
        response = await client.get('/v1/host/context/temporal',
            params={'contact_id': owner, 'channel': 'telegram'}, headers={'Authorization': 'Bearer ' + KEY})
    assert response.status_code == 200, response.text
    assert secret not in str(response.json()) and 'continuity' not in response.json()
    assert secret in (await brief(owner))['continuity']['body']


@pytest.mark.parametrize('private', [False, True])
def test_assembly_preserves_existing_private_turn_verdict(provider_mod, monkeypatch, private):
    secret = marker()
    fake = _FakeHttpx(routes={_RESOLVE: {'contact_id': 'cid-base'}, _ASSEMBLE: {'sections': [
        {'id': 'colonymind-owner-continuity', 'body': secret}]}})
    provider = _make_provider(provider_mod, fake, monkeypatch)
    verdicts = []
    def verdict(session, contact):
        verdicts.append((session, contact))
        return private
    monkeypatch.setattr(provider, '_owner_private_turn', verdict)
    sections, _ = provider._assemble(marker(), session_id='session', contact_id='cid-base')
    call = next(call for call in fake.requests if call['url'].endswith(_ASSEMBLE[1]))
    assert verdicts and call['json']['context']['metadata']['owner_private_turn'] is private
    assert (secret in str(sections)) is private


@pytest.mark.asyncio
async def test_summary_uses_central_redactor_without_window_handles(continuity, wired, monkeypatch):
    from colonymind import redact
    owner, _, log, _, now = continuity
    bare = marker()
    handles = AsyncMock(return_value=[SimpleNamespace(address=bare)])
    monkeypatch.setattr(wired, 'get_handles', handles)
    values = ['+1 (555) 123-4567', 'someone@example.test', '555123456789@lid',
              '555123456789@s.whatsapp.net', marker() + ':3@lid', '@' + marker(), bare]
    calls = []
    original = redact.redact_sensitive_text
    def redactor(text, **kwargs):
        calls.append(kwargs)
        return original(text, **kwargs)
    monkeypatch.setattr('colonymind.contacts.continuity.redact_sensitive_text', redactor)
    log.log(owner, channel='whatsapp', summary='Contact ' + ' '.join(values), ts=now.isoformat())
    text = (await brief(owner))['continuity']['body']
    assert calls and all(value not in text for value in values)
    handles.assert_awaited_once_with(owner)


from colonymind.memory import curated
@pytest.mark.asyncio
async def test_paging_skips_current_platform_and_erased_pages(continuity):
    owner, _, log, ledger, now = continuity
    source, secret = (marker(), marker())
    ledger.record_source(source, contact_id=owner, session_id=marker(), messages=[{'role': 'user', 'content': secret}], derive_claims=False)
    lineage, _ = log.source_input(source, owner)
    for i in range(105):
        log.log(owner, channel='\t Companion \xa0:' + marker(), summary=marker(), ts=now.isoformat())
        log.log(owner, channel='whatsapp:' + marker(), summary=secret, ts=(now - timedelta(seconds=1)).isoformat(), source_lineage=lineage)
    texts = [marker() for _ in range(6)]
    for i, text in enumerate(texts):
        log.log(owner, channel='\t TELEGRAM \xa0:' + marker(), summary=text, ts=(now - timedelta(minutes=6 - i)).isoformat())
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger)
    curated.forget_sources(ledger, owner, source_ids=[source])
    assert historical_bytes(ledger) == _native_before_withdrawal
    text = (await brief(owner))['continuity']['body']
    assert secret not in text and all((value in text for value in texts))
    assert text.count('Telegram,') == 6
