"""Offline owner recall exit observations through the real host and source hooks."""
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
import threading
import uuid
from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from onekey import KEY
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import host
from colonymind.memory import recall_exit, search, turn_budget
from colonymind.plasticity.learners.query_adapter import RecallRequest
from colonymind.turns import source_vectors
from test_plasticity_query_adapter_driver import runtime  # noqa: F401
from test_recall_query_adapter_live import Pipeline, Store

QUERY = 'private-query-sentinel'
TURN = 'private-turn-sentinel'
SESSION = 'private-session-sentinel'
PASSAGE = 'private-passage-sentinel'
ERROR = 'private-exception-message-sentinel'
OWNER = 'private-owner-sentinel'


@pytest.fixture(autouse=True)
def reset_observer(monkeypatch):
    monkeypatch.setattr(recall_exit, '_debug_at', float('-inf'))
    monkeypatch.setattr(recall_exit, '_debug_pending', False)


@pytest.fixture
async def world(runtime, monkeypatch, tmp_path):
    from colonymind import vector
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(tmp_path))
    monkeypatch.setenv('COLONYMIND_OWNER_CONTACT_ID', OWNER)
    monkeypatch.setattr(host, '_temporal_now', lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))
    monkeypatch.setenv('COLONYMIND_CONTEXT_STATUS', 'off')
    monkeypatch.setenv('COLONYMIND_CONTEXT_SELECTION', 'off')
    for name in ('_contacts_store', '_telemetry', '_reranker', '_context_recall_selector', '_comms_log',
                 '_initiative_store', '_commitment_store', '_preference_learner'):
        monkeypatch.setattr(host, name, None)
    monkeypatch.setattr(host, '_mind', lambda: SimpleNamespace(plasticity=runtime, enabled=False))
    driver = runtime.loaded['recall-test'].driver
    driver.document_identity = asdict(Pipeline.index_identity.document_identity)
    store, pipeline = Store(), Pipeline()
    monkeypatch.setattr(vector, '_store', store)
    monkeypatch.setattr(vector, '_pipeline', pipeline)
    monkeypatch.setattr(source_vectors, 'hydrate', lambda *a, **kw: {
        'turn_id': 'source-private', 'role': 'user', 'scope': 'person', 'content': PASSAGE})
    state = SimpleNamespace(runtime=runtime, driver=driver, store=store, pipeline=pipeline,
                            content=PASSAGE, collect_error=None, select_error=None,
                            before_select=None, collect_wait=False)

    async def collect(ledger, *, query, contact_id, session_id, recall_request=None, **kwargs):
        if state.collect_wait:
            await asyncio.Event().wait()
        if state.collect_error:
            raise state.collect_error
        # Preserve the vector lane's existing optional-failure behavior.
        try:
            await source_vectors.SourceVectors(None, store, pipeline).search(
                query, contact_id=contact_id, session_id=session_id, recall_request=recall_request)
        except Exception:
            pass
        return search.CollectedSources(None, contact_id, session_id, None)

    async def select(*args, **kwargs):
        if state.before_select:
            state.before_select()
        if state.select_error:
            raise state.select_error
        return SimpleNamespace(content=state.content, selected=[], source_refs=[])

    monkeypatch.setattr(search, 'collect_sources', collect)
    monkeypatch.setattr(search, 'select_memory', select)
    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware, api_key=KEY)
    app.include_router(host.router)
    state.app = app
    yield state
    if turn_budget._PENDING:
        await asyncio.gather(*tuple(turn_budget._PENDING))


async def assemble(world, *, query=QUERY, owner=True, turn_id=TURN, wait_ms=None):
    payload = {'identity': {'host_id': 'fixture'},
               'context': {'contact_id': OWNER if owner else 'guest-private',
                           'session_id': SESSION, 'turn_id': turn_id},
               'incoming_message': {'role': 'user', 'content': query}}
    if wait_ms is not None:
        payload['wait_ms'] = wait_ms
    async with AsyncClient(transport=ASGITransport(app=world.app), base_url='http://test',
                           headers={'Authorization': 'Bearer ' + KEY}) as client:
        response = await client.post('/v1/host/context/assemble', json=payload)
    assert response.status_code == 200, response.text
    if turn_budget._PENDING:
        await asyncio.gather(*tuple(turn_budget._PENDING))
    return response.json()


import importlib
import importlib.util
from pathlib import Path
import sys
import types

from test_colonymind_memory_provider import (provider_mod, _make_provider, _FakeHttpx,
                                            _ASSEMBLE, _TEMPORAL)  # noqa: F401


@pytest.fixture
def counter(monkeypatch):
    # The host plugin is deliberately not installed in the sidecar environment.
    package = types.ModuleType('recall_host_test')
    package.__path__ = [str(Path(__file__).resolve().parents[2] / 'plugins/hermes-plugin')]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    name = package.__name__ + '.capture'
    spec = importlib.util.spec_from_file_location(name, Path(package.__path__[0]) / 'capture.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, 'colonymind_capture', SimpleNamespace(recall_exit=module))
    return module


def memory(body=PASSAGE):
    return '<memory-context>\n[System note]\n\n## Relevant Memories\n' + body + '\n</memory-context>'


def request(counter, state, *, fresh=True, existing=False, session=SESSION):
    messages = []
    if existing:
        messages.append({'role': 'user', 'content': memory()})
        messages.append({'role': 'assistant', 'content': 'previous answer'})
    messages.append({'role': 'user', 'content': QUERY + ('\n' + memory() if fresh else '')})
    before = json.dumps(messages).encode()
    counter.pre_api_request(session_id=session, turn_id=state.turn_id, request_messages=messages)
    assert before == json.dumps(messages).encode()
    return messages


def finish(counter, state, session=SESSION):
    counter.post_llm_call(session_id=session, turn_id=state.turn_id)
    counter.drain()


def log_lines(counter, caplog):
    return [r.getMessage() for r in caplog.records if r.name == counter.__name__
            and r.getMessage().startswith('RECALL_EXIT ')]


def check(counter, caplog, code, *, submitted=0, returned=0, existing='none'):
    result = log_lines(counter, caplog)
    assert len(result) == 1, result
    line = result[0]
    assert f'exit={code} ' in line
    for field, value in (('sections_returned', returned), ('sections_submitted', submitted),
                         ('chars_submitted', submitted * len(PASSAGE)), ('existing_vs_new', existing)):
        assert f'{field}={value} ' in line
    for field in ('turn_id=', 'channel=', 'recall_ms=', 'budget_ms=', 'rerank=', 'capture='):
        assert field in line
    for sentinel in (QUERY, SESSION, PASSAGE, ERROR, OWNER, 'source-private'):
        assert sentinel not in line
    assert f'turn_id="{TURN}"' in line  # opaque host id is explicitly allowed now
    return line


async def joined(world, counter, *, fresh=True, existing=False, **kwargs):
    result = await assemble(world, **kwargs)
    state = counter.begin(SESSION, TURN, owner=True, platform='whatsapp')
    counter.update(state, **{k: v for k, v in result['recall_exit'].items() if k != 'turn_id'})
    request(counter, state, fresh=fresh and bool(result['recall_exit']['sections_returned']), existing=existing)
    finish(counter, state)
    return result, state


@pytest.mark.asyncio
@pytest.mark.parametrize('gate', ['no_turn_id', 'not_owner', 'runtime_unavailable', 'no_adapter'])
async def test_create_gates(world, counter, monkeypatch, caplog, gate):
    caplog.set_level(logging.INFO)
    original = RecallRequest.create
    if gate == 'not_owner':
        monkeypatch.setattr(RecallRequest, 'create', lambda runtime, **kw: original(runtime, owner=False, turn_id=kw['turn_id']))
    elif gate == 'runtime_unavailable':
        monkeypatch.setattr(world.runtime, 'enabled', False)
    elif gate == 'no_adapter':
        monkeypatch.setattr(world.runtime, 'find', lambda name: None)
    result, _ = await joined(world, counter, turn_id=None if gate == 'no_turn_id' else TURN)
    check(counter, caplog, 'runtime_unavailable' if gate == 'no_adapter' else gate,
          returned=1, submitted=1, existing='new')
    assert not any(r.name == recall_exit.__name__ and r.getMessage().startswith('RECALL_EXIT ') for r in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['embed', 'search', 'owner_search', 'rerank'])
async def test_exception_stages(world, counter, monkeypatch, caplog, stage):
    caplog.set_level(logging.INFO)
    async def broken(*args, **kwargs):
        raise RuntimeError(ERROR)
    if stage == 'embed':
        monkeypatch.setattr(world.pipeline, 'embed_query', broken)
    elif stage == 'owner_search':
        monkeypatch.setattr(world.store, 'search', broken)
    elif stage == 'search':
        world.collect_error = RuntimeError(ERROR)
    else:
        from colonymind.memory.selection import RecallSelector
        monkeypatch.setenv('COLONYMIND_RECALL_RERANK', 'on')
        async def select(*args, **kwargs):
            await RecallSelector(broken)._reranked(QUERY, [{'id': TURN, 'content': PASSAGE}], 0)
            return SimpleNamespace(content='', selected=[], source_refs=[])
        monkeypatch.setattr(search, 'select_memory', select)
    world.content = ''
    await joined(world, counter)
    line = check(counter, caplog, 'error')
    assert f'stage={stage}' in line and 'exception=RuntimeError' in line
    if stage == 'rerank':
        assert 'rerank=yes' in line
        # Invalid returned scores are caught inside selection, too.
        async def invalid(*a, **kw):
            return []
        with turn_budget.turn(turn_budget.TurnBudget(wait_s=7.5)) as budget:
            recall_exit.begin(True, True, TURN)
            await RecallSelector(invalid)._reranked(QUERY, [{'id': TURN, 'content': PASSAGE}], 0)
            assert budget.recall_exit.exit == 'error'
            assert budget.recall_exit.stage == 'rerank'
            assert budget.recall_exit.exception == 'ValueError'


@pytest.mark.asyncio
async def test_identity_refusal(world, counter, caplog):
    caplog.set_level(logging.INFO)
    world.driver.document_identity = {'different': True}
    world.content = ''
    await joined(world, counter)
    line = check(counter, caplog, 'error')
    assert 'stage=owner_search' in line and 'subcode=identity' in line and 'exception=ValueError' in line


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['embed', 'search', 'owner_search', 'rerank', 'assembly_left', 'recall'])
async def test_own_deadlines(world, counter, monkeypatch, caplog, stage):
    caplog.set_level(logging.INFO)
    async def timeout(*args, **kwargs):
        raise TimeoutError(ERROR)
    kwargs = {}
    if stage == 'embed':
        monkeypatch.setattr(world.pipeline, 'embed_query', timeout)
    elif stage == 'search':
        world.collect_error = TimeoutError(ERROR)
    elif stage == 'owner_search':
        monkeypatch.setattr(source_vectors.producers, 'expired', lambda: True)
    elif stage == 'rerank':
        from colonymind.memory.selection import RecallSelector
        monkeypatch.setenv('COLONYMIND_RECALL_RERANK', 'on')
        async def select(*args, **kw):
            await RecallSelector(timeout)._reranked(QUERY, [{'id': TURN, 'content': PASSAGE}], 0)
            return SimpleNamespace(content='', selected=[], source_refs=[])
        monkeypatch.setattr(search, 'select_memory', select)
    elif stage == 'assembly_left':
        world.collect_wait = True
        kwargs['wait_ms'] = 20
    else:
        with turn_budget.turn(turn_budget.TurnBudget(wait_s=7.5)) as budget:
            recall_exit.begin(True, True, TURN)
            recall_exit.failed(TimeoutError(ERROR), 'recall')
            outcome = recall_exit.snapshot(budget.recall_exit)
        state = counter.begin(SESSION, TURN, owner=True)
        counter.update(state, **{k:v for k,v in outcome.items() if k != 'turn_id'})
        request(counter, state, fresh=False)
        finish(counter, state)
        assert 'stage=recall' in check(counter, caplog, 'deadline')
        return
    world.content = ''
    await joined(world, counter, **kwargs)
    assert f'stage={stage}' in check(counter, caplog, 'deadline')


@pytest.mark.asyncio
async def test_caller_cancel_is_not_deadline(world, counter, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError()
    monkeypatch.setattr(world.pipeline, 'embed_query', cancelled)
    # Production propagates caller cancellation; no change to cancellation behavior.
    with pytest.raises(asyncio.CancelledError):
        with turn_budget.turn(turn_budget.TurnBudget(wait_s=7.5)) as budget:
            recall_exit.begin(True, True, TURN)
            try:
                await source_vectors.SourceVectors(None, world.store, world.pipeline).search(
                    QUERY, contact_id=OWNER, session_id=SESSION)
            finally:
                outcome = recall_exit.snapshot(budget.recall_exit)
    state = counter.begin(SESSION, TURN, owner=True)
    counter.update(state, **{k:v for k,v in outcome.items() if k != 'turn_id'})
    request(counter, state, fresh=False)
    finish(counter, state)
    assert 'stage=embed' in check(counter, caplog, 'cancelled')


@pytest.mark.asyncio
async def test_no_candidates_and_eligibility_zero(world, counter, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(source_vectors, 'hydrate', lambda *a, **kw: None)
    world.content = ''
    await joined(world, counter)
    check(counter, caplog, 'no_candidates')


@pytest.mark.asyncio
@pytest.mark.parametrize('capture', ['c', 'd', 'recorded'])
async def test_capture_is_separate_from_exit(world, counter, monkeypatch, caplog, capture):
    caplog.set_level(logging.INFO)
    if capture == 'c':
        async def omit(body, sections, **kwargs):
            return [s for s in sections if s.id != 'colonymind-memory']
        monkeypatch.setattr(host, '_turn_context_selection', omit)
    elif capture == 'd':
        # Change the joint pin after paired search, before its accepted callback.
        world.before_select = lambda: monkeypatch.setattr(world.runtime, 'pin', lambda *a, **kw:
                                                         SimpleNamespace(closure=SimpleNamespace(id='changed-private')))
    result, state = await joined(world, counter)
    capture_lines = [r.getMessage() for r in caplog.records if r.name == recall_exit.__name__
                     and r.getMessage().startswith('RECALL_CAPTURE ')]
    assert capture_lines == [f'RECALL_CAPTURE turn_id="{TURN}" capture={capture}']
    assert result['recall_exit']['capture'] == ('c' if capture == 'c' else 'pending')
    line = check(counter, caplog, 'no_candidates' if capture == 'c' else 'ok',
          returned=0 if capture == 'c' else 1, submitted=0 if capture == 'c' else 1,
          existing='none' if capture == 'c' else 'new')
    assert 'capture=' + ('c' if capture == 'c' else 'pending') in line
    assert state.exit in {'ok', 'no_candidates'}



@pytest.mark.parametrize('drop', ['prefetch_wait', 'context_fit', 'late_callback'])
def test_host_drops(counter, caplog, drop):
    caplog.set_level(logging.INFO)
    state = counter.begin(SESSION, TURN, owner=True)
    if drop == 'late_callback':
        request(counter, state, fresh=False)
        counter.update(state, sections_returned=1)
    else:
        counter.update(state, sections_returned=1, drop=drop)
        request(counter, state, fresh=False)
    finish(counter, state)
    assert f'drop={drop}' in check(counter, caplog, 'host_drop', returned=1)


@pytest.mark.parametrize('old,new,kind', [(False,True,'new'), (True,False,'existing'), (True,True,'both')])
def test_actual_submission_and_history(counter, caplog, old, new, kind):
    caplog.set_level(logging.INFO)
    state = counter.begin(SESSION, TURN, owner=True, platform='sms')
    counter.update(state, sections_returned=int(new))
    # An earlier attempt lacked memory; later actual submission must win.
    request(counter, state, fresh=False)
    messages = request(counter, state, fresh=new, existing=old)
    # Per-request retries and tool calls are not additional owner turns.
    counter.pre_api_request(session_id=SESSION, turn_id=TURN, request_messages=messages)
    finish(counter, state)
    finish(counter, state)
    check(counter, caplog, 'ok', returned=int(new), submitted=int(old)+int(new), existing=kind)


@pytest.mark.parametrize('first', ['no_turn_id', 'not_owner', 'runtime_unavailable', 'circuit_open', 'review',
                                    'binding_refused', 'error', 'cancelled', 'deadline', 'no_candidates', 'host_drop'])
def test_first_failed_gate_precedence(counter, caplog, first):
    caplog.set_level(logging.INFO)
    state = counter.begin(SESSION, TURN, owner=True)
    # Apply in reverse to prove precedence is gate order, independent of task scheduling.
    for code in reversed(counter.ORDER[counter.ORDER.index(first):]):
        counter.update(state, exit=code, stage='owner_search', exception='ValueError')
    request(counter, state, fresh=False)
    finish(counter, state)
    check(counter, caplog, first)


@pytest.mark.parametrize('gate', ['circuit_open', 'review', 'binding_refused', 'no_turn_id'])
def test_provider_early_gates(counter, provider_mod, monkeypatch, caplog, gate, tmp_path):
    caplog.set_level(logging.INFO)
    monkeypatch.setenv('COLONYMIND_HOME', str(tmp_path / 'instance'))
    provider = _make_provider(provider_mod, _FakeHttpx(), monkeypatch)
    provider.initialize(SESSION, platform='cli')
    if gate != 'no_turn_id':
        provider._turn_ids[SESSION] = TURN
    state = counter.begin(SESSION, TURN, owner=True)
    if gate == 'circuit_open':
        monkeypatch.setattr(provider, '_is_circuit_open', lambda: True)
    elif gate == 'review':
        monkeypatch.setattr(provider, '_review_turn', lambda sid: True)
    elif gate == 'binding_refused':
        monkeypatch.setattr(provider, '_prefetch_contact', lambda *a, **kw: '')
    provider.prefetch(QUERY, session_id=SESSION)
    request(counter, state, fresh=False)
    finish(counter, state)
    check(counter, caplog, gate)


def test_provider_builtin_request_timeout_is_recall_deadline(counter, provider_mod, monkeypatch, caplog, tmp_path):
    caplog.set_level(logging.INFO)
    monkeypatch.setenv('COLONYMIND_HOME', str(tmp_path / 'instance'))
    provider = _make_provider(provider_mod, _FakeHttpx(), monkeypatch)
    provider.initialize(SESSION, platform='cli')
    provider._turn_ids[SESSION] = TURN
    state = counter.begin(SESSION, TURN, owner=True)
    original = provider_mod._request_within

    def expired(timeout, method, url, **kwargs):
        return original(0, method, url, **kwargs)  # Real built-in timeout, before any client/socket.

    monkeypatch.setattr(provider_mod, '_request_within', expired)
    token = provider_mod._RECALL_EXIT.set(state)
    try:
        assert provider._assemble(QUERY, session_id=SESSION, contact_id=provider._contact_id) == ([], None)
    finally:
        provider_mod._RECALL_EXIT.reset(token)
    request(counter, state, fresh=False)
    finish(counter, state)
    line = check(counter, caplog, 'deadline')
    assert 'stage=recall' in line and 'exception=TimeoutError' in line
    assert 'no time left for the request' not in line


@pytest.mark.parametrize('channel', ['whatsapp', 'rcs', 'sms', 'telegram', 'private-channel-sentinel'])
def test_channel_is_closed_vocabulary(counter, caplog, channel):
    caplog.set_level(logging.INFO)
    state = counter.begin(SESSION, TURN, owner=True, platform=channel)
    request(counter, state)
    finish(counter, state)
    assert f'channel={channel if channel in {"whatsapp", "rcs", "sms"} else "other"} ' in check(
        counter, caplog, 'ok', submitted=1, existing='new')


@pytest.mark.parametrize('threshold', [2, 3])
def test_streak_exclusions_and_one_alarm_per_streak(counter, caplog, threshold):
    caplog.set_level(logging.INFO)
    counter.configure(threshold)
    def turn(number, *, ok=False, owner=True, excluded=False):
        state = counter.begin(SESSION, str(number), owner=owner, excluded=excluded)
        request(counter, state, fresh=ok)
        finish(counter, state)
    for i in range(threshold-1):
        turn(i)
    turn('internal', excluded=True)
    turn('replay', excluded=True)
    turn('rewind', excluded=True)
    turn('guest', owner=False)
    assert counter._streak == threshold-1
    turn('threshold')
    turn('still missing')
    assert counter._streak == threshold+1
    assert sum(r.getMessage().startswith('RECALL_ALARM ') for r in caplog.records) == 1
    turn('reset', ok=True)
    assert counter._streak == 0
    for i in range(threshold):
        turn('next' + str(i))
    assert sum(r.getMessage().startswith('RECALL_ALARM ') for r in caplog.records) == 2


@pytest.mark.asyncio
async def test_sections_writes_and_request_byte_identical(world, counter, provider_mod, monkeypatch, tmp_path):
    writes = []
    async def finish_writes(proposals, observation=None):
        writes.append([(section, getattr(call, '__name__', type(call).__name__), getattr(call, '__defaults__', None), args, kwargs)
                       for section, call, args, kwargs in proposals])
    monkeypatch.setattr(turn_budget, 'finish_writes', finish_writes)
    original = recall_exit.begin
    monkeypatch.setattr(recall_exit, 'begin', lambda *a, **kw: None)
    before = await assemble(world)
    monkeypatch.setattr(recall_exit, 'begin', original)
    after = await assemble(world)
    assert json.dumps(before['sections']).encode() == json.dumps(after['sections']).encode()
    assert repr(writes[0]).encode() == repr(writes[1]).encode()
    outputs = []
    monkeypatch.setenv('COLONYMIND_HOME', str(tmp_path / 'instance'))
    for payload in (before, after):
        fake = _FakeHttpx(routes={_ASSEMBLE: payload, _TEMPORAL: {}})
        provider = _make_provider(provider_mod, fake, monkeypatch)
        provider.initialize(SESSION, platform='cli')
        provider._turn_ids[SESSION] = TURN
        state = counter.begin(SESSION, TURN, owner=True)
        output = provider.prefetch(QUERY, session_id=SESSION)
        messages = [{'role': 'user', 'content': '<memory-context>' + output + '</memory-context>'}]
        data = json.dumps({'messages': messages}).encode()
        counter.pre_api_request(session_id=SESSION, turn_id=TURN, request_messages=messages)
        assert json.dumps({'messages': messages}).encode() == data
        outputs.append(data)
    assert outputs[0] == outputs[1]


@pytest.mark.asyncio
async def test_sidecar_hooks_and_emitters_cannot_affect_turn(world, monkeypatch):
    def broken(*args):
        raise RuntimeError(ERROR)
    monkeypatch.setattr(recall_exit, 'current', broken)
    monkeypatch.setattr(recall_exit, 'emit', broken)
    result = await assemble(world)
    assert any(s['id'] == 'colonymind-memory' for s in result['sections'])
    assert world.driver.queries.get(TURN) is not None


@pytest.mark.parametrize('target', ['_emit', 'raise_alarm', 'logger'])
def test_host_failures_contained(counter, monkeypatch, caplog, target):
    caplog.set_level(logging.DEBUG)
    def broken(*args, **kw):
        raise RuntimeError(ERROR)
    if target == 'logger':
        monkeypatch.setattr(counter.logger, 'info', broken)
    else:
        monkeypatch.setattr(counter, target, broken)
    for i in range(3):
        state = counter.begin(SESSION, str(i), owner=True)
        request(counter, state, fresh=False)
        finish(counter, state)
    debug = [r.getMessage() for r in caplog.records if r.name == counter.__name__ and r.levelno == logging.DEBUG]
    assert debug == ['RECALL_EXIT_COUNTER_FAILED']
    assert not any(ERROR in r.getMessage() for r in caplog.records)


def test_emission_is_deferred_and_alarm_seam_allowlisted(counter, caplog):
    caplog.set_level(logging.INFO)
    state = counter.begin(SESSION, TURN, owner=True)
    request(counter, state, fresh=False)
    counter.post_llm_call(session_id=SESSION, turn_id=TURN)
    assert not log_lines(counter, caplog) and counter._streak == 0
    counter.drain()
    check(counter, caplog, 'no_candidates')
    counter.raise_alarm(2, ['deadline', QUERY, PASSAGE, ERROR])
    assert caplog.records[-1].getMessage().endswith('exits: deadline')


def test_provider_late_join_cannot_cross_turns(counter, provider_mod):
    first = counter.begin(SESSION, 'first', owner=True)
    callback = counter.prefetch_start(SESSION, 'first', wait=8)
    second = counter.begin(SESSION, 'second', owner=True)
    token = provider_mod._RECALL_EXIT.set(callback)
    try:
        # A response with an intentionally omitted context id still belongs to
        # its exact original request, never to the session's newer turn.
        provider_mod._observe_recall_response({'recall_exit': {
            'turn_id': '', 'sections_returned': 1, 'exit': 'error', 'stage': 'embed',
            'exception': 'RuntimeError', 'owner': False, 'query': QUERY}}, [], 'first')
    finally:
        provider_mod._RECALL_EXIT.reset(token)
    assert first.owner and not hasattr(first, 'query')
    assert first.sections_returned == 1 and first.exit == 'error'
    assert second.sections_returned == 0 and second.exit == 'ok'


def test_opaque_id_is_single_line_and_payload_is_allowlisted(counter, caplog):
    caplog.set_level(logging.INFO)
    state = counter.begin(SESSION, 'opaque\n"id', owner=True)
    counter.update(state, exit='error', exception=ERROR, stage=QUERY, subcode=PASSAGE, query=QUERY)
    counter.pre_api_request(session_id=SESSION, turn_id=state.turn_id, request_messages=[])
    finish(counter, state)
    line = log_lines(counter, caplog)[0]
    assert '\n' not in line and 'opaque\\n\\"id' in line
    assert all(value not in line for value in [QUERY, ERROR, PASSAGE, SESSION])


def test_host_alarm_config(counter, monkeypatch, tmp_path):
    from colonymind.config import DEFAULTS, ConfigError, _validate_context, load_config, apply_environment
    assert DEFAULTS['context']['recall_alarm_turns'] == 2
    for value in (0, -1, True, '2', 1.5):
        with pytest.raises(ConfigError, match='recall_alarm_turns'):
            _validate_context({'recall_alarm_turns': value})
    (tmp_path / 'colonymind.yaml').write_text('context:\n  recall_alarm_turns: 4\n')
    config = load_config(tmp_path)
    target = {}
    apply_environment(config, environ=target)
    assert target[recall_exit.ALARM_ENV] == '4'
    counter.configure_from_settings(SimpleNamespace(home=tmp_path))
    assert counter._threshold == 4


@pytest.fixture
def alarm_sender(counter, monkeypatch):
    body = importlib.import_module(counter.__package__ + '.body')
    monkeypatch.setattr(body, 'capture_module', counter)
    calls = []
    response = {'result': 'sent'}

    def send(target, text):
        calls.append((target, text, threading.get_ident(), counter._lock._is_owned()))
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr(body, 'send_message', send)

    def change(value):
        nonlocal response
        response = value

    return calls, change


def alarm_turn(counter, *, ok=False, code=None, drain=True):
    session, turn = ('marker-' + str(uuid.uuid4()) for _ in range(2))
    state = counter.begin(session, turn, owner=True)
    if code:
        counter.update(state, exit=code)
    messages = [{'role': 'user', 'content': memory('marker-' + str(uuid.uuid4()))}] if ok else []
    counter.pre_api_request(session_id=session, turn_id=turn, request_messages=messages)
    counter.post_llm_call(session_id=session, turn_id=turn)
    if drain:
        counter.drain()
    return state


@pytest.fixture
def alarm_body(counter, alarm_sender, monkeypatch, tmp_path):
    name = counter.__package__
    spec = importlib.util.spec_from_file_location(name, Path(counter.__file__).with_name('__init__.py'),
                                                submodule_search_locations=[str(Path(counter.__file__).parent)])
    plugin = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, plugin)
    spec.loader.exec_module(plugin)
    body_module = importlib.import_module(name + '.body')
    assert body_module.capture_module is counter
    client_module = importlib.import_module(name + '.client')
    marker = 'marker-' + str(uuid.uuid4())
    settings = client_module.Settings(marker, tmp_path / marker, marker, tmp_path, tmp_path, tmp_path / marker)
    client = SimpleNamespace(mind_state=lambda: {}, has_mind_routes=lambda: False)
    outbox = SimpleNamespace(drain=lambda deliver: 0, pending_count=lambda: 0)
    body = body_module.Body(client, outbox, None, settings)
    monkeypatch.setattr(plugin, '_BODY', body)
    monkeypatch.delenv('HERMES_KANBAN_TASK', raising=False)
    try:
        yield plugin, body
    finally:
        activity = plugin._ACTIVITY
        if activity is not None:
            activity.shutdown()
            if activity._thread is not None:
                activity._thread.join(3)
                assert not activity._thread.is_alive()


@pytest.fixture
def body_loop(alarm_body, monkeypatch):
    _, body = alarm_body
    run_once = body.run_once

    def once():
        try:
            return run_once()
        finally:
            body.stop()  # One real loop iteration, with no timer or external I/O.

    def run():
        body._stop.clear()
        body._thread = None
        with monkeypatch.context() as patch:
            patch.setattr(body, 'run_once', once)
            body.start()
            body._thread.join(2)
        assert not body._thread.is_alive()
        return body._thread.ident

    return run


@pytest.mark.parametrize('target', [None, '', ' ', 0, False, [], {}, ['marker-' + str(uuid.uuid4())]])
def test_alarm_unset_or_invalid_target_is_log_only(counter, alarm_sender, body_loop, monkeypatch, target, caplog):
    caplog.set_level(logging.INFO)
    client = importlib.import_module(counter.__package__ + '.client')
    config = {'recall_alarm_turns': 2}
    if target is not None:
        config['recall_alarm_target'] = target
    monkeypatch.setattr(client, 'read_yaml', lambda path: {'context': config})
    counter.configure_from_settings(SimpleNamespace(home=Path('marker-' + str(uuid.uuid4()))))
    for _ in range(3):
        alarm_turn(counter)
    body_loop()
    assert counter._alarm_target is None
    assert not alarm_sender[0]
    assert sum(r.getMessage().startswith('RECALL_ALARM ') for r in caplog.records) == 1
    assert not any(r.getMessage().startswith('RECALL_ALARM_SEND ') for r in caplog.records)


def test_alarm_send_once_per_streak_and_rearm(counter, alarm_sender, body_loop, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    target = 'marker-' + str(uuid.uuid4())
    client = importlib.import_module(counter.__package__ + '.client')
    monkeypatch.setattr(client, 'read_yaml', lambda path: {
        'context': {'recall_alarm_turns': 2, 'recall_alarm_target': target}})
    counter.configure_from_settings(SimpleNamespace(home=Path('marker-' + str(uuid.uuid4()))))
    calls = alarm_sender[0]
    alarm_turn(counter, code='deadline')
    assert not calls
    alarm_turn(counter, code='error')
    alarm_turn(counter)
    assert not calls
    body_loop()
    expected = 'Memory context was missing on 2 consecutive owner turns - exit codes: deadline,error.'
    assert [(t, text) for t, text, _, _ in calls] == [(target, expected)]
    assert expected.isascii() and '\u2013' not in expected and '\u2014' not in expected
    assert 'MEDIA' not in expected and expected.count(' - ') == 1
    assert 'marker-' not in expected and 'turn_id' not in expected and 'session_id' not in expected
    alarm_turn(counter, ok=True)
    assert counter._streak == 0
    alarm_turn(counter)
    alarm_turn(counter)
    alarm_turn(counter)
    body_loop()
    assert [(t, text) for t, text, _, _ in calls] == [
        (target, expected),
        (target, 'Memory context was missing on 2 consecutive owner turns - exit codes: no_candidates.')]
    assert not any(locked for _, _, _, locked in calls)
    assert [r.getMessage() for r in caplog.records if r.getMessage().startswith('RECALL_ALARM_SEND ')] == [
        'RECALL_ALARM_SEND result=sent', 'RECALL_ALARM_SEND result=sent']
    alarm_logs = [r.getMessage() for r in caplog.records if r.getMessage().startswith('RECALL_ALARM')]
    assert all(target not in line and 'Memory context was missing' not in line for line in alarm_logs)


@pytest.mark.parametrize('response,outcome', [
    (RuntimeError('marker-' + str(uuid.uuid4())), 'uncertain'),
    ({'result': 'failed', 'error': 'marker-' + str(uuid.uuid4())}, 'failed'),
    ({'result': 'sent', 'error': 'marker-' + str(uuid.uuid4()),
      'detail': {'raw': 'marker-' + str(uuid.uuid4())}}, 'sent'),
    ({'result': 'uncertain'}, 'uncertain'), (None, 'uncertain'), ({'result': 'marker-' + str(uuid.uuid4())}, 'uncertain')])
def test_alarm_sender_failure_does_not_repeat_or_interrupt_turns(counter, alarm_sender, body_loop, caplog, response, outcome):
    caplog.set_level(logging.INFO)
    counter.configure(2, 'marker-' + str(uuid.uuid4()))
    calls, respond = alarm_sender
    respond(response)
    for _ in range(4):
        alarm_turn(counter)
        body_loop()
    assert len(calls) == 1 and counter._streak == 4
    assert len(log_lines(counter, caplog)) == 4
    assert [r.getMessage() for r in caplog.records if r.getMessage().startswith('RECALL_ALARM_SEND ')] == [
        'RECALL_ALARM_SEND result=' + outcome]
    alarm_turn(counter, ok=True)
    respond({'result': 'sent'})
    alarm_turn(counter)
    alarm_turn(counter)
    body_loop()
    assert len(calls) == 2 and len(log_lines(counter, caplog)) == 7
    assert not any('marker-' in r.getMessage() for r in caplog.records
                   if r.getMessage().startswith('RECALL_ALARM'))


def test_alarm_send_runs_only_when_body_drains(counter, alarm_sender, body_loop, caplog):
    caplog.set_level(logging.INFO)
    counter.configure(2, 'marker-' + str(uuid.uuid4()))
    hook_thread = threading.get_ident()
    for _ in range(2):
        alarm_turn(counter, drain=False)
    assert not alarm_sender[0] and not log_lines(counter, caplog)
    assert not any(r.getMessage().startswith('RECALL_ALARM') for r in caplog.records)
    body_thread = body_loop()
    assert len(alarm_sender[0]) == 1
    assert alarm_sender[0][0][2:] == (body_thread, False)
    assert body_thread != hook_thread
    assert len(log_lines(counter, caplog)) == 2


@pytest.mark.parametrize('api', ['flush', 'tick', 'run_once'])
def test_caller_body_pass_never_sends_pending_alarm(counter, alarm_sender, alarm_body, body_loop, api):
    plugin, body = alarm_body
    counter.configure(2, 'marker-' + str(uuid.uuid4()))
    for _ in range(2):
        alarm_turn(counter, drain=False)
    caller = threading.get_ident()
    call = body.run_once if api == 'run_once' else getattr(plugin, api)
    call()
    call()  # An already pending alarm also stays off the caller thread.
    assert not alarm_sender[0] and counter._alarm_notice is not None
    thread = body_loop()
    assert len(alarm_sender[0]) == 1
    assert alarm_sender[0][0][2:] == (thread, False) and thread != caller
    body_loop()
    assert len(alarm_sender[0]) == 1


def test_body_disabled_alarm_is_log_only(counter, alarm_sender, alarm_body, monkeypatch, caplog):
    plugin, body = alarm_body
    caplog.set_level(logging.INFO)
    monkeypatch.setenv('COLONYMIND_BODY_THREAD', '0')
    monkeypatch.setattr(plugin, 'load_settings', lambda: body.settings)
    monkeypatch.setattr(plugin, 'ColonyMindClient', lambda settings: body.client)
    monkeypatch.setattr(plugin, 'TurnOutbox', lambda path: body.outbox)
    monkeypatch.setattr(plugin, 'Body', lambda *args, **kwargs: body)
    monkeypatch.setattr(plugin.busy_ingress, 'register', lambda ctx: None)
    monkeypatch.setattr(plugin.native_settlement, 'register', lambda ctx, capture: None)
    ctx = SimpleNamespace(**{name: lambda *args, **kwargs: object() for name in (
        'register_hook', 'register_command', 'register_tool', 'register_system_prompt_section')})
    plugin.register(ctx)
    assert body._thread is None
    body.work_capture = None
    counter.configure(2, 'marker-' + str(uuid.uuid4()))
    for _ in range(2):
        alarm_turn(counter, drain=False)
    plugin.flush()
    plugin.tick()
    body.run_once()
    assert counter._alarm_notice is not None and not alarm_sender[0]
    assert sum(r.getMessage().startswith('RECALL_ALARM ') for r in caplog.records) == 1
    assert not any(r.getMessage().startswith('RECALL_ALARM_SEND ') for r in caplog.records)


def test_newer_streak_replaces_unsent_notice(counter, alarm_sender, body_loop, caplog):
    caplog.set_level(logging.INFO)
    counter.configure(2, 'marker-' + str(uuid.uuid4()))
    for _ in range(2):
        alarm_turn(counter, code='deadline')
    first = counter._alarm_notice
    alarm_turn(counter, ok=True)
    for _ in range(2):
        alarm_turn(counter, code='error')
    assert not alarm_sender[0] and counter._alarm_notice != first
    body_loop()
    assert [text for _, text, _, _ in alarm_sender[0]] == [
        'Memory context was missing on 2 consecutive owner turns - exit codes: error.']
    alarm_turn(counter, code='deadline')
    body_loop()
    assert len(alarm_sender[0]) == 1
    alarm_turn(counter, ok=True)
    for _ in range(2):
        alarm_turn(counter)
    body_loop()
    body_loop()
    assert [text for _, text, _, _ in alarm_sender[0]] == [
        'Memory context was missing on 2 consecutive owner turns - exit codes: error.',
        'Memory context was missing on 2 consecutive owner turns - exit codes: no_candidates.']
    assert sum(r.getMessage().startswith('RECALL_ALARM ') for r in caplog.records) == 3
    assert sum(r.getMessage().startswith('RECALL_ALARM_SEND ') for r in caplog.records) == 2


def test_recovery_before_the_loop_sends_drops_the_notice(counter, alarm_sender, body_loop, caplog):
    caplog.set_level(logging.INFO)
    counter.configure(2, 'marker-' + str(uuid.uuid4()))
    for _ in range(2):
        alarm_turn(counter, code='deadline')
    assert counter._alarm_notice is not None
    alarm_turn(counter, ok=True)
    body_loop()
    assert alarm_sender[0] == [] and counter._alarm_notice is None
    assert sum(r.getMessage().startswith('RECALL_ALARM ') for r in caplog.records) == 1
    assert not any(r.getMessage().startswith('RECALL_ALARM_SEND ') for r in caplog.records)


def test_late_callback_waits_off_response_path_and_preserves_order(counter, caplog):
    caplog.set_level(logging.INFO)
    first = counter.begin(SESSION, TURN, owner=True)
    callback = counter.prefetch_start(SESSION, TURN, wait=8)
    request(counter, first, fresh=False)
    counter.post_llm_call(session_id=SESSION, turn_id=TURN)
    second = counter.begin(SESSION, 'second', owner=True)
    request(counter, second, fresh=True)
    counter.post_llm_call(session_id=SESSION, turn_id='second')
    counter.drain()  # nonblocking: first's existing prefetch has not returned
    assert not log_lines(counter, caplog)
    counter.update(callback, sections_returned=1)
    counter.prefetch_done(callback)
    counter.drain()
    lines = log_lines(counter, caplog)
    assert len(lines) == 2 and 'exit=host_drop ' in lines[0] and 'drop=late_callback' in lines[0]
    assert 'exit=ok ' in lines[1] and counter._streak == 0


def test_abandoned_prefetch_drains_in_order_and_ignores_late_completion(
        counter, provider_mod, monkeypatch, caplog, tmp_path):
    caplog.set_level(logging.INFO)
    monkeypatch.setenv('COLONYMIND_HOME', str(tmp_path / 'instance'))
    provider = _make_provider(provider_mod, _FakeHttpx(), monkeypatch)
    provider.initialize(SESSION, platform='cli')
    provider._turn_ids[SESSION] = TURN
    manager = types.ModuleType('agent.memory_manager')
    manager._EXTERNAL_PREFETCH_TIMEOUT_S = 12.0
    monkeypatch.setitem(sys.modules, manager.__name__, manager)
    now = [100.0]
    monkeypatch.setattr(counter, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    started, release = threading.Event(), threading.Event()

    def stuck(query, *, session_id):
        started.set()
        release.wait(5)
        provider_mod._observe_recall_response({'recall_exit': {
            'exit': 'ok', 'sections_returned': 1}}, [], TURN)
        return memory()

    monkeypatch.setattr(provider, '_prefetch_observed', stuck)
    first = counter.begin(SESSION, TURN, owner=True)
    thread = threading.Thread(target=provider.prefetch, args=(QUERY,), kwargs={'session_id': SESSION})
    thread.start()
    try:
        assert started.wait(2)
        assert first.prefetch_deadline == now[0] + manager._EXTERNAL_PREFETCH_TIMEOUT_S
        request(counter, first, fresh=False)
        counter.post_llm_call(session_id=SESSION, turn_id=TURN)
        second = counter.begin(SESSION, 'second', owner=True)
        request(counter, second, fresh=False)
        counter.post_llm_call(session_id=SESSION, turn_id='second')
        now[0] = first.prefetch_deadline - 0.01
        counter.drain()
        assert not log_lines(counter, caplog) and counter._streak == 0
        now[0] = first.prefetch_deadline + 0.01
        counter.drain()
        lines = log_lines(counter, caplog)
        assert len(lines) == 2
        assert f'exit=host_drop turn_id="{TURN}"' in lines[0] and 'drop=late_callback' in lines[0]
        assert 'exit=no_candidates turn_id="second"' in lines[1]
        assert counter._streak == 2 and not counter._pending
        assert sum(r.getMessage().startswith('RECALL_ALARM ') for r in caplog.records) == 1
        frozen = asdict(first)
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive()
    counter.post_llm_call(session_id=SESSION, turn_id=TURN)
    counter.drain()
    assert asdict(first) == frozen  # Both the late response and prefetch_done were ignored.
    assert log_lines(counter, caplog) == lines and counter._streak == 2
    assert sum(r.getMessage().startswith('RECALL_ALARM ') for r in caplog.records) == 1
    assert all(value not in line for line in lines for value in (QUERY, PASSAGE, ERROR, SESSION))


def test_cli_internal_notification_history_excludes_records_and_streak(counter, caplog):
    caplog.set_level(logging.INFO)
    capture = counter.Capture.__new__(counter.Capture)
    capture._recall_owner_handles = {}
    info = SimpleNamespace(session_id=SESSION, turn_id='', owner=False, host=None, platform='cli', sender_id='')
    history = [{'role': 'user', 'content': QUERY}]
    for number in range(2):
        info.turn_id = str(number)
        history.append({'role': 'user', 'content': PASSAGE, 'display_kind': 'internal_notification'})
        capture._begin_recall_exit(info, {'conversation_history': history})
        state = counter._turns[(SESSION, info.turn_id)]
        assert state.owner and state.excluded
        counter.update(state, exit='review')
        request(counter, state, fresh=False)
        finish(counter, state)
    assert not log_lines(counter, caplog) and counter._streak == 0
    assert not any(r.getMessage().startswith('RECALL_ALARM ') for r in caplog.records)


@pytest.mark.parametrize('kind', ['owner', 'internal', 'replay', 'rewind', 'guest', 'cron'])
def test_real_capture_exclusions(counter, monkeypatch, caplog, kind):
    caplog.set_level(logging.INFO)
    capture_module = counter
    info = SimpleNamespace(session_id=SESSION, turn_id=TURN, owner=kind != 'guest', host=None,
        platform='cron' if kind == 'cron' else 'whatsapp', sender_id='sender-private',
        heard=SimpleNamespace(message_id='inbound-private') if kind != 'internal' else None)
    capture = capture_module.Capture.__new__(capture_module.Capture)
    capture.sessions = SimpleNamespace(get=lambda sid: info, observe=lambda **kw: info)
    capture._recall_owner_handles = {}
    capture.work = capture_module.WorkCapture(None, None)  # recall also observes with work records disabled
    capture.branch_reaction = lambda **kw: None
    capture._capture = lambda **kw: None
    capture.on_enqueue = None
    capture.pre_llm_call(session_id=SESSION, **({kind: True} if kind in {'replay', 'rewind'} else {}))
    state = counter._turns[(SESSION, TURN)]
    capture.work.pre_api_request(session_id=SESSION, turn_id=TURN, request_messages=[])
    assert state.requested == (kind in {'owner', 'guest'})  # excluded kinds never mark a request
    capture.post_llm_call(session_id=SESSION, turn_id=TURN)
    assert not log_lines(counter, caplog)
    counter.drain()
    assert len(log_lines(counter, caplog)) == (1 if kind == 'owner' else 0)
    assert counter._streak == (1 if kind == 'owner' else 0)


@pytest.mark.parametrize('shape', ['string', 'text', 'input_text'])
def test_submitted_request_content_shapes(counter, caplog, shape):
    caplog.set_level(logging.INFO)
    state = counter.begin(SESSION, TURN, owner=True)
    text = memory()
    content = text if shape == 'string' else [{'type': shape, 'text': text}]
    messages = [{'role': 'user', 'content': content}]
    encoded = json.dumps(messages).encode()
    counter.pre_api_request(session_id=SESSION, turn_id=TURN, request_messages=messages)
    assert json.dumps(messages).encode() == encoded
    finish(counter, state)
    check(counter, caplog, 'ok', submitted=1, existing='new')


def test_fence_scan_is_linear_and_outside_counter_lock(counter, caplog):
    caplog.set_level(logging.INFO)
    calls, locks = [], []

    class Text(str):
        def find(self, tag, start=0):
            calls.append((tag, start))
            locks.append(counter._lock._is_owned())
            return super().find(tag, start)

    state = counter.begin(SESSION, TURN, owner=True)
    text = Text(memory() * 2 + '<memory-context>' * 10000)
    messages = [{'role': 'user', 'content': text}]
    before = json.dumps(messages).encode()
    counter.pre_api_request(session_id=SESSION, turn_id=TURN, request_messages=messages)
    assert json.dumps(messages).encode() == before
    # One opening and closing search per complete fence, then one pair for
    # the entire unterminated suffix, regardless of its opening-tag count.
    assert calls == [
        ('<memory-context>', 0), ('</memory-context>', len('<memory-context>')),
        ('<memory-context>', len(memory())), ('</memory-context>', len(memory()) + len('<memory-context>')),
        ('<memory-context>', 2 * len(memory())), ('</memory-context>', 2 * len(memory()) + len('<memory-context>'))]
    assert not any(locks)
    finish(counter, state)
    check(counter, caplog, 'ok', submitted=2, existing='new')


def test_unchanged_marker_alone_is_not_memory(counter, caplog):
    caplog.set_level(logging.INFO)
    state = counter.begin(SESSION, TURN, owner=True)
    counter.update(state, sections_returned=1)
    counter.pre_api_request(session_id=SESSION, turn_id=TURN, request_messages=[{'role': 'user', 'content':
        '<memory-context>\nunchanged since your last turn: Relevant Memories\n</memory-context>'}])
    finish(counter, state)
    line = check(counter, caplog, 'host_drop', returned=1)
    assert 'drop=' not in line  # absence cannot prove a specific fitting cause


@pytest.mark.asyncio
async def test_lexical_lane_exception_is_observed_with_fallback(monkeypatch):
    async def bad(query):
        raise LookupError(ERROR)
    async def vector(query):
        return search.LaneResult([], [], 'ready')
    ledger = SimpleNamespace(erasure_watermark=lambda cid: 0)
    monkeypatch.setattr(search, 'LANES', (search.Lane('vector', vector), search.Lane('lexical', bad)))
    with turn_budget.turn(turn_budget.TurnBudget(wait_s=7.5)) as budget:
        recall_exit.begin(True, True, TURN)
        with pytest.raises(LookupError):
            await search.collect_sources(ledger, query=QUERY, contact_id=OWNER, session_id=SESSION)
        outcome = recall_exit.snapshot(budget.recall_exit)
    assert outcome['exit'] == 'error' and outcome['stage'] == 'search' and outcome['exception'] == 'LookupError'


@pytest.mark.asyncio
@pytest.mark.parametrize('pending', [True, False])
async def test_erasure_refusal_is_an_actual_owner_search_exception(world, counter, monkeypatch, caplog, pending):
    caplog.set_level(logging.INFO)
    if pending:
        monkeypatch.setattr(world.driver, 'pending_erasure', lambda: {'private': [TURN]})
        monkeypatch.setattr(world.driver, 'attach', lambda runtime: None)
    else:
        monkeypatch.setattr(world.driver, 'erased', lambda turn: True)
    world.content = ''
    await joined(world, counter)
    line = check(counter, caplog, 'error')
    assert 'stage=owner_search' in line and 'exception=ValueError' in line and 'subcode=identity' not in line


@pytest.mark.asyncio
async def test_sidecar_fit_zero_is_not_a_host_drop(world, counter, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    async def omit(body, sections, **kwargs):
        return [s for s in sections if s.id != 'colonymind-memory']
    monkeypatch.setattr(host, '_turn_context_selection', omit)
    await joined(world, counter)
    check(counter, caplog, 'no_candidates')


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['owner_search', 'rerank'])
async def test_cancellation_stages(world, counter, monkeypatch, caplog, stage):
    caplog.set_level(logging.INFO)
    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        with turn_budget.turn(turn_budget.TurnBudget(wait_s=7.5)) as budget:
            recall_exit.begin(True, True, TURN)
            try:
                if stage == 'owner_search':
                    monkeypatch.setattr(world.store, 'search', cancelled)
                    recall_request = RecallRequest.create(world.runtime, owner=True, turn_id=TURN)
                    await source_vectors.SourceVectors(None, world.store, world.pipeline).search(
                        QUERY, contact_id=OWNER, session_id=SESSION, recall_request=recall_request)
                else:
                    from colonymind.memory.selection import RecallSelector
                    monkeypatch.setenv('COLONYMIND_RECALL_RERANK', 'on')
                    await RecallSelector(cancelled)._reranked(QUERY, [{'id': TURN, 'content': PASSAGE}], 0)
            finally:
                outcome = recall_exit.snapshot(budget.recall_exit)
    state = counter.begin(SESSION, TURN, owner=True)
    counter.update(state, **{k:v for k,v in outcome.items() if k != 'turn_id'})
    request(counter, state, fresh=False)
    finish(counter, state)
    assert 'stage=' + stage in check(counter, caplog, 'cancelled')


@pytest.mark.asyncio
async def test_assembly_deadline_before_owner_flags_still_has_facts(world, counter, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    async def stalled(*a, **kw):
        await asyncio.Event().wait()
    monkeypatch.setattr(host, '_context_assemble', stalled)
    await joined(world, counter, wait_ms=20)
    line = check(counter, caplog, 'deadline')
    assert 'stage=assembly_left' in line


def test_create_uses_operator_gate_precedence(world):
    # Both early gates fail: absence of a hook turn id wins over owner/runtime.
    with turn_budget.turn(turn_budget.TurnBudget(wait_s=7.5)) as budget:
        recall_exit.begin(None, True, '')
        assert RecallRequest.create(None, owner=False, turn_id='') is None
        assert budget.recall_exit.exit == 'no_turn_id'


def test_abandoned_prefetch_with_history_memory_submitted_is_ok(counter, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    now = [100.0]
    monkeypatch.setattr(counter, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    miss = counter.begin(SESSION, 'miss', owner=True)
    request(counter, miss, fresh=False)
    counter.post_llm_call(session_id=SESSION, turn_id='miss')
    counter.drain()
    assert counter._streak == 1
    state = counter.begin(SESSION, TURN, owner=True)
    counter.prefetch_start(SESSION, TURN, 12.0)
    request(counter, state, fresh=False, existing=True)
    counter.post_llm_call(session_id=SESSION, turn_id=TURN)
    now[0] += 12.01
    counter.drain()
    lines = log_lines(counter, caplog)
    assert len(lines) == 2 and f'exit=ok turn_id="{TURN}"' in lines[1] and 'sections_submitted=1' in lines[1]
    assert counter._streak == 0
    assert not any(r.getMessage().startswith('RECALL_ALARM ') for r in caplog.records)


def test_a_concurrent_drain_cannot_erase_a_newer_streaks_notice(counter, alarm_sender, body_loop, monkeypatch):
    counter.configure(2, 'marker-' + str(uuid.uuid4()))
    in_ok, release = threading.Event(), threading.Event()
    info = counter.logger.info

    def held_info(message, *args):
        if message.startswith('RECALL_EXIT ') and args[0] == 'ok' and not release.is_set():
            in_ok.set()
            release.wait(5)
        return info(message, *args)

    monkeypatch.setattr(counter.logger, 'info', held_info)
    alarm_turn(counter, ok=True, drain=False)
    first = threading.Thread(target=counter.drain)
    first.start()
    assert in_ok.wait(5)
    for _ in range(2):
        alarm_turn(counter, code='deadline', drain=False)
    second = threading.Thread(target=counter.drain)
    second.start()
    second.join(0.5)
    release.set()
    first.join(5)
    second.join(5)
    assert counter._streak == 2 and counter._alarm_notice is not None
    body_loop()
    assert len(alarm_sender[0]) == 1



@pytest.mark.parametrize('broken', ['take', 'send_log', 'send_log_cancelled'])
def test_a_raising_take_or_send_log_never_ends_the_body_loop(counter, alarm_sender, body_loop, monkeypatch, broken):
    counter.configure(2, 'marker-' + str(uuid.uuid4()))
    for _ in range(2):
        alarm_turn(counter, code='deadline')
    escaped = []
    monkeypatch.setattr(threading, 'excepthook', escaped.append)
    body = importlib.import_module(counter.__package__ + '.body')

    class Broken:
        def __enter__(self):
            raise RuntimeError('marker-' + str(uuid.uuid4()))

        def __exit__(self, *_):
            return False

    if broken == 'take':
        monkeypatch.setattr(counter, '_lock', Broken())
    else:
        info = body.logger.info

        def raising(message, *args):
            if message.startswith('RECALL_ALARM_SEND '):
                if broken == 'send_log_cancelled':
                    raise asyncio.CancelledError
                raise RuntimeError('marker-' + str(uuid.uuid4()))
            return info(message, *args)
        monkeypatch.setattr(body.logger, 'info', raising)
    body_loop()
    assert escaped == []
    assert len(alarm_sender[0]) == (0 if broken == 'take' else 1)
