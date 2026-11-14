"""The phone's Hands surface is bounded by routes and verified owner actions."""
from itertools import product

from fastapi import Depends, FastAPI, HTTPException, Request
import httpx
import pytest
import pytest_asyncio

from colonymind.api.credentials import CredentialTable, secret_digest
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import hands
from colonymind.hands import controller, definitions, wire
from test_hands_controller import c, running, finish, EPOCH, REF  # noqa: F401
from test_hands_definitions import editable, events  # noqa: F401

pytestmark = pytest.mark.asyncio


def credentials():
    return CredentialTable.from_config({name: {
        'owner': owner, 'contacts': 'own', 'access': ['hands'], 'sha256': secret_digest(name)
    } for name, owner in [('companion-hands', ['companion']), ('narrow', [])]})


@pytest_asyncio.fixture
async def phone(c, monkeypatch):
    app = FastAPI()
    app.include_router(hands.runtime_router)
    app.include_router(hands.router)
    app.state.hands_store = c.store
    app.add_middleware(ApiKeyMiddleware, api_key='instance', credentials=credentials())
    async def current():
        return c
    monkeypatch.setattr(controller, 'get_controller', current)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test',
                                headers={'Authorization': 'Bearer companion-hands'}) as client:
        yield client


async def test_every_app_hands_route_checks_the_explicit_companion_allowlist(monkeypatch):
    from colonymind.api import auth
    from colonymind.server import create_app
    monkeypatch.setattr(auth, 'configured_api_key', lambda: 'instance')
    monkeypatch.setattr(CredentialTable, 'load', lambda: credentials())
    app = create_app()  # No lifespan or services: exercise the real app's mounted route catalog.
    allowed = {
        ('GET', '/v1/hands/definitions'), ('POST', '/v1/hands/definitions'),
        ('GET', '/v1/hands/definitions/{d}'), ('PATCH', '/v1/hands/definitions/{d}'),
        ('DELETE', '/v1/hands/definitions/{d}'),
        ('GET', '/v1/hands/definitions/{d}/tasks'), ('POST', '/v1/hands/definitions/{d}/tasks'),
        ('GET', '/v1/hands/'), ('GET', '/v1/hands/{id}'), ('GET', '/v1/hands/digest'),
        ('GET', '/v1/hands/tasks/{tid}'), ('GET', '/v1/hands/tasks/{tid}/messages'),
        ('POST', '/v1/hands/tasks/{tid}/accept'), ('POST', '/v1/hands/tasks/{tid}/cancel'),
        ('POST', '/v1/hands/tasks/{tid}/revise'), ('POST', '/v1/hands/questions/{qid}/answer'),
    }
    # New routes, methods, and lookalike suffixes must be denied without editing the allowlist.
    async def future():
        raise AssertionError('A future route must never execute')
    for path, method in [('/v1/hands/tasks/{tid}/future', 'POST'),
                         ('/v1/hands/definitions', 'PUT'),
                         ('/v1/hands/questions/{qid}/answer/again', 'POST')]:
        app.add_api_route(path, future, methods=[method], dependencies=[Depends(hands.require_owner)])
    async def checked(request: Request):
        await hands.require_owner(request)
        raise HTTPException(418, 'guard_allowed')
    app.dependency_overrides[hands.require_owner] = checked
    def routes(router):
        for route in router.routes:
            # Newer FastAPI versions keep included routers lazy.
            included = getattr(route, 'original_router', None)
            if included is not None:
                yield from routes(included)
            else:
                yield route
    seen = set()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test',
                                headers={'Authorization': 'Bearer companion-hands'}) as client:
        for route in routes(app):
            if not getattr(route, 'path', '').startswith('/v1/hands/'):
                continue
            path = route.path
            for name in route.param_convertors:
                path = path.replace('{' + name + '}', {'tid': 't-' + '1' * 12,
                    'id': 'h-' + '1' * 12}.get(name, 'test-' + name))
            for method in route.methods:
                pair = (method, route.path)
                seen.add(pair)
                response = await client.request(method, path, json={}, headers={'Idempotency-Key': 'walk'})
                assert response.status_code == (418 if pair in allowed else 403), (pair, response.text)
    assert allowed <= seen


async def write(phone, path, body, key='action'):
    return await phone.post('/v1/hands' + path, json=body, headers={'Idempotency-Key': key})


@pytest.mark.parametrize('supplied_actor', [None, 'spoofed-owner'])
async def test_answer_uses_verified_credential_and_preserves_raw_body_idempotency(c, phone, supplied_actor):
    p = await running(c)
    question = await c.question(p['hand_id'], wire.QuestionRequest(
        epoch=EPOCH, question='Which source?', assumption='The saved one', blocking=True), 'question')
    path = '/questions/' + question['id'] + '/answer'
    body = {'answer': 'Use the latest source'}
    if supplied_actor:
        body['answered_by'] = supplied_actor
    for headers in ({}, {'Idempotency-Key': ''}, {'Idempotency-Key': '  '}):
        assert (await phone.post('/v1/hands' + path, json=body, headers=headers)).status_code == 422
    assert (await c.require('questions', question['id']))['state'] == 'open'
    denied = await phone.post('/v1/hands' + path, json=body,
        headers={'Authorization': 'Bearer narrow', 'Idempotency-Key': 'denied'})
    assert denied.status_code == 403
    denied = await phone.post('/v1/hands' + path, json=body,
        headers={'X-ColonyMind-Session': 'plan-1:child', 'Idempotency-Key': 'child'})
    assert denied.status_code == 403
    response = await write(phone, path, body)
    assert response.status_code == 200, response.text
    saved = response.json()
    assert saved['state'] == 'answered' and saved['answer'] == body['answer']
    assert saved['answered_by'] == 'credential:companion-hands'
    assert (await write(phone, path, body)).json() == saved
    for changed in (body | {'answer': 'Different'}, body | {'answered_by': 'another-spoof'}):
        conflict = await write(phone, path, changed)
        assert conflict.status_code == 409 and conflict.json()['detail'] == 'idempotency_key_reuse'
    closed = await write(phone, path, body, 'new-key')
    assert closed.status_code == 409 and closed.json()['detail'] == 'question_closed'
    assert (await c.task_view(p['task_id']))['status'] != 'needs_owner'
    assert (await phone.post('/v1/mind/permit-asks/ask/answer', json={'answer': 'yes'},
                            headers={'Idempotency-Key': 'permit'})).status_code == 403


async def test_instance_answer_callers_can_keep_their_attribution(c, phone):
    p = await running(c)
    q = await c.question(p['hand_id'], wire.QuestionRequest(epoch=EPOCH, question='Which?'), 'q')
    response = await phone.post('/v1/hands/questions/' + q['id'] + '/answer',
        json={'answer': 'This one', 'answered_by': 'front'},
        headers={'Authorization': 'Bearer instance', 'Idempotency-Key': 'legacy'})
    assert response.status_code == 200 and response.json()['answered_by'] == 'front'


async def test_blocking_question_keeps_needs_owner_across_revision(c, phone):
    p = await running(c)
    q = await c.question(p['hand_id'], wire.QuestionRequest(
        epoch=EPOCH, question='Which source?', blocking=True), 'q')
    revised = await write(phone, '/tasks/' + p['task_id'] + '/revise', {'goal': 'New goal'}, 'revise')
    assert revised.status_code == 200 and revised.json()['intent_version'] == 2
    assert revised.json()['status'] == 'needs_owner'
    assert revised.json()['questions'][0]['id'] == q['id']
    await write(phone, '/questions/' + q['id'] + '/answer', {'answer': 'This one'}, 'answer')
    assert (await phone.get('/v1/hands/tasks/' + p['task_id'])).json()['status'] == 'running'


@pytest.mark.parametrize('operation', ['accept', 'cancel', 'revise'])
async def test_task_owner_actions_bodies_replay_and_changed_body_conflict(c, phone, operation):
    p = await running(c)
    body = {'accept': {'reason': 'Reviewed'}, 'cancel': {'reason': 'Stop'},
            'revise': {'goal': 'Use newer sources', 'criteria': {'required_sources': 2}, 'stop': True}}[operation]
    if operation == 'accept':
        await finish(c, p)
    path = '/tasks/' + p['task_id'] + '/' + operation
    assert (await phone.post('/v1/hands' + path, json=body)).status_code == 422
    response = await write(phone, path, body)
    assert response.status_code == 200, response.text
    task = response.json()
    assert task['id'] == p['task_id'] and 'questions' in task
    assert task['state'] == {'accept': 'done', 'cancel': 'cancelled', 'revise': 'open'}[operation]
    if operation == 'accept':
        assert task['status'] == 'done'
    if operation == 'revise':
        assert task['intent_version'] == 2
    assert (await write(phone, path, body)).json() == task
    changed = body | ({'goal': 'Changed goal'} if operation == 'revise' else {'reason': 'Changed reason'})
    response = await write(phone, path, changed)
    assert response.status_code == 409 and response.json()['detail'] == 'idempotency_key_reuse'


@pytest.mark.parametrize('fields', ['both', 'hand', 'version', 'neither'])
async def test_accept_optional_fields_and_stale_version(c, phone, fields):
    p = await running(c)
    await finish(c, p)
    path = '/tasks/' + p['task_id'] + '/accept'
    before = await c.get_task(p['task_id'])
    stale = await write(phone, path, {'reason': 'Old review', 'for_version': 2}, 'stale')
    assert stale.status_code == 409 and stale.json()['detail'] == 'stale_result'
    assert await c.get_task(p['task_id']) == before
    body = {'reason': 'Reviewed'}
    if fields in {'both', 'hand'}:
        body['hand'] = p['hand_id']
    if fields in {'both', 'version'}:
        body['for_version'] = 1
    response = await write(phone, path, body)
    assert response.status_code == 200 and response.json()['state'] == 'done'


async def test_accept_requires_current_result_and_does_not_pick_an_older_success(c, phone):
    p = await running(c)
    path = '/tasks/' + p['task_id'] + '/accept'
    response = await write(phone, path, {'reason': 'Too early'})
    assert response.status_code == 409 and response.json()['detail'] == 'acceptance_guard'
    await finish(c, p)
    await c.revise(p['task_id'], wire.ReviseRequest(goal='New goal'), 'revise')
    response = await write(phone, path, {'reason': 'Stale', 'for_version': 1})
    assert response.status_code == 409 and response.json()['detail'] == 'stale_result'
    response = await write(phone, path, {'reason': 'No current result'})
    assert response.status_code == 409 and response.json()['detail'] == 'acceptance_guard'


async def test_task_and_definition_list_questions_are_bounded_open_first_and_isolated(c, phone):
    definition = await c.save_definition(definitions.CreateDefinition(name='Reviewer'), 'definition')
    p = await running(c, hand=definition['id'])
    other = await running(c, 'other')
    for i in range(25):
        await c.insert('questions', id=f'q-{i:02}', hand_id=p['hand_id'], text=f'Question {i}',
            assumption='Default', blocking=i == 0, asked_at=f'2026-01-01T00:00:{i:02}Z',
            state='open' if i < 2 else 'answered', answer=None if i < 2 else 'Saved')
    await c.insert('questions', id='q-unrelated', hand_id=other['hand_id'], text='PRIVATE-OTHER',
                   asked_at=c.now(), blocking=True)
    await c.store.db.commit()
    task = (await phone.get('/v1/hands/tasks/' + p['task_id'])).json()
    assert task['status'] == 'needs_owner' and task['state'] == 'open'
    assert len(task['questions']) == 20
    assert [q['id'] for q in task['questions'][:3]] == ['q-01', 'q-00', 'q-24']
    assert [q['id'] for q in task['questions'][2:]] == [f'q-{i:02}' for i in range(24, 6, -1)]
    assert all(set(q) == {'id', 'text', 'assumption', 'blocking', 'asked_at', 'state', 'answer'}
               for q in task['questions'])
    assert task['questions'][1]['blocking'] is True
    listed = (await phone.get('/v1/hands/definitions/' + definition['id'] + '/tasks')).json()['tasks']
    assert listed == [task]
    await write(phone, '/questions/q-00/answer', {'answer': 'Resolved'})
    task = (await phone.get('/v1/hands/tasks/' + p['task_id'])).json()
    assert task['status'] == 'running'
    assert task['questions'][0]['id'] == 'q-01'
    assert (await c.task_view(other['task_id']))['status'] == 'needs_owner'


@pytest.mark.parametrize('color,shape', list(product(
    ['#B9A7D8', '#9EBBA0', '#D8B99A', '#A0B9D0'], ['circle', 'flower', 'square', 'star'])))
async def test_avatar_exact_enum_round_trip(c, phone, color, shape):
    response = await write(phone, '/definitions', {'name': 'Hand', 'avatar': {
        'color': color, 'shape': shape, 'ref': 'unused-reference'}})
    assert response.status_code == 200, response.text
    row = response.json()
    assert row['avatar'] == {'color': color, 'shape': shape, 'ref': 'unused-reference'}
    assert (await phone.get('/v1/hands/definitions/' + row['id'])).json() == row


@pytest.mark.parametrize('avatar', [
    {'color': '#b9a7d8'}, {'color': '#B9A7D8 '}, {'color': 'red'}, {'color': '#FFFFFF'},
    {'color': None}, {'color': 1}, {'shape': 'Circle'}, {'shape': 'triangle'}, {'shape': ''},
    {'shape': None}, {'shape': 1}, {'ref': 'x' * 2001}, {'unexpected': True},
])
async def test_avatar_invalid_on_create_and_replace_returns_422(c, phone, avatar):
    response = await write(phone, '/definitions', {'name': 'Invalid', 'avatar': avatar})
    assert response.status_code == 422
    row = await c.save_definition(definitions.CreateDefinition(name='Existing'), 'existing')
    body = editable(row).model_dump() | {'avatar': avatar}
    response = await phone.patch('/v1/hands/definitions/' + row['id'], json=body,
                                headers={'Idempotency-Key': 'replace'})
    assert response.status_code == 422


async def test_legacy_avatar_rows_and_snapshots_remain_readable(c, phone):
    import json
    row = await c.save_definition(definitions.CreateDefinition(name='Legacy'), 'legacy')
    row['avatar'] = {'color': 'old-color', 'shape': None, 'ref': None}
    for table in ('hand_definitions', 'hand_definition_versions'):
        await c.execute(f'UPDATE {table} SET document=?', (json.dumps(row),))
    await c.store.db.commit()
    assert (await phone.get('/v1/hands/definitions/' + row['id'])).json() == row
    assert (await phone.get('/v1/hands/definitions')).json()['definitions'] == [row]
    queued = await write(phone, '/definitions/' + row['id'] + '/tasks', {'brief': 'Read legacy'})
    assert queued.status_code == 200 and queued.json()['questions'] == []
    assert (await c.task_definition(queued.json()['id']))['avatar'] == row['avatar']
