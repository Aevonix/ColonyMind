"""Ideas wire contract, copy admission, private media and real queue."""
from copy import deepcopy
import hashlib
import json

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from colonymind.api.credentials import CredentialTable, default_credentials, secret_digest
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import host
from colonymind.hands import controller, definitions, wire
from colonymind.mind.ideas import Ideas
from colonymind.mind.ideas_build import HandsQueue
from colonymind.mind.ideas_copy import PROMPT, validate_copy
from colonymind.mind.ideas_presentation import KINDS, media
from test_grounded_ideas import ACCOUNT, KEY, app, env, generate, mailbox  # noqa: F401
from test_hands_controller import c, EPOCH  # noqa: F401
from test_hands_definitions import events  # noqa: F401

pytestmark = pytest.mark.asyncio
ROOT = '/v1/host/initiatives'


def check_presentation(value, *, item=False):
    # Unbuilt cards and every item: no kind/category labels and no badge; only cards carry the action.
    assert set(value) == ({'icon', 'labels', 'badges'} if item else {'icon', 'labels', 'badges', 'actionLabel'})
    assert item or value['actionLabel'] == 'Build'
    assert value['labels'] == [] and value['badges'] == []
    assert value['icon']['altText']
    assert len(value['icon']['variants']) == 1
    variant = value['icon']['variants'][0]
    assert set(variant) == {'url', 'width', 'height', 'mimeType', 'byteSize', 'version'}
    assert variant['mimeType'] == 'image/svg+xml'
    assert variant['width'] == variant['height'] == 96
    assert type(variant['version']) is int and variant['version'] == 1
    assert variant['byteSize'] == len(media(variant['url'].rsplit('/', 1)[1]))


async def test_every_summary_detail_item_and_private_media_field(env):
    await generate(env)
    async with AsyncClient(transport=ASGITransport(app=app()), base_url='http://test',
                           headers={'Authorization': 'Bearer ' + KEY}) as client:
        page = (await client.get(ROOT, params={'contract': 2})).json()
        assert page['viewerState'] == {'hasBuiltIdea': False}
        assert page['pagination'] == {'nextCursor': None, 'hasNextPage': False}
        assert page['capabilities']['sharing'] is False
        assert page['sections'][0]['presentation'] == {'layout': 'hero'}
        assert 3 <= len(page['sections'][0]['cards']) <= 8
        assert all(s['presentation'] == {'layout': 'rows'} for s in page['sections'][1:])
        cards = [card for section in page['sections'] for card in section['cards']]
        assert len(cards) == len({c['id'] for c in cards}) == 4
        assert {card['kind'] for card in cards} == set(KINDS)
        for section in page['sections']:
            assert set(section) == {'id', 'title', 'subtitle', 'presentation', 'cards'} and section['cards']
            if section['presentation']['layout'] == 'rows':
                assert {card['secondaryLabel'] for card in section['cards']} == {section['title']}
        for card in cards:
            assert card['sharing'] is None  # private; no invented share target
            assert set(card['content']) == {'title', 'summary', 'detailDescription', 'fitReason', 'buildSummary'}
            assert 'primaryLabel' not in card
            assert card['secondaryLabel'] == KINDS[card['kind']][1]
            assert not {'buildStatus', 'build_status', 'buildState'} & card.keys()
            validate_copy(card['content'])
            check_presentation(card['presentation'])
            assert 'previewImage' not in card['presentation']
            variant = card['presentation']['icon']['variants'][0]
            assert card['iconUrl'] == variant['url'] and card['mimeType'] == variant['mimeType']
            image = await client.get(card['iconUrl'])
            assert image.status_code == 200 and image.headers['content-type'] == 'image/svg+xml'
            assert len(image.content) == variant['byteSize']
            assert image.headers['cache-control'].startswith('private,')
            assert image.headers['x-content-type-options'] == 'nosniff'
            assert b'<script' not in image.content and b'href=' not in image.content
            detail = await client.get(f"{ROOT}/{card['id']}", params={'contract': 2})
            assert detail.json() == {'card': card}
            legacy = (await client.get(f"{ROOT}/{card['id']}")).json()
            assert legacy['id'] == card['id'] and 'description' in legacy and 'card' not in legacy
            assert 'content' not in legacy and 'INV-204' not in json.dumps(legacy)
            for item in card['items']:
                assert set(item) == {'id', 'selectable', 'isSelected', 'content', 'presentation'}
                assert item['isSelected'] is True
                assert set(item['content']) == {'title', 'summary', 'buildSummary'}
                assert all(item['content'].values())
                check_presentation(item['presentation'], item=True)
                assert 'previewImage' not in item['presentation']
        assert (await client.get(ROOT + '/media/unknown.svg')).status_code == 404
    # Same authority requirement as the catalog, including anonymous and model callers.
    credentials = default_credentials()
    credentials['model'] = {'owner': ['sms'], 'contacts': 'all', 'access': ['read', 'write', 'manage'],
                            'sha256': secret_digest('model-api-key')}
    guarded = FastAPI()
    guarded.add_middleware(ApiKeyMiddleware, api_key=KEY, credentials=CredentialTable.from_config(credentials))
    guarded.include_router(host.router)
    async with AsyncClient(transport=ASGITransport(app=guarded), base_url='http://test') as client:
        for endpoint in (ROOT + '/ideas?contract=2', cards[0]['iconUrl']):
            assert (await client.get(endpoint)).status_code == 401
            assert (await client.get(endpoint, headers={'Authorization': 'Bearer model-api-key'})).status_code == 403
    assert env.model.calls == 1


async def test_thin_and_empty_sections_and_retained_deselection(env):
    assert host._ideas_catalog_v2(6, None)['sections'] == []
    row = (await generate(env, mailbox()[:1]))[0]
    page = host._ideas_catalog_v2(6, None)
    assert [s['presentation']['layout'] for s in page['sections']] == ['rows']
    # A third genuine optional deliverable exercises the stored selection merge.
    idea = deepcopy(row.context['idea'])
    payload = env.ideas.open(idea['private'])
    payload['items'].append({'id': 'timeline', 'type': 'deliverable', 'selectable': True, 'isSelected': True,
                            'content': {'title': 'List the decision dates', 'summary': 'Collect the explicit deadlines.',
                                        'buildSummary': 'A dated decision checklist.'}, 'presentation': {}})
    idea['private'] = env.ideas.seal(payload)
    env.store.update(row.id, context={'idea': idea})
    await env.ideas.build(row, 'selectedItems', ['draft'], None)
    card = env.ideas.card(env.store.get(row.id))
    assert {item['id']: item['isSelected'] for item in card['items']} == {
        'evidence': True, 'draft': True, 'timeline': False}
    assert json.loads(env.queue.calls[0][1])['selectedItems'] == ['draft', 'evidence']


async def test_pagination_never_loses_cards_and_stale_cursor_is_explicit(env):
    rows = await generate(env)
    expected = {r.id for r in rows}
    for limit in (1, 2, 3, 4, 6, 1000):
        cursor, seen = None, []
        while True:
            page = host._ideas_catalog_v2(limit, cursor)
            assert len(json.dumps(page, ensure_ascii=True).encode()) < 1_048_576
            cards = [c for s in page['sections'] for c in s['cards']]
            assert len(cards) <= limit
            seen.extend(c['id'] for c in cards)
            if not page['pagination']['hasNextPage']:
                assert page['pagination']['nextCursor'] is None
                break
            cursor = page['pagination']['nextCursor']
            assert cursor
        assert set(seen) == expected and len(seen) == len(expected)
    cursor = host._ideas_catalog_v2(1, None)['pagination']['nextCursor']
    await env.ideas.build(rows[0], 'full', None, None)
    with pytest.raises(Exception, match='invalid_or_stale_ideas_cursor'):
        host._ideas_catalog_v2(1, cursor)
    assert host._ideas_catalog_v2(6, None)['viewerState']['hasBuiltIdea'] is True


async def test_relay_budget_counts_envelope_and_continuation(env, monkeypatch):
    await generate(env)
    original = host._ideas_v2_card
    def wide(row, state):
        card = original(row, state)
        card['content'] = {**card['content'], 'detailDescription': '界' * 50000}
        return card
    monkeypatch.setattr(host, '_ideas_v2_card', wide)
    first = host._ideas_catalog_v2(6, None)
    assert len(json.dumps(first, ensure_ascii=True).encode()) <= host.IDEAS_V2_BUDGET
    assert first['pagination']['hasNextPage'] is True
    second = host._ideas_catalog_v2(6, first['pagination']['nextCursor'])
    assert len(json.dumps(second, ensure_ascii=True).encode()) <= host.IDEAS_V2_BUDGET
    assert len({c['id'] for p in (first, second) for s in p['sections'] for c in s['cards']}) == 4


@pytest.mark.parametrize('native,api', [('accepted', 'in_progress'), ('working', 'in_progress'),
                                        ('done', 'complete'), ('failed', 'failed')])
async def test_all_honest_build_states_mirror_api_and_retained_selection(env, native, api):
    row = (await generate(env, mailbox()[:1]))[0]
    await env.ideas.build(row, 'selectedItems', ['draft'], None)
    env.queue.state = native
    await env.ideas.reconcile()
    card = env.ideas.card(env.store.get(row.id))
    assert card['buildStatus'] == card['build_status'] == api
    assert card['buildState'] == native
    state = env.store.idea_states([row.id])[0]
    assert state['state'] == state['buildState'] == native and state['buildStatus'] == api
    assert all(item['isSelected'] for item in card['items'])


@pytest.mark.parametrize('field,bad', [
    ('summary', 'I’ll use gmail_get to read the email.'),
    ('fitReason', 'calendar_events e-review has the details.'),
    ('fitReason', 'The account is synthetic-account.'),
    ('fitReason', 'The message is m-invoice.'),
    ('fitReason', 'The source is 01234567-1234-5678-abcd-123456789abc.'),
    ('summary', 'I’ll compare it.".'), ('summary', 'I’ll compare it..'),
    ('summary', 'I’ll compare it. Then reply.'), ('summary', 'A generic summary.'),
    ('title', 'I can ' + 'x' * 60), ('fitReason', 'Sources: email'),
])
async def test_copy_validator_rejects_internals_punctuation_and_bad_shape(env, field, bad):
    row = (await generate(env, mailbox()[:1]))[0]
    payload = env.ideas.open(row.context['idea']['private'])
    content = {**payload['content'], field: bad}
    with pytest.raises(ValueError):
        validate_copy(content, payload['sources'])


async def test_model_copy_is_used_when_valid_and_rejected_when_invalid(env, monkeypatch):
    monkeypatch.setenv('COLONYMIND_PERSONA_NAME', 'Nova')
    good = {
        'title': 'Reconcile INV-204 with PO-88',
        'summary': 'I’ll check the $30 gap between the $480 invoice and $450 order.',
        'detailDescription': 'I’ll compare the invoice and order and prepare an unsent reply before 2026-10-21.',
        'fitReason': 'The email asks you to confirm the amount by 2026-10-21.',
        'buildSummary': 'I’ll return a cited comparison, discrepancies and an unsent reply draft.',
    }
    # Computed amounts must be evidence-backed too; the quote states both amounts,
    # but does not itself state $30. Use a literal, supported summary for admission.
    good['summary'] = 'I’ll check the $480 charge against the $450 order total.'
    env.model.rewrite = lambda offers: [{**offers[0], 'content': good}]
    row = (await generate(env, mailbox()[:1]))[0]
    assert env.ideas.card(row)['content'] == good
    assert env.model.messages[0] == {'role': 'system', 'content': PROMPT.replace('{assistant}', 'Nova')}
    for summary in ('I’ll reconcile INV-204 with PO-88.', 'I’ll check the $999 charge.',
                    'I’ll check the $48 charge.', 'I’ll check INV-450.', 'I’ll use gmail_get.'):
        content = {**good, 'summary': summary}
        with pytest.raises(ValueError):
            validate_copy(content, evidence=mailbox()[0]['result']['data']['snippet'])
    # Rejected supplied prose uses the verified fallback without losing the offer.
    env.model.rewrite = lambda offers: [{**offers[1], 'content': {**good, 'summary': 'I’ll use gmail_get.'}}]
    rows = await generate(env)
    assert len(rows) == 2 and row in rows
    renewal = next(r for r in rows if r.id != row.id)
    assert env.ideas.card(renewal)['content']['title'] == 'Review the Workspace renewal'


async def test_copy_validator_rejects_short_title_echo():
    with pytest.raises(ValueError, match='copy_title_echo'):
        validate_copy({'title': 'Review terms', 'summary': 'I’ll review the terms.',
                       'detailDescription': 'I’ll check the price before the decision date.',
                       'fitReason': 'The email gives a decision deadline.',
                       'buildSummary': 'I’ll return a dated terms summary.'})


async def test_fallback_keeps_amounts_bound_to_their_documents(env):
    source = mailbox()[:1]
    source[0]['result']['data']['snippet'] = 'Order PO-88 lists $450; Invoice INV-204 is $480. Confirm by 2026-10-21.'
    row = (await generate(env, source))[0]
    content = env.ideas.card(row)['content']
    assert content['summary'] == 'I’ll check the $480 invoice against the $450 order.'


@pytest.mark.parametrize('mode,item_ids', [('full', None), ('selectedItems', ['draft'])])
async def test_real_owner_started_queue_build_retry_claim_bind_and_result(env, c, monkeypatch, mode, item_ids):
    async def get_controller():
        return c
    monkeypatch.setattr(controller, 'get_controller', get_controller)
    definition = await c.save_definition(definitions.CreateDefinition(name='Review packets',
        instructions='Compare originals; return a private packet and unsent draft.'), 'review-definition')
    env.ideas.config['definition_id'] = definition['id']
    env.ideas.queue = HandsQueue()
    row = (await generate(env, mailbox()[:1]))[0]
    async with AsyncClient(transport=ASGITransport(app=app()), base_url='http://test',
                           headers={'Authorization': 'Bearer ' + KEY}) as client:
        for invalid in ({'mode': 'unknown'}, {'mode': 'full', 'itemIds': ['draft']},
                        {'mode': 'selectedItems'}, {'mode': 'full', 'extra': 1}):
            assert (await client.post(f'{ROOT}/{row.id}/execute', json=invalid)).status_code == 422
        execution_body = {'mode': mode, **({'itemIds': item_ids} if item_ids is not None else {})}
        response = await client.post(f'{ROOT}/{row.id}/execute', json=execution_body)
        assert response.status_code == 200, response.text
        receipt = response.json()
        assert set(receipt) == {'cardId', 'executionId', 'status'} and receipt['status'] == 'accepted'
        assert receipt['cardId'] == row.id and receipt['executionId']
        saved = env.store.get(row.id).context['idea']['execution']
        task_id = saved['taskId']
        queued = await c.one('SELECT * FROM definition_task_queue WHERE task_id=?', (task_id,))
        assert queued['owner_started'] == 1 and queued['initiating_credential'] == 'api.key'
        assert json.loads(queued['brief'])['sourceRefs'][0]['account'] == ACCOUNT['account']
        snapshot = await c.task_definition(task_id)
        assert {k: v for k, v in snapshot.items() if k != 'source_reads'} == definition
        assert snapshot['source_reads'][0]['provider_id'] == 'm-invoice'
        assert (await c.one('SELECT count(*) n FROM tasks'))['n'] == 1
        assert await c.all('SELECT * FROM hands') == []
        env.ideas = Ideas(env.mind, vault=env.ideas.vault)  # reload durable binding
        env.ideas.config['definition_id'] = definition['id']
        env.store._ideas = env.ideas
        assert (await client.post(f'{ROOT}/{row.id}/execute', json=execution_body)).json() == receipt
        assert (await c.one('SELECT count(*) n FROM tasks'))['n'] == 1
        assert (await env.ideas.queue.status(task_id))['status'] == 'accepted'
        # Exercise the real runtime queue protocol in-process; no native child is launched.
        claim = (await c.claim_definition_tasks(definitions.QueueClaim(parent_session='owner-parent'), 'pickup'))['tasks'][0]
        planned = (await c.plan(wire.PlanRequest(parent_session='owner-parent', turn='owner-start',
            origin='owner_turn', origin_ref={'platform': 'test', 'chat': 'owner', 'message_id': 'build'},
            tasks=[{'continue': task_id, 'goal': 'ignored replacement', 'queue_claim': claim['claim_token']}]),
            'plan-idea'))['plans'][0]
        assert planned['admit'] == 'start' and queued['brief'] in planned['brief']
        await c.bind_definition_claim(definitions.QueueBind(task_id=task_id, hand_id=planned['hand_id'],
            claim_token=claim['claim_token']), 'bind-idea-claim')
        await c.dispatched(planned['hand_id'], wire.DispatchedRequest(epoch=EPOCH,
            delegation_id=planned['hand_id'], lease_token=planned['lease_token']), 'idea-dispatched')
        await c.bind(planned['hand_id'], wire.BindRequest(epoch=EPOCH,
            executor_ref=planned['hand_id'], child_session='private-idea-child'), 'idea-bound')
        await env.ideas.reconcile()
        assert env.ideas.card(env.store.get(row.id))['buildState'] == 'working'
        packet = 'INV-204 is $480; PO-88 is $450: $30 difference. Reply draft remains unsent.'
        result = await c.result(planned['hand_id'], wire.ResultRequest(epoch=EPOCH, intent_version=1,
            status='done', summary_ref='private-result-branch', summary_text=packet,
            summary_hash=hashlib.sha256(packet.encode()).hexdigest(),
            tool_digest=[{'tool': 'connectors__google__gmail_get', 'result_hash': hashlib.sha256(packet.encode()).hexdigest()}], receipts=[],
            families_used=['family-a']), 'idea-result')
        assert result['outcome'] == 'done'
        await env.ideas.reconcile()
        detail = (await client.get(f'{ROOT}/{row.id}?contract=2')).json()['card']
        assert detail['buildStatus'] == detail['build_status'] == 'complete'
        assert detail['buildState'] == 'done' and detail['result']['packet'] == packet
        assert detail['result']['branch'] == 'private-result-branch'
        assert detail['result']['evidence'] == result['evidence']
        assert (await client.get(ROOT + '?contract=2')).json()['viewerState']['hasBuiltIdea'] is True
    assert env.store._db.execute('SELECT count(*) FROM initiative_engagement').fetchone()[0] == 0
    assert not any(h.action.startswith('feedback_') or h.action.startswith('llm_') for h in env.store.get_history(row.id))


async def test_only_actionable_build_states_carry_a_badge():
    from colonymind.mind.ideas_presentation import presentation
    assert presentation('packet')['badges'] == [] and 'display_badge' not in presentation('packet')
    for state, text in {'accepted': 'Queued', 'working': 'Building', 'done': 'Ready', 'failed': 'Build failed'}.items():
        value = presentation('packet', state)
        assert value['badges'] == [{'text': text}] and value['display_badge'] == {'text': text}
        assert value['labels'] == [] and value['actionLabel'] == 'Build'
