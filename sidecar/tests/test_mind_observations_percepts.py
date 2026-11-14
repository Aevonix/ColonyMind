"""Percept routes retain original evidence and cross the existing work and message gates."""

import base64
from dataclasses import replace
from datetime import timedelta

import pytest

from colonymind.instinct.points import OWNER_SEND, PERCEPT_ATTEND
from colonymind.mind.affect import CAP
from colonymind.mind.gate import Message
from colonymind.turns.source_read import read
from test_instinct_core import Provider, engine, record
from test_mind_loop import OWNER, pinned_clock  # noqa: F401
from test_mind_observations_sources import clip, fx, observation, outside, post  # noqa: F401


def percept(fx, **changes):
    image, digest = clip()
    return observation(fx, kind='percept', image=image, frame_hash=digest,
                       device_principal='device:unit', **changes)


async def test_look_dispatch_names_the_original_and_the_source_reader_reopens_it(fx):
    fx.mind.config['observations'] = {'routes': {'sensor': 'look'}}
    item = percept(fx)
    await post(fx, items=[item])
    source, = outside(fx).active(viewer=OWNER)
    worker, = fx.mind.dispatch()
    assert worker['type'] == 'percept_look' and worker['kind'] == 'task'
    for key, value in {'source_view': 'image', 'source_id': source['source_id'],
                       'source_version': source['source_version'], 'asset_hash': item['frame_hash']}.items():
        assert key + '=' + value in worker['body']
    opened = read(fx.ledger, contact_id=OWNER, session_id='worker-session',
                  source_id=source['source_id'], source_version=source['source_version'],
                  view='image', asset_hash=item['frame_hash'])
    assert opened['image_bytes_included'] is True
    assert base64.b64decode(opened['image']['data_url'].split(',', 1)[1]) == base64.b64decode(
        item['image'].split(',', 1)[1])
    await post(fx, items=[item])
    assert [row['id'] for row in fx.mind.dispatch()] == [worker['id']]


@pytest.mark.parametrize('route', ['look', 'tell'])
async def test_no_original_keeps_a_note_without_work_or_a_message(fx, route):
    fx.mind.config['observations'] = {'routes': {'sensor': route}}
    await post(fx, items=[observation(fx, kind='percept')])
    concern, = fx.mind.concerns.open()
    assert concern.detail['note_only'] is True
    assert fx.store.list(limit=100) == []
    assert fx.mind.dispatch() == [] and await fx.mind.outbox_ready() == []


async def test_curiosity_from_many_sources_stays_within_the_existing_cap(fx):
    # Twelve sources is a synthetic stress count, an implementation test assumption.
    sources = [f'sensor-{index}' for index in range(12)]
    fx.mind.config['observations'] = {'routes': {source: 'note' for source in sources}}
    for source in sources:
        await post(fx, source, [percept(fx, subject='device:' + source)])
        assert 0 < fx.mind.mind_state.get('affect.curiosity')['level'] <= CAP
    assert fx.mind.mind_state.get('affect.curiosity')['level'] == CAP
    assert len(fx.mind.concerns.open()) == len(sources)
    assert fx.store.list(limit=100) == []


async def test_sender_route_fields_cannot_override_the_configured_note(fx):
    fx.mind.config['observations'] = {'routes': {'sensor': 'note'}}
    await post(fx, items=[percept(fx, route='tell', rule_route='look')])
    assert fx.store.list(limit=100) == []
    assert fx.mind.dispatch() == [] and await fx.mind.outbox_ready() == []


async def test_device_capture_does_not_count_as_the_owner_speaking(fx):
    assert fx.mind._owner_last_turn() is None
    await post(fx, items=[percept(fx)])
    assert fx.mind._owner_last_turn() is None


@pytest.mark.parametrize('switch', ['senses', 'mind'])
async def test_disabled_switch_cannot_dispatch_or_message(fx, switch):
    fx.mind.config['observations'] = {'routes': {'sensor': 'look'}}
    if switch == 'senses':
        fx.mind.faculties['senses'] = False
    else:
        fx.mind.off(reason='observation test')
    before = [row.id for row in fx.store.intentions(limit=100)]
    await post(fx, items=[percept(fx)])
    assert [row.id for row in fx.store.intentions(limit=100)] == before
    assert fx.mind.dispatch() == [] and await fx.mind.outbox_ready() == []
    assert fx.mind.owner_gate.ledger.recent(since=fx.now - timedelta(days=1)) == []


async def test_calibrated_tell_is_withheld_when_the_owner_gate_holds(fx):
    provider = Provider(percept_attend={name: float(name == 'tell') for name in PERCEPT_ATTEND.labels})
    fx.mind.instinct = engine([PERCEPT_ATTEND], reader=provider, records=[record(PERCEPT_ATTEND, 'reader')])
    fx.mind.owner_gate.enabled = True
    # The synthetic confidence forces a second opinion on the ordinary mind prior.
    fx.mind.owner_gate.settings = replace(fx.mind.owner_gate.settings, confidence=0.99)
    fx.mind.owner_gate.instinct = engine([OWNER_SEND])
    judged = []

    async def judge(system, prompt):
        judged.append(prompt)
        return {'choice': 'digest', 'reason': 'This observation can wait.'}

    fx.mind.owner_gate.judge = judge
    await post(fx, items=[percept(fx)])
    assert len(provider.calls) == 1
    message, = fx.store.intentions(kind=['message'], limit=100)
    assert message.type == 'percept_tell'
    assert message.status == 'approved', message.decision_reason
    assert await fx.mind.outbox_ready() == []
    verdict, = fx.mind.owner_gate.ledger.recent(since=fx.now - timedelta(days=1))
    assert len(judged) == 1, (verdict['stage'], verdict['reason'])
    assert verdict['decision'] == 'digest'
    assert fx.store.get(message.id).status == 'expired'


async def test_owner_gate_model_send_cannot_widen_a_rules_hold(fx):
    fx.mind.owner_gate.enabled = True
    # At this synthetic confidence the low-urgency digest prior asks its second opinion.
    fx.mind.owner_gate.settings = replace(fx.mind.owner_gate.settings, confidence=0.99)
    fx.mind.owner_gate.instinct = engine([OWNER_SEND])
    judged = []

    async def judge(system, prompt):
        judged.append(prompt)
        return {'choice': 'send_now', 'reason': 'The model would interrupt now.'}

    fx.mind.owner_gate.judge = judge
    verdict = await fx.mind.owner_gate.review(Message(type='percept_tell', topic='device observation',
        text='An unverified observation changed.', urgency='low', source='sensor', ref='observation-1'))
    assert len(judged) == 1
    assert verdict.decision == 'digest'


from colonymind.memory import curated
async def test_erasure_while_attention_is_deciding_prevents_a_worker(fx, monkeypatch):
    fx.mind.config['observations'] = {'routes': {'sensor': 'look'}}
    made = engine([PERCEPT_ATTEND])
    decide = made.decide

    async def erase_before_return(point, fields, *args, **kwargs):
        decision = await decide(point, fields, *args, **kwargs)
        from test_curated_memory import historical_bytes
        _native_before_withdrawal = historical_bytes(fx.ledger)
        curated.forget_sources(fx.ledger, OWNER, source_ids=[fields['source_id']])
        assert historical_bytes(fx.ledger) == _native_before_withdrawal
        return decision
    monkeypatch.setattr(made, 'decide', erase_before_return)
    fx.mind.instinct = made
    await post(fx, items=[percept(fx)])
    assert fx.mind.dispatch() == []
    assert fx.store.intentions(kind=['task'], limit=100) == []
