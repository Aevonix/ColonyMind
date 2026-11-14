"""Outside sources stay scoped, bounded and separate from the body's board."""
import base64
import hashlib
import io
import json
from datetime import timedelta

import pytest
from PIL import Image

from test_mind_loop import Fixture, OWNER, CONTACT, pinned_clock  # noqa: F401


@pytest.fixture
def fx(tmp_path, monkeypatch):
    monkeypatch.setenv('COLONYMIND_OWNER_CONTACT_ID', OWNER)
    fixture = Fixture(tmp_path, config={'faculties': {'outreach': False}})
    yield fixture
    fixture.store.close()


def observation(fx, *, key='sample', kind='numeric_surprise', **changes):
    return {'key': key, 'kind': kind, 'subject': 'device:unit', 'feature': 'capacity',
            'observed_at': fx.now.isoformat(), 'severity': 0.8,
            'summary': 'Capacity changed unexpectedly', **changes}


async def post(fx, source='sensor', items=None):
    return await fx.mind.observe_source(source, {'source': source, 'observations': items or []})


def outside(fx):
    from colonymind.mind.observations import for_mind
    return for_mind(fx.mind)


async def test_source_replacement_board_isolation_and_capture_order(fx):
    fx.mind.observe({'stale_tasks': [{'id': 'task', 'age_hours': 80}]})
    await post(fx, items=[observation(fx, key='late'), observation(
        fx, key='early', observed_at=(fx.now - timedelta(minutes=5)).isoformat())])
    assert [x['key'] for x in outside(fx).active(viewer=OWNER)] == ['early', 'late']
    assert fx.mind.observations['stale_task'][0]['id'] == 'task'
    fx.mind.observe({'stale_tasks': []})
    assert len(outside(fx).active(viewer=OWNER)) == 2
    await post(fx, 'second', [observation(fx, key='other')])
    await post(fx, items=[])
    assert [x['source'] for x in outside(fx).active(viewer=OWNER)] == ['second']


async def test_expiry_staleness_cap_and_restart(fx):
    stale = observation(fx, key='stale', observed_at=(fx.now - timedelta(hours=3)).isoformat())
    assert (await post(fx, items=[stale]))['accepted'] == 0
    assert not fx.mind.concerns.open()
    assert (await post(fx, items=[observation(fx, key=str(i)) for i in range(9)]))['accepted'] == 6
    fx.restart()
    assert (await post(fx, items=[observation(fx, key='extra')]))['accepted'] == 0
    assert outside(fx).active(source='missing', viewer=OWNER) == []
    fx.shift(hours=3)
    assert outside(fx).active(viewer=OWNER) == []


async def test_numeric_surprise_is_one_note_with_no_action_across_full_ticks(fx):
    await post(fx, items=[observation(fx)])
    await post(fx, 'second', [observation(fx, key='repeat')])
    for _ in range(3):
        await fx.mind.tick(force=True)
    concerns = fx.mind.concerns.open()
    assert len(concerns) == 1 and concerns[0].drive == 'upkeep'
    assert concerns[0].detail['note_only'] is True
    rows = fx.expectations.store.resolved_since(0, include_notes=True)
    assert len(rows) == 1 and rows[0].domain == 'upkeep' and rows[0].outcome == 'miss'
    assert fx.mind.dispatch() == []
    assert await fx.mind.outbox_ready() == []
    assert fx.store.list(limit=100) == []
    assert fx.mind.owner_gate.ledger.recent(since=fx.now - timedelta(days=1)) == []
    assert fx.mind.goals.open() == []
    assert fx.mind.feelings._misses(fx.now) == []
    assert (fx.mind.mind_state.get('affect.curiosity') or {}).get('level', 0) == 0


async def test_off_accepts_ignores_and_views_require_owner(fx):
    fx.mind.faculties['senses'] = False
    result = await post(fx, items=[observation(fx)])
    assert result['ignored'] == 1 and result['enabled'] is False
    assert not fx.mind.concerns.open()
    assert fx.expectations.store.resolved_since(0, include_notes=True) == []
    assert outside(fx).active(viewer=OWNER) == []
    fx.mind.faculties['senses'] = True
    await post(fx, items=[observation(fx, key='enabled')])
    assert outside(fx).active(viewer=CONTACT) == []
    assert outside(fx).active(viewer=None) == []


def clip():
    stream = io.BytesIO()
    Image.new('RGB', (4, 4), 'blue').save(stream, format='PNG')
    data = stream.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(data).decode(), hashlib.sha256(data).hexdigest()


async def test_percept_original_scope_capture_time_and_habituation(fx):
    image, digest = clip()
    capture = fx.now - timedelta(minutes=30)
    item = observation(fx, kind='percept', image=image, frame_hash=digest,
                       device_principal='device:unit', observed_at=capture.isoformat())
    await post(fx, items=[item])
    row = outside(fx).active(viewer=OWNER)[0]
    refs = fx.ledger.source_references([row['source_id']], contact_id=OWNER, session_id='later')
    assert refs and fx.ledger.source_references([row['source_id']], contact_id=CONTACT, session_id='later') == []
    with fx.ledger._connect() as conn:
        stored = conn.execute('SELECT occurred_at,messages_json FROM turn_sources WHERE turn_id=?',
                              (row['source_id'],)).fetchone()
    assert stored['occurred_at'] == capture.isoformat()
    provenance = json.loads(stored['messages_json'])[0]['content'][0]
    assert provenance['kind'] == 'observation' and provenance['epistemic_state'] == 'derived_unverified'
    assert provenance['frame_hash'] == digest and provenance['device_principal'] == 'device:unit'
    assert fx.mind.mind_state.get('affect.curiosity')['level'] <= 0.7
    concern = fx.mind.concerns.open()[0]
    await post(fx, items=[{**item, 'key': 'calm', 'severity': 0.0, 'observed_at': fx.now.isoformat()}])
    assert fx.mind.concerns.get(concern.id).status == 'open'
    assert len(fx.mind.concerns.open()) == 1


@pytest.mark.parametrize('changes', [
    {'observed_at': '2026-01-01T00:00:00'}, {'severity': float('nan')},
    {'kind': 'task'}, {'summary': ''}, {'expires_at': 'bad'},
])
async def test_invalid_batch_is_atomic(fx, changes):
    with pytest.raises(ValueError):
        await post(fx, items=[observation(fx), observation(fx, key='bad', **changes)])
    assert not fx.mind.concerns.open()


async def test_changed_replay_rejects_entire_batch_before_any_effect(fx):
    item = observation(fx, key='existing')
    await post(fx, items=[item])
    with pytest.raises(ValueError):
        await post(fx, items=[observation(fx, key='new', subject='device:other',
            observed_at=(fx.now - timedelta(minutes=1)).isoformat()), {**item, 'summary': 'Changed'}])
    assert len(fx.mind.concerns.open()) == 1


async def test_retry_completes_note_after_an_interrupted_concern_write(fx, monkeypatch):
    bump = fx.mind.concerns.bump
    def fail(**kwargs):
        raise RuntimeError('interrupted')
    monkeypatch.setattr(fx.mind.concerns, 'bump', fail)
    with pytest.raises(RuntimeError):
        await post(fx, items=[observation(fx)])
    monkeypatch.setattr(fx.mind.concerns, 'bump', bump)
    await post(fx, items=[observation(fx)])
    assert len(fx.mind.concerns.open()) == 1


async def test_clip_limit_and_frame_hash_are_verified_before_retention(fx):
    image, digest = clip()
    for changes in ({'image': 'data:image/png;base64,' + 'A' * 800000}, {'frame_hash': '0' * 64}):
        with pytest.raises(ValueError):
            await post(fx, items=[observation(fx, kind='percept', image=image, frame_hash=digest,
                                            device_principal='device:unit') | changes])
    assert not fx.mind.concerns.open()


async def test_percept_uncertain_reader_can_use_budgeted_deliberation_with_source_refs(fx):
    from types import SimpleNamespace
    from colonymind.instinct.points import PERCEPT_ATTEND
    from test_instinct_core import Provider, engine, record
    calls = []
    class Router:
        supports_function_routing = True
        async def complete(self, messages, *, context):
            calls.append(context)
            return SimpleNamespace(content='{"route":"note"}', usage={'total_tokens': 12})
    made = engine([PERCEPT_ATTEND], reader=Provider(percept_attend={
        label: 0.25 for label in PERCEPT_ATTEND.labels}), records=[record(PERCEPT_ATTEND, 'reader')])
    decide = made.decide
    observed = []
    async def capture(point, fields, refs=None, **kwargs):
        observed.append(refs)
        return await decide(point, fields, refs, **kwargs)
    made.decide = capture
    fx.mind.instinct, fx.mind.router = made, Router()
    image, digest = clip()
    await post(fx, items=[observation(fx, kind='percept', image=image, frame_hash=digest,
                                     device_principal='device:unit')])
    assert len(calls) == 1 and calls[0]['allow_fallback'] is False
    assert observed[0]['observation'].startswith('observation:')
    charged, = fx.store.intentions(kind=['note'])
    assert charged.cost_tokens == 12
    assert fx.mind.dispatch() == [] and await fx.mind.outbox_ready() == []


async def test_route_narrowed_while_deciding_cannot_dispatch(fx):
    from colonymind.instinct.points import PERCEPT_ATTEND
    from test_instinct_core import engine
    fx.mind.config['observations'] = {'routes': {'sensor': 'look'}}
    made = engine([PERCEPT_ATTEND])
    decide = made.decide
    async def narrow(*args, **kwargs):
        result = await decide(*args, **kwargs)
        fx.mind.config['observations']['routes']['sensor'] = 'ignore'
        return result
    made.decide = narrow
    fx.mind.instinct = made
    image, digest = clip()
    await post(fx, items=[observation(fx, kind='percept', image=image, frame_hash=digest,
                                     device_principal='device:unit')])
    assert fx.mind.dispatch() == []

from colonymind.memory import curated

async def test_active_percepts_require_the_current_retained_source(fx):
    image, digest = clip()
    await post(fx, items=[observation(fx, kind='percept', image=image, frame_hash=digest,
                                     device_principal='device:unit')])
    row, = outside(fx).active(viewer=OWNER)
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(fx.ledger)
    _source_forget_receipt = curated.forget_sources(fx.ledger, OWNER, source_ids=[row['source_id']])
    assert historical_bytes(fx.ledger) == _native_before_forget
    assert all(curated.is_forgotten(fx.ledger, sid) for sid in _source_forget_receipt['source_ids'])
    assert outside(fx).active(viewer=OWNER) == []


async def test_percept_descriptions_live_only_in_the_erasable_source(fx):
    image, digest = clip()
    summary = 'An unverified scene detail beside the window.'
    await post(fx, items=[observation(fx, kind='percept', image=image, frame_hash=digest,
                                     device_principal='device:unit', summary=summary)])
    source, = outside(fx).active(viewer=OWNER)
    assert source['summary'] == summary
    assert all(summary not in state['text'] for state in fx.mind.mind_state.items('observations.source:'))
    assert summary not in fx.mind.section()
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(fx.ledger)
    _source_forget_receipt = curated.forget_sources(fx.ledger, OWNER, source_ids=[source['source_id']])
    assert historical_bytes(fx.ledger) == _native_before_forget
    assert all(curated.is_forgotten(fx.ledger, sid) for sid in _source_forget_receipt['source_ids'])
    assert outside(fx).active(viewer=OWNER) == []
    assert summary not in fx.mind.section()
