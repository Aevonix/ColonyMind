"""Nightly titles have bounded model authority and durable retry selection."""
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json

import pytest

from colonymind.mind.consolidate import Night
from colonymind.work.night import WorkNight, ROOTS_PER_NIGHT, TASK
from test_work_projection import stores, admit, envelope

NOW = datetime(2026, 10, 3, tzinfo=timezone.utc)


def record():
    return Night(local_date=NOW.date().isoformat(), started_at=NOW.isoformat())


async def test_titles_bounded_and_backlog_survives_restart(stores):
    for i in range(19):
        admit(stores, envelope(f'root-{i:02}'), source_id=f'root-{i:02}',
              messages=[{'role': 'user', 'content': 'Please update the report. ' * 100}])
    stores.work.reconcile()
    calls = []
    async def call(night, **kw):
        inputs = json.loads(kw['user'])
        calls.append(inputs)
        assert len(inputs) == 1
        assert all(len(item['request']) <= 600 and not item.get('tools') for item in inputs)
        assert kw['task'] == TASK and kw['max_output_tokens'] == 768
        return {'titles': [{'id': item['id'], 'title': 'Update the report'} for item in inputs]}
    night = record()
    await WorkNight(stores.ledger, 'owner').run(night, NOW, call=call)
    assert night.counts['work_titled'] == ROOTS_PER_NIGHT
    await WorkNight(stores.ledger, 'owner').run(record(), NOW, call=call)
    assert len(calls) == 19 and all(len(items) == 1 for items in calls)
    assert all(row['title'] == 'Update the report' and row['status'] == 'proven'
               for row in stores.work.list_roots(contact_id='owner'))


@pytest.mark.parametrize('answer', [None, {}, {'titles': [{'id': 'w1', 'title': 'Done'}]},
    {'titles': [{'id': 'w1', 'title': 'one two three four five six seven eight nine'}]},
    {'titles': [{'id': 'w1', 'title': 'Update the report', 'status': 'proven'}]},
    {'titles': [{'id': 'w9', 'title': 'Update the report'}]},
    {'titles': [{'id': 'w1', 'title': 'Update the report'}] * 2}])
async def test_failed_night_leaves_untitled_and_retry_watermark(stores, answer):
    admit(stores)
    stores.work.reconcile()
    calls = []
    async def call(night, **kw):
        calls.append(kw)
        return answer
    await WorkNight(stores.ledger, 'owner').run(record(), NOW, call=call)
    assert stores.work.get_root('root', contact_id='owner')['title'] is None
    await WorkNight(stores.ledger, 'owner').run(record(), NOW + timedelta(minutes=30), call=call)
    assert len(calls) == 1
    await WorkNight(stores.ledger, 'owner').run(record(), NOW + timedelta(hours=1), call=call)
    assert len(calls) == 2
    with closing(stores.work._connect()) as conn:
        assert conn.execute('SELECT attempts FROM work_title_attempts').fetchone()[0] == 2


async def test_work_stage_uses_budget_without_lessons(tmp_path, monkeypatch):
    from test_mind_consolidate import Fixture, NightRouter, OWNER
    monkeypatch.setenv('COLONYMIND_OWNER_CONTACT_ID', OWNER)
    fx = Fixture(tmp_path, config={'faculties': {'consolidation': False, 'lessons': False, 'work_records': True}},
                 router=NightRouter({TASK: {'titles': [{'id': 'w1', 'title': 'Update the report'}]}}))
    fx.ledger.record_source('root', contact_id=OWNER, session_id='session',
        messages=[{'role': 'user', 'content': 'Update the report.'}],
        work={'envelope': envelope(), 'principal_id': 'host'}, derive_claims=False)
    work = WorkNight(fx.ledger, OWNER)
    work.store.reconcile()
    night = await fx.mind.consolidate()
    assert night['done'] == ['work'] and night['tokens'] == 100 and night['calls'] == 1
    assert work.store.get_root('root', contact_id=OWNER)['title'] == 'Update the report'


async def test_cancelled_call_persists_retry_selection(stores):
    import asyncio
    admit(stores)
    stores.work.reconcile()
    async def call(*args, **kw):
        raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await WorkNight(stores.ledger, 'owner').run(record(), NOW, call=call)
    assert WorkNight(stores.ledger, 'owner')._select(NOW) == []
    assert len(WorkNight(stores.ledger, 'owner')._select(NOW + timedelta(hours=1))) == 1


def test_published_title_schema_matches_runtime():
    from pathlib import Path
    from colonymind.work.models import WorkTitles
    from test_work_contract import validator
    published = json.loads((Path(__file__).parents[1] / 'colonymind/work/schema/work-titles.json').read_text())
    assert {key: value for key, value in published.items() if key not in {'$schema', '$id'}} == WorkTitles.model_json_schema()
    validator('work-titles.json').validate({'titles': [{'id': 'w1', 'title': 'Update the report'}]})


def test_failed_batch_cannot_starve_unattempted_roots(stores):
    for i in range(19):
        admit(stores, envelope(f'root-{i:02}'), source_id=f'root-{i:02}')
    stores.work.reconcile()
    worker = WorkNight(stores.ledger, 'owner')
    first = worker._select(NOW)
    second = worker._select(NOW + timedelta(days=1))
    old = {item['root']['id'] for item in first}
    assert all(item['root']['id'] not in old for item in second[:3])


from colonymind.memory import curated
async def test_erasure_during_title_call_cannot_resurrect(stores):
    admit(stores)
    stores.work.reconcile()

    async def call(night, **kw):
        from test_curated_memory import historical_bytes
        _native_before_withdrawal = historical_bytes(stores.ledger)
        curated.forget_sources(stores.ledger, 'owner', source_ids=['root'])
        assert historical_bytes(stores.ledger) == _native_before_withdrawal
        return {'titles': [{'id': 'w1', 'title': 'Update the report'}]}
    await WorkNight(stores.ledger, 'owner').run(record(), NOW, call=call)
    assert stores.work.get_root('root', contact_id='owner') is None

async def test_stale_title_never_adds_secret_to_vault(stores):
    admit(stores)
    stores.work.reconcile()
    before = stores.ledger.vault.entries()

    async def call(night, **kw):
        from test_curated_memory import historical_bytes
        _native_before_withdrawal = historical_bytes(stores.ledger)
        curated.forget_sources(stores.ledger, 'owner', source_ids=['root'])
        assert historical_bytes(stores.ledger) == _native_before_withdrawal
        return {'titles': [{'id': 'w1', 'title': 'Rotate token ghp_' + 'a' * 36}]}
    await WorkNight(stores.ledger, 'owner').run(record(), NOW, call=call)
    assert stores.ledger.vault.entries() == before
