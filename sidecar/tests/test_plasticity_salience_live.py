"""Salience acceptance: owner evidence, bounded observations, gates and erasure.

Fixture assumptions: twelve UTC weeks, a balanced weekly usefulness pattern,
fixed synthetic identities and a .01 learning rate for a separated shuffled
control, and the shared numeric comparator's thirty-outcome minimum.
"""

import copy
import json
import math
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from test_plasticity_wiring_runtime import learner_config as load_config
from colonymind.initiatives.store import InitiativeStore
from colonymind.initiatives import store as initiative_module
from colonymind.instinct.outcomes import Sources
from colonymind.mind.rank import Candidate
from colonymind.plasticity import _io
from colonymind.plasticity.gates import Check, GateReport
from colonymind.plasticity.learners import salience
from colonymind.plasticity.nightly import Runtime
from test_mind_drives_loop import fx, idle  # noqa: F401
from test_mind_loop import pinned_clock  # noqa: F401


NOW = datetime(2026, 4, 6, 12, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)
NAME = 'salience-test'
WORDS = 'The owner asked about the quiet garden by the blue gate.'


def runtime(path, now, **changes):
    path.mkdir(exist_ok=True)
    key = path / 'plasticity' / 'comparator.key'
    if not key.exists():
        key.parent.mkdir(exist_ok=True)
        _io.write_atomic(key, uuid.UUID('64967592-43e7-4f12-859c-e9976682db12').bytes * 2)
    declaration = copy.deepcopy(salience.DECLARATION)
    declaration.update(changes)
    (path / 'colonymind.yaml').write_text(yaml.safe_dump({
        'plasticity': {'learners': {NAME: declaration}}}))
    return Runtime(load_config(path, environ={}),
                   sources={NAME: SimpleNamespace(sources=Sources(path, owner_id='Owner'))},
                   clock=lambda: now[0])


def candidate(ref, **changes):
    fields = dict(type='research', kind='task', drive='curiosity', title=WORDS,
                  text=WORDS, rationale=WORDS, evidence=[WORDS], concern=WORDS,
                  dedup_key=ref, salience=.9, cost=.1, concern_kind='interest')
    fields.update(changes)
    return Candidate(**fields)


def intention(store, service, ref, at, *, verdict=None, by='owner',
              decision='act', status='done', dependencies=(), **changes):
    item = candidate(ref, **changes)
    key = service.loaded[NAME].driver.rows.key
    frame = salience.frame(item, at, {}, key)
    context = {'text': WORDS, 'body': WORDS, 'evidence': [WORDS], 'salience': {
        'frame': frame, 'source_ids': list(dependencies),
        'concern': salience.keyed(key, ['concern', ref])}}
    row, _ = store.create_intention(kind=item.kind, type=item.type, title=WORDS,
        drive=item.drive, cls='internal', decision=decision, decision_reason='fixture',
        status=status, dedup_key=ref, context=context, source_type=item.source_type,
        source_id=item.source_id, created_at=at)
    if verdict is not None:
        rate(store, row, verdict, at=at + HOUR / 2, by=by)
    return item, store.get(row.id)


def rate(store, row, verdict, *, at=NOW, by='owner'):
    return store.transition(row.id, row.status, action='rated',
                            details={'verdict': verdict, 'by': by},
                            at=at, verdict=verdict)


def observation(items, *, acted=()):
    return SimpleNamespace(eligible=tuple((item, 1. - i / 100.) for i, item in enumerate(items)),
        context={}, acted=tuple(acted), rescore=lambda item, overrides: item.salience *
            (1. - item.cost) * overrides.get('salience', 1.))


def assert_private(path):
    assert all(WORDS.encode() not in file.read_bytes()
               for file in (path / 'plasticity').rglob('*') if file.is_file())


def test_registered_rows_owner_verdicts_and_acted_discretionary_only(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    service = runtime(tmp_path, [NOW])
    try:
        assert service.find('salience') == NAME, service.rejected
        binding = service.loaded[NAME]
        assert binding.store.mode == 'rebuild'
        assert binding.comparator.variant == 'logistic'
        expected = {}
        for verdict in ('useful', 'actioned', 'not_useful', 'wrong', 'dismissed'):
            _, row = intention(store, service, 'rated-' + verdict, NOW - 2 * HOUR,
                               verdict=verdict, dependencies=('feature-turn',))
            expected[row.id] = float(verdict in {'useful', 'actioned'})
        excluded = []
        cases = [dict(by='operator'), dict(decision='ask', status='asked'),
                 dict(decision='defer', status='proposed'), dict(source_type='commitment'),
                 dict(type='outreach_answer'), dict(type='outreach_followup')]
        for index, changes in enumerate(cases):
            _, row = intention(store, service, f'excluded-{index}', NOW - 2 * HOUR,
                               verdict='useful', **changes)
            excluded.append(row.id)
        rows = list(binding.driver.source.rows(binding.driver.objective))
        assert len(rows) == len(expected)
        seen = set()
        for row in rows:
            deps = set(binding.driver.rows.source_ids(row['ref']))
            original, = deps.intersection(expected)
            seen.add(original)
            assert row['label'] == expected[original]
            assert row['teacher'] == 'owner_label' and not row['push_away']
            assert 'feature-turn' in deps
            assert not deps.intersection(excluded)
        assert seen == set(expected)
        assert set(binding.driver.rows.refs_resting_on(['feature-turn'])) >= {r['ref'] for r in rows}
        binding.driver.rows.sources.withheld = lambda ids: {'feature-turn'} & set(ids)
        assert list(binding.driver.source.rows(binding.driver.objective)) == []
        assert_private(tmp_path)
    finally:
        service.close()
        store.close()


def test_unreacted_intentions_wait_48_hours_and_only_push_away(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    now = [NOW]
    service = runtime(tmp_path, now)
    try:
        _, old = intention(store, service, 'old-ignore', NOW - 48 * HOUR)
        _, young = intention(store, service, 'young-ignore', NOW - 48 * HOUR + timedelta(seconds=1))
        _, unacted = intention(store, service, 'unacted-ignore', NOW - 60 * HOUR,
                                decision='ask', status='asked')
        driver = service.loaded[NAME].driver
        rows = list(driver.source.rows(driver.objective))
        assert len(rows) == 1
        assert old.id in driver.rows.source_ids(rows[0]['ref'])
        assert rows[0]['teacher'] == 'behavioural_proxy' and rows[0]['push_away']
        now[0] += timedelta(seconds=1)
        rows = list(driver.source.rows(driver.objective))
        assert len(rows) == 2
        assert {old.id, young.id} == {next(ref for ref in driver.rows.source_ids(row['ref'])
                                       if ref in {old.id, young.id, unacted.id}) for row in rows}
        assert all(row['teacher'] == 'behavioural_proxy' and row['push_away'] for row in rows)
    finally:
        service.close()
        store.close()


def test_propensity_log_is_bounded_keyed_and_only_eligible(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    service = runtime(tmp_path, [NOW])
    try:
        items = [candidate(f'candidate-sensitive-{i}') for i in range(25)]
        _, row = intention(store, service, items[0].dedup_key, NOW)
        salience.observe(service, NAME, observation(items, acted=((items[0], row.id),)), NOW)
        path = service.loaded[NAME].store.directory / 'salience-propensities.json'
        batches = json.loads(path.read_text())
        assert len(batches) == 1
        batch = batches[0]
        assert batch['probability'] == 0.0 and not batch['exploration']
        assert len(batch['candidates']) == 20
        assert sum(item['acted'] for item in batch['candidates']) == 1
        for item in batch['candidates']:
            assert len(item['ref']) == 64
            assert .5 <= item['served_factor'] <= 1.5
            assert .5 <= item['reference_factor'] <= 1.5
            assert item['served_rank'] >= 0 and item['reference_rank'] >= 0
        assert 'candidate-sensitive-' not in path.read_text() and row.id not in path.read_text()
        assert len(service.loaded[NAME].comparator.records()) == 1
        assert_private(tmp_path)
    finally:
        service.close()
        store.close()


def test_observe_serves_once_and_live_poll_labels_only_owner_verdict(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    service = runtime(tmp_path, [NOW])
    try:
        item, row = intention(store, service, 'served-once', NOW)
        unformed = candidate('never-formed')
        view = observation([item, unformed], acted=((item, row.id),))
        salience.observe(service, NAME, view, NOW)
        salience.observe(service, NAME, view, NOW)
        binding = service.loaded[NAME]
        binding.driver.poll(service)
        record, = binding.comparator.records()
        assert record['outcome'] is None and record['reference_output'] == .5
        rate(store, row, 'useful', by='operator')
        binding.driver.poll(service)
        assert binding.comparator.records()[0]['outcome'] is None
        rate(store, row, 'useful')
        binding.driver.poll(service)
        binding.driver.poll(service)
        record, = binding.comparator.records()
        assert record['outcome']['observed'] == 1.
        assert record['outcome']['teacher'] == 'owner_label'
        from colonymind.reports.plasticity import collect, lines
        report = ' '.join(lines(collect(service.config, runtime=service)))
        for fragment in ('Salience', 'Brier', 'log loss', 'selection bias: outcomes exist only for acted items'):
            assert fragment in report
        assert_private(tmp_path)
    finally:
        service.close()
        store.close()


def plant_weekly(store, service):
    # Shuffling canonical training rows also depends on their keyed references,
    # not just the RNG seed. Fix the synthetic key and row IDs for this control.
    start = NOW - timedelta(weeks=12)
    rows = []
    ids = iter(uuid.uuid5(uuid.NAMESPACE_URL, f'marker-64967592-43e7-4f12-859c-e9976682db12-{i}')
               for i in range(12 * 168))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(initiative_module, 'uuid', SimpleNamespace(uuid4=lambda: next(ids)))
        for index in range(12 * 168):
            at = start + index * HOUR
            useful = math.sin(2 * math.pi * (at.weekday() * 24 + at.hour) / 168) > 0
            _, row = intention(store, service, f'weekly-{index}', at,
                               verdict='useful' if useful else 'not_useful')
            rows.append(row)
    return rows


def test_nightly_beats_frozen_anchor_and_time_shuffled_control_does_not(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    service = runtime(tmp_path, [NOW], learning_rate=.01)
    try:
        plant_weekly(store, service)
        result, = service.run()
        driver = service.loaded[NAME].driver
        assert result['result'] == 'promoted', (result, driver.metrics)
        assert driver.last_report.passed
        assert driver.metrics['brier_gain_lower'] > 0
        assert driver.metrics['shuffled_gain_lower'] <= 0
        assert_private(tmp_path)
    finally:
        service.close()
        store.close()


def test_live_owner_outcomes_roll_back_a_planted_regression(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    service = runtime(tmp_path, [NOW])
    try:
        binding = service.loaded[NAME]
        state = binding.driver.learner.base()
        state['W'][0, -1] = 1.
        closure = binding.store.revisions.compose(state=_io.state_digest(state),
            objective=binding.driver.objective.identity, calibration=('fixture',), dependents={},
            parent=binding.store.revisions.serving().id, manifests=(), periods=(),
            audience='owner', mode='rebuild')
        binding.controller.propose(closure, state)
        binding.controller.consider(closure.id, GateReport(closure.id, 'fixture', (Check('budget', True),)))
        assert binding.store.revisions.serving().id == closure.id
        for index in range(30):
            item, row = intention(store, service, f'regression-{index}', NOW)
            salience.observe(service, NAME, observation([item], acted=((item, row.id),)), NOW)
            rate(store, row, 'not_useful')
            binding.driver.poll(service)
        assert len(binding.comparator.records()) == 30
        assert all(row['outcome']['teacher'] == 'owner_label' for row in binding.comparator.records())
        assert binding.store.revisions.serving().mode == 'base'
        assert binding.comparator.summary()['last_rollback']['rolled_back']
    finally:
        service.close()
        store.close()


def test_comparator_keeps_the_closures_used_to_rank_when_serving_changes(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    service = runtime(tmp_path, [NOW])
    try:
        binding = service.loaded[NAME]

        def promote(intercept):
            state = binding.driver.learner.base()
            state['W'][0, -1] = intercept
            closure = binding.store.revisions.compose(state=_io.state_digest(state),
                objective=binding.driver.objective.identity, calibration=('fixture',), dependents={},
                parent=binding.store.revisions.serving().id, manifests=(), periods=(),
                audience='owner', mode='rebuild')
            binding.controller.propose(closure, state)
            binding.controller.consider(closure.id,
                GateReport(closure.id, 'fixture', (Check('budget', True),)))
            return closure

        first = promote(.5)
        item, row = intention(store, service, 'pinned-pair', NOW)
        view = observation([item], acted=((item, row.id),))
        served_factor = salience.factor(service, NAME, item, NOW, context=view.context)
        assert served_factor > 1.
        second = promote(-.5)
        assert service.pin(NAME, 'owner').closure.id == second.id
        salience.observe(service, NAME, view, NOW)
        record, = binding.comparator.records()
        assert record['closure'] == first.id
        assert record['reference'] == binding.store.revisions.base.id
        assert record['output'] == pytest.approx(1. / (1. + math.exp(-.5)), rel=1e-6)
        assert record['reference_output'] == .5
        assert record['output'] / .5 == pytest.approx(served_factor)
    finally:
        service.close()
        store.close()


def test_propensity_history_has_count_and_age_bounds(tmp_path):
    now = [NOW]
    service = runtime(tmp_path, now)
    try:
        item = candidate('bounded-history')
        for index in range(205):
            now[0] = NOW + timedelta(minutes=index)
            salience.observe(service, NAME, observation([item]), now[0])
        path = service.loaded[NAME].store.directory / 'salience-propensities.json'
        history = json.loads(path.read_text())
        assert len(history) == 200
        assert datetime.fromisoformat(history[0]['at']) == NOW + timedelta(minutes=5)
        now[0] += timedelta(days=15)
        salience.observe(service, NAME, observation([item]), now[0])
        history = json.loads(path.read_text())
        assert len(history) == 1 and datetime.fromisoformat(history[0]['at']) == now[0]
        assert_private(tmp_path)
    finally:
        service.close()


def test_loaded_erasure_physically_removes_propensity_and_comparator_item(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    service = runtime(tmp_path, [NOW])
    try:
        first, first_row = intention(store, service, 'erase-first', NOW)
        second, second_row = intention(store, service, 'keep-second', NOW)
        salience.observe(service, NAME, observation([first, second],
            acted=((first, first_row.id), (second, second_row.id))), NOW)
        binding = service.loaded[NAME]
        removed_hash = salience.keyed(binding.driver.rows.key, ['intention', first_row.id])
        path = binding.store.directory / 'salience-propensities.json'
        assert removed_hash in path.read_text() and len(binding.comparator.records()) == 2
        store._db.execute('DELETE FROM assignment_history WHERE initiative_id=?', (first_row.id,))
        store._db.execute('DELETE FROM initiatives WHERE id=?', (first_row.id,))
        store._db.commit()
        service.erase_sources([first_row.id])
        assert not path.exists() or removed_hash not in path.read_text()
        record, = binding.comparator.records()
        assert record['ref_hash'] == binding.comparator._path(second_row.id).stem
    finally:
        service.close()
        store.close()


@pytest.mark.parametrize('state', ['disabled', 'omitted'])
def test_unloaded_source_erasure_removes_all_retained_propensity_bytes(tmp_path, state):
    store = InitiativeStore(state_dir=tmp_path)
    service = runtime(tmp_path, [NOW])
    try:
        item, row = intention(store, service, 'erase-unloaded', NOW)
        salience.observe(service, NAME, observation([item], acted=((item, row.id),)), NOW)
        binding = service.loaded[NAME]
        path = binding.store.directory / 'salience-propensities.json'
        assert path.exists() and path.read_text().strip() != '[]'
        config = service.config
        service.close()
        if state == 'omitted':
            config.data['plasticity']['learners'] = {}
        else:
            config.data['plasticity']['learners'][NAME]['enabled'] = False
        service = Runtime(config, clock=lambda: NOW)
        assert not service.loaded
        store._db.execute('DELETE FROM assignment_history WHERE initiative_id=?', (row.id,))
        store._db.execute('DELETE FROM initiatives WHERE id=?', (row.id,))
        store._db.commit()
        service.erase_sources([row.id])
        assert not path.exists() or path.read_text().strip() == '[]'
        assert not list((binding.store.directory / 'comparator').glob('numeric-*.jsonl'))
    finally:
        service.close()
        store.close()


async def test_async_live_driver_delivers_the_owner_outcome(tmp_path, monkeypatch):
    store = InitiativeStore(state_dir=tmp_path)
    service = runtime(tmp_path, [NOW])
    try:
        item, row = intention(store, service, 'live-loop', NOW)
        salience.observe(service, NAME, observation([item], acted=((item, row.id),)), NOW)
        rate(store, row, 'useful')

        class StopPolling(Exception):
            pass

        async def stop_after_poll(seconds):
            assert seconds == 60
            raise StopPolling

        monkeypatch.setattr(salience.asyncio, 'sleep', stop_after_poll)
        binding = service.loaded[NAME]
        with pytest.raises(StopPolling):
            await binding.driver.live(service)
        record, = binding.comparator.records()
        assert record['outcome']['observed'] == 1.
        assert record['outcome']['teacher'] == 'owner_label'
    finally:
        service.close()
        store.close()


async def test_real_mind_tick_captures_and_labels_one_salience_item(fx):
    service = runtime(fx.state, [fx.now])
    fx.mind.plasticity = service
    try:
        fx.mind.add_interest('local history')
        summary, = await idle(fx)
        formed, = summary['formed']
        assert formed['decision'] == 'act'
        binding = service.loaded[NAME]
        record, = binding.comparator.records()
        assert record['ref_hash'] == binding.comparator._path(formed['id']).stem
        source = fx.store.get(formed['id'])
        assert source.context['salience']['frame']
        assert source.context['learned']['salience'] == 1.
        fx.mind.rate(source.id, 'useful')
        binding.driver.poll(service)
        record, = binding.comparator.records()
        assert record['outcome']['teacher'] == 'owner_label'
        assert record['outcome']['observed'] == 1.
        await idle(fx)
        assert len(binding.comparator.records()) == 1
    finally:
        fx.mind.plasticity = None
        service.close()


def test_erasure_rebuild_matches_a_clean_build(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    service = runtime(tmp_path, [NOW], learning_rate=.01)
    try:
        plant_weekly(store, service)
        assert service.run()[0]['result'] == 'promoted'
        binding = service.loaded[NAME]
        old = binding.store.revisions.serving()
        target = next(row for row in binding.driver.source.rows(binding.driver.objective)
                      if not salience.gate_week(row['at']) and row['label'] == 1.)
        source_id = binding.driver.rows.source_ids(target['ref'])[0]
        store._db.execute('DELETE FROM assignment_history WHERE initiative_id=?', (source_id,))
        store._db.execute('DELETE FROM initiatives WHERE id=?', (source_id,))
        store._db.commit()
        service.erase_sources([source_id])
        assert not binding.store.revisions.exists(old.id)
        actual = service.pin(NAME, 'owner').state
        clean_dir = tmp_path / 'clean'
        clean_dir.mkdir()
        # Backup includes committed WAL pages, without copying the learned state.
        import sqlite3
        with sqlite3.connect(clean_dir / 'initiatives.db') as destination:
            store._db.backup(destination)
        (clean_dir / 'plasticity').mkdir()
        for filename in ('comparator.key', 'erased-refs.json', 'erased-sources.json'):
            source = tmp_path / 'plasticity' / filename
            if source.exists():
                shutil.copyfile(source, clean_dir / 'plasticity' / filename)
        clean = runtime(clean_dir, [NOW], learning_rate=.01)
        try:
            assert clean.run()[0]['result'] == 'promoted'
            expected = clean.pin(NAME, 'owner').state
            assert actual.keys() == expected.keys()
            for key in actual:
                np.testing.assert_allclose(actual[key], expected[key], rtol=1e-6, atol=1e-6)
        finally:
            clean.close()
    finally:
        service.close()
        store.close()
