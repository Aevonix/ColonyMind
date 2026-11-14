"""Action forecasts use verified outcomes, bounded rank and paired live evidence.

Fixture assumptions: five weeks of hourly tasks, a risky third succeeding 20%
of the time and the remaining tasks succeeding 95%, for a .7 overall prior.
"""
import copy
import json
import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from test_plasticity_wiring_runtime import learner_config as load_config
from colonymind.initiatives.store import InitiativeStore
from colonymind.instinct.outcomes import Sources
from colonymind.mind.learned import LearnedFactors
from colonymind.mind.rank import Candidate, eligible, rank
from colonymind.mind.tick import Mind
from colonymind.plasticity import _io
from colonymind.plasticity.gates import Check, GateReport
from colonymind.plasticity.learners.action_success import (
    DECLARATION, OBJECTIVE, ActionLearner, factor, label, note, report,
)
from colonymind.plasticity.nightly import Runtime

NOW = datetime(2026, 4, 6, 12, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)
WORDS = ('A private task title about the garden.', 'Inspect the garden and return a count.',
         'The garden irrigation topic.', 'The inspection found three faults.')


def runtime(path, now, **changes):
    path.mkdir(exist_ok=True)
    spec = {**copy.deepcopy(DECLARATION), **changes}
    (path / 'colonymind.yaml').write_text(yaml.safe_dump({'plasticity': {'learners': {'success': spec}}}))
    service = Runtime(load_config(path, environ={}),
        sources={'success': SimpleNamespace(sources=Sources(path, owner_id='Owner'))}, clock=lambda: now[0])
    assert not service.rejected
    return service


def task(store, at, *, kind='research', outcome=None, verified='none', topic=WORDS[2], turn='turn-1'):
    row, _ = store.create_intention(kind='task', type=kind, title=WORDS[0], drive='curiosity',
        cls='internal', decision='act', decision_reason='fixture', status='approved', dedup_key=None,
        context={'body': WORDS[1], 'topic': topic, 'evidence': ['turn:' + turn], 'open_ended': True,
                 'max_runtime_seconds': 600, 'max_retries': 1},
        success_check={'kind': 'result_field', 'field': 'finding'}, hermes_kind='none', created_at=at)
    if outcome:
        row = store.update(row.id, outcome=outcome, verified=verified, result=WORDS[3],
                           status=outcome, completed_at=at + HOUR / 2, assigned_at=at)
    return row


def candidate(**changes):
    return Candidate(**{'type': 'research', 'kind': 'task', 'drive': 'curiosity', 'title': WORDS[0],
                        'text': WORDS[1], 'topic': WORDS[2], 'dedup_key': 'task-fixture', **changes})


def dispatch_mind(store, service=None):
    from colonymind.mind.authority import Budgets
    mind = SimpleNamespace(store=store, clock=lambda: NOW, enabled=True, plasticity=service,
        _blocked=lambda row: False, feelings=SimpleNamespace(note_for=lambda topic: ''),
        policy=SimpleNamespace(budgets=Budgets()))
    mind.learned = LearnedFactors(mind)
    return mind


def promote(service, intercept):
    binding = service.loaded['success']
    state = binding.driver.learner.base()
    state['W'][0, -1] = intercept
    closure = binding.store.revisions.compose(state=_io.state_digest(state), objective=OBJECTIVE.identity,
        calibration=('fixture',), dependents={}, parent=binding.store.revisions.serving().id,
        manifests=(), periods=(), audience='owner', mode='rebuild')
    binding.controller.propose(closure, state)
    binding.controller.consider(closure.id, GateReport(closure.id, 'fixture', (Check('budget', True),)))
    return closure


def test_no_learner_preserves_ranking_and_dispatch(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    try:
        row = task(store, NOW)
        mind = dispatch_mind(store)
        candidates = [candidate(), candidate(kind='message', dedup_key='message')]
        for operation in (rank, eligible):
            assert operation(candidates) == operation(candidates, learned=mind.learned.at(NOW))
        first, = Mind.dispatch(mind)
        assert first['body'] == row.context['body']
        assert Mind.dispatch(mind) == [first]
    finally:
        store.close()


def test_anchor_exact_and_one_comparator_item_per_dispatch(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    service = runtime(tmp_path, [NOW])
    try:
        row = task(store, NOW)
        assert factor(service, 'success', candidate(), NOW) == 1.
        assert note(service, 'success', row, NOW) is None
        assert note(service, 'success', store.get(row.id), NOW) is None
        record, = service.loaded['success'].comparator.records()
        assert record['output'] == record['reference_output'] == .7
        assert record['outcome'] is None
        assert service.loaded['success'].store.mode == 'rebuild'
    finally:
        service.close()
        store.close()


@pytest.mark.parametrize('p', [.01, .35, .3999, .4, .7, .99])
def test_factor_only_lowers_discretionary_tasks_and_note_threshold(tmp_path, monkeypatch, p):
    store = InitiativeStore(state_dir=tmp_path)
    service = runtime(tmp_path, [NOW])
    try:
        driver = service.loaded['success'].driver
        monkeypatch.setattr(driver.learner, 'predict', lambda state, frames: np.full((len(frames), 1), p))
        chosen = candidate()
        assert factor(service, 'success', chosen, NOW) == min(1., max(.5, p / .7))
        for owed in (replace(chosen, source_type='commitment'), replace(chosen, type='outreach_answer'),
                     replace(chosen, type='outreach_followup'), replace(chosen, kind='message')):
            assert factor(service, 'success', owed, NOW) == 1.
        row = task(store, NOW)
        mind = dispatch_mind(store, service)
        before = store.get(row.id)
        first, = Mind.dispatch(mind)
        assert first['body'].count('A learned forecast') == int(p < .4)
        if p < .4:
            assert f'about {round(p * 10) * 10}%' in first['body']
        assert Mind.dispatch(mind) == [first]
        after = store.get(row.id)
        assert (after.status, after.decision, after.priority) == (before.status, before.decision, before.priority)
        assert len(service.loaded['success'].comparator.records()) == 1
    finally:
        service.close()
        store.close()


@pytest.mark.parametrize('outcome', ['done', 'failed', 'blocked', 'expired', 'cancelled', 'uncertain'])
@pytest.mark.parametrize('verifier', ['owner', 'check', 'hermes_failure', 'none'])
def test_labels_require_the_exact_verifier_and_class(outcome, verifier):
    expected = (1 if outcome == 'done' and verifier in {'owner', 'check'} else
                0 if outcome in {'failed', 'blocked', 'expired'} and verifier != 'none' else None)
    assert label({'outcome': outcome, 'verified': verifier}) == expected


def test_rows_reference_sources_and_use_only_past_topic_failures(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    service = runtime(tmp_path, [NOW])
    try:
        old = task(store, NOW - 2 * HOUR, outcome='failed', verified='hermes_failure', turn='old-turn')
        current = task(store, NOW - HOUR, outcome='done', verified='check')
        task(store, NOW - HOUR / 4, outcome='failed', verified='owner', turn='future-turn')
        task(store, NOW - 3 * HOUR, outcome='done')
        rows = service.loaded['success'].driver.rows
        example = next(r for r in rows.examples() if r['ref'] == current.id)
        frame = json.loads(example['state'])
        assert frame['prior_failures'] == 1 and frame['last_failure'] == 'failed'
        assert frame['open_ended'] and frame['success_check']
        assert frame['topic'] != WORDS[2]
        assert example['teacher'] == 'verified_outcome' and example['label'] == 1
        assert {current.id, 'turn-1', old.id, 'old-turn'} <= set(rows.source_ids(current.id))
        assert 'future-turn' not in rows.source_ids(current.id)
        assert current.id in rows.refs_resting_on(['old-turn'])
        assert not any(word in example['state'] for word in WORDS)
        rows.sources.withheld = lambda ids: {'turn-1'} & set(ids)
        assert current.id not in {r['ref'] for r in rows.examples()}
    finally:
        service.close()
        store.close()


def planted(store):
    for i in range(840):
        risky = i % 60 < 20
        success = i % 20 >= 16 if risky else i % 20 != 0
        task(store, NOW - timedelta(weeks=5) + i * HOUR, kind='inspection' if risky else 'research',
             outcome='done' if success else 'failed', verified='check', topic='', turn=f'turn-{i}')


def test_nightly_pattern_beats_anchor_with_clear_interval_and_shuffled_control_does_not(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    planted(store)
    service = runtime(tmp_path, [NOW])
    try:
        result, = service.run()
        driver = service.loaded['success'].driver
        assert result['result'] == 'promoted', (result, driver.metrics)
        assert driver.last_report.passed
        assert driver.metrics['brier_gain_lower'] > 0
        assert driver.metrics['shuffled_gain_lower'] <= 0
        assert driver.learner.budget_used(service.pin('success', 'owner').state) <= 2.
        for path in service.root.rglob('*'):
            if path.is_file():
                assert all(word.encode() not in path.read_bytes() for word in WORDS), path
    finally:
        service.close()
        store.close()


def test_live_poll_labels_only_served_items_and_rolls_back_regression(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    service = runtime(tmp_path, [NOW])
    try:
        promote(service, 1.)
        driver = service.loaded['success'].driver
        task(store, NOW - HOUR, outcome='failed', verified='check')
        for _ in range(30):
            row = task(store, NOW)
            note(service, 'success', row, NOW)
            store.update(row.id, outcome='failed', verified='hermes_failure', status='failed', failed_at=NOW)
        driver.poll(service)
        driver.poll(service)
        records = service.loaded['success'].comparator.records()
        assert len(records) == 30
        assert all(r['reference_output'] == .7 for r in records)
        assert all(r['outcome']['teacher'] == 'realized_value' for r in records)
        assert service.loaded['success'].store.revisions.serving().mode == 'base'
        detail = report(service.config, 'success', runtime=service)
        assert detail['served'] == detail['outcomes'] == 30
        assert detail['outcomes_by_verifier']['hermes_failure'] == 30
        assert detail['outcomes_by_class']['failed'] == 30
        assert detail['constant_brier'] == pytest.approx(.49)
        from colonymind.reports.plasticity import collect, lines
        text = ' '.join(lines(collect(service.config, runtime=service)))
        assert 'Action success' in text and 'Brier' in text and 'log loss' in text
        assert 'Rolled back' in text
    finally:
        service.close()
        store.close()


def test_erasing_turn_unserves_and_rebuilds_like_clean_input(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    planted(store)
    service = runtime(tmp_path, [NOW])
    try:
        assert service.run()[0]['result'] == 'promoted'
        binding = service.loaded['success']
        previous = binding.store.revisions.serving()
        source = binding.driver.rows
        forgotten = source.refs_resting_on(['turn-1'])
        assert forgotten
        service.erase_sources(['turn-1'])
        assert binding.store.revisions.status(previous.id).state == 'withdrawn'
        assert not binding.store.revisions.exists(previous.id)
        assert not any(r['ref'] in forgotten for r in binding.source.rows(OBJECTIVE))
        clean = runtime(tmp_path / 'clean', [NOW])
        try:
            # Same primary rows, with the erased turn withheld before a first build.
            clean.loaded['success'].driver.rows.sources = Sources(tmp_path, owner_id='Owner')
            clean.loaded['success'].driver.rows.sources.withheld = lambda ids: {'turn-1'} & set(ids)
            # Keyed categorical vocabulary must share the instance identity.
            (clean.root / 'comparator.key').write_bytes((service.root / 'comparator.key').read_bytes())
            assert clean.run()[0]['result'] == 'promoted'
            expected, actual = clean.pin('success', 'owner').state, service.pin('success', 'owner').state
            assert expected.keys() == actual.keys()
            for key in expected:
                np.testing.assert_allclose(actual[key], expected[key], rtol=1e-6, atol=0.)
        finally:
            clean.close()
    finally:
        service.close()
        store.close()


def test_type_vocabulary_is_training_only_and_anchor_is_exact():
    learner = ActionLearner('fixture')
    frame = {'at': NOW.isoformat(), 'task_type': 'unseen', 'drive': 'curiosity', 'open_ended': True,
             'goal_mode': False, 'success_check': True, 'runtime_seconds': 600., 'retries': 1.,
             'prior_failures': 0, 'last_failure': 'none', 'lessons': 0, 'topic': ''}
    assert learner.predict(learner.base(), [frame])[0, 0] == .7


def test_future_settlement_cannot_train_or_label_live_comparison(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    service = runtime(tmp_path, [NOW])
    try:
        row = task(store, NOW)
        note(service, 'success', row, NOW)
        store.update(row.id, outcome='done', verified='check', completed_at=NOW + HOUR)
        driver = service.loaded['success'].driver
        assert row.id not in {r['ref'] for r in driver.rows.examples()}
        driver.poll(service)
        record, = service.loaded['success'].comparator.records()
        assert record['outcome'] is None
    finally:
        service.close()
        store.close()


def test_report_prior_excludes_erased_evidence_and_counts_mind_events(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    service = runtime(tmp_path, [NOW])
    try:
        earlier = task(store, NOW - 2 * HOUR, outcome='failed', verified='check', turn='erased-turn')
        row = task(store, NOW)
        note(service, 'success', row, NOW)
        store.transition(row.id, 'failed', action='outcome_failed', at=NOW,
                         outcome='failed', verified='hermes_failure', failed_at=NOW)
        store.transition(earlier.id, 'failed', action='run_failed', at=NOW)
        store.transition(earlier.id, 'failed', action='breaker_trip', at=NOW)
        service.loaded['success'].driver.poll(service)
        service.loaded['success'].driver.rows.sources.withheld = lambda ids: {'erased-turn'} & set(ids)
        detail = report(service.config, 'success', runtime=service)
        assert detail['task_type_prior_brier'] == pytest.approx(.49)
        assert detail['failed_dispatches_per_week'] == 1
        assert detail['breaker_trips_per_week'] == 1
    finally:
        service.close()
        store.close()


def test_verified_blocked_history_counts_without_a_terminal_timestamp(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    now = [NOW - HOUR]
    service = runtime(tmp_path, now)
    try:
        prior = task(store, NOW - 2 * HOUR)
        store.update(prior.id, outcome='blocked', verified='hermes_failure')
        # A block has no source timestamp: first observation must not backdate it.
        service.loaded['success'].driver.poll(service)
        now[0] = NOW
        row = task(store, NOW)
        source = service.loaded['success'].driver.rows
        frame, _ = source.frame(next(r for r in source.snapshot() if r['id'] == row.id), NOW)
        assert frame['prior_failures'] == 1 and frame['last_failure'] == 'blocked'
    finally:
        service.close()
        store.close()


def test_erasing_indirect_dispatch_source_cannot_reuse_saved_forecast(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    service = runtime(tmp_path, [NOW])
    try:
        task(store, NOW - 2 * HOUR, outcome='failed', verified='check', turn='prior-turn')
        row = task(store, NOW, turn='current-turn')
        promote(service, -1.5)
        assert note(service, 'success', row, NOW)
        service.erase_sources(['prior-turn'])
        assert service.loaded['success'].comparator.records() == []
        assert note(service, 'success', store.get(row.id), NOW) is None
        service.close()
        service = runtime(tmp_path, [NOW])
        assert note(service, 'success', store.get(row.id), NOW) is None
    finally:
        service.close()
        store.close()


def test_pending_erasure_does_not_write_an_unlogged_dispatch_marker(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    service = runtime(tmp_path, [NOW])
    try:
        row = task(store, NOW)
        binding = service.loaded['success']
        _io.write_json(binding.store.rebuild_required, {'observe': True})
        assert note(service, 'success', row, NOW) is None
        assert 'action_success' not in store.get(row.id).context
        assert binding.comparator.records() == []
    finally:
        service.close()
        store.close()


def test_vocabulary_and_standardisation_are_fitted_only_from_training_rows():
    from colonymind.plasticity.builds import TrainingRow
    learner = ActionLearner('fixture')
    frames = [{'at': (NOW + i * HOUR).isoformat(), 'task_type': f'type-{i % 14}', 'drive': 'curiosity',
               'open_ended': True, 'goal_mode': False, 'success_check': True,
               'runtime_seconds': 60. * math.exp(i % 3), 'retries': 1., 'prior_failures': 0,
               'last_failure': 'none', 'lessons': 0, 'topic': ''} for i in range(30)]
    rows = [TrainingRow(str(i), 'action_success', json.dumps(frame), i % 2, 'verified_outcome',
                        False, 'logistic', 1., 'owner') for i, frame in enumerate(frames)]
    state = learner.fit(learner.base(), rows, objective=OBJECTIVE, seed=0)
    vocabulary = json.loads(state['types'].tobytes())
    assert len(vocabulary) == 12 and 'type-0' in vocabulary and 'type-1' in vocabulary
    np.testing.assert_allclose(state['runtime_scale'], [1., np.std([0., 1., 2.])])
    before = {key: value.copy() for key, value in state.items()}
    held_out = {**frames[0], 'task_type': 'gate-only', 'runtime_seconds': 1e8}
    encoded = learner.frame(held_out, state)['inputs']
    assert encoded['type_12'] == 1. and encoded['runtime'] == 3.
    learner.predict(state, [held_out])
    for key in state:
        np.testing.assert_array_equal(state[key], before[key])


def test_insufficient_labels_keep_base_and_offline_report_matches(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    service = runtime(tmp_path, [NOW])
    try:
        row = task(store, NOW - HOUR, outcome='failed', verified='check')
        assert service.run()[0]['result'] != 'promoted'
        assert service.pin('success', 'owner').closure.mode == 'base'
        note(service, 'success', row, NOW)
        service.loaded['success'].driver.poll(service)
        detail = report(service.config, 'success', runtime=service)
        from unittest.mock import patch
        with patch('colonymind.util.temporal.now_utc', return_value=NOW):
            assert report(service.config, 'success') == detail
    finally:
        service.close()
        store.close()


def test_dispatch_marker_deduplicates_beyond_comparator_retention(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    now = [NOW]
    service = runtime(tmp_path, now)
    try:
        row = task(store, NOW)
        promote(service, -1.5)
        first = note(service, 'success', row, NOW)
        now[0] += timedelta(days=15)
        assert service.loaded['success'].comparator.records() == []
        assert note(service, 'success', store.get(row.id), now[0]) == first
        assert service.loaded['success'].comparator.records() == []
    finally:
        service.close()
        store.close()


def test_owner_verification_after_dispatch_cannot_enter_earlier_features_or_prior(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    now = [NOW]
    service = runtime(tmp_path, now)
    try:
        prior = task(store, NOW - 3 * HOUR, outcome='failed', verified='none', turn='prior')
        # A rating before the terminal result never verified that later result.
        store.transition(prior.id, 'failed', action='rated', at=NOW - 2.75 * HOUR,
                         details={'verdict': 'useful', 'by': 'owner'})
        current = task(store, NOW, turn='current')
        note(service, 'success', current, NOW)
        now[0] += HOUR
        store.transition(prior.id, 'failed', action='rated', at=now[0],
                         details={'verdict': 'wrong', 'by': 'owner'}, verified='owner')
        store.update(current.id, outcome='done', verified='check', completed_at=now[0])
        driver = service.loaded['success'].driver
        frame = json.loads(next(r for r in driver.rows.examples() if r['ref'] == current.id)['state'])
        assert frame['prior_failures'] == 0
        driver.poll(service)
        detail = report(service.config, 'success', runtime=service)
        assert detail['task_type_prior_brier'] == pytest.approx(.09)
    finally:
        service.close()
        store.close()


def test_report_keeps_the_verifier_and_class_that_labelled_the_served_item(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    now = [NOW]
    service = runtime(tmp_path, now)
    try:
        row = task(store, NOW)
        note(service, 'success', row, NOW)
        store.update(row.id, outcome='blocked', verified='hermes_failure')
        driver = service.loaded['success'].driver
        driver.poll(service)
        now[0] += HOUR
        store.update(row.id, outcome='done', verified='check', completed_at=now[0])
        driver.poll(service)
        record, = service.loaded['success'].comparator.records()
        assert record['outcome']['observed'] == 0
        detail = report(service.config, 'success', runtime=service)
        assert detail['outcomes_by_verifier'] == {'hermes_failure': 1}
        assert detail['outcomes_by_class'] == {'blocked': 1}
    finally:
        service.close()
        store.close()


def test_dispatch_provenance_survives_restart_and_erasure(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    now = [NOW]
    service = runtime(tmp_path, now)
    try:
        row = task(store, NOW)
        promote(service, -1.5)
        assert note(service, 'success', row, NOW)
        service.close()
        service = runtime(tmp_path, now)
        assert note(service, 'success', store.get(row.id), NOW)
        assert len(service.loaded['success'].comparator.records()) == 1
        service.erase_sources(['turn-1'])
        assert note(service, 'success', store.get(row.id), NOW) is None
        assert service.loaded['success'].comparator.records() == []
    finally:
        service.close()
        store.close()
