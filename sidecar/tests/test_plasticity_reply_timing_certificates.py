"""Non-inferiority boundaries and durable, independently checkable certificates."""
import copy
import json
import math
from datetime import timedelta

import numpy as np
import pytest
import yaml

from colonymind.plasticity import _io
from colonymind.plasticity.builds import Build
from colonymind.plasticity.certificates import GateRecord, GateRecords
from colonymind.plasticity.compaction import compact
from colonymind.plasticity.gates import Check, GateReport, Z95
from colonymind.plasticity.learners.reply_timing import gate_week
from test_plasticity_reply_timing import H, NOW, runtime


@pytest.fixture
def gate_case(tmp_path, monkeypatch):
    service = runtime(tmp_path, [NOW])
    driver = service.loaded['timing'].driver
    reserved = NOW
    while not gate_week(reserved):
        reserved -= timedelta(days=7)

    def setup(candidate, serving, control=.25, count=30, training=40):
        losses = [np.broadcast_to(value, (count,)).copy() for value in (candidate, serving, control)]
        rows = [{'ref': f'gate-{i}', 'state': json.dumps({'at': (reserved + i * H).isoformat(),
                 'channel': 'a', 'hours': 4., 'outreach': 0.}), 'at': (reserved + i * H).isoformat(),
                 'label': 0.} for i in range(count)]
        monkeypatch.setattr(driver.source, 'rows', lambda objective: iter(rows))

        def state(values):
            return {**driver.learner.base(), 'probabilities': np.sqrt(values)}

        def build(values):
            fitted = state(values)
            return Build('rebuild', fitted, _io.state_digest(fitted), frozenset(), (), 'owner', training)

        incumbent = driver.propose(build(losses[1]))
        driver.controller.consider(incumbent.id, GateReport(incumbent.id, 'fixture', (Check('budget', True),)))

        def predict(fitted, states):
            # Preserve missing-input refusal for the real canary check.
            for frame in states:
                json.loads(frame)['channel']
            return np.asarray(fitted['probabilities']).reshape(-1, 1)

        monkeypatch.setattr(driver.learner, 'predict', predict)
        monkeypatch.setattr(driver.learner, 'fit', lambda *args, **kwargs: state(losses[2]))
        candidate_build = build(losses[0])
        closure = driver.propose(candidate_build)
        return driver, closure, candidate_build, losses

    yield setup
    service.close()


def lower(values):
    return float(np.mean(values) - Z95 * np.std(values, ddof=1) / math.sqrt(len(values)))


@pytest.mark.parametrize('candidate,serving', [
    (.04, .04), (.04, .1), (.04, .04 - 5e-13), (.04, .04 - 2e-12),
    (.04, np.tile([0., .09], 15)), (.25, .3), (.3, .4),
])
def test_default_replays_legacy_retention_predicate(gate_case, candidate, serving):
    driver, closure, build, losses = gate_case(candidate, serving)
    report = driver.gate(closure, build)
    candidate_loss, serving_loss, _ = [np.sqrt(loss) ** 2 for loss in losses]
    legacy = lower(.25 - candidate_loss) > 0 and lower(serving_loss - candidate_loss) >= -1e-12
    assert next(check.passed for check in report.checks if check.name == 'retention') == legacy
    assert driver.retention_margin == 1e-12


def test_margin_admits_exactly_the_lower_bound_and_preserves_strict_checks(gate_case):
    driver, closure, build, losses = gate_case(.04, np.tile([0., .09], 15))
    lb = lower(np.sqrt(losses[1]) ** 2 - np.sqrt(losses[0]) ** 2)
    assert lb < 0 < (losses[1] - losses[0]).mean()
    for margin, expected in [(0., False), (np.nextafter(-lb, 0.), False), (-lb, True), (-lb + .01, True)]:
        driver.retention_margin = float(margin)
        report = driver.gate(closure, build)
        assert report.passed == expected
        assert [check.name for check in report.checks if not check.passed] == ([] if expected else ['retention'])
    assert len(driver.store.certificates.records()) == 4


@pytest.mark.parametrize('candidate,control,failed', [(.25, .4, 'retention'), (.04, .04, 'specificity')])
def test_positive_margin_does_not_relax_anchor_or_specificity(gate_case, candidate, control, failed):
    driver, closure, build, _ = gate_case(candidate, 0., control)
    driver.retention_margin = 1.
    report = driver.gate(closure, build)
    assert failed in report.failed


@pytest.mark.parametrize('margin', [-1., float('inf'), -float('inf'), float('nan'), True, False,
                                    '0.01', None, [], {}, 10 ** 400])
def test_invalid_margin_refused_when_learner_loads(tmp_path, margin):
    service = runtime(tmp_path, [NOW], retention_margin=margin)
    try:
        assert 'timing' not in service.loaded
        assert 'retention_margin must be non-negative and finite' in service.rejected['timing']
    finally:
        service.close()


def test_omitted_default_and_configured_margin_reload_without_new_identity(tmp_path):
    service = runtime(tmp_path, [NOW])
    service.close()
    path = tmp_path / 'colonymind.yaml'
    spec = yaml.safe_load(path.read_text())
    del spec['plasticity']['learners']['timing']['retention_margin']
    path.write_text(yaml.safe_dump(spec))
    from colonymind.config import load_config
    from colonymind.plasticity.nightly import Runtime
    from colonymind.instinct.outcomes import Sources
    from types import SimpleNamespace

    for margin in (None, 0., .012):
        if margin is not None:
            spec['plasticity']['learners']['timing']['retention_margin'] = margin
            path.write_text(yaml.safe_dump(spec))
        service = Runtime(load_config(tmp_path, environ={}), clock=lambda: NOW,
                          sources={'timing': SimpleNamespace(sources=Sources(tmp_path, owner_id='Owner'))})
        try:
            assert service.loaded['timing'].driver.retention_margin == (1e-12 if margin is None else margin)
        finally:
            service.close()


def test_certificate_contains_actual_statistics_thresholds_and_survives_restart(gate_case, tmp_path):
    driver, closure, build, losses = gate_case(np.linspace(.01, .09, 30), .12)
    report = driver.gate(closure, build)
    record, = driver.store.certificates.records()
    assert record['schema'] == 'gate-certificate-1'
    assert record['learner'] == 'timing' and record['revision'] == closure.id
    assert record['serving_revision'] == driver.store.revisions.serving().id
    assert record['state'] == build.digest and record['gate_set'] == report.suite
    assert record['at'] == _io.stamp(NOW)
    assert record['loss'] == 'brier'
    actual = [np.sqrt(loss) ** 2 for loss in losses]
    for arm, loss in zip(('candidate', 'serving', 'shuffled'), actual):
        assert record['arms'][arm] == pytest.approx(loss.mean())
    assert record['arms']['anchor'] == .25
    for name, delta in [('anchor_candidate', .25 - actual[0]),
                        ('serving_candidate', actual[1] - actual[0]),
                        ('shuffled_candidate', actual[2] - actual[0]),
                        ('anchor_shuffled', .25 - actual[2])]:
        assert {key: record['comparisons'][name][key] for key in ('n', 'mean', 'sd', 'se', 'lower')} == pytest.approx({
            'n': 30, 'mean': delta.mean(), 'sd': delta.std(ddof=1),
            'se': delta.std(ddof=1) / math.sqrt(30), 'lower': lower(delta)})
    assert record['comparisons']['anchor_shuffled']['adjusted_lower'] == driver.metrics['shuffled_gain_lower']
    assert record['counts'] == {'gate': 30, 'training': 40, 'control_training': 0, 'shuffles': 8}
    assert record['thresholds'] == {
        'gate_min_rows': 30, 'training_min_rows': 1, 'z': Z95, 'sd_ddof': 1,
        'retention_margin': 1e-12, 'anchor_gain_lower': {'gt': 0.},
        'serving_gain_lower': {'ge': -1e-12}, 'specificity_gain_lower': {'gt': 0.},
        'canary_exposure': 'equal_to_base', 'budget_type': 'anchored_output', 'budget_bound': 2.,
        'finite': 'all_state_values', 'parity_rtol': 1e-6, 'parity_atol': 1e-8,
        'parity_equal_nan': False, 'shuffle_seeds': list(range(7, 15))}
    assert record['parity'] == {'status': 'measured', 'passed': True}
    assert record['outcome'] == 'passed' and record['reason'] == '' and record['error'] is None
    assert record['checks'] == report.to_dict()['checks']
    compact(driver.store)
    directory = driver.store.certificates.directory
    reader = GateRecords(directory)
    assert reader.records() == [record]
    assert reader.records(since=NOW, until=NOW + H) == [record]
    assert reader.records(since=NOW + H) == [] and reader.records(until=NOW) == []
    service = runtime(tmp_path, [NOW])
    try:
        assert service.loaded['timing'].store.certificates.records() == [record]
    finally:
        service.close()


def test_insufficient_rows_record_not_tested_statistics(gate_case):
    driver, closure, build, _ = gate_case(.04, .1, count=29)
    report = driver.gate(closure, build)
    record, = driver.store.certificates.records()
    assert {'retention', 'specificity', 'served_offline'} <= set(report.failed)
    assert record['counts']['gate'] == 29
    assert all(value is None for value in record['arms'].values())
    assert all(value is None for value in record['comparisons'].values())
    assert record['parity'] == {'status': 'not-tested', 'passed': None}
    assert record['outcome'] == 'failed'


@pytest.mark.parametrize('error', [False, True])
def test_parity_failure_or_error_is_recorded_and_error_propagates(gate_case, monkeypatch, error):
    driver, closure, build, _ = gate_case(.04, .1)
    original = driver.store.revisions.state

    def pinned(candidate):
        if candidate.id != closure.id:
            return original(candidate)
        if error:
            raise ValueError('fixture parity failure')
        state = copy.deepcopy(original(candidate))
        state['probabilities'] += .01
        return state

    monkeypatch.setattr(driver.store.revisions, 'state', pinned)
    if error:
        with pytest.raises(ValueError, match='fixture parity failure'):
            driver.gate(closure, build)
    else:
        assert driver.gate(closure, build).failed == ('served_offline',)
    record, = driver.store.certificates.records()
    assert record['parity'] == {'status': 'error' if error else 'measured', 'passed': None if error else False}
    assert record['outcome'] == ('error' if error else 'failed')
    assert record['error'] == ('ValueError' if error else None)
    assert record['arms']['candidate'] == pytest.approx(.04)


def test_generic_record_has_unique_runs_and_atomic_persistence(tmp_path, monkeypatch):
    records = GateRecords(tmp_path / 'certificates')
    record = GateRecord('generic', 'revision', 'state', 'serving', _io.stamp(NOW))
    records.write(record)
    original = records.records()

    def failed_replace(*args):
        raise OSError('fixture interrupted write')

    with monkeypatch.context() as patch:
        patch.setattr(_io, '_replace', failed_replace)
        with pytest.raises(OSError, match='interrupted write'):
            records.write(record)
    assert GateRecords(records.directory).records() == original
    records.write(record)
    assert len({item['id'] for item in records.records()}) == 2
