"""Sparse nights preserve evidence until each learner's existing label minimum.

The compatibility hashes cover
all persisted JSON bytes, including closures, manifests, periods, certificates,
verdicts, transitions, nightly.json and report.json. Only fixture entropy (the
comparator key and certificate UUID sequence) is fixed; no output is normalized.
Proposed closure payloads and gate reports are captured before compaction too.
"""
import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from colonymind.instinct.outcomes import Sources
from colonymind.plasticity import _io
from colonymind.plasticity.compaction import compact, discarded_rows
from colonymind.plasticity.gates import Check, GateReport, MIN_POOLED
from colonymind.plasticity.learners import (
    action_success, behaviour, contact_reply_timing, query_adapter, reply_timing, salience, service_health,
)
from colonymind.plasticity.nightly import Runtime

NOW = datetime(2026, 4, 6, 12, tzinfo=timezone.utc)


class Config:
    def __init__(self, home, specs):
        self.home, self.specs = home, specs

    def get(self, key, default=None):
        return {'plasticity.learners': self.specs}.get(key, default)


def create(path, module, name, **changes):
    path.mkdir(parents=True, exist_ok=True)
    (path / 'plasticity').mkdir(exist_ok=True)
    (path / 'plasticity' / 'comparator.key').write_bytes(b'f1 frozen key'.ljust(32, b'!'))
    spec = {**copy.deepcopy(module.DECLARATION), **changes}
    service = Runtime(Config(path, {name: spec}),
        sources={name: SimpleNamespace(sources=Sources(path, owner_id='owner'))}, clock=lambda: NOW)
    assert service.rejected == {}
    return service


def snapshot(service):
    return hashlib.sha256(_io.canonical({str(p.relative_to(service.root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in service.root.rglob('*.json')})).hexdigest()


BASE_BYTES = {
    'health-4': (
        '00ed07e4a51bdf9274e464b2c256fc4b4e8832547511b54941f39c7aeec544d1',
        '5d72b8ef148fd23a0614fe73c7365f1300e33b4368b2e5a9aee4af908be9a711',
        'd7ae10a27d574c9a1ead9be1bf9fd9c6a1c70f6f85a17e544ded4c4a2e988906',
        'ae0798f4c30c75153e7e6e9f18f2e2f7c0becd80aca137e98a282b0d2edefe0a',
    ),
    'mind-1': (
        'bfc3bbde57021481277b2cedb86d19ccda2d7719959764d6df588e9ac2a735bc',
        '7d5827973efe32fbb5e27966a08e71418be4997e5d70de9078893ace8e0bf9a9',
        'ee07fda778ce312d436c97cd72a50362333a6e25c07d0f8103c3d7efb1d9a578',
        'd1bc5c01db818bfe88b8e106cc408d7d5a263b0312cb01e6b03be8e9de143370',
    ),
}


@pytest.mark.parametrize('module,name', [(service_health, 'health-4'), (reply_timing, 'mind-1')])
def test_health4_and_mind1_match_recorded_base_bytes(tmp_path, monkeypatch, module, name):
    service = create(tmp_path, module, name)
    driver = service.loaded[name].driver
    assert 'minimum_labels' not in type(driver).__dict__
    assert service._minimum_labels(service.loaded[name]) == {}
    examples = []
    proposals, reports = [], []
    original_propose, original_gate = driver.controller.propose, driver.gate

    def propose(closure, state):
        proposals.append({'closure': closure.to_dict(), 'state': _io.state_digest(state)})
        return original_propose(closure, state)

    def gate(closure, build):
        report = original_gate(closure, build)
        reports.append(report.to_dict())
        return report

    monkeypatch.setattr(driver.controller, 'propose', propose)
    monkeypatch.setattr(driver, 'gate', gate)
    monkeypatch.setattr('colonymind.plasticity.certificates.uuid4',
                        lambda sequence=iter(range(20)): SimpleNamespace(hex=f'{next(sequence):032x}'))
    if module is reply_timing:
        monkeypatch.setattr(driver.rows, 'examples', lambda: list(examples))
    try:
        for night, size in enumerate((0, 4, 1200, 0)):
            if module is reply_timing:
                for i in range(size):
                    at = NOW - timedelta(hours=1300-i)
                    state = json.dumps({'at': at.isoformat(), 'channel': 'chat', 'hours': 4., 'outreach': False})
                    examples.append({'ref': f'fixture-{len(examples)}', 'point': driver.objective.points[0],
                        'version': 'fixture-1', 'state': state, 'state_hash': _io.digest(state),
                        'label': int(np.sin(at.hour * 2*np.pi/24) > 0), 'teacher': 'realized_value',
                        'push_away': False, 'source_kind': 'system', 'audience': 'owner', 'at': at.isoformat()})
            elif size:
                rows = []
                offset = len(driver.samples._read())
                for i in range(size):
                    at = NOW - timedelta(seconds=300*(1400-i-offset))
                    frame = {'at': at.isoformat(), 'streams': {'value': [float(np.sin((i-lag)*2*np.pi/288))
                             for lag in range(1, 13)]}, 'series': 'router.primary.latency_p95_ms'}
                    rows.append((frame, float(np.sin(i*2*np.pi/288))))
                driver.samples.add_many(rows)
            verdicts = service.run()
            assert verdicts[0]['result'] != 'error'
            actual = hashlib.sha256(_io.canonical({'files': snapshot(service), 'proposals': proposals,
                                                   'gates': reports, 'verdicts': verdicts})).hexdigest()
            assert actual == BASE_BYTES[name][night], (name, night)
    finally:
        service.close()


def rows_for(module, count, *, gate, prefix):
    """Numeric-only fixture rows; gate membership uses the real learner's split."""
    rows = []
    at = NOW - timedelta(days=70)
    while len(rows) < count:
        at += timedelta(hours=1)
        if module.gate_week(at) != gate:
            continue
        if module is action_success:
            frame = {'at': at.isoformat(), 'task_type': 'fixture', 'drive': 'curiosity',
                     'runtime_seconds': 600, 'last_failure': 'none', 'open_ended': False,
                     'goal_mode': False, 'success_check': True, 'retries': 0, 'prior_failures': 0, 'lessons': 0}
        elif module is salience:
            frame = {'at': at.isoformat(), 'type': '0'*64, 'verdict': 'useful'}
        else:
            frame = {'at': at.isoformat(), 'channel': 'chat', 'hours': 4., 'reply_share': .5,
                     'sends': 1, 'horizon_hours': 1.}
        state = json.dumps(frame)
        teacher = next(iter(module.OBJECTIVE.losses))
        rows.append({'ref': f'{prefix}-{len(rows)}', 'point': module.OBJECTIVE.points[0],
            'version': 'fixture-1', 'state': state, 'state_hash': _io.digest(state), 'label': len(rows) % 2,
            'teacher': teacher, 'push_away': False, 'source_kind': 'resolver', 'audience': 'owner',
            'at': at.isoformat()})
    return rows


def offer(monkeypatch, driver, rows, *, dependencies=None):
    def examples():
        if dependencies is not None:
            driver.rows.dependencies.update(dependencies)
        return list(rows)
    monkeypatch.setattr(driver.rows, 'examples', examples)
    if isinstance(driver, action_success.Driver):
        monkeypatch.setattr(driver.rows, 'observe', lambda: None)


@pytest.mark.parametrize('module', [action_success, contact_reply_timing, salience])
def test_sparse_nights_accumulate_then_propose_once_and_report_counts(tmp_path, monkeypatch, module):
    service = create(tmp_path, module, 'sparse')
    binding = service.loaded['sparse']
    driver = binding.driver
    training_need, gate_need = driver.minimum_labels['training'], driver.minimum_labels['gate']
    training = rows_for(module, training_need, gate=False, prefix='training')
    gates = rows_for(module, gate_need, gate=True, prefix='gate')
    offered = []
    offer(monkeypatch, driver, offered)
    propose = Mock(wraps=driver.propose)
    gate = Mock(side_effect=lambda closure, build: GateReport(closure.id, 'fixture', (Check('retention', True),)))
    monkeypatch.setattr(driver, 'propose', propose)
    monkeypatch.setattr(driver, 'gate', gate)
    serving = binding.store.revisions.serving().id
    try:
        for have_training, have_gate in ((0, 0), (training_need//3, gate_need//3),
                (training_need-1, gate_need-1), (training_need-1, gate_need)):
            offered[:] = training[:have_training] + gates[:have_gate]
            result, = service.run()
            expected = {'training': {'have': have_training, 'need': training_need},
                        'gate': {'have': have_gate, 'need': gate_need}}
            assert result == {'learner': 'sparse', 'closure': serving, 'result': 'unchanged',
                              'why': ['insufficient_labels'], 'counts': expected}
            assert driver.idle_reason == 'insufficient_labels'
            assert binding.last_build.rows == have_training
            assert propose.call_count == gate.call_count == 0
            assert binding.store.revisions.serving().id == serving
            assert all(period.state == ('closed' if any(m.entries for m in binding.store.manifests_of([period]))
                                        else 'discarded') for period in binding.store.periods.periods())
            assert discarded_rows(binding.store) == set()
            assert _io.read_json(service.root / 'nightly.json')['learners'] == [result]
            report_row, = _io.read_json(service.root / 'report.json')['learners']
            assert {key: report_row[key] for key in ('result', 'why', 'counts')} == {
                key: result[key] for key in ('result', 'why', 'counts')}
            # Only newly observed rows enter later manifests, while builds read the earlier ones too.
            assert sum(len(manifest.admitted()) for manifest in binding.store.current_manifests()) == have_training
        offered[:] = training + gates
        result, = service.run()
        assert result['result'] == 'promoted'
        assert driver.idle_reason is None
        assert propose.call_count == gate.call_count == 1
        assert binding.last_build.rows == training_need
        assert len(binding.store.periods.retained()) == 4
        assert discarded_rows(binding.store) == set()
        assert 'counts' not in _io.read_json(service.root / 'report.json')['learners'][0]
    finally:
        service.close()


@pytest.mark.parametrize('mode,accumulation', [('rebuild', True), ('rebuild', False), ('stack', True)])
def test_empty_idle_periods_are_discarded_without_burning_rows(tmp_path, monkeypatch, mode, accumulation):
    from test_plasticity_withdrawn_rederive import runtime
    from colonymind.plasticity.synthetic import SyntheticSource
    service = runtime(tmp_path, SyntheticSource([]), mode=mode, accumulation=accumulation)
    driver = service.loaded['probe'].driver
    monkeypatch.setattr(driver, 'minimum_labels', {'training': contact_reply_timing.TRAINING_MIN_ROWS}, raising=False)
    monkeypatch.setattr(driver, 'label_counts', lambda build: {'training': build.rows}, raising=False)
    try:
        for _ in range(3):
            assert service.run()[0]['why'] == ['insufficient_labels']
        assert [period.state for period in driver.store.periods.periods()] == ['discarded']*3
        assert driver.store.periods.retained() == []
        assert driver.store.manifests.ids() == []
        assert discarded_rows(driver.store) == set()
    finally:
        service.close()


@pytest.mark.parametrize('empty', [False, True])
def test_idle_period_survives_a_chain_siblings_error_unless_empty(tmp_path, monkeypatch, empty):
    from test_plasticity_withdrawn_rederive import chained
    from colonymind.plasticity.synthetic import SyntheticSource, make_rows
    service = chained(tmp_path, {'first': SyntheticSource([] if empty else make_rows(1)),
                                'second': SyntheticSource([])})
    driver = service.loaded['first'].driver
    monkeypatch.setattr(driver, 'minimum_labels', {'training': contact_reply_timing.TRAINING_MIN_ROWS}, raising=False)
    monkeypatch.setattr(driver, 'label_counts', lambda build: {'training': build.rows}, raising=False)
    monkeypatch.setattr(service.loaded['second'].driver, 'build', Mock(side_effect=ValueError('fixture')))
    try:
        results = service.run()
        assert results[0]['why'] == ['insufficient_labels']
        assert results[1]['result'] == 'error'
        assert driver.store.periods.periods()[0].state == ('discarded' if empty else 'closed')
        assert discarded_rows(driver.store) == set()
    finally:
        service.close()


def test_idle_rows_obey_erasure_and_existing_pruned_row_exclusions_after_restart(tmp_path, monkeypatch):
    module = contact_reply_timing
    service = create(tmp_path, module, 'sparse')
    binding = service.loaded['sparse']
    offered = rows_for(module, 1, gate=False, prefix='pruned')
    offer(monkeypatch, binding.driver, offered)
    # A preexisting, genuinely discarded period remains excluded as before.
    binding.driver.period()
    binding.store.periods.discard(binding.driver.last_period, reason='fixture prior gate failure')
    compact(binding.store)
    excluded = discarded_rows(binding.store)
    assert excluded == {_io.digest([offered[0]['ref'], offered[0]['teacher']])}
    erased = rows_for(module, 1, gate=False, prefix='erased')
    offered.extend(erased)
    offer(monkeypatch, binding.driver, offered, dependencies={row['ref']: ('erased-source',) for row in erased})
    try:
        result, = service.run()
        assert result['counts']['training']['have'] == 1
        assert service.erase_sources(['erased-source'])['clean']
        assert service.run()[0]['counts']['training']['have'] == 0
    finally:
        service.close()
    # Reoffer both the old ref and a new ref resting on the erased source, even if the source cache retained them.
    offered.extend(rows_for(module, 1, gate=False, prefix='new-ref-from-erased-source'))
    service = create(tmp_path, module, 'sparse')
    binding = service.loaded['sparse']
    offer(monkeypatch, binding.driver, offered,
          dependencies={row['ref']: ('erased-source',) for row in offered if row['ref'] != 'pruned-0'})
    try:
        result, = service.run()
        assert result['counts']['training']['have'] == 0
        assert list(binding.source.rows(binding.driver.objective)) == []
        assert discarded_rows(binding.store) == excluded
        assert all(period.state == 'discarded' for period in binding.store.periods.periods()[2:])
    finally:
        service.close()


def test_query_adapter_without_capture_accumulates_training_and_never_proposes(tmp_path, monkeypatch):
    from test_plasticity_query_adapter_driver import Config as RecallConfig, training
    service = Runtime(RecallConfig(tmp_path), clock=lambda: NOW)
    binding = service.loaded['recall-test']
    driver = binding.driver
    driver.attach(service)
    propose, gate = Mock(wraps=driver.propose), Mock(wraps=driver.gate)
    monkeypatch.setattr(driver, 'propose', propose)
    monkeypatch.setattr(driver, 'gate', gate)
    try:
        for count in range(3):
            if count:
                training(driver, turn=f'training-{count}', source=f'source-{count}')
            result, = service.run()
            assert result['why'] == ['no_gate_suite'] and result['capture_present'] is False
            assert result['counts'] == {'training': {'have': count, 'need': 0}}
            assert all(type(value) is int for pair in result['counts'].values() for value in pair.values())
            assert discarded_rows(binding.store) == set()
            assert binding.store.revisions.serving().mode == 'base'
            assert propose.call_count == gate.call_count == 0
            assert not list((driver.store.directory / 'cutoff-checks').glob('*.json'))
        assert len(binding.store.periods.retained()) == 2
    finally:
        service.close()
    service = Runtime(RecallConfig(tmp_path), clock=lambda: NOW)
    try:
        row, = service.report()['learners']
        assert row['why'] == result['why'] and row['capture_present'] is False
        assert row['counts'] == result['counts']
    finally:
        service.close()


def test_query_adapter_waits_for_its_declared_suite_size_without_a_training_floor(tmp_path, monkeypatch):
    from test_plasticity_query_adapter_driver import Config as RecallConfig, training, qualification
    service = Runtime(RecallConfig(tmp_path), clock=lambda: NOW)
    binding = service.loaded['recall-test']
    driver = binding.driver
    driver.attach(service)
    training(driver)
    try:
        result, = service.run()
        assert result['why'] == ['no_gate_suite']
        assert result['capture_present'] is False
        assert result['counts'] == {'training': {'have': 1, 'need': 0}}
        assert binding.last_build.rows == 1  # No invented training-row minimum.
        assert discarded_rows(binding.store) == set()
        for report in ('nightly.json', 'report.json'):
            row, = _io.read_json(service.root / report)['learners']
            assert row['why'] == result['why'] and row['counts'] == result['counts']
            assert row['capture_present'] is False
        propose = Mock(wraps=driver.propose)
        monkeypatch.setattr(driver, 'propose', propose)
        capture = qualification(driver, binding.last_build)
        for key in ('ids', 'base', 'reference', 'candidate', 'control'):
            capture['gate'][key] = capture['gate'][key][:MIN_POOLED-1]
        capture_path = driver.store.directory / 'qualification' / (binding.last_build.digest + '.json')
        _io.write_json(capture_path, capture)
        result, = service.run()
        assert result['counts'] == {'gate': {'have': MIN_POOLED-1, 'need': MIN_POOLED}}
        assert result['why'] == ['insufficient_labels'] and result['capture_present'] is True
        assert propose.call_count == 0
        assert binding.last_build.rows == 1 and discarded_rows(binding.store) == set()
        assert not list((driver.store.directory / 'cutoff-checks').glob('*.json'))
        capture = qualification(driver, binding.last_build)
        capture['gate']['ids'] = ['repeated-gate-id']*MIN_POOLED
        _io.write_json(capture_path, capture)
        result, = service.run()
        assert result['counts'] == {'gate': {'have': 1, 'need': MIN_POOLED}}
        assert propose.call_count == 0
        qualification(driver, binding.last_build)
        result, = service.run()
        assert result['result'] == 'promoted'
        assert propose.call_count == 1
        assert binding.last_build.rows == 1
        assert discarded_rows(binding.store) == set()
    finally:
        service.close()


def test_no_minimum_preserves_the_existing_behaviour_idle_reason(tmp_path):
    service = create(tmp_path, behaviour, 'no-minimum')
    try:
        assert service.loaded['no-minimum'].driver.minimum_labels == {}
        for _ in range(3):
            result, = service.run()
            assert result == {'learner': 'no-minimum', 'closure': service.loaded['no-minimum'].store.revisions.base.id,
                              'result': 'unchanged', 'why': ['awaiting_lesson']}
            assert 'counts' not in _io.read_json(service.root / 'report.json')['learners'][0]
            assert snapshot(service) == 'd068f70f8d507a1c44402a4d4f31fe4ae291f4feeaa288471dd6d44dc9f7fb12'
    finally:
        service.close()


def test_idle_counts_survive_reload(tmp_path, monkeypatch):
    service = create(tmp_path, contact_reply_timing, 'sparse')
    offer(monkeypatch, service.loaded['sparse'].driver,
          rows_for(contact_reply_timing, 1, gate=False, prefix='retained'))
    result, = service.run()
    service.close()
    service = create(tmp_path, contact_reply_timing, 'sparse')
    try:
        row, = _io.read_json(service.root / 'report.json')['learners']
        assert row['result'] == 'unchanged' and row['why'] == ['insufficient_labels']
        assert row['counts'] == result['counts']
        assert len(service.loaded['sparse'].store.periods.retained()) == 1
    finally:
        service.close()


@pytest.mark.parametrize('restart_before_erasure', [False, True])
def test_erasure_proposal_clears_persisted_idle_report_across_restarts(tmp_path, monkeypatch, restart_before_erasure):
    module = contact_reply_timing
    service = create(tmp_path, module, 'sparse')
    driver = service.loaded['sparse'].driver
    training = rows_for(module, driver.minimum_labels['training'] + 1, gate=False, prefix='training')
    gates = rows_for(module, driver.minimum_labels['gate'], gate=True, prefix='gate')
    offered = training + gates[:-1]
    offer(monkeypatch, driver, offered)
    try:
        result, = service.run()
        assert result['why'] == ['insufficient_labels']
        if restart_before_erasure:
            service.close()
            service = create(tmp_path, module, 'sparse')
            driver = service.loaded['sparse'].driver
        offered.append(gates[-1])
        offer(monkeypatch, driver, offered)
        propose = Mock(wraps=driver.propose)
        monkeypatch.setattr(driver, 'propose', propose)
        monkeypatch.setattr(driver, 'gate', lambda closure, build:
                            GateReport(closure.id, 'fixture', (Check('retention', True),)))
        assert service.erase([training[-1]['ref']])['clean']
        assert propose.call_count == 1
        assert service.loaded['sparse'].store.revisions.serving().mode != 'base'
        # Erasure rebuilds intentionally leave the last nightly verdict intact.
        assert _io.read_json(service.root / 'nightly.json')['learners'] == [result]
        for row in (service.report()['learners'][0], _io.read_json(service.root / 'report.json')['learners'][0]):
            assert not {'why', 'counts', 'result', 'capture_present'}.intersection(row)
    finally:
        service.close()
    service = create(tmp_path, module, 'sparse')
    try:
        assert not {'why', 'counts', 'result', 'capture_present'}.intersection(service.report()['learners'][0])
    finally:
        service.close()


@pytest.mark.parametrize('mode,accumulation', [('rebuild', True), ('rebuild', False), ('stack', True)])
@pytest.mark.parametrize('verdict', ['admitted', 'quarantine', 'refused'])
def test_nonempty_idle_period_keeps_every_manifest_entry(tmp_path, monkeypatch, mode, accumulation, verdict):
    from dataclasses import replace
    from test_plasticity_withdrawn_rederive import runtime
    from colonymind.plasticity.synthetic import SyntheticSource, make_rows
    from colonymind.plasticity.admission import Admitted
    service = runtime(tmp_path, SyntheticSource(make_rows(1)), mode=mode, accumulation=accumulation)
    binding = service.loaded['probe']
    driver = binding.driver
    monkeypatch.setattr(driver, 'minimum_labels', {'training': contact_reply_timing.TRAINING_MIN_ROWS})
    monkeypatch.setattr(driver, 'label_counts', lambda build: {'training': build.rows})
    admit = driver.admission.admit

    def decide(*args, **kwargs):
        admitted = admit(*args, **kwargs)
        manifest = replace(admitted.manifest,
                           entries=tuple(replace(entry, verdict=verdict) for entry in admitted.manifest.entries))
        return Admitted(manifest, admitted.rows if verdict == 'admitted' else ())

    monkeypatch.setattr(driver.admission, 'admit', decide)
    try:
        result, = service.run()
        assert result['why'] == ['insufficient_labels']
        period, = binding.store.periods.retained()
        manifest, = binding.store.manifests_of([period])
        assert [entry.verdict for entry in manifest.entries] == [verdict]
        assert discarded_rows(binding.store) == set()
        assert not (binding.store.directory / 'discarded-rows.json').exists()
    finally:
        service.close()


def test_salience_proxies_do_not_count_as_owner_gate_labels(tmp_path, monkeypatch):
    service = create(tmp_path, salience, 'sparse')
    binding = service.loaded['sparse']
    minimum = binding.driver.minimum_labels
    proxies = rows_for(salience, minimum['gate'], gate=True, prefix='proxy')
    for row in proxies:
        row.update(teacher='behavioural_proxy', push_away=True, source_kind='behavioural')
    training = rows_for(salience, minimum['training'], gate=False, prefix='owner-training')
    offer(monkeypatch, binding.driver, training + proxies)
    try:
        result, = service.run()
        assert result['counts'] == {'training': {'have': minimum['training'], 'need': minimum['training']},
                                    'gate': {'have': 0, 'need': minimum['gate']}}
        assert result['why'] == ['insufficient_labels']
        assert discarded_rows(binding.store) == set()
    finally:
        service.close()


def test_idle_does_not_mask_a_siblings_own_gate_failure(tmp_path, monkeypatch):
    from test_plasticity_withdrawn_rederive import chained, failing
    from colonymind.plasticity.synthetic import SyntheticSource, make_rows
    service = chained(tmp_path, {'first': SyntheticSource([]), 'second': SyntheticSource(make_rows(60))})
    first, second = service.loaded['first'], service.loaded['second']
    monkeypatch.setattr(first.driver, 'minimum_labels', {'training': contact_reply_timing.TRAINING_MIN_ROWS})
    monkeypatch.setattr(first.driver, 'label_counts', lambda build: {'training': build.rows})
    monkeypatch.setattr(second.driver, 'gate', failing)
    try:
        result = {row['learner']: row for row in service.run()}
        assert result['first']['why'] == ['insufficient_labels']
        assert first.store.periods.periods()[0].state == 'discarded'
        assert discarded_rows(first.store) == set()
        # The sibling's own failed gate is still judged, as before the sparse-idle change.
        assert result['second']['result'] == 'dropped'
        assert result['second']['why'] == ['calibration']
        assert second.store.periods.periods()[0].state == 'discarded'
    finally:
        service.close()
