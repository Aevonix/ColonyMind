"""Salience keeps the ranking closure pair through asynchronous formation.

Numeric fixtures and the planted bias are builder assumptions.
"""
import copy
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import yaml

from colonymind.config import load_config
from colonymind.initiatives.store import InitiativeStore
from colonymind.instinct.outcomes import Sources
from colonymind.mind.rank import Candidate, score
from colonymind.plasticity import _io
from colonymind.plasticity.gates import Check, GateReport
from colonymind.plasticity.nightly import Runtime

NOW = datetime(2026, 4, 6, 12, tzinfo=timezone.utc)


def setup_runtime(path):
    from colonymind.plasticity.learners.salience import DECLARATION
    path.mkdir(exist_ok=True)
    (path / 'colonymind.yaml').write_text(yaml.safe_dump({
        'plasticity': {'learners': {'salience-test': copy.deepcopy(DECLARATION)}}}))
    sources = Sources(path, owner_id='Owner')
    service = Runtime(load_config(path, environ={}),
                      sources={'salience-test': SimpleNamespace(sources=sources)}, clock=lambda: NOW)
    assert not service.rejected
    return service


def promote(binding, bias):
    from colonymind.plasticity.learners.salience import OBJECTIVE
    state = binding.driver.learner.base()
    state['W'][0, -1] = bias
    closure = binding.store.revisions.compose(state=_io.state_digest(state), objective=OBJECTIVE.identity,
        calibration=('fixture',), dependents={}, parent=binding.store.revisions.serving().id,
        manifests=(), periods=(), audience='owner', mode='rebuild')
    binding.controller.propose(closure, state)
    binding.controller.consider(closure.id, GateReport(closure.id, 'fixture', (Check('budget', True),)))
    return closure


def candidate(i=0):
    return Candidate(type='research', drive='curiosity', kind='task', title='A private fixture sentence.',
                     dedup_key=f'fixture:{i}', salience=.9, cost=.1)


def test_observer_serves_the_pair_that_ranked_before_promotion(tmp_path):
    from colonymind.plasticity.learners.salience import factor, observe
    service = setup_runtime(tmp_path)
    store = InitiativeStore(state_dir=tmp_path)
    try:
        binding = service.loaded['salience-test']
        old = promote(binding, .4)
        item = candidate()
        context = {'drives': {'curiosity': 1.}, 'base': {'curiosity': 1.}, 'threshold': .6,
                   'affect': None, 'last_owner_turn': None}
        value = factor(service, 'salience-test', item, NOW, context=context)
        promote(binding, -.4)
        row, _ = store.create_intention(kind='task', type=item.type, title=item.title, drive=item.drive,
            cls='internal', decision='act', decision_reason='fixture', status='approved', dedup_key='fixture:0',
            context={}, created_at=NOW)
        ranking = SimpleNamespace(eligible=((item, score(item) * value),), context=context,
            acted=((item, row.id),), rescore=lambda c, overrides: score(c) * overrides['salience'])
        observe(service, 'salience-test', ranking, NOW)
        records = binding.comparator.records()
        assert len(records) == 1
        assert records[0]['closure'] == old.id
        assert records[0]['reference'] == binding.store.revisions.base.id
        assert records[0]['output'] / .5 == pytest.approx(value)
        assert records[0]['reference_output'] == .5
        assert store.get(row.id).context['salience']['frame']
        assert row.id in binding.driver.rows.source_ids(row.id)
    finally:
        service.close()
        store.close()


def test_formed_ask_is_not_an_acted_comparator_item(tmp_path):
    from colonymind.plasticity.learners.salience import factor, observe
    service = setup_runtime(tmp_path)
    store = InitiativeStore(state_dir=tmp_path)
    try:
        item = candidate()
        context = {}
        factor(service, 'salience-test', item, NOW, context=context)
        row, _ = store.create_intention(kind='task', type=item.type, title=item.title, drive=item.drive,
            cls='internal', decision='ask', decision_reason='fixture', status='asked', dedup_key='fixture:0',
            context={}, created_at=NOW)
        observe(service, 'salience-test', SimpleNamespace(eligible=((item, .81),), context=context,
            acted=((item, row.id),), rescore=lambda c, overrides: .81), NOW)
        assert service.loaded['salience-test'].comparator.records() == []
    finally:
        service.close()
        store.close()


def test_reference_withdrawal_during_formation_keeps_ranked_pair(tmp_path):
    from colonymind.plasticity.learners.salience import factor, observe
    service = setup_runtime(tmp_path)
    store = InitiativeStore(state_dir=tmp_path)
    try:
        binding = service.loaded['salience-test']
        previous = promote(binding, .2)
        current = promote(binding, .4)
        item, context = candidate(), {}
        factor(service, 'salience-test', item, NOW, context=context)
        binding.controller._withdraw(previous.id, reason='fixture reference withdrawal')
        row, _ = store.create_intention(kind='task', type=item.type, title=item.title, drive=item.drive,
            cls='internal', decision='act', decision_reason='fixture', status='approved', dedup_key='fixture:0',
            context={}, created_at=NOW)
        observe(service, 'salience-test', SimpleNamespace(eligible=((item, .81),), context=context,
            acted=((item, row.id),), rescore=lambda c, overrides: .81), NOW)
        record, = binding.comparator.records()
        assert record['closure'] == current.id
        assert record['reference'] == previous.id
    finally:
        service.close()
        store.close()


def test_report_pairs_all_served_outcomes_with_their_frozen_anchor(tmp_path):
    from colonymind.plasticity.learners.salience import factor, observe, report
    service = setup_runtime(tmp_path)
    store = InitiativeStore(state_dir=tmp_path)
    try:
        binding = service.loaded['salience-test']
        items, acted, context = [], [], {}
        for i in range(21):
            item = candidate(i)
            factor(service, 'salience-test', item, NOW, context=context)
            row, _ = store.create_intention(kind='task', type=item.type, title=item.title, drive=item.drive,
                cls='internal', decision='act', decision_reason='fixture', status='approved',
                dedup_key=item.dedup_key, context={}, created_at=NOW)
            items.append((item, .81))
            acted.append((item, row.id))
            store.transition(row.id, 'done', action='rated', at=NOW,
                             details={'by': 'owner', 'verdict': 'useful'}, verdict='useful')
        observe(service, 'salience-test', SimpleNamespace(eligible=items, context=context,
            acted=acted, rescore=lambda c, overrides: .81), NOW)
        binding.driver.poll(service)
        detail = report(service.config, 'salience-test', runtime=service)
        assert detail['outcomes'] == detail['p0_outcomes'] == 21
        assert detail['served_brier'] == detail['p0_brier'] == .25
        assert all(store.get(ref).context['salience']['p0'] == .5 for _, ref in acted)
    finally:
        service.close()
        store.close()


def test_propensity_ranks_keep_owed_positions_in_the_full_eligible_order(tmp_path):
    from colonymind.plasticity.learners.salience import observe
    service = setup_runtime(tmp_path)
    store = InitiativeStore(state_dir=tmp_path)
    try:
        owed, optional = candidate(), candidate(1)
        owed.source_type = 'commitment'
        observe(service, 'salience-test', SimpleNamespace(eligible=((owed, 1.), (optional, .8)), context={},
            acted=(), rescore=lambda c, overrides: 1. if c is owed else .8), NOW)
        item, = service.loaded['salience-test'].driver.rows.propensities()[0]['candidates']
        assert item['served_rank'] == item['reference_rank'] == 2
    finally:
        service.close()
        store.close()


def test_source_erasure_during_formation_cannot_recreate_propensities(tmp_path):
    from colonymind.plasticity.learners.salience import factor, observe
    service = setup_runtime(tmp_path)
    store = InitiativeStore(state_dir=tmp_path)
    try:
        item, context = candidate(), {}
        item.source_id = 'feature-source'
        factor(service, 'salience-test', item, NOW, context=context)
        service.erase_sources(['feature-source'])
        observe(service, 'salience-test', SimpleNamespace(eligible=((item, .81),), context=context,
            acted=(), rescore=lambda c, overrides: .81), NOW)
        assert service.loaded['salience-test'].driver.rows.propensities() == []
        # A new pass with a stale candidate also cannot retain that source.
        context = {}
        factor(service, 'salience-test', item, NOW, context=context)
        observe(service, 'salience-test', SimpleNamespace(eligible=((item, .81),), context=context,
            acted=(), rescore=lambda c, overrides: .81), NOW)
        assert service.loaded['salience-test'].driver.rows.propensities() == []
    finally:
        service.close()
        store.close()


def test_corrected_owner_verdict_replaces_the_by_reference_training_row(tmp_path):
    from datetime import timedelta
    from colonymind.plasticity.learners.salience import frame
    service = setup_runtime(tmp_path)
    store = InitiativeStore(state_dir=tmp_path)
    try:
        binding = service.loaded['salience-test']
        item = candidate()
        at = NOW - timedelta(days=1)
        row, _ = store.create_intention(kind='task', type=item.type, title=item.title, drive=item.drive,
            cls='internal', decision='act', decision_reason='fixture', status='done', dedup_key='fixture:0',
            context={'salience': {'frame': frame(item, at, {}, binding.driver.rows.key)}}, created_at=at)
        store.transition(row.id, 'done', action='rated', at=NOW,
                         details={'by': 'owner', 'verdict': 'useful'}, verdict='useful')
        binding.driver.period()
        assert binding.driver.build().rows == 1
        store.transition(row.id, 'done', action='rated', at=NOW,
                         details={'by': 'owner', 'verdict': 'wrong'}, verdict='wrong')
        binding.driver.period()
        build = binding.driver.build()
        assert build.rows == 1
        assert binding.driver.learner.anchor(build.state) == 1e-6
    finally:
        service.close()
        store.close()


def test_distinct_passes_at_the_same_clock_time_keep_separate_propensities(tmp_path):
    from colonymind.plasticity.learners.salience import observe
    service = setup_runtime(tmp_path)
    store = InitiativeStore(state_dir=tmp_path)
    try:
        for i in range(2):
            observe(service, 'salience-test', SimpleNamespace(eligible=((candidate(i), .81),), context={},
                acted=(), rescore=lambda c, overrides: .81), NOW)
        passes = service.loaded['salience-test'].driver.rows.propensities()
        assert len(passes) == 2
        assert passes[0]['ref'] != passes[1]['ref']
    finally:
        service.close()
        store.close()
