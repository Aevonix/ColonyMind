"""Reviewed-2 clarifies two output contracts without changing answers or old data."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from colonymind.qualification import paired_cases as paired


DIRECTORY = Path(paired.__file__).parent / 'fixtures'


def baseline(arm='base_hermes'):
    return paired.cases(arm, dataset_version=paired.BASELINE_VERSION)


def test_current_default_is_reviewed_two_with_frozen_provenance():
    manifest, cases, checksum = paired.load_dataset(DIRECTORY / paired.VERSION)
    assert paired.VERSION == paired.BASELINE_VERSION == 'paired-agent-reviewed-2'
    assert paired.DATASET_VERSIONS == (paired.VERSION, paired.WORKFLOW_VERSION)
    assert len(cases) == manifest['scenario_count'] == 60
    assert checksum == paired.DATASET_SHA256
    assert manifest['derived_from'] == {
        'dataset_id': 'paired-agent-reviewed-1',
        'manifest_sha256': 'f0ed44967d6cb63b1318620e31ca9c25532c5dbf07d322fa0038aee29ecff373'}
    assert len(manifest['contract_changes']) == 3
    for retired in ('paired-agent-pilot-1', 'paired-agent-reviewed-1'):
        assert not (DIRECTORY / retired).exists()
        with pytest.raises(ValueError, match='dataset version'):
            paired.cases('base_hermes', dataset_version=retired)


def test_both_baseline_arms_keep_identical_inputs_oracles_and_existing_budgets():
    for left, right in zip(baseline(), baseline('colonymind')):
        assert left.version == right.version == paired.BASELINE_VERSION
        a, b = deepcopy(left.inputs), deepcopy(right.inputs)
        assert a.pop('arm') == 'base_hermes' and b.pop('arm') == 'colonymind'
        assert a == b and left.oracle == right.oracle
        assert 0 < left.timeout_seconds == right.timeout_seconds <= 600
        assert left.max_output_bytes == right.max_output_bytes == 262144
        # Frozen datasets keep stock Hermes tool loading and bare turns; only generated
        # families declare eager loading and message timestamps.
        for key in ('tool_loading', 'message_timestamps', 'environment_note'):
            assert key not in left.inputs and key not in right.inputs


@pytest.mark.parametrize('scenario,path,valid,invalid', [
    ('separate-peoples-preferences', 'refreshments.json',
     {'people': {'Mira': {'drink': 'water', 'avoid': ['peanuts']},
                 'Rowan': {'drink': 'coffee', 'avoid': []}}},
     {'Mira': {'drink': 'water', 'avoid': ['peanuts']},
      'Rowan': {'drink': 'coffee', 'avoid': []}}),
    ('independent-resources-critical-path', 'pipeline.json',
     {'tasks': [{'id': 'A', 'start': 0, 'finish': 2},
                {'id': 'B', 'start': 0, 'finish': 3},
                {'id': 'C', 'start': 2, 'finish': 4},
                {'id': 'D', 'start': 4, 'finish': 5}]},
     {'tasks': [{'source_id': 'A', 'start': 0, 'finish': 2},
                {'source_id': 'B', 'start': 0, 'finish': 3},
                {'source_id': 'C', 'start': 2, 'finish': 4},
                {'source_id': 'D', 'start': 4, 'finish': 5}]}),
])
def test_explicit_contract_accepts_correct_shape_and_rejects_previous_ambiguity(
        scenario, path, valid, invalid):
    case = next(c for c in baseline() if c.inputs['scenario'] == scenario)
    user = case.inputs['episodes'][-1]['user']
    assert 'top-level property named ' + ('people' if scenario.startswith('separate') else 'tasks') in user
    for value, passes in ((valid, True), (invalid, False)):
        effects = {'turns_completed': len(case.inputs['episodes']),
                   'declared_turns': len(case.inputs['episodes']),
                   'artifacts': {path: json.dumps(value)}}
        assert all(paired.assess({'effects': effects}, case.oracle).values()) is passes


def test_baseline_preserves_semantic_failures_despite_valid_shape():
    case = next(c for c in baseline() if c.inputs['scenario'] == 'separate-peoples-preferences')
    stale = {'people': {'Mira': {'drink': 'tea', 'avoid': ['peanuts']},
                        'Rowan': {'drink': 'coffee', 'avoid': []}}}
    effects = {'turns_completed': 3, 'declared_turns': 3,
               'artifacts': {'refreshments.json': json.dumps(stale)}}
    assert not all(paired.assess({'effects': effects}, case.oracle).values())
