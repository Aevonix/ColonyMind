"""The public memory scorer: evidence reach, right answers and traps, counted apart, with their statistics.

Synthetic packets and answers only; nothing here reads a model, a dataset or a trace.
"""
import importlib.util
from pathlib import Path

import pytest

from colonymind.qualification import paired_statistics

SCORING = Path(__file__).resolve().parents[2] / 'benchmarks' / 'public_memory' / 'scoring.py'


@pytest.fixture(scope='module')
def scoring():
    spec = importlib.util.spec_from_file_location('public_memory_scoring', SCORING)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Reach is structural: identifiers, never text.

def test_structural_reach_counts_gold_items_by_identifier_and_ranks_the_first(scoring):
    gold = [{'turn-a', 'hash-a'}, {'turn-b'}]
    shown = [{'turn-x'}, {'hash-a', 'turn-y'}, {'turn-z'}]
    result = scoring.reach(gold, shown)
    assert result == {'basis': 'structural', 'weaker': False, 'gold': 2, 'reached': 1, 'any': True,
                      'all': False, 'first_rank': 2}
    everything = scoring.reach(gold, [{'turn-b'}, {'hash-a'}])
    assert (everything['reached'], everything['any'], everything['all'], everything['first_rank']) == (2, True, True, 1)
    missed = scoring.reach(gold, [{'turn-x'}])
    assert (missed['reached'], missed['any'], missed['all'], missed['first_rank']) == (0, False, False, None)


def test_structural_reach_never_matches_text(scoring):
    # An identifier that merely contains a gold identifier is a different item.
    result = scoring.reach([{'turn:owner-1:abc'}], [{'turn:owner-1:abcd'}, {'TURN:OWNER-1:ABC'}])
    assert result['any'] is False


def test_a_question_without_gold_evidence_has_no_reach(scoring):
    result = scoring.reach([], [{'turn-a'}])
    assert (result['gold'], result['any'], result['all'], result['first_rank']) == (0, None, None, None)


def test_session_reach_is_a_returned_gold_session(scoring):
    result = scoring.session_reach({'chat-03'}, ['sms-01', 'chat-03', 'chat-03'])
    assert result == {'basis': 'session', 'weaker': False, 'gold': 1, 'reached': 1, 'any': True, 'all': True,
                      'first_rank': 2}
    assert scoring.session_reach({'chat-03'}, [])['any'] is False


def test_span_reach_is_the_weaker_check_on_whole_words(scoring):
    result = scoring.span_reach(['east room', '09:15'], ['nothing here', 'It moved to the East  Room at 09:15.'])
    assert result['basis'] == 'span' and result['weaker'] is True
    assert (result['reached'], result['all'], result['first_rank']) == (2, True, 2)
    # Whole words only: a span inside longer words is not shown.
    partial = scoring.span_reach(['east room'], ['a feast roommate'])
    assert partial['any'] is False
    assert scoring.span_reach([], ['east room'])['any'] is None


def test_two_by_two_from_synthetic_packets(scoring):
    rows = [
        {'question': 'q1', 'reach': scoring.reach([{'h1'}], [{'h9'}, {'h1'}]), 'right': True},
        {'question': 'q2', 'reach': scoring.reach([{'h2'}], [{'h9'}]), 'right': True},
        {'question': 'q3', 'reach': scoring.reach([{'h3'}, {'h4'}], [{'h3'}]), 'right': False},
        {'question': 'q4', 'reach': scoring.reach([], [{'h3'}]), 'right': False},
        {'question': 'q5', 'reach': scoring.reach([{'h5'}], []), 'right': False},
        {'question': 'q6', 'reach': scoring.reach([{'h6'}], [{'h6'}]), 'right': None},
    ]
    assert scoring.two_by_two(rows) == {'reached_right': 1, 'reached_wrong': 1, 'missed_right': 1,
                                        'missed_wrong': 1, 'unscored': 2}


# Traps: a stale value asserted, a side picked, an answer where none exists. Outcomes rest on structured judgments
# (the values an answer asserts or chooses), never on words found in its text.

@pytest.mark.parametrize('arguments, outcome', [
    (dict(right=True, asserted=['east room'], stale=['corner office']), 'correct'),
    (dict(right=False, asserted=['Corner Office'], stale=['corner office']), 'trap'),
    # Both values asserted as current, neither chosen, is a trap even beside a right verdict.
    (dict(right=True, asserted=['corner office', 'east room'], stale=['corner office']), 'trap'),
    (dict(right=False, asserted=['unknown'], stale=['corner office']), 'other'),
    (dict(right=False, asked=True, stale=['corner office']), 'asked'),
    # Asking while asserting the stale value as current still asserts it.
    (dict(right=False, asked=True, asserted=['corner office'], stale=['corner office']), 'trap'),
    # "It is no longer the corner office; I don't know the new location" names the old value and asserts none.
    (dict(right=False, asserted=[], stale=['corner office']), 'other'),
    # Text handed over as a value is one value, never searched for the stale one inside it.
    (dict(right=False, asserted=["It is no longer the corner office; I don't know the new location"],
          stale=['corner office']), 'other'),
])
def test_update_trap_outcomes(scoring, arguments, outcome):
    assert scoring.trap_outcome('update', **arguments)['outcome'] == outcome


def test_update_trap_names_the_stale_value_it_matched(scoring):
    result = scoring.trap_outcome('update', right=False, asserted=['Top shelf.'], stale=['top shelf', 'hall cupboard'])
    assert result == {'kind': 'update', 'outcome': 'trap', 'matched': ['top shelf']}


@pytest.mark.parametrize('arguments, outcome', [
    (dict(right=True, asked=True, sides=['09:15', '10:45']), 'asked'),
    (dict(right=False, asserted=['09:15'], sides=['09:15', '10:45']), 'trap'),
    # "09:15 is correct, not 10:45" chooses 09:15: the judgment names the choice, and the rejected side is no choice.
    (dict(right=False, asserted=['09:15'], asked=False, sides=['09:15', '10:45']), 'trap'),
    (dict(right=False, asserted=[], sides=['09:15', '10:45']), 'other'),
    # Both given, neither chosen: no side picked.
    (dict(right=False, asserted=['09:15', '10:45'], sides=['09:15', '10:45']), 'other'),
    # An ask that also chooses a side has picked it.
    (dict(right=False, asked=True, asserted=['10:45'], sides=['09:15', '10:45']), 'trap'),
])
def test_conflict_trap_outcomes(scoring, arguments, outcome):
    assert scoring.trap_outcome('conflict', **arguments)['outcome'] == outcome


@pytest.mark.parametrize('arguments, outcome', [
    (dict(right=True, answerable=False, asserted=[]), 'correct'),
    (dict(right=False, answerable=True, asserted=['hall cupboard']), 'trap'),
    (dict(right=False, answerable=True, asserted=['  ']), 'other'),
    (dict(right=True, asserted=[]), 'correct'),
])
def test_abstention_trap_outcomes(scoring, arguments, outcome):
    assert scoring.trap_outcome('abstain', **arguments)['outcome'] == outcome


def test_trap_outcomes_take_no_answer_text(scoring):
    with pytest.raises(TypeError):
        scoring.trap_outcome('update', right=False, answer='the corner office', stale=['corner office'])


def test_recall_questions_have_no_trap(scoring):
    assert scoring.trap_outcome('recall', right=True, asserted=['x'])['outcome'] == 'correct'
    assert scoring.trap_outcome('recall', right=False, asserted=['x'])['outcome'] == 'other'
    with pytest.raises(ValueError):
        scoring.trap_outcome('guess', right=True)


# Statistics: known values, and seeded determinism.

@pytest.mark.parametrize('successes, trials, lower, upper', [
    (8, 10, 0.4902, 0.9433),
    (0, 10, 0.0, 0.2775),
    (10, 10, 0.7225, 1.0),
    (50, 100, 0.4038, 0.5962),
])
def test_wilson_interval_matches_known_values(scoring, successes, trials, lower, upper):
    result = scoring.wilson(successes, trials)
    assert result['rate'] == successes / trials
    assert result['lower'] == pytest.approx(lower, abs=1e-4)
    assert result['upper'] == pytest.approx(upper, abs=1e-4)
    assert scoring.wilson(0, 0) is None
    with pytest.raises(ValueError):
        scoring.wilson(3, 2)


def _rows(values, prefix='q'):
    return [{'question': f'{prefix}{index}', 'right': value, 'cluster': f'c{index // 2}'}
            for index, value in enumerate(values)]


def test_paired_contrast_uses_the_exact_sign_test(scoring):
    treatment = _rows([True] * 9 + [False] + [True] * 5)
    comparator = _rows([False] * 9 + [True] + [True] * 5)
    result = scoring.contrast(treatment, comparator, key='right', seed=7)
    assert (result['wins'], result['losses'], result['ties']) == (9, 1, 5)
    assert result['sign_test']['p_two_sided'] == pytest.approx(0.021484375)
    assert result['sign_test'] == paired_statistics.sign_test(9, 1)
    assert (result['declared_units'], result['unavailable_units']) == (15, 0)
    with pytest.raises(ValueError):
        scoring.contrast(treatment[:-1], comparator, key='right', seed=7)
    with pytest.raises(ValueError):
        scoring.contrast(_rows(['yes'] * 15), comparator, key='right', seed=7)


def test_a_question_without_an_outcome_withholds_the_contrast(scoring):
    """The paired track's rule: a contrast with any unattributed question is unavailable; no partial cohort."""
    treatment = _rows([True] * 6 + [None])
    comparator = _rows([False] * 7)
    assert scoring.contrast(treatment, comparator, key='right', seed=7) == {
        'verdict': 'unavailable', 'declared_units': 7, 'unavailable_units': 1}
    assert scoring.contrast(comparator, treatment, key='right', seed=7)['verdict'] == 'unavailable'
    assert scoring.contrast([], [], key='right', seed=7) == {
        'verdict': 'unavailable', 'declared_units': 0, 'unavailable_units': 0}


def test_repetitions_are_averaged_per_scenario_before_testing(scoring):
    """The paired track's unit: each scenario's repetitions become one pass value per arm, then the sign test."""
    def attempts(values):
        return [{'question': f'{scenario}.repeat-{number}', 'scenario': scenario, 'right': value}
                for scenario, repeats in values.items() for number, value in enumerate(repeats, 1)]
    treatment = attempts({'a': [True, True], 'b': [True, False], 'c': [False, False]})
    comparator = attempts({'a': [False, False], 'b': [False, False], 'c': [False, True]})
    result = scoring.contrast(treatment, comparator, key='right', seed=3, unit_key='scenario')
    assert (result['units'], result['wins'], result['losses'], result['ties']) == (3, 2, 1, 0)
    assert result['treatment_pass_rate'] == pytest.approx(0.5)
    assert result['comparator_pass_rate'] == pytest.approx(1 / 6)
    assert result == {**paired_statistics.contrast({'a': (1.0, 0.0), 'b': (0.5, 0.0), 'c': (0.0, 0.5)}, seed=3),
                      'declared_units': 3, 'unavailable_units': 0}
    # One unattributed repetition makes its scenario unavailable, and the contrast is withheld.
    treatment[1]['right'] = None
    assert scoring.contrast(treatment, comparator, key='right', seed=3, unit_key='scenario') == {
        'verdict': 'unavailable', 'declared_units': 3, 'unavailable_units': 1}
    # A repetition filed under another scenario in one arm is refused.
    comparator[0]['scenario'] = 'b'
    with pytest.raises(ValueError):
        scoring.contrast(treatment, comparator, key='right', seed=3, unit_key='scenario')


def test_the_bootstrap_is_deterministic_under_a_seed_and_resamples_whole_clusters(scoring):
    rows = [{'question': f'q{i}', 'right': value, 'cluster': cluster}
            for i, (value, cluster) in enumerate([(True, 'a'), (True, 'a'), (True, 'a'), (False, 'b'),
                                                  (False, 'b'), (True, 'c'), (False, 'c')])]
    first = scoring.interval(rows, key='right', seed=11, cluster_key='cluster')
    again = scoring.interval(rows, key='right', seed=11, cluster_key='cluster')
    assert first == again and first['clusters'] == 3 and first['seed'] == 11
    assert 0 <= first['lower'] <= first['upper'] <= 1
    other = scoring.interval(rows, key='right', seed=12, cluster_key='cluster')
    assert other['seed'] == 12
    # Clustered contrasts resample conversations, not questions.
    treatment = [dict(row, right=True) for row in rows]
    clustered = scoring.contrast(treatment, rows, key='right', seed=5, cluster_key='cluster')
    assert clustered['clusters'] == 3


# Per-type tables.

def test_summary_tables_count_reach_right_and_traps_apart(scoring):
    rows = [
        {'question': 'a', 'type': 'update', 'reach': scoring.reach([{'h1'}], [{'h1'}]), 'right': True,
         'trap': scoring.trap_outcome('update', right=True, asserted=['east room'], stale=['main hall']),
         'chars': 3000, 'budget': 4000, 'tokens': 700},
        {'question': 'b', 'type': 'update', 'reach': scoring.reach([{'h2'}], [{'h9'}, {'h8'}]), 'right': False,
         'trap': scoring.trap_outcome('update', right=False, asserted=['main hall'], stale=['main hall']),
         'chars': 1000, 'budget': 4000, 'tokens': 300},
        {'question': 'c', 'type': 'abstain', 'reach': scoring.reach([], [{'h1'}]), 'right': True,
         'trap': scoring.trap_outcome('abstain', right=True, answerable=False), 'chars': 0, 'budget': 4000,
         'tokens': 0},
    ]
    tables = scoring.summarize(rows)
    update = tables['types']['update']
    assert update['questions'] == 2
    assert update['right']['successes'] == 1 and update['right']['trials'] == 2
    assert update['reach_any']['successes'] == 1 and update['reach_all']['trials'] == 2
    assert update['two_by_two'] == {'reached_right': 1, 'reached_wrong': 0, 'missed_right': 0,
                                    'missed_wrong': 1, 'unscored': 0}
    assert update['traps'] == {'correct': 1, 'trap': 1}
    assert update['trap_rate']['successes'] == 1 and update['trap_rate']['trials'] == 2
    assert update['first_rank_median'] == 1
    assert update['fill_ratio_mean'] == pytest.approx(0.5)
    assert update['tokens_mean'] == 500
    assert tables['types']['abstain']['reach_any'] is None
    assert tables['overall']['questions'] == 3
    assert tables['overall']['reach_basis'] == {'structural': 2}
