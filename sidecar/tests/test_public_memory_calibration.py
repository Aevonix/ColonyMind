"""The judge's calibration (``benchmarks/public_memory/calibration.py``): a blind labelling sheet drawn from a
verdict file, and the judge's agreement with the hand labels.

Synthetic verdict rows only; nothing here reads a model or a dataset file.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

BENCH = Path(__file__).resolve().parents[2] / 'benchmarks' / 'public_memory'
UTC = timezone.utc


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def calibration():
    return _load('colonymind_public_memory_calibration', BENCH / 'calibration.py')


@pytest.fixture(scope='module')
def records():
    return _load('colonymind_public_memory_records_for_calibration', BENCH / 'datasets' / 'records.py')


TYPES = ('temporal-reasoning', 'knowledge-update', 'multi-session')


def verdict(qid, kind, right, arm='all-on', **extra):
    return {'protocol': 'public-memory-verdict-1', 'question': qid, 'conversation': qid, 'type': kind, 'arm': arm,
            'budget_chars': 4000, 'right': right, 'undecided': right is None, 'reader_refused': False,
            'skipped': None, 'response': f'answer to {qid}', 'judge': {'name': 'open-judge-70b', 'revision': 'r'},
            **extra}


def questions(records, rows):
    return {row['question']: records.Question(
        row['question'], row['question'], row['type'], f'question {row["question"]}?', f'gold {row["question"]}',
        datetime(2023, 4, 9, 12, tzinfo=UTC), None, False, (), (), {}) for row in rows}


@pytest.fixture
def verdicts(tmp_path):
    rows = [verdict(f'{kind}-{n}', kind, n % 3 != 0) for kind in TYPES for n in range(30)]
    rows += [verdict('skipped-1', 'multi-session', None, skipped='capture_failed'),
             verdict('refused-1', 'multi-session', False, reader_refused=True),
             verdict('undecided-1', 'multi-session', None)]
    path = tmp_path / 'verdicts.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return path, rows


def test_the_sheet_is_a_blind_seeded_stratified_sample(calibration, records, verdicts):
    path, rows = verdicts
    known = questions(records, rows)
    sheet = calibration.sample(path, known, size=60, seed=1)
    again = calibration.sample(path, known, size=60, seed=1)
    other = calibration.sample(path, known, size=60, seed=2)
    assert sheet == again and sheet['items'] != other['items']
    assert sheet['verdicts_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    items = sheet['items']
    assert len(items) == 60
    # Every type in proportion; only rows with a verdict to agree with.
    assert {kind: sum(item['type'] == kind for item in items) for kind in TYPES} == {kind: 20 for kind in TYPES}
    assert not {'skipped-1', 'refused-1', 'undecided-1'} & {item['question'] for item in items}
    # Blind: the labeller sees the question, the reference and the response, never the judge's verdict.
    first = items[0]
    assert first['label'] is None and first['text'].startswith('question ') and first['reference'].startswith('gold')
    assert first['response'].startswith('answer to ')
    assert not {'right', 'verdict', 'judge', 'undecided'} & set(first)


def test_agreement_with_hand_labels_is_reported_with_kappa(calibration, records, verdicts, tmp_path):
    path, rows = verdicts
    sheet = calibration.sample(path, questions(records, rows), size=12, seed=3)
    judged = {(row['question'], row['arm'], row['budget_chars']): row['right'] for row in rows}
    # The labeller agrees on all but two items, and leaves one unlabelled.
    for number, item in enumerate(sheet['items']):
        right = judged[(item['question'], item['arm'], item['budget_chars'])]
        item['label'] = (not right) if number in (0, 1) else right
    sheet['items'][-1]['label'] = None
    report = calibration.agreement(path, sheet)
    assert report['labelled'] == 11 and report['unlabelled'] == 1
    assert report['agreement']['successes'] == 9 and report['agreement']['trials'] == 11
    table = report['confusion']
    assert sum(table.values()) == 11
    yes_yes, no_no = table['judge_yes_label_yes'], table['judge_no_label_no']
    judge_yes = yes_yes + table['judge_yes_label_no']
    label_yes = yes_yes + table['judge_no_label_yes']
    expected = (9 / 11 - (judge_yes * label_yes + (11 - judge_yes) * (11 - label_yes)) / 121) / (
        1 - (judge_yes * label_yes + (11 - judge_yes) * (11 - label_yes)) / 121)
    assert (yes_yes + no_no) == 9 and report['kappa'] == pytest.approx(expected)
    assert set(report['types']) <= set(TYPES)


def test_agreement_refuses_a_sheet_of_another_verdict_file_or_a_label_that_is_not_yes_or_no(calibration, records,
                                                                                            verdicts, tmp_path):
    path, rows = verdicts
    sheet = calibration.sample(path, questions(records, rows), size=6, seed=1)
    sheet['items'][0]['label'] = 'mostly'
    with pytest.raises(calibration.CalibrationError):
        calibration.agreement(path, sheet)
    sheet['items'][0]['label'] = True
    changed = tmp_path / 'changed.jsonl'
    changed.write_text(path.read_text() + json.dumps(verdict('extra', 'multi-session', True)) + '\n')
    with pytest.raises(calibration.CalibrationError):
        calibration.agreement(changed, sheet)
