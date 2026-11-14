"""The judge's calibration: about 60 answers labelled by hand, and how often the judge agrees with them.

1. ``sample`` draws a labelling sheet from a verdict file: ``size`` rows (60 by default) that have a verdict to agree
   with (a row the reader skipped or refused, or the judge left undecided, has none), shared across question types
   by largest remainder and taken within a type in the order of SHA-256 of ``"<seed>:<question>:<arm>:<budget>"``
   (``datasets/subsets.py``'s scheme), so anyone can draw the same sheet again. The sheet is blind: each item shows
   the question, its date, the reference answer (or rubric) and the reader's response, never the judge's verdict.
   It names the verdict file by its SHA-256.
2. A person writes each item's ``label``: true when the response is right by the benchmark's own rule, false when it
   is not.
3. ``agreement`` reads the labelled sheet against the same verdict file and reports how many labels the judge agrees
   with (a Wilson interval), Cohen's kappa, the confusion table and the counts per type. An unlabelled item is
   counted and left out; a label other than true or false, or a sheet drawn from another verdict file, is refused.

    python benchmarks/public_memory/calibration.py sample --dataset longmemeval_s --verdicts verdicts.jsonl \\
        --out sheet.json
    python benchmarks/public_memory/calibration.py agreement --verdicts verdicts.jsonl --sheet sheet.json \\
        --output agreement.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

# Support both package imports and the documented direct script commands.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.public_memory import reader, scoring
from benchmarks.public_memory.datasets import subsets

PROTOCOL = 'public-memory-judge-calibration-1'
SIZE = 60


class CalibrationError(ValueError):
    """A sheet that cannot be read against the verdict file."""


def _rows(path: Path) -> list:
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def _key(row: Mapping[str, Any]) -> tuple:
    return (row['question'], row['arm'], row.get('budget_chars'))


def _digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reference(question) -> str:
    rubric = question.detail.get('rubric') if isinstance(question.detail, Mapping) else None
    if isinstance(rubric, list) and rubric:
        return '\n'.join(f'- {item}' for item in rubric)
    return question.answer


def sample(verdicts: Path, questions: Mapping[str, Any], *, size: int = SIZE, seed=1) -> dict:
    """A blind labelling sheet of ``size`` judged rows of ``verdicts``, stratified by question type."""
    seed = str(seed)
    rows = [row for row in _rows(verdicts)
            if row.get('right') is not None and not row.get('skipped') and not row.get('reader_refused')]
    strata: dict = {}
    for row in rows:
        strata.setdefault(row['type'], []).append(row)
    quotas = subsets.allocate({name: len(items) for name, items in strata.items()}, size)
    chosen = []
    for name in sorted(quotas):
        ranked = sorted(strata[name], key=lambda row: subsets.rank(seed, ':'.join(map(str, _key(row)))))
        chosen.extend(ranked[:quotas[name]])
    chosen.sort(key=lambda row: subsets.rank(seed, ':'.join(map(str, _key(row)))))
    items = []
    for row in chosen:
        question = questions[row['question']]
        items.append({'question': row['question'], 'arm': row['arm'], 'budget_chars': row.get('budget_chars'),
                      'type': row['type'], 'text': question.question, 'date': reader.question_date(question),
                      'reference': _reference(question), 'abstention': bool(question.abstention),
                      'response': row.get('response'), 'label': None})
    return {'protocol': PROTOCOL, 'verdicts_sha256': _digest(verdicts), 'seed': seed, 'size': len(items),
            'strata': {name: count for name, count in sorted(quotas.items()) if count}, 'items': items}


def _kappa(table: Mapping[str, int]) -> Any:
    total = sum(table.values())
    if not total:
        return None
    judge_yes = table['judge_yes_label_yes'] + table['judge_yes_label_no']
    label_yes = table['judge_yes_label_yes'] + table['judge_no_label_yes']
    observed = (table['judge_yes_label_yes'] + table['judge_no_label_no']) / total
    chance = (judge_yes * label_yes + (total - judge_yes) * (total - label_yes)) / (total * total)
    return None if chance == 1 else (observed - chance) / (1 - chance)


def _table(pairs) -> dict:
    table = {'judge_yes_label_yes': 0, 'judge_yes_label_no': 0, 'judge_no_label_yes': 0, 'judge_no_label_no': 0}
    for judged, label in pairs:
        table[f'judge_{"yes" if judged else "no"}_label_{"yes" if label else "no"}'] += 1
    return table


def agreement(verdicts: Path, sheet: Mapping[str, Any]) -> dict:
    """The judge's agreement with the sheet's hand labels."""
    if sheet.get('protocol') != PROTOCOL:
        raise CalibrationError('Not a judge calibration sheet')
    if sheet.get('verdicts_sha256') != _digest(verdicts):
        raise CalibrationError('The sheet was drawn from another verdict file')
    judged = {_key(row): row for row in _rows(verdicts)}
    pairs, types, unlabelled = [], {}, 0
    for item in sheet.get('items') or []:
        label = item.get('label')
        if label is None:
            unlabelled += 1
            continue
        if type(label) is not bool:
            raise CalibrationError(f'{item.get("question")}: a label is true or false')
        row = judged.get(_key(item))
        if row is None or row.get('right') is None:
            raise CalibrationError(f'{item.get("question")}: no verdict to agree with')
        pairs.append((row['right'], label))
        types.setdefault(item['type'], []).append((row['right'], label))
    table = _table(pairs)
    agree = table['judge_yes_label_yes'] + table['judge_no_label_no']
    return {'protocol': PROTOCOL, 'verdicts_sha256': sheet['verdicts_sha256'], 'labelled': len(pairs),
            'unlabelled': unlabelled, 'agreement': scoring.wilson(agree, len(pairs)), 'kappa': _kappa(table),
            'confusion': table,
            'types': {name: {'labelled': len(members),
                             'agreement': scoring.wilson(sum(judged == label for judged, label in members),
                                                         len(members))}
                      for name, members in sorted(types.items())}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    sampling = commands.add_parser('sample', help='draw a blind labelling sheet')
    sampling.add_argument('--dataset', required=True, choices=('longmemeval_s', 'beam'))
    sampling.add_argument('--data-dir', type=Path)
    sampling.add_argument('--offline', action='store_true')
    sampling.add_argument('--verdicts', type=Path, required=True)
    sampling.add_argument('--size', type=int, default=SIZE)
    sampling.add_argument('--seed', default='1')
    sampling.add_argument('--out', type=Path, required=True)
    agreeing = commands.add_parser('agreement', help='the judge\'s agreement with a labelled sheet')
    agreeing.add_argument('--verdicts', type=Path, required=True)
    agreeing.add_argument('--sheet', type=Path, required=True)
    agreeing.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'sample':
        dataset = reader.runner.load_dataset(args.dataset, args.data_dir, args.offline)
        sheet = sample(args.verdicts, {question.id: question for question in dataset.questions}, size=args.size,
                       seed=args.seed)
        args.out.write_text(json.dumps(sheet, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
        print(json.dumps({'items': sheet['size'], 'strata': sheet['strata']}))
        return 0
    report = agreement(args.verdicts, json.loads(args.sheet.read_text(encoding='utf-8')))
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({'labelled': report['labelled'], 'kappa': report['kappa']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
