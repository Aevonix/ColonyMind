"""The frozen old-value labels: what counts as a stale value, or as a side of a conflict, before any arm runs.

A trap is scored against values fixed in advance (``scoring.trap_outcome``): for a knowledge-update question the
values that were current once and have been replaced, taken from the earlier evidence turn; for a contradiction
question its two sides. They are labelled by a person, reviewed, frozen and pinned by the SHA-256 of the frozen
file, which the pre-registration records, so no label can be adjusted after answers are seen.

1. ``draft`` lists every knowledge-update and contradiction question of a dataset (or a subset) with its evidence
   turns in time order: each turn's session, date, role, words and label where the dataset gives one
   (``original_info``, ``first_statement``), and whether it is an earlier turn or the latest. The draft labels
   nothing itself: ``values`` is empty and ``reviewed`` false. A draft is its own protocol (``DRAFT``), never read
   as labels.
2. A reviewer writes each entry's ``values`` (one or more old values for an update; exactly the two sides for a
   conflict) and sets ``reviewed`` to true.
3. ``freeze`` refuses anything not reviewed or not well formed (an entry for a question the dataset does not hold or
   of another kind, an old value that is the current answer, two sides that are one value, a draft of another
   dataset file) and writes the frozen file: only the question, its kind, its type and its values, in canonical
   JSON, so the same review always freezes to the same bytes (``FROZEN``). It prints the SHA-256 to pre-register.
4. ``load`` (and ``check``) reads a frozen file only when its bytes hash to the pinned value and are what ``freeze``
   writes: the frozen protocol, the canonical form, entries in order, each well formed as ``freeze`` checks it. A
   draft pinned by its own hash, or a hand-made file, is refused. Given the dataset a run reads (the judge always
   gives it), ``load`` also refuses a file frozen from another dataset file, an entry for a question the dataset
   does not hold or of another kind or type, and an old value that is the current answer.

    python benchmarks/public_memory/labels.py draft --dataset longmemeval_s --out draft.json
    python benchmarks/public_memory/labels.py freeze --dataset longmemeval_s --draft draft.json --out old-values.json
    python benchmarks/public_memory/labels.py check --labels old-values.json --sha256 HASH
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional

# Support both package imports and the documented direct script commands.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.public_memory import scoring

DRAFT = 'public-memory-old-values-draft-1'
FROZEN = 'public-memory-old-values-1'
BODY_FIELDS = frozenset({'protocol', 'dataset', 'dataset_sha256', 'entries'})
ENTRY_FIELDS = frozenset({'question', 'kind', 'type', 'values'})
UPDATE, CONFLICT, ABSTAIN, RECALL = 'update', 'conflict', 'abstain', 'recall'
LABELLED = (UPDATE, CONFLICT)


class LabelError(ValueError):
    """A draft that cannot be frozen, or a frozen file that is not the pinned one."""


def trap_kind(question) -> str:
    """The trap a question can fall into (``scoring.KINDS``): an abstention question's is ``abstain`` whatever its
    type; a knowledge update's ``update``; a contradiction's ``conflict``; every other question's ``recall``."""
    if question.abstention:
        return ABSTAIN
    kind = str(question.type).replace('_', '-')
    if kind == 'knowledge-update':
        return UPDATE
    if kind == 'contradiction-resolution':
        return CONFLICT
    return RECALL


def _evidence(dataset, question) -> list:
    sessions = {session.key: (rank, session) for rank, session in
                enumerate(dataset.conversations[question.conversation].sessions)}
    items = sorted(question.evidence, key=lambda item: (sessions[item.session][0], item.turn))
    latest = max((sessions[item.session][0] for item in items), default=None)
    rows = []
    for item in items:
        rank, session = sessions[item.session]
        turn = session.turns[item.turn]
        rows.append({'session': item.session, 'turn': item.turn, 'label': item.label, 'at': session.at.isoformat(),
                     'role': turn.role, 'content': turn.content,
                     'position': 'latest' if rank == latest else 'earlier'})
    return rows


def draft(dataset, ids=()) -> dict:
    """The entries to label: every update and conflict question (of ``ids``, when given), values empty."""
    wanted = set(ids)
    entries = [{'question': question.id, 'type': question.type, 'kind': trap_kind(question),
                'text': question.question, 'answer': question.answer, 'evidence': _evidence(dataset, question),
                'values': [], 'reviewed': False, 'note': ''}
               for question in dataset.questions
               if trap_kind(question) in LABELLED and (not wanted or question.id in wanted)]
    return {'protocol': DRAFT, 'dataset': dataset.name, 'dataset_sha256': dataset.sha256, 'entries': entries}


def _values(entry: Mapping[str, Any], qid: str, answer: Optional[str] = None) -> list:
    """An entry's values, stripped, when they are well formed for its kind; ``answer`` is the question's current
    answer, which no old value may be (checked where the dataset is at hand)."""
    values = entry.get('values')
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise LabelError(f'{qid}: values must be a list of strings')
    values = [value.strip() for value in values]
    if any(not scoring.words(value) for value in values):
        raise LabelError(f'{qid}: a value has no words')
    if entry['kind'] == UPDATE:
        if not values:
            raise LabelError(f'{qid}: an update needs at least one old value')
        if answer is not None and any(scoring.same_value(value, answer) for value in values):
            raise LabelError(f'{qid}: an old value is the current answer')
    elif len(values) != 2 or scoring.same_value(*values):
        raise LabelError(f'{qid}: a conflict needs exactly its two sides, two different values')
    return values


def _canonical(body: Mapping[str, Any]) -> bytes:
    return (json.dumps(body, indent=1, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8')


def freeze(document: Mapping[str, Any], dataset) -> tuple:
    """The frozen file's bytes and their SHA-256, from a reviewed draft of ``dataset``."""
    if document.get('protocol') != DRAFT:
        raise LabelError('Not an old-value label draft')
    if document.get('dataset') != dataset.name or document.get('dataset_sha256') != dataset.sha256:
        raise LabelError('The draft was made from another dataset file')
    questions = {question.id: question for question in dataset.questions}
    frozen, seen = [], set()
    for entry in document.get('entries') or []:
        question = questions.get(entry.get('question'))
        if question is None:
            raise LabelError(f'{entry.get("question")}: no such question in {dataset.name}')
        if question.id in seen:
            raise LabelError(f'{question.id}: labelled twice')
        seen.add(question.id)
        if entry.get('kind') != trap_kind(question) or entry['kind'] not in LABELLED:
            raise LabelError(f'{question.id}: the entry is not of the question\'s kind ({trap_kind(question)})')
        if entry.get('reviewed') is not True:
            raise LabelError(f'{question.id}: not reviewed')
        frozen.append({'question': question.id, 'kind': entry['kind'], 'type': question.type,
                       'values': _values(entry, question.id, question.answer)})
    frozen.sort(key=lambda item: item['question'])
    body = {'protocol': FROZEN, 'dataset': dataset.name, 'dataset_sha256': dataset.sha256, 'entries': frozen}
    data = _canonical(body)
    return data, hashlib.sha256(data).hexdigest()


def load(path: Path, sha256: str, dataset=None) -> dict:
    """``{question: {kind, type, values}}`` from a frozen file whose bytes hash to ``sha256`` and are what
    ``freeze`` writes; checked against ``dataset`` too when it is given."""
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != str(sha256).strip().lower():
        raise LabelError(f'{path} is not the pinned label file')
    try:
        body = json.loads(data)
    except ValueError:
        raise LabelError(f'{path} is not a frozen label file') from None
    if (not isinstance(body, dict) or set(body) != BODY_FIELDS or body['protocol'] != FROZEN
            or not isinstance(body['entries'], list)):
        raise LabelError(f'{path} is not a frozen label file (a draft is frozen with labels.py freeze)')
    if _canonical(body) != data:
        raise LabelError(f'{path} is not in the form freeze writes')
    if dataset is not None and (body['dataset'] != dataset.name or body['dataset_sha256'] != dataset.sha256):
        raise LabelError(f'{path} was frozen from another dataset file')
    questions = {question.id: question for question in dataset.questions} if dataset is not None else None
    loaded: dict = {}
    for entry in body['entries']:
        if (not isinstance(entry, dict) or set(entry) != ENTRY_FIELDS or not isinstance(entry['question'], str)
                or not isinstance(entry['type'], str)):
            raise LabelError(f'{path}: an entry is not a frozen label')
        qid = entry['question']
        if qid in loaded:
            raise LabelError(f'{qid}: labelled twice')
        if entry['kind'] not in LABELLED:
            raise LabelError(f'{qid}: {entry["kind"]} is not a labelled kind')
        answer = None
        if questions is not None:
            question = questions.get(qid)
            if question is None:
                raise LabelError(f'{qid}: no such question in {dataset.name}')
            if entry['kind'] != trap_kind(question) or entry['type'] != question.type:
                raise LabelError(f'{qid}: the entry is not of the question\'s kind and type')
            answer = question.answer
        if _values(entry, qid, answer) != entry['values']:
            raise LabelError(f'{qid}: values are not as freeze writes them')
        loaded[qid] = {'kind': entry['kind'], 'type': entry['type'], 'values': list(entry['values'])}
    if list(loaded) != sorted(loaded):
        raise LabelError(f'{path}: entries are not in the order freeze writes')
    return loaded


def load_dataset(name, data_dir, offline):
    from benchmarks.public_memory.runner import load_dataset as load
    return load(name, data_dir, offline)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--dataset', required=True, choices=('longmemeval_s', 'beam'))
    common.add_argument('--data-dir', type=Path)
    common.add_argument('--offline', action='store_true')
    drafting = commands.add_parser('draft', parents=[common], help='list the questions to label')
    drafting.add_argument('--question', action='append', help='a question id; repeat for more')
    drafting.add_argument('--subset', type=Path, help='a subset record (datasets subset --out)')
    drafting.add_argument('--out', type=Path, required=True)
    freezing = commands.add_parser('freeze', parents=[common], help='freeze a reviewed draft')
    freezing.add_argument('--draft', type=Path, required=True)
    freezing.add_argument('--out', type=Path, required=True)
    checking = commands.add_parser('check', help='check a frozen file against its pinned hash')
    checking.add_argument('--labels', type=Path, required=True)
    checking.add_argument('--sha256', required=True)
    args = parser.parse_args(argv)
    if args.command == 'check':
        try:
            entries = load(args.labels, args.sha256)
        except LabelError as error:
            print(json.dumps({'ok': False, 'error': str(error)}))
            return 1
        print(json.dumps({'ok': True, 'entries': len(entries)}))
        return 0
    dataset = load_dataset(args.dataset, args.data_dir, args.offline)
    if args.command == 'draft':
        ids = list(args.question or [])
        if args.subset:
            ids += json.loads(Path(args.subset).read_text(encoding='utf-8'))['ids']
        document = draft(dataset, ids)
        args.out.write_text(json.dumps(document, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
        print(json.dumps({'entries': len(document['entries'])}))
        return 0
    data, digest = freeze(json.loads(args.draft.read_text(encoding='utf-8')), dataset)
    args.out.write_bytes(data)
    print(json.dumps({'sha256': digest, 'entries': len(json.loads(data)['entries'])}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
