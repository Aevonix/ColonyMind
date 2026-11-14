"""The publication exporter: a memory benchmark run on a public dataset, as the fields that may be published.

This is its own exporter, apart from the qualification one (``colonymind.qualification.public_export``), which
refuses answers and generated datasets and is never relaxed to publish these. It reads a run's answer and verdict
files, the stores' storage rows and the judge's agreement, and writes one results file (``PROTOCOL``) that the
benchmark page and the evidence chart are generated from (``page`` below, ``scripts/render_evidence_chart.py``).

**Public datasets only.** The declared dataset is one this harness reads, reads by its own judge protocol and pins
(``PUBLIC``: LongMemEval-S and BEAM), and its SHA-256 is the pin. Every row's question is one of its questions, and
of the declared subset, which is drawn again from the data (``subsets.check``). A generated set, a private family, a
file with other bytes, or a question it does not hold is refused.

Every arm at every budget has a row for every question of the declared set (the subset, else the whole file): a
question cannot drop out of a published arm unseen, and a failure is published as a row with its reason.

**Allowlisted fields only.** Nothing is copied through: every published field is built here from a named source
and checked for its type (``ROW_FIELDS``, ``REACH_FIELDS``). A question's type and conversation come from the
dataset, never from a row. The judge's agreement is rebuilt from its counts (each rate and kappa must be the one its
counts give), and its per-type keys must be types of the declared set's questions. Never published: the reader's
answer and reply, the judge's calls and replies, the model id a server reported, the capture (sections, recall
trace, reranker calls), the store's formation record, any address or key.

**Declared names only.** The publisher writes a declaration (``DECLARATION``, a closed set of keys): the dataset,
the subset, the models by public name and pinned weights revision (the reader, the judge, the model that formed the
stores, the embedder and the reranker, with an optional public web page each), the harness commit, the frozen
labels' hash, the bootstrap seed and, where the pre-registration fixes them, the paired contrasts and the character
budgets (the full-context baseline reads the whole history, never a budget, and is published beside them). A key
outside it (an endpoint, a variant, a server) is refused by name, never by value. Every row must have been read by
the declared reader and judged by the declared judge, under the prompts this harness publishes
(``reader.READING_SHA256``, ``judge.PROMPTS_SHA256``) and against the pinned file's own reading of its question
(``judge.question_sha256``), and every arm is one of this harness's own (``runner.ARMS``, ``baselines.ARMS``). An arm
that reads a formed store needs its formation model declared. The frozen labels' hash is published only when the
label file given hashes to it and every verdict read its labels from that file (``judge.label_sha256``); a
knowledge update or a contradiction needs its label.

**Private terms.** A private-terms list names strings that must never be published (endpoint names, variant
labels, hosts: the list stays private and is never committed), one per line, ``#`` starting a comment. A publisher
configures it once: the file ``COLONYMIND_PRIVATE_TERMS`` names, else ``~/.config/colonymind/private-terms`` when it
exists. Every export applies a configured list (``configured_private_terms``), beside any ``--private-terms FILE``
names, and a configured list that cannot be read stops the export. A results file with a term anywhere is refused,
and the refusal names the field, never the term.

Per question, arm and budget a row holds the memory characters and item count the reader was shown, the reader's
prompt tokens, evidence reached (``scoring.reach``, structural for ColonyMind's arms), the reader's ``answerable``,
the verdict (``right``, ``wrong``, ``reader_refused``, ``undecided``, ``skipped``), the BEAM rubric score, the trap
outcome and, for a knowledge update or a contradiction read from a ColonyMind store, how that store holds it
(``storage.py``: every such question an arm answered needs exactly one storage outcome of the store it read).
Beside the rows: per arm and budget the scorer's tables with Wilson intervals per type (``scoring.summarize``), the
storage outcomes, and for BEAM the rubric mean and an interval resampling whole conversations; the paired contrasts
(the exact sign test and a seeded bootstrap, over the same questions in both arms, withheld when any is unscored);
every failure; the judge's agreement with its hand labels.

**The page.** ``docs/PUBLIC-MEMORY-BENCHMARK.md`` is written by hand around generated blocks: the method block (the
pinned datasets, the budgets, the reader's and the judge's own prompts and their hashes, from the code) and the
results block (one section per results file committed in ``RESULTS_DIR``, any name ending ``.json``, so a subset
run and an all-questions run of one dataset are two files, each section linking its own file and, when one is
committed beside it under the same name, its chart; before any, a statement that nothing is demonstrated). ``page``
rewrites both; ``page --check`` exits 1 when the page is not what the code and the committed results give.

    python benchmarks/public_memory/export.py results --declaration publication.json --answers answers.jsonl \\
        --verdicts verdicts.jsonl --storage storage.jsonl --agreement agreement.json --labels old-values.json \\
        --output results.json
    python benchmarks/public_memory/export.py page [--check]
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import ipaddress
import json
import os
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Iterable, Mapping, Optional
from urllib.parse import urlsplit

# Support both package imports and the documented direct script commands.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

HERE = Path(__file__).resolve().parent


from benchmarks.public_memory import baselines, calibration, judge, reader, storage
from benchmarks.public_memory.datasets import subsets
runner, scoring, labels_module = reader.runner, judge.scoring, judge.labels_module


def _adapter(name: str):
    """A dataset adapter imported under its package name."""
    return importlib.import_module(f'benchmarks.public_memory.datasets.{name}')


PROTOCOL = 'public-memory-results-1'
# The datasets a result may be published for: read by this harness's own judge protocol and pinned upstream.
PUBLIC = {name: {'sha256': _adapter(name).PIN.sha256, 'licence': _adapter(name).LICENSE, 'source': _adapter(name).PIN.url}
          for name in sorted(judge.SCHEMES)}
ARMS = frozenset(runner.ARMS) | frozenset(baselines.ARMS)
# The declaration's keys; the required ones, and each model's.
DECLARATION = frozenset({'dataset', 'subset', 'seed', 'harness_commit', 'labels_sha256', 'models', 'contrasts',
                         'budgets'})
REQUIRED = frozenset({'dataset', 'seed', 'harness_commit', 'models'})
MODEL_ROLES = ('reader', 'judge', 'formation', 'embedder', 'reranker')
MODEL_FIELDS = frozenset({'name', 'revision', 'source'})
SUBSET_FIELDS = frozenset({'scheme', 'dataset', 'sha256', 'seed', 'by', 'size', 'strata', 'ids', 'excluded'})
CONTRAST_FIELDS = frozenset({'treatment', 'comparator', 'measure', 'types'})
# What a contrast compares per question: the verdict, evidence reached, or keeping out of the trap (a trap is a
# failure, so a demonstrated contrast on ``no_trap`` means fewer traps).
MEASURES = ('right', 'reached', 'no_trap')
# The plan's primary contrasts: all-on against stock Hermes' session search, all-on against the store with no claims
# on knowledge updates and temporal questions, and the traps against stock Hermes. A declaration's ``contrasts``
# (the pre-registration's) replaces them.
PRIMARY = ({'treatment': 'all-on', 'comparator': 'session-search', 'measure': 'right', 'types': None},
           {'treatment': 'all-on', 'comparator': 'import-only', 'measure': 'right',
            'types': ['knowledge-update', 'temporal-reasoning']},
           {'treatment': 'all-on', 'comparator': 'session-search', 'measure': 'no_trap', 'types': None})
ROW_FIELDS = ('question', 'conversation', 'type', 'kind', 'arm', 'budget_chars', 'memory_chars', 'items',
              'reader_prompt_tokens', 'reach', 'answerable', 'verdict', 'rubric_score', 'trap', 'storage')
REACH_FIELDS = ('basis', 'weaker', 'gold', 'reached', 'any', 'all', 'first_rank')
AGREEMENT_FIELDS = ('labelled', 'unlabelled', 'agreement', 'kappa', 'confusion', 'types')
CONFUSION = ('judge_yes_label_yes', 'judge_yes_label_no', 'judge_no_label_yes', 'judge_no_label_no')
WILSON = ('successes', 'trials', 'rate', 'lower', 'upper')
RIGHT, WRONG, REFUSED, UNDECIDED, SKIPPED = 'right', 'wrong', 'reader_refused', 'undecided', 'skipped'
# The full-context baseline reads the whole history, not a budget: its rows have no budget, its tables this key, and
# it sorts after every budget. A declaration's budgets are the other arms' character budgets.
WHOLE = 'whole'
WHOLE_HISTORY = ('full-context',)
_HEX = re.compile(r'[0-9a-f]+')


class ExportError(ValueError):
    """Something a results file may not publish, or cannot be built from. The message never repeats a value that
    could be private."""


# ------------------------------------------------------------------ the declaration

def _closed(value: Any, allowed: Iterable[str], where: str) -> dict:
    if not isinstance(value, Mapping):
        raise ExportError(f'{where} is an object')
    extra = sorted(set(map(str, value)) - set(allowed))
    if extra:
        raise ExportError(f'{where} may not name {", ".join(extra)}: only {", ".join(sorted(allowed))} are published')
    return dict(value)


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExportError(f'{where} is a nonempty string')
    return value


def _source(value: Any, where: str) -> str:
    """A public web page for the weights: https, a named host, no port, credentials, query or fragment."""
    parts = urlsplit(_text(value, where))
    host = parts.hostname or ''
    try:
        ipaddress.ip_address(host)
        literal = True
    except ValueError:
        literal = False
    try:
        port = parts.port
    except ValueError:
        port = -1
    if (parts.scheme != 'https' or not host or '.' not in host or literal or port is not None or parts.username
            or parts.password or parts.query or parts.fragment):
        raise ExportError(f'{where} is a public https page for the weights (a named host, no port, credentials, '
                          'query or fragment)')
    return value


def _model(value: Any, where: str) -> Optional[dict]:
    if value is None:
        return None
    body = _closed(value, MODEL_FIELDS, where)
    model = {'name': _text(body.get('name'), f'{where}.name'),
             'revision': _text(body.get('revision'), f'{where}.revision')}
    if body.get('source') is not None:
        model['source'] = _source(body['source'], f'{where}.source')
    return model


def _hex(value: Any, where: str, lengths) -> str:
    if not isinstance(value, str) or len(value) not in lengths or not _HEX.fullmatch(value):
        raise ExportError(f'{where} is a lower-case hexadecimal hash')
    return value


def _contrast(value: Any, where: str) -> dict:
    body = _closed(value, CONTRAST_FIELDS, where)
    for name in ('treatment', 'comparator'):
        if body.get(name) not in ARMS:
            raise ExportError(f'{where}.{name} is one of this harness\'s arms')
    if body.get('measure') not in MEASURES:
        raise ExportError(f'{where}.measure is one of {", ".join(MEASURES)}')
    types = body.get('types')
    if types is not None and (not isinstance(types, list) or not all(isinstance(item, str) for item in types)):
        raise ExportError(f'{where}.types is a list of question types, or null')
    return {'treatment': body['treatment'], 'comparator': body['comparator'], 'measure': body['measure'],
            'types': None if types is None else sorted(_type(item) for item in types)}


def read_declaration(value: Any, dataset) -> dict:
    """The declaration, checked: a closed set of keys, a public dataset, declared names only."""
    body = _closed(value, DECLARATION, 'the declaration')
    missing = sorted(REQUIRED - set(body))
    if missing:
        raise ExportError(f'the declaration needs {", ".join(missing)}')
    name = body['dataset']
    if name not in PUBLIC or dataset.name != name:
        raise ExportError('the dataset is not one this harness publishes: public datasets only '
                          f'({", ".join(PUBLIC)})')
    if dataset.sha256 != PUBLIC[name]['sha256']:
        raise ExportError(f'the {name} file read is not the pinned public file (SHA-256 {PUBLIC[name]["sha256"]})')
    if type(body['seed']) is not int:
        raise ExportError('the seed is a whole number')
    models = _closed(body['models'], MODEL_ROLES, 'models')
    for role in ('reader', 'judge'):
        if models.get(role) is None:
            raise ExportError(f'models.{role} is declared')
    subset = body.get('subset')
    if subset is not None:
        subset = _closed(subset, SUBSET_FIELDS, 'subset')
        # Published as declared, so each question it left out before its draw must be one of the dataset's.
        questions = {question.id for question in dataset.questions}
        excluded = subset.get('excluded')
        if not isinstance(excluded, list) or not all(isinstance(item, str) and item in questions
                                                     for item in excluded):
            raise ExportError(f'subset.excluded lists questions of the {name} file only')
        try:
            subsets.check(subset, dataset)
        except (subsets.DatasetError, KeyError, TypeError) as error:
            raise ExportError(f'the subset is not one drawn from {name}: {error}') from None
    budgets = body.get('budgets')
    if budgets is not None and (not isinstance(budgets, list) or not all(type(item) is int for item in budgets)):
        raise ExportError('budgets is a list of whole numbers of characters')
    contrasts = body.get('contrasts')
    if contrasts is not None and not isinstance(contrasts, list):
        raise ExportError('contrasts is a list')
    return {'dataset': name, 'subset': subset, 'seed': body['seed'],
            'harness_commit': _hex(body['harness_commit'], 'harness_commit', range(7, 41)),
            'labels_sha256': None if body.get('labels_sha256') is None
            else _hex(body['labels_sha256'], 'labels_sha256', (64,)),
            'models': {role: _model(models.get(role), f'models.{role}') for role in MODEL_ROLES},
            'budgets': None if budgets is None else sorted(set(budgets)),
            'contrasts': [_contrast(item, f'contrasts[{index}]') for index, item in enumerate(contrasts)]
            if contrasts is not None else [dict(item, types=None if item['types'] is None else list(item['types']))
                                           for item in PRIMARY]}


# ------------------------------------------------------------------ the rows

def _type(value: str) -> str:
    return str(value).replace('_', '-')


def _order(budget: Optional[int]) -> tuple:
    return (budget is None, budget or 0)


def _budget_key(budget: Optional[int]) -> str:
    return WHOLE if budget is None else str(budget)


def _jsonl(paths: Iterable[Path]) -> list:
    rows = []
    for path in paths:
        rows += [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    return rows


def _identity(value: Any) -> Optional[tuple]:
    return (value.get('name'), value.get('revision')) if isinstance(value, Mapping) else None


def _declared(model: Mapping[str, Any]) -> tuple:
    return (model['name'], model['revision'])


def _where(row: Mapping[str, Any]) -> str:
    return f'{row.get("question")} (budget {row.get("budget_chars")})'


def _check_rows(declared: Mapping[str, Any], dataset, answers: list, verdicts: list) -> None:
    """Refuse a row from outside the harness's arms, the dataset or subset, the declared models, or its prompts."""
    readings = {question.id: judge.question_sha256(question) for question in dataset.questions}
    questions = set(readings)
    wanted = set(declared['subset']['ids']) if declared['subset'] else questions
    for row in [*answers, *verdicts]:
        if row.get('arm') not in ARMS:
            raise ExportError(f'{_where(row)}: the row\'s arm is not one of this harness\'s arms')
        if row.get('question') not in questions or row.get('question') not in wanted:
            raise ExportError(f'{row.get("question")} is not a question of the declared {dataset.name} '
                              f'{"subset" if declared["subset"] else "file"}')
        budget = row.get('budget_chars')
        if row['arm'] in WHOLE_HISTORY:
            if budget is not None:
                raise ExportError(f'{_where(row)}: {row["arm"]} reads the whole history, never a budget')
        elif type(budget) is not int:
            raise ExportError(f'{_where(row)}: only {", ".join(WHOLE_HISTORY)} reads the whole history; every other '
                              'arm reads a budget')
        elif declared['budgets'] is not None and budget not in declared['budgets']:
            raise ExportError(f'{_where(row)}: the budget is not a declared one')
    for row in answers:
        if row.get('protocol') != reader.PROTOCOL or row.get('reading_sha256') != reader.READING_SHA256:
            raise ExportError(f'{_where(row)} was read with other prompts or settings than this harness publishes')
        if _identity(row.get('reader')) != _declared(declared['models']['reader']):
            raise ExportError(f'{_where(row)} was read by another reader than the declared one')
    scheme, votes = judge.SCHEMES[dataset.name], {row.get('votes') for row in verdicts}
    for row in verdicts:
        if row.get('protocol') != judge.PROTOCOL or row.get('prompts_sha256') != judge.PROMPTS_SHA256:
            raise ExportError(f'{_where(row)} was judged with other prompts than this harness publishes')
        if row.get('scheme') != scheme:
            raise ExportError(f'{_where(row)} was judged under another scheme than {dataset.name}\'s')
        if row.get('question_sha256') != readings[row['question']]:
            raise ExportError(f'{_where(row)} was judged against another reading of its question (words, reference '
                              f'answer, type or rubric) than the pinned {dataset.name} file gives')
        if _identity(row.get('judge')) != _declared(declared['models']['judge']):
            raise ExportError(f'{_where(row)} was judged by another judge than the declared one')
        if _identity(row.get('reader')) != _declared(declared['models']['reader']):
            raise ExportError(f'{_where(row)} was read by another reader than the declared one')
    if len(votes) > 1:
        raise ExportError('the verdicts were not all given with one number of votes')
    # Published as the judge's number of votes: one of the counts it gives its items, nothing else.
    if any(type(value) is not int or value not in judge.VOTES for value in votes):
        raise ExportError('the verdicts\' votes are not a number of votes the judge gives '
                          f'({" or ".join(map(str, judge.VOTES))})')
    covered: dict = {}
    for row in verdicts:
        covered.setdefault((row['arm'], row.get('budget_chars')), set()).add(row['question'])
    for (arm, budget), found in sorted(covered.items(), key=lambda item: (item[0][0], _order(item[0][1]))):
        left_out = sorted(wanted - found)
        if left_out:
            raise ExportError(f'{arm} at {_budget_label(budget)} has no row for {len(left_out)} question(s) of the '
                              f'declared set ({", ".join(left_out[:5])}): every question is published, a failure '
                              'as a row with its reason')
    stores = {runner.ARMS[row['arm']]['store'] for row in verdicts if row['arm'] in runner.ARMS}
    if stores & {'formed', 'before_night'} and declared['models']['formation'] is None:
        raise ExportError('an arm reads a store formed by a model: models.formation is declared')


def _check_labels(declared: Mapping[str, Any], dataset, labels: Optional[Path], verdicts: list) -> None:
    """Refuse verdicts whose traps were not judged against the frozen labels the declaration pins: the label file
    must hash to ``labels_sha256`` and be the one ``labels.py freeze`` wrote for this dataset, every verdict must
    name the label it read from that file (``judge.label_sha256``), and a knowledge update or a contradiction the
    reader answered needs its label."""
    pinned = declared['labels_sha256']
    if pinned is None and labels is not None:
        raise ExportError('a label file is given but the declaration pins none: declare its labels_sha256')
    if pinned is not None and labels is None:
        raise ExportError('the declaration pins labels_sha256: give the frozen label file it pins (--labels)')
    try:
        loaded = labels_module.load(labels, pinned, dataset) if labels is not None else {}
    except labels_module.LabelError:
        raise ExportError(f'the label file is not the frozen {dataset.name} label file labels_sha256 pins '
                          '(labels.py freeze)') from None
    questions = {question.id: question for question in dataset.questions}
    for row in verdicts:
        question = questions[row['question']]
        if (labels_module.trap_kind(question) in labels_module.LABELLED and not row.get('skipped')
                and question.id not in loaded):
            raise ExportError(f'{_where(row)}: a {labels_module.trap_kind(question)} question needs its frozen label '
                              '(labels_sha256 and --labels)')
        if row.get('label_sha256') != judge.label_sha256(loaded, question, row):
            raise ExportError(f'{_where(row)} was judged against another frozen label than the declared file holds')


def _keyed(rows: list, what: str) -> dict:
    keyed: dict = {}
    for row in rows:
        key = reader.key(row)
        if key in keyed and keyed[key] != row:
            raise ExportError(f'{_where(row)} ({what}) appears twice with different contents')
        keyed[key] = row
    return keyed


def _verdict(verdict: Mapping[str, Any]) -> str:
    if verdict.get('skipped'):
        return SKIPPED
    if verdict.get('reader_refused'):
        return REFUSED
    right = verdict.get('right')
    return UNDECIDED if right is None else RIGHT if right is True else WRONG


def _trap(verdict: Mapping[str, Any]) -> Optional[str]:
    trap = verdict.get('trap')
    if not isinstance(trap, Mapping):
        return None
    if trap.get('undecided'):
        return UNDECIDED
    return trap.get('outcome') if trap.get('outcome') in scoring.OUTCOMES else None


def _count(value: Any) -> Optional[int]:
    return value if type(value) is int and value >= 0 else None


def _reach(score: Mapping[str, Any]) -> Optional[dict]:
    reach = score.get('reach') if isinstance(score, Mapping) else None
    if not isinstance(reach, Mapping) or reach.get('basis') not in (scoring.STRUCTURAL, scoring.SESSION,
                                                                    scoring.SPAN):
        return None
    flags = {name: reach.get(name) if type(reach.get(name)) is bool else None for name in ('weaker', 'any', 'all')}
    return {'basis': reach['basis'], 'weaker': bool(flags['weaker']), 'gold': _count(reach.get('gold')),
            'reached': _count(reach.get('reached')), 'any': flags['any'], 'all': flags['all'],
            'first_rank': _count(reach.get('first_rank'))}


def _row(question, answer: Mapping[str, Any], verdict: Mapping[str, Any], storage_rows: Mapping) -> dict:
    kind = labels_module.trap_kind(question)
    score = answer.get('score') if isinstance(answer.get('score'), Mapping) else {}
    usage = answer.get('usage') if isinstance(answer.get('usage'), Mapping) else {}
    outcome = _verdict(verdict)
    store = runner.ARMS[answer['arm']]['store'] if answer['arm'] in runner.ARMS else None
    stored = storage_rows.get((question.id, store)) if store and kind in labels_module.LABELLED else None
    rubric = verdict.get('rubric_score')
    return {'question': question.id, 'conversation': question.conversation, 'type': question.type, 'kind': kind,
            'arm': answer['arm'], 'budget_chars': _count(answer.get('budget_chars')),
            'memory_chars': _count(score.get('chars')), 'items': _count(score.get('items')),
            'reader_prompt_tokens': _count(usage.get('prompt_tokens')),
            'reach': None if outcome == SKIPPED else _reach(score),
            'answerable': answer.get('answerable') if type(answer.get('answerable')) is bool else None,
            'verdict': outcome,
            'rubric_score': float(rubric) if type(rubric) in (int, float) else None,
            'trap': _trap(verdict), 'storage': stored}


def _storage(rows: list) -> dict:
    """``{(question, store): outcome}``: one outcome per question and store. The same row given twice is one row;
    two that differ are refused."""
    keyed, seen = {}, {}
    for row in rows:
        if row.get('protocol') != storage.PROTOCOL or row.get('outcome') not in storage.OUTCOMES:
            raise ExportError(f'{row.get("question")}: not a storage row')
        key = (row.get('question'), row.get('store'))
        if key in seen and seen[key] != row:
            raise ExportError(f'{key[0]}: the {key[1]} store has two storage rows that differ: give one storage.py '
                              'reading of each store')
        seen[key], keyed[key] = row, row['outcome']
    return keyed


def _check_storage(answered: Mapping, questions: Mapping, storage_rows: Mapping) -> None:
    """Refuse a run whose storage outcomes are missing: every knowledge update or contradiction an arm answered
    from a store needs that store's storage row (``storage.py``)."""
    for (qid, arm, budget), answer in sorted(answered.items(), key=lambda item: (item[0][1], _order(item[0][2]),
                                                                                 item[0][0])):
        store = runner.ARMS[arm]['store'] if arm in runner.ARMS else None
        if (store is None or answer.get('skipped')
                or labels_module.trap_kind(questions[qid]) not in labels_module.LABELLED):
            continue
        if (qid, store) not in storage_rows:
            raise ExportError(f'{qid} ({arm}, {_budget_label(budget)}) was read from the {store} store, which has no '
                              'storage row: give the storage.py reading of every store the run read (--storage)')


# ------------------------------------------------------------------ tables, contrasts, failures

def _scored(row: Mapping[str, Any]) -> dict:
    scored = {'question': row['question'], 'conversation': row['conversation'], 'type': row['type'],
              'reach': row['reach'], 'right': {RIGHT: True, WRONG: False, REFUSED: False}.get(row['verdict']),
              'chars': row['memory_chars'], 'budget': row['budget_chars'], 'tokens': row['reader_prompt_tokens']}
    if row['trap'] in scoring.OUTCOMES:
        scored['trap'] = {'outcome': row['trap']}
    return scored


def _tables(rows: list, dataset: str, seed: int) -> dict:
    tables: dict = {}
    for arm in sorted({row['arm'] for row in rows}):
        for budget in sorted({row['budget_chars'] for row in rows if row['arm'] == arm}, key=_order):
            members = [row for row in rows if row['arm'] == arm and row['budget_chars'] == budget]
            table = scoring.summarize([_scored(row) for row in members])
            table['storage'] = dict(sorted(Counter(row['storage'] for row in members if row['storage']).items()))
            if dataset == 'beam':
                scores = [row['rubric_score'] for row in members if row['rubric_score'] is not None]
                table['rubric'] = {'mean': statistics.fmean(scores), 'scored': len(scores)} if scores else None
                decided = [_scored(row) for row in members if row['verdict'] in (RIGHT, WRONG, REFUSED)]
                table['clustered'] = (scoring.interval(decided, key='right', seed=seed, cluster_key='conversation')
                                      if decided else None)
            tables.setdefault(arm, {})[_budget_key(budget)] = table
    return tables


def _value(row: Mapping[str, Any], measure: str) -> Optional[bool]:
    if measure == 'right':
        return {RIGHT: True, WRONG: False, REFUSED: False}.get(row['verdict'])
    if measure == 'reached':
        return (row['reach'] or {}).get('any')
    return None if row['trap'] not in scoring.OUTCOMES else row['trap'] != scoring.TRAP


def _contrasts(rows: list, declared: list, seed: int) -> list:
    results = []
    for item in declared:
        types = None if item['types'] is None else set(item['types'])

        def members(arm, budget):
            return [row for row in rows if row['arm'] == arm and row['budget_chars'] == budget
                    and (types is None or _type(row['type']) in types)
                    and (item['measure'] != 'no_trap' or row['kind'] != labels_module.RECALL)]
        budgets = sorted({row['budget_chars'] for row in rows if row['arm'] == item['treatment']}
                         & {row['budget_chars'] for row in rows if row['arm'] == item['comparator']}, key=_order)
        for budget in budgets:
            first, second = members(item['treatment'], budget), members(item['comparator'], budget)
            if not first and not second:
                continue
            head = {**item, 'budget_chars': budget}
            only = ({row['question'] for row in first} - {row['question'] for row in second},
                    {row['question'] for row in second} - {row['question'] for row in first})
            if any(only):
                results.append({**head, 'verdict': 'unpaired', 'treatment_only': len(only[0]),
                                'comparator_only': len(only[1])})
                continue

            def paired(group):
                return [{'question': row['question'], 'conversation': row['conversation'],
                         'value': _value(row, item['measure'])} for row in group]
            results.append({**head, **scoring.contrast(paired(first), paired(second), key='value', seed=seed,
                                                       cluster_key='conversation')})
    return results


def _failures(rows: list, answers: Mapping) -> list:
    found = []
    for row in rows:
        failure = {SKIPPED: None, REFUSED: 'reader_refused', UNDECIDED: 'judge_undecided'}.get(row['verdict'], '')
        if failure is None:
            skipped = answers[(row['question'], row['arm'], row['budget_chars'])].get('skipped')
            failure = skipped if skipped in ('capture_failed', 'does_not_fit') else 'skipped'
        if failure:
            found.append({'arm': row['arm'], 'budget_chars': row['budget_chars'], 'question': row['question'],
                          'failure': failure})
        if row['trap'] == UNDECIDED:
            found.append({'arm': row['arm'], 'budget_chars': row['budget_chars'], 'question': row['question'],
                          'failure': 'trap_undecided'})
    return sorted(found, key=lambda item: (item['arm'], _order(item['budget_chars']), item['question'],
                                           item['failure']))


def _fields(value: Any, allowed: Iterable[str], where: str) -> Mapping[str, Any]:
    """``value`` as an object holding only ``allowed`` keys. Unlike the declaration's keys, a key here was never
    chosen to be published, so the refusal does not repeat it."""
    if not isinstance(value, Mapping):
        raise ExportError(f'{where} is an object')
    if not set(value) <= set(allowed):
        raise ExportError(f'{where} holds a field that is not published: only {", ".join(sorted(allowed))} are')
    return value


def _whole(value: Any, where: str) -> int:
    if type(value) is not int or value < 0:
        raise ExportError(f'{where} is a whole number')
    return value


def _wilson(value: Any, successes: int, trials: int, where: str) -> Optional[dict]:
    """The Wilson interval of ``successes`` in ``trials``, built here; ``value`` must be exactly that."""
    expected = scoring.wilson(successes, trials)
    if isinstance(value, Mapping):
        _fields(value, WILSON, where)
    if value != expected:
        raise ExportError(f'{where} is not the Wilson interval of its counts')
    return expected


def _agreement(value: Optional[Mapping[str, Any]], verdict_files: Iterable[Path],
               types: Iterable[str]) -> Optional[dict]:
    """The judge's agreement, rebuilt from its counts: every number is checked against the counts it comes from and
    written here, and a type is one of the declared set's questions' types. Nothing in the report is copied."""
    if value is None:
        return None
    if not isinstance(value, Mapping) or value.get('protocol') != calibration.PROTOCOL:
        raise ExportError('the agreement is not a judge calibration report (calibration.py agreement)')
    body = _fields(value, {'protocol', 'verdicts_sha256', *AGREEMENT_FIELDS}, 'the agreement')
    drawn = {hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in verdict_files}
    if body.get('verdicts_sha256') not in drawn:
        raise ExportError('the agreement was drawn from another verdict file than the ones published')
    labelled = _whole(body.get('labelled'), 'agreement.labelled')
    confusion = _fields(body.get('confusion'), CONFUSION, 'agreement.confusion')
    table = {name: _whole(confusion.get(name), f'agreement.confusion.{name}') for name in CONFUSION}
    if sum(table.values()) != labelled:
        raise ExportError('agreement.confusion does not add up to agreement.labelled')
    agree = table['judge_yes_label_yes'] + table['judge_no_label_no']
    kappa = calibration._kappa(table)
    if type(body.get('kappa')) not in (int, float, type(None)) or body.get('kappa') != kappa:
        raise ExportError('agreement.kappa is not the kappa of agreement.confusion')
    known = set(types)
    per_type = _fields(body.get('types'), known, 'agreement.types')
    rebuilt, counted, agreed = {}, 0, 0
    for name in sorted(per_type):
        item = _fields(per_type[name], ('labelled', 'agreement'), f'agreement.types.{name}')
        members = _whole(item.get('labelled'), f'agreement.types.{name}.labelled')
        rate = item.get('agreement')
        hits = _whole(rate.get('successes') if isinstance(rate, Mapping) else 0,
                      f'agreement.types.{name}.agreement.successes')
        if hits > members:
            raise ExportError(f'agreement.types.{name} agrees more often than it was labelled')
        where = f'agreement.types.{name}.agreement'
        rebuilt[name] = {'labelled': members, 'agreement': _wilson(item.get('agreement'), hits, members, where)}
        counted, agreed = counted + members, agreed + hits
    if (counted, agreed) != (labelled, agree):
        raise ExportError('agreement.types does not add up to agreement.labelled and its agreement')
    return {'labelled': labelled, 'unlabelled': _whole(body.get('unlabelled'), 'agreement.unlabelled'),
            'agreement': _wilson(body.get('agreement'), agree, labelled, 'agreement.agreement'), 'kappa': kappa,
            'confusion': table, 'types': rebuilt}


# ------------------------------------------------------------------ private terms

def _strings(value: Any, path: str = ''):
    if isinstance(value, Mapping):
        for key, item in value.items():
            where = f'{path}.{key}' if path else str(key)
            yield where, str(key)
            yield from _strings(item, where)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _strings(item, f'{path}[{index}]')
    elif isinstance(value, str):
        yield path, value


PRIVATE_TERMS_ENV = 'COLONYMIND_PRIVATE_TERMS'
DEFAULT_PRIVATE_TERMS = Path('~/.config/colonymind/private-terms')


def read_terms(path: Path) -> list:
    """The terms of a private-terms list: one per line, blank lines and ``#`` comments skipped."""
    lines = (line.strip() for line in Path(path).read_text(encoding='utf-8').splitlines())
    return [line for line in lines if line and not line.startswith('#')]


def configured_private_terms() -> list:
    """The publisher's configured private-terms list (the file ``COLONYMIND_PRIVATE_TERMS`` names, else the default
    file when it exists); none configured, no terms. A configured list that cannot be read is an ``ExportError`` that
    names neither the file nor a term: an export never goes ahead without the list its publisher keeps."""
    named = os.environ.get(PRIVATE_TERMS_ENV, '').strip()
    path = Path(named).expanduser() if named else DEFAULT_PRIVATE_TERMS.expanduser()
    if not named and not path.exists():
        return []
    try:
        return read_terms(path)
    except (OSError, UnicodeDecodeError) as error:
        where = PRIVATE_TERMS_ENV if named else 'the default location'
        raise ExportError(f'the configured private-terms list ({where}) cannot be read ({type(error).__name__}): '
                          'nothing is written') from None


def refuse_private_terms(results: Mapping[str, Any], terms: Iterable[str]) -> None:
    """Refuse ``results`` when a published key or string holds one of ``terms`` (any case); the refusal names the
    field alone."""
    terms = [term.casefold() for term in (str(item).strip() for item in terms) if term]
    for where, text in _strings(results):
        folded = text.casefold()
        if any(term in folded for term in terms):
            raise ExportError(f'{where} holds a private term: nothing is written')


# ------------------------------------------------------------------ the results

def build(declaration: Mapping[str, Any], dataset, *, answers: Iterable[Path], verdicts: Iterable[Path],
          storage: Iterable[Path] = (), agreement: Optional[Mapping[str, Any]] = None, labels: Optional[Path] = None,
          private_terms: Iterable[str] = ()) -> dict:
    """The results document for one run (module docstring), or ``ExportError``. ``labels`` is the frozen label
    file the declaration's ``labels_sha256`` pins; ``private_terms`` join the configured list, which always applies."""
    configured = configured_private_terms()
    declared = read_declaration(declaration, dataset)
    verdict_files = list(verdicts)
    answer_rows, verdict_rows = _jsonl(answers), _jsonl(verdict_files)
    _check_rows(declared, dataset, answer_rows, verdict_rows)
    _check_labels(declared, dataset, labels, verdict_rows)
    answered, judged = _keyed(answer_rows, 'answer'), _keyed(verdict_rows, 'verdict')
    missing = sorted(set(answered) - set(judged), key=str)
    if missing:
        raise ExportError(f'{missing[0][0]} ({missing[0][1]}, budget {missing[0][2]}) has no verdict: judge every '
                          'answer row')
    questions = {question.id: question for question in dataset.questions}
    wanted = set(declared['subset']['ids']) if declared['subset'] else set(questions)
    storage_rows = _storage(_jsonl(storage))
    _check_storage(answered, questions, storage_rows)
    rows = []
    for key, verdict in sorted(judged.items(), key=lambda item: (item[0][1], _order(item[0][2]), item[0][0])):
        answer = answered.get(key)
        if answer is None or judge._digest(dict(answer)) != verdict.get('answer_sha256'):
            raise ExportError(f'{_where(verdict)} ({key[1]}) has no answer row it was given on')
        rows.append(_row(questions[key[0]], answer, verdict, storage_rows))
    results = {'protocol': PROTOCOL,
               'dataset': {'name': dataset.name, 'sha256': dataset.sha256, 'licence': PUBLIC[dataset.name]['licence'],
                           'source': PUBLIC[dataset.name]['source'], 'questions': len(dataset.questions)},
               'subset': declared['subset'], 'questions': sorted({row['question'] for row in rows}),
               'models': declared['models'], 'harness_commit': declared['harness_commit'],
               'labels_sha256': declared['labels_sha256'], 'seed': declared['seed'],
               'budgets': sorted({row['budget_chars'] for row in rows}, key=_order),
               'arms': sorted({row['arm'] for row in rows}),
               'reader': {'protocol': reader.PROTOCOL, 'prompt_sha256': reader.READING_SHA256},
               'judge': {'protocol': judge.PROTOCOL, 'scheme': judge.SCHEMES[dataset.name],
                         'votes': next(iter({row['votes'] for row in verdict_rows}), None),
                         'prompts_sha256': judge.PROMPTS_SHA256},
               'rows': rows, 'tables': _tables(rows, dataset.name, declared['seed']),
               'contrasts': _contrasts(rows, declared['contrasts'], declared['seed']),
               'failures': _failures(rows, answered),
               'agreement': _agreement(agreement, verdict_files, {questions[qid].type for qid in wanted})}
    refuse_private_terms(results, [*configured, *private_terms])
    return results


def dumps(results: Mapping[str, Any]) -> str:
    """The results file's bytes: the same document always writes the same text."""
    return json.dumps(results, sort_keys=True, indent=1, ensure_ascii=False) + '\n'


# ------------------------------------------------------------------ the page

RESULTS_DIR = HERE / 'results'
PAGE = HERE.parents[1] / 'docs' / 'PUBLIC-MEMORY-BENCHMARK.md'
# How the page links a committed results file or chart: from ``docs/`` to ``RESULTS_DIR``.
PAGE_LINK = '../benchmarks/public_memory/results/'
TITLES = {'longmemeval_s': 'LongMemEval-S', 'beam': 'BEAM, 100K split'}
_BLOCK = re.compile(r'(<!-- generated: (\w+) -->\n)(.*?)(<!-- end generated: \2 -->)', re.DOTALL)
WITHHELD = {'unavailable': 'withheld: a question without a verdict in either arm',
            'unpaired': 'not paired: the arms read different questions',
            'demonstrated': 'demonstrated', 'not_demonstrated': 'not demonstrated'}


def _budget_text(budget: Optional[int]) -> str:
    return 'whole history' if budget is None else f'{budget:,}'


def _budget_label(budget: Optional[int]) -> str:
    return 'whole history' if budget is None else f'{budget:,} characters'


def _pct(value: Optional[float]) -> str:
    return 'n/a' if value is None else f'{100 * value:.1f}'


def _rate(item: Optional[Mapping[str, Any]], *, counts: bool = True) -> str:
    """A Wilson rate as the page prints it: ``54.2% [45.3, 62.8] (65/120)``."""
    if not item:
        return 'n/a'
    text = f'{_pct(item["rate"])}% [{_pct(item["lower"])}, {_pct(item["upper"])}]'
    return f'{text} ({item["successes"]}/{item["trials"]})' if counts else text


def _grid(results: Mapping[str, Any], cell) -> list:
    """A table with one row per arm and one column per budget."""
    budgets = results['budgets']
    lines = ['| Arm | ' + ' | '.join(_budget_label(budget) for budget in budgets) + ' |',
             '| --- |' + ' --- |' * len(budgets)]
    for arm in results['arms']:
        tables = results['tables'].get(arm, {})
        lines.append(f'| `{arm}` | ' + ' | '.join(cell(tables[_budget_key(budget)]) if _budget_key(budget) in tables
                                                  else 'not run' for budget in budgets) + ' |')
    return lines


def _traps(table: Mapping[str, Any]) -> str:
    rate, traps = table['overall'].get('trap_rate'), table['overall'].get('traps') or {}
    if not rate:
        return 'none judged'
    return f'{traps.get("trap", 0)} of {rate["trials"]}: {_rate(rate, counts=False)}; asked {traps.get("asked", 0)}'


def _contrast_line(item: Mapping[str, Any]) -> str:
    types = ', '.join(f'`{name}`' for name in item['types']) if item['types'] else 'all'
    head = (f'| `{item["treatment"]}` | `{item["comparator"]}` | {item["measure"]} | {types} | '
            f'{_budget_text(item["budget_chars"])} |')
    if item['verdict'] in ('unavailable', 'unpaired'):
        return head + ' | | | | ' + WITHHELD[item['verdict']] + ' |'
    ci = item['ci_pp']
    return (head + f' {item["units"]} | {item["wins"]} / {item["losses"]} / {item["ties"]} | '
            f'{item["delta_pp"]:+.1f} pp [{ci["lower"]:.1f}, {ci["upper"]:.1f}] | '
            f'p = {item["sign_test"]["p_two_sided"]:.3f} | {WITHHELD[item["verdict"]]} |')


def _model_line(role: str, model: Optional[Mapping[str, Any]]) -> str:
    if model is None:
        return f'| {role} | not used | | |'
    page = f'[page]({model["source"]})' if model.get('source') else ''
    return f'| {role} | `{model["name"]}` | `{model["revision"]}` | {page} |'


def render_run(results: Mapping[str, Any], *, chart: bool, stem: Optional[str] = None) -> str:
    """One results file's section of the page; ``stem`` is the committed file's name without ``.json`` (the
    dataset's name by default), which its links and its chart's share."""
    if results.get('protocol') != PROTOCOL:
        raise ExportError('not a results file (export.py results)')
    data, name = results['dataset'], results['dataset']['name']
    stem = stem or name
    subset = results.get('subset')
    scope = (f'a subset of {len(subset["ids"])} questions (seed {subset["seed"]})' if subset
             else f'all {data["questions"]} questions')
    lines = [f'### {TITLES.get(name, name)}, {scope}', '',
             f'- Dataset: `{name}`, SHA-256 `{data["sha256"]}`, licence {data["licence"]}, '
             f'[the pinned upstream file]({data["source"]}). The run read {len(results["questions"])} of its '
             f'{data["questions"]} questions (listed at the end of this section).',
             (f'- Subset: seed `{subset["seed"]}`, stratified by {subset["by"]} '
              f'({", ".join(f"{key} {value}" for key, value in sorted(subset["strata"].items()))}), drawn by '
              f'{subset["scheme"]}.') if subset else '- Subset: none declared; the rows name every question read.',
             f'- Harness commit `{results["harness_commit"]}`; frozen trap labels SHA-256 '
             f'`{results["labels_sha256"] or "none"}`; bootstrap seed {results["seed"]}.',
             f'- Reader prompt SHA-256 `{results["reader"]["prompt_sha256"]}`; judge prompts SHA-256 '
             f'`{results["judge"]["prompts_sha256"]}`, {results["judge"]["votes"]} vote(s) per item, scheme '
             f'`{results["judge"]["scheme"]}`.',
             f'- Budgets: {", ".join(_budget_label(budget) for budget in results["budgets"])}. Results file: '
             f'[`{stem}.json`]({PAGE_LINK}{stem}.json).', '',
             '| Role | Public name | Revision | Page |', '| --- | --- | --- | --- |',
             *(_model_line(role, results['models'].get(role)) for role in MODEL_ROLES), '']
    if chart:
        lines += [f'![Accuracy, evidence reached and traps by memory budget, per arm]({PAGE_LINK}{stem}.svg)', '']
    lines += ['#### Accuracy', '', 'Right answers over the answers with a verdict, with a Wilson interval; every '
              'answer without one is listed under failures.', '',
              *_grid(results, lambda table: _rate(table['overall']['right'])), '',
              '#### Evidence reached', '', 'Whether what the arm showed the reader cites a gold evidence message '
              "(structural for ColonyMind's arms; a baseline reaches when it shows a gold session).", '',
              *_grid(results, lambda table: _rate(table['overall']['reach_any'])), '',
              '#### Traps', '', 'An old value asserted as current, one side of a contradiction chosen, or a specific '
              'answer given to a question nothing answers; counted apart from accuracy, never averaged into it.', '',
              *_grid(results, _traps), '']
    for budget in results['budgets']:
        types = sorted({row['type'] for row in results['rows'] if row['budget_chars'] == budget})
        arms = [arm for arm in results['arms'] if _budget_key(budget) in results['tables'].get(arm, {})]
        lines += [f'#### Accuracy by question type, {_budget_label(budget)}', '',
                  'Right answers over the answers with a verdict, with a Wilson interval.', '',
                  '| Type | ' + ' | '.join(f'`{arm}`' for arm in arms) + ' |', '| --- |' + ' --- |' * len(arms)]
        for kind in types:
            cells = [_rate((results['tables'][arm][_budget_key(budget)]['types'].get(kind) or {}).get('right'))
                     for arm in arms]
            lines.append(f'| `{kind}` | ' + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['#### How each store held a changed value', '', 'For knowledge updates and contradictions read from a '
              'ColonyMind store: an old value replaced (superseded), both kept and the owner asked (conflict), both '
              'held as current (both_current), one value (single_value), or nothing stored.', '',
              *_grid(results, lambda table: ', '.join(f'{key} {value}' for key, value in table['storage'].items())
                     or 'none'), '']
    if name == 'beam':
        lines += ['#### Rubric score', '', *_grid(results, lambda table: 'n/a' if not table.get('rubric') else
                                                  f'{table["rubric"]["mean"]:.3f} ({table["rubric"]["scored"]})'), '']
    lines += ['#### Paired tests', '', 'Each over the same questions in both arms at one budget: the exact sign test '
              'and a seeded bootstrap interval resampling whole conversations. Demonstrated needs a two-sided '
              "rejection in the treatment's favour and an interval above zero; anything else is not demonstrated.",
              '', '| Treatment | Comparator | Measure | Types | Budget | Questions | Wins / losses / ties | Difference '
              '| Sign test | Verdict |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
              *(_contrast_line(item) for item in results['contrasts']), '']
    failures = results['failures']
    kinds = Counter(item['failure'] for item in failures)
    lines += ['#### Failures', '', (f'{len(failures)} failure(s): ' + ', '.join(f'{key} {kinds[key]}'
                                                                               for key in sorted(kinds)) + '.')
              if failures else 'None.', '']
    if failures:
        lines += ['| Arm | Budget | Question | Failure |', '| --- | --- | --- | --- |',
                  *(f'| `{item["arm"]}` | {_budget_text(item["budget_chars"])} | `{item["question"]}` | '
                    f'{item["failure"]} |' for item in failures), '']
    agreement = results.get('agreement')
    if agreement and agreement.get('agreement'):
        kappa = 'n/a' if agreement['kappa'] is None else f'{agreement["kappa"]:.2f}'
        lines += ['#### Judge agreement', '', f'{_rate(agreement["agreement"], counts=False)} of '
                  f'{agreement["labelled"]} labelled, kappa {kappa} ({agreement["unlabelled"]} left unlabelled).', '']
    else:
        lines += ['#### Judge agreement', '', 'The judge was not calibrated for this run: its agreement is not '
                  'demonstrated.', '']
    lines += ['#### Question ids', '', f'<details><summary>The {len(results["questions"])} questions</summary>', '',
              ', '.join(f'`{qid}`' for qid in results['questions']), '', '</details>', '']
    return '\n'.join(lines)


def render_results(directory: Path = RESULTS_DIR) -> str:
    """The results block: one section per committed results file, in dataset order, or the statement that nothing
    is demonstrated yet."""
    files = sorted(Path(directory).glob('*.json')) if Path(directory).is_dir() else []
    if not files:
        return ('No run has been published yet. Until one is, every memory benchmark result is not demonstrated: '
                'nothing on this page is a measurement.\n')
    sections = []
    for path in files:
        try:
            results = json.loads(path.read_text(encoding='utf-8'))
        except ValueError:
            raise ExportError(f'{path.name} is not a results file (export.py results)') from None
        if not isinstance(results, Mapping):
            raise ExportError(f'{path.name} is not a results file (export.py results)')
        sections.append(render_run(results, chart=path.with_suffix('.svg').is_file(), stem=path.stem))
    return '\n'.join(sections)


def render_method() -> str:
    """The method block: the pinned datasets, the budgets, and the reader's and judge's prompts with their hashes."""
    lines = ['| Dataset | Pinned upstream file | SHA-256 | Licence |', '| --- | --- | --- | --- |',
             *(f'| {TITLES[name]} (`{name}`) | [{pin["source"].split("/resolve/")[0].split("/datasets/")[-1]}]'
               f'({pin["source"]}) | `{pin["sha256"]}` | {pin["licence"]} |' for name, pin in PUBLIC.items()), '',
             f'Budgets: {" and ".join(f"{budget:,}" for budget in runner.BUDGETS)} characters by default; '
             f'{runner.HOST_INLINE_CAP:,} is the host\'s default inline cap.', '',
             f'**The reader** reads every arm alike, at temperature {reader.TEMPERATURE} with at most '
             f'{reader.MAX_TOKENS:,} output tokens and a strict JSON schema (`answer`, `answerable`). Its system '
             'prompt:', '', '```text', reader.SYSTEM, '```', '',
             'Its user message is the question date, the question and the memory the arm provided, as the host '
             f'renders it (`{reader.EMPTY_MEMORY}` when there is none). Reading SHA-256 (the fixed prompt, format and '
             f'settings): `{reader.READING_SHA256}`.', '',
             '**The judge** is never the reader\'s model. LongMemEval answers are checked with LongMemEval\'s own '
             'per-type prompts and its abstention prompt, and BEAM answers with BEAM\'s own rubric prompt, copied '
             'verbatim from the upstream code with their citations and licence notices '
             '([judge_prompts.py](../benchmarks/public_memory/judge_prompts.py)). One call at temperature 0, or three '
             f'votes at {judge.VOTE_TEMPERATURE} whose majority decides. Traps are judged as structures. The trap '
             'prompt (a knowledge update or a contradiction, shown the frozen values in an order that says nothing '
             'about which is which):', '', '```text', judge.TRAP_SYSTEM, '```', '',
             'The abstention prompt (for a question nothing answers, when the reader claimed an answer):', '',
             '```text', judge.ABSTAIN_SYSTEM, '```', '', f'Judge prompts SHA-256: `{judge.PROMPTS_SHA256}`.', '']
    return '\n'.join(lines)


def render_page(text: str, directory: Path = RESULTS_DIR) -> str:
    """``text`` with its generated blocks rewritten."""
    renderers = {'method': render_method, 'results': lambda: render_results(directory)}

    def block(match):
        if match.group(2) not in renderers:
            raise ExportError(f'the page has a generated block this exporter does not write: {match.group(2)}')
        return match.group(1) + renderers[match.group(2)]() + match.group(4)
    return _BLOCK.sub(block, text)


def load_dataset(name: str, data_dir: Optional[Path], offline: bool):
    return runner.load_dataset(name, data_dir, offline)


def _terms(path: Optional[Path]) -> list:
    return [] if path is None else read_terms(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    results = commands.add_parser('results', help='write the results file of one run')
    results.add_argument('--declaration', type=Path, required=True, help='the publisher\'s declaration (JSON)')
    results.add_argument('--data-dir', type=Path)
    results.add_argument('--offline', action='store_true')
    results.add_argument('--answers', type=Path, action='append', required=True)
    results.add_argument('--verdicts', type=Path, action='append', required=True)
    results.add_argument('--storage', type=Path, action='append', default=[])
    results.add_argument('--agreement', type=Path, help='calibration.py agreement output')
    results.add_argument('--labels', type=Path, help='the frozen old-value labels the declaration\'s labels_sha256 '
                         'pins (labels.py freeze)')
    results.add_argument('--private-terms', type=Path, help='strings never to publish, one per line (private); a '
                         f'configured list (${PRIVATE_TERMS_ENV}, else {DEFAULT_PRIVATE_TERMS}) always applies too')
    results.add_argument('--output', type=Path, required=True)
    page = commands.add_parser('page', help='rewrite the benchmark page\'s generated blocks')
    page.add_argument('--page', type=Path, default=PAGE)
    page.add_argument('--results-dir', type=Path, default=RESULTS_DIR)
    page.add_argument('--check', action='store_true', help='exit 1 when the page is not current; write nothing')
    args = parser.parse_args(argv)
    if args.command == 'page':
        current = args.page.read_text(encoding='utf-8')
        rendered = render_page(current, args.results_dir)
        if args.check:
            if rendered != current:
                print(f'{args.page} is not what the code and the committed results give: run export.py page',
                      file=sys.stderr)
                return 1
            return 0
        args.page.write_text(rendered, encoding='utf-8')
        return 0
    declaration = json.loads(args.declaration.read_text(encoding='utf-8'))
    if not isinstance(declaration, dict) or declaration.get('dataset') not in PUBLIC:
        raise SystemExit(f'the declaration names a dataset this harness publishes: {", ".join(PUBLIC)}')
    dataset = load_dataset(declaration['dataset'], args.data_dir, args.offline)
    agreement = json.loads(args.agreement.read_text(encoding='utf-8')) if args.agreement else None
    document = build(declaration, dataset, answers=args.answers, verdicts=args.verdicts, storage=args.storage,
                     agreement=agreement, labels=args.labels, private_terms=_terms(args.private_terms))
    args.output.write_text(dumps(document), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
