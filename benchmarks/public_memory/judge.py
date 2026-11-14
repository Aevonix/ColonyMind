"""The judge: whether each answer is right, by the benchmark's own protocol, and whether it fell into a trap.

**Right.** A LongMemEval answer is checked with LongMemEval's own prompt for its question type, or its abstention
prompt for an abstention question (``judge_prompts``, copied from the MIT upstream), and the judge's reply is read
as ``yes`` or ``no`` alone. A BEAM answer is scored with BEAM's own judge prompt once per rubric item (0, 0.5 or 1,
as a JSON object), and the question's score is the mean over its items, as BEAM computes it; the answer is counted
right when every item is fully met (a threshold this harness chose: LongMemEval likewise counts an answer that holds
only part of what is required as wrong). The mean is always reported beside it. BEAM's event-ordering questions are
scored by their rubric items only: its additional order measure (a rank correlation over events aligned by model
calls) is not computed here.

**Every verdict is treated alike.** Every item gets the same number of calls: one at temperature 0, or three votes
at ``VOTE_TEMPERATURE`` whose majority decides (``VOTES``). A reply that is not the asked-for structure is recorded
as refused and counts as no vote. Each decision needs a majority of the votes for one value (``_decided``): a yes or
no verdict, a rubric item's score, whether a listed value is asserted, whether the answer asks. Two replies that
disagree beside a refusal decide nothing. An item with no majority has no verdict (``undecided``) and is never asked
again; a trap is undecided when a decision without a majority could change its outcome.
Nothing is re-checked because it failed. A row is judged whole or not at all: a call that does not complete stops the
run with the row unwritten, and a resumed run judges that row from its first call.

**A verdict file holds one way of judging, over fixed inputs.** A resumed run extends a verdict file only with the same
judge, number of votes, scheme and prompts (``PROMPTS_SHA256``), and keeps a verdict already given only when it was
given on the same inputs: the same answer row (``answer_sha256``), the same question as the judge reads it (its
words, reference answer, type and rubric: ``question_sha256``) and the same frozen label (``label_sha256``).
Anything else is refused before the first call.

**The judge is never the reader's model.** A judge whose public name is the reader's is refused before any call,
and one whose server reports the reader's own model id is refused on its first reply.

**Traps** (``scoring.trap_outcome``) come from structured judgments, never from words found in an answer:

- ``abstain`` (an abstention question): a specific answer given as answerable is the trap. The reader's own
  ``answerable`` field says whether it claims an answer (false: it declined, no judge call); when it does, the judge
  says whether the answer states a specific answer at all (``ABSTAIN_SYSTEM``, a JSON object). An answer flagged
  answerable whose words give no specific answer is no trap, and the correctness verdict decides it, as it does an
  answer with no flag.
- ``update`` (a knowledge update) and ``conflict`` (a contradiction): the judge is shown the answer and a numbered
  list of values (the frozen old values and the current answer, or the conflict's two sides, from ``labels.py``,
  in an order that says nothing about which is which) and answers with the numbers of the values the answer states
  as the answer and whether it asks which one holds (``TRAP_SYSTEM``, a JSON object checked against the list).
- ``recall``: no trap.

A reader reply that was refused is a wrong answer with no call, and for BEAM it meets none of the rubric's items (0
on each), so the mean rubric score counts it as the accuracy does. A row the reader skipped (a capture the host
failed to serve) has no verdict and no score, and an item the judge left undecided leaves its row without a score:
both are counted apart, never as a zero.

    python benchmarks/public_memory/judge.py judge --dataset longmemeval_s --answers answers.jsonl \\
        --verdicts verdicts.jsonl --labels old-values.json --labels-sha256 HASH
    python benchmarks/public_memory/judge.py report --verdicts verdicts.jsonl --output report.json

The judge model is ``COLONYMIND_BENCH_JUDGE_MODEL`` and ``_REVISION`` (or ``--model`` and ``--revision``), served
at ``COLONYMIND_BENCH_JUDGE_BASE_URL`` with ``COLONYMIND_BENCH_JUDGE_API_KEY``; neither address nor key is written.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Mapping, Optional

# Support both package imports and the documented direct script commands.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.public_memory import models, scoring
from benchmarks.public_memory import judge_prompts as prompts
from benchmarks.public_memory import labels as labels_module

PROTOCOL = 'public-memory-verdict-1'
REPORT = 'public-memory-report-1'
LONGMEMEVAL, BEAM = 'longmemeval', 'beam-rubric'
SCHEMES = {'longmemeval_s': LONGMEMEVAL, 'beam': BEAM}
VOTES = (1, 3)
# The temperature of three votes: a harness choice (votes at temperature 0 would repeat one another).
VOTE_TEMPERATURE = 0.7
# LongMemEval's own evaluator gives the judge 10 tokens for its yes or no; the other two bounds are harness choices.
VERDICT_TOKENS, RUBRIC_TOKENS, TRAP_TOKENS = 10, 512, 128
RUBRIC_SCORES = (0.0, 0.5, 1.0)
RUBRIC_FORMAT = {'type': 'json_schema', 'json_schema': {'name': 'rubric_score', 'strict': True, 'schema': {
    'type': 'object', 'additionalProperties': False, 'required': ['score', 'reason'],
    'properties': {'score': {'type': 'number', 'enum': list(RUBRIC_SCORES)}, 'reason': {'type': 'string'}}}}}
TRAP_SYSTEM = (
    'You read a question, an answer to it, and a numbered list of values. Say which of the listed values the answer '
    'states as the answer to the question: for something that changed, the value it gives as holding now; for '
    'statements that disagree, the side it takes. A value the answer mentions only to say that it no longer holds, '
    'was replaced or is wrong is not stated as the answer. Say also whether the answer points out that the '
    'information changed or disagrees and asks which value holds. Reply with one JSON object and nothing else: '
    '{"asserted": [<the numbers of the values stated as the answer>], "asks_which_holds": <true or false>}.')
TRAP_FORMAT = {'type': 'json_schema', 'json_schema': {'name': 'trap_judgment', 'strict': True, 'schema': {
    'type': 'object', 'additionalProperties': False, 'required': ['asserted', 'asks_which_holds'],
    'properties': {'asserted': {'type': 'array', 'items': {'type': 'integer'}},
                   'asks_which_holds': {'type': 'boolean'}}}}}
ABSTAIN_SYSTEM = (
    'You read a question and an answer to it. Say whether the answer states a specific answer to the question: a '
    'particular value, name, date, number, place, event or description given as what holds, even with a hedge. An '
    'answer that says the information is missing, unknown or was never mentioned, or that declines or asks for more, '
    'states no specific answer, even when it names what is missing. Reply with one JSON object and nothing else: '
    '{"states_answer": <true or false>}.')
ABSTAIN_FORMAT = {'type': 'json_schema', 'json_schema': {'name': 'abstention_judgment', 'strict': True, 'schema': {
    'type': 'object', 'additionalProperties': False, 'required': ['states_answer'],
    'properties': {'states_answer': {'type': 'boolean'}}}}}
_FENCE = re.compile(r'```(?:json)?[ \t]*\n(.*)\n[ \t]*```', re.DOTALL)


class Refused(ValueError):
    """A judge reply that is not the asked-for structure. The message never repeats the reply."""


class JudgeIsReader(ValueError):
    """The judge would be the reader's own model."""


class JudgeMismatch(ValueError):
    """Verdicts given by one judge (or with one number of votes, scheme or set of prompts) would be extended by
    another."""


class ResumeMismatch(ValueError):
    """A verdict already given was given on other inputs than the row a resumed run would judge now."""


class LabelsMissing(ValueError):
    """A trap to judge has no frozen label."""


# ------------------------------------------------------------------ reading a judge's reply

def parse_yes_no(text: str) -> bool:
    """LongMemEval's verdict: the whole reply is ``yes`` or ``no`` (any case, one closing full stop or exclamation
    mark). Anything else is refused: no word is searched for inside a longer reply."""
    word = str(text or '').strip().casefold()
    if word[-1:] in ('.', '!'):
        word = word[:-1].strip()
    if word not in ('yes', 'no'):
        raise Refused('the verdict is not yes or no alone')
    return word == 'yes'


def _object(text: str, fields) -> dict:
    body = str(text or '').strip()
    fenced = _FENCE.fullmatch(body)
    if fenced:
        body = fenced.group(1).strip()
    try:
        value = json.loads(body)
    except ValueError:
        raise Refused('the reply is not a JSON object') from None
    if not isinstance(value, dict) or set(value) != set(fields):
        raise Refused('the reply is not the asked-for JSON object')
    return value


def parse_rubric(text: str) -> float:
    """BEAM's rubric score: ``{"score": 0, 0.5 or 1, "reason": "..."}`` and nothing else."""
    value = _object(text, ('score', 'reason'))
    score = value['score']
    if type(score) not in (int, float) or float(score) not in RUBRIC_SCORES or not isinstance(value['reason'], str):
        raise Refused('the score is not 0, 0.5 or 1 with a reason')
    return float(score)


def parse_trap(text: str, count: int) -> dict:
    """A trap judgment over ``count`` listed values: the numbers asserted (each listed, once) and whether it asks."""
    value = _object(text, ('asserted', 'asks_which_holds'))
    asserted, asks = value['asserted'], value['asks_which_holds']
    if (not isinstance(asserted, list) or any(type(number) is not int or not 1 <= number <= count
                                              for number in asserted) or len(set(asserted)) != len(asserted)):
        raise Refused('asserted must list the numbers of listed values, each once')
    if type(asks) is not bool:
        raise Refused('asks_which_holds must be true or false')
    return {'asserted': sorted(asserted), 'asks': asks}


def parse_abstain(text: str) -> bool:
    """An abstention judgment: ``{"states_answer": true or false}`` and nothing else."""
    value = _object(text, ('states_answer',))
    if type(value['states_answer']) is not bool:
        raise Refused('states_answer must be true or false')
    return value['states_answer']


# ------------------------------------------------------------------ prompts

def longmemeval_prompt(question, response: str) -> str:
    """LongMemEval's own answer check for ``question`` (``get_anscheck_prompt``)."""
    if question.abstention:
        return prompts.LONGMEMEVAL_ABSTENTION.format(question.question, question.answer, response)
    template = prompts.LONGMEMEVAL_BY_TYPE.get(question.type)
    if template is None:
        raise ValueError(f'LongMemEval has no judge prompt for {question.type}')
    return template.format(question.question, question.answer, response)


def beam_prompt(item: str, response: str) -> str:
    """BEAM's own judge prompt for one rubric item, filled as its evaluation fills it."""
    return prompts.BEAM_RUBRIC.replace('<rubric_item>', item).replace('<llm_response>', response)


def offered_values(question, label: Mapping[str, Any]) -> list:
    """The values a trap judgment lists: the old values and the current answer (an update), or the two sides (a
    conflict), in an order fixed by a hash of each value, which says nothing about which is which."""
    values = list(label['values'])
    if label['kind'] == labels_module.UPDATE and not any(scoring.same_value(value, question.answer)
                                                         for value in values):
        values.append(question.answer)
    return sorted(values, key=lambda value: hashlib.sha256(f'{question.id}\n{value}'.encode('utf-8')).hexdigest())


def trap_messages(question, response: str, values) -> list:
    listed = '\n'.join(f'{number}. {value}' for number, value in enumerate(values, 1))
    return [{'role': 'system', 'content': TRAP_SYSTEM},
            {'role': 'user', 'content': f'Question: {question.question}\n\nAnswer: {response}\n\nValues:\n{listed}'}]


# ------------------------------------------------------------------ what a verdict was given on

def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


# Every prompt and response format a verdict can use, and the settings of its calls.
PROMPTS_SHA256 = _digest({'protocol': PROTOCOL, 'longmemeval': prompts.LONGMEMEVAL_BY_TYPE,
                          'longmemeval_abstention': prompts.LONGMEMEVAL_ABSTENTION, 'beam': prompts.BEAM_RUBRIC,
                          'rubric_format': RUBRIC_FORMAT, 'trap_system': TRAP_SYSTEM, 'trap_format': TRAP_FORMAT,
                          'abstain_system': ABSTAIN_SYSTEM, 'abstain_format': ABSTAIN_FORMAT,
                          'vote_temperature': VOTE_TEMPERATURE,
                          'tokens': [VERDICT_TOKENS, RUBRIC_TOKENS, TRAP_TOKENS]})


def question_sha256(question) -> str:
    """What the judge reads of a question: its words, reference answer, type, abstention and rubric."""
    rubric = question.detail.get('rubric') if isinstance(question.detail, Mapping) else None
    return _digest({'id': question.id, 'type': question.type, 'question': question.question,
                    'answer': question.answer, 'abstention': bool(question.abstention), 'rubric': rubric})


def label_sha256(labels: Mapping[str, Any], question, row: Mapping[str, Any]) -> Optional[str]:
    """The frozen label a verdict on ``row`` reads (an update or a conflict that is not skipped), or None."""
    if row.get('skipped') or labels_module.trap_kind(question) not in labels_module.LABELLED:
        return None
    label = labels.get(question.id)
    return None if label is None else _digest(dict(label))


def abstain_messages(question, response: str) -> list:
    return [{'role': 'system', 'content': ABSTAIN_SYSTEM},
            {'role': 'user', 'content': f'Question: {question.question}\n\nAnswer: {response}'}]


# ------------------------------------------------------------------ judging one answer

def _temperature(votes: int) -> float:
    return 0 if votes == 1 else VOTE_TEMPERATURE


def _majority(count: int, votes: int) -> bool:
    return count * 2 > votes


def _decided(values, votes: int):
    """The value a majority of the ``votes`` gave, or None: a refused reply (None) is no vote, and a split between
    the others has no majority."""
    counts = Counter(value for value in values if value is not None)
    return next((value for value, count in counts.items() if _majority(count, votes)), None)


def _resolutions(decisions: list):
    """Every way the decisions without a majority (None) could have come out."""
    open_at = [index for index, decision in enumerate(decisions) if decision is None]
    for choice in itertools.product((False, True), repeat=len(open_at)):
        filled = list(decisions)
        for index, value in zip(open_at, choice):
            filled[index] = value
        yield filled


async def _ask(model, row, messages, *, votes, max_tokens, response_format=None):
    reply = await model.complete(messages, temperature=_temperature(votes), max_tokens=max_tokens,
                                 response_format=response_format)
    served = row.get('served_model')
    if reply.model and served and str(reply.model).strip().casefold() == str(served).strip().casefold():
        raise JudgeIsReader(f'the judge is served by the reader\'s own model ({served})')
    return reply


async def _votes(model, row, messages, *, votes, max_tokens, parse, name, calls, response_format=None) -> list:
    parsed = []
    for vote in range(votes):
        reply = await _ask(model, row, messages, votes=votes, max_tokens=max_tokens, response_format=response_format)
        call = {'prompt': name, 'vote': vote, 'reply': reply.content, 'served_model': reply.model}
        try:
            call['parsed'] = parse(reply.content)
        except Refused as refused:
            call.update(parsed=None, refusal=str(refused))
        calls.append(call)
        parsed.append(call['parsed'])
    return parsed


async def _right_longmemeval(model, row, question, *, votes, calls):
    parsed = await _votes(model, row, [{'role': 'user', 'content': longmemeval_prompt(question, row['answer'])}],
                          votes=votes, max_tokens=VERDICT_TOKENS, parse=parse_yes_no, name='answer', calls=calls)
    return _decided(parsed, votes), None, None


def _rubric(question) -> list:
    rubric = question.detail.get('rubric') if isinstance(question.detail, Mapping) else None
    return rubric if isinstance(rubric, list) else []


async def _right_beam(model, row, question, *, votes, calls):
    rubric = _rubric(question)
    if not rubric:
        return None, None, None
    items = []
    for number, item in enumerate(rubric):
        parsed = await _votes(model, row, [{'role': 'user', 'content': beam_prompt(str(item), row['answer'])}],
                              votes=votes, max_tokens=RUBRIC_TOKENS, parse=parse_rubric, name=f'rubric:{number}',
                              calls=calls, response_format=RUBRIC_FORMAT)
        items.append(_decided(parsed, votes))
    if any(score is None for score in items):
        return None, items, None
    mean = sum(items) / len(items)
    return mean == 1.0, items, mean


def _label(labels: Mapping[str, Any], question) -> Mapping[str, Any]:
    label = labels.get(question.id)
    if label is None or label.get('kind') != labels_module.trap_kind(question):
        raise LabelsMissing(f'{question.id}: no frozen {labels_module.trap_kind(question)} label')
    return label


def _outcome(kind, label, *, right, asserted=(), asked=False, answerable=None):
    values = list(label['values']) if label else []
    return scoring.trap_outcome(kind, right=bool(right), asserted=list(asserted), asked=asked,
                                answerable=answerable, stale=values if kind == labels_module.UPDATE else (),
                                sides=values if kind == labels_module.CONFLICT else ())


async def _trap(model, row, question, kind, labels, *, right, votes, calls):
    if kind == labels_module.RECALL:
        return None
    if kind == labels_module.ABSTAIN:
        if row['answerable'] is not True:
            # The reader declined: no specific answer is given as answerable, whatever its words add.
            return _outcome(kind, None, right=right, answerable=row['answerable'])
        parsed = await _votes(model, row, abstain_messages(question, row['answer']), votes=votes,
                              max_tokens=TRAP_TOKENS, parse=parse_abstain, name='abstain', calls=calls,
                              response_format=ABSTAIN_FORMAT)
        states = _decided(parsed, votes)
        if states is None:
            return {'kind': kind, 'outcome': None, 'matched': [], 'undecided': True}
        # A specific answer given as answerable is the trap; a flag the words do not bear out is not read, and the
        # correctness verdict decides.
        outcome = _outcome(kind, None, right=right, asserted=[row['answer']] if states else [],
                           answerable=True if states else None)
        return {**outcome, 'states_answer': states}
    label = _label(labels, question)
    values = offered_values(question, label)
    parsed = await _votes(model, row, trap_messages(question, row['answer'], values), votes=votes,
                          max_tokens=TRAP_TOKENS, parse=lambda text: parse_trap(text, len(values)), name='trap',
                          calls=calls, response_format=TRAP_FORMAT)
    # One decision per listed value (is it asserted?) and one for the ask, each by a majority of the votes; the
    # trap is decided when every way the decisions without a majority could come out gives one outcome.
    decisions = [_decided([None if item is None else number in item['asserted'] for item in parsed], votes)
                 for number in range(1, len(values) + 1)]
    decisions.append(_decided([None if item is None else item['asks'] for item in parsed], votes))
    outcomes = {json.dumps(_outcome(kind, label, right=right, asked=filled[-1],
                                    asserted=[value for value, chosen in zip(values, filled) if chosen]),
                           sort_keys=True)
                for filled in _resolutions(decisions)}
    if len(outcomes) != 1:
        return {'kind': kind, 'outcome': None, 'matched': [], 'undecided': True}
    return json.loads(outcomes.pop())


async def judge_row(row: Mapping[str, Any], question, model, *, dataset: str, votes: int = 1,
                    labels: Optional[Mapping[str, Any]] = None) -> dict:
    """One verdict row for one answer row (``reader.answer_row``)."""
    if votes not in VOTES:
        raise ValueError(f'votes is one of {VOTES}')
    if dataset not in SCHEMES:
        raise ValueError(f'no judge protocol for {dataset}')
    if model.spec.same_model(row.get('reader') or {}):
        raise JudgeIsReader('the judge is the reader\'s own model')
    labels = labels or {}
    kind = labels_module.trap_kind(question)
    verdict = {'protocol': PROTOCOL, 'question': row['question'], 'conversation': row['conversation'],
               'type': question.type, 'arm': row['arm'], 'budget_chars': row.get('budget_chars'),
               'reader': row.get('reader'), 'response': row.get('answer'), 'judge': model.spec.identity(),
               'votes': votes, 'prompts_sha256': PROMPTS_SHA256, 'answer_sha256': _digest(dict(row)),
               'question_sha256': question_sha256(question), 'label_sha256': label_sha256(labels, question, row),
               'temperature': _temperature(votes), 'scheme': SCHEMES[dataset], 'score': row.get('score'),
               'trap_kind': kind, 'calls': [], 'right': None, 'undecided': False, 'rubric': None,
               'rubric_score': None, 'trap': None, 'reader_refused': False, 'skipped': row.get('skipped')}
    if row.get('skipped'):
        return verdict
    if row.get('parse') != 'ok':
        # The reader did not answer in the structure: a wrong answer, and nothing to judge. It meets no rubric
        # item, so a BEAM mean counts it as the accuracy does.
        label = labels.get(question.id) if kind in labels_module.LABELLED else None
        trap = None if kind == labels_module.RECALL else _outcome(kind, label, right=False)
        refused = {**verdict, 'right': False, 'reader_refused': True, 'trap': trap}
        rubric = _rubric(question) if dataset == 'beam' else []
        if rubric:
            refused.update(rubric=[0.0] * len(rubric), rubric_score=0.0)
        return refused
    if kind in labels_module.LABELLED:
        _label(labels, question)
    calls = verdict['calls']
    if dataset == 'beam':
        right, items, mean = await _right_beam(model, row, question, votes=votes, calls=calls)
    else:
        right, items, mean = await _right_longmemeval(model, row, question, votes=votes, calls=calls)
    trap = await _trap(model, row, question, kind, labels, right=right, votes=votes, calls=calls)
    return {**verdict, 'right': right, 'undecided': right is None, 'rubric': items, 'rubric_score': mean,
            'trap': trap}


# ------------------------------------------------------------------ a run

def _rows(path: Path) -> list:
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def _key(row: Mapping[str, Any]) -> tuple:
    return (row['question'], row['arm'], row.get('budget_chars'))


async def judge(answers: Path, questions: Mapping[str, Any], model, out: Path, *, dataset: str, votes: int = 1,
                labels: Optional[Mapping[str, Any]] = None) -> dict:
    """Judge every answer row not yet in ``out``, appending each verdict row as it is complete. Everything a run
    could refuse is checked before its first call: the judge being the reader, a judge, vote count, scheme or set of
    prompts other than the ones ``out`` was judged with, a verdict already given on other inputs, a trap with no
    frozen label."""
    out, labels = Path(out), labels or {}
    rows = _rows(answers)
    done = _rows(out) if out.is_file() else []
    for row in done:
        if row.get('judge') != model.spec.identity() or row.get('votes') != votes:
            raise JudgeMismatch(f'{out} was judged by {row.get("judge")} with {row.get("votes")} vote(s): every '
                                'verdict is given alike')
        if row.get('scheme') != SCHEMES.get(dataset) or row.get('prompts_sha256') != PROMPTS_SHA256:
            raise JudgeMismatch(f'{out} was judged under another scheme or with other prompts: every verdict is '
                                'given alike')
    finished = {_key(row): row for row in done}
    for row in rows:
        previous = finished.get(_key(row))
        if previous is None:
            continue
        question = questions.get(row['question'])
        if question is None:
            raise KeyError(f'no question {row["question"]} in the dataset read')
        where = f'{row["question"]} ({row["arm"]}, {row.get("budget_chars")})'
        if previous.get('answer_sha256') != _digest(dict(row)):
            raise ResumeMismatch(f'{where} was judged on another answer row: judge it into a new file')
        if previous.get('question_sha256') != question_sha256(question):
            raise ResumeMismatch(f'{where} was judged against another reading of its question: judge it into a new '
                                 'file')
        if previous.get('label_sha256') != label_sha256(labels, question, row):
            raise ResumeMismatch(f'{where} was judged against another frozen label: judge it into a new file')
    pending = [row for row in rows if _key(row) not in finished]
    for row in pending:
        if model.spec.same_model(row.get('reader') or {}):
            raise JudgeIsReader('the judge is the reader\'s own model')
        question = questions.get(row['question'])
        if question is None:
            raise KeyError(f'no question {row["question"]} in the dataset read')
        if (labels_module.trap_kind(question) in labels_module.LABELLED and row.get('parse') == 'ok'
                and not row.get('skipped')):
            _label(labels, question)
    for row in pending:
        verdict = await judge_row(row, questions[row['question']], model, dataset=dataset, votes=votes,
                                  labels=labels)
        with out.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(verdict, sort_keys=True, ensure_ascii=False) + '\n')
    return {'judged': len(pending), 'already': len(rows) - len(pending)}


# ------------------------------------------------------------------ the report

def scored(verdict: Mapping[str, Any]) -> dict:
    """A verdict as a ``scoring.summarize`` row: the capture's reach and characters, the verdict and the trap."""
    score = verdict.get('score') or {}
    row = {'question': verdict['question'], 'conversation': verdict['conversation'], 'type': verdict['type'],
           'arm': verdict['arm'], 'budget': score.get('budget', verdict.get('budget_chars')),
           'reach': score.get('reach'), 'chars': score.get('chars'), 'right': verdict.get('right'),
           'rubric_score': verdict.get('rubric_score')}
    trap = verdict.get('trap')
    if trap and trap.get('outcome') is not None:
        row['trap'] = trap
    if verdict.get('skipped'):
        row['reach'] = None
    return row


def report(verdicts: Path) -> dict:
    """Per arm and budget: the scorer's tables (``scoring.summarize``), the mean rubric score over the rows that
    have one (a reader refusal scores 0; a skipped or undecided row has none), and how many rows the reader refused,
    the judge left undecided (the verdict, or the trap) and the reader skipped."""
    rows = _rows(verdicts)
    groups: dict = {}
    for verdict in rows:
        groups.setdefault(verdict['arm'], {}).setdefault(str(verdict.get('budget_chars')), []).append(verdict)
    summary, counts, rubric = {}, {}, {}
    for arm, budgets in sorted(groups.items()):
        for budget, members in sorted(budgets.items()):
            summary.setdefault(arm, {})[budget] = scoring.summarize([scored(verdict) for verdict in members])
            counts.setdefault(arm, {})[budget] = {
                'rows': len(members), 'reader_refused': sum(bool(v.get('reader_refused')) for v in members),
                'undecided': sum(bool(v.get('undecided')) and not v.get('skipped') for v in members),
                'trap_undecided': sum(bool((v.get('trap') or {}).get('undecided')) for v in members),
                'skipped': sum(bool(v.get('skipped')) for v in members)}
            scores = [v['rubric_score'] for v in members if v.get('rubric_score') is not None]
            if scores:
                rubric.setdefault(arm, {})[budget] = {'mean': statistics.fmean(scores), 'scored': len(scores)}
    return {'protocol': REPORT, 'rows': [scored(verdict) for verdict in rows], 'summary': summary,
            'counts': counts, 'rubric': rubric}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    judging = commands.add_parser('judge', help='judge an answer file')
    judging.add_argument('--dataset', required=True, choices=sorted(SCHEMES))
    judging.add_argument('--data-dir', type=Path)
    judging.add_argument('--offline', action='store_true')
    judging.add_argument('--answers', type=Path, required=True)
    judging.add_argument('--verdicts', type=Path, required=True)
    judging.add_argument('--votes', type=int, choices=VOTES, default=1)
    judging.add_argument('--labels', type=Path, help='the frozen old-value labels (labels.py freeze)')
    judging.add_argument('--labels-sha256', help='the pinned hash of --labels')
    judging.add_argument('--model', help='the judge\'s public name (else COLONYMIND_BENCH_JUDGE_MODEL)')
    judging.add_argument('--revision', help='its pinned weights revision (else COLONYMIND_BENCH_JUDGE_REVISION)')
    reporting = commands.add_parser('report', help='score a verdict file per arm and budget')
    reporting.add_argument('--verdicts', type=Path, required=True)
    reporting.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'report':
        args.output.write_text(json.dumps(report(args.verdicts), indent=2, sort_keys=True, default=str),
                               encoding='utf-8')
        return 0
    if bool(args.labels) != bool(args.labels_sha256):
        parser.error('--labels and --labels-sha256 go together')
    dataset = labels_module.load_dataset(args.dataset, args.data_dir, args.offline)
    labels = labels_module.load(args.labels, args.labels_sha256, dataset) if args.labels else {}
    model = models.OpenAICompatible('JUDGE', models.spec_from_environment('JUDGE', name=args.model,
                                                                          revision=args.revision))
    counts = asyncio.run(judge(args.answers, {question.id: question for question in dataset.questions}, model,
                               args.verdicts, dataset=args.dataset, votes=args.votes, labels=labels))
    print(json.dumps(counts), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
