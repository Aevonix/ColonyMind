"""The reader: one fixed prompt that answers a question from what an arm put in front of it.

Every arm is read the same way, by the same model: the question, the date it is asked, and the memory the arm
provided, with no hint of the question's category and no instruction on how to treat values that disagree. The
reader answers in a structure, ``{"answer": ..., "answerable": ...}``, at temperature 0; ``answerable`` is the
reader's own claim that the memory holds the answer, which the abstention traps read (``scoring.trap_outcome``). A
reply that is not that structure is recorded as refused and its text is never read for an answer.

**What the reader is shown is what the host injects.** A ColonyMind arm's capture row (``runner.py``) holds the
sections ``/v1/host/context/assemble`` returned for the budget, packed as a host's turn is packed (recall with no
count limit behind a verified cutoff, then packed again within the host's allowance); the reader is shown those
sections cut to the budget in the host's order (``runner.fit``) and rendered as the host renders them: each section
as ``## <title>`` and its body, a blank line between sections (``render``). A baseline's row (``baselines.py``)
holds one section its arm packed with the host's own packer. The reader never selects or packs anything itself.

Each answer row keeps the reader's identity (the public name and pinned revision), the raw reply, the parsed
fields, and the capture row's own score (reach, characters, budget), so the judge and the report need nothing but
the answer file. A capture the host failed to serve (a status other than 200) and a full-context row that does not
fit the reader are written as skipped, with no call.

    python benchmarks/public_memory/reader.py --dataset longmemeval_s --capture capture.jsonl --answers answers.jsonl

The reader model is ``COLONYMIND_BENCH_READER_MODEL`` and ``_REVISION`` (or ``--model`` and ``--revision``), served
at ``COLONYMIND_BENCH_READER_BASE_URL`` with ``COLONYMIND_BENCH_READER_API_KEY``; neither address nor key is written.
A run resumes: rows already answered are not asked again, and answers read by one model, or read another way, are
never extended. A row already answered is kept only when it was answered from the same inputs: the same capture row
(``capture_sha256``) and the same prompt (``prompt_sha256``: the fixed text, the question and its date, the memory
shown), under the same way of reading (``reading_sha256``: the fixed prompt, its response format, temperature and
output bound). Anything else is refused before the first call, never mixed into one answer file.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional

# Support both package imports and the documented direct script commands.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.public_memory import models, runner

PROTOCOL = 'public-memory-answer-1'
TEMPERATURE = 0
# Room for the longest answer a question asks for (a summary of a whole conversation): a harness choice.
MAX_TOKENS = 1024
EMPTY_MEMORY = '(no memory was provided)'
SYSTEM = (
    'You are an assistant answering a question from your user. The memory below is what your memory system '
    'recalled from your earlier conversations with this user; it may be incomplete, and it may hold nothing that '
    'answers the question. Answer from what the memory holds and the date the question is asked. Reply with one '
    'JSON object and nothing else: {"answer": "<your answer>", "answerable": <true or false>}. Set answerable to '
    'false when the question depends on something the memory does not hold, and then say in answer what is '
    'missing.')
SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['answer', 'answerable'],
          'properties': {'answer': {'type': 'string'}, 'answerable': {'type': 'boolean'}}}
RESPONSE_FORMAT = {'type': 'json_schema', 'json_schema': {'name': 'memory_answer', 'strict': True, 'schema': SCHEMA}}
FIELDS = frozenset(SCHEMA['properties'])
_FENCE = re.compile(r'```(?:json)?[ \t]*\n(.*)\n[ \t]*```', re.DOTALL)
WEEKDAYS = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')


class Refused(ValueError):
    """A reply that is not the answer structure. The message says what is wrong, never what the reply said."""


class ReaderMismatch(ValueError):
    """Answers read by one model, or read another way, would be extended: every arm is read alike."""


class ResumeMismatch(ValueError):
    """A row already answered was answered from other inputs than the row a resumed run would answer now."""


def question_date(question) -> str:
    """The question's date as a reader is given it: the dataset's own stamp where it gives one, else the time the
    question is asked, in UTC."""
    if question.date_text:
        return question.date_text
    moment = question.asked_at
    return f'{moment:%Y-%m-%d} ({WEEKDAYS[moment.weekday()]}) {moment:%H:%M} UTC'


def messages(question, memory: str) -> list:
    """The one prompt every arm is read with."""
    return [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': f'Question date: {question_date(question)}\n\n'
                                        f'Question: {question.question}\n\nMemory:\n{memory or EMPTY_MEMORY}'}]


def parse(text: str) -> dict:
    """The answer structure, or ``Refused``. The whole reply is one JSON object with exactly the two fields; one
    code fence around the object is the object. Nothing is searched for inside free text."""
    body = str(text or '').strip()
    fenced = _FENCE.fullmatch(body)
    if fenced:
        body = fenced.group(1).strip()
    try:
        value = json.loads(body)
    except ValueError:
        raise Refused('the reply is not a JSON object') from None
    if not isinstance(value, dict):
        raise Refused('the reply is not a JSON object')
    if set(value) != FIELDS:
        raise Refused('the object must have exactly the fields answer and answerable')
    if not isinstance(value['answer'], str):
        raise Refused('answer must be a string')
    if type(value['answerable']) is not bool:
        raise Refused('answerable must be true or false')
    return {'answer': value['answer'], 'answerable': value['answerable']}


def render(sections) -> str:
    """Sections as the host renders them into the prompt (the memory provider's ``_format_sections``)."""
    return '\n\n'.join(f"## {section.get('title') or section['id']}\n{section['body']}" for section in sections)


def _baselines():
    from benchmarks.public_memory import baselines
    return baselines


def shown_sections(row: Mapping[str, Any]) -> list:
    """The sections a reader is shown for a capture row: a ColonyMind arm's cut to its budget in the host's order,
    a baseline's as its arm packed them with the host's packer."""
    if row.get('protocol') == runner.PROTOCOL:
        return runner.fit(row['sections'], int(row['budget_chars']))
    if row.get('protocol') == _baselines().PROTOCOL:
        return list(row['sections'])
    raise ValueError('Not a public memory capture or baseline row')


def row_score(row: Mapping[str, Any]) -> dict:
    """The capture row's own score: reach, items, characters and budget (``runner.score``, ``baselines.score``)."""
    if row.get('protocol') == runner.PROTOCOL:
        return runner.score(row)
    if row.get('protocol') == _baselines().PROTOCOL:
        return _baselines().score(row)
    raise ValueError('Not a public memory capture or baseline row')


def _skip_reason(row: Mapping[str, Any]) -> Optional[str]:
    if row.get('protocol') == runner.PROTOCOL and row.get('status') != 200:
        return 'capture_failed'
    if row.get('fits') is False:
        return 'does_not_fit'
    return None


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


# How every answer is read, apart from the question and the memory: an answer file holds one way of reading only.
READING_SHA256 = _digest({'protocol': PROTOCOL, 'system': SYSTEM, 'empty_memory': EMPTY_MEMORY,
                          'response_format': RESPONSE_FORMAT, 'temperature': TEMPERATURE, 'max_tokens': MAX_TOKENS})


def key(row: Mapping[str, Any]) -> tuple:
    """What identifies a row across the capture, answer and verdict files."""
    return (row['question'], row['arm'], row.get('budget_chars'))


async def answer_row(row: Mapping[str, Any], question, model) -> dict:
    """One answer row: ``question`` read by ``model`` from what the capture ``row`` shows."""
    result = {'protocol': PROTOCOL, 'question': row['question'], 'conversation': row['conversation'],
              'type': row['type'], 'abstention': bool(row.get('abstention')), 'arm': row['arm'],
              'budget_chars': row.get('budget_chars'), 'capture_sha256': _digest(row), 'score': row_score(row),
              'reader': model.spec.identity(), 'reading_sha256': READING_SHA256, 'answer': None,
              'answerable': None}
    skipped = _skip_reason(row)
    if skipped:
        return {**result, 'skipped': skipped}
    memory = render(shown_sections(row))
    prompt = messages(question, memory)
    reply = await model.complete(prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                                 response_format=RESPONSE_FORMAT)
    result.update(prompt_sha256=_digest(prompt), memory_chars=len(memory), served_model=reply.model,
                  usage=dict(reply.usage or {}), reply=reply.content)
    try:
        result.update(parse(reply.content), parse='ok')
    except Refused as refused:
        result.update(parse='refused', refusal=str(refused))
    return result


def read_rows(path: Path) -> list:
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def _prompt_sha256(row: Mapping[str, Any], question) -> str:
    return _digest(messages(question, render(shown_sections(row))))


def _check_resumed(previous: Mapping[str, Any], row: Mapping[str, Any], question) -> None:
    """Refuse an answer already in the file unless it was read from ``row`` as ``row`` would be read now."""
    where = f'{row["question"]} ({row["arm"]}, {row.get("budget_chars")})'
    if previous.get('protocol') != PROTOCOL or previous.get('capture_sha256') != _digest(row):
        raise ResumeMismatch(f'{where} was answered from another capture row: answer it into a new file')
    if not previous.get('skipped') and previous.get('prompt_sha256') != _prompt_sha256(row, question):
        raise ResumeMismatch(f'{where} was answered from another prompt (the question, its date or the memory '
                             'shown): answer it into a new file')


async def answer(capture: Path, questions: Mapping[str, Any], model, out: Path) -> dict:
    """Answer every row of ``capture`` not yet in ``out``, appending each answer as it is read. A call that does
    not complete stops the run with its row unwritten; the next run asks it again. ``out`` holds one reader, reading
    one way, only. Everything a run could refuse is checked before its first call: another reader or way of
    reading, a row already answered from other inputs (``_check_resumed``), one row twice in ``capture`` with
    different contents, a question the dataset does not hold."""
    out = Path(out)
    done = read_rows(out) if out.is_file() else []
    for row in done:
        if row.get('reader') != model.spec.identity():
            raise ReaderMismatch(f'{out} was read by {row.get("reader")}, not {model.spec.identity()}: every arm '
                                 'is read by the same model')
        if row.get('reading_sha256') != READING_SHA256:
            raise ReaderMismatch(f'{out} was read with another prompt, format or settings: every arm is read the '
                                 'same way')
    answered = {key(row): row for row in done}
    rows, seen = read_rows(capture), {}
    for row in rows:
        if row['question'] not in questions:
            raise KeyError(f'no question {row["question"]} in the dataset read')
        if seen.setdefault(key(row), _digest(row)) != _digest(row):
            raise ResumeMismatch(f'{row["question"]} ({row["arm"]}, {row.get("budget_chars")}) is in {capture} '
                                 'twice with different contents')
        if key(row) in answered:
            _check_resumed(answered[key(row)], row, questions[row['question']])
    counts = {'answered': 0, 'already': 0, 'skipped': 0}
    for row in rows:
        if key(row) in answered:
            counts['already'] += 1
            continue
        result = await answer_row(row, questions[row['question']], model)
        with out.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(result, sort_keys=True, ensure_ascii=False) + '\n')
        answered[key(row)] = result
        counts['skipped' if result.get('skipped') else 'answered'] += 1
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, choices=('longmemeval_s', 'beam'))
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--capture', type=Path, action='append', required=True,
                        help='a capture or baseline file; repeat for more')
    parser.add_argument('--answers', type=Path, required=True)
    parser.add_argument('--model', help='the reader\'s public name (else COLONYMIND_BENCH_READER_MODEL)')
    parser.add_argument('--revision', help='its pinned weights revision (else COLONYMIND_BENCH_READER_REVISION)')
    args = parser.parse_args(argv)
    dataset = runner.load_dataset(args.dataset, args.data_dir, args.offline)
    questions = {question.id: question for question in dataset.questions}
    model = models.OpenAICompatible('READER', models.spec_from_environment('READER', name=args.model,
                                                                            revision=args.revision))
    for capture in args.capture:
        counts = asyncio.run(answer(capture, questions, model, args.answers))
        print(json.dumps({'capture': str(capture), **counts}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
