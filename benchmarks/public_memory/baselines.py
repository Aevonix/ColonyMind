"""The baselines: memory without ColonyMind, packed at the same budget by the packer the host's turn uses.

Five arms read the raw history (``ARMS``):

- ``bm25-sessions``: Okapi BM25 over whole sessions (user and assistant turns), the question as the query; sessions
  that share no term with it are not returned. Ties keep the sessions' time order.
- ``oracle``: the gold sessions only, in time order: what reading costs at the budget when retrieval is perfect.
- ``full-context``: every session in time order, not cut to a budget: the ceiling, where the reader's whole request
  fits its context (``--context-chars``, in characters): the messages the reader sends with the history rendered as
  it reads it, and its response format, as JSON (``request_chars``), plus its output bound reserved at
  ``CHARS_PER_TOKEN`` characters a token (``reserve_chars``). A question whose request does not fit is recorded and
  not read.
- ``session-search``: stock Hermes' ``session_search`` tool, called once with the question and its defaults, over a
  Hermes session store built by Hermes' own import (``SessionDB.import_sessions``), one session per dataset session
  at its own time. It needs stock Hermes installed beside the sidecar. This is the host's session search, not all
  of the host's memory: a replay writes none of the host's own notes. The tool's full-text query joins every term
  with AND, so a question asked as it is phrased rarely matches anything; the result says so and advises OR.
- ``session-search-or``: the same tool called once with the question's words joined by OR, as that advice says (a
  harness choice, recorded as the arm's query; the words are the question's own, in its order, each once).

Hermes shows a session's time in the process's local zone; each call runs with the zone set to UTC, so a result
does not depend on the machine it ran on.

**Packing.** A baseline's items (whole sessions, or the tool's result entries) are packed in the arm's own order by
``colonymind.memory.recall.pack_memory_context`` with no count limit: the packer a host's turn uses behind a
verified cutoff, with its rules for what does not fit (an item is shortened to the room left when at least 80 of its
characters fit, and skipped otherwise). The packed evidence is one section titled as ColonyMind's memory section,
so the reader cannot tell the arms apart, and the section costs no more than the budget as the host counts it
(``memory.blocks.cost``). A whole session longer than the budget is therefore shortened, never dropped.

**No gold leaks.** Sessions are shown by their date only; the dataset's own session ids (LongMemEval names its
evidence sessions ``answer_...``) never reach the reader, and the Hermes store gives each session a neutral id.

**Reach** is structural, by session (``scoring.session_reach``: a shown session that is a gold session), as the
evaluation plan specifies for baselines; ``turn_reach`` also counts, by turn, the gold turns whose words are shown
whole (a shortened session shows its first turns). A session-search result that carries no session id falls back
to the answer-span check, labelled ``weaker``.

    python benchmarks/public_memory/baselines.py --dataset longmemeval_s --arm bm25-sessions --capture base.jsonl
    python benchmarks/public_memory/baselines.py --dataset longmemeval_s --arm full-context --context-chars 400000 \\
        --capture base.jsonl

Rows (``PROTOCOL``) are read by ``reader.py`` exactly as a ColonyMind arm's capture rows are.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import unicodedata
from typing import Any, Iterable, Mapping, Optional

# Support both package imports and the documented direct script commands.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.public_memory import runner, scoring

PROTOCOL = 'public-memory-baseline-1'
ARMS = ('bm25-sessions', 'oracle', 'full-context', 'session-search', 'session-search-or')
SEARCH_ARMS = ('session-search', 'session-search-or')
SECTION, TITLE = 'baseline-memory', 'Relevant Memories'
BUDGETS = runner.BUDGETS
# Okapi BM25's usual constants: a harness choice, not tuned on any benchmark question.
K1, B = 1.2, 0.75
# The reader's output bound, reserved in characters at this many a token: a harness choice (a common ratio for
# English text). State --context-chars at the same ratio.
CHARS_PER_TOKEN = 4
_TOKEN = re.compile(r'\w+', re.UNICODE)


# ------------------------------------------------------------------ sessions as items

def tokens(text: str) -> list:
    return _TOKEN.findall(unicodedata.normalize('NFKC', str(text or '')).casefold())


def session_text(session) -> tuple:
    """A session's words as a reader is shown them, one ``role: words`` line per turn, and where each turn ends."""
    lines, ends, length = [], [], 0
    for turn in session.turns:
        line = f'{turn.role}: {turn.content}'
        length += len(line) + (1 if lines else 0)
        lines.append(line)
        ends.append(length)
    return '\n'.join(lines), ends


def bm25(query: str, documents: list) -> list:
    """``[(index, score)]`` for the documents that share a term with ``query``, best first, ties in the documents'
    own order."""
    docs = [Counter(tokens(text)) for text in documents]
    count = len(docs)
    if not count:
        return []
    average = sum(sum(doc.values()) for doc in docs) / count or 1.0
    terms = sorted(set(tokens(query)))
    frequency = {term: sum(term in doc for doc in docs) for term in terms}
    scores = []
    for index, doc in enumerate(docs):
        length, score = sum(doc.values()), 0.0
        for term in terms:
            tf = doc.get(term, 0)
            if not tf:
                continue
            idf = math.log(1 + (count - frequency[term] + 0.5) / (frequency[term] + 0.5))
            score += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / average))
        if score > 0:
            scores.append((index, score))
    return sorted(scores, key=lambda item: (-item[1], item[0]))


def ranked_sessions(arm: str, conversation, question) -> list:
    """The sessions an arm returns, in its own order (``session-search`` is ranked by Hermes, ``session_search``)."""
    sessions = list(conversation.sessions)
    if arm == 'bm25-sessions':
        return [sessions[index] for index, _ in bm25(question.question, [session_text(s)[0] for s in sessions])]
    if arm == 'oracle':
        gold = set(question.gold_sessions)
        return [session for session in sessions if session.key in gold]
    if arm == 'full-context':
        return sessions
    raise ValueError(f'{arm} does not rank sessions')


def _row(key: str, content: str, at: Optional[str], kind: str) -> dict:
    row = {'id': f'baseline:{key}', 'kind': kind, 'source_uri': '', 'epistemic_state': 'quotation',
           'content': content, '_item': key}
    if at:
        row['occurred_at'] = at
    return row


def pack(items: Iterable[dict], budget: int) -> tuple:
    """The items packed in their order within ``budget`` as the host counts a section: ``(selected, body)``."""
    from colonymind.memory import blocks
    from colonymind.memory.recall import pack_memory_context
    room = max(0, int(budget) - blocks.cost(TITLE, ''))
    return pack_memory_context(list(items), limit=None, max_chars=room)


def _whole(items: Iterable[dict]) -> str:
    """Every item rendered as the packer renders it, with no budget (the full-context arm)."""
    from colonymind.memory.recall import pack_memory_context
    return pack_memory_context(list(items), limit=None, max_chars=10 ** 12)[1]


def _reader():
    from benchmarks.public_memory import reader
    return reader


def request_fit(question, sections: list, context_chars: int) -> dict:
    """Whether the reader's whole request for ``sections`` fits ``context_chars``: its messages as it sends them (the
    fixed text, the question, its date, the memory rendered as it reads it) and its response format, as JSON, with
    its output bound reserved."""
    reader = _reader()
    messages = reader.messages(question, reader.render(sections))
    request = (len(json.dumps(messages, ensure_ascii=False))
               + len(json.dumps(reader.RESPONSE_FORMAT, ensure_ascii=False)))
    reserve = reader.MAX_TOKENS * CHARS_PER_TOKEN
    return {'request_chars': request, 'reserve_chars': reserve, 'fits': request + reserve <= int(context_chars)}


def _visible_turns(session, shown: Mapping[str, Any]) -> list:
    text, ends = session_text(session)
    length = len(str(shown.get('content') or '')) if shown.get('excerpt_truncated') else len(text)
    return [index for index, end in enumerate(ends) if end <= length]


def _base(question, arm: str, budget: Optional[int]) -> dict:
    return {'protocol': PROTOCOL, 'question': question.id, 'conversation': question.conversation,
            'type': question.type, 'abstention': bool(question.abstention), 'arm': arm, 'budget_chars': budget,
            'gold_sessions': list(question.gold_sessions),
            'gold_turns': [[item.session, item.turn] for item in question.evidence], 'answer_spans': [question.answer]}


def session_row(arm: str, conversation, question, budget: Optional[int] = None, *,
                context_chars: Optional[int] = None) -> dict:
    """A baseline row for a session arm: what it shows within ``budget`` (the whole history for ``full-context``,
    with ``fits``: whether the reader's whole request fits ``context_chars``, ``request_fit``)."""
    sessions = ranked_sessions(arm, conversation, question)
    items = [_row(session.key, session_text(session)[0], session.at.isoformat(), 'session') for session in sessions]
    row = _base(question, arm, None if arm == 'full-context' else int(budget))
    if arm == 'full-context':
        if context_chars is None:
            raise ValueError('full-context needs the reader\'s context size in characters')
        selected, body = items, _whole(items)
    else:
        selected, body = pack(items, int(budget))
    by_key = {session.key: session for session in sessions}
    sections = [{'id': SECTION, 'title': TITLE, 'body': body}] if body else []
    row.update(returned_sessions=[session.key for session in sessions],
               shown_sessions=[item['_item'] for item in selected],
               shortened=[item['_item'] for item in selected if item.get('excerpt_truncated')],
               shown_turns=[[item['_item'], index] for item in selected
                            for index in _visible_turns(by_key[item['_item']], item)],
               sections=sections)
    if arm == 'full-context':
        row.update(context_chars=int(context_chars), **request_fit(question, sections, int(context_chars)))
    return row


# ------------------------------------------------------------------ stock Hermes session search

def hermes_sessions(conversation) -> tuple:
    """The conversation as Hermes' session import reads it (``SessionDB.import_sessions``): one session per dataset
    session, in time order, each at its own time with its turns a second apart, under a neutral id; and the map
    from those ids back to the dataset's session keys."""
    payload, ids = [], {}
    for number, session in enumerate(conversation.sessions, 1):
        sid = f'session-{number}'
        ids[sid] = session.key
        start = session.at.timestamp()
        payload.append({'id': sid, 'source': 'cli', 'started_at': start,
                        'ended_at': start + max(0, len(session.turns) - 1),
                        'messages': [{'role': turn.role, 'content': turn.content, 'timestamp': start + index}
                                     for index, turn in enumerate(session.turns)]})
    return payload, ids


def search_query(arm: str, question) -> str:
    """What the arm passes the tool: the question as asked, or its words joined by OR, in order, each once."""
    if arm == 'session-search':
        return question.question
    if arm == 'session-search-or':
        return ' OR '.join(dict.fromkeys(tokens(question.question)))
    raise ValueError(f'{arm} is not a session-search arm')


@contextmanager
def _utc():
    before = os.environ.get('TZ')
    os.environ['TZ'] = 'UTC'
    time.tzset()
    try:
        yield
    finally:
        if before is None:
            os.environ.pop('TZ', None)
        else:
            os.environ['TZ'] = before
        time.tzset()


def session_search(conversation, question, directory: Path, *, arm: str = 'session-search') -> tuple:
    """Stock Hermes' ``session_search`` called once (``search_query``) over a fresh store in ``directory``:
    ``(result, ids)``, the tool's parsed result and the id map (``hermes_sessions``)."""
    from hermes_state import SessionDB
    from tools.session_search_tool import session_search as tool
    payload, ids = hermes_sessions(conversation)
    Path(directory).mkdir(parents=True, exist_ok=True)
    with _utc():
        db = SessionDB(db_path=Path(directory) / 'state.db')
        try:
            imported = db.import_sessions(payload)
            if not imported.get('ok', True) or imported.get('errors'):
                raise RuntimeError('the Hermes session import refused the history '
                                   f'({len(imported.get("errors") or [])} errors)')
            result = json.loads(tool(query=search_query(arm, question), db=db))
        finally:
            db.close()
    return result, ids


def search_row(question, budget: int, result: Mapping[str, Any], ids: Mapping[str, str], *,
               arm: str = 'session-search') -> dict:
    """A baseline row from a session-search result: its entries packed in the tool's order."""
    entries = [entry for entry in result.get('results') or [] if isinstance(entry, dict)]
    items = [_row(f'entry-{number}', json.dumps(entry, ensure_ascii=False), None, 'session_search_result')
             for number, entry in enumerate(entries)]
    selected, body = pack(items, int(budget))
    shown = {item['_item'] for item in selected}
    returned = [ids.get(str(entry.get('session_id'))) for entry in entries]
    row = _base(question, arm, int(budget))
    row.update(query=search_query(arm, question), returned_sessions=[key for key in returned if key],
               shown_sessions=[key for item_key, key in zip((item['_item'] for item in items), returned)
                               if item_key in shown and key],
               shortened=[item['_item'] for item in selected if item.get('excerpt_truncated')],
               entries=len(entries), entries_with_ids=sum(bool(key) for key in returned), tool_result=result,
               sections=[{'id': SECTION, 'title': TITLE, 'body': body}] if body else [])
    return row


# ------------------------------------------------------------------ scoring a row

def score(row: Mapping[str, Any]) -> dict:
    """A baseline row scored from the row alone, in ``runner.score``'s shape."""
    from colonymind.memory import blocks
    chars = sum(blocks.cost(section.get('title') or section['id'], section['body']) for section in row['sections'])
    if row['arm'] in SEARCH_ARMS and row.get('entries') and not row.get('entries_with_ids'):
        reach = scoring.span_reach(row['answer_spans'], [section['body'] for section in row['sections']])
    else:
        reach = scoring.session_reach(row['gold_sessions'], row['shown_sessions'])
    result = {'question': row['question'], 'conversation': row['conversation'], 'type': row['type'],
              'arm': row['arm'], 'reach': reach, 'items': len(row['shown_sessions']), 'chars': chars,
              'budget': row.get('budget_chars'), 'returned_chars': chars, 'cut': False}
    if 'shown_turns' in row and row['gold_turns']:
        result['turn_reach'] = scoring.reach([{f'{s}:{t}'} for s, t in row['gold_turns']],
                                             [{f'{s}:{t}'} for s, t in row['shown_turns']])
    return result


# ------------------------------------------------------------------ the command line

def _key(row: Mapping[str, Any]) -> tuple:
    return (row['question'], row['arm'], row.get('budget_chars'))


def build(arm: str, dataset, questions, *, budgets=BUDGETS, capture: Path, context_chars: Optional[int] = None,
          work: Optional[Path] = None) -> dict:
    """Append the arm's rows for ``questions`` to ``capture`` (rows already there are kept)."""
    capture = Path(capture)
    done = {_key(json.loads(line)) for line in capture.read_text(encoding='utf-8').splitlines()
            if line.strip()} if capture.is_file() else set()
    counts = {'written': 0, 'already': 0}
    for question in questions:
        conversation = dataset.conversations[question.conversation]
        for budget in ((None,) if arm == 'full-context' else budgets):
            if (question.id, arm, budget) in done:
                counts['already'] += 1
                continue
            if arm in SEARCH_ARMS:
                directory = Path(tempfile.mkdtemp(prefix='session-search-', dir=work))
                try:
                    result, ids = session_search(conversation, question, directory, arm=arm)
                finally:
                    shutil.rmtree(directory, ignore_errors=True)
                row = search_row(question, budget, result, ids, arm=arm)
            else:
                row = session_row(arm, conversation, question, budget, context_chars=context_chars)
            with capture.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n')
            counts['written'] += 1
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, choices=('longmemeval_s', 'beam'))
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--question', action='append', help='a question id; repeat for more')
    parser.add_argument('--subset', type=Path, help='a subset record (datasets subset --out)')
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--budget', type=int, action='append')
    parser.add_argument('--context-chars', type=int,
                        help=f'full-context: the reader\'s context, in characters at {CHARS_PER_TOKEN} a token (the '
                             'whole request and the reserved output must fit)')
    parser.add_argument('--work', type=Path, help='session-search: where each question\'s store is built')
    parser.add_argument('--capture', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.arm == 'full-context' and not args.context_chars:
        parser.error('full-context needs --context-chars')
    dataset = runner.load_dataset(args.dataset, args.data_dir, args.offline)
    ids = list(args.question or [])
    if args.subset:
        ids += json.loads(Path(args.subset).read_text(encoding='utf-8'))['ids']
    wanted = set(ids)
    questions = [question for question in dataset.questions if not wanted or question.id in wanted]
    counts = build(args.arm, dataset, questions, budgets=tuple(args.budget or BUDGETS), capture=args.capture,
                   context_chars=args.context_chars, work=args.work)
    print(json.dumps(counts), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
