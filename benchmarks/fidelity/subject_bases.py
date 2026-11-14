"""Claims whose subject an earlier message named, read from formed stores: how many a build forms, and whether each
names the right subject.

    python benchmarks/fidelity/subject_bases.py --stores <store or directory> [...] --out <results.json> [--judge]

A store is a public memory benchmark replay (``benchmarks/public_memory/replay.py``: a directory holding ``replay.json``
and ``instance/turn-idempotency.db``), a directory whose subdirectories are such stores, or a ledger file itself. Each
ledger is opened read-only and never written. For each store the report counts every claim formed, by extraction
version, and lists each claim whose subject an earlier message named (``subject_basis_message``): the claim, the words
that named its subject, the whole earlier message, the conversation it is judged in (``conversation``), the claim's own
message, whether the claim is live, and whether its basis still stands (whether the readers would show it).

The conversation is every message of the claim's conversation, the speaker's and the assistant's, from the first
earlier message its extraction is offered (or the basis message, when that came first) to the last before the claim's
own message: a message between them naming another subject ("My mother is staying with us.") is what can make a
pronoun someone else's or leave it open. At most ``CONVERSATION_MESSAGES`` of them, the nearest; the basis message is
always there, first when the conversation does not hold it (a correction in a later conversation keeps the basis of the
claim it corrects).

With ``--judge`` each listed claim is judged as the context fidelity suite judges one (``suites.CONTEXT_SYSTEM``), in
that conversation, each earlier message with its role (``judge_system``). The claim is correct when the value is its
own subject's (``own``); a value of another subject (``other``), or of a subject the conversation leaves open
(``ambiguous``), is a trap. The judge is ``COLONYMIND_BENCH_JUDGE_BASE_URL`` and ``COLONYMIND_BENCH_JUDGE_MODEL``
(``COLONYMIND_BENCH_JUDGE_API_KEY`` optional), one call per claim at temperature 0; never a model a live instance uses.

Run it on the stores ``benchmarks/public_memory/runner.py replay`` formed for the dev slice's single-session-user
questions, once per build (the version before and this one): the secondary measurement of pronoun subjects beside the
borrowed-value suites. Run it with the interpreter of the build under test.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import closing
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Dict, Iterable, List, Optional

HERE = Path(__file__).resolve().parent
LEDGER = Path('instance') / 'turn-idempotency.db'
MANIFEST = 'replay.json'
TRAPS = ('other', 'ambiguous')
#: The most messages a claim is judged in, the nearest to its own (my choice: the four earlier user messages an
#: extraction is offered, their replies and room for a longer exchange between them).
CONVERSATION_MESSAGES = 16
#: The suite judges a conversation of the speaker's messages alone; a formed store's conversation has the assistant's
#: replies too, and a reply can name the subject a pronoun then means ("How is your mother?").
SPEAKER_ONLY = ("The earlier messages are the same speaker's, in order, and the last message may refer back to someone "
                "or something they named.")
WITH_ROLES = ("The earlier messages are the conversation before the last message, in order, each with its role: the "
              "speaker's own (\"user\") and the assistant's replies (\"assistant\"). The last message is the speaker's "
              "and may refer back to someone or something either of them named.")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f'fidelity_{name}', HERE / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def stores(paths: Iterable[Any]) -> List[Dict[str, Any]]:
    """``{name, ledger, questions}`` for each store the paths name: a ledger file, a replay directory, or a directory
    of replay directories (one level). A path that is none of these is refused."""
    found = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            found.append({'name': path.name, 'ledger': path, 'questions': []})
        elif (path / LEDGER).is_file():
            manifest = path / MANIFEST
            questions = json.loads(manifest.read_text(encoding='utf-8')).get('questions', []) if manifest.is_file() \
                else []
            found.append({'name': path.name, 'ledger': path / LEDGER, 'questions': list(questions)})
        elif path.is_dir() and any((child / LEDGER).is_file() for child in path.iterdir()):
            found.extend(stores(sorted(child for child in path.iterdir() if (child / LEDGER).is_file())))
        else:
            raise ValueError(f'{path} is not a store, a directory of stores or a ledger')
    return found


def _message(conn, turn_id: str, message_hash: str) -> Optional[str]:
    from colonymind.turns.idempotency import source_message_hash
    row = conn.execute('SELECT session_id,messages_json FROM turn_sources WHERE turn_id=?', (turn_id,)).fetchone()
    if row is None:
        return None
    return next((message.get('content') for message in json.loads(row['messages_json'])
                 if source_message_hash(row['session_id'], message) == message_hash
                 and isinstance(message.get('content'), str)), None)


class _ReadOnly:
    """A store's ledger as the claim readers open it, read-only: one connection per call."""

    def __init__(self, uri: str) -> None:
        self.uri = uri

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn


def conversation(conn, uri: str, turn_id: str, message_hash: str, basis: Dict[str, Any]) -> List[Dict[str, str]]:
    """The conversation a claim named from an earlier message is judged in (the module's description), oldest first:
    ``[{role, text}]``, the speaker's (``user``) and the assistant's messages with text."""
    from colonymind.beliefs.source_projection import SourceClaimProjection
    from colonymind.turns.idempotency import source_message_hash
    source = conn.execute('SELECT * FROM turn_sources WHERE turn_id=?', (turn_id,)).fetchone()
    message = None if source is None else next((m for m in json.loads(source['messages_json'])
                                                if source_message_hash(source['session_id'], m) == message_hash), None)
    if message is None:
        return []
    offered = SourceClaimProjection(_ReadOnly(uri)).earlier_messages(dict(source), message)
    occurrence = 'coalesce(julianday({0}.occurred_at),julianday({0}.ingested_at))'.format
    rows = conn.execute(f'''SELECT s.turn_id,s.session_id,s.messages_json FROM turn_sources s, turn_sources c
        WHERE c.turn_id=? AND s.contact_id=c.contact_id AND s.session_id=c.session_id AND s.scope='person'
        AND NOT EXISTS (SELECT 1 FROM source_attribution_invalidations i WHERE i.source_id=s.turn_id)
        AND NOT EXISTS (SELECT 1 FROM source_projection_erasures e WHERE e.turn_id=s.turn_id)
        AND ({occurrence('s')},s.ingested_at,s.turn_id)<=({occurrence('c')},c.ingested_at,c.turn_id)
        ORDER BY {occurrence('s')},s.ingested_at,s.turn_id''', (turn_id,)).fetchall()
    said = []
    for row in rows:
        messages = [((row['turn_id'], source_message_hash(row['session_id'], raw)), raw)
                    for raw in json.loads(row['messages_json'])]
        if row['turn_id'] == turn_id:
            said.extend(messages[:[identity for identity, _ in messages].index((turn_id, message_hash))])
            break
        said.extend(messages)
    named = (basis.get('turn_id'), basis.get('message_hash'))
    marks = {named, *((item['turn_id'], item['message_hash']) for item in offered[:1])}
    start = min((index for index, (identity, _) in enumerate(said) if identity in marks), default=len(said))
    kept = [(identity, {'role': raw['role'], 'text': raw['content']}) for identity, raw in said[start:]
            if raw.get('role') in {'user', 'assistant'} and isinstance(raw.get('content'), str)
            and raw['content'].strip()][-CONVERSATION_MESSAGES:]
    if all(identity != named for identity, _ in kept):
        text = _message(conn, *named)
        if text:
            kept.insert(0, (named, {'role': 'user', 'text': text}))
    return [entry for _, entry in kept]


def read_store(store: Dict[str, Any]) -> Dict[str, Any]:
    """One store's counts and its claims named from an earlier message, read-only. The readers of such a claim
    (``message_subject_basis``, the offered earlier messages) are only a build's that forms them, imported for the
    first one: the build before reads its own stores here too."""
    uri = Path(store['ledger']).resolve().as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        versions: Counter = Counter()
        named = []
        for row in conn.execute('SELECT c.*,s.contact_id FROM source_claims c JOIN turn_sources s '
                                'ON s.turn_id=c.turn_id ORDER BY c.id'):
            data = json.loads(row['data_json'])
            versions[str(data.get('extraction_version') or 'unknown')] += 1
            basis = data.get('subject_basis_message')
            if not isinstance(basis, dict):
                continue
            from colonymind.beliefs.source_projection import message_subject_basis
            named.append({
                'claim_id': row['id'], 'extraction_version': data.get('extraction_version'),
                **{key: data.get(key) for key in ('subject', 'predicate', 'value', 'evidence', 'operation')},
                'named_by': basis.get('evidence'), 'earlier_message': _message(conn, basis.get('turn_id'),
                                                                               basis.get('message_hash')),
                'conversation': conversation(conn, uri, row['turn_id'], row['message_hash'], basis),
                'message': _message(conn, row['turn_id'], row['message_hash']),
                'live': not row['superseded_by'] and not row['retracted_by'],
                'basis_stands': message_subject_basis(conn, data, contact_id=row['contact_id']) is not None})
    return {'store': store['name'], 'questions': store['questions'], 'claims': sum(versions.values()),
            'by_version': dict(sorted(versions.items())), 'named': named}


def judge_system(suites) -> str:
    """The suite's judge instructions (``suites.CONTEXT_SYSTEM``) for a conversation whose earlier messages carry their
    roles (``WITH_ROLES``); refused when the suite no longer describes its earlier messages in the words replaced."""
    if suites.CONTEXT_SYSTEM.count(SPEAKER_ONLY) != 1:
        raise ValueError('the suite no longer describes its earlier messages as this report replaces them')
    return suites.CONTEXT_SYSTEM.replace(SPEAKER_ONLY, WITH_ROLES)


async def judge_named(report: Dict[str, Any], judge) -> None:
    """Each listed claim judged in its conversation (``conversation``, then its own message), in place: ``holder``,
    ``trap`` and ``correct``, or ``unscored`` with its reason."""
    suites = _load('suites')
    system = judge_system(suites)
    for store in report['stores']:
        for row in store['named']:
            if not row['conversation'] or not row['message']:
                row.update(grade={'scored': False, 'reason': 'no_text'})
                continue
            claim = {key: str(row.get(key) or '') for key in ('subject', 'predicate', 'value')}
            verdict = await judge(system, {
                'earlier_messages': row['conversation'], 'last_message': row['message'], 'facts': [],
                'claims': [{'index': 0, **claim}]}, suites.CONTEXT_SCHEMA)
            graded = suites.grade_context({'facts': [], 'expect': []}, [claim], verdict)
            if not graded['scored']:
                row.update(grade={'scored': False, 'reason': graded['reason']})
                continue
            holder = verdict['claims'][0]['holder']
            row.update(grade={'scored': True, 'holder': holder, 'trap': holder in TRAPS, 'correct': holder == 'own'})


def summarize(report: Dict[str, Any]) -> Dict[str, Any]:
    suites = _load('suites')
    named = [row for store in report['stores'] for row in store['named']]
    scored = [row for row in named if (row.get('grade') or {}).get('scored')]
    traps = sum(row['grade']['trap'] for row in scored)
    return {'stores': len(report['stores']), 'claims': sum(store['claims'] for store in report['stores']),
            'named': len(named), 'named_live': sum(row['live'] for row in named),
            'basis_stands': sum(row['basis_stands'] for row in named), 'judged': len(scored),
            'unscored': sum(1 for row in named if 'grade' in row and not row['grade']['scored']),
            'traps': traps, 'trap_rate': suites.wilson(traps, len(scored)),
            'correct_rate': suites.wilson(sum(row['grade']['correct'] for row in scored), len(scored)),
            'stores_with_a_trap': sorted({store['store'] for store in report['stores'] for row in store['named']
                                          if (row.get('grade') or {}).get('trap')})}


async def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--stores', nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--judge', action='store_true', help='judge each claim named from an earlier message')
    args = parser.parse_args(argv)
    try:
        import colonymind  # noqa: F401  (the build under test: the interpreter's own)
    except ImportError:
        sys.path.insert(0, str(HERE.parents[1] / 'sidecar'))
    from colonymind.beliefs.source_claims import EXTRACTION_VERSION
    report = {'build': {'extraction_version': EXTRACTION_VERSION},
              'stores': [read_store(store) for store in stores(args.stores)]}
    if args.judge:
        missing = [name for name in ('COLONYMIND_BENCH_JUDGE_BASE_URL', 'COLONYMIND_BENCH_JUDGE_MODEL')
                   if not os.environ.get(name)]
        if missing:
            raise ValueError('Missing benchmark variables: ' + ', '.join(missing))
        from colonymind.router.router import LLMRouter
        run, suites = _load('run'), _load('suites')
        router = LLMRouter(tiers={})
        router.configure(run.router_config(os.environ['COLONYMIND_BENCH_JUDGE_BASE_URL'],
                                           os.environ['COLONYMIND_BENCH_JUDGE_MODEL'],
                                           os.environ.get('COLONYMIND_BENCH_JUDGE_API_KEY', ''), temperature=0))
        judge = suites.Judge(router)
        await judge_named(report, judge)
        report['build']['judge'] = os.environ['COLONYMIND_BENCH_JUDGE_MODEL']
        report['judge_calls'] = judge.calls
    report['summary'] = summarize(report)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps(report['summary'], indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
