"""How a store holds each changed or disputed value: the knowledge-update storage outcome, read from the store alone.

For a question whose trap is a changed value (``update``) or two statements that disagree (``conflict``,
``labels.trap_kind``), the outcome says how the store holds what its evidence said, from the store's own links:
the replay's map from each evidence turn to the message it was stored as (``replay.json``), the claims anchored at or
witnessed by those messages (``source_claims``, ``source_claim_witnesses``), the links a claim carries
(``superseded_by``, ``retracted_by``) and the relations the night decided between two claims (``mind.db``). No text is
read or matched. The first that holds is the outcome:

- ``superseded``: an evidence claim was replaced or withdrawn; the store holds the old value as no longer current.
- ``conflict``: two live evidence claims the night decided cannot both stand (a ``replaces`` decision) and neither was
  replaced: the store keeps both and asks which holds.
- ``both_current``: two different live values, each from a different evidence message, with nothing settling them:
  both read as current. A value is a claim's value key: a claim witnessed by several evidence messages is one value
  (the witnesses are its provenance), and so are two claims filed under one value key.
- ``single_value``: one live value however many evidence messages state it, or values one evidence message alone
  states (``latest_live`` says whether the latest evidence message has a live claim).
- ``not_stored``: no claim from any evidence message (an import-only store holds none).

Counts beside it: the evidence turns, how many the store held, the evidence claims, how many are live, replaced,
withdrawn, the decided ``replaces`` pairs among them, the evidence messages with a live claim, the distinct live
values, and whether the latest evidence message the store held has one.

Each arm reads one store (``runner.ARMS``): the formed store, the copy before its night, or the import-only store.
``read`` writes one row per question and store for every store a directory holds.

    python benchmarks/public_memory/storage.py --dataset longmemeval_s --stores DIR --output storage.jsonl
"""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Iterable, Mapping, Optional

# Support both package imports and the documented direct script commands.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.public_memory import labels, runner
replay = runner.replay

PROTOCOL = 'public-memory-storage-1'
SUPERSEDED, CONFLICT, BOTH_CURRENT, SINGLE_VALUE, NOT_STORED = (
    'superseded', 'conflict', 'both_current', 'single_value', 'not_stored')
OUTCOMES = (SUPERSEDED, CONFLICT, BOTH_CURRENT, SINGLE_VALUE, NOT_STORED)
# The stores a directory can hold, and where their instance lives in it.
HOMES = {'formed': Path(replay.INSTANCE), 'before_night': Path(replay.BEFORE_NIGHT) / replay.INSTANCE,
         'import_only': Path(replay.INSTANCE)}


def _rows(conn, sql: str, args: Iterable[Any]) -> list:
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(sql, list(args))]
    except sqlite3.OperationalError:
        return []           # a store without the table holds none of its rows


def _read_only(path: Path) -> sqlite3.Connection:
    """``path`` opened read-only; a file URI needs an absolute path, so a relative one (``--stores stores``) is
    resolved first."""
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)


def evidence_messages(manifest: Mapping[str, Any], question) -> list:
    """The stored message hash of each evidence turn the store held, in the question's evidence order (time order)."""
    turns = {(row['session'], row['index']): row for row in manifest.get('turns') or []}
    hashes = []
    for evidence in question.evidence:
        row = turns.get((evidence.session, evidence.turn))
        if row is not None and row.get('message_hash') and row['message_hash'] not in hashes:
            hashes.append(row['message_hash'])
    return hashes


def evidence_claims(ledger: Path, hashes) -> dict:
    """``{claim id: {'live': bool, 'superseded': bool, 'withdrawn': bool, 'value': str, 'messages': set}}``: every
    claim anchored at or witnessed by one of the ``hashes``, its value key, and which of them it belongs to."""
    if not hashes or not Path(ledger).is_file():
        return {}
    marks = ','.join('?' for _ in hashes)
    with closing(_read_only(ledger)) as conn:
        anchored = _rows(conn, f'SELECT id,message_hash,value_key,superseded_by,retracted_by FROM source_claims '
                               f'WHERE message_hash IN ({marks})', hashes)
        witnessed = _rows(conn, f'SELECT claim_id,message_hash FROM source_claim_witnesses '
                                f'WHERE message_hash IN ({marks})', hashes)
        extra = sorted({row['claim_id'] for row in witnessed} - {row['id'] for row in anchored})
        if extra:
            anchored += _rows(conn, f'SELECT id,message_hash,value_key,superseded_by,retracted_by FROM source_claims '
                                    f'WHERE id IN ({",".join("?" for _ in extra)})', extra)
    wanted = set(hashes)
    claims = {}
    for row in anchored:
        claim = claims.setdefault(row['id'], {'superseded': bool(row['superseded_by']),
                                              'withdrawn': bool(row['retracted_by']), 'value': row['value_key'],
                                              'messages': set()})
        if row['message_hash'] in wanted:
            claim['messages'].add(row['message_hash'])
    for row in witnessed:
        if row['claim_id'] in claims:
            claims[row['claim_id']]['messages'].add(row['message_hash'])
    for claim in claims.values():
        claim['live'] = not claim['superseded'] and not claim['withdrawn']
    return claims


def replaces_pairs(mind_db: Path, ids) -> int:
    """How many pairs of ``ids`` the night decided ``replaces`` (in either order)."""
    ids = sorted(ids)
    if len(ids) < 2 or not Path(mind_db).is_file():
        return 0
    from colonymind.mind.consolidate import RELATION_KEY, _pair_id
    keys = {}
    for index, first in enumerate(ids):
        for second in ids[index + 1:]:
            keys[RELATION_KEY + _pair_id(first, second)] = (first, second)
            keys[RELATION_KEY + _pair_id(second, first)] = (first, second)
    names = sorted(keys)
    with closing(_read_only(mind_db)) as conn:
        found = []
        for start in range(0, len(names), 500):
            chunk = names[start:start + 500]
            found += _rows(conn, f'SELECT key,text FROM mind_state WHERE key IN ({",".join("?" for _ in chunk)})',
                           chunk)
    return len({keys[row['key']] for row in found if row.get('text') == 'replaces'})


def live_values(claims: Mapping[str, Mapping[str, Any]], hashes) -> dict:
    """``{value key: the evidence messages stating it}`` over the live claims: one entry per value, however many
    claims or witnesses state it."""
    wanted, values = set(hashes), {}
    for claim in claims.values():
        if claim['live']:
            values.setdefault(claim['value'], set()).update(claim['messages'] & wanted)
    return values


def classify(claims: Mapping[str, Mapping[str, Any]], hashes, replaces: int) -> str:
    """The outcome (module docstring) of an evidence's claims."""
    if not claims:
        return NOT_STORED
    if any(not claim['live'] for claim in claims.values()):
        return SUPERSEDED
    if replaces:
        return CONFLICT
    stated = list(live_values(claims, hashes).values())
    # Two different values compete when a different evidence message stands behind each.
    competing = any(first != second for index, one in enumerate(stated) for other in stated[index + 1:]
                    for first in one for second in other)
    return BOTH_CURRENT if competing else SINGLE_VALUE


def outcome(root: Path, question, *, store: str = 'formed') -> Optional[dict]:
    """The storage row of ``question`` in the ``store`` a replay directory ``root`` holds, or None for a question
    whose trap is neither a changed value nor a conflict."""
    kind = labels.trap_kind(question)
    if kind not in labels.LABELLED:
        return None
    root = Path(root)
    manifest = replay.load_manifest(root)
    home = root / HOMES[store]
    hashes = evidence_messages(manifest, question)
    claims = evidence_claims(home / 'turn-idempotency.db', hashes)
    live = [claim_id for claim_id, claim in claims.items() if claim['live']]
    replaces = replaces_pairs(home / 'mind.db', live)
    messages_live = set().union(*(claims[claim_id]['messages'] for claim_id in live)) if live else set()
    counts = {'evidence': len(question.evidence), 'stored': len(hashes), 'claims': len(claims), 'live': len(live),
              'superseded': sum(claim['superseded'] for claim in claims.values()),
              'withdrawn': sum(claim['withdrawn'] for claim in claims.values()), 'replaces': replaces,
              'messages_live': len(messages_live), 'values_live': len(live_values(claims, hashes)),
              'latest_live': bool(hashes) and hashes[-1] in messages_live}
    return {'protocol': PROTOCOL, 'question': question.id, 'conversation': question.conversation,
            'type': question.type, 'kind': kind, 'store': store, 'outcome': classify(claims, hashes, replaces),
            'counts': counts}


def stores_in(root: Path) -> list:
    """The stores a replay directory holds: the import-only store, or the formed store and its copy before the
    night when that copy exists."""
    manifest = replay.load_manifest(root)
    if manifest.get('store') == 'import_only':
        return ['import_only']
    return ['formed', *(['before_night'] if (Path(root) / HOMES['before_night']).is_dir() else [])]


def read(stores: Path, questions, output: Path) -> dict:
    """Append one row per question and store to ``output``, for each question whose conversation has a directory
    under ``stores`` (named as ``runner.py replay`` names it)."""
    written = 0
    with Path(output).open('a', encoding='utf-8') as stream:
        for question in questions:
            root = Path(stores) / runner.safe_name(question.conversation)
            if not (root / replay.MANIFEST).is_file():
                continue
            for store in stores_in(root):
                row = outcome(root, question, store=store)
                if row is not None:
                    stream.write(json.dumps(row, sort_keys=True) + '\n')
                    written += 1
    return {'rows': written}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, choices=('longmemeval_s', 'beam'))
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--stores', type=Path, required=True, help='the directory the stores were formed in')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    dataset = runner.load_dataset(args.dataset, args.data_dir, args.offline)
    print(json.dumps(read(args.stores, dataset.questions, args.output)), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
