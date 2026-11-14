"""A forget's canonical closure costs lookups, not sources times rules.

On a real owner history (38k sources, 281 MB of messages, 74 erasure rules) the
closure tested every message against every rule on every pass, parsed every
source on every pass and scanned the whole full-text table once per changed
source: 5.1 s for one forget, above the adapter's timeout. These tests pin the
cheaper closure: indexed rules give exactly the pairwise answers, a source no
rule can reach is never parsed, and the lexical index is rewritten in one scan.
"""

import json
import random

import pytest

from colonymind.turns import idempotency
from colonymind.turns.idempotency import TurnIdempotencyLedger, _ErasureRules, source_message_hash


def pairwise_causes(messages, session_id, rules):
    """The closure's original definition, every message against every rule."""
    causes = set()
    for message in messages:
        refs = (message.get('_supplied_sources', []) if message.get('role') == 'assistant' else
                message.get('_observation_sources', []) if message.get('role') == 'tool' and
                message.get('_native_tool_observation') == 'native-tool-observation-v1' else [])
        for rule in rules:
            exact = (rule['session_id'] == session_id and source_message_hash(session_id, message)
                     in json.loads(rule['message_hashes_json']))
            dependent = any(ref.get('source_id') == rule['turn_id'] and (
                rule['whole_source'] or ref.get('source_version') == rule['source_version'])
                for ref in refs if isinstance(ref, dict))
            if exact or dependent:
                causes.add(rule['turn_id'])
    return causes


def test_indexed_rules_give_the_pairwise_causes():
    rng = random.Random(20260924)
    sessions, turns = ['s1', 's2', 's3'], ['t1', 't2', 't3', 't4']
    versions = ['a' * 64, 'b' * 64, 'c' * 64]
    contents = ['one', 'two', 'three']

    def message():
        role = rng.choice(['user', 'assistant', 'tool', 'system'])
        value = {'role': role, 'content': rng.choice(contents)}
        key = rng.choice(['_supplied_sources', '_observation_sources', None])
        if key:
            value[key] = [rng.choice([
                {'source_id': rng.choice(turns), 'source_version': rng.choice(versions)},
                {'source_id': rng.choice(turns)},                      # no version
                {'source_id': 7, 'source_version': versions[0]},        # not a string
                {'source_id': rng.choice(turns), 'source_version': ['x']},
                'not-a-dict'])
                for _ in range(rng.randint(0, 3))]
        if role == 'tool' and rng.random() < 0.6:
            value['_native_tool_observation'] = 'native-tool-observation-v1'
        return value

    for _ in range(3000):
        rules = []
        for _ in range(rng.randint(0, 5)):
            session = rng.choice(sessions)
            whole = rng.random() < 0.5
            hashes = [source_message_hash(session, {'role': rng.choice(['user', 'assistant']),
                                                     'content': rng.choice(contents)})
                      for _ in range(rng.randint(0, 2))]
            rules.append({'turn_id': rng.choice(turns), 'session_id': session, 'message_hashes_json': json.dumps(hashes),
                          'whole_source': 1 if whole else 0, 'source_version': None if whole else rng.choice(versions)})
        messages = [message() for _ in range(rng.randint(1, 3))]
        session = rng.choice(sessions)
        index = _ErasureRules(rules)
        assert TurnIdempotencyLedger._erasure_causes(messages, session, index) == pairwise_causes(messages, session, rules)
        assert TurnIdempotencyLedger._erasure_causes(messages, session, rules) == pairwise_causes(messages, session, rules)
        assert (TurnIdempotencyLedger._retained_messages(messages, session, rules)
                == [m for m in messages if not pairwise_causes([m], session, rules)])


def history(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    ledger.record_source('target', contact_id='owner', session_id='s-target',
                         messages=[{'role': 'user', 'content': 'The neutral locker code is 4417.'}])
    version = ledger.source_references(['target'], contact_id='owner', session_id='s-target')[0]['source_version']
    ledger.record_source('answer', contact_id='owner', session_id='s-answer', messages=[
        {'role': 'user', 'content': 'What was the locker code?'},
        {'role': 'assistant', 'content': 'It was 4417.',
         '_supplied_sources': [{'source_id': 'target', 'source_version': version}]}])
    ledger.record_source('copy', contact_id='owner', session_id='s-target', scope='session',
                         messages=[{'role': 'user', 'content': 'The neutral locker code is 4417.'}])
    for index in range(20):
        ledger.record_source(f'other-{index}', contact_id='owner', session_id=f's-{index}',
                             messages=[{'role': 'user', 'content': f'Unrelated neutral note {index} about gardens.'}])
    with ledger._connect() as conn:
        raw = {row['turn_id']: row['messages_json'] for row in conn.execute('SELECT * FROM turn_sources')}
    return ledger, raw
