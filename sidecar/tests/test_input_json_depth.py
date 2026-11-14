"""Deeply nested stored and owner text follows the invalid-JSON fallback."""
from contextlib import closing

import pytest

from colonymind.memory.branches import BranchStore
from colonymind.turns.hermes_history import source_message
from colonymind.turns.idempotency import TurnIdempotencyLedger


# Deep enough to raise RecursionError in the JSON decoder on Python 3.11, 3.12 and 3.13.
DEEP_TEXT = '[' * 100000 + ']'


@pytest.mark.parametrize('role', ['user', 'assistant'])
def test_stored_message_keeps_deep_json_as_literal_text(role):
    row = {'id': 1, 'session_id': 'session-1', 'role': role, 'content': DEEP_TEXT,
           'active': True, 'compacted': False, '_compressed_summary': False,
           'observed': False, 'effect_disposition': None, 'display_kind': None,
           'tool_calls': None, 'finish_reason': None, 'timestamp': 0, 'platform_message_id': 'message-1'}
    binding = {'platform': 'telegram', 'actor_id': 'contact-1', 'chat_id': 'chat-1',
               'review_evidence': 'fixture-binding'}
    (message, occurred_at), reason = source_message(row, {}, binding, 'fixture')
    assert reason is None
    assert message['role'] == role and message['content'] == DEEP_TEXT
    assert message['provenance']['speaker'] == ('contact' if role == 'user' else 'assistant')
    assert occurred_at == '1970-01-01T00:00:00+00:00'


def test_deep_owner_ratings_consume_the_turn_without_labelling_it(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'turns.db')
    store = BranchStore(ledger, key=b'fixture-key')
    store.record_pair('ref-1', session_id='session-1', question='Which record?',
                      answer='First record', reference_answer='Second record')
    assert store.reaction(session_id='session-1', text=DEEP_TEXT, requester='owner',
                          turn_id='turn-1') == {'labelled': False}
    with closing(ledger._connect()) as conn:
        row = conn.execute('SELECT reaction_seen, served_correct, reference_correct FROM wm_consults').fetchone()
        assert tuple(row) == (1, None, None)
    assert store.reaction(session_id='session-1', text=DEEP_TEXT, requester='owner',
                          turn_id='turn-1') == {'labelled': False, 'replayed': True}
