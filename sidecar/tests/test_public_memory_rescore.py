"""Offline re-score of paired memory runs: reach, traps and the dead-value classification from recorded traces.

Every trace here is synthetic; nothing reads a model. Values and quotations stay out of the rendered report.
"""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from colonymind.qualification import paired_report
from colonymind.turns.idempotency import source_message_hash

BENCH = Path(__file__).resolve().parents[2] / 'benchmarks'
RESCORE = BENCH / 'public_memory' / 'paired_rescore.py'
GENERATORS = BENCH / 'paired' / 'generators'

OLD = 'The monthly budget draft review with p-11 is held in the east room.'
NEW = 'The budget draft review with p-11 is in the main hall from now on.'
HISTORY = [{'id': 'chat-01', 'at': 1, 'messages': [{'role': 'user', 'content': OLD},
                                                  {'role': 'assistant', 'content': 'Noted.'}]},
           {'id': 'email-02', 'at': 2, 'messages': [{'role': 'user', 'content': 'Long week.'},
                                                   {'role': 'assistant', 'content': 'Noted.'}]},
           {'id': 'sms-03', 'at': 3, 'messages': [{'role': 'user', 'content': NEW},
                                                 {'role': 'assistant', 'content': 'Noted.'}]}]
H_OLD = source_message_hash('chat-01', {'role': 'user', 'content': OLD})
H_NEW = source_message_hash('sms-03', {'role': 'user', 'content': NEW})
EPISODES = [{'session_id': 'owner-1', 'user': 'Nothing to do right now.'}, {'advance_clock': 86400}, {'tick': 1},
            {'session_id': 'owner-2', 'user': 'Where is the monthly budget draft review with p-11 held now?'}]


@pytest.fixture(scope='module')
def rescore():
    spec = importlib.util.spec_from_file_location('public_memory_paired_rescore', RESCORE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def artifact(expected, forbidden, gold=None, key='place'):
    spec = {'path': 'answer.json', 'format': 'json', 'forbidden': list(forbidden),
            'assertions': [{'path': [], 'op': 'keys_equal', 'value': [key]},
                           {'path': [key], 'op': 'label_one_of', 'value': [expected]}]}
    if gold is not None:
        spec['gold'] = gold
    return spec


def update_case(gold=True):
    return {'id': 'buried-correction.01',
            'inputs': {'scenario': 'buried-correction', 'episodes': copy.deepcopy(EPISODES),
                       'history': copy.deepcopy(HISTORY)},
            'oracle': {'artifacts': [artifact('main hall', ['east room'], {
                'kind': 'update', 'evidence': [{'session': 'sms-03', 'message': 0}], 'stale': ['east room']}
                if gold else None)]}}


def event(kind, data, thread='MainThread'):
    return json.dumps({'protocol': 'paired-private-trace-1', 'kind': kind, 'thread': thread, 'data': data})


def request(messages, thread='Thread-9 (<lambda>)'):
    return event('model_request', {'request_id': 1, 'payload': {'messages': messages}}, thread)


def context(*lines):
    return 'Where?\n\n<memory-context>\n## Relevant Memories\n' + '\n'.join(lines) + '\n</memory-context>'


def turn(index, session, *messages):
    return event('native_turn', {'index': index, 'session_id': session, 'kind': 'user', 'completed': True,
                                 'messages': [{'role': role, 'content': content} for role, content in messages]})


def trace(*probe_requests, before=()):
    """A trace whose last turn (index 3) is the probe: earlier requests, a tick, then the probe's requests."""
    return [event('episode', {'session_ids': ['owner-1']}),
            request([{'role': 'user', 'content': context(*before)}]) if before else event('drain', {}),
            turn(0, 'owner-1', ('user', '[Mon 2026-09-28 12:00:01 UTC] Nothing to do right now.'),
                 ('assistant', 'Noted.')),
            event('body_tick', {'index': 2}), event('episode', {'session_ids': ['owner-2']}),
            *probe_requests,
            turn(3, 'owner-2', ('user', '[Tue 2026-09-29 12:00:26 UTC] ' + EPISODES[3]['user']),
                 ('assistant', 'Written.'))]


def passage(ref, source, digest, quote):
    return '- ' + json.dumps({'evidence_ref': ref, 'source': 'turn:' + source, 'source_message_hash': digest,
                              'role': 'user', 'quote': quote})


def card(subject, predicate, *assertions, source='hermes-history:paired-history:1'):
    return '- ' + json.dumps({'kind': 'source_quote', 'source': 'turn:' + source, 'state': 'source_assertion',
                              'content': {'subject': subject, 'predicate': predicate, 'status': 'source_assertion',
                                          'assertions': [{'claim_id': f'claim:{index}', 'value': value,
                                                          'evidence_ref': reference}
                                                         for index, (value, reference) in enumerate(assertions)]}})


def quotation(source, digest, text):
    return '- ' + json.dumps({'kind': 'source_quote', 'source': 'turn:' + source, 'state': 'quotation',
                              'source_message_hash': digest, 'role': 'user'}) + ' ' + json.dumps(text)


def conversation_pair(source, digest, text):
    return '- ' + json.dumps({'kind': 'conversation_pair', 'source': 'turn:' + source, 'state': 'quotation',
                              'content': {'input': {'source_id': source, 'source_message_hash': digest,
                                                    'role': 'user', 'quote': text},
                                          'response': {'source_id': source, 'role': 'assistant',
                                                       'quote': 'Noted.'}}})


def notes_block(title, body, limit=2200):
    """A built-in notes block as the host renders it: its header states the body's own length."""
    rule = '\u2550' * 46
    usage = f'{min(100, len(body) * 100 // limit)}% \u2014 {len(body):,}/{limit:,} chars'
    return f'{rule}\n{title} [{usage}]\n{rule}\n{body}'


Q_OLD = passage('q1', 'hermes-history:paired-history:1', H_OLD, OLD)
Q_NEW = passage('q2', 'hermes-history:paired-history:5', H_NEW, NEW)


# Reach from the recorded section: identifiers only.

def test_reach_is_structural_on_the_message_hashes_the_section_cites(rescore):
    lines = trace(request([{'role': 'user', 'content': context(
        Q_OLD, Q_NEW, card('budget draft review with p-11', 'location', ('east room', 'q1'), ('main hall', 'q2')))}]))
    row = rescore.episode(update_case(), lines, completion=True, answer='{"place": "main hall"}')
    assert row['gold_basis'] == 'dataset' and row['type'] == 'buried-correction'
    assert (row['reach']['basis'], row['reach']['any'], row['reach']['first_rank']) == ('structural', True, 2)
    assert row['reach_surfaces']['injected']['any'] is True and row['reach_surfaces']['tools']['any'] is False
    assert row['reach_value']['weaker'] is True and row['reach_value']['any'] is True
    assert row['right'] is True and row['trap']['outcome'] == 'correct'


def test_the_probe_turn_alone_counts_not_earlier_requests(rescore):
    # The gold message was shown in an earlier turn's context, never in the probe's.
    lines = trace(request([{'role': 'user', 'content': context(Q_OLD)}]), before=[Q_NEW])
    row = rescore.episode(update_case(), lines, completion=False, answer='{"place": "east room"}')
    assert row['reach']['any'] is False
    assert row['trap'] == {'kind': 'update', 'outcome': 'trap', 'matched': ['east room']}


def test_a_session_search_result_reaches_by_session_and_by_message_hash(rescore):
    result = {'success': True, 'results': [
        {'session_id': 'email-02', 'messages': [{'id': 3, 'role': 'user', 'content': 'Long week.'}]},
        {'session_id': 'sms-03', 'snippet': 'main hall', 'messages': [{'id': 5, 'role': 'user', 'content': NEW}]}]}
    messages = [{'role': 'system', 'content': 'You are an agent.'},
                {'role': 'user', 'content': EPISODES[3]['user']},
                {'role': 'assistant', 'content': '', 'tool_calls': [
                    {'id': 'call-1', 'type': 'function', 'function': {'name': 'session_search', 'arguments': '{}'}},
                    {'id': 'call-2', 'type': 'function', 'function': {'name': 'write_file', 'arguments': '{}'}}]},
                {'role': 'tool', 'tool_call_id': 'call-1', 'content': json.dumps(result)},
                {'role': 'tool', 'tool_call_id': 'call-2', 'content': json.dumps({'content': NEW})}]
    row = rescore.episode(update_case(), trace(request(messages)), completion=True, answer='{"place": "main hall"}')
    assert (row['reach']['any'], row['reach']['first_rank']) == (True, 2)
    assert row['reach_surfaces']['tools']['any'] is True and row['reach_surfaces']['injected']['any'] is False
    assert (row['reach_session']['any'], row['reach_session']['first_rank']) == (True, 2)
    # A file tool's result is no memory surface: a copy of the words there reaches nothing.
    only_file = [message for message in messages if message.get('tool_call_id') != 'call-1']
    only_file[2] = dict(only_file[2], tool_calls=only_file[2]['tool_calls'][1:])
    assert rescore.episode(update_case(), trace(request(only_file)), completion=True,
                           answer='{}')['reach']['any'] is False


@pytest.mark.parametrize('lines', [
    [],  # no trace was recorded
    [event('episode', {'session_ids': ['owner-2']}),  # the probe turn ran, but no model request of it was recorded
     turn(3, 'owner-2', ('user', EPISODES[3]['user']), ('assistant', 'Written.'))],
])
def test_an_unobserved_probe_request_leaves_reach_unscored(rescore, lines):
    """The history gold resolves from the plan alone, but nothing shows what the reader was given: that is no miss."""
    row = rescore.episode(update_case(), lines, completion=True, answer='{"place": "main hall"}')
    assert row['observed'] is False
    assert (row['reach'], row['reach_surfaces'], row['reach_session']) == (None, None, None)
    assert (row['reach_value'], row['reach_value_surfaces'], row['reach_quote']) == (None, None, None)
    assert rescore.scoring.two_by_two([row])['unscored'] == 1
    # A request observed with nothing of the gold in it is a measured miss.
    seen = rescore.episode(update_case(), trace(request([{'role': 'user', 'content': 'no context'}])),
                           completion=True, answer='{"place": "main hall"}')
    assert seen['observed'] is True and seen['reach']['any'] is False and seen['reach_value']['any'] is False
    assert rescore.scoring.two_by_two([seen])['missed_right'] == 1


def test_the_weaker_checks_read_whole_memory_bodies_prose_included(rescore):
    """A resident block's paragraph is shown to the reader like any item: the value and quotation checks read it.
    Structural reach still reads only the parsed items, and prose cites nothing."""
    resident = ('## Resident\nStanding notes on the reviews.\n\nThe budget draft review with p-11\nmeets in the '
                'main hall now.\n\n## Relevant Memories\n' + Q_OLD)
    row = rescore.episode(update_case(), trace(request([{'role': 'user', 'content': context(resident)}])),
                          completion=True, answer='{"place": "main hall"}')
    assert row['reach_value']['any'] is True and row['reach_value_surfaces']['injected'] is True
    assert row['reach']['any'] is False
    # The same through the memory search tool: its result's prose, beside an escaped item.
    found = {'content': 'Two notes match.\n' + NEW + '\n' + Q_OLD}
    messages = [{'role': 'user', 'content': EPISODES[3]['user']},
                {'role': 'assistant', 'content': '', 'tool_calls': [
                    {'id': 'c1', 'type': 'function', 'function': {'name': 'colonymind_memory_search', 'arguments': '{}'}}]},
                {'role': 'tool', 'tool_call_id': 'c1', 'content': json.dumps(found)}]
    row = rescore.episode(update_case(), trace(request(messages)), completion=True, answer='{"place": "main hall"}')
    assert row['reach_value_surfaces'] == {'injected': False, 'tools': True, 'notes': False}
    assert row['reach_quote']['any'] is True and row['reach']['any'] is False


def test_a_notes_block_is_read_whole_to_the_length_its_header_states(rescore):
    """Entries are joined by the host's entry delimiter and may hold blank lines; the header states the length."""
    memory = ('Grant summary review: Friday.\n\nIt is held in the east room.\n\u00a7\nThe spare chess set lives in '
              'the top shelf.')
    user = 'Prefers minutes.\n\nWorks late on Fridays.'
    system = ('You are an agent.\n\n' + notes_block('MEMORY (your personal notes)', memory) + '\n\n'
              + notes_block('USER PROFILE (who the user is)', user, 1375) + '\n\nConversation started: Monday')
    assert rescore.notes_bodies(system) == [memory, user]
    _, _, notes = rescore.surfaces([json.loads(request([{'role': 'system', 'content': system}]))])
    assert notes == [memory, user]
    # A header that states no length cannot bound its body: it is read to the next block header or the end.
    rule = '\u2550' * 46
    bare = f'{rule}\nMEMORY (your personal notes) [1%]\n{rule}\n{memory}\n\nConversation started: Monday'
    assert rescore.notes_bodies(bare) == [memory + '\n\nConversation started: Monday']
    # Evidence in a later paragraph of an entry reaches the weaker check.
    case = update_case()
    case['oracle']['artifacts'] = [artifact('east room', [], {'kind': 'recall', 'evidence': []})]
    row = rescore.episode(case, trace(request([{'role': 'system', 'content': system},
                                               {'role': 'user', 'content': EPISODES[3]['user']}])),
                          completion=True, answer='{"place": "east room"}')
    assert row['reach_value_surfaces'] == {'injected': False, 'tools': False, 'notes': True}


def test_episode_turn_gold_is_read_from_the_recorded_turn(rescore):
    fact = 'For the record: the grant summary review with p-22 is on Friday at 10:45 in the east room.'
    case = {'id': 'fact-after-restart.01',
            'inputs': {'scenario': 'fact-after-restart', 'episodes': [
                {'session_id': 'owner-1', 'user': fact}, {'advance_clock': 86400}, {'tick': 1},
                {'session_id': 'owner-2', 'user': 'Write answer.json.'}]},
            'oracle': {'artifacts': [artifact('east room', [], {'kind': 'recall', 'evidence': [{'turn': 0}]})]}}
    said = '[Mon 2026-09-28 12:00:04 UTC] ' + fact
    digest = source_message_hash('owner-1', {'role': 'user', 'content': said})
    lines = [turn(0, 'owner-1', ('user', said), ('assistant', 'Noted.')), event('body_tick', {}),
             request([{'role': 'user', 'content': context(passage('q1', 'owner-1:x:y', digest, said))}]),
             turn(3, 'owner-2', ('user', 'Write answer.json.'))]
    row = rescore.episode(case, lines, completion=True, answer='{"place": "east room"}')
    assert row['reach']['any'] is True and row['reach_quote']['any'] is True
    # The turn never recorded: the gold cannot be resolved, so reach is not scored.
    unresolved = rescore.episode(case, lines[1:], completion=True, answer='{"place": "east room"}')
    assert unresolved['reach'] is None and unresolved['gold_unresolved'] == 1
    # A host that rewrote the fact into its own notes shows no source: only the weaker check sees it there.
    notes = ('You are an agent.\n\n' + notes_block('MEMORY (your personal notes)',
                                                    'Grant summary review: Friday 10:45, east room.')
             + '\n\nConversation started: today')
    search = {'results': [{'session_id': 'owner-1', 'messages': [{'role': 'user', 'content': 'Unrelated.'}]}]}
    stock = [turn(0, 'owner-1', ('user', said), ('assistant', 'Noted.')), event('body_tick', {}),
             request([{'role': 'system', 'content': notes}, {'role': 'user', 'content': 'Write answer.json.'},
                      {'role': 'assistant', 'content': '', 'tool_calls': [
                          {'id': 'c1', 'type': 'function', 'function': {'name': 'session_search', 'arguments': '{}'}}]},
                      {'role': 'tool', 'tool_call_id': 'c1', 'content': json.dumps(search)}]),
             turn(3, 'owner-2', ('user', 'Write answer.json.'))]
    host = rescore.episode(case, stock, completion=True, answer='{"place": "east room"}')
    assert host['reach']['any'] is False and host['reach_session']['any'] is True
    assert host['reach_value_surfaces'] == {'injected': False, 'tools': False, 'notes': True}


# The dead-value classification: why a stale value was shown as current.

@pytest.mark.parametrize('lines, expected', [
    # One key holds both values: the supersession was missed on an identical key.
    ([Q_OLD, Q_NEW, card('budget draft review with p-11', 'location', ('east room', 'q1'), ('main hall', 'q2')),
      card('budget draft review with p-11', 'location', ('east room', 'q1'))],
     ['identical_key', 'identical_key']),
    # The correction formed a claim under another key for the same fact.
    ([Q_OLD, Q_NEW, card('budget draft review with p-11', 'location', ('east room', 'q1')),
      card('budget draft review', 'venue', ('main hall', 'q2'))],
     ['different_key', 'different_key']),
    # The correction reached the section only as a quotation, with no claim shown for it.
    ([Q_OLD, card('budget draft review with p-11', 'location', ('east room', 'q1')),
      conversation_pair('hermes-history:paired-history:5', H_NEW, NEW)],
     ['raw_quotation', 'raw_quotation']),
    # The old statement reached the section only as a quotation; the correction has its claim.
    ([quotation('hermes-history:paired-history:1', H_OLD, OLD), Q_NEW,
      card('budget draft review', 'venue', ('main hall', 'q2'))],
     ['raw_quotation']),
    # The correction is not in the section at all: nothing shows its key.
    ([Q_OLD, card('budget draft review with p-11', 'location', ('east room', 'q1'))], ['unknown', 'unknown']),
])
def test_dead_value_lines_are_classified_by_the_keys_the_section_shows(rescore, lines, expected):
    row = rescore.episode(update_case(), trace(request([{'role': 'user', 'content': context(*lines)}])),
                          completion=False, answer='{"place": "east room"}')
    assert [line['class'] for line in row['dead_lines']] == expected
    assert all(line['basis'] == 'gold' for line in row['dead_lines'])


@pytest.mark.parametrize('extra, expected', [
    # The correction message also says the review is monthly. Only that fact formed a claim: the corrected place
    # reached the section as the quotation alone, and the frequency claim is never compared with the place's key.
    ([card('budget draft review with p-11', 'frequency', ('monthly', 'q2'))], ['raw_quotation', 'raw_quotation']),
    # The corrected place also formed a claim, under another key: that assertion is the one compared.
    ([card('budget draft review with p-11', 'frequency', ('monthly', 'q2')),
      card('budget draft review', 'venue', ('main hall', 'q2'))], ['different_key', 'different_key']),
])
def test_only_the_assertion_of_the_corrected_value_is_the_correction(rescore, extra, expected):
    lines = trace(request([{'role': 'user', 'content': context(
        Q_OLD, Q_NEW, card('budget draft review with p-11', 'location', ('east room', 'q1')), *extra)}]))
    row = rescore.episode(update_case(), lines, completion=False, answer='{"place": "east room"}')
    assert [line['class'] for line in row['dead_lines']] == expected


def scoped_case(stale_fields=True):
    """A three-field answer whose correction changes one field (the time) of a statement that named three."""
    first = 'The design brief handover with p-31 is on Monday at 09:15 in the main hall.'
    fix = 'Correction, only to the time of that design brief handover on Monday: it is 13:05.'
    gold = {'kind': 'update', 'evidence': [{'turn': 0}, {'turn': 1}], 'stale': ['09:15']}
    if stale_fields:
        gold['stale_fields'] = {'09:15': 'time'}
    case = {'id': 'scoped-correction.01',
            'inputs': {'scenario': 'scoped-correction', 'episodes': [
                {'session_id': 'owner-1', 'user': first}, {'session_id': 'owner-1', 'user': fix},
                {'advance_clock': 86400}, {'tick': 1}, {'session_id': 'owner-2', 'user': 'Write answer.json.'}]},
            'oracle': {'artifacts': [{'path': 'answer.json', 'format': 'json', 'forbidden': ['09:15'], 'assertions': [
                {'path': [], 'op': 'keys_equal', 'value': ['day', 'place', 'time']},
                {'path': ['day'], 'op': 'label_one_of', 'value': ['Monday']},
                {'path': ['place'], 'op': 'label_one_of', 'value': ['main hall']},
                {'path': ['time'], 'op': 'label_one_of', 'value': ['13:05']}], 'gold': gold}]}}
    said = ['[Mon 2026-09-28 12:00:04 UTC] ' + first, '[Mon 2026-09-28 12:00:20 UTC] ' + fix]
    hashes = [source_message_hash('owner-1', {'role': 'user', 'content': text}) for text in said]
    return case, said, hashes


@pytest.mark.parametrize('stale_fields, extra, expected', [
    # The correction message restates the day, and only that formed a claim: the day is another field's value,
    # so it is not the time's correction, however its key compares.
    (True, [('day', 'Monday')], 'raw_quotation'),
    # The corrected time formed a claim under another key.
    (True, [('day', 'Monday'), ('start', '13:05')], 'different_key'),
    # Gold that does not say which field the stale value held: nothing from the correction can be told apart as
    # the time's, so no key comparison is supported.
    (False, [('day', 'Monday'), ('start', '13:05')], 'unknown'),
])
def test_a_multi_field_correction_is_compared_on_the_stale_values_field(rescore, stale_fields, extra, expected):
    case, said, hashes = scoped_case(stale_fields)
    subject = 'design brief handover with p-31'
    lines = [turn(0, 'owner-1', ('user', said[0]), ('assistant', 'Noted.')),
             turn(1, 'owner-1', ('user', said[0]), ('assistant', 'Noted.'), ('user', said[1]), ('assistant', 'Noted.')),
             event('body_tick', {}),
             request([{'role': 'user', 'content': context(
                 passage('q1', 'owner-1:a:1', hashes[0], said[0]), passage('q2', 'owner-1:a:2', hashes[1], said[1]),
                 card(subject, 'time', ('09:15', 'q1'), source='owner-1:a:1'),
                 *[card(subject, predicate, (value, 'q2'), source='owner-1:a:2') for predicate, value in extra])}]),
             turn(4, 'owner-2', ('user', 'Write answer.json.'))]
    row = rescore.episode(case, lines, completion=False, answer='{"day": "Monday", "place": "main hall", '
                                                               '"time": "09:15"}')
    assert [(line['line'], line['class']) for line in row['dead_lines']] == [('claim', expected)]


def test_the_correction_is_the_evidence_after_the_stale_statement(rescore):
    """Gold may include the statement that also held the old value (a scoped correction keeps its other fields):
    only the evidence after it is the correction, so the old claim is never its own correction."""
    case, said, hashes = scoped_case()
    lines = [turn(0, 'owner-1', ('user', said[0]), ('assistant', 'Noted.')),
             turn(1, 'owner-1', ('user', said[0]), ('assistant', 'Noted.'), ('user', said[1]), ('assistant', 'Noted.')),
             event('body_tick', {}),
             request([{'role': 'user', 'content': context(
                 passage('q1', 'owner-1:a:1', hashes[0], said[0]),
                 card('design brief handover with p-31', 'time', ('09:15', 'q1'), source='owner-1:a:1'),
                 card('design brief handover with p-31', 'day', ('Monday', 'q1'), source='owner-1:a:1'),
                 conversation_pair('owner-1:a:2', hashes[1], said[1]))}]),
             turn(4, 'owner-2', ('user', 'Write answer.json.'))]
    row = rescore.episode(case, lines, completion=False, answer='{"day": "Monday", "place": "main hall", '
                                                               '"time": "09:15"}')
    # The first statement's quotation also shows the day and the place, which answer it: only the time claim is dead.
    assert [(line['line'], line['class']) for line in row['dead_lines']] == [('claim', 'raw_quotation')]
    assert row['reach']['all'] is True and row['trap']['outcome'] == 'trap'


@pytest.mark.parametrize('extra, expected', [
    # The withdrawal formed a claim, but the answer's value is "unknown": nothing says which claim is the place's.
    ([card('budget draft review with p-11', 'location', ('not known', 'q2'))], 'unknown'),
    # The withdrawal reached the section only as its quotation.
    ([], 'raw_quotation'),
])
def test_a_withdrawal_with_no_value_supports_no_key_comparison(rescore, extra, expected):
    case = update_case()
    case['oracle']['artifacts'] = [artifact('unknown', ['east room'], {
        'kind': 'abstain', 'evidence': [{'session': 'sms-03', 'message': 0}], 'stale': ['east room']})]
    lines = trace(request([{'role': 'user', 'content': context(
        Q_NEW, card('budget draft review with p-11', 'location', ('east room', 'q1')), *extra)}]))
    row = rescore.episode(case, lines, completion=False, answer='{"place": "east room"}')
    assert [line['class'] for line in row['dead_lines']] == [expected]


def test_a_forbidden_value_that_was_never_current_is_a_decoy(rescore):
    case = {'id': 'near-miss.01', 'inputs': {'scenario': 'near-miss', 'episodes': copy.deepcopy(EPISODES),
                                             'history': copy.deepcopy(HISTORY)},
            'oracle': {'artifacts': [artifact('unknown', ['east room'], {'kind': 'abstain', 'evidence': []})]}}
    lines = trace(request([{'role': 'user', 'content': context(Q_OLD)}]))
    row = rescore.episode(case, lines, completion=False, answer='{"place": "east room"}')
    assert [line['class'] for line in row['dead_lines']] == ['decoy']
    assert row['trap']['outcome'] == 'trap' and row['reach'] is None


def test_dead_lines_match_the_report_count(rescore, tmp_path):
    case = update_case()
    lines = trace(request([{'role': 'user', 'content': context(
        Q_OLD, Q_NEW, card('budget draft review with p-11', 'location', ('east room', 'q1')),
        card('budget draft review', 'venue', ('main hall', 'q2')))}]))
    row = rescore.episode(case, lines, completion=False, answer=None)
    path = tmp_path / 'runs/one/attempts' / case['id'] / 'private-trace.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text('\n'.join(lines))
    counted = paired_report._dead_values(tmp_path, [{'path': 'runs/one', 'case': case}])
    assert len(row['dead_lines']) == counted['dead_value_lines'] == 2


# Gold for runs whose plan predates it.

def test_gold_is_inferred_from_the_order_of_the_owner_turns_when_absent(rescore):
    episodes = [{'session_id': 'a', 'user': 'The window cleaning falls on Thursday.'},
                {'session_id': 'a', 'user': 'The spare chess set lives in the top shelf.'},
                {'session_id': 'b', 'user': 'New day for the window cleaning: Tuesday.'},
                {'session_id': 'c', 'user': 'Write answer.json with the day of the window cleaning.'}]
    case = {'id': 'update-day.01', 'inputs': {'scenario': 'update-day', 'episodes': episodes},
            'oracle': {'artifacts': [artifact('Tuesday', ['Thursday', 'top shelf'], key='day')]}}
    gold = rescore.inferred_gold(case)
    assert gold == [{'kind': 'update', 'evidence': [{'turn': 2}], 'stale': ['Thursday']}]
    never = copy.deepcopy(case)
    never['oracle']['artifacts'] = [artifact('unknown', ['top shelf'], key='day')]
    assert rescore.inferred_gold(never) == [{'kind': 'abstain', 'evidence': []}]
    conflict = copy.deepcopy(case)
    conflict['oracle']['artifacts'] = [{'path': 'answer.json', 'format': 'json', 'forbidden': [], 'assertions': [
        {'path': [], 'op': 'keys_equal', 'value': ['question', 'status', 'value']},
        {'path': ['status'], 'op': 'label_one_of', 'value': ['ask']}]}]
    assert rescore.inferred_gold(conflict)[0]['kind'] == 'conflict'


def conflict_case(sides=True):
    episodes = [{'session_id': 'a', 'user': 'The pickup is at 09:15.'},
                {'session_id': 'b', 'user': 'The pickup is at 10:45.'}, {'advance_clock': 86400}, {'tick': 1},
                {'session_id': 'c', 'user': 'When is the pickup?'}]
    spec = {'path': 'answer.json', 'format': 'json', 'forbidden': [], 'assertions': [
        {'path': [], 'op': 'keys_equal', 'value': ['question', 'status', 'time']},
        {'path': ['status'], 'op': 'label_one_of', 'value': ['ask']}]}
    if sides:
        spec['gold'] = {'kind': 'conflict', 'evidence': [{'turn': 0}, {'turn': 1}], 'sides': ['09:15', '10:45']}
    return {'id': 'contradiction-ask.01', 'inputs': {'scenario': 'contradiction-ask', 'episodes': episodes},
            'oracle': {'artifacts': [spec]}}


CONFLICT_LINES = [request([{'role': 'user', 'content': context()}]), turn(4, 'c', ('user', 'When is the pickup?'))]


def test_an_inferred_gold_is_labelled_and_the_conflict_trap_reads_the_status(rescore):
    picked = rescore.episode(conflict_case(False), CONFLICT_LINES, completion=False,
                             answer='{"status": "answer", "time": "10:45", "question": ""}')
    assert picked['gold_basis'] == 'inferred' and picked['trap']['outcome'] == 'trap'
    asked = rescore.episode(conflict_case(False), CONFLICT_LINES, completion=True,
                            answer='{"status": "ask", "time": "", "question": "Which time holds?"}')
    assert asked['trap']['outcome'] == 'asked'


@pytest.mark.parametrize('sides', [True, False])
@pytest.mark.parametrize('answer, outcome', [
    ('{"status": "ask", "time": "", "question": "Which time holds, 09:15 or 10:45?"}', 'asked'),
    # An "ask" that fills the time has chosen it, and asks nothing.
    ('{"status": "ask", "time": "09:15", "question": ""}', 'trap'),
    ('{"status": "answer", "time": "10:45", "question": ""}', 'trap'),
    # Malformed asks: no clarification question, one of no words, a time that is no side, or no time field.
    ('{"status": "ask", "time": "", "question": ""}', 'other'),
    ('{"status": "ask", "time": "", "question": " ? "}', 'other'),
    ('{"status": "ask", "time": "", "question": null}', 'other'),
    ('{"status": "ask", "question": "Which time holds?"}', 'other'),
])
def test_an_ask_is_the_specified_empty_time_and_a_question(rescore, sides, answer, outcome):
    row = rescore.episode(conflict_case(sides), CONFLICT_LINES, completion=outcome == 'asked', answer=answer)
    assert row['trap']['outcome'] == outcome


def test_an_ask_that_gives_a_time_that_is_no_side_is_no_ask(rescore):
    row = rescore.episode(conflict_case(), CONFLICT_LINES, completion=True,
                          answer='{"status": "ask", "time": "11:00", "question": "Which time holds?"}')
    assert row['trap']['outcome'] == 'other'


@pytest.mark.parametrize('answer, outcome', [
    ('{"place": "east room"}', 'trap'),
    ('{"place": " East Room "}', 'trap'),
    # A field that names the old value while denying it asserts no stale value.
    ('{"place": "It is no longer the east room; I do not know the new place"}', 'other'),
    ('{"place": "main hall"}', 'correct'),
    # A null field gives no value.
    ('{"place": null}', 'other'),
])
def test_an_answer_file_asserts_each_fields_whole_value(rescore, answer, outcome):
    lines = trace(request([{'role': 'user', 'content': context(Q_OLD)}]))
    row = rescore.episode(update_case(), lines, completion=outcome == 'correct', answer=answer)
    assert row['trap']['outcome'] == outcome


@pytest.mark.parametrize('answer, outcome', [
    ('{"place": null}', 'other'),
    ('{"place": "unknown"}', 'other'),
    ('{"place": "east room"}', 'trap'),
])
def test_an_abstention_trap_is_a_specific_value_given(rescore, answer, outcome):
    case = update_case()
    case['oracle']['artifacts'] = [artifact('unknown', ['east room'], {'kind': 'abstain', 'evidence': []})]
    row = rescore.episode(case, trace(request([{'role': 'user', 'content': context()}])), completion=False,
                          answer=answer)
    assert row['trap']['outcome'] == outcome


def test_rendered_gold_is_used_only_for_the_scenario_it_renders(rescore, tmp_path):
    generate = importlib.util.module_from_spec(
        spec := importlib.util.spec_from_file_location('paired_generate_rescore', GENERATORS / 'generate.py'))
    spec.loader.exec_module(generate)
    module = generate.load_templates(GENERATORS / 'memory_hard.py')
    generate.write(tmp_path / 'dev', module, 7, 'dev', 1, GENERATORS / 'memory_hard.py')
    rendered = rescore.RenderedGold([tmp_path / 'dev'])
    scenario = json.loads((tmp_path / 'dev' / 'scenarios.json').read_text())[0]
    stripped = copy.deepcopy(scenario['oracle']['artifacts'])
    for spec_ in stripped:
        del spec_['gold']
    case = {'id': scenario['id'], 'inputs': {key: scenario[key] for key in ('scenario', 'episodes', 'history',
                                                                            'initial_files')},
            'oracle': {'artifacts': stripped, 'declared_turns': 4}}
    assert rendered.gold(case) == [spec_['gold'] for spec_ in scenario['oracle']['artifacts']]
    changed = copy.deepcopy(case)
    changed['inputs']['history'][0]['messages'][0]['content'] += ' Changed.'
    assert rendered.gold(changed) is None
    assert rendered.mismatches == [scenario['id']]


# A whole results directory.

def results_directory(rescore, directory, monkeypatch, attempts, label='fixture-run'):
    """A paired results directory the re-score reads: ``attempts`` is a list of ``(case, {arm: (trace lines,
    completion, answer)}, workflow)``; a workflow names the scenario that repetitions share."""
    pairs, graded = [], []
    for index, (case, arms, workflow) in enumerate(attempts):
        episode = case['id'] if workflow is None else f"{case['id']}.repeat-{index + 1:02d}"
        members = {}
        for arm, (lines, _, _) in arms.items():
            path = directory / 'runs' / f'{index + 1:03d}-{arm}' / 'attempts' / case['id'] / 'private-trace.jsonl'
            path.parent.mkdir(parents=True)
            path.write_text('\n'.join(lines))
            members[arm] = {'path': f'runs/{index + 1:03d}-{arm}', 'case': case}
        pairs.append({'episode_id': episode, 'arms': members, **({'workflow_id': workflow} if workflow else {})})
        graded.append({'episode_id': episode, 'scenario_id': workflow or episode,
                       'completion': {arm: completion for arm, (_, completion, _) in arms.items()},
                       'results': {arm: {'effects': {'artifacts': {'answer.json': answer}}}
                                   for arm, (_, _, answer) in arms.items()}})
    manifest = {'label': label, 'dataset': {'version': 'mind-memory-2', 'split': 'dev'}, 'pairs': pairs}
    summary = {'pairs': graded, 'reference_arm': 'base', 'arm_order': ['base', 'full']}
    monkeypatch.setattr(rescore.paired_report, 'load_manifest', lambda _: manifest)
    monkeypatch.setattr(rescore.paired_report, 'arms_of', lambda _: (['base', 'full'], 'base', {}, {}))
    monkeypatch.setattr(rescore.paired_report, 'summarize', lambda _: summary)
    return directory


def numbered(index, gold=True):
    return dict(update_case(gold), id=f'buried-correction.{index:02d}')


REACHED = context(Q_OLD, Q_NEW, card('budget draft review with p-11', 'location', ('east room', 'q1')),
                  card('budget draft review', 'venue', ('main hall', 'q2')))


def test_a_results_directory_is_rescored_per_arm_without_copying_text(rescore, tmp_path, monkeypatch):
    case = update_case(gold=False)
    results_directory(rescore, tmp_path, monkeypatch, [(case, {
        'base': (trace(request([{'role': 'user', 'content': 'no context'}])), False, '{"place": "east room"}'),
        'full': (trace(request([{'role': 'user', 'content': REACHED}])), True, '{"place": "main hall"}')}, None)])
    result = rescore.rescore(tmp_path)
    assert result['label'] == 'fixture-run' and result['reference_arm'] == 'base'
    full, base = result['arms']['full'], result['arms']['base']
    assert full['summary']['overall']['two_by_two']['reached_right'] == 1
    assert base['summary']['overall']['two_by_two']['missed_wrong'] == 1
    assert base['summary']['overall']['traps'] == {'trap': 1}
    assert full['dead_lines']['by_class'] == {'different_key': 2} and full['dead_lines']['lines'] == 2
    assert full['dead_lines']['report_lines'] == 2
    assert result['gold_basis'] == {'inferred': 2}
    rendered = rescore.markdown(result)
    for text in (OLD, NEW, 'east room', 'main hall', 'budget draft'):
        assert text not in rendered and text not in json.dumps(result['arms']['full']['dead_lines'])
    assert 'different key' in rendered


# Paired contrasts: the paired track's contract, no partial cohort.

def test_a_contrast_with_an_unattributed_question_is_withheld(rescore, tmp_path, monkeypatch):
    """Six wins and one treatment attempt the report could not attribute: the cohort is not tested."""
    base = (trace(request([{'role': 'user', 'content': 'no context'}])), False, '{"place": "east room"}')
    full = trace(request([{'role': 'user', 'content': REACHED}]))
    attempts = [(numbered(index), {'base': base, 'full': (full, True if index < 6 else None, '{"place": "main hall"}')},
                 None) for index in range(7)]
    result = rescore.rescore(results_directory(rescore, tmp_path / 'one', monkeypatch, attempts))
    right = result['contrasts']['full vs base']['right']
    assert right == {'verdict': 'unavailable', 'declared_units': 7, 'unavailable_units': 1}
    # Reach was observed for all seven in both arms: its contrast stands, with its coverage.
    reach = result['contrasts']['full vs base']['reach_any']
    assert (reach['verdict'], reach['declared_units'], reach['unavailable_units'], reach['wins']) == \
        ('demonstrated', 7, 0, 7)
    assert '1 of 7 scenarios have an unattributed attempt' in rescore.markdown(result)
    # Attributed, the same cohort is tested.
    attempts[6][1]['full'] = (full, True, '{"place": "main hall"}')
    tested = rescore.rescore(results_directory(rescore, tmp_path / 'all', monkeypatch, attempts))
    tested = tested['contrasts']['full vs base']['right']
    assert (tested['verdict'], tested['wins'], tested['declared_units'], tested['unavailable_units']) == \
        ('demonstrated', 7, 7, 0)


def test_an_unobserved_probe_withholds_the_reach_contrast_not_the_answers(rescore, tmp_path, monkeypatch):
    base = (trace(request([{'role': 'user', 'content': 'no context'}])), False, '{"place": "east room"}')
    full = (trace(request([{'role': 'user', 'content': REACHED}])), True, '{"place": "main hall"}')
    attempts = [(numbered(index), {'base': base, 'full': full if index else ([], True, '{"place": "main hall"}')},
                 None) for index in range(7)]
    result = rescore.rescore(results_directory(rescore, tmp_path, monkeypatch, attempts))
    contrasts = result['contrasts']['full vs base']
    assert contrasts['reach_any'] == {'verdict': 'unavailable', 'declared_units': 7, 'unavailable_units': 1}
    assert contrasts['right']['verdict'] == 'demonstrated' and contrasts['right']['unavailable_units'] == 0
    full_arm = result['arms']['full']
    assert full_arm['probe_unobserved'] == 1 and result['arms']['base']['probe_unobserved'] == 0
    assert full_arm['summary']['overall']['reach_any']['trials'] == 6
    assert full_arm['summary']['overall']['two_by_two'] == {'reached_right': 6, 'reached_wrong': 0, 'missed_right': 0,
                                                            'missed_wrong': 0, 'unscored': 1}
    assert 'full 1' in rescore.markdown(result)


def test_repeated_runs_are_rescored_per_scenario(rescore, tmp_path, monkeypatch):
    """Repetitions share their case id: each attempt keeps its own identity, and the contrast averages them."""
    base = (trace(request([{'role': 'user', 'content': 'no context'}])), False, '{"place": "east room"}')
    full = trace(request([{'role': 'user', 'content': REACHED}]))
    # The last scenario passes once in its two repetitions.
    attempts = [(numbered(scenario), {'base': base, 'full': (full, scenario < 6 or repeat == 0,
                                                              '{"place": "main hall"}')},
                 f'buried-correction.{scenario:02d}') for repeat in range(2) for scenario in range(7)]
    result = rescore.rescore(results_directory(rescore, tmp_path, monkeypatch, attempts))
    rows = result['arms']['full']['rows']
    assert len(rows) == 14 and len({row['question'] for row in rows}) == 14
    assert {row['scenario'] for row in rows} == {f'buried-correction.{index:02d}' for index in range(7)}
    right = result['contrasts']['full vs base']['right']
    assert (right['units'], right['wins'], right['declared_units']) == (7, 7, 7)
    assert right['treatment_pass_rate'] == pytest.approx(13 / 14)
    assert result['contrasts']['full vs base']['reach_any']['units'] == 7
