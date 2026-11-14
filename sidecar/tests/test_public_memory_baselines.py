"""The memory benchmark's baselines (``benchmarks/public_memory/baselines.py``): BM25 over whole sessions, the
oracle, full context and stock Hermes' session search, packed by the packer a host's turn uses.

Synthetic conversations only. The session-search test that runs stock Hermes is skipped where Hermes is not
installed (the sidecar suite); the adapter environment has it.
"""
import asyncio
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

BENCH = Path(__file__).resolve().parents[2] / 'benchmarks' / 'public_memory'
UTC = timezone.utc


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def baselines():
    return _load('colonymind_public_memory_baselines', BENCH / 'baselines.py')


@pytest.fixture(scope='module')
def reader():
    return _load('colonymind_public_memory_reader', BENCH / 'reader.py')


@pytest.fixture(scope='module')
def records():
    return _load('colonymind_public_memory_records_for_baselines', BENCH / 'datasets' / 'records.py')


def conversation(records, long_note=0):
    """Four dated sessions under LongMemEval-like ids, the gold one named ``answer_...``."""
    Turn, Session = records.Turn, records.Session
    at = [datetime(2023, 3, day, 9, 0, tzinfo=UTC) for day in (1, 5, 9, 20)]
    sessions = (
        Session('sharegpt_a', 'sharegpt_a', 0, at[0], (Turn(0, 'user', 'Recommend a hiking trail near the lake.'),
                                                        Turn(1, 'assistant', 'Try the ridge loop.'))),
        Session('answer_b', 'answer_b', 1, at[1], (Turn(0, 'user', 'My office moved to Room 9 on the third floor.'),
                                                    Turn(1, 'assistant', 'Noted: Room 9.'),
                                                    Turn(2, 'user', 'x ' * long_note if long_note else 'Thanks.'),
                                                    Turn(3, 'assistant', 'Anything else?'))),
        Session('ultrachat_c', 'ultrachat_c', 2, at[2], (Turn(0, 'user', 'The office printer is jammed.'),
                                                          Turn(1, 'assistant', 'Restart it.'))),
        Session('sharegpt_d', 'sharegpt_d', 3, at[3], (Turn(0, 'user', 'What should I cook tonight?'),
                                                        Turn(1, 'assistant', 'A lentil soup.'))))
    conv = records.Conversation(id='q-1', sessions=sessions)
    question = records.Question('q-1', 'q-1', 'single-session-user', 'Which room is my office in?', 'Room 9',
                                datetime(2023, 3, 25, 12, 0, tzinfo=UTC), None, False, ('answer_b',),
                                (records.Evidence('answer_b', 0),))
    return conv, question


def test_bm25_ranks_whole_sessions_by_the_questions_terms(baselines, records):
    conv, question = conversation(records)
    ranked = baselines.ranked_sessions('bm25-sessions', conv, question)
    # Only sessions sharing a term with the question, best first: the office sessions, the gold one ahead.
    assert [session.key for session in ranked] == ['answer_b', 'ultrachat_c']
    # Ties keep the sessions' own order; a query with no shared term returns nothing.
    tied = baselines.bm25('alpha', ['alpha beta', 'alpha beta', 'gamma'])
    assert [index for index, _ in tied] == [0, 1] and tied[0][1] == tied[1][1] > 0
    assert baselines.bm25('delta', ['alpha', 'beta']) == []


def test_a_baseline_is_packed_by_the_hosts_turn_packer_within_the_budget(baselines, records):
    from colonymind.memory import blocks
    from colonymind.memory.recall import pack_memory_context
    conv, question = conversation(records, long_note=3000)
    row = baselines.session_row('oracle', conv, question, 2000)
    section, = row['sections']
    assert section['id'] == baselines.SECTION and section['title'] == 'Relevant Memories'
    assert blocks.cost(section['title'], section['body']) <= 2000
    # The gold session is longer than the budget: shortened to the room left, never dropped as a cut would.
    assert row['shown_sessions'][0] == 'answer_b' and row['shortened'] == ['answer_b']
    # Its first turns are shown whole; the long note and what follows it are not.
    assert row['shown_turns'][:2] == [['answer_b', 0], ['answer_b', 1]]
    assert ['answer_b', 3] not in row['shown_turns']
    # The body is the host packer's own output, with no count limit.
    items = [baselines._row(session.key, baselines.session_text(session)[0], session.at.isoformat(), 'session')
             for session in baselines.ranked_sessions('oracle', conv, question)]
    assert section['body'] == pack_memory_context(items, limit=None,
                                                  max_chars=2000 - blocks.cost('Relevant Memories', ''))[1]


def test_short_sessions_all_fit_with_no_count_limit(baselines, records):
    Turn, Session = records.Turn, records.Session
    sessions = tuple(Session(f's-{n}', f's-{n}', n, datetime(2023, 1, n + 1, tzinfo=UTC),
                             (Turn(0, 'user', f'Crate {n} is blue.'),)) for n in range(7))
    conv = records.Conversation(id='q', sessions=sessions)
    question = records.Question('q', 'q', 'multi-session', 'Which crates are blue?', 'all', datetime(
        2023, 2, 1, tzinfo=UTC), None, False, tuple(s.key for s in sessions), ())
    row = baselines.session_row('oracle', conv, question, 4000)
    assert row['shown_sessions'] == [f's-{n}' for n in range(7)] and row['shortened'] == []


def test_no_dataset_session_id_reaches_the_reader(baselines, reader, records):
    conv, question = conversation(records)
    for arm in ('bm25-sessions', 'oracle'):
        body = reader.render(reader.shown_sections(baselines.session_row(arm, conv, question, 4000)))
        assert 'answer_b' not in body and 'sharegpt' not in body and 'ultrachat' not in body
        assert '2023-03-05' in body and 'Room 9' in body


def test_the_oracle_shows_the_gold_sessions_and_reach_is_structural_by_session(baselines, records):
    conv, question = conversation(records)
    oracle = baselines.session_row('oracle', conv, question, 4000)
    assert oracle['shown_sessions'] == ['answer_b']
    scored = baselines.score(oracle)
    assert scored['reach']['basis'] == 'session' and scored['reach']['any'] is True
    assert scored['turn_reach']['any'] is True and scored['chars'] <= 4000 == scored['budget']
    missed = baselines.score({**oracle, 'shown_sessions': ['sharegpt_a'], 'shown_turns': [['sharegpt_a', 0]]})
    assert missed['reach']['any'] is False and missed['turn_reach']['any'] is False


def test_full_context_shows_everything_or_is_not_read(baselines, reader, records):
    conv, question = conversation(records)
    whole = baselines.session_row('full-context', conv, question, context_chars=100_000)
    assert whole['fits'] is True and whole['budget_chars'] is None
    assert whole['shown_sessions'] == [session.key for session in conv.sessions] and whole['shortened'] == []
    tight = baselines.session_row('full-context', conv, question, context_chars=200)
    assert tight['fits'] is False
    # The reader never reads a history that does not fit; it records why.
    models = reader.models
    unused = SimpleNamespace(spec=models.ModelSpec('open-reader-7b', 'rev-1'))
    answered = asyncio.run(reader.answer_row(tight, question, unused))
    assert answered['skipped'] == 'does_not_fit' and answered['answer'] is None


def test_full_context_fits_only_when_the_readers_whole_request_does(baselines, reader, records):
    from colonymind.memory import blocks
    conv, question = conversation(records)
    whole = baselines.session_row('full-context', conv, question, context_chars=10 ** 6)
    section, = whole['sections']
    memory = blocks.cost(section['title'], section['body'])
    # The history alone fits; the request the reader sends with it (its fixed text, the question and its date, the
    # response format) and the room its answer needs do not.
    assert baselines.session_row('full-context', conv, question, context_chars=memory)['fits'] is False
    prompt = reader.messages(question, reader.render(reader.shown_sections(whole)))
    assert whole['request_chars'] == len(json.dumps(prompt, ensure_ascii=False)) + len(json.dumps(
        reader.RESPONSE_FORMAT, ensure_ascii=False))
    assert whole['request_chars'] > sum(len(message['content']) for message in prompt) > memory
    assert whole['reserve_chars'] == reader.MAX_TOKENS * baselines.CHARS_PER_TOKEN
    edge = whole['request_chars'] + whole['reserve_chars']
    assert baselines.session_row('full-context', conv, question, context_chars=edge)['fits'] is True
    assert baselines.session_row('full-context', conv, question, context_chars=edge - 1)['fits'] is False


def test_baseline_rows_are_deterministic(baselines, records, tmp_path):
    conv, question = conversation(records, long_note=500)
    dataset = records.Dataset('longmemeval_s', '0' * 64, {'q-1': conv}, (question,), {})
    first, second = tmp_path / 'first.jsonl', tmp_path / 'second.jsonl'
    for capture in (first, second):
        for arm in ('bm25-sessions', 'oracle'):
            baselines.build(arm, dataset, [question], budgets=(1000, 4000), capture=capture)
        baselines.build('full-context', dataset, [question], capture=capture, context_chars=10_000)
    assert first.read_bytes() == second.read_bytes() and len(first.read_text().splitlines()) == 5
    # A second build keeps what is there and adds nothing.
    assert baselines.build('oracle', dataset, [question], budgets=(1000, 4000), capture=first) == {
        'written': 0, 'already': 2}


def test_the_reader_reads_a_baseline_row_as_packed(baselines, reader, records):
    conv, question = conversation(records)
    row = baselines.session_row('bm25-sessions', conv, question, 4000)
    assert reader.shown_sections(row) == row['sections']
    assert reader.row_score(row) == baselines.score(row)


# Stock Hermes' session search: the store Hermes' own import builds, the tool called once with the question.

def test_the_session_store_is_the_history_under_neutral_ids_at_its_own_times(baselines, records):
    conv, _ = conversation(records)
    payload, ids = baselines.hermes_sessions(conv)
    assert [item['id'] for item in payload] == ['session-1', 'session-2', 'session-3', 'session-4']
    assert ids == {'session-1': 'sharegpt_a', 'session-2': 'answer_b', 'session-3': 'ultrachat_c',
                   'session-4': 'sharegpt_d'}
    assert payload[1]['started_at'] == conv.sessions[1].at.timestamp()
    assert [message['timestamp'] - payload[1]['started_at'] for message in payload[1]['messages']] == [0, 1, 2, 3]


def test_a_search_result_is_packed_in_the_tools_order_and_mapped_back(baselines, records):
    conv, question = conversation(records)
    result = {'mode': 'discover', 'results': [
        {'session_id': 'session-3', 'when': '2023-03-09', 'snippet': 'The office printer is jammed.'},
        {'session_id': 'session-2', 'when': '2023-03-05', 'snippet': 'My office moved to Room 9'}]}
    ids = {'session-2': 'answer_b', 'session-3': 'ultrachat_c'}
    row = baselines.search_row(question, 4000, result, ids)
    assert row['returned_sessions'] == ['ultrachat_c', 'answer_b'] == row['shown_sessions']
    scored = baselines.score(row)
    assert scored['reach']['basis'] == 'session' and scored['reach']['first_rank'] == 2
    # Entries with no session id: only the weaker answer-span check can say anything.
    bare = baselines.search_row(question, 4000, {'results': [{'snippet': 'My office moved to Room 9'}]}, ids)
    weak = baselines.score(bare)
    assert weak['reach']['basis'] == 'span' and weak['reach']['weaker'] is True and weak['reach']['any'] is True
    empty = baselines.score(baselines.search_row(question, 4000, {'results': []}, ids))
    assert empty['reach']['any'] is False and empty['items'] == 0
    # The query each arm passes: the question as asked, or its own words joined by OR, in order, each once.
    assert row['query'] == 'Which room is my office in?'
    assert baselines.search_query('session-search-or', records.Question(
        'q-2', 'q-2', 't', 'Is my office in Room 9, or in room 4?', 'a', question.asked_at, None, False, (), ())
    ) == 'is OR my OR office OR in OR room OR 9 OR or OR 4'
    either = baselines.search_row(question, 4000, result, ids, arm='session-search-or')
    assert either['arm'] == 'session-search-or' and baselines.score(either)['reach']['any'] is True


def test_stock_hermes_session_search_returns_session_ids_for_the_history(baselines, records, tmp_path):
    native = pytest.importorskip('hermes_state')
    conv, question = conversation(records)
    result, ids = baselines.session_search(conv, records.Question(
        'q-1', 'q-1', 'single-session-user', 'office Room', 'Room 9', question.asked_at, None, False,
        ('answer_b',), ()), tmp_path)
    returned = [ids.get(entry.get('session_id')) for entry in result['results']]
    assert 'answer_b' in returned and all(returned)
    row = baselines.search_row(question, 4000, result, ids)
    assert baselines.score(row)['reach']['any'] is True
    # L keeps an empty implicit-AND result. S retries that miss using any term.
    # Keep the benchmark's one native tool call and assert each source's search contract.
    asked, asked_ids = baselines.session_search(conv, question, tmp_path / 'asked')
    assert asked['mode'] == 'discover'
    relax = getattr(native.SessionDB, '_or_relaxed_query', None)
    if callable(relax):
        sanitized = native.SessionDB._sanitize_fts5_query(question.question)
        assert sanitized == 'Which room is my office in'
        assert relax(sanitized) == 'Which OR room OR is OR my OR office OR in'
        assert relax('office OR Room') is None
        assert [entry['session_id'] for entry in asked['results']] == ['session-3', 'session-2']
        assert [asked_ids[entry['session_id']] for entry in asked['results']] == ['ultrachat_c', 'answer_b']
    else:
        assert asked['results'] == []
    # Explicit OR finds the gold session on both revisions, with UTC times on every host.
    either, ids = baselines.session_search(conv, question, tmp_path / 'either', arm='session-search-or')
    assert baselines.search_query('session-search-or', question) == 'which OR room OR is OR my OR office OR in'
    gold, = [entry for entry in either['results'] if ids[entry['session_id']] == 'answer_b']
    assert '09:00' in gold['when']
    row = baselines.search_row(question, 4000, either, ids, arm='session-search-or')
    assert row['query'] == baselines.search_query('session-search-or', question)
    assert baselines.score(row)['reach']['any'] is True
