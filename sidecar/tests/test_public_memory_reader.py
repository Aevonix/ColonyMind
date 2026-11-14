"""The memory benchmark's reader (``benchmarks/public_memory/reader.py``) and its model client (``models.py``).

A scripted model stands in for the reader; the HTTP client is exercised against an in-process transport. Nothing
here reaches a model endpoint.
"""
import asyncio
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import httpx
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
def reader():
    return _load('colonymind_public_memory_reader', BENCH / 'reader.py')


@pytest.fixture(scope='module')
def models():
    return _load('colonymind_public_memory_models', BENCH / 'models.py')


@pytest.fixture(scope='module')
def records():
    return _load('colonymind_public_memory_records_for_reader', BENCH / 'datasets' / 'records.py')


class Scripted:
    """A reader that answers from a script, one reply per call, and keeps every request it was sent."""

    def __init__(self, models, replies, name='open-reader-7b', revision='rev-1'):
        self.spec = models.ModelSpec(name, revision)
        self.replies, self.requests = list(replies), []

    async def complete(self, messages, **options):
        self.requests.append({'messages': messages, **options})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def question(records, qid='q-1', kind='knowledge-update', text='Where is my office now?', date_text=None):
    return records.Question(
        id=qid, conversation=qid, type=kind, question=text, answer='Room 9',
        asked_at=datetime(2023, 4, 9, 12, 0, tzinfo=UTC), date_text=date_text, abstention=False,
        gold_sessions=('s-3',), evidence=(records.Evidence('s-3', 0),))


def item(turn, text):
    return '- ' + json.dumps({'source_turn_id': turn, 'text': text})


def capture_row(reader, qid='q-1', budget=2000, arm='all-on'):
    """A capture row as the runner writes it: the sections the host returned, over the budget."""
    memory = '\n'.join(['Recalled:', item('t-1', 'a' * 900), item('t-2', 'b' * 900), item('t-3', 'c' * 900)])
    sections = [{'id': 'temporal-context', 'title': 'Current Time', 'body': 'Now: 2023-04-09 12:00 UTC'},
                {'id': 'colonymind-memory', 'title': 'Relevant Memories', 'body': memory},
                {'id': 'colonymind-status', 'title': 'Status', 'body': 'x' * 300}]
    return {'protocol': reader.runner.PROTOCOL, 'question': qid, 'conversation': qid, 'type': 'knowledge-update',
            'abstention': False, 'arm': arm, 'store': 'formed', 'budget_chars': budget, 'status': 200,
            'sections': sections, 'gold': [['turn:t-1'], ['turn:t-3']]}


# The fixed prompt: the question, its date and the memory section, nothing else.

def test_the_reader_prompt_is_fixed_and_carries_only_the_question_its_date_and_the_memory(reader, records):
    first = reader.messages(question(records, kind='knowledge-update'), 'Memory A')
    second = reader.messages(question(records, qid='q-2', kind='temporal-reasoning', text='When did I move?',
                                      date_text='2023/05/30 (Tue) 23:40'), 'Memory B')
    # One system text for every question: no category hint reaches the reader.
    assert first[0] == second[0] and first[0]['role'] == 'system'
    assert [message['role'] for message in first] == ['system', 'user']
    for messages, kind in ((first, 'knowledge-update'), (second, 'temporal-reasoning')):
        assert kind not in json.dumps(messages)
    assert 'Where is my office now?' in first[1]['content'] and 'Memory A' in first[1]['content']
    # The dataset's own stamp when it gives one; otherwise the time the question is asked, in UTC.
    assert '2023/05/30 (Tue) 23:40' in second[1]['content']
    assert reader.question_date(question(records)) == '2023-04-09 (Sun) 12:00 UTC'
    # An empty memory is said to be empty, never left as a blank the reader might misread.
    assert reader.EMPTY_MEMORY in reader.messages(question(records), '')[1]['content']


# The structured answer: a JSON object with the two fields, or a refusal. Free text is never read for an answer.

@pytest.mark.parametrize('text', [
    'Room 9',
    'The answer is {"answer": "Room 9", "answerable": true}',
    '{"answer": "Room 9", "answerable": true} I hope that helps.',
    '{"answer": "Room 9", "answerable": "yes"}',
    '{"answer": 9, "answerable": true}',
    '{"answer": "Room 9"}',
    '{"answer": "Room 9", "answerable": true, "confidence": 0.9}',
    '["Room 9", true]',
    '```\nRoom 9\n```',
    '',
])
def test_the_structured_parse_refuses_free_text(reader, text):
    with pytest.raises(reader.Refused):
        reader.parse(text)


def test_the_structured_parse_reads_the_object_and_nothing_else(reader):
    assert reader.parse('{"answer": "Room 9", "answerable": true}') == {'answer': 'Room 9', 'answerable': True}
    assert reader.parse('  {"answerable": false, "answer": ""}\n') == {'answer': '', 'answerable': False}
    # One fence around the object is the object; anything beside it is free text.
    assert reader.parse('```json\n{"answer": "Room 9", "answerable": true}\n```')['answer'] == 'Room 9'


# The reader reads what the host would inject, as the host renders it.

def test_the_reader_is_shown_what_the_host_injects_within_the_budget(reader, models, records):
    row = capture_row(reader)
    model = Scripted(models, [models.Reply('{"answer": "Room 9", "answerable": true}', model='open-reader-7b')])
    answered = asyncio.run(reader.answer_row(row, question(records), model))
    request, = model.requests
    kept = reader.runner.fit(row['sections'], row['budget_chars'])
    shown = reader.render(kept)
    # The host's rendering: each section as "## title" and its body, a blank line between sections.
    assert shown == '\n\n'.join(f"## {section['title']}\n{section['body']}" for section in kept)
    assert shown in request['messages'][1]['content'] and 'x' * 300 not in request['messages'][1]['content']
    assert request['temperature'] == 0 and request['max_tokens'] == reader.MAX_TOKENS
    assert request['response_format'] == reader.RESPONSE_FORMAT
    assert answered['answer'] == 'Room 9' and answered['answerable'] is True and answered['parse'] == 'ok'
    assert answered['reader'] == {'name': 'open-reader-7b', 'revision': 'rev-1'}
    assert answered['memory_chars'] == len(shown)
    # The row carries the capture's own score, so a verdict is scored without the capture file.
    assert answered['score'] == reader.runner.score(row)
    assert (answered['question'], answered['arm'], answered['budget_chars']) == ('q-1', 'all-on', 2000)


def test_a_reply_that_is_not_the_structure_is_recorded_as_refused_never_read(reader, models, records):
    model = Scripted(models, [models.Reply('I think it is Room 9.')])
    answered = asyncio.run(reader.answer_row(capture_row(reader), question(records), model))
    assert answered['parse'] == 'refused' and answered['answer'] is None and answered['answerable'] is None
    assert answered['reply'] == 'I think it is Room 9.' and answered['refusal']


def test_a_capture_the_host_failed_is_not_read(reader, models, records):
    model = Scripted(models, [])
    answered = asyncio.run(reader.answer_row({**capture_row(reader), 'status': 503, 'sections': []},
                                             question(records), model))
    assert answered['skipped'] == 'capture_failed' and model.requests == [] and answered['answer'] is None


def test_every_arm_is_read_by_one_model_and_a_run_resumes_where_it_stopped(reader, models, records, tmp_path):
    capture = tmp_path / 'capture.jsonl'
    rows = [capture_row(reader, arm='all-on'), capture_row(reader, arm='lexical-only'),
            capture_row(reader, arm='no-night')]
    capture.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    questions = {'q-1': question(records)}
    out = tmp_path / 'answers.jsonl'
    ok = models.Reply('{"answer": "Room 9", "answerable": true}')
    # A call that did not complete leaves its row unwritten and stops the run.
    first = Scripted(models, [ok, models.ModelError('status 502')])
    with pytest.raises(models.ModelError):
        asyncio.run(reader.answer(capture, questions, first, out))
    assert [json.loads(line)['arm'] for line in out.read_text().splitlines()] == ['all-on']
    # The same model resumes: rows already answered are not asked again.
    second = Scripted(models, [ok, ok])
    counts = asyncio.run(reader.answer(capture, questions, second, out))
    assert counts == {'answered': 2, 'already': 1, 'skipped': 0}
    assert [json.loads(line)['arm'] for line in out.read_text().splitlines()] == ['all-on', 'lexical-only',
                                                                                  'no-night']
    # Another model never adds to answers read by the first: every arm is read by the same reader.
    other = Scripted(models, [ok], name='another-reader', revision='rev-9')
    capture.write_text(capture.read_text() + json.dumps(capture_row(reader, arm='memory-core')) + '\n')
    with pytest.raises(reader.ReaderMismatch):
        asyncio.run(reader.answer(capture, questions, other, out))
    assert other.requests == []


def test_a_resumed_run_refuses_a_row_answered_from_other_inputs(reader, models, records, tmp_path):
    capture, out = tmp_path / 'capture.jsonl', tmp_path / 'answers.jsonl'
    row = capture_row(reader)
    capture.write_text(json.dumps(row) + '\n')
    ok = models.Reply('{"answer": "Room 9", "answerable": true}')
    asyncio.run(reader.answer(capture, {'q-1': question(records)}, Scripted(models, [ok]), out))
    before = out.read_bytes()

    def refused(rows, questions, error=reader.ResumeMismatch):
        capture.write_text(''.join(json.dumps(item) + '\n' for item in rows))
        model = Scripted(models, [ok, ok, ok])
        with pytest.raises(error):
            asyncio.run(reader.answer(capture, questions, model, out))
        # Everything a resume could refuse is checked before the first call; nothing is written.
        assert model.requests == [] and out.read_bytes() == before

    # The memory the answer was read from changed: the old answer is not an answer to the new row.
    changed = json.loads(json.dumps(row))
    changed['sections'][1]['body'] = changed['sections'][1]['body'].replace('a' * 900, 'z' * 900)
    refused([changed], {'q-1': question(records)})
    # The question as the reader is asked it changed: its words, or its date.
    refused([row], {'q-1': question(records, text='Where was my office?')})
    refused([row], {'q-1': question(records, date_text='2023/05/30 (Tue) 23:40')})
    # One row twice in a capture with different contents: which one was answered could not be told.
    twin = capture_row(reader, arm='lexical-only')
    refused([row, twin, {**twin, 'store': 'import-only'}], {'q-1': question(records)})
    # An answer read another way (the fixed prompt, its format or its settings) is never extended.
    out.write_text(json.dumps({**json.loads(before), 'reading_sha256': '0' * 64}) + '\n')
    before = out.read_bytes()
    refused([row, twin], {'q-1': question(records)}, error=reader.ReaderMismatch)
    out.write_bytes(before.replace(b'"reading_sha256": "' + b'0' * 64, b'"reading_sha256": "'
                                   + reader.READING_SHA256.encode()))
    before = out.read_bytes()
    # The same inputs resume with no call.
    capture.write_text(json.dumps(row) + '\n')
    model = Scripted(models, [])
    assert asyncio.run(reader.answer(capture, {'q-1': question(records)}, model, out)) == {
        'answered': 0, 'already': 1, 'skipped': 0}
    assert model.requests == []


# The HTTP client: where it is served and its key come from the environment, and never reach a row.

def test_the_http_client_reads_its_address_and_key_from_the_environment_only(models):
    seen = []

    def handler(request):
        seen.append({'url': str(request.url), 'auth': request.headers.get('authorization'),
                     'body': json.loads(request.content)})
        return httpx.Response(200, json={'model': 'served-reader', 'choices': [{'message': {
            'role': 'assistant', 'content': '{"answer": "", "answerable": false}'}}],
            'usage': {'prompt_tokens': 12, 'completion_tokens': 5}})

    key = 'bench-' + 'key-not-recorded'
    environ = {'COLONYMIND_BENCH_READER_BASE_URL': 'http://reader.invalid/v1',
               'COLONYMIND_BENCH_READER_API_KEY': key}
    client = models.OpenAICompatible('READER', models.ModelSpec('open-reader-7b', 'rev-1'), environ=environ,
                                     transport=httpx.MockTransport(handler))
    reply = asyncio.run(client.complete([{'role': 'user', 'content': 'q'}], temperature=0, max_tokens=64,
                                        response_format={'type': 'json_object'}))
    request, = seen
    assert request['url'] == 'http://reader.invalid/v1/chat/completions' and request['auth'] == f'Bearer {key}'
    assert request['body'] == {'model': 'open-reader-7b', 'messages': [{'role': 'user', 'content': 'q'}],
                               'temperature': 0, 'max_tokens': 64, 'response_format': {'type': 'json_object'}}
    assert reply == models.Reply('{"answer": "", "answerable": false}', model='served-reader',
                                 usage={'prompt_tokens': 12, 'completion_tokens': 5})
    # A failed call names its status only: a response body or an address can carry a secret.
    failing = models.OpenAICompatible('READER', models.ModelSpec('open-reader-7b', 'rev-1'), environ=environ,
                                      transport=httpx.MockTransport(lambda request: httpx.Response(
                                          500, text=f'upstream said {key} at http://reader.invalid')))
    with pytest.raises(models.ModelError) as raised:
        asyncio.run(failing.complete([{'role': 'user', 'content': 'q'}], temperature=0, max_tokens=8))
    assert key not in str(raised.value) and 'reader.invalid' not in str(raised.value)
    with pytest.raises(models.ModelError):
        models.OpenAICompatible('READER', models.ModelSpec('open-reader-7b', 'rev-1'), environ={})


def test_a_model_is_named_by_its_public_name_and_a_pinned_revision(models):
    with pytest.raises(ValueError):
        models.ModelSpec('open-reader-7b', '')
    with pytest.raises(ValueError):
        models.ModelSpec('', 'rev-1')
    spec = models.ModelSpec('Open-Reader-7B', 'rev-1')
    assert spec.identity() == {'name': 'Open-Reader-7B', 'revision': 'rev-1'}
    assert spec.same_model({'name': 'open-reader-7b', 'revision': 'rev-2'})
    assert not spec.same_model({'name': 'open-judge-70b', 'revision': 'rev-1'})
