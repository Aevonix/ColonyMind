"""The memory benchmark end to end on a small history: a store read through the host's context route and a
baseline, both read by one scripted reader, judged by a scripted judge, and reported per arm.

The sidecar runs in process on an import-only store (no model call); the reader and the judge are scripted.
"""
import asyncio
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from colonymind.qualification import paired_body

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


class Scripted:
    def __init__(self, models, name, reply):
        self.spec = models.ModelSpec(name, 'rev-1')
        self.reply, self.requests = reply, []

    async def complete(self, messages, **options):
        self.requests.append({'messages': messages, **options})
        return self.reply(messages)


def test_a_question_goes_from_the_store_to_a_report_through_what_the_host_injects(tmp_path):
    reader = _load('colonymind_public_memory_reader', BENCH / 'reader.py')
    judge = _load('colonymind_public_memory_judge', BENCH / 'judge.py')
    baselines = _load('colonymind_public_memory_baselines', BENCH / 'baselines.py')
    records = _load('colonymind_public_memory_records_for_pipeline', BENCH / 'datasets' / 'records.py')
    models, runner = reader.models, reader.runner
    Turn, Session = records.Turn, records.Session
    sessions = (
        Session('s-1', 's-1', 0, datetime(2023, 3, 1, 9, tzinfo=UTC),
                (Turn(0, 'user', 'My office is in Room 4 on the second floor.'), Turn(1, 'assistant', 'Noted.'))),
        Session('s-2', 's-2', 1, datetime(2023, 3, 20, 9, tzinfo=UTC),
                (Turn(0, 'user', 'We moved: my office is in Room 9 now.'), Turn(1, 'assistant', 'Updated.'))))
    conversation = records.Conversation('p-1', sessions)
    question = records.Question('p-1', 'p-1', 'knowledge-update', 'Where is my office now?', 'Room 9',
                                datetime(2023, 4, 1, 12, tzinfo=UTC), None, False, ('s-1', 's-2'),
                                (records.Evidence('s-1', 0), records.Evidence('s-2', 0)))
    dataset = records.Dataset('longmemeval_s', '0' * 64, {'p-1': conversation}, (question,), {})
    try:
        asyncio.run(runner.replay.import_only(conversation, [question], tmp_path / 'store'))
    finally:
        paired_body.uninstall_clock()
    capture, base = tmp_path / 'capture.jsonl', tmp_path / 'baselines.jsonl'
    asyncio.run(runner.read(tmp_path / 'store', [question], arm='import-only', budgets=(4000,), capture=capture,
                            work=tmp_path / 'work'))
    baselines.build('bm25-sessions', dataset, [question], budgets=(4000,), capture=base)

    answer = json.dumps({'answer': 'Room 9', 'answerable': True})
    the_reader = Scripted(models, 'open-reader-7b', lambda messages: models.Reply(answer, model='open-reader-7b'))
    answers = tmp_path / 'answers.jsonl'
    for path in (capture, base):
        asyncio.run(reader.answer(path, {'p-1': question}, the_reader, answers))
    colonymind_prompt, baseline_prompt = (request['messages'][1]['content'] for request in the_reader.requests)
    # The ColonyMind arm is shown the sections the host returned, as the host renders them.
    row = json.loads(capture.read_text())
    assert reader.render(runner.fit(row['sections'], 4000)) in colonymind_prompt
    assert 'Room 9 now' in colonymind_prompt and '## Relevant Memories' in colonymind_prompt
    # The baseline is shown its packed sessions under the same heading, by date and never by session id.
    assert 'Room 9 now' in baseline_prompt and '## Relevant Memories' in baseline_prompt
    assert '"s-2"' not in baseline_prompt

    def judged(messages):
        if messages[0]['role'] == 'system':         # the trap judgment: the value it states as the answer
            listed = messages[-1]['content'].split('Values:\n')[1].splitlines()
            chosen = [int(line.split('.')[0]) for line in listed if line.endswith('Room 9')]
            return models.Reply(json.dumps({'asserted': chosen, 'asks_which_holds': False}), model='open-judge')
        return models.Reply('yes', model='open-judge')

    the_judge = Scripted(models, 'open-judge-70b', judged)
    verdicts = tmp_path / 'verdicts.jsonl'
    labels = {'p-1': {'kind': 'update', 'type': 'knowledge-update', 'values': ['Room 4']}}
    counts = asyncio.run(judge.judge(answers, {'p-1': question}, the_judge, verdicts, dataset='longmemeval_s',
                                     labels=labels))
    assert counts == {'judged': 2, 'already': 0}
    report = judge.report(verdicts)
    for arm in ('import-only', 'bm25-sessions'):
        table = report['summary'][arm]['4000']['overall']
        assert table['right']['successes'] == 1 and table['reach_any']['successes'] == 1
        assert table['traps'] == {'correct': 1}
    assert report['summary']['import-only']['4000']['overall']['reach_basis'] == {'structural': 1}
    assert report['summary']['bm25-sessions']['4000']['overall']['reach_basis'] == {'session': 1}


@pytest.fixture(autouse=True)
def _clock():
    try:
        yield
    finally:
        paired_body.uninstall_clock()
