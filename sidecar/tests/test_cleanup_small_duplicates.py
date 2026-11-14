"""Behaviour at the shared parser, metric, import and batch boundaries."""
import asyncio
from datetime import datetime, timedelta, timezone
import fcntl
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from colonymind.util import temporal
from benchmarks.public_memory import export, judge, reader, runner, storage
from benchmarks.public_memory.datasets import longmemeval_s, records
from benchmarks.paired.anchors import longmemeval_s as anchor
from colonymind.qualification import benchmark, native_memory_batch


@pytest.mark.parametrize('module,helper', [
    ('instinct.outcomes', temporal.parse_utc), ('mind.lessons', temporal.parse_utc),
    ('mind.gate', temporal.parse_utc), ('mind.wishes', temporal.parse_utc),
    ('mind.questions', temporal.parse_utc), ('mind.tick', temporal.parse_utc_timestamp),
    ('mind.drives', temporal.parse_utc_timestamp), ('mind.consolidate', temporal.parse_utc_text),
    ('atlas.snapshot', temporal.parse_utc_text),
])
def test_stored_stamps_keep_their_input_rules(module, helper):
    consumer = importlib.import_module('colonymind.' + module)
    assert getattr(consumer, 'utc' if module == 'atlas.snapshot' else '_utc') is helper
    for value in (None, '', 'invalid', ' 2023-05-20T02:21:00Z ', '0'):
        assert helper(value) is None
    expected = datetime(2023, 5, 20, 2, 21, tzinfo=timezone.utc)
    assert helper('2023-05-20T02:21:00Z') == helper(expected.replace(tzinfo=None)) == expected
    offset = expected.replace(tzinfo=timezone(timedelta(hours=3)))
    assert helper(offset).isoformat() == '2023-05-20T02:21:00+03:00'
    if helper is not temporal.parse_utc_text:
        assert helper(offset) is offset
    if helper is temporal.parse_utc_timestamp:
        assert helper(0) == helper(False) == datetime(1970, 1, 1, tzinfo=timezone.utc)
        assert helper(1.5).microsecond == 500000
        assert helper(True).second == 1
        with pytest.raises(ValueError):
            helper(float('nan'))
        with pytest.raises((OverflowError, OSError)):
            helper(float('inf'))
    else:
        assert helper(0) is helper(False) is helper(1.5) is None


def test_timestamp_parser_accepts_a_stamp_even_when_its_object_is_false():
    class Stamp:
        def __bool__(self):
            return False

        def __str__(self):
            return '2023-05-20T02:21:00Z'

    assert temporal.parse_utc(Stamp()) is temporal.parse_utc_text(Stamp()) is None
    assert temporal.parse_utc_timestamp(Stamp()) == datetime(2023, 5, 20, 2, 21, tzinfo=timezone.utc)
    assert temporal.parse_iso(' 2023-05-20T02:21:00Z ') == temporal.parse_utc('2023-05-20T02:21:00Z')


def test_stored_stamp_parsers_keep_string_conversion_error_semantics():
    class InvalidStamp:
        def __str__(self):
            raise ValueError('invalid stamp')

    class BrokenStamp:
        def __str__(self):
            raise TypeError('broken stamp')

    for helper in (temporal.parse_utc, temporal.parse_utc_text, temporal.parse_utc_timestamp):
        assert helper(InvalidStamp()) is None
        with pytest.raises(TypeError, match='broken stamp'):
            helper(BrokenStamp())


def test_benchmark_ece_keeps_boundaries_zero_empty_and_custom_bins():
    from colonymind.instinct.fit import ece
    from benchmarks.decisions import analysis, measure
    assert analysis.ece is measure.ece is ece
    assert ece([]) is None
    assert ece([(0.0, False), (1.0, True)]) == 0
    pairs = [(0.0, True), (0.1, False), (0.2, True), (0.5, False), (1.0, True)]
    assert ece(pairs) == pytest.approx(0.44)
    assert ece(pairs, bins=1) == pytest.approx(0.24)


def test_public_memory_stages_share_modules_and_dataset_records():
    assert reader.runner is storage.runner is export.runner is runner
    assert export.reader is reader and export.judge is judge
    assert judge.scoring is runner.scoring is export.scoring
    assert export.calibration.reader is reader
    assert runner.rescoring.scoring is runner.scoring
    assert export._adapter('longmemeval_s') is longmemeval_s
    assert export.subsets.DatasetError is records.DatasetError


@pytest.mark.parametrize('script', [
    'runner', 'reader', 'judge', 'labels', 'baselines', 'storage', 'export', 'calibration', 'paired_rescore',
])
def test_public_memory_direct_commands_import_from_outside_the_checkout(script, tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'benchmarks/public_memory' / (script + '.py')), '--help'],
        cwd=tmp_path, env={**os.environ, 'PYTHONPATH': str(ROOT / 'sidecar')},
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout


def question_item():
    return {'question_id': 'q1', 'question_type': 'knowledge-update', 'question': 'Which room?',
        'answer': 'The Blue Room.', 'question_date': '2023/05/20 (Sat) 02:00',
        'haystack_session_ids': ['late', 'early'],
        'haystack_dates': ['2023/05/20 (Sat) 02:21', '2023/05/19 (Fri) 03:00'],
        'haystack_sessions': [[{'role': 'user', 'content': 'The room is Blue.', 'has_answer': True}],
                             [{'role': 'user', 'content': 'The room was Red.'}]],
        'answer_session_ids': ['late']}


def test_anchor_keeps_source_order_and_question_date_while_public_replay_adjusts_time(tmp_path):
    path = tmp_path / 'questions.json'
    path.write_text(json.dumps([question_item()]))
    dataset = longmemeval_s.parse(path)
    question = dataset.questions[0]
    assert question.date_text == '2023/05/20 (Sat) 02:00'
    assert question.asked_at == datetime(2023, 5, 20, 2, 22, tzinfo=timezone.utc)
    assert dataset.findings['asked_after_last_session'] == ('q1',)
    [entry] = anchor.load_questions(path)
    assert entry == question_item()
    scenario = anchor.render_question(entry)
    assert scenario['history'] == [
        {'id': 'late', 'at': 1684549260, 'messages': [{'role': 'user', 'content': 'The room is Blue.'}]},
        {'id': 'early', 'at': int(datetime(2023, 5, 19, 3, tzinfo=timezone.utc).timestamp()),
         'messages': [{'role': 'user', 'content': 'The room was Red.'}]},
    ]
    assert question.date_text in scenario['episodes'][0]['user']
    assert '02:22' not in scenario['episodes'][0]['user']
    assert 'asked_at' not in scenario['episodes'][0]
    assert scenario['artifacts'][0]['assertions'][1]['value'] == ['The Blue Room.', 'The Blue Room', 'Blue Room']


@pytest.mark.parametrize('stamp,message', [
    ('unparsable', 'stamp'), ('2023/05/20 (Sun) 02:21', 'weekday'),
    ('2023/05/21 (Sun) 02:21', 'after the question date'),
])
def test_public_replay_refuses_bad_dates_while_anchor_keeps_its_date_rules(tmp_path, stamp, message):
    item = question_item()
    item['haystack_dates'][0] = stamp
    path = tmp_path / 'questions.json'
    path.write_text(json.dumps([item]))
    with pytest.raises(records.DatasetError, match=message):
        longmemeval_s.parse(path)
    [entry] = anchor.load_questions(path)
    assert entry == item
    assert anchor.history_for(entry)[0]['at'] == {
        'unparsable': anchor.FALLBACK_EPOCH,
        '2023/05/20 (Sun) 02:21': 1684549260,
        '2023/05/21 (Sun) 02:21': 1684635660,
    }[stamp]


def test_anchor_command_keeps_seeded_selection_and_loadable_scenarios_with_valid_dates(tmp_path):
    from colonymind.qualification.paired_cases import load_generated_dataset
    items = []
    for index, kind in enumerate(('single-session-user', 'multi-session', 'knowledge-update',
                                  'temporal-reasoning', 'single-session-user')):
        item = question_item()
        item.update(question_id=f'q{index}' + ('_abs' if index == 4 else ''), question_type=kind)
        items.append(item)
    path = tmp_path / 'questions.json'
    path.write_text(json.dumps(items))
    output = tmp_path / 'anchor'
    assert anchor.main(['--dataset', str(path), '--seed', '7', '--per-ability', '1', '--output', str(output)]) == 0
    manifest, scenarios, _ = load_generated_dataset(output)
    assert len(scenarios) == 5 and manifest['generator']['split'] == 'anchor'
    assert manifest['families'] == dict.fromkeys(anchor.ABILITIES, 1)
    assert set(manifest['generator']['question_ids'].values()) == {item['question_id'] for item in items}
    entries = anchor.load_questions(path)
    assert anchor.select(entries, 7, 1) == anchor.select(list(reversed(entries)), 7, 1)


@pytest.fixture
def batch(tmp_path, monkeypatch):
    calls = []
    groups = [({'id': f'shard-{i}', 'path': f'runs/shard-{i}', 'recipe': {'shard': i}}, ['case'])
              for i in range(2)]
    manifest = {'evidence_mode': 'controlled', 'suite_version': 'fixture-version'}
    args = SimpleNamespace(output=tmp_path, resume=False)
    supporting = {'nested': {'value': 'original'}}

    class Router:
        def __init__(self, **kwargs):
            pass

        def configure(self, config):
            assert config == supporting
            config['nested']['value'] = 'mutated in one case'

    monkeypatch.setattr('colonymind.router.LLMRouter', Router)
    monkeypatch.setattr(native_memory_batch, 'native_context', lambda config, recipe: recipe)
    monkeypatch.setattr(native_memory_batch, 'MemoryRouter', lambda router, context: (router, context))

    async def evaluate(target, recipe, cases, consumers, evaluators, factory, **kwargs):
        assert consumers is native_memory_batch.CONSUMERS and evaluators is native_memory_batch.EVALUATORS
        first, second = factory(cases[0]), factory(cases[0])
        assert first[0] is not second[0] and first[1] is second[1] is recipe
        calls.append((target.name, kwargs))
        target.mkdir(parents=True, exist_ok=True)
        (target / 'run.json').write_text('{}')

    monkeypatch.setattr(benchmark, 'evaluate', evaluate)
    monkeypatch.setattr(benchmark, 'summarize', lambda target: {'cases': []})
    monkeypatch.setattr(benchmark, 'inspect', lambda directory: {'outcomes': {'pass': len(calls)}})
    monkeypatch.setattr(benchmark, 'markdown', lambda report: 'report')
    return args, manifest, groups, supporting, calls


def execute(batch):
    args, manifest, groups, supporting, _ = batch
    return asyncio.run(native_memory_batch.execute(args, manifest, groups, supporting, {}))


def test_native_memory_common_executor_keeps_resume_reports_and_fresh_routers(batch):
    args, _, groups, supporting, calls = batch
    assert execute(batch) == {'outcomes': {'pass': 2}}
    assert supporting == {'nested': {'value': 'original'}}
    assert all(options == {'resume': False, 'evidence_mode': 'controlled', 'suite_version': 'fixture-version'}
               for _, options in calls)
    with pytest.raises(ValueError, match='resume'):
        execute(batch)
    assert len(calls) == 2
    args.resume = True
    assert execute(batch) == {'outcomes': {'pass': 4}}
    assert all(options['resume'] for _, options in calls[2:])
    for shard, _ in groups:
        target = args.output / shard['path']
        assert len(list(target.glob('report-*.json'))) == len(list(target.glob('report-*.md'))) == 2
    assert len(list(args.output.glob('report-*.json'))) == 2


def test_native_memory_common_executor_stops_on_retained_state_and_holds_lock(batch, monkeypatch):
    args, _, _, _, calls = batch
    with (args.output / '.benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            execute(batch)
    assert calls == []
    monkeypatch.setattr(benchmark, 'summarize', lambda target: {'cases': [{'cleanup': 'state_directory_retained'}]})
    assert execute(batch) == {'outcomes': {'pass': 1}}
    assert [name for name, _ in calls] == ['shard-0']
    assert not (args.output / 'runs/shard-1').exists()
