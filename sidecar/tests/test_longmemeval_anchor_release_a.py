"""Paired anchor compatibility with release A, using its actual code as the oracle.

Set COLONYMIND_BENCH_DATA_DIR to a cache containing the pinned LongMemEval-S
file to run the frozen-data comparisons. Nothing is downloaded or vendored.
"""
import copy
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.paired.anchors import longmemeval_s as anchor
from benchmarks.public_memory.datasets import longmemeval_s, pinned

# The frozen oracle is independent of the implementation under test.
BASELINE = Path(__file__).parent / "fixtures/compatibility_sources"
ANCHOR_PATH = 'benchmarks/paired/anchors/longmemeval_s.py'


@pytest.fixture(scope='module')
def release_a():
    source = (BASELINE / ANCHOR_PATH).read_text()
    module = ModuleType('release_a_longmemeval_anchor')
    module.__file__ = str(ROOT / ANCHOR_PATH)
    exec(compile(source, str(BASELINE / ANCHOR_PATH), 'exec'), module.__dict__)
    return module


@pytest.fixture(scope='module')
def frozen_entries(release_a):
    directory = os.environ.get(pinned.DATA_DIR_ENV)
    if not directory or not (Path(directory) / longmemeval_s.PIN.name).exists():
        pytest.skip('the pinned LongMemEval-S file is not in COLONYMIND_BENCH_DATA_DIR')
    path = pinned.fetch(longmemeval_s.PIN, directory, download=False)
    return release_a.load_questions(path), anchor.load_questions(path)


def test_frozen_raw_records_and_eligible_ids_match_release_a(release_a, frozen_entries):
    before, after = frozen_entries
    assert after == before
    assert [entry['question_id'] for entry in after if anchor.eligible(entry)] == [
        entry['question_id'] for entry in before if release_a.eligible(entry)]
    for old, new in zip(before, after):
        assert type(new['answer']) is type(old['answer'])
        assert anchor.eligible(new) == release_a.eligible(old)


@pytest.mark.parametrize('seed', [7, 8, 42])
def test_frozen_selected_ids_and_generated_scenarios_match_release_a(release_a, frozen_entries, seed):
    before, after = frozen_entries
    old_module, old_chosen = release_a.module_for(before, seed)
    new_module, new_chosen = anchor.module_for(after, seed)
    assert [entry['question_id'] for entry in new_chosen] == [entry['question_id'] for entry in old_chosen]
    assert list(new_module.TEMPLATES) == list(old_module.TEMPLATES)
    generate = anchor.engine()
    assert generate.render(new_module, seed, 1) == generate.render(old_module, seed, 1)


def test_every_frozen_eligible_scenario_matches_release_a(release_a, frozen_entries):
    before, after = frozen_entries
    eligible = [entry for entry in before if release_a.eligible(entry)]
    by_id = {entry['question_id']: entry for entry in after}
    assert eligible
    for old in eligible:
        new = by_id[old['question_id']]
        assert anchor.template_name(new) == release_a.template_name(old)
        assert anchor.render_question(new) == release_a.render_question(old), old['question_id']


def raw_entry():
    # No answer_session_ids: paired accepts raw inputs without replay evidence.
    return {'question_id': 'q1', 'question_type': 'knowledge-update',
            'question': 'Which room?', 'answer': 'The Blue Room.',
            'question_date': 'original question date',
            'haystack_session_ids': ['same', 'empty', 'unsafe/id', 'same', 123],
            'haystack_dates': ['2023/05/21 (Wrong) 03:00', 'invalid', 'invalid',
                              '2023/05/20 02:21', '2023/05/22 (Mon) 03:00'],
            'haystack_sessions': [
                [{'role': 'user', 'content': 'The room is Blue.', 'has_answer': True},
                 {'role': 'system', 'content': 'Discard'}, None,
                 {'role': 'assistant', 'content': '  '}, {'role': 'user', 'content': 42}],
                [{'role': 'tool', 'content': 'Discard'}],
                [{'role': 'assistant', 'content': 'Kept'}],
                [{'role': 'user', 'content': 'The room was Red.'}],
                [{'role': 'user', 'content': 'Also kept'}]]}


def test_raw_entry_loading_and_rendering_keep_release_a_contract(release_a, tmp_path):
    entry = raw_entry()
    path = tmp_path / 'questions.json'
    path.write_text(json.dumps([entry]))
    assert anchor.load_questions(path) == release_a.load_questions(path) == [entry]
    before = copy.deepcopy(entry)
    assert anchor.history_for(entry) == release_a.history_for(entry)
    assert anchor.render_question(entry) == release_a.render_question(entry)
    history = anchor.history_for(entry)
    assert [session['id'] for session in history] == ['same', 'h-003', 'same', '123']
    assert history[1]['at'] == release_a.FALLBACK_EPOCH + 2 * release_a.DAY
    assert anchor.probe(entry) == release_a.probe(entry)
    assert entry == before


@pytest.mark.parametrize('stamp', [
    None, '', 'unparsable', '2023/05/20 02:21', '2023/05/20 (Sun) 02:21',
    'prefix 2023/05/20 (Anyweekday) 02:21 suffix',
    '2023/05/32 (Mon) 25:61', '2023/02/29 00:00', '2023/13/01 00:00',
    '0000/01/01 00:00', '9999/12/31 23:59',
])
def test_session_date_parsing_and_bad_date_fallback_match_release_a(release_a, stamp):
    entry = raw_entry()
    entry['haystack_dates'] = [stamp] * len(entry['haystack_dates'])
    assert anchor.epoch(stamp) == release_a.epoch(stamp)
    assert anchor.render_question(entry) == release_a.render_question(entry)


@pytest.mark.parametrize('field', ['haystack_session_ids', 'haystack_dates', 'haystack_sessions'])
def test_mismatched_session_lists_keep_release_a_zip_behavior(release_a, field):
    entry = raw_entry()
    entry[field] = entry[field][:3]
    assert anchor.history_for(entry) == release_a.history_for(entry)


def test_empty_filtered_history_keeps_release_a_error(release_a):
    entry = raw_entry()
    entry['haystack_sessions'] = [[{'role': 'system', 'content': 'Discard'}]]
    for module in (anchor, release_a):
        with pytest.raises(ValueError, match='at least one haystack session with conversation turns'):
            module.history_for(entry)


@pytest.mark.parametrize('answer', [42, None, True, ['Blue'], {'room': 'Blue'}])
def test_non_string_answers_keep_release_a_eligibility(release_a, tmp_path, answer):
    entry = raw_entry()
    entry['answer'] = answer
    path = tmp_path / 'questions.json'
    path.write_text(json.dumps([entry]))
    [loaded] = anchor.load_questions(path)
    assert loaded == release_a.load_questions(path)[0]
    assert not anchor.eligible(loaded)
    loaded['question_id'] += '_abs'
    assert anchor.eligible(loaded) == release_a.eligible(loaded) is True
    assert anchor.render_question(loaded) == release_a.render_question(loaded)


def test_raw_question_id_types_keep_release_a_distinctness_rules(release_a, tmp_path):
    first, second = raw_entry(), raw_entry()
    first['question_id'], second['question_id'] = 1, '1'
    path = tmp_path / 'questions.json'
    path.write_text(json.dumps([first, second]))
    assert anchor.load_questions(path) == release_a.load_questions(path) == [first, second]


def test_anchor_direct_command_imports_outside_checkout(tmp_path):
    import subprocess
    result = subprocess.run([sys.executable, str(ROOT / ANCHOR_PATH), '--help'],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout
