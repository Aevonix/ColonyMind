"""The memory benchmark's judge (``benchmarks/public_memory/judge.py``): verdicts, rubric scores and traps.

Scripted judges give fixed verdicts; nothing here reaches a model endpoint. The copied LongMemEval prompts are
compared with the pinned upstream file when ``COLONYMIND_BENCH_DATA_DIR`` holds it.
"""
import ast
import asyncio
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys

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
def judge():
    return _load('colonymind_public_memory_judge', BENCH / 'judge.py')


@pytest.fixture(scope='module')
def models():
    return _load('colonymind_public_memory_models', BENCH / 'models.py')


@pytest.fixture(scope='module')
def records():
    return _load('colonymind_public_memory_records_for_judge', BENCH / 'datasets' / 'records.py')


class Scripted:
    """A judge that answers from a script and keeps every request."""

    def __init__(self, models, replies, name='open-judge-70b', revision='rev-j', served='open-judge-70b'):
        self.spec = models.ModelSpec(name, revision)
        self.replies, self.requests, self.served = list(replies), [], served

    async def complete(self, messages, **options):
        self.requests.append({'messages': messages, **options})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if not isinstance(reply, str) else _reply(reply, self.served)


def _reply(content, served='open-judge-70b'):
    models = sys.modules['colonymind_public_memory_models']
    return models.Reply(content, model=served)


def at(day):
    return datetime(2023, 4, day, 12, 0, tzinfo=UTC)


def question(records, qid='q-1', kind='knowledge-update', answer='Room 9', abstention=False, detail=None,
             text='Where is my office now?'):
    return records.Question(id=qid, conversation=qid, type=kind, question=text, answer=answer, asked_at=at(9),
                            date_text=None, abstention=abstention, gold_sessions=('s-3',),
                            evidence=(records.Evidence('s-3', 0),), detail=detail or {})


def answer_row(qid='q-1', kind='knowledge-update', answer='Room 9', answerable=True, arm='all-on', budget=4000,
               parse='ok', reader=None, served='open-reader-7b', **extra):
    row = {'protocol': 'public-memory-answer-1', 'question': qid, 'conversation': qid, 'type': kind,
           'abstention': qid.endswith('_abs'), 'arm': arm, 'budget_chars': budget,
           'score': {'question': qid, 'conversation': qid, 'type': kind, 'arm': arm, 'items': 2, 'chars': 1800,
                     'budget': budget, 'reach': {'basis': 'structural', 'weaker': False, 'gold': 1, 'reached': 1,
                                                 'any': True, 'all': True, 'first_rank': 1}},
           'reader': reader or {'name': 'open-reader-7b', 'revision': 'rev-r'}, 'served_model': served,
           'reply': json.dumps({'answer': answer, 'answerable': answerable}),
           'answer': answer if parse == 'ok' else None, 'answerable': answerable if parse == 'ok' else None,
           'parse': parse}
    row.update(extra)
    return row


def run(judge, rows, questions, model, *, dataset='longmemeval_s', votes=1, labels=None):
    return [asyncio.run(judge.judge_row(row, questions[row['question']], model, dataset=dataset, votes=votes,
                                        labels=labels or {})) for row in rows]


# The copied prompts are LongMemEval's own.

def test_the_longmemeval_prompts_are_the_upstream_ones(judge):
    prompts = judge.prompts
    assert set(prompts.LONGMEMEVAL_BY_TYPE) == {'single-session-user', 'single-session-assistant', 'multi-session',
                                               'temporal-reasoning', 'knowledge-update', 'single-session-preference'}
    assert 'do not penalize off-by-one errors' in prompts.LONGMEMEVAL_TEMPORAL
    assert 'Does the model correctly identify the question as unanswerable?' in prompts.LONGMEMEVAL_ABSTENTION
    assert prompts.BEAM_RUBRIC.count('<rubric_item>') == 1 and prompts.BEAM_RUBRIC.count('<llm_response>') == 1
    data_dir = os.environ.get('COLONYMIND_BENCH_DATA_DIR')
    upstream = Path(data_dir) / 'longmemeval-evaluate_qa.py' if data_dir else None
    if upstream is None or not upstream.is_file():
        pytest.skip('the pinned LongMemEval judge file is not in COLONYMIND_BENCH_DATA_DIR')
    tree = ast.parse(upstream.read_text(encoding='utf-8'))
    function = next(node for node in tree.body if getattr(node, 'name', '') == 'get_anscheck_prompt')
    texts = {node.value.value for node in ast.walk(function)
             if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)}
    assert texts == {prompts.LONGMEMEVAL_ANSWER, prompts.LONGMEMEVAL_TEMPORAL, prompts.LONGMEMEVAL_UPDATE,
                     prompts.LONGMEMEVAL_PREFERENCE, prompts.LONGMEMEVAL_ABSTENTION}


# Scripted judges give fixed verdicts; a verdict is read strictly.

def test_a_longmemeval_verdict_uses_the_question_types_prompt_and_reads_yes_or_no(judge, models, records):
    questions = {'q-1': question(records, kind='temporal-reasoning', text='How many days between?', answer='18'),
                 'q-2_abs': question(records, qid='q-2_abs', kind='multi-session', abstention=True,
                                     answer='You did not mention a car.')}
    rows = [answer_row('q-1', kind='temporal-reasoning', answer='19 days'),
            answer_row('q-2_abs', kind='multi-session', answer='The memory does not say.', answerable=False)]
    model = Scripted(models, ['yes', 'No.'])
    first, second = run(judge, rows, questions, model)
    temporal, abstention = (request['messages'][-1]['content'] for request in model.requests)
    assert temporal == judge.prompts.LONGMEMEVAL_TEMPORAL.format('How many days between?', '18', '19 days')
    assert abstention == judge.prompts.LONGMEMEVAL_ABSTENTION.format(
        questions['q-2_abs'].question, 'You did not mention a car.', 'The memory does not say.')
    assert all(request['temperature'] == 0 for request in model.requests)
    assert first['right'] is True and second['right'] is False
    assert first['judge'] == {'name': 'open-judge-70b', 'revision': 'rev-j'} and first['votes'] == 1
    # The response judged is kept with the verdict, for the calibration sheet.
    assert first['response'] == '19 days' and second['response'] == 'The memory does not say.'
    assert [call['parsed'] for call in first['calls']] == [True]


@pytest.mark.parametrize('text', ['Yes, the response is correct.', 'yes and no', 'The answer is yes', 'y', '',
                                  '{"verdict": "yes"}'])
def test_the_verdict_parse_refuses_free_text(judge, text):
    with pytest.raises(judge.Refused):
        judge.parse_yes_no(text)


def test_the_verdict_parse_reads_yes_or_no_alone(judge):
    assert [judge.parse_yes_no(text) for text in ('yes', 'Yes.', ' NO ', 'no!')] == [True, True, False, False]


def test_a_refused_verdict_is_recorded_and_never_asked_again(judge, models, records):
    model = Scripted(models, ['I believe so.'])
    verdict, = run(judge, [answer_row(kind='multi-session')], {'q-1': question(records, kind='multi-session')},
                   model)
    assert verdict['right'] is None and verdict['undecided'] is True
    assert verdict['calls'][0]['parsed'] is None and verdict['calls'][0]['refusal']
    assert len(model.requests) == 1


def test_three_votes_are_taken_for_every_item_and_the_majority_decides(judge, models, records):
    questions = {'q-1': question(records, kind='multi-session'), 'q-2': question(records, qid='q-2',
                                                                                 kind='multi-session')}
    rows = [answer_row('q-1', kind='multi-session'), answer_row('q-2', kind='multi-session')]
    model = Scripted(models, ['yes', 'no', 'yes', 'no', 'maybe', 'yes'])
    first, second = run(judge, rows, questions, model, votes=3)
    assert len(model.requests) == 6
    assert all(request['temperature'] == judge.VOTE_TEMPERATURE for request in model.requests)
    assert first['right'] is True and first['votes'] == 3
    # One yes, one no and one refusal: no majority, so no verdict, and nothing is asked again.
    assert second['right'] is None and second['undecided'] is True
    with pytest.raises(ValueError):
        run(judge, rows, questions, Scripted(models, []), votes=2)


def test_three_rubric_votes_need_a_majority_for_one_score(judge, models, records):
    qid = 'c:information_extraction:0'
    questions = {qid: question(records, qid=qid, kind='information_extraction', answer='About 250ms',
                               detail={'rubric': ['LLM response should state: 250ms']})}

    def scored(*replies):
        verdict, = run(judge, [answer_row(qid, kind='information_extraction', answer='About 250ms.')], questions,
                       Scripted(models, [json.dumps({'score': reply, 'reason': 'r'}) if reply is not None
                                         else 'I will not score this.' for reply in replies]),
                       dataset='beam', votes=3)
        return verdict

    # Two replies that disagree beside a refusal decide nothing, as the same split does for a yes or no verdict.
    for split in ((0, 1, None), (0, 0.5, 1), (1, None, None)):
        verdict = scored(*split)
        assert verdict['rubric'] == [None] and verdict['rubric_score'] is None
        assert verdict['right'] is None and verdict['undecided'] is True
    # A majority of the three votes for one score decides it, a refusal counting as no vote.
    assert scored(1, 1, None)['rubric'] == [1.0] and scored(1, 1, None)['right'] is True
    assert scored(0.5, 1, 0.5)['rubric_score'] == 0.5


def test_three_trap_votes_need_a_majority_for_each_value(judge, models, records):
    questions = {'q-1': question(records)}
    order = judge.offered_values(questions['q-1'], LABELS['q-1'])
    stale, current = order.index('Room 4') + 1, order.index('Room 9') + 1

    def trap(*replies):
        verdict, = run(judge, [answer_row(answer='Room 4, or Room 9')], questions,
                       Scripted(models, ['no', 'no', 'no', *replies]), votes=3, labels=LABELS)
        return verdict['trap']

    # One vote finds the old value asserted, one the new one, one is refused: whether the old value is asserted
    # has no majority, so the trap is undecided rather than read as no trap.
    split = trap(_trap([stale]), _trap([current]), 'Room 4 is asserted.')
    assert split['outcome'] is None and split['undecided'] is True
    split = trap(_trap([current]), _trap([current], asks=True), 'Room 4 is asserted.')
    assert split['outcome'] is None and split['undecided'] is True
    assert trap(_trap([stale]), _trap([stale, current]), 'Room 4 is asserted.')['outcome'] == 'trap'
    # Three well-formed votes always give each value a majority.
    assert trap(_trap([stale]), _trap([current]), _trap([current]))['outcome'] == 'other'


def test_a_reader_refusal_is_a_wrong_answer_with_no_judge_call(judge, models, records):
    model = Scripted(models, [])
    refused, skipped = run(judge, [answer_row(parse='refused'), answer_row(arm='x', skipped='capture_failed')],
                           {'q-1': question(records)}, model,
                           labels={'q-1': {'kind': 'update', 'type': 'knowledge-update', 'values': ['Room 4']}})
    assert model.requests == []
    assert refused['right'] is False and refused['trap']['outcome'] == 'other'
    # A capture the host failed to serve has no answer to judge and no score: it is left out of every count.
    assert skipped['right'] is None and skipped['trap'] is None and skipped['skipped'] == 'capture_failed'


# BEAM: every rubric item judged with BEAM's own prompt, the score averaged over them.

def test_a_beam_answer_is_scored_per_rubric_item(judge, models, records):
    rubric = ['LLM response should state: 250ms', 'LLM response should mention: caching']
    questions = {'c:information_extraction:0': question(records, qid='c:information_extraction:0',
                                                         kind='information_extraction', answer='About 250ms',
                                                         detail={'rubric': rubric}),
                 'c:information_extraction:1': question(records, qid='c:information_extraction:1',
                                                         kind='information_extraction', answer='About 250ms',
                                                         detail={'rubric': rubric})}
    rows = [answer_row(qid, kind='information_extraction', answer='Around 250ms, after caching.')
            for qid in questions]
    model = Scripted(models, ['{"score": 1.0, "reason": "states 250ms"}', '{"score": 0.5, "reason": "partly"}',
                              '{"score": 1, "reason": "yes"}', '{"score": 1.0, "reason": "yes"}'])
    partial, full = run(judge, rows, questions, model, dataset='beam')
    first = model.requests[0]['messages'][-1]['content']
    assert first == judge.prompts.BEAM_RUBRIC.replace('<rubric_item>', rubric[0]).replace(
        '<llm_response>', 'Around 250ms, after caching.')
    assert partial['rubric'] == [1.0, 0.5] and partial['rubric_score'] == 0.75 and partial['right'] is False
    # Right is every rubric item fully met, as LongMemEval counts an answer with part of what is required wrong.
    assert full['rubric_score'] == 1.0 and full['right'] is True


def test_a_beam_reader_failure_meets_no_rubric_item_and_counts_in_the_mean(judge, models, records, tmp_path):
    rubric = ['LLM response should state: 250ms', 'LLM response should mention: caching']
    ids = [f'c:information_extraction:{number}' for number in range(4)]
    questions = {qid: question(records, qid=qid, kind='information_extraction', answer='About 250ms',
                               detail={'rubric': rubric}) for qid in ids}
    rows = [answer_row(ids[0], kind='information_extraction', answer='Around 250ms, after caching.'),
            answer_row(ids[1], kind='information_extraction', parse='refused'),
            answer_row(ids[2], kind='information_extraction', skipped='capture_failed'),
            answer_row(ids[3], kind='information_extraction', answer='Fast, I think.')]
    model = Scripted(models, ['{"score": 1, "reason": "yes"}', '{"score": 1, "reason": "yes"}',
                              'I cannot score this.', '{"score": 0, "reason": "no"}'])
    full, refused, skipped, unsure = run(judge, rows, questions, model, dataset='beam')
    # The reader's malformed reply is a wrong answer with no call, and it meets none of the rubric's items.
    assert len(model.requests) == 4
    assert refused['right'] is False and refused['reader_refused'] is True
    assert refused['rubric'] == [0.0, 0.0] and refused['rubric_score'] == 0.0
    # A row the reader skipped, or the judge left undecided, has no score: counted apart, never as a zero.
    assert skipped['rubric_score'] is None and unsure['rubric_score'] is None and unsure['undecided'] is True
    path = tmp_path / 'verdicts.jsonl'
    path.write_text(''.join(json.dumps(verdict) + '\n' for verdict in (full, refused, skipped, unsure)))
    report = judge.report(path)
    # One answer fully right and one the reader failed: the mean agrees with the accuracy beside it.
    assert report['rubric']['all-on']['4000'] == {'mean': 0.5, 'scored': 2}
    assert report['summary']['all-on']['4000']['overall']['right']['rate'] == 0.5
    assert report['counts']['all-on']['4000'] == {'rows': 4, 'reader_refused': 1, 'undecided': 1,
                                                  'trap_undecided': 0, 'skipped': 1}


@pytest.mark.parametrize('text', ['{"score": "1.0", "reason": "x"}', '{"score": 0.7, "reason": "x"}',
                                  '{"score": 1.0}', 'Score: 1.0', '{"score": 1.0, "reason": "x"} extra',
                                  '{"score": true, "reason": "x"}', '{"score": 1.0, "reason": "x", "more": 1}'])
def test_the_rubric_parse_refuses_anything_but_the_scored_object(judge, text):
    with pytest.raises(judge.Refused):
        judge.parse_rubric(text)


# Traps: structured judgments of which listed values the answer asserts, never words found in the answer.

LABELS = {'q-1': {'kind': 'update', 'type': 'knowledge-update', 'values': ['Room 4']}}


def _trap(asserted, asks=False):
    return json.dumps({'asserted': asserted, 'asks_which_holds': asks})


def test_an_update_trap_is_judged_against_the_frozen_old_values(judge, models, records):
    questions = {'q-1': question(records)}
    model = Scripted(models, ['no', None])
    order = judge.offered_values(questions['q-1'], LABELS['q-1'])
    assert sorted(order) == ['Room 4', 'Room 9']
    stale = order.index('Room 4') + 1
    model.replies[1] = _trap([stale])
    verdict, = run(judge, [answer_row(answer='Room 4')], questions, model, labels=LABELS)
    request = model.requests[1]
    assert request['response_format'] == judge.TRAP_FORMAT
    assert f'{stale}. Room 4' in request['messages'][-1]['content']
    assert verdict['trap']['outcome'] == 'trap' and verdict['trap']['matched'] == ['Room 4']
    # The new value asserted with a right verdict is correct; a flagged change that asks is "asked".
    current = order.index('Room 9') + 1
    right, = run(judge, [answer_row(answer='Room 9, it moved from Room 4')], questions,
                 Scripted(models, ['yes', _trap([current])]), labels=LABELS)
    assert right['trap']['outcome'] == 'correct'
    asked, = run(judge, [answer_row(answer='It was Room 4, then Room 9: which holds?')], questions,
                 Scripted(models, ['no', _trap([], asks=True)]), labels=LABELS)
    assert asked['trap']['outcome'] == 'asked'
    # A trap judgment with no majority is left undecided, counted apart and never asked again.
    unsure, = run(judge, [answer_row(answer='Room 4?')], questions,
                  Scripted(models, ['no', 'Room 4 is asserted.']), labels=LABELS)
    assert unsure['trap']['outcome'] is None and unsure['trap']['undecided'] is True
    assert 'trap' not in judge.scored(unsure)


@pytest.mark.parametrize('text', ['[1]', '{"asserted": [3], "asks_which_holds": false}',
                                  '{"asserted": [0], "asks_which_holds": false}',
                                  '{"asserted": ["1"], "asks_which_holds": false}',
                                  '{"asserted": [1, 1], "asks_which_holds": false}',
                                  '{"asserted": [1], "asks_which_holds": "no"}', 'Value 1 is asserted.'])
def test_the_trap_parse_refuses_anything_but_listed_value_numbers(judge, text):
    with pytest.raises(judge.Refused):
        judge.parse_trap(text, 2)


def test_a_conflict_trap_counts_one_side_chosen(judge, models, records):
    qid = 'c:contradiction_resolution:0'
    rubric = ['LLM response should state: there is contradictory information']
    questions = {qid: question(records, qid=qid, kind='contradiction_resolution', answer='Ask which holds.',
                               detail={'rubric': rubric})}
    labels = {qid: {'kind': 'conflict', 'type': 'contradiction_resolution', 'values': ['May 1', 'June 3']}}
    order = judge.offered_values(questions[qid], labels[qid])
    assert sorted(order) == ['June 3', 'May 1']
    chose, = run(judge, [answer_row(qid, kind='contradiction_resolution', answer='You start on May 1.')], questions,
                 Scripted(models, ['{"score": 0.0, "reason": "no"}', _trap([order.index('May 1') + 1])]),
                 dataset='beam', labels=labels)
    assert chose['trap']['kind'] == 'conflict' and chose['trap']['outcome'] == 'trap'
    asked, = run(judge, [answer_row(qid, kind='contradiction_resolution', answer='May 1 or June 3?')], questions,
                 Scripted(models, ['{"score": 1.0, "reason": "asks"}', _trap([1, 2], asks=True)]),
                 dataset='beam', labels=labels)
    assert asked['trap']['outcome'] == 'asked' and asked['right'] is True


def _states(value):
    return json.dumps({'states_answer': value})


def test_an_abstention_trap_is_a_specific_answer_given_as_answerable(judge, models, records):
    questions = {'q-3_abs': question(records, qid='q-3_abs', kind='multi-session', abstention=True,
                                     answer='No car was mentioned.')}
    model = Scripted(models, ['yes', 'no', _states(True)])
    declined, claimed = run(judge, [answer_row('q-3_abs', kind='multi-session', answer='Not in memory.',
                                               answerable=False),
                                    answer_row('q-3_abs', kind='multi-session', arm='lexical-only',
                                               answer='Level 2 of the garage.', answerable=True)],
                            questions, model)
    # The reader's own field says whether it claims an answer; a structured judgment, never the words of the
    # answer, says whether it gives a specific one. A reader that declines needs no judgment.
    assert len(model.requests) == 3
    request = model.requests[2]
    assert request['response_format'] == judge.ABSTAIN_FORMAT and request['messages'][0]['content'] == \
        judge.ABSTAIN_SYSTEM
    assert 'Level 2 of the garage.' in request['messages'][-1]['content']
    assert declined['trap']['outcome'] == 'correct' and claimed['trap']['outcome'] == 'trap'
    assert claimed['trap']['states_answer'] is True


def test_an_answer_flagged_answerable_that_gives_no_value_is_not_a_trap(judge, models, records):
    questions = {'q-3_abs': question(records, qid='q-3_abs', kind='multi-session', abstention=True,
                                     answer='No car was mentioned.')}
    words = 'The provided memory does not say, so I cannot answer.'
    # The flag says answerable, the words decline: the judgment finds no specific answer, so there is no trap, and
    # the correctness verdict on the words decides, as for an answer with no flag.
    accepted, = run(judge, [answer_row('q-3_abs', kind='multi-session', answer=words, answerable=True)], questions,
                    Scripted(models, ['yes', _states(False)]))
    assert accepted['right'] is True and accepted['trap']['outcome'] == 'correct'
    assert accepted['trap']['states_answer'] is False
    rejected, = run(judge, [answer_row('q-3_abs', kind='multi-session', answer=words, answerable=True)], questions,
                    Scripted(models, ['no', _states(False)]))
    assert rejected['trap']['outcome'] == 'other'
    # The flag says not answerable, the words give a value: the trap is a value given as answerable, so this is no
    # trap; the correctness verdict still reads the words.
    flagged, = run(judge, [answer_row('q-3_abs', kind='multi-session', answer='Level 2.', answerable=False)],
                   questions, Scripted(models, ['no']))
    assert flagged['right'] is False and flagged['trap']['outcome'] == 'correct'
    # The judgment is voted like every other: a split beside a refusal leaves the trap undecided.
    split, = run(judge, [answer_row('q-3_abs', kind='multi-session', answer='Maybe level 2.', answerable=True)],
                 questions, Scripted(models, ['no', 'no', 'no', _states(True), _states(False), 'Yes it does.']),
                 votes=3)
    assert split['trap']['outcome'] is None and split['trap']['undecided'] is True


@pytest.mark.parametrize('text', ['true', '{"states_answer": "yes"}', '{"states_answer": 1}', '{}',
                                  '{"states_answer": true, "why": "x"}', 'It states an answer.'])
def test_the_abstention_parse_refuses_anything_but_the_flag(judge, text):
    with pytest.raises(judge.Refused):
        judge.parse_abstain(text)


def test_a_plain_recall_question_has_no_trap(judge, models, records):
    verdict, = run(judge, [answer_row(kind='single-session-user')],
                   {'q-1': question(records, kind='single-session-user')}, Scripted(models, ['yes']))
    assert verdict['trap'] is None and verdict['trap_kind'] == 'recall'


# The judge is never the reader's model, and a run judges every row alike.

def test_the_judge_is_never_the_readers_model(judge, models, records, tmp_path):
    answers = tmp_path / 'answers.jsonl'
    answers.write_text(json.dumps(answer_row()) + '\n')
    questions = {'q-1': question(records, kind='single-session-user')}
    same = Scripted(models, ['yes'], name='Open-Reader-7B', revision='rev-other')
    with pytest.raises(judge.JudgeIsReader):
        asyncio.run(judge.judge(answers, questions, same, tmp_path / 'verdicts.jsonl', dataset='longmemeval_s'))
    assert same.requests == []
    # Another name served by the reader's own model is refused on its first reply, before anything is written.
    alias = Scripted(models, ['yes'], served='open-reader-7b')
    with pytest.raises(judge.JudgeIsReader):
        asyncio.run(judge.judge(answers, questions, alias, tmp_path / 'verdicts.jsonl', dataset='longmemeval_s'))
    assert not (tmp_path / 'verdicts.jsonl').exists()


def test_a_run_needs_a_frozen_label_for_every_trap_it_judges(judge, models, records, tmp_path):
    answers = tmp_path / 'answers.jsonl'
    answers.write_text(json.dumps(answer_row()) + '\n')
    model = Scripted(models, [])
    with pytest.raises(judge.LabelsMissing):
        asyncio.run(judge.judge(answers, {'q-1': question(records)}, model, tmp_path / 'v.jsonl',
                                dataset='longmemeval_s', labels={}))
    assert model.requests == []


def test_a_row_is_judged_whole_or_not_at_all_and_a_run_resumes(judge, models, records, tmp_path):
    answers = tmp_path / 'answers.jsonl'
    rows = [answer_row(arm='all-on'), answer_row(arm='lexical-only')]
    answers.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    questions = {'q-1': question(records)}
    out = tmp_path / 'verdicts.jsonl'
    stalled = Scripted(models, ['yes', _trap([1]), 'no', models.ModelError('status 502')])
    with pytest.raises(models.ModelError):
        asyncio.run(judge.judge(answers, questions, stalled, out, dataset='longmemeval_s', labels=LABELS))
    assert [json.loads(line)['arm'] for line in out.read_text().splitlines()] == ['all-on']
    # The unfinished row is judged again from its first call; the finished one is never asked again.
    resumed = Scripted(models, ['no', _trap([])])
    counts = asyncio.run(judge.judge(answers, questions, resumed, out, dataset='longmemeval_s', labels=LABELS))
    assert counts == {'judged': 1, 'already': 1} and len(resumed.requests) == 2
    # A different judge never extends verdicts another one gave.
    with pytest.raises(judge.JudgeMismatch):
        asyncio.run(judge.judge(answers, questions, Scripted(models, [], name='other-judge'), out,
                                dataset='longmemeval_s', labels=LABELS))


def test_a_resumed_run_refuses_a_verdict_given_on_other_inputs(judge, models, records, tmp_path):
    answers, out = tmp_path / 'answers.jsonl', tmp_path / 'verdicts.jsonl'
    row = answer_row()
    answers.write_text(json.dumps(row) + '\n')
    questions = {'q-1': question(records)}
    current = judge.offered_values(questions['q-1'], LABELS['q-1']).index('Room 9') + 1
    asyncio.run(judge.judge(answers, questions, Scripted(models, ['yes', _trap([current])]), out,
                            dataset='longmemeval_s', labels=LABELS))
    before = out.read_bytes()

    def refused(error, rows=(row,), questions=questions, labels=LABELS, dataset='longmemeval_s'):
        answers.write_text(''.join(json.dumps(item) + '\n' for item in rows))
        model = Scripted(models, ['yes', _trap([current])] * 2)
        with pytest.raises(error):
            asyncio.run(judge.judge(answers, questions, model, out, dataset=dataset, labels=labels))
        # Everything a resume could refuse is checked before the first call; nothing is written.
        assert model.requests == [] and out.read_bytes() == before

    # The answer judged changed: its verdict is not the new answer's verdict.
    refused(judge.ResumeMismatch, rows=[{**row, 'answer': 'Room 4'}])
    refused(judge.ResumeMismatch, rows=[{**row, 'answerable': False}])
    # The question as the judge reads it changed (its reference answer), or the frozen label its trap was
    # judged against.
    refused(judge.ResumeMismatch, questions={'q-1': question(records, answer='Room 11')})
    refused(judge.ResumeMismatch, labels={'q-1': {**LABELS['q-1'], 'values': ['Room 7']}})
    # Verdicts given under another scheme, or with other prompts, are never extended.
    refused(judge.JudgeMismatch, dataset='beam')
    out.write_text(json.dumps({**json.loads(before), 'prompts_sha256': '0' * 64}) + '\n')
    before = out.read_bytes()
    refused(judge.JudgeMismatch)
    out.write_bytes(before.replace(b'"prompts_sha256": "' + b'0' * 64, b'"prompts_sha256": "'
                                   + judge.PROMPTS_SHA256.encode()))
    # The same inputs resume with no call.
    answers.write_text(json.dumps(row) + '\n')
    model = Scripted(models, [])
    assert asyncio.run(judge.judge(answers, questions, model, out, dataset='longmemeval_s', labels=LABELS)) == {
        'judged': 0, 'already': 1}
    assert model.requests == []


# The report: the scorer's tables per arm and budget, from the verdict file alone.

def test_the_report_scores_each_arm_and_budget_from_the_verdicts(judge, models, records, tmp_path):
    questions = {'q-1': question(records, kind='single-session-user'),
                 'q-2': question(records, qid='q-2', kind='single-session-user')}
    rows = [answer_row('q-1', kind='single-session-user'), answer_row('q-2', kind='single-session-user'),
            answer_row('q-1', kind='single-session-user', arm='lexical-only'),
            answer_row('q-2', kind='single-session-user', arm='lexical-only', parse='refused')]
    verdicts = run(judge, rows, questions, Scripted(models, ['yes', 'no', 'yes']))
    path = tmp_path / 'verdicts.jsonl'
    path.write_text(''.join(json.dumps(verdict) + '\n' for verdict in verdicts))
    report = judge.report(path)
    table = report['summary']['all-on']['4000']
    assert table['overall']['right']['successes'] == 1 and table['overall']['right']['trials'] == 2
    assert table['overall']['two_by_two']['reached_right'] == 1
    assert report['counts']['lexical-only']['4000'] == {'rows': 2, 'reader_refused': 1, 'undecided': 0,
                                                       'trap_undecided': 0, 'skipped': 0}
    assert {row['question'] for row in report['rows']} == {'q-1', 'q-2'}
