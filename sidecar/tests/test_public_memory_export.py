"""The publication exporter (``benchmarks/public_memory/export.py``): public datasets only, allowlisted fields only.

A small campaign is judged by the real judge over scripted replies (nothing reaches a model endpoint): four
LongMemEval-shaped questions (a knowledge update, a temporal question, a single-session question and an abstention)
read by three arms. The exporter publishes one row per question, arm and budget with the fields the plan names
(tokens, evidence reached, ``answerable``, the verdict, the trap and the storage outcome), tables with intervals,
paired tests, every failure and the judge's agreement, and refuses what is not public: a dataset that is not a
pinned public one, a name the declaration does not give, a field outside its allowlist, a private term.
"""
import asyncio
import dataclasses
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import sys

import pytest

BENCH = Path(__file__).resolve().parents[2] / 'benchmarks' / 'public_memory'
UTC = timezone.utc
SERVED = 'private-variant-label-x'          # what a server reports: never published
JUDGE_SERVED = 'private-judge-label-y'
ANSWER_TEXT = 'It is Room 9 on the second floor.'


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def export():
    return _load('colonymind_public_memory_export', BENCH / 'export.py')


@pytest.fixture(scope='module')
def records():
    return _load('colonymind_public_memory_records_for_export', BENCH / 'datasets' / 'records.py')


def _question(records, qid, kind, answer, abstention=False):
    return records.Question(id=qid, conversation=qid, type=kind, question=f'Question {qid}?', answer=answer,
                            asked_at=datetime(2023, 4, 9, 12, tzinfo=UTC), date_text=None, abstention=abstention,
                            gold_sessions=('s-1',), evidence=(records.Evidence('s-1', 0),))


def dataset(export, records, name='longmemeval_s', sha256=None):
    questions = (_question(records, 'q-ku', 'knowledge-update', 'Room 9'),
                 _question(records, 'q-tr', 'temporal-reasoning', '3 days'),
                 _question(records, 'q-su', 'single-session-user', 'blue'),
                 _question(records, 'q-ab_abs', 'knowledge-update', 'never mentioned', abstention=True))
    pinned = export.PUBLIC[name]['sha256'] if name in export.PUBLIC else '0' * 64
    return records.Dataset(name, sha256 or pinned, {}, questions, {})


class Judge:
    """A judge scripted by what it is asked: yes when the answer is the reference, the trap and abstention
    judgments from the answer itself, and an unreadable reply for one answer."""

    def __init__(self, export, name='open-judge-70b', revision='rev-j'):
        self.models = export.judge.models
        self.spec = self.models.ModelSpec(name, revision)

    async def complete(self, messages, **options):
        system = messages[0]['content']
        user = messages[-1]['content']
        if system == export_module().judge.TRAP_SYSTEM:
            answer = re.search(r'Answer: (.*)\n\nValues:', user, re.S).group(1)
            listed = re.findall(r'^(\d+)\. (.*)$', user.split('Values:\n', 1)[1], re.M)
            content = json.dumps({'asserted': [int(n) for n, value in listed if value in answer],
                                  'asks_which_holds': False})
        elif system == export_module().judge.ABSTAIN_SYSTEM:
            content = json.dumps({'states_answer': 'never mentioned' not in user})
        else:
            reference = re.search(r'(?:Correct Answer|Rubric|Explanation): (.*)\n', user).group(1)
            response = user.split('Model Response: ', 1)[1].split('\n', 1)[0]
            content = 'Perhaps.' if 'UNREADABLE' in response else 'yes' if reference in response else 'no'
        return self.models.Reply(content, model=JUDGE_SERVED)


def export_module():
    return sys.modules['colonymind_public_memory_export']


def answer(export, qid, kind, arm, *, text, answerable=True, budget=4000, parse='ok', skipped=None, reached=True,
           tokens=1200, abstention=False, reader=None):
    reader_module = export.reader
    row = {'protocol': reader_module.PROTOCOL, 'question': qid, 'conversation': qid, 'type': kind,
           'abstention': abstention, 'arm': arm, 'budget_chars': budget, 'capture_sha256': 'c' * 64,
           'score': {'question': qid, 'conversation': qid, 'type': kind, 'arm': arm, 'items': 3, 'chars': 1800,
                     'budget': budget, 'returned_chars': 1800, 'cut': False,
                     'reach': {'basis': 'session' if arm.startswith('session') else 'structural', 'weaker': False,
                               'gold': 1, 'reached': int(reached), 'any': reached, 'all': reached,
                               'first_rank': 1 if reached else None}},
           'reader': reader or {'name': 'open-reader-7b', 'revision': 'rev-r'},
           'reading_sha256': reader_module.READING_SHA256, 'answer': None, 'answerable': None}
    if skipped:
        return {**row, 'skipped': skipped}
    row.update(prompt_sha256='d' * 64, memory_chars=1800, served_model=SERVED, usage={'prompt_tokens': tokens},
               reply=json.dumps({'answer': text, 'answerable': answerable}), parse=parse)
    if parse == 'ok':
        row.update(answer=text, answerable=answerable)
    else:
        row['refusal'] = 'the reply is not a JSON object'
    return row


LABELS = {'q-ku': {'kind': 'update', 'type': 'knowledge-update', 'values': ['Room 4']}}
ARMS = ('all-on', 'import-only', 'session-search')


def frozen_labels(labels=None):
    """``labels`` (``LABELS`` by default) frozen by ``labels.py freeze`` from the test dataset: the file's bytes
    and the SHA-256 a declaration pins."""
    export = _load('colonymind_public_memory_export', BENCH / 'export.py')
    records = _load('colonymind_public_memory_records_for_export', BENCH / 'datasets' / 'records.py')
    data = dataset(export, records)
    draft = {'protocol': export.labels_module.DRAFT, 'dataset': data.name, 'dataset_sha256': data.sha256,
             'entries': [{'question': qid, 'kind': label['kind'], 'values': list(label['values']), 'reviewed': True}
                         for qid, label in sorted((labels or LABELS).items())]}
    return export.labels_module.freeze(draft, data)


def labels_beside(answers):
    """The frozen label file ``campaign`` writes beside its answer file."""
    return answers.parent / 'old-values.json'


def campaign(export, records, tmp_path, *, judge=None):
    """Answer and verdict files for the three arms at 4,000 characters (all-on also at 24,000), the frozen label file
    the verdicts read (``labels_beside``), the storage rows, and the judge's agreement on the verdict file."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    data = dataset(export, records)
    questions = {question.id: question for question in data.questions}
    rows = []
    for arm in ARMS:
        rows += [answer(export, 'q-ku', 'knowledge-update', arm, text=ANSWER_TEXT if arm == 'all-on' else 'Room 4'),
                 answer(export, 'q-tr', 'temporal-reasoning', arm, text='3 days' if arm != 'session-search'
                        else 'UNREADABLE'),
                 answer(export, 'q-su', 'single-session-user', arm, text='blue', parse='refused'
                        if arm == 'import-only' else 'ok', reached=arm != 'session-search'),
                 answer(export, 'q-ab_abs', 'knowledge-update', arm, abstention=True,
                        text='It was never mentioned.' if arm == 'all-on' else 'Room 12', answerable=arm != 'all-on')]
    rows += [answer(export, 'q-ku', 'knowledge-update', 'all-on', text='Room 9', budget=24000),
             answer(export, 'q-tr', 'temporal-reasoning', 'all-on', text='', budget=24000, skipped='capture_failed'),
             answer(export, 'q-su', 'single-session-user', 'all-on', text='blue', budget=24000),
             answer(export, 'q-ab_abs', 'knowledge-update', 'all-on', abstention=True, budget=24000,
                    text='It was never mentioned.', answerable=False)]
    answers, verdicts = tmp_path / 'answers.jsonl', tmp_path / 'verdicts.jsonl'
    answers.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    model = judge or Judge(export)
    with verdicts.open('w') as stream:
        for row in rows:
            verdict = asyncio.run(export.judge.judge_row(row, questions[row['question']], model,
                                                         dataset='longmemeval_s', labels=LABELS))
            stream.write(json.dumps(verdict, sort_keys=True) + '\n')
    labels_beside(answers).write_bytes(frozen_labels()[0])
    storage = tmp_path / 'storage.jsonl'
    storage.write_text(''.join(json.dumps(row) + '\n' for row in (
        {'protocol': export.storage.PROTOCOL, 'question': 'q-ku', 'conversation': 'q-ku', 'type': 'knowledge-update',
         'kind': 'update', 'store': 'formed', 'outcome': 'superseded', 'counts': {}},
        {'protocol': export.storage.PROTOCOL, 'question': 'q-ku', 'conversation': 'q-ku', 'type': 'knowledge-update',
         'kind': 'update', 'store': 'import_only', 'outcome': 'not_stored', 'counts': {}})))
    return data, answers, verdicts, storage, agreement_of(export, data, verdicts)


def sheet_of(export, data, verdicts):
    """A labelling sheet drawn from ``verdicts`` by the calibration tool, labelled as a person would: nine items, one
    left unlabelled and one labelled against the judge."""
    sheet = export.calibration.sample(verdicts, {question.id: question for question in data.questions}, size=9,
                                      seed=1)
    judged = {(row['question'], row['arm'], row.get('budget_chars')): row
              for row in map(json.loads, verdicts.read_text().splitlines())}
    for index, item in enumerate(sheet['items'][:-1]):
        right = judged[(item['question'], item['arm'], item['budget_chars'])]['right']
        item['label'] = (not right) if index == 0 else right
    return sheet


def agreement_of(export, data, verdicts, sheet=None):
    """The judge's agreement on ``verdicts``, as ``calibration.py agreement`` reports it."""
    return export.calibration.agreement(verdicts, sheet or sheet_of(export, data, verdicts))


def declaration(**changes):
    body = {'dataset': 'longmemeval_s', 'subset': None, 'seed': 7, 'harness_commit': 'a' * 40,
            'labels_sha256': frozen_labels()[1],
            'models': {'reader': {'name': 'open-reader-7b', 'revision': 'rev-r',
                                  'source': 'https://huggingface.co/example/open-reader-7b'},
                       'judge': {'name': 'open-judge-70b', 'revision': 'rev-j'},
                       'formation': {'name': 'open-former-30b', 'revision': 'rev-f'},
                       'embedder': {'name': 'open-embed-1b', 'revision': 'rev-e'}, 'reranker': None}}
    body.update(changes)
    return body


def build(export, records, tmp_path, *, private_terms=(), **changes):
    data, answers, verdicts, storage, agreement = campaign(export, records, tmp_path)
    return export.build(declaration(**changes), data, answers=[answers], verdicts=[verdicts], storage=[storage],
                        agreement=agreement, labels=labels_beside(answers), private_terms=private_terms)


def by(results, arm, question, budget=4000):
    found, = [row for row in results['rows'] if (row['arm'], row['question'], row['budget_chars'])
              == (arm, question, budget)]
    return found


def test_one_allowlisted_row_per_question_arm_and_budget(export, records, tmp_path):
    results = build(export, records, tmp_path)
    assert results['protocol'] == export.PROTOCOL
    assert len(results['rows']) == 16
    assert all(set(row) == set(export.ROW_FIELDS) for row in results['rows'])
    ku = by(results, 'all-on', 'q-ku')
    assert (ku['type'], ku['kind'], ku['memory_chars'], ku['reader_prompt_tokens']) == (
        'knowledge-update', 'update', 1800, 1200)
    assert ku['reach'] == {'basis': 'structural', 'weaker': False, 'gold': 1, 'reached': 1, 'any': True, 'all': True,
                           'first_rank': 1}
    assert (ku['answerable'], ku['verdict'], ku['trap'], ku['storage']) == (True, 'right', 'correct', 'superseded')
    # The import-only arm reads the import-only store; a baseline reads no store.
    assert by(results, 'import-only', 'q-ku')['storage'] == 'not_stored'
    assert by(results, 'session-search', 'q-ku')['storage'] is None
    assert by(results, 'import-only', 'q-ku')['trap'] == 'trap'
    # An abstention: the reader declined (no trap), or gave a value as answerable (the trap).
    assert by(results, 'all-on', 'q-ab_abs')['trap'] == 'correct'
    assert by(results, 'session-search', 'q-ab_abs')['trap'] == 'trap'
    assert by(results, 'all-on', 'q-tr')['trap'] is None and by(results, 'all-on', 'q-tr')['storage'] is None


def test_every_failure_is_published_with_its_reason(export, records, tmp_path):
    results = build(export, records, tmp_path)
    assert by(results, 'all-on', 'q-tr', 24000)['verdict'] == 'skipped'
    assert by(results, 'import-only', 'q-su')['verdict'] == 'reader_refused'
    assert by(results, 'import-only', 'q-su')['answerable'] is None
    assert by(results, 'session-search', 'q-tr')['verdict'] == 'undecided'
    assert results['failures'] == [
        {'arm': 'all-on', 'budget_chars': 24000, 'question': 'q-tr', 'failure': 'capture_failed'},
        {'arm': 'import-only', 'budget_chars': 4000, 'question': 'q-su', 'failure': 'reader_refused'},
        {'arm': 'session-search', 'budget_chars': 4000, 'question': 'q-tr', 'failure': 'judge_undecided'}]


def test_nothing_but_the_allowlisted_fields_reaches_the_output(export, records, tmp_path):
    """No answer text, reply, judge call, server-reported model id, capture or prompt hash of a row is published."""
    text = json.dumps(build(export, records, tmp_path), sort_keys=True)
    for private in (SERVED, JUDGE_SERVED, ANSWER_TEXT, 'It was never mentioned', 'Room 12', 'Perhaps', 'c' * 64, 'd' * 64,
                    'asks_which_holds', 'served_model', 'reply', 'calls'):
        assert private not in text


def test_the_declared_models_prompts_and_hashes_are_published(export, records, tmp_path):
    results = build(export, records, tmp_path)
    assert results['models']['reader'] == {'name': 'open-reader-7b', 'revision': 'rev-r',
                                           'source': 'https://huggingface.co/example/open-reader-7b'}
    assert results['models']['reranker'] is None
    assert results['dataset'] == {'name': 'longmemeval_s', 'sha256': export.PUBLIC['longmemeval_s']['sha256'],
                                  'licence': 'MIT', 'source': export.PUBLIC['longmemeval_s']['source'],
                                  'questions': 4}
    assert results['questions'] == ['q-ab_abs', 'q-ku', 'q-su', 'q-tr']
    assert results['reader'] == {'protocol': export.reader.PROTOCOL, 'prompt_sha256': export.reader.READING_SHA256}
    assert results['judge'] == {'protocol': export.judge.PROTOCOL, 'scheme': 'longmemeval', 'votes': 1,
                                'prompts_sha256': export.judge.PROMPTS_SHA256}
    assert (results['harness_commit'], results['labels_sha256'], results['seed']) == ('a' * 40, frozen_labels()[1], 7)
    assert results['budgets'] == [4000, 24000] and results['arms'] == ['all-on', 'import-only', 'session-search']


def test_tables_with_intervals_per_arm_budget_and_type(export, records, tmp_path):
    results = build(export, records, tmp_path)
    table = results['tables']['all-on']['4000']
    assert table['overall']['right']['successes'] == 4 and table['overall']['right']['trials'] == 4
    assert 0 < table['overall']['right']['lower'] < 0.75 < table['overall']['right']['upper'] <= 1
    assert results['tables']['import-only']['4000']['overall']['right']['successes'] == 1
    assert set(table['types']) == {'knowledge-update', 'single-session-user', 'temporal-reasoning'}
    assert table['storage'] == {'superseded': 1}
    assert results['tables']['import-only']['4000']['overall']['traps'] == {'trap': 2}


def test_the_primary_paired_tests_are_computed_over_the_same_questions(export, records, tmp_path):
    results = build(export, records, tmp_path)
    named = {(item['treatment'], item['comparator'], item['measure'], tuple(item['types'] or ())): item
             for item in results['contrasts']}
    assert set(named) == {('all-on', 'session-search', 'right', ()),
                          ('all-on', 'import-only', 'right', ('knowledge-update', 'temporal-reasoning')),
                          ('all-on', 'session-search', 'no_trap', ())}
    # The session-search arm left one verdict undecided: the contrast is withheld, never tested on part.
    assert named[('all-on', 'session-search', 'right', ())]['verdict'] == 'unavailable'
    update = named[('all-on', 'import-only', 'right', ('knowledge-update', 'temporal-reasoning'))]
    # The two types, abstention questions of those types included: q-ku and q-ab_abs won, q-tr tied.
    assert (update['units'], update['wins'], update['ties'], update['budget_chars']) == (3, 2, 1, 4000)
    # The trap measure passes when the answer kept out of the trap, so a demonstrated verdict means fewer traps.
    trap = named[('all-on', 'session-search', 'no_trap', ())]
    assert (trap['units'], trap['wins'], trap['losses']) == (2, 2, 0)


def test_declared_contrasts_replace_the_primary_ones(export, records, tmp_path):
    results = build(export, records, tmp_path, contrasts=[
        {'treatment': 'all-on', 'comparator': 'import-only', 'measure': 'reached', 'types': None}])
    item, = results['contrasts']
    assert (item['measure'], item['units']) == ('reached', 4)


def test_the_judges_agreement_is_published_from_the_verdicts_it_was_drawn_from(export, records, tmp_path):
    results = build(export, records, tmp_path)
    assert results['agreement']['labelled'] == 8 and results['agreement']['agreement']['successes'] == 7
    data, answers, verdicts, storage, agreement = campaign(export, records, tmp_path / 'again')
    with pytest.raises(export.ExportError, match='agreement'):
        export.build(declaration(), data, answers=[answers], verdicts=[verdicts], storage=[storage],
                     labels=labels_beside(answers), agreement={**agreement, 'verdicts_sha256': 'e' * 64})


def _nested(agreement, path, value):
    """``agreement`` with ``value`` set at ``path`` (keys, outermost first)."""
    changed = json.loads(json.dumps(agreement))
    place = changed
    for key in path[:-1]:
        place = place[key]
    place[path[-1]] = value
    return changed


def test_the_agreement_is_rebuilt_from_its_counts_never_copied(export, records, tmp_path):
    results = build(export, records, tmp_path)
    agreement = results['agreement']
    assert set(agreement) == set(export.AGREEMENT_FIELDS)
    table = agreement['confusion']
    assert agreement['labelled'] == sum(table.values()) == 8 and agreement['unlabelled'] == 1
    assert agreement['agreement'] == export.scoring.wilson(table['judge_yes_label_yes'] + table['judge_no_label_no'], 8)
    assert agreement['agreement']['successes'] == 7
    assert agreement['kappa'] == export.calibration._kappa(table)
    assert set(agreement['types']) <= {question.type for question in dataset(export, records).questions}


def test_a_private_label_written_into_the_labelling_sheet_is_never_published(export, records, tmp_path):
    """The sheet is edited by hand: a type written into it that the declared set does not hold (a private variant
    label, say) reaches the calibration report as a key, and the export refuses it without repeating it."""
    data, answers, verdicts, storage, _ = campaign(export, records, tmp_path)
    sheet = sheet_of(export, data, verdicts)
    sheet['items'][1]['type'] = SERVED
    agreement = agreement_of(export, data, verdicts, sheet)
    assert SERVED in agreement['types']                 # the calibration tool keeps it
    with pytest.raises(export.ExportError, match='agreement.types') as refused:
        export.build(declaration(), data, answers=[answers], verdicts=[verdicts], storage=[storage],
                     agreement=agreement, labels=labels_beside(answers))
    assert SERVED not in str(refused.value)


@pytest.mark.parametrize('path, value', [
    (('agreement', 'endpoint'), 'private-endpoint-z'),
    (('confusion', 'private-endpoint-z'), 0),
    (('types', 'knowledge-update', 'agreement', 'host'), 'private-endpoint-z'),
    (('types', 'knowledge-update', 'private-endpoint-z'), {'key': 'x'}),
    (('kappa',), {'value': 0.5, 'server': 'private-endpoint-z'}),
    (('labelled',), 'private-endpoint-z'),
    (('served_by',), 'private-endpoint-z'),
])
def test_a_field_nested_anywhere_in_the_agreement_is_refused_without_repeating_it(export, records, tmp_path, path,
                                                                                   value):
    data, answers, verdicts, storage, agreement = campaign(export, records, tmp_path)
    with pytest.raises(export.ExportError, match='agreement') as refused:
        export.build(declaration(), data, answers=[answers], verdicts=[verdicts], storage=[storage],
                     agreement=_nested(agreement, path, value), labels=labels_beside(answers))
    assert 'private-endpoint-z' not in str(refused.value)


@pytest.mark.parametrize('path, value', [
    (('agreement', 'rate'), 0.99), (('agreement', 'lower'), 0.9), (('kappa',), 0.99), (('labelled',), 9),
    (('confusion', 'judge_yes_label_no'), 3), (('types', 'knowledge-update', 'labelled'), 1),
])
def test_agreement_numbers_that_are_not_their_counts_are_refused(export, records, tmp_path, path, value):
    data, answers, verdicts, storage, agreement = campaign(export, records, tmp_path)
    with pytest.raises(export.ExportError, match='agreement'):
        export.build(declaration(), data, answers=[answers], verdicts=[verdicts], storage=[storage],
                     agreement=_nested(agreement, path, value), labels=labels_beside(answers))


def test_the_export_is_deterministic(export, records, tmp_path):
    first = export.dumps(build(export, records, tmp_path / 'one'))
    second = export.dumps(build(export, records, tmp_path / 'two'))
    assert first == second


@pytest.mark.parametrize('name, sha256', [('mind-memory-1', None), ('precisionmembench', None),
                                          ('longmemeval_s', 'f' * 64)])
def test_a_dataset_that_is_not_a_pinned_public_one_is_refused(export, records, tmp_path, name, sha256):
    _, answers, verdicts, storage, _ = campaign(export, records, tmp_path)
    with pytest.raises(export.ExportError, match='public'):
        export.build(declaration(dataset=name), dataset(export, records, name, sha256), answers=[answers],
                     verdicts=[verdicts])


def test_a_question_the_dataset_or_subset_does_not_hold_is_refused(export, records, tmp_path):
    data, answers, verdicts, storage, _ = campaign(export, records, tmp_path)
    smaller = records.Dataset(data.name, data.sha256, {}, data.questions[:3], {})
    with pytest.raises(export.ExportError, match='q-ab_abs'):
        export.build(declaration(), smaller, answers=[answers], verdicts=[verdicts])


@pytest.mark.parametrize('changes', [{'endpoint': 'x'}, {'models': {**declaration()['models'],
                                                                     'reader': {'name': 'open-reader-7b',
                                                                                'revision': 'rev-r',
                                                                                'variant': 'y'}}},
                                     {'models': {**declaration()['models'], 'server': {'name': 'z', 'revision': 'r'}}}])
def test_a_declared_field_outside_the_allowlist_is_refused_by_name_alone(export, records, tmp_path, changes):
    data, answers, verdicts, storage, _ = campaign(export, records, tmp_path)
    with pytest.raises(export.ExportError) as refused:
        export.build(declaration(**changes), data, answers=[answers], verdicts=[verdicts])
    message = str(refused.value)
    assert any(key in message for key in ('endpoint', 'variant', 'server'))
    assert not any(value in message for value in ("'x'", "'y'", "'z'"))


@pytest.mark.parametrize('source', ['http://example.org/model', 'https://192.0.2.1/model', 'https://host:8443/m',
                                    'https://user@example.org/m', 'https://example.org/m?key=1', 'ftp://example.org'])
def test_a_model_source_that_is_not_a_public_web_page_is_refused(export, records, tmp_path, source):
    data, answers, verdicts, storage, _ = campaign(export, records, tmp_path)
    models = declaration()['models']
    with pytest.raises(export.ExportError, match='source'):
        export.build(declaration(models={**models, 'judge': {**models['judge'], 'source': source}}), data,
                     answers=[answers], verdicts=[verdicts])


@pytest.mark.parametrize('role, name', [('reader', 'another-reader'), ('judge', 'another-judge')])
def test_rows_from_a_model_the_declaration_does_not_name_are_refused(export, records, tmp_path, role, name):
    data, answers, verdicts, storage, _ = campaign(export, records, tmp_path)
    models = declaration()['models']
    with pytest.raises(export.ExportError, match=role):
        export.build(declaration(models={**models, role: {'name': name, 'revision': 'rev-r'}}), data,
                     answers=[answers], verdicts=[verdicts])


def test_a_store_formed_by_an_undeclared_model_is_refused(export, records, tmp_path):
    data, answers, verdicts, storage, _ = campaign(export, records, tmp_path)
    models = {**declaration()['models'], 'formation': None}
    with pytest.raises(export.ExportError, match='formation'):
        export.build(declaration(models=models), data, answers=[answers], verdicts=[verdicts])


def test_an_arm_that_is_not_the_harnesss_own_is_refused(export, records, tmp_path):
    data, answers, verdicts, storage, _ = campaign(export, records, tmp_path)
    for path in (answers, verdicts):
        path.write_text(path.read_text().replace('"import-only"', '"all-on-private-variant"'))
    with pytest.raises(export.ExportError, match='arm'):
        export.build(declaration(), data, answers=[answers], verdicts=[verdicts])


def test_rows_read_or_judged_another_way_than_this_harness_publishes_are_refused(export, records, tmp_path):
    data, answers, verdicts, storage, _ = campaign(export, records, tmp_path)
    judged = verdicts.read_text().replace(export.judge.PROMPTS_SHA256, '9' * 64)
    (tmp_path / 'other.jsonl').write_text(judged)
    with pytest.raises(export.ExportError, match='prompts'):
        export.build(declaration(), data, answers=[answers], verdicts=[tmp_path / 'other.jsonl'])


def rejudge(export, data, answers, verdicts, question):
    """``verdicts`` judged again with ``question`` in place of the dataset's question of that id."""
    questions = {item.id: item for item in data.questions}
    questions[question.id] = question
    rows = [json.loads(line) for line in answers.read_text().splitlines()]
    with verdicts.open('w') as stream:
        for row in rows:
            verdict = asyncio.run(export.judge.judge_row(row, questions[row['question']], Judge(export),
                                                         dataset='longmemeval_s', labels=LABELS))
            stream.write(json.dumps(verdict, sort_keys=True) + '\n')


def test_a_verdict_given_against_another_reading_of_its_question_is_refused(export, records, tmp_path):
    """A verdict names the question it was judged against (its words, reference answer, type and rubric); one judged
    against another reference answer is not a verdict on the pinned dataset's question."""
    data, answers, verdicts, storage, _ = campaign(export, records, tmp_path)
    question, = [item for item in data.questions if item.id == 'q-ku']
    rejudge(export, data, answers, verdicts, dataclasses.replace(question, answer='Room 4'))
    with pytest.raises(export.ExportError, match='q-ku.*another reading of its question'):
        export.build(declaration(), data, answers=[answers], verdicts=[verdicts], storage=[storage],
                     labels=labels_beside(answers))
    rejudge(export, data, answers, verdicts, question)
    assert export.build(declaration(), data, answers=[answers], verdicts=[verdicts], storage=[storage],
                        labels=labels_beside(answers))['rows']


def test_the_published_label_hash_is_the_label_file_every_verdict_read(export, records, tmp_path):
    """The declared labels_sha256 is published only when the label file hashes to it and every verdict read its label
    from that file: verdicts judged against other old values are refused under the original hash."""
    data, answers, verdicts, storage, _ = campaign(export, records, tmp_path)
    assert build(export, records, tmp_path / 'fine')['labels_sha256'] == frozen_labels()[1]
    other = {'q-ku': {**LABELS['q-ku'], 'values': ['Room 5']}}
    questions = {question.id: question for question in data.questions}
    rows = [json.loads(line) for line in answers.read_text().splitlines()]
    verdicts.write_text(''.join(json.dumps(asyncio.run(export.judge.judge_row(
        row, questions[row['question']], Judge(export), dataset='longmemeval_s', labels=other)), sort_keys=True) + '\n'
        for row in rows))
    with pytest.raises(export.ExportError, match='q-ku.*another frozen label'):
        export.build(declaration(), data, answers=[answers], verdicts=[verdicts], storage=[storage],
                     labels=labels_beside(answers))
    # The file of those other values, under the hash the declaration pins, is not the pinned file either.
    (tmp_path / 'other.json').write_bytes(frozen_labels(other)[0])
    with pytest.raises(export.ExportError, match='not the frozen longmemeval_s label file'):
        export.build(declaration(), data, answers=[answers], verdicts=[verdicts], storage=[storage],
                     labels=tmp_path / 'other.json')


def test_a_label_hash_without_its_file_or_a_file_without_its_hash_is_refused(export, records, tmp_path):
    data, answers, verdicts, storage, _ = campaign(export, records, tmp_path)
    with pytest.raises(export.ExportError, match='--labels'):
        export.build(declaration(), data, answers=[answers], verdicts=[verdicts], storage=[storage])
    with pytest.raises(export.ExportError, match='pins none'):
        export.build(declaration(labels_sha256=None), data, answers=[answers], verdicts=[verdicts],
                     storage=[storage], labels=labels_beside(answers))


def test_a_knowledge_update_judged_without_frozen_labels_is_refused(export, records, tmp_path):
    """A trap on a knowledge update is judged against frozen old values; with none declared it cannot be published."""
    data, answers, verdicts, storage, _ = campaign(export, records, tmp_path)
    with pytest.raises(export.ExportError, match='q-ku.*needs its frozen label'):
        export.build(declaration(labels_sha256=None), data, answers=[answers], verdicts=[verdicts],
                     storage=[storage])


def test_a_knowledge_update_read_from_a_store_needs_that_stores_storage_row(export, records, tmp_path):
    """An arm that reads a store publishes how that store holds each changed value: a run exported without its
    storage rows, or without one store's, is refused, never published with the outcome missing."""
    data, answers, verdicts, storage, _ = campaign(export, records, tmp_path)
    with pytest.raises(export.ExportError, match='q-ku.*formed store.*no storage row'):
        export.build(declaration(), data, answers=[answers], verdicts=[verdicts], labels=labels_beside(answers))
    rows = storage.read_text().splitlines()
    formed_only = tmp_path / 'formed-only.jsonl'
    formed_only.write_text(''.join(line + '\n' for line in rows if '"formed"' in line))
    with pytest.raises(export.ExportError, match='q-ku.*import_only store.*no storage row'):
        export.build(declaration(), data, answers=[answers], verdicts=[verdicts], storage=[formed_only],
                     labels=labels_beside(answers))


def test_two_storage_rows_that_disagree_are_refused(export, records, tmp_path):
    data, answers, verdicts, storage, _ = campaign(export, records, tmp_path)
    row = json.loads(storage.read_text().splitlines()[0])
    other = tmp_path / 'other-storage.jsonl'
    other.write_text(json.dumps({**row, 'outcome': 'both_current'}) + '\n')
    with pytest.raises(export.ExportError, match='q-ku.*formed store.*two storage rows'):
        export.build(declaration(), data, answers=[answers], verdicts=[verdicts], storage=[storage, other],
                     labels=labels_beside(answers))
    # The same row given twice (a storage file read twice) is one row.
    results = export.build(declaration(), data, answers=[answers], verdicts=[verdicts], storage=[storage, storage],
                           labels=labels_beside(answers))
    assert by(results, 'all-on', 'q-ku')['storage'] == 'superseded'


def test_a_verdict_with_no_answer_row_it_was_given_on_is_refused(export, records, tmp_path):
    data, answers, verdicts, storage, _ = campaign(export, records, tmp_path)
    lines = answers.read_text().splitlines()
    answers.write_text('\n'.join(lines[1:]) + '\n')
    with pytest.raises(export.ExportError, match='answer row'):
        export.build(declaration(), data, answers=[answers], verdicts=[verdicts], storage=[storage],
                     labels=labels_beside(answers))


def test_a_private_term_anywhere_in_the_output_is_refused_without_repeating_it(export, records, tmp_path):
    with pytest.raises(export.ExportError) as refused:
        build(export, records, tmp_path, private_terms=['FORMER-30B'])
    assert 'models.formation.name' in str(refused.value) and 'former-30b' not in str(refused.value).lower()


def test_the_command_writes_the_results_file(export, records, tmp_path, monkeypatch):
    data, answers, verdicts, storage, agreement = campaign(export, records, tmp_path)
    (tmp_path / 'declaration.json').write_text(json.dumps(declaration()))
    (tmp_path / 'agreement.json').write_text(json.dumps(agreement))
    (tmp_path / 'terms.txt').write_text('never-published-host\n\n')
    monkeypatch.setattr(export, 'load_dataset', lambda name, data_dir, offline: data)
    assert export.main(['results', '--declaration', str(tmp_path / 'declaration.json'),
                        '--answers', str(answers), '--verdicts', str(verdicts), '--storage', str(storage),
                        '--agreement', str(tmp_path / 'agreement.json'), '--private-terms', str(tmp_path / 'terms.txt'),
                        '--labels', str(labels_beside(answers)), '--output', str(tmp_path / 'results.json')]) == 0
    written = (tmp_path / 'results.json').read_text()
    assert json.loads(written)['protocol'] == export.PROTOCOL and written == export.dumps(json.loads(written))


def with_full_context(export, records, tmp_path, **changes):
    """The campaign plus the full-context baseline, which reads the whole history and so has no budget."""
    data, answers, verdicts, storage, agreement = campaign(export, records, tmp_path)
    questions = {question.id: question for question in data.questions}
    whole = [answer(export, 'q-ku', 'knowledge-update', 'full-context', text='Room 9', budget=None),
             *(answer(export, qid, kind, 'full-context', text='', budget=None, skipped='does_not_fit',
                      abstention=qid.endswith('_abs'))
               for qid, kind in (('q-tr', 'temporal-reasoning'), ('q-su', 'single-session-user'),
                                 ('q-ab_abs', 'knowledge-update')))]
    with answers.open('a') as stream:
        stream.write(''.join(json.dumps(row, sort_keys=True) + '\n' for row in whole))
    with verdicts.open('a') as stream:
        for row in whole:
            verdict = asyncio.run(export.judge.judge_row(row, questions[row['question']], Judge(export),
                                                         dataset='longmemeval_s', labels=LABELS))
            stream.write(json.dumps(verdict, sort_keys=True) + '\n')
    return export.build(declaration(**changes), data, answers=[answers], verdicts=[verdicts], storage=[storage],
                        labels=labels_beside(answers))


def test_the_full_context_arm_is_published_as_the_whole_history(export, records, tmp_path):
    results = with_full_context(export, records, tmp_path)
    assert results['budgets'] == [4000, 24000, None]
    row = by(results, 'full-context', 'q-ku', None)
    assert (row['verdict'], row['storage'], row['budget_chars']) == ('right', None, None)
    assert results['tables']['full-context'] == {export.WHOLE: results['tables']['full-context'][export.WHOLE]}
    assert results['tables']['full-context'][export.WHOLE]['overall']['right']['trials'] == 1
    assert {'arm': 'full-context', 'budget_chars': None, 'question': 'q-su', 'failure': 'does_not_fit'} \
        in results['failures']
    assert export.dumps(results) == export.dumps(with_full_context(export, records, tmp_path / 'again'))


def test_declared_budgets_are_the_budgeted_arms_and_the_full_context_arm_reads_the_whole_history(export, records,
                                                                                                  tmp_path):
    """A declaration's budgets are character budgets; the full-context baseline has none and is published beside
    them. A budget the declaration does not name is refused."""
    results = with_full_context(export, records, tmp_path / 'both', budgets=[4000, 24000])
    assert results['budgets'] == [4000, 24000, None] and 'full-context' in results['arms']
    with pytest.raises(export.ExportError, match='budget 24000.*not a declared one'):
        with_full_context(export, records, tmp_path / 'one', budgets=[4000])


@pytest.mark.parametrize('arm, budget, match', [('full-context', 4000, 'reads the whole history'),
                                                ('import-only', None, 'only full-context')])
def test_a_budget_that_is_not_the_arms_kind_is_refused(export, records, tmp_path, arm, budget, match):
    data, answers, verdicts, storage, _ = campaign(export, records, tmp_path)
    row = answer(export, 'q-su', 'single-session-user', arm, text='blue', budget=budget)
    verdict = asyncio.run(export.judge.judge_row(row, data.questions[2], Judge(export), dataset='longmemeval_s',
                                                 labels=LABELS))
    with answers.open('a') as stream:
        stream.write(json.dumps(row, sort_keys=True) + '\n')
    with verdicts.open('a') as stream:
        stream.write(json.dumps(verdict, sort_keys=True) + '\n')
    with pytest.raises(export.ExportError, match=match):
        export.build(declaration(budgets=[4000, 24000]), data, answers=[answers], verdicts=[verdicts],
                     storage=[storage], labels=labels_beside(answers))


def test_a_question_an_arm_left_out_is_refused(export, records, tmp_path):
    """Every arm at every budget answers every question of the declared set (the subset, else the file), so no
    question can drop out of a published arm unseen: a failure is a row, listed with its reason."""
    data, answers, verdicts, storage, _ = campaign(export, records, tmp_path)
    for path in (answers, verdicts):
        path.write_text(''.join(line + '\n' for line in path.read_text().splitlines()
                                if not ('"q-su"' in line and '"import-only"' in line)))
    with pytest.raises(export.ExportError, match='import-only.*1 question'):
        export.build(declaration(), data, answers=[answers], verdicts=[verdicts])


def test_a_declared_subset_is_the_question_set_every_arm_answers(export, records, tmp_path):
    data, answers, verdicts, storage, _ = campaign(export, records, tmp_path)
    subset = export.subsets.draw(data, seed=1, quotas={'knowledge-update': 2, 'temporal-reasoning': 1,
                                                       'single-session-user': 1}).record()
    results = export.build(declaration(subset=subset), data, answers=[answers], verdicts=[verdicts],
                           storage=[storage], labels=labels_beside(answers))
    assert results['subset'] == subset and results['questions'] == sorted(subset['ids'])
    smaller = export.subsets.draw(data, seed=1, quotas={'knowledge-update': 2, 'temporal-reasoning': 1,
                                                        'single-session-user': 0}).record()
    with pytest.raises(export.ExportError, match='q-su'):
        export.build(declaration(subset=smaller), data, answers=[answers], verdicts=[verdicts],
                     labels=labels_beside(answers))


def _rewritten(path, change):
    """The JSON lines file at ``path`` with ``change`` applied to every row."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    path.write_text(''.join(json.dumps(change(row), sort_keys=True) + '\n' for row in rows))


@pytest.mark.parametrize('votes', ['3 at a private host', 2, 3.0, True, None])
def test_the_judges_votes_are_one_of_its_own_counts_or_nothing_is_published(export, records, tmp_path, votes):
    """``judge.votes`` is published, so it is checked like every other published field: a whole number the judge
    gives its items (``judge.VOTES``). A verdict file whose votes hold anything else (text, another count, a float
    or a flag) is refused, and the refusal does not repeat it."""
    data, answers, verdicts, storage, agreement = campaign(export, records, tmp_path)
    _rewritten(verdicts, lambda row: {**row, 'votes': votes})
    with pytest.raises(export.ExportError, match='votes') as refused:
        export.build(declaration(), data, answers=[answers], verdicts=[verdicts], storage=[storage],
                     labels=labels_beside(answers))
    assert 'private host' not in str(refused.value)


def test_a_subsets_excluded_questions_are_questions_of_the_dataset(export, records, tmp_path):
    """A subset record lists the questions left out before its draw (``excluded``), and it is published as
    declared: each must be a question of the dataset, so nothing else can ride on it into the results file. The
    refusal names the field, never the value."""
    data, answers, verdicts, storage, _ = campaign(export, records, tmp_path)
    quotas = {'knowledge-update': 1, 'temporal-reasoning': 1, 'single-session-user': 1}
    held = export.subsets.draw(data, seed=1, quotas=quotas, exclude=['q-ab_abs']).record()
    assert held['excluded'] == ['q-ab_abs']
    kept = {'q-ku', 'q-tr', 'q-su'}
    for path in (answers, verdicts):
        path.write_text(''.join(line + '\n' for line in path.read_text().splitlines()
                                if json.loads(line)['question'] in kept))
    results = export.build(declaration(subset=held), data, answers=[answers], verdicts=[verdicts], storage=[storage],
                           labels=labels_beside(answers))
    assert results['subset']['excluded'] == ['q-ab_abs']
    for excluded in (['a private note', 'q-ab_abs'], 'q-ab_abs', ['q-ab_abs', 7]):
        ghost = export.subsets.draw(data, seed=1, quotas=quotas, exclude=['q-ab_abs']).record()
        ghost['excluded'] = excluded
        with pytest.raises(export.ExportError, match='subset.excluded') as refused:
            export.build(declaration(subset=ghost), data, answers=[answers], verdicts=[verdicts], storage=[storage],
                         labels=labels_beside(answers))
        assert 'private note' not in str(refused.value)


def test_a_configured_private_terms_list_always_applies(export, records, tmp_path, monkeypatch):
    """A publisher that keeps a private-terms list outside the repository configures it once (the file
    ``COLONYMIND_PRIVATE_TERMS`` names, else ``~/.config/colonymind/private-terms``): every export then applies it,
    with or without ``--private-terms``, and a configured list that cannot be read stops the export. With none
    configured and none given, nothing is refused for its terms."""
    data, answers, verdicts, storage, agreement = campaign(export, records, tmp_path)
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.delenv('COLONYMIND_PRIVATE_TERMS', raising=False)

    def build_it(**kwargs):
        return export.build(declaration(), data, answers=[answers], verdicts=[verdicts], storage=[storage],
                            agreement=agreement, labels=labels_beside(answers), **kwargs)
    assert build_it()['protocol'] == export.PROTOCOL
    default = home / '.config' / 'colonymind' / 'private-terms'
    default.parent.mkdir(parents=True)
    default.write_text('# a comment line\n\nopen-former\n')
    with pytest.raises(export.ExportError, match='models.formation.name'):
        build_it()
    with pytest.raises(export.ExportError, match='models.formation.name'):
        build_it(private_terms=['never-published-host'])
    named = tmp_path / 'terms.txt'
    named.write_text('open-judge\n')
    monkeypatch.setenv('COLONYMIND_PRIVATE_TERMS', str(named))
    with pytest.raises(export.ExportError, match='models.judge.name'):
        build_it()
    monkeypatch.setenv('COLONYMIND_PRIVATE_TERMS', str(tmp_path / 'missing.txt'))
    with pytest.raises(export.ExportError, match='private-terms list') as refused:
        build_it()
    assert 'missing.txt' not in str(refused.value)
    (tmp_path / 'declaration.json').write_text(json.dumps(declaration()))
    monkeypatch.setattr(export, 'load_dataset', lambda name, data_dir, offline: data)
    monkeypatch.setenv('COLONYMIND_PRIVATE_TERMS', str(named))
    with pytest.raises(export.ExportError, match='models.judge.name'):
        export.main(['results', '--declaration', str(tmp_path / 'declaration.json'), '--answers', str(answers),
                     '--verdicts', str(verdicts), '--storage', str(storage), '--labels', str(labels_beside(answers)),
                     '--output', str(tmp_path / 'results.json')])
    assert not (tmp_path / 'results.json').exists()

